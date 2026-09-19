#!/usr/bin/env python3
"""Select a Fourier cutoff on pilot trajectories; this is not calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_icassp2027_readout_audit import fit_ridge, load_tensor, measure, synthesize  # noqa: E402


def infer_max_modes(output_dim: int) -> int:
    if output_dim % 2 or ((output_dim // 2) - 1) % 2:
        raise ValueError("targets must contain two [real 0..K, imaginary 1..K] fields")
    return ((output_dim // 2) - 1) // 2


def coefficient_indices(max_modes: int, modes: int, device: torch.device) -> torch.Tensor:
    per_field = 2 * max_modes + 1
    one = torch.cat([
        torch.arange(modes + 1, device=device),
        torch.arange(max_modes + 1, max_modes + 1 + modes, device=device),
    ])
    return torch.cat([one, one + per_field])


def flatten_samples(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1])


def predict_full(latent: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, modes: int, output_dim: int) -> torch.Tensor:
    flat = flatten_samples(latent)
    prediction = flat @ weight + bias
    full = torch.zeros(flat.shape[0], output_dim, device=flat.device, dtype=prediction.dtype)
    max_modes = infer_max_modes(output_dim)
    full[:, coefficient_indices(max_modes, modes, flat.device)] = prediction
    return full.reshape(*latent.shape[:-1], output_dim)


def trajectory_scores(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=1)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    if scores.ndim != 1 or not len(scores) or not 0.0 < alpha < 1.0:
        raise ValueError("conformal scores must be a nonempty vector and alpha in (0, 1)")
    if not torch.isfinite(scores).all():
        raise ValueError("conformal scores must be finite")
    ordered = scores.sort().values
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    if rank > len(ordered):
        raise ValueError(f"{len(ordered)} calibration samples cannot support alpha={alpha}; need at least {math.ceil(1.0 / alpha) - 1}")
    return float(ordered[rank - 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-latents", type=Path, required=True)
    parser.add_argument("--fit-targets", type=Path, required=True)
    parser.add_argument("--calibration-latents", type=Path, required=True)
    parser.add_argument("--calibration-targets", type=Path, required=True)
    parser.add_argument("--eval-latents", type=Path, required=True)
    parser.add_argument("--eval-targets", type=Path, required=True)
    parser.add_argument("--levels", default="4,8,12,16,20,24")
    parser.add_argument("--grid-sizes", default="64,128,256,512")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--risk-ratio", type=float, default=1.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=484)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    fit_x = load_tensor(args.fit_latents, device)
    fit_y = load_tensor(args.fit_targets, device)
    cal_x = load_tensor(args.calibration_latents, device)
    cal_y = load_tensor(args.calibration_targets, device)
    eval_x = load_tensor(args.eval_latents, device)
    eval_y = load_tensor(args.eval_targets, device)
    if any(value.ndim != 3 for value in (fit_x, fit_y, cal_x, cal_y, eval_x, eval_y)):
        raise ValueError("all tensors must preserve [trajectory, time, feature] dimensions")
    max_modes = infer_max_modes(fit_y.shape[-1])
    levels = sorted(set(int(item) for item in args.levels.split(",") if item.strip() and int(item) <= max_modes))
    if not levels or levels[-1] != max_modes:
        levels.append(max_modes)

    fits = {}
    calibration = {}
    fit_x_flat = flatten_samples(fit_x)
    fit_y_flat = flatten_samples(fit_y)
    for modes in levels:
        indices = coefficient_indices(max_modes, modes, device)
        weight, bias = fit_ridge(fit_x_flat, fit_y_flat[:, indices], args.ridge)
        fits[modes] = (weight, bias)
        scores = trajectory_scores(predict_full(cal_x, weight, bias, modes, fit_y.shape[-1]), cal_y)
        # Archived eight-trajectory pilot used its maximum, a clipped rank.
        # ceil(9*.9)=9 > 8, so this cannot be a 90% conformal certificate.
        calibration[modes] = {"quantile": float(scores.max()), "scores": scores.tolist()}

    full_quantile = float(calibration[max_modes]["quantile"])
    risk_budget = args.risk_ratio * full_quantile
    selected = next((modes for modes in levels if float(calibration[modes]["quantile"]) <= risk_budget), max_modes)
    weight, bias = fits[selected]
    eval_prediction = predict_full(eval_x, weight, bias, selected, eval_y.shape[-1])
    eval_scores = trajectory_scores(eval_prediction, eval_y)
    selected_quantile = float(calibration[selected]["quantile"])
    coverage = float((eval_scores <= selected_quantile).float().mean())

    rows = []
    for grid in [int(item) for item in args.grid_sizes.split(",") if item.strip()]:
        with torch.no_grad():
            prediction_field = synthesize(flatten_samples(eval_prediction), grid)
            target_field = synthesize(flatten_samples(eval_y), grid)
            field_mse = float(torch.mean((prediction_field - target_field) ** 2))
            timing = measure(
                lambda: synthesize(
                    flatten_samples(predict_full(eval_x[:1], weight, bias, selected, eval_y.shape[-1]))[: args.batch_size], grid
                ), device, args.warmups, args.blocks, args.repeats,
            )
        rows.append({
            "grid_size": grid, "max_exported_modes": max_modes, "selected_modes": selected,
            "alpha": args.alpha, "risk_budget": risk_budget, "pilot_max_score": selected_quantile,
            "empirical_test_coverage": coverage, "mean_test_spectrum_mse": float(eval_scores.mean()),
            "field_mse": field_mse, "parameters": int(weight.numel() + bias.numel()), "device": str(device), **timing,
        })
    with (args.out_dir / "adaptive_readout_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"settings": vars(args), "levels": levels, "calibration": calibration,
               "selection_statistic": "pilot trajectory maximum (legacy quantile key)",
               "coverage_is_diagnostic_only": True, "rows": rows}
    (args.out_dir / "adaptive_readout_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# FPUT Pilot Cutoff Selection", "",
        f"Selected `K={selected}` from `{levels}` using {len(cal_x)} pilot trajectories. The maximum pilot spectrum error is `{selected_quantile:.6e}`; the diagnostic held-out fraction below it is `{coverage:.3f}`.", "",
        "No conformal guarantee is attached to this selection statistic. Freeze K, then use independent native-field calibration trajectories in run_icassp2027_full_field_conformal.py.", "",
        "| Grid | Selected K | Field MSE | Spectrum MSE | Coverage | Values | Median seconds |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['grid_size']} | {selected} | {row['field_mse']:.6e} | {row['mean_test_spectrum_mse']:.6e} | {coverage:.3f} | {row['parameters']} | {row['median_seconds']:.6e} |")
    (args.out_dir / "adaptive_readout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "adaptive_readout_summary.md")


if __name__ == "__main__":
    main()
