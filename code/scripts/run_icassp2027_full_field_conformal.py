#!/usr/bin/env python3
"""Calibrate adaptive Fourier readouts against native-grid physical fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_icassp2027_adaptive_readout_audit import coefficient_indices, conformal_quantile, flatten_samples, infer_max_modes  # noqa: E402
from run_icassp2027_readout_audit import fit_ridge, load_tensor, synthesize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-latents", type=Path, required=True)
    parser.add_argument("--fit-targets", type=Path, required=True)
    parser.add_argument("--calibration-bundle", type=Path, required=True)
    parser.add_argument("--eval-bundle", type=Path, required=True)
    parser.add_argument("--levels", default="4,8,12,16,20,24,32,48,63")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--risk-ratio", type=float, default=1.10)
    parser.add_argument("--fixed-modes", type=int, required=True, help="Mode count frozen on an independent pilot before calibration.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def field_prediction(latents: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, grid: int) -> torch.Tensor:
    coefficients = flatten_samples(latents) @ weight + bias
    return synthesize(coefficients, grid).reshape(latents.shape[0], latents.shape[1], 2, grid)


def field_scores(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=1)


def fput_energy(fields: torch.Tensor, beta: float) -> torch.Tensor:
    q, p = fields[:, :, 0], fields[:, :, 1]
    stretch = torch.roll(q, shifts=-1, dims=-1) - q
    return (0.5 * p.square() + 0.5 * stretch.square() + (beta / 4.0) * stretch.pow(4)).sum(dim=-1)


def physics_scores(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_scale = target.square().reshape(target.shape[0], -1).mean(dim=1).clamp_min(1e-12)
    state = field_scores(prediction, target) / target_scale
    predicted_energy = fput_energy(prediction, beta)
    target_energy = fput_energy(target, beta)
    energy = (predicted_energy - target_energy).abs().div(target_energy.abs().clamp_min(1e-12)).max(dim=1).values
    return torch.maximum(state, energy), state, energy


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    fit_x = load_tensor(args.fit_latents, device)
    fit_y = load_tensor(args.fit_targets, device)
    calibration_bundle = torch.load(args.calibration_bundle, map_location=device, weights_only=False)
    eval_bundle = torch.load(args.eval_bundle, map_location=device, weights_only=False)
    max_modes = infer_max_modes(fit_y.shape[-1])
    levels = sorted(set(int(item) for item in args.levels.split(",") if int(item) <= max_modes))
    if levels[-1] != max_modes:
        levels.append(max_modes)
    fits = {}
    quantiles = {}
    for modes in levels:
        indices = coefficient_indices(max_modes, modes, device)
        weight, bias = fit_ridge(flatten_samples(fit_x), flatten_samples(fit_y)[:, indices], args.ridge)
        fits[modes] = (weight, bias)
        quantiles[modes] = {}
        for label, bundle in calibration_bundle.items():
            prediction = field_prediction(bundle["latents"].to(device), weight, bias, int(bundle["grid_size"]))
            _, state_scores, energy_scores = physics_scores(prediction, bundle["fields"].to(device), float(bundle["beta"]))
            # Bonferroni split conformal gives a simultaneous guarantee for
            # the full state and the trajectory-wide energy constraint.
            component_alpha = args.alpha / 2.0
            quantiles[modes][label] = {
                "state": conformal_quantile(state_scores, component_alpha),
                "energy": conformal_quantile(energy_scores, component_alpha),
            }
    selected = args.fixed_modes
    if selected not in fits:
        raise ValueError("--fixed-modes must be one of --levels")
    weight, bias = fits[selected]
    rows = []
    for label, bundle in eval_bundle.items():
        prediction = field_prediction(bundle["latents"].to(device), weight, bias, int(bundle["grid_size"]))
        field_mse_scores = field_scores(prediction, bundle["fields"].to(device))
        scores, state_scores, energy_scores = physics_scores(prediction, bundle["fields"].to(device), float(bundle["beta"]))
        covered = bool(bundle["coverage_condition"])
        component_quantiles = quantiles[selected].get(label, {}) if covered else {}
        state_quantile = component_quantiles.get("state", float("nan"))
        energy_quantile = component_quantiles.get("energy", float("nan"))
        joint_coverage = float(((state_scores <= state_quantile) & (energy_scores <= energy_quantile)).float().mean()) if covered else float("nan")
        state_coverage = float((state_scores <= state_quantile).float().mean()) if covered else float("nan")
        energy_coverage = float((energy_scores <= energy_quantile).float().mean()) if covered else float("nan")
        rows.append({
            "target": label, "grid_size": int(bundle["grid_size"]), "beta": float(bundle["beta"]),
            "coverage_condition": covered, "selected_modes": selected, "state_quantile": state_quantile,
            "energy_quantile": energy_quantile, "state_coverage": state_coverage,
            "energy_coverage": energy_coverage, "joint_coverage": joint_coverage,
            "mean_field_mse": float(field_mse_scores.mean()), "worst_field_mse": float(field_mse_scores.max()),
            "mean_relative_state_error": float(state_scores.mean()), "mean_max_relative_energy_error": float(energy_scores.mean()),
            "mean_joint_physics_score": float(scores.mean()), "worst_joint_physics_score": float(scores.max()),
            "trajectories": len(scores), "parameters": int(weight.numel() + bias.numel()),
        })
    with (args.out_dir / "full_field_conformal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out_dir / "full_field_conformal.json").write_text(json.dumps({"levels": levels, "fixed_modes": args.fixed_modes, "quantiles": quantiles, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    selection_text = "frozen before calibration" if args.fixed_modes is not None else "selected using calibration"
    lines = ["# Joint Full-State/Energy Split-Conformal Audit", "", f"`K={selected}` was {selection_text}. State and trajectory-wide relative-energy scores use separate split-conformal quantiles at `alpha/2`; their intersection is therefore covered at least `1-alpha` by the union bound. Guarantees apply only to exchangeable covered conditions; OOD rows are diagnostic.", "", "| Target | Grid | Beta | Covered? | Field MSE | State q | Energy q | State cov. | Energy cov. | Joint cov. |", "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        values = ["n/a"] * 5 if not row["coverage_condition"] else [f"{row[key]:.3f}" for key in ("state_quantile", "energy_quantile", "state_coverage", "energy_coverage", "joint_coverage")]
        lines.append(f"| {row['target']} | {row['grid_size']} | {row['beta']:.1f} | {row['coverage_condition']} | {row['mean_field_mse']:.6e} | " + " | ".join(values) + " |")
    (args.out_dir / "full_field_conformal.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_dir / "full_field_conformal.md")


if __name__ == "__main__":
    main()
