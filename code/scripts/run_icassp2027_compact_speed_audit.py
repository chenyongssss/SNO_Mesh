#!/usr/bin/env python3
"""Benchmark compact adaptive ridge and a matched-output spectral MLP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_icassp2027_adaptive_readout_audit import coefficient_indices, flatten_samples, infer_max_modes  # noqa: E402
from run_icassp2027_readout_audit import SharedSpectralReadout, load_tensor, fit_ridge, measure, model_size, select_mlp, synthesize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-latents", type=Path, required=True)
    parser.add_argument("--fit-targets", type=Path, required=True)
    parser.add_argument("--eval-latents", type=Path, required=True)
    parser.add_argument("--eval-targets", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--grid-sizes", default="64,128,256,512")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=484)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    fit_x = flatten_samples(load_tensor(args.fit_latents, device)); eval_x = flatten_samples(load_tensor(args.eval_latents, device))
    fit_full = flatten_samples(load_tensor(args.fit_targets, device)); eval_full = flatten_samples(load_tensor(args.eval_targets, device))
    indices = coefficient_indices(infer_max_modes(fit_full.shape[-1]), args.modes, device)
    fit_y, eval_y = fit_full[:, indices], eval_full[:, indices]
    weight, bias = fit_ridge(fit_x, fit_y, args.ridge)
    ridge = SharedSpectralReadout(weight, bias, fit_y.shape[-1]).to(device).eval()
    mlp, hidden, epochs = select_mlp(fit_x.shape[-1], fit_y.shape[-1], [64, 128], fit_x, fit_y, 500, 1e-2, 0.2, 50, args.seed)
    rows = []
    batch = eval_x[:args.batch_size]
    for grid in [int(item) for item in args.grid_sizes.split(",") if item.strip()]:
        target = synthesize(eval_y, grid)
        basis = synthesize(torch.eye(fit_y.shape[-1], device=device), grid).reshape(fit_y.shape[-1], -1)
        fused_weight = weight @ basis
        fused_bias = bias @ basis
        fused_decode = lambda: (batch @ fused_weight + fused_bias).reshape(-1, 2, grid)
        timings = {}
        for name, model in (("adaptive_ridge", ridge), ("matched_spectral_mlp", mlp)):
            with torch.no_grad():
                prediction = synthesize(model(eval_x), grid)
                timing = measure(lambda m=model: synthesize(m(batch), grid), device, args.warmups, args.blocks, args.repeats)
            timings[name] = timing["median_seconds"]
            rows.append({"readout": name, "grid_size": grid, "field_mse": float(torch.mean((prediction-target)**2)), "parameters": model_size(model), "selected_hidden": hidden if name.endswith("mlp") else "", "selected_epochs": epochs if name.endswith("mlp") else "", "device": str(device), **timing})
        with torch.no_grad():
            fused_prediction = (eval_x @ fused_weight + fused_bias).reshape(-1, 2, grid)
            fused_timing = measure(fused_decode, device, args.warmups, args.blocks, args.repeats)
        timings["fused_adaptive_ridge"] = fused_timing["median_seconds"]
        rows.append({"readout": "fused_adaptive_ridge", "grid_size": grid, "field_mse": float(torch.mean((fused_prediction-target)**2)), "parameters": model_size(ridge), "selected_hidden": "", "selected_epochs": "", "device": str(device), **fused_timing})
        rows.append({"readout": "ridge_speedup_vs_mlp", "grid_size": grid, "field_mse": "", "parameters": "", "selected_hidden": "", "selected_epochs": "", "device": str(device), "median_seconds": timings["matched_spectral_mlp"] / timings["adaptive_ridge"], "iqr_seconds": "", "peak_memory_bytes_median": ""})
        rows.append({"readout": "fused_speedup_vs_mlp", "grid_size": grid, "field_mse": "", "parameters": "", "selected_hidden": "", "selected_epochs": "", "device": str(device), "median_seconds": timings["matched_spectral_mlp"] / timings["fused_adaptive_ridge"], "iqr_seconds": "", "peak_memory_bytes_median": ""})
    with (args.out_dir / "compact_speed_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {
        "settings": vars(args), "resolved_device": str(device),
        "timer": "perf_counter with CUDA synchronization around each block",
        "synthesis_convention": "legacy ortho-IFFT array units; not amplitude-preserving interpolation",
        "excludes": ["encoder", "latent rollout", "fit", "cache construction"],
        "actual_batch_size": len(batch), "rows": rows,
    }
    (args.out_dir / "compact_speed_audit.json").write_text(json.dumps(payload,indent=2,default=str)+"\n",encoding="utf-8")
    print(args.out_dir / "compact_speed_audit.csv")


if __name__ == "__main__": main()
