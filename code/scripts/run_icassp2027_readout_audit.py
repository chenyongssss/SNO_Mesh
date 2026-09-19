#!/usr/bin/env python3
"""Benchmark queryable readouts on identical frozen latent states.

This utility is deliberately independent of training. It fits the proposed
shared Fourier readout on a declared fit split, evaluates all readouts on a
disjoint split, and records repeated synchronized latency and CUDA memory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn


def fit_ridge(x: torch.Tensor, y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.double()
    y = y.double()
    design = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)
    gram = design.T @ design + ridge * torch.eye(design.shape[1], device=x.device, dtype=x.dtype)
    solution = torch.linalg.solve(gram, design.T @ y)
    return solution[:-1].float(), solution[-1].float()


class SharedSpectralReadout(nn.Module):
    def __init__(self, weights: torch.Tensor, bias: torch.Tensor, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)
        self.output_dim = output_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent @ self.weights + self.bias


class SharedSpectralMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def synthesize(
    coefficients: torch.Tensor,
    grid_size: int,
    *,
    reference_grid_size: int | None = None,
) -> torch.Tensor:
    """Synthesize [Re F_0..F_K, Im F_1..F_K] for each of two channels.

    Coefficients use rfft(norm='ortho'). With reference_grid_size=N0, scaling
    by sqrt(N/N0) preserves the trigonometric field's amplitude across grids.
    Both grids must resolve K strictly below Nyquist. None retains the legacy
    array audit: amplitude scales as 1/sqrt(N) and unresolved modes are dropped.
    That legacy convention is not physical resolution transfer.
    """
    if grid_size < 1 or not coefficients.is_floating_point():
        raise ValueError("synthesis requires a positive grid size and real floating coefficients")
    if coefficients.shape[-1] % 2:
        raise ValueError("Fourier target dimension must contain paired q/p coefficient vectors")
    per_field = coefficients.shape[-1] // 2
    modes = per_field // 2
    expected = 2 * modes + 1
    if per_field != expected:
        raise ValueError("Each field must use [real modes 0..K, imaginary modes 1..K]")
    if reference_grid_size is not None and (
        reference_grid_size <= 2 * modes or grid_size <= 2 * modes
    ):
        raise ValueError("amplitude-preserving synthesis requires N0 > 2K and N > 2K")
    state = coefficients.reshape(*coefficients.shape[:-1], 2, per_field)
    kept = min(modes + 1, grid_size // 2 + 1)
    spectrum = torch.zeros(
        *state.shape[:-1],
        grid_size // 2 + 1,
        device=state.device,
        dtype=torch.complex128 if coefficients.dtype == torch.float64 else torch.complex64,
    )
    real = state[..., :kept]
    imag = state[..., modes + 1 : modes + kept]
    spectrum[..., :kept] = torch.complex(real, torch.zeros_like(real))
    if kept > 1:
        spectrum[..., 1:kept] = torch.complex(real[..., 1:kept], imag)
    if reference_grid_size is not None:
        spectrum = spectrum * (grid_size / reference_grid_size) ** 0.5
    return torch.fft.irfft(spectrum, n=grid_size, dim=-1, norm="ortho")


def train_mlp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()


def select_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int],
    x: torch.Tensor,
    y: torch.Tensor,
    max_epochs: int,
    learning_rate: float,
    validation_fraction: float,
    patience: int,
    seed: int,
) -> tuple[SharedSpectralMLP, int, int]:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    order = torch.randperm(x.shape[0], generator=generator, device=x.device)
    validation_count = max(1, int(x.shape[0] * validation_fraction))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    best_choice = None
    for hidden_dim in hidden_dims:
        torch.manual_seed(seed + hidden_dim)
        candidate = SharedSpectralMLP(input_dim, output_dim, hidden_dim).to(x.device)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=learning_rate)
        best_validation = float("inf")
        best_epoch = 1
        stale = 0
        for epoch in range(1, max_epochs + 1):
            candidate.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((candidate(x[train_indices]) - y[train_indices]) ** 2)
            loss.backward()
            optimizer.step()
            candidate.eval()
            with torch.no_grad():
                validation = torch.mean((candidate(x[validation_indices]) - y[validation_indices]) ** 2).item()
            if validation < best_validation - 1e-8:
                best_validation = validation
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        choice = (best_validation, hidden_dim, best_epoch)
        if best_choice is None or choice < best_choice:
            best_choice = choice
    if best_choice is None:
        raise RuntimeError("no MLP candidate was trained")
    _, selected_hidden, selected_epochs = best_choice
    torch.manual_seed(seed + selected_hidden)
    selected = SharedSpectralMLP(input_dim, output_dim, selected_hidden).to(x.device)
    train_mlp(selected, x, y, selected_epochs, learning_rate)
    return selected, selected_hidden, selected_epochs


def model_size(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters()) + sum(value.numel() for value in model.buffers())


def synchronized(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device: torch.device, warmups: int, blocks: int, repeats: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    synchronized(device)
    samples = []
    peak = []
    for _ in range(blocks):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronized(device)
        start = perf_counter()
        for _ in range(repeats):
            fn()
        synchronized(device)
        samples.append((perf_counter() - start) / repeats)
        peak.append(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    values = torch.tensor(samples, dtype=torch.float64)
    return {
        "median_seconds": float(values.median()),
        "iqr_seconds": float(torch.quantile(values, 0.75) - torch.quantile(values, 0.25)),
        "peak_memory_bytes_median": float(torch.tensor(peak, dtype=torch.float64).median()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=66)
    parser.add_argument("--fit-samples", type=int, default=4096)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", default="64,128")
    parser.add_argument("--mlp-epochs", type=int, default=500)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-2)
    parser.add_argument("--mlp-validation-fraction", type=float, default=0.2)
    parser.add_argument("--mlp-patience", type=int, default=50)
    parser.add_argument("--grid-sizes", default="128,256")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fit-latents", type=Path, help="Optional .pt/.npy tensor with fit latents.")
    parser.add_argument("--eval-latents", type=Path, help="Optional .pt/.npy tensor with evaluation latents.")
    parser.add_argument("--fit-targets", type=Path, help="Optional .pt/.npy tensor with fit Fourier targets.")
    parser.add_argument("--eval-targets", type=Path, help="Optional .pt/.npy tensor with evaluation Fourier targets.")
    parser.add_argument("--out-dir", type=Path, default=Path("reproducibility/outputs/readout_audit"))
    return parser.parse_args()


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix == ".npy":
        import numpy as np
        return torch.from_numpy(np.load(path)).to(device=device, dtype=torch.float32)
    value = torch.load(path, map_location=device, weights_only=False)
    if isinstance(value, dict):
        for key in ("latent", "latents", "target", "targets", "x", "y"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must contain a tensor or a recognized tensor dictionary key")
    return value.to(device=device, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tensor_args = (args.fit_latents, args.eval_latents, args.fit_targets, args.eval_targets)
    if any(tensor_args) and not all(tensor_args):
        raise ValueError("--fit/eval-latents and --fit/eval-targets must be supplied as a complete set")
    if all(tensor_args):
        fit_x = load_tensor(args.fit_latents, device)
        eval_x = load_tensor(args.eval_latents, device)
        fit_y = load_tensor(args.fit_targets, device)
        eval_y = load_tensor(args.eval_targets, device)
        if fit_x.ndim != 2 or eval_x.ndim != 2 or fit_y.ndim != 2 or eval_y.ndim != 2:
            raise ValueError("saved audit tensors must have shape [samples, features]")
        if fit_x.shape[1] != args.input_dim or fit_y.shape[1] != args.output_dim:
            raise ValueError("saved tensor dimensions do not match --input-dim/--output-dim")
    else:
        fit_x = torch.randn(args.fit_samples, args.input_dim, device=device)
        eval_x = torch.randn(args.eval_samples, args.input_dim, device=device)
        target_w = torch.randn(args.input_dim, args.output_dim, device=device)
        fit_y = fit_x @ target_w + 0.02 * torch.randn(args.fit_samples, args.output_dim, device=device)
        eval_y = eval_x @ target_w + 0.02 * torch.randn(args.eval_samples, args.output_dim, device=device)
    ridge_w, ridge_b = fit_ridge(fit_x, fit_y, args.ridge)
    mlp, selected_hidden, selected_epochs = select_mlp(
        args.input_dim,
        args.output_dim,
        [int(item) for item in args.hidden_dims.split(",") if item.strip()],
        fit_x,
        fit_y,
        args.mlp_epochs,
        args.mlp_learning_rate,
        args.mlp_validation_fraction,
        args.mlp_patience,
        args.seed,
    )
    models = {
        "shared_spectral_ridge": SharedSpectralReadout(ridge_w, ridge_b, args.output_dim).to(device).eval(),
        "shared_spectral_mlp": mlp,
    }
    rows = []
    for grid_size in [int(item) for item in args.grid_sizes.split(",") if item.strip()]:
        fit_fields = synthesize(fit_y, grid_size).reshape(fit_y.shape[0], -1)
        eval_fields = synthesize(eval_y, grid_size)
        grid_w, grid_b = fit_ridge(fit_x, fit_fields, args.ridge)
        grid_model = SharedSpectralReadout(grid_w, grid_b, fit_fields.shape[1]).to(device).eval()
        for name, model in {**models, "per_grid_physical_ridge": grid_model}.items():
            with torch.no_grad():
                raw_prediction = model(eval_x)
                if name == "per_grid_physical_ridge":
                    prediction = raw_prediction.reshape_as(eval_fields)
                    coefficient_mse = float("nan")
                    field_mse = torch.mean((prediction - eval_fields) ** 2).item()
                    decode = lambda: model(eval_x[: args.batch_size]).reshape(-1, 2, grid_size)
                else:
                    coefficient_mse = torch.mean((raw_prediction - eval_y) ** 2).item()
                    field_mse = torch.mean((synthesize(raw_prediction, grid_size) - eval_fields) ** 2).item()
                    decode = lambda m=model: synthesize(m(eval_x[: args.batch_size]), grid_size)
                timing = measure(
                    decode,
                    device,
                    args.warmups,
                    args.blocks,
                    args.repeats,
                )
            rows.append(
                {
                    "readout": name,
                    "grid_size": grid_size,
                    "coefficient_mse": coefficient_mse,
                    "field_mse": field_mse,
                    "parameters": model_size(model),
                    "selected_hidden_dim": selected_hidden if name == "shared_spectral_mlp" else "",
                    "selected_epochs": selected_epochs if name == "shared_spectral_mlp" else "",
                    "device": str(device),
                    **timing,
                }
            )

    with (args.out_dir / "readout_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"settings": vars(args), "resolved_device": str(device), "rows": rows}
    (args.out_dir / "readout_audit.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(args.out_dir / "readout_audit.csv")


if __name__ == "__main__":
    main()
