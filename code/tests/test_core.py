from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from run_icassp2027_adaptive_readout_audit import (  # noqa: E402
    coefficient_indices,
    conformal_quantile,
)
from run_icassp2027_full_field_conformal import physics_scores  # noqa: E402
from run_icassp2027_readout_audit import fit_ridge, synthesize  # noqa: E402


class CoreReproducibilityTests(unittest.TestCase):
    def test_reference_grid_synthesis_preserves_amplitude(self) -> None:
        n0 = 128
        modes = 4

        def field(n: int) -> torch.Tensor:
            x = torch.arange(n, dtype=torch.float64) / n
            return torch.stack(
                [
                    1.3 + 0.7 * torch.cos(4 * math.pi * x) - 0.4 * torch.sin(6 * math.pi * x),
                    -0.8 + 0.2 * torch.sin(2 * math.pi * x),
                ]
            )[None]

        spectrum = torch.fft.rfft(field(n0), norm="ortho")
        packed = torch.cat(
            [spectrum[..., : modes + 1].real, spectrum[..., 1 : modes + 1].imag], dim=-1
        ).flatten(1)
        for n in (64, 128, 256, 512):
            actual = synthesize(packed, n, reference_grid_size=n0)
            self.assertTrue(torch.allclose(actual, field(n), atol=1e-12, rtol=1e-12))

    def test_ridge_solution_satisfies_penalized_normal_equations(self) -> None:
        generator = torch.Generator().manual_seed(2027)
        x = torch.randn(40, 5, generator=generator)
        y = torch.randn(40, 10, generator=generator)
        ridge = 0.01
        weight, bias = fit_ridge(x, y, ridge)
        design = torch.cat([x.double(), torch.ones(40, 1)], dim=1)
        solution = torch.cat([weight.double(), bias.double()[None]], dim=0)
        residual = design.T @ (design @ solution - y.double()) + ridge * solution
        self.assertLess(float(residual.abs().max()), 1e-5)

    def test_coefficient_indices_preserve_channel_blocks(self) -> None:
        actual = coefficient_indices(4, 2, torch.device("cpu"))
        expected = torch.tensor([0, 1, 2, 5, 6, 9, 10, 11, 14, 15])
        self.assertTrue(torch.equal(actual, expected))

    def test_conformal_rank_and_small_sample_guard(self) -> None:
        self.assertEqual(conformal_quantile(torch.arange(100.0), 0.05), 95.0)
        with self.assertRaises(ValueError):
            conformal_quantile(torch.arange(8.0), 0.10)

    def test_fixed_grid_fusion_matches_two_stage_decode(self) -> None:
        generator = torch.Generator().manual_seed(17)
        latent = torch.randn(12, 6, generator=generator)
        coefficient_weight = torch.randn(6, 18, generator=generator)
        coefficient_bias = torch.randn(18, generator=generator)
        grid = 64
        basis = synthesize(torch.eye(18), grid).reshape(18, -1)
        two_stage = synthesize(latent @ coefficient_weight + coefficient_bias, grid)
        fused = (
            latent @ (coefficient_weight @ basis) + coefficient_bias @ basis
        ).reshape(-1, 2, grid)
        self.assertTrue(torch.allclose(two_stage, fused, atol=2e-5, rtol=2e-5))

    def test_physics_scores_vanish_for_exact_prediction(self) -> None:
        generator = torch.Generator().manual_seed(31)
        target = torch.randn(5, 7, 2, 32, generator=generator)
        joint, state, energy = physics_scores(target.clone(), target, beta=1.0)
        self.assertTrue(torch.equal(joint, torch.zeros_like(joint)))
        self.assertTrue(torch.equal(state, torch.zeros_like(state)))
        self.assertTrue(torch.equal(energy, torch.zeros_like(energy)))


if __name__ == "__main__":
    unittest.main()
