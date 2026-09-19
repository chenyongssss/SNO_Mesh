# Reproduction Scope

## Directly Audited

- affine ridge fitting from frozen latent states to packed Fourier coefficients;
- matched-output spectral MLP comparison;
- real Fourier synthesis and amplitude-preserving reference-grid scaling;
- exact algebraic equivalence of spectral decoding and fixed-grid fusion;
- independent-pilot cutoff selection;
- native-grid FPUT state and energy scores; and
- split-conformal component quantiles with a union-bound joint guarantee.

## Requires Archived Inputs

- the numerical rows reported for model seeds 11, 43, and 53;
- GPU latency and allocator measurements;
- the NLS matched-causal comparison; and
- learned-inverse decoder comparisons.

These results depend on frozen latent trajectories, native physical fields,
trained checkpoints, or a CUDA device. The code package does not silently
replace missing inputs with synthetic data when file arguments are supplied.

## Not Claimed

- coefficient-array queries at different inverse-FFT lengths are not evidence
  of physical resolution transfer;
- the shared spectral readout is not claimed to be symplectic;
- a latent-core residual does not certify conditioning, whitening, or physical
  reconstruction;
- the eight-trajectory pilot maximum is not a conformal quantile;
- coverage outside the calibrated parameter/grid condition is not guaranteed;
- nominal coverage alone is not evidence of useful predictive-set sharpness;
  and
- timing ratios are not expected to be invariant across hardware or software.
