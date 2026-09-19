# Data Bundle

Large tensors, checkpoints, and generated trajectories are intentionally not
stored in Git. A complete numerical reproduction requires a versioned external
archive containing one directory per model seed.

## Expected Files Per Seed

```text
fput_seed<SEED>/
|-- fit_latents.pt
|-- fit_targets.pt
|-- pilot_latents.pt
|-- pilot_targets.pt
|-- eval_latents.pt
|-- eval_targets.pt
|-- calibration_bundle.pt
|-- eval_bundle.pt
`-- manifest.json
```

The paper uses seeds `11`, `43`, and `53` for the cutoff-4 timing and
native-field calibration audits. The separate cutoff-24 comparison uses model
seeds `11`, `23`, and `37` and should be published as a separately named
protocol bundle.

## Tensor Contract

- `*_latents.pt`: floating tensor `[trajectory, time, latent_feature]`.
- `*_targets.pt`: floating tensor `[trajectory, time, 2 * (2K + 1)]`.
- Each channel packs `[Re F_0, ..., Re F_K, Im F_1, ..., Im F_K]` from
  `torch.fft.rfft(..., norm="ortho")`; Nyquist is excluded.
- `calibration_bundle.pt` and `eval_bundle.pt`: dictionaries keyed by condition.
  Each value contains `latents`, `fields`, integer `grid_size`, and numeric
  `beta`. Evaluation values additionally contain boolean `coverage_condition`.
- Native fields have shape `[trajectory, time, 2, grid_size]`, ordered as FPUT
  displacement and momentum.

PyTorch files are pickle-based. Load only bundles obtained from the official
project archive or another trusted source.

## Release Checklist

Before publishing the repository, replace this paragraph with:

- the immutable archive URL or DOI;
- archive version and creation date;
- SHA-256 for every file;
- license and redistribution terms;
- compressed and extracted sizes; and
- the exact script/commit used to create the bundle.

Do not upload the current workspace's entire `artifacts/` directory. It
contains redundant intermediate experiments and is substantially larger than
the minimal evidence bundle.
