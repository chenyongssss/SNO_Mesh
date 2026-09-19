# ICASSP 2027 Reproducibility Package

This directory contains the code-level reproducibility package for
**Efficient Grid-Query Spectral Readout for Symplectic Latent PDE Forecasting**.
It isolates the evaluation procedures used for the paper from the larger
research workspace.

The package reproduces four operations on frozen latent trajectories:

1. fit and compare affine and nonlinear spectral readouts;
2. select a Fourier cutoff on an independent pilot split;
3. benchmark ordinary and fixed-grid-fused readouts; and
4. calibrate joint state/energy predictive sets on native-grid FPUT fields.

The code does not claim to reproduce GPU timings on different hardware, and it
does not treat coefficient synthesis at a new grid size as physical resolution
generalization. See `SCOPE.md` for the exact claim boundaries.

## Directory Layout

```text
reproducibility/
|-- README.md
|-- SCOPE.md
|-- CITATION.cff
|-- requirements.txt
|-- code/
|   |-- scripts/
|   |   |-- run_icassp2027_readout_audit.py
|   |   |-- run_icassp2027_adaptive_readout_audit.py
|   |   |-- run_icassp2027_compact_speed_audit.py
|   |   `-- run_icassp2027_full_field_conformal.py
|   `-- tests/
|       `-- test_core.py
|-- data/
|   `-- README.md
`-- outputs/
    `-- .gitignore
```

## Environment

Python 3.10 or newer is recommended. The archived results were produced with
PyTorch/CUDA; the mathematical smoke tests also run on CPU.

```bash
python3 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install -r reproducibility/requirements.txt
```

## Quick Verification

Run this from the repository root:

```bash
python -m unittest discover -s reproducibility/code/tests -v
```

The tests verify Fourier sign and amplitude conventions, ridge normal
equations, cutoff indexing, finite conformal ranks, exact fixed-grid fusion,
and FPUT state/energy score behavior. They use synthetic data and require no
download.

The readout script can also run end-to-end with synthetic tensors:

```bash
python reproducibility/code/scripts/run_icassp2027_readout_audit.py \
  --device cpu \
  --input-dim 18 \
  --output-dim 18 \
  --fit-samples 256 \
  --eval-samples 128 \
  --mlp-epochs 10 \
  --warmups 1 \
  --blocks 2 \
  --repeats 2
```

This is an implementation check, not a reproduction of the paper's numerical
tables.

## Reproducing Archived FPUT Audits

Obtain the separately archived tensor bundle described in `data/README.md` and
place it under `reproducibility/data/fput_seed<SEED>/`. For each seed in
`11, 43, 53`, run the following commands after replacing `<SEED>`:

```bash
python reproducibility/code/scripts/run_icassp2027_adaptive_readout_audit.py \
  --fit-latents reproducibility/data/fput_seed<SEED>/fit_latents.pt \
  --fit-targets reproducibility/data/fput_seed<SEED>/fit_targets.pt \
  --calibration-latents reproducibility/data/fput_seed<SEED>/pilot_latents.pt \
  --calibration-targets reproducibility/data/fput_seed<SEED>/pilot_targets.pt \
  --eval-latents reproducibility/data/fput_seed<SEED>/eval_latents.pt \
  --eval-targets reproducibility/data/fput_seed<SEED>/eval_targets.pt \
  --levels 4,8,12,16,20,24,32,48,63 \
  --device cuda \
  --out-dir reproducibility/outputs/fput_seed<SEED>/pilot
```

```bash
python reproducibility/code/scripts/run_icassp2027_compact_speed_audit.py \
  --fit-latents reproducibility/data/fput_seed<SEED>/fit_latents.pt \
  --fit-targets reproducibility/data/fput_seed<SEED>/fit_targets.pt \
  --eval-latents reproducibility/data/fput_seed<SEED>/eval_latents.pt \
  --eval-targets reproducibility/data/fput_seed<SEED>/eval_targets.pt \
  --modes 4 \
  --device cuda \
  --out-dir reproducibility/outputs/fput_seed<SEED>/timing
```

```bash
python reproducibility/code/scripts/run_icassp2027_full_field_conformal.py \
  --fit-latents reproducibility/data/fput_seed<SEED>/fit_latents.pt \
  --fit-targets reproducibility/data/fput_seed<SEED>/fit_targets.pt \
  --calibration-bundle reproducibility/data/fput_seed<SEED>/calibration_bundle.pt \
  --eval-bundle reproducibility/data/fput_seed<SEED>/eval_bundle.pt \
  --fixed-modes 4 \
  --device cuda \
  --out-dir reproducibility/outputs/fput_seed<SEED>/conformal
```

The paper's CUDA latency ratios are hardware- and software-dependent. Report
the GPU model, PyTorch/CUDA versions, batch size, warmups, blocks, and repeats
when producing new timing results.

## From-Training Reproduction

The compact package begins from frozen latent trajectories so that the paper's
readout and calibration claims can be audited without retraining the upstream
model. The full research workspace retains the model and data-generation
pipeline under `experiments/`:

| Stage | Workspace entry point |
| --- | --- |
| FPUT data generation | `experiments/scripts/generate_fput_lattice.py` |
| Frozen-protocol FPUT validation | `experiments/scripts/write_stage5_fput_fresh_seed_frozen_protocol_validation.py` |
| Export frozen audit tensors | `experiments/scripts/export_icassp2027_fput_tensors.py` |
| Materialize native-field bundles | `experiments/scripts/materialize_icassp2027_field_bundles.py` |
| NLS matched causal audit | `experiments/scripts/write_stage4_nls_matched_causal_ablation.py` |
| Manuscript evidence verification | `experiments/scripts/verify_icassp2027_revision.py` |

These upstream scripts require archived checkpoints and datasets and are not
duplicated here. A public release should attach those files as a versioned data
archive and record its URL and SHA-256 manifest in `data/README.md`.

## Outputs

Each audit writes CSV and JSON machine-readable results. Some scripts also
write a Markdown summary. Generated outputs are ignored by Git except for the
placeholder file.

## Citation and License

Citation metadata, including author ORCIDs, is provided in `CITATION.cff`.
Replace its repository URL and add the paper DOI after publication. No reuse
license has yet been selected for this standalone package; choose and add one
before the public release.
