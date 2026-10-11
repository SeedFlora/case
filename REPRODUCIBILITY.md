# Reproducibility

This file describes the original non-nested pipeline. The nested experiment and
archive checks are documented in
[revision_20261011/REPRODUCE.txt](revision_20261011/REPRODUCE.txt).

## Setup and execution

Use Python 3.13 and install `requirements.txt`, or create the environment from
`environment.yml`. The reference run used PyTorch 2.11.0+cu128 and CUDA 12.8;
choose a PyTorch build suitable for your hardware. Configure all dataset paths
in `configs/data.yaml` for the extracted CASE interpolated variant. Saved paths
refer to `D:/case` on the original computer.

From the repository root:

```bash
python run_all.py
```

The driver performs discovery, preprocessing and validation, 20-second window
generation with zero delay, training, participant statistics, sanity checks,
scientific reports and tests. Outputs overwrite the original pipeline's
`results/`, `manifests/` and `reports/`; use a separate checkout to preserve the
archive. Raw dataset files are read without modification.

The sensitivity choice is frozen by default. To repeat the optional sweep:

```bash
python run_all.py --no-skip-sensitivity
```

Skipping training and preprocessing still requires the data for discovery and
existing signal/window caches for sanity checks:

```bash
python run_all.py --skip-train --skip-preprocess
```

For archive checks without data or caches, use the nested reproduction guide.

## Splits and operation order

Seeds 13, 42 and 2026 each produce five participant-grouped folds, giving 15
outer folds. `set_all_seeds` seeds Python, NumPy and PyTorch, including CUDA,
and requests deterministic cuDNN. Corruption uses fixed seed 777 so models
receive matched corrupted inputs. The condition catalog is saved in
`manifests/corruption_manifest_main.json`.

1. Split participants before fitting scalers or models.
2. Fit scalers, feature scalers, Ridge alpha and early stopping using only the
   fold's training and validation participants.
3. Corrupt a copy of raw test signals before scaling.
4. Recompute quality indicators from corrupted signals.
5. Inverse-transform predictions before computing MAE and RMSE.

## Recorded evidence and checks

`results/experiment_manifest.json` records environment versions, hardware,
configuration, participant membership, scaler states, training scores, epochs,
timings and model sizes. `manifests/dataset_inventory.json` records dataset
file sizes and selected SHA-256 hashes. The nested driver uses the original
experiment manifest to verify its exact outer folds; keep that file available.

After a completed original run, inspect:

- `manifests/leakage_audit_main.json`: `passed` should be true.
- `results/sanity_checks.json`: `all_passed` should be true.
- `reports/leakage_audit.md`, `experiment_report.md` and `limitations.md`.

Run the tests with:

```bash
python -m pytest tests -q
```

Tests needing absent caches or saved metrics are skipped. Passing these tests
alone does not validate a complete trained experiment.

## Reproduction limits

GPU kernels and mixed precision can prevent bitwise identical results across
hardware, drivers and library versions. Deterministic cuDNN is requested, and
LightGBM is single-threaded per model with fixed configuration. The archived
environment documents the run rather than guaranteeing availability on every
platform. `.gitattributes` disables line-ending conversion to preserve recorded
source hashes.

On the reference RTX 3070 Ti Laptop GPU, the original full training grid took
about one hour; preprocessing and checks added several minutes. Signal caches
can be regenerated and are excluded from Git along with checkpoints and logs.
