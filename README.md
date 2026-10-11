# case

CASE reliability fusion project: Python code and archived results for estimating
continuous valence and arousal from physiological signals when sensors are
missing or corrupted. CASE supplies 30 participants and eight channels: ECG,
BVP, GSR, respiration, skin temperature and three EMG channels.

Authors: Gabrielle Jeconiah Christiando, Darrel Immanuel, Dharma Vija Putra
Sugianto and Budi Juarto, Bina Nusantara University, Jakarta, Indonesia.

## Project layout

```text
src/, configs/, tests/   preprocessing, windowing, features, models, training,
                         statistics, validation and tests
run_all.py               original non-nested experiment and scientific reports
results/, manifests/     historical results and their provenance
revision_20261011/
  src/                   nested experiment and implementation validation
  analysis/              participant statistics and result verification scripts
  results/               combined and per-fold metrics, selected settings,
                         gate diagnostics and held-out predictions
  manifests/             folds, selection records, leakage and validation audits
  REPRODUCE.txt          archive checks and fresh retraining instructions
requirements.txt, environment.yml
DATASET_CARD.md, LICENSE_NOTES.md, REPRODUCIBILITY.md
```

Raw recordings, signal caches, checkpoints, run logs, manuscripts and conference
templates are excluded. Derived predictions and metrics are retained so the
archived analysis can be checked without retraining.

## Environment and data

The recorded environment used Python 3.13.5, PyTorch 2.11.0+cu128, CUDA 12.8 and
an NVIDIA RTX 3070 Ti Laptop GPU on Windows 11. Exact versions are in the
experiment manifests. Install the PyTorch build appropriate for your hardware,
then install the remaining dependencies:

```bash
git clone -c core.autocrlf=false https://github.com/SeedFlora/case.git
cd case
pip install -r requirements.txt
# Alternative: conda env create -f environment.yml
```

Obtain `CASE_full.zip` from the official
[CASE_Dataset-full record](https://doi.org/10.6084/m9.figshare.8869157.v1)
and extract it. The data record is CC0 1.0; the
[data descriptor](https://doi.org/10.1038/s41597-019-0209-0) is CC BY 4.0.
See [DATASET_CARD.md](DATASET_CARD.md) for details.

In `configs/data.yaml`, configure `archive`, `root`, `physiological_dir`,
`annotations_dir` and `metadata_dir` for your local extraction. The checked-in
paths point to `D:/case`; use the `data/interpolated/` variant. Dataset files are
read without modification.

## Run and verify

Run commands from the repository root. The tests need no raw dataset; tests
requiring unavailable caches or results are skipped:

```bash
python -m pytest tests -q
```

Check archived statistics and independently recompute participant metrics from
the saved nested predictions, without data caches or a GPU:

```bash
python revision_20261011/analysis/reanalyse_existing_results.py
python revision_20261011/analysis/reanalyse_existing_results.py --subject-metrics revision_20261011/results/subject_metrics.csv --out-dir revision_20261011/analysis/nested --analysis-origin retrospective
python revision_20261011/analysis/verify_predictions.py
```

These commands rewrite regenerated analysis and verification files. Preserve a
copy if you need an unchanged archive.

After configuring the data paths, run the original experiment:

```bash
python run_all.py
```

This rebuilds preprocessing, trains the original non-nested grid, computes
statistics and sanity checks, generates scientific reports and runs the tests.
It overwrites the original pipeline's outputs. Its fixed setting is a 20-second
window with zero annotation delay. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

For the nested experiment, rebuild all three window caches and use a fresh
`revision_rerun/` folder as described in
[revision_20261011/REPRODUCE.txt](revision_20261011/REPRODUCE.txt). Archived
completion markers must not be copied into a fresh training run.

## Protocol and interpretation

The nested experiment compares nine models using participant-grouped five-fold
cross-validation repeated with seeds 13, 42 and 2026: 15 outer folds, each with
18 training, six validation and six test participants. Each fold selects a
window of 5, 10 or 20 seconds and a delay of 0, 1, 2 or 3 seconds using only its
training and validation participants. Models include training mean, Ridge,
LightGBM, early fusion, equal-weight fusion, quality gating and matched dropout
and constant-quality controls. Modality dropout is 0.2.

The 55 conditions cover clean input, fixed missing-channel subsets, individual
channel removal, a wearable subset, and synthetic noise, flatline, stalls,
clipping and impulses. Per-fold results and predictions support verification of
participant metrics, matched targets/windows and fold integrity.

No significant quality-gate advantage over equal-weight fusion with dropout was
detected in any of the six condition families after participant pairing and
Holm correction. LightGBM had the highest mean CCC in each family. Failure to
detect a difference does not establish equivalence. Full model, condition and
target statistics are in `revision_20261011/analysis/nested/`.

This is retrospective analysis of participants used during earlier development.
Architectures, loss weights and dropout were fixed before nested reanalysis; no
record of the stated earlier loss-weight sweep was archived. Pointwise
participant-bootstrap intervals condition on fitted folds and do not account
for all training or cross-validation dependence. Every participant viewed the
same clips, so evaluation does not establish generalization to unseen stimuli.
Faults are synthetic, and self-reported affect is not a clinical measure.

## License

No license has been chosen for this project's code, results or documentation.
The authors retain all rights; reuse, modification or redistribution requires
their permission. CASE data retain their separate terms. See
[LICENSE_NOTES.md](LICENSE_NOTES.md). Contact: Budi Juarto,
budi.juarto@binus.ac.id.
