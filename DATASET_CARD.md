# Dataset Card: CASE (Continuously Annotated Signals of Emotion)

The statements below were checked against `CASE_full.zip`: its top-level
`README.txt`, the `metadata/` folder and the data files inside the archive. The
archive itself is not part of this repository (see Source). The evidence that
is distributed here:

- `manifests/dataset_inventory.json`: the extracted files with sizes, plus
  SHA-256 hashes for the README, metadata and script files and for three
  participants' interpolated CSV files.
- `manifests/subject_scan.csv`: row counts, time spans and emotional-stimulus
  duration for every participant.
- `manifests/preprocess_manifest.json` and `manifests/validation_summary.json`:
  preprocessing records and signal plausibility checks for all 240 trials.

## Source

- Data descriptor: Sharma, K., Castellini, C., van den Broek, E. L.,
  Albu-Schaeffer, A., & Schwenker, F. (2019). *A dataset of continuous affect
  annotations and physiological signals for emotion analysis.* Scientific Data,
  6, 196. https://doi.org/10.1038/s41597-019-0209-0. The article is published
  under CC BY 4.0.
- Data record: figshare collection
  https://doi.org/10.6084/m9.figshare.c.4260668.v1 (published 2019-10-01). Its
  full data item, CASE_Dataset-full
  (https://doi.org/10.6084/m9.figshare.8869157.v1), contains `CASE_full.zip`
  (4,553,720,945 bytes, MD5 `5e679ce69d6602c5f8225f03c489df76`) and is
  released under **CC0 1.0**.
- Dataset authors' code: https://gitlab.com/karan-shr/case_dataset

## Composition (verified)

| Property | Value |
|---|---|
| Participants | 30 (`sub_1` ... `sub_30`), all with both physiological and annotation files |
| Physiological channels | 8: `ecg`, `bvp`, `gsr`, `rsp`, `skt`, `emg_zygo`, `emg_coru`, `emg_trap` |
| Physiological sampling rate | 1000 Hz (measured `daqtime` step = 1 ms) |
| Annotation channels | `valence`, `arousal` |
| Annotation sampling rate | 20 Hz (measured `jstime` step = 50 ms) |
| Annotation scale | [0.5, 9.5], midpoint 5.0 |
| Rows per participant | 2,451,584 physiological / 49,032 annotation (identical for all 30) |
| Full session duration | 2451.6 s per participant |
| Non-finite samples | 0 in all 240 emotional-stimulus segments |

## Stimuli

Eight emotional videos (IDs 1-8), each viewed by every participant in a
pseudo-random, participant-specific order, interleaved with 120 s blue screens
(ID 11), preceded by a start video (ID 10) and followed by an end video (ID 12).

| ID | Label | Intended category |
|---|---|---|
| 1, 2 | amusing-1, amusing-2 | amusing |
| 3, 4 | boring-1, boring-2 | boring |
| 5, 6 | relaxed-1, relaxed-2 | relaxing |
| 7, 8 | scary-1, scary-2 | scary |

Only IDs 1-8 are used for modeling. Emotional-stimulus duration is **1270.1 s
per participant** (identical across participants), giving 30 x 8 = **240
trials**.

The stimulus videos are not distributed with the dataset; the CASE `README.txt`
states that they cannot be shared because of copyright restrictions. They are
not needed: the dataset authors embedded per-sample video IDs during their own
preprocessing.

## Variant used

The **interpolated** variant. The CASE `README.txt` recommends it for downstream
analysis because per-video sample counts are consistent across participants,
which is what makes a fixed windowing grid comparable across the cohort.
Interpolation is linear (`interp1`) onto the common 2451583.333 ms session grid,
performed by the dataset authors' script `scripts/s03_v2_interTransformData.m`
inside `CASE_full.zip`.

## Demographics

`metadata/participants.xlsx` inside `CASE_full.zip` records participant ID, sex,
age group, and the video sequence used. The cohort is balanced by sex (15
female, 15 male). Demographic attributes are **not** used as model inputs.

## Intended use in this project

Subject-independent continuous regression of valence and arousal from
physiological signals, plus a robustness benchmark under missing and corrupted
sensors. Participant ID, file name, recording ID and stimulus ID are never used
as predictive features.

## Known characteristics and caveats

- **Laboratory video-elicitation protocol.** Findings do not transfer
  automatically to in-the-wild or clinical settings.
- **Shared stimulus set.** Every participant sees the same 8 clips, so
  subject-independent evaluation does *not* demonstrate generalization to
  unseen stimuli.
- **Continuous affect is not a clinical or diagnostic measure**, and is not
  equivalent to "happiness". The targets are self-reported joystick positions.
- **Participant 21** shows a stable peripheral skin temperature near 20 C across
  all trials (per-trial medians in `manifests/validation_summary.json`). This is
  consistent with vasoconstriction or cold hands rather than sensor failure (it
  is stable, not a dropout or spike), and the participant is retained; the
  plausibility bound that admits it is documented in `src/validate_data.py`.
- **Annotation is a motor act.** Joystick reports lag the underlying affective
  response by an unknown, participant-specific amount. The original non-nested
  pipeline applied no shift, because a 0-3 s delay sweep on validation
  participants found no stable improvement
  (`manifests/sensitivity_selection.json`). The nested protocol reported in the
  paper instead selects a delay of 0, 1, 2 or 3 s, together with a window length
  of 5, 10 or 20 s, inside each outer fold, using only that fold's training and
  validation participants. The selected settings are in
  `revision_20261011/results/selected_settings.csv`: delays of 0 s in 5 folds,
  1 s in 3, 2 s in 5 and 3 s in 2; windows of 20 s in 14 folds and 5 s in 1.
