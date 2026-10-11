# Licensing and redistribution notes

## CASE dataset

This project uses CASE (Continuously Annotated Signals of Emotion). Native
recordings and stimulus videos are excluded from this repository.

- Official collection: https://doi.org/10.6084/m9.figshare.c.4260668.v1.
  Its full item, [CASE_Dataset-full](https://doi.org/10.6084/m9.figshare.8869157.v1),
  provides `CASE_full.zip` (4,553,720,945 bytes; MD5
  `5e679ce69d6602c5f8225f03c489df76`) under CC0 1.0.
- Data descriptor: Sharma, K., Castellini, C., van den Broek, E. L.,
  Albu-Schaeffer, A., and Schwenker, F. (2019), *A dataset of continuous affect
  annotations and physiological signals for emotion analysis*, Scientific Data
  6:196, https://doi.org/10.1038/s41597-019-0209-0. CC BY 4.0 applies to this
  article, separately from the data record.
- Dataset authors' code: https://gitlab.com/karan-shr/case_dataset.

CC0 does not require attribution, but please cite both the data descriptor and
data record in scholarly work. The CASE README states that stimulus videos
cannot be shared because of copyright restrictions. This project uses embedded
per-sample video IDs and does not need the videos.

## Derived outputs and local data

Archived outputs include window-level targets and held-out predictions,
participant metrics, statistics and manifests. Native recordings, signal
caches and model checkpoints are excluded.

Code reads extracted dataset files without modifying them. The original
pipeline writes under `cache/`, `checkpoints/`, `logs/`, `results/`, `reports/`
and `manifests/`. Nested training and analysis write under their revision
directory or the fresh sibling `revision_rerun/` described in
`revision_20261011/REPRODUCE.txt`; they read root signal/window caches.

## Project code, results and documentation

No license has been chosen yet. The authors retain all rights; reuse,
modification or redistribution of the project's code, results or documentation
requires their permission. This does not change the separate terms of the
official CASE data or third-party dependencies.

Contact: Budi Juarto, budi.juarto@binus.ac.id.
