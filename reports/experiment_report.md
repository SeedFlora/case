# Experiment Report

All numbers below are read from `results/*.csv` and `results/experiment_manifest.json`.

## 1. Configuration

- Window: **20 s**, stride **5 s**, annotation delay **0 s** (frozen on validation participants; see `manifests/sensitivity_selection.json`)
- Analysis rate: **64 Hz**, **1280** samples per window per channel
- Windows: **6,810** kept, **0** excluded, from **240** trials
- Modalities: ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap
- Protocol: repeated grouped 5-fold CV, seeds [13, 42, 2026], **15 outer folds**
- Robustness conditions per fold: **55**
- Modality dropout: **0.2**
- Runtime: **72.6 min** on NVIDIA GeForce RTX 3070 Ti Laptop GPU

## 2. Clean performance (participant-averaged)

| Model | CCC valence | CCC arousal | mean CCC | MAE | RMSE |
|---|---|---|---|---|---|
| `naive` | 0.000 | 0.000 | 0.000 +/- 0.000 | 1.153 | 1.513 |
| `ridge` | 0.244 | 0.262 | 0.253 +/- 0.030 | 1.439 | 2.080 |
| `lightgbm` | 0.279 | 0.322 | 0.300 +/- 0.031 | 1.189 | 1.510 |
| `early_fusion` | 0.197 | 0.177 | 0.187 +/- 0.029 | 1.374 | 1.737 |
| `late_fusion` | 0.186 | 0.176 | 0.181 +/- 0.028 | 1.319 | 1.687 |
| `reliability_gated` | 0.183 | 0.179 | 0.181 +/- 0.027 | 1.241 | 1.597 |
| `abl_no_quality` | 0.187 | 0.177 | 0.182 +/- 0.027 | 1.234 | 1.586 |
| `abl_no_moddropout` | 0.178 | 0.164 | 0.171 +/- 0.028 | 1.297 | 1.672 |
| `abl_equal_weights` | 0.176 | 0.174 | 0.175 +/- 0.027 | 1.233 | 1.596 |

## 3. Robustness (participant-averaged mean CCC)

| Model | clean | 1 sensor missing | 2 missing | 4 missing | wearable subset only | corrupted | rel. drop @4 |
|---|---|---|---|---|---|---|---|
| `naive` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | nan% |
| `ridge` | 0.253 | 0.222 | 0.209 | 0.115 | 0.183 | 0.083 | 54.4% |
| `lightgbm` | 0.300 | 0.263 | 0.243 | 0.138 | 0.228 | 0.165 | 54.1% |
| `early_fusion` | 0.187 | 0.165 | 0.142 | 0.081 | 0.129 | 0.162 | 56.9% |
| `late_fusion` | 0.181 | 0.178 | 0.162 | 0.120 | 0.112 | 0.153 | 33.8% |
| `reliability_gated` | 0.181 | 0.180 | 0.165 | 0.131 | 0.114 | 0.149 | 27.4% |
| `abl_no_quality` | 0.182 | 0.180 | 0.164 | 0.132 | 0.121 | 0.152 | 27.3% |
| `abl_no_moddropout` | 0.171 | 0.167 | 0.155 | 0.113 | 0.112 | 0.138 | 34.2% |
| `abl_equal_weights` | 0.175 | 0.176 | 0.162 | 0.134 | 0.115 | 0.145 | 23.3% |

## 4. Per-modality ablation (which sensor matters most)

| Removed sensor | reliability-gated | best conventional |
|---|---|---|
| `bvp` | 0.176 | 0.166 |
| `ecg` | 0.150 | 0.175 |
| `emg_coru` | 0.180 | 0.172 |
| `emg_trap` | 0.189 | 0.181 |
| `emg_zygo` | 0.144 | 0.141 |
| `gsr` | 0.160 | 0.126 |
| `rsp` | 0.192 | 0.209 |
| `skt` | 0.195 | 0.181 |

## 5. Statistical tests

Confirmatory family (Holm-corrected across 6 predeclared conditions), paired at participant level (n = 30):

| Condition | baseline | proposed | diff | 95% CI | p | p (Holm) | r |
|---|---|---|---|---|---|---|---|
| clean | 0.187 | 0.181 | -0.006 | [-0.040, +0.029] | 0.7766 | 1.0000 | -0.062 |
| missing_random_1 | 0.165 | 0.180 | +0.015 | [-0.017, +0.047] | 0.4161 | 1.0000 | +0.174 |
| missing_random_2 | 0.142 | 0.165 | +0.023 | [-0.007, +0.053] | 0.2621 | 1.0000 | +0.239 |
| missing_random_4 | 0.081 | 0.131 | +0.051 | [+0.024, +0.079] | 0.0016 | 0.0095 | +0.639 |
| wearable | 0.129 | 0.114 | -0.015 | [-0.059, +0.030] | 0.5561 | 1.0000 | -0.127 |
| corrupt_all | 0.162 | 0.149 | -0.013 | [-0.043, +0.017] | 0.5028 | 1.0000 | -0.144 |

## 6. Sanity checks

| Check | Result | Detail |
|---|---|---|
| Permuted-label control collapses | PASS | test mean CCC = -0.0009 |
| Random-label control collapses | PASS | test mean CCC = +0.0010 |
| No suspiciously high performance | PASS | best clean mean CCC = 0.300 (threshold 0.85) |
| Scalers are fold-local | PASS | 15 distinct states |
| Test participants unseen in training | PASS | |

### Between- vs within-participant decomposition

A model that only learns 'this participant sits high on valence' would show a high between-participant CCC and a near-zero within-participant CCC. Both components are reported:

| Model | target | CCC (full) | CCC (within-participant) | CCC (between-participant) |
|---|---|---|---|---|
| `abl_equal_weights` | valence | 0.150 | 0.191 | -0.294 |
| `abl_equal_weights` | arousal | 0.217 | 0.216 | 0.224 |
| `abl_no_moddropout` | valence | 0.158 | 0.204 | -0.223 |
| `abl_no_moddropout` | arousal | 0.198 | 0.201 | 0.184 |
| `abl_no_quality` | valence | 0.166 | 0.211 | -0.330 |
| `abl_no_quality` | arousal | 0.223 | 0.217 | 0.258 |
| `early_fusion` | valence | 0.169 | 0.218 | -0.043 |
| `early_fusion` | arousal | 0.221 | 0.233 | 0.142 |
| `late_fusion` | valence | 0.150 | 0.213 | -0.373 |
| `late_fusion` | arousal | 0.208 | 0.216 | 0.160 |
| `lightgbm` | valence | 0.261 | 0.315 | -0.140 |
| `lightgbm` | arousal | 0.327 | 0.361 | 0.084 |
| `naive` | valence | -0.002 | -0.000 | -0.077 |
| `naive` | arousal | -0.008 | -0.000 | -0.155 |
| `reliability_gated` | valence | 0.159 | 0.206 | -0.353 |
| `reliability_gated` | arousal | 0.219 | 0.217 | 0.234 |
| `ridge` | valence | 0.072 | 0.088 | -0.123 |
| `ridge` | arousal | 0.079 | 0.091 | 0.017 |

### Pooled vs participant-averaged aggregation

| Model | pooled-window | participant-averaged | inflation |
|---|---|---|---|
| `abl_equal_weights` | 0.179 | 0.175 | +0.004 |
| `abl_no_moddropout` | 0.172 | 0.171 | +0.001 |
| `abl_no_quality` | 0.191 | 0.182 | +0.009 |
| `early_fusion` | 0.201 | 0.187 | +0.014 |
| `late_fusion` | 0.183 | 0.181 | +0.002 |
| `lightgbm` | 0.296 | 0.300 | -0.004 |
| `naive` | 0.000 | 0.000 | +0.000 |
| `reliability_gated` | 0.184 | 0.181 | +0.003 |
| `ridge` | 0.203 | 0.253 | -0.050 |

## 7. Window/delay sensitivity (validation participants only)

- Selected window: **20 s** - 20 s outperformed 10 s and 5 s at EVERY delay candidate: all four 20 s configurations scored above every 10 s and 5 s configuration in the 3-fold sweep. The ordering 20 s > 10 s > 5 s was consistent, so the effect is treated as real.
- Selected delay: **0 s** - In the 3-fold sweep the delay optimum was inconsistent across window lengths (best at 1 s for 5 s windows, 2 s for 10 s windows, 3 s for 20 s windows), the signature of noise-chasing rather than a consistent physiological lag. A confirmation run at 20 s using ALL FIVE validation folds shrank the delay-3 advantage from +0.027 to +0.008 mean CCC (0.2411 +/- 0.0567 versus 0.2334 +/- 0.0521), an order of magnitude smaller than the fold-to-fold standard deviation.
- The 3-fold argmax would have selected a 3 s delay. It was rejected after the 5-fold confirmation showed the advantage was not stable. This is reported in the manuscript rather than silently adopting the larger number.

## 8. Gates

| Gate | Status |
|---|---|
| G0 data | PASS |
| G1 leakage | PASS |
| G2 baselines | PASS |
| G3 proposed model | PASS |
| G4 robustness | PASS |
| G5 statistics | PASS |