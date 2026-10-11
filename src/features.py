"""Phase 6 (Baseline 1): compact time- and frequency-domain features per sensor.

Deliberately generic and cheap: the point of the classical baseline is to show
what a well-regularised model can do on standard descriptors, not to hand-craft
a state-of-the-art feature set. Features are computed per window and per
channel, then flattened, so a missing modality can be zeroed out consistently
with the neural models.
"""
from __future__ import annotations

from typing import List

import numpy as np

FEATURE_NAMES = [
    "mean", "std", "min", "max", "median", "iqr", "skew", "kurtosis",
    "rms", "mean_abs_diff", "ptp", "energy_log",
    "bandpow_low", "bandpow_mid", "bandpow_high", "spec_centroid",
]
N_FEATURES = len(FEATURE_NAMES)
_EPS = 1e-12


def extract(x: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Features for windows `x` [N, C, T] -> [N, C*F] (channel-major)."""
    n, c, t = x.shape
    out = np.zeros((n, c, N_FEATURES), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = _chunk(x[s:e].astype(np.float64))
    return out.reshape(n, c * N_FEATURES)


def feature_columns(modalities: List[str]) -> List[str]:
    return [f"{m}__{f}" for m in modalities for f in FEATURE_NAMES]


def _chunk(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    mn = x.min(axis=-1)
    mx = x.max(axis=-1)
    med = np.median(x, axis=-1)
    q25, q75 = np.percentile(x, [25, 75], axis=-1)
    iqr = q75 - q25

    xc = x - mean[..., None]
    sd = np.maximum(std, _EPS)
    skew = (xc ** 3).mean(axis=-1) / sd ** 3
    kurt = (xc ** 4).mean(axis=-1) / sd ** 4 - 3.0

    rms = np.sqrt((x ** 2).mean(axis=-1))
    mad_ = np.abs(np.diff(x, axis=-1)).mean(axis=-1)
    ptp = mx - mn
    energy_log = np.log1p((x ** 2).sum(axis=-1))

    spec = np.abs(np.fft.rfft(xc, axis=-1)) ** 2
    total = spec.sum(axis=-1) + _EPS
    nb = spec.shape[-1]
    a, b = int(0.1 * nb), int(0.4 * nb)
    low = spec[..., :a].sum(axis=-1) / total
    mid = spec[..., a:b].sum(axis=-1) / total
    high = spec[..., b:].sum(axis=-1) / total
    freqs = np.arange(nb)
    centroid = (spec * freqs).sum(axis=-1) / total

    feats = np.stack([mean, std, mn, mx, med, iqr, skew, kurt, rms, mad_, ptp,
                      energy_log, low, mid, high, centroid], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def apply_modality_mask(feats: np.ndarray, mask: np.ndarray, n_modalities: int) -> np.ndarray:
    """Zero out feature blocks of absent modalities. feats [N, C*F], mask [N, C]."""
    f = feats.reshape(feats.shape[0], n_modalities, -1).copy()
    f[~mask] = 0.0
    return f.reshape(feats.shape[0], -1)
