"""Phase 3: label-free signal-quality indicators.

Every indicator is WINDOW-LOCAL: it depends only on the samples inside the
window being scored, never on dataset statistics and never on labels. That
matters for two reasons:

  * leakage -- a quality feature computed from global statistics would leak
    information across the train/test boundary;
  * deployability -- the reliability gate must be computable at inference time
    from a single incoming window, with no access to a corpus.

Returns a [N, C, 10] tensor of quality vectors q_m.
"""
from __future__ import annotations

import numpy as np

QUALITY_FEATURE_NAMES = [
    "finite_ratio",
    "flatline_ratio",
    "saturation_ratio",
    "log_variance",
    "mad",
    "mean_abs_derivative",
    "outlier_ratio",
    "hf_ratio",
    "spectral_entropy",
    "deriv_zero_crossing_rate",
]
N_QUALITY_FEATURES = len(QUALITY_FEATURE_NAMES)

_EPS = 1e-12


def compute_quality(x: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Quality vectors for windows `x` of shape [N, C, T] -> [N, C, 10]."""
    if x.ndim != 3:
        raise ValueError(f"expected [N, C, T], got {x.shape}")
    n, c, t = x.shape
    out = np.zeros((n, c, N_QUALITY_FEATURES), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = _quality_chunk(x[s:e].astype(np.float64))
    return out


def _quality_chunk(x: np.ndarray) -> np.ndarray:
    n, c, t = x.shape
    finite = np.isfinite(x)
    finite_ratio = finite.mean(axis=-1)

    # Work on a NaN-free copy so the remaining statistics are well defined.
    xf = np.where(finite, x, 0.0)

    d = np.diff(xf, axis=-1)
    absd = np.abs(d)

    flatline_ratio = (absd < 1e-8).mean(axis=-1)

    xmin = xf.min(axis=-1, keepdims=True)
    xmax = xf.max(axis=-1, keepdims=True)
    rng = np.maximum(xmax - xmin, _EPS)
    near_lo = (xf - xmin) / rng < 1e-3
    near_hi = (xmax - xf) / rng < 1e-3
    saturation_ratio = (near_lo | near_hi).mean(axis=-1)

    var = xf.var(axis=-1)
    log_variance = np.log1p(var)

    med = np.median(xf, axis=-1, keepdims=True)
    mad = np.median(np.abs(xf - med), axis=-1)

    mean_abs_derivative = absd.mean(axis=-1)

    robust_sigma = np.maximum(1.4826 * mad, _EPS)[..., None]
    outlier_ratio = (np.abs(xf - med) / robust_sigma > 4.0).mean(axis=-1)

    # Spectral indicators
    xc = xf - xf.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(xc, axis=-1)) ** 2
    total = spec.sum(axis=-1) + _EPS
    n_bins = spec.shape[-1]
    hf_start = int(0.4 * n_bins)
    hf_ratio = spec[..., hf_start:].sum(axis=-1) / total

    p = spec / total[..., None]
    p = np.clip(p, _EPS, 1.0)
    spectral_entropy = (-(p * np.log(p)).sum(axis=-1)) / np.log(n_bins)

    sign_changes = np.diff(np.signbit(d).astype(np.int8), axis=-1) != 0
    deriv_zcr = sign_changes.mean(axis=-1)

    feats = np.stack([
        finite_ratio, flatline_ratio, saturation_ratio, log_variance, mad,
        mean_abs_derivative, outlier_ratio, hf_ratio, spectral_entropy, deriv_zcr,
    ], axis=-1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def invalid_fraction(x: np.ndarray) -> np.ndarray:
    """Fraction of samples per window/channel that are non-finite or flatlined."""
    finite = np.isfinite(x)
    d = np.abs(np.diff(np.where(finite, x, 0.0), axis=-1))
    flat = np.concatenate([d < 1e-12, d[..., -1:] < 1e-12], axis=-1)
    return (~finite | flat).mean(axis=-1).astype(np.float32)
