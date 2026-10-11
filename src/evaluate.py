"""Phase 5: metrics and aggregation.

CCC (Lin's concordance correlation coefficient) is the primary metric because it
penalises both correlation errors and calibration errors -- a model that tracks
the shape of an affect trajectory but sits at the wrong level is penalised,
unlike Pearson r.

    CCC = 2*cov(x,y) / (var(x) + var(y) + (mean(x)-mean(y))^2)

Aggregation is PARTICIPANT-LEVEL by default: metrics are computed within each
held-out participant and then averaged across participants, so that participants
contributing more windows cannot dominate the reported score.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TARGETS = ("valence", "arousal")


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size < 2:
        return float("nan")

    vt, vp = y_true.var(), y_pred.var()
    mt, mp = y_true.mean(), y_pred.mean()
    denom = vt + vp + (mt - mp) ** 2
    if denom <= 0:
        # Both series constant: perfect agreement iff the constants coincide.
        return 1.0 if np.isclose(mt, mp) else 0.0
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    return float(2.0 * cov / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(d * d)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """All metrics for a [N, 2] prediction block, plus the selection score."""
    out: Dict[str, float] = {}
    for i, t in enumerate(TARGETS):
        out[f"ccc_{t}"] = ccc(y_true[:, i], y_pred[:, i])
        out[f"mae_{t}"] = mae(y_true[:, i], y_pred[:, i])
        out[f"rmse_{t}"] = rmse(y_true[:, i], y_pred[:, i])
        out[f"r_{t}"] = pearson(y_true[:, i], y_pred[:, i])
    out["mean_ccc"] = float(np.nanmean([out[f"ccc_{t}"] for t in TARGETS]))
    out["mean_mae"] = float(np.nanmean([out[f"mae_{t}"] for t in TARGETS]))
    out["mean_rmse"] = float(np.nanmean([out[f"rmse_{t}"] for t in TARGETS]))
    return out


def per_subject_metrics(subject_ids: np.ndarray, y_true: np.ndarray,
                        y_pred: np.ndarray) -> pd.DataFrame:
    """Metrics computed independently within every held-out participant."""
    rows: List[Dict[str, float]] = []
    for s in np.unique(subject_ids):
        m = subject_ids == s
        if m.sum() < 2:
            continue
        rec = {"subject_id": int(s), "n_windows": int(m.sum())}
        rec.update(metrics_block(y_true[m], y_pred[m]))
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_participant_level(df_subject: pd.DataFrame) -> Dict[str, float]:
    """Equal-weight average over participants (the primary aggregation)."""
    cols = [c for c in df_subject.columns if c not in ("subject_id", "n_windows")]
    return {c: float(df_subject[c].mean()) for c in cols}


def quadrant_labels(y: np.ndarray, midpoint: float = 5.0) -> np.ndarray:
    """Four-quadrant affect label from the documented scale midpoint.

    0: low valence / low arousal   1: low valence / high arousal
    2: high valence / low arousal  3: high valence / high arousal
    """
    hv = (y[:, 0] >= midpoint).astype(int)
    ha = (y[:, 1] >= midpoint).astype(int)
    return (hv * 2 + ha).astype(int)


def quadrant_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     midpoint: float = 5.0) -> Dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    t = quadrant_labels(y_true, midpoint)
    p = quadrant_labels(y_pred, midpoint)
    return {
        "quadrant_macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "quadrant_balanced_acc": float(balanced_accuracy_score(t, p)),
    }
