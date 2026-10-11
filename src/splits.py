"""Phase 4: subject-grouped splits and fold-local scaling.

The single most important property of this project: **all windows from one
participant live in exactly one outer partition**. Because windows overlap by
50%, a window-level random split would place two windows sharing 5 seconds of
signal on opposite sides of the train/test boundary and inflate scores. Grouping
by participant makes that structurally impossible.

Scalers are fitted on the training partition of each outer fold ONLY and applied
unchanged to validation and test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    seed: int
    fold: int
    train_subjects: Tuple[int, ...]
    val_subjects: Tuple[int, ...]
    test_subjects: Tuple[int, ...]

    @property
    def name(self) -> str:
        return f"s{self.seed}_f{self.fold}"


def make_folds(subjects: List[int], seeds: List[int], n_folds: int,
               val_fraction: float) -> List[Fold]:
    """Repeated grouped k-fold over PARTICIPANTS.

    For each seed the participant list is shuffled and partitioned into
    `n_folds` test blocks. Within each outer training partition a further
    participant-disjoint validation block is carved out for model selection.
    """
    folds: List[Fold] = []
    subs = np.array(sorted(subjects))
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = subs[rng.permutation(len(subs))]
        blocks = np.array_split(perm, n_folds)
        for k in range(n_folds):
            test = np.sort(blocks[k])
            trainval = np.sort(np.concatenate([blocks[j] for j in range(n_folds) if j != k]))
            # deterministic validation carve-out, seeded per (seed, fold)
            rng2 = np.random.RandomState(seed * 1000 + k)
            tv = trainval[rng2.permutation(len(trainval))]
            n_val = max(1, int(round(val_fraction * len(tv))))
            val = np.sort(tv[:n_val])
            train = np.sort(tv[n_val:])
            folds.append(Fold(seed, k, tuple(int(x) for x in train),
                              tuple(int(x) for x in val), tuple(int(x) for x in test)))
    return folds


def fold_masks(meta: pd.DataFrame, f: Fold) -> Dict[str, np.ndarray]:
    s = meta["subject_id"].to_numpy()
    return {
        "train": np.isin(s, f.train_subjects),
        "val": np.isin(s, f.val_subjects),
        "test": np.isin(s, f.test_subjects),
    }


# --------------------------------------------------------------------------- #
# fold-local scaling
# --------------------------------------------------------------------------- #
class RobustChannelScaler:
    """Per-channel robust scaling: (x - median) / IQR, fitted on train only.

    Operates on [N, C, T] signal tensors, pooling over windows and time within
    each channel, and on [N, C, F] quality tensors, pooling over windows.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "RobustChannelScaler":
        # pool every axis except the channel axis (axis=1)
        axes = tuple(i for i in range(x.ndim) if i != 1)
        flat = np.moveaxis(x, 1, 0).reshape(x.shape[1], -1)
        q25, med, q75 = np.percentile(flat, [25, 50, 75], axis=1)
        iqr = q75 - q25
        sd = flat.std(axis=1)
        # Same degenerate-feature guard as QualityScaler: never divide a
        # constant channel by an epsilon floor.
        self.center_ = med.astype(np.float64)
        self.scale_ = np.where(iqr > 1e-8, iqr, np.where(sd > 1e-8, sd, 1.0)).astype(np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("scaler not fitted")
        shape = [1] * x.ndim
        shape[1] = -1
        c = self.center_.reshape(shape)
        s = self.scale_.reshape(shape)
        return ((x - c) / s).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def state(self) -> Dict[str, Any]:
        return {"center": self.center_.tolist(), "scale": self.scale_.tolist()}


class QualityScaler(RobustChannelScaler):
    """Robust scaling of quality vectors, per (channel, feature) pair.

    Degenerate features need care. Several indicators are *perfectly constant*
    on clean CASE data (`finite_ratio` is 1.0 everywhere, `flatline_ratio` 0.0),
    so their IQR and standard deviation are both exactly zero. Falling back to a
    small epsilon floor would divide floating-point dust by 1e-6 and produce
    values of order 1e5, which saturates the gating MLP and destroys training.
    A constant feature carries no information and must map to ~0 instead.

    Note these indicators are NOT useless: they become informative precisely
    when a sensor fails, which is the regime this study stress-tests. The scaler
    is fitted on clean training data, so the corrupted test-time values land far
    from the training centre -- exactly the signal the gate needs.
    """

    CLIP = 10.0

    def fit(self, x: np.ndarray) -> "QualityScaler":
        # x: [N, C, F] -> statistics per (C, F)
        q25, med, q75 = np.percentile(x, [25, 50, 75], axis=0)
        iqr = q75 - q25
        sd = x.std(axis=0)
        # IQR where usable, else standard deviation, else unit scale.
        scale = np.where(iqr > 1e-8, iqr, np.where(sd > 1e-8, sd, 1.0))
        self.center_ = med.astype(np.float64)
        self.scale_ = scale.astype(np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.center_[None]) / self.scale_[None]
        # Bounded because these are gate INPUTS: an extreme corruption should
        # saturate the reliability signal, not detonate the network.
        return np.clip(z, -self.CLIP, self.CLIP).astype(np.float32)


class TargetScaler:
    """Affine target scaling fitted on training targets only.

    CCC is invariant to a shared affine transform of prediction and reference,
    so training in scaled space and reporting CCC there is equivalent; MAE and
    RMSE are always inverse-transformed back to the native [0.5, 9.5] scale.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        self.mean_ = y.mean(axis=0)
        self.std_ = np.maximum(y.std(axis=0), 1e-6)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return ((y - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return (y * self.std_ + self.mean_).astype(np.float32)

    def state(self) -> Dict[str, Any]:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


# --------------------------------------------------------------------------- #
# leakage audit
# --------------------------------------------------------------------------- #
def audit_folds(meta: pd.DataFrame, folds: List[Fold]) -> Dict[str, Any]:
    """Verify the properties the whole study depends on."""
    issues: List[str] = []
    per_fold: List[Dict[str, Any]] = []
    all_subjects = set(meta["subject_id"].unique().tolist())

    for f in folds:
        tr, va, te = set(f.train_subjects), set(f.val_subjects), set(f.test_subjects)

        if tr & te:
            issues.append(f"{f.name}: train/test participant overlap {sorted(tr & te)}")
        if tr & va:
            issues.append(f"{f.name}: train/val participant overlap {sorted(tr & va)}")
        if va & te:
            issues.append(f"{f.name}: val/test participant overlap {sorted(va & te)}")
        if (tr | va | te) != all_subjects:
            issues.append(f"{f.name}: partition does not cover all participants")

        m = fold_masks(meta, f)
        if (m["train"] & m["test"]).any() or (m["train"] & m["val"]).any() or (m["val"] & m["test"]).any():
            issues.append(f"{f.name}: window mask overlap")

        # window-level check: no identical (subject, trial, start) across splits
        keyed = {}
        for split in ("train", "val", "test"):
            sub = meta.loc[m[split], ["subject_id", "trial_id", "start_s"]]
            keyed[split] = set(map(tuple, sub.to_numpy()))
        for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
            inter = keyed[a] & keyed[b]
            if inter:
                issues.append(f"{f.name}: {len(inter)} identical windows in {a} and {b}")

        # temporal-overlap check: any two windows sharing signal must be co-located
        overlap_cross = 0
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            sa = meta.loc[m[a], ["subject_id", "trial_id", "start_s", "end_s"]]
            sb = meta.loc[m[b], ["subject_id", "trial_id", "start_s", "end_s"]]
            common = set(map(tuple, sa[["subject_id", "trial_id"]].to_numpy())) & \
                     set(map(tuple, sb[["subject_id", "trial_id"]].to_numpy()))
            if common:
                overlap_cross += len(common)
        if overlap_cross:
            issues.append(f"{f.name}: {overlap_cross} (subject,trial) pairs span two splits")

        per_fold.append({
            "fold": f.name, "seed": f.seed, "k": f.fold,
            "n_train_subjects": len(tr), "n_val_subjects": len(va), "n_test_subjects": len(te),
            "train_subjects": sorted(tr), "val_subjects": sorted(va), "test_subjects": sorted(te),
            "n_train_windows": int(m["train"].sum()),
            "n_val_windows": int(m["val"].sum()),
            "n_test_windows": int(m["test"].sum()),
        })

    # every participant must be tested exactly n_seeds times
    counts: Dict[int, int] = {s: 0 for s in all_subjects}
    for f in folds:
        for s in f.test_subjects:
            counts[s] += 1
    n_seeds = len({f.seed for f in folds})
    bad = {s: c for s, c in counts.items() if c != n_seeds}
    if bad:
        issues.append(f"participants not tested exactly {n_seeds} times: {bad}")

    return {
        "n_folds": len(folds),
        "n_subjects": len(all_subjects),
        "test_coverage_per_subject": counts,
        "issues": issues,
        "passed": not issues,
        "per_fold": per_fold,
    }
