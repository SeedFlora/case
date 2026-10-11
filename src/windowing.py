"""Phase 3: window generation with trial-safe boundaries and annotation delay.

Windows NEVER cross a trial (stimulus) boundary, and every window carries the
participant and trial identifiers needed to keep grouped splits honest.

Labels are the mean annotation over the window, optionally shifted forward in
time by `delay_s`: the joystick annotation lags the physiological response
(the participant must perceive the change and then move), so the annotation
that corresponds to physiological window [a, b] is sampled over
[a + delay, b + delay].

Artifacts (per window-length / delay configuration):
  cache/windows/X_{tag}.npy      float32 [N, C, T]  signals
  cache/windows/Q_{tag}.npy      float32 [N, C, 10] quality vectors
  cache/windows/y_{tag}.npy      float32 [N, 2]     valence, arousal
  cache/windows/meta_{tag}.parquet                  per-window metadata
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from quality import N_QUALITY_FEATURES, QUALITY_FEATURE_NAMES, compute_quality, invalid_fraction
from utils import REPO_ROOT, get_logger, load_configs, modality_names, write_json

LOG = get_logger("windowing", REPO_ROOT / "logs" / "windowing.log")
CACHE_PRE = REPO_ROOT / "cache" / "preprocessed"
CACHE_WIN = REPO_ROOT / "cache" / "windows"
FS = 64


def tag_for(window_s: float, delay_s: float) -> str:
    return f"w{window_s:g}_d{delay_s:g}"


def _window_labels(
    anno_t_ms: np.ndarray, anno_v: np.ndarray, anno_a: np.ndarray,
    t0_ms: float, starts_s: np.ndarray, window_s: float, delay_s: float,
    n_probe: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean valence/arousal over each (delay-shifted) window.

    The 20 Hz annotation stream is sampled at `n_probe` evenly spaced points
    inside each window and averaged, which is equivalent to a mean over the
    window for a slowly varying joystick signal and avoids per-window searches.
    """
    offs = (np.arange(n_probe) + 0.5) / n_probe * window_s          # [P]
    abs_t = t0_ms + (starts_s[:, None] + offs[None, :] + delay_s) * 1000.0  # [N, P]

    v = np.interp(abs_t.ravel(), anno_t_ms, anno_v).reshape(abs_t.shape)
    a = np.interp(abs_t.ravel(), anno_t_ms, anno_a).reshape(abs_t.shape)

    # coverage: fraction of probe points that fall inside the annotated span
    inside = ((abs_t >= anno_t_ms[0]) & (abs_t <= anno_t_ms[-1])).mean(axis=1)
    return v.mean(axis=1).astype(np.float32), a.mean(axis=1).astype(np.float32), inside.astype(np.float32)


def build_windows(cfg: Dict[str, Any], window_s: float, delay_s: float,
                  subjects: List[int] | None = None) -> Dict[str, Any]:
    names = modality_names(cfg)
    emotional = cfg["data"]["videos"]["emotional_ids"]
    labels = cfg["data"]["videos"]["labels"]
    stride_s = float(cfg["data"]["windowing"]["stride_seconds"])
    max_invalid = float(cfg["data"]["windowing"]["max_invalid_fraction"])

    win_n = int(round(window_s * FS))
    stride_n = int(round(stride_s * FS))

    sids = subjects or sorted(int(p.stem.split("_")[1]) for p in CACHE_PRE.glob("sub_*.npz"))

    Xs: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    for sid in sids:
        z = np.load(CACHE_PRE / f"sub_{sid}.npz")
        for vid in emotional:
            key = f"sig_{vid}"
            if key not in z:
                continue
            sig = z[key]                       # [C, T]
            at = z[f"annt_{vid}"].astype(np.float64)
            av, aa = z[f"annv_{vid}"], z[f"anna_{vid}"]
            t0_ms = float(at[0])
            total = sig.shape[1]
            if total < win_n:
                continue

            starts = np.arange(0, total - win_n + 1, stride_n)
            # [n, C, win_n] view via stride tricks, then copy once
            idx = starts[:, None] + np.arange(win_n)[None, :]
            w = sig[:, idx]                    # [C, n, win_n]
            w = np.transpose(w, (1, 0, 2)).copy()   # [n, C, win_n]

            starts_s = starts / FS
            v, a, cover = _window_labels(at, av, aa, t0_ms, starts_s, window_s, delay_s)

            inv = invalid_fraction(w)          # [n, C]
            Xs.append(w)
            for j in range(w.shape[0]):
                rows.append({
                    "subject_id": sid,
                    "trial_id": int(vid),
                    "trial_label": labels[vid],
                    "window_idx": int(j),
                    "start_s": float(starts_s[j]),
                    "end_s": float(starts_s[j] + window_s),
                    "start_ms_abs": float(t0_ms + starts_s[j] * 1000.0),
                    "end_ms_abs": float(t0_ms + (starts_s[j] + window_s) * 1000.0),
                    "valence": float(v[j]),
                    "arousal": float(a[j]),
                    "annotation_coverage": float(cover[j]),
                    "max_invalid_fraction": float(inv[j].max()),
                    "mean_invalid_fraction": float(inv[j].mean()),
                })

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    meta = pd.DataFrame(rows)

    # Exclusion: windows whose worst channel exceeds the invalid threshold, or
    # whose annotation coverage is incomplete.
    meta["excluded"] = (
        (meta["max_invalid_fraction"] > max_invalid)
        | (meta["annotation_coverage"] < 0.999)
    )
    keep = ~meta["excluded"].to_numpy()

    LOG.info("built %d windows (%d excluded) for window=%.0fs delay=%.0fs",
             len(meta), int((~keep).sum()), window_s, delay_s)

    Xk = X[keep]
    metak = meta.loc[keep].reset_index(drop=True)
    Q = compute_quality(Xk)
    y = metak[["valence", "arousal"]].to_numpy(dtype=np.float32)

    CACHE_WIN.mkdir(parents=True, exist_ok=True)
    tag = tag_for(window_s, delay_s)
    np.save(CACHE_WIN / f"X_{tag}.npy", Xk)
    np.save(CACHE_WIN / f"Q_{tag}.npy", Q)
    np.save(CACHE_WIN / f"y_{tag}.npy", y)
    metak.to_parquet(CACHE_WIN / f"meta_{tag}.parquet", index=False)

    info = {
        "tag": tag,
        "window_seconds": window_s,
        "stride_seconds": stride_s,
        "delay_seconds": delay_s,
        "fs_hz": FS,
        "samples_per_window": win_n,
        "n_windows_built": int(len(meta)),
        "n_windows_excluded": int((~keep).sum()),
        "n_windows_kept": int(keep.sum()),
        "exclusion_reasons": {
            "invalid_fraction_gt_threshold": int((meta["max_invalid_fraction"] > max_invalid).sum()),
            "incomplete_annotation_coverage": int((meta["annotation_coverage"] < 0.999).sum()),
        },
        "max_invalid_fraction_threshold": max_invalid,
        "n_subjects": int(metak["subject_id"].nunique()),
        "n_trials": int(metak.groupby(["subject_id", "trial_id"]).ngroups),
        "windows_per_subject": metak.groupby("subject_id").size().describe().to_dict(),
        "modalities": names,
        "quality_features": QUALITY_FEATURE_NAMES,
        "shapes": {"X": list(Xk.shape), "Q": list(Q.shape), "y": list(y.shape)},
        "valence_stats": {"mean": float(y[:, 0].mean()), "std": float(y[:, 0].std()),
                          "min": float(y[:, 0].min()), "max": float(y[:, 0].max())},
        "arousal_stats": {"mean": float(y[:, 1].mean()), "std": float(y[:, 1].std()),
                          "min": float(y[:, 1].min()), "max": float(y[:, 1].max())},
        "size_mb": {
            "X": round(Xk.nbytes / 1024**2, 2),
            "Q": round(Q.nbytes / 1024**2, 2),
        },
    }
    write_json(info, REPO_ROOT / "manifests" / f"windows_{tag}.json")
    return info


def load_windows(window_s: float, delay_s: float):
    tag = tag_for(window_s, delay_s)
    X = np.load(CACHE_WIN / f"X_{tag}.npy", mmap_mode="r")
    Q = np.load(CACHE_WIN / f"Q_{tag}.npy")
    y = np.load(CACHE_WIN / f"y_{tag}.npy")
    meta = pd.read_parquet(CACHE_WIN / f"meta_{tag}.parquet")
    return X, Q, y, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=float, nargs="*", default=None)
    ap.add_argument("--delays", type=float, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_configs()
    wcfg = cfg["data"]["windowing"]
    windows = args.windows if args.windows is not None else [float(wcfg["window_seconds"])]
    delays = args.delays if args.delays is not None else [0.0]

    infos = []
    for w in windows:
        for d in delays:
            infos.append(build_windows(cfg, w, d))
            i = infos[-1]
            LOG.info("%s -> %d windows kept, X %.1f MB, valence mu=%.3f sd=%.3f",
                     i["tag"], i["n_windows_kept"], i["size_mb"]["X"],
                     i["valence_stats"]["mean"], i["valence_stats"]["std"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
