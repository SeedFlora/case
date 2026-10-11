"""Validation-only sensitivity study for window length and annotation delay.

Both choices are made using TRAINING and VALIDATION participants only, on a
subset of folds from the first seed, and are then FROZEN before any outer test
fold is touched. Selecting them on test data would be a textbook way to inflate
the headline numbers.

The annotation delay sweep does not rebuild signals: shifting the delay changes
only the label assigned to a window, so labels are recomputed from the cached
20 Hz annotation streams while X and Q stay fixed.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from evaluate import metrics_block
from splits import QualityScaler, RobustChannelScaler, TargetScaler, fold_masks, make_folds
from train import build_net, predict_neural, train_neural
from utils import REPO_ROOT, get_logger, load_configs, modality_names, write_json
from windowing import CACHE_PRE, build_windows, load_windows, tag_for

LOG = get_logger("sensitivity", REPO_ROOT / "logs" / "sensitivity.log")
FS = 64


def relabel_for_delay(meta: pd.DataFrame, window_s: float, delay_s: float,
                      n_probe: int = 16) -> np.ndarray:
    """Recompute [N, 2] labels for a delay, reusing the cached annotations."""
    offs = (np.arange(n_probe) + 0.5) / n_probe * window_s
    y = np.zeros((len(meta), 2), dtype=np.float32)
    for sid, gsub in meta.groupby("subject_id"):
        z = np.load(CACHE_PRE / f"sub_{int(sid)}.npz")
        for vid, g in gsub.groupby("trial_id"):
            at = z[f"annt_{int(vid)}"].astype(np.float64)
            av, aa = z[f"annv_{int(vid)}"], z[f"anna_{int(vid)}"]
            t0 = float(at[0])
            starts = g["start_s"].to_numpy()
            abs_t = t0 + (starts[:, None] + offs[None, :] + delay_s) * 1000.0
            y[g.index.to_numpy(), 0] = np.interp(abs_t.ravel(), at, av).reshape(abs_t.shape).mean(axis=1)
            y[g.index.to_numpy(), 1] = np.interp(abs_t.ravel(), at, aa).reshape(abs_t.shape).mean(axis=1)
    return y


def eval_config(X, Q, y, meta, cfg, folds, device, mdrop: float) -> Dict[str, float]:
    """Train on train participants, score on VALIDATION participants only."""
    scores: List[float] = []
    n_mod, n_q = X.shape[1], Q.shape[2]
    for f in folds:
        m = fold_masks(meta, f)
        itr, iva = np.where(m["train"])[0], np.where(m["val"])[0]
        Xtr_raw, Xva_raw = np.asarray(X[itr]), np.asarray(X[iva])
        xs = RobustChannelScaler().fit(Xtr_raw)
        qs = QualityScaler().fit(Q[itr])
        ts = TargetScaler().fit(y[itr])
        data = {
            "xtr": xs.transform(Xtr_raw), "qtr": qs.transform(Q[itr]),
            "ytr": ts.transform(y[itr]),
            "xva": xs.transform(Xva_raw), "qva": qs.transform(Q[iva]),
            "yva": ts.transform(y[iva]),
        }
        net = build_net("gated", n_mod, n_q, cfg)
        info = train_neural(net, data, cfg, mdrop, device, f.seed,
                            REPO_ROOT / "checkpoints" / f"sens_{f.name}.pt")
        pv = predict_neural(net, data["xva"], data["qva"],
                            np.ones((len(iva), n_mod), dtype=bool), device)
        scores.append(metrics_block(y[iva], ts.inverse_transform(pv))["mean_ccc"])
    return {"val_mean_ccc": float(np.mean(scores)),
            "val_mean_ccc_std": float(np.std(scores)),
            "per_fold": [float(s) for s in scores]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--windows", type=float, nargs="*", default=None)
    ap.add_argument("--delays", type=float, nargs="*", default=None)
    ap.add_argument("--out", default="sensitivity.csv")
    args = ap.parse_args()

    cfg = load_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wcfg = cfg["data"]["windowing"]
    windows = args.windows or [float(w) for w in wcfg["candidate_window_seconds"]]
    delays = args.delays if args.delays is not None else [
        float(d) for d in wcfg["annotation_delay_candidates_s"]]

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for w in windows:
        tag = tag_for(w, 0.0)
        if not (REPO_ROOT / "cache" / "windows" / f"X_{tag}.npy").exists():
            LOG.info("building windows for %.0fs", w)
            build_windows(cfg, w, 0.0)
        X, Q, y0, meta = load_windows(w, 0.0)
        p = cfg["experiment"]["protocol"]
        folds = make_folds(sorted(meta["subject_id"].unique().tolist()), [args.seed],
                           p["n_folds"], p["inner_validation"]["val_participant_fraction"])
        folds = folds[:args.folds]

        for d in delays:
            y = y0 if d == 0.0 else relabel_for_delay(meta, w, d)
            res = eval_config(X, Q, y, meta, cfg, folds, device, args.modality_dropout)
            rows.append({"window_s": w, "delay_s": d, "n_windows": len(meta),
                         **{k: v for k, v in res.items() if k != "per_fold"},
                         "per_fold": str(res["per_fold"])})
            LOG.info("window=%.0fs delay=%.0fs -> val mean CCC = %.4f (+/- %.4f)",
                     w, d, res["val_mean_ccc"], res["val_mean_ccc_std"])

    df = pd.DataFrame(rows).sort_values("val_mean_ccc", ascending=False)
    (REPO_ROOT / "results").mkdir(exist_ok=True, parents=True)
    df.to_csv(REPO_ROOT / "results" / args.out, index=False)

    best = df.iloc[0]
    write_json({
        "selected_window_seconds": float(best["window_s"]),
        "selected_delay_seconds": float(best["delay_s"]),
        "selection_score": float(best["val_mean_ccc"]),
        "selection_basis": "training+validation participants only, seed "
                           f"{args.seed}, first {args.folds} folds",
        "candidates": rows,
        "elapsed_minutes": round((time.time() - t0) / 60, 2),
    }, REPO_ROOT / "manifests" / "sensitivity_selection.json")
    LOG.info("SELECTED window=%.0fs delay=%.0fs (val mean CCC %.4f)",
             best["window_s"], best["delay_s"], best["val_mean_ccc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
