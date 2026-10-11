"""Phase 12: sanity checks run BEFORE any result is accepted.

The governing assumption is adversarial: if performance looks good, assume
leakage until it is disproven. These checks are built to expose leakage,
label-shortcutting, and metric inflation.

  1. Permutation test      -- destroy the signal, expect CCC ~ 0.
  2. Random-label baseline -- train on noise, expect CCC ~ 0.
  3. Participant-identity probe -- can predictions be explained purely by a
     per-participant offset? (the classic confound in affect regression)
  4. Pooled vs participant-averaged metrics -- quantify aggregation inflation.
  5. Prediction/target distribution inspection.
  6. Fold-local scaler independence -- verify no scaler state was shared.
  7. Test-participant isolation -- re-verify every held-out participant.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from evaluate import ccc, metrics_block, per_subject_metrics
from splits import QualityScaler, RobustChannelScaler, TargetScaler, fold_masks, make_folds
from train import build_net, predict_neural, train_neural
from utils import REPO_ROOT, get_logger, load_configs, read_json, write_json
from windowing import load_windows

LOG = get_logger("sanity", REPO_ROOT / "logs" / "sanity_checks.log")
RESULTS = REPO_ROOT / "results"

SUSPICIOUS_CCC = 0.85   # cross-subject affect regression at this level implies leakage


def permutation_and_random_label(cfg, X, Q, y, meta, folds, device,
                                 n_folds: int = 2) -> Dict[str, Any]:
    """Retrain with (a) permuted and (b) random labels; both must collapse."""
    out: Dict[str, Any] = {"permuted": [], "random": []}
    n_mod, n_q = X.shape[1], Q.shape[2]

    for f in folds[:n_folds]:
        m = fold_masks(meta, f)
        itr, iva, ite = (np.where(m[k])[0] for k in ("train", "val", "test"))
        Xtr, Xva, Xte = (np.asarray(X[i]) for i in (itr, iva, ite))
        xs = RobustChannelScaler().fit(Xtr)
        qs = QualityScaler().fit(Q[itr])

        for mode in ("permuted", "random"):
            rng = np.random.RandomState(13)
            if mode == "permuted":
                ytr = y[itr][rng.permutation(len(itr))]
                yva = y[iva][rng.permutation(len(iva))]
            else:
                lo, hi = float(y.min()), float(y.max())
                ytr = rng.uniform(lo, hi, size=y[itr].shape).astype(np.float32)
                yva = rng.uniform(lo, hi, size=y[iva].shape).astype(np.float32)

            ts = TargetScaler().fit(ytr)
            data = {"xtr": xs.transform(Xtr), "qtr": qs.transform(Q[itr]),
                    "ytr": ts.transform(ytr),
                    "xva": xs.transform(Xva), "qva": qs.transform(Q[iva]),
                    "yva": ts.transform(yva)}
            net = build_net("gated", n_mod, n_q, cfg)
            train_neural(net, data, cfg, 0.2, device, f.seed,
                         REPO_ROOT / "checkpoints" / f"sanity_{mode}_{f.name}.pt")
            pt = ts.inverse_transform(
                predict_neural(net, xs.transform(Xte), qs.transform(Q[ite]),
                               np.ones((len(ite), n_mod), dtype=bool), device))
            # scored against the TRUE test labels
            mb = metrics_block(y[ite], pt)
            out[mode].append({"fold": f.name, "mean_ccc": mb["mean_ccc"],
                              "ccc_valence": mb["ccc_valence"],
                              "ccc_arousal": mb["ccc_arousal"]})
            LOG.info("  %s labels, fold %s -> test mean CCC = %+.4f",
                     mode, f.name, mb["mean_ccc"])
    for k in ("permuted", "random"):
        vals = [r["mean_ccc"] for r in out[k]]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def participant_identity_probe(preds: pd.DataFrame) -> Dict[str, Any]:
    """How much of the prediction is just a per-participant constant?

    If a model merely learns 'this participant sits high on valence', its
    predictions correlate with the participant mean but carry no within-
    participant information. We therefore compare the full CCC against the CCC
    after removing each participant's mean from BOTH series.
    """
    out: Dict[str, Any] = {}
    for model, d in preds.groupby("model"):
        rows = {}
        for tgt in ("valence", "arousal"):
            yt = d[f"{tgt}_true"].to_numpy()
            yp = d[f"{tgt}_pred"].to_numpy()
            full = ccc(yt, yp)
            g = d.groupby("subject_id")
            yt_c = yt - g[f"{tgt}_true"].transform("mean").to_numpy()
            yp_c = yp - g[f"{tgt}_pred"].transform("mean").to_numpy()
            within = ccc(yt_c, yp_c)
            # between-participant component: participant means only
            bt = g[f"{tgt}_true"].mean().to_numpy()
            bp = g[f"{tgt}_pred"].mean().to_numpy()
            rows[tgt] = {"ccc_full": full, "ccc_within_participant": within,
                         "ccc_between_participant": ccc(bt, bp)}
        out[model] = rows
    return out


def pooled_vs_participant(subj: pd.DataFrame, fold: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for model in sorted(set(fold["model"])):
        f = fold[fold["model"] == model]
        s = subj[(subj["model"] == model) & (subj["condition"] == "clean")]
        if f.empty or s.empty:
            continue
        pooled = float(f["mean_ccc"].mean())
        part = float(s.groupby("subject_id")["mean_ccc"].mean().mean())
        out[model] = {"pooled_window_mean_ccc": pooled,
                      "participant_averaged_mean_ccc": part,
                      "inflation_pooled_minus_participant": pooled - part}
    return out


def distributions(preds: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_model_prediction_spread": {}}
    for model, d in preds.groupby("model"):
        out["per_model_prediction_spread"][model] = {
            t: {"pred_mean": float(d[f"{t}_pred"].mean()),
                "pred_std": float(d[f"{t}_pred"].std()),
                "true_std": float(d[f"{t}_true"].std()),
                "std_ratio": float(d[f"{t}_pred"].std() / max(d[f"{t}_true"].std(), 1e-9))}
            for t in ("valence", "arousal")}
    bysub = meta.groupby("subject_id")[["valence", "arousal"]].agg(["mean", "std"])
    bytrial = meta.groupby("trial_id")[["valence", "arousal"]].agg(["mean", "std"])
    out["target_by_participant"] = {
        "valence_mean_range": [float(bysub[("valence", "mean")].min()),
                               float(bysub[("valence", "mean")].max())],
        "arousal_mean_range": [float(bysub[("arousal", "mean")].min()),
                               float(bysub[("arousal", "mean")].max())],
        "between_participant_std_valence": float(bysub[("valence", "mean")].std()),
        "between_participant_std_arousal": float(bysub[("arousal", "mean")].std()),
    }
    out["target_by_trial"] = {
        str(int(t)): {"valence_mean": float(bytrial.loc[t, ("valence", "mean")]),
                      "arousal_mean": float(bytrial.loc[t, ("arousal", "mean")])}
        for t in bytrial.index}
    return out


def scaler_independence(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Fold-local scalers must differ across folds; identical state implies sharing."""
    states = {}
    for fold, info in manifest.get("runs", {}).items():
        sc = info.get("scalers", {}).get("signal")
        if sc:
            states[fold] = np.asarray(sc["center"] + sc["scale"], dtype=float)
    keys = sorted(states)
    identical: List[List[str]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.allclose(states[keys[i]], states[keys[j]], atol=1e-12):
                identical.append([keys[i], keys[j]])
    return {"n_folds_with_scaler_state": len(keys),
            "identical_scaler_pairs": identical,
            "all_distinct": not identical}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--perm-folds", type=int, default=2)
    args = ap.parse_args()

    cfg = load_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Q, y, meta = load_windows(args.window, args.delay)
    manifest = read_json(RESULTS / "experiment_manifest.json")
    fold_df = pd.read_csv(RESULTS / "fold_metrics.csv")
    subj_df = pd.read_csv(RESULTS / "subject_metrics.csv")
    preds = pd.read_parquet(RESULTS / "predictions.parquet")

    p = cfg["experiment"]["protocol"]
    folds = make_folds(sorted(meta["subject_id"].unique().tolist()),
                       cfg["experiment"]["seeds"], p["n_folds"],
                       p["inner_validation"]["val_participant_fraction"])

    LOG.info("1-2) permutation and random-label controls ...")
    perm = permutation_and_random_label(cfg, X, Q, y, meta, folds, device, args.perm_folds)

    LOG.info("3) participant-identity probe ...")
    ident = participant_identity_probe(preds)

    LOG.info("4) pooled vs participant-averaged ...")
    agg = pooled_vs_participant(subj_df, fold_df)

    LOG.info("5) distributions ...")
    dist = distributions(preds, meta)

    LOG.info("6) scaler independence ...")
    sc = scaler_independence(manifest)

    LOG.info("7) test-participant isolation ...")
    folds_meta = manifest.get("folds", [])
    iso_issues = []
    for f in folds_meta:
        tr = set(f["train_subjects"]) | set(f["val_subjects"])
        te = set(f["test_subjects"])
        if tr & te:
            iso_issues.append({"fold": f"s{f['seed']}_f{f['fold']}",
                               "overlap": sorted(tr & te)})

    best_clean = float(subj_df[subj_df["condition"] == "clean"]
                       .groupby("model")["mean_ccc"].mean().max())

    verdicts = {
        "permutation_collapses": abs(perm["permuted_mean"]) < 0.05,
        "random_label_collapses": abs(perm["random_mean"]) < 0.05,
        "no_suspiciously_high_performance": best_clean < SUSPICIOUS_CCC,
        "scalers_fold_local": sc["all_distinct"],
        "test_participants_isolated": not iso_issues,
    }
    report = {
        "permutation_and_random_label": perm,
        "participant_identity_probe": ident,
        "pooled_vs_participant": agg,
        "distributions": dist,
        "scaler_independence": sc,
        "test_participant_isolation_issues": iso_issues,
        "best_clean_participant_mean_ccc": best_clean,
        "suspicious_threshold": SUSPICIOUS_CCC,
        "verdicts": verdicts,
        "all_passed": all(verdicts.values()),
    }
    write_json(report, RESULTS / "sanity_checks.json")
    LOG.info("VERDICTS: %s", verdicts)
    LOG.info("permuted mean CCC %.4f | random mean CCC %.4f | best clean %.4f",
             perm["permuted_mean"], perm["random_mean"], best_clean)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
