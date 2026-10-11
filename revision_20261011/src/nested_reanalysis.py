"""Post-review nested reanalysis; all output stays in revision_20261011.

Each original outer fold selects window/delay using its own disjoint inner
validation participants. No test prediction or metric is computed until that
fold's selection is complete. Neural budgets and robustness recipes are the
original ones. Run with the original Python 3.13 environment.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import sys
import time
import warnings
from dataclasses import asdict

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names", category=UserWarning)
REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
sys.path.insert(0, str(ROOT / "src"))
for sub in ("logs", "results", "checkpoints", "manifests", "cache"):
    (REV / sub).mkdir(parents=True, exist_ok=True)

import utils
_original_logger = utils.get_logger
def revision_logger(name, logfile=None):
    return _original_logger("revision." + name, REV / "logs" / (name + ".log"))
utils.get_logger = revision_logger

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import features
from corruption import build_conditions, corrupt_signals, fixed_missing_mask
from evaluate import metrics_block, per_subject_metrics, quadrant_metrics
from models.classical import LightGBMModel, NaivePredictor, RidgeModel
from quality import compute_quality, QUALITY_FEATURE_NAMES
from splits import Fold, RobustChannelScaler, QualityScaler, TargetScaler, fold_masks, audit_folds, make_folds
from train import build_net, train_neural, predict_neural, apply_mask_np, NEURAL_SPECS
from windowing import load_windows, CACHE_PRE, _window_labels
from utils import load_configs, modality_names, write_json, set_all_seeds, capture_environment, sha256_file

_worker_tag = sys.argv[sys.argv.index("--worker-tag")+1] if "--worker-tag" in sys.argv else ""
LOG = revision_logger("nested_reanalysis" + ("_"+_worker_tag if _worker_tag else ""))
WINDOWS = (5., 10., 20.)
DELAYS = (0., 1., 2., 3.)
MDROP = 0.2


def candidate_labels(meta, w, d, y0):
    """Reuse cached signals, but apply the original annotation coverage rule."""
    path = REV / "cache" / f"labels_w{w:g}_d{d:g}.npz"
    if path.exists():
        z = np.load(path)
        return z["y"], z["keep"], z["coverage"]
    y = np.empty_like(y0)
    coverage = np.zeros(len(meta), dtype=np.float32)
    for sid, group in meta.groupby("subject_id"):
        with np.load(CACHE_PRE / f"sub_{int(sid)}.npz") as z:
            for vid, trial in group.groupby("trial_id"):
                ix = trial.index.to_numpy()
                at = z[f"annt_{int(vid)}"].astype(np.float64)
                v, a, cov = _window_labels(at, z[f"annv_{int(vid)}"], z[f"anna_{int(vid)}"],
                    float(at[0]), trial["start_s"].to_numpy(), w, d)
                y[ix, 0], y[ix, 1], coverage[ix] = v, a, cov
    keep = coverage >= 0.999
    # Signal-invalid windows already excluded by original d0 cache.
    np.savez_compressed(path, y=y, keep=keep, coverage=coverage)
    return y, keep, coverage


def load_datasets(write_manifest=True):
    out = {}
    label_info = []
    for w in WINDOWS:
        X, Q, y0, meta = load_windows(w, 0.)
        candidates = {}
        for d in DELAYS:
            y, keep, coverage = candidate_labels(meta, w, d, y0)
            candidates[d] = {"y": y, "keep": keep, "coverage": coverage}
            label_info.append({"window_s": w, "delay_s": d, "base_windows": len(meta),
                "kept_windows": int(keep.sum()), "excluded_incomplete_coverage": int((~keep).sum()),
                "participants": int(meta.loc[keep, "subject_id"].nunique())})
        out[w] = {"X": X, "Q": Q, "meta": meta, "candidates": candidates}
    if write_manifest:
        write_json({"rule": "16 midpoint probes; coverage >= 0.999; existing signal-invalid exclusion preserved",
            "candidates": label_info}, REV / "manifests" / "label_coverage.json")
    return out


def prepare(ds, w, d, f, include_test=False):
    candidate = ds["candidates"][d]
    masks = fold_masks(ds["meta"], f)
    ids = {split: np.flatnonzero(mask & candidate["keep"]) for split, mask in masks.items()}
    itr, iva = ids["train"], ids["val"]
    Xtr = np.asarray(ds["X"][itr])
    Xva = np.asarray(ds["X"][iva])
    y = candidate["y"]
    xs, qs, ts = RobustChannelScaler().fit(Xtr), QualityScaler().fit(ds["Q"][itr]), TargetScaler().fit(y[itr])
    data = {"xtr": xs.transform(Xtr), "qtr": qs.transform(ds["Q"][itr]), "ytr": ts.transform(y[itr]),
        "xva": xs.transform(Xva), "qva": qs.transform(ds["Q"][iva]), "yva": ts.transform(y[iva])}
    return data, xs, qs, ts, ids, Xtr.std(axis=(0, 2)).astype(np.float32)


def initialized_net(kind, nmod, nq, cfg, seed, common=None, **kw):
    # Seed before construction. Equal/late heads otherwise receive different RNG
    # draws because the gated constructor inserts MLPs before its heads.
    set_all_seeds(seed)
    model = build_net(kind, nmod, nq, cfg, **kw)
    if common is not None and hasattr(model, "encoders"):
        state = model.state_dict()
        for key, value in common.items():
            if key.startswith(("encoders.", "heads.")):
                assert key in state and state[key].shape == value.shape
                state[key] = value.clone()
        model.load_state_dict(state)
    return model


def fit_or_load(net, data, cfg, dropout, device, seed, ckpt):
    infofile = ckpt.with_suffix(".json")
    if ckpt.exists() and infofile.exists():
        net.to(device).load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        return json.loads(infofile.read_text(encoding="utf-8"))
    start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    info = train_neural(net, data, cfg, dropout, device, seed, ckpt)
    info.update(train_seconds=round(time.time()-start, 3), modality_dropout=dropout,
        initialization_seed=seed, training_seed=seed,
        checkpoint=str(ckpt.relative_to(REV)))
    if device.type == "cuda":
        info["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated()/1024**2, 1)
    write_json(info, infofile)
    return info


def global_features(ds, w):
    path = REV / "cache" / f"features_w{w:g}.npy"
    if not path.exists():
        LOG.info("Extracting label-free clean feature cache w%g", w)
        np.save(path, features.extract(ds["X"]))
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def gate_outputs(net, q, mask, device):
    net.eval()
    outs = []
    for start in range(0, len(q), 1024):
        qq = torch.from_numpy(q[start:start+1024]).to(device)
        mm = torch.from_numpy(mask[start:start+1024].astype(np.float32)).to(device)
        outs.append(net.gates(qq, mm).float().cpu().numpy())
    raw = np.concatenate(outs)
    weights = raw / (raw.sum(axis=1, keepdims=True) + 1e-6)
    return raw, weights


def run_fold(f, datasets, cfg, device, conditions, mods):
    start_fold = time.time()
    seed = f.seed * 1000 + f.fold
    nmod = len(mods)
    # All late/gated conditions have identical encoders AND heads initially.
    reference = initialized_net("gated", nmod, 10, cfg, seed)
    common = {k: v.detach().clone() for k, v in reference.state_dict().items()
              if k.startswith(("encoders.", "heads."))}
    del reference
    candidates = []
    for w in WINDOWS:
        ds = datasets[w]
        for d in DELAYS:
            data, xs, qs, ts, ids, refstd = prepare(ds, w, d, f)
            net = initialized_net("gated", nmod, 10, cfg, seed, common)
            ckpt = REV / "checkpoints" / f"candidate_{f.name}_w{w:g}_d{d:g}.pt"
            info = fit_or_load(net, data, cfg, MDROP, device, seed, ckpt)
            pv = predict_neural(net, data["xva"], data["qva"],
                np.ones((len(ids["val"]), nmod), bool), device)
            score = metrics_block(ds["candidates"][d]["y"][ids["val"]], ts.inverse_transform(pv))["mean_ccc"]
            candidates.append({"fold": f.name, "seed": f.seed, "k": f.fold,
                "window_s": w, "delay_s": d, "val_mean_ccc": score,
                "n_train_windows": len(ids["train"]), "n_val_windows": len(ids["val"]),
                "n_test_windows": len(ids["test"]), "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"], "train_seconds": info["train_seconds"],
                "checkpoint": str(ckpt.relative_to(REV)),
                "signal_scaler": xs.state(), "quality_scaler": qs.state(), "target_scaler": ts.state()})
            LOG.info("%s candidate w%g d%g val=%.5f epochs=%d fit=%.1fs", f.name, w, d,
                score, info["epochs_run"], info["train_seconds"])
            del net, data
    # Deterministic tie-break: first in predeclared w ascending, delay ascending grid.
    selected = max(candidates, key=lambda r: r["val_mean_ccc"])
    w, d = selected["window_s"], selected["delay_s"]
    selection = {"fold": asdict(f), "selection_criterion": "inner validation pooled-window mean CCC in native units",
        "test_metrics_used": False, "selected": selected, "candidates": candidates}
    write_json(selection, REV / "manifests" / f"selection_{f.name}.json")
    LOG.info("%s SELECTED w%g d%g using inner val=%.5f", f.name, w, d, selected["val_mean_ccc"])

    ds = datasets[w]
    data, xs, qs, ts, ids, refstd = prepare(ds, w, d, f, include_test=True)
    itr, iva, ite = ids["train"], ids["val"], ids["test"]
    y = ds["candidates"][d]["y"]
    yte = y[ite]
    meta_te = ds["meta"].iloc[ite]
    subj = meta_te["subject_id"].to_numpy()
    Xte = np.asarray(ds["X"][ite])
    Qte = ds["Q"][ite]
    trained, runs = {}, {}
    for name, spec in NEURAL_SPECS.items():
        net = initialized_net(spec["cls"], nmod, 10, cfg, seed, common, **spec["kw"])
        if name == "reliability_gated":
            ckpt = REV / selected["checkpoint"]
            net.to(device).load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            info = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
            info["reused_selected_candidate"] = True
        else:
            ckpt = REV / "checkpoints" / f"{f.name}_{name}.pt"
            info = fit_or_load(net, data, cfg, MDROP if spec["mdrop"] else 0., device, seed, ckpt)
        trained[name], runs[name] = net, info
        LOG.info("%s %-20s epochs=%d best=%.4f fit=%.1fs", f.name, name,
            info["epochs_run"], info["best_val_mean_ccc"], info["train_seconds"])
    F = global_features(ds, w)
    Ftr, Fva, Fclean = np.asarray(F[itr]), np.asarray(F[iva]), np.asarray(F[ite])
    center = np.median(Ftr, axis=0)
    scale = np.maximum(np.percentile(Ftr, 75, axis=0)-np.percentile(Ftr, 25, axis=0), 1e-6)
    def fscale(a):
        return ((a-center)/scale).astype(np.float32)
    naive = NaivePredictor().fit(Ftr, y[itr])
    ridge = RidgeModel(alphas=cfg["experiment"]["models"]["ridge"]["alphas"])
    ridge.select(fscale(Ftr), y[itr], fscale(Fva), y[iva])
    ridge.fit(fscale(Ftr), y[itr])
    lgbparams = dict(cfg["experiment"]["models"]["lightgbm"])
    lgbparams.update(random_state=seed, n_jobs=4)
    lgbm = LightGBMModel(**lgbparams).fit(fscale(Ftr), y[itr])
    runs["ridge_alpha"] = ridge.alpha_
    runs["scalers"] = {"signal": xs.state(), "quality": qs.state(), "target": ts.state(),
        "features": {"center": center, "scale": scale}, "training_ref_std": refstd}
    runs["selected_window_s"], runs["selected_delay_s"] = w, d
    runs["initialization"] = "seed before construction; clone gated reference encoders/heads into all late/gated models"
    runs["fold"] = asdict(f)
    runs["seed_used"] = seed
    runs["training_settings"] = cfg["experiment"]["training"]
    rows, subject_rows, gate_rows = [], [], []
    pred_path = REV / "results" / f"predictions_{f.name}.parquet"
    writer = None
    clean_q = qs.transform(Qte)
    clean_x = xs.transform(Xte)
    clean_f = fscale(Fclean)
    clean_gate, clean_weights = gate_outputs(trained["reliability_gated"], clean_q,
        np.ones((len(ite), nmod), bool), device)
    for ci, cond in enumerate(conditions):
        # Missing-only conditions reuse clean quality/features. Corruption requires
        # recomputation only for affected channels (all descriptors window-local).
        xr, qraw, fraw = Xte, Qte, Fclean
        if cond["corrupt"] is not None:
            cc = cond["corrupt"]
            ch = np.asarray(cc["channels"])
            xr = corrupt_signals(Xte, cc["kind"], cc["severity"],
                cond["seed"]+int(cond.get("draw", 0)), refstd, channels=ch)
            qraw = Qte.copy()
            qraw[:, ch] = compute_quality(xr[:, ch, :])
            fraw = Fclean.reshape(len(ite), nmod, -1).copy()
            fraw[:, ch] = features.extract(xr[:, ch, :]).reshape(len(ite), len(ch), -1)
            fraw = fraw.reshape(len(ite), -1)
        mask = fixed_missing_mask(len(ite), nmod, cond["missing"])
        xscaled = clean_x if xr is Xte else xs.transform(xr)
        qscaled = clean_q if qraw is Qte else qs.transform(qraw)
        xm, qm = apply_mask_np(xscaled, qscaled, mask)
        fm = features.apply_modality_mask(clean_f if fraw is Fclean else fscale(fraw), mask, nmod)
        pred = {"naive": naive.predict(fm), "ridge": ridge.predict(fm), "lightgbm": lgbm.predict(fm)}
        for name, net in trained.items():
            pred[name] = ts.inverse_transform(predict_neural(net, xm, qm, mask, device))
        for name, yp in pred.items():
            ps = per_subject_metrics(subj, yte, yp)
            identity = {"fold": f.name, "seed": f.seed, "k": f.fold, "model": name,
                "condition": cond["condition"], "family": cond["family"], "level": cond["level"],
                "window_s": w, "delay_s": d}
            rec = {**identity, **metrics_block(yte, yp), "n_test_windows": len(ite), "n_test_subjects": len(f.test_subjects)}
            rec.update({"subj_"+metric: float(ps[metric].mean()) for metric in
                ("ccc_valence", "ccc_arousal", "mean_ccc", "mean_mae", "mean_rmse")})
            if cond["condition"] == "clean":
                rec.update(quadrant_metrics(yte, yp))
            rows.append(rec)
            subject_rows.extend([{**identity, **row} for row in ps.to_dict(orient="records")])
            frame = pd.DataFrame({"fold": f.name, "seed": f.seed, "k": f.fold, "model": name,
                "condition": cond["condition"], "window_s": w, "delay_s": d,
                "source_window_index": ite, "subject_id": subj, "trial_id": meta_te["trial_id"].to_numpy(),
                "start_s": meta_te["start_s"].to_numpy(), "valence_true": yte[:, 0], "arousal_true": yte[:, 1],
                "valence_pred": yp[:, 0], "arousal_pred": yp[:, 1]})
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(pred_path, table.schema, compression="zstd")
            writer.write_table(table)
        raw, weights = gate_outputs(trained["reliability_gated"], qm, mask, device)
        affected = set(cond["missing"])
        if cond["corrupt"]:
            affected.update(cond["corrupt"]["channels"])
        for sid in np.unique(subj):
            ix = subj == sid
            for channel, modality in enumerate(mods):
                role = "absent" if channel in cond["missing"] else "affected" if channel in affected else "healthy"
                gr, gw = raw[ix, channel], weights[ix, channel]
                gate_rows.append({"fold": f.name, "seed": f.seed, "k": f.fold, "subject_id": int(sid),
                    "condition": cond["condition"], "family": cond["family"], "level": cond["level"],
                    "draw": cond.get("draw", -1), "modality": modality, "role": role,
                    "raw_gate_mean": float(gr.mean()), "raw_gate_std": float(gr.std()),
                    "normalized_weight_mean": float(gw.mean()), "normalized_weight_std": float(gw.std()),
                    "clean_raw_gate_mean": float(clean_gate[ix, channel].mean()),
                    "clean_weight_mean": float(clean_weights[ix, channel].mean()),
                    "raw_gate_delta_from_clean": float((gr-clean_gate[ix, channel]).mean()),
                    "weight_delta_from_clean": float((gw-clean_weights[ix, channel]).mean()),
                    "saturation_fraction": float(((gr<.05)|(gr>.95)).mean()) if role!="absent" else np.nan,
                    "n_windows": int(ix.sum()), "available_modalities": int(mask[0].sum()),
                    "quality_clip_fraction": float((np.abs(qscaled[ix, channel])>=QualityScaler.CLIP).mean())})
        if ci % 5 == 0 or ci == len(conditions)-1:
            LOG.info("%s evaluation %d/%d %s elapsed %.1fs", f.name, ci+1, len(conditions),
                cond["condition"], time.time()-start_fold)
    if writer:
        writer.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(REV / "results" / f"robustness_metrics_{f.name}.csv", index=False)
    frame.loc[frame.condition=="clean"].to_csv(REV / "results" / f"fold_metrics_{f.name}.csv", index=False)
    pd.DataFrame(subject_rows).to_csv(REV / "results" / f"subject_metrics_{f.name}.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(REV / "results" / f"gate_diagnostics_{f.name}.csv", index=False)
    runs["fold_seconds"] = round(time.time()-start_fold, 3)
    runs["prediction_file"] = str(pred_path.relative_to(REV))
    write_json(runs, REV / "manifests" / f"run_{f.name}.json")
    write_json({"complete": True, "fold": f.name, "seconds": runs["fold_seconds"]},
        REV / "manifests" / f"complete_{f.name}.json")
    return candidates, runs


def combine(folds, cfg, conditions, audit, start):
    complete = [f for f in folds if (REV/"manifests"/f"complete_{f.name}.json").exists()]
    if not complete:
        return
    for category in ("fold_metrics", "subject_metrics", "robustness_metrics", "gate_diagnostics"):
        pd.concat([pd.read_csv(REV/"results"/f"{category}_{f.name}.csv") for f in complete],
            ignore_index=True).to_csv(REV/"results"/f"{category}.csv", index=False)
    selections = [json.loads((REV/"manifests"/f"selection_{f.name}.json").read_text()) for f in complete]
    selection_rows = [{k:v for k,v in c.items() if not k.endswith("scaler")}
        for s in selections for c in s["candidates"]]
    pd.DataFrame(selection_rows).to_csv(REV/"results"/"nested_selection_scores.csv", index=False)
    pd.DataFrame([{k:v for k,v in s["selected"].items() if not k.endswith("scaler")}
        for s in selections]).to_csv(REV/"results"/"selected_settings.csv", index=False)
    runs = {f.name:json.loads((REV/"manifests"/f"run_{f.name}.json").read_text()) for f in complete}
    write_json({"tag": "nested_post_review", "n_folds": len(complete), "n_folds_planned": len(folds),
        "complete": len(complete)==len(folds), "n_conditions": len(conditions), "modalities": modality_names(cfg),
        "windows": WINDOWS, "delays": DELAYS, "modality_dropout": MDROP,
        "selection_basis": "Independent 12-candidate inner grouped holdout within EACH original outer fold; pooled validation CCC; no test selection",
        "annotation_coverage": "Recomputed at each candidate delay; incomplete windows excluded before partition scalers fit",
        "protocol": cfg["experiment"]["protocol"], "training": cfg["experiment"]["training"],
        "initialization": "fold_seed = outer_seed*1000+fold; seed before model construction; exact shared encoder/head state among late/gated controls",
        "quality_during_training": "clean quality features; stochastic modality masking only; no corruption augmentation",
        "folds": [asdict(f) for f in folds], "runs": runs,
        "leakage_audit_passed": audit["passed"], "environment": capture_environment(),
        "elapsed_seconds_this_invocation": round(time.time()-start, 3),
        "total_fold_seconds": sum(r["fold_seconds"] for r in runs.values()),
        "torch_num_threads": torch.get_num_threads(),
        "revision_source_sha256": {str(p.relative_to(REV)): sha256_file(p) for p in
            [Path(__file__), REV/"src"/"validate_nested_implementation.py", REV/"manifests"/"frozen_original_config.json"]},
        "original_source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
            [ROOT/"src"/"train.py", ROOT/"src"/"sensitivity.py", ROOT/"configs"/"experiment.yaml"]}},
        REV/"results"/"experiment_manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--fold-name", nargs="*")
    parser.add_argument("--worker", action="store_true", help="Disjoint worker; no shared manifests or combined outputs")
    parser.add_argument("--worker-tag", type=str)
    parser.add_argument("--prebuild-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(4)
    cfg = load_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = load_datasets(write_manifest=not args.worker)
    original = json.loads((ROOT/"results"/"experiment_manifest.json").read_text())
    folds = [Fold(int(f["seed"]), int(f["fold"]), tuple(f["train_subjects"]), tuple(f["val_subjects"]),
        tuple(f["test_subjects"])) for f in original["folds"]]
    generated = make_folds(sorted(datasets[20.]["meta"].subject_id.unique()), cfg["experiment"]["seeds"],
        cfg["experiment"]["protocol"]["n_folds"], cfg["experiment"]["protocol"]["inner_validation"]["val_participant_fraction"])
    assert folds == generated, "Original exact folds differ from regenerated folds"
    audit = audit_folds(datasets[20.]["meta"], folds)
    assert audit["passed"], audit["issues"]
    if not args.worker:
        write_json(audit, REV/"manifests"/"leakage_audit_nested.json")
    conditions = build_conditions(cfg, modality_names(cfg))
    assert len(conditions)==55
    if not args.worker:
        write_json(conditions, REV/"manifests"/"corruption_manifest_nested.json")
        write_json(cfg, REV/"manifests"/"frozen_original_config.json")
    if args.prebuild_only:
        for w in WINDOWS:
            global_features(datasets[w], w)
        return
    if args.aggregate_only:
        combine(folds, cfg, conditions, audit, start)
        return
    chosen = folds
    if args.fold_name:
        chosen = [f for f in folds if f.name in args.fold_name]
    if args.fold_limit:
        chosen = chosen[:args.fold_limit]
    for i, f in enumerate(chosen):
        if (REV/"manifests"/f"complete_{f.name}.json").exists():
            LOG.info("Skipping completed %s", f.name)
            continue
        LOG.info("[%d/%d] %s train=%s val=%s test=%s", i+1, len(chosen), f.name,
            f.train_subjects, f.val_subjects, f.test_subjects)
        run_fold(f, datasets, cfg, device, conditions, modality_names(cfg))
        if not args.worker:
            combine(folds, cfg, conditions, audit, start)
    if not args.worker:
        combine(folds, cfg, conditions, audit, start)
    LOG.info("Invocation finished in %.2f min", (time.time()-start)/60)


if __name__ == "__main__":
    main()
