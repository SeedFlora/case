"""Phase 8: the experiment driver.

One entry point runs the whole grid: for every outer fold it fits fold-local
scalers on the TRAINING participants only, trains every model, and evaluates the
selected checkpoints on the held-out participants under every robustness
condition. Nothing about the test partition is ever used for fitting, scaling,
model selection, or early stopping.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import features as featmod
from corruption import build_conditions, corrupt_signals, fixed_missing_mask
from evaluate import metrics_block, per_subject_metrics, quadrant_metrics
from models.classical import LightGBMModel, NaivePredictor, RidgeModel
from models.early_fusion import EarlyFusionNet
from models.encoders import count_parameters
from models.late_fusion import LateFusionNet
from models.reliability_gated import ReliabilityGatedNet
from quality import compute_quality
from splits import (Fold, QualityScaler, RobustChannelScaler, TargetScaler,
                    audit_folds, fold_masks, make_folds)
from utils import (REPO_ROOT, capture_environment, get_logger, load_configs,
                   modality_names, set_all_seeds, write_json)
from windowing import load_windows

LOG = get_logger("train", REPO_ROOT / "logs" / "train.log")
RESULTS = REPO_ROOT / "results"
CKPT = REPO_ROOT / "checkpoints"

NEURAL_SPECS: Dict[str, Dict[str, Any]] = {
    # name -> (class key, kwargs override, uses modality dropout)
    "early_fusion":       {"cls": "early", "kw": {}, "mdrop": False},
    "late_fusion":        {"cls": "late", "kw": {}, "mdrop": False},
    "reliability_gated":  {"cls": "gated", "kw": {}, "mdrop": True},
    "abl_no_quality":     {"cls": "gated", "kw": {"use_quality": False}, "mdrop": True},
    "abl_no_moddropout":  {"cls": "gated", "kw": {}, "mdrop": False},
    "abl_equal_weights":  {"cls": "gated", "kw": {"learned_gates": False}, "mdrop": True},
}


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - CCC, averaged over targets, computed within the batch."""
    loss = 0.0
    for j in range(pred.shape[1]):
        p, t = pred[:, j], target[:, j]
        pm, tm = p.mean(), t.mean()
        pv, tv = p.var(unbiased=False), t.var(unbiased=False)
        cov = ((p - pm) * (t - tm)).mean()
        denom = pv + tv + (pm - tm) ** 2 + 1e-8
        loss = loss + (1.0 - 2.0 * cov / denom)
    return loss / pred.shape[1]


class CombinedLoss(nn.Module):
    def __init__(self, lambda_mae: float, lambda_ccc: float):
        super().__init__()
        self.l1 = nn.SmoothL1Loss()
        self.a, self.b = lambda_mae, lambda_ccc

    def forward(self, pred, target):
        return self.a * self.l1(pred, target) + self.b * ccc_loss(pred, target)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_net(kind: str, n_mod: int, n_q: int, cfg: Dict[str, Any],
              **override) -> nn.Module:
    mc = cfg["experiment"]["models"]
    if kind == "early":
        return EarlyFusionNet(n_mod, **{k: v for k, v in mc["early_fusion"].items()
                                        if k != "enabled"})
    if kind == "late":
        return LateFusionNet(n_mod, **{k: v for k, v in mc["late_fusion"].items()
                                       if k != "enabled"})
    kw = {k: v for k, v in mc["reliability_gated"].items()
          if k not in ("enabled", "modality_dropout_candidates")}
    kw.update(override)
    return ReliabilityGatedNet(n_mod, n_quality=n_q, **kw)


def apply_mask_np(x: np.ndarray, q: np.ndarray, mask: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Zero the signal and quality vector of absent modalities."""
    xm = x * mask[:, :, None]
    qm = q * mask[:, :, None]
    return xm.astype(np.float32), qm.astype(np.float32)


def sample_modality_dropout(n: int, c: int, p: float, rng: np.random.RandomState
                            ) -> np.ndarray:
    """Random availability mask, guaranteeing at least one live modality."""
    m = rng.rand(n, c) >= p
    dead = ~m.any(axis=1)
    if dead.any():
        keep = rng.randint(0, c, size=dead.sum())
        m[np.where(dead)[0], keep] = True
    return m


@torch.no_grad()
def predict_neural(model: nn.Module, x: np.ndarray, q: np.ndarray, mask: np.ndarray,
                   device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    for s in range(0, x.shape[0], batch):
        e = min(s + batch, x.shape[0])
        xb = torch.from_numpy(x[s:e]).to(device)
        qb = torch.from_numpy(q[s:e]).to(device)
        mb = torch.from_numpy(mask[s:e].astype(np.float32)).to(device)
        outs.append(model(xb, qb, mb).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_neural(model: nn.Module, data: Dict[str, np.ndarray], cfg: Dict[str, Any],
                 mdrop: float, device: torch.device, seed: int,
                 ckpt_path: Path) -> Dict[str, Any]:
    tc = cfg["experiment"]["training"]
    set_all_seeds(seed)
    model = model.to(device)

    xtr = torch.from_numpy(data["xtr"]).to(device)
    qtr = torch.from_numpy(data["qtr"]).to(device)
    ytr = torch.from_numpy(data["ytr"]).to(device)
    n = xtr.shape[0]
    c = xtr.shape[1]

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["max_epochs"])
    lossfn = CombinedLoss(tc["loss"]["lambda_mae"], tc["loss"]["lambda_ccc"])
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    rng = np.random.RandomState(seed)
    bs = int(tc["batch_size"])
    best, best_ep, bad = -np.inf, -1, 0
    history: List[Dict[str, float]] = []

    for ep in range(int(tc["max_epochs"])):
        model.train()
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            if idx.size < 4:
                continue
            xb, qb, yb = xtr[idx], qtr[idx], ytr[idx]
            if mdrop > 0:
                mnp = sample_modality_dropout(idx.size, c, mdrop, rng)
                mb = torch.from_numpy(mnp.astype(np.float32)).to(device)
                xb = xb * mb.unsqueeze(-1)
                qb = qb * mb.unsqueeze(-1)
            else:
                mb = torch.ones(idx.size, c, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb, qb, mb)
                loss = lossfn(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tot += float(loss.item()); nb += 1
        sched.step()

        # validation: clean, fully-available inputs
        pv = predict_neural(model, data["xva"], data["qva"],
                            np.ones((data["xva"].shape[0], c), dtype=bool), device)
        vm = metrics_block(data["yva"], pv)
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                        "val_mean_ccc": vm["mean_ccc"]})

        if vm["mean_ccc"] > best + 1e-5:
            best, best_ep, bad = vm["mean_ccc"], ep, 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= int(tc["patience"]):
                break

    # weights_only=True: these checkpoints hold tensors only, and unpickling
    # arbitrary objects from disk is never needed here.
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return {"best_val_mean_ccc": float(best), "best_epoch": int(best_ep),
            "epochs_run": len(history), "history": history,
            "n_parameters": count_parameters(model)}


# --------------------------------------------------------------------------- #
# per-fold execution
# --------------------------------------------------------------------------- #
def run_fold(f: Fold, X, Q, y, meta, cfg, device, conditions, args) -> Dict[str, Any]:
    mods = modality_names(cfg)
    n_mod, n_q = len(mods), Q.shape[2]
    m = fold_masks(meta, f)
    t_fold = time.time()

    idx_tr, idx_va, idx_te = np.where(m["train"])[0], np.where(m["val"])[0], np.where(m["test"])[0]
    Xtr_raw = np.asarray(X[idx_tr]); Xva_raw = np.asarray(X[idx_va]); Xte_raw = np.asarray(X[idx_te])
    Qtr_raw, Qva_raw, Qte_raw = Q[idx_tr], Q[idx_va], Q[idx_te]
    ytr_raw, yva_raw, yte_raw = y[idx_tr], y[idx_va], y[idx_te]
    subj_te = meta["subject_id"].to_numpy()[idx_te]

    # ---- fold-local scalers: FIT ON TRAIN ONLY -----------------------------
    xs = RobustChannelScaler().fit(Xtr_raw)
    qs = QualityScaler().fit(Qtr_raw)
    ts = TargetScaler().fit(ytr_raw)
    ref_std = Xtr_raw.std(axis=(0, 2)).astype(np.float32)   # raw-unit per-channel std

    Xtr, Xva = xs.transform(Xtr_raw), xs.transform(Xva_raw)
    Qtr, Qva = qs.transform(Qtr_raw), qs.transform(Qva_raw)
    ytr_s, yva_s = ts.transform(ytr_raw), ts.transform(yva_raw)

    full_tr = np.ones((len(idx_tr), n_mod), dtype=bool)
    full_va = np.ones((len(idx_va), n_mod), dtype=bool)

    rows_fold: List[Dict[str, Any]] = []
    rows_subj: List[Dict[str, Any]] = []
    rows_rob: List[Dict[str, Any]] = []
    preds_out: List[pd.DataFrame] = []
    runinfo: Dict[str, Any] = {}

    def record(model_name: str, cond: Dict[str, Any], pred_native: np.ndarray,
               store_predictions: bool = False) -> None:
        blk = metrics_block(yte_raw, pred_native)
        rec = {"fold": f.name, "seed": f.seed, "k": f.fold, "model": model_name,
               "condition": cond["condition"], "family": cond["family"],
               "level": cond["level"], "n_test_windows": len(idx_te),
               "n_test_subjects": len(set(subj_te.tolist()))}
        rec.update(blk)
        ps = per_subject_metrics(subj_te, yte_raw, pred_native)
        agg = {f"subj_{k}": float(ps[k].mean()) for k in
               ("ccc_valence", "ccc_arousal", "mean_ccc", "mean_mae", "mean_rmse")}
        rec.update(agg)
        if cond["condition"] == "clean":
            rec.update(quadrant_metrics(yte_raw, pred_native))
        rows_rob.append(rec)
        if cond["condition"] == "clean":
            rows_fold.append(rec)
            for _, r in ps.iterrows():
                d = {"fold": f.name, "seed": f.seed, "k": f.fold, "model": model_name,
                     "condition": "clean"}
                d.update(r.to_dict())
                rows_subj.append(d)
        else:
            for _, r in ps.iterrows():
                d = {"fold": f.name, "seed": f.seed, "k": f.fold, "model": model_name,
                     "condition": cond["condition"], "family": cond["family"],
                     "level": cond["level"]}
                d.update(r.to_dict())
                rows_subj.append(d)
        if store_predictions:
            preds_out.append(pd.DataFrame({
                "fold": f.name, "model": model_name, "condition": cond["condition"],
                "subject_id": subj_te,
                "trial_id": meta["trial_id"].to_numpy()[idx_te],
                "start_s": meta["start_s"].to_numpy()[idx_te],
                "valence_true": yte_raw[:, 0], "arousal_true": yte_raw[:, 1],
                "valence_pred": pred_native[:, 0], "arousal_pred": pred_native[:, 1],
            }))

    # ---------------- classical models -------------------------------------
    Ftr = featmod.extract(Xtr_raw); Fva = featmod.extract(Xva_raw)
    fs = RobustChannelScaler()
    # treat the flattened feature vector as a single "channel" axis of length F
    fs.center_ = np.median(Ftr, axis=0)
    fs.scale_ = np.maximum(np.percentile(Ftr, 75, axis=0) - np.percentile(Ftr, 25, axis=0), 1e-6)
    def fscale(a): return ((a - fs.center_) / fs.scale_).astype(np.float32)

    clean_cond = conditions[0]
    naive = NaivePredictor().fit(Ftr, ytr_raw)

    ridge = RidgeModel(alphas=cfg["experiment"]["models"]["ridge"]["alphas"])
    ridge.select(fscale(Ftr), ytr_raw, fscale(Fva), yva_raw)
    ridge.fit(fscale(Ftr), ytr_raw)

    lgbm = None
    if cfg["experiment"]["models"]["lightgbm"]["enabled"]:
        lgbm = LightGBMModel(**cfg["experiment"]["models"]["lightgbm"]).fit(fscale(Ftr), ytr_raw)

    runinfo["ridge_alpha"] = ridge.alpha_

    # ---------------- neural models ----------------------------------------
    trained: Dict[str, nn.Module] = {}
    mdrop_p = float(args.modality_dropout)
    for name, spec in NEURAL_SPECS.items():
        if args.models and name not in args.models:
            continue
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        net = build_net(spec["cls"], n_mod, n_q, cfg, **spec["kw"])
        info = train_neural(
            net, {"xtr": Xtr, "qtr": Qtr, "ytr": ytr_s,
                  "xva": Xva, "qva": Qva, "yva": yva_s},
            cfg, mdrop_p if spec["mdrop"] else 0.0, device, f.seed,
            CKPT / f"{f.name}_{name}.pt")
        info["train_seconds"] = round(time.time() - t0, 2)
        info["modality_dropout"] = mdrop_p if spec["mdrop"] else 0.0
        if device.type == "cuda":
            info["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        runinfo[name] = info
        trained[name] = net
        LOG.info("  %-20s val_ccc=%.4f ep=%d/%d %.1fs params=%d",
                 name, info["best_val_mean_ccc"], info["best_epoch"],
                 info["epochs_run"], info["train_seconds"], info["n_parameters"])

    # ---------------- evaluation under every condition ----------------------
    for cond in conditions:
        # 1) corrupt raw test signals (sensor-level fault), 2) then scale
        xr = Xte_raw
        if cond["corrupt"] is not None:
            cc = cond["corrupt"]
            xr = corrupt_signals(Xte_raw, cc["kind"], cc["severity"],
                                 cond["seed"] + int(cond.get("draw", 0)),
                                 ref_std, channels=np.asarray(cc["channels"]))
        mask = fixed_missing_mask(len(idx_te), n_mod, cond["missing"])

        # Quality indicators are RECOMPUTED from the corrupted signal: that is
        # the whole point of the gate -- it must notice the degradation itself,
        # with no privileged knowledge of which condition is active.
        Xte_s = xs.transform(xr)
        Qte_s = qs.transform(compute_quality(xr))
        Xte_m, Qte_m = apply_mask_np(Xte_s, Qte_s, mask)

        Fte = featmod.extract(xr)
        Fte_m = featmod.apply_modality_mask(fscale(Fte), mask, n_mod)

        is_clean = cond["condition"] == "clean"
        record("naive", cond, naive.predict(Fte_m), store_predictions=is_clean)
        record("ridge", cond, ridge.predict(Fte_m), store_predictions=is_clean)
        if lgbm is not None:
            record("lightgbm", cond, lgbm.predict(Fte_m), store_predictions=is_clean)

        for name, net in trained.items():
            p_s = predict_neural(net, Xte_m, Qte_m, mask, device)
            record(name, cond, ts.inverse_transform(p_s), store_predictions=is_clean)

    runinfo["scalers"] = {"signal": xs.state(), "target": ts.state()}
    runinfo["fold_seconds"] = round(time.time() - t_fold, 1)
    return {"fold_rows": rows_fold, "subj_rows": rows_subj, "rob_rows": rows_rob,
            "preds": preds_out, "runinfo": {f.name: runinfo}}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--modality-dropout", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=None, help="limit number of folds")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--models", type=str, nargs="*", default=None)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    cfg = load_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = modality_names(cfg)

    X, Q, y, meta = load_windows(args.window, args.delay)
    LOG.info("windows: X=%s Q=%s y=%s subjects=%d", X.shape, Q.shape, y.shape,
             meta["subject_id"].nunique())

    seeds = args.seeds or cfg["experiment"]["seeds"]
    p = cfg["experiment"]["protocol"]
    folds = make_folds(sorted(meta["subject_id"].unique().tolist()), seeds,
                       p["n_folds"], p["inner_validation"]["val_participant_fraction"])

    audit = audit_folds(meta, folds)
    write_json(audit, REPO_ROOT / "manifests" / f"leakage_audit_{args.tag}.json")
    if not audit["passed"]:
        LOG.error("LEAKAGE AUDIT FAILED: %s", audit["issues"][:10])
        return 2
    LOG.info("leakage audit PASSED for %d folds", len(folds))

    if args.folds:
        folds = folds[:args.folds]
    conditions = build_conditions(cfg, mods)
    if args.pilot:
        conditions = [c for c in conditions if c["condition"] in
                      ("clean", "missing1_d0", "missing4_d0", "gaussian_noise_s1_d0")]
    LOG.info("running %d folds x %d conditions", len(folds), len(conditions))
    write_json(conditions, REPO_ROOT / "manifests" / f"corruption_manifest_{args.tag}.json")

    all_fold, all_subj, all_rob, all_pred = [], [], [], []
    runinfos: Dict[str, Any] = {}
    t0 = time.time()
    for i, f in enumerate(folds, 1):
        LOG.info("[%d/%d] fold %s | train=%d val=%d test=%s",
                 i, len(folds), f.name, len(f.train_subjects), len(f.val_subjects),
                 list(f.test_subjects))
        out = run_fold(f, X, Q, y, meta, cfg, device, conditions, args)
        all_fold += out["fold_rows"]; all_subj += out["subj_rows"]
        all_rob += out["rob_rows"]; all_pred += out["preds"]
        runinfos.update(out["runinfo"])

    RESULTS.mkdir(parents=True, exist_ok=True)
    suf = "" if args.tag == "main" else f"_{args.tag}"
    pd.DataFrame(all_fold).to_csv(RESULTS / f"fold_metrics{suf}.csv", index=False)
    pd.DataFrame(all_subj).to_csv(RESULTS / f"subject_metrics{suf}.csv", index=False)
    pd.DataFrame(all_rob).to_csv(RESULTS / f"robustness_metrics{suf}.csv", index=False)
    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_parquet(
            RESULTS / f"predictions{suf}.parquet", index=False)

    manifest = {
        "tag": args.tag,
        "window_seconds": args.window, "delay_seconds": args.delay,
        "modality_dropout": args.modality_dropout,
        "seeds": seeds, "n_folds": len(folds), "n_conditions": len(conditions),
        "modalities": mods,
        "protocol": cfg["experiment"]["protocol"],
        "environment": capture_environment(),
        "leakage_audit_passed": audit["passed"],
        "total_seconds": round(time.time() - t0, 1),
        "runs": runinfos,
        "folds": [asdict(f) for f in folds],
    }
    write_json(manifest, RESULTS / f"experiment_manifest{suf}.json")
    LOG.info("DONE in %.1f min -> results/*%s.csv", (time.time() - t0) / 60, suf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
