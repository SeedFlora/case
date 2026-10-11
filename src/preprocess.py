"""Phase 2: modality-specific preprocessing and trial segmentation.

For each participant and each of the 8 emotional stimuli we:
  1. slice the physiological stream by the official `video` ID column,
  2. apply a MODALITY-SPECIFIC zero-phase filter at the native 1000 Hz,
  3. for sEMG only, rectify and low-pass to obtain a linear envelope BEFORE
     decimation (20-450 Hz sEMG power cannot survive decimation to 64 Hz),
  4. decimate to a common 64 Hz analysis grid with `resample_poly`, whose
     polyphase FIR provides the anti-aliasing filter,
  5. keep the 20 Hz annotation stream in native resolution so that annotation
     delay can be swept later without recomputing signals.

Output: one compressed .npz per participant in cache/preprocessed/.
No normalisation happens here -- scaling is fold-local (see src/splits.py).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from utils import REPO_ROOT, get_logger, load_configs, write_json

LOG = get_logger("preprocess", REPO_ROOT / "logs" / "preprocess.log")

CACHE = REPO_ROOT / "cache" / "preprocessed"
FS_RAW = 1000
FS_TARGET = 64
# 64/1000 = 8/125 exactly -> integer polyphase resampling, no drift.
RESAMPLE_UP, RESAMPLE_DOWN = 8, 125


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def _design(spec: Dict[str, Any], fs: int):
    nyq = fs / 2.0
    order = int(spec.get("order", 4))
    if spec["type"] == "bandpass":
        lo = spec["low_hz"] / nyq
        hi = min(spec["high_hz"] / nyq, 0.99)
        return butter(order, [lo, hi], btype="bandpass", output="sos")
    if spec["type"] == "lowpass":
        return butter(order, min(spec["high_hz"] / nyq, 0.99), btype="lowpass", output="sos")
    if spec["type"] == "highpass":
        return butter(order, spec["low_hz"] / nyq, btype="highpass", output="sos")
    raise ValueError(f"unknown filter type {spec['type']}")


def process_channel(x: np.ndarray, mod: Dict[str, Any]) -> np.ndarray:
    """Filter one channel at 1000 Hz, optionally envelope it, then decimate."""
    x = np.asarray(x, dtype=np.float64)

    # Guard: replace any non-finite sample by linear interpolation so the
    # filter does not propagate NaNs across the whole trial. The *fraction* of
    # such samples is recorded separately and drives window exclusion.
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(x.size)
        good = ~bad
        x = np.interp(idx, idx[good], x[good]) if good.any() else np.zeros_like(x)

    sos = _design(mod["filter"], FS_RAW)
    # padlen must stay below the segment length; trials are >100 k samples so
    # the default is always safe, but we clamp defensively.
    padlen = min(3 * (2 * len(sos) + 1), max(x.size - 1, 1))
    y = sosfiltfilt(sos, x, padlen=padlen)

    if mod.get("envelope", False):
        # RMS envelope: sqrt(lowpass(x^2)). Unlike rectify-then-lowpass, this is
        # non-negative by construction. A low-pass rings around the sharp bursts
        # that characterise sEMG, and on a rectified signal that ringing dips
        # below zero -- physically meaningless for an amplitude envelope. Here
        # any ringing lands in the POWER domain, where a negative value is
        # unambiguously an artifact and is clipped before the square root.
        sos_env = butter(4, mod["envelope_lowpass_hz"] / (FS_RAW / 2.0),
                         btype="lowpass", output="sos")
        power = sosfiltfilt(sos_env, y * y, padlen=padlen)
        y = np.sqrt(np.clip(power, 0.0, None))

    y = resample_poly(y, RESAMPLE_UP, RESAMPLE_DOWN)

    if mod.get("envelope", False):
        # The envelope is already band-limited to 8 Hz, so polyphase resampling
        # to 64 Hz introduces only negligible ringing; clip residuals at zero.
        y = np.clip(y, 0.0, None)

    return y.astype(np.float32)


# --------------------------------------------------------------------------- #
# per-subject worker
# --------------------------------------------------------------------------- #
def preprocess_subject(sid: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg["data"]["dataset"]
    mods: List[Dict[str, Any]] = cfg["data"]["modalities"]
    emotional = cfg["data"]["videos"]["emotional_ids"]

    ph = pd.read_csv(Path(d["physiological_dir"]) / f"sub_{sid}.csv")
    an = pd.read_csv(Path(d["annotations_dir"]) / f"sub_{sid}.csv")

    store: Dict[str, np.ndarray] = {}
    meta: Dict[str, Any] = {"subject_id": sid, "fs_target": FS_TARGET, "trials": {}}

    for vid in emotional:
        pm = ph["video"] == vid
        am = an["video"] == vid
        if pm.sum() == 0 or am.sum() == 0:
            meta["trials"][str(vid)] = {"status": "missing"}
            continue

        seg = ph.loc[pm]
        aseg = an.loc[am]
        t0 = float(seg["daqtime"].iloc[0])

        # per-channel non-finite fraction on the RAW segment (pre-repair)
        nonfinite_frac = {}
        chans = []
        for m in mods:
            raw = pd.to_numeric(seg[m["column"]], errors="coerce").to_numpy(dtype=np.float64)
            nonfinite_frac[m["name"]] = float((~np.isfinite(raw)).mean())
            chans.append(process_channel(raw, m))

        n = min(c.size for c in chans)
        sig = np.stack([c[:n] for c in chans], axis=0)          # [8, T64]

        store[f"sig_{vid}"] = sig
        store[f"annt_{vid}"] = aseg["jstime"].to_numpy(dtype=np.float64)
        store[f"annv_{vid}"] = aseg["valence"].to_numpy(dtype=np.float32)
        store[f"anna_{vid}"] = aseg["arousal"].to_numpy(dtype=np.float32)

        meta["trials"][str(vid)] = {
            "status": "ok",
            "t0_ms": t0,
            "n_raw_samples": int(pm.sum()),
            "n_target_samples": int(n),
            "duration_s": round(n / FS_TARGET, 3),
            "n_anno_samples": int(am.sum()),
            "nonfinite_fraction": nonfinite_frac,
            "anno_valence_mean": float(aseg["valence"].mean()),
            "anno_arousal_mean": float(aseg["arousal"].mean()),
        }

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"sub_{sid}.npz"
    np.savez_compressed(out, **store)
    meta["cache_file"] = out.name
    meta["cache_size_mb"] = round(out.stat().st_size / 1024**2, 3)
    return meta


def _worker(args):
    sid, cfg = args
    try:
        return preprocess_subject(sid, cfg)
    except Exception as exc:  # surfaced, never silently swallowed
        return {"subject_id": sid, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_configs()
    d = cfg["data"]["dataset"]
    sids = args.subjects or sorted(
        int(p.stem.split("_")[1]) for p in Path(d["physiological_dir"]).glob("sub_*.csv")
    )
    LOG.info("Preprocessing %d subjects with %d workers -> %s", len(sids), args.workers, CACHE)

    metas: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (s, cfg)): s for s in sids}
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            metas.append(m)
            if "error" in m:
                LOG.error("subject %s FAILED: %s", m["subject_id"], m["error"])
            else:
                LOG.info("[%d/%d] subject %s -> %s trials, %.1f MB",
                         i, len(sids), m["subject_id"],
                         sum(1 for t in m["trials"].values() if t.get("status") == "ok"),
                         m["cache_size_mb"])

    metas.sort(key=lambda m: m["subject_id"])
    failed = [m for m in metas if "error" in m]
    ok = [m for m in metas if "error" not in m]

    manifest = {
        "n_subjects": len(metas),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failed": failed,
        "fs_target_hz": FS_TARGET,
        "resample_ratio": f"{RESAMPLE_UP}/{RESAMPLE_DOWN}",
        "modalities": [m["name"] for m in cfg["data"]["modalities"]],
        "filters": {m["name"]: {"filter": m["filter"],
                                "envelope": m.get("envelope", False),
                                "envelope_lowpass_hz": m.get("envelope_lowpass_hz")}
                    for m in cfg["data"]["modalities"]},
        "total_cache_mb": round(sum(m.get("cache_size_mb", 0) for m in ok), 2),
        "subjects": metas,
    }
    write_json(manifest, REPO_ROOT / "manifests" / "preprocess_manifest.json")
    LOG.info("Done. %d ok / %d failed. Cache %.1f MB",
             len(ok), len(failed), manifest["total_cache_mb"])
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
