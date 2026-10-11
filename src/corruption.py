"""Phase 9: reproducible missing-modality and sensor-corruption stress tests.

Design rules enforced here:
  * corruption is applied ONLY to a copy of the held-out test inputs,
  * it is applied in RAW signal units, BEFORE fold-local scaling, because a
    real sensor fault happens at the sensor, not after normalisation,
  * the scaler is never refitted on corrupted data,
  * every condition is driven by a fixed seed so that all competing models see
    bit-identical corrupted inputs.

Corrupting all eight channels equally would leave a reliability gate nothing to
discriminate: if every modality is equally bad, down-weighting cannot help. The
scientifically informative test therefore corrupts a SUBSET of modalities, which
is also the realistic failure mode (one electrode detaches, not all of them).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

CORRUPTION_TYPES = ("gaussian_noise", "flatline", "sample_dropout", "clipping", "impulse")


# --------------------------------------------------------------------------- #
# missing modalities
# --------------------------------------------------------------------------- #
def missing_mask(n_windows: int, n_modalities: int, n_missing: int,
                 seed: int, draw: int = 0) -> np.ndarray:
    """Availability mask [N, C]; True = present.

    One modality subset is drawn per (seed, draw) and applied to EVERY window in
    the condition, which models a sensor that is absent for the whole session
    rather than flickering window to window.
    """
    mask = np.ones((n_windows, n_modalities), dtype=bool)
    if n_missing <= 0:
        return mask
    rng = np.random.RandomState(seed + 9173 * draw)
    drop = rng.choice(n_modalities, size=min(n_missing, n_modalities), replace=False)
    mask[:, drop] = False
    return mask


def fixed_missing_mask(n_windows: int, n_modalities: int,
                       missing_idx: List[int]) -> np.ndarray:
    mask = np.ones((n_windows, n_modalities), dtype=bool)
    if missing_idx:
        mask[:, np.asarray(missing_idx, dtype=int)] = False
    return mask


# --------------------------------------------------------------------------- #
# corruption
# --------------------------------------------------------------------------- #
def corrupt_signals(x: np.ndarray, kind: str, severity: float, seed: int,
                    ref_std: np.ndarray, channels: np.ndarray | None = None,
                    ) -> np.ndarray:
    """Return a corrupted COPY of `x` ([N, C, T]) in raw units.

    `ref_std` is the per-channel standard deviation of the TRAINING signals, so
    noise amplitude is expressed relative to the training signal scale.
    `channels` selects which modalities are affected (default: all).
    """
    if kind not in CORRUPTION_TYPES:
        raise ValueError(f"unknown corruption {kind}")
    out = np.array(x, dtype=np.float32, copy=True)
    n, c, t = out.shape
    ch = np.arange(c) if channels is None else np.asarray(channels, dtype=int)
    rng = np.random.RandomState(seed)
    sd = np.asarray(ref_std, dtype=np.float32).reshape(1, -1, 1)

    if kind == "gaussian_noise":
        noise = rng.randn(n, len(ch), t).astype(np.float32) * (severity * sd[:, ch, :])
        out[:, ch, :] += noise

    elif kind == "flatline":
        # a contiguous dead segment covering `severity` of the window, held at
        # the value present when the sensor stalled (fully vectorised)
        seg = max(1, int(round(severity * t)))
        starts = rng.randint(0, max(1, t - seg + 1), size=(n, len(ch)))
        sub = out[:, ch, :].copy()                                  # [n, K, T]
        vals = np.take_along_axis(sub, starts[:, :, None], axis=2)  # [n, K, 1]
        idx = starts[:, :, None] + np.arange(seg)[None, None, :]    # [n, K, seg]
        np.put_along_axis(sub, idx, np.broadcast_to(vals, idx.shape).copy(), axis=2)
        out[:, ch, :] = sub

    elif kind == "sample_dropout":
        drop = rng.rand(n, len(ch), t) < severity
        sub = out[:, ch, :]
        # dropped samples are held at the last valid value (sensor stall)
        for j in range(len(ch)):
            d = drop[:, j, :]
            s = sub[:, j, :]
            idx = np.where(d, 0, np.arange(t)[None, :])
            idx = np.maximum.accumulate(idx, axis=1)
            sub[:, j, :] = np.take_along_axis(s, idx, axis=1)
        out[:, ch, :] = sub

    elif kind == "clipping":
        # saturate at a quantile of |x| within each window
        q = np.quantile(np.abs(out[:, ch, :]), max(1e-3, 1.0 - severity), axis=-1,
                        keepdims=True)
        out[:, ch, :] = np.clip(out[:, ch, :], -q, q)

    elif kind == "impulse":
        rate = 0.005 * max(severity, 1e-6)
        hit = rng.rand(n, len(ch), t) < rate
        amp = (5.0 * severity) * sd[:, ch, :]
        signs = rng.choice([-1.0, 1.0], size=(n, len(ch), t)).astype(np.float32)
        out[:, ch, :] += hit * signs * amp

    return out


def corruption_channels(n_modalities: int, n_affected: int, seed: int,
                        draw: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed + 4441 * draw)
    return np.sort(rng.choice(n_modalities, size=min(n_affected, n_modalities),
                              replace=False))


# --------------------------------------------------------------------------- #
# condition catalogue
# --------------------------------------------------------------------------- #
def build_conditions(cfg: Dict, modalities: List[str]) -> List[Dict]:
    """Enumerate every robustness condition, with its exact reproduction recipe."""
    rc = cfg["experiment"]["robustness"]
    seed = int(rc["corruption_seed"])
    n_c = len(modalities)
    conds: List[Dict] = [{"condition": "clean", "family": "clean", "level": 0.0,
                          "seed": seed, "missing": [], "corrupt": None}]

    # random missing-modality levels
    for lvl in rc["missing_modality"]["n_missing_levels"]:
        if lvl == 0:
            continue
        for draw in range(int(rc["missing_modality"]["n_random_draws"])):
            m = missing_mask(1, n_c, lvl, seed, draw)[0]
            miss = [i for i in range(n_c) if not m[i]]
            conds.append({
                "condition": f"missing{lvl}_d{draw}", "family": "missing_random",
                "level": float(lvl), "seed": seed, "draw": draw,
                "missing": miss,
                "missing_names": [modalities[i] for i in miss],
                "corrupt": None,
            })

    # each individual modality removed
    if rc["missing_modality"]["per_modality_ablation"]:
        for i, name in enumerate(modalities):
            conds.append({
                "condition": f"drop_{name}", "family": "missing_single",
                "level": 1.0, "seed": seed, "missing": [i],
                "missing_names": [name], "corrupt": None,
            })

    # wearable-only subset
    if rc["missing_modality"]["wearable_subset_only"]:
        keep = set(cfg["data"]["wearable_subset"])
        miss = [i for i, n in enumerate(modalities) if n not in keep]
        conds.append({
            "condition": "wearable_only", "family": "wearable",
            "level": float(len(miss)), "seed": seed, "missing": miss,
            "missing_names": [modalities[i] for i in miss], "corrupt": None,
        })

    # corruption: half the modalities affected, three severities
    n_aff = max(1, n_c // 2)
    for kind in rc["corruption"]["types"]:
        for sev in rc["corruption"]["severities"]:
            for draw in range(2):
                ch = corruption_channels(n_c, n_aff, seed, draw)
                conds.append({
                    "condition": f"{kind}_s{sev:g}_d{draw}", "family": f"corrupt_{kind}",
                    "level": float(sev), "seed": seed, "draw": draw, "missing": [],
                    "corrupt": {"kind": kind, "severity": float(sev),
                                "channels": ch.tolist(),
                                "channel_names": [modalities[i] for i in ch]},
                })
    return conds
