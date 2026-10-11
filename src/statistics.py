"""Phase 11: participant-level statistical analysis.

Windows are NOT independent samples: ~240 windows come from each participant and
overlap by 50%. Treating them as independent would inflate the effective sample
size by two orders of magnitude and make trivial differences "significant".
Every test here is therefore paired at the PARTICIPANT level, n = 30.

Confirmatory tests are predeclared (clean + the missing/corrupted conditions
that define the research question) and Holm-corrected as a family. Everything
else is reported as exploratory and is not used to support the paper's claims.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from utils import REPO_ROOT, get_logger, write_json

LOG = get_logger("statistics", REPO_ROOT / "logs" / "statistics.log")
RESULTS = REPO_ROOT / "results"

# Predeclared confirmatory families: the conditions the research question is about.
CONFIRMATORY = ["clean", "missing_random_1", "missing_random_2", "missing_random_4",
                "wearable", "corrupt_all"]


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
    d = diff[diff != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    rp = ranks[d > 0].sum()
    rn = ranks[d < 0].sum()
    total = rp + rn
    return float((rp - rn) / total) if total > 0 else 0.0


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 13) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference, resampling participants."""
    rng = np.random.RandomState(seed)
    n = diff.size
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.randint(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction; returns adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for i, j in enumerate(order):
        val = (n - i) * pvals[j]
        running = max(running, val)
        adj[j] = min(1.0, running)
    return adj.tolist()


def participant_scores(subj: pd.DataFrame, model: str, family: str | None = None,
                       condition: str | None = None, metric: str = "mean_ccc"
                       ) -> pd.Series:
    """Mean metric per participant for one model under one condition family.

    Each participant is held out once per seed, so this averages over seeds and
    over the random draws within a condition family.
    """
    d = subj[subj["model"] == model]
    if condition is not None:
        d = d[d["condition"] == condition]
    elif family is not None:
        d = d[d["family"] == family]
    if d.empty:
        return pd.Series(dtype=float)
    return d.groupby("subject_id")[metric].mean()


def compare(subj: pd.DataFrame, model_a: str, model_b: str, label: str,
            family: str | None = None, condition: str | None = None,
            metric: str = "mean_ccc", kind: str = "confirmatory",
            n_boot: int = 10000) -> Dict[str, Any] | None:
    """Paired participant-level comparison of model_b (proposed) vs model_a (baseline)."""
    sa = participant_scores(subj, model_a, family, condition, metric)
    sb = participant_scores(subj, model_b, family, condition, metric)
    common = sorted(set(sa.index) & set(sb.index))
    if len(common) < 5:
        return None
    a = sa.loc[common].to_numpy(dtype=float)
    b = sb.loc[common].to_numpy(dtype=float)
    diff = b - a

    if np.allclose(diff, 0):
        p, w = 1.0, 0.0
    else:
        w, p = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
    lo, hi = bootstrap_ci(diff, n_boot=n_boot)
    return {
        "comparison": label,
        "kind": kind,
        "baseline_model": model_a,
        "proposed_model": model_b,
        "family": family or "",
        "condition": condition or "",
        "metric": metric,
        "n_participants": len(common),
        "baseline_mean": float(a.mean()),
        "proposed_mean": float(b.mean()),
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci95_low": lo,
        "ci95_high": hi,
        "wilcoxon_W": float(w),
        "p_value": float(p),
        "effect_size_rank_biserial": rank_biserial(diff),
        "n_participants_improved": int((diff > 0).sum()),
    }


def normalise_family(subj: pd.DataFrame) -> pd.DataFrame:
    """Collapse random-draw conditions into analysis families."""
    d = subj.copy()
    if "family" not in d.columns:
        d["family"] = "clean"
    d["family"] = d["family"].fillna("clean")
    fam = d["family"].astype(str)
    lvl = d["level"].fillna(0) if "level" in d.columns else 0
    d["family"] = np.where(fam == "missing_random",
                           "missing_random_" + lvl.astype(int).astype(str), fam)
    d.loc[fam.str.startswith("corrupt_"), "family"] = "corrupt_all"
    d["corruption_type"] = np.where(fam.str.startswith("corrupt_"),
                                    fam.str.replace("corrupt_", "", regex=False), "")
    return d


def run(subject_csv, out_csv, baseline: str | None = None,
        proposed: str = "reliability_gated", n_boot: int = 10000) -> pd.DataFrame:
    subj = normalise_family(pd.read_csv(subject_csv))

    # The "strongest conventional fusion baseline" is chosen from CLEAN results,
    # data-driven and decided only after the experiments finished.
    if baseline is None:
        clean = subj[subj["condition"] == "clean"]
        cand = ["early_fusion", "late_fusion"]
        means = {m: clean[clean["model"] == m]["mean_ccc"].mean() for m in cand}
        means = {k: v for k, v in means.items() if np.isfinite(v)}
        baseline = max(means, key=means.get)
        LOG.info("strongest conventional fusion baseline (clean): %s %s", baseline, means)

    rows: List[Dict[str, Any]] = []

    # ---- confirmatory family -------------------------------------------------
    for fam in CONFIRMATORY:
        r = compare(subj, baseline, proposed, f"{proposed} vs {baseline} [{fam}]",
                    family=fam, kind="confirmatory", n_boot=n_boot)
        if r:
            rows.append(r)

    conf_idx = [i for i, r in enumerate(rows) if r["kind"] == "confirmatory"]
    adj = holm([rows[i]["p_value"] for i in conf_idx])
    for i, a in zip(conf_idx, adj):
        rows[i]["p_value_holm"] = a
        rows[i]["significant_holm_0.05"] = bool(a < 0.05)

    # ---- exploratory ---------------------------------------------------------
    others = [m for m in subj["model"].unique()
              if m not in (proposed,) and isinstance(m, str)]
    for m in others:
        r = compare(subj, m, proposed, f"{proposed} vs {m} [clean]",
                    family="clean", kind="exploratory", n_boot=n_boot)
        if r:
            rows.append(r)

    # per corruption type
    for ct in sorted(x for x in subj["corruption_type"].unique() if x):
        d = subj[subj["corruption_type"] == ct]
        sa = participant_scores(d, baseline, family="corrupt_all")
        sb = participant_scores(d, proposed, family="corrupt_all")
        common = sorted(set(sa.index) & set(sb.index))
        if len(common) < 5:
            continue
        a, b = sa.loc[common].to_numpy(), sb.loc[common].to_numpy()
        diff = b - a
        w, p = (0.0, 1.0) if np.allclose(diff, 0) else stats.wilcoxon(b, a)
        lo, hi = bootstrap_ci(diff, n_boot=n_boot)
        rows.append({
            "comparison": f"{proposed} vs {baseline} [corrupt:{ct}]",
            "kind": "exploratory", "baseline_model": baseline, "proposed_model": proposed,
            "family": f"corrupt_{ct}", "condition": "", "metric": "mean_ccc",
            "n_participants": len(common),
            "baseline_mean": float(a.mean()), "proposed_mean": float(b.mean()),
            "mean_difference": float(diff.mean()),
            "median_difference": float(np.median(diff)),
            "ci95_low": lo, "ci95_high": hi, "wilcoxon_W": float(w), "p_value": float(p),
            "effect_size_rank_biserial": rank_biserial(diff),
            "n_participants_improved": int((diff > 0).sum()),
        })

    # ---- ablation contrasts (exploratory) -----------------------------------
    # diff = proposed - ablation, so a POSITIVE difference means the removed
    # component helps. The 4-missing condition is included because that is where
    # the confirmatory effect appears, and it is the contrast that decides
    # whether robustness comes from the gate or from modality dropout.
    ABL_FAMS = ("clean", "missing_random_2", "missing_random_4", "wearable", "corrupt_all")
    for abl, what in [("abl_no_quality", "quality indicators"),
                      ("abl_no_moddropout", "modality dropout"),
                      ("abl_equal_weights", "learned gating")]:
        for fam in ABL_FAMS:
            r = compare(subj, abl, proposed,
                        f"effect of {what} [{fam}]", family=fam,
                        kind="exploratory", n_boot=n_boot)
            if r:
                rows.append(r)

    # Isolate modality dropout with gating held OFF: equal-weight fusion trained
    # WITH dropout versus plain late fusion trained WITHOUT it.
    for fam in ABL_FAMS:
        r = compare(subj, "late_fusion", "abl_equal_weights",
                    f"effect of modality dropout, no gates [{fam}]", family=fam,
                    kind="exploratory", n_boot=n_boot)
        if r:
            rows.append(r)

    # The overall strongest model on clean data, whatever it is, versus the
    # proposed model under every family.
    clean_all = subj[subj["condition"] == "clean"]
    best_overall = (clean_all.groupby("model")["mean_ccc"].mean()
                    .drop(labels=[proposed], errors="ignore").idxmax())
    for fam in ABL_FAMS:
        r = compare(subj, best_overall, proposed,
                    f"{proposed} vs best overall ({best_overall}) [{fam}]",
                    family=fam, kind="exploratory", n_boot=n_boot)
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    LOG.info("wrote %s (%d tests: %d confirmatory)", out_csv, len(df),
             int((df["kind"] == "confirmatory").sum()))
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-metrics", default=str(RESULTS / "subject_metrics.csv"))
    ap.add_argument("--out", default=str(RESULTS / "statistical_tests.csv"))
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--proposed", default="reliability_gated")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    df = run(args.subject_metrics, args.out, args.baseline, args.proposed, args.n_boot)
    conf = df[df["kind"] == "confirmatory"]
    for _, r in conf.iterrows():
        LOG.info("%-52s d=%+.4f p=%.4g p_holm=%.4g r=%+.3f",
                 r["comparison"], r["mean_difference"], r["p_value"],
                 r.get("p_value_holm", np.nan), r["effect_size_rank_biserial"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
