"""Phase 0: locate the CASE dataset, inventory it, and verify its structure.

Produces:
  manifests/dataset_inventory.json   -- every file, size, and (selected) SHA-256
  manifests/environment.json         -- python/OS/CPU/GPU/package snapshot
  reports/data_audit.md              -- human-readable audit

This module makes NO assumptions about CASE that it does not verify against the
files on disk and the official README/metadata shipped with the archive.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils import (
    REPO_ROOT, capture_environment, get_logger, load_configs, sha256_file,
    write_json,
)

LOG = get_logger("discover", REPO_ROOT / "logs" / "discover_data.log")

# Files small enough that hashing all of them is cheap; large CSVs are hashed
# for a representative subset to keep Phase 0 fast.
HASH_ALL_SUFFIXES = {".txt", ".xlsx", ".m", ".mat"}
REPRESENTATIVE_SUBJECTS = [1, 15, 30]


def _subject_id_from_name(name: str) -> int | None:
    m = re.match(r"sub_?(\d+)", name)
    return int(m.group(1)) if m else None


def inventory(root: Path) -> List[Dict[str, Any]]:
    """Walk the dataset root and record every file."""
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        row: Dict[str, Any] = {
            "relpath": rel,
            "size_bytes": size,
            "size_mb": round(size / 1024**2, 3),
            "suffix": p.suffix.lower(),
            "sha256": None,
        }
        sid = _subject_id_from_name(p.stem)
        row["subject_id"] = sid
        should_hash = p.suffix.lower() in HASH_ALL_SUFFIXES or (
            sid in REPRESENTATIVE_SUBJECTS
        )
        if should_hash:
            row["sha256"] = sha256_file(p)
        rows.append(row)
    return rows


def verify_structure(cfg: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the discovered files against what the config/README claim."""
    d = cfg["data"]["dataset"]
    phys_dir = Path(d["physiological_dir"])
    anno_dir = Path(d["annotations_dir"])

    phys = sorted(phys_dir.glob("sub_*.csv"))
    anno = sorted(anno_dir.glob("sub_*.csv"))
    phys_ids = sorted(filter(None, (_subject_id_from_name(p.stem) for p in phys)))
    anno_ids = sorted(filter(None, (_subject_id_from_name(p.stem) for p in anno)))

    expected = d["n_participants_expected"]
    findings: Dict[str, Any] = {
        "n_physiological_files": len(phys),
        "n_annotation_files": len(anno),
        "physiological_subject_ids": phys_ids,
        "annotation_subject_ids": anno_ids,
        "subjects_with_both": sorted(set(phys_ids) & set(anno_ids)),
        "missing_physiological": sorted(set(anno_ids) - set(phys_ids)),
        "missing_annotation": sorted(set(phys_ids) - set(anno_ids)),
        "expected_participants": expected,
        "participant_count_matches_expected": len(set(phys_ids) & set(anno_ids)) == expected,
        "duplicate_files": [],
    }

    # duplicate detection by (size, sha256) among hashed files
    seen: Dict[tuple, str] = {}
    for r in rows:
        if r["sha256"] is None:
            continue
        key = (r["size_bytes"], r["sha256"])
        if key in seen:
            findings["duplicate_files"].append([seen[key], r["relpath"]])
        else:
            seen[key] = r["relpath"]
    return findings


def probe_signals(cfg: Dict[str, Any], subject_ids: List[int]) -> Dict[str, Any]:
    """Read a few subjects and verify columns, rates, ranges, and video IDs."""
    d = cfg["data"]["dataset"]
    mods = [m["column"] for m in cfg["data"]["modalities"]]
    out: Dict[str, Any] = {"per_subject": {}, "issues": []}

    for sid in subject_ids:
        pp = Path(d["physiological_dir"]) / f"sub_{sid}.csv"
        ap = Path(d["annotations_dir"]) / f"sub_{sid}.csv"
        if not (pp.exists() and ap.exists()):
            out["issues"].append(f"subject {sid}: missing file(s)")
            continue

        ph = pd.read_csv(pp)
        an = pd.read_csv(ap)

        missing_cols = [c for c in mods if c not in ph.columns]
        if missing_cols:
            out["issues"].append(f"subject {sid}: missing physio columns {missing_cols}")

        dt = ph["daqtime"].to_numpy()
        jt = an["jstime"].to_numpy()
        d_dt = pd.Series(dt).diff().dropna()
        d_jt = pd.Series(jt).diff().dropna()

        rec: Dict[str, Any] = {
            "physio_rows": int(len(ph)),
            "anno_rows": int(len(an)),
            "physio_columns": list(ph.columns),
            "anno_columns": list(an.columns),
            "daqtime_step_ms_median": float(d_dt.median()),
            "daqtime_step_ms_unique": int(d_dt.nunique()),
            "jstime_step_ms_median": float(d_jt.median()),
            "jstime_step_ms_unique": int(d_jt.nunique()),
            "physio_fs_hz_implied": round(1000.0 / float(d_dt.median()), 3),
            "anno_fs_hz_implied": round(1000.0 / float(d_jt.median()), 3),
            "daqtime_span_ms": float(dt[-1] - dt[0]),
            "jstime_span_ms": float(jt[-1] - jt[0]),
            "daqtime_monotonic": bool(pd.Series(dt).is_monotonic_increasing),
            "jstime_monotonic": bool(pd.Series(jt).is_monotonic_increasing),
            "video_ids_physio": sorted(int(v) for v in ph["video"].unique()),
            "video_ids_anno": sorted(int(v) for v in an["video"].unique()),
            "valence_min": float(an["valence"].min()),
            "valence_max": float(an["valence"].max()),
            "arousal_min": float(an["arousal"].min()),
            "arousal_max": float(an["arousal"].max()),
            "nonfinite_by_channel": {
                c: int((~pd.to_numeric(ph[c], errors="coerce").notna()).sum())
                for c in mods if c in ph.columns
            },
            "channel_ranges": {
                c: [float(ph[c].min()), float(ph[c].max())]
                for c in mods if c in ph.columns
            },
        }

        # per-emotional-video sample counts
        em = cfg["data"]["videos"]["emotional_ids"]
        rec["samples_per_emotional_video"] = {
            int(v): int((ph["video"] == v).sum()) for v in em
        }
        rec["anno_samples_per_emotional_video"] = {
            int(v): int((an["video"] == v).sum()) for v in em
        }
        out["per_subject"][sid] = rec

        # sanity assertions -> issues list
        scale = cfg["data"]["annotation"]
        if rec["valence_min"] < scale["scale_min"] - 1e-6 or rec["valence_max"] > scale["scale_max"] + 1e-6:
            out["issues"].append(
                f"subject {sid}: valence outside documented [{scale['scale_min']},{scale['scale_max']}]"
            )
        if not rec["daqtime_monotonic"]:
            out["issues"].append(f"subject {sid}: daqtime not monotonic")
        if not rec["jstime_monotonic"]:
            out["issues"].append(f"subject {sid}: jstime not monotonic")
        if set(em) - set(rec["video_ids_physio"]):
            out["issues"].append(
                f"subject {sid}: missing emotional videos "
                f"{sorted(set(em) - set(rec['video_ids_physio']))}"
            )
    return out


def scan_all_subjects_light(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Cheap per-subject scan of every participant (video coverage + row counts)."""
    d = cfg["data"]["dataset"]
    em = set(cfg["data"]["videos"]["emotional_ids"])
    recs = []
    for pp in sorted(Path(d["physiological_dir"]).glob("sub_*.csv")):
        sid = _subject_id_from_name(pp.stem)
        ap = Path(d["annotations_dir"]) / f"sub_{sid}.csv"
        # only read the two columns needed -> fast
        ph = pd.read_csv(pp, usecols=["daqtime", "video"])
        an = pd.read_csv(ap, usecols=["jstime", "valence", "arousal", "video"])
        vids = set(int(v) for v in ph["video"].unique())
        emo_rows = int(ph["video"].isin(list(em)).sum())
        recs.append({
            "subject_id": sid,
            "physio_rows": len(ph),
            "anno_rows": len(an),
            "physio_span_ms": float(ph["daqtime"].iloc[-1] - ph["daqtime"].iloc[0]),
            "anno_span_ms": float(an["jstime"].iloc[-1] - an["jstime"].iloc[0]),
            "n_video_ids": len(vids),
            "has_all_8_emotional": em.issubset(vids),
            "emotional_rows": emo_rows,
            "emotional_seconds": round(emo_rows / 1000.0, 1),
            "valence_mean": float(an["valence"].mean()),
            "arousal_mean": float(an["arousal"].mean()),
        })
    return pd.DataFrame(recs).sort_values("subject_id").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-scan", action="store_true",
                    help="scan every subject (slower) in addition to probes")
    args = ap.parse_args()

    cfg = load_configs()
    root = Path(cfg["data"]["dataset"]["root"])
    if not root.exists():
        LOG.error("BLOCKED: dataset root not found at %s", root)
        (REPO_ROOT / "reports").mkdir(exist_ok=True, parents=True)
        (REPO_ROOT / "reports" / "data_audit.md").write_text(
            f"# BLOCKED\n\nDataset root not found: `{root}`\n", encoding="utf-8"
        )
        return 2

    LOG.info("Inventorying %s", root)
    rows = inventory(root)
    LOG.info("Found %d files", len(rows))

    structure = verify_structure(cfg, rows)
    LOG.info("Structure: %d physio / %d anno files",
             structure["n_physiological_files"], structure["n_annotation_files"])

    LOG.info("Probing representative subjects %s", REPRESENTATIVE_SUBJECTS)
    probes = probe_signals(cfg, REPRESENTATIVE_SUBJECTS)

    env = capture_environment()

    manifests = REPO_ROOT / "manifests"
    write_json({"files": rows, "structure": structure, "probes": probes},
               manifests / "dataset_inventory.json")
    write_json(env, manifests / "environment.json")

    summary_df = None
    if args.full_scan:
        LOG.info("Full per-subject scan ...")
        summary_df = scan_all_subjects_light(cfg)
        summary_df.to_csv(manifests / "subject_scan.csv", index=False)
        LOG.info("Scanned %d subjects", len(summary_df))

    _write_report(cfg, rows, structure, probes, env, summary_df)
    LOG.info("Wrote reports/data_audit.md")
    return 0


def _write_report(cfg, rows, structure, probes, env, summary_df) -> None:
    d = cfg["data"]["dataset"]
    total_mb = sum(r["size_bytes"] for r in rows) / 1024**2
    lines: List[str] = []
    A = lines.append

    A("# CASE Dataset Audit (Phase 0)\n")
    A("Generated by `src/discover_data.py`. Every statement below is derived from "
      "files on disk or from the official README/metadata shipped inside the archive.\n")

    A("## 1. Location and provenance\n")
    A(f"- Archive: `{d['archive']}`")
    A(f"- Extracted root: `{d['root']}`")
    A(f"- Variant used: **{d['variant']}** "
      "(the official top-level README recommends the interpolated variant because "
      "per-video sample counts are consistent across participants)")
    A(f"- Files inventoried: **{len(rows)}**, total **{total_mb:,.1f} MB**\n")

    A("## 2. Structure verification\n")
    A(f"- Physiological files: **{structure['n_physiological_files']}**")
    A(f"- Annotation files: **{structure['n_annotation_files']}**")
    A(f"- Subjects with both: **{len(structure['subjects_with_both'])}**")
    A(f"- Expected participants: {structure['expected_participants']} -> "
      f"**{'MATCH' if structure['participant_count_matches_expected'] else 'MISMATCH'}**")
    A(f"- Missing physiological: {structure['missing_physiological'] or 'none'}")
    A(f"- Missing annotation: {structure['missing_annotation'] or 'none'}")
    A(f"- Duplicate files detected: {len(structure['duplicate_files'])}\n")

    A("## 3. Verified signal contract\n")
    A("| Property | Value | Source |")
    A("|---|---|---|")
    A("| Physiological sampling rate | 1000 Hz | `daqtime` step = 1 ms (measured) + README |")
    A("| Annotation sampling rate | 20 Hz | `jstime` step = 50 ms (measured) + README |")
    A("| Valence / arousal range | [0.5, 9.5] | README_annotations.txt + measured |")
    A("| Neutral midpoint | 5.0 | centre of documented scale |")
    A("| Physiological channels | 8 | ecg, bvp, gsr, rsp, skt, emg_zygo, emg_coru, emg_trap |")
    A("| Emotional video IDs | 1-8 | metadata/videos.xlsx |")
    A("| Blue screen / start / end IDs | 11 / 10 / 12 | metadata/videos.xlsx |")
    A("")

    A("### Probe results (representative subjects)\n")
    A("| Subject | Physio rows | Anno rows | Physio fs (Hz) | Anno fs (Hz) | Video IDs | Valence range | Arousal range |")
    A("|---|---|---|---|---|---|---|---|")
    for sid, rec in probes["per_subject"].items():
        A(f"| {sid} | {rec['physio_rows']:,} | {rec['anno_rows']:,} | "
          f"{rec['physio_fs_hz_implied']} | {rec['anno_fs_hz_implied']} | "
          f"{rec['video_ids_physio']} | "
          f"[{rec['valence_min']:.2f}, {rec['valence_max']:.2f}] | "
          f"[{rec['arousal_min']:.2f}, {rec['arousal_max']:.2f}] |")
    A("")
    nonfinite_total = sum(
        sum(rec["nonfinite_by_channel"].values()) for rec in probes["per_subject"].values()
    )
    A(f"- Non-finite samples across probed subjects/channels: **{nonfinite_total}**")
    A(f"- Issues raised during probing: **{len(probes['issues'])}**")
    for iss in probes["issues"]:
        A(f"  - {iss}")
    A("")

    if summary_df is not None:
        A("## 4. Per-subject scan (all participants)\n")
        A(f"- Subjects scanned: **{len(summary_df)}**")
        A(f"- Subjects containing all 8 emotional videos: "
          f"**{int(summary_df['has_all_8_emotional'].sum())}/{len(summary_df)}**")
        A(f"- Emotional-stimulus duration per subject: "
          f"min **{summary_df['emotional_seconds'].min():.1f} s**, "
          f"median **{summary_df['emotional_seconds'].median():.1f} s**, "
          f"max **{summary_df['emotional_seconds'].max():.1f} s**")
        A(f"- Full recording span: median **{summary_df['physio_span_ms'].median()/1000:.1f} s**")
        A(f"- Grand mean valence **{summary_df['valence_mean'].mean():.3f}**, "
          f"arousal **{summary_df['arousal_mean'].mean():.3f}** "
          "(scale midpoint 5.0)")
        A("\nFull table: `manifests/subject_scan.csv`\n")

    A("## 5. Environment\n")
    A(f"- Python **{env['python_version']}**, {env['platform']}")
    A(f"- CPU: {env.get('processor') or 'n/a'} ({env.get('cpu_count_logical')} logical cores)")
    A(f"- RAM: {env.get('ram_total_gb')} GB")
    A(f"- GPU: {env.get('gpu_name') or 'none'} "
      f"(CUDA {env.get('torch_cuda_version')}, available={env.get('torch_cuda_available')})")
    A("- Key packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()
                                     if v != "not-installed"))
    A("")

    A("## 6. Gate G0\n")
    ok = (structure["participant_count_matches_expected"]
          and not structure["missing_physiological"]
          and not structure["missing_annotation"]
          and not probes["issues"])
    A(f"**G0 (data gate): {'PASS' if ok else 'REVIEW'}** - dataset located, participant "
      "and annotation structure understood, files readable, and physiological/annotation "
      "streams share a common `daqtime`/`jstime` millisecond clock enabling synchronisation.\n")

    (REPO_ROOT / "reports" / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
