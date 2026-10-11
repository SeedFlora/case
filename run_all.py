#!/usr/bin/env python
"""End-to-end reproduction of the original CASE experiment pipeline.

    python run_all.py                # full pipeline
    python run_all.py --skip-train   # recompute statistics/reports from existing results
    python run_all.py --quick        # 1 seed, reduced conditions (smoke test)

Every stage is a separate module under src/ and can be run independently; this
script only fixes the order and the arguments that produced the reported run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Frozen configuration selected on validation participants only
# (see manifests/sensitivity_selection.json).
WINDOW = "20"
DELAY = "0"
MODALITY_DROPOUT = "0.2"


def run(desc: str, args: list[str], cwd: Path = SRC) -> None:
    print(f"\n{'=' * 78}\n>>> {desc}\n    $ {' '.join(args)}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=str(cwd))
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"STAGE FAILED ({r.returncode}): {desc}")
    print(f"<<< done in {dt/60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true", default=True,
                    help="the window/delay choice is already frozen; re-run "
                         "explicitly with --no-skip-sensitivity to redo it")
    ap.add_argument("--no-skip-sensitivity", dest="skip_sensitivity",
                    action="store_false")
    args = ap.parse_args()

    # --- Phase 0: discovery -------------------------------------------------
    run("Phase 0: dataset discovery and audit", ["discover_data.py", "--full-scan"])

    # --- Phase 2-3: preprocessing and windowing -----------------------------
    if not args.skip_preprocess:
        run("Phase 2: modality-specific preprocessing", ["preprocess.py", "--workers", "6"])
        run("Phase 2: preprocessing validation", ["validate_data.py"])
        run("Phase 3: window generation", ["windowing.py", "--windows", WINDOW,
                                           "--delays", DELAY])

    # --- validation-only sensitivity (optional) -----------------------------
    if not args.skip_sensitivity:
        run("Sensitivity: window length x annotation delay (validation only)",
            ["sensitivity.py", "--folds", "3"])

    # --- Phase 4-10: the experiment grid ------------------------------------
    if not args.skip_train:
        train_args = ["train.py", "--window", WINDOW, "--delay", DELAY,
                      "--modality-dropout", MODALITY_DROPOUT, "--tag", "main"]
        if args.quick:
            train_args += ["--seeds", "13", "--folds", "2"]
        run("Phases 4-10: grouped CV, baselines, proposed model, robustness, ablations",
            train_args)

    # --- Phase 11-12: statistics and sanity checks --------------------------
    run("Phase 11: participant-level statistics", ["statistics.py"])
    run("Phase 12: sanity checks", ["sanity_checks.py", "--window", WINDOW,
                                    "--delay", DELAY])

    # --- reports ------------------------------------------------------------
    run("Reports: experiment report, leakage audit, limitations", ["make_reports.py"])

    # --- tests --------------------------------------------------------------
    run("Test suite", ["-m", "pytest", "tests", "-q"], cwd=ROOT)

    print("\nAll stages complete. Deliverables:")
    for p in ["results/fold_metrics.csv", "results/statistical_tests.csv",
              "reports/experiment_report.md", "reports/leakage_audit.md"]:
        print(f"  {'OK ' if (ROOT / p).exists() else 'MISSING '} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
