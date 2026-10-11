"""Validate completeness, selection integrity, budgets and gate diagnostics.

Prediction values and independently recomputed outcome metrics are verified by
verify_predictions.py; this checker validates the experiment's saved structure.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
MODELS = {"naive", "ridge", "lightgbm", "early_fusion", "late_fusion",
          "reliability_gated", "abl_no_quality", "abl_no_moddropout",
          "abl_equal_weights"}
NEURAL = MODELS - {"naive", "ridge", "lightgbm"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read_json(REV / "results/experiment_manifest.json")
    original = read_json(ROOT / "results/experiment_manifest.json")
    assert manifest["complete"] and manifest["n_folds"] == 15
    assert manifest["n_conditions"] == 55
    assert manifest["folds"] == original["folds"]
    assert manifest["leakage_audit_passed"]
    training = manifest["training"]
    assert training["max_epochs"] == 60 and training["patience"] == 12
    for relative, digest in manifest["revision_source_sha256"].items():
        assert hashlib.sha256((REV / relative).read_bytes()).hexdigest() == digest
    for relative, digest in manifest["original_source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

    categories = {name: pd.read_csv(REV / "results" / f"{name}.csv") for name in
                  ("fold_metrics", "robustness_metrics", "subject_metrics", "gate_diagnostics")}
    expected = {"fold_metrics": 135, "robustness_metrics": 7425,
                "subject_metrics": 44550, "gate_diagnostics": 39600}
    for name, frame in categories.items():
        assert len(frame) == expected[name], (name, len(frame))
        assert frame["fold"].nunique() == 15
    for name in ("fold_metrics", "robustness_metrics", "subject_metrics"):
        frame = categories[name]
        assert set(frame.model) == MODELS
        # The two quadrant summaries are intentionally computed for clean
        # evaluation only; native continuous outcomes are saved for all cases.
        numeric = frame.select_dtypes("number")
        if name == "robustness_metrics":
            numeric = numeric.drop(columns=["quadrant_macro_f1", "quadrant_balanced_acc"])
            assert frame.loc[frame.condition == "clean", ["quadrant_macro_f1", "quadrant_balanced_acc"]].notna().all().all()
            assert frame.loc[frame.condition != "clean", ["quadrant_macro_f1", "quadrant_balanced_acc"]].isna().all().all()
        assert np.isfinite(numeric.to_numpy()).all(), name
    subjects = categories["subject_metrics"]
    assert subjects.subject_id.nunique() == 30
    assert not subjects.duplicated(["fold", "model", "condition", "subject_id"]).any()
    assert subjects.groupby(["model", "condition", "subject_id"]).size().eq(3).all()

    selection_scores = pd.read_csv(REV / "results/nested_selection_scores.csv")
    selected_settings = pd.read_csv(REV / "results/selected_settings.csv")
    assert len(selection_scores) == 180 and len(selected_settings) == 15
    assert not selection_scores.duplicated(["fold", "window_s", "delay_s"]).any()
    assert np.isfinite(selection_scores.val_mean_ccc).all()
    prediction_rows = 0
    fits = 0
    for split in manifest["folds"]:
        fold = f"s{split['seed']}_f{split['fold']}"
        train, val, test = map(set, (split["train_subjects"], split["val_subjects"], split["test_subjects"]))
        assert (len(train), len(val), len(test)) == (18, 6, 6)
        assert not train & val and not train & test and not val & test
        assert train | val | test == set(range(1, 31))
        assert read_json(REV / "manifests" / f"complete_{fold}.json")["complete"]
        selection = read_json(REV / "manifests" / f"selection_{fold}.json")
        assert selection["fold"] == split and not selection["test_metrics_used"]
        candidates = selection["candidates"]
        assert {(c["window_s"], c["delay_s"]) for c in candidates} == {
            (w, d) for w in (5., 10., 20.) for d in (0., 1., 2., 3.)}
        selected = selection["selected"]
        assert selected == max(candidates, key=lambda c: c["val_mean_ccc"])
        run = read_json(REV / "manifests" / f"run_{fold}.json")
        assert run["training_settings"] == training
        assert run["seed_used"] == split["seed"] * 1000 + split["fold"]
        assert (run["selected_window_s"], run["selected_delay_s"]) == (
            selected["window_s"], selected["delay_s"])
        gated = run["reliability_gated"]
        assert gated["reused_selected_candidate"]
        assert gated["checkpoint"] == selected["checkpoint"]
        assert gated["epochs_run"] == selected["epochs_run"]
        for candidate in candidates:
            info = read_json((REV / candidate["checkpoint"]).with_suffix(".json"))
            assert (REV / candidate["checkpoint"]).is_file()
            assert 1 <= info["epochs_run"] <= 60
            assert len(info["history"]) == info["epochs_run"]
            assert 0 <= info["best_epoch"] < info["epochs_run"]
            fits += 1
        for name in NEURAL:
            info = run[name]
            assert (REV / info["checkpoint"]).is_file()
            assert 1 <= info["epochs_run"] <= 60
            assert len(info["history"]) == info["epochs_run"]
            assert info["training_seed"] == run["seed_used"]
            assert info["initialization_seed"] == run["seed_used"]
            assert info["modality_dropout"] == (0 if name in {"early_fusion", "late_fusion", "abl_no_moddropout"} else .2)
            if name != "reliability_gated":
                fits += 1
        metrics = categories["robustness_metrics"].query("fold == @fold")
        assert len(metrics) == 495 and metrics.condition.nunique() == 55
        assert metrics.groupby("condition").model.nunique().eq(9).all()
        assert metrics.n_test_windows.eq(selected["n_test_windows"]).all()
        assert set(subjects.query("fold == @fold").subject_id) == test
        nrows = pq.ParquetFile(REV / run["prediction_file"]).metadata.num_rows
        assert nrows == selected["n_test_windows"] * 55 * 9
        prediction_rows += nrows
    assert fits == 255

    gate = categories["gate_diagnostics"]
    assert not gate.duplicated(["fold", "condition", "subject_id", "modality"]).any()
    assert set(gate.role) == {"affected", "healthy", "absent"}
    assert gate.raw_gate_mean.between(0, 1).all()
    assert gate.normalized_weight_mean.between(0, 1).all()
    assert gate.quality_clip_fraction.between(0, 1).all()
    present = gate.role != "absent"
    assert gate.loc[present, "saturation_fraction"].between(0, 1).all()
    assert gate.loc[~present, "saturation_fraction"].isna().all()
    assert gate.loc[~present, "normalized_weight_mean"].eq(0).all()
    assert gate.loc[gate.role == "healthy", "raw_gate_delta_from_clean"].eq(0).all()
    weight_sum_error = float((gate.groupby(["fold", "condition", "subject_id"])
                             .normalized_weight_mean.sum() - 1).abs().max())
    assert weight_sum_error < 2e-5, weight_sum_error
    result = {"passed": True, "n_folds": 15, "n_models": 9,
              "n_named_conditions": 55, "n_candidates": 180,
              "n_neural_fits": fits, "n_prediction_files": 15,
              "n_predictions": prediction_rows, "row_counts": expected,
              "max_normalized_gate_sum_error": weight_sum_error,
              "checks": ["original exact outer splits and disjoint train/validation/test",
                         "all 12 candidate settings and maximum validation selection",
                         "no outer metrics used for selection",
                         "all 60-epoch/patience-12 budgets retained",
                         "selected candidate checkpoint reused and all checkpoints present",
                         "complete nine-model by 55-condition catalogue",
                         "30 participants with three repeated outer tests per scenario",
                         "finite saved outcome metrics and complete parquet row counts",
                         "valid gate normalization, availability and matched healthy raw deltas",
                         "manifest source and configuration fingerprints"]}
    path = REV / "manifests/artifact_validation.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
