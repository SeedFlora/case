"""Automated tests for the CASE reliability-gated fusion pipeline.

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from corruption import (build_conditions, corrupt_signals, fixed_missing_mask,  # noqa: E402
                        missing_mask)
from evaluate import ccc, mae, metrics_block, per_subject_metrics, quadrant_labels  # noqa: E402
from features import extract as feat_extract  # noqa: E402
from models.reliability_gated import ReliabilityGatedNet  # noqa: E402
from quality import N_QUALITY_FEATURES, compute_quality  # noqa: E402
from splits import (Fold, QualityScaler, RobustChannelScaler, TargetScaler,  # noqa: E402
                    audit_folds, fold_masks, make_folds)
from utils import load_configs, modality_names  # noqa: E402


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def test_config_loads_and_declares_eight_modalities():
    cfg = load_configs()
    mods = modality_names(cfg)
    assert len(mods) == 8
    assert mods == ["ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"]
    assert cfg["data"]["videos"]["emotional_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_dataset_discovery_found_all_participants():
    inv = ROOT / "manifests" / "dataset_inventory.json"
    if not inv.exists():
        pytest.skip("run src/discover_data.py first")
    import json
    s = json.loads(inv.read_text(encoding="utf-8"))["structure"]
    assert s["n_physiological_files"] == 30
    assert s["n_annotation_files"] == 30
    assert s["participant_count_matches_expected"]
    assert not s["missing_physiological"] and not s["missing_annotation"]


# --------------------------------------------------------------------------- #
# splits / leakage
# --------------------------------------------------------------------------- #
def test_participant_group_split_isolation():
    subs = list(range(1, 31))
    folds = make_folds(subs, [13, 42, 2026], 5, 0.25)
    assert len(folds) == 15
    for f in folds:
        tr, va, te = set(f.train_subjects), set(f.val_subjects), set(f.test_subjects)
        assert not (tr & te), "train/test participant overlap"
        assert not (tr & va), "train/val participant overlap"
        assert not (va & te), "val/test participant overlap"
        assert tr | va | te == set(subs), "partition must cover every participant"


def test_every_participant_tested_once_per_seed():
    subs = list(range(1, 31))
    folds = make_folds(subs, [13, 42, 2026], 5, 0.25)
    counts = {s: 0 for s in subs}
    for f in folds:
        for s in f.test_subjects:
            counts[s] += 1
    assert set(counts.values()) == {3}


def test_audit_detects_injected_leakage():
    meta = pd.DataFrame({
        "subject_id": np.repeat(np.arange(1, 11), 4),
        "trial_id": np.tile([1, 1, 2, 2], 10),
        "start_s": np.tile([0.0, 5.0, 0.0, 5.0], 10),
        "end_s": np.tile([10.0, 15.0, 10.0, 15.0], 10),
    })
    good = make_folds(list(range(1, 11)), [13], 5, 0.25)
    assert audit_folds(meta, good)["passed"]

    bad = list(good)
    f = bad[0]
    leaked = Fold(f.seed, f.fold, tuple(sorted(set(f.train_subjects) | {f.test_subjects[0]})),
                  f.val_subjects, f.test_subjects)
    bad[0] = leaked
    assert not audit_folds(meta, bad)["passed"]


# --------------------------------------------------------------------------- #
# windowing / alignment
# --------------------------------------------------------------------------- #
def test_window_boundaries_do_not_cross_trials():
    p = ROOT / "cache" / "windows" / "meta_w10_d0.parquet"
    if not p.exists():
        pytest.skip("run src/windowing.py first")
    meta = pd.read_parquet(p)
    assert (meta["end_s"] > meta["start_s"]).all()
    assert np.allclose(meta["end_s"] - meta["start_s"], 10.0)
    # within a (subject, trial), starts must be strictly increasing by the stride
    for _, g in meta.groupby(["subject_id", "trial_id"]):
        s = g.sort_values("start_s")["start_s"].to_numpy()
        assert (np.diff(s) > 0).all()
        assert np.allclose(np.diff(s), 5.0)


def test_annotation_alignment_and_coverage():
    p = ROOT / "cache" / "windows" / "meta_w10_d0.parquet"
    if not p.exists():
        pytest.skip("run src/windowing.py first")
    meta = pd.read_parquet(p)
    assert (meta["annotation_coverage"] >= 0.999).all()
    # labels must respect the documented joystick scale
    assert meta["valence"].between(0.5, 9.5).all()
    assert meta["arousal"].between(0.5, 9.5).all()


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
def test_scaler_uses_training_statistics_only():
    rng = np.random.RandomState(0)
    train = rng.randn(50, 3, 40).astype(np.float32)
    test = rng.randn(20, 3, 40).astype(np.float32) * 100.0 + 500.0

    sc = RobustChannelScaler().fit(train)
    before = (sc.center_.copy(), sc.scale_.copy())
    _ = sc.transform(test)
    assert np.allclose(sc.center_, before[0]), "transform must not refit"
    assert np.allclose(sc.scale_, before[1])
    # test data is far off-centre precisely because train statistics were used
    assert np.abs(sc.transform(test)).mean() > 5.0


def test_quality_scaler_handles_constant_features():
    """Regression test: a constant indicator must map to ~0, not explode.

    `finite_ratio` is exactly 1.0 on clean CASE data, so IQR and std are both
    zero. An epsilon floor would divide floating-point dust by 1e-6 and produce
    values of order 1e5, saturating the gate MLP.
    """
    q = np.random.RandomState(1).rand(100, 4, N_QUALITY_FEATURES).astype(np.float32)
    q[:, :, 0] = 1.0          # perfectly constant feature
    sc = QualityScaler().fit(q)
    z = sc.transform(q)
    assert np.isfinite(z).all()
    assert np.abs(z[:, :, 0]).max() < 1e-3, "constant feature must scale to ~0"
    assert np.abs(z).max() <= QualityScaler.CLIP + 1e-6


def test_target_scaler_roundtrip():
    y = np.random.RandomState(2).rand(100, 2).astype(np.float32) * 9 + 0.5
    ts = TargetScaler().fit(y)
    assert np.allclose(ts.inverse_transform(ts.transform(y)), y, atol=1e-4)


# --------------------------------------------------------------------------- #
# quality indicators
# --------------------------------------------------------------------------- #
def test_quality_shapes_and_finiteness():
    x = np.random.RandomState(3).randn(16, 8, 640).astype(np.float32)
    q = compute_quality(x)
    assert q.shape == (16, 8, N_QUALITY_FEATURES)
    assert np.isfinite(q).all()


def test_quality_detects_flatline_and_noise():
    rng = np.random.RandomState(4)
    normal = rng.randn(8, 1, 640).astype(np.float32)
    flat = np.zeros((8, 1, 640), dtype=np.float32)
    noisy = normal + rng.randn(8, 1, 640).astype(np.float32) * 10.0

    qn, qf, qz = compute_quality(normal), compute_quality(flat), compute_quality(noisy)
    # flatline ratio (index 1) must be far higher on the dead signal
    assert qf[:, :, 1].mean() > qn[:, :, 1].mean() + 0.5
    # log-variance (index 3) must rise with injected noise
    assert qz[:, :, 3].mean() > qn[:, :, 3].mean()


# --------------------------------------------------------------------------- #
# masking / gating
# --------------------------------------------------------------------------- #
def test_absent_modality_receives_zero_gate_and_is_excluded():
    torch.manual_seed(0)
    net = ReliabilityGatedNet(n_modalities=4, n_quality=N_QUALITY_FEATURES,
                              enc_channels=8, depth=2)
    x = torch.randn(6, 4, 128)
    q = torch.randn(6, 4, N_QUALITY_FEATURES)
    mask = torch.ones(6, 4)
    mask[:, 2] = 0.0

    g = net.gates(q, mask)
    assert torch.allclose(g[:, 2], torch.zeros(6)), "absent modality must gate to 0"
    assert (g[:, [0, 1, 3]] > 0).all()


def test_absent_modality_cannot_change_prediction():
    """The value on a masked channel must be irrelevant to the output."""
    torch.manual_seed(0)
    net = ReliabilityGatedNet(n_modalities=4, n_quality=N_QUALITY_FEATURES,
                              enc_channels=8, depth=2).eval()
    x = torch.randn(5, 4, 128)
    q = torch.randn(5, 4, N_QUALITY_FEATURES)
    mask = torch.ones(5, 4)
    mask[:, 1] = 0.0
    x1, q1 = x.clone(), q.clone()
    x1[:, 1] = 999.0            # garbage on the absent channel
    q1[:, 1] = -42.0
    # the caller zeroes absent inputs; emulate that contract
    with torch.no_grad():
        a = net(x * mask.unsqueeze(-1), q * mask.unsqueeze(-1), mask)
        b = net(x1 * mask.unsqueeze(-1), q1 * mask.unsqueeze(-1), mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_equal_weight_ablation_has_uniform_gates():
    net = ReliabilityGatedNet(n_modalities=5, n_quality=N_QUALITY_FEATURES,
                              enc_channels=8, depth=2, learned_gates=False)
    q = torch.randn(3, 5, N_QUALITY_FEATURES)
    mask = torch.ones(3, 5); mask[:, 4] = 0
    g = net.gates(q, mask)
    assert torch.allclose(g[:, :4], torch.ones(3, 4))
    assert torch.allclose(g[:, 4], torch.zeros(3))


# --------------------------------------------------------------------------- #
# corruption
# --------------------------------------------------------------------------- #
def test_corruption_is_reproducible_and_non_destructive():
    x = np.random.RandomState(5).randn(10, 4, 256).astype(np.float32)
    ref = x.std(axis=(0, 2))
    orig = x.copy()
    a = corrupt_signals(x, "gaussian_noise", 0.5, 777, ref)
    b = corrupt_signals(x, "gaussian_noise", 0.5, 777, ref)
    assert np.array_equal(a, b), "same seed must give identical corruption"
    assert np.array_equal(x, orig), "input must not be mutated"
    assert not np.array_equal(a, orig), "corruption must actually change the signal"


def test_corruption_only_touches_selected_channels():
    x = np.random.RandomState(6).randn(8, 5, 128).astype(np.float32)
    ref = x.std(axis=(0, 2))
    y = corrupt_signals(x, "gaussian_noise", 1.0, 1, ref, channels=np.array([1, 3]))
    for c in (0, 2, 4):
        assert np.array_equal(y[:, c], x[:, c])
    for c in (1, 3):
        assert not np.array_equal(y[:, c], x[:, c])


@pytest.mark.parametrize("kind", ["gaussian_noise", "flatline", "sample_dropout",
                                  "clipping", "impulse"])
def test_all_corruptions_produce_finite_output(kind):
    x = np.random.RandomState(7).randn(6, 3, 256).astype(np.float32)
    ref = x.std(axis=(0, 2))
    y = corrupt_signals(x, kind, 0.5, 99, ref)
    assert y.shape == x.shape and np.isfinite(y).all()


def test_flatline_actually_flatlines():
    x = np.random.RandomState(8).randn(4, 2, 256).astype(np.float32)
    ref = x.std(axis=(0, 2))
    y = corrupt_signals(x, "flatline", 0.5, 3, ref)
    # the corrupted signal must contain a long run of identical values
    d = np.abs(np.diff(y, axis=-1))
    assert (d < 1e-12).mean() > 0.3


def test_missing_masks_are_deterministic_and_correct_size():
    m1 = missing_mask(10, 8, 3, 777, 0)
    m2 = missing_mask(10, 8, 3, 777, 0)
    assert np.array_equal(m1, m2)
    assert (~m1[0]).sum() == 3
    assert (~missing_mask(10, 8, 0, 777, 0)).sum() == 0
    fixed = fixed_missing_mask(5, 8, [2, 5])
    assert not fixed[:, 2].any() and not fixed[:, 5].any()
    assert fixed[:, [0, 1, 3, 4, 6, 7]].all()


def test_condition_catalogue_is_complete():
    cfg = load_configs()
    conds = build_conditions(cfg, modality_names(cfg))
    names = {c["condition"] for c in conds}
    assert "clean" in names and "wearable_only" in names
    for m in modality_names(cfg):
        assert f"drop_{m}" in names
    assert conds[0]["condition"] == "clean"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_ccc_perfect_and_known_value():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ccc(y, y) == pytest.approx(1.0)
    # a pure shift keeps r = 1 but must reduce CCC
    assert 0.0 < ccc(y, y + 1.0) < 1.0
    # a scale change likewise
    assert 0.0 < ccc(y, 2.0 * y) < 1.0
    # anti-correlated
    assert ccc(y, -y) < 0.0


def test_ccc_matches_reference_formula():
    rng = np.random.RandomState(9)
    a, b = rng.randn(200), rng.randn(200)
    num = 2 * np.cov(a, b, bias=True)[0, 1]
    den = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    assert ccc(a, b) == pytest.approx(num / den, rel=1e-9)


def test_ccc_edge_cases():
    const = np.full(10, 3.0)
    assert ccc(const, const) == pytest.approx(1.0)      # identical constants
    assert ccc(const, np.full(10, 5.0)) == pytest.approx(0.0)  # different constants
    assert np.isnan(ccc(np.array([1.0]), np.array([1.0])))     # too few points
    # NaNs are dropped pairwise
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isfinite(ccc(a, b))


def test_metrics_block_and_selection_score():
    rng = np.random.RandomState(10)
    yt = rng.rand(100, 2) * 9 + 0.5
    yp = yt + rng.randn(100, 2) * 0.1
    m = metrics_block(yt, yp)
    assert m["mean_ccc"] == pytest.approx((m["ccc_valence"] + m["ccc_arousal"]) / 2)
    assert m["mae_valence"] > 0 and m["rmse_valence"] >= m["mae_valence"]


def test_per_subject_metrics_are_participant_local():
    subs = np.repeat([1, 2], 50)
    yt = np.random.RandomState(11).rand(100, 2)
    yp = yt.copy()
    yp[subs == 2] = 0.5                       # participant 2 predicted flat
    ps = per_subject_metrics(subs, yt, yp)
    assert len(ps) == 2
    s1 = ps.loc[ps.subject_id == 1, "mean_ccc"].item()
    s2 = ps.loc[ps.subject_id == 2, "mean_ccc"].item()
    assert s1 > 0.9 and s2 < 0.1


def test_quadrant_labels_use_documented_midpoint():
    y = np.array([[6.0, 6.0], [6.0, 4.0], [4.0, 6.0], [4.0, 4.0]])
    assert list(quadrant_labels(y, 5.0)) == [3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_feature_extraction_shape_and_finiteness():
    x = np.random.RandomState(12).randn(20, 8, 640).astype(np.float32)
    f = feat_extract(x)
    assert f.shape[0] == 20 and f.shape[1] == 8 * 16
    assert np.isfinite(f).all()


def test_features_of_constant_signal_are_finite():
    x = np.zeros((4, 3, 128), dtype=np.float32)
    assert np.isfinite(feat_extract(x)).all()


# --------------------------------------------------------------------------- #
# result artifacts
# --------------------------------------------------------------------------- #
def test_result_tables_exist_and_are_consistent():
    fm = ROOT / "results" / "fold_metrics.csv"
    if not fm.exists():
        pytest.skip("run src/train.py first")
    df = pd.read_csv(fm)
    assert len(df) > 0
    for c in ("fold", "model", "ccc_valence", "ccc_arousal", "mean_ccc"):
        assert c in df.columns
    assert df["mean_ccc"].between(-1.0, 1.0).all()
    assert np.isfinite(df["mean_ccc"]).all()


def test_checkpoint_reload_is_deterministic():
    """Same checkpoint must reproduce identical predictions."""
    torch.manual_seed(0)
    net = ReliabilityGatedNet(n_modalities=3, n_quality=N_QUALITY_FEATURES,
                              enc_channels=8, depth=2).eval()
    x, q = torch.randn(4, 3, 128), torch.randn(4, 3, N_QUALITY_FEATURES)
    mask = torch.ones(4, 3)
    with torch.no_grad():
        a = net(x, q, mask)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.pt"
        torch.save(net.state_dict(), p)
        net2 = ReliabilityGatedNet(n_modalities=3, n_quality=N_QUALITY_FEATURES,
                                   enc_channels=8, depth=2)
        net2.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        net2.eval()
        with torch.no_grad():
            b = net2(x, q, mask)
    assert torch.allclose(a, b, atol=1e-6)
