"""GT-verified loop precision/recall from synthetic run-dir artifacts."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# `run_lc_parity.py`-style same-dir import: put evals/runners itself on sys.path
# (mirrors tests/evals/test_run_lc_parity.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))

from lc_loop_pr import _load_gt, label_pair, loop_precision_recall, scene_covisibility, scene_diameter


########################################
############ Fixture helpers ###########
########################################


def _dec(qs: int, qf: int, ds: int, df: int, accepted: bool = True) -> dict:
    """One lc_decisions entry in the on-disk schema."""
    return {
        "query_submap": qs,
        "query_frame": qf,
        "detected_submap": ds,
        "detected_frame": df,
        "accepted": accepted,
        "reject_reason": None if accepted else "verify_ratio",
        "l2_score": 0.5,
    }


def _cluster_xs(cluster_centers: list[float], per: int = 8, step: float = 0.1) -> list[float]:
    """Camera x-positions: `per` frames near each cluster center (all looking +z)."""
    return [c + k * step for c in cluster_centers for k in range(per)]


def _write_run_dir(root: Path, xs: list[float], decisions: list[dict], submap_size: int,
                   quats: list[tuple] | None = None) -> Path:
    """Synthetic arm dir: gt.tum (cameras on x-axis), lc_decisions_lc.json, metrics.json."""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, x in enumerate(xs):
        q = quats[i] if quats is not None else (0.0, 0.0, 0.0, 1.0)
        lines.append(f"{float(i):.6f} {x:.6f} 0.0 0.0 {q[0]} {q[1]} {q[2]} {q[3]}")
    (root / "gt.tum").write_text("\n".join(lines) + "\n")
    (root / "lc_decisions_lc.json").write_text(json.dumps(decisions))
    (root / "metrics.json").write_text(json.dumps({"_config": {"submap_size": submap_size}}))
    return root


########################################
########## label_pair geometry #########
########################################


_Z = np.array([0.0, 0.0, 1.0])


def test_label_pair_positive_close_and_aligned():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    assert label_pair(a, _Z, b, _Z, diameter=100.0) == 1


def test_label_pair_negative_far_apart():
    a, b = np.array([0.0, 0.0, 0.0]), np.array([60.0, 0.0, 0.0])
    assert label_pair(a, _Z, b, _Z, diameter=100.0) == -1


def test_label_pair_negative_opposed_views():
    # Same spot, cameras looking opposite ways — not covisible.
    a = np.array([0.0, 0.0, 0.0])
    assert label_pair(a, _Z, a, -_Z, diameter=100.0) == -1


def test_label_pair_ambiguous_distance_band():
    # 42 m apart with 100 m diameter: between 0.35 and 0.5 fractions → excluded.
    a, b = np.array([0.0, 0.0, 0.0]), np.array([42.0, 0.0, 0.0])
    assert label_pair(a, _Z, b, _Z, diameter=100.0) == 0


def test_label_pair_ambiguous_view_angle_band():
    # Close centers, 80° between view dirs: between 75° and 90° → excluded.
    a = np.array([0.0, 0.0, 0.0])
    d80 = np.array([math.sin(math.radians(80.0)), 0.0, math.cos(math.radians(80.0))])
    assert label_pair(a, _Z, a, d80, diameter=100.0) == 0


def test_scene_covisibility_matrix_symmetric():
    centers = np.array([[0.0, 0, 0], [1.0, 0, 0], [60.0, 0, 0]])
    dirs = np.tile(_Z, (3, 1))
    labels = scene_covisibility(centers, dirs)
    assert scene_diameter(centers) == pytest.approx(60.0)
    assert labels[0, 1] == 1 and labels[1, 0] == 1  # 1 m ≤ 0.35 × 60
    assert labels[0, 2] == -1 and labels[2, 0] == -1  # 60 m > 0.5 × 60
    assert (np.diag(labels) == 1).all()


def test_load_gt_view_dir_from_quaternion(tmp_path):
    # 180° about x (qx=1): camera z-axis flips to (0, 0, -1).
    p = tmp_path / "gt.tum"
    p.write_text("0.0 0 0 0 0 0 0 1\n1.0 0 0 0 1 0 0 0\n")
    centers, dirs = _load_gt(p)
    np.testing.assert_allclose(dirs[0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(dirs[1], [0, 0, -1], atol=1e-12)


########################################
###### loop_precision_recall e2e #######
########################################


def test_genuine_revisit_precision_and_recall_one(tmp_path):
    # Submaps 0 and 2 share a location (revisit); submap 1 is far away.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    run = _write_run_dir(tmp_path / "arm", xs, [_dec(2, 0, 0, 0)], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=5)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(1.0)
    assert out["accepted_positive"] == 1
    assert out["opportunities"] == 1  # 64 covisible frame pairs collapse to one submap pair


def test_false_accepted_loop_precision_zero(tmp_path):
    # Accepted loop links submap 2 (x≈0) to far-away submap 1 (x≈100) — GT-negative.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    run = _write_run_dir(tmp_path / "arm", xs, [_dec(2, 0, 1, 0)], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=5)
    assert out["precision"] == pytest.approx(0.0)
    assert out["accepted_negative"] == 1
    assert out["recall"] == pytest.approx(0.0)  # genuine (2, 0) opportunity missed


def test_ambiguous_pair_excluded_from_both_metrics(tmp_path):
    # Submap 2 sits in the ambiguous distance band relative to submap 0.
    xs = _cluster_xs([0.0, 100.0, 42.0])
    run = _write_run_dir(tmp_path / "arm", xs, [_dec(2, 0, 0, 0)], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=5)
    assert out["accepted_ambiguous"] == 1
    assert out["precision"] is None  # zero labeled accepted loops
    assert out["opportunities"] == 0
    assert out["recall"] is None


def test_recall_denominator_collapses_frame_pairs(tmp_path):
    # No loops applied: recall 0 over exactly ONE submap-pair opportunity, even
    # though 64 individual frame pairs are covisible.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    run = _write_run_dir(tmp_path / "arm", xs, [], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=5)
    assert out["opportunities"] == 1
    assert out["recall"] == pytest.approx(0.0)
    assert out["precision"] is None  # no accepted loops to label


def test_max_loops_per_submap_caps_opportunities(tmp_path):
    # Submap 3 revisits both submap 0 and submap 1 → two partners, cap trims to one.
    xs = _cluster_xs([0.0, 2.0, 100.0, 1.0])
    run = _write_run_dir(tmp_path / "arm", xs, [], submap_size=8)
    uncapped = loop_precision_recall(run, nms_frame_distance=5)
    capped = loop_precision_recall(run, nms_frame_distance=5, max_loops_per_submap=1)
    assert uncapped["opportunities"] == 2
    assert capped["opportunities"] == 1


def test_nms_frame_distance_filters_opportunities(tmp_path):
    # All covisible pairs are closer than 100 frames apart → no opportunities.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    run = _write_run_dir(tmp_path / "arm", xs, [], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=100)
    assert out["opportunities"] == 0
    assert out["recall"] is None


def test_min_submap_gap_excludes_adjacent_submaps(tmp_path):
    # Submaps 0 and 1 co-located, but gap 1 is unreachable by the pipeline
    # (it queries submaps[:len - min_submap_gap] before appending the current one).
    xs = _cluster_xs([0.0, 0.5])
    run = _write_run_dir(tmp_path / "arm", xs, [], submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=1)
    assert out["opportunities"] == 0


def test_rejected_candidates_ignored_by_precision(tmp_path):
    # A rejected candidate must not enter the precision denominator.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    decisions = [_dec(2, 0, 0, 0), _dec(2, 1, 1, 0, accepted=False)]
    run = _write_run_dir(tmp_path / "arm", xs, decisions, submap_size=8)
    out = loop_precision_recall(run, nms_frame_distance=5)
    assert out["accepted"] == 1
    assert out["precision"] == pytest.approx(1.0)


def test_out_of_range_decision_frame_raises(tmp_path):
    # A decision mapping past the GT trajectory means the frame-index mapping is wrong.
    xs = _cluster_xs([0.0, 100.0, 0.0])
    run = _write_run_dir(tmp_path / "arm", xs, [_dec(5, 0, 0, 0)], submap_size=8)
    with pytest.raises(ValueError, match="out of range"):
        loop_precision_recall(run, nms_frame_distance=5)


def test_missing_decisions_file_raises_file_not_found(tmp_path):
    run = _write_run_dir(tmp_path / "arm", _cluster_xs([0.0]), [], submap_size=8)
    (run / "lc_decisions_lc.json").unlink()
    with pytest.raises(FileNotFoundError):
        loop_precision_recall(run)
