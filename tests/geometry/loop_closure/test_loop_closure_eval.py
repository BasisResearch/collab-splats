from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure import PoseGraph, Submap
from collab_splats.geometry.loop_closure.eval import (
    _classify_edges,
    _per_edge_error,
    ate_translation,
    capture_pose_graph_loss,
    rpe,
    umeyama_align,
)


@pytest.fixture
def pg_with_loop():
    pg = PoseGraph()
    H0 = np.eye(4, dtype=np.float64)
    H1 = np.eye(4, dtype=np.float64)
    H2 = np.eye(4, dtype=np.float64)
    H3 = np.eye(4, dtype=np.float64)
    # nodes
    pg.add_homography(0, H0)
    pg.add_homography(1, H1)
    pg.add_homography(2, H2)
    pg.add_homography(3, H3)
    pg.add_prior_factor(0, H0)
    # sequential edges (3 total)
    pg.add_between_factor(0, 1, H1)
    pg.add_between_factor(1, 2, H2)
    pg.add_between_factor(2, 3, H3)
    # loop edge (1 total) — loop-chain edges share the sequential-edge API/noise
    # (add_loop_edge was removed with the scale-reconciled 3-edge chain)
    pg.add_between_factor(0, 3, H3)
    return pg


def test_classify_edges_returns_two_keys(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    assert set(groups) == {"sequential", "loop"}


def test_classify_edges_sequential_count_matches_expected(pg_with_loop):
    """All 4 BetweenFactors classify as 'sequential' under VGGT-SLAM noise parity.

    Since 011c56f ("match VGGT-SLAM PGO noise exactly"), loop edges use the
    same plain Gaussian (Diagonal.Sigmas) noise as sequential edges — neither
    is Robust(Huber). _classify_edges keys off isinstance(nm, noiseModel.Robust),
    so with no Robust factors all 3 sequential + 1 loop BetweenFactors land in
    'sequential' (4 total); the prior is skipped.
    """
    groups = _classify_edges(pg_with_loop._graph)
    assert len(groups["sequential"]) == 4


def test_classify_edges_loop_count_matches_expected(pg_with_loop):
    """No edge is Robust-noised post-VGGT-SLAM-parity, so 'loop' is empty.

    See test_classify_edges_sequential_count_matches_expected — loop edges are no
    longer noise-distinguishable from sequential ones (011c56f), so _classify_edges
    cannot separate them and 'loop' is empty.
    """
    groups = _classify_edges(pg_with_loop._graph)
    assert len(groups["loop"]) == 0


def test_classify_edges_skips_prior_factor(pg_with_loop):
    """The PriorFactor on frame 0 should not appear in any group."""
    groups = _classify_edges(pg_with_loop._graph)
    total_classified = sum(len(v) for v in groups.values())
    # 3 sequential + 1 loop = 4; total factors = 5 (incl. prior)
    assert total_classified == pg_with_loop._graph.size() - 1


def test_per_edge_error_keys_match_classifier(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    assert set(errors) == {"sequential", "loop"}


def test_per_edge_error_returns_floats(pg_with_loop):
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    for v in errors.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_per_edge_error_sum_matches_classified_total(pg_with_loop):
    """Sum of per-edge errors == sum of individual classified factor errors."""
    groups = _classify_edges(pg_with_loop._graph)
    errors = _per_edge_error(pg_with_loop._graph, pg_with_loop._initial, groups)
    summed = sum(errors.values())
    expected = sum(
        pg_with_loop._graph.at(i).error(pg_with_loop._initial) for indices in groups.values() for i in indices
    )
    assert abs(summed - expected) < 1e-9


def test_capture_loss_returns_required_keys(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert set(trace) >= {"iterations", "per_edge_initial", "per_edge_final", "optimized_values"}


def test_capture_loss_iterations_is_nonempty_list_of_floats(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert isinstance(trace["iterations"], list)
    assert len(trace["iterations"]) >= 1
    assert all(isinstance(x, float) for x in trace["iterations"])


def test_capture_loss_curve_decreases_or_holds(pg_with_loop):
    """LM should never increase total error step-to-step; final ≤ initial."""
    trace = capture_pose_graph_loss(pg_with_loop)
    assert trace["iterations"][-1] <= trace["iterations"][0] + 1e-9


def test_capture_loss_does_not_mutate_input_graph(pg_with_loop):
    """Capture must not mutate pg._initial or pg._graph."""
    keys_before = set(pg_with_loop._initial.keys())
    poses_before = {k: pg_with_loop._initial.atSL4(k).matrix().copy() for k in keys_before}
    graph_size_before = pg_with_loop._graph.size()

    capture_pose_graph_loss(pg_with_loop)

    assert set(pg_with_loop._initial.keys()) == keys_before
    assert pg_with_loop._graph.size() == graph_size_before
    for k in keys_before:
        assert np.allclose(poses_before[k], pg_with_loop._initial.atSL4(k).matrix())


def test_capture_loss_per_edge_keys_match_classifier(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert set(trace["per_edge_initial"]) == {"sequential", "loop"}
    assert set(trace["per_edge_final"]) == {"sequential", "loop"}


def test_capture_loss_returns_converged_flag(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert "converged" in trace
    assert isinstance(trace["converged"], bool)


def test_capture_loss_returns_n_iterations(pg_with_loop):
    trace = capture_pose_graph_loss(pg_with_loop)
    assert "n_iterations" in trace
    assert trace["n_iterations"] == len(trace["iterations"]) - 1
    assert trace["n_iterations"] >= 0


def test_capture_loss_max_iterations_respected(pg_with_loop):
    """If max_iterations=1, iterations should have at most 2 entries (initial + 1 step)."""
    trace = capture_pose_graph_loss(pg_with_loop, max_iterations=1)
    assert len(trace["iterations"]) <= 2


def test_capture_loss_rejects_max_iterations_zero(pg_with_loop):
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        capture_pose_graph_loss(pg_with_loop, max_iterations=0)


def test_umeyama_align_returns_aligned_poses_and_transform():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    aligned, T_align = umeyama_align(pred, gt)
    assert aligned.shape == (2, 4, 4)
    assert T_align.shape == (4, 4)


def test_ate_translation_returns_error_dict():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    result = ate_translation(pred, gt)
    assert set(result) >= {"rmse", "mean", "median", "max", "per_frame"}
    assert result["rmse"] >= 0.0


def test_rpe_returns_error_dict():
    pred = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    gt = np.tile(np.eye(4), (2, 1, 1)).astype(np.float32)
    result = rpe(pred, gt)
    assert set(result) >= {"trans_rmse", "rot_rmse_deg"}
    assert result["trans_rmse"] >= 0.0


def test_capture_pose_graph_loss_importable_from_package():
    from collab_splats.geometry.loop_closure import capture_pose_graph_loss as cpl

    assert callable(cpl)


# NOTE: the batch output-merge helper + its _dedup_rows index array were deleted in
# P4.3c (output assembly now lives in GraphMap.get_world_pointcloud /
# get_corrected_extrinsics). The former dedup-index tests covered a code path that no
# longer exists in production.
