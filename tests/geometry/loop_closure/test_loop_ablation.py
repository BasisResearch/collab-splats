"""Per-loop ablation metric on the LoopClosure wrapper: delta-ATE attribution (GPU-free)."""
from __future__ import annotations

import numpy as np

from collab_splats.geometry.loop_closure import LoopClosureConfig
from collab_splats.geometry.loop_closure.closure import run_pose_graph_optimization
from collab_splats.geometry.loop_closure.wrapper import LoopClosure
from tests.geometry.loop_closure.test_loop_edge_chain import (
    D_GLOBAL,
    Q_GLOBAL,
    _centre,
    _gt_trajectory,
    _make_lc_submap,
    _make_regular_submap,
)


########################################
####### Fixture helpers ################
########################################


class _StubBase:
    """Bare attribute bag standing in for a feedforward creator."""


def _make_wrapper() -> LoopClosure:
    """LoopClosure wrapper with PGO params matching the loop-chain fixtures."""
    cfg = LoopClosureConfig(
        submap_size=4, submap_overlap=1, scale_method="rotation_only",
        verify_match_ratio=0.85,
    )
    return LoopClosure(_StubBase(), config=cfg)


def _drifted_submaps(gt: np.ndarray) -> list:
    """2-submap graph with drift on submap 1 (same shape as the loop-chain fixtures)."""
    return [
        _make_regular_submap(gt, 0, 4, submap_id=0),
        _make_regular_submap(gt, 3, 7, submap_id=1, drift=True),
    ]


def _run_full(submaps, lc_submaps) -> np.ndarray:
    """Full-LC PGO with the exact kwargs the wrapper uses."""
    return run_pose_graph_optimization(
        submaps, lc_submaps, total_frames=7, overlap_frames=1,
        conf_threshold=25.0, scale_method="rotation_only",
    )


########################################
####### Ablation extrinsics ############
########################################


def test_ablation_extrinsics_len_and_shapes():
    """One ablation trajectory per loop, each (total_frames, 4, 4) and finite."""
    gt = _gt_trajectory(7)
    submaps = _drifted_submaps(gt)
    loops = [
        _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL),
        _make_lc_submap(gt, 6, 2),
    ]
    lc = _make_wrapper()
    lc._ablate_loops(submaps, loops, total_frames=7)
    abl = lc.base._lc_ablation_extrinsics
    assert len(abl) == len(loops)
    for a in abl:
        assert a.shape == (7, 4, 4)
        assert np.isfinite(a).all()


def test_ablating_good_loop_moves_query_away_from_gt():
    """Drifted graph + GT loop: removing the loop must worsen the query frame (Δ > 0)."""
    gt = _gt_trajectory(7)
    submaps = _drifted_submaps(gt)
    loops = [_make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)]
    full = _run_full(submaps, loops)
    lc = _make_wrapper()
    lc._ablate_loops(submaps, loops, total_frames=7)
    ablated = lc.base._lc_ablation_extrinsics[0]
    gt_c = _centre(gt[Q_GLOBAL])
    err_full = np.linalg.norm(_centre(full[Q_GLOBAL]) - gt_c)
    err_ablated = np.linalg.norm(_centre(ablated[Q_GLOBAL]) - gt_c)
    assert err_ablated > err_full


def test_ablating_single_loop_equals_no_loop_run():
    """With one loop, its ablation is exactly the no-loop optimization."""
    gt = _gt_trajectory(7)
    submaps = _drifted_submaps(gt)
    loops = [_make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)]
    lc = _make_wrapper()
    lc._ablate_loops(submaps, loops, total_frames=7)
    no_loop = _run_full(submaps, [])
    np.testing.assert_array_equal(lc.base._lc_ablation_extrinsics[0], no_loop)


def test_ablation_leaves_full_optimization_unchanged():
    """Regression guard: ablations must not mutate submaps or perturb the full-LC output."""
    gt = _gt_trajectory(7)
    submaps = _drifted_submaps(gt)
    loops = [
        _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL),
        _make_lc_submap(gt, 6, 2),
    ]
    before = _run_full(submaps, loops)
    poses_before = [s.poses.copy() for s in submaps + loops]
    lc = _make_wrapper()
    lc._ablate_loops(submaps, loops, total_frames=7)
    after = _run_full(submaps, loops)
    np.testing.assert_array_equal(before, after)
    for s, p in zip(submaps + loops, poses_before):
        np.testing.assert_array_equal(s.poses, p)


def test_ablation_logs_single_info_line(caplog):
    """One INFO line summarizing loop count and elapsed time."""
    import logging

    gt = _gt_trajectory(7)
    submaps = _drifted_submaps(gt)
    loops = [_make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)]
    lc = _make_wrapper()
    with caplog.at_level(logging.INFO, logger="collab_splats.geometry.loop_closure.wrapper"):
        lc._ablate_loops(submaps, loops, total_frames=7)
    lines = [r for r in caplog.records if "loop ablation" in r.getMessage()]
    assert len(lines) == 1
    assert "1 re-optimizations" in lines[0].getMessage()
