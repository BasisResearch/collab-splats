"""Numeric tests for the scale-reconciled 3-edge loop chain (VGGT-SLAM parity).

Pure numpy/gtsam — GPU-free. Fixtures build a 2-submap graph with known
ground-truth poses plus a synthetic LC submap whose 2 frames duplicate the
query/detected images, and prove:
- consistent graph + GT loop → optimized output stays at ground truth
  (the old direct edge inv(P0)@P1 is the exact inverse and fails this);
- drifted graph + GT loop → query frame moves toward ground truth;
- LC run at 2x scale → anchor scales recover 1/2 and 2, trajectory matches GT;
- LC nodes are graph-only (no output-frame collision);
- missing LC world points → scale-1.0 fallback with one warning per loop.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure.closure import (
    _lc_anchor_scale,
    _loop_chain_relatives,
    run_pose_graph_optimization,
)
from collab_splats.geometry.loop_closure.submap import Submap
from evals.runners.compare_loop_edges import compose_slam_chain

# Image/grid geometry: full-res H*W LC grids vs subsample=8 strided regular grids
H_IMG, W_IMG, STRIDE = 80, 80, 8


########################################
####### Geometry helpers ###############
########################################


def _rz(rad: float) -> np.ndarray:
    """Rotation about z by `rad`."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _w2c(R_c2w: np.ndarray, p: np.ndarray) -> np.ndarray:
    """World-to-cam 4x4 from cam-to-world rotation R and camera centre p."""
    M = np.eye(4)
    M[:3, :3] = R_c2w.T
    M[:3, 3] = -R_c2w.T @ p
    return M


def _apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homogeneous 4x4 T to (N, 3) points."""
    h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T @ h.T).T[:, :3]


def _gt_trajectory(n: int = 7) -> np.ndarray:
    """(n, 4, 4) ground-truth world-to-cam poses; frame 0 at identity."""
    poses = []
    for f in range(n):
        phi = 0.3 * f
        p = 1.5 * np.array([np.cos(phi) - 1.0, np.sin(phi), 0.05 * f])
        poses.append(_w2c(_rz(0.25 * f), p))
    return np.stack(poses)


def _full_grid() -> np.ndarray:
    """(H*W, 3) camera-local synthetic depth grid, row-major over (v, u)."""
    vv, uu = np.meshgrid(np.arange(H_IMG), np.arange(W_IMG), indexing="ij")
    return np.stack(
        [(uu.ravel() - W_IMG / 2) / 20.0, (vv.ravel() - H_IMG / 2) / 20.0,
         2.0 + 0.01 * (uu.ravel() + vv.ravel())],
        axis=1,
    )


def _strided_flat_idx() -> np.ndarray:
    """Flat indices of the subsample=8 grid inside the full-res row-major grid."""
    us = np.arange(0, W_IMG, STRIDE)
    vs = np.arange(0, H_IMG, STRIDE)
    uu, vv = np.meshgrid(us, vs)
    return (vv * W_IMG + uu).ravel()


########################################
####### Submap fixtures ################
########################################


def _make_regular_submap(
    gt: np.ndarray,
    lo: int,
    hi: int,
    submap_id: int,
    drift: bool = False,
    with_conf: bool = False,
) -> Submap:
    """Regular submap over global frames [lo, hi); local frame 0 = identity."""
    k = hi - lo
    grid = _full_grid()[_strided_flat_idx()]
    poses, wps = [], []
    for li in range(k):
        P_local = gt[lo + li] @ np.linalg.inv(gt[lo])
        # Drift: multiplicative camera-frame error growing with local index
        if drift and li > 0:
            E = np.eye(4)
            E[:3, :3] = _rz(np.deg2rad(2.0 * li))
            E[:3, 3] = np.array([0.05 * li, -0.03 * li, 0.02 * li])
            P_local = E @ P_local
        poses.append(P_local)
        # world_points in submap-local world frame: c2w @ camera-local grid
        wps.append(_apply(np.linalg.inv(P_local), grid))
    conf = np.full((k, grid.shape[0]), 100.0, dtype=np.float32) if with_conf else None
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=np.stack(poses).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 8),
        image_paths=[Path(f"frame_{lo + li:04d}.png") for li in range(k)],
        frame_start=lo,
        world_points=np.stack(wps).astype(np.float32),
        world_points_conf=conf,
    )


def _make_lc_submap(
    gt: np.ndarray,
    q_global: int,
    d_global: int,
    scale: float = 1.0,
    no_points: bool = False,
    garbage_offgrid: bool = False,
    poison_inf: bool = False,
    with_conf: bool = False,
) -> Submap:
    """2-frame LC submap for query/detected images; optionally at `scale`x LC units."""
    Sk = np.diag([scale, scale, scale, 1.0])
    P0 = np.eye(4)
    P1_metric = gt[d_global] @ np.linalg.inv(gt[q_global])
    P1 = Sk @ P1_metric @ np.linalg.inv(Sk)  # scale translation by `scale`
    grid = _full_grid()
    wp0 = scale * grid                                   # P0 = I: world == camera-local
    wp1 = _apply(np.linalg.inv(P1), scale * grid)        # LC-local world frame
    if garbage_offgrid:
        # Corrupt every pixel NOT on the stride grid — pixel-aligned resampling
        # must ignore them; naive flatten-pairing would be poisoned.
        off = np.ones(H_IMG * W_IMG, dtype=bool)
        off[_strided_flat_idx()] = False
        wp0 = wp0.copy(); wp1 = wp1.copy()
        wp0[off] = 1e6
        wp1[off] = -1e6
    if poison_inf:
        # Corrupt ALL pixels (on-grid included) with inf — mimics depth-unprojection
        # NaN/inf on invalid pixels driving the scale estimate degenerate.
        wp0 = np.full_like(wp0, np.inf)
        wp1 = np.full_like(wp1, np.inf)
    conf = np.full((2, H_IMG * W_IMG), 100.0, dtype=np.float32) if with_conf else None
    return Submap(
        submap_id=99,
        frames=torch.zeros(2, 3, H_IMG, W_IMG),
        poses=np.stack([P0, P1]).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(2, 8),
        image_paths=[Path(f"frame_{q_global:04d}.png"), Path(f"frame_{d_global:04d}.png")],
        is_lc_submap=True,
        world_points=None if no_points else np.stack([wp0, wp1]).astype(np.float32),
        world_points_conf=None if no_points else conf,
    )


def _centre(w2c: np.ndarray) -> np.ndarray:
    """Camera centre from a world-to-cam pose."""
    return -w2c[:3, :3].T @ w2c[:3, 3]


def _run(submaps, lc_submaps, total_frames=7):
    return run_pose_graph_optimization(
        submaps, lc_submaps, total_frames=total_frames, overlap_frames=1,
        manifold="sl4", conf_threshold=25.0, scale_method="rotation_only",
    )


def _run_recording_graph_errors(monkeypatch, submaps, lc_submaps):
    """Run PGO while recording factor-graph error before each optimize() call."""
    from collab_splats.geometry.loop_closure.closure import _SL4PoseGraph

    pre_errors: list[float] = []
    orig = _SL4PoseGraph.optimize

    def patched(self):
        pre_errors.append(float(self._graph.error(self._initial)))
        orig(self)

    monkeypatch.setattr(_SL4PoseGraph, "optimize", patched)
    out = _run(submaps, lc_submaps)
    return out, pre_errors


@pytest.fixture
def gt():
    return _gt_trajectory(7)


@pytest.fixture
def consistent_submaps(gt):
    # Submap 0: global frames 0-3; submap 1: global 3-6 (overlap frame 3)
    return [
        _make_regular_submap(gt, 0, 4, submap_id=0),
        _make_regular_submap(gt, 3, 7, submap_id=1),
    ]


Q_GLOBAL, D_GLOBAL = 5, 1  # query frame (submap 1) loops back to detected frame (submap 0)


########################################
####### Chain relative derivation ######
########################################


def test_chain_relatives_compose_to_direct_for_identity_anchors(gt):
    """With sA=sB=1 and K=I the composed chain equals P_lc0 @ inv(P_lc1)."""
    P0 = np.eye(4)
    P1 = gt[D_GLOBAL] @ np.linalg.inv(gt[Q_GLOBAL])
    h_a, h_inner, h_b = _loop_chain_relatives(P0, P1, 1.0, 1.0)
    composed = compose_slam_chain(h_a, h_inner, h_b)
    np.testing.assert_allclose(composed, P0 @ np.linalg.inv(P1), atol=1e-12)
    # each anchor is pure identity here; the inner edge carries the whole relative
    np.testing.assert_allclose(h_a, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(h_b, np.eye(4), atol=1e-12)


def test_chain_relatives_scale_reconciliation_cancels_lc_scale(gt):
    """LC at 2x: sA=1/2, sB=2 makes the composed chain equal the metric relative."""
    k = 2.0
    Sk = np.diag([k, k, k, 1.0])
    P1_metric = gt[D_GLOBAL] @ np.linalg.inv(gt[Q_GLOBAL])
    P0_k, P1_k = np.eye(4), Sk @ P1_metric @ np.linalg.inv(Sk)
    h_a, h_inner, h_b = _loop_chain_relatives(P0_k, P1_k, 1.0 / k, k)
    composed = compose_slam_chain(h_a, h_inner, h_b)
    np.testing.assert_allclose(composed, np.eye(4) @ np.linalg.inv(P1_metric), atol=1e-10)


########################################
####### Anchor scale estimation ########
########################################


def test_anchor_scale_recovers_2x_lc_scale(gt, consistent_submaps):
    """Pixel-aligned anchors: sA = 1/2 (LC→query units), sB = 2 (detected→LC units)."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, scale=2.0)
    sub_q, sub_d = consistent_submaps[1], consistent_submaps[0]
    s_a = _lc_anchor_scale(lc, 0, sub_q, Q_GLOBAL - 3, 25.0, "rotation_only")
    s_b = _lc_anchor_scale(sub_d, D_GLOBAL, lc, 1, 25.0, "rotation_only")
    assert s_a == pytest.approx(0.5, rel=1e-6)
    assert s_b == pytest.approx(2.0, rel=1e-6)


def test_anchor_scale_resamples_lc_grid_pixel_aligned(gt, consistent_submaps):
    """Off-stride LC pixels are garbage: correct only if resampled on the stride grid."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, scale=2.0, garbage_offgrid=True)
    sub_q = consistent_submaps[1]
    s_a = _lc_anchor_scale(lc, 0, sub_q, Q_GLOBAL - 3, 25.0, "rotation_only")
    assert s_a == pytest.approx(0.5, rel=1e-6)


def test_anchor_scale_with_conf_joint_mask(gt):
    """Conf present on both sides (all above threshold) → joint chain, same estimate."""
    submaps = [
        _make_regular_submap(gt, 0, 4, submap_id=0, with_conf=True),
        _make_regular_submap(gt, 3, 7, submap_id=1, with_conf=True),
    ]
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, scale=2.0, with_conf=True)
    s_b = _lc_anchor_scale(submaps[0], D_GLOBAL, lc, 1, 25.0, "rotation_only")
    assert s_b == pytest.approx(2.0, rel=1e-6)


def test_anchor_scale_none_when_lc_points_missing(gt, consistent_submaps):
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, no_points=True)
    s_a = _lc_anchor_scale(lc, 0, consistent_submaps[1], Q_GLOBAL - 3, 25.0, "rotation_only")
    assert s_a is None


def test_anchor_scale_none_when_lc_points_nonfinite(gt, consistent_submaps):
    """All-inf LC world points (invalid depth pixels) on the prior side drive the
    median scale non-finite → None, and the loop falls back to a finite graph."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, poison_inf=True)
    # s_b pairs finite detected-frame norms (X) against inf/NaN LC norms (Y):
    # the median ratio is non-finite and must be rejected by the guard.
    s_b = _lc_anchor_scale(consistent_submaps[0], D_GLOBAL, lc, 1, 25.0, "rotation_only")
    assert s_b is None
    # End-to-end: fallback keeps the graph finite and at GT (consistent fixture)
    out = _run(consistent_submaps, [lc])
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, gt.astype(np.float32), atol=1e-3)


########################################
####### End-to-end pose graph ##########
########################################


def test_consistent_loop_keeps_gt_trajectory(gt, consistent_submaps, monkeypatch):
    """Zero-error graph: GT loop adds ~0 residual and leaves the trajectory at GT.

    The old direct edge inv(P0)@P1 is the exact inverse of the correct relative:
    it injects a huge pre-optimization residual (measured ~2.6e3 on this fixture).
    """
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)
    out, pre_errors = _run_recording_graph_errors(monkeypatch, consistent_submaps, [lc])
    assert pre_errors[-1] < 1e-6  # graph incl. 3-edge loop chain is consistent pre-optimization
    np.testing.assert_allclose(out, gt.astype(np.float32), atol=1e-3)


def test_scaled_lc_recovers_gt(gt, consistent_submaps, monkeypatch):
    """LC run at 2x scale: scale-reconciled anchors cancel it; output stays at GT."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, scale=2.0)
    out, pre_errors = _run_recording_graph_errors(monkeypatch, consistent_submaps, [lc])
    assert pre_errors[-1] < 1e-6
    np.testing.assert_allclose(out, gt.astype(np.float32), atol=1e-3)


def test_drifted_loop_moves_query_toward_gt(gt):
    """Drifted submap 1 + exact GT loop: query frame moves toward ground truth."""
    submaps = [
        _make_regular_submap(gt, 0, 4, submap_id=0),
        _make_regular_submap(gt, 3, 7, submap_id=1, drift=True),
    ]
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)
    out_noloop = _run(submaps, [])
    out_loop = _run(submaps, [lc])
    gt_c = _centre(gt[Q_GLOBAL])
    err_noloop = np.linalg.norm(_centre(out_noloop[Q_GLOBAL]) - gt_c)
    err_loop = np.linalg.norm(_centre(out_loop[Q_GLOBAL]) - gt_c)
    assert err_noloop > 0.05  # drift fixture is actually drifted
    assert err_loop < 0.7 * err_noloop


def test_lc_nodes_excluded_from_output(gt, consistent_submaps):
    """LC nodes are graph-only: output has exactly total_frames rows, all finite."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL)
    out = _run(consistent_submaps, [lc])
    assert out.shape == (7, 4, 4)
    assert np.isfinite(out).all()
    # every output row is a real camera pose (bottom row [0,0,0,1]) — no LC leakage
    np.testing.assert_allclose(out[:, 3, :], np.tile([0, 0, 0, 1.0], (7, 1)), atol=1e-6)


def test_missing_lc_points_falls_back_scale1_with_one_warning(gt, consistent_submaps, caplog):
    """poses-only LC (vggtx/omega): direction fix still applies; warn once per loop."""
    lc = _make_lc_submap(gt, Q_GLOBAL, D_GLOBAL, no_points=True)
    with caplog.at_level(logging.WARNING, logger="collab_splats.geometry.loop_closure.closure"):
        out = _run(consistent_submaps, [lc])
    np.testing.assert_allclose(out, gt.astype(np.float32), atol=1e-3)
    warnings = [r for r in caplog.records if "scale" in r.getMessage().lower()]
    assert len(warnings) == 1
