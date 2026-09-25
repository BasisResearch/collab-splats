from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure import graph as graph_mod
from collab_splats.geometry.loop_closure.graph import (
    PoseGraph,
    decompose_camera,
    estimate_scale_pairwise,
)
from collab_splats.geometry.loop_closure.submap import Submap
from tests.geometry.loop_closure._helpers import drive_pose_graph


def test_decompose_camera_round_trip():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    R = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    t = np.array([0.1, -0.2, 0.5])
    P34 = K @ np.hstack([R, t[:, None]])  # (3, 4)
    K_out, R_out, t_out, scale = decompose_camera(P34)
    assert np.allclose(K_out[:3, :3] / K_out[0, 0], K / K[0, 0], atol=1e-6)
    assert np.allclose(np.abs(R_out), np.abs(R), atol=1e-5)
    assert np.allclose(t_out, t, atol=1e-5)


def test_decompose_camera_accepts_4x4():
    K = np.eye(3, dtype=np.float64) * 400.0
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3)
    P34 = K @ np.hstack([R, t[:, None]])
    P44 = np.vstack([P34, [0, 0, 0, 1]])
    K_out, R_out, t_out, scale = decompose_camera(P44)
    assert K_out.shape[0] == 3


def test_estimate_scale_pairwise_known():
    rng = np.random.default_rng(1)
    X = rng.random((50, 3)).astype(np.float64)
    Y = X * 2.5  # exact scale = 2.5
    scale = estimate_scale_pairwise(X, Y)
    assert abs(scale - 2.5) < 0.01


def test_estimate_scale_pairwise_no_div_zero():
    X = np.zeros((5, 3), dtype=np.float64)  # all at origin
    Y = np.ones((5, 3), dtype=np.float64)
    scale = estimate_scale_pairwise(X, Y)  # should not raise
    assert np.isfinite(scale)


########################################
########## PoseGraph tests #############
########################################


def _identity_H() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _translate_H(tx: float, ty: float, tz: float) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [tx, ty, tz]
    return H


def test_sl4_add_homography_initializes():
    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    pg.add_homography(1, _translate_H(0.1, 0, 0))
    assert 0 in pg._node_ids
    assert 1 in pg._node_ids


def test_sl4_add_homography_duplicate_noop():
    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    pg.add_homography(0, _translate_H(1, 1, 1))  # duplicate — must not raise or re-insert
    assert len(pg._node_ids) == 1


def test_sl4_sequential_edge_optimize():
    pg = PoseGraph()
    # Translation matrices have det=1, so they already satisfy the SL(4)
    # constraint; add_homography/add_prior_factor SL4-normalize on insert regardless.
    H0 = _identity_H()
    H1 = _translate_H(0.1, 0, 0)
    pg.add_homography(0, H0)
    pg.add_homography(1, H1)
    pg.add_prior_factor(0, H0)
    H_rel = np.linalg.inv(H0) @ H1
    pg.add_between_factor(0, 1, H_rel)
    pg.optimize()
    H0_out = pg.get_homography(0)
    assert H0_out.shape == (4, 4)
    assert np.allclose(H0_out, H0, atol=0.05)


def test_sl4_loop_edge_no_crash():
    pg = PoseGraph()
    Hs = [_translate_H(i * 0.1, 0, 0) for i in range(3)]
    for i, H in enumerate(Hs):
        pg.add_homography(i, H)
    pg.add_prior_factor(0, Hs[0])
    pg.add_between_factor(0, 1, np.linalg.inv(Hs[0]) @ Hs[1])
    pg.add_between_factor(1, 2, np.linalg.inv(Hs[1]) @ Hs[2])
    # Loop-chain edges share the sequential-edge API and Gaussian noise
    # (add_loop_edge was removed with the scale-reconciled 3-edge chain).
    pg.add_between_factor(2, 0, np.linalg.inv(Hs[2]) @ Hs[0])
    pg.optimize()  # must not raise
    for i in range(3):
        assert np.isfinite(pg.get_homography(i)).all()


def test_get_homography_post_optimize():
    pg = PoseGraph()
    H0 = _identity_H()
    pg.add_homography(0, H0)
    pg.add_prior_factor(0, H0)
    pg.optimize()
    H_out = pg.get_homography(0)
    assert H_out.shape == (4, 4)
    assert np.isfinite(H_out).all()


########################################
####### incremental PoseGraph drive ####
########################################


def _make_real_submap(submap_id: int, k: int = 4, frame_start: int = 0) -> Submap:
    rng = np.random.default_rng(submap_id)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = (rng.standard_normal((k, 3)) * 0.05).astype(np.float32)
    intrinsics = np.tile(np.diag([400.0, 400.0, 1.0]).astype(np.float32), (k, 1, 1))
    world_points = rng.standard_normal((k, 20, 3)).astype(np.float32)
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 8, 8),
        poses=poses,
        intrinsics=intrinsics,
        retrieval_vectors=torch.zeros(k, 32),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
        frame_start=frame_start,
        world_points=world_points,
    )


def test_incremental_pose_graph_returns_correct_shape():
    k = 4
    submaps = [_make_real_submap(0, k=k, frame_start=0), _make_real_submap(1, k=k, frame_start=k)]
    result = drive_pose_graph(
        submaps,
        lc_submaps=[],
        total_frames=k * 2,
        overlap_frames=1,
    )
    assert result.shape == (k * 2, 4, 4)
    assert np.isfinite(result).all()


########################################
####### error handling #################
########################################


def test_decompose_camera_rejects_bad_shape():
    with pytest.raises(ValueError, match="expected"):
        decompose_camera(np.ones((2, 4)))


def _optimizer_raising(exc):
    def build(*args, **kwargs):
        raise exc

    return build


def test_optimize_keeps_initial_values_on_gtsam_runtime_error(monkeypatch):
    pg = PoseGraph()
    H0 = _translate_H(0.3, 0, 0)
    pg.add_homography(0, H0)
    before = pg.get_homography(0)
    monkeypatch.setattr(
        graph_mod,
        "gtsam",
        SimpleNamespace(
            LevenbergMarquardtParams=lambda: None,
            LevenbergMarquardtOptimizer=_optimizer_raising(RuntimeError("indeterminant system")),
        ),
    )
    pg.optimize()
    monkeypatch.undo()
    assert np.array_equal(pg.get_homography(0), before)


def test_optimize_propagates_non_gtsam_errors(monkeypatch):
    pg = PoseGraph()
    pg.add_homography(0, _identity_H())
    monkeypatch.setattr(
        graph_mod,
        "gtsam",
        SimpleNamespace(
            LevenbergMarquardtParams=lambda: None,
            LevenbergMarquardtOptimizer=_optimizer_raising(TypeError("bad argument")),
        ),
    )
    with pytest.raises(TypeError, match="bad argument"):
        pg.optimize()


########################################
####### confidence-mask floor ##########
########################################


# Three confidence groups, each voting a different scale
# - A (20 pts): both sides confident, scale 2 -> the joint mask
# - B (60 pts): prior side confident only, scale 4 -> A+B is the prior-only mask
# - C (120 pts): low but positive confidence on both sides, scale 8 -> everything
N_A, N_B, N_C = 20, 60, 120


def _conf_group_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Curr/prior camera-local points plus confidences for the three groups.

    Returns:
        curr points, prior points (both (N, 3)), curr confidence, prior confidence (both (N,)).
    """
    rng = np.random.default_rng(7)
    n = N_A + N_B + N_C
    prior = rng.standard_normal((n, 3)) + np.array([0, 0, 5.0])
    scale = np.concatenate([np.full(N_A, 2.0), np.full(N_B, 4.0), np.full(N_C, 8.0)])
    curr = prior / scale[:, None]

    # Joint confidence on A only; prior confidence on A+B; C passes only a `> 0` test
    curr_conf = np.concatenate([np.full(N_A, 50.0), np.zeros(N_B), np.ones(N_C)]).astype(np.float32)
    prior_conf = np.concatenate([np.full(N_A + N_B, 50.0), np.ones(N_C)]).astype(np.float32)
    return curr, prior, curr_conf, prior_conf


def _submap_pair_with_conf_groups():
    """Two submaps whose single overlap frame carries the three confidence groups."""
    curr, prior, curr_conf, prior_conf = _conf_group_points()
    k = 2
    s0 = _make_real_submap(0, k=k, frame_start=0)
    s1 = _make_real_submap(1, k=k, frame_start=k - 1)
    s0.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    s1.poses[1, 0, 3] = 0.1  # a baseline inside s1, so its scale shows in frame 2
    s0.world_points = np.tile(prior, (k, 1, 1)).astype(np.float32)
    s1.world_points = np.tile(curr, (k, 1, 1)).astype(np.float32)
    s0.world_points_conf = np.tile(prior_conf, (k, 1))
    s1.world_points_conf = np.tile(curr_conf, (k, 1))
    return s0, s1


def _drive(pg, submaps):
    for s in submaps:
        pg.add_submap(s, 1, 25.0, "rotation_only")
        pg.optimize()
    return pg.extract_extrinsics(3)


def test_min_conf_points_sets_the_confidence_mask_floor():
    """
    Each floor picks its own mask on the sequential edge.

    - 100 (default): joint 20 and prior 80 both fall short -> every point
    - 30: joint falls short, prior clears -> the prior-only mask
    - 10: joint clears -> the joint mask
    """
    s0, s1 = _submap_pair_with_conf_groups()
    default = _drive(PoseGraph(), [s0, s1])
    explicit = _drive(PoseGraph(min_conf_points=100), [s0, s1])
    prior_only = _drive(PoseGraph(min_conf_points=30), [s0, s1])
    joint = _drive(PoseGraph(min_conf_points=10), [s0, s1])
    assert np.array_equal(default, explicit)
    assert not np.allclose(default, prior_only)
    assert not np.allclose(prior_only, joint)
    assert not np.allclose(default, joint)


def _single_frame_submap(sid: int, points: np.ndarray, conf: np.ndarray, is_lc: bool) -> Submap:
    """One-frame submap at the identity pose with identity intrinsics."""
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32)[None],
        retrieval_vectors=np.zeros((1, 8), dtype=np.float32),
        image_paths=[f"s{sid}_f0.jpg"],
        is_lc_submap=is_lc,
        world_points=points[None].astype(np.float32),
        world_points_conf=conf[None],
    )


@pytest.mark.parametrize("min_conf_points, expected", [(100, 8.0), (30, 4.0), (10, 2.0)])
def test_lc_anchor_scale_confidence_floor_picks_the_mask(min_conf_points, expected):
    """100: prior > 0 fallback (all); 30: prior-only mask (A+B); 10: joint mask (A)."""
    curr, prior, curr_conf, prior_conf = _conf_group_points()
    lc = _single_frame_submap(9, curr, curr_conf, is_lc=True)
    reg = _single_frame_submap(0, prior, prior_conf, is_lc=False)
    s = graph_mod._lc_anchor_scale(lc, 0, reg, 0, 25.0, "rotation_only", min_conf_points)
    assert s == pytest.approx(expected, rel=1e-5)
