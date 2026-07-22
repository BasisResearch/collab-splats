from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure.graph import (
    decompose_camera,
    estimate_scale_pairwise,
)


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

import gtsam

from collab_splats.geometry.loop_closure.graph import PoseGraph


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

from pathlib import Path

import torch

from collab_splats.geometry.loop_closure.submap import Submap
from tests.geometry.loop_closure._helpers import drive_pose_graph


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
