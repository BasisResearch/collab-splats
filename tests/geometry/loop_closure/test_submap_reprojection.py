from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry.loop_closure.submap import Submap


def _make_submap(k: int = 3, n_pts: int = 10) -> Submap:
    rng = np.random.default_rng(42)
    poses = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    poses[:, :3, 3] = rng.standard_normal((k, 3)).astype(np.float32) * 0.01
    world_points = rng.standard_normal((k, n_pts, 3)).astype(np.float32)
    return Submap(
        submap_id=0,
        frames=torch.zeros(k, 3, 8, 8),
        poses=poses,
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=torch.zeros(k, 32),
        image_paths=[Path(f"f{i}.jpg") for i in range(k)],
        world_points=world_points,
    )


def test_get_world_points_none_returns_local():
    sm = _make_submap()
    out = sm.get_world_points(H=None)
    assert out.shape == sm.world_points.shape
    assert np.allclose(out, sm.world_points)


def test_get_world_points_identity_noop():
    sm = _make_submap()
    out = sm.get_world_points(H=np.eye(4))
    assert np.allclose(out, sm.world_points, atol=1e-5)


def test_get_world_points_translation():
    sm = _make_submap(k=2, n_pts=5)
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [1.0, 2.0, 3.0]
    out = sm.get_world_points(H=H)   # shape (k*n_pts, 3) or (k, n_pts, 3)
    local_flat = sm.world_points.reshape(-1, 3).astype(np.float64)
    expected = local_flat + np.array([1.0, 2.0, 3.0])
    assert np.allclose(out.reshape(-1, 3), expected, atol=1e-5)


def test_get_world_points_projective_dehomogenizes():
    sm = _make_submap(k=1, n_pts=4)
    # Non-trivial SL4 H with projective component
    H = np.eye(4, dtype=np.float64)
    H[3, :3] = [0.01, 0.01, 0.01]  # projective row → w != 1
    H /= np.linalg.det(H) ** 0.25
    out = sm.get_world_points(H=H)
    assert out.shape == sm.world_points.shape
    assert np.isfinite(out).all()


def test_get_world_points_no_world_points_raises():
    sm = _make_submap()
    sm.world_points = None
    with pytest.raises(ValueError):
        sm.get_world_points(H=None)


def test_get_poses_world_none_returns_poses():
    sm = _make_submap()
    out = sm.get_poses_world(H=None)
    assert np.allclose(out, sm.poses)


def test_get_world_points_near_zero_w_clamps():
    sm = _make_submap(k=1, n_pts=2)
    # Force w to be near-zero by making the projective row cancel pts
    H = np.eye(4, dtype=np.float64)
    pts_flat = sm.world_points[0]  # (2, 3)
    # Set projective row so w = pts_x * a + ... ≈ 0 for one point
    H[3, :3] = [-1.0 / pts_flat[0, 0], 0, 0] if abs(pts_flat[0, 0]) > 1e-3 else [0, 0, 0]
    H[3, 3] = 1.0  # keeps w ≈ 0 for first point
    H /= abs(np.linalg.det(H)) ** 0.25 if abs(np.linalg.det(H)) > 1e-12 else 1.0
    out = sm.get_world_points(H=H)  # must not raise
    assert np.isfinite(out).all()


def test_get_poses_world_known_H():
    sm = _make_submap(k=3)
    H = np.eye(4, dtype=np.float64)
    H[:3, 3] = [5.0, 0.0, 0.0]
    out = sm.get_poses_world(H=H)
    assert out.shape == (3, 4, 4)
    # first pose in output ≈ H @ poses[0]
    expected0 = (H @ sm.poses[0].astype(np.float64)).astype(np.float32)
    assert np.allclose(out[0], expected0, atol=1e-5)
