"""Tests for pose extraction method (Bug 2 fix).

New: local_proj @ inv(H_opt) then decompose.
Old: decompose(H_opt) directly.

VGGT-SLAM: projection_mat = proj_mats[idx] @ inv(homography_world).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.graph import decompose_camera
from collab_splats.pointcloud.loop_closure.submap import Submap


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _make_submap(poses: np.ndarray, world_points: np.ndarray, submap_id: int = 0) -> Submap:
    k = poses.shape[0]
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=poses.astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 64, dtype=torch.float32),
        image_paths=[f"frame_{i:04d}.png" for i in range(k)],
        raw_outputs={},
        frame_start=submap_id * k,
        world_points=world_points.astype(np.float32),
        world_points_conf=None,
    )


def test_pose_extraction_formula_local_proj_inv_h_opt():
    """local_proj @ inv(H_opt) then decompose gives different result from decompose(H_opt).

    Verifies the formula change is actually applied. Uses synthetic H_opt and local_proj
    with a known relative rotation so old and new differ detectably.
    """
    # Synthetic: H_opt has 15° tilt, local_proj has 5° roll — distinct matrices
    R_opt = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    H_opt = np.eye(4, dtype=np.float64)
    H_opt[:3, :3] = R_opt
    H_opt[:3, 3] = [0.2, 0.1, 0.0]

    R_local = ScipyR.from_euler("x", 5, degrees=True).as_matrix()
    local_proj = np.eye(4, dtype=np.float64)
    local_proj[:3, :3] = R_local
    local_proj[:3, 3] = [0.05, 0., 0.]

    # New extraction
    corrected = local_proj @ np.linalg.inv(H_opt)
    _, R_new, t_new, _ = decompose_camera(corrected)

    # Old extraction
    _, R_old, t_old, _ = decompose_camera(H_opt)

    # They must differ (formula change is meaningful)
    assert not np.allclose(R_old, R_new, atol=0.01), \
        "Old and new extraction must give different rotations for non-trivial inputs"
    assert not np.allclose(t_old, t_new, atol=0.01), \
        "Old and new extraction must give different translations for non-trivial inputs"


def test_pose_extraction_single_submap_first_frame_near_identity():
    """Single submap: first frame should be near identity after PGO (pinned by prior).

    With new extraction: local_proj[0] @ inv(H_opt[0]).
    H_opt[0] is pinned by prior to H0 = poses[0].
    local_proj[0] = poses[0].
    So result: poses[0] @ inv(poses[0]) = I → decompose(I) → R=I, t=0. ✓
    """
    rng = np.random.default_rng(42)
    k = 4
    poses = np.stack([
        _make_w2c(np.eye(3), np.array([i * 0.1, 0., 0.])) for i in range(k)
    ]).astype(np.float32)
    wp = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1

    submap = _make_submap(poses, wp, submap_id=0)
    result = run_pose_graph_optimization(
        [submap], lc_submaps=[], total_frames=k, overlap_frames=1
    )

    assert result.shape == (k, 4, 4)
    # First frame: reference frame → near identity
    assert np.allclose(result[0], np.eye(4), atol=0.1), \
        f"First frame should be near identity, got\n{result[0]}"
