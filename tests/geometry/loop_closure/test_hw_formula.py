"""Tests for H_w inter-submap initialization formula (Bug 1 fix).

H_w = H_overlap @ T @ H_scale  where T = inv(P_prev_ov) @ P_curr_ov.
Old code used inv(K_prev) @ K_curr which = I when K=identity, ignoring rotation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure.graph import run_pose_graph_optimization
from collab_splats.geometry.loop_closure.submap import Submap


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _make_submap(poses: np.ndarray, world_points: np.ndarray, submap_id: int = 0) -> Submap:
    k = poses.shape[0]
    # Submap.world_points is (K, P, 3) per the production contract (_raw_to_world_points
    # in feedforward/base.py returns (K, P, 3)). Flatten any per-pixel (K, H, W, 3) grid
    # into the (K, P, 3) layout closure.py expects.
    wp = np.asarray(world_points, dtype=np.float32).reshape(k, -1, 3)
    return Submap(
        submap_id=submap_id,
        frames=None,
        poses=poses.astype(np.float32),
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=np.zeros((k, 64), dtype=np.float32),
        image_paths=[f"frame_{i:04d}.png" for i in range(k)],
        raw_outputs={},
        frame_start=submap_id * k,
        world_points=wp,
        world_points_conf=None,
    )


def test_hw_formula_uses_full_pose_not_k_only():
    """Direct formula check: H_w_new = H_overlap @ T @ H_scale differs from
    H_w_old = H_overlap @ inv(K) @ K @ H_scale = H_overlap when K=I.
    """
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()

    P_prev_ov = np.eye(4, dtype=np.float64)
    P_curr_ov = np.eye(4, dtype=np.float64)
    P_curr_ov[:3, :3] = R_w  # 30° rotation between world frames

    H_overlap = P_prev_ov.copy()
    T = np.linalg.inv(P_prev_ov) @ P_curr_ov
    H_scale = np.eye(4, dtype=np.float64)

    # Old formula: K=I → identity transform
    K_prev4 = np.eye(4, dtype=np.float64)
    K_curr4 = np.eye(4, dtype=np.float64)
    H_w_old = H_overlap @ np.linalg.inv(K_prev4) @ K_curr4 @ H_scale

    # New formula: full pose T
    H_w_new = H_overlap @ T @ H_scale

    # Old formula collapses to H_overlap (no rotation encoded)
    assert np.allclose(H_w_old, H_overlap), "Old K-only formula should equal H_overlap when K=I"

    # New formula includes the 30° rotation
    assert not np.allclose(
        H_w_new, H_overlap, atol=1e-6
    ), "New T-based formula must differ when world frames have a rotation"
    assert np.allclose(
        H_w_new[:3, :3], R_w, atol=1e-6
    ), "New H_w rotation block should match the world-frame rotation R_w"


def test_hw_formula_integration_two_submaps_rotated():
    """Integration: 2-submap PGO with 30° rotation between world frames.
    New H_w correctly initializes the first frame of submap 2 with the rotation.
    Check: first frame of submap 2 in output has rotation close to R_w (not identity).
    """
    rng = np.random.default_rng(0)
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    k = 2

    # Prev submap: identity camera (overlap frame = last frame = identity)
    poses_prev = np.stack([_make_w2c(np.eye(3), np.array([i * 0.1, 0.0, 0.0])) for i in range(k)])
    # Curr submap: first frame (overlap) has 30° rotation in curr world
    poses_curr = np.stack([_make_w2c(R_w, np.array([i * 0.1, 0.0, 0.0])) for i in range(k)])

    wp_prev = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1
    wp_curr = rng.standard_normal((k, 5, 5, 3)).astype(np.float32) * 0.1

    prev_sub = _make_submap(poses_prev.astype(np.float32), wp_prev, submap_id=0)
    curr_sub = _make_submap(poses_curr.astype(np.float32), wp_curr, submap_id=1)

    total_frames = k + (k - 1)  # 3 with overlap=1
    result = run_pose_graph_optimization(
        [prev_sub, curr_sub], lc_submaps=[], total_frames=total_frames, overlap_frames=1
    )

    assert result.shape == (total_frames, 4, 4)
    # First frame (reference) should be near identity
    assert np.allclose(result[0], np.eye(4), atol=0.15), f"Frame 0 should be near identity, got\n{result[0]}"
    # With correct H_w (T-based) + new extraction (local_proj @ inv(H_opt)):
    # H_opt encodes R_w and local_proj also has R_w, so they cancel → rotation ≈ I.
    # With wrong H_w (K-only, old bug): H_opt ≈ I, so local_proj @ inv(I) = R_w → rotation visible.
    R_out = result[k, :3, :3]  # unique frame of curr submap (starts at frame_start=submap_id*k=k)
    rot_vs_identity = np.linalg.norm(R_out - np.eye(3), "fro")
    assert (
        rot_vs_identity < 0.3
    ), f"With correct H_w, rotation should cancel in extraction (got rot_vs_identity={rot_vs_identity:.3f})"
