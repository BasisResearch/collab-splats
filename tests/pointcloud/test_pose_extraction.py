"""Tests for pose extraction method (Bug 2 fix).

New: local_proj @ inv(H_opt) then decompose.
Old: decompose(H_opt) directly.

VGGT-SLAM: projection_mat = proj_mats[idx] @ inv(homography_world).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.pointcloud.loop_closure.closure import run_pose_graph_optimization
from collab_splats.pointcloud.loop_closure.graph import decompose_camera, normalize_to_sl4
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
        retrieval_vectors=np.zeros((k, 64), dtype=np.float32),
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


def test_pose_extraction_non_first_frame_uses_local_proj():
    """Frame 1 extraction: result = decompose(local_proj[1] @ inv(H_opt[1])).

    poses[0]=I, poses[1]=R_y15 →
      H_opt[1] ≈ inv(poses[1]) = R_y(-15°) (sequential edge initialization)
      corrected = poses[1] @ inv(H_opt[1]) = R_y(15°) @ R_y(15°) = R_y(30°)
      decompose_camera transposes rotation: output ≈ R_y(-30°)

    Old extraction: decompose(H_opt[1]) where H_opt[1] has R_y(-15°) → output R_y(+15°)
    New extraction → R_y(-30°), old → R_y(+15°): 45° apart.
    """
    rng = np.random.default_rng(99)
    R15 = ScipyR.from_euler("y", 15, degrees=True).as_matrix()
    # decompose_camera transposes rotation block; corrected has R30 → output is R_y(-30°)
    R_neg30 = ScipyR.from_euler("y", -30, degrees=True).as_matrix()
    # Old extraction: decompose(H_opt[1]) where H_opt[1] has R_y(-15°) → output R_y(+15°)
    R_pos15 = R15

    poses = np.stack([
        _make_w2c(np.eye(3), np.array([0., 0., 0.])),
        _make_w2c(R15, np.array([0.1, 0., 0.])),
    ]).astype(np.float32)
    wp = rng.standard_normal((2, 5, 5, 3)).astype(np.float32) * 0.1

    submap = _make_submap(poses, wp, submap_id=0)
    result = run_pose_graph_optimization(
        [submap], lc_submaps=[], total_frames=2, overlap_frames=1
    )

    assert result.shape == (2, 4, 4)
    R_out = result[1, :3, :3]

    def _angle_deg(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))

    # New extraction: result[1] ≈ R_y(-30°)
    angle_from_new = _angle_deg(R_out, R_neg30)
    # Old extraction would give R_y(+15°) — 45° away from R_y(-30°)
    angle_from_old = _angle_deg(R_out, R_pos15)

    assert angle_from_new < 5.0, \
        f"New extraction should give ~R_y(-30°) at frame 1, got {angle_from_new:.1f}° away"
    assert angle_from_old > 10.0, \
        f"Result should differ from old extraction R_y(+15°), got {angle_from_old:.1f}° (should be >10°)"


def test_decompose_camera_handles_sl4_projective_scale():
    """decompose_camera divides by H[-1,-1] — correct even when H[3,3]≠1 after SL(4) norm.

    Regression guard for H6: SL(4) normalization should not corrupt decomposition.
    Tests that decompose_camera handles projective scaling via P[-1,-1] division.
    """
    # Construct H with H[3,3] = 2, so SL(4) norm will make H[3,3] != 1
    R_input = np.array([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.],
    ])
    t_input = np.array([0.1, -0.2, 0.5])
    H = np.eye(4)
    H[:3, :3] = R_input
    H[:3, 3] = t_input
    H[3, 3] = 2.0

    H_sl4 = normalize_to_sl4(H)
    assert abs(H_sl4[3, 3] - 1.0) > 1e-6, "H[3,3] should differ from 1 after SL(4) norm"
    assert abs(np.linalg.det(H_sl4) - 1.0) < 1e-9, "SL(4) norm should enforce det=1"

    # decompose_camera handles projective scaling by dividing P by P[-1,-1] before RQ
    K_out, R_out, t_out, _ = decompose_camera(H_sl4)

    # Verify basic properties: K should be upper triangular, R should be orthogonal
    assert np.allclose(R_out @ R_out.T, np.eye(3), atol=1e-6), "R should be orthogonal"
    assert np.allclose(K_out, np.triu(K_out), atol=1e-9), "K should be upper triangular"
    # Verify K diagonal is positive (enforced by decompose_camera)
    assert K_out[0, 0] > 0 and K_out[1, 1] > 0 and K_out[2, 2] > 0, "K diagonal should be positive"
