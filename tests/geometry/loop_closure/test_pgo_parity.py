"""Tests verifying VGGT-SLAM parity changes:

1. Scale estimation uses full extrinsic poses, not just intrinsics.
   When there is a rotation between submaps, the old intrinsics-only
   code would compare points in misaligned frames (wrong scale).
   The new code transforms via the overlap frame's w2c matrices first.

2. submap_overlap default is 1 (VGGT-SLAM default, not our old 4).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyR

from collab_splats.geometry.loop_closure.graph import (
    estimate_scale_pairwise,
    run_pose_graph_optimization,
)
from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig
from collab_splats.geometry.loop_closure.submap import Submap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 world-to-camera matrix from R (3×3) and t (3,)."""
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
        # Submap.world_points is (K, P, 3) per the production contract
        # (_raw_to_world_points returns (K, P, 3)). Flatten any per-pixel
        # (K, H, W, 3) grid into (K, P, 3) for closure.py.
        world_points=np.asarray(world_points, dtype=np.float32).reshape(k, -1, 3),
        world_points_conf=None,
    )


# ---------------------------------------------------------------------------
# Test 1: Scale estimation with rotation between submaps
# ---------------------------------------------------------------------------


def test_scale_estimation_survives_intersubmap_rotation():
    """Scale estimation with full w2c poses vs intrinsics-only (K=I→T=I).

    The key failure mode: curr world origin is far from prev world origin.
    Old code leaves curr_pts at their small curr-world magnitudes while
    prev_pts are large → scale wildly off.  New code applies the full T
    (rotation + translation) first, recovering the correct 1/true_scale ratio.

    Why 1/true_scale?  H_scale = diag([scale, scale, scale, 1]) is right-multiplied
    into H_w.  When curr world is at true_scale times prev world, the SL4 homography
    for the overlap frame satisfies P_prev = P_curr @ diag([true_scale,...,1]), so
    H_scale must be diag([1/true_scale,...,1]) — i.e. scale = 1/true_scale.
    """
    rng = np.random.default_rng(42)
    true_scale = 3.0

    # Canonical scene: small cluster of points near the origin
    N = 200
    X_scene = rng.standard_normal((N, 3)) * 0.1

    # Prev world: scene lives far from prev-world origin at [D, 0, 0].
    # Overlap-frame camera sits at the prev-world origin (P_prev_ov = I).
    D = 10.0
    X_prev = X_scene + np.array([D, 0.0, 0.0])
    P_prev_ov = np.eye(4, dtype=np.float64)

    # Curr world: 30° rotation + true_scale relative to prev world.
    # Same physical points expressed in curr coords: true_scale * R_w @ X_scene
    # (scene is near the curr-world origin, which is shifted to [D,0,0] in prev frame).
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    X_curr = true_scale * (R_w @ X_scene.T).T

    # Overlap-frame camera (prev-world origin [0,0,0]) expressed in curr world:
    #   C_cam_curr = true_scale * R_w @ (C_cam_prev - scene_offset)
    #              = true_scale * R_w @ [-D, 0, 0]
    C_cam_curr = true_scale * (R_w @ np.array([-D, 0.0, 0.0]))
    R_cam_curr = R_w.T  # compensate for world rotation
    t_cam_curr = -R_cam_curr @ C_cam_curr
    P_curr_ov = np.eye(4, dtype=np.float64)
    P_curr_ov[:3, :3] = R_cam_curr
    P_curr_ov[:3, 3] = t_cam_curr

    # --- Old code: K=I → P_temp=I → no frame alignment ---
    curr_in_prev_old = X_curr.copy()
    scale_old = estimate_scale_pairwise(curr_in_prev_old, X_prev)

    # --- New code: full w2c poses ---
    T = np.linalg.inv(P_prev_ov) @ P_curr_ov
    curr_h = np.hstack([X_curr, np.ones((N, 1))])
    curr_in_prev_new = (T @ curr_h.T).T[:, :3]
    scale_new = estimate_scale_pairwise(curr_in_prev_new, X_prev)

    # New code: scale = 1/true_scale (the H_scale correction for a 3× curr world)
    expected = 1.0 / true_scale
    assert abs(scale_new - expected) / expected < 0.05, f"New scale {scale_new:.4f} far from expected {expected:.4f}"
    # Old code: X_curr is near origin (magnitude ~0.3) but X_prev is far (magnitude ~10)
    # → scale_old >> expected (regression guard)
    assert scale_old > 5.0, f"Old code should give a large wrong scale, got {scale_old:.3f}"


# ---------------------------------------------------------------------------
# Test 2: submap_overlap default is 1
# ---------------------------------------------------------------------------


def test_loop_closure_config_default_overlap_is_1():
    cfg = LoopClosureConfig()
    assert cfg.submap_overlap == 1, f"Expected default submap_overlap=1 (VGGT-SLAM parity), got {cfg.submap_overlap}"


# ---------------------------------------------------------------------------
# Test 3: PGO with overlap=1 builds correct edge topology
# ---------------------------------------------------------------------------


def test_pgo_overlap_1_connects_submaps():
    """With overlap=1, PGO should produce N total unique frames (no duplication)
    and the result shape should match total_frames.
    """
    rng = np.random.default_rng(7)
    k = 4  # frames per submap
    n_submaps = 3

    # Build simple sequential submaps with known geometry
    submaps = []
    global_t = 0.0
    for si in range(n_submaps):
        poses = np.stack([_make_w2c(np.eye(3), np.array([global_t + i * 0.1, 0.0, 0.0])) for i in range(k)])
        wp = rng.standard_normal((k, 5, 5, 3)).astype(np.float64) * 0.1
        submaps.append(_make_submap(poses.astype(np.float32), wp, submap_id=si))
        global_t += (k - 1) * 0.1  # advance by k-1 (1-frame overlap)

    total_frames = k + (k - 1) * (n_submaps - 1)  # 4 + 3 + 3 = 10 for overlap=1
    result = run_pose_graph_optimization(submaps, lc_submaps=[], total_frames=total_frames, overlap_frames=1)
    assert result.shape == (total_frames, 4, 4), f"Expected ({total_frames}, 4, 4), got {result.shape}"
    # First frame should be near identity (pinned by prior)
    assert np.allclose(result[0], np.eye(4), atol=0.1), "First frame should be near identity"


# ---------------------------------------------------------------------------
# Test 4: Confidence masking reduces scale noise
# ---------------------------------------------------------------------------


def test_confidence_masking_reduces_scale_noise():
    """Confidence filtering excludes noisy low-conf points from scale estimation.

    estimate_scale_pairwise(X, Y) = median(||Y[i]|| / ||X[i]||).
    Production call: estimate_scale_pairwise(curr_in_prev, prev_pts).
    When curr world is world_scale× larger than prev world, the function
    returns ~1/world_scale (the SL4 correction factor).

    Setup: 20 good points (conf=50, ratio=0.5) + 80 noisy points (conf=5,
    ratio≈50, far from truth). Noisy majority corrupts the unmasked median
    but the joint mask (conf>25) selects only the 20 good points.
    """
    rng = np.random.default_rng(42)
    world_scale = 2.0
    expected_scale = 1.0 / world_scale  # 0.5

    N_good = 20
    N_noisy = 80  # majority — enough to shift unmasked median far from truth

    # Good points: prev far from origin, curr = world_scale × prev → ratio = 0.5
    X_prev_good = rng.standard_normal((N_good, 3)) + np.array([10.0, 0.0, 0.0])
    X_curr_in_prev_good = world_scale * X_prev_good  # ratio ||prev||/||curr|| = 0.5

    # Noisy points: large curr norms, small prev norms → ratio ≈ 50 (far from 0.5)
    X_prev_noisy = rng.standard_normal((N_noisy, 3)) * 0.01 + np.array([0.1, 0.0, 0.0])
    X_curr_in_prev_noisy = rng.standard_normal((N_noisy, 3)) * 5.0 + np.array([10.0, 0.0, 0.0])

    X_prev = np.vstack([X_prev_good, X_prev_noisy])
    X_curr_in_prev = np.vstack([X_curr_in_prev_good, X_curr_in_prev_noisy])

    # Confidence: good=50 (above threshold 25), noisy=5 (below threshold)
    conf_threshold = 25.0
    conf_prev = np.array([50.0] * N_good + [5.0] * N_noisy, dtype=np.float32)
    conf_curr = np.array([50.0] * N_good + [5.0] * N_noisy, dtype=np.float32)

    # Without masking: 80 noisy points dominate the median, pulling it away from 0.5
    scale_unmasked = estimate_scale_pairwise(X_curr_in_prev, X_prev)

    # With joint mask (conf > threshold on both sides): only the 20 good points
    joint_mask = (conf_curr > conf_threshold) & (conf_prev > conf_threshold)
    assert joint_mask.sum() == N_good, f"Should have exactly {N_good} good points in mask"
    scale_masked = estimate_scale_pairwise(X_curr_in_prev[joint_mask], X_prev[joint_mask])

    err_masked = abs(scale_masked - expected_scale) / expected_scale
    err_unmasked = abs(scale_unmasked - expected_scale) / expected_scale

    assert err_masked < 0.05, f"Masked scale {scale_masked:.3f} far from expected {expected_scale:.3f}"
    assert err_unmasked > err_masked, (
        f"Masking should improve estimate: masked_err={err_masked:.3f}, " f"unmasked_err={err_unmasked:.3f}"
    )
