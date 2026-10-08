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

from collab_splats.geometry.loop_closure.graph import calculate_pairwise_frame_scale
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.loop_closure.wrapper import LoopClosureConfig
from tests.geometry.loop_closure._helpers import drive_pose_graph

########################################################################
# Helpers
########################################################################


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
        frame_start=submap_id * k,
        # Submap.points is a dense (K, H, W, 3) grid; lay any (K, P, 3) input out as H = 1
        points=np.asarray(world_points, dtype=np.float32).reshape(k, 1, -1, 3),
    )


def _one_frame_submap(submap_id, pts, pose=None, conf=None, conf_threshold=None) -> Submap:
    """1-frame submap with K = I; P points laid out as a (1, 1, P, 3) grid."""
    n = len(pts)
    pose = np.eye(4) if pose is None else pose
    return Submap(
        submap_id=submap_id,
        poses=pose[None].astype(np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        image_paths=[f"frame_{submap_id}.png"],
        points=np.asarray(pts, dtype=np.float32).reshape(1, 1, n, 3),
        conf=None if conf is None else np.asarray(conf, dtype=np.float32).reshape(1, 1, n),
        conf_threshold=conf_threshold,
    )


def _frame_scale(curr_pts, prior_pts, curr_pose=None, curr_conf=None, prior_conf=None, conf_threshold=None) -> float:
    """calculate_pairwise_frame_scale between two 1-frame submaps, min_conf_points 10."""
    curr = _one_frame_submap(1, curr_pts, curr_pose, curr_conf, conf_threshold)
    prior = _one_frame_submap(0, prior_pts, None, prior_conf, conf_threshold)
    return calculate_pairwise_frame_scale(curr, 0, prior, 0, min_conf_points=10)


########################################################################
# Test 0: median norm ratio
########################################################################


def test_frame_scale_known_ratio():
    rng = np.random.default_rng(1)
    X = rng.random((50, 3))
    assert abs(_frame_scale(X, X * 2.5) - 2.5) < 0.01


def test_frame_scale_degenerate_source_falls_back_to_one():
    # Every source point at the origin: no valid ratio, scale 1.0
    assert _frame_scale(np.zeros((5, 3)), np.ones((5, 3))) == 1.0


########################################################################
# Test 1: Scale estimation with rotation between submaps
########################################################################


def test_scale_estimation_survives_intersubmap_rotation():
    """Scale estimation moves each frame's points into its own camera first.

    The key failure mode: curr world origin is far from prev world origin.
    Skipping the camera move leaves curr_pts at their small curr-world magnitudes while
    prev_pts are large → scale wildly off.  Applying the full w2c pose
    (rotation + translation) first recovers the correct 1/true_scale ratio.

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

    # Prev world: scene at [D, 0, 0], overlap-frame camera at the origin (P_prev_ov = I)
    D = 10.0
    X_prev = X_scene + np.array([D, 0.0, 0.0])
    P_prev_ov = np.eye(4, dtype=np.float64)

    # Curr world: same points as true_scale * R_w @ X_scene, near the curr origin (prev [D, 0, 0])
    R_w = ScipyR.from_euler("z", 30, degrees=True).as_matrix()
    X_curr = true_scale * (R_w @ X_scene.T).T

    # Overlap-frame camera in curr world: true_scale * R_w @ (0 - scene_offset) = true_scale * R_w @ [-D, 0, 0]
    C_cam_curr = true_scale * (R_w @ np.array([-D, 0.0, 0.0]))
    R_cam_curr = R_w.T  # compensate for world rotation
    t_cam_curr = -R_cam_curr @ C_cam_curr
    P_curr_ov = np.eye(4, dtype=np.float64)
    P_curr_ov[:3, :3] = R_cam_curr
    P_curr_ov[:3, 3] = t_cam_curr

    # Without the camera move (identity curr pose) vs with the full w2c pose
    assert np.array_equal(P_prev_ov, np.eye(4))
    scale_old = _frame_scale(X_curr, X_prev)
    scale_new = _frame_scale(X_curr, X_prev, curr_pose=P_curr_ov)

    # With the camera move: scale = 1/true_scale (the H_scale correction for a 3× curr world)
    expected = 1.0 / true_scale
    assert abs(scale_new - expected) / expected < 0.05, f"New scale {scale_new:.4f} far from expected {expected:.4f}"
    # Without it: X_curr near origin (~0.3) vs X_prev far (~10), so scale_old >> expected
    assert scale_old > 5.0, f"Unmoved points should give a large wrong scale, got {scale_old:.3f}"


########################################################################
# Test 2: submap_overlap default is 1
########################################################################


def test_loop_closure_config_default_overlap_is_1():
    cfg = LoopClosureConfig()
    assert cfg.submap_overlap == 1, f"Expected default submap_overlap=1 (VGGT-SLAM parity), got {cfg.submap_overlap}"


########################################################################
# Test 3: PGO with overlap=1 builds correct edge topology
########################################################################


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
    result = drive_pose_graph(submaps, lc_submaps=[], total_frames=total_frames, overlap_frames=1)
    assert result.shape == (total_frames, 4, 4), f"Expected ({total_frames}, 4, 4), got {result.shape}"
    # First frame should be near identity (pinned by prior)
    assert np.allclose(result[0], np.eye(4), atol=0.1), "First frame should be near identity"


########################################################################
# Test 4: Confidence masking reduces scale noise
########################################################################


def test_confidence_masking_reduces_scale_noise():
    """Confidence filtering excludes noisy low-conf points from scale estimation.

    The scale is median(||prior[i]|| / ||curr[i]||) over the masked points.
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

    # Without conf: 80 noisy points dominate the median, pulling it away from 0.5
    scale_unmasked = _frame_scale(X_curr_in_prev, X_prev)

    # With conf on both sides: the joint tier (both > threshold) keeps only the 20 good points
    joint_mask = (conf_curr > conf_threshold) & (conf_prev > conf_threshold)
    assert joint_mask.sum() == N_good, f"Should have exactly {N_good} good points in mask"
    scale_masked = _frame_scale(
        X_curr_in_prev, X_prev, curr_conf=conf_curr, prior_conf=conf_prev, conf_threshold=conf_threshold
    )

    err_masked = abs(scale_masked - expected_scale) / expected_scale
    err_unmasked = abs(scale_unmasked - expected_scale) / expected_scale

    assert err_masked < 0.05, f"Masked scale {scale_masked:.3f} far from expected {expected_scale:.3f}"
    assert err_unmasked > err_masked, (
        f"Masking should improve estimate: masked_err={err_masked:.3f}, " f"unmasked_err={err_unmasked:.3f}"
    )
