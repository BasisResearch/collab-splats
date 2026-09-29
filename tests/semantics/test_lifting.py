import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.semantics.lifting import lift_features

########################################################
########## lift_features (multi-view) ##################
########################################################


def _make_lift_result(
    pts3d: np.ndarray,
    pixel_indices: np.ndarray,
    depth: np.ndarray,
    conf: np.ndarray,
    *,
    n: int,
    h: int,
    w: int,
) -> PointcloudResult:
    """Build a minimal PointcloudResult sufficient for lift_features.

    Uses identity world-to-cam extrinsics and a pinhole intrinsics centered at (w/2, h/2).
    """
    extrinsics_3x4 = np.tile(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32), (n, 1, 1)
    )
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1, 1))
    extrinsics_4x4 = np.concatenate([extrinsics_3x4, bottom], axis=1)
    intrinsics = np.tile(
        np.array([[100, 0, w / 2], [0, 100, h / 2], [0, 0, 1]], dtype=np.float32), (n, 1, 1)
    )
    return PointcloudResult(
        points=pts3d,
        colors=np.zeros((pts3d.shape[0], 3), dtype=np.uint8),
        extrinsics=extrinsics_4x4,
        intrinsics=None,
        model_intrinsics=intrinsics,
        image_paths=[Path(f"frame_{i}.png") for i in range(n)],
        original_coords=np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (n, 1)),  # full-frame box
        model_width=w,
        model_height=h,
        pixel_indices=pixel_indices,
        depth=depth,
        confidence=torch.from_numpy(conf),
    )


def test_lift_features_aggregates_across_frames():
    """Point visible in 2 frames with conf (1, 2) → weighted mean of feats (1, 2)."""
    n, h, w, D = 2, 8, 8, 4
    # Point at world (0, 0, 1); identity extrinsics → both frames project to (cx, cy) = (4, 4)
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth at the projected pixel matches z=1 in both frames → depth-consistent
    depth = np.ones((n, h, w), dtype=np.float32)
    # Distinct conf per frame: 1 in frame 0, 2 in frame 1
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[0, 4, 4] = 1.0
    conf[1, 4, 4] = 2.0
    # Feature maps: frame 0 all ones, frame 1 all twos
    feature_maps = [
        torch.full((D, h, w), 1.0, dtype=torch.float32),
        torch.full((D, h, w), 2.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    # Weighted mean: (1*1 + 2*2) / (1 + 2) = 5/3
    expected = 5.0 / 3.0
    np.testing.assert_allclose(feats[0].numpy(), [expected] * D, atol=1e-4)


def test_lift_features_source_fallback_for_unseen():
    """Point with zero accumulated weight everywhere → source-frame sample fallback."""
    n, h, w, D = 2, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    # Depth far from z=1 in both frames → depth-consistency fails everywhere → zero weight
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    # Distinct features per frame; fallback should sample from frame 0 (source)
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    # Source frame is 0 (pixel_indices says so) → expect feature 7.0
    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_reads_depth_at_rounded_pixel():
    """Visibility reads the depth map at the ROUNDED projected pixel (depth_residual's nearest)."""
    n, h, w, D = 2, 8, 8, 4
    # Point projects to u = 4 + 100 * 0.006 = 4.6, v = 4: floor reads col 4, round reads col 5
    # - col 4 agrees with z=1, col 5 does not, so only a floor read would see the point
    # - seen nowhere → source-frame fallback (frame 0 → 7.0); a floor read gives mean(7, 9) = 8.0
    pts3d = np.array([[0.006, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.full((n, h, w), 100.0, dtype=np.float32)
    depth[:, :, 4] = 1.0
    conf = np.ones((n, h, w), dtype=np.float32)
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_weights_by_confidence_at_rounded_pixel():
    """The confidence weight is read at the same ROUNDED pixel as the depth test."""
    n, h, w, D = 2, 8, 8, 4
    # Point projects to u = 4.6, v = 4 and is depth-consistent everywhere
    # - frame 0 has confidence only at col 5 (round), frame 1 only at col 4 (floor)
    # - a round read weights frame 0 alone → 7.0; a floor read weights frame 1 alone → 9.0
    pts3d = np.array([[0.006, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[0, :, 5] = 1.0
    conf[1, :, 4] = 1.0
    feature_maps = [
        torch.full((D, h, w), 7.0, dtype=torch.float32),
        torch.full((D, h, w), 9.0, dtype=torch.float32),
    ]

    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feats = lift_features(feature_maps, result)

    np.testing.assert_allclose(feats[0].numpy(), [7.0] * D, atol=1e-4)


def test_lift_features_asserts_missing_depth():
    """Missing required field (depth) raises AssertionError with clear message."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    # Drop depth to trigger assert
    result = dataclasses.replace(result, depth=None)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(AssertionError, match="depth"):
        lift_features(feature_maps, result)


def test_lift_features_asserts_frame_count_mismatch():
    """feature_maps length must match number of frames in extrinsics."""
    n, h, w, D = 2, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    feature_maps = [torch.zeros(D, h, w)]  # only 1 map, 2 frames

    with pytest.raises(AssertionError, match="frame count"):
        lift_features(feature_maps, result)


def test_lift_features_rejects_4d_depth():
    """depth must be (N, H, W); a trailing channel dim raises ValueError."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    result = dataclasses.replace(result, depth=depth[..., None])  # (N, H, W, 1)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(ValueError, match="depth"):
        lift_features(feature_maps, result)


def test_lift_features_rejects_3x4_extrinsics():
    """extrinsics must be (N, 4, 4); a 3x4 (no homogeneous row) matrix raises ValueError."""
    n, h, w, D = 1, 8, 8, 4
    pts3d = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    pixel_indices = np.array([[0, 4, 4]], dtype=np.int32)
    depth = np.ones((n, h, w), dtype=np.float32)
    conf = np.ones((n, h, w), dtype=np.float32)
    result = _make_lift_result(pts3d, pixel_indices, depth, conf, n=n, h=h, w=w)
    result = dataclasses.replace(result, extrinsics=result.extrinsics[:, :3, :])  # (N, 3, 4)
    feature_maps = [torch.zeros(D, h, w)]

    with pytest.raises(ValueError, match="4, 4"):
        lift_features(feature_maps, result)
