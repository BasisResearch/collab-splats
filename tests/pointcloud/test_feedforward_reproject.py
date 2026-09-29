"""Unit tests for PointcloudResult.reproject()."""
from pathlib import Path

import numpy as np
import pytest

from collab_splats.pointcloud.base import PointcloudResult


def _make_result(*, depth=None, pixel_indices=None) -> PointcloudResult:
    """Minimal PointcloudResult with configurable depth and pixel_indices."""
    N, H, W, P = 2, 4, 4, 3
    return PointcloudResult(
        points=np.zeros((P, 3), dtype=np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=None,
        model_intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img_{i}.png") for i in range(N)],
        original_coords=np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (N, 1)),  # full-frame box
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
    )


def test_reproject_unprojects_source_pixels_under_current_poses():
    # Non-identity poses and an off-center K, so R vs R.T and u vs v are both visible
    N, H, W = 2, 4, 5
    rng = np.random.default_rng(0)
    depth = (rng.random((N, H, W)) + 1.0).astype(np.float32)
    pixel_indices = np.array([[0, 1, 3], [1, 2, 0], [0, 3, 4]], dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)
    result.model_intrinsics[:] = np.array([[50.0, 0, 1.5], [0, 60.0, 2.25], [0, 0, 1]], dtype=np.float32)
    for i in range(N):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        result.extrinsics[i, :3, :3] = q * np.sign(np.linalg.det(q))
        result.extrinsics[i, :3, 3] = rng.normal(size=3)

    reprojected = result.reproject()

    # Pinhole ray at each source pixel, scaled by its depth, taken camera -> world
    K = result.model_intrinsics[0].astype(np.float64)
    for (f, r, c), point in zip(pixel_indices, reprojected.points):
        d = float(depth[f, r, c])
        cam = np.array([(c - K[0, 2]) / K[0, 0] * d, (r - K[1, 2]) / K[1, 1] * d, d])
        R, t = result.extrinsics[f, :3, :3].astype(np.float64), result.extrinsics[f, :3, 3].astype(np.float64)
        np.testing.assert_allclose(point, R.T @ (cam - t), rtol=1e-5, atol=1e-5)
    assert reprojected.points.dtype == np.float32

    # world_points is the full per-pixel grid the points were sampled from
    assert reprojected.world_points.shape == (N, H, W, 3)
    frame, row, col = pixel_indices.T
    np.testing.assert_array_equal(reprojected.world_points[frame, row, col], reprojected.points)


def test_reproject_preserves_colors_and_extrinsics():
    N, H, W, P = 2, 4, 4, 3
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.zeros((P, 3), dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)

    reprojected = result.reproject()

    # World positions updated; source-pixel-derived fields unchanged
    np.testing.assert_array_equal(reprojected.colors, result.colors)
    np.testing.assert_array_equal(reprojected.extrinsics, result.extrinsics)
    assert reprojected is not result


def test_reproject_rejects_4d_depth():
    result = _make_result(depth=np.ones((2, 4, 4, 1), np.float32), pixel_indices=np.zeros((1, 3), np.int32))
    with pytest.raises(ValueError, match="depth"):
        result.reproject()


def test_reproject_raises_without_depth():
    P = 3
    pixel_indices = np.zeros((P, 3), dtype=np.int32)
    result = _make_result(depth=None, pixel_indices=pixel_indices)
    with pytest.raises(ValueError, match="reproject\\(\\) requires depth"):
        result.reproject()


def test_reproject_raises_without_pixel_indices():
    N, H, W = 2, 4, 4
    depth = np.ones((N, H, W), dtype=np.float32)
    result = _make_result(depth=depth, pixel_indices=None)
    with pytest.raises(ValueError, match="pixel_indices"):
        result.reproject()
