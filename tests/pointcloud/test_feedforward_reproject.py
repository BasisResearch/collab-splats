"""Unit tests for FeedforwardResult.reproject()."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from collab_splats.pointcloud.feedforward.base import FeedforwardResult


def _make_result(*, depth=None, pixel_indices=None) -> FeedforwardResult:
    """Minimal FeedforwardResult with configurable depth and pixel_indices."""
    N, H, W, P = 2, 4, 4, 3
    return FeedforwardResult(
        pts3d=np.zeros((P, 3), dtype=np.float32),
        colors=np.zeros((P, 3), dtype=np.uint8),
        extrinsics=np.tile(np.eye(4), (N, 1, 1)).astype(np.float32),
        intrinsics=np.tile(np.eye(3), (N, 1, 1)).astype(np.float32),
        image_paths=[Path(f"img_{i}.png") for i in range(N)],
        original_coords=np.zeros((N, 6), dtype=np.float32),
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
    )


def test_reproject_calls_reproject_pixels_with_correct_args():
    N, H, W, P = 2, 4, 4, 3
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.array([[0, 1, 1], [1, 2, 2], [0, 3, 3]], dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)
    new_pts = np.ones((P, 3), dtype=np.float32) * 99.0

    with patch(
        "collab_splats.pointcloud.feedforward.base.reproject_pixels",
        return_value=new_pts,
    ) as mock_rp:
        reprojected = result.reproject()

    assert mock_rp.call_count == 1
    call_args = mock_rp.call_args[0]
    np.testing.assert_array_equal(call_args[0], depth)
    np.testing.assert_array_equal(call_args[1], pixel_indices)
    np.testing.assert_array_equal(call_args[2], result.extrinsics[:, :3, :])
    np.testing.assert_array_equal(call_args[3], result.intrinsics)
    np.testing.assert_array_equal(reprojected.pts3d, new_pts)


def test_reproject_preserves_colors_and_extrinsics():
    N, H, W, P = 2, 4, 4, 3
    depth = np.ones((N, H, W), dtype=np.float32)
    pixel_indices = np.zeros((P, 3), dtype=np.int32)
    result = _make_result(depth=depth, pixel_indices=pixel_indices)
    new_pts = np.full((P, 3), 7.0, dtype=np.float32)

    with patch(
        "collab_splats.pointcloud.feedforward.base.reproject_pixels",
        return_value=new_pts,
    ):
        reprojected = result.reproject()

    # World positions updated; source-pixel-derived fields unchanged
    np.testing.assert_array_equal(reprojected.colors, result.colors)
    np.testing.assert_array_equal(reprojected.extrinsics, result.extrinsics)
    assert reprojected is not result


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
