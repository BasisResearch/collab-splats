from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.utils import (
    clean_pointcloud,
    cross_frame_attention_ratio,
    frame_depths,
    outlier_mask,
)

########################################################
########## outlier_mask ################################
########################################################


def test_outlier_mask_masks_the_far_outlier():
    """
    A tight cluster plus one distant point: the mask keeps the cluster, drops the outlier.
    """
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3))
    points = np.vstack([cluster, [[50.0, 50.0, 50.0]]])

    keep = outlier_mask(points)

    assert keep.dtype == bool
    assert keep.shape == (201,)
    assert keep[:200].all()
    assert not keep[200]


@pytest.mark.parametrize("n", [0, 5])
def test_outlier_mask_keeps_everything_when_too_few_points(n):
    """
    Below the neighborhood size open3d rejects the whole cloud — keep all; empty is not an error.
    """
    keep = outlier_mask(np.zeros((n, 3), dtype=np.float32))

    assert keep.shape == (n,)
    assert keep.all()


def test_outlier_mask_degenerate_cloud_keeps_all():
    """
    Duplicate points above the neighborhood size: open3d rejects every point, so the
    all-rejected guard keeps the cloud rather than emptying the reconstruction.
    """
    keep = outlier_mask(np.tile([1.0, 2.0, 3.0], (50, 1)))

    assert keep.shape == (50,)
    assert keep.all()


@pytest.mark.parametrize("remove_outliers, max_points, n_kept", [(True, 50, 50), (False, 1000, 200)])
def test_clean_pointcloud_row_aligned(remove_outliers, max_points, n_kept):
    """
    SOR on: far point dropped, cap binds. Off: all kept, outlier too. Colors follow the points.
    """
    points = np.random.default_rng(0).normal(scale=0.01, size=(200, 3))
    points[0] = 50.0
    result = PointcloudResult(
        points=points.astype(np.float32),
        colors=np.arange(600).reshape(200, 3).astype(np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None],
        intrinsics=None,
        model_intrinsics=np.array([[[10.0, 0, 5], [0, 10.0, 4], [0, 0, 1]]], np.float32),
        image_paths=[Path("frame_000000.png")],
        original_coords=np.array([[0, 0, 10, 8, 10, 8]], np.float32),
        model_width=10,
        model_height=8,
        pixel_indices=np.arange(600).reshape(200, 3).astype(np.int32),
    )

    out = clean_pointcloud(result, remove_outliers=remove_outliers, max_points=max_points)

    rows = out.pixel_indices[:, 0] // 3
    assert len(out.points) == n_kept
    assert (0 in rows) != remove_outliers
    np.testing.assert_array_equal(out.points, result.points[rows])
    np.testing.assert_array_equal(out.colors, result.colors[rows])


########################################################
########## cross_frame_attention_ratio #################
########################################################


def test_cross_frame_attention_ratio_returns_per_token_array():
    B, heads, N, hd = 1, 2, 20, 4
    g = torch.Generator().manual_seed(0)
    k = torch.randn(B, heads, N, hd, generator=g)
    q = torch.randn(B, heads, N, hd, generator=g)
    result = cross_frame_attention_ratio(k, q, token_offset=0)
    assert isinstance(result, np.ndarray)


def test_cross_frame_attention_ratio_similar_frames_high():
    """Identical content in both frame halves → ratio close to 1.0."""
    B, heads, hd = 1, 2, 4
    g = torch.Generator().manual_seed(0)

    # N=20: 10 tokens per frame, both frames have identical feature vectors
    half = torch.randn(B, heads, 10, hd, generator=g) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    ratios = cross_frame_attention_ratio(k, q, token_offset=0)
    thresh = np.percentile(ratios, 75)
    ratio = ratios[ratios >= thresh].mean()
    assert ratio > 0.8


def test_cross_frame_attention_ratio_orthogonal_frames_low():
    """Second frame tokens orthogonal to first → low cross-frame attention ratio."""
    B, heads, hd = 1, 1, 4
    N = 20
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    # First frame activations in dim 0
    k[:, :, :10, 0] = 10.0
    q[:, :, :10, 0] = 10.0
    # Second frame activations in dim 1 (orthogonal to first)
    k[:, :, 10:, 1] = 10.0
    q[:, :, 10:, 1] = 10.0
    ratios = cross_frame_attention_ratio(k, q, token_offset=0)
    thresh = np.percentile(ratios, 75)
    ratio = ratios[ratios >= thresh].mean()
    assert ratio < 0.2


def test_cross_frame_attention_ratio_empty_raises():
    """If token_offset >= tokens_per_img, k_first is empty → raises ValueError."""
    B, heads, N, hd = 1, 2, 20, 4
    tokens_per_img = N // 2
    g = torch.Generator().manual_seed(0)
    k = torch.randn(B, heads, N, hd, generator=g)
    q = torch.randn(B, heads, N, hd, generator=g)
    # token_offset=10 means k_first = k[:, :, 10:10, :] which is empty
    with pytest.raises(ValueError, match="no patch tokens"):
        cross_frame_attention_ratio(k, q, token_offset=tokens_per_img)


########################################################
########## frame_depths ################################
########################################################


def _depth_result(depth, confidence, frame_hw=(8, 8)):
    """Stand-in result carrying depth, confidence and full-frame original_coords."""
    n = len(depth)
    h, w = frame_hw

    return SimpleNamespace(
        depth=depth,
        confidence=confidence,
        original_coords=np.tile(np.array([0, 0, w, h, w, h]), (n, 1)),
    )


def test_frame_depths_masks_low_confidence_then_lifts_to_the_frame_grid():
    depth = np.full((1, 4, 4), 2.0, dtype=np.float32)
    confidence = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    rgbs = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    out = frame_depths(_depth_result(depth, confidence), rgbs, conf_percentile=50)

    assert out.shape == (1, 8, 8)
    assert (out == 0).mean() == 0.5
    assert (out[0, :4] == 0).all()
    assert np.isclose(out[0, 4:], 2.0).all()


def test_frame_depths_without_confidence_keeps_every_pixel():
    depth = np.full((1, 4, 4), 2.0, dtype=np.float32)
    rgbs = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    out = frame_depths(_depth_result(depth, None), rgbs, conf_percentile=50)

    assert (out > 0).all()


def test_frame_depths_refuses_a_result_without_depth():
    result = SimpleNamespace(depth=None, confidence=None, original_coords=None)

    with pytest.raises(ValueError, match="no depth"):
        frame_depths(result, np.zeros((1, 8, 8, 3), np.uint8), conf_percentile=None)
