import numpy as np
import torch

from collab_splats.pointcloud.utils import (
    clean_pointcloud,
    cross_frame_attention_ratio,
    fit_dominant_plane,
)


########################################################
########## clean_pointcloud ############################
########################################################


def test_clean_pointcloud_masks_the_far_outlier():
    """
    A tight cluster plus one distant point: the mask keeps the cluster, drops the outlier.
    """
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3))
    points = np.vstack([cluster, [[50.0, 50.0, 50.0]]])

    keep = clean_pointcloud(points)

    assert keep.dtype == bool
    assert keep.shape == (201,)
    assert keep[:200].all()
    assert not keep[200]


def test_clean_pointcloud_keeps_everything_when_too_few_points():
    """
    Below the neighborhood size open3d rejects the whole cloud — keep all instead.
    """
    keep = clean_pointcloud(np.zeros((5, 3), dtype=np.float32))

    assert keep.shape == (5,)
    assert keep.all()


def test_clean_pointcloud_empty_cloud():
    """
    An empty cloud yields an empty mask, not an error.
    """
    keep = clean_pointcloud(np.zeros((0, 3)))

    assert keep.shape == (0,)


def test_clean_pointcloud_degenerate_cloud_keeps_all():
    """
    Duplicate points above the neighborhood size: open3d rejects every point, so the
    all-rejected guard keeps the cloud rather than emptying the reconstruction.
    """
    keep = clean_pointcloud(np.tile([1.0, 2.0, 3.0], (50, 1)))

    assert keep.shape == (50,)
    assert keep.all()


########################################################
########## cross_frame_attention_ratio #################
########################################################


def test_cross_frame_attention_ratio_returns_float():
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    result = cross_frame_attention_ratio(k, q, token_offset=0)
    assert isinstance(result, float)


def test_cross_frame_attention_ratio_similar_frames_high():
    """Identical content in both frame halves → ratio close to 1.0."""
    torch.manual_seed(0)
    B, heads, hd = 1, 2, 4
    # N=20: 10 tokens per frame, both frames have identical feature vectors
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    ratio = cross_frame_attention_ratio(k, q, token_offset=0)
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
    ratio = cross_frame_attention_ratio(k, q, token_offset=0)
    assert ratio < 0.2


def test_cross_frame_attention_ratio_empty_returns_zero():
    """If token_offset >= tokens_per_img, k_first is empty → return 0.0."""
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    # token_offset=10 means k_first = k[:, :, 10:10, :] which is empty
    result = cross_frame_attention_ratio(k, q, token_offset=10)
    assert result == 0.0


########################################################
########## fit_dominant_plane ##########################
########################################################


def test_fit_dominant_plane_flat_z_up():
    """Flat ground at z=-1 → R≈I, t brings floor to z=0."""
    rng = np.random.default_rng(42)
    # Ground plane at z = -1 with small noise
    xy = rng.uniform(-5, 5, (800, 2)).astype(np.float32)
    z = rng.normal(-1.0, 0.005, (800,)).astype(np.float32)
    ground = np.column_stack([xy, z])
    # Scatter above-ground points
    above_xy = rng.uniform(-5, 5, (100, 2)).astype(np.float32)
    above_z = rng.uniform(-0.5, 2.0, (100,)).astype(np.float32)
    above = np.column_stack([above_xy, above_z])
    points = np.vstack([ground, above])

    R, t = fit_dominant_plane(points)

    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # After applying transform, floor z-mean should be ≈ 0
    pts_aligned = (R @ points[:800].T).T + t
    np.testing.assert_allclose(pts_aligned[:, 2].mean(), 0.0, atol=0.1)


def test_fit_dominant_plane_returns_valid_rotation():
    """R is a proper rotation matrix (det=1, orthogonal)."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    pts[:400, 2] = rng.normal(0, 0.01, 400)  # flat-ish ground at z=0
    R, t = fit_dominant_plane(pts)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
