import numpy as np
import open3d as o3d
import pytest
import torch

from collab_splats.pointcloud.utils import (
    filter_distance,
    fit_dominant_plane,
    voxel_downsample,
    clean_pointcloud,
)


def _make_pcd(n: int = 200, seed: int = 0) -> o3d.geometry.PointCloud:
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rng.standard_normal((n, 3)).astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(rng.random((n, 3)))
    return pcd


# ---------------------------------------------------------------------------
# filter_distance — radial mode
# ---------------------------------------------------------------------------

def test_filter_distance_radial_max_distance():
    pcd = _make_pcd(500)
    # Plant obvious outlier far from centroid
    pts = np.asarray(pcd.points).copy()
    pts[0] = [1000.0, 1000.0, 1000.0]
    pcd.points = o3d.utility.Vector3dVector(pts)

    filtered = filter_distance(pcd, method="radial", max_distance=5.0)
    assert len(filtered.points) < 500
    assert np.asarray(filtered.points).max() < 10.0


def test_filter_distance_radial_n_points():
    pcd = _make_pcd(100)
    filtered = filter_distance(pcd, method="radial", n_points=50)
    assert len(filtered.points) == 50


def test_filter_distance_radial_return_mask():
    pcd = _make_pcd(100)
    filtered, mask = filter_distance(pcd, method="radial", max_distance=3.0, return_mask=True)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == 100
    assert mask.sum() == len(filtered.points)


def test_filter_distance_radial_origin_reference():
    pcd = _make_pcd(100)
    filtered = filter_distance(pcd, method="radial", max_distance=2.0, reference="origin")
    assert isinstance(filtered, o3d.geometry.PointCloud)


def test_filter_distance_radial_invalid_reference():
    pcd = _make_pcd(50)
    with pytest.raises(ValueError, match="reference"):
        filter_distance(pcd, method="radial", max_distance=1.0, reference="moon")


def test_filter_distance_radial_missing_params():
    pcd = _make_pcd(50)
    with pytest.raises(ValueError):
        filter_distance(pcd, method="radial")


def test_filter_distance_radial_n_points_too_large():
    pcd = _make_pcd(50)
    with pytest.raises(ValueError, match="n_points"):
        filter_distance(pcd, method="radial", n_points=999)


# ---------------------------------------------------------------------------
# filter_distance — bbox mode
# ---------------------------------------------------------------------------

def test_filter_distance_bbox_removes_outliers():
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    pts[:5] = 1000.0  # extreme outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((1000, 3)))

    filtered = filter_distance(pcd, method="bbox", percentile_range=(1.0, 99.0))
    assert len(filtered.points) < 1000
    assert np.asarray(filtered.points).max() < 100.0


def test_filter_distance_bbox_max_extent():
    pcd = _make_pcd(500)
    filtered = filter_distance(pcd, method="bbox", percentile_range=(0.0, 100.0), max_extent=1.0)
    pts = np.asarray(filtered.points)
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert extent.max() <= 1.01  # small tolerance for float precision


def test_filter_distance_bbox_empty():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    filtered = filter_distance(pcd, method="bbox")
    assert len(filtered.points) == 0


def test_filter_distance_unknown_method():
    pcd = _make_pcd(50)
    with pytest.raises(ValueError, match="Unknown filter_distance method"):
        filter_distance(pcd, method="sphere")


# ---------------------------------------------------------------------------
# voxel_downsample
# ---------------------------------------------------------------------------

def test_voxel_downsample_reduces_points():
    pcd = _make_pcd(5000)
    down, indices = voxel_downsample(pcd, voxel_size=0.3, adaptive=False)
    assert len(down.points) < 5000
    assert len(indices) == len(down.points)


def test_voxel_downsample_returns_valid_indices():
    pcd = _make_pcd(200)
    down, indices = voxel_downsample(pcd, adaptive=False)
    assert indices.max() < 200
    assert indices.min() >= 0


def test_voxel_downsample_adaptive():
    pcd = _make_pcd(15000)
    down, indices = voxel_downsample(pcd, voxel_size=0.3, adaptive=True)
    assert len(down.points) < 15000


def test_voxel_downsample_empty():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
    down, indices = voxel_downsample(pcd)
    assert len(down.points) == 0
    assert len(indices) == 0


# ---------------------------------------------------------------------------
# clean_pointcloud
# ---------------------------------------------------------------------------

def test_clean_pointcloud_returns_tuple():
    pcd = _make_pcd(300)
    result_pcd, indices = clean_pointcloud(pcd)
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert isinstance(indices, np.ndarray)


def test_clean_pointcloud_disable_all_stages():
    pcd = _make_pcd(100)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs=None,
    )
    assert len(result_pcd.points) == 100
    assert len(indices) == 100


def test_clean_pointcloud_downsample_kwargs():
    pcd = _make_pcd(5000)
    result_pcd, _ = clean_pointcloud(
        pcd,
        outlier_kwargs=None,
        distance_kwargs=None,
        downsample_kwargs={"voxel_size": 0.5, "adaptive": False},
    )
    assert len(result_pcd.points) < 5000


def test_clean_pointcloud_distance_kwargs():
    pcd = _make_pcd(300)
    result_large, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 100.0},
    )
    result_small, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 0.5},
    )
    assert len(result_large.points) >= len(result_small.points)


# ---------------------------------------------------------------------------
# cross_frame_attention_ratio
# ---------------------------------------------------------------------------

def test_cross_frame_attention_ratio_returns_float():
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    result = cross_frame_attention_ratio(k, q, token_offset=0)
    assert isinstance(result, float)


def test_cross_frame_attention_ratio_similar_frames_high():
    """Identical content in both frame halves → ratio close to 1.0."""
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
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
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
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
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    # token_offset=10 means k_first = k[:, :, 10:10, :] which is empty
    result = cross_frame_attention_ratio(k, q, token_offset=10)
    assert result == 0.0


# ---------------------------------------------------------------------------
# clean_pointcloud — new API (None=skip, kwargs merge)
# ---------------------------------------------------------------------------

def test_clean_pointcloud_skip_all_via_none():
    pcd = _make_pcd(300)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs=None,
    )
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert isinstance(indices, np.ndarray)


def test_clean_pointcloud_skip_downsample_via_none():
    """Skipping downsample only — point count may reduce from outlier/distance steps."""
    pcd = _make_pcd(5000)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
    )
    # All 5000 points available since no downsampling, but outlier/distance may remove some
    assert len(result_pcd.points) <= 5000
    assert len(indices) == len(result_pcd.points)


def test_clean_pointcloud_kwargs_merge_preserves_method():
    """Passing only max_distance should keep method='radial' from defaults."""
    pcd = _make_pcd(300)
    result_pcd, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={"max_distance": 100.0},
    )
    # Should not raise — merge must have kept method='radial'
    assert isinstance(result_pcd, o3d.geometry.PointCloud)


def test_clean_pointcloud_outlier_kwargs_overridable():
    """outlier_kwargs should now be overridable (previously hardcoded)."""
    pcd = _make_pcd(300)
    result_pcd, indices = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs={"nb_neighbors": 5, "std_ratio": 1.0},
        distance_kwargs=None,
    )
    assert isinstance(result_pcd, o3d.geometry.PointCloud)
    assert len(indices) <= 300


def test_clean_pointcloud_logging(caplog):
    """DEBUG logs must emit point counts for each active step."""
    import logging
    pcd = _make_pcd(300)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.pointcloud.utils"):
        clean_pointcloud(pcd)
    messages = caplog.text
    assert "downsample" in messages
    assert "outlier_removal" in messages
    assert "distance_removal" in messages
    assert "→" in messages


def test_clean_pointcloud_logging_skipped_step_absent(caplog):
    """Skipped steps must not appear in DEBUG logs."""
    import logging
    pcd = _make_pcd(300)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.pointcloud.utils"):
        clean_pointcloud(pcd, downsample_kwargs=None, outlier_kwargs=None)
    messages = caplog.text
    assert "downsample" not in messages
    assert "outlier" not in messages
    assert "distance_removal" in messages


def test_clean_pointcloud_empty_dict_uses_defaults():
    """Passing {} for a step should run with full module defaults (same as omitting it)."""
    pcd = _make_pcd(300)
    result_default, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
    )
    result_empty_dict, _ = clean_pointcloud(
        pcd,
        downsample_kwargs=None,
        outlier_kwargs=None,
        distance_kwargs={},
    )
    # Both should produce the same result since {} merges to full defaults
    assert len(result_default.points) == len(result_empty_dict.points)


########################################################
########## compute_obb_from_points #####################
########################################################

def test_compute_obb_axis_aligned_cube():
    """Axis-aligned cube: center at origin, extents all 2.0, rotation near identity."""
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
    ], dtype=np.float32)
    center, extent, rotation = compute_obb_from_points(pts)
    np.testing.assert_allclose(center, [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(sorted(extent), sorted([2.0, 2.0, 2.0]), atol=1e-4)
    # rotation must be orthonormal
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)


def test_compute_obb_all_nan_raises():
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.full((10, 3), np.nan)
    with pytest.raises(ValueError, match="empty or invalid"):
        compute_obb_from_points(pts)


def test_compute_obb_empty_raises():
    from collab_splats.pointcloud.utils import compute_obb_from_points
    pts = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="empty or invalid"):
        compute_obb_from_points(pts)


########################################################
########## get_points_in_mask ##########################
########################################################

def test_get_points_in_mask_basic():
    from collab_splats.pointcloud.utils import get_points_in_mask
    # 4 points from frame 0, 4 from frame 1
    points = np.arange(24, dtype=np.float32).reshape(8, 3)
    pixel_indices = np.array([
        [0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3],
        [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3],
    ], dtype=np.int32)
    # mask: only (row=1,col=1) and (row=3,col=3) are True in frame 0
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[3, 3] = True
    result = get_points_in_mask(0, mask, points, pixel_indices)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, points[[1, 3]])


def test_get_points_in_mask_no_match():
    from collab_splats.pointcloud.utils import get_points_in_mask
    points = np.zeros((4, 3), dtype=np.float32)
    pixel_indices = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=np.int32)
    mask = np.zeros((5, 5), dtype=bool)  # nothing selected
    result = get_points_in_mask(0, mask, points, pixel_indices)
    assert result.shape == (0, 3)


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
