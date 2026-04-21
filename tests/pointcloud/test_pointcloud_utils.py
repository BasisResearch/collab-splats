import numpy as np
import pytest
from collab_splats.pointcloud.utils import filter_points_by_spatial_extent, voxel_downsample_point_cloud

try:
    import open3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def test_filter_removes_outliers():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    colors = np.zeros((1000, 3), dtype=np.uint8)
    pts[:10] = 1000.0
    out_pts, out_colors = filter_points_by_spatial_extent(pts, colors, percentile_range=(1.0, 99.0))
    assert len(out_pts) < 1000
    assert out_pts.max() < 100.0
    assert out_colors.shape[0] == out_pts.shape[0]


def test_filter_empty():
    pts = np.zeros((0, 3), dtype=np.float32)
    colors = np.zeros((0, 3), dtype=np.uint8)
    out_pts, out_colors = filter_points_by_spatial_extent(pts, colors)
    assert out_pts.shape == (0, 3)


@pytest.mark.skipif(not HAS_OPEN3D, reason="open3d not installed")
def test_voxel_downsample_reduces_points():
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((10000, 3)).astype(np.float32)
    colors = np.zeros((10000, 3), dtype=np.uint8)
    down_pts, down_colors = voxel_downsample_point_cloud(pts, colors, voxel_fraction=0.05)
    assert len(down_pts) < 10000
    assert down_colors.shape[0] == down_pts.shape[0]
