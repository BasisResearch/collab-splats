import numpy as np
import open3d as o3d
import pytest
from collab_splats.mesh.utils import align_geometry_floor


def _make_flat_pcd(floor_z: float = -1.0, n: int = 500, seed: int = 0) -> o3d.geometry.PointCloud:
    """Synthetic flat floor at floor_z with scatter above."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-3, 3, (n, 2))
    z = rng.normal(floor_z, 0.005, n)
    pts = np.column_stack([xy, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_align_geometry_floor_pcd_floor_at_zero():
    """After alignment the dominant floor plane should be at z ≈ 0."""
    pcd = _make_flat_pcd(floor_z=-1.0)
    aligned_pcd, R, t = align_geometry_floor(pcd)
    pts = np.asarray(aligned_pcd.points)
    # Bottom 30% of points (the floor inliers) should be near z=0
    bottom = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    np.testing.assert_allclose(bottom[:, 2].mean(), 0.0, atol=0.1)


def test_align_geometry_floor_returns_r_t_shapes():
    """Return types and shapes are correct."""
    pcd = _make_flat_pcd()
    _, R, t = align_geometry_floor(pcd)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_align_geometry_floor_mesh():
    """Works on TriangleMesh input without error."""
    mesh = o3d.geometry.TriangleMesh.create_box(2, 2, 0.1)
    mesh.translate([-1, -1, -0.5])
    _, R, t = align_geometry_floor(mesh)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
