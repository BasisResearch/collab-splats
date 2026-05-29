from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.utils import transfer_features_to_mesh


def _make_mesh(n_vertices=20):
    """Toy mesh: random vertices, no real geometry needed."""
    rng = np.random.default_rng(42)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(rng.standard_normal((n_vertices, 3)).astype(np.float64))
    return mesh


def _make_result(n_points=30, dim=8, features=None):
    """Minimal duck-type result — transfer_features_to_mesh only reads .points and .features."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n_points, 3)).astype(np.float32)
    feats = features if features is not None else rng.standard_normal((n_points, dim)).astype(np.float32)
    return SimpleNamespace(points=pts, features=feats)


def test_transfer_features_to_mesh_basic():
    """Output shape is (M, D) and contains non-zero values when points are near vertices."""
    mesh = _make_mesh(n_vertices=20)
    # Place pointcloud points close to mesh vertices so KNN finds matches within sdf_trunc
    mesh_verts = np.asarray(mesh.vertices).astype(np.float32)
    # Jitter mesh vertices slightly to make pointcloud near them
    rng = np.random.default_rng(1)
    pts = mesh_verts + rng.standard_normal(mesh_verts.shape).astype(np.float32) * 0.001
    feats = rng.standard_normal((len(pts), 8)).astype(np.float32)
    result = SimpleNamespace(points=pts, features=feats)

    vertex_features = transfer_features_to_mesh(result, mesh, sdf_trunc=0.1)

    assert vertex_features.shape == (20, 8)
    assert vertex_features.dtype == np.float32
    # At least some vertices should have non-zero features
    assert np.any(vertex_features != 0.0)


def test_transfer_features_to_mesh_none_raises():
    """AssertionError when result.features is None."""
    mesh = _make_mesh()
    result = SimpleNamespace(points=np.zeros((10, 3), dtype=np.float32), features=None)

    with pytest.raises(AssertionError, match="result.features is None"):
        transfer_features_to_mesh(result, mesh)


def test_transfer_features_to_mesh_zero_points_in_range():
    """Returns all-zero array when all points are outside sdf_trunc."""
    mesh = _make_mesh(n_vertices=10)
    # Place pointcloud 100 units away from mesh vertices (which are ~unit-scale)
    pts = np.zeros((5, 3), dtype=np.float32) + 100.0
    feats = np.ones((5, 8), dtype=np.float32)
    result = SimpleNamespace(points=pts, features=feats)

    vertex_features = transfer_features_to_mesh(result, mesh, sdf_trunc=0.03)

    assert vertex_features.shape == (10, 8)
    assert np.all(vertex_features == 0.0)
