import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from collab_splats.mesh.features import features2vertex, mesh_clustering

######## features2vertex


def _features2vertex_numpy_reference(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """CPU reference: same kernel as the torch path, written with np.add.at."""
    vertices = np.asarray(mesh_vertices)
    M, D = len(vertices), features.shape[1]
    distances, indices = cKDTree(vertices).query(points, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((M, D), dtype=features.dtype)
    distances, indices, feats = distances[valid_mask], indices[valid_mask], features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((M, D), dtype=np.float64)
    wsum = np.zeros((M, 1), dtype=np.float64)
    for j in range(k):
        np.add.at(out, indices[:, j], feats * weights[:, j : j + 1])
        np.add.at(wsum, indices[:, j], weights[:, j : j + 1])
    nz = wsum[:, 0] > 0
    out[nz] /= wsum[nz]
    return out.astype(features.dtype)


def test_features2vertex_output_shape():
    rng = np.random.default_rng(0)
    verts = rng.random((50, 3))
    pts = rng.random((200, 3))
    feats = rng.random((200, 16)).astype(np.float32)
    assert features2vertex(verts, pts, feats, k=5).shape == (50, 16)


def test_features2vertex_matches_numpy_reference():
    rng = np.random.default_rng(0)
    verts = rng.random((200, 3))
    pts = rng.random((1000, 3))
    feats = rng.random((1000, 8)).astype(np.float32)
    out = features2vertex(verts, pts, feats, k=5, sdf_trunc=0.1)
    ref = _features2vertex_numpy_reference(verts, pts, feats, k=5, sdf_trunc=0.1)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_features2vertex_all_far_returns_zeros():
    verts = np.zeros((10, 3))
    pts = np.full((20, 3), 100.0)
    feats = np.ones((20, 4), dtype=np.float32)
    out = features2vertex(verts, pts, feats, k=3)
    assert out.shape == (10, 4) and not out.any()


def test_features2vertex_dtype_preserved():
    rng = np.random.default_rng(1)
    verts = rng.random((50, 3)).astype(np.float32)
    pts = rng.random((100, 3)).astype(np.float32)
    feats = rng.random((100, 6)).astype(np.float32)
    out = features2vertex(verts, pts, feats, k=4, sdf_trunc=0.2)
    assert out.dtype == np.float32 and out.shape == (50, 6)


######## mesh_clustering


def _two_blobs_mesh():
    """60 vertices in two tight blobs 1 unit apart plus 10 scattered low-similarity vertices."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.005, (30, 3))
    b = rng.normal(0.0, 0.005, (30, 3)) + [1.0, 0.0, 0.0]
    stray = rng.random((10, 3)) * [0.5, 1.0, 1.0] + [0.25, 0.0, 0.0]
    verts = np.vstack([a, b, stray])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(np.zeros((0, 3), np.int32))
    )
    similarity = np.r_[np.ones(60), np.zeros(10)]
    return mesh, similarity


def test_mesh_clustering_groups_nearby_high_similarity_vertices():
    mesh, similarity = _two_blobs_mesh()
    clusters = mesh_clustering(mesh, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=10)
    assert len(clusters) == 2
    assert {frozenset(c.tolist()) for c in clusters} == {frozenset(range(30)), frozenset(range(30, 60))}


def test_mesh_clustering_min_cluster_size_drops_small_clusters():
    mesh, similarity = _two_blobs_mesh()
    assert mesh_clustering(mesh, similarity, similarity_threshold=0.5, spatial_radius=0.05, min_cluster_size=31) == []


def test_mesh_clustering_no_valid_vertices_returns_empty_list():
    mesh, similarity = _two_blobs_mesh()
    assert mesh_clustering(mesh, np.zeros_like(similarity)) == []
