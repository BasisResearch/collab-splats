import numpy as np
import pytest


def test_find_depth_edges_shape():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.random.rand(64, 64).astype(np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=1)
    assert edges.shape == (64, 64)
    assert edges.dtype == bool


def test_find_depth_edges_constant_depth_no_edges():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.ones((32, 32), dtype=np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=0)
    assert not edges.any()


def test_normals2vertex_output_shape():
    from collab_splats.mesh.utils import normals2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    normals = rng.random((200, 3)).astype(np.float32)
    result = normals2vertex(mesh_vertices, points, normals, k=5)
    assert result.shape == (50, 3)


def test_features2vertex_output_shape():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    features = rng.random((200, 16)).astype(np.float32)
    result = features2vertex(mesh_vertices, points, features, k=5)
    assert result.shape == (50, 16)
