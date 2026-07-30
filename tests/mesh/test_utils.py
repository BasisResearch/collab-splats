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


def _features2vertex_numpy_reference(mesh_vertices, points, features, k=5, sdf_trunc=0.03):
    """Frozen copy of the original CPU implementation — parity oracle for the GPU rewrite."""
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh_vertices)
    tree = cKDTree(vertices)
    distances, indices = tree.query(points, k=k)
    valid_mask = distances[:, 0] <= sdf_trunc
    if not np.any(valid_mask):
        return np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    distances = distances[valid_mask]
    indices = indices[valid_mask]
    features = features[valid_mask]
    sigma = np.mean(distances)
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    out = np.zeros((len(vertices), features.shape[1]), dtype=features.dtype)
    wsum = np.zeros((len(vertices), 1), dtype=features.dtype)
    for i in range(k):
        np.add.at(out, indices[:, i], features * weights[:, i : i + 1])
        np.add.at(wsum, indices[:, i], weights[:, i : i + 1])
    nz = wsum.squeeze() > 0
    out[nz] /= wsum[nz]
    return out


def test_features2vertex_matches_numpy_reference():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    vertices = rng.random((200, 3)).astype(np.float64)
    points = rng.random((1000, 3)).astype(np.float64)
    features = rng.random((1000, 8)).astype(np.float32)

    got = features2vertex(vertices, points, features, k=5, sdf_trunc=0.1)
    want = _features2vertex_numpy_reference(vertices, points, features, k=5, sdf_trunc=0.1)

    assert got.shape == (200, 8)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_features2vertex_all_far_returns_zeros():
    from collab_splats.mesh.utils import features2vertex

    vertices = np.zeros((10, 3), dtype=np.float64)
    points = np.full((20, 3), 100.0, dtype=np.float64)  # all far beyond sdf_trunc
    features = np.ones((20, 4), dtype=np.float32)

    out = features2vertex(vertices, points, features, k=3, sdf_trunc=0.03)
    assert out.shape == (10, 4)
    assert np.all(out == 0.0)


def test_features2vertex_dtype_preserved():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(1)
    vertices = rng.random((50, 3))
    points = rng.random((100, 3))
    features = rng.random((100, 6)).astype(np.float32)

    out = features2vertex(vertices, points, features, k=4, sdf_trunc=0.2)
    assert out.dtype == np.float32
    assert out.shape == (50, 6)


def test_meshresult_has_vertex_features_field():
    from pathlib import Path

    from collab_splats.mesh.base import MeshResult

    r = MeshResult(mesh_path=Path("/tmp/m.ply"))
    assert r.vertex_features is None  # default

    r2 = MeshResult(mesh_path=Path("/tmp/m.ply"), vertex_features=np.zeros((3, 2)))
    assert r2.vertex_features.shape == (3, 2)


def test_persist_mesh_vertex_features(tmp_path):
    import open3d as o3d

    from collab_splats.mesh.utils import persist_mesh_vertex_features

    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh_path = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    points = verts.copy()
    feats = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)

    out = persist_mesh_vertex_features(mesh_path, points, feats, k=1, sdf_trunc=0.5)

    assert out.shape == (3, 2)
    saved = np.load(mesh_path.parent / "vertex_features.npy")
    np.testing.assert_allclose(saved, out)


########
# clean_repair_mesh — component filtering + hole filling (meshlib)
########


def _holed_sphere_with_strays(path, radius=1.0, resolution=20):
    """Sphere missing a cap, plus one stray blob inside its bbox and one far outside."""
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    tris = np.asarray(sphere.triangles)
    sphere.triangles = o3d.utility.Vector3iVector(tris[:-12])  # punch a hole
    sphere.remove_unreferenced_vertices()

    inside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    inside.translate((0.2, 0.0, 0.0))
    outside = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=6)
    outside.translate((radius * 9, 0.0, 0.0))

    o3d.io.write_triangle_mesh(str(path), sphere + inside + outside)
    return path


def test_clean_repair_mesh_drops_out_of_bounds_components_and_fills_holes(tmp_path):
    """The two jobs of the cleanup, on a mesh built to need both.

    A TSDF scene comes out with floating specks from stray depth and small holes where coverage
    thinned. Detached geometry *inside* the room (furniture) must survive — that is why the
    bounding-box test exists instead of a plain keep-the-largest.
    """
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    before = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(before.cluster_connected_triangles()[2]) == 3
    assert not before.is_watertight()

    out = clean_repair_mesh(mesh_path, max_hole_size=3.0)

    assert out == mesh_path  # rewritten in place, not to a new name
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    # The far blob is dropped, the one inside the bbox is kept, and the sphere's hole is closed.
    assert len(after.cluster_connected_triangles()[2]) == 2
    assert after.is_watertight()


def test_clean_repair_mesh_leaves_large_holes_alone(tmp_path):
    """A hole bigger than max_hole_size is a real opening (unscanned wall), not a defect."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    clean_repair_mesh(mesh_path, max_hole_size=1e-6)  # below any real perimeter → fill nothing

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert not after.is_watertight()  # the hole survived
    assert len(after.get_non_manifold_edges(allow_boundary_edges=False)) == 14  # same boundary
    assert len(after.cluster_connected_triangles()[2]) == 2  # component filtering still ran


def test_clean_repair_mesh_use_largest_keeps_only_the_main_component(tmp_path):
    """use_largest=True is the aggressive mode: everything but the biggest body is discarded."""
    import open3d as o3d

    from collab_splats.mesh.utils import clean_repair_mesh

    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    clean_repair_mesh(mesh_path, use_largest=True)

    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(after.cluster_connected_triangles()[2]) == 1  # the in-bbox blob went too
    assert after.is_watertight()
