import numpy as np
import open3d as o3d

from collab_splats.mesh.utils import (
    adjacent_face_pairs,
    face_components,
    from_meshlib,
    to_meshlib,
)


def _spheres_and_strays():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=12)
    mesh += o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=6).translate((4.0, 0.0, 0.0))
    mesh += o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1).translate((0.0, 5.0, 0.0))
    return mesh


def test_face_components_matches_open3d_partition_sizes_and_areas():
    mesh = _spheres_and_strays()
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    ids, sizes, areas = face_components(verts, faces)
    ref_ids, ref_sizes, ref_areas = (np.asarray(a) for a in mesh.cluster_connected_triangles())

    # Same partition: each of our labels maps onto exactly one Open3D label
    pairs = np.unique(np.stack([ids, ref_ids], 1), axis=0)
    assert len(pairs) == len(np.unique(ids)) == len(np.unique(ref_ids)) == 3

    np.testing.assert_array_equal(sizes[pairs[:, 0]], ref_sizes[pairs[:, 1]])
    np.testing.assert_allclose(areas[pairs[:, 0]], ref_areas[pairs[:, 1]])


def test_face_components_empty_faces_gives_no_components():
    ids, sizes, areas = face_components(np.zeros((4, 3)), np.zeros((0, 3), dtype=np.int64))

    assert len(ids) == len(sizes) == len(areas) == 0


def test_adjacent_face_pairs_connects_every_face_on_a_non_manifold_edge():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    ids = face_components(verts, faces)[0]

    assert len(adjacent_face_pairs(faces, len(verts))) == 2
    assert len(np.unique(ids)) == 1


def test_meshlib_round_trip_keeps_geometry_and_colors():
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    colors = np.random.default_rng(0).random((len(mesh.vertices), 3))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    back = from_meshlib(to_meshlib(mesh), mesh)

    np.testing.assert_allclose(np.asarray(back.vertices), np.asarray(mesh.vertices))
    np.testing.assert_array_equal(np.asarray(back.triangles), np.asarray(mesh.triangles))
    np.testing.assert_allclose(np.asarray(back.vertex_colors), colors)
