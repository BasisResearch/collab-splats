from __future__ import annotations

import numpy as np
import open3d as o3d
import pyvista as pv

from collab_splats.utils.visualization import o3d_mesh_to_polydata


def _make_o3d_mesh(with_colors: bool = True) -> o3d.geometry.TriangleMesh:
    """Minimal 2-triangle mesh (4 vertices, 2 faces)."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if with_colors:
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def test_returns_polydata():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh())
    assert isinstance(pd, pv.PolyData)


def test_vertex_count():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh())
    assert pd.n_points == 4


def test_face_count():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh())
    assert pd.n_cells == 2


def test_rgb_scalar_present_when_colors():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh(with_colors=True))
    assert "RGB" in pd.point_data


def test_rgb_dtype_uint8():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh(with_colors=True))
    assert pd.point_data["RGB"].dtype == np.uint8


def test_no_rgb_when_no_colors():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh(with_colors=False))
    assert "RGB" not in pd.point_data


def test_vertex_positions_preserved():
    pd = o3d_mesh_to_polydata(_make_o3d_mesh())
    expected = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_allclose(pd.points, expected, atol=1e-6)
