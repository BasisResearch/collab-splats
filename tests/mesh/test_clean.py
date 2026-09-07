import numpy as np
import open3d as o3d
import pytest

from collab_splats.mesh.clean import (
    clean_repair_mesh,
    fill_holes,
    get_scene_scale,
    remove_floaters,
)


def _holed_sphere_with_strays(path, radius=1.0, resolution=20, color=None):
    """Sphere with one hole (last 12 triangles removed) plus a near and a far stray sphere."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    tris = np.asarray(sphere.triangles)
    sphere.triangles = o3d.utility.Vector3iVector(tris[:-12])  # punch a hole
    sphere.remove_unreferenced_vertices()
    near = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 0.1, resolution=6)
    near.translate((radius * 1.15, 0.0, 0.0))
    far = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 0.1, resolution=6)
    far.translate((radius * 9, 0.0, 0.0))
    combined = sphere + near + far
    if color is not None:
        combined.paint_uniform_color(color)
    o3d.io.write_triangle_mesh(str(path), combined)
    return path


def _n_components(mesh):
    _, sizes, _ = mesh.cluster_connected_triangles()
    return len(sizes)


def _n_boundary_edges(mesh):
    return len(mesh.get_non_manifold_edges(allow_boundary_edges=False))


def test_get_scene_scale_ignores_outliers():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.random((1000, 3)), [[100.0, 100.0, 100.0]]])
    scale = get_scene_scale(pts)
    assert 1.6 < scale < 1.8  # ~sqrt(3) for the unit cube; the outlier would make it ~170


def test_remove_floaters_drops_far_component_keeps_near(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    assert _n_components(mesh) == 3
    out = remove_floaters(mesh, max_gap_frac=0.1)
    assert out is mesh and _n_components(mesh) == 2


def test_remove_floaters_area_floor_drops_small_components(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    remove_floaters(mesh, min_area_frac=0.1, max_gap_frac=0.1)
    assert _n_components(mesh) == 1


def test_remove_floaters_empty_mesh_is_a_no_op():
    mesh = o3d.geometry.TriangleMesh()
    assert remove_floaters(mesh) is mesh


def test_fill_holes_closes_small_hole(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    assert _n_boundary_edges(mesh) == 14
    filled = fill_holes(mesh, max_hole_frac=0.2)
    assert _n_boundary_edges(filled) == 0
    assert filled.is_edge_manifold(allow_boundary_edges=False)
    assert len(filled.triangles) > len(mesh.triangles)
    # Guards the Open3D tensor round-trip: a freed vertex buffer blows the extent up
    orig_extent = mesh.get_axis_aligned_bounding_box().get_extent()
    assert np.allclose(filled.get_axis_aligned_bounding_box().get_extent(), orig_extent, atol=1e-5)


def test_fill_holes_leaves_large_holes_alone(tmp_path):
    mesh = o3d.io.read_triangle_mesh(str(_holed_sphere_with_strays(tmp_path / "m.ply")))
    # The fixture's scene scale is 10.38 (the far stray sets it): the hole fills at
    # max_hole_frac >= 0.017 and survives below it
    filled = fill_holes(mesh, max_hole_frac=0.005)
    assert _n_boundary_edges(filled) == 14


def test_clean_repair_mesh_writes_in_place(tmp_path):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply")
    out = clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    assert out == mesh_path
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert _n_components(after) == 2
    assert _n_boundary_edges(after) == 0


@pytest.mark.parametrize("radius", [1.0, 10.0])
def test_clean_repair_thresholds_follow_mesh_scale(tmp_path, radius):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", radius=radius)
    clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert (_n_components(after), _n_boundary_edges(after)) == (2, 0)


def test_clean_repair_preserves_vertex_colors(tmp_path):
    mesh_path = _holed_sphere_with_strays(tmp_path / "mesh.ply", color=(0.2, 0.6, 0.9))
    clean_repair_mesh(mesh_path, max_gap_frac=0.1, max_hole_frac=0.2)
    after = o3d.io.read_triangle_mesh(str(mesh_path))
    assert after.has_vertex_colors()
    colors = np.asarray(after.vertex_colors)
    assert np.allclose(colors, [0.2, 0.6, 0.9], atol=0.02)
