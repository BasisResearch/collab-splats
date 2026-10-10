import meshlib.mrmeshnumpy as mn
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
from scipy.spatial import cKDTree

from collab_splats.mesh.clean import (
    _local_median,
    _mesh_coverage_mask,
    bridge_mesh_edges,
    clean_repair_mesh,
    decimate_mesh,
    fill_holes,
    get_scene_scale,
    make_convex_hull,
    make_manifold,
    prepare_mesh,
    remove_floaters,
    trim_mesh_edges,
)


def _holed_sphere_with_strays(radius=1.0, resolution=20, color=None):
    """Sphere with one hole (last 12 triangles removed) plus a near and a far stray sphere."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
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

    return combined


def _n_components(mesh):
    _, sizes, _ = mesh.cluster_connected_triangles()
    return len(sizes)


def _n_boundary_edges(mesh):
    return len(mesh.get_non_manifold_edges(allow_boundary_edges=False))


def test_get_scene_scale_ignores_outliers():
    rng = np.random.default_rng(0)
    pts = np.vstack([rng.random((1000, 3)), [[100.0, 100.0, 100.0]]])
    scale = get_scene_scale(pts)
    assert (
        1.6 < scale < 1.8
    )  # ~sqrt(3) for the unit cube; the outlier would make it ~170


def test_remove_floaters_drops_far_component_keeps_near():
    mesh = _holed_sphere_with_strays()
    assert _n_components(mesh) == 3
    out = remove_floaters(mesh, max_gap_frac=0.1)
    assert out is mesh and _n_components(mesh) == 2


def test_remove_floaters_area_floor_drops_small_components():
    mesh = _holed_sphere_with_strays()
    remove_floaters(mesh, min_area_frac=0.1, max_gap_frac=0.1)
    assert _n_components(mesh) == 1


def test_remove_floaters_empty_mesh_is_a_no_op():
    mesh = o3d.geometry.TriangleMesh()
    assert remove_floaters(mesh) is mesh


def test_fill_holes_closes_small_hole():
    mesh = _holed_sphere_with_strays()
    assert _n_boundary_edges(mesh) == 14
    filled = fill_holes(mesh, max_hole_perimeter_ratio=0.2)
    assert _n_boundary_edges(filled) == 0
    assert filled.is_edge_manifold(allow_boundary_edges=False)
    assert len(filled.triangles) > len(mesh.triangles)
    # Guards the meshlib round-trip: source geometry comes back where it was
    orig_extent = mesh.get_axis_aligned_bounding_box().get_extent()
    assert np.allclose(
        filled.get_axis_aligned_bounding_box().get_extent(), orig_extent, atol=1e-5
    )


def test_fill_holes_without_subdivide_fill_adds_a_flat_lid():
    mesh = _holed_sphere_with_strays()
    flat = fill_holes(mesh, max_hole_perimeter_ratio=0.2, subdivide_fill=False)
    fine = fill_holes(mesh, max_hole_perimeter_ratio=0.2, subdivide_fill=True)
    assert _n_boundary_edges(flat) == 0
    assert len(flat.vertices) == len(
        mesh.vertices
    )  # the rim is triangulated, never split
    assert (
        len(flat.triangles) == len(mesh.triangles) + 12
    )  # a 14-edge loop takes 14 - 2 triangles
    assert len(fine.triangles) > len(flat.triangles)


def test_fill_holes_gives_small_loops_a_plain_lid_even_when_subdividing():
    mesh = _holed_sphere_with_strays()
    plain = fill_holes(
        mesh, max_hole_perimeter_ratio=0.2, subdivide_fill=True, max_plain_edges=14
    )
    nicely = fill_holes(
        mesh, max_hole_perimeter_ratio=0.2, subdivide_fill=True, max_plain_edges=13
    )

    assert _n_boundary_edges(plain) == 0
    assert len(plain.vertices) == len(
        mesh.vertices
    )  # the 14-edge loop is at the bound: no subdivision
    assert len(plain.triangles) == len(mesh.triangles) + 12
    assert len(nicely.vertices) > len(
        mesh.vertices
    )  # one edge over the bound: subdivided as before


def test_fill_holes_keeps_outer_rim_open_at_any_bound():
    plane = _dense_plane(30)
    centers = np.asarray(plane.vertices)[np.asarray(plane.triangles)].mean(axis=1)
    plane.remove_triangles_by_mask(np.abs(centers[:, :2] - 0.5).max(axis=1) < 0.1)
    rim_edges = 4 * 29
    assert _n_boundary_edges(plane) > rim_edges
    filled = fill_holes(plane, max_hole_perimeter_ratio=100.0)
    assert (
        _n_boundary_edges(filled) == rim_edges
    )  # the interior hole closes, the rim never does


def test_fill_holes_keeps_outer_rim_open_when_an_interior_loop_is_longer():
    plane = _dense_plane(60)
    cells = np.floor(
        np.asarray(plane.vertices)[np.asarray(plane.triangles)].mean(axis=1)[:, :2] * 59
    ).astype(int)

    # Comb-shaped hole in the middle: a spine row plus every other column, longer than the rim but narrow
    col, row = cells[:, 0], cells[:, 1]
    in_box = (col >= 18) & (col < 42) & (row >= 18) & (row < 42)
    plane.remove_triangles_by_mask(in_box & ((row == 18) | (col % 2 == 0)))
    rim_edges = 4 * 59
    assert _n_boundary_edges(plane) - rim_edges > rim_edges

    filled = fill_holes(plane, max_hole_perimeter_ratio=100.0)
    assert (
        _n_boundary_edges(filled) == rim_edges
    )  # the widest loop is the rim, not the longest


def test_fill_holes_leaves_large_holes_alone():
    mesh = _holed_sphere_with_strays()
    # Hole perimeter 0.75 over scene scale 10.38: fills above ratio 0.072, survives below
    filled = fill_holes(mesh, max_hole_perimeter_ratio=0.005)
    assert _n_boundary_edges(filled) == 14


def test_clean_repair_mesh_drops_strays_and_fills_holes():
    cleaned, _ = clean_repair_mesh(
        _holed_sphere_with_strays(), max_gap_frac=0.1, max_hole_perimeter_ratio=0.5
    )
    assert _n_components(cleaned) == 2
    assert _n_boundary_edges(cleaned) == 0


def test_clean_repair_mesh_returns_the_floater_cut_surface_unfilled():
    mesh = _holed_sphere_with_strays()
    _, real = clean_repair_mesh(mesh, max_gap_frac=0.1, max_hole_perimeter_ratio=0.5)
    assert real is mesh  # the input, cut in place; the fill never touched it
    assert _n_components(real) == 2
    assert _n_boundary_edges(real) > 0


@pytest.mark.parametrize("radius", [1.0, 10.0])
def test_clean_repair_thresholds_follow_mesh_scale(radius):
    cleaned, _ = clean_repair_mesh(
        _holed_sphere_with_strays(radius=radius),
        max_gap_frac=0.1,
        max_hole_perimeter_ratio=0.5,
    )
    assert (_n_components(cleaned), _n_boundary_edges(cleaned)) == (2, 0)


def test_clean_repair_preserves_vertex_colors():
    mesh = _holed_sphere_with_strays(color=(0.2, 0.6, 0.9))
    cleaned, _ = clean_repair_mesh(mesh, max_gap_frac=0.1, max_hole_perimeter_ratio=0.5)
    assert cleaned.has_vertex_colors()
    colors = np.asarray(cleaned.vertex_colors)
    assert np.allclose(colors, [0.2, 0.6, 0.9], atol=0.02)


######## make_convex_hull fixtures


def _ground_scene(n=140, res=0.17, upside_down=False):
    """Height-field ground at TSDF-like edge length: a raised box, a star-shaped outline, a slit; normals up."""
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = (cols - n / 2) * res
    y = (rows - n / 2) * res
    z = np.where((np.abs(x) < 1.5) & (np.abs(y) < 1.0), 1.0, 0.0)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Two counter-clockwise triangles per grid quad, so normals point +z
    idx = (rows * n + cols)[:-1, :-1].reshape(-1)
    quads = np.stack([idx, idx + 1, idx + n + 1, idx + n], axis=1)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])

    # Star-shaped outline plus a two-cell slit cut in from the rim
    center = verts[faces].mean(axis=1)
    radius = np.linalg.norm(center[:, :2], axis=1)
    angle = np.arctan2(center[:, 1], center[:, 0])
    reach = 0.4 * n * res * (0.8 + 0.2 * np.sin(5 * angle))
    slit = (center[:, 0] > 4.0) & (np.abs(center[:, 1] - 3.0) < res)
    faces = faces[(radius < reach) & ~slit]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(0.5 + 0.1 * verts, 0, 1))
    mesh.remove_unreferenced_vertices()

    if upside_down:
        mesh.rotate(np.diag([1.0, -1.0, -1.0]), center=(0, 0, 0))

    return mesh


def _loop_perimeters(mesh):
    mmesh = mn.meshFromFacesVerts(
        np.ascontiguousarray(np.asarray(mesh.triangles), dtype=np.int32),
        np.ascontiguousarray(np.asarray(mesh.vertices), dtype=np.float64),
    )
    return sorted(
        mmesh.holePerimeter(hole) for hole in mmesh.topology.findHoleRepresentiveEdges()
    )


######## make_convex_hull


def test_make_convex_hull_then_fill_holes_leaves_one_manifold_piece_with_only_the_outer_rim():
    filled = fill_holes(make_convex_hull(_ground_scene()), max_hole_perimeter_ratio=3.9)
    assert len(_loop_perimeters(filled)) == 1
    assert len(filled.get_non_manifold_edges(allow_boundary_edges=True)) == 0
    assert _n_components(filled) == 1


def test_make_convex_hull_patch_lies_on_the_ground_and_is_kept():
    scene = _ground_scene()
    hull = make_convex_hull(scene)
    dist, _ = cKDTree(np.asarray(scene.vertices)).query(np.asarray(hull.vertices))
    patch = np.asarray(hull.vertices)[dist > 1e-6]
    assert len(patch) > 500
    assert np.abs(patch[:, 2]).max() < 0.01


def test_bridge_mesh_edges_turns_a_narrow_inlet_into_an_interior_hole():
    """A one-cell slit from the rim to the middle of a plane: bridged at its mouth, it is a hole."""
    plane = _dense_plane(n=60)
    faces = np.asarray(plane.triangles)
    center = np.asarray(plane.vertices)[faces].mean(axis=1)
    step = 1.0 / 59
    slit = (np.abs(center[:, 1] - 0.5) < 0.5 * step) & (center[:, 0] > 0.5)
    plane.triangles = o3d.utility.Vector3iVector(faces[~slit])
    assert len(_loop_perimeters(plane)) == 1
    bridged = bridge_mesh_edges(plane, radius=3.0)
    assert len(_loop_perimeters(bridged)) == 2
    assert _loop_perimeters(bridged)[-1] < _loop_perimeters(plane)[-1] - 0.5


def test_trim_mesh_edges_cuts_a_thin_spur_off_the_rim_and_keeps_the_vertices():
    """A three-cell-wide spur out of a disc's rim falls outside the opened outline and touches the rim."""
    plane = _dense_plane(n=100)
    faces = np.asarray(plane.triangles)
    center = np.asarray(plane.vertices)[faces].mean(axis=1)
    step = 1.0 / 99
    disc = np.linalg.norm(center[:, :2] - 0.5, axis=1) < 0.3
    spur = (np.abs(center[:, 1] - 0.5) < 1.5 * step) & (center[:, 0] > 0.5)
    plane.triangles = o3d.utility.Vector3iVector(faces[disc | spur])

    trimmed = trim_mesh_edges(plane)
    kept = np.asarray(trimmed.vertices)[np.asarray(trimmed.triangles)].mean(axis=1)
    assert len(trimmed.vertices) == len(plane.vertices)
    assert np.linalg.norm(kept[:, :2] - 0.5, axis=1).max() < 0.3 + 2 * step
    assert len(kept) > 0.95 * disc.sum()


def test_make_convex_hull_shortens_the_outline():
    scene = _ground_scene()
    assert (
        _loop_perimeters(make_convex_hull(scene))[-1]
        < 0.8 * _loop_perimeters(scene)[-1]
    )


def test_make_convex_hull_caps_patch_edges_that_touch_the_rim():
    # Jitter the outer ground band, so rim heights jump but stay inside the rim height gate
    scene = _ground_scene()
    verts = np.asarray(scene.vertices).copy()
    band = (verts[:, 2] < 1e-6) & (
        np.linalg.norm(verts[:, :2], axis=1) > 0.2 * 140 * 0.17
    )
    verts[band, 2] += np.random.default_rng(1).uniform(-3.5, 3.5, band.sum()) * 0.17
    scene.vertices = o3d.utility.Vector3dVector(verts)
    faces = np.asarray(scene.triangles)
    res = np.median(
        np.linalg.norm(verts[faces] - verts[np.roll(faces, 1, axis=1)], axis=2)
    )

    # Bridges off: they span two rim edges by design, the cap is on patch faces
    hull = make_convex_hull(scene, rim_max_edge=3.0, bridge_radius=0.0)

    # Patch faces with both a source vertex and a patch vertex: none reaches past the cap
    hull_verts = np.asarray(hull.vertices)
    dist, _ = cKDTree(verts).query(hull_verts)
    is_source = dist < 1e-6
    hull_faces = np.asarray(hull.triangles)
    mixed = hull_faces[
        is_source[hull_faces].any(axis=1) & ~is_source[hull_faces].all(axis=1)
    ]
    assert len(mixed) > 0
    longest = np.linalg.norm(
        hull_verts[mixed] - hull_verts[np.roll(mixed, 1, axis=1)], axis=2
    ).max(axis=1)
    assert longest.max() < 3.0 * res


def test_make_convex_hull_rejects_a_mesh_without_ground():
    box = o3d.geometry.TriangleMesh.create_box(5.0, 5.0, 3.0)
    box.compute_triangle_normals()
    walls = np.abs(np.asarray(box.triangle_normals)[:, 2]) < 0.5
    box.triangles = o3d.utility.Vector3iVector(np.asarray(box.triangles)[walls])

    with pytest.raises(ValueError, match="no dominant ground"):
        make_convex_hull(box)


def test_make_convex_hull_is_the_same_either_way_up():
    upright = make_convex_hull(_ground_scene())
    flipped = make_convex_hull(_ground_scene(upside_down=True))
    flipped.rotate(np.diag([1.0, -1.0, -1.0]), center=(0, 0, 0))
    assert abs(len(flipped.triangles) - len(upright.triangles)) < 0.05 * len(
        upright.triangles
    )
    assert (
        abs(flipped.get_surface_area() - upright.get_surface_area())
        < 0.02 * upright.get_surface_area()
    )
    dist, _ = cKDTree(np.asarray(upright.vertices)).query(np.asarray(flipped.vertices))
    assert np.percentile(dist, 99) < 0.17


def test_make_convex_hull_leaves_the_input_untouched():
    scene = _ground_scene()
    verts, faces = np.asarray(scene.vertices).copy(), np.asarray(scene.triangles).copy()
    make_convex_hull(scene)
    np.testing.assert_array_equal(np.asarray(scene.vertices), verts)
    np.testing.assert_array_equal(np.asarray(scene.triangles), faces)


def test_clean_repair_mesh_runs_the_convex_hull_only_when_asked():
    default, _ = clean_repair_mesh(_ground_scene())
    off, _ = clean_repair_mesh(_ground_scene(), use_convex_hull=False)
    on, _ = clean_repair_mesh(_ground_scene(), use_convex_hull=True)
    np.testing.assert_array_equal(
        np.asarray(off.triangles), np.asarray(default.triangles)
    )
    assert _loop_perimeters(on)[-1] < _loop_perimeters(default)[-1]


######## Prepare for UV unwrap fixtures


def _dense_plane(n=60, noise=0.0, seed=0):
    """Unit plane in z=0 tessellated n×n, optional gaussian z-noise; normals face +z."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([xs.ravel(), ys.ravel(), rng.normal(0, noise, n * n)], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = (
        i[:-1, :-1].ravel(),
        i[:-1, 1:].ravel(),
        i[1:, :-1].ravel(),
        i[1:, 1:].ravel(),
    )
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )


def _sphere(res=40):
    return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=res)


def _bowtie():
    """Two triangles sharing only vertex 0 (a non-manifold vertex, no non-manifold edge)."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    f = np.array([[0, 1, 2], [0, 3, 4]])
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )


def _deviation_p99(reference, decimated):
    """99th-percentile distance from reference vertices to the decimated surface."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(decimated))
    d = scene.compute_distance(
        o3c.Tensor(np.asarray(reference.vertices), dtype=o3c.float32)
    ).numpy()
    return float(np.percentile(d, 99))


######## make_manifold


def test_make_manifold_splits_bowtie_vertex():
    m = _bowtie()
    assert len(m.get_non_manifold_vertices()) == 1
    out = make_manifold(m)
    assert len(out.get_non_manifold_vertices()) == 0
    assert len(out.vertices) == 6 and len(out.triangles) == 2
    v0, v1 = np.asarray(m.vertices), np.asarray(out.vertices)
    f0, f1 = np.asarray(m.triangles), np.asarray(out.triangles)
    assert np.allclose(v0[f0], v1[f1])  # every corner keeps its position
    assert np.allclose(v1[5], v0[0])  # the copy sits exactly on vertex 0
    assert len(m.vertices) == 5  # input not mutated


def test_make_manifold_bowtie_copy_keeps_colors():
    m = _bowtie()
    m.vertex_colors = o3d.utility.Vector3dVector(np.linspace(0, 1, 15).reshape(5, 3))
    out = make_manifold(m)
    c = np.asarray(out.vertex_colors)
    assert c.shape == (6, 3) and np.allclose(c[5], c[0])


def test_make_manifold_noop_on_manifold():
    m = _sphere(10)
    out = make_manifold(m)
    assert len(out.vertices) == len(m.vertices) and len(out.triangles) == len(
        m.triangles
    )


def test_make_manifold_drops_opposite_winding_duplicates_and_fold_overs():
    # A reversed duplicate face and a fold-over: manifold to Open3D, still repaired
    m = _dense_plane(n=4)
    f = np.asarray(m.triangles)
    dup = f[0][[0, 2, 1]]
    fold = np.array([f[3][0], f[3][1], 99])
    v = np.vstack([np.asarray(m.vertices), [[5.0, 5.0, 5.0]] * 84])
    f = np.vstack([f, dup[None], fold[None]])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )
    assert not m.is_orientable()
    n_faces_in = len(m.triangles)
    out = make_manifold(m)
    assert len(m.triangles) == n_faces_in  # input not mutated
    assert out.is_orientable()
    assert (
        len(out.get_non_manifold_edges()) == 0
        and len(out.get_non_manifold_vertices()) == 0
    )
    fo = np.asarray(out.triangles)
    de = np.concatenate([fo[:, [0, 1]], fo[:, [1, 2]], fo[:, [2, 0]]])
    assert np.unique(de, axis=0).shape[0] == len(de)  # every directed edge used once
    assert len(fo) == 18  # dup dropped, fold dropped, f3 (first owner) kept


def test_make_manifold_removes_degenerate_before_fold_over_check():
    # Degenerate [0,1,1] shares directed edge (0,1) with valid [0,1,2]; only the degenerate goes
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1]])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )
    fo = np.asarray(make_manifold(m).triangles)
    assert len(fo) == 2 and [0, 1, 2] in fo.tolist()


def test_make_manifold_drops_zero_area_face_with_distinct_ids():
    # Vertex 4 sits on vertex 1: [1, 4, 3] has three distinct ids but no area
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )
    out = make_manifold(m)

    assert np.asarray(out.triangles).tolist() == [[0, 1, 2], [1, 3, 2]]
    assert len(out.vertices) == 4


def test_make_manifold_splits_every_extra_fan_in_vertex_then_face_order():
    # Vertex 0: two fans of two faces; vertex 5: a two-face fan plus a lone face; [9, 2, 1] has zero area
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [-1, -1, 0],
            [0, -1, 0],
            [-2, -1, 0],
            [-1, -2, 0],
            [1, 0, 0],
        ],
        float,
    )
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 6], [5, 7, 8], [9, 2, 1]])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )
    m.vertex_colors = o3d.utility.Vector3dVector(np.linspace(0, 1, 30).reshape(10, 3))
    out = make_manifold(m)

    # Expected faces are the pre-vectorization output: copy 9 is vertex 0, copy 10 is vertex 5
    fo = np.asarray(out.triangles)
    assert fo.tolist() == [[0, 1, 2], [0, 2, 3], [9, 4, 5], [9, 5, 6], [10, 7, 8]]
    assert len(out.get_non_manifold_vertices()) == 0

    vo, co = np.asarray(out.vertices), np.asarray(out.vertex_colors)
    np.testing.assert_array_equal(vo[[9, 10]], v[[0, 5]])
    np.testing.assert_array_equal(co[[9, 10]], np.asarray(m.vertex_colors)[[0, 5]])


def test_local_median_matches_per_point_median():
    rng = np.random.default_rng(0)
    xy = rng.random((300, 2))
    values = rng.random(300)
    tree = cKDTree(xy)
    expected = np.array(
        [np.median(values[ix]) for ix in tree.query_ball_point(xy, 0.1)]
    )
    counts = tree.query_ball_point(xy, 0.1, return_length=True)

    # Both parities present, so both middle-element branches are checked
    assert (counts % 2 == 0).any() and (counts % 2 == 1).any()
    np.testing.assert_array_equal(_local_median(xy, values, 0.1), expected)
    assert len(_local_median(np.zeros((0, 2)), np.zeros(0), 0.1)) == 0


def test_mesh_coverage_mask_fills_each_face():
    xy = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 3.0], [4.0, 3.0], [3.0, 4.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    mask = _mesh_coverage_mask(xy, faces, np.zeros(2), 0.1, (50, 50))

    assert mask[2, 2] == 255 and mask[32, 32] == 255
    assert mask[20, 20] == 0 and mask[9, 9] == 0
    assert set(np.unique(mask)) == {0, 255}


def test_make_manifold_after_decimate_yields_manifold_mesh():
    m = make_manifold(_dense_plane(n=80, noise=0.003))
    out, _ = decimate_mesh(m, max_error=0.01)
    out = make_manifold(out)
    assert len(out.get_non_manifold_edges()) == 0
    assert len(out.get_non_manifold_vertices()) == 0
    assert out.is_orientable()


######## decimate_mesh


def test_decimate_mesh_respects_absolute_bound():
    m = _dense_plane(noise=0.002)
    bound = 0.01
    out, result_error = decimate_mesh(m, max_error=bound)
    assert len(out.triangles) < len(m.triangles)
    assert _deviation_p99(m, out) <= 1.5 * bound
    assert result_error <= bound + 1e-6


def test_decimate_mesh_keeps_curvature_relative_to_planes():
    plane, sphere = _dense_plane(n=60), _sphere(res=40)
    p, _ = decimate_mesh(plane, max_error=0.01)
    s, _ = decimate_mesh(sphere, max_error=0.01)
    assert len(p.triangles) < 0.05 * len(
        plane.triangles
    )  # a plane collapses to a handful
    assert len(s.triangles) > len(p.triangles)  # a sphere keeps many to stay in bound


def test_decimate_mesh_never_moves_vertices():
    m = _dense_plane(n=20, noise=0.001)
    out, _ = decimate_mesh(m, max_error=0.01)
    # meshlib holds float32 coordinates; a kept vertex lands within float32 rounding of its source
    dist, _ = cKDTree(np.asarray(m.vertices)).query(np.asarray(out.vertices))
    assert dist.max() < 1e-6


def test_decimate_mesh_max_faces_caps_past_the_bound_without_moving_vertices():
    m = _sphere(res=40)
    free, _ = decimate_mesh(m, max_error=1e-5)
    capped, _ = decimate_mesh(m, max_error=1e-5, max_faces=500)
    assert len(free.triangles) > 500 >= len(capped.triangles) > 400
    dist, _ = cKDTree(np.asarray(m.vertices)).query(np.asarray(capped.vertices))
    assert dist.max() < 1e-6


def test_prepare_mesh_smoothing_flattens_noise_and_keeps_faces():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    vertices = np.asarray(sphere.vertices)
    noise = np.random.default_rng(0).normal(0.0, 0.01, len(vertices))
    sphere.vertices = o3d.utility.Vector3dVector(vertices * (1.0 + noise)[:, None])

    # A tiny voxel keeps decimation from collapsing the noise away first
    rough = prepare_mesh(sphere, voxel_size=1e-5)
    smooth = prepare_mesh(sphere, voxel_size=1e-5, smooth_iterations=30)

    radial_rough = np.linalg.norm(np.asarray(rough.vertices), axis=1)
    radial_smooth = np.linalg.norm(np.asarray(smooth.vertices), axis=1)
    assert radial_smooth.std() < 0.5 * radial_rough.std()
    assert abs(radial_smooth.mean() - radial_rough.mean()) < 0.01
    assert len(smooth.triangles) == len(rough.triangles)


def test_prepare_mesh_max_faces_caps_the_face_count():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    free = prepare_mesh(sphere, voxel_size=1e-5)
    capped = prepare_mesh(sphere, voxel_size=1e-5, max_faces=500)
    assert len(free.triangles) > 500 >= len(capped.triangles)
