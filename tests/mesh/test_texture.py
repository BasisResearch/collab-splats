import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest
from plyfile import PlyData

pytest.importorskip("warp")
pytest.importorskip("meshoptimizer")

from collab_splats.mesh.texture import (  # noqa: E402
    _make_manifold,
    _split_non_manifold_vertices,
    decimate_mesh,
    project_images_to_texture,
    texture_mesh,
    unwrap_mesh_uvs,
)

######## Fixtures


def _dense_plane(n=60, noise=0.0, seed=0):
    """Unit plane in z=0 tessellated n×n, optional gaussian z-noise; normals face +z."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    v = np.stack([xs.ravel(), ys.ravel(), rng.normal(0, noise, n * n)], axis=1)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    f = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))


def _sphere(res=40):
    return o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=res)


def _bowtie():
    """Two triangles sharing only vertex 0 (a non-manifold vertex, no non-manifold edge)."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
    f = np.array([[0, 1, 2], [0, 3, 4]])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))


def _deviation_p99(reference, decimated):
    """99th-percentile distance from reference vertices to the decimated surface."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(decimated))
    d = scene.compute_distance(o3c.Tensor(np.asarray(reference.vertices), dtype=o3c.float32)).numpy()
    return float(np.percentile(d, 99))


def _camera():
    """One camera 2 units up +z looking down at the origin; K f=100 c=64 on a 128² image."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = [0.0, 0.0, 2.0]
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    return c2w[None], K[None]


def _squares_tm(squares):
    """Tensor mesh of xy rectangles; each entry is (z, xy_scale, uv_offset, uv_scale, flip_winding)."""
    verts, faces, uvs = [], [], []
    for z, xy_scale, uv_offset, uv_scale, flip in squares:
        v = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], np.float32)
        v[:, :2] *= np.asarray(xy_scale, np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3]], np.int32) + 4 * len(verts)
        if flip:
            f = f[:, ::-1].copy()
        verts.append(v)
        faces.append(f)
        uvs.append(
            (np.asarray(uv_offset) + np.asarray(uv_scale) * v[f - 4 * (len(verts) - 1)][..., :2]).astype(np.float32)
        )
    tm = o3d.t.geometry.TriangleMesh(o3c.Tensor(np.vstack(verts)), o3c.Tensor(np.vstack(faces)))
    tm.triangle.texture_uvs = o3c.Tensor(np.concatenate(uvs))
    return tm


def _constant_image(rgb=(51, 128, 204)):
    return np.full((1, 128, 128, 3), rgb, dtype=np.uint8)


######## _split_non_manifold_vertices / _make_manifold


def test_split_non_manifold_vertices_bowtie():
    m = _bowtie()
    assert len(m.get_non_manifold_vertices()) == 1
    out, n_split = _split_non_manifold_vertices(m)
    assert n_split == 1
    assert len(out.get_non_manifold_vertices()) == 0
    assert len(out.vertices) == 6 and len(out.triangles) == 2
    v0, v1 = np.asarray(m.vertices), np.asarray(out.vertices)
    f0, f1 = np.asarray(m.triangles), np.asarray(out.triangles)
    assert np.allclose(v0[f0], v1[f1])  # every corner keeps its position
    assert np.allclose(v1[5], v0[0])  # the copy sits exactly on vertex 0
    assert len(m.vertices) == 5  # input not mutated


def test_split_non_manifold_vertices_keeps_colors():
    m = _bowtie()
    m.vertex_colors = o3d.utility.Vector3dVector(np.linspace(0, 1, 15).reshape(5, 3))
    out, _ = _split_non_manifold_vertices(m)
    c = np.asarray(out.vertex_colors)
    assert c.shape == (6, 3) and np.allclose(c[5], c[0])


def test_split_non_manifold_vertices_noop_on_manifold():
    m = _sphere(10)
    out, n_split = _split_non_manifold_vertices(m)
    assert n_split == 0 and len(out.vertices) == len(m.vertices)


def test_make_manifold_drops_opposite_winding_duplicates_and_fold_overs():
    # One face duplicated in reverse winding and one folded back over a neighbor's directed
    # edge: Open3D calls both manifold, UVAtlas calls both non-manifold
    m = _dense_plane(n=4)
    f = np.asarray(m.triangles)
    dup = f[0][[0, 2, 1]]
    fold = np.array([f[3][0], f[3][1], 99])
    v = np.vstack([np.asarray(m.vertices), [[5.0, 5.0, 5.0]] * 84])
    f = np.vstack([f, dup[None], fold[None]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    assert not m.is_orientable()
    n_faces_in = len(m.triangles)
    out = _make_manifold(m)
    assert len(m.triangles) == n_faces_in  # input not mutated
    assert out.is_orientable()
    assert len(out.get_non_manifold_edges()) == 0 and len(out.get_non_manifold_vertices()) == 0
    fo = np.asarray(out.triangles)
    de = np.concatenate([fo[:, [0, 1]], fo[:, [1, 2]], fo[:, [2, 0]]])
    assert np.unique(de, axis=0).shape[0] == len(de)  # every directed edge used once
    assert len(fo) == 18  # dup dropped, fold dropped, f3 (first owner) kept
    o3d.t.geometry.TriangleMesh.from_legacy(out).compute_uvatlas(size=64)


def test_make_manifold_removes_degenerate_before_fold_over_check():
    # Degenerate [0,1,1] carries directed edge (0,1) shared with valid [0,1,2]; only the
    # degenerate face may go
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2], [1, 3, 2], [0, 1, 1]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    fo = np.asarray(_make_manifold(m).triangles)
    assert len(fo) == 2 and [0, 1, 2] in fo.tolist()


def test_make_manifold_after_decimate_yields_uvatlas_ready_mesh():
    m = _make_manifold(_dense_plane(n=80, noise=0.003))
    out, _ = decimate_mesh(m, 0.01)
    out = _make_manifold(out)
    assert len(out.get_non_manifold_edges()) == 0
    assert len(out.get_non_manifold_vertices()) == 0
    tm = o3d.t.geometry.TriangleMesh.from_legacy(out)
    tm.compute_uvatlas(size=256)
    assert tm.triangle.texture_uvs.shape[0] == len(out.triangles)


######## decimate_mesh


def test_decimate_mesh_respects_absolute_bound():
    m = _dense_plane(noise=0.002)
    bound = 0.01
    out, result_error = decimate_mesh(m, bound)
    assert len(out.triangles) < len(m.triangles)
    assert _deviation_p99(m, out) <= 1.5 * bound
    assert result_error <= bound + 1e-6


def test_decimate_mesh_keeps_curvature_relative_to_planes():
    plane, sphere = _dense_plane(n=60), _sphere(res=40)
    p, _ = decimate_mesh(plane, 0.01)
    s, _ = decimate_mesh(sphere, 0.01)
    assert len(p.triangles) < 0.05 * len(plane.triangles)  # a plane collapses to a handful
    assert len(s.triangles) > len(p.triangles)  # a sphere keeps many to stay in bound


def test_decimate_mesh_never_moves_vertices():
    m = _dense_plane(n=20, noise=0.001)
    out, _ = decimate_mesh(m, 0.01)
    v_in = {tuple(np.round(x, 9)) for x in np.asarray(m.vertices)}
    assert all(tuple(np.round(x, 9)) in v_in for x in np.asarray(out.vertices))


######## unwrap_mesh_uvs


def test_unwrap_mesh_uvs_gives_per_corner_uvs_in_unit_square():
    mesh = _sphere(10)
    tm = unwrap_mesh_uvs(mesh, tex_size=64)
    uv = tm.triangle.texture_uvs.numpy()
    assert uv.shape == (len(mesh.triangles), 3, 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0


######## project_images_to_texture


def test_project_images_to_texture_constant_view_gives_constant_albedo():
    tm = _squares_tm([(0.0, (1, 1), (0, 0), (1, 1), False)])
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo.shape == (64, 64, 3) and albedo.dtype == np.uint8
    filled = albedo.any(axis=-1)
    assert filled.mean() >= 0.95
    assert np.abs(albedo[filled].astype(int) - [51, 128, 204]).max() <= 2


def test_project_images_to_texture_occluded_texels_stay_black():
    # Floor square (uv u=x/2 → left half of the atlas) under an occluder at z=0.5 spanning
    # x<0.5 only (uv u=0.5+x → right half). By perspective the occluder hides floor x<0.667.
    tm = _squares_tm([(0.0, (1, 1), (0, 0), (0.5, 1), False), (0.5, (0.5, 1), (0.5, 0), (1, 1), False)])
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo[:, :16].max() == 0  # floor x<0.5: hidden
    assert albedo[:, 23:32].any(axis=-1).mean() > 0.95  # floor x>0.72: seen
    assert albedo[:, 32:].any(axis=-1).mean() > 0.95  # the occluder itself


def test_project_images_to_texture_back_faces_get_nothing():
    tm = _squares_tm([(0.0, (1, 1), (0, 0), (1, 1), True)])  # winding flipped: normal points away
    c2w, K = _camera()
    albedo = project_images_to_texture(tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=0)
    assert albedo.max() == 0


def test_project_images_to_texture_gutter_grows_filled_region():
    tm = _squares_tm([(0.0, (1, 1), (0.25, 0.25), (0.5, 0.5), False)])  # chart in the atlas center
    c2w, K = _camera()
    filled = {}
    for gutter in (0, 4):
        albedo = project_images_to_texture(
            tm, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01, gutter_px=gutter
        )
        filled[gutter] = albedo.any(axis=-1)
    assert filled[4][32].sum() - filled[0][32].sum() == 8
    assert filled[4][filled[0]].all()  # dilation never clears a filled texel


######## texture_mesh


def test_texture_mesh_writes_textured_ply(tmp_path):
    mesh_path = tmp_path / "mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), _dense_plane(30))
    c2w, K = _camera()
    out = texture_mesh(mesh_path, tmp_path / "texture", _constant_image(), c2w, K, voxel_size=0.01, tex_size=64)
    assert out == tmp_path / "texture" / "mesh.ply" and out.exists()
    assert (tmp_path / "texture" / "albedo.png").exists()
    vertex = PlyData.read(str(out))["vertex"]
    assert "s" in vertex.data.dtype.names and "t" in vertex.data.dtype.names
    albedo = cv2.imread(str(tmp_path / "texture" / "albedo.png"))
    assert albedo.shape == (64, 64, 3) and albedo.max() > 0
    assert mesh_path.exists()  # the fused mesh is never modified
