import numpy as np
import open3d as o3d
import pytest
import trimesh

pytest.importorskip("meshoptimizer")

from collab_splats.mesh import texture  # noqa: E402
from collab_splats.mesh.clean import prepare_mesh  # noqa: E402
from collab_splats.mesh.texture import (  # noqa: E402
    _apply_view_gains,
    _solve_view_gains,
    create_texture_mesh,
    project_images_to_texture,
    unwrap_view_charts,
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


def _camera():
    """One camera 2 units up +z looking down at the origin; K f=100 c=64 on a 128² image."""
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])
    c2w[:3, 3] = [0.0, 0.0, 2.0]
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    return c2w[None], K[None]


def _squares(squares):
    """
    (verts, faces, normals, uvs) of xy rectangles; each entry is (z, xy_scale, uv_offset, uv_scale, flip_winding).
    """
    verts, faces, normals, uvs = [], [], [], []

    for z, xy_scale, uv_offset, uv_scale, flip in squares:
        v = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], np.float32)
        v[:, :2] *= np.asarray(xy_scale, np.float32)
        f = np.array([[0, 1, 2], [0, 2, 3]], np.int32) + 4 * len(verts)

        if flip:
            f = f[:, ::-1].copy()

        verts.append(v)
        faces.append(f)
        normals.append(np.tile([0, 0, -1.0 if flip else 1.0], (4, 1)))
        uvs.append(
            (np.asarray(uv_offset) + np.asarray(uv_scale) * v[f - 4 * (len(verts) - 1)][..., :2]).astype(np.float32)
        )

    return np.vstack(verts), np.vstack(faces), np.vstack(normals), np.concatenate(uvs)


def _constant_image(rgb=(51, 128, 204)):
    return np.full((1, 128, 128, 3), rgb, dtype=np.uint8)


######## color gains


def _gain_views(gains, n=40):
    """
    Views of a smooth albedo plane from 2 units up, each times its gain; plane spans the footprints.

    - albedo varies per channel in [0.25, 0.65], so no gain pushes a sample past the clip gate
    """
    plane = _dense_plane(n)
    verts = np.asarray(plane.vertices) * [4.0, 4.0, 0.0] - [1.5, 1.5, 0.0]
    c2w, K = _camera()
    poses, images = [], []
    v, u = np.mgrid[:128, :128] + 0.5

    for k, g in enumerate(gains):
        # Ray from the camera through each pixel center hits z=0 at (X, Y)
        center = np.array([0.3 + 0.1 * k, 0.4 + 0.05 * k])
        x = center[0] + 2 * (u - 64) / 100
        y = center[1] - 2 * (v - 64) / 100
        albedo = np.stack([0.45 + 0.2 * np.sin(1.5 * x), 0.45 + 0.2 * np.cos(1.2 * y), 0.4 + 0.15 * np.sin(x + y)], -1)
        images.append(np.round(255 * albedo * g).astype(np.uint8))
        pose = c2w[0].copy()
        pose[:2, 3] = center
        poses.append(pose)

    return verts, np.asarray(plane.triangles), np.stack(images), np.stack(poses), np.repeat(K, len(gains), 0)


def test_solve_view_gains_recovers_injected_gains():
    truth = np.array([[1.0, 1.0, 1.0], [1.2, 0.9, 1.1], [0.85, 1.1, 0.95], [1.1, 1.15, 0.8], [0.95, 0.85, 1.2]])
    verts, faces, rgbs, c2w, K = _gain_views(truth)
    gains = _solve_view_gains(verts, faces, rgbs, c2w, K, n_points=20_000, smooth=0.0)
    error = np.log(gains) - np.log(truth)
    assert np.abs(error - np.median(error, 0)).max() < 1e-2  # gains are only defined up to one shared scale


def test_solve_view_gains_single_view_is_unity():
    verts, faces, rgbs, c2w, K = _gain_views(np.ones((1, 3)))
    gains = _solve_view_gains(verts, faces, rgbs, c2w, K, n_points=1_000)
    assert np.array_equal(gains, np.ones((1, 3)))


def test_apply_view_gains_divides_below_knee_and_rolls_off_above():
    rgbs = np.array([[[[100, 100, 100], [250, 250, 250]]]], dtype=np.uint8)
    out = _apply_view_gains(rgbs, np.array([[0.8, 0.8, 0.5]]))
    assert out[0, 0, 0].tolist() == [125, 125, 200]  # exact division up to the knee
    assert 200 < out[0, 0, 1].min() and out[0, 0, 1].max() <= 255  # 312 and 500 roll off, never wrap
    assert rgbs[0, 0, 0].tolist() == [100, 100, 100]  # the input is not modified


######## unwrap_view_charts


def _two_planes(n=10):
    """
    Unit plane at z=0 facing the camera, over a copy at z=-0.5 the top one hides completely.
    """
    top = _dense_plane(n)
    bottom = _dense_plane(n).translate((0, 0, -0.5))
    bottom.triangles = o3d.utility.Vector3iVector(np.asarray(bottom.triangles)[:, ::-1])
    return top + bottom


def test_unwrap_view_charts_uvs_in_unit_square_at_one_texel_density():
    mesh = _two_planes()
    c2w, K = _camera()
    uv, boxes = unwrap_view_charts(mesh, c2w, K, (128, 128), tex_size=256)
    assert uv.shape == (len(mesh.triangles), 3, 2)
    assert uv.min() >= 0.0 and uv.max() <= 1.0
    assert len(boxes) >= 2  # at least the seen and the hidden plane

    # Seen faces (camera chart) and hidden faces (flat patch) share one texel density
    fv = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    world = 0.5 * np.linalg.norm(np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0]), axis=1)
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    atlas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1])
    ratio = atlas / world
    assert np.abs(ratio / np.median(ratio) - 1).max() < 0.02


def test_unwrap_view_charts_faces_never_share_texels():
    mesh = _two_planes()
    c2w, K = _camera()
    uv, _ = unwrap_view_charts(mesh, c2w, K, (128, 128), tex_size=256)

    # Each face's texel-center count matches its UV area, so no face is drawn over another
    ids = texture._rasterize_uvs(uv, 256)[0, ..., 3].long() - 1
    owned = np.bincount(ids[ids >= 0].cpu().numpy(), minlength=len(uv))
    assert owned.sum() == int((ids >= 0).sum())
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    expect = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]) * 256**2
    assert abs(owned.sum() / expect.sum() - 1) < 0.05


######## project_images_to_texture


def test_project_images_to_texture_constant_view_gives_constant_albedo():
    mesh = _squares([(0.0, (1, 1), (0, 0), (1, 1), False)])
    c2w, K = _camera()
    albedo = project_images_to_texture(*mesh, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01)
    assert albedo.shape == (64, 64, 3) and albedo.dtype == np.uint8
    assert np.abs(albedo.astype(int) - [51, 128, 204]).max() <= 2  # the fill reaches every texel


def _two_color_image():
    """Red left of column 97 (where the occluder lands), blue right of it (the visible floor)."""
    image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
    image[..., :97, 0] = 255
    image[..., 97:, 2] = 255
    return image


def _floor_and_occluder():
    """
    Floor square (uv u=x/2, left atlas half) under an occluder at z=0.5 over x<0.5 (right half).

    - by perspective the occluder hides floor x<0.667, atlas columns below ~21
    - occluder overhangs the floor to y=-0.05: an edge at y=0 would sit on pixel row 64's center, a lookup tie
    """
    verts, faces, normals, uvs = _squares(
        [(0.0, (1, 1), (0, 0), (0.5, 1), False), (0.5, (0.5, 1), (0.5, 0), (1, 1), False)]
    )
    verts[4:, 1] = verts[4:, 1] * 1.05 - 0.05
    return verts, faces, normals, uvs


def test_project_images_to_texture_occluded_texels_take_fill_not_occluder_color():
    c2w, K = _camera()
    albedo = project_images_to_texture(
        *_floor_and_occluder(), _two_color_image(), c2w, K, tex_size=64, occlusion_eps=0.01
    ).astype(int)
    assert (albedo[:, 32:] == [255, 0, 0]).all(axis=-1).mean() > 0.9  # the occluder sees red
    assert (albedo[:, 23:32] == [0, 0, 255]).all(axis=-1).mean() > 0.9  # floor x>0.72 sees blue
    assert albedo[:, :16, 2].min() > 0  # hidden floor: filled, never the red it would sample


def test_project_images_to_texture_separate_occluder_replaces_mesh():
    floor = (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]))
    c2w, K = _camera()
    albedo = project_images_to_texture(
        *_floor_and_occluder(), _two_color_image(), c2w, K, tex_size=64, occlusion_eps=0.01, occluder=floor
    ).astype(int)
    assert (albedo[:, :16] == [255, 0, 0]).all(axis=-1).mean() > 0.9  # floor under the square now seen


def test_project_images_to_texture_back_faces_get_nothing():
    mesh = _squares([(0.0, (1, 1), (0, 0), (1, 1), True)])  # winding flipped: normal points away
    c2w, K = _camera()
    albedo = project_images_to_texture(*mesh, _constant_image(), c2w, K, tex_size=64, occlusion_eps=0.01)
    assert albedo.max() == 0  # nothing seen, so the fill has nothing to spread


######## create_texture_mesh


def test_create_texture_mesh_writes_obj_mtl_and_albedo(tmp_path):
    plane = _dense_plane(30)
    c2w, K = _camera()
    # The plane's rim is 2.8 × its scene scale, under the 3.9 gate; it stays open as the outer rim
    prepared = prepare_mesh(plane, voxel_size=0.01)
    out = create_texture_mesh(
        prepared, plane, tmp_path / "texture", _constant_image(), c2w, K, voxel_size=0.01, tex_size=64
    )
    assert out == tmp_path / "texture" / "mesh.obj"
    assert sorted(p.name for p in out.parent.iterdir()) == ["albedo.png", "mesh.mtl", "mesh.obj"]
    assert "Kd 1.0 1.0 1.0" in (out.parent / "mesh.mtl").read_text()
    assert "\nvn " in out.read_text()  # smooth normals; without them viewers shade split corners flat
    loaded = trimesh.load(out, process=False)
    assert loaded.visual.uv.shape == (len(loaded.vertices), 2)
    albedo = np.asarray(loaded.visual.material.image)
    assert albedo.shape[:2] == (64, 64) and np.abs(albedo[..., :3].astype(int) - [51, 128, 204]).max() <= 2
    assert len(plane.triangles) == 2 * 29 * 29  # the cleaned mesh is never modified
