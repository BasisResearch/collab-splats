import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
import torch
import trimesh
from plyfile import PlyData

from collab_splats.mesh.io import (
    render_tsdf_inputs,
    upsample_depths,
    write_textured_ply,
)
from collab_splats.preproc.frames import write_frames

######## upsample_depths


def _step_scene(factor=4):
    """Model-res depth with a vertical step edge + RGB guide whose edge aligns with it."""
    h, w = 32, 32
    depth = np.full((h, w), 1.0, dtype=np.float32)
    depth[:, w // 2 :] = 2.0
    H, W = h * factor, w * factor
    rgb = np.full((H, W, 3), 40, dtype=np.uint8)
    rgb[:, W // 2 :] = 200
    return depth, rgb


def test_upsample_depths_places_crop():
    depth, rgb = _step_scene(factor=2)
    canvas_hw = (100, 120)
    full_rgb = np.zeros((*canvas_hw, 3), dtype=np.uint8)
    full_rgb[10:74, 20:84] = rgb
    out = upsample_depths(depth[None], full_rgb[None], [[20, 10, 84, 74]])
    assert out.shape == (1, *canvas_hw) and out.dtype == np.float32
    out = out[0]
    assert np.all(out[:10] == 0) and np.all(out[74:] == 0)
    assert np.all(out[:, :20] == 0) and np.all(out[:, 84:] == 0)
    assert (out[10:74, 20:84] > 0).mean() > 0.99


def test_upsample_depths_masked_pixels_stay_zero():
    depth, rgb = _step_scene(factor=4)
    depth[8:16, 8:16] = 0.0
    H, W = rgb.shape[:2]
    out = upsample_depths(depth[None], rgb[None], [[0, 0, W, H]])[0]
    assert np.all(out[32:64, 32:64] == 0)
    valid = out[out > 0]
    assert valid.min() >= 1.0 - 1e-3 and valid.max() <= 2.0 + 1e-3


def test_upsample_depths_step_edge_stays_sharp():
    depth, rgb = _step_scene(factor=4)
    H, W = rgb.shape[:2]
    out = upsample_depths(depth[None], rgb[None], [[0, 0, W, H]])[0]
    interior = out[:, np.r_[0 : W // 2 - 8, W // 2 + 8 : W]]
    fabricated = (interior > 1.1) & (interior < 1.9)
    assert fabricated.mean() < 0.01


def test_upsample_depths_rejects_count_mismatch():
    depth, rgb = _step_scene(factor=2)
    with pytest.raises(ValueError, match="crop boxes"):
        upsample_depths(depth[None], rgb[None], [[0, 0, 64, 64], [0, 0, 64, 64]])


def test_upsample_depths_rejects_box_outside_canvas():
    depth, rgb = _step_scene(factor=2)
    with pytest.raises(ValueError, match="outside"):
        upsample_depths(depth[None], rgb[None], [[0, 0, 65, 64]])


######## render_tsdf_inputs


def _fake_rendering(views, image_ids, hw):
    """
    Stand-in for collab_splats.splats.rendering: fixed poses, canned views.
    """
    n = len(views)

    def load_checkpoint(path, device):
        return (
            "model",
            "camera_opt",
            torch.eye(4).repeat(n, 1, 1),
            torch.eye(3).repeat(n, 1, 1),
            list(image_ids),
            hw,
        )

    def render_views(model, camera_opt, cam_to_world, intrinsics, height, width):
        yield from views

    return SimpleNamespace(load_checkpoint=load_checkpoint, render_views=render_views)


def _view(h, w, depth, rgb=0.5, alpha=1.0, median_depth=None):
    view = {
        "rgb": torch.full((1, h, w, 3), rgb),
        "depth": torch.full((1, h, w, 1), depth),
        "alpha": torch.full((1, h, w, 1), alpha),
    }
    if median_depth is not None:
        view["median_depth"] = torch.full((1, h, w, 1), median_depth)
    return view


def test_render_tsdf_inputs_stacks_views_and_zeroes_empty_pixels(tmp_path, monkeypatch):
    h, w = 4, 6
    v0 = _view(h, w, depth=2.0)
    v0["alpha"][0, 0, 0, 0] = 0.0
    monkeypatch.setitem(
        sys.modules,
        "collab_splats.splats.rendering",
        _fake_rendering([v0, _view(h, w, depth=3.0)], [0, 1], (h, w)),
    )
    depths, rgbs, c2w, K = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert depths.shape == (2, h, w) and depths.dtype == np.float32
    assert depths[0, 0, 0] == 0.0 and depths[0, 1, 1] == 2.0 and np.all(depths[1] == 3.0)
    assert rgbs.shape == (2, h, w, 3) and rgbs.dtype == np.uint8 and np.all(rgbs == 127)
    assert c2w.shape == (2, 4, 4) and K.shape == (2, 3, 3)
    assert c2w.dtype == np.float32 and K.dtype == np.float32


def test_render_tsdf_inputs_prefers_median_depth(tmp_path, monkeypatch):
    h, w = 4, 6
    monkeypatch.setitem(
        sys.modules,
        "collab_splats.splats.rendering",
        _fake_rendering([_view(h, w, depth=9.0, median_depth=5.0)], [0], (h, w)),
    )
    depths, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert np.all(depths == 5.0)


def test_render_tsdf_inputs_swaps_in_source_frames_by_image_id(tmp_path, monkeypatch):
    h, w = 8, 8
    images_dir = tmp_path / "images"
    frames = [np.full((h, w, 3), idx * 10, dtype=np.uint8) for idx in (0, 5, 7)]
    write_frames(images_dir, frames, [{"frame_idx": idx} for idx in (0, 5, 7)], {})
    monkeypatch.setitem(
        sys.modules,
        "collab_splats.splats.rendering",
        _fake_rendering([_view(h, w, 1.0), _view(h, w, 1.0)], [7, 0], (h, w)),
    )
    _, rgbs, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")
    assert np.all(rgbs[0] == 70) and np.all(rgbs[1] == 0)


def test_render_tsdf_inputs_rejects_frame_size_mismatch(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    write_frames(images_dir, [np.zeros((8, 8, 3), np.uint8)], [{"frame_idx": 0}], {})
    monkeypatch.setitem(
        sys.modules,
        "collab_splats.splats.rendering",
        _fake_rendering([_view(4, 6, 1.0)], [0], (4, 6)),
    )
    with pytest.raises(ValueError, match="checkpoint renders"):
        render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")


######## write_textured_ply


def _unit_square_mesh():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    uv = verts[faces][..., :2]  # (F, 3, 2): u = x, v = y
    return mesh, uv


def test_write_textured_ply_writes_uv_and_texture_comment(tmp_path):
    mesh, uv = _unit_square_mesh()
    albedo = np.zeros((8, 8, 3), np.uint8)
    albedo[..., 0] = np.arange(8)[None, :] * 32  # red grows with column (u)
    albedo[..., 1] = np.arange(8)[:, None] * 32  # green grows with row
    out = write_textured_ply(mesh, uv, albedo, tmp_path)
    assert out == tmp_path / "mesh.ply" and (tmp_path / "albedo.png").exists()
    header = out.read_bytes()[:600].decode("ascii", "ignore")
    assert "property float s" in header and "property float t" in header
    assert "comment TextureFile albedo.png" in header

    # Six corners (two triangles), color sampled from the atlas at each corner's uv
    vertex = PlyData.read(str(out))["vertex"]
    assert len(vertex) == 6
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1)
    origin = np.flatnonzero((xyz == [0, 0, 0]).all(axis=1))[0]
    assert tuple(rgb[origin]) == tuple(albedo[7, 0]) == (0, 224, 0)  # v=0 → bottom row, u=0 → col 0
    right = np.flatnonzero((xyz == [1, 0, 0]).all(axis=1))[0]
    assert tuple(rgb[right]) == tuple(albedo[7, 7]) == (224, 224, 0)


def test_write_textured_ply_round_trips_through_trimesh_and_open3d(tmp_path):
    mesh, uv = _unit_square_mesh()
    albedo = np.full((8, 8, 3), 200, np.uint8)
    out = write_textured_ply(mesh, uv, albedo, tmp_path)
    tm = trimesh.load(str(out), process=False)
    assert tm.visual.uv.shape == (6, 2)
    assert tm.visual.material.image.size == (8, 8)
    o3 = o3d.io.read_triangle_mesh(str(out))
    assert o3.has_vertex_colors() and len(o3.triangles) == 2
    assert np.allclose(np.asarray(o3.vertex_colors), 200 / 255, atol=1 / 255)
