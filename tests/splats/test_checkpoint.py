"""
Tests for collab_splats.splats.checkpoint: render_tsdf_inputs with a stubbed checkpoint.
"""

import numpy as np
import pytest
import torch

from collab_splats.preproc.frames import write_frames
from collab_splats.splats.checkpoint import render_tsdf_inputs


def _fake_rendering(monkeypatch, views, image_ids, hw):
    """
    Stub load_checkpoint and render_views: fixed poses, canned views.
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

    monkeypatch.setattr("collab_splats.splats.checkpoint.load_checkpoint", load_checkpoint)
    monkeypatch.setattr("collab_splats.splats.checkpoint.render_views", render_views)


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
    _fake_rendering(monkeypatch, [v0, _view(h, w, depth=3.0)], [0, 1], (h, w))
    depths, rgbs, c2w, K, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert depths.shape == (2, h, w) and depths.dtype == np.float32
    assert depths[0, 0, 0] == 0.0 and depths[0, 1, 1] == 2.0 and np.all(depths[1] == 3.0)
    # 0.5 x 255 = 127.5 rounds half-to-even up to 128
    assert rgbs.shape == (2, h, w, 3) and rgbs.dtype == np.uint8 and np.all(rgbs == 128)
    assert c2w.shape == (2, 4, 4) and K.shape == (2, 3, 3)
    assert c2w.dtype == np.float32 and K.dtype == np.float32


def test_render_tsdf_inputs_defaults_to_the_expected_depth(tmp_path, monkeypatch):
    h, w = 4, 6
    _fake_rendering(monkeypatch, [_view(h, w, depth=9.0, median_depth=5.0)], [0], (h, w))
    depths, _, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu")
    assert np.all(depths == 9.0)


def test_render_tsdf_inputs_median_source_takes_the_median_depth(tmp_path, monkeypatch):
    h, w = 4, 6
    _fake_rendering(monkeypatch, [_view(h, w, depth=9.0, median_depth=5.0)], [0], (h, w))
    depths, _, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu", depth_source="median")
    assert np.all(depths == 5.0)


def test_render_tsdf_inputs_expected_source_ignores_median_depth(tmp_path, monkeypatch):
    h, w = 4, 6
    _fake_rendering(monkeypatch, [_view(h, w, depth=9.0, median_depth=5.0)], [0], (h, w))
    depths, _, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu", depth_source="expected")
    assert np.all(depths == 9.0)


def test_render_tsdf_inputs_median_source_falls_back_on_a_3dgs_checkpoint(tmp_path, monkeypatch):
    h, w = 4, 6
    _fake_rendering(monkeypatch, [_view(h, w, depth=9.0)], [0], (h, w))
    depths, _, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu", depth_source="median")
    assert np.all(depths == 9.0)


def test_render_tsdf_inputs_rejects_an_unknown_depth_source(tmp_path):
    with pytest.raises(ValueError, match="depth_source"):
        render_tsdf_inputs(tmp_path / "ckpt.pt", device="cpu", depth_source="nearest")


def test_render_tsdf_inputs_swaps_in_source_frames_by_image_id(tmp_path, monkeypatch):
    h, w = 8, 8
    images_dir = tmp_path / "images"
    frames = [np.full((h, w, 3), idx * 10, dtype=np.uint8) for idx in (0, 5, 7)]
    write_frames(images_dir, frames, [{"frame_idx": idx} for idx in (0, 5, 7)], {})
    _fake_rendering(monkeypatch, [_view(h, w, 1.0), _view(h, w, 1.0)], [7, 0], (h, w))
    _, rgbs, _, _, _ = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")
    assert np.all(rgbs[0] == 70) and np.all(rgbs[1] == 0)


def test_render_tsdf_inputs_returns_the_image_ids_it_read(tmp_path, monkeypatch):
    h, w = 8, 8
    images_dir = tmp_path / "images"
    frames = [np.full((h, w, 3), idx * 10, dtype=np.uint8) for idx in (0, 5, 7)]
    write_frames(images_dir, frames, [{"frame_idx": idx} for idx in (0, 5, 7)], {})
    _fake_rendering(monkeypatch, [_view(h, w, 1.0), _view(h, w, 1.0)], [7, 0], (h, w))
    _, rgbs, _, _, image_ids = render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")

    # The ids must be the order the RGB rows are actually in, not the directory's order
    assert image_ids == [7, 0]
    assert np.all(rgbs[0] == 70) and np.all(rgbs[1] == 0)


def test_render_tsdf_inputs_rejects_frame_size_mismatch(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    write_frames(images_dir, [np.zeros((8, 8, 3), np.uint8)], [{"frame_idx": 0}], {})
    _fake_rendering(monkeypatch, [_view(4, 6, 1.0)], [0], (4, 6))
    with pytest.raises(ValueError, match="checkpoint renders"):
        render_tsdf_inputs(tmp_path / "ckpt.pt", images_dir=images_dir, device="cpu")
