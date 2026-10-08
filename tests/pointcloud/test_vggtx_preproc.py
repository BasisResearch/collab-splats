"""Tests for VGGTXCreator preprocessing — upstream crop mode."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from tests.pointcloud.conftest import _frame_files, _frames


def _make_frames(tmp_path, n=2, width=1080, height=1920):
    """Black frame files for preprocess tests."""
    return _frame_files(_frames([(width, height)] * n), tmp_path)


def test_vggtx_creator_no_resize_mode_attribute():
    """resize_mode field removed — upstream crop mode is the only preprocessing."""
    c = VGGTXCreator()
    assert not hasattr(c, "resize_mode"), "resize_mode attribute should be removed"


def test_vggtx_creator_no_post_init_resize_mode_check():
    """Passing resize_mode= as kwarg raises TypeError (unexpected keyword), not ValueError."""
    with pytest.raises(TypeError):
        VGGTXCreator(resize_mode="square")  # type: ignore[call-arg]


def test_preprocess_calls_crop_mode(tmp_path):
    """_preprocess must call load_and_preprocess_images with mode='crop'."""
    paths = _make_frames(tmp_path, n=2)
    c = VGGTXCreator()
    fake_images = torch.zeros(2, 3, 518, 518)
    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images",
        return_value=fake_images,
    ) as m:
        images, coords = c._preprocess(paths)
        assert m.called
        _, kwargs = m.call_args
        assert kwargs.get("mode") == "crop", (
            f"expected mode='crop', got {kwargs.get('mode')!r}"
        )


def test_preprocess_original_coords_shape(tmp_path):
    """_preprocess returns original_coords with shape (N, 6)."""
    paths = _make_frames(tmp_path, n=3)
    c = VGGTXCreator()
    fake_images = torch.zeros(3, 3, 518, 518)
    with patch(
        "collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images",
        return_value=fake_images,
    ):
        _, coords = c._preprocess(paths)
    assert coords.shape == (3, 6), f"expected (3, 6), got {coords.shape}"
    assert coords.dtype == np.float32


########################################################################
########## in-memory frame handoff #####################################
########################################################################


@pytest.mark.parametrize(
    "sizes",
    [
        [(96, 48)] * 9,
        [(96, 48)] * 4 + [(48, 96)] * 3,
        [(200, 50)] * 3,
        [(64, 300)] * 2,
        [(518, 280), (300, 301)],
        [(1920, 1080), (1080, 1920), (1920, 1080)],
    ],
)
def test_preprocess_frames_match_files(tmp_path, sizes):
    """
    Handed-off arrays preprocess bit-exact to the same frames read back from PNG.
    """
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for w, h in sizes]
    paths = _frame_files(frames, tmp_path)

    views_f, coords_f = VGGTXCreator()._preprocess(paths)

    from_arrays = VGGTXCreator()
    from_arrays.frames = {p.name: f for p, f in zip(paths, frames, strict=True)}
    views_a, coords_a = from_arrays._preprocess(paths)

    assert torch.equal(views_a, views_f)
    np.testing.assert_array_equal(coords_a, coords_f)
