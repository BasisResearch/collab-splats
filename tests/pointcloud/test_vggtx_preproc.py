"""Tests for VGGTXCreator preprocessing — upstream crop mode."""
import pytest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from PIL import Image as PILImage


def _make_image_dir(tmp_path, n=2, width=1080, height=1920):
    """Create real JPEG images (PIL-readable) for preprocess tests."""
    for i in range(n):
        p = tmp_path / f"frame_{i:04d}.jpg"
        PILImage.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(p)
    return tmp_path


def test_vggtx_creator_no_resize_mode_attribute():
    """resize_mode field removed — upstream crop mode is the only preprocessing."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    c = VGGTXCreator()
    assert not hasattr(c, "resize_mode"), "resize_mode attribute should be removed"


def test_vggtx_creator_no_post_init_resize_mode_check():
    """Passing resize_mode= as kwarg raises TypeError (unexpected keyword), not ValueError."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    with pytest.raises(TypeError):
        VGGTXCreator(resize_mode="square")  # type: ignore[call-arg]


def test_preprocess_calls_crop_mode(tmp_path):
    """_preprocess must call load_and_preprocess_images with mode='crop'."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path, n=2)
    c = VGGTXCreator()
    fake_images = torch.zeros(2, 3, 518, 518)
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images",
               return_value=fake_images) as m:
        images, paths, coords = c._preprocess(img_dir)
        assert m.called
        _, kwargs = m.call_args
        assert kwargs.get("mode") == "crop", f"expected mode='crop', got {kwargs.get('mode')!r}"


def test_preprocess_original_coords_shape(tmp_path):
    """_preprocess returns original_coords with shape (N, 6)."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    img_dir = _make_image_dir(tmp_path, n=3)
    c = VGGTXCreator()
    fake_images = torch.zeros(3, 3, 518, 518)
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images",
               return_value=fake_images):
        _, paths, coords = c._preprocess(img_dir)
    assert coords.shape == (3, 6), f"expected (3, 6), got {coords.shape}"
    assert coords.dtype == np.float32
