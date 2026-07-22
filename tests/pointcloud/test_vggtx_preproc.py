"""Tests for VGGTXCreator preprocessing — upstream crop mode."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image as PILImage


def _make_frames(n=2, width=1080, height=1920):
    """Create in-memory decoded frames + frame_idxs for preprocess tests."""
    frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(n)]
    return frames, list(range(n))


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


def test_preprocess_calls_crop_mode():
    """_preprocess must call load_and_preprocess_images with mode='crop'."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    frames, frame_idxs = _make_frames(n=2)
    c = VGGTXCreator()
    fake_images = torch.zeros(2, 3, 518, 518)
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images", return_value=fake_images) as m:
        images, paths, coords = c._preprocess(frames, frame_idxs)
        assert m.called
        _, kwargs = m.call_args
        assert kwargs.get("mode") == "crop", f"expected mode='crop', got {kwargs.get('mode')!r}"


def test_preprocess_original_coords_shape():
    """_preprocess returns original_coords with shape (N, 6)."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    frames, frame_idxs = _make_frames(n=3)
    c = VGGTXCreator()
    fake_images = torch.zeros(3, 3, 518, 518)
    with patch("collab_splats.pointcloud.feedforward.vggtx.load_and_preprocess_images", return_value=fake_images):
        _, paths, coords = c._preprocess(frames, frame_idxs)
    assert coords.shape == (3, 6), f"expected (3, 6), got {coords.shape}"
    assert coords.dtype == np.float32
