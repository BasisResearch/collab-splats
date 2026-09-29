"""Tests for collab_splats.utils.image — pure PIL image utilities."""

import numpy as np
import pytest
from PIL import Image

from collab_splats.utils.image import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    fill_missing_pixels,
    open_image,
    resize_image,
    upsample_depths,
)


def test_open_image_pil_passthrough():
    img = Image.new("RGB", (100, 100))
    assert open_image(img) is img


def test_open_image_from_ndarray():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    result = open_image(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (100, 100)


def test_open_image_from_path(tmp_path):
    p = tmp_path / "test.png"
    Image.new("RGB", (50, 50)).save(p)
    result = open_image(str(p))
    assert result.size == (50, 50)


def test_open_image_from_pathlib(tmp_path):
    p = tmp_path / "test.png"
    Image.new("RGB", (50, 50)).save(p)
    result = open_image(p)
    assert isinstance(result, Image.Image)


def test_open_image_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported image type"):
        open_image(42)


def test_resize_image_landscape():
    img = Image.new("RGB", (200, 100))
    resized = resize_image(img, longest_edge=100)
    assert resized.size[0] == 100
    assert resized.size[1] == 50


def test_resize_image_portrait():
    img = Image.new("RGB", (100, 200))
    resized = resize_image(img, longest_edge=100)
    assert resized.size[0] == 50
    assert resized.size[1] == 100


def test_resize_image_square():
    img = Image.new("RGB", (200, 200))
    resized = resize_image(img, longest_edge=100)
    assert resized.size == (100, 100)


def test_normalization_constants():
    """The shared normalization constants hold the standard ImageNet and CLIP values."""
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
    assert CLIP_MEAN == [0.48145466, 0.4578275, 0.40821073]
    assert CLIP_STD == [0.26862954, 0.26130258, 0.27577711]


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


def test_fill_missing_pixels_keeps_known_and_fills_the_rest():
    image = np.zeros((32, 48, 3), dtype=np.float32)
    known = np.zeros((32, 48), dtype=bool)
    image[:, :10] = 0.25
    known[:, :10] = True

    out = fill_missing_pixels(image, known)

    assert out.shape == image.shape and out.dtype == np.float32
    np.testing.assert_allclose(out[known], 0.25, atol=1e-6)
    np.testing.assert_allclose(out[~known], 0.25, atol=1e-4)


def test_fill_missing_pixels_single_channel_and_nothing_known():
    heights = np.full((20, 20), 2.0, dtype=np.float32)
    known = np.zeros((20, 20), dtype=bool)
    known[5, 5] = True

    np.testing.assert_allclose(fill_missing_pixels(heights, known), 2.0, atol=1e-4)
    assert not fill_missing_pixels(heights, np.zeros_like(known)).any()
