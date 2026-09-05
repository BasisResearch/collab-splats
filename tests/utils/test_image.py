"""Tests for collab_splats.utils.image — pure PIL image utilities."""

import numpy as np
import pytest
from PIL import Image

from collab_splats.utils.image import open_image, resize_image


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
    """The stats extractors normalize with live next to the resize helpers that feed them."""
    from collab_splats.utils.image import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD

    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
    assert CLIP_MEAN == [0.48145466, 0.4578275, 0.40821073]
    assert CLIP_STD == [0.26862954, 0.26130258, 0.27577711]
