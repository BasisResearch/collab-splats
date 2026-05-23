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
