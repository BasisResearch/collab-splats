import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import collab_splats.semantics.features as feat_mod
from collab_splats.utils.image import open_image
from collab_splats.semantics.utils import compute_semantic_contrast, pytorch_gc


def test_maskclip_onnx_importable():
    """maskclip_onnx must be installed — it is a required dependency."""
    import maskclip_onnx  # raises ImportError if missing


def test_pytorch_gc_safe_on_cpu(monkeypatch):
    """pytorch_gc must not raise RuntimeError on CPU-only systems."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pytorch_gc()  # must not raise


def test_open_image_from_str_path(tmp_path):
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    p = tmp_path / "test.png"
    img.save(p)
    result = open_image(str(p))
    assert isinstance(result, Image.Image)


def test_open_image_from_path_object(tmp_path):
    img = Image.new("RGB", (10, 10))
    p = tmp_path / "test.png"
    img.save(p)
    result = open_image(p)
    assert isinstance(result, Image.Image)


def test_open_image_from_ndarray():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    result = open_image(arr)
    assert isinstance(result, Image.Image)


def test_open_image_from_pil_returns_same():
    img = Image.new("RGB", (10, 10))
    result = open_image(img)
    assert result is img


def test_open_image_invalid_type():
    with pytest.raises(ValueError, match="Unsupported image type"):
        open_image(42)


def test_compute_semantic_contrast_max_shape():
    raw = torch.rand(3, 6)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="max")
    assert result.shape == (6,)


def test_compute_semantic_contrast_pool_shape():
    raw = torch.rand(3, 6)
    result = compute_semantic_contrast(raw, num_positive=1, temperature=0.05, reduction="pool")
    assert result.shape == (6,)


def test_compute_semantic_contrast_unknown_raises():
    raw = torch.rand(2, 4)
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(raw, num_positive=1, temperature=0.05, reduction="bad")
