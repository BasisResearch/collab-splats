import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import collab_splats.semantics.features as feat_mod


def test_maskclip_extractor_raises_without_package(monkeypatch):
    """MaskCLIPExtractor raises ImportError when maskclip_onnx unavailable."""
    orig = feat_mod._MASKCLIP_AVAILABLE
    try:
        feat_mod._MASKCLIP_AVAILABLE = False
        with pytest.raises(ImportError, match="maskclip_onnx"):
            feat_mod.MaskCLIPExtractor()
    finally:
        feat_mod._MASKCLIP_AVAILABLE = orig


def test_pytorch_gc_safe_on_cpu(monkeypatch):
    """pytorch_gc must not raise RuntimeError on CPU-only systems."""
    import torch
    import importlib
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import collab_splats.semantics.features as mod
    importlib.reload(mod)
    mod.pytorch_gc()  # must not raise


def test_open_image_from_str_path(tmp_path):
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    p = tmp_path / "test.png"
    img.save(p)
    result = feat_mod._open_image(str(p))
    assert isinstance(result, Image.Image)


def test_open_image_from_path_object(tmp_path):
    img = Image.new("RGB", (10, 10))
    p = tmp_path / "test.png"
    img.save(p)
    result = feat_mod._open_image(p)
    assert isinstance(result, Image.Image)


def test_open_image_from_ndarray():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    result = feat_mod._open_image(arr)
    assert isinstance(result, Image.Image)


def test_open_image_from_pil_returns_same():
    img = Image.new("RGB", (10, 10))
    result = feat_mod._open_image(img)
    assert result is img


def test_open_image_invalid_type():
    with pytest.raises(ValueError, match="Unsupported image type"):
        feat_mod._open_image(42)
