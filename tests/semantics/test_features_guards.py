import pytest
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
