"""
maskclip_onnx imports when installed; a semantics-only env skips it.
"""

import pytest

maskclip_onnx = pytest.importorskip("maskclip_onnx")


def test_maskclip_onnx_importable():
    assert callable(maskclip_onnx.clip.load)
