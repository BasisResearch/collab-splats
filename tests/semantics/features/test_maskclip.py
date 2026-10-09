"""
maskclip_onnx is a required dependency: its absence fails the suite, not just skips it.
"""

import maskclip_onnx


def test_maskclip_onnx_importable():
    """maskclip_onnx must be installed — it is a required dependency."""
    assert callable(maskclip_onnx.clip.load)
