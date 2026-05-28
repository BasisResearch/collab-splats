"""Tests for BaseFeatureExtractor.features_to_rgb and extract_and_cache_from_zarr."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from collab_splats.semantics.features.base import BaseFeatureExtractor


########################################################################
# Minimal concrete extractor for tests — no model weights needed
########################################################################

@BaseFeatureExtractor.register("_test_extractor")
class _TestExtractor(BaseFeatureExtractor):
    """Returns constant (D, H_p, W_p) tensors — no model needed."""

    patch_size = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._D = 8
        self._H_p = 4
        self._W_p = 4

    def forward(self, images: list) -> list[torch.Tensor]:
        return [torch.ones(self._D, self._H_p, self._W_p) for _ in images]


def _make_frames_zarr(n: int, H: int, W: int, tmp_dir: str) -> Path:
    """Write a minimal frames.zarr with n random uint8 frames."""
    zarr_path = Path(tmp_dir) / "frames.zarr"
    store = zarr.open(str(zarr_path), mode="w")
    store.attrs.update({"n_frames": n, "height": H, "width": W})
    arr = store.create_array(
        "frames",
        shape=(n, H, W, 3),
        chunks=(1, H, W, 3),
        dtype="uint8",
        fill_value=0,
    )
    for i in range(n):
        arr[i] = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    return zarr_path


########################################################################
# features_to_rgb
########################################################################

def test_features_to_rgb_shape():
    feat = torch.randn(16, 6, 8)  # (D, H_p, W_p)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    assert rgb.shape == (6, 8, 3)
    assert rgb.dtype == np.uint8


def test_features_to_rgb_range():
    feat = torch.randn(32, 4, 4)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    assert rgb.min() >= 0
    assert rgb.max() <= 255


def test_features_to_rgb_constant_returns_zero():
    # All-identical feature vectors → PCA variance is zero → output is 0 after normalization
    feat = torch.ones(16, 4, 4)
    rgb = BaseFeatureExtractor.features_to_rgb(feat)
    assert rgb.max() == 0


########################################################################
# extract_and_cache_from_zarr
########################################################################

def test_extract_and_cache_from_zarr_creates_zarr():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features" / "_test_extractor"
        result = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        assert result.exists()
        z = zarr.open(str(result), mode="r")
        assert z["features"].shape[0] == 3
        assert z.attrs["extractor"] == "_test_extractor"
        assert z.attrs["n_frames"] == 3


def test_extract_and_cache_from_zarr_feature_shape():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=4, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features"
        result = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        z = zarr.open(str(result), mode="r")
        N, D, H_p, W_p = z["features"].shape
        assert N == 4
        assert D == extractor._D
        assert H_p == extractor._H_p
        assert W_p == extractor._W_p


def test_extract_and_cache_from_zarr_skip_existing():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=2, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "features"
        result1 = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        mtime1 = result1.stat().st_mtime
        result2 = extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
        assert result1 == result2
        assert result2.stat().st_mtime == mtime1
