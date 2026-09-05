"""Tests for BaseFeatureExtractor.features_to_rgb and semantics.utils.extract_feature_cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import zarr

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.semantics.features.base import BaseFeatureExtractor
from collab_splats.semantics.utils import extract_feature_cache

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
    """Write a frames.zarr of n random uint8 frames using the real writer.

    Must go through FrameStore.create, not a hand-rolled zarr: the previous fixture invented
    an `n_frames` attr and a `frames` array, so extract_feature_cache passed its tests
    while crashing on every real store (which has `images` + record_keys).
    """
    zarr_path = Path(tmp_dir) / "frames.zarr"
    frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]
    records = [{"frame_idx": i} for i in range(n)]
    FrameStore.create(zarr_path, frames, records, provenance={"source": "test"})
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
# extract_feature_cache
########################################################################


def test_extract_feature_cache_creates_zarr():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result = extract_feature_cache(extractor, frames_zarr, cache_dir)
        assert result == cache_dir / "_test_extractor.zarr"
        assert result.exists()


def test_extract_feature_cache_feature_shape():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result = extract_feature_cache(extractor, frames_zarr, cache_dir)
        store = zarr.open(str(result), mode="r")
        N, D, H_p, W_p = store["features"].shape
        assert N == 3
        assert D == extractor._D
        assert H_p == extractor._H_p
        assert W_p == extractor._W_p
        assert store.attrs["extractor"] == "_test_extractor"
        assert store.attrs["n_frames"] == 3
        assert store.attrs["patch_size"] == 16
        # created_at / feature_dim were dropped — nothing read them
        assert "created_at" not in store.attrs
        assert "feature_dim" not in store.attrs


def test_extract_feature_cache_reuses_a_matching_cache():
    extractor = _TestExtractor()
    with tempfile.TemporaryDirectory() as tmp:
        frames_zarr = _make_frames_zarr(n=3, H=64, W=64, tmp_dir=tmp)
        cache_dir = Path(tmp) / "cache"
        result1 = extract_feature_cache(extractor, frames_zarr, cache_dir)
        mtime = result1.stat().st_mtime_ns
        result2 = extract_feature_cache(extractor, frames_zarr, cache_dir)
        assert result1 == result2
        assert result2.stat().st_mtime_ns == mtime  # untouched -> extraction was skipped
