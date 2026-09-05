"""Tests for positional debiasing in BaseFeatureExtractor.

Uses _FakeExtractor — a lightweight subclass with no model weights — to test
the infrastructure in BaseFeatureExtractor without requiring GPU or HuggingFace downloads.
"""

import logging
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from collab_splats.semantics.features import BaseFeatureExtractor, _DEBIAS_VALIDATED


# ---------------------------------------------------------------------------
# Fake extractor: minimal concrete implementation for testing base-class logic
# ---------------------------------------------------------------------------

class _FakeExtractor(BaseFeatureExtractor):
    """Returns deterministic features based on pixel mean so image content affects output.

    patch_size=14 matches standard ViT-B stride — required by _build_positional_basis
    so the zero-image dimensions are computed correctly.
    """

    patch_size = 14

    def __init__(self, feature_dim: int = 16, h_p: int = 4, w_p: int = 4, **kwargs):
        super().__init__("max_size", 512, **kwargs)  # resize/resolution unused: forward is synthetic
        self._feature_dim = feature_dim  # D: output patch feature dimensionality
        self._h_p = h_p                  # H_p: fixed patch grid height (regardless of input size)
        self._w_p = w_p                  # W_p: fixed patch grid width

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")  # CPU-only — no GPU needed in tests

    def forward(self, images: list) -> list[torch.Tensor]:
        results = []
        for img in images:
            arr = np.array(img)
            # Seed from pixel mean: zero-pixel image (mean=0) gives different features than
            # content images (mean>0), which is required for debiasing tests to be meaningful.
            seed = int(arr.mean() * 1000) % (2 ** 31)
            gen = torch.Generator()
            gen.manual_seed(seed)
            feat = torch.randn(self._feature_dim, self._h_p, self._w_p, generator=gen)
            results.append(F.normalize(feat, p=2, dim=0))  # unit-norm patches, matches real extractor contract
        return results


class _UnvalidatedExtractor(_FakeExtractor):
    """Subclass of _FakeExtractor NOT listed in _DEBIAS_VALIDATED — used to test warning path."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(h: int = 64, w: int = 64, value: int = 100) -> Image.Image:
    """Create a solid-color PIL image for use as test input."""
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Task 1: smoke test — base class state is initialised
# ---------------------------------------------------------------------------

def test_base_state_initialised():
    """BaseFeatureExtractor.__init__ must set svd_components, _pos_basis_cache, _zero_feats_cache."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=8)
    assert extractor.svd_components == 8
    assert extractor._pos_basis_cache == {}
    assert extractor._zero_feats_cache == {}


def test_forward_unchanged():
    """Subclass forward() must work exactly as before — debias state must not affect it."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4)
    img = _make_rgb_image()
    result = extractor.forward([img])
    assert isinstance(result, list) and len(result) == 1
    assert result[0].shape == (16, 4, 4)


# ---------------------------------------------------------------------------
# Task 2: debias() — SVD projection, shape, caching, warning
# ---------------------------------------------------------------------------

def test_debias_returns_list_of_tensors_same_shape():
    """debias() must return list[Tensor] with same shape as input features."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    features = extractor.forward([img])
    debiased = extractor.debias(features)
    assert isinstance(debiased, list) and len(debiased) == 1
    assert debiased[0].shape == features[0].shape


def test_debias_output_differs_from_input():
    """debias() must produce different features than the raw forward() output."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)  # non-zero pixel value → differs from zero-image basis
    features = extractor.forward([img])
    debiased = extractor.debias(features)
    # Debiasing subtracts a structured positional component — outputs must differ.
    assert not torch.allclose(features[0], debiased[0])


def test_debias_output_is_unit_norm():
    """_apply_debias re-normalizes L2 — each patch vector in output must be unit norm."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    debiased = extractor.debias(extractor.forward([img]))
    norms = debiased[0].norm(dim=0)  # (H_p, W_p) — norm of each patch vector along feature dim
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_positional_basis_cached_after_first_debias_call():
    """_pos_basis_cache must be populated after first debias() call and NOT rebuilt on second."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    assert len(extractor._pos_basis_cache) == 0  # empty before first call

    features = extractor.forward([img])
    extractor.debias(features)
    assert (4, 4) in extractor._pos_basis_cache  # cached after first call

    # Monkey-patch to detect if _build_positional_basis is called a second time.
    calls = []
    original = extractor._build_positional_basis
    extractor._build_positional_basis = lambda *a, **kw: calls.append(1) or original(*a, **kw)

    extractor.debias(features)  # second call at same resolution
    assert len(calls) == 0  # basis must NOT be rebuilt — must reuse from cache


def test_unvalidated_extractor_warns_on_debias(caplog):
    """debias() on an extractor not in _DEBIAS_VALIDATED must log WARNING (not raise)."""
    assert _UnvalidatedExtractor.__name__ not in _DEBIAS_VALIDATED  # confirm test precondition
    extractor = _UnvalidatedExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    features = extractor.forward([img])
    with caplog.at_level(logging.WARNING, logger="collab_splats.semantics.features"):
        debiased = extractor.debias(features)  # must not raise
    assert any("not yet validated" in r.message for r in caplog.records)
    assert debiased[0].shape == features[0].shape  # still returns correct output despite warning


# ---------------------------------------------------------------------------
# Task 3: get_bias_visualization
# ---------------------------------------------------------------------------

def test_get_bias_visualization_correct_shape_and_dtype():
    """get_bias_visualization must return (H_p, W_p, 3) uint8 after a debias() call."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    extractor.debias(extractor.forward([img]))  # populates _zero_feats_cache
    viz = extractor.get_bias_visualization(4, 4)
    assert isinstance(viz, np.ndarray)
    assert viz.shape == (4, 4, 3)
    assert viz.dtype == np.uint8


def test_get_bias_visualization_raises_without_prior_debias_call():
    """get_bias_visualization must raise KeyError if debias() has not been called first."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    with pytest.raises(KeyError, match="No positional bias cached"):
        extractor.get_bias_visualization(4, 4)


def test_get_bias_visualization_values_in_range():
    """get_bias_visualization output values must lie in [0, 255]."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    extractor.debias(extractor.forward([img]))
    viz = extractor.get_bias_visualization(4, 4)
    assert int(viz.min()) >= 0
    assert int(viz.max()) <= 255


def test_missing_patch_size_raises():
    """_build_positional_basis must raise AttributeError if patch_size is not set on the extractor."""
    class _NoPatchSizeExtractor(BaseFeatureExtractor):
        # patch_size intentionally absent — simulates a subclass that forgot to set it
        def forward(self, images: list) -> list:
            return []

    extractor = _NoPatchSizeExtractor("max_size", 512)
    with pytest.raises(AttributeError, match="patch_size"):
        extractor._build_positional_basis(4, 4)
