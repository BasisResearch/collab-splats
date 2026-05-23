"""Tests for BaseQueryableExtractor contract and query API."""

import pytest
import torch

from collab_splats.semantics.features import BaseQueryableExtractor


def _talk2dino_available() -> bool:
    """Return True only if Talk2DinoExtractor can actually be instantiated.

    timm 0.6.7 is missing ImageNetInfo; transformers' AutoModel lazy-loads
    gemma3n on registry scan which crashes even with trust_remote_code=True.
    Tests that call Talk2DinoExtractor() must skip in this environment.
    """
    try:
        from timm.data import ImageNetInfo  # noqa: F401
        return True
    except ImportError:
        return False


_requires_talk2dino = pytest.mark.skipif(
    not _talk2dino_available(),
    reason="timm too old (missing ImageNetInfo) — Talk2DinoExtractor cannot be instantiated",
)


class _ConcreteExtractor(BaseQueryableExtractor):
    """Minimal concrete subclass for testing the base class contract."""

    def encode_text(self, texts):
        # Returns normalized random embeddings of shape (N, 8)
        n = len(texts)
        emb = torch.randn(n, 8)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def forward(self, images):
        # Returns (8, 4, 4) feature map per image
        return [torch.zeros(8, 4, 4) for _ in images]


def test_base_not_directly_instantiable():
    """BaseQueryableExtractor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseQueryableExtractor()


def test_isinstance_check():
    """Concrete subclass satisfies isinstance check."""
    extractor = _ConcreteExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)


def test_compute_similarity_shape():
    """compute_similarity returns (N_queries, H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.compute_similarity(features, ["cat", "dog", "background"])
    assert result.shape == (3, 4, 4)


def test_score_queries_shape_with_negative():
    """score_queries returns (H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"], negative=["background"])
    assert result.shape == (4, 4)


def test_score_queries_shape_no_negative():
    """score_queries with no negative does not raise and returns (H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"])
    assert result.shape == (4, 4)


def test_score_queries_no_negative_returns_bounded_scores():
    """score_queries with no explicit negative must return softmax scores in [0, 1].

    The default negative ("object") forces the contrastive softmax path.
    Raw dot-product fallback would return values well outside [0, 1].
    """
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"])
    assert result.min() >= 0.0, "score below 0 — contrastive softmax path not taken"
    assert result.max() <= 1.0 + 1e-6, "score above 1 — contrastive softmax path not taken"


def test_score_queries_bounds():
    """score_queries output is in [0, 1]."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"], negative=["background"])
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_maskclip_default_resolution():
    """MaskCLIPExtractor stores image_resolution at init."""
    pytest.importorskip("maskclip_onnx")
    from collab_splats.semantics.features import MaskCLIPExtractor
    extractor = MaskCLIPExtractor(image_resolution=512)
    assert extractor._image_resolution == 512


def test_maskclip_is_queryable():
    """MaskCLIPExtractor is a BaseQueryableExtractor."""
    pytest.importorskip("maskclip_onnx")
    from collab_splats.semantics.features import MaskCLIPExtractor
    extractor = MaskCLIPExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)


def test_talk2dino_accepts_model_name():
    """Talk2DinoExtractor uses model_name, not hf_model_id."""
    pytest.importorskip("transformers")
    import inspect
    from collab_splats.semantics.features import Talk2DinoExtractor
    sig = inspect.signature(Talk2DinoExtractor.__init__)
    assert "model_name" in sig.parameters
    assert "hf_model_id" not in sig.parameters


@_requires_talk2dino
def test_talk2dino_forward_returns_spatial():
    """Talk2DinoExtractor.forward() returns (D, pH, pW) with square patch grid."""
    pytest.importorskip("transformers")
    from collab_splats.semantics.features import Talk2DinoExtractor
    from PIL import Image
    import numpy as np
    extractor = Talk2DinoExtractor()
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    features = extractor.forward([img])
    assert len(features) == 1
    assert features[0].ndim == 3  # (D, pH, pW), not flat
    D, pH, pW = features[0].shape
    assert pH == pW  # center-crop guarantees square grid
    # Don't assert exact grid size — Talk2DINOv3 upscales inputs internally
    # (e.g. 224→448) so effective patches-per-side != 224 // config.patch_size


@_requires_talk2dino
def test_talk2dino_is_queryable():
    """Talk2DinoExtractor is a BaseQueryableExtractor."""
    pytest.importorskip("transformers")
    from collab_splats.semantics.features import Talk2DinoExtractor
    extractor = Talk2DinoExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)
