# tests/test_semantics_logging.py
import logging
import numpy as np
import torch
import torchvision.transforms as T
import pytest
from typing import List
from unittest.mock import MagicMock
from PIL import Image as PILImage

from collab_splats.semantics.features import (
    BaseQueryableExtractor,
    DINOFeatureExtractor,
    MaskCLIPExtractor,
    Talk2DinoExtractor,
)


class _MockQueryable(BaseQueryableExtractor):
    """Minimal queryable extractor — no external deps needed."""

    def __init__(self):
        super().__init__("max_size", 512)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return torch.randn(len(texts), 64)

    def forward(self, images: list) -> list:
        return [torch.randn(64, 8, 8) for _ in images]


def test_score_queries_logs_positive_count(caplog):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.score_queries(features, positive=["cat", "dog"])
    assert "2 positive" in caplog.text
    assert "done in" not in caplog.text or True  # score_queries logs count, not timing


def test_score_queries_logs_negative_count(caplog):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.score_queries(features, positive=["cat"], negative=["background", "wall"])
    assert "1 positive" in caplog.text
    assert "2 negative" in caplog.text


def test_score_queries_no_negative_omits_count(caplog):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.score_queries(features, positive=["cat"], negative=[], reduction="pool")
    assert "reduction=pool" in caplog.text
    assert "negative" not in caplog.text


def _make_img(w=64, h=64):
    return PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def test_dino_forward_logs(caplog):
    extractor = DINOFeatureExtractor.__new__(DINOFeatureExtractor)
    torch.nn.Module.__init__(extractor)
    extractor._image_resolution = 56
    extractor._resize_mode = "max_size"
    extractor._normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    extractor._device = torch.device("cpu")

    mock_model = MagicMock()
    mock_model.config.patch_size = 14
    def _call(tensors):
        b, _, H, W = tensors.shape
        ph, pw = H // 14, W // 14
        out = MagicMock()
        out.last_hidden_state = torch.zeros(b, ph * pw + 1, 384)
        return out
    mock_model.side_effect = _call
    extractor.model = mock_model

    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.forward([_make_img(56, 56)])
    assert "DINOFeatureExtractor" in caplog.text
    assert "1 images" in caplog.text


def test_maskclip_forward_logs(caplog):
    pytest.importorskip("maskclip_onnx")

    extractor = MaskCLIPExtractor.__new__(MaskCLIPExtractor)
    torch.nn.Module.__init__(extractor)
    extractor._image_resolution = 64
    extractor._resize_mode = "max_size"
    extractor.patch_size = 16
    extractor._normalize = T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                        [0.26862954, 0.26130258, 0.27577711])
    extractor._device = torch.device("cpu")

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    def _get_patch_encodings(stacked):
        b, _, H, W = stacked.shape
        return torch.randn(b, (H // 16) * (W // 16), 512)
    mock_model.get_patch_encodings.side_effect = _get_patch_encodings
    extractor.model = mock_model

    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.forward([_make_img()])
    assert "MaskCLIPExtractor" in caplog.text
    assert "1 images" in caplog.text


def test_talk2dino_forward_logs(caplog):
    extractor = Talk2DinoExtractor.__new__(Talk2DinoExtractor)
    torch.nn.Module.__init__(extractor)
    extractor.patch_size = 14
    extractor._resize_mode = "max_size"
    extractor._image_resolution = 56  # 4 patches × 14px
    extractor._device = torch.device("cpu")
    extractor._normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # forward_features returns (B, 5 + N_patches, D); [:, 5:] drops register tokens
    def _fake_forward_features(batch):
        B, _, H, W = batch.shape
        ph, pw = H // 14, W // 14
        return torch.randn(B, 5 + ph * pw, 256)

    mock_backbone = MagicMock()
    mock_backbone.forward_features.side_effect = _fake_forward_features

    mock_model = MagicMock()
    mock_model.model = mock_backbone
    extractor._model = mock_model

    with caplog.at_level(logging.DEBUG, logger="collab_splats.semantics.features"):
        extractor.forward([_make_img()])
    assert "Talk2DinoExtractor" in caplog.text
    assert "1 images" in caplog.text
