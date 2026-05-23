# tests/test_semantics_logging.py
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

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        return torch.randn(len(texts), 64)

    def forward(self, images: list) -> list:
        return [torch.randn(64, 8, 8) for _ in images]


def test_score_queries_logs_positive_count(capsys):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    extractor.score_queries(features, positive=["cat", "dog"])
    out = capsys.readouterr().out
    assert "2 positive" in out
    assert "done in" in out


def test_score_queries_logs_negative_count(capsys):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    extractor.score_queries(features, positive=["cat"], negative=["background", "wall"])
    out = capsys.readouterr().out
    assert "1 positive" in out
    assert "2 negative" in out
    assert "done in" in out


def test_score_queries_no_negative_omits_count(capsys):
    extractor = _MockQueryable()
    features = torch.randn(64, 8, 8)
    extractor.score_queries(features, positive=["cat"], reduction="pool")
    out = capsys.readouterr().out
    assert "reduction=pool" in out
    assert "negative" not in out


def _make_img(w=64, h=64):
    return PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def test_dino_forward_logs(capsys):
    extractor = DINOFeatureExtractor.__new__(DINOFeatureExtractor)
    extractor.patch_size = 14
    extractor.resolution = 56
    extractor.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    def _call(tensors):
        b, _, H, W = tensors.shape
        ph, pw = H // 14, W // 14
        out = MagicMock()
        out.last_hidden_state = torch.zeros(b, ph * pw + 1, 384)
        return out
    mock_model.side_effect = _call
    extractor.model = mock_model

    extractor.forward([_make_img(56, 56)])
    out = capsys.readouterr().out
    assert "DINOFeatureExtractor" in out
    assert "1 images" in out
    assert "done in" in out


def test_maskclip_forward_logs(capsys):
    pytest.importorskip("maskclip_onnx")

    extractor = MaskCLIPExtractor.__new__(MaskCLIPExtractor)
    extractor.default_resolution = 64
    extractor.patch_size = 16
    extractor.transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    def _get_patch_encodings(stacked):
        b, _, H, W = stacked.shape
        return torch.randn(b, (H // 16) * (W // 16), 512)
    mock_model.get_patch_encodings.side_effect = _get_patch_encodings
    extractor.model = mock_model

    extractor.forward([_make_img()])
    out = capsys.readouterr().out
    assert "MaskCLIPExtractor" in out
    assert "1 images" in out
    assert "done in" in out


def test_talk2dino_forward_logs(capsys):
    extractor = Talk2DinoExtractor.__new__(Talk2DinoExtractor)
    extractor.patch_size = 14
    extractor._device = torch.device("cpu")

    n_patches = 16  # 4×4 square grid
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.randn(1, n_patches, 256)
    extractor._model = mock_model

    extractor.forward([_make_img()])
    out = capsys.readouterr().out
    assert "Talk2DinoExtractor" in out
    assert "1 images" in out
    assert "done in" in out
