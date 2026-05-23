"""Tests for BaseFeatureExtractor._get_memory_per_image."""

import torch
import torch.nn as nn
from PIL import Image

from collab_splats.semantics.features import BaseFeatureExtractor


class DummyExtractor(BaseFeatureExtractor):
    """Minimal extractor for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, images: list) -> list[torch.Tensor]:
        return [torch.randn(10) for _ in images]


def test_get_memory_returns_positive_float():
    ext = DummyExtractor()
    sample = Image.new("RGB", (64, 64))
    result = ext._get_memory_per_image(sample)
    assert isinstance(result, float)
    assert result > 0.0


def test_get_memory_cpu_returns_fallback():
    if not torch.cuda.is_available():
        ext = DummyExtractor()
        sample = Image.new("RGB", (64, 64))
        result = ext._get_memory_per_image(sample)
        assert result == 2.0
