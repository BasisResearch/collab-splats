"""Tests for BaseFeatureExtractor.features_to_rgb."""

from __future__ import annotations

import numpy as np
import torch

from collab_splats.semantics.features.base import BaseFeatureExtractor

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
