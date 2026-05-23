"""Tests for unified extractor preprocessing utilities and interface."""
import torch
import torch.nn.functional as F
import pytest
from PIL import Image

from collab_splats.semantics.utils import _tokens_to_feature_map


def test_tokens_to_feature_map_shape():
    patch_size = 14
    H, W = 196, 280  # multiples of 14
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D)
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    assert out.shape == (D, ph, pw)


def test_tokens_to_feature_map_l2_normalized():
    patch_size = 14
    H, W = 196, 196
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D) * 10  # large values
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    # F.normalize(feat, dim=0) normalizes along channel dim per spatial position
    norms = out.norm(dim=0)  # (H_p, W_p)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tokens_to_feature_map_wrong_count_raises():
    with pytest.raises(AssertionError):
        _tokens_to_feature_map(torch.randn(99, 8), 196, 196, 14)
