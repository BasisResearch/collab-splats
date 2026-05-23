"""Tests for shape-aware compute_similarity and score_queries."""

import torch
import torch.nn.functional as F

from collab_splats.semantics.features import BaseQueryableExtractor

D = 8  # feature dimension for the fake extractor


class _FakeExtractor(BaseQueryableExtractor):
    """Minimal concrete extractor — avoids loading real model weights."""

    def encode_text(self, texts):
        emb = torch.randn(len(texts), D)
        return emb / emb.norm(dim=-1, keepdim=True)

    def forward(self, images):
        return [torch.zeros(D, 4, 4) for _ in images]


def test_compute_similarity_point_array():
    ext = _FakeExtractor()
    P = 100
    features = F.normalize(torch.randn(P, D), dim=-1)
    out = ext.compute_similarity(features, ["tree", "ground"])
    assert out.shape == (2, P)


def test_score_queries_point_array():
    ext = _FakeExtractor()
    P = 100
    features = F.normalize(torch.randn(P, D), dim=-1)
    scores = ext.score_queries(features, positive=["tree"], negative=["background"])
    assert scores.shape == (P,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_compute_similarity_image_map_unchanged():
    ext = _FakeExtractor()
    H, W = 12, 16
    features = torch.randn(D, H, W)
    out = ext.compute_similarity(features, ["tree", "ground"])
    assert out.shape == (2, H, W)


def test_score_queries_image_map_unchanged():
    ext = _FakeExtractor()
    H, W = 12, 16
    features = torch.randn(D, H, W)
    scores = ext.score_queries(features, positive=["tree"], negative=["background"])
    assert scores.shape == (H, W)
