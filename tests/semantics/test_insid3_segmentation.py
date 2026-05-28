# tests/semantics/test_insid3_segmentation.py
"""Tests for INSID3 in-context segmentation backend."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def test_agglomerative_clustering_returns_integer_labels():
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(16, 32), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    assert labels.shape == (16,)
    assert labels.dtype == torch.long
    assert labels.min() >= 0


def test_agglomerative_clustering_device_preserved():
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(8, 16), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    assert labels.device == X.device


def test_cluster_prototypes_shape_and_normalized():
    from collab_splats.semantics.segmentation.insid3 import _cluster_prototypes
    X = F.normalize(torch.randn(20, 32), p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 1])
    K = 3
    protos = _cluster_prototypes(X, labels, K)
    assert protos.shape == (K, 32)
    norms = protos.norm(dim=1)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5)


def test_agglomerative_clustering_identical_vectors_merge():
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    # All identical vectors → cosine distance = 0 → any tau merges into 1 cluster
    X = F.normalize(torch.ones(8, 16), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.5)
    assert labels.shape == (8,)
    assert int(labels.max().item()) == 0  # all merged into one cluster


def test_cluster_prototypes_handles_missing_cluster():
    from collab_splats.semantics.segmentation.insid3 import _cluster_prototypes
    # K=3 but only clusters 0 and 2 exist — cluster 1 is empty
    X = F.normalize(torch.randn(10, 16), p=2, dim=1)
    labels = torch.tensor([0, 0, 2, 2, 0, 2, 0, 2, 0, 2])
    protos = _cluster_prototypes(X, labels, K=3)
    assert protos.shape == (3, 16)
    norms = protos.norm(dim=1)
    # Non-empty clusters (0 and 2) have unit norm; empty cluster (1) has norm 0
    assert torch.isclose(norms[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(norms[1], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(norms[2], torch.tensor(1.0), atol=1e-5)


def test_downsample_mask_reduces_to_patch_resolution():
    from collab_splats.semantics.segmentation.insid3 import _downsample_mask
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[20:44, 20:44] = True
    down = _downsample_mask(mask, h=8, w=8)
    assert down.shape == (8, 8)
    assert down.any(), "mask should be non-empty after downsampling"


def test_downsample_mask_tiny_mask_fallback():
    from collab_splats.semantics.segmentation.insid3 import _downsample_mask
    # Single pixel mask — bilinear would vanish at patch resolution
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[32, 32] = True
    down = _downsample_mask(mask, h=8, w=8)
    assert down.shape == (8, 8)
    assert down.sum() == 1, "fallback should produce exactly one True patch"


def test_locate_candidates_returns_bool_mask():
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    D, H, W = 16, 6, 6
    tgt = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref_mask_down = torch.zeros(H, W, dtype=torch.bool)
    ref_mask_down[1:4, 1:4] = True
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    out = _locate_candidates(tgt, ref, ref_mask_down, proto)
    assert out.shape == (H, W)
    assert out.dtype == torch.bool


def test_locate_candidates_perfect_match():
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    # When target IS the reference, backward NN of each target patch is itself → inside mask if in masked region
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref_mask_down = torch.zeros(H, W, dtype=torch.bool)
    ref_mask_down[0:3, 0:3] = True
    # prototype = mean of masked ref features
    proto = F.normalize(feat[:, ref_mask_down].mean(dim=1), p=2, dim=0)
    out = _locate_candidates(feat, feat, ref_mask_down, proto)
    assert out.shape == (H, W)
    # At least some candidates should be inside the masked region
    assert (out & ref_mask_down).any()


def test_downsample_mask_all_zero_returns_empty():
    from collab_splats.semantics.segmentation.insid3 import _downsample_mask
    mask = torch.zeros(64, 64, dtype=torch.bool)
    down = _downsample_mask(mask, h=8, w=8)
    assert down.shape == (8, 8)
    assert not down.any(), "all-zero mask should produce all-zero output without crashing"


def test_upsample_mask_basic():
    from collab_splats.semantics.segmentation.insid3 import _upsample_mask
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:5, 2:5] = True
    up = _upsample_mask(mask, H=64, W=64)
    assert up.shape == (64, 64)
    assert up.dtype == torch.bool
    assert up.any()


def test_tensor_to_pil_shape_and_mode():
    from collab_splats.semantics.segmentation.insid3 import _tensor_to_pil
    t = torch.rand(3, 32, 32)
    img = _tensor_to_pil(t)
    assert img.size == (32, 32)
    assert img.mode == "RGB"


def test_seed_and_aggregate_returns_bool_mask():
    from collab_splats.semantics.segmentation.insid3 import _seed_and_aggregate
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    feat_deb = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    candidate_mask = torch.zeros(H, W, dtype=torch.bool)
    candidate_mask[1:4, 1:4] = True
    labels = torch.zeros(H, W, dtype=torch.long)
    labels[::2, ::2] = 1
    K = 2
    out = _seed_and_aggregate(candidate_mask, feat, feat_deb, proto, labels, K, merge_threshold=0.2)
    assert out.shape == (H, W)
    assert out.dtype == torch.bool


def test_seed_and_aggregate_empty_candidate_returns_empty():
    from collab_splats.semantics.segmentation.insid3 import _seed_and_aggregate
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    feat_deb = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    candidate_mask = torch.zeros(H, W, dtype=torch.bool)  # all-zero
    labels = torch.zeros(H, W, dtype=torch.long)
    out = _seed_and_aggregate(candidate_mask, feat, feat_deb, proto, labels, K=1, merge_threshold=0.2)
    assert out.shape == (H, W)
    assert not out.any()


def test_seed_and_aggregate_missing_cluster_no_crash():
    from collab_splats.semantics.segmentation.insid3 import _seed_and_aggregate
    # K=3 but only clusters 0 and 2 present — cluster 1 is absent
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    feat_deb = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    candidate_mask = torch.zeros(H, W, dtype=torch.bool)
    candidate_mask[0:3, 0:3] = True
    labels = torch.zeros(H, W, dtype=torch.long)
    labels[3:, 3:] = 2  # only clusters 0 and 2 — cluster 1 absent
    out = _seed_and_aggregate(candidate_mask, feat, feat_deb, proto, labels, K=3, merge_threshold=0.2)
    assert out.shape == (H, W)
    assert out.dtype == torch.bool


from unittest.mock import MagicMock
from PIL import Image


def _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6):
    """Helper: INSID3Segmentation with mocked DINOFeatureExtractor."""
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    feat = F.normalize(torch.randn(D, H_p, W_p), p=2, dim=0)
    extractor = MagicMock()
    extractor.forward.return_value = [feat.clone()]
    extractor.debias.return_value = [feat.clone()]
    seg = INSID3Segmentation.__new__(INSID3Segmentation)
    seg._extractor = extractor
    seg._tau = 0.6
    seg._merge_threshold = 0.2
    seg._prototype = None
    seg._ref_feat_deb = None
    seg._ref_mask_down = None
    return seg, extractor, feat


def test_set_context_caches_prototype_and_features():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    assert seg._prototype is not None
    assert seg._ref_feat_deb is not None
    assert seg._ref_mask_down is not None
    assert seg._prototype.shape == (16,)
    assert seg._prototype.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_clear_context_resets_state():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    seg.clear_context()
    assert seg._prototype is None
    assert seg._ref_feat_deb is None
    assert seg._ref_mask_down is None


def test_set_context_accepts_numpy_mask():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = np.zeros((48, 48), dtype=bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    assert seg._prototype is not None


def test_set_context_empty_mask_does_not_set_context():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)  # all-zero mask
    seg.set_context(ref_image, ref_mask)
    # Context should NOT be set — prototype should remain None
    assert seg._prototype is None


def test_registered_as_insid3():
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    from collab_splats.semantics.segmentation.base import BaseSegmentation
    assert BaseSegmentation.get("insid3") is INSID3Segmentation


def test_segment_raises_without_context():
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    seg = INSID3Segmentation.__new__(INSID3Segmentation)
    seg._prototype = None
    seg._ref_feat_deb = None
    seg._ref_mask_down = None
    tgt = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="set_context"):
        seg.segment(tgt)


def test_segment_output_shape_matches_input():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    H, W = 48, 48
    seg._prototype = F.normalize(torch.randn(16), p=2, dim=0)
    seg._ref_feat_deb = feat.clone()
    seg._ref_mask_down = torch.zeros(6, 6, dtype=torch.bool)
    seg._ref_mask_down[1:4, 1:4] = True
    tgt = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    pred_mask, meta = seg.segment(tgt)
    assert pred_mask.shape == (H, W)
    assert pred_mask.dtype == torch.bool
    assert "candidate_mask" in meta
    assert "cluster_labels" in meta
    assert "n_clusters" in meta


def test_segment_context_reused_across_calls():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    seg._prototype = F.normalize(torch.randn(16), p=2, dim=0)
    seg._ref_feat_deb = feat.clone()
    seg._ref_mask_down = torch.zeros(6, 6, dtype=torch.bool)
    seg._ref_mask_down[1:4, 1:4] = True
    tgt = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    seg.segment(tgt)
    seg.segment(tgt)
    seg.segment(tgt)
    # forward() called 3 times (one per target); context already cached so no ref call
    assert extractor.forward.call_count == 3


def test_segment_with_mask_one_shot_clears_context():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    tgt_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = np.zeros((48, 48), dtype=bool)
    ref_mask[16:32, 16:32] = True
    pred_mask, meta = seg.segment_with_mask(tgt_image, ref_image, ref_mask)
    assert pred_mask.shape == (48, 48)
    assert pred_mask.dtype == torch.bool
    assert seg._prototype is None  # cleared after one-shot
    assert seg._ref_feat_deb is None
