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
    seg._fallback_quantile = 0.9
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


def test_set_context_empty_mask_raises_and_keeps_no_context():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    with pytest.raises(ValueError, match="no True pixel"):
        seg.set_context(ref_image, torch.zeros(48, 48, dtype=torch.bool))
    assert seg._prototype is None


def test_set_context_empty_mask_raises_before_extraction():
    """An empty mask is rejected before the backbone runs, so no forward pass is wasted."""
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="no True pixel"):
        seg.set_context(ref_image, np.zeros((48, 48), dtype=bool))
    extractor.forward.assert_not_called()


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


def test_segment_with_mask_clears_context_when_segment_raises():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.segment = MagicMock(side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        seg.segment_with_mask(ref_image, ref_image, ref_mask)
    assert seg._prototype is None


def test_agglomerative_labels_cover_every_cluster():
    """Cluster labels are compact: every id in 0..K-1 has a member."""
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(40, 16), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    K = int(labels.max()) + 1
    assert torch.equal(labels.unique(), torch.arange(K))


def test_locate_candidates_fallback_quantile():
    """With no positive-similarity patch, the fallback keeps the top (1 - q) share."""
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    D = 4
    prototype = F.normalize(torch.ones(D), dim=0)
    tgt = -F.normalize(torch.rand(D, 4, 5) + 0.1, dim=0)
    ref = F.normalize(torch.rand(D, 2, 2), dim=0)
    mask = torch.ones(2, 2, dtype=torch.bool)
    few = _locate_candidates(tgt, ref, mask, prototype, fallback_quantile=0.9).sum()
    many = _locate_candidates(tgt, ref, mask, prototype, fallback_quantile=0.5).sum()
    assert 0 < few < many


@pytest.mark.parametrize("device", [None, "cpu"])
def test_insid3_forwards_device_to_extractor(monkeypatch, device):
    """INSID3 passes device through untouched; DINOFeatureExtractor resolves None itself."""
    from collab_splats.semantics.segmentation import insid3
    seen = {}
    monkeypatch.setattr(insid3, "DINOFeatureExtractor", lambda **kw: seen.update(kw))
    insid3.INSID3Segmentation(device=device)
    assert seen["device"] == device


def test_segment_forwards_fallback_quantile(monkeypatch):
    from collab_splats.semantics.segmentation import insid3
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    seg._fallback_quantile = 0.5
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)

    # Stub records its kwargs and returns no candidates on the 6x6 patch grid
    seen = {}

    def fake_locate(*args, **kwargs):
        seen.update(kwargs)
        return torch.zeros(6, 6, dtype=torch.bool)

    monkeypatch.setattr(insid3, "_locate_candidates", fake_locate)
    seg.segment(ref_image)
    assert seen["fallback_quantile"] == 0.5
