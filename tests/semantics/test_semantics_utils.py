"""Tests for collab_splats.semantics.utils — contrastive scoring, point clustering, package surface."""

import numpy as np
import pytest
import torch

import collab_splats.semantics as semantics
import collab_splats.semantics.store as store
import collab_splats.semantics.utils as su
from collab_splats.semantics.utils import cluster_points, compute_semantic_contrast


def test_max_contrast_shape():
    raw = torch.randn(5, 100)
    result = compute_semantic_contrast(
        raw, num_positive=2, temperature=0.05, reduction="max"
    )
    assert result.shape == (100,)


def test_max_contrast_bounds():
    raw = torch.randn(3, 50)
    result = compute_semantic_contrast(
        raw, num_positive=2, temperature=0.05, reduction="max"
    )
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_pool_contrast_shape():
    raw = torch.randn(4, 100)
    result = compute_semantic_contrast(
        raw, num_positive=2, temperature=0.05, reduction="pool"
    )
    assert result.shape == (100,)


def test_pool_contrast_bounds():
    raw = torch.randn(4, 50)
    result = compute_semantic_contrast(
        raw, num_positive=2, temperature=0.05, reduction="pool"
    )
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_unknown_reduction_raises():
    raw = torch.randn(3, 50)
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(
            raw, num_positive=1, temperature=0.05, reduction="invalid"
        )


def test_unknown_reduction_raises_even_without_negatives():
    """Unknown reduction raises regardless of whether negatives are present."""
    sims = torch.randn(2, 10)  # 2 positives, no negatives
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(
            sims, num_positive=2, temperature=0.05, reduction="bad"
        )


def test_no_negatives_max_returns_raw_max():
    """No negatives: falls back to max over positives, no softmax."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])  # 2 positives, 2 patches
    result = compute_semantic_contrast(
        sims, num_positive=2, temperature=0.05, reduction="max"
    )
    assert torch.allclose(result, sims.max(dim=0).values)


def test_no_negatives_pool_returns_raw_mean():
    """No negatives: falls back to mean over positives."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])
    result = compute_semantic_contrast(
        sims, num_positive=2, temperature=0.05, reduction="pool"
    )
    assert torch.allclose(result, sims.mean(dim=0))


def test_max_single_positive_wins_on_high_sim_patch():
    """Patch with high positive similarity and low negative similarity scores > 0.5."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])  # (pos, neg) x 2 patches
    result = compute_semantic_contrast(
        sims, num_positive=1, temperature=0.05, reduction="max"
    )
    assert result[0] > 0.5  # patch 0: positive dominates
    assert result[1] < 0.5  # patch 1: negative dominates


def test_max_not_inflated_by_multiple_positives():
    """Adding synonym positives should not drastically inflate or deflate score."""
    # 1 positive, 1 negative
    sims_1pos = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    # 3 synonym positives, same negative — patch 0 should still score high
    sims_3pos = torch.tensor([[0.9, 0.1], [0.85, 0.05], [0.88, 0.08], [-0.1, 0.9]])
    r1 = compute_semantic_contrast(
        sims_1pos, num_positive=1, temperature=0.05, reduction="max"
    )
    r3 = compute_semantic_contrast(
        sims_3pos, num_positive=3, temperature=0.05, reduction="max"
    )
    assert abs(r1[0].item() - r3[0].item()) < 0.15


def test_pool_single_pos_matches_max():
    """With one positive, pool and max should return identical results."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    r_max = compute_semantic_contrast(
        sims, num_positive=1, temperature=0.05, reduction="max"
    )
    r_pool = compute_semantic_contrast(
        sims, num_positive=1, temperature=0.05, reduction="pool"
    )
    assert torch.allclose(r_max, r_pool)


########################################################################
# Surviving module surface: what the shim removal took with it
########################################################################


def test_utils_no_longer_re_exports_torch_helpers():
    """
    The back-compat shim is gone — torch helpers come from collab_splats.utils.torch_utils.

    - `batch_iterator` is checked against __all__ only: the re-export must not come back
    """
    for name in (
        "get_device",
        "pytorch_gc",
        "infer_batch_size",
        "load_hf_weights",
        "load_torchhub_model",
        "interpolate_to_patch_size",
    ):
        assert not hasattr(su, name), f"semantics.utils still exposes {name}"

    assert "batch_iterator" not in su.__all__, (
        "batch_iterator is an internal dependency, not public surface"
    )


def test_tokens_to_feature_map_is_private():
    """Only the three feature backends call it; it is not user surface."""
    assert "tokens_to_feature_map" not in su.__all__
    assert not hasattr(su, "tokens_to_feature_map")
    assert callable(su._tokens_to_feature_map)


def test_package_re_exports_every_utils_public_name():
    """
    Every name in utils/store __all__ is reachable as `from collab_splats.semantics import <name>`.

    - utils.__all__ is the module surface; this is the package surface, and they drifted once.
    - The two halves of a re-export fail independently: a name dropped from the `.utils` import
      is listed but unbound, a name dropped from the package __all__ is bound but not public.
    - The second sweep covers the rest of the package the same way, so `import *` cannot
      raise AttributeError on a name __all__ promises.
    """
    missing = [
        n
        for module in (su, store)
        for n in module.__all__
        if n not in semantics.__all__ or not hasattr(semantics, n)
    ]
    assert not missing, f"semantics package does not re-export: {missing}"

    unbound = [n for n in semantics.__all__ if not hasattr(semantics, n)]
    assert not unbound, (
        f"semantics.__all__ lists names the package never binds: {unbound}"
    )


def test_package_exports_segmentation_backends():
    for name in ("INSID3Segmentation", "SkyWaterSegmentation", "sky_masks"):
        assert name in semantics.__all__ and hasattr(semantics, name), name
    for name in ("load_mobile_sam", "tokens_to_feature_map"):
        assert name not in semantics.__all__ and not hasattr(semantics, name), name


########################################################################
# cluster_points
########################################################################


def _two_blobs():
    """60 points in two tight blobs 1 unit apart plus 10 scattered low-similarity points."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.005, (30, 3))
    b = rng.normal(0.0, 0.005, (30, 3)) + [1.0, 0.0, 0.0]
    stray = rng.random((10, 3)) * [0.5, 1.0, 1.0] + [0.25, 0.0, 0.0]
    points = np.vstack([a, b, stray])
    similarity = np.r_[np.ones(60), np.zeros(10)]
    return points, similarity


def test_cluster_points_groups_nearby_high_similarity_points():
    points, similarity = _two_blobs()
    clusters = cluster_points(
        points,
        similarity,
        similarity_threshold=0.5,
        spatial_radius=0.05,
        min_cluster_size=10,
    )
    assert len(clusters) == 2
    assert {frozenset(c.tolist()) for c in clusters} == {
        frozenset(range(30)),
        frozenset(range(30, 60)),
    }


def test_cluster_points_min_cluster_size_drops_small_clusters():
    points, similarity = _two_blobs()
    assert (
        cluster_points(
            points,
            similarity,
            similarity_threshold=0.5,
            spatial_radius=0.05,
            min_cluster_size=31,
        )
        == []
    )


def test_cluster_points_no_valid_points_returns_empty_list():
    points, similarity = _two_blobs()
    assert cluster_points(points, np.zeros_like(similarity)) == []
