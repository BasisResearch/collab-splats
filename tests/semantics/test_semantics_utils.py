"""Tests for collab_splats.semantics.utils — torch/model utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from collab_splats.semantics.utils import (
    batch_iterator,
    cache_store_path,
    compute_semantic_contrast,
    infer_batch_size,
    interpolate_to_patch_size,
    load_feature_maps,
    pytorch_gc,
)


def test_max_contrast_shape():
    raw = torch.randn(5, 100)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="max")
    assert result.shape == (100,)


def test_max_contrast_bounds():
    raw = torch.randn(3, 50)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="max")
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_pool_contrast_shape():
    raw = torch.randn(4, 100)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="pool")
    assert result.shape == (100,)


def test_pool_contrast_bounds():
    raw = torch.randn(4, 50)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="pool")
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_unknown_reduction_raises():
    raw = torch.randn(3, 50)
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(raw, num_positive=1, temperature=0.05, reduction="invalid")


def test_unknown_reduction_raises_even_without_negatives():
    """Unknown reduction raises regardless of whether negatives are present."""
    sims = torch.randn(2, 10)  # 2 positives, no negatives
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(sims, num_positive=2, temperature=0.05, reduction="bad")


def test_no_negatives_max_returns_raw_max():
    """No negatives: falls back to max over positives, no softmax."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])  # 2 positives, 2 patches
    result = compute_semantic_contrast(sims, num_positive=2, temperature=0.05, reduction="max")
    assert torch.allclose(result, sims.max(dim=0).values)


def test_no_negatives_pool_returns_raw_mean():
    """No negatives: falls back to mean over positives."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])
    result = compute_semantic_contrast(sims, num_positive=2, temperature=0.05, reduction="pool")
    assert torch.allclose(result, sims.mean(dim=0))


def test_max_single_positive_wins_on_high_sim_patch():
    """Patch with high positive similarity and low negative similarity scores > 0.5."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])  # (pos, neg) x 2 patches
    result = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="max")
    assert result[0] > 0.5  # patch 0: positive dominates
    assert result[1] < 0.5  # patch 1: negative dominates


def test_max_not_inflated_by_multiple_positives():
    """Adding synonym positives should not drastically inflate or deflate score."""
    # 1 positive, 1 negative
    sims_1pos = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    # 3 synonym positives, same negative — patch 0 should still score high
    sims_3pos = torch.tensor([[0.9, 0.1], [0.85, 0.05], [0.88, 0.08], [-0.1, 0.9]])
    r1 = compute_semantic_contrast(sims_1pos, num_positive=1, temperature=0.05, reduction="max")
    r3 = compute_semantic_contrast(sims_3pos, num_positive=3, temperature=0.05, reduction="max")
    assert abs(r1[0].item() - r3[0].item()) < 0.15


def test_pool_single_pos_matches_max():
    """With one positive, pool and max should return identical results."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    r_max = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="max")
    r_pool = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="pool")
    assert torch.allclose(r_max, r_pool)


def test_interpolate_divisible():
    img = torch.randn(1, 3, 224, 224)
    result, h, w = interpolate_to_patch_size(img, patch_size=14)
    assert h % 14 == 0
    assert w % 14 == 0
    assert result.shape == (1, 3, h, w)


def test_interpolate_non_divisible():
    img = torch.randn(1, 3, 230, 230)
    result, h, w = interpolate_to_patch_size(img, patch_size=14)
    assert h % 14 == 0
    assert w % 14 == 0   # restore this line
    assert h == 224


def test_pytorch_gc_no_error():
    pytorch_gc()


def test_infer_batch_size_cpu():
    if not torch.cuda.is_available():
        assert infer_batch_size(3.0) == 1


def test_infer_batch_size_negative_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(-1.0)


def test_infer_batch_size_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(0.0)


def test_batch_iterator_basic():
    items = list(range(10))
    batches = list(batch_iterator(3, items))
    assert len(batches) == 4
    assert batches[0] == [[0, 1, 2]]
    assert batches[-1] == [[9]]


def test_batch_iterator_multiple_args():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    batches = list(batch_iterator(2, a, b))
    assert len(batches) == 2
    assert batches[0] == [[1, 2], [5, 6]]


def test_batch_iterator_mismatched_raises():
    with pytest.raises(AssertionError):
        list(batch_iterator(2, [1, 2], [3]))


########################################################################
# On-disk layout: the 2D patch cache vs the lifted per-point store
########################################################################


def _write_both_stores(tmp_path):
    """Write a `talk2dino.zarr` 2D cache and a `talk2dino_lifted.zarr` per-point store side by side."""
    cache = zarr.open(str(tmp_path / "talk2dino.zarr"), mode="w")
    cache["features"] = np.zeros((2, 4, 3, 3), dtype=np.float32)
    cache.attrs.update({"extractor": "talk2dino", "patch_size": 14, "n_frames": 2})

    lifted = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    lifted["features"] = np.zeros((6, 4), dtype=np.float32)
    lifted.attrs.update({"input_dim": 4, "latent_dim": 4})


def _force_lifted_first(monkeypatch, tmp_path):
    """Make `Path.glob` yield the lifted store first, so suffix filtering is what does the work."""
    real_glob = Path.glob

    def ordered(self, pattern):
        return sorted(real_glob(self, pattern), key=lambda p: "_lifted" not in p.name)

    monkeypatch.setattr(Path, "glob", ordered)


def test_cache_store_path_ignores_the_lifted_store(tmp_path, monkeypatch):
    _write_both_stores(tmp_path)
    _force_lifted_first(monkeypatch, tmp_path)
    assert cache_store_path(tmp_path).name == "talk2dino.zarr"
    # Callers read the extractor name off `.stem`; "talk2dino_lifted" would be the answer
    # if the lifted store were picked up
    assert cache_store_path(tmp_path).stem == "talk2dino"


def test_cache_store_path_raises_when_there_is_no_2d_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="no 2D feature cache"):
        cache_store_path(tmp_path)


def test_load_feature_maps_takes_a_store_path(tmp_path):
    _write_both_stores(tmp_path)
    maps = load_feature_maps(cache_store_path(tmp_path))
    assert len(maps) == 2
    assert maps[0].shape == (4, 3, 3)
