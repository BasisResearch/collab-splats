"""Tests for collab_splats.semantics.utils — contrastive scoring and the on-disk artifact pair."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

import collab_splats.semantics as semantics
import collab_splats.semantics.utils as su
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.utils import (
    ae_path,
    cache_store_path,
    compute_semantic_contrast,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    write_point_features,
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


########################################################################
# Surviving module surface: what the shim removal took with it
########################################################################


def test_utils_no_longer_re_exports_torch_helpers():
    """The back-compat shim is gone — torch helpers come from collab_splats.utils.torch_utils.

    `batch_iterator` is deliberately absent from this list: load_point_features streams its
    decode through it, so a top-level import legitimately binds the name. What must not come
    back is the RE-EXPORT — hence the __all__ check below, which is the actual public surface.
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

    assert "batch_iterator" not in su.__all__, "batch_iterator is an internal dependency, not public surface"


def test_tokens_to_feature_map_is_public():
    """Three modules import it — the leading underscore was a lie about its visibility.

    Asserting on `__all__` rather than on an import: three other modules already import the
    name at module scope, so a revert of the rename breaks collection elsewhere long before
    an import-based check here could report. `__all__` is the public surface itself.
    """
    assert "tokens_to_feature_map" in su.__all__, "tokens_to_feature_map dropped from the public surface"
    assert callable(su.tokens_to_feature_map)


def test_package_re_exports_every_utils_public_name():
    """
    Every name in utils.__all__ is reachable as `from collab_splats.semantics import <name>`.

    - utils.__all__ is the module surface; this is the package surface, and they drifted once.
    - The two halves of a re-export fail independently: a name dropped from the `.utils` import
      is listed but unbound, a name dropped from the package __all__ is bound but not public.
    - The second sweep covers the rest of the package the same way, so `import *` cannot
      raise AttributeError on a name __all__ promises.
    """
    missing = [n for n in su.__all__ if n not in semantics.__all__ or not hasattr(semantics, n)]
    assert not missing, f"semantics package does not re-export: {missing}"

    unbound = [n for n in semantics.__all__ if not hasattr(semantics, n)]
    assert not unbound, f"semantics.__all__ lists names the package never binds: {unbound}"


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


########################################################################
# Per-point pair: what load_point_features accepts and what it refuses
########################################################################


def _write_semantics(semantics_dir, n_points=32, latent=8, input_dim=32, extractor="talk2dino"):
    """Write the semantics artifact pair a scene dir is expected to carry.

    latent MUST stay != input_dim: with equal widths latent codes and decoded features have
    the same shape and a reader that skips the decode is indistinguishable from a correct one.
    """
    assert latent != input_dim, "fixture must not hide a latent/decoded mixup behind equal widths"
    torch.manual_seed(0)
    codes = np.random.default_rng(0).random((n_points, latent), dtype=np.float32)
    write_point_features(semantics_dir, extractor, codes, FeatureAutoencoder(input_dim=input_dim, latent_dim=latent))


def test_load_point_features_full_dim_pair_needs_no_weights(tmp_path):
    """semantics.n_components: null writes full-dim codes and no autoencoder — still readable."""
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(tmp_path, "talk2dino", feats)  # ae=None -> uncompressed

    assert not (tmp_path / "talk2dino_ae.pt").exists()
    assert point_features_cached(tmp_path)
    out = load_point_features(tmp_path)
    # Returned as-is apart from the L2 normalization every consumer expects
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


def test_load_point_features_rejects_latent_codes_without_weights(tmp_path):
    """Half-written pair (codes, no weights): raise, never hand back undecoded codes."""
    _write_semantics(tmp_path, n_points=4, latent=2, input_dim=5)
    (tmp_path / "talk2dino_ae.pt").unlink()  # crash between the two writes

    assert not point_features_cached(tmp_path)  # -> caller re-lifts instead of getting stuck
    with pytest.raises(FileNotFoundError, match="re-lift"):
        load_point_features(tmp_path)


def test_load_point_features_rejects_legacy_codes_without_weights(tmp_path):
    """Pre-attrs scenes with no weights are indistinguishable from orphans — raise, don't guess."""
    store = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    store["features"] = np.zeros((4, 2), dtype=np.float32)  # no dim attrs

    assert not point_features_cached(tmp_path)
    with pytest.raises(FileNotFoundError, match="re-lift"):
        load_point_features(tmp_path)


def test_point_features_cached_reports_not_cached_on_a_null_latent_dim(tmp_path):
    """A non-int latent_dim is an unusable store, not a half-readable one — report NOT cached.

    `int(None)` raises inside the width comparison; the predicate must swallow that and say
    False so the caller re-lifts, rather than propagating into the dashboard's UI-state check.
    """
    store = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    store["features"] = np.zeros((4, 8), dtype=np.float32)
    store.attrs.update({"input_dim": 8, "latent_dim": None})

    assert not (tmp_path / "talk2dino_ae.pt").exists()
    assert not point_features_cached(tmp_path)


def test_load_point_features_decodes_in_batches_matching_the_unbatched_result(tmp_path, monkeypatch):
    """Batched decode must be numerically identical to the one-shot decode it replaces."""
    sem_dir = tmp_path / "semantics"
    _write_semantics(sem_dir, n_points=37, latent=8, input_dim=32)

    # Reference: decode every code in ONE call through the same weights.
    codes = torch.from_numpy(np.asarray(zarr.open(str(sem_dir / "talk2dino_lifted.zarr"), mode="r")["features"]))
    ae = FeatureAutoencoder.load(ae_path(sem_dir, "talk2dino"))
    with torch.no_grad():
        expected = torch.nn.functional.normalize(ae.per_point_decode(codes), dim=1).numpy()

    # Shrink the batch so 37 points genuinely span several calls, and count them.
    sizes = []
    real_decode = FeatureAutoencoder.per_point_decode

    def counting_decode(self, x):
        sizes.append(len(x))
        return real_decode(self, x)

    monkeypatch.setattr(FeatureAutoencoder, "per_point_decode", counting_decode)
    out = load_point_features(sem_dir, batch_size=5)

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert len(sizes) == 8 and max(sizes) <= 5  # 37 = 7*5 + 2; never the whole array at once
