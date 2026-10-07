"""Tests for collab_splats.semantics.store — the 2D patch cache and the lifted per-point store."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

import collab_splats.semantics.store as store
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.store import read_point_features, write_point_features

########################################################################
# 2D patch cache: an unreadable store re-extracts, a bug propagates
########################################################################


def _frames_dir(tmp_path: Path, n: int) -> Path:
    images = tmp_path / "images"
    images.mkdir()

    for i in range(n):
        (images / f"frame_{i:06d}.png").touch()

    return images


def _maps(n: int, dim: int = 3) -> list[torch.Tensor]:
    return [torch.full((dim, 2, 2), float(i)) for i in range(n)]


ATTRS = {"extractor": "fake", "patch_size": 2, "n_frames": 2, "extractor_kwargs": {"layer": 17}, "latent_dim": 3}


def test_write_feature_cache_writes_fp16_one_chunk_per_frame(tmp_path):
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    arr = zarr.open(str(path), mode="r")["features"]
    assert arr.dtype == np.float16
    assert arr.shape == (2, 3, 2, 2) and arr.chunks == (1, 3, 2, 2)
    assert float(arr[1, 0, 0, 0]) == 1.0
    assert dict(zarr.open(str(path), mode="r").attrs) == ATTRS


def test_write_feature_cache_saves_the_autoencoder_before_the_attrs(tmp_path, monkeypatch):
    """A store whose AE never landed must read invalid."""
    path = tmp_path / "fake_codes.zarr"

    def boom(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS, ae=FeatureAutoencoder(8, 3))

    assert "extractor" not in zarr.open(str(path), mode="r").attrs


def test_write_feature_cache_crash_mid_frames_reads_invalid(tmp_path):
    images = _frames_dir(tmp_path, 2)
    path = tmp_path / "fake_codes.zarr"

    def crashing():
        yield _maps(1)[0]
        raise RuntimeError("crash mid-extraction")

    with pytest.raises(RuntimeError):
        store.write_feature_cache(path, crashing(), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) is None


def test_write_feature_cache_rejects_a_short_frame_stream(tmp_path):
    with pytest.raises(ValueError, match="1 of 2"):
        store.write_feature_cache(tmp_path / "fake_codes.zarr", iter(_maps(1)), 2, ATTRS)


def test_write_feature_cache_rejects_a_long_frame_stream(tmp_path):
    with pytest.raises(ValueError, match="more than 1"):
        store.write_feature_cache(tmp_path / "fake_codes.zarr", iter(_maps(2)), 1, ATTRS)


def test_write_feature_cache_rejects_zero_frames(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        store.write_feature_cache(tmp_path / "fake_codes.zarr", iter([]), 0, ATTRS)


def test_valid_feature_cache_checks_name_kwargs_frames_and_latent_dim(tmp_path):
    images = _frames_dir(tmp_path, 2)
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) == path
    assert store.valid_feature_cache(path, "other", images, {"layer": 17}, 3) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 18}, 3) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 8) is None
    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, None) is None


def test_valid_feature_cache_rejects_a_frame_count_mismatch(tmp_path):
    images = _frames_dir(tmp_path, 3)
    path = tmp_path / "fake_codes.zarr"
    store.write_feature_cache(path, iter(_maps(2)), 2, ATTRS)

    assert store.valid_feature_cache(path, "fake", images, {"layer": 17}, 3) is None


def test_valid_feature_cache_propagates_unexpected_errors(tmp_path, monkeypatch):
    """A bug inside the validity check must surface, not read as a miss."""
    images = _frames_dir(tmp_path, 1)
    (tmp_path / "fake_codes.zarr").mkdir()

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(store.zarr, "open", boom)

    with pytest.raises(RuntimeError, match="not a store error"):
        store.valid_feature_cache(tmp_path / "fake_codes.zarr", "fake", images, {}, None)


########################################################################
# Lifted store: AE inside, atomic write, decode
########################################################################


def _write_lifted(store_path, n_points=32, latent=8, input_dim=32):
    """
    Write a compressed lifted store.

    - latent != input_dim: equal widths would hide a reader that skips the decode
    """
    torch.manual_seed(0)
    codes = np.random.default_rng(0).random((n_points, latent), dtype=np.float32)
    write_point_features(store_path, codes, FeatureAutoencoder(input_dim=input_dim, latent_dim=latent))


def test_write_point_features_puts_the_autoencoder_inside_the_store(tmp_path):
    store_path = tmp_path / "dinov2_lifted.zarr"
    _write_lifted(store_path)

    assert (store_path / "autoencoder.pt").is_file()
    assert dict(zarr.open(str(store_path), mode="r").attrs) == {"input_dim": 32, "latent_dim": 8}


def test_read_point_features_full_dim_store_needs_no_weights(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert not (store_path / "autoencoder.pt").exists()
    out = read_point_features(store_path)
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-3)


def test_write_point_features_replaces_the_store_whole(tmp_path):
    """An uncompressed rewrite leaves no earlier autoencoder.pt to decode full-dim codes."""
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=6, latent=2, input_dim=5)
    feats = np.random.default_rng(1).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert not (store_path / "autoencoder.pt").exists()
    out = read_point_features(store_path)
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-3)


def test_write_point_features_killed_before_rename_leaves_no_store(tmp_path, monkeypatch):
    """A write that dies before the rename leaves only the tmp dir, so the stage is not done."""
    store_path = tmp_path / "talk2dino_lifted.zarr"

    def killed(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "rename", killed)

    with pytest.raises(KeyboardInterrupt):
        _write_lifted(store_path)

    assert not store_path.exists()
    assert (tmp_path / "talk2dino_lifted.zarr.tmp").exists()


def test_write_point_features_leaves_no_store_when_weights_fail(tmp_path, monkeypatch):
    """A failed weight save removes the tmp dir and never touches the store path."""
    store_path = tmp_path / "talk2dino_lifted.zarr"

    def boom(self, path):
        raise OSError("disk full")

    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        _write_lifted(store_path)

    assert not store_path.exists()
    assert not (tmp_path / "talk2dino_lifted.zarr.tmp").exists()


def test_read_point_features_rejects_latent_codes_without_weights(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=4, latent=2, input_dim=5)
    (store_path / "autoencoder.pt").unlink()

    with pytest.raises(FileNotFoundError, match="autoencoder.pt"):
        read_point_features(store_path)


def test_read_point_features_decodes_in_batches_matching_the_unbatched_result(tmp_path, monkeypatch):
    """Batched decode must be numerically identical to a one-shot decode."""
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=37, latent=8, input_dim=32)

    # Reference: decode every code in one call through the same weights
    codes = torch.from_numpy(np.asarray(zarr.open(str(store_path), mode="r")["features"], dtype=np.float32))
    ae = FeatureAutoencoder.load(store_path / "autoencoder.pt")

    with torch.no_grad():
        expected = torch.nn.functional.normalize(ae.per_point_decode(codes), dim=1).numpy()

    # Shrink the batch so 37 points span several calls, and count them
    sizes = []
    real_decode = FeatureAutoencoder.per_point_decode

    def counting_decode(self, x):
        sizes.append(len(x))
        return real_decode(self, x)

    monkeypatch.setattr(FeatureAutoencoder, "per_point_decode", counting_decode)
    out = read_point_features(store_path, batch_size=5)

    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert len(sizes) == 8 and max(sizes) <= 5  # 37 = 7*5 + 2; never the whole array at once


def test_write_point_features_stores_fp16_and_reads_float32_unit_rows(tmp_path):
    store_path = tmp_path / "talk2dino_lifted.zarr"
    feats = np.random.default_rng(0).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert zarr.open(str(store_path), mode="r")["features"].dtype == np.float16
    out = read_point_features(store_path)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
