"""Tests for collab_splats.semantics.store — the 2D patch cache and the lifted per-point store."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import zarr
from PIL import Image

import collab_splats.semantics.store as store
from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.store import read_point_features, write_point_features

########################################################################
# 2D patch cache: an unreadable store re-extracts, a bug propagates
########################################################################


def _one_frame_scene(tmp_path: Path, n: int = 1) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    for i in range(n):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / f"frame_{i:06d}.png")
    return images


def _fake_extractor(n_calls_before_crash: int | None = None) -> MagicMock:
    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    calls = []

    def forward(frames):
        calls.append(1)
        if n_calls_before_crash is not None and len(calls) > n_calls_before_crash:
            raise RuntimeError("crash mid-extraction")
        return [torch.full((3, 2, 2), 0.5) for _ in frames]

    extractor.forward.side_effect = forward
    return extractor


def test_extract_feature_cache_propagates_unexpected_errors(tmp_path, monkeypatch):
    """A bug inside the validity check must surface, not trigger a silent re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    (tmp_path / "fake.zarr").mkdir()

    def boom(*args, **kwargs):
        raise RuntimeError("not a store error")

    monkeypatch.setattr(store.zarr, "open", boom)
    extractor = MagicMock()
    extractor.name = "fake"
    with pytest.raises(RuntimeError, match="not a store error"):
        store.extract_feature_cache(extractor, images, tmp_path)


def test_extract_feature_cache_reextracts_corrupt_store(tmp_path):
    """A store with unreadable metadata is an expected stale cache: re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    store_path = tmp_path / "fake.zarr"
    store_path.mkdir()
    (store_path / "zarr.json").write_text("{not json")

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    store.extract_feature_cache(extractor, images, tmp_path)
    assert zarr.open(str(store_path), mode="r").attrs["n_frames"] == 1


def test_extract_feature_cache_hands_the_extractor_rgb_arrays(tmp_path):
    """Frames reach forward() as decoded RGB ndarrays, not lazily-decoded PIL handles."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((4, 4, 3), (200, 10, 10), dtype=np.uint8)).save(images / "frame_000000.png")

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    store.extract_feature_cache(extractor, images, tmp_path)

    [frame] = extractor.forward.call_args.args[0]
    assert isinstance(frame, np.ndarray)
    np.testing.assert_array_equal(frame[0, 0], [200, 10, 10])


def test_extract_feature_cache_reextracts_on_frame_count_mismatch(tmp_path):
    """A store whose n_frames disagrees with the directory is stale: re-extract."""
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images / "frame_000000.png")
    stale = zarr.open(str(tmp_path / "fake.zarr"), mode="w")
    stale.attrs.update({"extractor": "fake", "n_frames": 7})

    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    extractor.forward.return_value = [torch.zeros(3, 2, 2)]
    store.extract_feature_cache(extractor, images, tmp_path)

    assert extractor.forward.called
    assert zarr.open(str(tmp_path / "fake.zarr"), mode="r").attrs["n_frames"] == 1


def test_extract_feature_cache_writes_float16(tmp_path):
    images = _one_frame_scene(tmp_path)
    path = store.extract_feature_cache(_fake_extractor(), images, tmp_path)
    arr = zarr.open(str(path), mode="r")["features"]
    assert arr.dtype == np.float16
    np.testing.assert_array_equal(np.asarray(arr[0]), np.full((3, 2, 2), 0.5, np.float16))


def test_extract_feature_cache_valid_hit_skips_extraction(tmp_path):
    images = _one_frame_scene(tmp_path)
    store.extract_feature_cache(_fake_extractor(), images, tmp_path, {"layer": 17})
    again = _fake_extractor()
    store.extract_feature_cache(again, images, tmp_path, {"layer": 17})
    assert not again.forward.called


def test_extract_feature_cache_overwrite_reextracts_a_valid_store(tmp_path):
    images = _one_frame_scene(tmp_path)
    store.extract_feature_cache(_fake_extractor(), images, tmp_path, {"layer": 17})
    again = _fake_extractor()
    store.extract_feature_cache(again, images, tmp_path, {"layer": 17}, overwrite=True)
    assert again.forward.called


def test_extract_feature_cache_changed_kwargs_reextract(tmp_path):
    images = _one_frame_scene(tmp_path)
    store.extract_feature_cache(_fake_extractor(), images, tmp_path, {"layer": 17})
    other_layer = _fake_extractor()
    path = store.extract_feature_cache(other_layer, images, tmp_path, {"layer": 20})
    assert other_layer.forward.called
    assert zarr.open(str(path), mode="r").attrs["extractor_kwargs"] == {"layer": 20}


def test_extract_feature_cache_crash_leaves_invalid_store(tmp_path):
    images = _one_frame_scene(tmp_path, n=2)
    with pytest.raises(RuntimeError, match="crash mid-extraction"):
        store.extract_feature_cache(_fake_extractor(n_calls_before_crash=1), images, tmp_path, batch_size=1)
    assert (tmp_path / "fake.zarr").exists()
    assert store.valid_feature_cache(tmp_path, "fake", images) is None


def test_extract_feature_cache_batches_keep_frame_order(tmp_path):
    """Batched forward: ragged last batch, every frame lands in its own row, in order."""
    images = _one_frame_scene(tmp_path, n=5)
    extractor = MagicMock()
    extractor.name = "fake"
    extractor.patch_size = 2
    counter = iter(range(5))
    extractor.forward.side_effect = lambda frames: [torch.full((3, 2, 2), float(next(counter))) for _ in frames]

    path = store.extract_feature_cache(extractor, images, tmp_path, batch_size=2)

    sizes = [len(call.args[0]) for call in extractor.forward.call_args_list]
    assert sizes == [2, 2, 1]
    features = zarr.open(str(path), mode="r")["features"][:]
    np.testing.assert_array_equal(features[:, 0, 0, 0], [0, 1, 2, 3, 4])


def test_valid_feature_cache_returns_path_on_match(tmp_path):
    images = _one_frame_scene(tmp_path)
    path = store.extract_feature_cache(_fake_extractor(), images, tmp_path, {"layer": 17})
    assert store.valid_feature_cache(tmp_path, "fake", images, {"layer": 17}) == path
    assert store.valid_feature_cache(tmp_path, "fake", images, {"layer": 18}) is None
    assert store.valid_feature_cache(tmp_path, "other", images, {"layer": 17}) is None


def test_valid_feature_cache_rejects_legacy_store_without_kwargs_attr(tmp_path):
    images = _one_frame_scene(tmp_path)
    group = zarr.open(str(tmp_path / "fake.zarr"), mode="w")
    group.attrs.update({"extractor": "fake", "n_frames": 1})
    assert store.valid_feature_cache(tmp_path, "fake", images) is None


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
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


def test_write_point_features_replaces_the_store_whole(tmp_path):
    """An uncompressed rewrite leaves no earlier autoencoder.pt to decode full-dim codes."""
    store_path = tmp_path / "talk2dino_lifted.zarr"
    _write_lifted(store_path, n_points=6, latent=2, input_dim=5)
    feats = np.random.default_rng(1).random((6, 5), dtype=np.float32)
    write_point_features(store_path, feats, None)

    assert not (store_path / "autoencoder.pt").exists()
    out = read_point_features(store_path)
    np.testing.assert_allclose(out, feats / np.linalg.norm(feats, axis=1, keepdims=True), rtol=1e-6)


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
    codes = torch.from_numpy(np.asarray(zarr.open(str(store_path), mode="r")["features"]))
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
