"""Both lifting paths must land on <extractor>_lifted.zarr + <extractor>_ae.pt."""

import numpy as np
import pytest
import torch
import zarr

from collab_splats.semantics.compression import (
    FeatureAutoencoder,
    find_lifted_extractor,
    write_point_features,
)


def test_load_features_and_decode_round_trip(tmp_path):
    """The lifted store holds latent codes; the _ae.pt beside it decodes them to input_dim."""
    torch.manual_seed(0)
    latent = torch.randn(64, 8)

    store = zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="w")
    store["features"] = latent.numpy()

    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.save(tmp_path, "talk2dino")

    assert (tmp_path / "talk2dino_ae.pt").is_file()

    codes = np.asarray(zarr.open(str(tmp_path / "talk2dino_lifted.zarr"), mode="r")["features"])
    assert codes.shape == (64, 8)

    decoded = FeatureAutoencoder.load(tmp_path, "talk2dino").per_point_decode(torch.from_numpy(codes))
    assert decoded.shape == (64, 32)


def test_write_point_features_names_both_halves_after_the_extractor(tmp_path):
    """The extractor is in both filenames, so a second extractor cannot overwrite the first."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    write_point_features(tmp_path, "dinov2", np.zeros((4, 8), dtype=np.float32), ae)

    assert (tmp_path / "dinov2_lifted.zarr").exists()
    assert (tmp_path / "dinov2_ae.pt").is_file()


def test_find_lifted_extractor_ignores_the_2d_cache(tmp_path):
    """A dir holding both stores resolves to the lifted one — the suffix is the discriminator."""
    zarr.open(str(tmp_path / "talk2dino.zarr"), mode="w")  # 2D patch cache, same stem
    write_point_features(tmp_path, "talk2dino", np.zeros((4, 8), dtype=np.float32))

    assert find_lifted_extractor(tmp_path) == "talk2dino"


def test_find_lifted_extractor_refuses_to_guess_between_two(tmp_path):
    """Two lifted stores in one dir must raise — the wrong pick mixes codes with a foreign decoder."""
    write_point_features(tmp_path, "talk2dino", np.zeros((4, 8), dtype=np.float32))
    write_point_features(tmp_path, "dinov2", np.zeros((4, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="several lifted stores"):
        find_lifted_extractor(tmp_path)


def test_write_point_features_leaves_no_orphan_codes_when_weights_fail(tmp_path, monkeypatch):
    """A failed weight save must take the codes with it — orphaned codes are unreadable.

    Codes on disk without their autoencoder look cached to every consumer while decoding to
    nothing, which is how a scene gets permanently stuck rather than simply re-lifting.
    """

    def boom(self, path, extractor):
        raise OSError("disk full")

    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        write_point_features(tmp_path, "talk2dino", np.zeros((4, 8), dtype=np.float32), ae)

    assert not (tmp_path / "talk2dino_lifted.zarr").exists()
    assert not (tmp_path / "talk2dino_ae.pt").exists()
