"""Both lifting paths must land on semantics/features.zarr + semantics/autoencoder.pt."""

import numpy as np
import pytest
import torch
import zarr

from collab_splats.semantics.compression import FeatureAutoencoder, write_point_features


def test_load_features_and_decode_round_trip(tmp_path):
    """features.zarr holds latent codes; autoencoder.pt decodes them back to input_dim."""
    torch.manual_seed(0)
    latent = torch.randn(64, 8)

    store = zarr.open(str(tmp_path / "features.zarr"), mode="w")
    store["features"] = latent.numpy()

    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.save(tmp_path)

    assert (tmp_path / "autoencoder.pt").is_file()

    codes = np.asarray(zarr.open(str(tmp_path / "features.zarr"), mode="r")["features"])
    assert codes.shape == (64, 8)

    decoded = FeatureAutoencoder.load(tmp_path).per_point_decode(torch.from_numpy(codes))
    assert decoded.shape == (64, 32)


def test_write_point_features_leaves_no_orphan_codes_when_weights_fail(tmp_path, monkeypatch):
    """A failed weight save must take features.zarr with it — orphaned codes are unreadable.

    Codes on disk without their autoencoder look cached to every consumer while decoding to
    nothing, which is how a scene gets permanently stuck rather than simply re-lifting.
    """
    def boom(self, path):
        raise OSError("disk full")

    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    monkeypatch.setattr(FeatureAutoencoder, "save", boom)

    with pytest.raises(OSError):
        write_point_features(tmp_path, np.zeros((4, 8), dtype=np.float32), ae)

    assert not (tmp_path / "features.zarr").exists()
    assert not (tmp_path / "autoencoder.pt").exists()
