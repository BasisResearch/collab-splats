"""Tests for FeatureAutoencoder — encode (spatial + point) and per_point_decode."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from collab_splats.semantics.compression import FeatureAutoencoder

# Small dims so all tests run fast on CPU
INPUT_DIM = 32
LATENT_DIM = 4
N = 64
H, W = 8, 8


def _make_ae() -> FeatureAutoencoder:
    return FeatureAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)


########################################################################
# Shape — point branch
########################################################################


def test_per_point_encode_shape():
    ae = _make_ae()
    out = ae.per_point_encode(torch.randn(N, INPUT_DIM))
    assert out.shape == (N, LATENT_DIM)


def test_per_point_decode_shape():
    ae = _make_ae()
    out = ae.per_point_decode(torch.randn(N, LATENT_DIM))
    assert out.shape == (N, INPUT_DIM)


def test_iter_decode_chunks_match_one_decode_and_cast_to_the_ae_dtype():
    torch.manual_seed(0)
    ae = _make_ae()
    codes = torch.randn(N, LATENT_DIM)
    expected = ae.per_point_decode(codes).detach()

    # Odd chunk size leaves a short last chunk; half codes are cast up to the AE's float32
    chunks = list(ae.iter_decode(codes.half(), batch_size=7))

    assert [len(c) for c in chunks][:-1] == [7] * (len(chunks) - 1)
    assert all(c.dtype == torch.float32 for c in chunks)
    torch.testing.assert_close(torch.cat(chunks), ae.per_point_decode(codes.half().float()).detach())
    torch.testing.assert_close(torch.cat(chunks), expected, atol=1e-2, rtol=1e-2)


########################################################################
# Shape — image branch
########################################################################


def test_encode_spatial_shape():
    ae = _make_ae()
    out = ae.encode(torch.randn(INPUT_DIM, H, W))
    assert out.shape == (LATENT_DIM, H, W)


########################################################################
# Training
########################################################################


def test_fit_reduces_loss():
    """Cosine similarity should improve after fitting vs. random init."""
    torch.manual_seed(42)
    features = torch.randn(N, INPUT_DIM)

    # Baseline: cosine sim before training
    ae = _make_ae()
    with torch.no_grad():
        sim_before = F.cosine_similarity(ae.per_point_decode(ae.per_point_encode(features)), features).mean().item()

    ae.fit(features, epochs=30, batch_size=N, lr=1e-2)

    with torch.no_grad():
        sim_after = F.cosine_similarity(ae.per_point_decode(ae.per_point_encode(features)), features).mean().item()

    assert sim_after > sim_before, f"cosine sim did not improve: {sim_before:.3f} → {sim_after:.3f}"


def test_reconstruction_cosine_sim():
    """After fitting, mean cosine similarity > 0.50 on training data.

    Threshold relaxed: LATENT_DIM=4 is an 8x bottleneck on pure Gaussian
    noise (hardest case). Real semantic features easily exceed 0.90.
    """
    torch.manual_seed(7)
    features = torch.randn(N, INPUT_DIM)

    ae = _make_ae()
    ae.fit(features, epochs=200, batch_size=N, lr=1e-2)

    with torch.no_grad():
        recon = ae.per_point_decode(ae.per_point_encode(features))
        sim = F.cosine_similarity(recon, features).mean().item()

    assert sim > 0.50, f"cosine similarity too low: {sim:.3f}"


########################################################################
# Persistence
########################################################################


def test_save_load_roundtrip():
    """Save + load produces identical per_point_encode output."""
    torch.manual_seed(99)
    features = torch.randn(N, INPUT_DIM)

    ae = _make_ae()
    ae.fit(features, epochs=2, batch_size=N)

    x = torch.randn(N, INPUT_DIM)
    with torch.no_grad():
        expected = ae.per_point_encode(x)

    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "talk2dino_ae.pt"
        ae.save(weights)
        ae2 = FeatureAutoencoder.load(weights)

    with torch.no_grad():
        actual = ae2.per_point_encode(x)

    assert torch.allclose(expected, actual, atol=1e-6), "encode output changed after save/load"


def test_save_creates_the_parent_dir_not_a_dir_named_after_the_file():
    """save() takes a FILE path: it mkdirs the parent, never the path itself.

    The old dir+extractor signature mkdir'd whatever it was handed, so passing a filename
    silently produced a DIRECTORY of that name and the weights went inside it.
    """
    ae = _make_ae()
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "nested" / "talk2dino_ae.pt"
        ae.save(weights)
        assert weights.is_file()
        assert weights.parent.is_dir()


def test_save_restores_the_device_when_the_write_fails(monkeypatch):
    """A failed write must not strand the model on CPU — save() restores in a finally."""
    # Discriminating on a CUDA box; on CPU it still pins "the device is what it was"
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    ae = _make_ae().to(device)

    # Fail the write itself, after save() has already moved the model to CPU to serialize it
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(OSError, match="disk full"):
            ae.save(Path(tmp) / "talk2dino_ae.pt")

    # The caller's model is still on its own device, and still usable there
    assert next(ae.parameters()).device == device
    ae.per_point_encode(torch.randn(N, INPUT_DIM, device=device))


########################################################################
# Architecture
########################################################################


def test_hidden_width_is_derived_from_latent_dim():
    """Width is max(64, 2 * latent_dim) — the shape on-disk checkpoints were written with."""
    assert FeatureAutoencoder(input_dim=INPUT_DIM, latent_dim=8).encoder[0].out_features == 64
    assert FeatureAutoencoder(input_dim=INPUT_DIM, latent_dim=64).encoder[0].out_features == 128
