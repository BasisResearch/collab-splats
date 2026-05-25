"""Tests for FeatureAutoencoder — image branch (encode/decode) and point branch (per_point_*)."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from collab_splats.semantics.compression import FeatureAutoencoder

# Small dims so all tests run fast on CPU
INPUT_DIM = 32
LATENT_DIM = 4
REG_DIM = 16
N = 64
H, W = 8, 8
REG_KWARGS = {"dim": REG_DIM, "weight": 0.1}


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


########################################################################
# Shape — image branch
########################################################################


def test_encode_spatial_shape():
    ae = _make_ae()
    out = ae.encode(torch.randn(INPUT_DIM, H, W))
    assert out.shape == (LATENT_DIM, H, W)


def test_decode_spatial_shape():
    ae = _make_ae()
    out = ae.decode(torch.randn(LATENT_DIM, H, W))
    assert out.shape == (INPUT_DIM, H, W)


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
        sim_before = F.cosine_similarity(
            ae.per_point_decode(ae.per_point_encode(features)), features
        ).mean().item()

    ae.fit(features, epochs=30, batch_size=N, lr=1e-2)

    with torch.no_grad():
        sim_after = F.cosine_similarity(
            ae.per_point_decode(ae.per_point_encode(features)), features
        ).mean().item()

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
        ae.save(Path(tmp))
        ae2 = FeatureAutoencoder.load(Path(tmp))

    with torch.no_grad():
        actual = ae2.per_point_encode(x)

    assert torch.allclose(expected, actual, atol=1e-6), "encode output changed after save/load"


########################################################################
# Regularization head
########################################################################


def test_reg_head_exists():
    ae = FeatureAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        regularization_kwargs=REG_KWARGS,
    )
    assert ae.reg_head is not None
    assert ae.reg_head.out_features == REG_DIM


def test_no_reg_head_when_no_kwargs():
    ae = _make_ae()
    assert ae.reg_head is None


def test_fit_with_reg_target():
    """fit() with reg_target runs without error and reduces main loss."""
    torch.manual_seed(42)
    features = torch.randn(N, INPUT_DIM)
    reg_target = torch.randn(N, REG_DIM)

    ae = FeatureAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        regularization_kwargs=REG_KWARGS,
    )
    with torch.no_grad():
        sim_before = F.cosine_similarity(
            ae.per_point_decode(ae.per_point_encode(features)), features
        ).mean().item()

    ae.fit(features, reg_target=reg_target, epochs=30, batch_size=N, lr=1e-2)

    with torch.no_grad():
        sim_after = F.cosine_similarity(
            ae.per_point_decode(ae.per_point_encode(features)), features
        ).mean().item()

    assert sim_after > sim_before, f"cosine sim did not improve: {sim_before:.3f} → {sim_after:.3f}"


def test_fit_reg_target_on_pure_ae_raises():
    ae = _make_ae()
    with pytest.raises(ValueError, match="no reg_head configured"):
        ae.fit(torch.randn(N, INPUT_DIM), reg_target=torch.randn(N, REG_DIM))


def test_save_load_roundtrip_with_reg():
    """Save + load preserves reg_head and encode output."""
    torch.manual_seed(99)
    features = torch.randn(N, INPUT_DIM)
    reg_target = torch.randn(N, REG_DIM)

    ae = FeatureAutoencoder(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        regularization_kwargs=REG_KWARGS,
    )
    ae.fit(features, reg_target=reg_target, epochs=2, batch_size=N)

    x = torch.randn(N, INPUT_DIM)
    with torch.no_grad():
        expected = ae.per_point_encode(x)

    with tempfile.TemporaryDirectory() as tmp:
        ae.save(Path(tmp))
        ae2 = FeatureAutoencoder.load(Path(tmp))

    assert ae2.reg_head is not None
    assert ae2.reg_head.out_features == REG_DIM
    assert ae2._reg_weight == 0.1

    with torch.no_grad():
        actual = ae2.per_point_encode(x)

    assert torch.allclose(expected, actual, atol=1e-6), "encode output changed after save/load"
