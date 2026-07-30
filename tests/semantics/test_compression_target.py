"""target_cosine early-stop and persisted reconstruction metrics."""

import pytest
import torch

from collab_splats.semantics.compression import FeatureAutoencoder


def _features(n=512, d=32):
    torch.manual_seed(0)
    return torch.randn(n, d)


def _seeded_fit(epochs, target_cosine=None, on_epoch=None):
    """Bit-reproducible fit — seeds features, weight init, and fit's own randperm stream."""
    torch.manual_seed(0)
    features = torch.randn(512, 32)
    torch.manual_seed(1234)
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    torch.manual_seed(1234)
    ae.fit(features, epochs=epochs, target_cosine=target_cosine, on_epoch=on_epoch)
    return ae


def test_fit_records_metrics_on_the_model():
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.fit(_features(), epochs=2)
    assert -1.0 <= ae.recon_cosine <= 1.0
    assert ae.recon_mse >= 0.0
    assert ae.epochs_run == 2


def test_target_cosine_stops_early():
    """A trivially-reachable target must stop before the epoch ceiling."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=32)
    ae.fit(_features(), epochs=50, target_cosine=-1.0)
    assert ae.epochs_run == 1


def test_target_cosine_unreachable_runs_to_ceiling():
    ae = FeatureAutoencoder(input_dim=32, latent_dim=2)
    ae.fit(_features(), epochs=3, target_cosine=1.5)
    assert ae.epochs_run == 3


def test_target_cosine_exactly_met_stops():
    """Boundary: cosine landing exactly on the target must stop (>= not >)."""
    # Learn the exact cosine one seeded epoch produces, and confirm it is bit-reproducible
    baseline = _seeded_fit(epochs=1)
    c = baseline.recon_cosine
    assert _seeded_fit(epochs=1).recon_cosine == c

    # Same seeded run with that value as the target: epoch 1 hits it exactly, so it must stop
    ae = _seeded_fit(epochs=10, target_cosine=c)
    assert ae.recon_cosine == c
    assert ae.epochs_run == 1


def test_on_epoch_fires_on_the_early_stop_epoch():
    """The callback must report the epoch it stopped on, not be skipped by the break."""
    calls = []
    ae = FeatureAutoencoder(input_dim=32, latent_dim=32)
    ae.fit(_features(), epochs=50, target_cosine=-1.0, on_epoch=lambda e, total, loss: calls.append((e, total)))
    assert ae.epochs_run == 1
    assert calls == [(1, 50)]


def test_fit_rejects_empty_features():
    """Zero samples is a caller bug: it must raise, not report a false trained state."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    with pytest.raises(ValueError, match=r"at least one sample.*\(0, 32\)"):
        ae.fit(torch.randn(0, 32), epochs=50, target_cosine=0.0)
    assert ae.epochs_run == 0
    assert ae.recon_cosine == 0.0


def test_metrics_survive_save_load(tmp_path):
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)
    ae.fit(_features(), epochs=2)
    ae.save(tmp_path, "talk2dino")
    loaded = FeatureAutoencoder.load(tmp_path, "talk2dino")
    assert loaded.recon_cosine == ae.recon_cosine
    assert loaded.recon_mse == ae.recon_mse
    assert loaded.epochs_run == ae.epochs_run


def test_load_legacy_checkpoint_without_metrics(tmp_path):
    """Checkpoints predating the metric keys still load, defaulting to the untrained values."""
    ae = FeatureAutoencoder(input_dim=32, latent_dim=8)

    # Hand-write a pre-metrics payload: the three metric keys are simply absent
    payload = {
        "input_dim": 32,
        "latent_dim": 8,
        "hidden_dim": ae.hidden_dim,
        "regularization_kwargs": {},
        "state_dict": ae.state_dict(),
    }
    torch.save(payload, tmp_path / "talk2dino_ae.pt")

    loaded = FeatureAutoencoder.load(tmp_path, "talk2dino")
    assert loaded.recon_cosine == 0.0
    assert loaded.recon_mse == 0.0
    assert loaded.epochs_run == 0
