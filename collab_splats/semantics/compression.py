"""
collab_splats.semantics.compression — lightweight feature autoencoder for
compressing patch features before 3D lifting.

Trained once on keyframe features; used at inference time to reduce the
dimensionality of semantic features stored per point.

Mirrors the two-branch convention of TwoLayerMLP:
  encode / decode         — spatial image branch  (D, H, W) ↔ (latent_dim, H, W)
  per_point_encode/decode — flat point branch     (N, D)    ↔ (N, latent_dim)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

########################################################################
# FeatureAutoencoder
########################################################################


class FeatureAutoencoder(nn.Module):
    """Two-layer MLP autoencoder for compressing semantic patch features.

    Architecture
    ------------
    encoder: Linear(input_dim → hidden_dim) → ReLU → Linear(hidden_dim → latent_dim)
    decoder: Linear(latent_dim → hidden_dim) → ReLU → Linear(hidden_dim → input_dim)

    Two entry points (mirrors TwoLayerMLP convention):
      encode / decode         — spatial image maps  (D, H, W) ↔ (latent_dim, H, W)
      per_point_encode/decode — flat point arrays   (N, D)    ↔ (N, latent_dim)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        # Resolve hidden dimension — default to twice the latent dim, min 64
        if hidden_dim is None:
            hidden_dim = max(64, 2 * latent_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: input_dim → hidden_dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: latent_dim → hidden_dim → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    ####################################################################
    # Image branch — operates on spatial patch maps (D, H, W)
    ####################################################################

    def encode(self, x: Tensor) -> Tensor:
        """Encode a spatial patch map: (input_dim, H, W) → (latent_dim, H, W)."""
        D, H, W = x.shape
        # Flatten spatial dims, encode flat patches, reshape back to spatial
        return self.encoder(x.flatten(1).T).T.reshape(self.latent_dim, H, W)

    def decode(self, x: Tensor) -> Tensor:
        """Decode a spatial patch map: (latent_dim, H, W) → (input_dim, H, W)."""
        K, H, W = x.shape
        # Flatten, decode, reshape back to spatial
        return self.decoder(x.flatten(1).T).T.reshape(self.input_dim, H, W)

    ####################################################################
    # Point branch — operates on flat point arrays (N, D)
    ####################################################################

    def per_point_encode(self, x: Tensor) -> Tensor:
        """Encode flat point features: (N, input_dim) → (N, latent_dim)."""
        return self.encoder(x)

    def per_point_decode(self, x: Tensor) -> Tensor:
        """Decode flat point codes: (N, latent_dim) → (N, input_dim)."""
        return self.decoder(x)

    ####################################################################
    # Training
    ####################################################################

    def fit(
        self,
        features: Tensor,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        """Train the autoencoder in-place on flat feature tensor (N, input_dim).

        Loss = MSE(recon, x) + (1 − mean cosine_similarity(recon, x))

        The model is moved to the device of ``features`` before training and
        left in eval mode afterwards.
        """
        # Move model to match feature device, then set train mode
        self.to(features.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        N = features.shape[0]

        for epoch in range(epochs):
            # Shuffle patch indices each epoch for unbiased mini-batches
            perm = torch.randperm(N, device=features.device)
            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch gradient updates
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x = features[idx]

                # Reconstruction via point branch; combined MSE + cosine loss
                recon = self.per_point_decode(self.per_point_encode(x))
                loss = F.mse_loss(recon, x) + (
                    1 - F.cosine_similarity(recon, x).mean()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.debug("epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg_loss)

            # Advance LR schedule once per epoch if provided
            if lr_scheduler is not None:
                lr_scheduler.step()

        self.eval()

    ####################################################################
    # Persistence
    ####################################################################

    def save(self, path: Path) -> None:
        """Save autoencoder to ``path/autoencoder.pt``.

        Moves model to CPU before saving so the checkpoint is device-agnostic,
        then restores the original device.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Snapshot current device, move to CPU for serialisation
        device = next(self.parameters()).device
        self.cpu()

        payload = {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path / "autoencoder.pt")

        # Restore original device
        self.to(device)
        logger.info("saved autoencoder → %s/autoencoder.pt", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureAutoencoder":
        """Load autoencoder from ``path/autoencoder.pt``."""
        path = Path(path)

        # Reconstruct architecture from saved hyperparams, then load weights
        payload = torch.load(
            path / "autoencoder.pt", map_location="cpu", weights_only=True
        )
        ae = cls(
            input_dim=payload["input_dim"],
            latent_dim=payload["latent_dim"],
            hidden_dim=payload["hidden_dim"],
        )
        ae.load_state_dict(payload["state_dict"])
        ae.eval()

        logger.info("loaded autoencoder ← %s/autoencoder.pt", path)
        return ae
