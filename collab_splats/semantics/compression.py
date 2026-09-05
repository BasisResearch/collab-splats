"""
Lightweight feature autoencoder for compressing patch features before 3D lifting.

Trained once on keyframe features, then used to reduce the dimensionality of the semantic
features stored per point. Compression happens on either shape, decompression only on points:
  encode                  — spatial patch maps (D, H, W) -> (latent_dim, H, W)
  per_point_encode/decode — flat point arrays  (P, D)    <-> (P, latent_dim)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

########################################################################
# FeatureAutoencoder
########################################################################


class FeatureAutoencoder(nn.Module):
    """
    Two-layer MLP autoencoder for compressing semantic patch features.

    Args:
        input_dim: width of the features being compressed.
        latent_dim: width of the codes written to disk.
    """

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()

        # Hidden width is derived, never configured: twice the latent dim, floor 64
        hidden_dim = max(64, 2 * latent_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # decoder: latent_dim -> hidden_dim -> input_dim. Kept as two submodules rather than
        # one Sequential so the state_dict keys match checkpoints already on disk.
        self.decoder_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        # Fit quality from the last fit(), persisted in the checkpoint. Both are measured on the
        # TRAINING set, so gate on epochs_run > 0 before trusting recon_cosine/recon_mse at all.
        self.recon_cosine: float = 0.0
        self.recon_mse: float = 0.0
        self.epochs_run: int = 0

    ####################################################################
    # Image branch — spatial patch maps (D, H, W)
    ####################################################################

    # Compress-only: the dashboard encodes patch maps then lifts the latent maps to points,
    # so decoding a map back to (D, H, W) never happens. per_point_decode is the read path.

    def encode(self, feature_map: Tensor) -> Tensor:
        """
        Encode a spatial patch map.

        Args:
            feature_map: (input_dim, H, W).

        Returns:
            (latent_dim, H, W).
        """
        _, H, W = feature_map.shape
        return self.encoder(feature_map.flatten(1).T).T.reshape(self.latent_dim, H, W)

    ####################################################################
    # Point branch — flat point arrays (P, D)
    ####################################################################

    def per_point_encode(self, feats: Tensor) -> Tensor:
        """
        Encode flat point features.

        Args:
            feats: (P, input_dim).

        Returns:
            (P, latent_dim).
        """
        return self.encoder(feats)

    def per_point_decode(self, codes: Tensor) -> Tensor:
        """
        Decode flat point codes.

        Args:
            codes: (P, latent_dim).

        Returns:
            (P, input_dim).
        """
        return self.decoder_out(self.decoder_hidden(codes))

    ####################################################################
    # Training
    ####################################################################

    def fit(
        self,
        features: Tensor,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
    ) -> None:
        """
        Train the autoencoder in place; loss is MSE(recon, x) + (1 - cosine(recon, x)).

        Args:
            features: (N, input_dim) training features.
            epochs: epoch ceiling.
            batch_size: mini-batch size.
            lr: Adam learning rate.
            on_epoch: callback(epoch, epochs, avg_loss) fired once per epoch — the dashboard
                reads progress through it, since tqdm and logger.debug do not reach the UI.
            target_cosine: stop once mean reconstruction cosine reaches this, with `epochs`
                as the ceiling. Measured on the training set, so optimistic at small N.

        Fit quality lands on `self` as recon_cosine / recon_mse / epochs_run.

        Raises:
            ValueError: if `features` has no samples.
        """
        # Reject empty input up front: zero gradient steps would still record metrics and could
        # satisfy target_cosine, publishing a false "trained" signal downstream.
        if features.shape[0] == 0:
            raise ValueError(f"fit() requires at least one sample; got features with shape {tuple(features.shape)}")

        self.to(features.device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        N = features.shape[0]

        pbar = tqdm(range(epochs), desc="fit autoencoder", unit="epoch")
        for epoch in pbar:
            # Shuffle patch indices each epoch for unbiased mini-batches
            perm = torch.randperm(N, device=features.device)
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x = features[idx]

                recon = self.decoder_out(self.decoder_hidden(self.encoder(x)))
                mse = F.mse_loss(recon, x)
                cos = F.cosine_similarity(recon, x).mean()
                loss = mse + (1 - cos)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cos += cos.item()
                epoch_mse += mse.item()
                n_batches += 1

            # N >= 1 is guaranteed above and range(0, N, batch_size) yields >= 1 batch,
            # so n_batches is never 0 here.
            avg_loss = epoch_loss / n_batches
            self.recon_cosine = epoch_cos / n_batches
            self.recon_mse = epoch_mse / n_batches
            self.epochs_run = epoch + 1

            pbar.set_postfix(loss=f"{avg_loss:.6f}", cos=f"{self.recon_cosine:.4f}")
            logger.debug("epoch %d/%d  loss=%.6f  cos=%.4f", epoch + 1, epochs, avg_loss, self.recon_cosine)
            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, avg_loss)

            # Early stop once reconstruction is good enough; epochs is the ceiling
            if target_cosine is not None and self.recon_cosine >= target_cosine:
                logger.info(
                    "target cosine %.4f reached at epoch %d/%d (cos=%.4f) — stopping",
                    target_cosine,
                    epoch + 1,
                    epochs,
                    self.recon_cosine,
                )
                break

        self.eval()

    ####################################################################
    # Persistence
    ####################################################################

    def save(self, path: Path) -> None:
        """
        Write the checkpoint to a weights file path.

        Args:
            path: destination `.pt` file — its parent dir is created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialise from CPU so the checkpoint is device-agnostic. The restore is in a finally
        # because a failed write must not strand the caller's model on CPU — the next forward
        # pass would then raise a device mismatch far from the save that caused it.
        device = next(self.parameters()).device
        self.cpu()
        try:
            torch.save(
                {
                    "input_dim": self.input_dim,
                    "latent_dim": self.latent_dim,
                    "state_dict": self.state_dict(),
                    "recon_cosine": self.recon_cosine,
                    "recon_mse": self.recon_mse,
                    "epochs_run": self.epochs_run,
                },
                path,
            )
        finally:
            self.to(device)

        logger.info("saved autoencoder → %s", path)

    @classmethod
    def load(cls, path: Path) -> "FeatureAutoencoder":
        """
        Read a checkpoint from a weights file path.

        Args:
            path: the `.pt` file written by `save`.

        Returns:
            An eval-mode FeatureAutoencoder with the checkpoint's weights and fit metrics.
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ae = cls(input_dim=payload["input_dim"], latent_dim=payload["latent_dim"])
        ae.load_state_dict(payload["state_dict"])
        # Checkpoints predating the metric keys default to the untrained values rather than failing
        ae.recon_cosine = payload.get("recon_cosine", 0.0)
        ae.recon_mse = payload.get("recon_mse", 0.0)
        ae.epochs_run = payload.get("epochs_run", 0)
        ae.eval()

        logger.info("loaded autoencoder ← %s", path)
        return ae
