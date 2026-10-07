"""
Lightweight feature autoencoder for compressing patch features before 3D lifting.

- trained once on every frame's patch features, streamed in blocks, then used to cut the
  dimensionality of the semantic features stored per point
- compression takes either shape, decompression only points
- encode: spatial patch maps (D, H, W) -> (latent_dim, H, W)
- per_point_encode / per_point_decode: flat point arrays (P, D) <-> (P, latent_dim)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from torch import Tensor
from tqdm.auto import tqdm

from collab_splats.utils.torch_utils import batch_iterator, get_device

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

        # decoder as two submodules, not one Sequential: keeps checkpoint state_dict keys
        self.decoder_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        # Fit quality from the last fit(), on the training set; trust only when epochs_run > 0
        self.recon_cosine: float = 0.0
        self.recon_mse: float = 0.0
        self.epochs_run: int = 0

    ####################################################################
    # Image branch — spatial patch maps (D, H, W)
    ####################################################################

    # No image decode: lifted codes are decoded per point (per_point_decode)

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

    @torch.no_grad()
    def iter_decode(self, codes: Tensor, batch_size: int = 65_536) -> Iterator[Tensor]:
        """
        Decode flat point codes one chunk at a time, on this autoencoder's device and dtype.

        - peak memory is one (batch_size, input_dim) chunk, never the full (P, input_dim)

        Args:
            codes: (P, latent_dim), on any device.
            batch_size: codes per chunk.

        Yields:
            (c, input_dim) decoded chunks, in order.
        """
        param = next(self.parameters())

        for (chunk,) in batch_iterator(batch_size, codes):
            chunk = chunk.to(device=param.device, dtype=param.dtype)
            yield self.per_point_decode(chunk)

    ####################################################################
    # Training
    ####################################################################

    def fit(
        self,
        features: Tensor | np.ndarray | zarr.Array,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
        read_gb: float = 1.0,
    ) -> None:
        """
        Train in place on MSE(recon, x) + (1 - cosine(recon, x)), streaming axis-0 blocks.

        - axis 1 = features; axis 0 + trailing axes = samples ((N, D) rows or (N, D, H, W) patch maps)
        - each epoch reads every item: blocks of about read_gb in random order, rows shuffled per block
        - a zarr never sits in memory whole; a tensor trains on its own device, others on get_device()
        - fit quality lands on `self` as recon_cosine, recon_mse, epochs_run (training set, optimistic)

        Args:
            features: tensor, ndarray or zarr array, feature width on axis 1.
            epochs: epoch ceiling.
            batch_size: mini-batch size.
            lr: Adam learning rate.
            on_epoch: callback(epoch, epochs, avg_loss), once per epoch; a progress hook for UIs.
            target_cosine: stop once mean reconstruction cosine reaches this; None runs all epochs.
            read_gb: float32 size of one axis-0 block, in GiB.

        Raises:
            ValueError: if `features` has no samples.
        """
        n_items, dim = features.shape[:2]
        per_item = math.prod(features.shape[2:])

        # Empty input raises a clear ValueError, not a ZeroDivisionError at the epoch mean
        if n_items * per_item == 0:
            raise ValueError(f"fit() requires at least one sample; got features with shape {tuple(features.shape)}")

        device = features.device if isinstance(features, Tensor) else get_device()
        self.to(device)
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Axis-0 blocks of about read_gb float32
        item_gb = per_item * dim * 4 / 2**30
        items_per_block = max(1, math.floor(read_gb / item_gb))
        starts = list(range(0, n_items, items_per_block))

        pbar = tqdm(range(epochs), desc="fit autoencoder", unit="epoch")

        for epoch in pbar:
            epoch_loss = 0.0
            epoch_cos = 0.0
            epoch_mse = 0.0
            n_batches = 0

            for b in torch.randperm(len(starts)).tolist():
                # Read one block as (rows, dim) float32 on device
                start = starts[b]
                block = features[start : start + items_per_block]

                # zarr and ndarray slices arrive as numpy; tensors stay on their device
                if not isinstance(block, Tensor):
                    block = torch.from_numpy(np.asarray(block))

                block = block.to(device=device, dtype=torch.float32)
                block = block.reshape(len(block), dim, per_item)
                block = block.transpose(1, 2)
                block = block.reshape(-1, dim)

                # Shuffle rows within the block for unbiased mini-batches
                perm = torch.randperm(len(block), device=device)

                for first in range(0, len(block), batch_size):
                    x = block[perm[first : first + batch_size]]

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

            # At least one sample (checked above), so n_batches >= 1
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

        # Serialize from CPU so the checkpoint is device-agnostic
        # - the restore sits in a finally: a failed write must not strand the model on CPU
        # - the next forward pass would raise a device mismatch far from the save that caused it
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

        Raises:
            KeyError: when the checkpoint lacks a key `save` writes (older checkpoints must be refit).
        """
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ae = cls(input_dim=payload["input_dim"], latent_dim=payload["latent_dim"])
        ae.load_state_dict(payload["state_dict"])
        ae.recon_cosine = payload["recon_cosine"]
        ae.recon_mse = payload["recon_mse"]
        ae.epochs_run = payload["epochs_run"]
        ae.eval()

        logger.info("loaded autoencoder ← %s", path)
        return ae
