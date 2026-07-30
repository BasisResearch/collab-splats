"""
collab_splats.semantics.compression — lightweight feature autoencoder for
compressing patch features before 3D lifting.

Trained once on keyframe features; used at inference time to reduce the
dimensionality of semantic features stored per point.

Mirrors the two-branch convention of TwoLayerMLP:
  encode / decode         — spatial image branch  (D, H, W) ↔ (latent_dim, H, W)
  per_point_encode/decode — flat point branch     (N, D)    ↔ (N, latent_dim)

Optional regularization branch (via regularization_kwargs) adds an auxiliary decoder
head trained with a weighted cosine loss — e.g. DINOv2 regularizing a MaskCLIP AE.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from torch import Tensor
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

########################################################################
# FeatureAutoencoder
########################################################################


class FeatureAutoencoder(nn.Module):
    """Two-layer MLP autoencoder for compressing semantic patch features.

    Architecture
    ------------
    encoder:        Linear(input_dim → hidden_dim) → ReLU → Linear(hidden_dim → latent_dim)
    decoder_hidden: Linear(latent_dim → hidden_dim) → ReLU  [shared with reg head]
    decoder_out:    Linear(hidden_dim → input_dim)           [main reconstruction head]
    reg_head:       Linear(hidden_dim → reg_dim)             [optional]

    Two entry points (mirrors TwoLayerMLP convention):
      encode / decode         — spatial image maps  (D, H, W) ↔ (latent_dim, H, W)
      per_point_encode/decode — flat point arrays   (N, D)    ↔ (N, latent_dim)

    regularization_kwargs
    ---------------------
    Optional dict with keys:
      "dim":    int   — output dim of the regularization head
      "weight": float — loss weight applied to the reg head during fit()
    Defaults to None (pure autoencoder, no reg head).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        regularization_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        # Resolve hidden dimension — default to twice the latent dim, min 64
        if hidden_dim is None:
            hidden_dim = max(64, 2 * latent_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.regularization_kwargs = regularization_kwargs or {}

        # Derived regularization config
        self._reg_dim: Optional[int] = self.regularization_kwargs.get("dim")
        self._reg_weight: float = self.regularization_kwargs.get("weight", 0.1)

        # Encoder: input_dim → hidden_dim → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder shared hidden: latent_dim → hidden_dim
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        # Main reconstruction head: hidden_dim → input_dim
        self.decoder_out = nn.Linear(hidden_dim, input_dim)

        # Optional regularization head: hidden_dim → reg_dim
        self.reg_head: Optional[nn.Linear] = nn.Linear(hidden_dim, self._reg_dim) if self._reg_dim is not None else None

        # Reconstruction quality from the last fit(), persisted in the checkpoint so a consumer
        # can judge whether decoded 768-D codes are trustworthy. Both metrics are measured on
        # the TRAINING set and are optimistic at small sample counts (a few hundred patches can
        # score ~1.0 in-sample while generalizing far worse); treat them as a fit-completed
        # signal, not a held-out score. 0.0 here means one of three things — never trained,
        # genuinely-terrible training, or a legacy checkpoint predating these keys — so gate on
        # ``epochs_run > 0`` before trusting recon_cosine/recon_mse at all.
        self.recon_cosine: float = 0.0
        self.recon_mse: float = 0.0
        # Epochs actually completed by the last fit(); 0 = never trained OR legacy checkpoint.
        self.epochs_run: int = 0

    ####################################################################
    # Image branch — operates on spatial patch maps (D, H, W)
    ####################################################################

    def encode(self, x: Tensor) -> Tensor:
        """Encode a spatial patch map: (input_dim, H, W) → (latent_dim, H, W)."""
        D, H, W = x.shape
        # Flatten spatial dims, encode, reshape back to spatial
        return self.encoder(x.flatten(1).T).T.reshape(self.latent_dim, H, W)

    def decode(self, x: Tensor) -> Tensor:
        """Decode a spatial patch map: (latent_dim, H, W) → (input_dim, H, W)."""
        K, H, W = x.shape
        # Flatten, decode via shared hidden + main head, reshape back to spatial
        flat = x.flatten(1).T  # (H*W, latent_dim)
        return self.decoder_out(self.decoder_hidden(flat)).T.reshape(self.input_dim, H, W)

    ####################################################################
    # Point branch — operates on flat point arrays (N, D)
    ####################################################################

    def per_point_encode(self, x: Tensor) -> Tensor:
        """Encode flat point features: (N, input_dim) → (N, latent_dim)."""
        return self.encoder(x)

    def per_point_decode(self, x: Tensor) -> Tensor:
        """Decode flat point codes: (N, latent_dim) → (N, input_dim)."""
        return self.decoder_out(self.decoder_hidden(x))

    ####################################################################
    # Training
    ####################################################################

    def fit(
        self,
        features: Tensor,
        reg_target: Optional[Tensor] = None,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        on_epoch: Optional[Callable[[int, int, float], None]] = None,
        target_cosine: Optional[float] = None,
    ) -> None:
        """Train the autoencoder in-place on flat feature tensor (N, input_dim).

        Loss = MSE(recon, x) + (1 − cosine(recon, x))
             + reg_weight * (1 − cosine(reg_head(h), reg_target))

        Args:
            features: (N, input_dim) main feature tensor.
            reg_target: optional (N, reg_dim) tensor for the regularization head.
            on_epoch: optional callback(epoch, total_epochs, avg_loss) fired once per epoch.
                Use to surface progress to a UI (tqdm/logger.debug do not reach the dashboard).
            target_cosine: stop once mean reconstruction cosine reaches this value, using
                ``epochs`` as a ceiling. None (default) always runs the full ``epochs``.
                Decoded 768-D features are what cross-scene comparison uses, so
                reconstruction fidelity is the right stop signal, not a fixed epoch count.
                Measured on the TRAINING set — it is optimistic at small sample counts, so a
                target met on a few hundred patches does not promise the same fidelity on
                unseen ones. At production scale (10^5–10^6 patches) the gap largely collapses.

        Raises:
            ValueError: if ``features`` has no samples, or if ``reg_target`` is supplied
                without a configured reg head.
        """
        # Validate reg_target against configured head
        if reg_target is not None and self.reg_head is None:
            raise ValueError(
                "reg_target supplied but no reg_head configured; " "pass regularization_kwargs to __init__"
            )

        # Reject empty input up front: zero gradient steps would still record metrics and
        # could satisfy target_cosine, publishing a false "trained" signal downstream.
        if features.shape[0] == 0:
            raise ValueError(f"fit() requires at least one sample; got features with shape {tuple(features.shape)}")

        # Move model to match feature device, then set train mode
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

            # Mini-batch gradient updates
            for start in range(0, N, batch_size):
                idx = perm[start : start + batch_size]
                x = features[idx]

                # Shared hidden representation from encoded latent
                h = self.decoder_hidden(self.encoder(x))
                recon = self.decoder_out(h)

                # Main reconstruction: MSE + cosine loss (tracked separately for metrics)
                mse = F.mse_loss(recon, x)
                cos = F.cosine_similarity(recon, x).mean()
                loss = mse + (1 - cos)

                # Regularization head loss (weighted cosine)
                if reg_target is not None:
                    reg_batch = reg_target[idx].to(features.device)
                    loss = loss + self._reg_weight * (1 - F.cosine_similarity(self.reg_head(h), reg_batch).mean())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cos += cos.item()
                epoch_mse += mse.item()
                n_batches += 1

            # Epoch tail: average the accumulators and record achieved fidelity. N >= 1 is
            # guaranteed above and range(0, N, batch_size) yields ceil(N/batch_size) >= 1
            # batches, so n_batches is never 0 here.
            avg_loss = epoch_loss / n_batches
            self.recon_cosine = epoch_cos / n_batches
            self.recon_mse = epoch_mse / n_batches
            self.epochs_run = epoch + 1

            pbar.set_postfix(loss=f"{avg_loss:.6f}", cos=f"{self.recon_cosine:.4f}")
            logger.debug("epoch %d/%d  loss=%.6f  cos=%.4f", epoch + 1, epochs, avg_loss, self.recon_cosine)
            if on_epoch is not None:
                on_epoch(epoch + 1, epochs, avg_loss)

            # Advance LR schedule once per epoch if provided
            if lr_scheduler is not None:
                lr_scheduler.step()

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
            "regularization_kwargs": self.regularization_kwargs,
            "state_dict": self.state_dict(),
            # Fit quality — lets a consumer decide whether to trust decoded 768-D codes
            "recon_cosine": self.recon_cosine,
            "recon_mse": self.recon_mse,
            "epochs_run": self.epochs_run,
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
        payload = torch.load(path / "autoencoder.pt", map_location="cpu", weights_only=True)
        ae = cls(
            input_dim=payload["input_dim"],
            latent_dim=payload["latent_dim"],
            hidden_dim=payload["hidden_dim"],
            regularization_kwargs=payload.get("regularization_kwargs"),
        )
        ae.load_state_dict(payload["state_dict"])
        # Older checkpoints predate the metrics — default to 0.0 rather than failing
        ae.recon_cosine = payload.get("recon_cosine", 0.0)
        ae.recon_mse = payload.get("recon_mse", 0.0)
        ae.epochs_run = payload.get("epochs_run", 0)
        ae.eval()

        logger.info("loaded autoencoder ← %s/autoencoder.pt", path)
        return ae


########################################################################
# Per-point artifact pair (features.zarr + autoencoder.pt)
########################################################################


def write_point_features(out_dir: Path, codes: np.ndarray, ae: Optional[FeatureAutoencoder] = None) -> None:
    """Write the canonical per-point pair: out_dir/features.zarr (+ autoencoder.pt when compressed).

    Single writer for every lifting path. Records input_dim/latent_dim on the zarr group's attrs
    so the artifact is self-describing: equal widths (ae=None) mean full-dim codes that need no
    weights, unequal widths mean the weights are REQUIRED to decode them. Without that marker a
    missing autoencoder.pt is ambiguous — uncompressed-by-design vs latent codes orphaned by a
    crash — and a reader would have to guess, silently handing back undecoded codes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = np.asarray(codes)
    features_zarr = out_dir / "features.zarr"

    # Codes first, then attrs, then weights; a failure anywhere after the store is created
    # removes it again so no half-pair (unreadable codes) is ever left behind on disk.
    try:
        store = zarr.open(str(features_zarr), mode="w")
        store["features"] = codes
        store.attrs.update(
            {
                "input_dim": int(ae.input_dim) if ae is not None else int(codes.shape[1]),
                "latent_dim": int(ae.latent_dim) if ae is not None else int(codes.shape[1]),
            }
        )
        if ae is not None:
            ae.save(out_dir)
    except Exception:
        shutil.rmtree(features_zarr, ignore_errors=True)
        raise
