"""Base classes and shared constants for feature extractors.

Provides:
  BaseFeatureExtractor     — abstract registry-based extractor with caching and debiasing
  BaseQueryableExtractor   — extends base with text-query scoring
  TORCH_HOME               — resolved torch cache directory
  _DEBIAS_VALIDATED        — set of extractor class names with validated debiasing
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel

from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    get_device,
    interpolate_to_patch_size,
)
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin

# TORCH_HOME: respects $TORCH_HOME env var, falls back to ~/.cache/torch (torch default)
TORCH_HOME = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

logger = logging.getLogger(__name__)

# Extractors where positional debiasing has been empirically validated against DINO-family models.
# Add a class name here after verifying that debiasing improves quality for that model family.
# Extractors NOT in this set still work with debias() but receive a warning at call time.
_DEBIAS_VALIDATED: frozenset = frozenset({"DINOFeatureExtractor", "Talk2DinoExtractor"})


########################################################
########## Registry and abstract base ##################
########################################################


class BaseFeatureExtractor(RegistryMixin, nn.Module, ABC):
    """Abstract base for image feature extractors with a name-based registry.

    Register subclasses via ``@BaseFeatureExtractor.register("name")``.
    Retrieve with ``BaseFeatureExtractor.get("name")``.
    """

    _registry: Dict[str, type["BaseFeatureExtractor"]] = {}
    _FALLBACK_MEM_GB: float = 2.0

    def __init__(self, svd_components: int = 500, **kwargs) -> None:
        # svd_components: number of top singular vectors kept for the positional subspace.
        # 500 matches the INSID3 default (see reference implementation).
        super().__init__(**kwargs)  # passes remaining kwargs up to nn.Module
        self.svd_components = svd_components  # stored so _build_positional_basis can read it later
        # Caches keyed by (H_p, W_p) so different input resolutions each get their own basis.
        self._pos_basis_cache: dict = {}   # (H_p, W_p) → Tensor(D, K) positional subspace basis
        self._zero_feats_cache: dict = {}  # (H_p, W_p) → Tensor(D, H_p, W_p) zero-image features for viz

    @abstractmethod
    def forward(self, images: list) -> list[torch.Tensor]:
        """Preprocess, run inference, reshape. Returns one feature tensor per image."""
        ...

    @property
    def name(self) -> str:
        """Registry key for this extractor — reverses the _registry dict lookup."""
        # Walk the registry to find the key that maps to this concrete class
        for key, cls in self._registry.items():
            if cls is type(self):
                return key
        raise AttributeError(
            f"{type(self).__name__} is not registered — use @BaseFeatureExtractor.register('name')"
        )

    @staticmethod
    def features_to_rgb(feat: "torch.Tensor") -> "np.ndarray":
        """Project a feature map onto its top-3 PCs to produce an RGB display image.

        Args:
            feat: (D, H_p, W_p) float tensor — output of forward() for one frame.

        Returns:
            np.ndarray of shape (H_p, W_p, 3) dtype uint8.
        """
        D, H_p, W_p = feat.shape
        E = feat.reshape(D, -1).T.float()  # (N, D) where N = H_p * W_p
        E = E - E.mean(dim=0, keepdim=True)  # center per-channel
        _, _, Vt = torch.linalg.svd(E, full_matrices=False)  # Vt: (min(N,D), D)
        rgb = (E @ Vt[:3].T).detach().cpu().numpy()  # (N, 3) — project onto top-3 PCs
        # Normalize to [0, 255]: shift to zero, scale by range, guard zero-variance case
        rgb -= rgb.min()
        rgb /= rgb.max() + 1e-8
        return (rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8)

    def _build_positional_basis(self, H_p: int, W_p: int) -> None:
        """Estimate the positional subspace from a zero-pixel image using SVD (INSID3 algorithm).

        A black (zero-pixel) image, after each extractor's own normalization, produces features
        driven entirely by the model's positional embeddings — no semantic content to interfere.
        SVD then extracts the top-K directions of variance, representing the positional subspace.

        Calls self.forward() so each subclass handles its own preprocessing identically to
        real inference — no separate code path, no reimplementation of preprocessing.

        Stores results in _pos_basis_cache[(H_p, W_p)] and _zero_feats_cache[(H_p, W_p)].
        Only called once per resolution; all subsequent debias() calls at (H_p, W_p) reuse the cache.

        Args:
            H_p: Number of patch rows in the target feature map.
            W_p: Number of patch columns in the target feature map.
        """
        # Require patch_size to be set explicitly — no silent fallback.
        # If missing, the AttributeError here is far better than producing a silently wrong zero image.
        if not hasattr(self, "patch_size"):
            raise AttributeError(
                f"{type(self).__name__} must define `self.patch_size` "
                "(the pixel stride of each patch token) before calling debias()."
            )
        patch_size = self.patch_size  # pixel stride per patch; set in each concrete subclass __init__
        H_img = H_p * patch_size  # image height that produces H_p patch rows under stride-exact preprocessing
        W_img = W_p * patch_size  # image width  that produces W_p patch cols under stride-exact preprocessing

        # Zero-pixel (black) image — matches INSID3's torch.zeros approach, expressed as a PIL Image
        # so it passes naturally through each subclass's own forward() without special-casing.
        # After normalization inside forward(), zero pixels become a fixed non-semantic input
        # that elicits the model's positional response with no image-content signal.
        zero_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        zero_pil = Image.fromarray(zero_arr)  # PIL Image so forward() preprocessing runs unchanged

        with torch.no_grad():
            [zero_feat] = self.forward([zero_pil])  # (D, H_p', W_p') — subclass forward handles preprocessing

        if zero_feat.shape[1:] != (H_p, W_p):
            # The subclass's internal preprocessing (e.g. Talk2DINO upscaling) changed the grid size.
            # Interpolate to the target (H_p, W_p) — positional bias is spatially smooth so this is valid.
            zero_feat = F.interpolate(
                zero_feat.unsqueeze(0),  # (1, D, H_p', W_p') — F.interpolate requires a leading batch dim
                size=(H_p, W_p),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # remove batch dim → (D, H_p, W_p)

        self._zero_feats_cache[(H_p, W_p)] = zero_feat  # saved for get_bias_visualization

        D = zero_feat.shape[0]  # feature dimensionality — needed to reshape E for SVD
        E = zero_feat.reshape(D, -1)               # (D, H_p*W_p) — one column per patch, one row per feature
        E = E - E.mean(dim=1, keepdim=True)        # center: remove mean activation per channel before SVD

        # SVD: columns of U are principal directions of feature variance across spatial patch positions.
        # Because the image has no content, these directions capture positional variation only.
        U, _, _ = torch.linalg.svd(E, full_matrices=False)  # U: (D, min(D, H_p*W_p))

        # Keep top-K directions — they represent the positional subspace; the tail captures noise.
        self._pos_basis_cache[(H_p, W_p)] = U[:, : self.svd_components].contiguous()  # (D, K)

    def _apply_debias(self, fmap: torch.Tensor) -> torch.Tensor:
        """Project fmap onto the orthogonal complement of the positional subspace.

        Removes the positional component from each patch feature vector, then re-normalizes
        L2 so downstream cosine-similarity comparisons remain well-defined.

        Args:
            fmap: (D, H_p, W_p) patch feature map — same spatial resolution as a cached basis.

        Returns:
            (D, H_p, W_p) debiased feature map with unit-norm patch vectors.
        """
        D, H_p, W_p = fmap.shape
        basis = self._pos_basis_cache[(H_p, W_p)].to(fmap.device)  # (D, K) — move to same device as features

        # P_perp = I - U @ U.T projects out the component in span(U) (the positional subspace).
        # Applying P_perp to a feature vector zeroes its positional component, keeping only semantics.
        P_perp = torch.eye(D, device=fmap.device, dtype=fmap.dtype) - basis @ basis.T  # (D, D)

        X = fmap.reshape(D, -1)    # (D, H_p*W_p) — flatten spatial dims for matrix multiply
        X_deb = P_perp @ X         # (D, H_p*W_p) — positional component removed from each patch vector

        # Re-normalize: after projection the vectors are no longer unit-norm.
        # Cosine-similarity comparisons in downstream code require unit-norm patch vectors.
        X_deb = F.normalize(X_deb, p=2, dim=0)  # normalize along the feature axis (dim=0) → each patch column becomes unit-norm

        return X_deb.reshape(D, H_p, W_p)  # restore spatial layout

    def get_bias_visualization(self, H_p: int, W_p: int) -> "np.ndarray":
        """Return an RGB heatmap of the positional bias at a given patch grid resolution.

        Uses PCA (via SVD, no sklearn required) to project the zero-image features onto the
        top 3 principal components, producing an image where color encodes the dominant axes
        of positional variation. Useful for verifying that debiasing captures spatial structure.

        Requires a prior debias() call at this (H_p, W_p) resolution to populate the cache.

        Args:
            H_p: Patch grid height — must match a resolution used in a prior debias() call.
            W_p: Patch grid width — must match a resolution used in a prior debias() call.

        Returns:
            np.ndarray of shape (H_p, W_p, 3) dtype uint8 — PCA-derived RGB visualization.

        Raises:
            KeyError: if (H_p, W_p) has not been cached — call debias() at this resolution first.
        """
        if (H_p, W_p) not in self._zero_feats_cache:
            raise KeyError(
                f"No positional bias cached for patch grid ({H_p}, {W_p}). "
                "Call debias() at this resolution first."
            )
        zero_feats = self._zero_feats_cache[(H_p, W_p)]  # (D, H_p, W_p) — zero-image patch features
        D = zero_feats.shape[0]  # feature dimensionality — needed to reshape before SVD

        # PCA via SVD on the (N, D) matrix where N = H_p*W_p patch locations.
        # Each row is one patch's feature vector; SVD gives the principal axes of spatial variation.
        E = zero_feats.reshape(D, -1).T  # (N, D) — transpose so rows are per-patch observations
        E = E - E.mean(dim=0, keepdim=True)  # center per channel (dim=0 on (N,D) ≡ dim=1 on (D,N) in _build_positional_basis — same operation)

        # SVD: right singular vectors Vt span the principal directions in feature space.
        # We only need Vt[:3] (top-3 right singular vectors) to project onto the 3 dominant PCs.
        _, _, Vt = torch.linalg.svd(E, full_matrices=False)  # Vt: (min(N,D), D)

        rgb = (E @ Vt[:3].T).cpu().numpy()  # (N, 3) — project each patch onto the top-3 PCs

        # Normalize to [0, 1] for display: shift minimum to zero, then scale by range.
        # +1e-8 prevents divide-by-zero when all features are identical (e.g. all-black image edge case).
        rgb -= rgb.min()
        rgb /= rgb.max() + 1e-8

        return (rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8)  # scale to uint8 RGB for display

    def debias(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Remove positional bias from extracted patch features using SVD projection (INSID3 algorithm).

        Operates on features already returned by forward(). Builds and caches the positional
        basis from a zero-pixel image (using this extractor's own forward()) on first call at
        each patch-grid resolution, then reuses the cached basis for all subsequent calls.

        Args:
            features: list of (D, H_p, W_p) tensors — output of forward(). All tensors must
                      have the same spatial resolution (H_p, W_p).

        Returns:
            list of (D, H_p, W_p) tensors with positional bias removed and L2 re-normalized.
        """
        if type(self).__name__ not in _DEBIAS_VALIDATED:
            # Algorithm is general but has only been verified for DINO-family models.
            # Warn rather than raise so researchers can experiment with other extractors.
            logger.warning(
                "[%s] Positional debiasing not yet validated for this extractor. "
                "Proceeding — results may be suboptimal.",
                type(self).__name__,
            )
        _, H_p, W_p = features[0].shape  # read patch grid dimensions from the first feature map
        if (H_p, W_p) not in self._pos_basis_cache:
            # First debias() call at this resolution — build and cache the positional basis.
            self._build_positional_basis(H_p, W_p)
        return [self._apply_debias(f) for f in features]  # project each image's feature map


########################################################
########## Queryable extractor base ####################
########################################################


class BaseQueryableExtractor(BaseFeatureExtractor, ABC):
    """Abstract base for feature extractors that support text-conditioned similarity queries.

    Subclasses must implement encode_text() and forward(). compute_similarity()
    and score_queries() are provided here and shared across all queryable extractors.
    All subclasses accept model_name as the first __init__ parameter.
    """

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Return normalized text embeddings of shape (N, D)."""
        ...

    @abstractmethod
    def forward(self, images: list) -> list:
        """Preprocess, run inference, reshape. Returns one (C, H, W) feature tensor per image.

        Subclasses may accept additional optional keyword arguments (e.g., resolution)
        as extensions of this contract. Callers using the base type receive the standard
        no-kwarg interface; callers with concrete type access extensions directly.
        """
        ...

    def compute_similarity(
        self,
        features: torch.Tensor,
        queries: List[str],
    ) -> torch.Tensor:
        """Raw cosine similarities between features and text queries.

        Args:
            features: (C, H, W) patch feature map, or (P, D) point feature array.
            queries: text strings to compare against features.

        Returns:
            (N_queries, H, W) when input is (C, H, W).
            (N_queries, P)    when input is (P, D).
        """
        is_points = features.ndim == 2
        if is_points:
            features = features.T.unsqueeze(-1)  # (D, P, 1)
        text_embs = self.encode_text(queries)
        # Move features to the text encoder's device (forward returns CPU tensors)
        features = features.to(text_embs.device)
        out = torch.einsum("chw,nc->nhw", features, text_embs)
        return out.squeeze(-1) if is_points else out

    def score_queries(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]] = None,
        temperature: float = 0.05,
        reduction: str = "max",
    ) -> torch.Tensor:
        """Score positive queries against negative queries using contrastive softmax.

        Args:
            features: (C, H, W) patch feature map, or (P, D) point feature array.
            positive: Text queries that should score high.
            negative: Text queries that should score low. Defaults to ["object"] —
                matches Talk2DINO paper convention; ensures contrastive softmax is always
                used. Pass [] to skip contrast and return raw cosine similarities directly
                (not recommended for visualization).
            temperature: Softmax temperature. Lower = sharper. Defaults to 0.05.
            reduction: How to reduce across positive queries. "max" or "mean". Defaults to "max".

        Returns:
            (H, W) score map in [0, 1] when input is (C, H, W).
            (P,)   score array in [0, 1] when input is (P, D).
        """
        logger.debug(
            "[%s] Scoring queries: %d positive%s (reduction=%s)",
            type(self).__name__, len(positive),
            f", {len(negative)} negative" if negative else "",
            reduction,
        )
        if negative is None:
            negative = ["object"]
        queries = positive + negative
        similarity = self.compute_similarity(features, queries)
        result = compute_semantic_contrast(similarity, len(positive), temperature, reduction)
        return result
