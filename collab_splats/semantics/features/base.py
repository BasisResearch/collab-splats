"""
Base classes and shared constants for feature extractors.

- BaseFeatureExtractor: abstract registry-based extractor with caching and debiasing
- BaseQueryableExtractor: extends the base with text-query scoring
- _DEBIAS_VALIDATED: extractor class names whose debiasing has been validated
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from collab_splats.semantics.utils import compute_semantic_contrast
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)

# Extractors where positional debiasing is empirically validated (DINO-family)
# - add a class name here after verifying debiasing improves that model family
# - extractors NOT listed still work with debias(), but warn at call time
_DEBIAS_VALIDATED: frozenset = frozenset({"DINOFeatureExtractor", "Talk2DinoExtractor"})


########################################################
########## Registry and abstract base ##################
########################################################


class BaseFeatureExtractor(RegistryMixin, nn.Module, ABC):
    """
    Abstract base for image feature extractors with a name-based registry.

    - Register subclasses via `@BaseFeatureExtractor.register("name")`, retrieve with `.get("name")`.
    - Subclasses set `self._normalize` and `self.patch_size`; `preprocess` is shared.
    """

    _registry: Dict[str, type["BaseFeatureExtractor"]] = {}

    def __init__(self, resize_mode: str, image_resolution: int, svd_components: int = 500) -> None:
        """
        Store the shared preprocessing and debiasing configuration.

        Subclasses must set `self._normalize` (a `T.Normalize`) and `self.patch_size` before
        `preprocess` is called — the stats and stride are backbone-specific.

        Args:
            resize_mode: "max_size" (proportional longest-edge) or "square" (center-crop + resize).
            image_resolution: longest-edge target for "max_size", square side for "square".
            svd_components: top singular vectors kept for the positional subspace. 500 matches
                the INSID3 default (see reference implementation).
        """
        super().__init__()

        self._resize_mode = resize_mode
        self._image_resolution = image_resolution
        self.svd_components = svd_components

        # Caches keyed by (H_p, W_p) so different input resolutions each get their own basis.
        self._pos_basis_cache: dict = {}   # (H_p, W_p) → Tensor(D, K) positional subspace basis
        self._zero_feats_cache: dict = {}  # (H_p, W_p) → Tensor(D, H_p, W_p) zero-image features for viz

    @abstractmethod
    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Preprocess, run inference, and reshape to patch grids.

        Args:
            images: anything `open_image` accepts, one entry per frame.

        Returns:
            One (D, H_p, W_p) tensor per image, on CPU.
        """
        ...

    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Resize to the configured resolution, round to patch multiples, and normalize.

        Args:
            image: anything `open_image` accepts — path, ndarray, or PIL image.

        Returns:
            (C, H, W) float32 CPU tensor. H and W are multiples of `patch_size`.
        """
        img = open_image(image).convert("RGB")

        if self._resize_mode == "square":
            # Center-crop to square, then resize to target resolution
            w, h = img.size
            crop = min(w, h)
            img = img.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
            img = img.resize((self._image_resolution, self._image_resolution), Image.BILINEAR)
        else:
            # Proportional longest-edge resize
            img = resize_image(img, longest_edge=self._image_resolution)

        # Round H and W to the nearest patch_size multiple
        w, h = img.size
        ph = round(h / self.patch_size) * self.patch_size
        pw = round(w / self.patch_size) * self.patch_size
        if (ph, pw) != (h, w):
            img = img.resize((pw, ph), Image.BILINEAR)

        return self._normalize(T.ToTensor()(img))

    @property
    def name(self) -> str:
        """
        Registry key this extractor was registered under.

        Returns:
            The key string — also the stem of its `.zarr` cache.

        Raises:
            AttributeError: if the class was never registered.
        """
        # Walk the registry to find the key that maps to this concrete class
        for key, cls in self._registry.items():
            if cls is type(self):
                return key
        raise AttributeError(
            f"{type(self).__name__} is not registered — use @BaseFeatureExtractor.register('name')"
        )

    @staticmethod
    def features_to_rgb(feat: "torch.Tensor") -> "np.ndarray":
        """
        Project a feature map onto its top-3 PCs for display.

        Args:
            feat: (D, H_p, W_p) float tensor — one frame's output from forward().

        Returns:
            (H_p, W_p, 3) uint8 ndarray. All-zero when the features have no variance.
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
        """
        Estimate the positional subspace from a zero-pixel image using SVD (INSID3 algorithm).

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

        # Zero-pixel (black) image, matching INSID3's torch.zeros approach
        # - expressed as a PIL Image so it passes through each subclass's own forward()
        # - after forward()'s normalization, zero pixels are a fixed non-semantic input,
        #   eliciting the model's positional response with no image-content signal
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
        """
        Project fmap onto the orthogonal complement of the positional subspace.

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
        """
        RGB heatmap of the positional bias at one patch-grid resolution.

        - PCA via SVD (no sklearn): zero-image features projected onto the top 3 components
        - color therefore encodes the dominant axes of positional variation
        - use it to check that debiasing actually captured spatial structure
        - needs a prior debias() call at this (H_p, W_p) to populate the cache

        Args:
            H_p: patch grid height — must match a resolution used in a prior debias() call.
            W_p: patch grid width — same constraint.

        Returns:
            (H_p, W_p, 3) uint8 RGB visualization.

        Raises:
            KeyError: when (H_p, W_p) is not cached — call debias() at this resolution first.
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
        """
        Remove positional bias from patch features by SVD projection (INSID3 algorithm).

        - operates on features already returned by forward()
        - the positional basis comes from a zero-pixel image through this extractor's own forward()
        - built once per patch-grid resolution, then cached and reused

        Args:
            features: (D, H_p, W_p) tensors from forward(); all must share (H_p, W_p).

        Returns:
            The same list with positional bias removed and each patch column L2 re-normalized.
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
    """
    Feature extractor that also embeds text, for cosine queries against patch features.

    - Subclasses implement `encode_text` and `forward`; `compute_similarity` and
      `score_queries` are shared.
    - All subclasses take `model_name` as their first constructor parameter.
    """

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Embed text into the same space as the patch features.

        Args:
            texts: query strings.

        Returns:
            (N, D) L2-normalized embeddings.
        """
        ...

    def compute_similarity(
        self,
        features: torch.Tensor,
        queries: List[str],
    ) -> torch.Tensor:
        """
        Raw cosine similarities between features and text queries.

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
        """
        Score positive queries against negative queries using contrastive softmax.

        Args:
            features: (C, H, W) patch feature map, or (P, D) point feature array.
            positive: Text queries that should score high.
            negative: Text queries that should score low. Defaults to ["object"] —
                matches Talk2DINO paper convention; ensures contrastive softmax is always
                used. Pass [] to skip contrast and return raw cosine similarities directly
                (not recommended for visualization).
            temperature: Softmax temperature. Lower = sharper. Defaults to 0.05.
            reduction: How to reduce across positive queries. "max" or "pool". Defaults to "max".

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
