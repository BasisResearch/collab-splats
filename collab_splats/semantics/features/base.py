"""
Base classes for patch-feature extractors.

- BaseFeatureExtractor: registry, shared preprocess, positional debiasing
- BaseQueryableExtractor: adds text embedding and contrastive query scoring
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

from collab_splats.semantics.utils import _tokens_to_feature_map, compute_semantic_contrast
from collab_splats.utils.image import open_image, resize_image
from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)


########################################################
########## Registry and abstract base ##################
########################################################


class BaseFeatureExtractor(RegistryMixin, nn.Module, ABC):
    """
    Image patch-feature extractor, registered by name.

    - register with `@BaseFeatureExtractor.register("name")`, look up with `.get("name")`
    - subclasses set `_normalize`, `patch_size`, `_device` and implement `_patch_tokens`
    - a backend whose tokens lead with CLS or registers sets `n_prefix_tokens` to their count
    - `preprocess` and `forward` are shared
    """

    _registry: Dict[str, type["BaseFeatureExtractor"]] = {}

    # Positional debiasing checked on this backend; unchecked backends warn in debias()
    debias_validated: bool = False

    # Tokens ahead of the patch grid in `_patch_tokens` output (CLS, registers)
    n_prefix_tokens: int = 0

    def __init__(self, resize_mode: str, image_resolution: int, svd_components: int = 500) -> None:
        """
        Store the shared preprocessing and debiasing configuration.

        Args:
            resize_mode: "max_size" (longest edge) or "square" (center-crop, then resize).
            image_resolution: longest-edge target for "max_size", side for "square".
            svd_components: positional-subspace rank for debias(); 500 is the INSID3 default.
        """
        super().__init__()

        self._resize_mode = resize_mode
        self._image_resolution = image_resolution
        self.svd_components = svd_components

        # Per patch-grid caches: positional basis (D, K) and zero-image features (D, H_p, W_p)
        self._pos_basis_cache: dict = {}
        self._zero_feats_cache: dict = {}

    @property
    def device(self) -> torch.device:
        """
        Device the backend's model was moved to at construction.

        Returns:
            `self._device`, set by each backend's `__init__`.
        """
        return self._device

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone on a preprocessed batch.

        Args:
            batch: (B, C, H, W) preprocessed images on `self.device`.

        Returns:
            (B, n_prefix_tokens + H_p * W_p, D) tokens; patch tokens last, in raster order.

        Raises:
            NotImplementedError: in a backend that does not override it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _patch_tokens")

    def forward(self, images: list) -> list[torch.Tensor]:
        """
        Preprocess, run the backbone, and reshape each image's patch tokens to a grid.

        - the backend's `n_prefix_tokens` (CLS, registers) are dropped
        - all images in one call must preprocess to the same size (they are stacked)

        Args:
            images: anything `open_image` accepts, one entry per frame.

        Returns:
            One (D, H_p, W_p) float32 CPU tensor per image, unit-norm per patch.

        Raises:
            ValueError: when the backbone's token count is not n_prefix_tokens + H_p * W_p.
        """
        logger.debug("[%s] Extracting features: %d images", type(self).__name__, len(images))

        # Preprocess and batch; torch.stack needs every image at one grid
        preprocessed = [self.preprocess(img) for img in images]
        batch = torch.stack(preprocessed).to(self.device)

        # Backbone tokens, prefix included
        with torch.no_grad():
            tokens_all = self._patch_tokens(batch)

        # One grid for the whole batch; any other token count means a wrong prefix
        _, H, W = batch.shape[1:]
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        if tokens_all.shape[1] != self.n_prefix_tokens + n_patches:
            raise ValueError(
                f"{type(self).__name__}: {tokens_all.shape[1]} tokens, expected n_prefix_tokens="
                f"{self.n_prefix_tokens} + {n_patches} patches for {H}x{W}"
            )

        # Drop the prefix and reshape each image to (D, H_p, W_p)
        patch_tokens = tokens_all[:, self.n_prefix_tokens :].cpu()
        return [_tokens_to_feature_map(tokens, H, W, self.patch_size) for tokens in patch_tokens]

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
        E = feat.reshape(D, -1).T.float()
        E = E - E.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(E, full_matrices=False)
        rgb = (E @ Vt[:3].T).detach().cpu().numpy()
        # Normalize to [0, 255]: shift to zero, scale by range, guard zero-variance case
        rgb -= rgb.min()
        rgb /= rgb.max() + 1e-8
        return (rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8)

    def _build_positional_basis(self, H_p: int, W_p: int) -> None:
        """
        Positional subspace at one patch grid, from a zero-pixel image (INSID3).

        - a black image carries no content, so its features are positional only
        - runs through self.forward(), so each backend preprocesses exactly as at inference
        - fills _pos_basis_cache and _zero_feats_cache at (H_p, W_p)

        Args:
            H_p: patch rows.
            W_p: patch columns.
        """
        # patch_size must be set; an AttributeError beats a wrong zero image
        if not hasattr(self, "patch_size"):
            raise AttributeError(
                f"{type(self).__name__} must define `self.patch_size` "
                "(the pixel stride of each patch token) before calling debias()."
            )
        patch_size = self.patch_size
        H_img = H_p * patch_size
        W_img = W_p * patch_size

        # Zero-pixel image through the backend's own forward()
        zero_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        zero_pil = Image.fromarray(zero_arr)

        with torch.no_grad():
            [zero_feat] = self.forward([zero_pil])

        if zero_feat.shape[1:] != (H_p, W_p):
            # Backend preprocessing changed the grid; bias is smooth, so interpolate
            zero_feat = F.interpolate(
                zero_feat.unsqueeze(0),
                size=(H_p, W_p),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        self._zero_feats_cache[(H_p, W_p)] = zero_feat

        # Top-K left singular vectors of the centered features span the positional subspace
        D = zero_feat.shape[0]
        E = zero_feat.reshape(D, -1)
        E = E - E.mean(dim=1, keepdim=True)

        U, _, _ = torch.linalg.svd(E, full_matrices=False)

        self._pos_basis_cache[(H_p, W_p)] = U[:, : self.svd_components].contiguous()

    def _apply_debias(self, fmap: torch.Tensor) -> torch.Tensor:
        """
        Project out the positional subspace, then re-normalize each patch.

        Args:
            fmap: (D, H_p, W_p) features at a grid with a cached basis.

        Returns:
            (D, H_p, W_p) debiased features, unit-norm per patch.
        """
        D, H_p, W_p = fmap.shape
        basis = self._pos_basis_cache[(H_p, W_p)].to(fmap.device)

        # P_perp = I - U U^T removes the span(U) component
        P_perp = torch.eye(D, device=fmap.device, dtype=fmap.dtype) - basis @ basis.T

        X = fmap.reshape(D, -1)
        X_deb = P_perp @ X

        # Projection breaks unit norm; cosine queries need it back
        X_deb = F.normalize(X_deb, p=2, dim=0)

        return X_deb.reshape(D, H_p, W_p)

    def get_bias_visualization(self, H_p: int, W_p: int) -> "np.ndarray":
        """
        RGB view of the positional bias at one patch grid (top-3 PCs of the zero image).

        - needs a prior debias() call at this grid

        Args:
            H_p: patch rows of a prior debias() call.
            W_p: patch columns of a prior debias() call.

        Returns:
            (H_p, W_p, 3) uint8 RGB.

        Raises:
            KeyError: when (H_p, W_p) is not cached.
        """
        if (H_p, W_p) not in self._zero_feats_cache:
            raise KeyError(
                f"No positional bias cached for patch grid ({H_p}, {W_p}). "
                "Call debias() at this resolution first."
            )
        return self.features_to_rgb(self._zero_feats_cache[(H_p, W_p)])

    def debias(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Remove positional bias from patch features by SVD projection (INSID3 algorithm).

        - operates on features already returned by forward()
        - the positional basis comes from a zero-pixel image through this extractor's own forward()
        - built once per patch-grid resolution, then cached and reused

        Args:
            features: (D, H_p, W_p) tensors from forward(); all must share (H_p, W_p).

        Returns:
            A new list, positional bias removed and each patch re-normalized.
        """
        if not self.debias_validated:
            # Checked only on DINO-family models; warn so other backends stay usable
            logger.warning(
                "[%s] Positional debiasing not yet validated for this extractor. "
                "Proceeding — results may be suboptimal.",
                type(self).__name__,
            )
        _, H_p, W_p = features[0].shape
        if (H_p, W_p) not in self._pos_basis_cache:
            # First call at this grid builds the basis
            self._build_positional_basis(H_p, W_p)
        return [self._apply_debias(f) for f in features]


########################################################
########## Queryable extractor base ####################
########################################################


class BaseQueryableExtractor(BaseFeatureExtractor, ABC):
    """
    Extractor that also embeds text, so patches can be queried by cosine similarity.

    - subclasses implement `encode_text` and `_patch_tokens`
    - `compute_similarity` and `score_queries` are shared
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
        Contrastive score of positive queries against negative ones.

        Args:
            features: (C, H, W) patch feature map, or (P, D) point feature array.
            positive: queries that should score high.
            negative: queries that should score low; None uses ["object"] (Talk2DINO's
                convention), [] skips the contrast and returns raw similarity.
            temperature: softmax temperature; lower is sharper.
            reduction: "max" or "pool", see `compute_semantic_contrast`.

        Returns:
            (H, W) for a patch map, (P,) for point features; in [0, 1] unless negative is [].
        """
        if negative is None:
            negative = ["object"]
        logger.debug(
            "[%s] Scoring queries: %d positive%s (reduction=%s)",
            type(self).__name__, len(positive),
            f", {len(negative)} negative" if negative else "",
            reduction,
        )
        queries = positive + negative
        similarity = self.compute_similarity(features, queries)
        result = compute_semantic_contrast(similarity, len(positive), temperature, reduction)
        return result
