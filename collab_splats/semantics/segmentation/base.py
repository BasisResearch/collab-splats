"""
Segmentation base class and mask utilities.

- BaseSegmentation: backend registry and interface
- create_composite_mask, mask_id_to_binary_mask, convert_matched_mask: integer-ID masks
- create_patch_mask, aggregate_masked_features: patch grids and per-mask feature pooling
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from collab_splats.utils.torch_utils import RegistryMixin

logger = logging.getLogger(__name__)


########################################################
########## Registry and abstract base ##################
########################################################


class BaseSegmentation(RegistryMixin, ABC):
    """
    Abstract base for segmentation backends with a name-based registry.

    - register subclasses via `@BaseSegmentation.register("name")`
    - retrieve with `.get("name")`
    """

    _registry: dict[str, type["BaseSegmentation"]] = {}

    @abstractmethod
    def segment(self, image: np.ndarray | Image.Image) -> tuple[torch.Tensor, Any]:
        """
        Segment one frame with the backend's default prompt or cached context.

        Args:
            image: the frame; accepted types are per backend (ndarray, PIL image, or tensor).

        Returns:
            (masks, metadata) — masks rank and dtype are backend-specific: (H, W) bool
            for insid3 and skywater, (N, H, W) float32 for mobilesamv2, (N, 1, H, W) bool
            for sam3.
            metadata is backend-specific.
        """

    def segment_with_text(
        self, image: Image.Image, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Text-prompted segmentation.

        Args:
            image: the frame to segment.
            prompt: text prompt.

        Returns:
            (masks, boxes, scores) from the backend.

        Raises:
            NotImplementedError: for backends without text prompts.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support text-prompted segmentation. "
            "Use backend='sam3'."
        )


########################################################
########## Mask utilities ##############################
########################################################


def create_patch_mask(image: np.ndarray, num_patches: int = 32) -> torch.Tensor:
    """
    Divide an image into a spatial patch grid; return a boolean occupancy tensor.

    Args:
        image: (H, W, ...) array — only H and W are read.
        num_patches: patches along each axis.

    Returns:
        (num_patches, num_patches, H*W) bool tensor — True where a pixel belongs to a patch.
    """
    H, W = image.shape[:2]

    patch_width = math.ceil(W / num_patches)
    patch_height = math.ceil(H / num_patches)

    total_pixels = H * W
    y_coords = torch.arange(H).unsqueeze(1).expand(-1, W).flatten()
    x_coords = torch.arange(W).unsqueeze(0).expand(H, -1).flatten()

    patch_y_indices = torch.clamp(y_coords // patch_height, 0, num_patches - 1)
    patch_x_indices = torch.clamp(x_coords // patch_width, 0, num_patches - 1)

    flatten_patch_mask = torch.zeros(
        (num_patches, num_patches, total_pixels), dtype=torch.bool
    )

    pixel_indices = torch.arange(total_pixels)
    flatten_patch_mask[patch_y_indices, patch_x_indices, pixel_indices] = True

    return flatten_patch_mask


def create_composite_mask(
    results: list[dict], confidence_threshold: float = 0.85, min_visible_frac: float = 0.1
) -> np.ndarray:
    """
    Merge SAM results into one (H, W) uint16 mask of integer IDs.

    - higher-confidence masks paint over lower ones; a mask left with ≤min_visible_frac of its pixels is dropped
    - uint16: SAM routinely yields >255 masks, and numpy>=2 raises on uint8 overflow
    - IDs skip dropped masks, so they are not always contiguous

    Args:
        results: SAM mask-generator dicts with "segmentation" (H, W) bool and "predicted_iou".
        confidence_threshold: drop masks with predicted_iou below this (or above 1).
        min_visible_frac: drop a mask left with this share of its pixels or less.

    Returns:
        (H, W) uint16; 1-indexed mask IDs, 0 is background.

    Raises:
        ValueError: when results is empty.
    """
    if not results:
        raise ValueError("create_composite_mask: no results to merge")

    selected_masks = []
    for mask in results:
        if mask["predicted_iou"] < confidence_threshold or mask["predicted_iou"] > 1.0:
            continue
        selected_masks.append((mask["segmentation"], mask["predicted_iou"]))

    if not selected_masks:
        return np.zeros_like(results[0]["segmentation"], dtype=np.uint16)
    masks, confs = zip(*selected_masks)

    H, W = masks[0].shape[:2]
    mask_id = np.zeros((H, W), dtype=np.uint16)

    sorted_idxs = np.argsort(confs)
    for i, idx in enumerate(sorted_idxs, start=1):
        current_mask = masks[idx]
        mask_id[current_mask == 1] = i

    mask_indices = np.unique(mask_id)
    mask_indices = np.setdiff1d(mask_indices, [0])

    composite_mask = np.zeros((H, W), dtype=np.uint16)

    for i, idx in enumerate(mask_indices, start=1):
        mask = mask_id == idx
        logger.debug("Mask %d has %d pixels", i, mask.sum())

        # ID idx was painted from masks[sorted_idxs[idx - 1]], not masks[idx - 1]
        if mask.sum() > 0 and (mask.sum() / masks[sorted_idxs[idx - 1]].sum()) > min_visible_frac:
            composite_mask[mask] = i

    return composite_mask


def mask_id_to_binary_mask(composite_mask: np.ndarray) -> np.ndarray:
    """
    Expand an integer-ID mask to a (N, H, W) boolean array.

    Args:
        composite_mask: (H, W) integer-ID mask; 0 is background.

    Returns:
        (N, H, W) bool array where N is the number of distinct mask IDs.
    """
    unique_ids = np.unique(composite_mask)
    unique_ids = unique_ids[unique_ids > 0]
    binary_masks = composite_mask[None, ...] == unique_ids[:, None, None]
    return binary_masks


def convert_matched_mask(labels: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    """
    Remap sequential mask IDs 1..N to matched label IDs.

    Args:
        labels: (N,) matched label per mask ID; label k is written as k + 1.
        masks: (H, W) sequential mask IDs, 1..N.

    Returns:
        (H, W) uint16 with each ID replaced by its label + 1.

    Raises:
        ValueError: when the label count differs from the highest mask ID.
    """
    if labels.shape[0] != np.max(masks):
        raise ValueError(f"{labels.shape[0]} labels for {int(np.max(masks))} mask IDs")

    matched_mask = np.zeros(masks.shape, dtype=np.uint16)

    for label_idx in range(labels.shape[0]):
        mask_id = label_idx + 1
        matched_label = labels[label_idx].item() + 1
        matched_mask[masks == mask_id] = matched_label

    return matched_mask  # uint16 — preserves IDs > 255


def aggregate_masked_features(
    features: torch.Tensor,
    masks: torch.Tensor,
    resolution: Tuple[int, int],
    final_resolution: Tuple[int, int],
) -> torch.Tensor:
    """
    Pool image features per segment mask and return a spatial feature map.

    Args:
        features: (C, H, W) image feature tensor.
        masks: (N, H, W) segmentation masks from SAM.
        resolution: intermediate spatial resolution for feature interpolation.
        final_resolution: output spatial resolution.

    Returns:
        (C, H, W) aggregated feature map at final_resolution.
    """
    features = F.interpolate(
        features.unsqueeze(0), size=resolution, mode="bilinear", align_corners=False
    )[0]

    masks = F.interpolate(masks.unsqueeze(1), size=resolution, mode="nearest").bool()[:, 0]
    masks = masks.to(features.device)

    weighted_features = torch.einsum("nhw,chw->chw", masks.float(), features)
    mask_counts = masks.sum(0).float()
    aggregated_feat_map = weighted_features / (mask_counts + 1e-6).unsqueeze(0)

    aggregated_feat_map = F.interpolate(
        aggregated_feat_map.unsqueeze(0),
        size=final_resolution,
        mode="bilinear",
        align_corners=False,
    )[0]

    return aggregated_feat_map
