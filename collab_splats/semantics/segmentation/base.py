"""
Base class and mask utilities for segmentation backends.

- BaseSegmentation: abstract registry-based segmentation interface
- create_patch_mask: divide an image into a spatial patch grid
- create_composite_mask: merge SAM results into a single integer-ID mask
- mask_id_to_binary_mask: expand an integer-ID mask to an (N, H, W) boolean array
- convert_matched_mask: remap sequential IDs to matched label IDs
- aggregate_masked_features: pool features per segment mask
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
    def segment(self, image: np.ndarray | Image.Image) -> tuple[torch.Tensor, Any] | None:
        """
        Class-agnostic segmentation with no prompt.

        Args:
            image: (H, W, 3) uint8 array or PIL Image, per backend.

        Returns:
            (masks, metadata) — masks rank and dtype are backend-specific: (H, W) bool
            for insid3, (N, H, W) float32 for mobilesamv2, (N, 1, H, W) float32 for
            sam3, and mobilesamv2 returns None outright when nothing is detected.
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


def create_composite_mask(results: list[dict], confidence_threshold: float = 0.85) -> np.ndarray:
    """
    Merge SAM segment results into a single (H, W) uint16 mask of integer IDs.

    - uint16 preserves IDs > 255: SAM's 32x32 point grid routinely clears 256 proposals
    - numpy>=2 raises OverflowError on the 256th rather than wrapping around

    Args:
        results: dicts from a SAM mask generator; each needs "segmentation" (H, W) bool
            and "predicted_iou" float.
        confidence_threshold: masks with iou below this are discarded.

    Returns:
        (H, W) uint16 array — pixel value is the 1-indexed mask ID, 0 is background.
    """
    selected_masks = []
    for mask in results:
        if mask["predicted_iou"] < confidence_threshold or mask["predicted_iou"] > 1.0:
            continue
        selected_masks.append((mask["segmentation"], mask["predicted_iou"]))

    if not selected_masks:
        return (
            np.zeros_like(results[0]["segmentation"], dtype=np.uint16) if results else np.zeros((0, 0), dtype=np.uint16)
        )
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

        # Paint ID `idx` was written from masks[sorted_idxs[idx - 1]] — not masks[idx - 1],
        # which is a different mask whenever the confidences are not already ascending.
        if mask.sum() > 0 and (mask.sum() / masks[sorted_idxs[idx - 1]].sum()) > 0.1:
            composite_mask[mask] = i

    return composite_mask


def mask_id_to_binary_mask(composite_mask: np.ndarray) -> np.ndarray:
    """
    Expand an integer-ID mask to a (N, H, W) boolean array.

    Args:
        composite_mask: (H, W) uint16 array where each unique positive integer
                        represents a separate object mask.

    Returns:
        (N, H, W) bool array where N is the number of distinct mask IDs.
    """
    unique_ids = np.unique(composite_mask)
    unique_ids = unique_ids[unique_ids > 0]
    binary_masks = composite_mask[None, ...] == unique_ids[:, None, None]
    return binary_masks


def convert_matched_mask(labels: torch.Tensor, masks: np.ndarray) -> np.ndarray:
    """
    Remap sequential mask IDs to matched label IDs.

    Args:
        labels: (N,) tensor of matched labels, one per mask ID.
        masks:  (H, W) array of sequential mask IDs from 1 to N.

    Returns:
        (H, W) uint16 array with IDs replaced by matched labels.
        uint16 preserves label IDs > 255.
    """
    assert labels.shape[0] == np.max(masks), (
        "Number of labels must match number of unique masks"
    )

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
        masks:    (N, H, W) segmentation masks from SAM.
        resolution: Intermediate spatial resolution for feature interpolation.
        final_resolution: Output spatial resolution.

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
