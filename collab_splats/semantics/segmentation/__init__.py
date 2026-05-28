"""Segmentation backends and mask utilities.

Import from here — submodule structure is an implementation detail.
"""
from __future__ import annotations

# ── Abstract base and mask utilities ──────────────────────────────────────────
from .base import (
    BaseSegmentation,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
    aggregate_masked_features,
)

# ── Concrete backends ─────────────────────────────────────────────────────────
from .mobile_sam import MobileSAMSegmentation, load_mobile_sam
from .sam3 import SAM3Segmentation
from .insid3 import INSID3Segmentation

__all__ = [
    # abstract base
    "BaseSegmentation",
    # mask utilities
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
    # backends
    "MobileSAMSegmentation",
    "load_mobile_sam",
    "SAM3Segmentation",
    "INSID3Segmentation",
]
