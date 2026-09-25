"""
Segmentation backends, looked up by name via `BaseSegmentation.get`, plus mask utilities.

- insid3: in-context masks from one reference image and mask
- mobilesamv2, sam3: class-agnostic instance masks; sam3 also takes text prompts
- skywater: per-pixel sky masks; `sky_masks` caches sky probability per scene
"""

from __future__ import annotations

########################################################################
# Abstract base and mask utilities
########################################################################
from .base import (
    BaseSegmentation,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
    aggregate_masked_features,
)

########################################################################
# Concrete backends
########################################################################
from .mobile_sam import MobileSAMSegmentation
from .sam3 import SAM3Segmentation
from .insid3 import INSID3Segmentation
from .sky import SkyWaterSegmentation, sky_masks

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
    "SAM3Segmentation",
    "INSID3Segmentation",
    "SkyWaterSegmentation",
    "sky_masks",
]
