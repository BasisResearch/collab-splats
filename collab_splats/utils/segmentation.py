"""
Backward-compatibility shim.

All segmentation code now lives in collab_splats.semantics.segmentation.
This module re-exports everything so existing callers require no changes.
"""

from collab_splats.semantics.segmentation import (
    Segmentation,
    load_mobile_sam,
    auto_segment_image,
    get_object_masks,
    object_segment_image,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
    aggregate_masked_features,
)

__all__ = [
    "Segmentation",
    "load_mobile_sam",
    "auto_segment_image",
    "get_object_masks",
    "object_segment_image",
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
]
