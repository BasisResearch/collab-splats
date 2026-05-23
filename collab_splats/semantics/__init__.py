"""
collab_splats.semantics — feature extraction, segmentation, and query interfaces.
"""

from .features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
)
from .utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
    pytorch_gc,
    infer_batch_size,
    load_hf_weights,
    load_torchhub_model,
    batch_iterator,
)
from .compression import FeatureAutoencoder
from .segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
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
    # compression
    "FeatureAutoencoder",
    # extractors
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    # semantics utilities
    "compute_semantic_contrast",
    "interpolate_to_patch_size",
    "pytorch_gc",
    "infer_batch_size",
    "load_hf_weights",
    "load_torchhub_model",
    "batch_iterator",
    # segmentation
    "BaseSegmentation",
    "MobileSAMSegmentation",
    "SAM3Segmentation",
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
