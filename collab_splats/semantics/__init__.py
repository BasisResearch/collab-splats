"""
collab_splats.semantics — feature extraction, segmentation, and capability protocols.
"""

from .features import (
    BaseFeatureExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
    load_hf_weights,
    load_torchhub_model,
    pytorch_gc,
    resize_image,
    interpolate_to_patch_size,
    batch_iterator,
)
from .segmentation import (
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
from .protocols import SupportsTextQuery
from .frame_sampling import (
    OpticalFlowFrameSelector,
    sample_frames_fps,
    sample_frames_optical_flow,
)

__all__ = [
    # extractors
    "BaseFeatureExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    # helpers
    "load_hf_weights",
    "load_torchhub_model",
    "pytorch_gc",
    "resize_image",
    "interpolate_to_patch_size",
    "batch_iterator",
    # segmentation
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
    # protocols
    "SupportsTextQuery",
    # frame sampling
    "OpticalFlowFrameSelector",
    "sample_frames_fps",
    "sample_frames_optical_flow",
]
