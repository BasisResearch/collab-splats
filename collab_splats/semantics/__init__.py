"""
collab_splats.semantics — feature extraction, segmentation, and query interfaces.
"""

from .compression import FeatureAutoencoder
from .features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
    DINOFeatureExtractor,
    MaskCLIPExtractor,
    Talk2DinoExtractor,
)
from .segmentation import (
    BaseSegmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
    aggregate_masked_features,
    convert_matched_mask,
    create_composite_mask,
    create_patch_mask,
    load_mobile_sam,
    mask_id_to_binary_mask,
)
from .utils import (
    ae_path,
    cache_store_path,
    compute_semantic_contrast,
    extract_feature_cache,
    find_lifted_extractor,
    lifted_store_path,
    load_feature_maps,
    load_point_features,
    point_features_cached,
    tokens_to_feature_map,
    write_point_features,
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
    # semantic helpers + artifact layout
    "compute_semantic_contrast",
    "tokens_to_feature_map",
    "cache_store_path",
    "extract_feature_cache",
    "load_feature_maps",
    "write_point_features",
    "load_point_features",
    "point_features_cached",
    "lifted_store_path",
    "ae_path",
    "find_lifted_extractor",
    # segmentation
    "BaseSegmentation",
    "MobileSAMSegmentation",
    "SAM3Segmentation",
    "load_mobile_sam",
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
]
