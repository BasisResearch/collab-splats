"""
Semantic features for reconstructed scenes: extract, compress, store, query, segment.

- features: patch-feature extractors (dinov2, maskclip, talk2dino) behind one registry
- segmentation: mask backends (insid3, mobilesamv2, sam3, skywater) behind one registry
- compression: FeatureAutoencoder, per-point codes for the lifted store
- utils: on-disk layout of the 2D cache and the lifted per-point store
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
    INSID3Segmentation,
    MobileSAMSegmentation,
    SAM3Segmentation,
    SkyWaterSegmentation,
    aggregate_masked_features,
    convert_matched_mask,
    create_composite_mask,
    create_patch_mask,
    mask_id_to_binary_mask,
    sky_masks,
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
    "INSID3Segmentation",
    "SkyWaterSegmentation",
    "sky_masks",
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
]
