"""
Semantic features for reconstructed scenes: extract, compress, store, query, segment.

- features: patch-feature extractors (dinov2, maskclip, talk2dino) behind one registry
- segmentation: mask backends (insid3, mobilesamv2, sam3, skywater) behind one registry
- compression: FeatureAutoencoder, per-point codes for the lifted store
- store: the 2D cache and the lifted per-point store
- utils: contrastive scoring and point clustering
"""

from collab_splats.semantics.compression import FeatureAutoencoder
from collab_splats.semantics.features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
    DINOFeatureExtractor,
    MaskCLIPExtractor,
    Talk2DinoExtractor,
)
from collab_splats.semantics.segmentation import (
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
from collab_splats.semantics.store import (
    read_point_features,
    valid_feature_cache,
    write_feature_cache,
    write_point_features,
)
from collab_splats.semantics.utils import cluster_points, compute_semantic_contrast

__all__ = [
    # compression
    "FeatureAutoencoder",
    # extractors
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    # feature math
    "compute_semantic_contrast",
    "cluster_points",
    # stores
    "write_feature_cache",
    "valid_feature_cache",
    "write_point_features",
    "read_point_features",
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
