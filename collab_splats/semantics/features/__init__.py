"""
Patch-feature extractors, looked up by name via `BaseFeatureExtractor.get`.

- dinov2: DINOv2 patch features, no text tower
- maskclip, talk2dino: queryable; they also embed text for `score_queries`
"""

from collab_splats.semantics.features.base import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
)
from collab_splats.semantics.features.dino import DINOFeatureExtractor
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
]
