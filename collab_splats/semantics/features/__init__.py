"""
Patch-feature extractors, looked up by name via `BaseFeatureExtractor.get`.

- dinov2: DINOv2 patch features, no text tower
- maskclip, talk2dino: queryable; they also embed text for `score_queries`
"""

from .base import BaseFeatureExtractor, BaseQueryableExtractor
from .dino import DINOFeatureExtractor
from .maskclip import MaskCLIPExtractor
from .talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
]
