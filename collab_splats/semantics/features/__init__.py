"""collab_splats.semantics.features — feature extractor registry and backends."""

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
