"""collab_splats.semantics.features — feature extractor registry and backends."""

from .base import BaseFeatureExtractor, BaseQueryableExtractor, _DEBIAS_VALIDATED, TORCH_HOME
from .maskclip import MaskCLIPExtractor
from .dino import DINOFeatureExtractor
from .talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    "_DEBIAS_VALIDATED",
    "TORCH_HOME",
]
