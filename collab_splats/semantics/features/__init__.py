"""
Patch-feature extractors, looked up by name via `BaseFeatureExtractor.get`.

- dinov2: DINOv2 patch features, no text tower
- maskclip, talk2dino: queryable; they also embed text for `score_queries`
- ocr_lens: LLaVA-1.6 OCR-head verbalization lens; decode with word_vocabulary + verbalize
"""

from collab_splats.semantics.features.base import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
)
from collab_splats.semantics.features.dino import DINOFeatureExtractor
from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
from collab_splats.semantics.features.ocr_lens import (
    OCRLensExtractor,
    WordVocab,
    load_decoder,
    score_ocr_heads,
    verbalize,
    word_vocabulary,
)
from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

__all__ = [
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
    "MaskCLIPExtractor",
    "DINOFeatureExtractor",
    "Talk2DinoExtractor",
    "OCRLensExtractor",
    "WordVocab",
    "load_decoder",
    "score_ocr_heads",
    "verbalize",
    "word_vocabulary",
]
