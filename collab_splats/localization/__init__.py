"""Camera localization pipeline: find pose of a query image in a known reconstruction.

Three-stage pipeline:

  Stage 1 — Global retrieval: top-K visually similar reference frames via compact
             global descriptors (DINOv2-SALAD). Current use: loop closure detection.

  Stage 2 — Local feature extraction + matching: DISK+LightGlue (default) or
             XFeat+MNN. Each extractor owns its detect→match logic.

  Stage 3 — Pose estimation: 2D→3D keypoint assignment via torch.cdist NN,
             then absolute pose via LO-RANSAC + Ceres refinement (pycolmap).
"""

from .extractors import (
    BaseLocalExtractor,
    DiskExtractor,
    LocalFeatures,
    LomaExtractor,
    LomaGExtractor,
    XFeatExtractor,
)
from .localizer import CameraLocalizer, LocalizationResult
from .retrieval import BaseRetrievalExtractor, DinoSaladExtractor, PECLIPExtractor
from .viz import correspondences_for_ref, plot_correspondences, plot_inlier_distribution

__all__ = [
    "BaseLocalExtractor",
    "BaseRetrievalExtractor",
    "CameraLocalizer",
    "DinoSaladExtractor",
    "DiskExtractor",
    "LocalFeatures",
    "LocalizationResult",
    "LomaExtractor",
    "LomaGExtractor",
    "PECLIPExtractor",
    "XFeatExtractor",
    "correspondences_for_ref",
    "plot_correspondences",
    "plot_inlier_distribution",
]
