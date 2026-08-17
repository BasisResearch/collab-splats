"""Camera localization pipeline: find pose of a query image in a known reconstruction.

Three-stage pipeline:

  Stage 1 — Global retrieval: top-K visually similar reference frames via compact
             global descriptors (DINOv2-SALAD). Current use: loop closure detection.

  Stage 2 — Local feature extraction + matching: DISK+LightGlue (default),
             XFeat+MNN, XFeat* semi-dense, or LoMa/LoMa-G. Each extractor owns
             its detect→match logic and returns MatchResult pixel pairs.

  Stage 3 — Pose estimation: 2D→3D depth lookup — bilinear-sample each
             reference frame's dense world_points at matched ref pixels
             (hloc pose_from_cluster analog), then absolute pose via
             LO-RANSAC + Ceres refinement (pycolmap).
"""

from .extractors import (
    BaseLocalExtractor,
    DiskExtractor,
    LocalFeatures,
    LomaExtractor,
    LomaGExtractor,
    MatchResult,
    XFeatExtractor,
    XFeatStarExtractor,
)
from .localizer import (
    CameraLocalizer,
    LocalizationResult,
    load_reconstruction_features,
    sample_world_points,
)
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
    "MatchResult",
    "PECLIPExtractor",
    "XFeatExtractor",
    "XFeatStarExtractor",
    "correspondences_for_ref",
    "load_reconstruction_features",
    "plot_correspondences",
    "plot_inlier_distribution",
    "sample_world_points",
]
