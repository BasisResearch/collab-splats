"""Camera localization pipeline: find pose of a query image in a known reconstruction.

Three-stage pipeline:

  Stage 1 — Global retrieval: top-K visually similar reference frames via compact
             global descriptors (DINOv2-SALAD). Current use: loop closure detection.

  Stage 2 — Local feature matching: vismatch-backed LocalMatcher (any
             vismatch model name, e.g. "loma", "xfeat", "disk-lightglue").
             Pairwise image-vs-image matching returning MatchResult pixel pairs.

  Stage 3 — Pose estimation: 2D→3D depth lookup — bilinear-sample each
             reference frame's dense world_points at matched ref pixels
             (hloc pose_from_cluster analog), then absolute pose via
             LO-RANSAC + Ceres refinement (pycolmap).
"""

from .extractors import LocalFeatures, LocalMatcher, MatchResult
from .localizer import (
    CameraLocalizer,
    LocalizationResult,
    load_reconstruction_features,
    sample_world_points,
)
from .retrieval import BaseRetrievalExtractor, DinoSaladExtractor, PECLIPExtractor
from .viz import correspondences_for_ref, plot_correspondences, plot_inlier_distribution

__all__ = [
    "BaseRetrievalExtractor",
    "CameraLocalizer",
    "DinoSaladExtractor",
    "LocalFeatures",
    "LocalMatcher",
    "LocalizationResult",
    "MatchResult",
    "PECLIPExtractor",
    "correspondences_for_ref",
    "load_reconstruction_features",
    "plot_correspondences",
    "plot_inlier_distribution",
    "sample_world_points",
]
