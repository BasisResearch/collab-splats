"""
Camera localization: pose of a query image in a known reconstruction.

- retrieval: DINO-SALAD global descriptors rank reference frames
- extractors: vismatch xfeat / loma features, cached and matched pair by pair
- localizer: matched ref px -> world_points lookup -> pycolmap absolute pose
"""

from collab_splats.localization.extractors import (
    LocalFeatures,
    LocalMatcher,
    MatchResult,
)
from collab_splats.localization.localizer import (
    CameraLocalizer,
    LocalizationResult,
    localization_db_exists,
    read_localization_db,
    seed_intrinsics,
)
from collab_splats.localization.retrieval import (
    BaseRetrievalExtractor,
    DinoSaladExtractor,
    PECLIPExtractor,
)
from collab_splats.localization.viz import (
    correspondences_for_ref,
    plot_correspondences,
    plot_inlier_distribution,
)

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
    "localization_db_exists",
    "read_localization_db",
    "plot_correspondences",
    "plot_inlier_distribution",
    "seed_intrinsics",
]
