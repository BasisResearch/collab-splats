"""collab-splats: Extension tools for nerfstudio"""

__version__ = "0.0.1"

from collab_splats.models.rade_gs_model import RadegsModel, RadegsModelConfig
from collab_splats.models.rade_features_model import (
    RadegsFeaturesModel,
    RadegsFeaturesModelConfig,
)

from collab_splats.wrapper.splatter import Splatter, SplatterConfig

from collab_splats.utils.camera_utils import ColmapCamera
from collab_splats.utils.trainer_config import _TrainerConfig, _ExperimentConfig

__all__ = [
    # General wrapper class for running splats
    "SplatterConfig",
    "Splatter",
    "RadegsModelConfig",
    "RadegsModel",
    "RadegsFeaturesModelConfig",
    "RadegsFeaturesModel",
    "ColmapCamera",
    "_TrainerConfig",
    "_ExperimentConfig",
]
