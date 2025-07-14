"""ns-extension: Extension tools for nerfstudio"""

__version__ = "0.0.1"

from ns_extension.models.rade_gs_model import RadegsModel, RadegsModelConfig
from ns_extension.models.rade_features_model import RadegsFeaturesModel, RadegsFeaturesModelConfig

from ns_extension.wrapper.splatter import Splatter, SplatterConfig

# from ns_extension.submodules.tetra_triangulation.tetranerf.utils.extension import cpp
from ns_extension.utils.camera_utils import ColmapCamera
from ns_extension.utils.trainer_config import _TrainerConfig, _ExperimentConfig

__all__ = [
    # General wrapper class for running splats
    "SplatterConfig",
    "Splatter",
    "RadegsModelConfig",
    "RadegsModel",
    "RadegsFeaturesModelConfig",
    "RadegsFeaturesModel",
    # "cpp",
    "ColmapCamera",
    "_TrainerConfig",
    "_ExperimentConfig"
]