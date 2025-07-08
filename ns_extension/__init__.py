"""ns-extension: Extension tools for nerfstudio"""

__version__ = "0.0.1"

from ns_extension.models.rade_gs_model import RadegsModel, RadegsModelConfig
# from ns_extension.submodules.tetra_triangulation.tetranerf.utils.extension import cpp
from ns_extension.utils.camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal
from ns_extension.utils.trainer_config import _TrainerConfig, _ExperimentConfig

__all__ = [
    "RadegsModelConfig",
    "RadegsModel",
    # "cpp",
    "ColmapCamera",
    "convert_to_colmap_camera",
    "depth_double_to_normal",
    "_TrainerConfig",
    "_ExperimentConfig"
]