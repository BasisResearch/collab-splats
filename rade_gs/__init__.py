"""RaDeGS: Rasterizing Depth in Gaussian Splatting"""

__version__ = "0.0.1"

from rade_gs.models.rade_gs_model import RadegsModel, RadegsModelConfig
from rade_gs.submodules.tetra_triangulation.tetranerf.utils.extension import cpp
from rade_gs.utils.camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal

__all__ = [
    "RadegsModelConfig",
    "RadegsModel",
    "cpp",
    "ColmapCamera",
    "convert_to_colmap_camera",
    "depth_double_to_normal"
]