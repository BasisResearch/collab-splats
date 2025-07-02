"""RaDeGS: Rasterizing Depth in Gaussian Splatting"""

__version__ = "0.0.1"

from .rade_gs.models import RadegsModel, RadegsModelConfig
from .rade_gs.submodules.tetra_triangulation.tetranerf.utils.extension import cpp

__all__ = [
    "RadegsModel",
    "RadegsModelConfig",
    "cpp",
]