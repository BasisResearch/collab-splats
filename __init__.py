"""RaDeGS: Rasterizing Depth in Gaussian Splatting"""

__version__ = "0.0.1"

from .rade_gs.models import RaDeGSModel, RaDeGSModelConfig
from .rade_gs.utils import depths_double_to_points, point_double_to_normal, depth_double_to_normal

__all__ = [
    "RaDeGSModel",
    "RaDeGSModelConfig",
    "depths_double_to_points",
    "point_double_to_normal",
    "depth_double_to_normal",
]