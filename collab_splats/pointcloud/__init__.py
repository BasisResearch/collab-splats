# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator
from .sfm import ColmapCreator, HlocCreator

try:
    from .feedforward import VGGTOmegaCreator

    _OMEGA_AVAILABLE = True
except ImportError:
    _OMEGA_AVAILABLE = False

try:
    from .feedforward import VGGTSPARKCreator

    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False

try:
    from .feedforward import LoGeRCreator

    _LOGER_AVAILABLE = True
except ImportError:
    _LOGER_AVAILABLE = False
from .utils import compute_obb_from_points, get_points_in_mask

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap": ColmapCreator,
    "hloc": HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx": VGGTXCreator,
}
if _OMEGA_AVAILABLE:
    _REGISTRY["vggt_omega"] = VGGTOmegaCreator
if _SPARK_AVAILABLE:
    _REGISTRY["vggt_spark"] = VGGTSPARKCreator
if _LOGER_AVAILABLE:
    _REGISTRY["loger"] = LoGeRCreator


def get_creator(name: str) -> type[BasePointcloudCreator]:
    """
    Look up a pointcloud creator class by registry name.

    - name: one of _REGISTRY's keys (colmap, hloc, mapanything, vggtx, plus vggt_omega /
      vggt_spark / loger when their optional deps are installed).
    - Returns the class; raises KeyError with the available names when unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def make_creator(name: str, **kwargs) -> BasePointcloudCreator:
    """
    Construct a registered pointcloud creator.

    - name: registry key; see get_creator.
    - kwargs: forwarded to the creator's constructor.
    - Returns the creator instance. Wrap it in LoopClosure yourself if you want loop closure.
    """
    return get_creator(name)(**kwargs)


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "CoordinateFrame",
    "ColmapCreator",
    "HlocCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "compute_obb_from_points",
    "get_points_in_mask",
    "get_creator",
    "make_creator",
]
