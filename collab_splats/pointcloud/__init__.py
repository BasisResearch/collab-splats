"""
Pointcloud reconstruction: the creator registry and its two backend families.

- feedforward backbones (vggtx, mapanything, vggt_omega, vggt_spark, loger) and sfm
  backends (colmap, hloc) all resolve through get_creator / make_creator
- the optional backbones register only when their dependencies import
- every creator returns a PointcloudResult; see base.py for that contract
"""
from .base import BasePointcloudCreator, PointcloudResult
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

    - not the `pointcloud.method: sfm` allowlist: colmap/hloc resolve here, but
      Reconstructor.validate_config rejects them as sfm backends

    Args:
        name: colmap, hloc, mapanything or vggtx, plus vggt_omega / vggt_spark / loger
            when their optional deps are installed.

    Returns:
        The creator class.

    Raises:
        KeyError: listing the available names, when the key is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def make_creator(name: str, **kwargs) -> BasePointcloudCreator:
    """
    Construct a registered pointcloud creator.

    - for loop closure, wrap it yourself:
      `collab_splats.geometry.LoopClosure(creator, config=...)`

    Args:
        name: registry key; see get_creator.
        kwargs: forwarded to the creator's constructor.

    Returns:
        The creator instance.
    """
    return get_creator(name)(**kwargs)


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "ColmapCreator",
    "HlocCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "get_creator",
    "make_creator",
]
