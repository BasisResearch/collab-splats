# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    """Get a pointcloud creator by name.

    Args:
        name: Creator name ('colmap', 'hloc', 'mapanything', 'vggtx')

    Returns:
        The creator class

    Raises:
        KeyError: If creator name not found
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "BasePointcloudCreator",
    "BaseFeedforwardCreator",
    "CoordinateFrame",
    "ColmapCreator",
    "HlocCreator",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "get_creator",
]
