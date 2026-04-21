# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap": ColmapCreator,
    "hloc": HlocCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    """Get a pointcloud creator by name.

    Args:
        name: Creator name ('colmap', 'hloc', 'feedforward', etc.)

    Returns:
        The creator class

    Raises:
        KeyError: If creator name not found
    """
    # Lazy import feedforward to avoid heavy dependencies
    if name == "feedforward" and name not in _REGISTRY:
        try:
            from .feedforward import MapAnythingCreator
            _REGISTRY[name] = MapAnythingCreator
        except (ImportError, AttributeError):
            pass

    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "BasePointcloudCreator",
    "PointcloudResult",
    "ColmapCreator",
    "HlocCreator",
    "get_creator",
]
