# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, PointcloudResult, _colmap_recon_to_result
from .sfm import NerfstudioSfmCreator
from .feedforward import MapAnythingCreator

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "sfm": NerfstudioSfmCreator,
    "feedforward": MapAnythingCreator,
}


def get_creator(name: str) -> type[BasePointcloudCreator]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown pointcloud backend '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


__all__ = [
    "BasePointcloudCreator",
    "PointcloudResult",
    "NerfstudioSfmCreator",
    "MapAnythingCreator",
    "get_creator",
]
