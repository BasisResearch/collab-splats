# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator

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
from .utils import compute_obb_from_points, get_points_in_mask

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}
if _OMEGA_AVAILABLE:
    _REGISTRY["vggt_omega"] = VGGTOmegaCreator
if _SPARK_AVAILABLE:
    _REGISTRY["vggt_spark"] = VGGTSPARKCreator


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


def make_creator(
    name: str,
    *,
    use_lc: bool = False,
    lc_config=None,
    **kwargs,
):
    """Construct a pointcloud creator, optionally wrapped with LoopClosure."""
    creator = get_creator(name)(**kwargs)
    if use_lc:
        # Deferred import — avoids circular dependency: geometry.loop_closure.wrapper
        # imports pointcloud.feedforward, so geometry cannot be imported at module load.
        from collab_splats.geometry import LoopClosure

        creator = LoopClosure(creator, config=lc_config)
    return creator


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
