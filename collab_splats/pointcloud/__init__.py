# collab_splats/pointcloud/__init__.py
from .base import BasePointcloudCreator, CoordinateFrame, PointcloudResult, _colmap_recon_to_result
from .sfm import ColmapCreator, HlocCreator
from .feedforward import BaseFeedforwardCreator, MapAnythingCreator, VGGTXCreator

try:
    from .feedforward import VGGTOmegaCreator
    _OMEGA_AVAILABLE = True
except ImportError:
    _OMEGA_AVAILABLE = False
from .bundle_adjustment import BundleAdjustmentConfig
from .loop_closure import LoopClosureConfig
from .wrappers import BundleAdjustment, LoopClosure
from .localization import (
    BaseRetrievalExtractor,
    CameraLocalizer,
    DiskExtractor,
    LocalFeatures,
    PECLIPExtractor,
    XFeatExtractor,
)
from .utils import compute_obb_from_points, get_points_in_mask

_REGISTRY: dict[str, type[BasePointcloudCreator]] = {
    "colmap":      ColmapCreator,
    "hloc":        HlocCreator,
    "mapanything": MapAnythingCreator,
    "vggtx":       VGGTXCreator,
}
if _OMEGA_AVAILABLE:
    _REGISTRY["vggt_omega"] = VGGTOmegaCreator


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
    use_ba: bool = False,
    lc_config=None,
    ba_config=None,
    **kwargs,
):
    """Construct a pointcloud creator, optionally wrapped with LoopClosure and/or BundleAdjustment."""
    creator = get_creator(name)(**kwargs)
    if use_lc:
        creator = LoopClosure(creator, config=lc_config)
    if use_ba:
        creator = BundleAdjustment(creator, config=ba_config)
    return creator


__all__ = [
    "BasePointcloudCreator",
    "BaseRetrievalExtractor",
    "BaseFeedforwardCreator",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "CameraLocalizer",
    "CoordinateFrame",
    "ColmapCreator",
    "DiskExtractor",
    "HlocCreator",
    "PECLIPExtractor",
    "LoopClosure",
    "LoopClosureConfig",
    "MapAnythingCreator",
    "PointcloudResult",
    "VGGTXCreator",
    "XFeatExtractor",
    "compute_obb_from_points",
    "get_points_in_mask",
    "get_creator",
    "make_creator",
]
