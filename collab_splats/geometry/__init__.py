"""
Pose and geometry backend: shared camera math, bundle adjustment, loop closure, scene metrics.

- transforms: pose conversions, Umeyama, intrinsics, decompose_camera, project_to_so3
- projection: unproject / project, world-point lookup, cross-view depth agreement
- tracks: BA track extraction (VGGSfM or matcher star tracks), zarr-cached
- bundle_adjustment: LM solve over given tracks (feedforward only at refine)
- photometric: brightness residual and pixel samples for the BA photometric term
- metrics: report-only quality tables on arrays; loop_closure: submap pose graph around a creator
"""

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
)
from collab_splats.geometry.projection import reprojection_error


def __getattr__(name: str) -> type:
    """
    Lazy LoopClosure and LoopClosureConfig, breaking a load-time cycle.

    - cycle: loop_closure.wrapper -> collab_splats.pointcloud -> geometry.transforms

    Args:
        name: attribute requested from the package.

    Returns:
        The requested class.

    Raises:
        AttributeError: any other name.
    """
    if name in ("LoopClosure", "LoopClosureConfig"):
        # Import on first access to break the load-time cycle
        from collab_splats.geometry import loop_closure

        return getattr(loop_closure, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "LoopClosure",
    "LoopClosureConfig",
    "reprojection_error",
]
