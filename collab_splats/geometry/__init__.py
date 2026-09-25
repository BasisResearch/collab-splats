"""
Pose and geometry backend: bundle adjustment, loop closure, verification, scene metrics.

- transforms: pose conversions, Umeyama alignment, intrinsics from points
- bundle_adjustment: LM refinement of a `pointcloud.zarr` result (feedforward only at refine)
- verification / metrics: report-only geometric checks over a `pointcloud.zarr` result
- loop_closure: submap pose graph around a feedforward creator's forward pass
"""

from .bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from .loop_closure import PoseGraph, Submap
from .transforms import (
    OPENGL_TO_OPENCV,
    estimate_intrinsics_from_points,
    extract_intrinsics,
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
    rotation_angle_deg,
)


def __getattr__(name: str) -> type:
    # Lazy import breaks a load-time cycle
    # - loop_closure.wrapper -> collab_splats.pointcloud -> geometry.transforms
    # - delegates to loop_closure's own lazy hook
    if name == "LoopClosure":
        from .loop_closure import LoopClosure

        return LoopClosure
    if name == "LoopClosureConfig":
        from .loop_closure import LoopClosureConfig

        return LoopClosureConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OPENGL_TO_OPENCV",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "LoopClosure",
    "LoopClosureConfig",
    "PoseGraph",
    "Submap",
    "estimate_intrinsics_from_points",
    "extract_intrinsics",
    "extrinsics_to_homogeneous",
    "invert_poses",
    "rotation_align_vectors",
    "rotation_angle_deg",
]
