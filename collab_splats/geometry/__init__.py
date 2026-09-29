"""
Pose and geometry backend: bundle adjustment, loop closure, scene metrics.

- transforms: pose conversions, Umeyama, intrinsics, decompose_camera, project_to_so3
- bundle_adjustment: LM refinement on arrays (feedforward only at refine)
- metrics: report-only quality tables on arrays; the Reconstructor stage owns the zarr
- loop_closure: submap pose graph around a feedforward creator's forward pass
"""

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
)
from collab_splats.geometry.loop_closure import PoseGraph, Submap
from collab_splats.geometry.transforms import (
    OPENGL_TO_OPENCV,
    estimate_intrinsics_from_points,
    extract_intrinsics,
    extrinsics_to_homogeneous,
    fit_dominant_plane,
    invert_poses,
    rotation_align_vectors,
    rotation_angle_deg,
)


def __getattr__(name: str) -> type:
    """
    Lazy import of LoopClosure and LoopClosureConfig to break a load-time cycle.

    - cycle: loop_closure.wrapper -> collab_splats.pointcloud -> geometry.transforms
    - delegates to loop_closure's own lazy hook

    Args:
        name: attribute requested from the package.

    Returns:
        The requested class.

    Raises:
        AttributeError: any other name.
    """
    if name == "LoopClosure":
        from collab_splats.geometry.loop_closure import LoopClosure

        return LoopClosure
    if name == "LoopClosureConfig":
        from collab_splats.geometry.loop_closure import LoopClosureConfig

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
    "fit_dominant_plane",
    "invert_poses",
    "rotation_align_vectors",
    "rotation_angle_deg",
]
