"""Geometry backend: loop closure, bundle adjustment, and SE(3)/pose transforms."""

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


def __getattr__(name):
    # Lazy import — loop_closure.wrapper imports collab_splats.pointcloud, which
    # imports geometry.transforms; an eager import here would cycle at load time.
    # Delegates to loop_closure's own lazy hook (one canonical lazy site).
    if name == "LoopClosure":
        from .loop_closure import LoopClosure

        return LoopClosure
    if name == "LoopClosureConfig":
        from .loop_closure import LoopClosureConfig

        return LoopClosureConfig
    if name == "run_global_alignment":
        # Parked VGGT-X native alignment; lazy for consistency with the hook above.
        from .global_alignment import run_global_alignment

        return run_global_alignment
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
    "run_global_alignment",
]
