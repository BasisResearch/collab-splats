"""Geometry backend: loop closure, bundle adjustment, and SE(3)/pose transforms."""

from .bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig
from .loop_closure import LoopClosureConfig, PoseGraph, Submap
from .transforms import (
    OPENGL_TO_OPENCV,
    extract_intrinsics,
    extrinsics_to_homogeneous,
    invert_poses,
    rotation_align_vectors,
)


def __getattr__(name):
    # Lazy import — loop_closure.wrapper imports collab_splats.pointcloud, which
    # imports geometry.transforms; an eager import here would cycle at load time.
    # Delegates to loop_closure's own lazy hook (one canonical lazy site).
    if name == "LoopClosure":
        from .loop_closure import LoopClosure

        return LoopClosure
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OPENGL_TO_OPENCV",
    "BundleAdjustment",
    "BundleAdjustmentConfig",
    "LoopClosure",
    "LoopClosureConfig",
    "PoseGraph",
    "Submap",
    "extract_intrinsics",
    "extrinsics_to_homogeneous",
    "invert_poses",
    "rotation_align_vectors",
]
