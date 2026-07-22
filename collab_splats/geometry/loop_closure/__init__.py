from .eval import capture_pose_graph_loss
from .graph import (
    PoseGraph,
    decompose_camera,
    dedup_overlap,
    estimate_scale_pairwise,
)
from .map import GraphMap
from .matching import (
    LoopMatch,
    LoopMatchQueue,
    find_loop_closures,
    translation_jump_check,
)
from .submap import Submap, assert_world_to_cam


def __getattr__(name):
    # Lazy import — wrapper imports collab_splats.pointcloud (feedforward/base), which
    # itself imports geometry.transforms; an eager import here would cycle at load time.
    if name == "LoopClosure":
        from .wrapper import LoopClosure

        return LoopClosure
    if name == "LoopClosureConfig":
        from .wrapper import LoopClosureConfig

        return LoopClosureConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LoopClosure",
    "GraphMap",
    "Submap",
    "assert_world_to_cam",
    "LoopClosureConfig",
    "LoopMatch",
    "LoopMatchQueue",
    "find_loop_closures",
    "dedup_overlap",
    "translation_jump_check",
    "PoseGraph",
    "decompose_camera",
    "estimate_scale_pairwise",
    "capture_pose_graph_loss",
]
