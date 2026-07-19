from .eval import capture_pose_graph_loss
from .graph import (
    PoseGraph,
    decompose_camera,
    estimate_scale_pairwise,
    run_pose_graph_optimization,
)
from .matching import LoopMatch, LoopMatchQueue, find_loop_closures, translation_jump_check
from .merge import dedup_overlap, merge_submap_outputs
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
    "Submap",
    "assert_world_to_cam",
    "LoopClosureConfig",
    "LoopMatch",
    "LoopMatchQueue",
    "find_loop_closures",
    "run_pose_graph_optimization",
    "merge_submap_outputs",
    "dedup_overlap",
    "translation_jump_check",
    "PoseGraph",
    "decompose_camera",
    "estimate_scale_pairwise",
    "capture_pose_graph_loss",
]
