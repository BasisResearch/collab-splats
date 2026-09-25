"""
Submap pose-graph loop closure around a feedforward creator's forward pass.

- wrapper: LoopClosure / LoopClosureConfig, the creator wrapper (lazy import)
- submap / map / graph: per-submap state, submap collection, factor graph
- matching: loop candidate retrieval
"""

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
)
from .submap import Submap, assert_world_to_cam


def __getattr__(name: str) -> type:
    # Lazy import: wrapper -> collab_splats.pointcloud -> geometry.transforms cycles
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
    "PoseGraph",
    "decompose_camera",
    "estimate_scale_pairwise",
]
