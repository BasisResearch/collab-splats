"""
Submap pose-graph loop closure around a feedforward creator's forward pass.

- wrapper: LoopClosure / LoopClosureConfig, the creator wrapper (lazy import)
- submap / map / graph: per-submap state, submap collection, factor graph
- matching: loop candidate retrieval
- exports only what non-test callers import; import the rest from submodules
"""

from .graph import PoseGraph
from .submap import Submap


def __getattr__(name: str) -> type:
    """
    Lazy export of the wrapper's LoopClosure and LoopClosureConfig.

    - lazy because wrapper -> collab_splats.pointcloud -> geometry.transforms is a cycle

    Args:
        name: attribute looked up on the package.

    Returns:
        The requested wrapper class.

    Raises:
        AttributeError: for any other name.
    """
    if name == "LoopClosure":
        from .wrapper import LoopClosure

        return LoopClosure
    if name == "LoopClosureConfig":
        from .wrapper import LoopClosureConfig

        return LoopClosureConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LoopClosure", "LoopClosureConfig", "PoseGraph", "Submap"]
