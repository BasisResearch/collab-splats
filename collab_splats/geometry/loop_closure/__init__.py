"""
Submap pose-graph loop closure around a feedforward creator's forward pass.

- wrapper: LoopClosure / LoopClosureConfig, the creator wrapper (lazy import)
- submap / map / graph: per-submap state, submap collection, factor graph
- matching: loop candidate retrieval
- exports only what non-test callers import; import the rest from submodules
"""

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.submap import Submap


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
    # Import the wrapper on first access (cycle)
    if name in ("LoopClosure", "LoopClosureConfig"):
        from collab_splats.geometry.loop_closure import wrapper

        return getattr(wrapper, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LoopClosure", "LoopClosureConfig", "PoseGraph", "Submap"]
