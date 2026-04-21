"""collab-splats: Extension tools for nerfstudio"""

__version__ = "0.0.1"

def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless needed."""
    if name == "Splatter":
        from collab_splats.wrapper.splatter import Splatter
        return Splatter
    elif name == "SplatterConfig":
        from collab_splats.wrapper.splatter import SplatterConfig
        return SplatterConfig
    elif name == "ColmapCamera":
        from collab_splats.utils.camera_utils import ColmapCamera
        return ColmapCamera
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # General wrapper class for running splats
    "SplatterConfig",
    "Splatter",
    "ColmapCamera",
]
