"""collab-splats: Extension tools for nerfstudio"""

__version__ = "0.0.1"

from collab_splats.wrapper.splatter import Splatter, SplatterConfig

from collab_splats.utils.camera_utils import ColmapCamera

__all__ = [
    # General wrapper class for running splats
    "SplatterConfig",
    "Splatter",
    "ColmapCamera",
]
