# collab_splats/pointcloud/sfm/__init__.py
"""
SfM backends: global-solver InstantSfM, plus the unwired ColmapCreator and HlocCreator.

- InstantSfMCreator is the only backend `pointcloud.method: sfm` accepts — see _SFM_BACKENDS.
- ColmapCreator (classical) and HlocCreator (learned-feature) are unit-tested against mocked
  pycolmap/hloc and never exercised end-to-end; nothing dispatches to them today.
"""

from .colmap import ColmapCreator
from .hloc import HlocCreator
from .instantsfm import InstantSfMCreator

__all__ = ["ColmapCreator", "HlocCreator", "InstantSfMCreator"]
