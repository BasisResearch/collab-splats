"""
Feedforward pointcloud creators: VGGT-X, MapAnything, VGGT-Omega, VGGT-SPARK, LoGeR.

- import from here; the submodule layout is an implementation detail
- vggt-omega is optional and re-exported only when its package is present
"""

from __future__ import annotations

# ── Public types and utilities ────────────────────────────────────────────────
from .base import (
    FeedforwardResult,
    BaseFeedforwardCreator,
    MultiviewConfidence,
    build_pycolmap_reconstruction,
    compute_multiview_depth_confidence,
    multiview_mask,
)

# ── Concrete creators ─────────────────────────────────────────────────────────
from .vggtx import VGGTXCreator
from .mapanything import MapAnythingCreator
from .vggt_spark_creator import VGGTSPARKCreator

# vggt-omega is an optional backend
# - present only if setup/feedforward.sh ran with the vggt-omega submodule initialized
# - the hard import lives in vggt_omega.py; this exposes it only when the package is there
try:
    from .vggt_omega import VGGTOmegaCreator
except ImportError:
    pass

# loger is an optional backend
# - the model tree is vendored by setup/loger.sh into the gitignored third_party/
# - the vendored import is deferred to _load_model, so this guards only against the
#   module itself being absent
try:
    from .loger import LoGeRCreator
except ImportError:
    pass

# ── Internal helpers (re-exported for wrappers and tests) ─────────────────────
# _raw_to_world_points is re-exported for geometry/loop_closure/wrapper.py
# - that caller runs it directly on raw VGGT-X outputs, outside the normal postprocess
from .base import _raw_to_world_points

# ── Test-patchable symbols ────────────────────────────────────────────────────
# Explicit re-export so the patch target never moves
# - tests patch collab_splats.pointcloud.feedforward.unproject_and_filter_points
# - the name stays valid whichever submodule defines the function
from .vggtx import unproject_and_filter_points

__all__ = [
    "FeedforwardResult",
    "BaseFeedforwardCreator",
    "build_pycolmap_reconstruction",
    "VGGTXCreator",
    "MapAnythingCreator",
    "VGGTOmegaCreator",
    "VGGTSPARKCreator",
    "LoGeRCreator",
    "unproject_and_filter_points",
    "MultiviewConfidence",
    "compute_multiview_depth_confidence",
    "multiview_mask",
]
