"""Feedforward pointcloud creators: VGGT-X, MapAnything, VGGT-Omega, VGGT-SPARK, and LoGeR backends.

Import from here — submodule structure is an implementation detail.
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

# vggt-omega is an optional backend — only available if setup/feedforward.sh
# was run with the vggt-omega submodule initialized.  Hard import lives in
# vggt_omega.py itself; here we expose it only when the package is present.
try:
    from .vggt_omega import VGGTOmegaCreator
except ImportError:
    pass

# loger is an optional backend — the model tree is vendored by setup/loger.sh into
# the gitignored third_party/. The vendored import is deferred to _load_model, so
# this only guards against the module itself being absent.
try:
    from .loger import LoGeRCreator
except ImportError:
    pass

# ── Internal helpers (re-exported for wrappers and tests) ─────────────────────
# _raw_to_world_points re-exported for geometry/loop_closure/wrapper.py, which calls it
# directly on raw VGGT-X outputs outside the normal postprocess pipeline.
from .base import _raw_to_world_points

# ── Test-patchable symbols ────────────────────────────────────────────────────
# Explicitly re-exported so tests can patch
# collab_splats.pointcloud.feedforward.unproject_and_filter_points
# regardless of which submodule defines the function.
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
