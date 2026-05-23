"""Feedforward pointcloud creators: VGGT-X, MapAnything, and VGGT-Omega backends.

Import from here — submodule structure is an implementation detail.
"""
from __future__ import annotations

# ── Public types and utilities ────────────────────────────────────────────────
from .base import FeedforwardResult, BaseFeedforwardCreator, build_pycolmap_reconstruction

# ── Concrete creators ─────────────────────────────────────────────────────────
from .vggtx import VGGTXCreator
from .mapanything import MapAnythingCreator
from .vggt_omega import VGGTOmegaCreator

# ── Internal helpers (re-exported for wrappers and tests) ─────────────────────
# _raw_to_world_points re-exported for wrappers.py BundleAdjustment, which calls it
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
    "unproject_and_filter_points",
]
