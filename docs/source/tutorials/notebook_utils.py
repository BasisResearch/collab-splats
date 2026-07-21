"""Presentation helpers shared by the tutorial notebooks. `%run ../notebook_utils.py`.

Notebook-only concerns (kept out of collab_splats/): pyvista backend selection and a
canonical keyframe loader. Import via `%run` so the functions land in the notebook namespace.
"""

import os
from pathlib import Path

import pyvista as pv

from collab_splats.preproc import FrameStore


def set_notebook_backend() -> None:
    """Static pyvista backend when headless (nbconvert), interactive trame otherwise."""
    pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")


def load_keyframe_paths(frames_zarr, export_dir) -> list[Path]:
    """Open the canonical frames.zarr and export its keyframes as sorted jpg paths.

    Transient bridge for path-locked model preprocessing: pixels come from the
    tutorial-owned frames.zarr, not a pipeline jpg dir. Returns frame_NNNNNN.jpg paths.
    """
    store = FrameStore.open(frames_zarr)
    return sorted(store.export(export_dir, ext="jpg"))
