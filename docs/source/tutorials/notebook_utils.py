"""Presentation helpers shared by the tutorial notebooks. `%run ../notebook_utils.py`.

Notebook-only concerns (kept out of collab_splats/): pyvista backend selection. Import
via `%run` so the functions land in the notebook namespace.
"""

import os

import pyvista as pv


def set_notebook_backend() -> None:
    """Static pyvista backend when headless (nbconvert), interactive trame otherwise."""
    pv.set_jupyter_backend("static" if os.environ.get("PYVISTA_OFF_SCREEN") else "trame")
