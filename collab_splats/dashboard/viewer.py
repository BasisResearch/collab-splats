"""Side-by-side PyVista viewer: RGB pointcloud/mesh (left) + similarity heatmap (right)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import panel as pn
import pyvista as pv

from collab_splats.dashboard.viz_utils import apply_viridis, pointcloud_to_polydata
from collab_splats.semantics.features.base import BaseQueryableExtractor

logger = logging.getLogger(__name__)


class SplitViewer:
    """Two linked plotters; left = RGB pcd/mesh, right = query similarity."""

    def __init__(self, off_screen: bool = False) -> None:
        self._off_screen = off_screen
        self._left = pv.Plotter(off_screen=off_screen)
        self._right = pv.Plotter(off_screen=off_screen)
        self._left_pane = pn.pane.VTK(self._left.ren_win, sizing_mode="stretch_both", min_height=500)
        self._right_pane = pn.pane.VTK(self._right.ren_win, sizing_mode="stretch_both", min_height=500)
        self.layout = pn.Row(self._left_pane, self._right_pane, sizing_mode="stretch_both")

        # State
        self.mode = "pointcloud"
        self.left_actor = None
        self._right_actor = None
        self._result = None
        self._mesh_path: Path | None = None
        self._lifted_normed: np.ndarray | None = None
        self._extractor_cache: dict = {}
        self._status = ""

    # ---- loading -------------------------------------------------------

    def load(self, result, mesh_path: Path | None, lifted_normed: np.ndarray | None = None) -> None:
        """Load a FeedforwardResult (+ optional mesh + lifted features) into both panes."""
        self._result = result
        self._mesh_path = Path(mesh_path) if mesh_path else None
        self._lifted_normed = lifted_normed
        self._render_left()
        self._render_right(None)

    def _render_left(self) -> None:
        """Render RGB pointcloud or mesh into the left plotter."""
        self._left.clear()
        if self.mode == "mesh" and self._mesh_path and self._mesh_path.exists():
            self.left_actor = self._left.add_mesh(pv.read(str(self._mesh_path)), rgb=True)
        else:
            if self.mode == "mesh":
                self._status = "mesh.ply not found."
                logger.warning("mesh.ply not found; falling back to pointcloud for left pane")
            cloud = pointcloud_to_polydata(self._result.points, RGB=self._result.colors)
            self.left_actor = self._left.add_mesh(
                cloud, scalars="RGB", rgb=True, point_size=2
            )
        if not self._off_screen:
            self._left_pane.synchronize()

    def _render_right(self, colors: np.ndarray | None) -> None:
        """Render RGB or similarity-colored pointcloud into the right plotter."""
        self._right.clear()
        rgb = colors if colors is not None else self._result.colors
        cloud = pointcloud_to_polydata(self._result.points, RGB=rgb)
        self._right_actor = self._right.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
        # Link right camera to left so the two views stay in sync
        self._right.camera = self._left.camera
        if not self._off_screen:
            self._right_pane.synchronize()

    # ---- interactions --------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch the left pane between 'pointcloud' and 'mesh'."""
        self.mode = mode
        if self._result is not None:
            self._render_left()

    def _get_extractor(self, name: str):
        """Construct (and cache) a queryable extractor by registry name."""
        if name not in self._extractor_cache:
            self._extractor_cache[name] = BaseQueryableExtractor.get(name)()
        return self._extractor_cache[name]

    def query(self, text: str, extractor_name: str) -> np.ndarray:
        """Recolour the right pane by cosine similarity to text query; return RGB colours."""
        if not text or self._lifted_normed is None:
            self._render_right(None)
            return self._result.colors
        extractor = self._get_extractor(extractor_name)
        text_emb = extractor.encode_text([text])   # (1, D) torch tensor
        vec = text_emb.detach().cpu().numpy()[0]
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        sims = self._lifted_normed @ vec           # (P,) cosine similarities
        colors = apply_viridis(sims)
        self._render_right(colors)
        return colors
