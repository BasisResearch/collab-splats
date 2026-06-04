"""Side-by-side PyVista viewer: RGB pointcloud/mesh (left) + similarity heatmap (right)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import panel as pn
import pyvista as pv
import torch
import zarr

from collab_splats.dashboard.viz_utils import apply_viridis, pointcloud_to_polydata

# NB: lift_features (pointcloud.utils) and BaseQueryableExtractor (semantics.features)
# pull in the heavy feedforward stack (~14s import). They are imported lazily inside the
# load/query methods so the viewer panes construct immediately at launch.

logger = logging.getLogger(__name__)


########################################################################
# Module-level feature helpers
########################################################################


def _load_feature_maps(semantics_dir) -> list:
    """Load per-frame dense feature maps (D, H_p, W_p) from the cached zarr."""
    # Layout: cache_dir/{name}.zarr is a zarr group; array "features" is (N, D, H_p, W_p)
    store_path = next(Path(semantics_dir).glob("*.zarr"))
    arr = zarr.open(str(store_path), mode="r")["features"]
    return [torch.from_numpy(np.asarray(arr[i])) for i in range(arr.shape[0])]


def load_lifted_normed(result, semantics_dir) -> np.ndarray:
    """Lift cached features to points and L2-normalise -> (P, D) float32."""
    # Lazy import: pointcloud.utils pulls the heavy feedforward stack.
    from collab_splats.pointcloud.utils import lift_features

    feature_maps = _load_feature_maps(semantics_dir)
    lifted = lift_features(feature_maps, result)
    lifted = lifted.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(lifted, axis=1, keepdims=True)
    return lifted / (norms + 1e-8)


class SplitViewer:
    """Two linked plotters; left = RGB pcd/mesh, right = query similarity."""

    def __init__(self, off_screen: bool = False) -> None:
        self._off_screen = off_screen
        self._left = pv.Plotter(off_screen=off_screen)
        self._right = pv.Plotter(off_screen=off_screen)
        self._left_pane = pn.pane.VTK(self._left.ren_win, sizing_mode="stretch_both", min_height=500)
        self._right_pane = pn.pane.VTK(self._right.ren_win, sizing_mode="stretch_both", min_height=500)
        self.layout = pn.Row(self._left_pane, self._right_pane, sizing_mode="stretch_both")

        # Browser-side bidirectional camera sync between the two VTK panes.
        if not off_screen:
            self._left_pane.jslink(self._right_pane, camera="camera", bidirectional=True)

        # State
        self.mode = "pointcloud"
        self.left_actor = None
        self.right_actor = None
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
            self.left_actor = self._left.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
        if not self._off_screen:
            self._left_pane.synchronize()

    def _render_right(self, colors: np.ndarray | None) -> None:
        """Render RGB or similarity-colored pointcloud into the right plotter."""
        self._right.clear()
        rgb = colors if colors is not None else self._result.colors
        cloud = pointcloud_to_polydata(self._result.points, RGB=rgb)
        self.right_actor = self._right.add_mesh(cloud, scalars="RGB", rgb=True, point_size=2)
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
        # Lazy import: semantics.features pulls the heavy feedforward stack.
        from collab_splats.semantics.features.base import BaseQueryableExtractor

        if name not in self._extractor_cache:
            self._extractor_cache[name] = BaseQueryableExtractor.get(name)()
        return self._extractor_cache[name]

    def query(
        self,
        positive: list[str],
        negative: list[str] | None = None,
        extractor_name: str = "talk2dino",
        op_log=None,
    ) -> np.ndarray:
        """Recolour the right pane by contrastive query score; return RGB colours.

        Reuses BaseQueryableExtractor.score_queries (contrastive softmax, [0, 1]).
        positive/negative are lists of phrases. Empty negative -> API default ["object"].
        """

        def _stage(msg: str) -> None:
            if op_log is not None:
                op_log.append_line(msg)

        # No query terms or no cached features: reset right pane to RGB.
        if not positive or self._lifted_normed is None:
            self._render_right(None)
            return self._result.colors

        _stage(f"query: encoding {len(positive)} positive / {len(negative or [])} negative")
        extractor = self._get_extractor(extractor_name)
        features = torch.from_numpy(self._lifted_normed)  # (P, D)

        _stage(f"query: scoring {features.shape[0]} points")
        scores = extractor.score_queries(features, positive=positive, negative=negative or None)
        sims = scores.detach().cpu().numpy()

        colors = apply_viridis(sims)
        self._render_right(colors)
        _stage("query: recolour done")
        return colors
