"""LocalizePane — Tab 5 of the unified dashboard.

Camera localization in a known reconstruction: single-image + batch modes.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.colors as mc
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.utils.visualization import (
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
)

logger = logging.getLogger(__name__)

########################################################################
# Colours

_COLOR_DEFAULT = "cornflowerblue"
_COLOR_QUERY   = "tomato"
_COLOR_REF     = "gold"


def _rgb(name: str) -> tuple[float, float, float]:
    """Convert matplotlib colour name to (r, g, b) 0–1 floats for VTK."""
    return mc.to_rgb(name)


_RGB_DEFAULT = _rgb(_COLOR_DEFAULT)
_RGB_QUERY   = _rgb(_COLOR_QUERY)
_RGB_REF     = _rgb(_COLOR_REF)

########################################################################


class LocalizeScenePanel(param.Parameterized):
    """Minimal PyVista 3D viewer for the Localize tab.

    Renders point cloud + camera frustums. Highlights a query/ref pair
    after localization via highlight(); reset() clears the highlight.
    """

    def __init__(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        image_paths: list,
        _off_screen: bool = False,
        **params: Any,
    ):
        """Build scene from pts3d (P,3), extrinsics (N,4,4), image_paths length-N."""
        super().__init__(**params)
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._image_paths = image_paths
        self._highlighted_query_idx: int | None = None
        self._highlighted_ref_idx: int | None = None
        self._query_actor = None
        self._connector_actor = None

        self._plotter = pv.Plotter(off_screen=_off_screen, notebook=False)
        try:
            self._vtk_pane = pn.pane.VTK(
                self._plotter.ren_win,
                sizing_mode="stretch_both",
                min_height=300,
            )
        except Exception as e:
            # ren_win may be unavailable (e.g. in tests with a mocked Plotter)
            logger.debug("VTK pane init failed (headless?): %s", e)
            self._vtk_pane = None
        self._build_scene()

    def _build_scene(self) -> None:
        """Render point cloud + all camera frustums at default colour."""
        self._plotter.clear()

        # Point cloud
        if len(self._pts3d) > 0:
            cloud = pointcloud_to_polydata(self._pts3d)
            self._plotter.add_mesh(
                cloud, color="lightgray", point_size=2, render_points_as_spheres=True
            )

        # Camera frustums — store actors so we can recolour on highlight
        self._frustum_actors: list[Any] = []
        for ext in self._extrinsics:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
            actor = self._plotter.add_mesh(frustum, color=_RGB_DEFAULT, line_width=1)
            self._frustum_actors.append(actor)

        self._plotter.reset_camera()

    def highlight(
        self,
        query_ext: np.ndarray,
        ref_ext: np.ndarray,
        query_idx: int,
        ref_idx: int,
    ) -> None:
        """Highlight query camera (red) and ref camera (yellow); grey out others.

        When query_idx == -1, query_ext is not in self._extrinsics — add it
        as a temporary actor instead of recolouring an existing one.
        """
        self._highlighted_query_idx = query_idx
        self._highlighted_ref_idx = ref_idx

        # Remove previous temporary query actor if any
        if self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None

        # Remove previous connector if any
        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None

        # Recolour reference frustums
        for i, actor in enumerate(self._frustum_actors):
            if i == ref_idx:
                actor.GetProperty().SetColor(*_RGB_REF)
                actor.GetProperty().SetLineWidth(3)
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetColor(*_RGB_DEFAULT)
                actor.GetProperty().SetOpacity(0.25)
                actor.GetProperty().SetLineWidth(1)

        # Add query camera as temporary red frustum (query_idx == -1 means not in array)
        if query_idx == -1:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(query_ext))
            self._query_actor = self._plotter.add_mesh(
                frustum, color=_COLOR_QUERY, line_width=3
            )
        else:
            self._frustum_actors[query_idx].GetProperty().SetColor(*_RGB_QUERY)
            self._frustum_actors[query_idx].GetProperty().SetLineWidth(3)
            self._frustum_actors[query_idx].GetProperty().SetOpacity(1.0)

        # Draw connector between query and ref camera centres
        q_pos = np.linalg.inv(query_ext)[:3, 3]
        r_pos = np.linalg.inv(ref_ext)[:3, 3]
        line = pv.Line(q_pos.tolist(), r_pos.tolist())
        self._connector_actor = self._plotter.add_mesh(
            line, color=_COLOR_REF, line_width=2
        )

        if self._vtk_pane is not None:
            self._vtk_pane.param.trigger("object")

    def reset(self) -> None:
        """Clear highlight; restore all cameras to default colour."""
        self._highlighted_query_idx = None
        self._highlighted_ref_idx = None

        if self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None

        if self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None

        for actor in self._frustum_actors:
            actor.GetProperty().SetColor(*_RGB_DEFAULT)
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetLineWidth(1)

        if self._vtk_pane is not None:
            self._vtk_pane.param.trigger("object")

    def panel(self) -> pn.pane.VTK | None:
        """Return the VTK Panel pane."""
        return self._vtk_pane


