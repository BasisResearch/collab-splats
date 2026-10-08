"""
Side-by-side PyVista viewer: RGB pointcloud or mesh (left), query similarity heatmap (right).

- both panes share one decimated subsample and one display-only view transform
- query scoring runs on the GPU worker; rendering runs on the IOLoop
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import panel as pn
import pyvista as pv
import torch
import zarr
from scipy.spatial.transform import Rotation

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.semantics.features import BaseQueryableExtractor
from collab_splats.semantics.store import read_point_features
from collab_splats.utils.visualization import PCD_KWARGS as _BASE_PCD_KWARGS
from collab_splats.utils.visualization import (
    VIZ_KWARGS,
    apply_viridis,
    pointcloud_to_polydata,
)

logger = logging.getLogger(__name__)

# Flat GL points: far cheaper than sphere impostors for 500k points; notebooks keep the shared defaults
PCD_KWARGS = {**_BASE_PCD_KWARGS, "render_points_as_spheres": False, "point_size": 2.0}


########
# Helpers
########


def _decimate_indices(n: int, max_points: int) -> np.ndarray:
    """
    Display indices into n points, evenly subsampled to at most max_points.

    - evenly strided (deterministic, no RNG), so RGB and heatmap panes stay registered
    - max_points <= 0 or n <= max_points -> identity
    """
    if max_points <= 0 or n <= max_points:
        return np.arange(n)

    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


def compute_view_transform(
    points: np.ndarray,
    extrinsics: np.ndarray | None = None,
    target_radius: float = 0.7,
    percentile: float = 95.0,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """
    Display-only similarity transform that recenters, up-aligns and scales a scene.

    - center: bbox midpoint of percentile-clipped inliers; a plain median sits off-center for skewed clouds
    - rotation: mean camera up (-R_w2c[1], OpenCV Y-down) onto up_axis
    - rotation is identity with no extrinsics or a degenerate mean up (e.g. a nadir orbit)
    - scale: percentile-th radius to target_radius; < 1 leaves margin for the fixed VIZ_KWARGS camera
    - render-time only: the scene itself is never mutated

    Args:
        points: world points, (N, 3).
        extrinsics: world-to-camera poses, (F, 3|4, 4); None skips the rotation.
        target_radius: display radius the percentile-th point radius maps to.
        percentile: inlier percentile for both centering and scale.
        up_axis: display up direction the mean camera up is rotated onto.

    Returns:
        Homogeneous transform, (4, 4).
    """
    pts = np.asarray(points, dtype=np.float64)

    # Center on the inlier bbox midpoint: clip flyers by radius about the median, then take the mid
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    inliers = pts[d <= np.percentile(d, percentile)]

    if len(inliers) == 0:
        inliers = pts

    center = (inliers.min(axis=0) + inliers.max(axis=0)) / 2.0

    # Up-align: mean camera up in world = -R_w2c[1, :]; skip rotation if degenerate
    R = np.eye(3)

    if extrinsics is not None and len(extrinsics) > 0:
        E = np.asarray(extrinsics, dtype=np.float64)
        mean_up = -E[:, 1, :3].mean(axis=0)

        if np.linalg.norm(mean_up) > 1e-6:
            rotation, _ = Rotation.align_vectors(
                np.asarray(up_axis, dtype=np.float64)[None], mean_up[None]
            )
            R = rotation.as_matrix()
        else:
            logger.warning(
                "view transform: degenerate mean camera up-vector; skipping rotation"
            )

    # Scale: percentile radius about center -> target_radius (robust to flyer points)
    radii = np.linalg.norm(pts - center, axis=1)
    r = float(np.percentile(radii, percentile))
    s = target_radius / r if r > 1e-9 else 1.0

    # Compose homogeneous T: p' = s * R @ (p - center)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = -s * (R @ center)
    return T


########
# SplitViewer
########


class SplitViewer:
    """
    Two linked plotters: left shows the RGB pointcloud or mesh, right the query similarity.

    - cameras are synced browser-side between the two VTK panes
    """

    def __init__(self, op_log: OperationLog, off_screen: bool = False) -> None:
        """
        Build both plotters and their linked panes.

        Args:
            op_log: shared log that viewer status lines surface in.
            off_screen: render without a window (tests).
        """
        self._off_screen = off_screen
        self._op_log = op_log
        self._left = pv.Plotter(off_screen=off_screen)
        self._right = pv.Plotter(off_screen=off_screen)
        self._left_pane = pn.pane.VTK(
            self._left.ren_win, sizing_mode="stretch_both", min_height=500
        )
        self._right_pane = pn.pane.VTK(
            self._right.ren_win, sizing_mode="stretch_both", min_height=500
        )
        self.layout = pn.Row(
            self._left_pane, self._right_pane, sizing_mode="stretch_both"
        )

        # Browser-side bidirectional camera sync between the two VTK panes
        if not off_screen:
            self._left_pane.jslink(
                self._right_pane, camera="camera", bidirectional=True
            )

        # Scene state
        self.mode = "pointcloud"
        self._result: PointcloudResult | None = None
        self._mesh_path: Path | None = None
        self._mesh_polydata: pv.PolyData | None = None
        self._mesh_vertex_features: np.ndarray | None = None
        self._mesh_observed: np.ndarray | None = None
        self._point_features: np.ndarray | None = None
        self._lifted_store: Path | None = None
        self._display_idx: np.ndarray | None = None

        # Right-pane cloud (pointcloud mode only), so render_query recolors in place
        self._right_cloud: pv.PolyData | None = None
        self._extractor_cache: dict = {}

        # Active query and per-mode similarity colors, kept across pointcloud/mesh switches
        self._last_query: tuple | None = None
        self._query_colors: dict[str, np.ndarray | None] = {
            "pointcloud": None,
            "mesh": None,
        }

        # Display-only normalization (orientation + scale); see compute_view_transform
        self._normalize_view = True
        self._fit_T: np.ndarray | None = None
        self._view_T: np.ndarray | None = None

        # View-space geometry built once per load / toggle; each pane renders a shallow copy
        self._cloud_view: pv.PolyData | None = None
        self._mesh_view: pv.PolyData | None = None

    ########
    # Loading
    ########

    def load(
        self,
        result: PointcloudResult,
        mesh_path: Path | None,
        lifted_store: Path | None = None,
        max_points: int = 500_000,
    ) -> None:
        """
        Load a scene into both panes; features are read on the first query.

        - the mesh read is deferred to ensure_mesh_polydata (worker thread), off the IOLoop
        - prior query state is dropped, so the right pane starts on plain RGB

        Args:
            result: the reconstruction to display.
            mesh_path: mesh .ply beside the result, or None.
            lifted_store: semantics zarr with lifted point and vertex features, or None.
            max_points: display cap; <= 0 shows every point.
        """
        self._result = result
        self._mesh_path = Path(mesh_path) if mesh_path else None
        self._mesh_polydata = None
        self._mesh_vertex_features = None
        self._mesh_observed = None
        self._point_features = None
        self._lifted_store = Path(lifted_store) if lifted_store else None

        # New scene: drop prior query state and the stale right-pane cloud
        self._last_query = None
        self._query_colors = {"pointcloud": None, "mesh": None}
        self._right_cloud = None

        # Decimate, fit the view transform and render both panes
        self._display_idx = _decimate_indices(len(result.points), max_points)
        self._fit_T = None
        self._update_view_transform()
        self._render_left()
        self._render_right(None)

    def ensure_mesh_polydata(self, preloaded: pv.PolyData | None = None) -> bool:
        """
        Lazily materialize the mesh PolyData on the worker thread.

        - renders reuse the cached PolyData; no per-interaction disk read

        Args:
            preloaded: shared-cache PolyData that skips the disk read.

        Returns:
            True when a mesh is available.
        """
        if self._mesh_polydata is not None:
            return True

        # New mesh: its view-space copy is rebuilt on the next render
        self._mesh_view = None

        if preloaded is not None:
            self._mesh_polydata = preloaded
        elif self._mesh_path and self._mesh_path.exists():
            with self._op_log.step("reading mesh"):
                self._mesh_polydata = pv.read(str(self._mesh_path))
        else:
            return False

        return True

    def mesh_polydata(self) -> pv.PolyData | None:
        """
        Cached mesh PolyData.

        Returns:
            The mesh, or None until ensure_mesh_polydata succeeds.
        """
        return self._mesh_polydata

    ########
    # Rendering
    ########

    def _update_view_transform(self) -> None:
        """
        Pick the display transform for the normalize toggle and drop stale view-space geometry.

        - fit on the FULL pointcloud once per load, so toggling never refits
        - normalize off -> None (raw world space)
        """
        if self._normalize_view and self._fit_T is None:
            assert self._result is not None
            self._fit_T = compute_view_transform(
                self._result.points, extrinsics=self._result.extrinsics
            )

        self._view_T = self._fit_T if self._normalize_view else None
        self._cloud_view = None
        self._mesh_view = None

    def _cloud_in_view(self) -> pv.PolyData:
        """
        Decimated RGB cloud in view space, built once per load / toggle.
        """
        if self._cloud_view is None:
            assert self._result is not None
            idx = self._display_idx
            self._cloud_view = pointcloud_to_polydata(
                self._result.points[idx], RGB=self._result.colors[idx]
            )

            # Fresh geometry, so transform it in place
            if self._view_T is not None:
                self._cloud_view.transform(self._view_T, inplace=True)

        return self._cloud_view

    def _mesh_in_view(self) -> pv.PolyData:
        """
        Mesh in view space, built once per mesh / toggle; the cached raw mesh is never touched.
        """
        if self._mesh_view is None:
            assert self._mesh_polydata is not None
            if self._view_T is None:
                self._mesh_view = self._mesh_polydata.copy(deep=False)
            else:
                self._mesh_view = self._mesh_polydata.transform(
                    self._view_T, inplace=False
                )

        return self._mesh_view

    def set_normalize_view(self, enabled: bool) -> None:
        """
        Toggle display-only orientation and scale normalization, then re-render both panes.

        Args:
            enabled: normalize the view; False shows raw world space.
        """
        self._normalize_view = enabled

        if self._result is not None:
            self._update_view_transform()
            self._render_left()
            self._render_right(self._query_colors.get(self.mode))

    def set_mode(self, mode: str) -> None:
        """
        Switch both panes between pointcloud and mesh, keeping the right similarity map.

        - the right pane renders this mode's cached query colors; None shows plain RGB
        - the app re-scores an unscored mode on the worker (see active_query / cached_query_colors)

        Args:
            mode: "pointcloud" or "mesh".
        """
        self.mode = mode

        if self._result is not None:
            self._render_left()
            self._render_right(self._query_colors.get(mode))

    def _apply_view(self, plotter: pv.Plotter) -> None:
        """
        Pin camera and lighting from VIZ_KWARGS, mirroring the notebook visualize_splat.

        - without it pyvista auto-frames to data bounds, so one flyer shrinks the scene to a speck
        - idempotent: view_angle is set absolutely and lights cleared, since Zoom and add_light accumulate
        """
        plotter.camera_position = [
            VIZ_KWARGS["position"],
            VIZ_KWARGS["focal_point"],
            VIZ_KWARGS["view_up"],
        ]
        plotter.camera.azimuth = VIZ_KWARGS["azimuth"]
        plotter.camera.elevation = VIZ_KWARGS["elevation"]
        plotter.camera.view_angle = 30.0 / VIZ_KWARGS["zoom"]
        plotter.remove_all_lights()

        for light in VIZ_KWARGS["lighting"]:
            plotter.add_light(pv.Light(**light))

    def _render_left(self) -> None:
        """
        Render the RGB pointcloud or mesh into the left plotter.
        """
        self._left.clear()

        if self.mode == "mesh" and self._mesh_polydata is not None:
            # Mesh .ply shares result.points' raw world space, so the same transform applies
            self._left.add_mesh(self._mesh_in_view().copy(deep=False), rgb=True)
        else:
            if self.mode == "mesh":
                self._op_log.append_line("mesh not found — showing pointcloud")
                logger.warning(
                    "mesh not found; falling back to pointcloud for left pane"
                )

            self._left.add_mesh(self._cloud_in_view().copy(deep=False), **PCD_KWARGS)

        self._apply_view(self._left)

        if not self._off_screen:
            self._left_pane.synchronize()

    def _render_right(self, colors: np.ndarray | None) -> None:
        """
        Render the RGB or similarity-colored scene into the right plotter.

        - mesh mode: colors are per-vertex, (M, 3), aligned with mesh.vertices
        - pointcloud mode: colors are (P, 3), aligned with result.points, then decimated
        - mesh colors of the wrong length revert to the plain RGB mesh
        """
        self._right.clear()

        if self.mode == "mesh" and self._mesh_polydata is not None:
            # Mesh displayed: the cached right-pane cloud no longer matches the pane
            self._right_cloud = None

            # Shallow copy: query colors land on this pane's copy only
            mesh = self._mesh_in_view().copy(deep=False)

            if colors is not None and len(colors) == mesh.n_points:
                # Per-vertex query colors, aligned with mesh.vertices order
                mesh.point_data["RGB"] = np.asarray(colors, dtype=np.uint8)
                self._right.add_mesh(mesh, scalars="RGB", rgb=True)
            else:
                # Plain RGB mesh (PLY already carries vertex colors)
                self._right.add_mesh(mesh, rgb=True)
        else:
            # Shallow copy: query colors land on this pane's copy only
            cloud = self._cloud_in_view().copy(deep=False)

            if colors is not None:
                cloud["RGB"] = np.asarray(colors[self._display_idx], dtype=np.uint8)

            self._right.add_mesh(cloud, **PCD_KWARGS)

            # Keep a handle to the displayed cloud so render_query can recolor in place
            self._right_cloud = cloud

        self._apply_view(self._right)

        if not self._off_screen:
            self._right_pane.synchronize()

    ########
    # Query
    ########

    def ensure_lifted(self) -> None:
        """
        Read the lifted per-point features on the first query (worker thread).
        """
        if self._point_features is not None or self._lifted_store is None:
            return

        self._op_log.append_line("query: loading lifted point features")
        self._point_features = read_point_features(self._lifted_store)

    def ensure_mesh_features(self) -> None:
        """
        Read the stored per-vertex features on the first mesh query (worker thread).

        - no store, or a store without `vertex_features`, leaves them None
        - unobserved vertices: all-zero raw codes, as the scene viewer reads them
        """
        if self._mesh_vertex_features is not None or self._lifted_store is None:
            return

        store = zarr.open(str(self._lifted_store), mode="r")

        if "vertex_features" not in store:
            return

        self._op_log.append_line("query: loading stored vertex features")
        self._mesh_observed = np.asarray(store["vertex_features"]).any(axis=1)
        self._mesh_vertex_features = read_point_features(
            self._lifted_store, name="vertex_features"
        )

    def active_query(self) -> tuple | None:
        """
        Last scored query.

        Returns:
            (positive, negative, extractor name), or None if no query is active.
        """
        return self._last_query

    def cached_query_colors(self, mode: str) -> np.ndarray | None:
        """
        Cached similarity colors for one mode.

        Args:
            mode: "pointcloud" or "mesh".

        Returns:
            RGB colors, or None if the mode must be (re)scored.
        """
        return self._query_colors.get(mode)

    def _get_extractor(self, name: str) -> BaseQueryableExtractor:
        """
        Construct (and cache) a queryable extractor by registry name.

        - built with the lifted store's `extractor_kwargs`, so text lands in the stored feature space
        - cached per name and kwargs, so a scene lifted with other kwargs gets its own extractor
        """
        store = zarr.open(str(self._lifted_store), mode="r")
        kwargs = store.attrs.get("extractor_kwargs", {})
        key = (name, json.dumps(kwargs, sort_keys=True))

        if key not in self._extractor_cache:
            self._extractor_cache[key] = BaseQueryableExtractor.get(name)(**kwargs)

        return self._extractor_cache[key]

    def score_query(
        self,
        positive: list[str],
        negative: list[str],
        extractor_name: str,
        mode: str | None = None,
    ) -> np.ndarray | None:
        """
        Per-element query colors; pure compute, no rendering (GPU worker).

        - BaseQueryableExtractor.score_queries: contrastive softmax in [0, 1]
        - empty positive or no lifted features -> the plain RGB colors
        - mode switches pass their target: set_mode runs later, so self.mode is still the outgoing one
        - pass the returned colors to render_query on the IOLoop

        Args:
            positive: phrases to score toward.
            negative: phrases to score away from.
            extractor_name: registry name of the queryable extractor.
            mode: target feature space, "pointcloud" or "mesh"; None uses the current mode.

        Returns:
            RGB colors, (P, 3) or (M, 3) uint8, or None when no scene is loaded.
        """
        target = mode or self.mode

        # No scene loaded yet (e.g. query pressed before a load completed): nothing to score
        if self._result is None:
            self._op_log.append_line("query: no scene loaded")
            return None

        # No query terms: fall back to plain RGB (skip the store read entirely)
        if not positive:
            return self._result.colors

        # Mesh on screen scores its stored vertex features; no mesh falls back to points
        on_mesh = target == "mesh" and self.ensure_mesh_polydata()

        if on_mesh:
            self.ensure_mesh_features()
            feature_array = self._mesh_vertex_features
        else:
            self.ensure_lifted()
            feature_array = self._point_features

        # Nothing to score: plain RGB, with the fix in the op log
        if feature_array is None:
            if on_mesh and self._lifted_store is not None:
                self._op_log.append_line(
                    "mesh query needs vertex_features — re-run the semantics stage with overwrite"
                )
            else:
                self._op_log.append_line(
                    "query: no lifted features for this backend — run the semantics stage"
                )

            return self._result.colors

        # Encode the query terms and score every element
        self._op_log.append_line(
            f"query: encoding {len(positive)} positive / {len(negative)} negative"
        )
        extractor = self._get_extractor(extractor_name)
        features = torch.from_numpy(feature_array)

        self._op_log.append_line(f"query: scoring {features.shape[0]} elements")
        scores = extractor.score_queries(features, positive=positive, negative=negative)
        sims = scores.detach().float().cpu().numpy()

        # Unobserved vertices score NaN, drawn grey
        if on_mesh:
            assert self._mesh_observed is not None
            sims[~self._mesh_observed] = np.nan

        colors = apply_viridis(sims)

        # Cache colors for the target mode only; the other mode re-scores on switch
        self._last_query = (list(positive), list(negative), extractor_name)
        self._query_colors = {"pointcloud": None, "mesh": None}
        self._query_colors[target] = colors
        self._op_log.append_line("query: scored")
        return colors

    def render_query(self, colors: np.ndarray | None) -> None:
        """
        Recolor the right pane with precomputed query colors (IOLoop thread).

        - fast path (pointcloud mode, same geometry on screen): swap the RGB scalars in place

        Args:
            colors: query colors from score_query; None shows plain RGB.
        """
        # Length guard: a stale or wrong-space result would index out of bounds, so show plain RGB
        if self.mode != "mesh" and colors is not None and self._result is not None:
            if len(colors) != len(self._result.points):
                self._op_log.append_line(
                    "query colors don't match the displayed scene — showing plain RGB"
                )
                logger.warning(
                    "render_query: %d colors vs %d points (stale result?)",
                    len(colors),
                    len(self._result.points),
                )
                colors = None

        # Same decimated geometry on screen: swap the active RGB array and flag it dirty
        if self.mode != "mesh" and colors is not None and self._right_cloud is not None:
            idx = self._display_idx
            self._right_cloud["RGB"] = np.asarray(colors[idx], dtype=np.uint8)
            self._right_cloud.Modified()

            if not self._off_screen:
                self._right_pane.synchronize()

            return

        self._render_right(colors)
