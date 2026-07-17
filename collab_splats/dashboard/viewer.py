"""Side-by-side PyVista viewer: RGB pointcloud/mesh (left) + similarity heatmap (right)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import panel as pn
import pyvista as pv
import torch
import zarr

from collab_splats.dashboard.viz_utils import (
    PCD_KWARGS,
    VIZ_KWARGS,
    apply_viridis,
    compute_view_transform,
    pointcloud_to_polydata,
)

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
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    from collab_splats.pointcloud.utils import lift_features

    # The display path loads the result lean (dense arrays skipped). Lifting needs
    # pixel_indices/depth/confidence — reload them from the source zarr on demand.
    _lift_fields = ("pixel_indices", "depth", "confidence")
    if any(getattr(result, f, None) is None for f in _lift_fields) and getattr(result, "_zarr_path", None):
        # world_points/features are unused by the lift; skip them to halve peak memory.
        result = FeedforwardResult.load_zarr(result._zarr_path, load_world_points=False, load_features=False)

    feature_maps = _load_feature_maps(semantics_dir)
    lifted = lift_features(feature_maps, result)
    lifted = lifted.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(lifted, axis=1, keepdims=True)
    return lifted / (norms + 1e-8)


def load_mesh_vertex_features(mesh_dir) -> "np.ndarray | None":
    """Load cached mesh vertex features and L2-normalise -> (M, D) float32, or None if absent.

    Matches load_lifted_normed's normalization so mesh features share the point feature
    space and can be scored by the same extractor.score_queries call.
    """
    path = Path(mesh_dir) / "vertex_features.npy"
    if not path.exists():
        return None
    feats = np.load(path).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / (norms + 1e-8)


def _decimate_indices(n: int, max_points: int) -> np.ndarray:
    """Return display indices into n points, evenly subsampled to at most max_points.

    Evenly-strided (deterministic, no RNG) so RGB and heatmap panes share the same
    subsample and stay registered. max_points <= 0 or n <= max_points -> identity.
    """
    if max_points <= 0 or n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, num=max_points, dtype=np.int64)


class SplitViewer:
    """Two linked plotters; left = RGB pcd/mesh, right = query similarity."""

    def __init__(self, off_screen: bool = False, op_log=None) -> None:
        self._off_screen = off_screen
        # Optional shared OperationLog: viewer status lines (e.g. mesh fallback) surface there.
        self._op_log = op_log
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
        self._mesh_polydata: pv.PolyData | None = None
        self._mesh_vertex_features: np.ndarray | None = None
        self._lifted_normed: np.ndarray | None = None
        self._semantics_dir: Path | None = None
        self._display_idx: np.ndarray | None = None
        # Decimated cloud currently displayed in the right pane (pointcloud mode only);
        # lets render_query recolor in place instead of rebuilding geometry.
        self._right_cloud: pv.PolyData | None = None
        self._extractor_cache: dict = {}
        self._status = ""
        # Active query (positive, negative, extractor) + per-mode similarity colours, so the right
        # pane keeps showing similarity across pointcloud/mesh switches instead of reverting to RGB.
        self._last_query: tuple | None = None
        self._query_colors: dict[str, np.ndarray | None] = {"pointcloud": None, "mesh": None}
        # Display-only normalization (orientation + scale); see compute_view_transform.
        self._normalize_view = True
        self._view_T: np.ndarray | None = None

    def _log(self, msg: str) -> None:
        """Append a status line to the shared op log, if one was provided."""
        if self._op_log is not None:
            self._op_log.append_line(msg)

    # ---- loading -------------------------------------------------------

    def load(
        self,
        result,
        mesh_path: Path | None,
        lifted_normed: np.ndarray | None = None,
        semantics_dir: Path | None = None,
        max_points: int = 500_000,
    ) -> None:
        """Load a FeedforwardResult (+ optional mesh) into both panes.

        Features are NOT lifted here — lifting 500k points takes minutes and is only
        needed for queries. semantics_dir is stashed so the first query can lift lazily
        (see ensure_lifted). lifted_normed may be passed pre-computed (tests).
        """
        self._result = result
        self._mesh_path = Path(mesh_path) if mesh_path else None
        # Mesh read is deferred to ensure_mesh_polydata (worker thread): the default pointcloud
        # mode must not pay a blocking pv.read + feature np.load on the IOLoop at every load.
        self._mesh_polydata = None
        self._mesh_vertex_features = None
        self._lifted_normed = lifted_normed
        self._semantics_dir = Path(semantics_dir) if semantics_dir else None
        # New scene -> drop any prior query state so the right pane starts on plain RGB.
        self._last_query = None
        self._query_colors = {"pointcloud": None, "mesh": None}
        # New scene -> the displayed right-pane cloud is stale; force a full rebuild.
        self._right_cloud = None
        self._display_idx = _decimate_indices(len(result.points), max_points)
        self._recompute_view_transform()
        self._render_left()
        self._render_right(None)

    def ensure_mesh_polydata(self, preloaded: "pv.PolyData | None" = None, op_log=None) -> bool:
        """Lazily materialise the mesh PolyData (worker thread). True when a mesh is available.

        preloaded lets the app hand in a shared-cache PolyData so the disk read is skipped;
        renders reuse the cached result (no per-interaction disk read).
        """
        if self._mesh_polydata is not None:
            return True
        if preloaded is not None:
            self._mesh_polydata = preloaded
        elif self._mesh_path and self._mesh_path.exists():
            # Blocking disk read — log it as a step when any op log is available.
            log = op_log or self._op_log
            if log is not None:
                with log.step("reading mesh"):
                    self._mesh_polydata = pv.read(str(self._mesh_path))
            else:
                self._mesh_polydata = pv.read(str(self._mesh_path))
        else:
            return False
        # Per-vertex mesh features (if the pipeline persisted them) — same space as point features.
        self._mesh_vertex_features = load_mesh_vertex_features(self._mesh_path.parent) if self._mesh_path else None
        return True

    def mesh_polydata(self) -> "pv.PolyData | None":
        """Return the cached mesh PolyData (None until ensure_mesh_polydata succeeds)."""
        return self._mesh_polydata

    def ensure_lifted(self, op_log=None) -> None:
        """Lazily load/lift per-point features on first query (off-loop)."""
        if self._lifted_normed is not None or self._semantics_dir is None:
            return
        # Fast path: the pipeline caches L2-normalised per-point features next to the scene;
        # loading the npy is instant vs the minutes-long lift from the feature zarr below.
        cached_path = self._semantics_dir / "lifted_normed.npy"
        if cached_path.exists():
            if op_log is not None:
                op_log.append_line("query: loading cached point features")
            self._lifted_normed = np.load(cached_path)
            return
        if op_log is not None:
            op_log.append_line("query: lifting features to points (first query — may take minutes)")
        try:
            self._lifted_normed = load_lifted_normed(self._result, self._semantics_dir)
        except Exception as exc:
            logger.warning("feature lift failed: %s", exc)
            self._lifted_normed = None

    def ensure_mesh_features(self, op_log=None) -> None:
        """Transfer cached point features onto mesh vertices on first mesh query (if not cached).

        Older runs have no persisted vertex_features.npy; compute it on demand from the
        already-lifted point features + the mesh so mesh-mode queries work without a re-run.
        L2-normalises to match load_mesh_vertex_features / point-feature scoring.
        """
        if self._mesh_vertex_features is not None:
            return
        if self._lifted_normed is None or self._mesh_polydata is None:
            return
        # Lazy import: mesh.utils pulls the heavy feedforward stack (matches load_lifted_normed).
        from collab_splats.mesh.utils import features2vertex

        if op_log is not None:
            op_log.append_line("query: transferring features to mesh vertices (first mesh query)")
        # Vertices come straight from the cached PolyData — no second disk read of the .ply.
        vertices = np.asarray(self._mesh_polydata.points)
        vf = features2vertex(vertices, self._result.points, self._lifted_normed)
        norms = np.linalg.norm(vf, axis=1, keepdims=True)
        self._mesh_vertex_features = (vf / (norms + 1e-8)).astype(np.float32)

    def _recompute_view_transform(self) -> None:
        """Compute the display-only recenter/up-align/scale transform for the loaded scene.

        Computed on the FULL pointcloud (not the decimated subset) for a stable center and
        scale. extrinsics may be absent (e.g. test fixtures) -> orientation falls back to no
        rotation. Disabled by the normalize toggle -> identity (raw world-space).
        """
        if self._result is None or not self._normalize_view:
            self._view_T = None
            return
        extrinsics = getattr(self._result, "extrinsics", None)
        self._view_T = compute_view_transform(self._result.points, extrinsics=extrinsics)

    def _normalize(self, mesh: pv.PolyData) -> pv.PolyData:
        """Return a view-normalized copy of a mesh/cloud (plain copy when normalization off).

        Always returns a new object so callers may freely mutate the result without
        touching a cached input; in-place transforms would compound across renders.
        """
        if self._view_T is None:
            return mesh.copy()
        return mesh.transform(self._view_T, inplace=False)

    def set_normalize_view(self, enabled: bool) -> None:
        """Toggle display-only orientation+scale normalization and re-render both panes."""
        self._normalize_view = enabled
        if self._result is not None:
            self._recompute_view_transform()
            self._render_left()
            self._render_right(None)

    def _apply_view(self, plotter: pv.Plotter) -> None:
        """Pin camera + lighting from VIZ_KWARGS, mirroring notebook visualize_splat.

        Without this, pyvista auto-frames to data bounds on every add_mesh, so a single
        outlier flyer zooms the camera out and the real scene collapses to a speck. The
        fixed origin-centred camera matches the notebook and ignores outlier extent.

        MUST be idempotent: it runs on every (re)render (mode/normalize toggles, queries).
        camera.Zoom() and add_light() are CUMULATIVE — calling them per render drifts the
        view_angle and stacks duplicate lights over a session. We set view_angle absolutely
        (30° base / zoom) and clear lights first so repeated renders reproduce one framing.
        """
        plotter.camera_position = [
            VIZ_KWARGS.get("position", (2, 2, 1)),
            VIZ_KWARGS.get("focal_point", (0, 0, 0)),
            VIZ_KWARGS.get("view_up", (0, 0, 1)),
        ]
        plotter.camera.azimuth = VIZ_KWARGS.get("azimuth", 235)
        plotter.camera.elevation = VIZ_KWARGS.get("elevation", 15)
        plotter.camera.view_angle = 30.0 / VIZ_KWARGS.get("zoom", 0.9)
        plotter.remove_all_lights()
        for light in VIZ_KWARGS.get("lighting", []):
            plotter.add_light(pv.Light(**light))

    def _render_left(self) -> None:
        """Render RGB pointcloud or mesh into the left plotter."""
        self._left.clear()
        if self.mode == "mesh" and self._mesh_polydata is not None:
            # Mesh .ply is in the same raw world-space as result.points -> same transform.
            # _normalize returns a copy, so the cached PolyData is never touched.
            self.left_actor = self._left.add_mesh(self._normalize(self._mesh_polydata), rgb=True)
        else:
            if self.mode == "mesh":
                self._status = "mesh not found."
                self._log("mesh not found — showing pointcloud")
                logger.warning("mesh not found; falling back to pointcloud for left pane")
            idx = self._display_idx
            cloud = self._normalize(pointcloud_to_polydata(self._result.points[idx], RGB=self._result.colors[idx]))
            # PCD_KWARGS = dashboard point style (flat GL points override; see viz_utils)
            self.left_actor = self._left.add_mesh(cloud, **PCD_KWARGS)
        self._apply_view(self._left)
        if not self._off_screen:
            self._left_pane.synchronize()

    def _render_right(self, colors: np.ndarray | None) -> None:
        """Render RGB or similarity-colored scene into the right plotter.

        Mesh mode colors per-vertex (colors is (M, 3) aligned with mesh.vertices); pointcloud
        mode colors the decimated cloud (colors is (P, 3) aligned with result.points). A
        colors length that doesn't match the mesh vertex count (e.g. no persisted vertex
        features -> point-length fallback) reverts to the plain RGB mesh.
        """
        self._right.clear()
        if self.mode == "mesh" and self._mesh_polydata is not None:
            # Mesh displayed -> the cached right-pane cloud no longer matches the pane.
            self._right_cloud = None
            # _normalize returns a copy of the cache; write query colours on the copy only.
            mesh = self._normalize(self._mesh_polydata)
            if colors is not None and len(colors) == mesh.n_points:
                # Per-vertex query colors, aligned with mesh.vertices order.
                mesh.point_data["RGB"] = np.ascontiguousarray(colors).astype(np.uint8)
                self.right_actor = self._right.add_mesh(mesh, scalars="RGB", rgb=True)
            else:
                # Plain RGB mesh (PLY already carries vertex colors).
                self.right_actor = self._right.add_mesh(mesh, rgb=True)
        else:
            idx = self._display_idx
            rgb = colors if colors is not None else self._result.colors
            cloud = self._normalize(pointcloud_to_polydata(self._result.points[idx], RGB=rgb[idx]))
            self.right_actor = self._right.add_mesh(cloud, **PCD_KWARGS)
            # Keep a handle to the displayed cloud so render_query can recolor in place.
            self._right_cloud = cloud
        self._apply_view(self._right)
        if not self._off_screen:
            self._right_pane.synchronize()

    # ---- interactions --------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch both panes between 'pointcloud' and 'mesh', keeping the right similarity map.

        Right pane renders this mode's cached query colours (None -> plain RGB until the app
        re-scores the new mode on the worker; see active_query / cached_query_colors).
        """
        self.mode = mode
        if self._result is not None:
            self._render_left()
            self._render_right(self._query_colors.get(mode))

    def active_query(self) -> tuple | None:
        """Return the last (positive, negative, extractor) query, or None if none is active."""
        return self._last_query

    def cached_query_colors(self, mode: str) -> np.ndarray | None:
        """Return cached similarity colours for a mode, or None if it must be (re)scored."""
        return self._query_colors.get(mode)

    def _get_extractor(self, name: str):
        """Construct (and cache) a queryable extractor by registry name."""
        # Lazy import: semantics.features pulls the heavy feedforward stack.
        from collab_splats.semantics.features.base import BaseQueryableExtractor

        if name not in self._extractor_cache:
            self._extractor_cache[name] = BaseQueryableExtractor.get(name)()
        return self._extractor_cache[name]

    def score_query(
        self,
        positive: list[str],
        negative: list[str] | None = None,
        extractor_name: str = "talk2dino",
        op_log=None,
        mode: "str | None" = None,
    ) -> np.ndarray:
        """Compute per-point query colours (RGB uint8). Pure compute — no rendering.

        Reuses BaseQueryableExtractor.score_queries (contrastive softmax, [0, 1]).
        Empty positive or no cached features -> returns the plain RGB colours.
        Call from the GPU worker; pass the returned colours to render_query on the IOLoop.
        mode selects the TARGET feature space ("pointcloud"/"mesh"); default is the current
        one. Mode switches must pass their target: set_mode runs later on the IOLoop, so
        self.mode is still the outgoing mode while this scores on the worker.
        """
        target = mode or self.mode

        def _stage(msg: str) -> None:
            if op_log is not None:
                op_log.append_line(msg)

        # No scene loaded yet (e.g. query pressed before a load completed) -> nothing to score.
        if self._result is None:
            _stage("query: no scene loaded")
            return None

        # No query terms: fall back to plain RGB (skip the expensive lift entirely).
        if not positive:
            return self._result.colors

        # Lift features on first query (cached thereafter); no semantics -> plain RGB.
        self.ensure_lifted(op_log)
        if self._lifted_normed is None:
            _stage("query: no semantic features for this scene — showing plain RGB")
            return self._result.colors

        # In mesh mode, materialise the mesh (worker thread) then transfer point features to
        # mesh vertices on first query if not already cached/persisted (no vertex_features.npy).
        if target == "mesh":
            self.ensure_mesh_polydata(op_log=op_log)
            self.ensure_mesh_features(op_log)

        _stage(f"query: encoding {len(positive)} positive / {len(negative or [])} negative")
        extractor = self._get_extractor(extractor_name)

        # In mesh mode score per-vertex features (same feature space); else score points.
        if target == "mesh" and self._mesh_vertex_features is not None:
            feature_array = self._mesh_vertex_features
        else:
            feature_array = self._lifted_normed
        features = torch.from_numpy(feature_array)  # (N, D)

        _stage(f"query: scoring {features.shape[0]} elements")
        scores = extractor.score_queries(features, positive=positive, negative=negative or None)
        sims = scores.detach().cpu().numpy()
        colors = apply_viridis(sims)
        # Remember the query and cache colours for the TARGET mode; invalidate the other mode so
        # a switch re-scores against the new terms (point vs mesh-vertex feature space).
        self._last_query = (list(positive), list(negative or []), extractor_name)
        self._query_colors = {"pointcloud": None, "mesh": None}
        self._query_colors[target] = colors
        _stage("query: scored")
        return colors

    def render_query(self, colors: np.ndarray) -> None:
        """Recolour the right pane with precomputed query colours (IOLoop thread).

        Fast path (pointcloud mode, same geometry already displayed): update the existing
        PolyData's RGB scalars in place instead of clearing + rebuilding the whole scene.
        """
        # Length guard: colours must cover the full pointcloud in pointcloud mode. A stale
        # or wrong-space result (e.g. mesh-vertex colours after a rapid mode flip, or a
        # scene swap mid-job) would index out of bounds — show plain RGB instead of crashing.
        if self.mode != "mesh" and colors is not None and self._result is not None:
            if len(colors) != len(self._result.points):
                self._log("query colours don't match the displayed scene — showing plain RGB")
                logger.warning(
                    "render_query: %d colours vs %d points (stale result?)",
                    len(colors),
                    len(self._result.points),
                )
                colors = None
        if (
            self.mode != "mesh"
            and colors is not None
            and self._right_cloud is not None
            and self._display_idx is not None
            and len(self._display_idx) == self._right_cloud.n_points
        ):
            # Same decimated geometry on screen -> swap the "RGB" point array (the active
            # scalars bound via PCD_KWARGS) and flag the dataset dirty; no clear/_apply_view.
            idx = self._display_idx
            self._right_cloud["RGB"] = np.ascontiguousarray(colors[idx]).astype(np.uint8)
            self._right_cloud.Modified()
            if not self._off_screen:
                self._right_pane.synchronize()
            return
        self._render_right(colors)
