"""Localization page: localize an rgb_X field-camera frame against a reconstruction."""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.dashboard.async_utils import run_off_loop
from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource

# NB: pipeline / localization viz imports are lazy (inside run/render paths) — they pull
# the heavy reconstruction stack, and the page must render immediately on launch.

logger = logging.getLogger(__name__)

########
# Constants + pure helpers (unit-tested)
########

_METHODS = ["disk", "xfeat", "loma", "loma-g"]
_DEFAULT_METHOD = "loma-g"
_SUBSAMPLE_ABOVE = 60  # plot every 3rd camera beyond this many reconstruction frames


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """World-space camera centers C = -R^T t from (N, 4, 4) world-to-camera transforms."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def subsample_step(n_cameras: int) -> int:
    """1 (all cameras) up to the threshold, 3 (every 3rd) beyond it."""
    return 1 if n_cameras <= _SUBSAMPLE_ABOVE else 3


def preselect_method(
    available_dbs: list[str], registered: list[str], default: str = _DEFAULT_METHOD
) -> tuple[list[str], str]:
    """Dropdown (options, value): prefer the default method's DB, then any existing DB,
    else the default (which will build on demand)."""
    if default in available_dbs:
        return registered, default
    if available_dbs:
        return registered, available_dbs[0]
    return registered, default


class SceneCache:
    """Session-level cache of expensive loads, keyed ((session, stem), kind).

    Known kinds: 'loaded' = SplatsApp's (result, mesh_path, semantics_dir, lifted_normed)
    tuple; 'mesh' = LocalizePage's pyvista mesh; 'localizer:*' = GPU-holding localizers.
    CPU loads (mesh, arrays) persist across tabs; GPU-holding entries use the
    'localizer:*' kind prefix so drop_kind('localizer') can evict them on tab switch."""

    # Kinds holding heavyweight objects get keep-last-N eviction; others are unbounded
    # ("loaded" is bounded by SplatsApp._remember_loaded; "localizer:*" by drop_kind).
    _KIND_KEEP = {"mesh": 3}

    def __init__(self) -> None:
        self._store: dict = {}
        self._order: dict[str, deque] = {}  # kind -> scene_key insertion order

    def get(self, scene_key, kind: str):
        return self._store.get((scene_key, kind))

    def put(self, scene_key, kind: str, value) -> None:
        self._store[(scene_key, kind)] = value
        keep = self._KIND_KEEP.get(kind)
        if keep is None:
            return
        order = self._order.setdefault(kind, deque())
        if scene_key in order:
            order.remove(scene_key)
        order.append(scene_key)
        while len(order) > keep:
            self._store.pop((order.popleft(), kind), None)

    def drop(self, scene_key, kind: str) -> None:
        """Remove one cache entry if present."""
        self._store.pop((scene_key, kind), None)

    def drop_scene(self, scene_key) -> None:
        """Remove every cached kind for a scene (fresh outputs invalidate them all)."""
        for key in [k for k in self._store if k[0] == scene_key]:
            del self._store[key]

    def drop_kind(self, prefix: str) -> None:
        """Evict every entry whose kind starts with prefix (e.g. GPU-holding localizers)."""
        for key in [k for k in self._store if k[1].startswith(prefix)]:
            del self._store[key]

    def clear(self) -> None:
        self._store.clear()


########
# Page
########


class LocalizePage(param.Parameterized):
    """Sidebar (scene + query + method) and three-panel result layout."""

    def __init__(
        self,
        base_dir: Path,
        source: SessionSource,
        gpu_worker: GpuWorker,
        op_log: OperationLog,
        cache: SceneCache | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source
        self._gpu = gpu_worker
        self._op_log = op_log
        self._cache = cache if cache is not None else SceneCache()
        self._build_sidebar()
        self._build_main()
        self._refresh_listings()

    # ---- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        """Scene (reconstruction) + query (field camera) + method widgets."""
        self.scene_session = pn.widgets.Select(name="Scene session", options=[])
        self.scene_video = pn.widgets.Select(name="Scene video", options=[])
        self.field_session = pn.widgets.Select(name="Field session", options=[])
        self.camera = pn.widgets.Select(name="Camera (rgb only)", options=[])
        self.query_video = pn.widgets.Select(name="Query video", options=[])
        self.frame_slider = pn.widgets.IntSlider(name="Frame", start=0, end=0, value=0)
        self.method = pn.widgets.Select(name="Method", options=_METHODS, value=_DEFAULT_METHOD)
        self.db_note = pn.pane.HTML("", sizing_mode="stretch_width")
        self.append_db = pn.widgets.Checkbox(name="Append localized frame to DB", value=True)
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")

        self.scene_session.param.watch(self._on_scene_session, "value")
        self.scene_video.param.watch(self._on_scene_video, "value")
        self.field_session.param.watch(self._on_field_session, "value")
        self.camera.param.watch(self._on_camera, "value")
        self.query_video.param.watch(self._on_query_video, "value")
        self.method.param.watch(self._on_method, "value")
        self.run_btn.on_click(self._on_run)

        self._sidebar = pn.Column(
            "## Scene",
            self.scene_session,
            self.scene_video,
            "## Query",
            self.field_session,
            self.camera,
            self.query_video,
            self.frame_slider,
            "## Localization",
            self.method,
            self.db_note,
            self.append_db,
            self.run_btn,
        )

    def sidebar(self) -> pn.Column:
        return self._sidebar

    # ---- main layout ---------------------------------------------------

    def _build_main(self) -> None:
        """Cheap result panes only — watchers fired during __init__ (e.g. _show_frame via
        _on_query_video) may touch these before main() is ever called. The heavy pyvista
        plotter + VTK pane are deferred to _ensure_plotter()."""
        self._frame_pane = pn.pane.Image(None, sizing_mode="scale_width")
        self._matches_col = pn.Column(self._frame_pane, sizing_mode="stretch_width", scroll=True, max_height=700)
        self._plotter: pv.Plotter | None = None
        self._vtk_pane: pn.pane.VTK | None = None
        self._dist_pane = pn.pane.Matplotlib(None, sizing_mode="stretch_width", tight=True)
        self._stats = pn.pane.HTML("", sizing_mode="stretch_width")

        # Progress strip: identical polling pattern to SplatsApp.main()
        self._progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_width")

    def _ensure_plotter(self) -> None:
        """Build the off-screen pyvista plotter + VTK pane on first use (lazy: main/_render_scene)."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=True)
            self._vtk_pane = pn.pane.VTK(self._plotter.ren_win, sizing_mode="stretch_both", min_height=500)

    def main(self) -> pn.Column:
        self._ensure_plotter()

        def _tick() -> None:
            self._progress.object = self._op_log.render_html()

        try:
            pn.state.add_periodic_callback(_tick, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

        top = pn.Row(self._matches_col, self._vtk_pane, sizing_mode="stretch_both")
        bottom = pn.Column(self._dist_pane, self._stats, sizing_mode="stretch_width")
        return pn.Column(top, bottom, self._progress, sizing_mode="stretch_both")

    def release_gpu(self) -> None:
        """Free GPU memory when the user leaves this tab (models reload on next run).

        Warm localizers hold the extractor model — evict them first or pytorch_gc
        cannot actually release the VRAM they reference."""
        from collab_splats.utils.torch_utils import pytorch_gc

        self._cache.drop_kind("localizer")
        pytorch_gc()

    # ---- listings (background threads, options set on the IOLoop) -------

    def _refresh_listings(self) -> None:
        """Populate scene sessions and field sessions off the IOLoop (rclone is blocking)."""
        doc = pn.state.curdoc

        def work():
            try:
                scenes = self._source.list_sessions()
            except Exception as exc:
                logger.warning("scene session listing failed: %s", exc)
                scenes = []
            try:
                fields = self._source.list_field_sessions()
            except Exception as exc:
                logger.warning("field session listing failed: %s", exc)
                fields = []

            def setter():
                self.scene_session.options = scenes
                self.field_session.options = fields

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="localize-list", daemon=True).start()

    def _on_scene_session(self, event) -> None:
        """Populate scene-video dropdown off the IOLoop (rclone list is blocking)."""
        if not event.new:
            return
        session = event.new
        run_off_loop(
            lambda: [Path(v).stem for v in self._source.list_videos(session)],
            lambda stems: setattr(self.scene_video, "options", stems),
            label="scene-video-list",
            doc=pn.state.curdoc,
        )

    def _on_scene_video(self, event) -> None:
        """Scene chosen → discover remote feature DBs and preselect the method."""
        if not event.new:
            return
        session, stem = self.scene_session.value, event.new
        doc = pn.state.curdoc

        def work():
            dbs = self._source.list_localization_dbs(session, stem)

            def setter():
                options, value = preselect_method(dbs, _METHODS)
                self.method.options = options
                self.method.value = value
                self._dbs = dbs
                self._update_db_note()

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="db-list", daemon=True).start()

    def _on_method(self, event) -> None:
        self._update_db_note()

    def _update_db_note(self) -> None:
        """Warn when the selected method has no DB yet (run will build it on GPU)."""
        dbs = getattr(self, "_dbs", [])
        if self.method.value in dbs:
            self.db_note.object = "<span style='color:#50c050;font-size:11px'>DB exists — will reuse</span>"
        else:
            self.db_note.object = (
                "<span style='color:#e0a050;font-size:11px'>no DB for this method — "
                "Run will build it (GPU, minutes)</span>"
            )

    def _on_field_session(self, event) -> None:
        """Populate camera dropdown off the IOLoop (rclone list is blocking)."""
        if not event.new:
            return
        fs = event.new
        run_off_loop(
            lambda: self._source.list_rgb_cameras(fs),
            lambda cams: setattr(self.camera, "options", cams),
            label="camera-list",
            doc=pn.state.curdoc,
        )

    def _on_camera(self, event) -> None:
        """Populate query-video dropdown off the IOLoop (rclone list is blocking)."""
        if not event.new:
            return
        fs, cam = self.field_session.value, event.new
        run_off_loop(
            lambda: self._source.list_camera_videos(fs, cam),
            lambda videos: setattr(self.query_video, "options", videos),
            label="camera-video-list",
            doc=pn.state.curdoc,
        )

    def _on_query_video(self, event) -> None:
        """Fetch the video in the background; set slider bound + preview frame 0."""
        if not event.new:
            return
        fs, cam, name = self.field_session.value, self.camera.value, event.new
        doc = pn.state.curdoc

        def work():
            try:
                video = self._ensure_local_query_video(fs, cam, name)
                from collab_splats.preproc import extract_frame, get_video_info

                total = int(get_video_info(str(video)).get("total_frames") or 1)
                frame = extract_frame(video, 0)
            except Exception:
                logger.warning("query video fetch/preview failed", exc_info=True)
                return

            def setter():
                self.frame_slider.end = max(total - 1, 0)
                self.frame_slider.value = 0
                self._show_frame(frame)

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="query-video", daemon=True).start()

    def _ensure_local_query_video(self, field_session: str, camera: str, name: str) -> Path:
        local = self._base_dir / "queries" / field_session / camera / name
        if local.exists():
            return local
        on_line = self._op_log.rclone_progress("⬇ fetching query video")
        return self._source.fetch_field_video(field_session, camera, name, local.parent, on_line=on_line)

    def _show_frame(self, frame: np.ndarray) -> None:
        """Show the selected query frame in the left panel (pre-run state)."""
        from PIL import Image as PILImage

        # pn.pane.Image renders PIL images directly; raw bytes are not accepted
        self._frame_pane.object = PILImage.fromarray(frame)
        self._matches_col[:] = [self._frame_pane]

    # ---- run -----------------------------------------------------------

    def _current_config(self) -> LocalizationConfig:
        return LocalizationConfig(extractor=self.method.value, append_to_db=self.append_db.value)

    def _on_run(self, event) -> None:
        scene_session = self.scene_session.value
        stem = self.scene_video.value
        fs, cam, name = self.field_session.value, self.camera.value, self.query_video.value
        frame_idx = self.frame_slider.value
        if not (scene_session and stem and fs and cam and name):
            self._op_log.error_op("select a scene and a query video first")
            return
        config = self._current_config()
        provenance = {
            "video_ref": f"{fs}/{cam}/{name}",
            "session": fs,
            "camera": cam,
            "frame_idx": int(frame_idx),
        }
        doc = pn.state.curdoc

        def job():
            # Lazy import: pulls the heavy stack only when a run starts (mirrors SplatsApp)
            from collab_splats.dashboard.pipeline import run_localization

            video = self._ensure_local_query_video(fs, cam, name)
            return run_localization(
                query_video=video,
                frame_idx=frame_idx,
                session=scene_session,
                stem=stem,
                config=config,
                op_log=self._op_log,
                source=self._source,
                base_dir=self._base_dir,
                provenance=provenance,
                cache=self._cache,  # keeps the localizer (and its extractor) warm across runs
            )

        def on_done(res):
            self.run_btn.disabled = False
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._render_result(res, config)

        self.run_btn.disabled = True
        self._gpu.submit(job, on_done, doc)

    # ---- rendering -----------------------------------------------------

    def _render_result(self, out, config: LocalizationConfig) -> None:
        """Fill all three panels from a LocalizationRunOutput (IOLoop thread)."""
        # Lazy: viz builds figures via plt.subplots(), so pull pyplot to close superseded ones
        import matplotlib.pyplot as plt

        from collab_splats.localization.viz import (
            plot_correspondences,
            plot_inlier_distribution,
        )

        try:
            loc = out.result
            n_frames = len(out.ref_image_paths)

            # Close the outgoing match-pair figures before rebuilding the column so they do
            # not accumulate in pyplot's global registry (the pre-run frame pane is an Image)
            for child in list(self._matches_col):
                if isinstance(child, pn.pane.Matplotlib) and child.object is not None:
                    plt.close(child.object)

            # Bottom: inlier distribution + summary stats (close the superseded dist figure)
            old_dist = self._dist_pane.object
            fig = plot_inlier_distribution(loc, n_frames=n_frames, frame_sources=out.frame_sources)
            self._dist_pane.object = fig
            if old_dist is not None:
                plt.close(old_dist)
            ratio = 100 * loc.n_inliers / max(loc.n_correspondences, 1)
            pose_msg = "" if loc.pose is not None else " — <b style='color:#e05050'>POSE FAILED</b>"
            self._stats.object = (
                f"<div style='font-size:12px'>inliers {loc.n_inliers}/{loc.n_correspondences} "
                f"({ratio:.0f}%) · intrinsics: {out.intrinsics_source} "
                f"(fx={out.query_intrinsics[0, 0]:.0f}){pose_msg}</div>"
            )

            # Left: top-k match-pair figures, best-first (replaces the frame preview)
            if loc.ref_frame_indices is not None and loc.inlier_mask is not None:
                counts = np.bincount(loc.ref_frame_indices[loc.inlier_mask].astype(np.intp), minlength=n_frames)
                top = np.argsort(counts)[::-1][: config.top_k_viz]
                panes = []
                for ref in top:
                    if counts[ref] == 0 or not Path(out.ref_image_paths[ref]).exists():
                        continue
                    mfig = plot_correspondences(
                        loc,
                        out.query_frame,
                        out.ref_image_paths,
                        max_pairs=config.max_pairs,
                        ref_idx=int(ref),
                        show=False,
                    )
                    if mfig is not None:
                        panes.append(pn.pane.Matplotlib(mfig, sizing_mode="stretch_width", tight=True))
                if panes:
                    self._matches_col[:] = panes

            # Right: mesh + viridis reconstruction cameras + red localized camera
            scene_key = (self.scene_session.value, self.scene_video.value)
            mesh_path = self._base_dir / scene_key[0] / scene_key[1] / "mesh" / "mesh_tsdf.ply"
            self._render_scene(scene_key, mesh_path, out.ref_extrinsics, loc.pose)
        except Exception as exc:
            # Surface a render failure via the op_log instead of escaping to the IOLoop
            logger.warning("localize render failed", exc_info=True)
            self._op_log.error_op(str(exc))

    def _render_scene(
        self, scene_key, mesh_path: Path, extrinsics: np.ndarray, localized_pose: "np.ndarray | None"
    ) -> None:
        """Rebuild the 3D pane: mesh, time-coloured cameras, red localized camera."""
        self._ensure_plotter()
        self._plotter.clear()

        # Mesh (cached across runs and tabs — expensive read)
        mesh = self._cache.get(scene_key, "mesh")
        if mesh is None and mesh_path.exists():
            mesh = pv.read(str(mesh_path))
            self._cache.put(scene_key, "mesh", mesh)
        if mesh is not None:
            self._plotter.add_mesh(mesh, rgb="RGB" in mesh.array_names, opacity=0.9)

        # Reconstruction cameras: viridis by time; subsampled with an on-plot note
        centers = camera_centers(np.asarray(extrinsics))
        step = subsample_step(len(centers))
        sub = centers[::step]
        poly = pv.PolyData(sub)
        poly["time"] = np.arange(len(sub), dtype=np.float32)
        self._plotter.add_mesh(
            poly, scalars="time", cmap="viridis", point_size=14, render_points_as_spheres=True, show_scalar_bar=False
        )
        if step > 1:
            self._plotter.add_text(f"showing every {step}rd camera", font_size=8, position="lower_left")

        # Localized camera in red, drawn larger
        if localized_pose is not None:
            loc_center = camera_centers(localized_pose[np.newaxis])
            self._plotter.add_mesh(pv.PolyData(loc_center), color="red", point_size=22, render_points_as_spheres=True)

        self._plotter.reset_camera()
        self._vtk_pane.synchronize()
