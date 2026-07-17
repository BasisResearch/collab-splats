"""Localization page: localize an rgb_X field-camera frame against a reconstruction."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.dashboard.async_utils import run_off_loop
from collab_splats.dashboard.config import LocalizationConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog, busy_html
from collab_splats.dashboard.sources import SessionSource

# NB: pipeline / localization viz imports are lazy (inside run/render paths) — they pull
# the heavy reconstruction stack, and the page must render immediately on launch.

logger = logging.getLogger(__name__)

########
# Constants + pure helpers (unit-tested)
########

_METHODS = ["disk", "xfeat", "loma", "loma-g"]
_DEFAULT_METHOD = "loma"
_SUBSAMPLE_ABOVE = 60  # plot every 3rd camera beyond this many reconstruction frames
_PREVIEW_DEBOUNCE_S = 0.3  # slider settles this long before a frame decode fires
_PREVIEW_MAX_W = 640  # thumbnail width pushed to the browser (full-res is wasteful)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """World-space camera centers C = -R^T t from (N, 4, 4) world-to-camera transforms."""
    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    return -np.einsum("nji,nj->ni", R, t)


def subsample_step(n_cameras: int) -> int:
    """1 (all cameras) up to the threshold, 3 (every 3rd) beyond it."""
    return 1 if n_cameras <= _SUBSAMPLE_ABOVE else 3


def select_options(items: list, hint: str = "— select —") -> dict:
    """Blank-first dropdown map: nothing auto-selects or cascades until the user picks.

    Auto-picking the first option cascaded listings/fetches for a scene the user never
    chose (and displayed the wrong video names against fieldwork_curated).
    """
    options = {hint: ""}
    options.update({i: i for i in items})
    return options


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
        self._preview_token = 0  # latest slider request; stale extracts are dropped
        self._preview_timer: threading.Timer | None = None
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
        # Cross-tab busy indicator: filled while any GpuWorker job is in flight.
        self.busy_note = pn.pane.HTML("", sizing_mode="stretch_width")

        self.scene_session.param.watch(self._on_scene_session, "value")
        self.scene_video.param.watch(self._on_scene_video, "value")
        self.field_session.param.watch(self._on_field_session, "value")
        self.camera.param.watch(self._on_camera, "value")
        self.query_video.param.watch(self._on_query_video, "value")
        self.frame_slider.param.watch(self._on_frame_slider, "value")
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
            self.busy_note,
        )

    def sidebar(self) -> pn.Column:
        return self._sidebar

    def set_busy(self, busy: bool) -> None:
        """Enable/disable this page's mutating widgets while a GPU job is in flight."""
        widgets = (
            self.run_btn,
            self.scene_session,
            self.scene_video,
            self.field_session,
            self.camera,
            self.query_video,
            self.method,
            self.append_db,
        )
        for w in widgets:
            w.disabled = busy
        self.busy_note.object = busy_html(self._op_log.current_op) if busy else ""

    def _sync_busy(self) -> None:
        """Poll hook: mirror the shared worker's busy flag onto this page's widgets."""
        busy = bool(self._gpu.busy)
        if busy != self.run_btn.disabled:
            self.set_busy(busy)
        elif busy:
            # Refresh the label while busy (current_op advances through the run).
            self.busy_note.object = busy_html(self._op_log.current_op)

    # ---- main layout ---------------------------------------------------

    def _build_main(self) -> None:
        """Cheap result panes only — watchers fired during __init__ (e.g. _show_frame via
        _on_query_video) may touch these before main() is ever called. The heavy pyvista
        plotter + VTK pane are deferred to _ensure_plotter()."""
        self._frame_pane = pn.pane.Image(None, sizing_mode="scale_width")
        # Placeholder until a query video is selected (frame preview replaces it) or a run
        # completes (correspondence figures replace it) — a blank pane reads as broken.
        placeholder = pn.pane.HTML(
            "<i style='color:#888'>Select a scene and a query video, then Run. "
            "The selected frame previews here; progress shows in the Operations console.</i>"
        )
        self._matches_col = pn.Column(placeholder, sizing_mode="stretch_width", scroll=True, max_height=700)
        self._plotter: pv.Plotter | None = None
        self._vtk_pane: pn.pane.VTK | None = None
        self._dist_pane = pn.pane.Matplotlib(None, sizing_mode="stretch_width", tight=True)
        self._stats = pn.pane.HTML("", sizing_mode="stretch_width")

    def _ensure_plotter(self) -> None:
        """Build the off-screen pyvista plotter + VTK pane on first use (lazy: main/_render_scene)."""
        if self._plotter is None:
            self._plotter = pv.Plotter(off_screen=True)
            self._vtk_pane = pn.pane.VTK(self._plotter.ren_win, sizing_mode="stretch_both", min_height=500)

    def main(self) -> pn.Column:
        self._ensure_plotter()

        # Busy-state poll only: the operations console is rendered ONCE by DashboardShell,
        # outside the tabs, so it stays visible on both tabs.
        try:
            pn.state.add_periodic_callback(self._sync_busy, period=300, start=True)
        except Exception:
            logger.debug("no periodic callback (no server doc)", exc_info=True)

        top = pn.Row(self._matches_col, self._vtk_pane, sizing_mode="stretch_both")
        bottom = pn.Column(self._dist_pane, self._stats, sizing_mode="stretch_width")
        return pn.Column(top, bottom, sizing_mode="stretch_both")

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
            # step()'s FAILED lines are the user-visible surface (no error_op: that would
            # clobber a concurrent run's is_running); fall through with empty lists
            # so the dropdowns don't wedge.
            try:
                with self._op_log.step("listing scene sessions"):
                    scenes = self._source.list_sessions()
            except Exception as exc:
                logger.warning("scene session listing failed: %s", exc)
                scenes = []
            try:
                with self._op_log.step("listing field sessions"):
                    fields = self._source.list_field_sessions()
            except Exception as exc:
                logger.warning("field session listing failed: %s", exc)
                fields = []

            def setter():
                # Blank-first: no session auto-selects, so no listing cascade fires
                # until the user explicitly picks one.
                self.scene_session.options = select_options(scenes, "— select scene session —")
                self.field_session.options = select_options(fields, "— select field session —")

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="localize-list", daemon=True).start()

    def _on_scene_session(self, event) -> None:
        """Populate scene-video dropdown off the IOLoop (rclone list is blocking)."""
        if not event.new:
            return
        session = event.new

        # step()'s FAILED line surfaces listing errors in the op log (run_off_loop swallows).
        def fetch():
            with self._op_log.step("listing scene videos"):
                return [Path(v).stem for v in self._source.list_videos(session)]

        run_off_loop(
            fetch,
            lambda stems: setattr(self.scene_video, "options", select_options(stems, "— select scene video —")),
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
            # step() logs start/done and FAILED; bail on failure (note stays as-is).
            try:
                with self._op_log.step("listing feature DBs"):
                    dbs = self._source.list_localization_dbs(session, stem)
            except Exception as exc:
                logger.warning("feature DB listing failed: %s", exc, exc_info=True)
                return

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

        # step()'s FAILED line surfaces listing errors in the op log (run_off_loop swallows).
        def fetch():
            with self._op_log.step("listing cameras"):
                return self._source.list_rgb_cameras(fs)

        run_off_loop(
            fetch,
            lambda cams: setattr(self.camera, "options", select_options(cams, "— select camera —")),
            label="camera-list",
            doc=pn.state.curdoc,
        )

    def _on_camera(self, event) -> None:
        """Populate query-video dropdown off the IOLoop (rclone list is blocking)."""
        if not event.new:
            return
        fs, cam = self.field_session.value, event.new

        # step()'s FAILED line surfaces listing errors in the op log (run_off_loop swallows).
        def fetch():
            with self._op_log.step("listing camera videos"):
                return self._source.list_camera_videos(fs, cam)

        run_off_loop(
            fetch,
            lambda videos: setattr(self.query_video, "options", select_options(videos, "— select query video —")),
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
            # step() logs fetch start/done/FAILED; the extra line marks the preview loss.
            try:
                with self._op_log.step(f"fetching query video {name}"):
                    video = self._ensure_local_query_video(fs, cam, name)
                    from collab_splats.preproc import extract_frame, get_video_info

                    total = int(get_video_info(str(video)).get("total_frames") or 1)
                    frame = extract_frame(video, 0)
            except Exception as exc:
                logger.warning("query video fetch/preview failed", exc_info=True)
                self._op_log.append_line(f"query video preview FAILED: {exc}")
                return

            def setter():
                self.frame_slider.end = max(total - 1, 0)
                self.frame_slider.value = 0
                self._show_frame(frame)

            doc.add_next_tick_callback(setter) if doc is not None else setter()

        threading.Thread(target=work, name="query-video", daemon=True).start()

    def _on_frame_slider(self, event) -> None:
        """Debounced live preview: decode + show the frame shortly after the slider settles."""
        if self._gpu.busy:
            return  # a run owns the panes; the slider still sets the run's frame_idx
        if not (self.field_session.value and self.camera.value and self.query_video.value):
            return
        self._preview_token += 1
        if self._preview_timer is not None:
            self._preview_timer.cancel()
        self._preview_timer = threading.Timer(
            _PREVIEW_DEBOUNCE_S,
            self._preview_frame,
            kwargs={"token": self._preview_token, "frame_idx": event.new, "doc": pn.state.curdoc},
        )
        self._preview_timer.daemon = True
        self._preview_timer.start()

    def _preview_frame(self, token: int, frame_idx: int, doc) -> None:
        """Timer thread: fast-seek decode, then marshal display back to the IOLoop."""
        if token != self._preview_token:
            return  # superseded by a newer slider position
        fs, cam, name = self.field_session.value, self.camera.value, self.query_video.value
        t0 = time.perf_counter()
        try:
            video = self._ensure_local_query_video(fs, cam, name)
            from collab_splats.preproc import extract_frame_fast

            frame = extract_frame_fast(video, frame_idx)
        except Exception as exc:
            logger.warning("frame preview failed", exc_info=True)
            self._op_log.append_line(f"frame {frame_idx} preview FAILED: {exc}")
            return
        if token != self._preview_token:
            return
        elapsed = time.perf_counter() - t0

        def show() -> None:
            if token != self._preview_token:
                return
            self._show_frame(frame)
            self._op_log.append_line(f"frame {frame_idx} loaded ({elapsed:.1f}s)")

        doc.add_next_tick_callback(show) if doc is not None else show()

    def _ensure_local_query_video(self, field_session: str, camera: str, name: str) -> Path:
        local = self._base_dir / "queries" / field_session / camera / name
        if local.exists():
            return local
        on_line = self._op_log.rclone_progress("⬇ fetching query video")
        return self._source.fetch_field_video(field_session, camera, name, local.parent, on_line=on_line)

    def _show_frame(self, frame: np.ndarray) -> None:
        """Show the selected query frame in the left panel, downscaled to a thumbnail."""
        from PIL import Image as PILImage

        # pn.pane.Image renders PIL images directly; full-res frames push MBs of base64
        # into the doc, so cap the preview width (display is scale_width anyway).
        img = PILImage.fromarray(frame)
        if img.width > _PREVIEW_MAX_W:
            img = img.resize((_PREVIEW_MAX_W, max(1, int(img.height * _PREVIEW_MAX_W / img.width))))
        self._frame_pane.object = img
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
        # Captured for the worker: on_done must only assign panes, never hit the disk
        scene_key = (scene_session, stem)
        mesh_path = self._base_dir / scene_session / stem / "mesh" / "mesh_tsdf.ply"
        doc = pn.state.curdoc

        def job():
            # Lazy import: pulls the heavy stack only when a run starts (mirrors SplatsApp)
            from collab_splats.dashboard.pipeline import run_localization

            video = self._ensure_local_query_video(fs, cam, name)
            out = run_localization(
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
            # Figures + mesh read are slow — build them here so on_done only assigns panes.
            with self._op_log.step("building result figures"):
                figs = self._build_result_figures(out, config)
            mesh = self._ensure_scene_mesh(scene_key, mesh_path)
            return (out, figs, mesh)

        def on_done(res):
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            out, figs, mesh = res
            self._render_result(out, figs, mesh)

        self.set_busy(True)
        self._gpu.submit(job, on_done, doc)

    # ---- rendering -----------------------------------------------------

    def _build_result_figures(self, out, config: LocalizationConfig) -> dict:
        """Build all matplotlib figures + stats HTML for a run output (worker thread — pure).

        Runs on the worker with the Agg backend. Figures are pyplot-managed (Gcf), so
        there is a theoretical cross-thread window vs the IOLoop's plt.close — benign
        under CPython/Agg; migrate viz to direct Figure() construction if it ever bites.
        """
        from collab_splats.localization.viz import (
            plot_correspondences,
            plot_inlier_distribution,
        )

        loc = out.result
        n_frames = len(out.ref_image_paths)

        # Bottom: inlier distribution + summary stats
        dist_fig = plot_inlier_distribution(loc, n_frames=n_frames, frame_sources=out.frame_sources)
        ratio = 100 * loc.n_inliers / max(loc.n_correspondences, 1)
        pose_msg = "" if loc.pose is not None else " — <b style='color:#e05050'>POSE FAILED</b>"
        stats_html = (
            f"<div style='font-size:12px'>inliers {loc.n_inliers}/{loc.n_correspondences} "
            f"({ratio:.0f}%) · intrinsics: {out.intrinsics_source} "
            f"(fx={out.query_intrinsics[0, 0]:.0f}){pose_msg}</div>"
        )

        # Top-k match-pair figures, best-first
        match_figs = []
        if loc.ref_frame_indices is not None and loc.inlier_mask is not None:
            counts = np.bincount(loc.ref_frame_indices[loc.inlier_mask].astype(np.intp), minlength=n_frames)
            top = np.argsort(counts)[::-1][: config.top_k_viz]
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
                    match_figs.append(mfig)
        return {"dist_fig": dist_fig, "match_figs": match_figs, "stats_html": stats_html}

    def _ensure_scene_mesh(self, scene_key, mesh_path: Path):
        """Read the scene mesh with cache (worker thread — pv.read is a blocking disk read)."""
        mesh = self._cache.get(scene_key, "mesh")
        if mesh is None and mesh_path.exists():
            with self._op_log.step("reading scene mesh"):
                mesh = pv.read(str(mesh_path))
            self._cache.put(scene_key, "mesh", mesh)
        return mesh

    def _render_result(self, out, figs: dict, mesh) -> None:
        """Fill all three panels from pre-built figures + mesh (IOLoop thread — assignment only)."""
        # Lazy: pyplot pulled only to close superseded figures (viz built them on the worker)
        import matplotlib.pyplot as plt

        try:
            # Close the outgoing match-pair figures before rebuilding the column so they do
            # not accumulate in pyplot's global registry (the pre-run frame pane is an Image)
            for child in list(self._matches_col):
                if isinstance(child, pn.pane.Matplotlib) and child.object is not None:
                    plt.close(child.object)

            # Bottom: inlier distribution + summary stats (close the superseded dist figure)
            old_dist = self._dist_pane.object
            self._dist_pane.object = figs["dist_fig"]
            if old_dist is not None:
                plt.close(old_dist)
            self._stats.object = figs["stats_html"]

            # Left: top-k match-pair figures, best-first (replaces the frame preview)
            if figs["match_figs"]:
                self._matches_col[:] = [
                    pn.pane.Matplotlib(f, sizing_mode="stretch_width", tight=True) for f in figs["match_figs"]
                ]

            # Right: mesh + viridis reconstruction cameras + red localized camera
            self._render_scene(mesh, out.ref_extrinsics, out.result.pose)
        except Exception as exc:
            # Surface a render failure via the op_log instead of escaping to the IOLoop
            logger.warning("localize render failed", exc_info=True)
            self._op_log.error_op(str(exc))

    def _render_scene(self, mesh, extrinsics: np.ndarray, localized_pose: "np.ndarray | None") -> None:
        """Rebuild the 3D pane: mesh, time-coloured cameras, red localized camera."""
        self._ensure_plotter()
        self._plotter.clear()

        # Mesh arrives preloaded (worker read it via _ensure_scene_mesh)
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
