# collab_splats/dashboard/app.py
"""Dashboard entry points: single-page SplatsApp."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import panel as pn
import param
import yaml

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource
from collab_splats.dashboard.viewer import SplitViewer

# NB: collab_splats.dashboard.pipeline and pointcloud.feedforward pull in the full
# reconstruction + TSDF mesh stack (~18s import). They are imported lazily inside the
# run/load paths so the server binds and the page renders immediately on launch.

logger = logging.getLogger(__name__)

########
# Helpers (kept for backward compat — used by tests and __init__)
########


def _scan_output_dirs(base_dir: Path) -> list[str]:
    """Return sorted names of subdirs in base_dir that contain run_config.yaml."""
    if not base_dir.is_dir():
        return []
    return sorted(p.name for p in base_dir.iterdir() if p.is_dir() and (p / "run_config.yaml").exists())


########
# SplatsApp — new single-page dashboard
########

_ENV_MODELS = ["vggt_omega", "vggtx", "mapanything"]
_EXTRACTORS = ["talk2dino", "maskclip", "dinov2"]
_SAMPLERS = ["balanced", "optical_flow"]

# Per-model confidence default — mirrors each creator's own class default so the dashboard
# reproduces the notebook (which instantiates creators with no conf override). A single shared
# default (35) silently ran VGGT-Omega far below its native 50 cutoff, keeping low-confidence
# flyers that distort the cloud. Keep in sync with VGGTOmegaCreator.conf_threshold=50,
# VGGTXCreator.conf_threshold=35, MapAnythingCreator.confidence_percentile=35.
_MODEL_CONF_DEFAULTS = {"vggt_omega": 50.0, "vggtx": 35.0, "mapanything": 35.0}


def _bind_visibility(widget, selector, predicate) -> None:
    """Show widget only when predicate(selector.value) holds; re-evaluate on change."""
    widget.visible = predicate(selector.value)
    selector.param.watch(lambda e: setattr(widget, "visible", predicate(e.new)), "value")


def _split_terms(text: str) -> list[str]:
    """Split a comma-separated query box into a list of non-empty phrases."""
    return [t.strip() for t in text.split(",") if t.strip()]


class SplatsApp(param.Parameterized):
    """Single-page dashboard wiring source, pipeline, and the split viewer."""

    def __init__(
        self,
        base_dir: Path = Path("/workspace/outputs"),
        source: SessionSource | None = None,
        gpu_worker: GpuWorker | None = None,
        op_log: OperationLog | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
        self._gpu = gpu_worker if gpu_worker is not None else GpuWorker()
        # Shared across sessions (passed from run_app) so a page refresh re-attaches to an
        # in-flight run's progress instead of spawning a fresh, disconnected log.
        self._op_log = op_log if op_log is not None else OperationLog()
        self._viewer = SplitViewer()
        # Persisted UI state survives browser reloads (each reload rebuilds widgets fresh).
        self._state_path = self._base_dir / ".dashboard_state.yaml"
        self._state = self._load_state()
        self._restored_selection = False  # session/video restored once, after options load
        self._suppress_autoload = False  # gate _on_video during programmatic option/restore churn
        self._build_sidebar()
        self._refresh_sessions()

    def _load_state(self) -> dict:
        """Load persisted widget values; empty dict if absent/unreadable."""
        try:
            if self._state_path.exists():
                return yaml.safe_load(self._state_path.read_text()) or {}
        except Exception:
            logger.warning("could not read dashboard state", exc_info=True)
        return {}

    def _persist_state(self, *_event) -> None:
        """Write current widget values to disk so they survive a browser reload."""
        data = {k: w.value for k, w in self._persisted.items()}
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(yaml.safe_dump(data, sort_keys=False))
        except Exception:
            logger.warning("could not persist dashboard state", exc_info=True)

    # ---- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        """Build all sidebar widgets and wire callbacks."""
        # Seed each widget from persisted state (falls back to the literal default).
        s = self._state
        self.session_select = pn.widgets.Select(name="Session", options=[])
        self.video_select = pn.widgets.Select(name="Video", options=[])
        self.sampling = pn.widgets.Select(name="Frame sampling", options=_SAMPLERS, value=s.get("sampling", "balanced"))
        # Number input (not a slider); upper bound + label set to the video's frame count on select.
        self.max_frames = pn.widgets.IntInput(name="Max frames", value=s.get("max_frames", 100), start=1, step=1)
        self.env_model = pn.widgets.Select(
            name="Environment model", options=_ENV_MODELS, value=s.get("env_model", "vggt_omega")
        )
        self.conf = pn.widgets.FloatSlider(
            name="Confidence",
            start=0,
            end=100,
            value=s.get("conf", _MODEL_CONF_DEFAULTS[s.get("env_model", "vggt_omega")]),
        )
        self.extractor = pn.widgets.Select(
            name="Semantic model", options=_EXTRACTORS, value=s.get("extractor", "talk2dino")
        )
        self.pos_query = pn.widgets.TextInput(
            name="Positive query", placeholder="e.g. chair, stool", value=s.get("pos_query", "")
        )
        self.neg_query = pn.widgets.TextInput(
            name="Negative query", placeholder="e.g. floor, wall", value=s.get("neg_query", "background, sky")
        )
        self.run_query_btn = pn.widgets.Button(label="Run query", button_type="primary")
        self.min_disparity = pn.widgets.FloatInput(name="min_disparity", value=s.get("min_disparity", 50.0))
        self.mesh_voxel = pn.widgets.FloatInput(name="voxel_size", value=s.get("mesh_voxel", 0.005))
        self.mesh_sdf = pn.widgets.FloatInput(name="sdf_trunc", value=s.get("mesh_sdf", 0.02))
        self.mesh_depth = pn.widgets.FloatInput(name="depth_trunc", value=s.get("mesh_depth", 1.0))
        self.mesh_clean = pn.widgets.Checkbox(name="clean_repair", value=s.get("mesh_clean", False))
        self.max_display_points = pn.widgets.IntInput(
            name="Max display points", value=s.get("max_display_points", 500_000), step=50_000
        )
        self.view_mode = pn.widgets.RadioButtonGroup(options=["pointcloud", "mesh"], value="pointcloud")
        self.normalize_view = pn.widgets.Checkbox(
            name="Normalize view (orient + scale)", value=s.get("normalize_view", True)
        )
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(label="Force re-run", button_type="warning")

        self.session_select.param.watch(self._on_session, "value")
        self.video_select.param.watch(self._on_video, "value")
        self.env_model.param.watch(self._on_env_model, "value")
        self.run_query_btn.on_click(self._on_query)
        self.view_mode.param.watch(self._on_view_mode, "value")
        self.normalize_view.param.watch(lambda e: self._viewer.set_normalize_view(e.new), "value")
        self.run_btn.on_click(lambda e: self._on_run(e, force=False))
        self.force_btn.on_click(lambda e: self._on_run(e, force=True))

        # Persist these widgets' values to disk on any change so a browser reload restores them.
        # session/video are restored once, after their options populate (see _restore_selection).
        self._persisted = {
            "session_select": self.session_select,
            "video_select": self.video_select,
            "sampling": self.sampling,
            "max_frames": self.max_frames,
            "env_model": self.env_model,
            "conf": self.conf,
            "extractor": self.extractor,
            "pos_query": self.pos_query,
            "neg_query": self.neg_query,
            "min_disparity": self.min_disparity,
            "mesh_voxel": self.mesh_voxel,
            "mesh_sdf": self.mesh_sdf,
            "mesh_depth": self.mesh_depth,
            "mesh_clean": self.mesh_clean,
            "max_display_points": self.max_display_points,
            "normalize_view": self.normalize_view,
        }
        for _w in self._persisted.values():
            _w.param.watch(self._persist_state, "value")

        # min_disparity is consumed only by the optical-flow sampler; hide it otherwise.
        _bind_visibility(self.min_disparity, self.sampling, lambda v: v == "optical_flow")

        self._sidebar = pn.Column(
            "## Source",
            self.session_select,
            self.video_select,
            pn.Card(self.sampling, self.max_frames, self.min_disparity, title="Frame sampling", collapsed=True),
            pn.Card(self.env_model, self.conf, title="Environment model", collapsed=True),
            pn.Card(
                self.extractor, self.pos_query, self.neg_query, self.run_query_btn, title="Semantics", collapsed=False
            ),
            pn.Card(
                self.mesh_voxel, self.mesh_sdf, self.mesh_depth, self.mesh_clean, title="Mesh params", collapsed=True
            ),
            self.max_display_points,
            "### View",
            self.view_mode,
            self.normalize_view,
            pn.Row(self.run_btn, self.force_btn),
        )

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable the action buttons while a GPU job is in flight (IOLoop thread)."""
        for btn in (self.run_btn, self.force_btn, self.run_query_btn):
            btn.disabled = busy

    # ---- data wiring ---------------------------------------------------

    def _refresh_sessions(self) -> None:
        """List sessions on a background thread; set options back on the IOLoop.

        rclone listing is a blocking network call; running it inline would stall the
        IOLoop during document init (the same class of freeze as the heavy imports).
        """
        doc = pn.state.curdoc

        def work():
            try:
                names = self._source.list_sessions()
            except Exception as exc:
                logger.warning("session listing failed: %s", exc)
                names = []
            self._apply_sessions(names, doc)

        self._session_thread = threading.Thread(target=work, name="session-list", daemon=True)
        self._session_thread.start()

    def _apply_sessions(self, names: list[str], doc) -> None:
        """Set the session dropdown options on the IOLoop (or inline if no doc)."""

        def setter():
            # Populating options flips value→options[0], cascading _on_session/_on_video and
            # auto-loading the WRONG (first) scene. Gate auto-load while we churn options + restore,
            # then issue exactly one load for the final selection.
            self._suppress_autoload = True
            try:
                self.session_select.options = names
                if not self._restored_selection:
                    self._restored_selection = True
                    self._restore_selection(names)
            finally:
                self._suppress_autoload = False
            self._autoload_current()

        if doc is not None:
            doc.add_next_tick_callback(setter)
        else:
            setter()

    def _restore_selection(self, names: list[str]) -> None:
        """Re-apply the persisted session + video once their option lists are available."""
        sess = self._state.get("session_select")
        if not sess or sess not in names:
            return
        self.session_select.value = sess
        # Populate the video options for this session, then re-select the saved video.
        try:
            videos = self._source.list_videos(sess)
        except Exception:
            logger.warning("could not list videos for restored session %s", sess, exc_info=True)
            return
        self.video_select.options = videos
        vid = self._state.get("video_select")
        if vid and vid in videos:
            self.video_select.value = vid  # final load is issued once by the setter (suppressed here)

    def _on_session(self, event) -> None:
        """Populate video dropdown when session changes."""
        if not event.new:
            return
        self.video_select.options = self._source.list_videos(event.new)

    def _on_video(self, event) -> None:
        """Auto-load cached outputs when a video is selected (skipped during programmatic churn)."""
        if self._suppress_autoload or not event.new:
            return
        self._update_max_frames_bound(self.session_select.value, event.new)
        self._autoload_current()

    def _update_max_frames_bound(self, session: str, name: str) -> None:
        """Set the Max-frames upper bound + label to the selected video's total frame count.

        Reading frame count needs the file. If it isn't local yet (remote bucket), fetch it on a
        background thread (Run needs it anyway) and apply the bound on the IOLoop when it arrives.
        """
        if not session or not name:
            return
        local = self._base_dir / session / Path(name).stem / name
        if local.exists():
            self._apply_max_frames_bound(local)
            return
        # Not local: fetch in the background, then set the bound back on the IOLoop.
        doc = pn.state.curdoc

        def work() -> None:
            try:
                video = self._ensure_local_video(session, name)
            except Exception:
                logger.warning("could not fetch video for frame count: %s/%s", session, name, exc_info=True)
                return
            if doc is not None:
                doc.add_next_tick_callback(lambda: self._apply_max_frames_bound(video))
            else:
                self._apply_max_frames_bound(video)

        threading.Thread(target=work, name="video-meta", daemon=True).start()

    def _apply_max_frames_bound(self, video: Path) -> None:
        """Read total frame count from a local video and reflect it in the Max-frames widget."""
        try:
            from collab_splats.preproc import get_video_info

            total = int(get_video_info(str(video)).get("total_frames") or 0)
        except Exception:
            logger.warning("could not read frame count for %s", video, exc_info=True)
            return
        if total > 0:
            self.max_frames.end = total
            self.max_frames.name = f"Max frames (video has {total})"

    def _autoload_current(self) -> None:
        """Load cached outputs for the currently-selected session/video, if present."""
        name = self.video_select.value
        if not name:
            return
        session, stem = self.session_select.value, Path(name).stem
        self._update_max_frames_bound(session, name)
        # A refresh mid-run must not clobber the shared op_log or queue a load behind the pipeline.
        if self._op_log.is_running:
            return
        out = self._base_dir / session / stem
        if (out / "feedforward.zarr").exists() or self._source.has_processed(session, stem):
            self._load_outputs(session, stem)

    def _on_env_model(self, event) -> None:
        """Reset confidence to the selected model's native default (matches the notebook)."""
        if event.new in _MODEL_CONF_DEFAULTS:
            self.conf.value = _MODEL_CONF_DEFAULTS[event.new]

    def _current_config(self) -> RunConfig:
        """Build RunConfig from current widget values."""
        return RunConfig(
            sampling_method=self.sampling.value,
            max_frames=self.max_frames.value,
            min_disparity=self.min_disparity.value,
            env_model=self.env_model.value,
            conf_threshold=self.conf.value,
            semantic_extractor=self.extractor.value,
            query_positive=self.pos_query.value,
            query_negative=self.neg_query.value,
            mesh_voxel_size=self.mesh_voxel.value,
            mesh_sdf_trunc=self.mesh_sdf.value,
            mesh_depth_trunc=self.mesh_depth.value,
            mesh_clean_repair=self.mesh_clean.value,
        )

    def _ensure_local_video(self, session: str, name: str) -> Path:
        """Return local video path, fetching from source if needed."""
        local = self._base_dir / session / Path(name).stem / name
        if not local.exists():
            local = self._source.fetch_video(session, name, local.parent)
        return local

    def _on_run(self, event, force: bool) -> None:
        """Run or reload the pipeline, respecting cache and force flag."""
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            return
        stem = Path(name).stem
        out = self._base_dir / session / stem
        cached = (out / "feedforward.zarr").exists() or self._source.has_processed(session, stem)
        # Cached and not forced: just load existing outputs without recomputing
        if cached and not force:
            self._load_outputs(session, stem)
            return
        config = self._current_config()
        doc = pn.state.curdoc

        def job():
            # Lazy import: pulls the heavy reconstruction/mesh stack only when a run starts.
            from collab_splats.dashboard.pipeline import run_pipeline

            video = self._ensure_local_video(session, name)
            run_pipeline(
                video_path=video,
                session=session,
                stem=stem,
                config=config,
                op_log=self._op_log,
                source=self._source,
                base_dir=self._base_dir,
            )
            return True

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._load_outputs(session, stem)  # re-enqueues a load job

        self._set_busy(True)
        self._op_log.start_op(f"running {stem}")
        self._gpu.submit(job, on_done, doc)

    def _dispatch_load(self, session: str, stem: str) -> None:
        """Schedule an outputs load (called from a worker job's completion)."""
        self._load_outputs(session, stem)

    def _load_outputs(self, session: str, stem: str) -> None:
        """Enqueue loading FeedforwardResult + semantics; render on the IOLoop when done."""
        out = self._base_dir / session / stem
        doc = pn.state.curdoc  # captured on the IOLoop at call time
        max_points = self.max_display_points.value

        def job():
            # Lazy import: FeedforwardResult lives in the heavy feedforward package.
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult

            if not (out / "feedforward.zarr").exists():
                self._source.pull_processed(session, stem, out)
            result = FeedforwardResult.load_zarr(out / "feedforward.zarr")
            semantics_dir = out / "semantics"
            # The pipeline lifts + compresses features eagerly during a run and caches the
            # L2-normalised per-point features here, so queries are instant. If absent (older
            # run), the viewer falls back to lazy lifting on first query.
            lifted_path = semantics_dir / "lifted_normed.npy"
            lifted_normed = np.load(lifted_path) if lifted_path.exists() else None
            # TSDF writes mesh_tsdf.ply (see mesh/tsdf.py), not mesh.ply.
            mesh_path = out / "mesh" / "mesh_tsdf.ply"
            return (
                result,
                mesh_path if mesh_path.exists() else None,
                semantics_dir if semantics_dir.exists() else None,
                lifted_normed,
            )

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            result, mesh_path, semantics_dir, lifted_normed = res
            self._viewer.load(
                result,
                mesh_path=mesh_path,
                semantics_dir=semantics_dir,
                lifted_normed=lifted_normed,
                max_points=max_points,
            )
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"loading {stem}")
        self._gpu.submit(job, on_done, doc)

    def _on_view_mode(self, event) -> None:
        """Switch pointcloud/mesh; keep the right pane's similarity map across the switch.

        set_mode renders the new mode (right pane uses this mode's cached similarity if present).
        If a query is active but this mode hasn't been scored yet, re-score it on the worker —
        point and mesh use different feature spaces, so colours can't be reused across modes.
        """
        self._viewer.set_mode(event.new)
        query = self._viewer.active_query()
        if not query or self._viewer.cached_query_colors(event.new) is not None:
            return
        positive, negative, extractor_name = query
        doc = pn.state.curdoc

        def job():
            return self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
            )

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            if res is None:
                self._op_log.finish_op()
                return
            self._viewer.render_query(res)

        self._set_busy(True)
        self._op_log.start_op(f"query: {event.new} similarity")
        self._gpu.submit(job, on_done, doc)

    def _on_query(self, event) -> None:
        """Score the positive/negative query off the IOLoop; recolour the right pane on done."""
        positive = _split_terms(self.pos_query.value)
        negative = _split_terms(self.neg_query.value)
        extractor_name = self.extractor.value
        doc = pn.state.curdoc

        def job():
            return self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
            )

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            if res is None:  # no scene loaded -> nothing to recolour
                self._op_log.finish_op()
                return
            self._viewer.render_query(res)

        self._set_busy(True)
        self._op_log.start_op("query")
        self._gpu.submit(job, on_done, doc)

    # ---- layout --------------------------------------------------------

    def sidebar(self) -> pn.Column:
        """Sidebar contents — composed by view() or by the tabbed shell."""
        return self._sidebar

    def main(self) -> pn.Column:
        """Main-area contents: split viewer + live progress strip."""
        # Live operations strip: stage label + progress bar + scrolling per-step log.
        # Poll the shared op_log on THIS session's IOLoop (op_log is mutated from the GpuWorker
        # thread; pushing Bokeh updates cross-thread glitches). Polling reads a locked snapshot and
        # updates the pane on the IOLoop → flicker-free, and a refreshed page re-attaches live.
        progress = pn.pane.HTML(self._op_log.render_html(), sizing_mode="stretch_width")

        def _tick() -> None:
            progress.object = self._op_log.render_html()

        try:
            pn.state.add_periodic_callback(_tick, period=300, start=True)
        except Exception:
            # No live server (tests) — leave the static snapshot.
            logger.debug("no periodic callback (no server doc); progress is static", exc_info=True)

        return pn.Column(self._viewer.layout, progress, sizing_mode="stretch_both")

    def view(self) -> pn.template.MaterialTemplate:
        """Standalone single-page layout (kept for tests and direct serving).

        The 'vtk' extension is loaded once in run_app (main thread, before serving) —
        loading it here per-session fails to inject the VTK JS and the panes hang.
        """
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self.sidebar()],
            main=[self.main()],
            header_background="#2596be",
            sidebar_width=340,
        )


# Spec 2026-07-14 names this class SplatsPage; alias until callers migrate.
SplatsPage = SplatsApp


########
# Entry points
########


def _ensure_display() -> None:
    """Start a headless Xvfb display if none is set, so VTK gets an OpenGL context.

    pn.pane.VTK builds a vtkXOpenGLRenderWindow when the document is created; with no
    DISPLAY this blocks in a C-level GL call (page spins, Ctrl-C is swallowed). On a
    headless host we spin up Xvfb and point DISPLAY at it before any VTK init.
    """
    if os.environ.get("DISPLAY"):
        return
    if not shutil.which("Xvfb"):
        logger.warning("no DISPLAY and Xvfb not installed; VTK rendering will fail on a headless host")
        return
    # Software GL via Mesa — containers rarely expose GLX on the GPU.
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    display = ":99"
    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    os.environ["DISPLAY"] = display
    time.sleep(1.0)  # let Xvfb come up before VTK probes the display
    logger.info("started Xvfb on %s for headless VTK rendering", display)


def _warm_heavy_stack() -> None:
    """Import the heavy reconstruction/semantics stack once at startup (background thread).

    Pre-pays the ~17s import + torch.compile so the first Run/load/query doesn't. Runs off
    the IOLoop (the server already binds and the page renders before this finishes).
    """
    try:
        # Lazy heavy-dep imports (intentional warm) — module-process-wide once loaded.
        import collab_splats.pointcloud.feedforward.base  # noqa: F401
        import collab_splats.semantics.features.base  # noqa: F401

        logger.info("heavy stack warmed")
    except Exception as exc:
        logger.warning("heavy-stack warm failed: %s", exc)


def run_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    base_dir: str = "/workspace/outputs",
    websocket_origin: str | list[str] | None = None,
) -> None:
    """Serve the splats dashboard.

    websocket_origin=None restricts connections to host:port + localhost:port. Pass an
    explicit list (or "*") to allow remote-IP / SSH-tunnel access.
    """
    # Headless host: ensure an OpenGL context exists before any VTK initialisation.
    _ensure_display()

    # Load the VTK extension ONCE here, in the main thread, before serving. Panel requires
    # pn.extension() at startup; deferring it into the per-session factory hangs the panes.
    # inline=True serves all JS/CSS (incl. the large vtk.js bundle) from this server instead
    # of cdn.holoviz.org — a headless/air-gapped host can't reach the CDN, so the page would
    # otherwise spin forever waiting on resources that never load.
    pn.extension("vtk", inline=True)

    # One shared GPU worker for every session: serializes all CUDA work across tabs,
    # preventing parallel model loads from OOMing the GPU.
    gpu_worker = GpuWorker()

    # One shared operation log too, so a page refresh re-attaches to the live run's progress.
    op_log = OperationLog()

    # Warm the heavy import in the background so the first interaction isn't a cold start.
    threading.Thread(target=_warm_heavy_stack, name="warm", daemon=True).start()

    if websocket_origin is None:
        origin: str | list[str] = [f"{host}:{port}", f"localhost:{port}"]
    else:
        origin = websocket_origin

    def factory() -> pn.template.MaterialTemplate:
        # Tabbed shell: splats + localize pages share one session, worker, and op_log.
        # Import stays local to avoid an app.py <-> shell.py circular import at module load.
        from collab_splats.dashboard.shell import DashboardShell

        return DashboardShell(base_dir=Path(base_dir), gpu_worker=gpu_worker, op_log=op_log).view()

    pn.serve(
        factory,
        address=host,
        port=port,
        show=False,
        title="splats",
        websocket_origin=origin,
        session_token_expiration=1800,
    )
