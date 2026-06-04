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

import panel as pn
import param

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource
from collab_splats.dashboard.viewer import SplitViewer, load_lifted_normed

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
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
        self._gpu = gpu_worker if gpu_worker is not None else GpuWorker()
        self._op_log = OperationLog()
        self._viewer = SplitViewer()
        self._build_sidebar()
        self._refresh_sessions()

    # ---- sidebar -------------------------------------------------------

    def _build_sidebar(self) -> None:
        """Build all sidebar widgets and wire callbacks."""
        self.session_select = pn.widgets.Select(name="Session", options=[])
        self.video_select = pn.widgets.Select(name="Video", options=[])
        self.sampling = pn.widgets.Select(name="Frame sampling", options=_SAMPLERS, value="balanced")
        self.max_frames = pn.widgets.IntSlider(name="Max frames", start=10, end=200, value=50)
        self.env_model = pn.widgets.Select(name="Environment model", options=_ENV_MODELS, value="vggt_omega")
        self.conf = pn.widgets.FloatSlider(name="Confidence", start=0, end=100, value=50.0)
        self.extractor = pn.widgets.Select(name="Semantic model", options=_EXTRACTORS, value="talk2dino")
        self.pos_query = pn.widgets.TextInput(name="Positive query", placeholder="e.g. chair, stool")
        self.neg_query = pn.widgets.TextInput(name="Negative query", placeholder="e.g. floor, wall")
        self.run_query_btn = pn.widgets.Button(label="Run query", button_type="primary")
        self.min_disparity = pn.widgets.FloatInput(name="min_disparity", value=50.0)
        self.mesh_voxel = pn.widgets.FloatInput(name="voxel_size", value=0.005)
        self.mesh_sdf = pn.widgets.FloatInput(name="sdf_trunc", value=0.02)
        self.mesh_depth = pn.widgets.FloatInput(name="depth_trunc", value=1.0)
        self.mesh_clean = pn.widgets.Checkbox(name="clean_repair", value=False)
        self.max_display_points = pn.widgets.IntInput(name="Max display points", value=150_000, step=50_000)
        self.view_mode = pn.widgets.RadioButtonGroup(options=["pointcloud", "mesh"], value="pointcloud")
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(label="Force re-run", button_type="warning")

        self.session_select.param.watch(self._on_session, "value")
        self.video_select.param.watch(self._on_video, "value")
        self.run_query_btn.on_click(self._on_query)
        self.view_mode.param.watch(lambda e: self._viewer.set_mode(e.new), "value")
        self.run_btn.on_click(lambda e: self._on_run(e, force=False))
        self.force_btn.on_click(lambda e: self._on_run(e, force=True))

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
            self.session_select.options = names

        if doc is not None:
            doc.add_next_tick_callback(setter)
        else:
            setter()

    def _on_session(self, event) -> None:
        """Populate video dropdown when session changes."""
        if not event.new:
            return
        self.video_select.options = self._source.list_videos(event.new)

    def _on_video(self, event) -> None:
        """Auto-load cached outputs when a video is selected."""
        if not event.new:
            return
        session, stem = self.session_select.value, Path(event.new).stem
        out = self._base_dir / session / stem
        if (out / "feedforward.zarr").exists() or self._source.has_processed(session, stem):
            self._load_outputs(session, stem)

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
            try:
                lifted = load_lifted_normed(result, out / "semantics")
            except Exception:
                lifted = None
            mesh_path = out / "mesh" / "mesh.ply"
            return (result, mesh_path if mesh_path.exists() else None, lifted)

        def on_done(res):
            self._set_busy(False)
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            result, mesh_path, lifted = res
            self._viewer.load(result, mesh_path=mesh_path, lifted_normed=lifted, max_points=max_points)
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"loading {stem}")
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
            self._viewer.render_query(res)

        self._set_busy(True)
        self._op_log.start_op("query")
        self._gpu.submit(job, on_done, doc)

    # ---- layout --------------------------------------------------------

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble the full single-page layout.

        The 'vtk' extension is loaded once in run_app (main thread, before serving) —
        loading it here per-session fails to inject the VTK JS and the panes hang.
        """
        # Progress strip bound reactively to op_log params
        progress_bar = pn.widgets.Progress(
            value=pn.bind(lambda v: v, self._op_log.param.progress),
            max=100,
            sizing_mode="stretch_width",
            height=8,
        )
        status_str = pn.pane.Str(pn.bind(lambda s: s, self._op_log.param.current_op))
        progress = pn.Column(progress_bar, status_str)

        main = pn.Column(self._viewer.layout, progress, sizing_mode="stretch_both")
        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self._sidebar],
            main=[main],
            header_background="#2596be",
            sidebar_width=340,
        )


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


def run_app(
    host: str = "0.0.0.0",
    port: int = 7860,
    base_dir: str = "/workspace/outputs",
    websocket_origin: str | list[str] | None = "*",
) -> None:
    """Serve the splats dashboard.

    websocket_origin defaults to "*" so the app renders when reached via a remote host
    IP or SSH tunnel; bokeh otherwise refuses the websocket and the page hangs blank.
    """
    # Headless host: ensure an OpenGL context exists before any VTK initialisation.
    _ensure_display()

    # Load the VTK extension ONCE here, in the main thread, before serving. Panel requires
    # pn.extension() at startup; deferring it into the per-session factory hangs the panes.
    # inline=True serves all JS/CSS (incl. the large vtk.js bundle) from this server instead
    # of cdn.holoviz.org — a headless/air-gapped host can't reach the CDN, so the page would
    # otherwise spin forever waiting on resources that never load.
    pn.extension("vtk", inline=True)

    def factory() -> pn.template.MaterialTemplate:
        return SplatsApp(base_dir=Path(base_dir)).view()

    pn.serve(
        factory,
        address=host,
        port=port,
        show=False,
        title="splats",
        websocket_origin=websocket_origin,
    )
