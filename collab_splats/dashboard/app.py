# collab_splats/dashboard/app.py
"""Dashboard entry points: single-page SplatsApp."""

from __future__ import annotations

import atexit
import importlib
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import panel as pn
import param
import yaml

from collab_splats.dashboard.async_utils import run_off_loop
from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.localize import SceneCache
from collab_splats.dashboard.operation_log import OperationLog, busy_html
from collab_splats.dashboard.sources import PULL_EXCLUDES, SessionSource
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
_SAMPLERS = ["uniform", "optical_flow"]

# Per-model confidence default — mirrors each creator's own class default so the dashboard
# reproduces the notebook (which instantiates creators with no conf override). A single shared
# default (35) silently ran VGGT-Omega far below its native 50 cutoff, keeping low-confidence
# flyers that distort the cloud. Keep in sync with VGGTOmegaCreator.conf_threshold=50,
# VGGTXCreator.conf_threshold=35, MapAnythingCreator.confidence_percentile=35.
_MODEL_CONF_DEFAULTS = {"vggt_omega": 50.0, "vggtx": 35.0, "mapanything": 35.0}

# Keep at most this many scenes' "loaded" tuples resident. Each holds a full
# FeedforwardResult (can be GBs) and the container cgroup caps memory at ~46.6 GB.
_LOADED_CACHE_KEEP = 3


def _bind_visibility(widget, selector, predicate) -> None:
    """Show widget only when predicate(selector.value) holds; re-evaluate on change."""
    widget.visible = predicate(selector.value)
    selector.param.watch(lambda e: setattr(widget, "visible", predicate(e.new)), "value")


def _split_terms(text: str) -> list[str]:
    """Split a comma-separated query box into a list of non-empty phrases."""
    return [t.strip() for t in text.split(",") if t.strip()]


def _video_options(videos: list, processed: "set[str]") -> dict:
    """Dropdown label -> value map: union of curated videos and processed scenes.

    Leads with a blank entry (nothing loads until the user picks). Curated videos with
    processed outputs get a ✓; processed scenes whose source video is missing from
    fieldwork_curated still appear (value = stem — the load path only needs the stem).
    """
    options = {"— select a video —": ""}
    options.update({(f"{v} ✓" if Path(v).stem in processed else v): v for v in videos})
    curated_stems = {Path(v).stem for v in videos}
    options.update({f"{stem} ✓ (no source video)": stem for stem in sorted(processed - curated_stems)})
    return options


class SplatsApp(param.Parameterized):
    """Single-page dashboard wiring source, pipeline, and the split viewer."""

    def __init__(
        self,
        base_dir: Path = Path("/workspace/outputs"),
        source: SessionSource | None = None,
        gpu_worker: GpuWorker | None = None,
        op_log: OperationLog | None = None,
        cache: SceneCache | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
        self._gpu = gpu_worker if gpu_worker is not None else GpuWorker()
        # Shared across sessions (passed from run_app) so a page refresh re-attaches to an
        # in-flight run's progress instead of spawning a fresh, disconnected log.
        self._op_log = op_log if op_log is not None else OperationLog()
        # Session-level cache of expensive loads, shared with LocalizePage via the shell.
        self._cache = cache if cache is not None else SceneCache()
        self._current_scene: tuple[str, str] | None = None  # (session, stem) currently displayed
        self._loading_scene: tuple[str, str] | None = None  # (session, stem) load in flight
        self._loaded_order: deque = deque()  # "loaded" insertion order for keep-last-N eviction
        self._viewer = SplitViewer(op_log=self._op_log)
        # Persisted UI state survives browser reloads (each reload rebuilds widgets fresh).
        self._state_path = self._base_dir / ".dashboard_state.yaml"
        self._state = self._load_state()
        self._state_dirty = False  # set by _persist_state, cleared by _flush_state
        self._restored_selection = False  # session/video restored once, after options load
        self._suppress_autoload = False  # gate _on_video during programmatic option/restore churn
        self._cache_check_token = 0  # latest-wins token for Run's remote-cache check
        self._cache_check_active = False  # keeps widgets locked through the check window
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
        """Mark UI state dirty; the debounce flush writes it (coalesces rapid changes)."""
        self._state_dirty = True

    def _flush_state(self) -> None:
        """Write current widget values to disk if dirty (debounce timer / run start)."""
        if not self._state_dirty:
            return
        self._state_dirty = False
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
        # Migrate the pre-rename "balanced" label from saved settings to "uniform".
        _sampling = s.get("sampling", "uniform")
        _sampling = "uniform" if _sampling == "balanced" else _sampling
        self.sampling = pn.widgets.Select(name="Frame sampling", options=_SAMPLERS, value=_sampling)
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

        # Density change must bust the reselect short-circuit so a reselect re-renders at the
        # new density; the cache stays valid since decimation happens inside viewer.load.
        def _on_density(event) -> None:
            """Bust the reselect short-circuit and tell the user how to apply the new density."""
            self._current_scene = None
            self._op_log.append_line(f"display density {event.new:,} — reselect the scene to apply")

        self.max_display_points.param.watch(_on_density, "value")
        self.normalize_view = pn.widgets.Checkbox(
            name="Normalize view (orient + scale)", value=s.get("normalize_view", True)
        )
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(label="Force re-run", button_type="warning")
        # Cross-tab busy indicator: filled while any GpuWorker job is in flight.
        self.busy_note = pn.pane.HTML("", sizing_mode="stretch_width")

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
            self.busy_note,
        )

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable every mutating widget while a GPU job is in flight (IOLoop thread).

        Covers view widgets too: toggling view_mode mid-load would fire set_mode on a
        half-loaded viewer and queue a second job.
        """
        widgets = (
            self.run_btn,
            self.force_btn,
            self.run_query_btn,
            self.view_mode,
            self.normalize_view,
            self.session_select,
            self.video_select,
        )
        for w in widgets:
            w.disabled = busy
        self.busy_note.object = busy_html(self._op_log.current_op) if busy else ""

    def _sync_busy(self) -> None:
        """Poll hook: mirror the shared worker's busy flag onto this page's widgets."""
        # A pending Run cache-check also counts as busy, or the 300ms poll would
        # re-enable Run mid-check and a second click could queue a duplicate pipeline.
        busy = bool(self._gpu.busy) or self._cache_check_active
        if busy != self.run_btn.disabled:
            self._set_busy(busy)
        elif busy:
            # Refresh the label while busy (current_op advances through the run).
            self.busy_note.object = busy_html(self._op_log.current_op)

    # ---- data wiring ---------------------------------------------------

    def _refresh_sessions(self) -> None:
        """List sessions on a background thread; set options back on the IOLoop.

        rclone listing is a blocking network call; running it inline would stall the
        IOLoop during document init (the same class of freeze as the heavy imports).
        """
        doc = pn.state.curdoc

        def work():
            # step()'s FAILED line is the user-visible surface (no error_op: that would
            # clobber a concurrent run's is_running); fall through with an empty list
            # so the dropdown doesn't wedge.
            try:
                with self._op_log.step("listing sessions"):
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
            # Blank-first: plain list options auto-flip value→options[0], firing a videos
            # listing for a session nobody picked — its late apply then overwrote the
            # restored session's video list (session said 2024_02_06, videos showed
            # 2023_11_05's). With the blank entry nothing fires until an explicit pick.
            self._suppress_autoload = True
            try:
                options = {"— select a session —": ""}
                options.update({n: n for n in names})
                self.session_select.options = options
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
        """Re-apply the persisted session once options are available (session only).

        Setting the value fires _on_session, which populates the video options off the
        IOLoop — no listing here: a synchronous rclone call would block the IOLoop and
        race the watcher's own fetch (two writers left the dropdown empty on reload).
        """
        sess = self._state.get("session_select")
        if not sess or sess not in names:
            return
        self.session_select.value = sess

    def _on_session(self, event) -> None:
        """Populate video dropdown when session changes (rclone list runs off the IOLoop)."""
        if not event.new:
            return
        session = event.new

        # Blocking rclone listings off the IOLoop; set options back on the loop.
        # step() logs start/done and its FAILED line surfaces listing errors. The two
        # listings fail independently: curated-only or processed-only beats an empty
        # dropdown, and a total failure shows a retry hint instead of silence.
        def fetch():
            with self._op_log.step(f"listing videos ({session})"):
                try:
                    videos = self._source.list_videos(session)
                except Exception as exc:
                    logger.warning("curated video listing failed: %s", exc)
                    self._op_log.append_line(f"curated listing FAILED: {exc}")
                    videos = []
                try:
                    processed = set(self._source.list_processed_stems(session))
                except Exception as exc:
                    logger.warning("processed listing failed: %s", exc)
                    self._op_log.append_line(f"processed listing FAILED: {exc}")
                    processed = set()
            if not videos and not processed:
                return {"— listing failed; reselect the session to retry —": ""}
            return _video_options(videos, processed)

        def apply(options: dict) -> None:
            # Latest-wins: a slow listing for a superseded session must not clobber the
            # current session's video list (the session/video mismatch bug).
            if self.session_select.value != session:
                return
            self.video_select.options = options
            # Explicit-select UX: a session switch never auto-loads. The blank entry is
            # selected until the user picks a video, which fires _on_video -> load.
            self.video_select.value = ""

        self._video_list_thread = run_off_loop(
            fetch,
            apply,
            label="video-list",
            doc=pn.state.curdoc,
        )

    def _on_video(self, event) -> None:
        """Auto-load cached outputs when a video is selected (skipped during programmatic churn)."""
        if self._suppress_autoload or not event.new:
            return
        # _autoload_current probes the frame bound itself — calling it here too started a
        # duplicate remote download while the first fetch was still mid-flight.
        self._autoload_current()

    def _update_max_frames_bound(self, session: str, name: str) -> None:
        """Set the Max-frames bound to the video's frame count — fetch/probe off the IOLoop.

        ffprobe is a subprocess even for local files; never run it inline in a watcher.
        """
        if not session or not name:
            return

        def work() -> int:
            video = self._ensure_local_video(session, name)  # no-op when already local
            from collab_splats.preproc import get_video_info

            return int(get_video_info(str(video)).get("total_frames") or 0)

        def apply(total: int) -> None:
            # Latest-wins: a slow probe for a superseded video must not clobber the bound.
            if self.video_select.value != name or total <= 0:
                return
            self.max_frames.end = total
            self.max_frames.name = f"Max frames (video has {total})"

        self._video_meta_thread = run_off_loop(
            work,
            apply,
            label="video-meta",
            doc=pn.state.curdoc,
            on_error=lambda exc: self._op_log.append_line(f"frame count unavailable: {exc}"),
        )

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
        if (out / "feedforward.zarr").exists():
            self._load_outputs(session, stem)
            return
        # Remote check is a blocking rclone list -> run off the IOLoop, then load if present.
        run_off_loop(
            lambda: self._source.has_processed(session, stem),
            lambda ok: self._load_outputs(session, stem) if ok else None,
            label="has-processed",
            doc=pn.state.curdoc,
            on_error=lambda exc: self._op_log.append_line(f"server check failed: {exc}"),
        )

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
        """Return local video path, fetching from source if needed (progress -> op_log)."""
        local = self._base_dir / session / Path(name).stem / name
        if local.exists():
            return local
        on_line = self._op_log.rclone_progress("⬇ fetching video")
        return self._source.fetch_video(session, name, local.parent, on_line=on_line)

    def _on_run(self, event, force: bool) -> None:
        """Run or reload the pipeline, respecting cache and force flag."""
        # Flush pending UI state so the config driving this run is durable on disk.
        self._flush_state()
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            self._op_log.error_op("select a session and a video first")
            return
        stem = Path(name).stem
        out = self._base_dir / session / stem
        if force:
            # Force re-run: drop cached loads so the post-run load re-reads fresh outputs.
            self._invalidate_scene(session, stem)
            self._start_run(session, name, stem)
            return
        if (out / "feedforward.zarr").exists():
            self._load_outputs(session, stem)
            return
        # Remote-cache check is a blocking rclone list — off the IOLoop (a cold check
        # inside the click handler froze the whole page), then load or run on the result.
        # Lock the widgets for the check window (a second click would queue a duplicate
        # pipeline run) and keep only the latest check's verdict (stale applies bail out).
        self._cache_check_token += 1
        token = self._cache_check_token
        self._cache_check_active = True
        self._set_busy(True)
        self._op_log.append_line(f"checking server for {stem}…")

        def apply(ok: bool) -> None:
            if token != self._cache_check_token:
                return  # superseded by a newer click
            self._cache_check_active = False
            self._sync_busy()  # re-enable unless the shared worker is mid-job
            if ok:
                self._load_outputs(session, stem)
            else:
                self._start_run(session, name, stem)

        def on_error(exc: Exception) -> None:
            if token != self._cache_check_token:
                return
            self._cache_check_active = False
            self._sync_busy()
            self._op_log.error_op(f"server check failed: {exc}")

        self._cache_check_thread = run_off_loop(
            lambda: self._source.has_processed(session, stem),
            apply,
            label="has-processed",
            doc=pn.state.curdoc,
            on_error=on_error,
        )

    def _start_run(self, session: str, name: str, stem: str) -> None:
        """Enqueue the full pipeline for a video (IOLoop thread)."""
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
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            self._invalidate_scene(session, stem)  # fresh outputs -> stale cache/display
            self._load_outputs(session, stem)  # re-enqueues a load job

        self._set_busy(True)
        self._op_log.start_op(f"running {stem}")
        self._gpu.submit(job, on_done, doc)

    def _dispatch_load(self, session: str, stem: str) -> None:
        """Schedule an outputs load (called from a worker job's completion)."""
        self._load_outputs(session, stem)

    def _invalidate_scene(self, session: str, stem: str) -> None:
        """Drop cached loads for a scene (Force re-run / fresh pipeline output)."""
        # Drop every kind, not just "loaded" — LocalizePage caches its "mesh" (and
        # localizer) entries under the same shared cache and must not serve stale ones.
        self._cache.drop_scene((session, stem))
        if self._current_scene == (session, stem):
            self._current_scene = None  # ensure the next load is not short-circuited
        self._source.invalidate(("has_processed", session, stem))

    def _remember_loaded(self, key: tuple[str, str]) -> None:
        """Track "loaded" insertion order; evict beyond the last N scenes (memory cap)."""
        # Re-loads move the key to the back instead of duplicating it in the deque.
        if key in self._loaded_order:
            self._loaded_order.remove(key)
        self._loaded_order.append(key)
        while len(self._loaded_order) > _LOADED_CACHE_KEEP:
            self._cache.drop(self._loaded_order.popleft(), "loaded")

    def _load_outputs(self, session: str, stem: str) -> None:
        """Enqueue loading FeedforwardResult + semantics; render on the IOLoop when done."""
        # Already displayed and idle -> nothing to do (reselect of the same scene). The
        # is_running guard keeps a mid-run reselect loading: _current_scene may point at
        # soon-to-be-stale output while a run/load is in flight, so don't trust it then.
        if self._current_scene == (session, stem) and not self._op_log.is_running:
            return
        # In-flight dedupe: the session-switch path and the video watcher can both request
        # the same load in one churn; queueing it twice doubles the pull + render.
        if self._loading_scene == (session, stem):
            return
        self._loading_scene = (session, stem)
        out = self._base_dir / session / stem
        doc = pn.state.curdoc  # captured on the IOLoop at call time
        max_points = self.max_display_points.value

        def job():
            # Lazy import: FeedforwardResult lives in the heavy feedforward package.
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult

            # Session cache: skip the pull + zarr/npy reads when this scene was already loaded.
            cached = self._cache.get((session, stem), "loaded")
            if cached is not None:
                self._op_log.append_line(f"{stem}: using in-memory cache")
                return cached
            if not (out / "feedforward.zarr").exists():
                with self._op_log.step(f"{stem}: pulling from server"):
                    self._source.pull_processed(
                        session,
                        stem,
                        out,
                        excludes=PULL_EXCLUDES,
                        on_line=self._op_log.rclone_progress("⬇ pulling from server"),
                    )
            # Display needs only points/colors/extrinsics; skip decoding dense arrays
            # (GBs when present locally). The lift path reloads them on demand.
            with self._op_log.step(f"{stem}: reading feedforward.zarr"):
                result = FeedforwardResult.load_zarr(
                    out / "feedforward.zarr",
                    load_depth=False,
                    load_world_points=False,
                    load_confidence=False,
                    load_features=False,
                    load_pixel_indices=False,
                )
            semantics_dir = out / "semantics"
            # TSDF writes mesh_tsdf.ply (see mesh/tsdf.py), not mesh.ply.
            mesh_path = out / "mesh" / "mesh_tsdf.ply"
            # lifted_normed=None defers the (P, D) feature read to the first query: the viewer's
            # ensure_lifted loads the cached lifted_normed.npy from semantics_dir (or lifts from
            # the feature zarr for older runs). Tuple keeps 4 slots so cache/on_done unpack as-is.
            value = (
                result,
                mesh_path if mesh_path.exists() else None,
                semantics_dir if semantics_dir.exists() else None,
                None,
            )
            self._cache.put((session, stem), "loaded", value)
            self._remember_loaded((session, stem))
            return value

        def on_done(res):
            self._loading_scene = None
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return
            result, mesh_path, semantics_dir, lifted_normed = res
            # Timed: the render phase (geometry build + software-GL + full-scene websocket
            # serialize for TWO panes) is the slow tail of a load — make it visible.
            try:
                n_pts = len(result.points)
                n_shown = min(n_pts, max_points) if max_points > 0 else n_pts
                label = f"rendering {n_shown:,} points × 2 panes"
            except Exception:  # test doubles without real arrays
                label = "rendering scene"
            with self._op_log.step(label):
                self._viewer.load(
                    result,
                    mesh_path=mesh_path,
                    semantics_dir=semantics_dir,
                    lifted_normed=lifted_normed,
                    max_points=max_points,
                )
            self._current_scene = (session, stem)  # reselects of this scene now short-circuit
            self._op_log.finish_op()

        self._set_busy(True)
        # Session in the label: stems repeat across sessions ("loading C0043" is ambiguous
        # right after a session switch).
        self._op_log.start_op(f"loading {session}/{stem}")
        self._gpu.submit(job, on_done, doc)

    def _on_view_mode(self, event) -> None:
        """Switch pointcloud/mesh; keep the right pane's similarity map across the switch.

        The mesh PolyData is loaded lazily (viewer.load defers it), so entering mesh mode
        first materialises it on the WORKER (shared-cache hit skips the disk read), then
        set_mode renders in on_done — VTK mutation stays on the IOLoop, the blocking read
        does not. If a query is active but this mode hasn't been scored yet, the same job
        re-scores it — point and mesh use different feature spaces, so colours can't be
        reused across modes.
        """
        mode = event.new
        scene_key = self._current_scene
        query = self._viewer.active_query()
        need_score = bool(query) and self._viewer.cached_query_colors(mode) is None
        positive, negative, extractor_name = query if query else ([], [], "")
        doc = pn.state.curdoc

        def job():
            # Mesh mode: materialise the polydata off the IOLoop; share it with LocalizePage
            # via the SceneCache so neither page re-reads the .ply the other already loaded.
            if mode == "mesh":
                preloaded = self._cache.get(scene_key, "mesh") if scene_key else None
                loaded = self._viewer.ensure_mesh_polydata(preloaded=preloaded, op_log=self._op_log)
                if loaded and scene_key:
                    self._cache.put(scene_key, "mesh", self._viewer.mesh_polydata())
            if need_score:
                fetched = self._ensure_lift_inputs(scene_key)
                # Explicit target mode: set_mode runs later in on_done, so self._viewer.mode
                # is still the OUTGOING mode here — scoring on it produced wrong-length
                # colours (mesh-vertex vs point) and an IndexError in render_query.
                result = self._viewer.score_query(
                    positive=positive,
                    negative=negative,
                    extractor_name=extractor_name,
                    op_log=self._op_log,
                    mode=mode,
                )
                if fetched:
                    # Lift is cached now (viewer saved the npy) -> drop the fetched GBs.
                    self._cleanup_lift_inputs(scene_key)
                return result
            return None

        def on_done(res):
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                # Mode switch failed (e.g. mesh read error): snap the radio back to what is
                # actually displayed, without re-firing this watcher.
                with param.parameterized.discard_events(self.view_mode):
                    self.view_mode.value = self._viewer.mode
                return
            # Mesh (if any) is resident now — render the new mode on the IOLoop.
            self._viewer.set_mode(mode)
            if res is not None:
                self._viewer.render_query(res)
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"switching to {mode}")
        self._gpu.submit(job, on_done, doc)

    _LIFT_MEMBERS = ("pixel_indices", "depth", "confidence", "conf")  # 'conf' = legacy key

    def _ensure_lift_inputs(self, scene_key) -> bool:
        """Fetch the dense zarr members a first-query feature lift needs (worker thread).

        New runs cache semantics/lifted_normed.npy so the lift never runs; legacy scenes
        lift from pixel_indices/depth/confidence, which the display pull excludes
        (PULL_EXCLUDES) — fetch them on demand or the lift fails. Returns True when a
        fetch happened, so the caller can clean the members up once the lift is cached.
        """
        if scene_key is None:
            return False
        session, stem = scene_key
        out = self._base_dir / session / stem
        if (out / "semantics" / "lifted_normed.npy").exists():
            return False  # cached per-point features -> no lift, no dense arrays needed
        zarr_dir = out / "feedforward.zarr"
        if not zarr_dir.exists():
            return False  # nothing local yet; the load path owns the initial pull
        core_missing = any(not (zarr_dir / m).exists() for m in ("pixel_indices", "depth"))
        conf_missing = not (zarr_dir / "confidence").exists() and not (zarr_dir / "conf").exists()
        if not core_missing and not conf_missing:
            return False
        with self._op_log.step(f"{stem}: fetching dense arrays for feature lift (legacy scene)"):
            self._source.pull_zarr_members(
                session,
                stem,
                out,
                self._LIFT_MEMBERS,
                on_line=self._op_log.rclone_progress("⬇ fetching dense arrays"),
            )
        return True

    def _cleanup_lift_inputs(self, scene_key) -> None:
        """Delete on-demand-fetched dense members once the lift is cached (worker thread).

        Runs ONLY when _ensure_lift_inputs fetched this call — fresh local runs keep
        their dense arrays (the pipeline wrote them; push_outputs uploads them). The
        npy guard keeps the members when the lift failed, so a retry can still run.
        """
        if scene_key is None:
            return
        session, stem = scene_key
        out = self._base_dir / session / stem
        if not (out / "semantics" / "lifted_normed.npy").exists():
            return  # lift didn't complete -> keep the inputs for a retry
        freed = 0
        for member in self._LIFT_MEMBERS:
            member_dir = out / "feedforward.zarr" / member
            if member_dir.exists():
                freed += sum(f.stat().st_size for f in member_dir.rglob("*") if f.is_file())
                shutil.rmtree(member_dir, ignore_errors=True)
        if freed:
            self._op_log.append_line(f"{stem}: removed fetched dense arrays ({freed / 1e9:.1f} GB freed)")

    def _on_query(self, event) -> None:
        """Score the positive/negative query off the IOLoop; recolour the right pane on done."""
        positive = _split_terms(self.pos_query.value)
        negative = _split_terms(self.neg_query.value)
        extractor_name = self.extractor.value
        scene_key = self._current_scene
        doc = pn.state.curdoc

        def job():
            fetched = self._ensure_lift_inputs(scene_key)
            result = self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name, op_log=self._op_log
            )
            if fetched:
                # Lift is cached now (viewer saved the npy) -> the fetched GBs are dead weight.
                self._cleanup_lift_inputs(scene_key)
            return result

        def on_done(res):
            # Sync, not unconditional re-enable: another queued job must keep widgets locked.
            self._sync_busy()
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
        """Main-area contents: split viewer (the shared op-log console lives in the shell)."""
        # Busy-state poll only: the operations console is rendered ONCE by DashboardShell,
        # outside the tabs, so it stays visible on both tabs. This tick just mirrors the
        # worker's busy flag onto this page's widgets.
        try:
            pn.state.add_periodic_callback(self._sync_busy, period=300, start=True)
        except Exception:
            # No live server (tests) — busy state syncs on job boundaries only.
            logger.debug("no periodic callback (no server doc)", exc_info=True)

        # Slow debounce flush: widget changes only mark state dirty; this writes it to disk.
        # A final flush on session teardown closes the ≤1s loss window on tab close.
        try:
            pn.state.add_periodic_callback(self._flush_state, period=1000, start=True)
            pn.state.on_session_destroyed(lambda _ctx: self._flush_state())
        except Exception:
            logger.debug("no periodic callback (no server doc); state flushes on run only", exc_info=True)

        return pn.Column(self._viewer.layout, sizing_mode="stretch_both")

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
    """Import the heavy reconstruction/semantics/localization stack once at startup (bg thread).

    Pre-pays the ~17s import + torch.compile so the first Run/load/query/Localize isn't a
    cold start. Runs off the IOLoop (the server binds and the page renders before this finishes).
    """
    modules = (
        "collab_splats.dashboard.pipeline",  # pulls feedforward + mesh/TSDF (~17s)
        "collab_splats.semantics.features.base",
        "collab_splats.localization.localizer",  # localize tab's first run (~10s)
    )
    ok = 0
    for name in modules:
        # Each import isolated: one missing optional dep must not skip the rest.
        try:
            importlib.import_module(name)
            ok += 1
        except Exception as exc:
            logger.warning("heavy-stack warm failed for %s: %s", name, exc)
    logger.info("heavy stack warmed (%d/%d)", ok, len(modules))


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
