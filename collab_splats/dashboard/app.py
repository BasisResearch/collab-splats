"""
Single-page dashboard: run a scene through Reconstructor per backend, then view and query it.

- every heavy job (run, load, mode switch, query) goes through the shared GpuWorker
- widget values persist to base_dir/.dashboard_state.yaml across browser reloads
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import panel as pn
import param
import yaml
from mergedeep import merge

from collab_splats.dashboard.gpu_worker import GpuWorker
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.viewer import SplitViewer
from collab_splats.reconstructor import STAGES, Reconstructor, backends
from collab_splats.remote import SceneSource

logger = logging.getLogger(__name__)

########
# Helpers
########

# Dense per-pixel zarr members the viewer never reads; unanchored, so they match under every <backend>/
PULL_EXCLUDES = (
    "pointcloud.zarr/depth/**",
    "pointcloud.zarr/confidence/**",
    "pointcloud.zarr/pixel_indices/**",
    "pointcloud.zarr/images/**",
)


def _busy_html(op: str) -> str:
    """
    Sidebar busy-note span shown while a GPU job is in flight.
    """
    return f"<span style='color:#e0a050;font-size:11px'>busy: {op or 'working'}…</span>"


def _split_terms(text: str) -> list[str]:
    """
    Split a comma-separated query box into a list of non-empty phrases.
    """
    return [t.strip() for t in text.split(",") if t.strip()]


def _scene_options(scenes: list[str], processed: set[str]) -> dict[str, str]:
    """
    Dropdown label -> value map: union of curated and processed scene ids.

    - leads with a blank entry, so nothing loads until the user picks
    - curated scenes with processed outputs get a ✓
    - processed scenes whose curated source is gone still appear; loading needs only the id
    """
    options = {"— select a scene —": ""}
    options.update({(f"{s} ✓" if s in processed else s): s for s in scenes})
    options.update(
        {f"{s} ✓ (no source video)": s for s in sorted(processed - set(scenes))}
    )
    return options


########
# Scene cache
########


class SceneCache:
    """
    Session cache of expensive loads, keyed (scene_key, kind).

    - kinds: "loaded" (result, mesh path, lifted store, extractor) and "mesh" (PolyData)
    - each kind keeps its most recent scenes; a loaded tuple can hold GBs of PointcloudResult
    """

    def __init__(self, keep: int = 3) -> None:
        """
        Start empty.

        Args:
            keep: scenes kept resident per kind.
        """
        self._keep = keep
        self._store: dict = {}
        self._order: dict[str, deque] = {}

    def get(self, scene_key: tuple[str, str], kind: str) -> object | None:
        """
        Cached value for one scene and kind.

        Args:
            scene_key: (scene, backend).
            kind: cache kind.

        Returns:
            The cached value, or None.
        """
        return self._store.get((scene_key, kind))

    def put(self, scene_key: tuple[str, str], kind: str, value: object) -> None:
        """
        Cache a value, evicting the kind's least recently put scene beyond `keep`.

        Args:
            scene_key: (scene, backend).
            kind: cache kind.
            value: what to cache.
        """
        self._store[(scene_key, kind)] = value
        order = self._order.setdefault(kind, deque())

        if scene_key in order:
            order.remove(scene_key)

        order.append(scene_key)

        while len(order) > self._keep:
            self._store.pop((order.popleft(), kind), None)

    def drop(self, scene_key: tuple[str, str]) -> None:
        """
        Remove every cached kind for one scene, since fresh outputs invalidate them all.

        Args:
            scene_key: (scene, backend).
        """
        for kind, order in self._order.items():
            self._store.pop((scene_key, kind), None)

            if scene_key in order:
                order.remove(scene_key)


########
# SplatsApp
########


class SplatsApp:
    """
    Single-page dashboard wiring the scene source, Reconstructor, and the split viewer.
    """

    ########
    # Setup and persisted state
    ########

    def __init__(
        self,
        gpu_worker: GpuWorker,
        base_dir: Path,
        source: SceneSource | None = None,
        op_log: OperationLog | None = None,
    ) -> None:
        """
        Build the sidebar and viewer, then list scenes in the background.

        Args:
            gpu_worker: the server's single worker thread every job runs on.
            base_dir: local outputs root, one directory per scene.
            source: remote scene listing / transfer; the default bucket when None.
            op_log: the server's shared log, so a page reload re-attaches to a running op.
        """
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SceneSource()
        self._gpu = gpu_worker
        self._op_log = op_log if op_log is not None else OperationLog()
        self._cache = SceneCache()
        self._current_key: tuple[str, str] | None = (
            None  # (scene, backend) currently displayed
        )
        self._loading_key: tuple[str, str] | None = (
            None  # (scene, backend) whose load is in flight
        )
        self._extractor_name = ""  # extractor of the displayed backend's semantics run
        self._viewer = SplitViewer(self._op_log)

        # Persisted UI state survives browser reloads (each reload rebuilds widgets fresh)
        self._state_path = self._base_dir / ".dashboard_state.yaml"
        self._state = self._load_state()
        self._state_dirty = False  # set by _persist_state, cleared by _flush_state
        self._restored_selection = False  # scene restored once, after options load
        self._suppress_autoload = (
            False  # gate _on_scene during programmatic option/restore churn
        )
        self._build_sidebar()
        self._refresh_scenes()

    def _load_state(self) -> dict:
        """
        Persisted widget values; empty dict if absent or unreadable.
        """
        try:
            if self._state_path.exists():
                return yaml.safe_load(self._state_path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            logger.warning("could not read dashboard state", exc_info=True)

        return {}

    def _persist_state(self, *_event: object) -> None:
        """
        Mark UI state dirty; the debounced flush writes it, coalescing rapid changes.
        """
        self._state_dirty = True

    def _flush_state(self) -> None:
        """
        Write current widget values to disk if dirty (poll tick, run start, tab close).
        """
        if not self._state_dirty:
            return

        self._state_dirty = False
        data = {k: w.value for k, w in self._persisted.items()}

        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(yaml.safe_dump(data, sort_keys=False))
        except (OSError, yaml.YAMLError):
            logger.warning("could not persist dashboard state", exc_info=True)

    ########
    # Sidebar
    ########

    def _build_sidebar(self) -> None:
        """
        Build all sidebar widgets and wire their callbacks.
        """
        # Seed each widget from persisted state (falls back to the literal default)
        s = self._state
        self.scene_select = pn.widgets.Select(name="Scene", options=[])
        registered = backends()
        self.backend = pn.widgets.Select(
            name="Backend",
            options=registered["feedforward"] + registered["sfm"],
            value=s.get("backend", "vggt_omega"),
        )
        self.stages = pn.widgets.MultiChoice(
            name="Stages",
            options=list(STAGES),
            value=[
                st for st in s.get("stages", ["preproc", "pointcloud"]) if st in STAGES
            ],
        )
        self.overrides = pn.widgets.TextAreaInput(
            name="Config overrides (YAML)",
            placeholder="semantics:\n  enabled: false\n# recorded into run_config.yaml; later runs keep them",
            description="Merged over the backend's run_config.yaml and recorded back into it, so they persist.",
            value=s.get("overrides", ""),
        )
        self.pos_query = pn.widgets.TextInput(
            name="Positive query",
            placeholder="e.g. chair, stool",
            value=s.get("pos_query", ""),
        )
        self.neg_query = pn.widgets.TextInput(
            name="Negative query",
            placeholder="e.g. floor, wall",
            value=s.get("neg_query", "background, sky"),
        )
        self.run_query_btn = pn.widgets.Button(label="Run query", button_type="primary")
        self.max_display_points = pn.widgets.IntInput(
            name="Max display points",
            value=s.get("max_display_points", 500_000),
            step=50_000,
        )
        self.view_mode = pn.widgets.RadioButtonGroup(
            options=["pointcloud", "mesh"], value="pointcloud"
        )

        # Density change reloads the displayed scene; the cache stays valid since decimation happens in viewer.load
        def _on_density(event: param.parameterized.Event) -> None:
            """
            Re-render the displayed scene at the new density.
            """
            key = self._current_key
            self._current_key = None

            if key is not None:
                self._load_outputs(*key)

        self.max_display_points.param.watch(_on_density, "value")
        self.normalize_view = pn.widgets.Checkbox(
            name="Normalize view (orient + scale)", value=s.get("normalize_view", True)
        )
        self.run_btn = pn.widgets.Button(label="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(label="Force re-run", button_type="warning")

        # Cross-session busy indicator: filled while any GpuWorker job is in flight
        self.busy_note = pn.pane.HTML("", sizing_mode="stretch_width")

        # Wire scene, query and view callbacks
        self.scene_select.param.watch(self._on_scene, "value")
        self.backend.param.watch(self._on_scene, "value")
        self.run_query_btn.on_click(self._on_query)
        self.view_mode.param.watch(self._on_view_mode, "value")
        self.normalize_view.param.watch(
            lambda e: self._viewer.set_normalize_view(e.new), "value"
        )

        # Push the restored normalize toggle into the viewer
        self._viewer.set_normalize_view(self.normalize_view.value)

        self.run_btn.on_click(lambda e: self._on_run(e, force=False))
        self.force_btn.on_click(lambda e: self._on_run(e, force=True))

        # Persist these widgets on change; scene is restored once its options load (_restore_selection)
        self._persisted = {
            "scene_select": self.scene_select,
            "backend": self.backend,
            "stages": self.stages,
            "overrides": self.overrides,
            "pos_query": self.pos_query,
            "neg_query": self.neg_query,
            "max_display_points": self.max_display_points,
            "normalize_view": self.normalize_view,
        }

        for _w in self._persisted.values():
            _w.param.watch(self._persist_state, "value")

        self._sidebar = pn.Column(
            "## Source",
            self.scene_select,
            self.backend,
            pn.Card(self.stages, self.overrides, title="Run", collapsed=False),
            pn.Card(
                self.pos_query,
                self.neg_query,
                self.run_query_btn,
                title="Semantics",
                collapsed=False,
            ),
            self.max_display_points,
            "### View",
            self.view_mode,
            self.normalize_view,
            pn.Row(self.run_btn, self.force_btn),
            self.busy_note,
        )

    def _set_busy(self, busy: bool) -> None:
        """
        Enable or disable every mutating widget while a GPU job is in flight (IOLoop thread).

        - covers view widgets too: a mid-load view_mode toggle would hit a half-loaded viewer
        """
        widgets = (
            self.run_btn,
            self.force_btn,
            self.run_query_btn,
            self.view_mode,
            self.normalize_view,
            self.scene_select,
            self.backend,
            self.stages,
            self.overrides,
        )

        for w in widgets:
            w.disabled = busy

        self.busy_note.object = _busy_html(self._op_log.current_op) if busy else ""

    def _sync_busy(self) -> None:
        """
        Poll hook: mirror the shared worker's busy flag onto this page's widgets.
        """
        busy = bool(self._gpu.busy)

        if busy != self.run_btn.disabled:
            self._set_busy(busy)
        elif busy:
            # Refresh the label while busy (current_op advances through the run)
            self.busy_note.object = _busy_html(self._op_log.current_op)

    ########
    # Scene listing
    ########

    def _refresh_scenes(self) -> None:
        """
        List curated and processed scenes on a background thread; set options on the IOLoop.

        - rclone listing is a blocking network call; inline it would stall the IOLoop at document init
        """
        doc = pn.state.curdoc

        def work() -> None:
            # Listings fail independently and never error_op, which would clobber a concurrent run
            try:
                with self._op_log.step("listing scenes"):
                    scenes = self._source.list_scenes()
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("curated scene listing failed: %s", exc)
                self._op_log.append_line(f"curated listing FAILED: {exc}")
                scenes = []

            try:
                with self._op_log.step("listing processed scenes"):
                    processed = set(self._source.list_processed_scenes())
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("processed scene listing failed: %s", exc)
                self._op_log.append_line(f"processed listing FAILED: {exc}")
                processed = set()

            self._apply_scenes(scenes, processed, doc)

        self._scene_thread = threading.Thread(
            target=work, name="scene-list", daemon=True
        )
        self._scene_thread.start()

    def _apply_scenes(self, scenes: list[str], processed: set[str], doc: Any) -> None:
        """
        Set the scene dropdown options on the IOLoop (or inline if no doc).
        """

        def setter() -> None:
            # Blank-first options: a plain list would auto-select and load options[0]
            restored = False
            self._suppress_autoload = True

            try:
                if not scenes and not processed:
                    self.scene_select.options = {
                        "— listing failed; reload the page to retry —": ""
                    }
                else:
                    self.scene_select.options = _scene_options(scenes, processed)

                if not self._restored_selection:
                    self._restored_selection = True
                    restored = self._restore_selection(scenes, processed)
            finally:
                self._suppress_autoload = False

            # A restored selection never auto-loads; the user reselects to load it
            if not restored:
                self._autoload_current()

        if doc is not None:
            doc.add_next_tick_callback(setter)
        else:
            setter()

    def _restore_selection(self, scenes: list[str], processed: set[str]) -> bool:
        """
        Re-apply the persisted scene once options are available; True when restored.
        """
        scene = self._state.get("scene_select")

        if not scene or (scene not in scenes and scene not in processed):
            return False

        self.scene_select.value = scene
        return True

    def _on_scene(self, event: param.parameterized.Event) -> None:
        """
        Load (scene, backend) when either dropdown changes, skipped during programmatic churn.
        """
        if self._suppress_autoload or not event.new:
            return

        self._autoload_current()

    def _autoload_current(self) -> None:
        """
        Load the selected scene's selected backend; the load job pulls it when only the server has it.
        """
        scene = self.scene_select.value

        if not scene or self._op_log.is_running:
            return

        self._load_outputs(scene, self.backend.value)

    ########
    # Run
    ########

    def _reconstructor(
        self, scene: str, backend: str, overrides: dict | None = None
    ) -> Reconstructor:
        """
        Reconstructor for one scene's backend: recorded run config, then overrides, then local paths.
        """
        out = self._base_dir / scene
        recorded = Reconstructor.run_config_path(out, backend)
        config = (
            (yaml.safe_load(recorded.read_text()) or {}) if recorded.exists() else {}
        )
        method = next(m for m, names in backends().items() if backend in names)

        # input_path is only read by preproc; the run swaps in the video when preproc must run
        paths = {
            "input_path": str(out),
            "output_path": str(out),
            "pointcloud": {"method": method, "backend": backend},
        }
        return Reconstructor(merge({}, config, overrides or {}, paths))

    def _pulled_reconstructor(
        self,
        scene: str,
        backend: str,
        overrides: dict | None,
        excludes: tuple[str, ...],
    ) -> Reconstructor:
        """
        Reconstructor for the backend, pulling the scene first when its pointcloud is not done locally.

        - the pull skips `excludes`; the Reconstructor is rebuilt over the pulled run_config
        """
        rec = self._reconstructor(scene, backend, overrides)

        if (
            rec.done("pointcloud")
            or not self._source.check_available()
            or not self._source.has_processed(scene)
        ):
            return rec

        with self._op_log.step(f"{scene}: pulling from server"):
            on_line = self._op_log.rclone_progress("⬇ pulling from server")
            self._source.pull_processed(
                scene, self._base_dir / scene, excludes=excludes, on_line=on_line
            )

        return self._reconstructor(scene, backend, overrides)

    def _ensure_local_video(self, scene: str) -> Path:
        """
        Scene video at base_dir/<scene>/<filename>, fetched once if absent.

        - jobs run one at a time on the single GPU worker, so no two fetches race
        """
        local = self._base_dir / scene / self._source.scene_video(scene)

        if local.exists():
            return local

        on_line = self._op_log.rclone_progress("⬇ fetching video")
        return self._source.fetch_video(scene, local.parent, on_line=on_line)

    def _push_async(self, scene: str) -> None:
        """
        Push the scene's outputs to the processed bucket in a detached, non-fatal thread.
        """

        def _worker() -> None:
            t0 = time.perf_counter()
            self._op_log.append_line("push: uploading to environments-processed")

            try:
                self._source.push_outputs(
                    self._base_dir / scene, scene, on_line=self._op_log.append_line
                )
                self._op_log.append_line(
                    f"push: done in {time.perf_counter() - t0:.1f}s"
                )
            except (
                RuntimeError,
                OSError,
            ) as exc:  # non-fatal: outputs already on local disk
                logger.exception("push failed")
                self._op_log.append_line(f"push: FAILED ({exc})")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run(self, event: object, force: bool) -> None:
        """
        Run the chosen stages for (scene, backend); Force re-run overwrites finished stages.
        """
        if self._gpu.busy:
            self._op_log.append_line("busy — wait for the current job")
            return

        self._flush_state()
        scene = self.scene_select.value

        if not scene:
            self._op_log.error_op("select a scene first")
            return

        # Overrides box: a YAML mapping merged over the recorded run config
        try:
            overrides = yaml.safe_load(self.overrides.value) or {}
        except yaml.YAMLError as exc:
            self._op_log.error_op(f"overrides are not valid YAML: {exc}")
            return

        if not isinstance(overrides, dict):
            self._op_log.error_op("overrides must be a YAML mapping")
            return

        self._start_run(
            scene,
            self.backend.value,
            list(self.stages.value) or None,
            overrides,
            overwrite=force,
        )

    def _start_run(
        self,
        scene: str,
        backend: str,
        stages: list[str] | None,
        overrides: dict,
        overwrite: bool,
    ) -> None:
        """
        Enqueue a Reconstructor run on the GPU worker, then reload the backend.
        """
        doc = pn.state.curdoc
        other_backends = tuple(f"/{b}/**" for b in self.backend.options if b != backend)

        def job() -> None:
            # The run pulls the server's backend in full: later stages may need the dense arrays
            rec = self._pulled_reconstructor(scene, backend, overrides, other_backends)

            # Fetch the video only when preproc will run; it reads input_path then
            runs_preproc = (stages is None or "preproc" in stages) and (
                overwrite or not rec.done("preproc")
            )

            if runs_preproc:
                rec.config["input_path"] = str(self._ensure_local_video(scene))

            # Record the config beside the outputs, as the CLI does
            rec.write_run_config()

            # Stream the Reconstructor's log records into the console while the stages run
            with self._op_log.attach_logging():
                rec.run(stages, overwrite=overwrite)

            self._push_async(scene)

        def on_done(res: Any) -> None:
            self._sync_busy()

            if isinstance(res, Exception):
                hint = " — use Force re-run" if "already exists" in str(res) else ""
                self._op_log.error_op(f"{res}{hint}")
                return

            # Fresh outputs invalidate every cached load; clearing the key stops the reload short-circuiting
            self._cache.drop((scene, backend))
            self._current_key = None
            self._load_outputs(scene, backend)

        self._set_busy(True)
        self._op_log.start_op(f"running {scene}/{backend}")
        self._gpu.submit(job, on_done, doc)

    ########
    # Load
    ########

    def _load_outputs(self, scene: str, backend: str) -> None:
        """
        Enqueue loading one backend's result, mesh path and lifted store; render on the IOLoop.
        """
        key = (scene, backend)

        # Already displayed and idle, or already loading: nothing to do
        if self._current_key == key and not self._op_log.is_running:
            return

        if self._loading_key == key:
            return

        self._loading_key = key
        doc = pn.state.curdoc
        max_points = self.max_display_points.value
        other_backends = tuple(f"/{b}/**" for b in self.backend.options if b != backend)

        def job() -> tuple | None:
            cached = self._cache.get(key, "loaded")

            if cached is not None:
                self._op_log.append_line(f"{scene}/{backend}: using in-memory cache")
                return cast(tuple, cached)

            # Viewing pulls the scene root and this backend without the dense per-pixel arrays
            rec = self._pulled_reconstructor(
                scene, backend, None, PULL_EXCLUDES + other_backends
            )

            if not rec.done("pointcloud"):
                return None

            with self._op_log.step(f"{scene}/{backend}: reading pointcloud.zarr"):
                result = rec.result

            value = (
                result,
                rec.outputs["mesh"] if rec.done("mesh") else None,
                rec.outputs["semantics"] if rec.done("semantics") else None,
                rec.config["semantics"]["extractor"],
            )
            self._cache.put(key, "loaded", value)
            return value

        def on_done(res: Any) -> None:
            if self._loading_key == key:
                self._loading_key = None

            self._sync_busy()

            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return

            if res is None:
                self._op_log.append_line(f"{backend} not run for {scene} — press Run")
                self._op_log.finish_op()
                return

            result, mesh_path, lifted_store, extractor_name = res

            # Label the render step with the displayed point count
            n_pts = len(result.points)
            n_shown = min(n_pts, max_points) if max_points > 0 else n_pts

            with self._op_log.step(f"rendering {n_shown:,} points × 2 panes"):
                self._viewer.load(
                    result,
                    mesh_path=mesh_path,
                    lifted_store=lifted_store,
                    max_points=max_points,
                )

            self._extractor_name = extractor_name
            self._current_key = key
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"loading {scene}/{backend}")
        self._gpu.submit(job, on_done, doc)

    ########
    # View mode and query
    ########

    def _on_view_mode(self, event: param.parameterized.Event) -> None:
        """
        Switch pointcloud/mesh, keeping the right pane's similarity map across the switch.

        - the mesh materializes on the WORKER (shared-cache hit skips the read); set_mode renders on the IOLoop
        - an active query unscored in the new mode is re-scored in the same job
        - point and mesh use different feature spaces, so colors are never reused across modes
        """
        mode = event.new
        key = self._current_key
        query = self._viewer.active_query()
        need_score = bool(query) and self._viewer.cached_query_colors(mode) is None
        positive, negative, extractor_name = query if query else ([], [], "")
        doc = pn.state.curdoc

        def job() -> Any:
            # Mesh mode: materialize the polydata off the IOLoop; the SceneCache skips a re-read
            if mode == "mesh":
                preloaded = self._cache.get(key, "mesh") if key else None
                loaded = self._viewer.ensure_mesh_polydata(preloaded=preloaded)

                if loaded and key:
                    self._cache.put(key, "mesh", self._viewer.mesh_polydata())

            # Explicit target mode: the viewer's mode is still the OUTGOING one until on_done
            if need_score:
                return self._viewer.score_query(
                    positive=positive,
                    negative=negative,
                    extractor_name=extractor_name,
                    mode=mode,
                )

            return None

        def on_done(res: Any) -> None:
            # Sync, not unconditional re-enable: another queued job must keep widgets locked
            self._sync_busy()

            # Failed switch (e.g. mesh read error): snap the radio back without re-firing this watcher
            if isinstance(res, Exception):
                self._op_log.error_op(str(res))

                with param.parameterized.discard_events(self.view_mode):
                    self.view_mode.value = self._viewer.mode

                return

            # Mesh (if any) is resident now: render the new mode on the IOLoop
            self._viewer.set_mode(mode)

            if res is not None:
                self._viewer.render_query(res)

            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op(f"switching to {mode}")
        self._gpu.submit(job, on_done, doc)

    def _on_query(self, event: object) -> None:
        """
        Score the positive/negative query off the IOLoop; recolor the right pane on done.
        """
        positive = _split_terms(self.pos_query.value)
        negative = _split_terms(self.neg_query.value)
        extractor_name = self._extractor_name
        doc = pn.state.curdoc

        def job() -> Any:
            return self._viewer.score_query(
                positive=positive, negative=negative, extractor_name=extractor_name
            )

        def on_done(res: Any) -> None:
            # Sync, not unconditional re-enable: another queued job must keep widgets locked
            self._sync_busy()

            if isinstance(res, Exception):
                self._op_log.error_op(str(res))
                return

            if res is None:  # no scene loaded -> nothing to recolor
                self._op_log.finish_op()
                return

            self._viewer.render_query(res)
            self._op_log.finish_op()

        self._set_busy(True)
        self._op_log.start_op("query")
        self._gpu.submit(job, on_done, doc)

    ########
    # Layout
    ########

    def view(self) -> pn.template.MaterialTemplate:
        """
        Page: sidebar, split viewer, and the operations console pinned under it.

        Returns:
            The page template.
        """
        # Resizable console showing the shared op log
        self._progress = pn.pane.HTML(
            self._op_log.render_html(), sizing_mode="stretch_both"
        )
        self._console = pn.Column(
            self._progress,
            sizing_mode="stretch_width",
            height=190,
            styles={
                "resize": "vertical",
                "overflow": "auto",
                "min-height": "70px",
                "border-top": "2px solid #2596be",
                "background": "#0d1117",
                "padding": "4px 8px",
            },
        )
        self._seen_log_version = -1

        # One poll drives busy state, the console and the debounced state flush; tab close flushes too
        try:
            pn.state.add_periodic_callback(self._tick, period=300, start=True)
            pn.state.on_session_destroyed(lambda _ctx: self._flush_state())
        except RuntimeError:
            logger.debug(
                "no server doc; busy state, console and state flush are static",
                exc_info=True,
            )

        return pn.template.MaterialTemplate(
            title="splats",
            sidebar=[self._sidebar],
            main=[
                pn.Column(
                    self._viewer.layout, self._console, sizing_mode="stretch_both"
                )
            ],
            header_background="#2596be",
            sidebar_width=340,
        )

    def _tick(self) -> None:
        """
        300 ms poll: mirror the worker's busy flag, refresh a changed console, flush dirty state.
        """
        self._sync_busy()

        if self._op_log.version != self._seen_log_version:
            self._seen_log_version = self._op_log.version
            self._progress.object = self._op_log.render_html()

        self._flush_state()
