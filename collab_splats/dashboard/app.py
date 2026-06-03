# collab_splats/dashboard/app.py
"""Dashboard entry points: new single-page SplatsApp and legacy 5-tab App."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import panel as pn
import param
import yaml

from collab_splats.dashboard.config import RunConfig
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.pipeline import run_pipeline
from collab_splats.dashboard.sources import SessionSource
from collab_splats.dashboard.viewer import SplitViewer, load_lifted_normed
from collab_splats.pointcloud.feedforward.base import FeedforwardResult

logger = logging.getLogger(__name__)

########
# Helpers (kept for backward compat — used by tests and __init__)
########


def _scan_output_dirs(base_dir: Path) -> list[str]:
    """Return sorted names of subdirs in base_dir that contain run_config.yaml."""
    if not base_dir.is_dir():
        return []
    return sorted(
        p.name for p in base_dir.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )


########
# SplatsApp — new single-page dashboard
########

_ENV_MODELS = ["vggt_omega", "vggtx", "mapanything"]
_EXTRACTORS = ["talk2dino", "maskclip", "dinov2"]
_SAMPLERS = ["balanced", "optical_flow"]


class SplatsApp(param.Parameterized):
    """Single-page dashboard wiring source, pipeline, and the split viewer."""

    def __init__(self, base_dir: Path = Path("/workspace/outputs"),
                 source: SessionSource | None = None, **params) -> None:
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._source = source if source is not None else SessionSource()
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
        self.query = pn.widgets.TextInput(name="Query", placeholder="e.g. chair")
        self.mesh_voxel = pn.widgets.FloatInput(name="voxel_size", value=0.01)
        self.mesh_sdf = pn.widgets.FloatInput(name="sdf_trunc", value=0.04)
        self.mesh_depth = pn.widgets.FloatInput(name="depth_trunc", value=10.0)
        self.view_mode = pn.widgets.RadioButtonGroup(options=["pointcloud", "mesh"], value="pointcloud")
        self.run_btn = pn.widgets.Button(name="Run", button_type="primary")
        self.force_btn = pn.widgets.Button(name="Force re-run", button_type="warning")

        self.session_select.param.watch(self._on_session, "value")
        self.video_select.param.watch(self._on_video, "value")
        self.query.param.watch(self._on_query, "value")
        self.view_mode.param.watch(lambda e: self._viewer.set_mode(e.new), "value")
        self.run_btn.on_click(lambda e: self._on_run(e, force=False))
        self.force_btn.on_click(lambda e: self._on_run(e, force=True))

        self._sidebar = pn.Column(
            "## Source", self.session_select, self.video_select,
            pn.Card(self.sampling, self.max_frames, title="Frame sampling", collapsed=True),
            pn.Card(self.env_model, self.conf, title="Environment model", collapsed=True),
            pn.Card(self.extractor, self.query, title="Semantics", collapsed=False),
            pn.Card(self.mesh_voxel, self.mesh_sdf, self.mesh_depth, title="Mesh params", collapsed=True),
            "### View", self.view_mode,
            pn.Row(self.run_btn, self.force_btn),
        )

    # ---- data wiring ---------------------------------------------------

    def _refresh_sessions(self) -> None:
        """Populate session dropdown from source."""
        try:
            self.session_select.options = self._source.list_sessions()
        except Exception as exc:
            logger.warning("session listing failed: %s", exc)
            self.session_select.options = []

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
            env_model=self.env_model.value,
            conf_threshold=self.conf.value,
            semantic_extractor=self.extractor.value,
            query=self.query.value,
            mesh_voxel_size=self.mesh_voxel.value,
            mesh_sdf_trunc=self.mesh_sdf.value,
            mesh_depth_trunc=self.mesh_depth.value,
        )

    def _ensure_local_video(self, session: str, name: str) -> Path:
        """Return local video path, fetching from source if needed."""
        local = self._base_dir / session / Path(name).stem / name
        if not local.exists():
            local = self._source.fetch_video(session, name, local.parent)
        return local

    def _on_run(self, event, force: bool) -> None:
        """Spawn background thread to run the full pipeline."""
        session, name = self.session_select.value, self.video_select.value
        if not session or not name:
            return
        stem = Path(name).stem
        config = self._current_config()

        def worker() -> None:
            video = self._ensure_local_video(session, name)
            run_pipeline(
                video_path=video, session=session, stem=stem, config=config,
                op_log=self._op_log, source=self._source, base_dir=self._base_dir,
            )
            self._load_outputs(session, stem)

        threading.Thread(target=worker, daemon=True).start()

    def _load_outputs(self, session: str, stem: str) -> None:
        """Load FeedforwardResult and semantics into the viewer."""
        out = self._base_dir / session / stem
        if not (out / "feedforward.zarr").exists():
            self._source.pull_processed(session, stem, out)
        result = FeedforwardResult.load_zarr(out / "feedforward.zarr")
        try:
            lifted = load_lifted_normed(result, out / "semantics")
        except Exception:
            lifted = None
        mesh_path = out / "mesh" / "mesh.ply"
        self._viewer.load(result, mesh_path=mesh_path if mesh_path.exists() else None,
                          lifted_normed=lifted)

    def _on_query(self, event) -> None:
        """Forward query text to viewer for live recolouring."""
        self._viewer.query(event.new, extractor_name=self.extractor.value)

    # ---- layout --------------------------------------------------------

    def view(self) -> pn.template.MaterialTemplate:
        """Assemble the full single-page layout."""
        pn.extension("vtk")

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
            title="splats", sidebar=[self._sidebar], main=[main],
            header_background="#2596be", sidebar_width=340,
        )


########
# Legacy App — 5-tab dashboard (kept for backward compat)
########

_HEADER_CSS = """
.bk-tab.bk-active {
    border-bottom: 3px solid #2596be !important;
    font-weight: 700 !important;
    color: #2596be !important;
}
"""


class App(param.Parameterized):
    """Unified collab-splats dashboard — 5-tab pipeline app (legacy)."""

    _tab_names = ("Preprocess", "Semantics", "Reconstruct", "Visualize", "Localize")

    def __init__(self, base_dir: str = "/workspace/outputs", video_server=None, **params: Any):
        super().__init__(**params)
        from collab_splats.dashboard.panes.preprocess import PreprocessPane
        from collab_splats.dashboard.panes.reconstruct import ReconstructPane
        from collab_splats.dashboard.panes.semantics import SemanticsPane
        from collab_splats.dashboard.panes.localize import LocalizePane
        from collab_splats.dashboard.panes.visualize import ScenePanel, _scan_datasets, _scan_backends, _scan_extractors
        from collab_splats.dashboard.state import AppState
        from collab_splats.dashboard.video_server import VideoFileServer, VideoStreamHandler, start_video_server

        self._VideoStreamHandler = VideoStreamHandler
        self._start_video_server = start_video_server
        self._scan_backends = _scan_backends
        self._scan_extractors = _scan_extractors
        self._scan_datasets = _scan_datasets

        self._base_dir = Path(base_dir)
        self._state = AppState()
        self._op_log = OperationLog()
        self._tabs: pn.Tabs | None = None
        self._video_server = video_server if video_server is not None else start_video_server(port=7863)

        self._preprocess = PreprocessPane(
            state=self._state, op_log=self._op_log, video_server=self._video_server
        )
        self._reconstruct = ReconstructPane(state=self._state, op_log=self._op_log)
        self._panes = {
            "Preprocess": self._preprocess,
            "Semantics": SemanticsPane(state=self._state, op_log=self._op_log),
            "Reconstruct": self._reconstruct,
            "Visualize": ScenePanel(base_dir=self._base_dir, state=self._state, op_log=self._op_log),
            "Localize": LocalizePane(state=self._state, op_log=self._op_log),
        }

        # Sidebar LOAD RESULTS widgets
        datasets = _scan_datasets(self._base_dir)
        dataset_names = [p.name for p in datasets]
        self._results_dataset_dd = pn.widgets.Select(
            name="Dataset", options=dataset_names or ["(none)"], width=280,
        )
        self._results_load_btn = pn.widgets.Button(
            name="⚡  Load Results", button_type="primary", width=280,
        )
        self._results_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'>No results loaded</p>", width=280,
        )
        self._results_dataset_dd.param.watch(self._on_results_dataset_change, "value")
        self._results_load_btn.on_click(self._on_results_load)

        # Sidebar SEMANTICS MODEL widgets
        self._sem_extractor_dd = pn.widgets.Select(name="Extractor", options=[], width=280)
        self._sem_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'>No dataset loaded</p>", width=280,
        )
        self._sem_extractor_dd.param.watch(self._on_sem_extractor_change, "value")

        # Sidebar POINTCLOUD MODEL widgets
        self._pc_creator_dd = pn.widgets.Select(
            name="Creator", options=["vggtx", "mapanything", "vggt_omega"], value="vggtx", width=280,
        )
        self._pc_conf_slider = pn.widgets.FloatSlider(
            name="Conf threshold", start=0.0, end=100.0, step=1.0, value=35.0, width=280,
        )
        self._pc_creator_dd.param.watch(self._on_pc_creator_change, "value")
        self._pc_conf_slider.param.watch(self._on_pc_conf_change, "value")
        self._state.pointcloud_creator = self._pc_creator_dd.value
        self._state.pointcloud_creator_conf = self._pc_conf_slider.value

        if dataset_names:
            self._on_results_dataset_change(None)

        # Localize sidebar widgets
        self._localize_extractor_dd = pn.widgets.Select(
            name="Localize method", options=["DISK+LightGlue", "XFeat+MNN"],
            value="DISK+LightGlue", width=280,
        )
        self._localize_extractor_dd.param.watch(self._on_localize_extractor_changed, ["value"])
        self._state.param.watch(self._on_pointcloud_backend_changed_localize, ["pointcloud_backend"])

        # Sidebar VIEW widgets
        self._ground_plane_status = pn.pane.HTML(
            "<p style='color:#666;font-size:11px'></p>", width=280,
        )
        self._redetect_btn = pn.widgets.Button(
            name="Detect ground plane", button_type="primary", width=170,
        )
        self._sidebar_frustum_check = pn.widgets.Checkbox(name="Show frustums", value=False)
        self._redetect_btn.on_click(self._on_redetect_ground_plane)
        self._sidebar_frustum_check.param.watch(self._on_sidebar_frustum_toggle, "value")

        # Sidebar MESH widgets
        self._mesh_voxel_input = pn.widgets.FloatInput(
            name="voxel_size", value=0.005, step=0.005, start=0.001, end=1.0, width=85,
        )
        self._mesh_sdf_input = pn.widgets.FloatInput(
            name="sdf_trunc", value=0.02, step=0.001, start=0.001, end=5.0, width=85,
        )
        self._mesh_depth_input = pn.widgets.FloatInput(
            name="depth_trunc", value=1.0, step=0.5, start=0.1, end=200.0, width=85,
        )
        self._mesh_run_btn = pn.widgets.Button(
            name="⚙  Run Mesh", button_type="primary", width=280, disabled=True,
        )
        self._mesh_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'>Load data to enable</p>", width=280,
        )
        self._mesh_progress = pn.widgets.Progress(
            active=False, visible=False, width=275, bar_color="primary"
        )
        self._mesh_stop_btn = pn.widgets.Button(
            name="⏹  Stop", button_type="danger", width=280, visible=False,
        )
        self._mesh_run_btn.on_click(self._on_run_mesh_sidebar)
        self._mesh_stop_btn.on_click(self._on_stop_mesh_sidebar)
        self._state.param.watch(self._on_feedforward_for_mesh, "feedforward_result")
        for _w in (self._mesh_voxel_input, self._mesh_sdf_input, self._mesh_depth_input):
            _w.param.watch(lambda _e: self._check_mesh_cache(), "value")

        # Sidebar section containers
        self._localize_section = pn.Column(
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Localize</h3>"),
            self._localize_extractor_dd,
            pn.layout.Divider(),
            visible=False,
        )
        self._mesh_section = pn.Column(
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Mesh</h3>"),
            pn.Row(self._mesh_voxel_input, self._mesh_sdf_input, self._mesh_depth_input),
            pn.Row(self._mesh_run_btn, self._mesh_stop_btn),
            self._mesh_progress,
            self._mesh_status,
            visible=False,
        )
        self._view_section = pn.Column(
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>View</h3>"),
            pn.Row(self._redetect_btn, self._sidebar_frustum_check),
            self._ground_plane_status,
            self._mesh_section,
            visible=False,
        )
        self._panes["Visualize"]._mode_selector.param.watch(
            lambda e: setattr(self._mesh_section, "visible", e.new == "Mesh"), "value"
        )

        self._sidebar = self._build_sidebar()

    def _build_sidebar(self) -> pn.Column:
        """Build the persistent sidebar: dataset controls + model info."""
        self._new_video_btn = pn.widgets.Button(
            name="📹  New from video", button_type="light", width=280
        )
        self._video_input = pn.widgets.TextInput(
            name="Video path", placeholder="/workspace/fieldwork-data/.../video.MP4",
            width=280, visible=False,
        )
        self._confirm_btn = pn.widgets.Button(
            name="Start preprocessing", button_type="success", width=280, visible=False
        )
        self._session_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'></p>", width=280
        )
        self._new_video_btn.on_click(self._on_new_video)
        self._confirm_btn.on_click(self._on_confirm_session)

        return pn.Column(
            pn.pane.HTML("<h3 style='color:#2596be;margin:0 0 8px 0'>Dataset</h3>"),
            self._results_dataset_dd,
            self._results_load_btn,
            self._results_status,
            pn.layout.Divider(),
            self._new_video_btn,
            self._video_input,
            self._confirm_btn,
            self._session_status,
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Semantics Model</h3>"),
            self._sem_extractor_dd,
            self._sem_status,
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Pointcloud Model</h3>"),
            self._pc_creator_dd,
            self._pc_conf_slider,
            pn.layout.Divider(),
            self._localize_section,
            self._view_section,
            width=300,
        )

    def _on_new_video(self, event: Any) -> None:
        self._video_input.visible = True
        self._confirm_btn.visible = True

    def _on_confirm_session(self, event: Any) -> None:
        """Handle session confirmation for both new-video and load-existing flows."""
        try:
            self._do_confirm_session()
        except Exception as exc:
            logger.exception("Confirm session failed")
            self._session_status.object = (
                f"<p style='color:#e05050;font-size:11px'>Error: {exc}</p>"
            )

    def _do_confirm_session(self) -> None:
        """Inner confirm logic — exceptions surface to _on_confirm_session."""
        self._session_status.object = "<p style='color:#aaa;font-size:12px'>Loading…</p>"
        video_path = Path(self._video_input.value.strip())
        if not video_path.exists():
            self._session_status.object = (
                f"<p style='color:#e05050;font-size:12px'>Not found: {video_path}</p>"
            )
            return
        if self._tabs is not None:
            self._tabs.active = 0
        self._state.video_path = video_path
        auto_out = Path("/workspace/outputs") / video_path.stem
        self._state.output_dir = auto_out
        zarr_candidate = auto_out / "frames.zarr"
        if zarr_candidate.exists():
            self._state.frames_zarr_path = zarr_candidate
        self._session_status.object = (
            f"<p style='color:#50c050;font-size:12px'>Video: {video_path.name}<br/>"
            f"Output: {auto_out}</p>"
        )
        self._video_input.visible = False
        self._confirm_btn.visible = False

    def _on_results_dataset_change(self, event: Any) -> None:
        """Repopulate extractor dropdown when dataset selection changes."""
        name = self._results_dataset_dd.value
        if not name or name == "(none)":
            self._sem_extractor_dd.options = []
            return
        ds_dir = self._base_dir / name
        backends = self._scan_backends(ds_dir)
        if backends:
            extractors = self._scan_extractors(ds_dir, backends[0])
            self._sem_extractor_dd.options = extractors or ["(none)"]
            if extractors:
                self._sem_status.object = (
                    f"<p style='color:#50c050;font-size:12px'>{extractors[0]}</p>"
                )

    def _on_results_load(self, event: Any) -> None:
        """Kick off background load of FeedforwardResult."""
        self._results_load_btn.disabled = True
        self._results_status.object = "<p style='color:#aaa;font-size:12px'>Loading…</p>"
        t = threading.Thread(target=self._do_load_results, daemon=True)
        t.start()

    def _do_load_results(self) -> None:
        """Background: load FeedforwardResult and populate AppState for Visualize tab."""
        try:
            ds_name = self._results_dataset_dd.value
            if not ds_name or ds_name == "(none)":
                self._results_status.object = (
                    "<p style='color:#e05050;font-size:12px'>Select dataset</p>"
                )
                return
            ds_dir = self._base_dir / ds_name
            backends = self._scan_backends(ds_dir)
            if not backends:
                self._results_status.object = (
                    "<p style='color:#e05050;font-size:12px'>No backends found in dataset</p>"
                )
                return
            backend = backends[0]
            zarr_path = ds_dir / backend / "feedforward.zarr"
            if not zarr_path.exists():
                self._results_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>feedforward.zarr not found in {backend}</p>"
                )
                return
            result = FeedforwardResult.load_zarr(zarr_path)
            n_pts = len(result.points)

            import json
            import numpy as np
            transforms_path = ds_dir / backend / "transforms.json"
            gp_R = None
            gp_t = None
            gp_status = ""
            try:
                from collab_splats.pointcloud.utils import fit_dominant_plane
                gp_R, gp_t = fit_dominant_plane(result.points)
                transforms_path.write_text(
                    json.dumps({"ground_plane": {"R": gp_R.tolist(), "t": gp_t.tolist()}}, indent=2)
                )
                gp_status = "auto-detected · saved"
            except Exception as exc:
                logger.warning("Ground plane auto-detect failed: %s", exc)
                if transforms_path.exists():
                    try:
                        data = json.loads(transforms_path.read_text())
                        gp = data.get("ground_plane")
                        if gp and "R" in gp and "t" in gp:
                            gp_R = np.array(gp["R"], dtype=np.float64)
                            gp_t = np.array(gp["t"], dtype=np.float64)
                            gp_status = "loaded from file (detect failed)"
                    except Exception:
                        logger.warning("Could not parse transforms.json at %s", transforms_path)

            def _set_state() -> None:
                self._state.output_dir = ds_dir
                self._state.pointcloud_backend = backend
                if backend in self._pc_creator_dd.options:
                    self._pc_creator_dd.value = backend
                extractor_val = self._sem_extractor_dd.value
                if extractor_val and extractor_val not in ("", "(none)"):
                    self._state.semantic_extractor = extractor_val
                self._state.ground_plane_R = gp_R
                self._state.ground_plane_t = gp_t
                zarr_candidate = ds_dir / "frames.zarr"
                if zarr_candidate.exists():
                    self._state.frames_zarr_path = zarr_candidate
                config_file = ds_dir / "run_config.yaml"
                if config_file.exists():
                    try:
                        config = yaml.safe_load(config_file.read_text())
                        raw_vp = config.get("video_path") or config.get("input_path")
                        if raw_vp:
                            vp = Path(raw_vp)
                            if vp.exists():
                                self._state.video_path = vp
                    except Exception as exc:
                        logger.warning("Could not parse video_path from %s: %s", config_file, exc)
                self._state.feedforward_result = result
                if gp_status:
                    self._ground_plane_status.object = (
                        f"<p style='color:#50c050;font-size:11px'>{gp_status}</p>"
                    )
                self._results_status.object = (
                    f"<p style='color:#50c050;font-size:12px'>"
                    f"Loaded {n_pts:,} pts<br/>"
                    f"<span style='color:#666'>{ds_name} / {backend}</span></p>"
                )

            pn.io.state.execute(_set_state)
        except Exception as exc:
            logger.exception("_do_load_results failed")
            self._results_status.object = (
                f"<p style='color:#e05050;font-size:12px'>Load failed: {exc}</p>"
            )
        finally:
            self._results_load_btn.disabled = False

    def _on_sem_extractor_change(self, event: Any) -> None:
        """Push selected semantic extractor to AppState."""
        self._state.semantic_extractor = event.new or ""
        if event.new and event.new != "(none)":
            self._sem_status.object = (
                f"<p style='color:#50c050;font-size:12px'>{event.new}</p>"
            )

    def _on_pc_creator_change(self, event: Any) -> None:
        """Push selected pointcloud creator to AppState."""
        self._state.pointcloud_creator = event.new or "vggtx"

    def _on_pc_conf_change(self, event: Any) -> None:
        """Push conf threshold to AppState."""
        self._state.pointcloud_creator_conf = event.new

    def _on_redetect_ground_plane(self, event: Any) -> None:
        """Recompute ground plane from current result and save to transforms.json."""
        if self._state.feedforward_result is None:
            return
        self._redetect_btn.disabled = True
        t = threading.Thread(target=self._do_detect_ground_plane, daemon=True)
        t.start()

    def _on_sidebar_frustum_toggle(self, event: Any) -> None:
        """Forward frustum toggle to ScenePanel."""
        scene = self._panes.get("Visualize")
        if scene is not None and hasattr(scene, "_on_frustum_toggle_from_sidebar"):
            scene._on_frustum_toggle_from_sidebar(event.new)

    def _on_pointcloud_backend_changed_localize(self, event: Any) -> None:
        """Sync localize_method from pointcloud_backend."""
        self._state.localize_method = event.new or ""

    def _on_localize_extractor_changed(self, event: Any) -> None:
        self._state.localize_extractor = event.new or "DISK+LightGlue"

    def _mesh_params_path(self) -> "Path | None":
        """Return path to mesh_params.json for the currently loaded dataset/backend."""
        output_dir = self._state.output_dir
        backend = self._state.pointcloud_backend
        if not output_dir or not backend:
            return None
        return Path(output_dir) / backend / "mesh" / "mesh_params.json"

    def _check_mesh_cache(self) -> None:
        """Compare saved mesh params to current widget values; update status label."""
        if self._state.feedforward_result is None:
            return
        path = self._mesh_params_path()
        if path is None or not path.exists():
            self._mesh_status.object = "<p style='color:#666;font-size:12px'>Ready</p>"
            return
        try:
            import json
            cached = json.loads(path.read_text())
            match = (
                cached.get("voxel_size") == self._mesh_voxel_input.value
                and cached.get("sdf_trunc") == self._mesh_sdf_input.value
                and cached.get("depth_trunc") == self._mesh_depth_input.value
                and not cached.get("clean_repair", False)
            )
            if match:
                self._mesh_status.object = "<p style='color:#50c050;font-size:12px'>Cached ✓</p>"
            else:
                self._mesh_status.object = (
                    "<p style='color:#e0a000;font-size:12px'>Params changed — rerun?</p>"
                )
        except Exception:
            self._mesh_status.object = "<p style='color:#666;font-size:12px'>Ready</p>"

    def _on_feedforward_for_mesh(self, event: Any) -> None:
        """Enable sidebar Run Mesh button when feedforward_result is available."""
        self._mesh_run_btn.disabled = event.new is None
        self._check_mesh_cache()

    def _on_run_mesh_sidebar(self, event: Any) -> None:
        """Trigger mesh generation via ScenePanel with sidebar params."""
        scene = self._panes.get("Visualize")
        if scene is None:
            return
        self._mesh_run_btn.disabled = True
        self._mesh_stop_btn.visible = True
        self._mesh_status.object = "<p style='color:#aaa;font-size:12px'>Running…</p>"
        self._mesh_progress.value = 0
        self._mesh_progress.active = False
        self._mesh_progress.visible = True
        scene.run_mesh(
            voxel_size=self._mesh_voxel_input.value,
            sdf_trunc=self._mesh_sdf_input.value,
            depth_trunc=self._mesh_depth_input.value,
            clean_repair=False,
            on_done=self._on_mesh_done,
            on_progress=self._on_mesh_progress,
        )

    def _on_mesh_progress(self, desc: str, pct: int) -> None:
        """Update sidebar progress bar from tqdm messages."""
        self._mesh_progress.value = pct
        self._mesh_status.object = (
            f"<p style='color:#aaa;font-size:12px'>{desc}: {pct}%</p>"
        )

    def _on_mesh_done(self, ok: bool, msg: str) -> None:
        """Re-enable sidebar Run Mesh button and update status after generation."""
        self._mesh_run_btn.disabled = False
        self._mesh_stop_btn.visible = False
        self._mesh_progress.active = False
        self._mesh_progress.visible = False
        if ok:
            self._check_mesh_cache()
        else:
            color = "#e05050" if msg != "Mesh cancelled." else "#e0a000"
            self._mesh_status.object = f"<p style='color:{color};font-size:12px'>{msg}</p>"

    def _on_stop_mesh_sidebar(self, event: Any) -> None:
        """Terminate the mesh subprocess and reset sidebar controls."""
        scene = self._panes.get("Visualize")
        if scene is not None:
            scene.stop_mesh()
        self._mesh_stop_btn.visible = False
        self._mesh_run_btn.disabled = False
        self._mesh_status.object = "<p style='color:#e0a000;font-size:12px'>Cancelling…</p>"

    def _do_detect_ground_plane(self) -> None:
        """Background: RANSAC ground plane detection; save transforms.json."""
        try:
            import json
            import numpy as np
            from collab_splats.pointcloud.utils import fit_dominant_plane

            result = self._state.feedforward_result
            R, t = fit_dominant_plane(result.points)
            transforms_path = (
                Path(self._state.output_dir) / self._state.pointcloud_backend / "transforms.json"
            )
            transforms_path.write_text(
                json.dumps({"ground_plane": {"R": R.tolist(), "t": t.tolist()}}, indent=2)
            )

            def _apply() -> None:
                self._state.ground_plane_R = R
                self._state.ground_plane_t = t
                self._ground_plane_status.object = (
                    "<p style='color:#50c050;font-size:11px'>auto-detected · saved</p>"
                )

            pn.io.state.execute(_apply)
        except Exception as exc:
            logger.exception("Ground plane detection failed")
            self._ground_plane_status.object = (
                f"<p style='color:#e05050;font-size:11px'>Detection failed: {exc}</p>"
            )
        finally:
            self._redetect_btn.disabled = False

    def servable(self) -> pn.template.MaterialTemplate:
        """Build and return the full MaterialTemplate for serving."""
        pn.extension("tabulator", "vtk", css_files=[], raw_css=[_HEADER_CSS])

        self._tabs = pn.Tabs(
            *[(name, self._panes[name].panel()) for name in self._tab_names],
            dynamic=True,
            sizing_mode="stretch_width",
        )

        preprocess_tab_index = list(self._panes.keys()).index("Preprocess")
        reconstruct_tab_index = list(self._panes.keys()).index("Reconstruct")
        visualize_tab_index = list(self._panes.keys()).index("Visualize")
        localize_tab_index = list(self._panes.keys()).index("Localize")
        self._panes["Preprocess"].wire_tabs(self._tabs, preprocess_tab_index)
        self._panes["Reconstruct"].wire_tabs(self._tabs, reconstruct_tab_index)
        self._panes["Visualize"].wire_tabs(self._tabs, visualize_tab_index)

        def _on_tab_change(event: Any) -> None:
            self._view_section.visible = (event.new == visualize_tab_index)
            self._localize_section.visible = (event.new == localize_tab_index)

        self._tabs.param.watch(_on_tab_change, "active")

        main_content = pn.Column(
            self._tabs,
            self._op_log.panel(),
            sizing_mode="stretch_width",
        )

        return pn.template.MaterialTemplate(
            title="collab-splats",
            sidebar=[self._sidebar],
            main=[main_content],
            header_background="#2596be",
            sidebar_width=320,
        )


########
# Entry points
########


def run_app(host: str = "0.0.0.0", port: int = 7860,
            base_dir: str = "/workspace/outputs") -> None:
    """Serve the splats dashboard."""
    def factory() -> pn.template.MaterialTemplate:
        return SplatsApp(base_dir=Path(base_dir)).view()

    pn.serve(factory, address=host, port=port, show=False, title="splats")
