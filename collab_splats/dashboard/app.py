from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import logging

import panel as pn
import param
import yaml

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.preprocess import PreprocessPane
from collab_splats.dashboard.panes.reconstruct import ReconstructPane
from collab_splats.dashboard.panes.semantics import SemanticsPane
from collab_splats.dashboard.panes.localize import LocalizePane
from collab_splats.dashboard.panes.visualize import ScenePanel, _scan_datasets, _scan_backends, _scan_extractors
from collab_splats.dashboard.state import AppState

from collab_splats.dashboard.video_server import VideoFileServer, start_video_server

logger = logging.getLogger(__name__)


def _scan_output_dirs(base_dir: Path) -> list[str]:
    """Return sorted names of subdirs in base_dir that contain run_config.yaml."""
    if not base_dir.is_dir():
        return []
    return sorted(
        p.name for p in base_dir.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )


_HEADER_CSS = """
.bk-tab.bk-active {
    border-bottom: 3px solid #2596be !important;
    font-weight: 700 !important;
    color: #2596be !important;
}
"""


class App(param.Parameterized):
    """Unified collab-splats dashboard — 5-tab pipeline app."""

    _tab_names = ("Preprocess", "Semantics", "Reconstruct", "Visualize", "Localize")

    def __init__(self, base_dir: str = "/workspace/outputs", video_server: VideoFileServer | None = None, **params: Any):
        super().__init__(**params)
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
        # Sidebar MODELS widgets — created before _build_sidebar
        datasets = _scan_datasets(self._base_dir)
        dataset_names = [p.name for p in datasets]
        self._models_dataset_dd = pn.widgets.Select(
            name="Dataset", options=dataset_names or ["(none)"], width=280,
        )
        self._models_backend_dd = pn.widgets.Select(
            name="Pointcloud backend", options=[], width=280,
        )
        self._models_extractor_dd = pn.widgets.Select(
            name="Semantic model", options=[], width=280,
        )
        self._models_load_btn = pn.widgets.Button(
            name="⚡  Load", button_type="primary", width=280,
        )
        self._models_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'>No data loaded</p>", width=280,
        )
        self._models_dataset_dd.param.watch(self._on_models_dataset_change, "value")
        self._models_load_btn.on_click(self._on_models_load)
        # Populate backend options for initial dataset selection
        if dataset_names:
            self._on_models_dataset_change(None)

        # Sidebar VIEW widgets
        self._ground_plane_check = pn.widgets.Checkbox(
            name="Align ground plane", value=True,
        )
        self._ground_plane_status = pn.pane.HTML(
            "<p style='color:#666;font-size:11px'></p>", width=280,
        )
        self._redetect_btn = pn.widgets.Button(
            name="↺  Re-detect ground plane", button_type="light", width=280,
        )
        self._sidebar_frustum_check = pn.widgets.Checkbox(
            name="Show frustums", value=False,
        )
        self._ground_plane_check.param.watch(self._on_ground_plane_toggle, "value")
        self._redetect_btn.on_click(self._on_redetect_ground_plane)
        self._sidebar_frustum_check.param.watch(self._on_sidebar_frustum_toggle, "value")

        self._sidebar = self._build_sidebar()

    def _build_sidebar(self) -> pn.Column:
        """Build the persistent sidebar: session controls + active session info."""
        self._new_video_btn = pn.widgets.Button(
            name="📹  New from video", button_type="primary", width=280
        )
        self._load_existing_btn = pn.widgets.Button(
            name="📁  Load existing results", button_type="light", width=280
        )
        self._video_input = pn.widgets.TextInput(
            name="Video path", placeholder="/workspace/fieldwork-data/.../video.MP4",
            width=280, visible=False,
        )
        self._refresh_dirs_btn = pn.widgets.Button(name="↻", width=40, visible=False)
        _dir_options = _scan_output_dirs(self._base_dir)
        self._output_dir_select = pn.widgets.Select(
            name="Output directory",
            options=_dir_options if _dir_options else ["(no sessions found)"],
            width=230,
            visible=False,
            disabled=not bool(_dir_options),
        )
        self._refresh_dirs_btn.on_click(self._on_refresh_dirs)
        self._confirm_btn = pn.widgets.Button(
            name="Confirm", button_type="success", width=280, visible=False
        )
        self._session_status = pn.pane.HTML(
            "<p style='color:#666;font-size:12px'>No session loaded</p>", width=280
        )

        self._new_video_btn.on_click(self._on_new_video)
        self._load_existing_btn.on_click(self._on_load_existing)
        self._confirm_btn.on_click(self._on_confirm_session)

        return pn.Column(
            pn.pane.HTML("<h3 style='color:#2596be;margin:0 0 8px 0'>Session</h3>"),
            self._new_video_btn,
            self._load_existing_btn,
            self._video_input,
            pn.Row(self._output_dir_select, self._refresh_dirs_btn),
            self._confirm_btn,
            pn.layout.Divider(),
            self._session_status,
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>Models</h3>"),
            self._models_dataset_dd,
            self._models_backend_dd,
            self._models_extractor_dd,
            self._models_load_btn,
            self._models_status,
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>View</h3>"),
            self._ground_plane_check,
            self._ground_plane_status,
            self._redetect_btn,
            self._sidebar_frustum_check,
            width=300,
        )

    def _on_new_video(self, event: Any) -> None:
        self._video_input.visible = True
        self._output_dir_select.visible = False
        self._refresh_dirs_btn.visible = False
        self._confirm_btn.name = "Start session"
        self._confirm_btn.visible = True

    def _on_load_existing(self, event: Any) -> None:
        self._video_input.visible = False
        self._output_dir_select.visible = True
        self._refresh_dirs_btn.visible = True
        self._confirm_btn.name = "Load session"
        self._confirm_btn.visible = True

    def _on_refresh_dirs(self, event: Any) -> None:
        """Re-scan base_dir and refresh the output directory selector options."""
        dirs = _scan_output_dirs(self._base_dir)
        if dirs:
            self._output_dir_select.options = dirs
            self._output_dir_select.disabled = False
        else:
            self._output_dir_select.options = ["(no sessions found)"]
            self._output_dir_select.disabled = True

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
        if self._video_input.visible:
            video_path = Path(self._video_input.value.strip())
            if not video_path.exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>Not found: {video_path}</p>"
                )
                return
            self._state.video_path = video_path
            auto_out = Path("/workspace/outputs") / video_path.stem
            self._state.output_dir = auto_out
            self._session_status.object = (
                f"<p style='color:#50c050;font-size:12px'>Video: {video_path.name}<br/>"
                f"Output: {auto_out}</p>"
            )
        else:
            # Reject sentinel value set when base_dir contains no valid sessions
            selected = self._output_dir_select.value
            if selected == "(no sessions found)" or not selected:
                self._session_status.object = (
                    "<p style='color:#e05050;font-size:12px'>No session selected</p>"
                )
                return
            out_dir = self._base_dir / selected
            config_file = out_dir / "run_config.yaml"
            if not config_file.exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>No run_config.yaml in {out_dir}</p>"
                )
                return
            # Switch to Preprocess tab so the pane is rendered before state updates fire
            if self._tabs is not None:
                self._tabs.active = 0
            self._state.output_dir = out_dir

            # Parse video_path from config if present and file exists
            try:
                config = yaml.safe_load(config_file.read_text())
                raw_vp = config.get("video_path") or config.get("input_path")
                if raw_vp:
                    vp = Path(raw_vp)
                    if vp.exists():
                        self._state.video_path = vp
            except Exception as exc:
                logger.warning("Could not parse video_path from %s: %s", config_file, exc)

            self._session_status.object = (
                f"<p style='color:#50c050;font-size:12px'>Loaded: {out_dir.name}</p>"
            )

        self._video_input.visible = False
        self._output_dir_select.visible = False
        self._refresh_dirs_btn.visible = False
        self._confirm_btn.visible = False

    def _on_models_dataset_change(self, event: Any) -> None:
        """Repopulate backend dropdown when dataset selection changes."""
        name = self._models_dataset_dd.value
        if not name or name == "(none)":
            self._models_backend_dd.options = []
            self._models_extractor_dd.options = []
            return
        ds_dir = self._base_dir / name
        backends = _scan_backends(ds_dir)
        self._models_backend_dd.options = backends or ["(none)"]
        if backends:
            extractors = _scan_extractors(ds_dir, backends[0])
            self._models_extractor_dd.options = extractors or ["(none)"]

    def _on_models_load(self, event: Any) -> None:
        """Kick off background load of FeedforwardResult."""
        self._models_load_btn.disabled = True
        self._models_status.object = "<p style='color:#aaa;font-size:12px'>Loading…</p>"
        t = threading.Thread(target=self._do_load_models, daemon=True)
        t.start()

    def _do_load_models(self) -> None:
        """Background: load FeedforwardResult and populate AppState."""
        try:
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult

            ds_name = self._models_dataset_dd.value
            backend = self._models_backend_dd.value
            extractor = self._models_extractor_dd.value

            if not ds_name or ds_name == "(none)" or not backend or backend == "(none)":
                self._models_status.object = (
                    "<p style='color:#e05050;font-size:12px'>Select dataset and backend</p>"
                )
                return

            ds_dir = self._base_dir / ds_name
            zarr_path = ds_dir / backend / "feedforward.zarr"
            if not zarr_path.exists():
                self._models_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>feedforward.zarr not found in {backend}</p>"
                )
                return

            result = FeedforwardResult.load_zarr(zarr_path)
            n_pts = len(result.points)

            # Load or auto-detect ground plane
            import json
            import numpy as np
            transforms_path = ds_dir / backend / "transforms.json"
            gp_R = None
            gp_t = None
            gp_status = ""
            if transforms_path.exists():
                try:
                    data = json.loads(transforms_path.read_text())
                    gp = data.get("ground_plane")
                    if gp and "R" in gp and "t" in gp:
                        gp_R = np.array(gp["R"], dtype=np.float64)
                        gp_t = np.array(gp["t"], dtype=np.float64)
                        gp_status = "loaded from file"
                except Exception:
                    logger.warning("Could not parse transforms.json at %s", transforms_path)
            if gp_R is None:
                try:
                    from collab_splats.pointcloud.utils import fit_dominant_plane
                    gp_R, gp_t = fit_dominant_plane(result.points)
                    transforms_path.write_text(
                        json.dumps({"ground_plane": {"R": gp_R.tolist(), "t": gp_t.tolist()}}, indent=2)
                    )
                    gp_status = "auto-detected · saved"
                except Exception as exc:
                    logger.warning("Ground plane auto-detect failed: %s", exc)

            # Update state on IOLoop thread
            def _set_state() -> None:
                self._state.output_dir = ds_dir
                self._state.pointcloud_backend = backend
                self._state.semantic_extractor = extractor if extractor and extractor != "(none)" else ""
                self._state.feedforward_result = result
                self._state.ground_plane_R = gp_R
                self._state.ground_plane_t = gp_t
                if gp_status:
                    self._ground_plane_status.object = (
                        f"<p style='color:#50c050;font-size:11px'>{gp_status}</p>"
                    )
                self._models_status.object = (
                    f"<p style='color:#50c050;font-size:12px'>"
                    f"Loaded {n_pts:,} pts<br/>"
                    f"<span style='color:#666'>{ds_name} / {backend}</span></p>"
                )

            pn.io.state.execute(_set_state)
        except Exception as exc:
            logger.exception("_do_load_models failed")
            self._models_status.object = (
                f"<p style='color:#e05050;font-size:12px'>Load failed: {exc}</p>"
            )
        finally:
            self._models_load_btn.disabled = False

    def _on_ground_plane_toggle(self, event: Any) -> None:
        """Propagate ground plane enable/disable to AppState."""
        self._state.ground_plane_enabled = event.new

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

    def _do_detect_ground_plane(self) -> None:
        """Background: RANSAC ground plane detection; save transforms.json."""
        try:
            import json
            import numpy as np
            from collab_splats.pointcloud.utils import fit_dominant_plane

            result = self._state.feedforward_result
            R, t = fit_dominant_plane(result.points)

            ds_name = self._models_dataset_dd.value
            backend = self._models_backend_dd.value
            transforms_path = self._base_dir / ds_name / backend / "transforms.json"
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

        # Wire tab activation → ScenePanel mode rescan
        visualize_tab_index = list(self._panes.keys()).index("Visualize")
        self._panes["Visualize"].wire_tabs(self._tabs, visualize_tab_index)

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


def run_app(host: str = "0.0.0.0", port: int = 7860, base_dir: str = "/workspace/outputs") -> None:
    """Launch the dashboard via pn.serve()."""
    video_server = start_video_server(port=7863)

    def app_factory():
        return App(base_dir=base_dir, video_server=video_server).servable()

    pn.serve(app_factory, host=host, port=port, show=False)
