from __future__ import annotations

from pathlib import Path
from typing import Any

import logging

import panel as pn
import param
import yaml

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes._placeholder import PlaceholderPane
from collab_splats.dashboard.panes.preprocess import PreprocessPane
from collab_splats.dashboard.panes.reconstruct import ReconstructPane
from collab_splats.dashboard.panes.semantics import SemanticsPane
from collab_splats.dashboard.panes.visualize import VisualizePane
from collab_splats.dashboard.state import AppState

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

    def __init__(self, base_dir: str = "/workspace/outputs", **params: Any):
        super().__init__(**params)
        self._base_dir = Path(base_dir)
        self._state = AppState()
        self._op_log = OperationLog()
        self._tabs: pn.Tabs | None = None

        self._preprocess = PreprocessPane(state=self._state, op_log=self._op_log)
        self._reconstruct = ReconstructPane(state=self._state, op_log=self._op_log)
        self._panes = {
            "Preprocess": self._preprocess,
            "Semantics": SemanticsPane(state=self._state, op_log=self._op_log),
            "Reconstruct": self._reconstruct,
            "Visualize": VisualizePane(state=self._state, op_log=self._op_log, base_dir=self._base_dir),
            "Localize": PlaceholderPane("Localize", "Coming in Phase 5 — camera localization in known scene"),
        }
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
                raw_vp = config.get("video_path")
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

    def servable(self) -> pn.template.MaterialTemplate:
        """Build and return the full MaterialTemplate for serving."""
        pn.extension("tabulator", "vtk", css_files=[], raw_css=[_HEADER_CSS])

        self._tabs = pn.Tabs(
            *[(name, self._panes[name].panel()) for name in self._tab_names],
            dynamic=True,
            sizing_mode="stretch_width",
        )

        # Wire tab activation → VisualizePane mode rescan
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
    def app_factory():
        return App(base_dir=base_dir).servable()

    pn.serve(app_factory, host=host, port=port, show=False)
