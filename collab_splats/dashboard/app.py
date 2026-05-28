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
from collab_splats.dashboard.state import AppState

logger = logging.getLogger(__name__)

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

        self._preprocess = PreprocessPane(state=self._state, op_log=self._op_log)
        self._panes = {
            "Preprocess": self._preprocess,
            "Semantics": PlaceholderPane("Semantics", "Coming in Phase 2 — 2D feature extraction and comparison"),
            "Reconstruct": PlaceholderPane("Reconstruct", "Coming in Phase 3 — run feedforward reconstruction with BA/LC"),
            "Visualize": PlaceholderPane("Visualize", "Coming in Phase 4 — interactive PyVista 3D comparison"),
            "Localize": PlaceholderPane("Localize", "Coming in Phase 5 — camera localization in known scene"),
        }

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
        self._output_dir_input = pn.widgets.TextInput(
            name="Output directory", placeholder="/workspace/outputs/birds_c0043",
            width=280, visible=False,
        )
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
            self._output_dir_input,
            self._confirm_btn,
            pn.layout.Divider(),
            self._session_status,
            width=300,
        )

    def _on_new_video(self, event: Any) -> None:
        self._video_input.visible = True
        self._output_dir_input.visible = False
        self._confirm_btn.name = "Start session"
        self._confirm_btn.visible = True

    def _on_load_existing(self, event: Any) -> None:
        self._video_input.visible = False
        self._output_dir_input.visible = True
        self._confirm_btn.name = "Load session"
        self._confirm_btn.visible = True

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
            out_dir = Path(self._output_dir_input.value.strip())
            config_file = out_dir / "run_config.yaml"
            if not config_file.exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>No run_config.yaml in {out_dir}</p>"
                )
                return
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
        self._output_dir_input.visible = False
        self._confirm_btn.visible = False

    def servable(self) -> pn.template.MaterialTemplate:
        """Build and return the full MaterialTemplate for serving."""
        pn.extension("tabulator", css_files=[], raw_css=[_HEADER_CSS])

        tabs = pn.Tabs(
            *[(name, self._panes[name].panel()) for name in self._tab_names],
            dynamic=True,
            sizing_mode="stretch_width",
        )

        main_content = pn.Column(
            tabs,
            self._op_log.panel(),
            sizing_mode="stretch_width",
        )

        return pn.template.MaterialTemplate(
            title="collab-splats",
            sidebar=[self._build_sidebar()],
            main=[main_content],
            header_background="#2596be",
            sidebar_width=320,
        )


def run_app(host: str = "0.0.0.0", port: int = 7860, base_dir: str = "/workspace/outputs") -> None:
    """Launch the dashboard via pn.serve()."""
    def app_factory():
        return App(base_dir=base_dir).servable()

    pn.serve(app_factory, host=host, port=port, show=False)
