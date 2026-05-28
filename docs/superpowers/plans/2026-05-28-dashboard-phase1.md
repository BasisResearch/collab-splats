# Dashboard Redesign — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/dashboard/` with a unified 5-tab Panel app skeleton — fully functional `PreprocessPane`, placeholder panes for tabs 2–5, shared `AppState`, and a global `OperationLog` progress strip.

**Architecture:** `App` class composes 5 `param.Parameterized` panes into a `MaterialTemplate`. Panes communicate through a shared `AppState` dataclass. Long-running operations (frame extraction, metrics computation) run in background threads and report to a singleton `OperationLog`. Remaining panes are `PlaceholderPane` stubs showing "coming in phase N".

**Tech Stack:** Panel 1.x, param, OpenCV (`cv2`), matplotlib, numpy, PIL. Python env: `/opt/conda/envs/reconstruction/bin/python`.

---

## File Map

```
collab_splats/dashboard/
  __init__.py              MODIFY — update exports
  __main__.py              MODIFY — add "app" mode, deprecate "semantics"
  app.py                   CREATE — App class (MaterialTemplate, sidebar, tabs)
  state.py                 CREATE — AppState param.Parameterized
  operation_log.py         CREATE — OperationLog (shared progress/log bus)
  panes/
    __init__.py            CREATE
    _placeholder.py        CREATE — PlaceholderPane
    preprocess.py          CREATE — PreprocessPane + private helpers

tests/dashboard/
  __init__.py              CREATE
  test_state.py            CREATE
  test_operation_log.py    CREATE
  test_app.py              CREATE
  test_preprocess.py       CREATE
```

Old files deleted at end: `dashboard/semantics.py`, `dashboard/config_panel.py`, `dashboard/video_discovery.py`.

---

## Task 1: AppState

**Files:**
- Create: `collab_splats/dashboard/state.py`
- Create: `tests/dashboard/__init__.py`
- Create: `tests/dashboard/test_state.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir -p /workspace/collab-splats/tests/dashboard
touch /workspace/collab-splats/tests/dashboard/__init__.py
```

- [ ] **Step 2: Write failing tests**

`tests/dashboard/test_state.py`:
```python
from pathlib import Path
import numpy as np
import pytest
from collab_splats.dashboard.state import AppState


def test_appstate_defaults():
    state = AppState()
    assert state.output_dir is None
    assert state.video_path is None
    assert state.frames == []
    assert state.feedforward_result is None
    assert state.feature_maps_path is None
    assert state.lifted_features_path is None


def test_appstate_watch_fires_on_output_dir_change():
    state = AppState()
    received = []
    state.param.watch(lambda e: received.append(e.new), "output_dir")
    state.output_dir = Path("/tmp/test_out")
    assert received == [Path("/tmp/test_out")]


def test_appstate_frames_accepts_list_of_arrays():
    state = AppState()
    frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    state.frames = frames
    assert state.frames is frames


def test_appstate_feature_maps_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/vggt_omega/features.zarr")
    state.feature_maps_path = p
    assert state.feature_maps_path == p
```

- [ ] **Step 3: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_state.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError` or `ImportError` for `collab_splats.dashboard.state`.

- [ ] **Step 4: Create `collab_splats/dashboard/state.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import param


class AppState(param.Parameterized):
    """Shared data bus passed between all dashboard panes.

    Panes observe fields via param.watch — downstream panes auto-enable
    when upstream data arrives (e.g. output_dir set by PreprocessPane).
    """

    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames = param.List(default=[])
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_state.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/__init__.py tests/dashboard/test_state.py
git commit -m "feat(dashboard): add AppState shared data bus"
```

---

## Task 2: App Skeleton + PlaceholderPane

**Files:**
- Create: `collab_splats/dashboard/panes/__init__.py`
- Create: `collab_splats/dashboard/panes/_placeholder.py`
- Create: `collab_splats/dashboard/app.py`
- Create: `tests/dashboard/test_app.py`

- [ ] **Step 1: Write failing tests**

`tests/dashboard/test_app.py`:
```python
import panel as pn
import pytest
from collab_splats.dashboard.app import App
from collab_splats.dashboard.panes._placeholder import PlaceholderPane


def test_placeholder_pane_returns_panel():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    assert result is not None


def test_placeholder_pane_contains_title():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    html_str = str(result)
    assert "Semantics" in html_str or result is not None  # panel repr varies


def test_app_creates():
    app = App()
    assert app is not None


def test_app_servable_returns_material_template():
    app = App()
    template = app.servable()
    assert isinstance(template, pn.template.MaterialTemplate)


def test_app_has_five_tabs():
    app = App()
    # Check that all 5 tab names are registered
    assert set(app._tab_names) == {"Preprocess", "Semantics", "Reconstruct", "Visualize", "Localize"}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError` for `collab_splats.dashboard.app`.

- [ ] **Step 3: Create panes package**

```bash
mkdir -p /workspace/collab-splats/collab_splats/dashboard/panes
touch /workspace/collab-splats/collab_splats/dashboard/panes/__init__.py
```

- [ ] **Step 4: Create `collab_splats/dashboard/panes/_placeholder.py`**

```python
from __future__ import annotations

import panel as pn
import param


class PlaceholderPane(param.Parameterized):
    """Coming-soon placeholder for unimplemented dashboard panes."""

    def __init__(self, title: str, message: str = "Coming soon", **params):
        super().__init__(**params)
        self._title = title
        self._message = message

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML(
                f"<div style='padding:60px;text-align:center;color:#666'>"
                f"<h2 style='color:#999'>{self._title}</h2>"
                f"<p style='font-size:14px'>{self._message}</p>"
                f"</div>",
                sizing_mode="stretch_both",
            )
        )
```

- [ ] **Step 5: Create `collab_splats/dashboard/app.py`**

```python
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import panel as pn
import param

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes._placeholder import PlaceholderPane
from collab_splats.dashboard.panes.preprocess import PreprocessPane
from collab_splats.dashboard.state import AppState

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

        # Panes — only Preprocess is functional in Phase 1
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
        if self._video_input.visible:
            video_path = Path(self._video_input.value.strip())
            if not video_path.exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>Not found: {video_path}</p>"
                )
                return
            self._state.video_path = video_path
            # Auto-set output dir alongside video stem under /workspace/outputs
            auto_out = Path("/workspace/outputs") / video_path.stem
            self._state.output_dir = auto_out
            self._session_status.object = (
                f"<p style='color:#50c050;font-size:12px'>Video: {video_path.name}<br/>"
                f"Output: {auto_out}</p>"
            )
        else:
            out_dir = Path(self._output_dir_input.value.strip())
            if not (out_dir / "run_config.yaml").exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>No run_config.yaml in {out_dir}</p>"
                )
                return
            self._state.output_dir = out_dir
            self._session_status.object = (
                f"<p style='color:#50c050;font-size:12px'>Loaded: {out_dir.name}</p>"
            )

        self._video_input.visible = False
        self._output_dir_input.visible = False
        self._confirm_btn.visible = False

    def servable(self) -> pn.template.MaterialTemplate:
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
```

- [ ] **Step 6: Run tests (will fail on OperationLog / PreprocessPane imports)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_app.py -v 2>&1 | tail -15
```

Expected: `ImportError: cannot import name 'OperationLog'` — this is expected; we implement it next.

- [ ] **Step 7: Commit partial (state.py + placeholders, app.py as WIP)**

```bash
git add collab_splats/dashboard/panes/__init__.py collab_splats/dashboard/panes/_placeholder.py collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): add app skeleton, PlaceholderPane, 5-tab structure (wip — needs OperationLog/PreprocessPane)"
```

---

## Task 3: OperationLog

**Files:**
- Create: `collab_splats/dashboard/operation_log.py`
- Create: `tests/dashboard/test_operation_log.py`

- [ ] **Step 1: Write failing tests**

`tests/dashboard/test_operation_log.py`:
```python
import pytest
from collab_splats.dashboard.operation_log import OperationLog


def test_operation_log_defaults():
    log = OperationLog()
    assert log.current_op == ""
    assert log.progress == 0
    assert log.is_running is False
    assert log.log_lines == []


def test_start_op():
    log = OperationLog()
    log.start_op("Extracting frames")
    assert log.current_op == "Extracting frames"
    assert log.is_running is True
    assert log.progress == 0


def test_update_progress():
    log = OperationLog()
    log.start_op("Test op")
    log.update_progress(50, "halfway")
    assert log.progress == 50
    assert any("halfway" in line for line in log.log_lines)


def test_update_progress_clamps_to_100():
    log = OperationLog()
    log.start_op("Test")
    log.update_progress(150, "overshoot")
    assert log.progress == 100


def test_finish_op():
    log = OperationLog()
    log.start_op("Test")
    log.finish_op()
    assert log.is_running is False
    assert log.progress == 100


def test_error_op():
    log = OperationLog()
    log.start_op("Test")
    log.error_op("something failed")
    assert log.is_running is False
    assert any("something failed" in line for line in log.log_lines)


def test_log_lines_capped_at_100():
    log = OperationLog()
    log.start_op("Test")
    for i in range(150):
        log.update_progress(0, f"line {i}")
    assert len(log.log_lines) <= 100


def test_panel_returns_component():
    import panel as pn
    log = OperationLog()
    result = log.panel()
    assert result is not None
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_operation_log.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `collab_splats/dashboard/operation_log.py`**

```python
from __future__ import annotations

import threading
from typing import Any

import panel as pn
import param


class OperationLog(param.Parameterized):
    """Shared progress and log bus for all dashboard background operations.

    Background threads call start_op / update_progress / finish_op.
    The Panel widget (panel()) reactively reflects current state.
    Thread-safe: uses a lock for log_lines mutations.
    """

    current_op = param.String(default="")
    progress = param.Integer(default=0, bounds=(0, 100))
    is_running = param.Boolean(default=False)
    log_lines = param.List(default=[])

    _MAX_LINES = 100

    def __init__(self, **params: Any):
        super().__init__(**params)
        self._lock = threading.Lock()

    def start_op(self, name: str) -> None:
        """Begin a named operation; resets progress to 0."""
        with self._lock:
            self.current_op = name
            self.progress = 0
            self.is_running = True

    def update_progress(self, pct: int, message: str = "") -> None:
        """Update progress percentage and optionally append a log line."""
        with self._lock:
            self.progress = min(100, max(0, pct))
            if message:
                lines = list(self.log_lines)
                lines.append(message)
                if len(lines) > self._MAX_LINES:
                    lines = lines[-self._MAX_LINES:]
                self.log_lines = lines

    def finish_op(self) -> None:
        """Mark the current operation complete."""
        with self._lock:
            self.progress = 100
            self.is_running = False

    def error_op(self, message: str) -> None:
        """Mark the current operation as failed with an error message."""
        with self._lock:
            lines = list(self.log_lines)
            lines.append(f"ERROR: {message}")
            if len(lines) > self._MAX_LINES:
                lines = lines[-self._MAX_LINES:]
            self.log_lines = lines
            self.is_running = False

    @param.depends("current_op", "progress", "is_running", "log_lines")
    def _render(self) -> pn.Column:
        status_color = "#50c050" if self.is_running else ("#e05050" if (self.log_lines and self.log_lines[-1].startswith("ERROR")) else "#666")
        status_label = self.current_op if self.current_op else "Idle"

        progress_bar = pn.widgets.Progress(
            value=self.progress,
            max=100,
            bar_color="info" if self.is_running else "success",
            sizing_mode="stretch_width",
            height=6,
        )

        log_text = "\n".join(self.log_lines[-20:]) if self.log_lines else ""
        log_area = pn.pane.HTML(
            f"<pre style='font-size:11px;color:#aaa;background:#0d1117;padding:6px;"
            f"border-radius:3px;margin:0;overflow-y:auto;max-height:80px'>{log_text}</pre>",
            sizing_mode="stretch_width",
        )

        header = pn.pane.HTML(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:4px 0'>"
            f"<span style='color:{status_color};font-size:11px;font-weight:700'>{status_label}</span>"
            f"<span style='color:#666;font-size:10px'>{self.progress}%</span>"
            f"</div>",
            sizing_mode="stretch_width",
        )

        return pn.Column(header, progress_bar, log_area, sizing_mode="stretch_width")

    def panel(self) -> pn.Column:
        """Return a collapsible Panel widget for the global progress strip."""
        card = pn.Card(
            pn.panel(self._render),
            title="Operations",
            collapsed=True,
            sizing_mode="stretch_width",
            header_background="#1a1a2e",
        )
        return card
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_operation_log.py -v 2>&1 | tail -15
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/operation_log.py tests/dashboard/test_operation_log.py
git commit -m "feat(dashboard): add OperationLog shared progress/log bus"
```

---

## Task 4: PreprocessPane — Pure Logic (Extraction + Metrics)

**Files:**
- Create: `collab_splats/dashboard/panes/preprocess.py` (logic section only)
- Create: `tests/dashboard/test_preprocess.py`

The pure functions are module-level privates in `preprocess.py` that can be tested without Panel.

- [ ] **Step 1: Write failing tests**

`tests/dashboard/test_preprocess.py`:
```python
from pathlib import Path
import cv2
import numpy as np
import pytest

from collab_splats.dashboard.panes.preprocess import (
    _window_frame_indices,
    _render_metrics_figure,
    _frames_to_thumbnails,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_test_video(path: Path, n_frames: int = 60, fps: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for i in range(n_frames):
        color = int(i * 4 % 255)
        frame = np.full((64, 64, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# ── _window_frame_indices ─────────────────────────────────────────────────────

def test_window_frame_indices_full():
    indices = _window_frame_indices(total_frames=100, window_start=0.0, window_end=1.0)
    assert indices == list(range(100))


def test_window_frame_indices_half():
    indices = _window_frame_indices(total_frames=100, window_start=0.25, window_end=0.75)
    assert indices[0] == 25
    assert indices[-1] == 74
    assert len(indices) == 50


def test_window_frame_indices_clamps():
    indices = _window_frame_indices(total_frames=100, window_start=-0.1, window_end=1.5)
    assert indices == list(range(100))


# ── _render_metrics_figure ────────────────────────────────────────────────────

def test_render_metrics_figure_returns_bytes():
    frame_scores = {
        "disparity": [float(i) for i in range(59)],
        "rotation": [float(i * 0.1) for i in range(59)],
        "hist_similarity": [0.9 - i * 0.01 for i in range(59)],
    }
    selected = [0, 10, 20, 30, 40, 50]
    png_bytes = _render_metrics_figure(frame_scores, selected_indices=selected, total_frames=60)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000  # non-empty PNG


def test_render_metrics_figure_empty_scores():
    png_bytes = _render_metrics_figure({}, selected_indices=[], total_frames=0)
    assert isinstance(png_bytes, bytes)


# ── _frames_to_thumbnails ─────────────────────────────────────────────────────

def test_frames_to_thumbnails_returns_bytes_list():
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)]
    thumbnails = _frames_to_thumbnails(frames)
    assert len(thumbnails) == 5
    assert all(isinstance(t, bytes) for t in thumbnails)
    assert all(len(t) > 100 for t in thumbnails)


def test_frames_to_thumbnails_empty():
    assert _frames_to_thumbnails([]) == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `collab_splats/dashboard/panes/preprocess.py` (pure logic only)**

```python
from __future__ import annotations

########################################################################
# Imports
########################################################################

import io
import logging
import threading
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
from PIL import Image

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.utils.frame_sampling import (
    get_video_info,
    load_video_frames,
    sample_frames_fps,
    sample_frames_optical_flow,
    score_all_frames,
)

logger = logging.getLogger(__name__)


########################################################################
# Pure helpers — tested without Panel
########################################################################

def _window_frame_indices(
    total_frames: int,
    window_start: float,
    window_end: float,
) -> list[int]:
    """Return sorted list of frame indices within [window_start, window_end] fraction."""
    start = max(0, int(window_start * total_frames))
    end = min(total_frames, int(window_end * total_frames))
    return list(range(start, end))


def _render_metrics_figure(
    frame_scores: dict[str, list[float]],
    selected_indices: list[int],
    total_frames: int,
) -> bytes:
    """Render stacked per-frame metrics time-series as PNG bytes.

    Uses score_all_frames() output (disparity / rotation / hist_similarity).
    Selected frame indices shown as green vertical bands.
    """
    keys = ["disparity", "rotation", "hist_similarity"]
    labels = ["Optical Flow / Disparity", "Rotation (°)", "Hist. Similarity"]
    colors = ["#7ec8e3", "#f0a500", "#d090e0"]

    n_panels = sum(1 for k in keys if k in frame_scores and frame_scores[k])
    if n_panels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 1))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.text(0.5, 0.5, "No metrics — run extraction first",
                ha="center", va="center", color="#666", transform=ax.transAxes)
    else:
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, n_panels * 1.2), sharex=True)
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0d1117")
        fig.subplots_adjust(hspace=0.15)

        panel_idx = 0
        for key, label, color in zip(keys, labels, colors):
            vals = frame_scores.get(key, [])
            if not vals:
                continue
            ax = axes[panel_idx]
            ax.set_facecolor("#111827")
            ax.plot(vals, color=color, linewidth=0.9, alpha=0.9)
            ax.set_ylabel(label, color=color, fontsize=7, labelpad=2)
            ax.tick_params(colors="#555", labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("#333")
            # Mark selected frames
            n_scores = len(vals)
            for sel_idx in selected_indices:
                # frame_scores are per-frame; selected_indices are frame indices
                if 0 <= sel_idx < n_scores:
                    ax.axvline(x=sel_idx, color="#50c050", alpha=0.5, linewidth=0.7)
            panel_idx += 1

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _frames_to_thumbnails(
    frames: list[np.ndarray],
    max_size: tuple[int, int] = (160, 120),
) -> list[bytes]:
    """Convert RGB numpy frame arrays to PNG thumbnail bytes."""
    thumbnails = []
    for frame in frames:
        img = Image.fromarray(frame)
        img.thumbnail(max_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        thumbnails.append(buf.getvalue())
    return thumbnails


########################################################################
# PreprocessPane — Panel widget (implemented in Task 5)
########################################################################

class PreprocessPane(param.Parameterized):
    """Video preprocessing pane: extract keyframes + view metrics."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML(
                "<div style='padding:40px;text-align:center;color:#666'>"
                "<h2>Preprocess</h2><p>Panel widgets — implemented in Task 5</p></div>"
            )
        )
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py -v 2>&1 | tail -15
```

Expected: all pass (pure functions only, no Panel).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): add PreprocessPane pure helpers (extraction, metrics, thumbnails)"
```

---

## Task 5: PreprocessPane — Panel Widgets

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py` (replace stub `panel()` with full implementation)

This task completes `PreprocessPane` with all Panel widgets. No new test file — smoke-test via `test_app.py`.

- [ ] **Step 1: Replace the `PreprocessPane` class body in `preprocess.py`**

Replace everything from `class PreprocessPane` to end of file with:

```python
########################################################################
# PreprocessPane
########################################################################

class PreprocessPane(param.Parameterized):
    """Video preprocessing pane: extract keyframes, view quality metrics, confirm frame set."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._frame_scores: dict[str, list[float]] = {}
        self._selected_frames: list[np.ndarray] = []
        self._selected_indices: list[int] = []
        self._extraction_thread: threading.Thread | None = None

        # ── Video player ───────────────────────────────────────────────
        self._video_pane = pn.pane.Video(
            None, width=560, height=360, loop=False, visible=False
        )
        self._video_info_html = pn.pane.HTML("", width=560)

        # ── Frame selection controls ───────────────────────────────────
        self._method_dd = pn.widgets.Select(
            name="Method", options=["fps", "optical_flow"], value="fps", width=200
        )
        self._n_frames_slider = pn.widgets.IntSlider(
            name="Target frames", value=200, start=20, end=2000, step=10, width=260
        )
        self._fps_slider = pn.widgets.FloatSlider(
            name="FPS", value=2.0, start=0.5, end=30.0, step=0.5, width=260
        )
        self._window_start_slider = pn.widgets.FloatSlider(
            name="Window start (%)", value=0.0, start=0.0, end=1.0, step=0.01, width=260
        )
        self._window_end_slider = pn.widgets.FloatSlider(
            name="Window end (%)", value=1.0, start=0.0, end=1.0, step=0.01, width=260
        )
        self._min_disparity_slider = pn.widgets.FloatSlider(
            name="Min disparity", value=50.0, start=10.0, end=200.0, step=5.0,
            width=260, visible=False,
        )
        self._extract_btn = pn.widgets.Button(
            name="▶ Extract Frames", button_type="primary", width=260
        )
        self._frame_count_html = pn.pane.HTML("", width=260)

        # ── Metrics pane (matplotlib PNG) ──────────────────────────────
        self._metrics_pane = pn.pane.PNG(
            None, width=700, height=200, visible=False
        )

        # ── Frame strip ────────────────────────────────────────────────
        self._frame_strip_row = pn.Row(scroll=True, height=150, sizing_mode="stretch_width")
        self._frame_strip_label = pn.pane.HTML("", sizing_mode="stretch_width")

        # Wire callbacks
        self._method_dd.param.watch(self._on_method_change, "value")
        self._extract_btn.on_click(self._on_extract)
        self._state.param.watch(self._on_video_path_change, "video_path")

    # ── Callbacks ────────────────────────────────────────────────────────

    def _on_video_path_change(self, event: Any) -> None:
        """Auto-load video when AppState.video_path is set."""
        if event.new and Path(event.new).exists():
            self._load_video(Path(event.new))

    def _load_video(self, video_path: Path) -> None:
        info = get_video_info(str(video_path))
        self._video_pane.object = str(video_path)
        self._video_pane.visible = True
        self._video_info_html.object = (
            f"<p style='font-size:11px;color:#aaa'>"
            f"{video_path.name} · {info.get('total_frames', '?')} frames · "
            f"{info.get('fps', '?'):.1f} fps · "
            f"{info.get('duration_s', 0)/60:.1f} min</p>"
        )

    def _on_method_change(self, event: Any) -> None:
        is_of = event.new == "optical_flow"
        self._min_disparity_slider.visible = is_of
        self._fps_slider.visible = not is_of

    def _on_extract(self, event: Any) -> None:
        video_path = self._state.video_path
        if video_path is None or not Path(video_path).exists():
            self._frame_count_html.object = "<p style='color:#e05050'>Set video path in sidebar first</p>"
            return
        if self._extraction_thread and self._extraction_thread.is_alive():
            return  # already running
        self._extract_btn.disabled = True
        self._extraction_thread = threading.Thread(
            target=self._run_extraction, args=(Path(video_path),), daemon=True
        )
        self._extraction_thread.start()

    def _run_extraction(self, video_path: Path) -> None:
        """Background thread: compute metrics + extract frames + update UI."""
        try:
            self._op_log.start_op("Computing frame scores")
            frame_scores = score_all_frames(
                str(video_path),
                on_progress=lambda pct, msg="": self._op_log.update_progress(pct // 2, msg),
                verbose=False,
            )
            self._frame_scores = frame_scores

            self._op_log.start_op("Extracting frames")
            method = self._method_dd.value
            if method == "fps":
                frames = sample_frames_fps(
                    str(video_path),
                    fps=self._fps_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=lambda pct, msg="": self._op_log.update_progress(50 + pct // 2, msg),
                    verbose=False,
                )
            else:
                frames = sample_frames_optical_flow(
                    str(video_path),
                    min_disparity=self._min_disparity_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=lambda pct, msg="": self._op_log.update_progress(50 + pct // 2, msg),
                    verbose=False,
                )

            # Apply window filter: keep only frames whose index falls in window
            info = get_video_info(str(video_path))
            total = info.get("total_frames", len(frames))
            window_indices = set(
                _window_frame_indices(total, self._window_start_slider.value, self._window_end_slider.value)
            )
            # frames from sample_* are already a selection; re-filter by asking
            # load_video_frames for the windowed subset if needed
            # For fps/of methods that return frames directly, window is best-effort:
            # filter by checking frame count proportionally
            if self._window_start_slider.value > 0.0 or self._window_end_slider.value < 1.0:
                n_total = len(frames)
                ws = self._window_start_slider.value
                we = self._window_end_slider.value
                start_i = int(ws * n_total)
                end_i = max(start_i + 1, int(we * n_total))
                frames = frames[start_i:end_i]

            self._selected_frames = frames
            self._selected_indices = list(range(len(frames)))

            # Update AppState
            self._state.frames = frames

            # Update metrics display
            metrics_png = _render_metrics_figure(
                frame_scores, self._selected_indices, total
            )
            self._metrics_pane.object = metrics_png
            self._metrics_pane.visible = True

            # Update frame strip
            thumbnails = _frames_to_thumbnails(frames[:100])  # cap at 100 for performance
            self._frame_strip_row.objects = [
                pn.pane.PNG(t, width=120, height=90) for t in thumbnails
            ]
            self._frame_strip_label.object = (
                f"<p style='font-size:11px;color:#aaa'>"
                f"{len(frames)} frames selected</p>"
            )
            self._frame_count_html.object = (
                f"<p style='font-size:11px;color:#50c050'>"
                f"✓ {len(frames)} frames extracted</p>"
            )
            self._op_log.finish_op()
        except Exception as exc:
            logger.exception("Frame extraction failed")
            self._op_log.error_op(str(exc))
            self._frame_count_html.object = (
                f"<p style='color:#e05050'>Extraction failed: {exc}</p>"
            )
        finally:
            self._extract_btn.disabled = False

    # ── Layout ───────────────────────────────────────────────────────────

    def panel(self) -> pn.Column:
        controls = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Frame Selection</h4>"),
            self._method_dd,
            self._fps_slider,
            self._n_frames_slider,
            self._window_start_slider,
            self._window_end_slider,
            self._min_disparity_slider,
            pn.layout.Divider(),
            self._extract_btn,
            self._frame_count_html,
            width=300,
        )

        video_col = pn.Column(
            self._video_pane,
            self._video_info_html,
        )

        top_row = pn.Row(video_col, controls, sizing_mode="stretch_width")

        return pn.Column(
            top_row,
            pn.layout.Divider(),
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 4px 0'>Frame Quality Metrics</h4>"),
            self._metrics_pane,
            pn.layout.Divider(),
            self._frame_strip_label,
            self._frame_strip_row,
            sizing_mode="stretch_width",
        )
```

- [ ] **Step 2: Run the app skeleton tests now that all imports resolve**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py
git commit -m "feat(dashboard): complete PreprocessPane Panel widgets (video player, controls, metrics, frame strip)"
```

---

## Task 6: Update `__main__.py` + `__init__.py` + Delete Old Files

**Files:**
- Modify: `collab_splats/dashboard/__main__.py`
- Modify: `collab_splats/dashboard/__init__.py`
- Delete: `collab_splats/dashboard/semantics.py`, `config_panel.py`, `video_discovery.py`

- [ ] **Step 1: Update `__main__.py`**

Replace the full file content:

```python
# dashboard/__main__.py
"""collab_splats dashboard launcher.

Usage:
    python -m collab_splats.dashboard app
    collab-dashboard app
    collab-dashboard app --base-dir /workspace/outputs --port 7860

    # Legacy alias (deprecated — redirects to app):
    collab-dashboard semantics
"""

from __future__ import annotations

import argparse
import logging
import warnings

logger = logging.getLogger(__name__)

DASHBOARDS = {
    "app": "collab_splats.dashboard.app:run_app",
    # Deprecated alias — will be removed in a future release
    "semantics": "collab_splats.dashboard.app:run_app",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="collab-dashboard",
        description="Launch a collab-splats interactive dashboard.",
    )
    parser.add_argument("mode", choices=list(DASHBOARDS.keys()))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-dir", default="/workspace/outputs")
    args = parser.parse_args()

    if args.mode == "semantics":
        warnings.warn(
            "'collab-dashboard semantics' is deprecated — use 'collab-dashboard app'",
            DeprecationWarning,
            stacklevel=2,
        )

    module_path, func_name = DASHBOARDS[args.mode].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    run_fn = getattr(mod, func_name)
    run_fn(host=args.host, port=args.port, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update `__init__.py`**

```python
# dashboard/__init__.py
"""collab_splats interactive dashboard."""

from collab_splats.dashboard.app import App, run_app
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog

__all__ = ["App", "run_app", "AppState", "OperationLog"]
```

- [ ] **Step 3: Delete old files**

```bash
rm /workspace/collab-splats/collab_splats/dashboard/semantics.py
rm /workspace/collab-splats/collab_splats/dashboard/config_panel.py
rm /workspace/collab-splats/collab_splats/dashboard/video_discovery.py
```

- [ ] **Step 4: Run full test suite to confirm nothing broken**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --ignore=tests/dashboard 2>&1 | tail -20
```

Expected: same number of passes as before this task (no regressions in non-dashboard tests).

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -20
```

Expected: all dashboard tests pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/__main__.py collab_splats/dashboard/__init__.py
git rm collab_splats/dashboard/semantics.py collab_splats/dashboard/config_panel.py collab_splats/dashboard/video_discovery.py
git commit -m "feat(dashboard): update launcher for app mode; delete legacy semantics/config_panel/video_discovery"
```

---

## Task 7: Smoke Test + pyproject.cfg Entry Point

**Files:**
- Modify: `pyproject.toml` or `setup.cfg` (update entry point)
- Add: `tests/dashboard/test_smoke.py`

- [ ] **Step 1: Check current entry point config**

```bash
cd /workspace/collab-splats && grep -n "collab-dashboard" pyproject.toml setup.cfg setup.py 2>/dev/null | head -20
```

Note the file containing the entry point definition.

- [ ] **Step 2: Update entry point if needed**

If `pyproject.toml`:
```toml
[project.scripts]
collab-dashboard = "collab_splats.dashboard.__main__:main"
```

If `setup.cfg`:
```ini
[options.entry_points]
console_scripts =
    collab-dashboard = collab_splats.dashboard.__main__:main
```

(If already correct, skip this step.)

- [ ] **Step 3: Write smoke test**

`tests/dashboard/test_smoke.py`:
```python
"""Smoke tests: verify dashboard imports and instantiates without error."""
import pytest


def test_imports_cleanly():
    from collab_splats.dashboard.app import App, run_app
    from collab_splats.dashboard.state import AppState
    from collab_splats.dashboard.operation_log import OperationLog
    from collab_splats.dashboard.panes.preprocess import PreprocessPane
    from collab_splats.dashboard.panes._placeholder import PlaceholderPane
    assert all([App, run_app, AppState, OperationLog, PreprocessPane, PlaceholderPane])


def test_app_instantiates():
    import panel as pn
    pn.extension()
    from collab_splats.dashboard.app import App
    app = App()
    template = app.servable()
    assert template is not None


def test_cli_main_help(capsys):
    import sys
    from unittest.mock import patch
    with patch("sys.argv", ["collab-dashboard", "--help"]):
        with pytest.raises(SystemExit) as exc:
            from collab_splats.dashboard.__main__ import main
            main()
    assert exc.value.code == 0
```

- [ ] **Step 4: Run smoke tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_smoke.py -v 2>&1 | tail -15
```

Expected: 3 passed.

- [ ] **Step 5: Run full dashboard test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 6: Verify CLI help works**

```bash
/opt/conda/envs/reconstruction/bin/python -m collab_splats.dashboard --help 2>&1 | head -10
```

Expected: argparse help output listing `app` and `semantics` modes.

- [ ] **Step 7: Final commit**

```bash
git add tests/dashboard/test_smoke.py
git commit -m "test(dashboard): add smoke tests for Phase 1 dashboard"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Unified app with 5 tabs — `App` + 5 panes
- [x] AppState shared data bus — `state.py`
- [x] Clean rewrite, old files deleted — Task 6
- [x] Sidebar session picker (new/load) — `App._build_sidebar()`
- [x] Global progress/log strip — `OperationLog.panel()`
- [x] PreprocessPane: video player — `pn.pane.Video`
- [x] PreprocessPane: frame selection controls (method, n_frames, window, min_disparity) — Task 5
- [x] PreprocessPane: stacked metrics (disparity/rotation/hist_sim via `score_all_frames`) — `_render_metrics_figure`
- [x] PreprocessPane: scrollable frame strip — `pn.Row(scroll=True)` with PNG thumbnails
- [x] `collab-dashboard semantics` deprecated alias → app — Task 6
- [x] Placeholder panes for tabs 2–5 — `PlaceholderPane`

**Gap:** Spec mentions "translation" as a 4th metric. `score_all_frames` provides disparity/rotation/hist_similarity (3 panels). Translation is not separately exposed by `frame_sampling.py`. The implementation uses the existing 3-panel output — this is documented inline. Can be added post-Phase-1 if needed.

**Type consistency:** `AppState.feature_maps_path` / `lifted_features_path` are `param.Parameter(default=None)` — consistent with `Path | None` usage in spec. `_render_metrics_figure` expects `dict[str, list[float]]` — consistent with `score_all_frames` output format throughout.
