# ReconstructPane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `ReconstructPane` (Tab 3) — backend picker, conf_threshold slider, BA/LC stubs, background reconstruction thread, log drain, state wiring.

**Architecture:** Single `param.Parameterized` class following the `PreprocessPane` pattern. Background thread calls feedforward creator directly (in-process), drains log lines via `pn.state.add_periodic_callback`. BA and LC rendered as permanently-disabled stubs.

**Tech Stack:** Panel, param, threading, `VGGTXCreator` / `MapAnythingCreator` / `VGGTOmegaCreator`, `FeedforwardResult`, `OperationLog`, `AppState`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `collab_splats/dashboard/panes/reconstruct.py` | Full `ReconstructPane` implementation |
| Create | `tests/dashboard/test_reconstruct_pane.py` | All pane tests |
| Modify | `collab_splats/dashboard/app.py` | Wire `ReconstructPane` into Tab 3 |

---

## Context you need to read before starting

- `collab_splats/dashboard/panes/preprocess.py` — exact `__init__` / `panel()` / thread pattern to follow
- `collab_splats/dashboard/state.py` — `AppState` fields
- `collab_splats/dashboard/operation_log.py` — `OperationLog.start_op / update_progress / finish_op / error_op`
- `collab_splats/dashboard/app.py` — where to wire in the new pane
- `collab_splats/pointcloud/feedforward/__init__.py` — public creator exports

**Python env:** always use `/opt/conda/envs/nerfstudio/bin/python` (py3.11). Run tests as:
```
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v
```

---

## Task 1: Test skeleton + gate logic

**Files:**
- Create: `tests/dashboard/test_reconstruct_pane.py`
- Create: `collab_splats/dashboard/panes/reconstruct.py` (minimal skeleton)

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_reconstruct_pane.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import panel as pn
import pytest

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.reconstruct import ReconstructPane
from collab_splats.dashboard.state import AppState

pn.extension()  # required for widget construction


def _make_pane():
    state = AppState()
    op_log = OperationLog()
    return ReconstructPane(state=state, op_log=op_log), state, op_log


def test_run_btn_disabled_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._run_btn.disabled is True


def test_run_btn_enabled_after_output_dir_set(tmp_path):
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert pane._run_btn.disabled is False


def test_run_btn_disabled_again_when_output_dir_cleared(tmp_path):
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    state.output_dir = None
    assert pane._run_btn.disabled is True


def test_ba_toggle_disabled():
    pane, _, _ = _make_pane()
    assert pane._ba_toggle.disabled is True


def test_lc_toggle_disabled():
    pane, _, _ = _make_pane()
    assert pane._lc_toggle.disabled is True


def test_backend_dd_options():
    pane, _, _ = _make_pane()
    assert set(pane._backend_dd.options) == {"vggtx", "mapanything", "vggt_omega"}


def test_conf_slider_default():
    pane, _, _ = _make_pane()
    assert pane._conf_slider.value == 35.0
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` — `reconstruct` module doesn't exist yet.

- [ ] **Step 3: Create skeleton reconstruct.py**

Create `collab_splats/dashboard/panes/reconstruct.py`:

```python
"""ReconstructPane — Tab 3 of the unified dashboard.

Runs feedforward reconstruction (VGGTXCreator / MapAnythingCreator / VGGTOmegaCreator)
in a background thread against frames already written by PreprocessPane.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import panel as pn
import param

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator

logger = logging.getLogger(__name__)

########################################################################
# ReconstructPane
########################################################################


class ReconstructPane(param.Parameterized):
    """Feedforward reconstruction pane: pick backend, run pipeline, stream logs."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._recon_thread: threading.Thread | None = None
        self._log_lines: list[str] = []
        self._log_lock = threading.Lock()

        # Config widgets
        self._backend_dd = pn.widgets.Select(
            name="Backend",
            options=["vggtx", "mapanything", "vggt_omega"],
            value="vggtx",
            width=260,
        )
        self._conf_slider = pn.widgets.FloatSlider(
            name="Conf threshold", start=0.0, end=100.0, step=1.0, value=35.0, width=260
        )

        # BA stub (disabled — not yet implemented)
        self._ba_toggle = pn.widgets.Toggle(
            name="Bundle Adjustment", disabled=True, width=260
        )
        self._ba_stub_html = pn.pane.HTML(
            "<span style='color:#e0a020;font-size:12px'>⚠ Not yet implemented</span>",
            width=260,
        )

        # LC stub (disabled — not yet implemented)
        self._lc_toggle = pn.widgets.Toggle(
            name="Loop Closure", disabled=True, width=260
        )
        self._lc_stub_html = pn.pane.HTML(
            "<span style='color:#e0a020;font-size:12px'>⚠ Not yet implemented</span>",
            width=260,
        )

        # Run button + status
        self._run_btn = pn.widgets.Button(
            name="Run Reconstruction", button_type="primary", disabled=True, width=260
        )
        self._status_html = pn.pane.HTML("", width=260)

        # Log area (readonly via disabled=True)
        self._log_area = pn.widgets.TextAreaInput(
            value="",
            disabled=True,
            height=500,
            width=500,
            placeholder="Reconstruction log will appear here…",
        )

        # Wire callbacks
        self._run_btn.on_click(self._on_run)
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])

        # Initialize gate state
        self._on_output_dir_changed(None)

    def _on_output_dir_changed(self, event: Any) -> None:
        """Enable Run button when output_dir is set."""
        self._run_btn.disabled = self._state.output_dir is None

    def _on_run(self, event: Any) -> None:
        """Spawn background reconstruction thread on button click."""
        if self._recon_thread and self._recon_thread.is_alive():
            return
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Running…</span>"
        self._log_area.value = ""
        self._recon_thread = threading.Thread(
            target=self._run_reconstruction, daemon=True
        )
        self._recon_thread.start()

    def _run_reconstruction(self) -> None:
        """Background thread: validate, run creator, save zarr, set state."""
        try:
            output_dir = Path(self._state.output_dir)
            backend = self._backend_dd.value
            conf = self._conf_slider.value
            images_dir = output_dir / "frames"

            # Validate frames directory written by PreprocessPane
            if not images_dir.exists():
                msg = f"frames dir not found: {images_dir}. Run Preprocess first."
                self._op_log.error_op(msg)
                self._append_log(f"ERROR: {msg}")
                self._finish(success=False)
                return

            self._op_log.start_op("Running reconstruction")
            self._append_log(f"Backend: {backend}  conf_threshold: {conf}")

            creator = self._build_creator(backend, conf)
            backend_dir = output_dir / backend

            self._append_log(f"Starting {backend} inference…")
            creator.reconstruct(images_dir, backend_dir)

            ff = creator.outputs
            if ff is None:
                raise RuntimeError("Creator produced no outputs after reconstruct()")

            zarr_path = backend_dir / "feedforward.zarr"
            self._append_log(f"Saving {zarr_path}")
            ff.save_zarr(zarr_path)

            self._state.feedforward_result = ff
            self._op_log.finish_op()
            self._append_log(f"Done. {len(ff.points):,} points.")
            self._finish(success=True)

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._op_log.error_op(str(exc))
            self._append_log(f"ERROR: {exc}\n{tb}")
            self._finish(success=False)

    def _build_creator(self, backend: str, conf: float) -> Any:
        """Instantiate the appropriate feedforward creator for the selected backend."""
        if backend == "vggtx":
            return VGGTXCreator(conf_threshold=conf)
        if backend == "mapanything":
            return MapAnythingCreator(confidence_percentile=conf)
        if backend == "vggt_omega":
            try:
                from collab_splats.pointcloud.feedforward import VGGTOmegaCreator
            except ImportError as exc:
                raise ImportError(
                    "vggt_omega requires the VGGT-Omega submodule. "
                    "Run setup_feedforward.sh with the submodule initialized."
                ) from exc
            return VGGTOmegaCreator(conf_threshold=conf)
        raise ValueError(f"Unknown backend: {backend!r}")

    def _append_log(self, line: str) -> None:
        """Thread-safe: queue a log line for drain into _log_area."""
        with self._log_lock:
            self._log_lines.append(line)

    def _drain_log(self) -> None:
        """Periodic callback (500ms): flush queued log lines into _log_area."""
        with self._log_lock:
            lines = self._log_lines[:]
            self._log_lines.clear()
        if lines:
            self._log_area.value = (self._log_area.value or "") + "\n".join(lines) + "\n"

    def _finish(self, *, success: bool) -> None:
        """Update status HTML and re-enable Run button."""
        if success:
            self._status_html.object = "<span style='color:#50e050'>✓ Complete</span>"
        else:
            self._status_html.object = "<span style='color:#e05050'>✗ Failed</span>"
        self._run_btn.disabled = False

    def panel(self) -> pn.viewable.Viewable:
        """Return the full ReconstructPane Panel layout."""
        pn.state.add_periodic_callback(self._drain_log, period=500)

        config_col = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Reconstruction Config</h4>"),
            self._backend_dd,
            self._conf_slider,
            pn.layout.Divider(),
            self._ba_toggle,
            self._ba_stub_html,
            pn.layout.Divider(),
            self._lc_toggle,
            self._lc_stub_html,
            pn.layout.Divider(),
            self._run_btn,
            self._status_html,
            width=380,
        )

        log_col = pn.Column(
            pn.pane.HTML("<h4 style='color:#7ec8e3;margin:0 0 6px 0'>Log</h4>"),
            self._log_area,
        )

        return pn.Row(config_col, log_col, sizing_mode="stretch_width")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v -k "test_run_btn or test_ba or test_lc or test_backend or test_conf" 2>&1 | tail -20
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/reconstruct.py tests/dashboard/test_reconstruct_pane.py
git commit -m "feat(dashboard): ReconstructPane skeleton + gate logic"
```

---

## Task 2: Background thread — validation and happy path

**Files:**
- Modify: `tests/dashboard/test_reconstruct_pane.py`
- (reconstruct.py already complete from Task 1)

- [ ] **Step 1: Add thread tests**

Append to `tests/dashboard/test_reconstruct_pane.py`:

```python
def test_run_reconstruction_errors_when_frames_dir_missing(tmp_path):
    # output_dir exists but frames/ subdir does not
    pane, state, op_log = _make_pane()
    state.output_dir = tmp_path
    pane._run_reconstruction()
    assert op_log.is_running is False
    assert any("frames dir not found" in line for line in pane._log_lines)


def test_run_reconstruction_sets_feedforward_result(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    mock_ff = MagicMock()
    mock_ff.points = list(range(100))

    mock_creator = MagicMock()
    mock_creator.outputs = mock_ff

    pane, state, op_log = _make_pane()
    state.output_dir = tmp_path

    with patch.object(pane, "_build_creator", return_value=mock_creator):
        pane._run_reconstruction()

    assert state.feedforward_result is mock_ff
    mock_ff.save_zarr.assert_called_once_with(tmp_path / "vggtx" / "feedforward.zarr")
    assert op_log.progress == 100
    assert op_log.is_running is False


def test_run_reconstruction_error_path(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    mock_creator = MagicMock()
    mock_creator.outputs = None  # simulate missing outputs

    pane, state, op_log = _make_pane()
    state.output_dir = tmp_path

    with patch.object(pane, "_build_creator", return_value=mock_creator):
        pane._run_reconstruction()

    assert op_log.is_running is False
    assert state.feedforward_result is None
    assert any("ERROR" in line for line in pane._log_lines)


def test_run_reconstruction_re_enables_run_btn_on_success(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    mock_ff = MagicMock()
    mock_ff.points = [1]
    mock_creator = MagicMock()
    mock_creator.outputs = mock_ff

    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    pane._run_btn.disabled = True

    with patch.object(pane, "_build_creator", return_value=mock_creator):
        pane._run_reconstruction()

    assert pane._run_btn.disabled is False


def test_run_reconstruction_re_enables_run_btn_on_failure(tmp_path):
    # frames/ missing → error path → button re-enabled
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    pane._run_btn.disabled = True
    pane._run_reconstruction()
    assert pane._run_btn.disabled is False
```

- [ ] **Step 2: Run to verify new tests fail or pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v -k "reconstruction" 2>&1 | tail -25
```

Note: `test_run_reconstruction_errors_when_frames_dir_missing` will fail because `pane._log_lines` is checked after the method appends to it — this is fine, `_run_reconstruction` is synchronous in tests (no thread spawn, called directly). Other tests should pass.

If `test_run_reconstruction_errors_when_frames_dir_missing` fails due to log buffering: the method calls `_append_log` which puts lines into `_log_lines` under a lock, then `_finish` is called. Verify `pane._log_lines` contains the expected message after direct call.

- [ ] **Step 3: Fix test if needed**

If `pane._log_lines` is empty because lines were drained: note that `_drain_log` is only called by the periodic callback (not in tests). So `_log_lines` will contain all lines after `_run_reconstruction()` returns. The test should pass as written.

- [ ] **Step 4: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/dashboard/test_reconstruct_pane.py
git commit -m "test(dashboard): ReconstructPane thread + happy/error path coverage"
```

---

## Task 3: _build_creator and log drain

**Files:**
- Modify: `tests/dashboard/test_reconstruct_pane.py`

- [ ] **Step 1: Add _build_creator and _drain_log tests**

Append to `tests/dashboard/test_reconstruct_pane.py`:

```python
def test_build_creator_vggtx_returns_correct_type():
    from collab_splats.pointcloud.feedforward import VGGTXCreator
    pane, _, _ = _make_pane()
    creator = pane._build_creator("vggtx", 40.0)
    assert isinstance(creator, VGGTXCreator)
    assert creator.conf_threshold == 40.0


def test_build_creator_mapanything_returns_correct_type():
    from collab_splats.pointcloud.feedforward import MapAnythingCreator
    pane, _, _ = _make_pane()
    creator = pane._build_creator("mapanything", 50.0)
    assert isinstance(creator, MapAnythingCreator)
    assert creator.confidence_percentile == 50.0


def test_build_creator_unknown_backend_raises():
    pane, _, _ = _make_pane()
    with pytest.raises(ValueError, match="Unknown backend"):
        pane._build_creator("bad_backend", 35.0)


def test_drain_log_appends_to_log_area():
    pane, _, _ = _make_pane()
    pane._append_log("line one")
    pane._append_log("line two")
    pane._drain_log()
    assert "line one" in pane._log_area.value
    assert "line two" in pane._log_area.value


def test_drain_log_clears_buffer_after_drain():
    pane, _, _ = _make_pane()
    pane._append_log("line one")
    pane._drain_log()
    pane._drain_log()  # second drain should add nothing
    assert pane._log_area.value.count("line one") == 1


def test_drain_log_is_noop_when_empty():
    pane, _, _ = _make_pane()
    pane._drain_log()  # must not raise
    assert pane._log_area.value == ""
```

- [ ] **Step 2: Run to verify**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v -k "build_creator or drain" 2>&1 | tail -20
```

Expected: all PASS. If `test_build_creator_vggtx` or `test_build_creator_mapanything` fail with `ImportError`, the feedforward env is not set up — confirm `setup_feedforward.sh` was run.

- [ ] **Step 3: Run full suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_reconstruct_pane.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/dashboard/test_reconstruct_pane.py
git commit -m "test(dashboard): _build_creator type checks + log drain coverage"
```

---

## Task 4: Wire into app.py

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `tests/dashboard/test_app.py` (add smoke assertion)

- [ ] **Step 1: Read app.py first**

Read `collab_splats/dashboard/app.py` to find the exact lines for the Reconstruct placeholder. Locate:

```python
"Reconstruct": PlaceholderPane("Reconstruct", "Com…
```

- [ ] **Step 2: Update app.py**

In `collab_splats/dashboard/app.py`, add the import at the top with other pane imports:

```python
from collab_splats.dashboard.panes.reconstruct import ReconstructPane
```

Replace the placeholder line for Reconstruct:

```python
"Reconstruct": PlaceholderPane("Reconstruct", "Coming in Phase 3 — feedforward reconstruction"),
```

with:

```python
"Reconstruct": ReconstructPane(state=self._state, op_log=self._op_log),
```

Also add the instance variable assignment before `self._panes` dict (following the PreprocessPane pattern):

```python
self._reconstruct = ReconstructPane(state=self._state, op_log=self._op_log)
```

And update the panes dict to reference it:

```python
"Reconstruct": self._reconstruct,
```

- [ ] **Step 3: Check test_app.py for smoke test**

Read `tests/dashboard/test_app.py`. Find the existing smoke test. Add an assertion that the Reconstruct tab is no longer a placeholder:

```python
def test_reconstruct_pane_wired(tmp_path):
    from collab_splats.dashboard.app import App
    from collab_splats.dashboard.panes.reconstruct import ReconstructPane
    app = App(base_dir=str(tmp_path))
    assert isinstance(app._reconstruct, ReconstructPane)
```

Add this test to `tests/dashboard/test_app.py`.

- [ ] **Step 4: Run app tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_app.py tests/dashboard/test_reconstruct_pane.py -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "feat(dashboard): wire ReconstructPane into Tab 3"
```

---

## Task 5: Full dashboard test pass

- [ ] **Step 1: Run all dashboard tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v 2>&1 | tail -40
```

Expected: all tests PASS. Fix any failures before proceeding.

- [ ] **Step 2: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/examples 2>&1 | tail -40
```

Expected: no regressions. Fix any failures.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -p  # stage only relevant fixes
git commit -m "fix(dashboard): address test failures after ReconstructPane wiring"
```
