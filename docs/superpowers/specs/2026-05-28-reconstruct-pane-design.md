# ReconstructPane — Design Spec

**Date:** 2026-05-28
**Phase:** 3 (Tab 3 of unified dashboard)
**Depends on:** Phase 1 (App skeleton, AppState, PreprocessPane)

---

## Overview

`ReconstructPane` implements Tab 3 of the dashboard. It lets the user pick a feedforward backend, tune the confidence threshold, and run the reconstruction pipeline against the frames already extracted by PreprocessPane. On completion it sets `state.feedforward_result`, enabling downstream tabs (Visualize, Localize).

BA and LC are explicitly out of scope for this phase — both sections appear as disabled stubs with a "not yet implemented" message.

---

## Architecture

### File

```
collab_splats/dashboard/panes/reconstruct.py
```

### Class

```python
class ReconstructPane(param.Parameterized):
    def __init__(self, state: AppState, op_log: OperationLog, **params): ...
    def panel(self) -> pn.viewable.Viewable: ...
```

Follows the same pattern as `PreprocessPane`: `param.Parameterized`, receives `state` and `op_log`, exposes `panel()`.

---

## AppState contract

| Field | Direction | Notes |
|---|---|---|
| `state.output_dir` | reads | Gate: pane active only when set |
| `state.feedforward_result` | writes | Set after successful run |
| `state.frames_zarr_path` | ignored | Not needed by this pane |

`state.frames` no longer exists (removed in Phase 2 refactor). Do not reference it.

---

## Layout

Two-column split panel, progress via global `op_log` strip.

```
┌─ Config (left ~380px) ───────┐  ┌─ Log stream (right) ─────────┐
│ Backend:  [vggtx ▼]          │  │                               │
│ Conf threshold: [35 ──────]  │  │  scrollable textarea          │
│                              │  │  green = success lines        │
│ ▼ Bundle Adjustment          │  │  white = info                 │
│   [disabled toggle]          │  │  red   = errors               │
│   ⚠ Not yet implemented      │  │                               │
│                              │  │                               │
│ ▼ Loop Closure               │  │                               │
│   [disabled toggle]          │  │                               │
│   ⚠ Not yet implemented      │  │                               │
│                              │  │                               │
│ [Run Reconstruction]         │  │                               │
│ [status html]                │  │                               │
└──────────────────────────────┘  └───────────────────────────────┘
```

---

## Widgets (`self._widget_name` convention)

| Name | Type | Details |
|---|---|---|
| `_backend_dd` | `pn.widgets.Select` | options: `["vggtx", "mapanything", "vggt_omega"]`, value: `"vggtx"` |
| `_conf_slider` | `pn.widgets.FloatSlider` | name="Conf threshold", start=0, end=100, step=1, value=35 |
| `_ba_toggle` | `pn.widgets.Toggle` | name="Bundle Adjustment", disabled=True |
| `_ba_stub_html` | `pn.pane.HTML` | "⚠ Not yet implemented" |
| `_lc_toggle` | `pn.widgets.Toggle` | name="Loop Closure", disabled=True |
| `_lc_stub_html` | `pn.pane.HTML` | "⚠ Not yet implemented" |
| `_run_btn` | `pn.widgets.Button` | name="Run Reconstruction", button_type="primary" |
| `_status_html` | `pn.pane.HTML` | idle / running / done / error indicator |
| `_log_area` | `pn.widgets.TextAreaInput` | readonly, height=500, auto-scrolls |

---

## Gate logic

```python
def _on_state_output_dir(self, event):
    ready = event.new is not None
    self._run_btn.disabled = not ready
```

`param.watch` on `state.output_dir`. Button disabled until `output_dir` is set.

---

## Background thread

`_run_btn.on_click` → `_on_run` → spawns `threading.Thread(target=_run_reconstruction, daemon=True)`.

Guard: if thread already alive, return immediately.

### `_run_reconstruction` steps

```
1. Validate (output_dir / "frames").exists() → op_log.error_op + return if not
2. op_log.start_op("Running reconstruction")
3. Instantiate creator:
     vggtx       → VGGTXCreator(conf_threshold=slider_value)
     mapanything → MapAnythingCreator(confidence_percentile=slider_value)
     vggt_omega  → VGGTOmegaCreator(conf_threshold=slider_value)
4. images_dir  = output_dir / "frames"
   backend_dir = output_dir / backend_name
5. result = creator.reconstruct(images_dir, backend_dir)
6. ff = creator.outputs
   ff.save_zarr(backend_dir / "feedforward.zarr")
7. state.feedforward_result = ff
8. op_log.finish_op()
```

On any exception: `op_log.error_op(str(e))`, re-enable Run button, append traceback to log.

### Log streaming

Thread appends lines to `self._log_lines: list[str]` under `self._log_lock: threading.Lock`.
`pn.state.add_periodic_callback(_drain_log, period=500)` drains into `_log_area.value`.
Callback registered in `panel()`, cancelled in `__del__` if needed.

---

## Creator imports

```python
from collab_splats.pointcloud.feedforward import (
    VGGTXCreator,
    MapAnythingCreator,
    VGGTOmegaCreator,
)
```

`VGGTOmegaCreator` available only when the vggt-omega submodule is initialised. Import inside the thread (not at module top) so a missing submodule doesn't prevent the pane from loading. If import fails, log error and abort.

---

## App wiring

`app.py` replaces the `PlaceholderPane` for "Reconstruct" with `ReconstructPane`:

```python
from collab_splats.dashboard.panes.reconstruct import ReconstructPane

self._reconstruct = ReconstructPane(state=self._state, op_log=self._op_log)
self._panes["Reconstruct"] = self._reconstruct
```

---

## AppState additions

None required. `feedforward_result` already exists in `AppState`.

---

## Out of scope (this phase)

- Bundle Adjustment (wrappers exist but not wired at Reconstructor level)
- Loop Closure (functional but deferred to a future phase)
- Kill / cancel running thread
- Per-backend advanced params (chunk_size, minibatch_size, etc.)
- Subprocess isolation (in-process thread chosen for simplicity, consistent with PreprocessPane)

---

## Testing

`tests/dashboard/test_reconstruct_pane.py` — flat test functions, no class-based.

Key test cases:
- `test_run_btn_disabled_without_output_dir` — button disabled on init
- `test_run_btn_enabled_after_output_dir_set` — watch fires, button enables
- `test_run_reconstruction_validates_frames_dir` — error path when `frames/` missing
- `test_run_reconstruction_sets_feedforward_result` — happy path with mock creator
- `test_log_drain_appends_to_log_area` — periodic callback drains lines
