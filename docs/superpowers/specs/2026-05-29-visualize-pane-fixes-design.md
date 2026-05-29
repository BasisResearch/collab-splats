# Visualize Pane Fixes — Design

**Date:** 2026-05-29
**Status:** Approved

## Summary

Four issues in the dashboard Visualize tab:

1. Camera frustums render at wrong positions (double-inversion bug)
2. VTK EGL stderr spam during queries
3. Timm/transformers warnings during model load
4. Two-scene compositor adds complexity without value; AppState is the authoritative config source

Approach: fix all four in one pass. No new abstractions.

---

## 1. Frustum Double-Inversion Fix

### Root cause

`create_camera_frustum_pyvista(pose)` in `utils/visualization.py` expects a **world-to-camera (w2c)** matrix and inverts it internally:

```python
c2w = np.linalg.inv(pose)
frustum.points = (c2w @ pts_h.T).T[:, :3]
```

Four call sites erroneously pass `np.linalg.inv(ext)` (already c2w), causing a second inversion back to w2c. Frustums are placed at `(w2c @ camera_origin)` instead of the camera centre.

### Fix

Remove `np.linalg.inv()` at each call site — pass `ext` (w2c) directly:

| File | Line | Change |
|------|------|--------|
| `dashboard/panes/visualize.py` | 456 | `create_camera_frustum_pyvista(ext)` |
| `dashboard/panes/localize.py` | 122 | `create_camera_frustum_pyvista(ext)` |
| `dashboard/panes/localize.py` | 166 | `create_camera_frustum_pyvista(query_ext)` |
| `utils/notebook.py` | 94 | `create_camera_frustum_pyvista(ext, scale=scale)` |

`create_camera_frustum_pyvista` itself and `visualize_splat` are already correct — no changes there.

---

## 2. VTK EGL Warning Suppression

### Root cause

PyVista uses a VTK EGL render window for headless rendering. In this container environment (no display), every `vtk_pane.synchronize()` call triggers `vtkEGLRenderWindow: Unable to eglMakeCurrent: 12290` on stderr. Pure noise — rendering still works via software fallback.

### Fix

Add to `dashboard/__main__.py` before any pane imports:

```python
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
```

This disables all VTK C++ warning output for the process lifetime. Applied once at entry point, not buried in library code.

---

## 3. Timm / Transformers Warning Suppression

### Root cause

Two distinct sources when semantic models (Talk2DINO, DINOv2) load:

- **torch.nn.modules.module**: `UserWarning: copying from a non-meta parameter in the checkpoint to a meta parameter` — emitted per-weight during `load_state_dict` when model was initialized with `device="meta"`.
- **HuggingFace transformers**: `Some weights of the model checkpoint... were not used` — informational, not actionable.

### Fix

Add to `dashboard/__main__.py` at startup:

```python
import warnings
import transformers

warnings.filterwarnings("ignore", message=".*non-meta parameter.*")
transformers.logging.set_verbosity_error()
```

Applied at process entry point so it covers all downstream model loads. Not applied inside library modules.

---

## 4. Single-Scene Simplification

### Motivation

`VisualizePane` currently wraps two `ScenePanel` instances ("A" and "B") side-by-side. Scene B has no AppState wiring — it is a fully independent manual loader. Since AppState is the authoritative source of the active dataset/backend (set when a reconstruction runs), Scene B's independent dropdowns are redundant and create user confusion.

### Changes

**Remove `VisualizePane` compositor class** (`dashboard/panes/visualize.py`):
- Delete the class (~40 lines)
- Delete `_rescan_both_scenes`, vertical divider, `scenes_row` layout

**Simplify `ScenePanel`**:
- Remove `scene_id: str` constructor param
- Remove `if scene_id == "A":` guards — AppState watchers become unconditional
- Internal label `f"Scene {self._scene_id}"` → `"Scene"`

**Update `app.py`**:
- Instantiate `ScenePanel(base_dir, state, op_log)` directly instead of `VisualizePane`
- Tab activation calls `scene_panel.rescan()` directly

### What stays the same

- Dataset/backend dropdowns + Load button (manual confirm-before-load pattern preserved)
- AppState auto-suggest (`_on_feedforward_result`, `_on_lifted_features_path`)
- All mode switching (PCD / Mesh / Similarity)
- Query, frustum toggle, point-size, snapshot

### Net change

~60–80 lines deleted, one class removed, no behavioral regressions.

---

## Affected Files

| File | Change |
|------|--------|
| `dashboard/__main__.py` | Add VTK warning off + warnings filters + transformers verbosity |
| `dashboard/app.py` | Replace `VisualizePane` with direct `ScenePanel` instantiation |
| `dashboard/panes/visualize.py` | Remove `VisualizePane` class; simplify `ScenePanel` (drop `scene_id`) |
| `dashboard/panes/localize.py` | Fix frustum double-inversion (lines 122, 166) |
| `utils/notebook.py` | Fix frustum double-inversion (line 94) |

## Out of Scope

- Fully centralizing dataset/backend into AppState (no dropdowns on ScenePanel) — deferred
- Any changes to `create_camera_frustum_pyvista` itself
- Changes to `visualize_splat` in `utils/visualization.py`
