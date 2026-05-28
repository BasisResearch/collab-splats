# LocalizePane Design Spec

**Date:** 2026-05-28  
**Branch:** refactor/cu121  
**Status:** Approved — ready for implementation plan

---

## Overview

Implement the `LocalizePane` dashboard tab (currently a `PlaceholderPane`) for camera localization in a known reconstruction. Supports single-image localization and batch-folder runs. Uses the existing `CameraLocalizer` / `plot_correspondences()` backend in `collab_splats/pointcloud/localization.py`.

---

## Layout

50/50 horizontal split with a gap between panels. Controls bar spans the full width at the top. Batch mode is a collapsible section below the split.

```
┌──────────────────────────────────────────────────────────────────┐
│ [📁 query.jpg] [Browse]   Recon: [vggtx▾]  [DISK+LG▾]  [▶ Localize] │
├─────────────────────────────┬──────────────────────────────────────┤
│ inliers ● 142  out ● 45     │                                      │
│ best ref: frame_042  [warp] │   3D scene (PyVista VTK)             │
│                             │   ▲ query (red)                      │
│  [dark bg]  [query] [ref]   │   ▲ frame_042 (yellow) - - -        │
│   stitched correspondence   │   ▲ ▲ ▲ other cameras (faint blue)  │
│   image with keypoint lines │   · point cloud                      │
│                             │                                      │
│  legend  pose: t=[…] ✓      │                                      │
├─────────────────────────────┴──────────────────────────────────────┤
│ ▼ Batch mode  [📁 folder]  [▶ Run batch]  [⬇ Export CSV]          │
│  image | inliers | status | t-err (m) | pose t                    │
└──────────────────────────────────────────────────────────────────┘
```

**Correspondence panel (left 50%):**
- Renders `plot_correspondences()` output as a matplotlib PNG (`pn.pane.PNG`)
- Dark background shows gap between the two padded image boxes
- Green lines = inliers, red lines = outliers, crossing the gap between images
- `warp_corners=True` when checkbox checked: cyan quad on ref side (query corners → ref space), yellow quad on query side (ref corners → query space)
- Checkbox lives in the panel header row, right-aligned next to inlier stats
- Pose matrix shown as text below the figure on success; failure message on `pose=None`

**3D scene panel (right 50%):**
- Interactive PyVista VTK widget via `pn.pane.VTK`
- Shows all reference cameras as faint blue frustums + point cloud
- Query camera = red frustum; best-ref camera = yellow frustum; dashed connector line between them
- Implemented by new `LocalizeScenePanel` class (not reusing `ScenePanel` from `VisualizePane`)

---

## Components

### `collab_splats/dashboard/panes/localize.py`

Two classes in one file.

#### `LocalizeScenePanel(param.Parameterized)`

Owns the PyVista VTK widget for the 3D right panel.

- `__init__(output_dir, recon_method)` — loads point cloud + camera frustums from reconstruction dir
- `highlight(query_pose, ref_pose)` — recolors query frustum red, ref frustum yellow, draws dashed connector; all others faint blue
- `reset()` — clears highlight, returns all cameras to faint blue
- `panel()` → `pn.pane.VTK`

Loads reconstruction data from `{output_dir}/{recon_method}/colmap/sparse/0/` (cameras, images, points3D).

#### `LocalizePane(param.Parameterized)`

The full tab pane. Owns all controls, split layout, correspondence PNG pane, and batch table.

**Controls:**
- File picker (`pn.widgets.FileInput` or path text input + Browse button) for query image
- Recon method dropdown — populated by scanning `output_dir` for available method subdirs (`vggtx`, `mapanything`, `colmap`); defaults to whatever the Reconstruct tab used
- Extractor dropdown: `DISK+LightGlue` (default) | `XFeat+MNN`
- Run button (▶ Localize) — disabled until `output_dir` set and query image selected
- Warp corners checkbox — lives in correspondence panel header

**Batch mode (collapsible):**
- Folder picker for query image directory
- Run batch button
- Export CSV button
- `pn.widgets.Tabulator` with columns: `image`, `inliers`, `status`, `t-err (m)`, `pose t`
  - `t-err` column is optional: populated only if a ground-truth pose sidecar file exists alongside the query image; format determined at implementation time. Shows `—` otherwise.
  - Rows stream in as batch progresses via `pn.state.execute`

**AppState integration:**
- Watches `state.output_dir` via `param.watch` — pane enables/disables reactively
- Read-only consumer: no new AppState fields needed

---

## Data Flow

### Single-image

1. User selects query image + recon method + extractor → clicks ▶ Localize
2. Background thread (via `pn.state.execute` or `threading.Thread`):
   a. Load reconstruction from disk via `colmap_reconstruction_to_result(recon_dir)` → `PointcloudResult`; construct `CameraLocalizer(pts3d, extrinsics, intrinsics, image_paths, extractor=chosen_extractor)` directly. (`from_feedforward()` takes a live `FeedforwardResult` object and is not used here.)
   b. Call `localizer.localize(query_image)` → `LocalizationResult`
3. On success:
   - Call `plot_correspondences(loc, query_image, image_paths, warp_corners=checkbox.value)` → save to `io.BytesIO` → update `pn.pane.PNG`
   - Call `scene_panel.highlight(loc.pose, best_ref_pose)` to update VTK view
   - Show pose text below correspondence panel
4. On failure (`pose is None`):
   - Correspondence panel shows: "Localization failed — N inliers (threshold: 4)"
   - op_log records error entry
   - 3D scene unchanged

### Batch

1. User picks query folder → clicks ▶ Run batch
2. Background thread iterates image files, calls `localizer.localize()` per image
3. Each result appended to Tabulator data reactively
4. On completion: Export CSV button becomes active

### Recon method switch

Switching the dropdown reloads `CameraLocalizer` (and `LocalizeScenePanel`) from the new method dir but does not re-run localization. Run button must be clicked again.

---

## Error Handling

| Condition | Behaviour |
|---|---|
| `output_dir` not set | Pane disabled (same pattern as other panes) |
| No recon dir found for method | Dropdown shows "none available"; run disabled |
| Reconstruction load fails (missing colmap files / bad transforms.json) | op_log error entry; pane stays idle |
| `localize()` returns `pose=None` | Failure message in correspondence panel; 3D scene unchanged |
| Batch image fails | Row shows ✗ status; batch continues |

---

## Testing

File: `tests/dashboard/test_localize_pane.py`

- `LocalizeScenePanel.highlight()` — unit test updates actor colors without raising (mock PyVista)
- `LocalizePane` initializes disabled when `state.output_dir` is empty
- `LocalizePane` enables when `state.output_dir` set via `param.watch`
- Batch table populates correctly — mock `CameraLocalizer.localize()` to return fixed results, verify Tabulator rows
- No full VTK integration test (headless rendering is fragile) — mock `LocalizeScenePanel`

---

## Files Changed

| File | Change |
|---|---|
| `collab_splats/dashboard/panes/localize.py` | **New** — `LocalizeScenePanel` + `LocalizePane` |
| `collab_splats/dashboard/app.py` | Replace `PlaceholderPane("Localize", …)` with `LocalizePane(state=…, op_log=…)` |
| `collab_splats/dashboard/__init__.py` | Export `LocalizePane` |
| `tests/dashboard/test_localize_pane.py` | **New** — unit tests |

No changes to `localization.py`, `AppState`, or other panes.
