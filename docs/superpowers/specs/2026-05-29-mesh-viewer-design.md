# Mesh Viewer Integration — Design Spec

**Date:** 2026-05-29
**Branch:** refactor/cu121

## Problem

`ScenePanel` can display a pre-existing `mesh.ply` but:
1. No "Run Mesh" button — users can't generate or regenerate a mesh from the viewer.
2. Mesh params (voxel_size, sdf_trunc, depth_trunc, clean_repair) are not exposed.
3. Displaying a mesh requires clicking Load first, even though `_rebuild_mesh_viewer` doesn't need `_result`.

## Goal

- Mesh mode auto-displays `mesh.ply` on tab activate (no Load click needed).
- When `feedforward.zarr` exists, a "Run Mesh" button with TSDF params appears beneath the mode selector.
- User can tweak params and regenerate; viewer refreshes automatically.

## What Changes

### `collab_splats/dashboard/panes/visualize.py` — only file touched

#### New widgets (`_mesh_options_row`)

| Widget | Type | Default |
|--------|------|---------|
| `_mesh_voxel_input` | `FloatInput` | `0.01` |
| `_mesh_sdf_input` | `FloatInput` | `0.04` |
| `_mesh_depth_input` | `FloatInput` | `10.0` |
| `_mesh_clean_check` | `Checkbox` | `True` |
| `_mesh_run_btn` | `Button` | disabled until zarr exists |

Row is `visible=False` by default; shown when mode == "Mesh" (same pattern as `_points_options_row` / `_sim_query_row`).

#### Layout (mode selector + contextual rows)

```
[ Points | Mesh | Similarity ]

Active=Mesh:
  voxel [0.01] sdf_trunc [0.04] depth_trunc [10.0] [✓ clean] [Run Mesh]

Active=Points:
  (existing _points_options_row)

Active=Similarity:
  (existing _sim_query_row)
```

#### New / changed methods

**`_auto_display_mesh()`** — called from `_scan_available_modes` when `mesh.ply` exists. Calls `_rebuild_mesh_viewer()` via `pn.io.state.execute` (IOLoop thread). Does not require `_result`. Replaces current "Mesh mode requires Load" flow.

**`_on_run_mesh(event)`** — button callback. Background thread:
1. If `_result is None`: `FeedforwardResult.load_zarr(zarr_path)`
2. `pointcloud_to_mesh(_result, mesh_dir, method="open3d_tsdf", voxel_size=..., sdf_trunc=..., depth_trunc=..., clean_repair=...)`
3. On success: `pn.io.state.execute(_rebuild_mesh_viewer)` + status "Mesh done (N verts)"
4. On error: `_set_status(str(exc))`
5. `finally`: `_mesh_run_btn.disabled = False`

**`_scan_available_modes()` changes:**
- When `mesh.ply` exists: call `_auto_display_mesh()`.
- When `feedforward.zarr` exists at `{ds_dir}/{backend}/feedforward.zarr`: set `_mesh_run_btn.disabled = False`.
- When zarr missing but mesh.ply exists: `_set_status("No zarr — cannot regenerate")`.

**`_on_mode_change()` changes:**
- `"Mesh"` branch: `_mesh_options_row.visible = True`, others False.
- Other branches: `_mesh_options_row.visible = False` (existing options rows logic extended).

#### What does NOT change

- `_rebuild_mesh_viewer` logic unchanged — reads `mesh.ply`, calls `pv.read`, `add_mesh`.
- Mode availability gating unchanged — Mesh still requires `mesh.ply` on disk.
- `ReconstructPane` unchanged.
- No new files.

## Data Flow

```
Tab activate
  → _scan_available_modes()
      mesh.ply exists → _auto_display_mesh() → render immediately
      zarr exists     → _mesh_run_btn.disabled = False

Run Mesh click
  → _on_run_mesh()
      _mesh_run_btn.disabled = True
      thread:
        _result or load_zarr()
        pointcloud_to_mesh(voxel, sdf_trunc, depth_trunc, clean_repair)
        pn.io.state.execute(_rebuild_mesh_viewer)
      finally: btn re-enabled
```

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| No zarr, no mesh.ply | Mesh mode disabled |
| No zarr, mesh.ply exists | Render mesh; Run btn disabled; status note |
| Mesh gen failure | Status badge with error; Run btn re-enabled |
| Mesh gen success | Viewer refreshes; status "Mesh done (N verts)" |

## Testing

| Test | Assertion |
|------|-----------|
| `test_mesh_options_row_hidden_by_default` | `_mesh_options_row.visible is False` |
| `test_mesh_options_row_visible_in_mesh_mode` | after `_on_mode_change("Mesh")` → visible |
| `test_run_mesh_btn_disabled_without_zarr` | no zarr → btn disabled after scan |
| `test_run_mesh_btn_enabled_with_zarr` | zarr present → btn enabled after scan |
| `test_auto_display_mesh_on_scan` | mesh.ply present → `_rebuild_mesh_viewer` called (mocked) |
| `test_default_mode_is_mesh` | already passes |

## Out of Scope

- Poisson methods (stubs only; not implemented in `mesh/poisson.py`).
- Mesh method selector dropdown (only `open3d_tsdf` is functional today).
- Progress bar / log drain for mesh generation.
- Separate Mesh tab / MeshPane.
