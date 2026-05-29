# Dashboard Visualization Improvements — Design Spec

Date: 2026-05-29

## Overview

Three improvements to the dashboard's Visualize tab and sidebar:

1. **Unified sidebar load** — dataset, pointcloud backend, and semantic model selectors move to the persistent sidebar. One Load button populates all panes. Per-pane controls removed.
2. **VTK viewer performance** — decimation cap for display, actor caching, throttled `synchronize()`, and reset camera fix.
3. **Ground plane alignment** — RANSAC plane detection saved to `transforms.json` per scene; checkbox toggle applies/inverts in viewer.

Also includes a **geometry refactor**: extract `fit_dominant_plane` and `rotation_align_vectors` so RANSAC floor alignment is reusable across mesh and pointcloud pipelines.

---

## Section 1: Architecture & AppState Changes

### AppState (`collab_splats/dashboard/state.py`)

Five new fields:

```python
pointcloud_backend  = param.String(default="")      # e.g. "vggt_x"
semantic_extractor  = param.String(default="")      # e.g. "dino_v2"
ground_plane_enabled = param.Boolean(default=True)
ground_plane_R      = param.Parameter(default=None) # (3,3) np.ndarray or None
ground_plane_t      = param.Parameter(default=None) # (3,)  np.ndarray or None
```

`feedforward_result` already exists — the sidebar Load button populates it directly, moving load responsibility from `ScenePanel` into `App._do_load_models()` (background thread).

### Sidebar (`collab_splats/dashboard/app.py → _build_sidebar`)

Two new sections added after SESSION:

**MODELS section:**
- `_dataset_dd` — `Select`, options from `_scan_output_dirs(base_dir)` (same scan as session selector)
- `_backend_dd` — `Select`, repopulates when dataset changes via existing `_scan_backends(dataset_dir)` helper
- `_extractor_dd` — `Select`, options from `BaseFeatureExtractor._registry`
- `_load_models_btn` — `Button(type="primary", name="⚡ Load")` — triggers background load
- Status HTML showing point counts after load

**VIEW section:**
- `_ground_plane_check` — `Checkbox(name="Align ground plane")` — toggles `state.ground_plane_enabled`
- `_ground_plane_status` — HTML showing "auto-detected · saved" / "not detected"
- `_redetect_btn` — `Button(name="↺ Re-detect ground plane")` — recomputes + saves
- `_frustum_check` — moved here from `ScenePanel`

### ScenePanel (`collab_splats/dashboard/panes/visualize.py`)

**Removed:** `_dataset_dd`, `_backend_dd`, `_load_btn`, `_frustum_check`, `_load_thread`, all load logic.

**Added watchers:**
- `state.watch("feedforward_result", self._on_feedforward_result)` — triggers display result prep + ground plane application
- `state.watch("ground_plane_enabled", self._on_ground_plane_toggle)` — re-renders with/without transform

**Keeps:** mode selector (PCD/Mesh/Similarity), point size slider, similarity query row, reset/snapshot buttons.

### SemanticsPane (`collab_splats/dashboard/panes/semantics.py`)

**Removed:** `_method_dd` (per-pane extractor selector).

**Added:** `state.watch("semantic_extractor", self._on_extractor_change)` — sets active extractor on state change, triggers cache discovery for the new extractor.

---

## Section 2: Ground Plane Alignment

### Geometry refactor

**`collab_splats/utils/geometry.py`** — add pure-numpy helper:

```python
def rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix R such that R @ src ≈ dst.

    Args:
        src: (3,) unit vector to rotate from.
        dst: (3,) unit vector to rotate to.
    Returns:
        (3, 3) rotation matrix. Identity if src ≈ dst.
    """
```

Uses Rodrigues axis-angle: `axis = cross(src, dst)`, `angle = arccos(dot(src, dst))`. Falls back to identity when `‖axis‖ < 1e-6`.

**`collab_splats/pointcloud/utils.py`** — add:

```python
def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit dominant plane via RANSAC; return (R_3x3, t_3) aligning plane to Z-up.

    Args:
        points: (N, 3) float32/64 point cloud.
    Returns:
        R: (3, 3) rotation matrix aligning floor normal to [0, 0, 1].
        t: (3,) translation placing floor at z=0.
    """
```

Implementation:
1. Build `o3d.PointCloud` from `points`
2. Call `pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)` → `[a, b, c, d]`
3. `normal = [a,b,c] / ‖[a,b,c]‖`; flip if `normal[2] < 0`
4. `R = rotation_align_vectors(normal, [0,0,1])`
5. `t = [0, 0, -d / ‖[a,b,c]‖]`
6. Return `(R, t)`

**`collab_splats/mesh/utils.py`** — refactor `align_geometry_floor`:

```python
def align_geometry_floor(geometry, dist_threshold=0.02, ransac_n=3, num_iterations=1000, num_sample_points=10000):
    """Align O3D PointCloud or TriangleMesh to floor using fit_dominant_plane."""
    from collab_splats.pointcloud.utils import fit_dominant_plane
    # sample surface points if mesh
    pts = np.asarray(pcd_for_detection.points)
    R, t = fit_dominant_plane(pts)
    geometry.rotate(R, center=(0, 0, 0))
    geometry.translate(t)
    return geometry, R, t
```

`get_floor_plane` is kept as an internal helper (used by `fit_dominant_plane` indirectly via O3D).

### Persistence

Saved to `{dataset_dir}/{backend}/transforms.json`:

```json
{
  "ground_plane": {
    "R": [[...], [...], [...]],
    "t": [0.0, 0.0, -1.23]
  }
}
```

### Load path

`App._do_load_models()` after loading `FeedforwardResult`:
1. Check for `transforms.json` in `{dataset_dir}/{backend}/`
2. If missing: call `fit_dominant_plane(result.points)` → save → set `_ground_plane_status` to "auto-detected · saved"
3. If present: load R + t → set status to "loaded from file"
4. Set `state.ground_plane_R = R` and `state.ground_plane_t = t` — ScenePanel watches both

### Apply/invert

`ScenePanel._apply_ground_plane(result) -> FeedforwardResult`:
- Shallow-copies `result`
- When `state.ground_plane_enabled=True`: `pts_out = (R @ pts.T).T + t`; extrinsics: left-multiply each by `R_4x4`
- When `False`: apply `R.T` and `-t` (inverse rotation, inverse translation)
- Always called before rendering; no separate code path

Re-detect button: recomputes `fit_dominant_plane(self._result.points)` → overwrites `transforms.json` → re-renders.

---

## Section 3: VTK Performance + Reset Camera Fix

### Display decimation

`ScenePanel._prepare_display_result(result, max_pts=150_000) -> FeedforwardResult`:
- If `len(result.points) <= max_pts`: return result unchanged
- Else: voxel subsample via `clean_pcd` (from `pointcloud/utils.py`, `downsample=True`, `outlier_removal=False`, `distance_removal=False`) to get display copy
- `self._result` = full result (kept for similarity queries — needs all points + lifted features)
- `self._display_result` = decimated (used for PCD/Mesh rendering only)
- Status: `"Loaded 487k pts (display: 142k)"`

### Actor caching

Build PCD and Mesh actors once on load; toggle visibility on mode switch:

```python
self._pcd_actor: Any | None = None   # cached VTK actor
self._mesh_actor: Any | None = None  # cached VTK actor
```

On mode switch:
- PCD/Mesh: `actor.VisibilityOn()` / `actor.VisibilityOff()` — no `plotter.clear()` + rebuild
- Rebuild only when: new data loaded, ground plane toggled, point size changed (PCD only)
- Similarity: always rebuilds (colors change per query — cache provides no benefit)

### Throttle `synchronize()`

- Remove `synchronize()` from `_on_frustum_toggle` and `_on_point_size_change`
- Point size: `actor.GetProperty().SetPointSize(n)` directly, then `self._vtk_pane.param.trigger("object")` (lightweight refresh)
- Frustum toggle: add/remove frustum actors directly, single `synchronize()` at end
- `synchronize()` only called after full actor rebuild (mode switch or new data)

### Reset camera fix

Current bug: `plotter.reset_camera()` uses stale actor bounds from prior scene.

Fix in `_on_reset_camera`:
```python
def _on_reset_camera(self, event):
    self._plotter.reset_camera()
    self._vtk_pane.synchronize()
```

Ensure `reset_camera()` is called **after** actors are added (i.e., inside `_rebuild_*_viewer()`), not before. This gives correct bounds for the new scene.

---

## File Change Summary

| File | Change |
|------|--------|
| `collab_splats/utils/geometry.py` | Add `rotation_align_vectors` |
| `collab_splats/pointcloud/utils.py` | Add `fit_dominant_plane` |
| `collab_splats/mesh/utils.py` | Refactor `align_geometry_floor` to call `fit_dominant_plane` |
| `collab_splats/dashboard/state.py` | Add `pointcloud_backend`, `semantic_extractor`, `ground_plane_enabled` |
| `collab_splats/dashboard/app.py` | Add MODELS + VIEW sidebar sections; add `_do_load_models()` |
| `collab_splats/dashboard/panes/visualize.py` | Remove load controls; add ground plane apply/invert; add actor caching; fix reset camera |
| `collab_splats/dashboard/panes/semantics.py` | Remove `_method_dd`; watch `state.semantic_extractor` |

---

## Out of Scope

- Dashboard visual re-styling (noted as a future improvement — user prefers the mockup aesthetic)
- Scene B (second viewer panel) — Phase 4 work
- Ground plane for mesh display (Mesh mode uses stored `mesh.ply` which may already be aligned by `align_geometry_floor` during meshing)
- Automatic ground plane re-detection on every load (only auto-detects when `transforms.json` absent)
