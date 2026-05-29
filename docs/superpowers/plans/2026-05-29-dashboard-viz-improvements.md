# Dashboard Visualization Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unified sidebar load (dataset / pointcloud backend / semantic model), VTK actor caching + decimation for perf, and RANSAC ground plane alignment saved to `transforms.json`.

**Architecture:** Geometry refactor extracts `rotation_align_vectors` (pure numpy, `utils/geometry.py`) and `fit_dominant_plane` (O3D RANSAC wrapper, `pointcloud/utils.py`). `align_geometry_floor` in `mesh/utils.py` is refactored to call `fit_dominant_plane`. AppState gains 5 new fields. Sidebar gains MODELS + VIEW sections owning all load controls. ScenePanel strips per-pane load widgets and gains ground plane apply/invert, actor caching, and reset camera fix. SemanticsPane watches `state.semantic_extractor` instead of owning its own method dropdown.

**Tech Stack:** panel, pyvista, open3d, numpy, param, dataclasses

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/utils/geometry.py` | Add `rotation_align_vectors` |
| `collab_splats/pointcloud/utils.py` | Add `fit_dominant_plane` |
| `collab_splats/mesh/utils.py` | Refactor `align_geometry_floor` to call `fit_dominant_plane` |
| `collab_splats/dashboard/state.py` | Add 5 new param fields |
| `collab_splats/dashboard/app.py` | Add MODELS + VIEW sidebar sections; add `_do_load_models()` |
| `collab_splats/dashboard/panes/visualize.py` | Strip load controls; add actor caching + ground plane + reset fix |
| `collab_splats/dashboard/panes/semantics.py` | Strip `_method_dd`; watch `state.semantic_extractor` |
| `tests/utils/test_geometry.py` | New: `rotation_align_vectors` tests |
| `tests/pointcloud/test_utils.py` | Add `fit_dominant_plane` tests |
| `tests/mesh/test_utils_ground_plane.py` | New: refactored `align_geometry_floor` tests |
| `tests/dashboard/test_state.py` | New: AppState field tests |
| `tests/dashboard/test_visualize.py` | Update existing tests; add ground plane + actor tests |

---

### Task 1: `rotation_align_vectors` in `utils/geometry.py`

**Files:**
- Modify: `collab_splats/utils/geometry.py`
- Create: `tests/utils/test_geometry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_geometry.py
import numpy as np
import pytest
from collab_splats.utils.geometry import rotation_align_vectors


def test_rotation_align_vectors_identity():
    """Aligning a vector to itself returns identity."""
    src = np.array([0.0, 0.0, 1.0])
    R = rotation_align_vectors(src, src)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_rotation_align_vectors_aligns_correctly():
    """R @ src ≈ dst."""
    src = np.array([0.0, 1.0, 0.0])
    dst = np.array([0.0, 0.0, 1.0])
    R = rotation_align_vectors(src, dst)
    result = R @ src
    np.testing.assert_allclose(result, dst, atol=1e-10)


def test_rotation_align_vectors_is_rotation():
    """det(R) == 1 and R @ R.T == I."""
    src = np.array([1.0, 0.0, 0.0])
    dst = np.array([0.0, 1.0, 0.0])
    R = rotation_align_vectors(src, dst)
    assert abs(np.linalg.det(R) - 1.0) < 1e-10
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_rotation_align_vectors_antiparallel():
    """180-degree case: src = -dst still returns a valid rotation."""
    src = np.array([0.0, 0.0, 1.0])
    dst = np.array([0.0, 0.0, -1.0])
    R = rotation_align_vectors(src, dst)
    result = R @ src
    np.testing.assert_allclose(result, dst, atol=1e-6)
```

- [ ] **Step 2: Run — verify FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_geometry.py -v 2>&1 | head -20
```
Expected: `ImportError` or `AttributeError: module has no attribute rotation_align_vectors`.

- [ ] **Step 3: Implement `rotation_align_vectors`**

In `collab_splats/utils/geometry.py`, after `extract_intrinsics`, add:

```python
def rotation_align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix R such that R @ src ≈ dst.

    Args:
        src: (3,) unit vector to rotate from.
        dst: (3,) unit vector to rotate to.
    Returns:
        (3, 3) rotation matrix. Identity if src ≈ dst or antiparallel fallback.
    """
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    axis = np.cross(src, dst)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        # parallel (identity) or antiparallel (180° rotation around arbitrary perp axis)
        if np.dot(src, dst) > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return -np.eye(3) + 2 * np.outer(axis, axis)
    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(src, dst), -1.0, 1.0))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
```

- [ ] **Step 4: Run — verify PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_geometry.py -v 2>&1 | tail -10
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/geometry.py tests/utils/test_geometry.py
git commit -m "feat(geometry): add rotation_align_vectors — pure-numpy Rodrigues rotation"
```

---

### Task 2: `fit_dominant_plane` in `pointcloud/utils.py`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Modify: `tests/pointcloud/test_utils.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/pointcloud/test_utils.py`:

```python
import numpy as np
from collab_splats.pointcloud.utils import fit_dominant_plane


def test_fit_dominant_plane_flat_z_up():
    """Flat ground at z=-1 → R≈I, t brings floor to z=0."""
    rng = np.random.default_rng(42)
    # Ground plane at z = -1 with small noise
    xy = rng.uniform(-5, 5, (800, 2)).astype(np.float32)
    z = rng.normal(-1.0, 0.005, (800,)).astype(np.float32)
    ground = np.column_stack([xy, z])
    # Scatter above-ground points
    above_xy = rng.uniform(-5, 5, (100, 2)).astype(np.float32)
    above_z = rng.uniform(-0.5, 2.0, (100,)).astype(np.float32)
    above = np.column_stack([above_xy, above_z])
    points = np.vstack([ground, above])

    R, t = fit_dominant_plane(points)

    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # After applying transform, floor z-mean should be ≈ 0
    pts_aligned = (R @ points[:800].T).T + t
    np.testing.assert_allclose(pts_aligned[:, 2].mean(), 0.0, atol=0.1)


def test_fit_dominant_plane_returns_valid_rotation():
    """R is a proper rotation matrix (det=1, orthogonal)."""
    rng = np.random.default_rng(7)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    pts[:400, 2] = rng.normal(0, 0.01, 400)  # flat-ish ground at z=0
    R, t = fit_dominant_plane(pts)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
```

- [ ] **Step 2: Run — verify FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_utils.py::test_fit_dominant_plane_flat_z_up tests/pointcloud/test_utils.py::test_fit_dominant_plane_returns_valid_rotation -v 2>&1 | tail -10
```
Expected: `ImportError` or `AttributeError`.

- [ ] **Step 3: Implement `fit_dominant_plane`**

In `collab_splats/pointcloud/utils.py`, find the section heading `########## Geometry: OBB + mask lifting ################` and add `fit_dominant_plane` before `compute_obb_from_points`:

```python
def fit_dominant_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit dominant plane via RANSAC; return (R_3x3, t_3) aligning plane to Z-up.

    Uses Open3D's segment_plane on the full point cloud. No heuristic percentile —
    the dominant plane (largest inlier set) is taken as the floor.

    Args:
        points: (N, 3) float32 or float64 point cloud.
    Returns:
        R: (3, 3) rotation matrix aligning floor normal to [0, 0, 1].
        t: (3,) translation placing floor at z=0 after rotation is applied.
    """
    import open3d as o3d  # optional heavy dep
    from collab_splats.utils.geometry import rotation_align_vectors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    n_mag = np.linalg.norm([a, b, c])
    normal = np.array([a, b, c]) / n_mag
    d_norm = d / n_mag  # plane: normal · x + d_norm = 0; floor at z = -d_norm after rotation

    # Ensure normal points upward (positive Z component after alignment)
    if normal[2] < 0:
        normal = -normal
        d_norm = -d_norm

    R = rotation_align_vectors(normal, np.array([0.0, 0.0, 1.0]))
    # After R, floor is at z = -d_norm. Translate by d_norm to bring to z = 0.
    t = np.array([0.0, 0.0, d_norm])
    return R.astype(np.float64), t.astype(np.float64)
```

- [ ] **Step 4: Run — verify PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_utils.py::test_fit_dominant_plane_flat_z_up tests/pointcloud/test_utils.py::test_fit_dominant_plane_returns_valid_rotation -v 2>&1 | tail -10
```
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_utils.py
git commit -m "feat(pointcloud): add fit_dominant_plane — RANSAC floor detection returning (R, t)"
```

---

### Task 3: Refactor `align_geometry_floor` in `mesh/utils.py`

**Files:**
- Modify: `collab_splats/mesh/utils.py`
- Create: `tests/mesh/test_utils_ground_plane.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/mesh/test_utils_ground_plane.py
import numpy as np
import open3d as o3d
import pytest
from collab_splats.mesh.utils import align_geometry_floor


def _make_flat_pcd(floor_z: float = -1.0, n: int = 500, seed: int = 0) -> o3d.geometry.PointCloud:
    """Synthetic flat floor at floor_z with scatter above."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-3, 3, (n, 2))
    z = rng.normal(floor_z, 0.005, n)
    pts = np.column_stack([xy, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def test_align_geometry_floor_pcd_floor_at_zero():
    """After alignment the dominant floor plane should be at z ≈ 0."""
    pcd = _make_flat_pcd(floor_z=-1.0)
    aligned_pcd, R, t = align_geometry_floor(pcd)
    pts = np.asarray(aligned_pcd.points)
    # Bottom 30% of points (the floor inliers) should be near z=0
    bottom = pts[pts[:, 2] < np.percentile(pts[:, 2], 30)]
    np.testing.assert_allclose(bottom[:, 2].mean(), 0.0, atol=0.1)


def test_align_geometry_floor_returns_r_t_shapes():
    """Return types and shapes are correct."""
    pcd = _make_flat_pcd()
    _, R, t = align_geometry_floor(pcd)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_align_geometry_floor_mesh():
    """Works on TriangleMesh input without error."""
    mesh = o3d.geometry.TriangleMesh.create_box(2, 2, 0.1)
    mesh.translate([-1, -1, -0.5])
    _, R, t = align_geometry_floor(mesh)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
```

- [ ] **Step 2: Run — verify current tests PASS (baseline)**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/test_utils_ground_plane.py -v 2>&1 | tail -15
```
Tests should fail (file doesn't exist yet). After creating the file, confirm they fail before refactor.

- [ ] **Step 3: Refactor `align_geometry_floor`**

In `collab_splats/mesh/utils.py`, replace the body of `align_geometry_floor` (lines 261–352) with:

```python
def align_geometry_floor(
    geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
    dist_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    num_sample_points: int = 10000,
) -> tuple[Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh], np.ndarray, np.ndarray]:
    """Align point cloud or triangle mesh to the floor plane.

    Uses fit_dominant_plane (single RANSAC call) to detect the dominant floor
    and compute rotation + translation, then applies both to the geometry.

    Args:
        geometry: Input geometry (PointCloud or TriangleMesh).
        dist_threshold: RANSAC distance threshold for inlier classification.
        ransac_n: Points sampled per RANSAC iteration.
        num_iterations: Number of RANSAC iterations.
        num_sample_points: Surface sample count for mesh inputs only.
    Returns:
        Tuple of (aligned_geometry, R (3,3), t (3,)).
    """
    from collab_splats.pointcloud.utils import fit_dominant_plane

    is_mesh = isinstance(geometry, o3d.geometry.TriangleMesh)
    if is_mesh:
        sample_pcd = geometry.sample_points_uniformly(number_of_points=num_sample_points)
        pts = np.asarray(sample_pcd.points)
    else:
        pts = np.asarray(geometry.points)

    R, t = fit_dominant_plane(pts)
    geometry.rotate(R, center=(0, 0, 0))
    geometry.translate(t)
    return geometry, R, t
```

Also delete `get_floor_plane` (lines 355–369) — it is now internal to `fit_dominant_plane` via O3D's `segment_plane`. If other callers exist, keep it but mark `# internal — prefer fit_dominant_plane`.

Check for other callers first:
```bash
grep -rn "get_floor_plane" /workspace/collab-splats/collab_splats/ --include="*.py"
```
If no other callers: delete it. If callers exist: keep it unchanged.

- [ ] **Step 4: Run — verify PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/mesh/test_utils_ground_plane.py -v 2>&1 | tail -10
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_utils_ground_plane.py
git commit -m "refactor(mesh): align_geometry_floor uses fit_dominant_plane — single RANSAC call"
```

---

### Task 4: AppState new fields

**Files:**
- Modify: `collab_splats/dashboard/state.py`
- Create: `tests/dashboard/test_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/dashboard/test_state.py
from collab_splats.dashboard.state import AppState


def test_appstate_new_fields_defaults():
    state = AppState()
    assert state.pointcloud_backend == ""
    assert state.semantic_extractor == ""
    assert state.ground_plane_enabled is True
    assert state.ground_plane_R is None
    assert state.ground_plane_t is None


def test_appstate_ground_plane_fields_settable():
    import numpy as np
    state = AppState()
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1.0])
    state.ground_plane_R = R
    state.ground_plane_t = t
    assert state.ground_plane_R is R
    assert state.ground_plane_t is t


def test_appstate_ground_plane_enabled_toggle():
    state = AppState()
    state.ground_plane_enabled = False
    assert state.ground_plane_enabled is False
```

- [ ] **Step 2: Run — verify FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_state.py -v 2>&1 | tail -10
```
Expected: AttributeError on `pointcloud_backend`.

- [ ] **Step 3: Add fields to AppState**

In `collab_splats/dashboard/state.py`, add after `lifted_features_path`:

```python
    pointcloud_backend = param.String(default="")
    semantic_extractor = param.String(default="")
    ground_plane_enabled = param.Boolean(default=True)
    ground_plane_R = param.Parameter(default=None)
    ground_plane_t = param.Parameter(default=None)
```

- [ ] **Step 4: Run — verify PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_state.py -v 2>&1 | tail -10
```
Expected: 3 PASSED.

- [ ] **Step 5: Run full suite to check no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -15
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/test_state.py
git commit -m "feat(dashboard): add pointcloud_backend, semantic_extractor, ground_plane fields to AppState"
```

---

### Task 5: Sidebar MODELS section + `_do_load_models()`

**Files:**
- Modify: `collab_splats/dashboard/app.py`

This task adds the MODELS sidebar section and the background load method. Ground plane detection is wired in Task 6.

- [ ] **Step 1: Add MODELS widgets to `__init__`**

In `App.__init__`, after `self._reconstruct = ReconstructPane(...)`, add imports at top of `app.py` for `_scan_datasets` and `_scan_backends` from `panes/visualize.py`:

```python
from collab_splats.dashboard.panes.visualize import _scan_datasets, _scan_backends, _scan_extractors
```

Then in `__init__` after panes dict, add:

```python
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
```

- [ ] **Step 2: Add `_on_models_dataset_change`, `_on_models_load`, `_do_load_models`**

Add these methods to `App` (after `_on_confirm_session`):

```python
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

            # Update state on IOLoop thread
            def _set_state() -> None:
                self._state.output_dir = ds_dir
                self._state.pointcloud_backend = backend
                self._state.semantic_extractor = extractor if extractor != "(none)" else ""
                self._state.feedforward_result = result
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
```

`threading` is already imported in `app.py` via `panes/reconstruct`; add `import threading` at top if absent.

- [ ] **Step 3: Add MODELS section to `_build_sidebar` return value**

In `_build_sidebar`, extend the returned `pn.Column` to include the MODELS section. Replace the final `return pn.Column(...)` block:

```python
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
            width=300,
        )
```

- [ ] **Step 4: Smoke test — dashboard starts without error**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import panel as pn
pn.extension('vtk')
from collab_splats.dashboard.app import App
app = App(base_dir='/workspace/outputs')
print('App constructed OK')
" 2>&1 | grep -E "OK|Error|Traceback"
```
Expected: `App constructed OK`.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/app.py
git commit -m "feat(dashboard): add sidebar MODELS section and _do_load_models background loader"
```

---

### Task 6: Sidebar VIEW section + ground plane transforms.json

**Files:**
- Modify: `collab_splats/dashboard/app.py`

Adds the VIEW section (ground plane checkbox + re-detect, frustum checkbox moved from ScenePanel). Wires ground plane detection into `_do_load_models`.

- [ ] **Step 1: Add VIEW widgets to `__init__`**

In `App.__init__`, after the MODELS widgets block, add:

```python
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
```

- [ ] **Step 2: Add VIEW callbacks**

```python
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
            from collab_splats.pointcloud.utils import fit_dominant_plane

            result = self._state.feedforward_result
            R, t = fit_dominant_plane(result.points)

            # Save to transforms.json alongside the backend output
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
```

- [ ] **Step 3: Load transforms.json in `_do_load_models`**

In `_do_load_models`, after `result = FeedforwardResult.load_zarr(zarr_path)`, add:

```python
            # Load or auto-detect ground plane
            import json
            transforms_path = ds_dir / backend / "transforms.json"
            gp_R: np.ndarray | None = None
            gp_t: np.ndarray | None = None
            gp_status = ""
            if transforms_path.exists():
                try:
                    data = json.loads(transforms_path.read_text())
                    gp = data.get("ground_plane", {})
                    gp_R = np.array(gp["R"], dtype=np.float64)
                    gp_t = np.array(gp["t"], dtype=np.float64)
                    gp_status = "loaded from file"
                except Exception:
                    logger.warning("Could not parse transforms.json at %s", transforms_path)
            else:
                # Auto-detect and save
                try:
                    from collab_splats.pointcloud.utils import fit_dominant_plane
                    gp_R, gp_t = fit_dominant_plane(result.points)
                    transforms_path.write_text(
                        json.dumps({"ground_plane": {"R": gp_R.tolist(), "t": gp_t.tolist()}}, indent=2)
                    )
                    gp_status = "auto-detected · saved"
                except Exception as exc:
                    logger.warning("Ground plane auto-detect failed: %s", exc)
```

And extend `_set_state` inside `_do_load_models` to also set ground plane fields:

```python
            def _set_state() -> None:
                self._state.output_dir = ds_dir
                self._state.pointcloud_backend = backend
                self._state.semantic_extractor = extractor if extractor != "(none)" else ""
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
```

Also add `import numpy as np` to `app.py` if not present.

- [ ] **Step 4: Add VIEW section to `_build_sidebar` return**

Extend the `pn.Column(...)` in `_build_sidebar` to append the VIEW section at the end:

```python
            pn.layout.Divider(),
            pn.pane.HTML("<h3 style='color:#2596be;margin:8px 0 8px 0'>View</h3>"),
            self._ground_plane_check,
            self._ground_plane_status,
            self._redetect_btn,
            self._sidebar_frustum_check,
```

- [ ] **Step 5: Smoke test**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import panel as pn
pn.extension('vtk')
from collab_splats.dashboard.app import App
app = App(base_dir='/workspace/outputs')
print('App VIEW section OK')
" 2>&1 | grep -E "OK|Error|Traceback"
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/app.py
git commit -m "feat(dashboard): add sidebar VIEW section — ground plane checkbox, re-detect, frustum toggle"
```

---

### Task 7: ScenePanel — strip load controls, actor caching, reset camera fix

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`
- Modify: `tests/dashboard/test_visualize.py`

**What gets removed:** `_dataset_dd`, `_backend_dd`, `_load_btn`, `_frustum_check`, `_load_thread`, `_do_load`, `_on_load`, `_on_dataset_change`, `wire_tabs` (if only used for rescan on tab switch — check first). `_scan_available_modes` is kept but now triggered by watching `state.feedforward_result`.

**What gets added:** `_pcd_actor`, `_mesh_actor` caches; `state.watch` on `feedforward_result`; `_on_frustum_toggle_from_sidebar`; fix `_on_reset_camera`.

- [ ] **Step 1: Update tests that reference removed widgets**

In `tests/dashboard/test_visualize.py`, update or remove tests that reference `sp._dataset_dd`, `sp._backend_dd`, `sp._load_btn`, `sp._frustum_check`:

- `test_scene_panel_constructs` — change to check `sp._mode_selector` instead of `sp._dataset_dd.options`
- `test_scene_panel_frustum_is_checkbox` — remove (frustum check now in sidebar)

Replace:
```python
def test_scene_panel_constructs(tmp_path):
    _make_dataset(tmp_path, "scene_01")
    state = AppState()
    op_log = _make_op_log()
    sp = ScenePanel(
        base_dir=tmp_path,
        state=state,
        op_log=op_log,
        _off_screen=True,
    )
    assert "scene_01" in sp._dataset_dd.options
```

With:
```python
def test_scene_panel_constructs(tmp_path):
    state = AppState()
    op_log = _make_op_log()
    sp = ScenePanel(
        base_dir=tmp_path,
        state=state,
        op_log=op_log,
        _off_screen=True,
    )
    assert isinstance(sp._mode_selector, pn.widgets.RadioButtonGroup)
    assert sp._mode_selector.disabled is True  # disabled until result loaded
```

Delete `test_scene_panel_frustum_is_checkbox`.

- [ ] **Step 2: Verify existing tests PASS before changes**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v 2>&1 | tail -20
```

- [ ] **Step 3: Strip removed widgets from `ScenePanel.__init__`**

In `collab_splats/dashboard/panes/visualize.py`, remove from `__init__`:
- `self._dataset_dd = ...`
- `self._backend_dd = ...`
- `self._load_btn = ...`
- `self._frustum_check = ...`
- `self._points_options_row` reference to frustum (keep point size slider)
- All `on_click` / `param.watch` calls for those widgets
- `self._load_thread: threading.Thread | None = None`

Add actor cache fields after `self._result`:
```python
        self._pcd_actor: Any | None = None
        self._mesh_actor: Any | None = None
```

Add state watcher:
```python
        state.param.watch(self._on_feedforward_result_change, "feedforward_result")
```

- [ ] **Step 4: Add `_on_feedforward_result_change`**

Replace the old `_on_load` / `_do_load` with:

```python
    def _on_feedforward_result_change(self, event: Any) -> None:
        """Called on IOLoop thread when state.feedforward_result is set."""
        result = event.new
        if result is None:
            return
        self._result = result
        self._display_result = self._prepare_display_result(result)
        # Reset cached actors — new data, rebuild on next mode render
        self._pcd_actor = None
        self._mesh_actor = None
        self._lifted_normed = None
        self._compressor = None
        self._current_dataset_dir = Path(str(self._state.output_dir)) if self._state.output_dir else None
        self._current_backend = self._state.pointcloud_backend or ""
        self._scan_available_modes()
        n_pts = len(result.points)
        n_disp = len(self._display_result.points)
        status = f"Loaded {n_pts:,} pts"
        if n_disp < n_pts:
            status += f" (display: {n_disp:,})"
        self._set_status(status)

    def _prepare_display_result(self, result: "FeedforwardResult", max_pts: int = 150_000) -> "FeedforwardResult":
        """Return decimated copy for display if result exceeds max_pts; else return unchanged."""
        import dataclasses
        from collab_splats.pointcloud.utils import clean_pcd
        import open3d as o3d

        if len(result.points) <= max_pts:
            return result

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(result.points.astype(np.float64))
        if result.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(result.colors.astype(np.float64) / 255.0)

        _, indices = clean_pcd(
            pcd,
            downsample=True,
            outlier_removal=False,
            distance_removal=False,
            downsample_kwargs={"voxel_size": None, "adaptive": True},
        )
        # If clean_pcd didn't reduce enough, hard cap via random subsample
        if len(indices) > max_pts:
            rng = np.random.default_rng(0)
            indices = rng.choice(indices, size=max_pts, replace=False)

        pts = result.points[indices]
        colors = result.colors[indices] if result.colors is not None else None
        return dataclasses.replace(result, points=pts, colors=colors)
```

Note: `_display_result` is used for PCD/Mesh rendering; `self._result` is kept full for similarity queries.

- [ ] **Step 5: Update actor caching in `_on_mode_change`**

Replace the current `_on_mode_change` body:

```python
    def _on_mode_change(self, new_mode: str) -> None:
        """Switch 3D view mode; use cached actors where available."""
        if self._result is None:
            return

        # Hide all cached actors before showing the new one
        if self._pcd_actor is not None:
            self._pcd_actor.VisibilityOff()
        if self._mesh_actor is not None:
            self._mesh_actor.VisibilityOff()

        self._sim_query_row.visible = new_mode == "Similarity"
        self._points_options_row.visible = new_mode == "Points"

        if new_mode == "Points":
            if self._pcd_actor is None:
                self._plotter.clear()
                self._rebuild_pcd_viewer()
            else:
                self._pcd_actor.VisibilityOn()
        elif new_mode == "Mesh":
            if self._mesh_actor is None:
                self._plotter.clear()
                self._rebuild_mesh_viewer()
            else:
                self._mesh_actor.VisibilityOn()
        elif new_mode == "Similarity":
            if self._lifted_normed is None:
                self._load_lifted_features_for_current_extractor()
            self._rebuild_sim_viewer(colors=None)

        self._vtk_pane.synchronize()
```

- [ ] **Step 6: Store actors in `_rebuild_pcd_viewer` and `_rebuild_mesh_viewer`**

Update `_rebuild_pcd_viewer` to store the returned actor:

```python
    def _rebuild_pcd_viewer(self) -> None:
        """Render points + optional frustums; store actor for caching."""
        if self._display_result is None:
            return
        result_to_render = self._get_render_result()
        point_size = self._point_size_slider.value
        cloud = pointcloud_to_polydata(result_to_render.points, RGB=result_to_render.colors)
        self._pcd_actor = self._plotter.add_mesh(
            cloud, scalars="RGB", rgb=True, point_size=point_size, render_points_as_spheres=False
        )
        if self._state_frustum_enabled:
            self._add_frustums()
```

Update `_rebuild_mesh_viewer` similarly:

```python
    def _rebuild_mesh_viewer(self) -> None:
        """Load and render mesh.ply; store actor for caching."""
        if self._current_dataset_dir is None or not self._current_backend:
            return
        mesh_path = self._current_dataset_dir / self._current_backend / "mesh" / "mesh.ply"
        if not mesh_path.exists():
            self._set_status("mesh.ply not found.")
            return
        mesh = pv.read(str(mesh_path))
        self._mesh_actor = self._plotter.add_mesh(mesh, rgb=True)
```

Add `_get_render_result` helper (used in both PCD and Sim) that returns ground-plane-transformed result:
```python
    def _get_render_result(self) -> "FeedforwardResult":
        """Return display result with ground plane transform applied if enabled."""
        return self._apply_ground_plane(self._display_result)
```

(Ground plane apply is implemented in Task 8.)

- [ ] **Step 7: Fix point size change — no full rebuild**

Replace `_on_point_size_change`:

```python
    def _on_point_size_change(self, event: Any) -> None:
        """Update point size property directly; no full geometry rebuild."""
        if self.mode != "Points" or self._pcd_actor is None:
            return
        self._pcd_actor.GetProperty().SetPointSize(event.new)
        self._vtk_pane.synchronize()
```

- [ ] **Step 8: Fix reset camera**

Replace `_on_reset_camera` (currently `_on_snapshot` area — find the reset btn click handler):

```python
    def _on_reset_camera(self, event: Any) -> None:
        """Reset camera to fit current actors, then synchronize."""
        self._plotter.reset_camera()
        self._vtk_pane.synchronize()
```

Ensure `_reset_btn.on_click(self._on_reset_camera)` is wired (replace any existing wire).

- [ ] **Step 9: Add `_on_frustum_toggle_from_sidebar`**

```python
    def _on_frustum_toggle_from_sidebar(self, enabled: bool) -> None:
        """Called from App sidebar frustum checkbox."""
        self._state_frustum_enabled = enabled
        if self.mode != "Points" or self._result is None:
            return
        self._plotter.clear()
        self._pcd_actor = None
        self._rebuild_pcd_viewer()
        self._vtk_pane.synchronize()
```

Add `self._state_frustum_enabled = False` to `__init__`.

- [ ] **Step 10: Update `panel()` layout — remove controls_row**

At `ScenePanel.panel()` (line 636), remove `controls_row` (dataset + backend + load btn). Keep mode selector, extractor, action row:

```python
    def panel(self) -> pn.Column:
        """Return the Panel layout for this scene viewer."""
        extractor_row = pn.Row(self._extractor_dd)
        action_row = pn.Row(self._reset_btn, self._snapshot_btn)
        return pn.Column(
            pn.Row(self._mode_selector, align="end"),
            self._points_options_row,
            extractor_row,
            self._vtk_pane,
            self._sim_query_row,
            action_row,
            self._status_html,
            sizing_mode="stretch_both",
        )
```

- [ ] **Step 11: Run tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py -v 2>&1 | tail -20
```
Fix any remaining failures from widget removal.

- [ ] **Step 12: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "refactor(dashboard): strip per-pane load controls from ScenePanel; add actor caching + reset camera fix"
```

---

### Task 8: ScenePanel — ground plane apply/invert

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`
- Modify: `tests/dashboard/test_visualize.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/dashboard/test_visualize.py`:

```python
import dataclasses
from unittest import mock


def _make_mock_result(n: int = 20):
    """Return a minimal FeedforwardResult-like object for ground plane tests."""
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    # Build minimal valid FeedforwardResult with zeros for unused fields
    extrinsics = np.eye(4, dtype=np.float64)[np.newaxis].repeat(n, axis=0)
    return FeedforwardResult(
        points=pts,
        extrinsics=extrinsics,
        colors=np.zeros((n, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64)[np.newaxis].repeat(n, axis=0),
        image_paths=[],
        confidence=None,
    )


def test_apply_ground_plane_identity_when_disabled(tmp_path):
    """When ground_plane_R is None, result passes through unchanged."""
    from collab_splats.dashboard.panes.visualize import ScenePanel
    state = AppState()
    sp = ScenePanel(tmp_path, state, OperationLog(), _off_screen=True)
    result = _make_mock_result()
    out = sp._apply_ground_plane(result)
    np.testing.assert_array_equal(out.points, result.points)


def test_apply_ground_plane_transforms_points(tmp_path):
    """When ground_plane_R/t set, points are rotated and translated."""
    from collab_splats.dashboard.panes.visualize import ScenePanel
    state = AppState()
    state.ground_plane_R = np.eye(3)
    state.ground_plane_t = np.array([0.0, 0.0, 1.0])
    state.ground_plane_enabled = True
    sp = ScenePanel(tmp_path, state, OperationLog(), _off_screen=True)
    result = _make_mock_result(5)
    out = sp._apply_ground_plane(result)
    np.testing.assert_allclose(out.points[:, 2], result.points[:, 2] + 1.0, atol=1e-6)


def test_apply_ground_plane_invert_when_disabled(tmp_path):
    """Applying then inverting (enabled=False) round-trips to original."""
    from collab_splats.dashboard.panes.visualize import ScenePanel
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([1.0, 2.0, 3.0])
    state = AppState()
    state.ground_plane_R = R
    state.ground_plane_t = t

    state.ground_plane_enabled = True
    sp = ScenePanel(tmp_path, state, OperationLog(), _off_screen=True)
    result = _make_mock_result(10)
    aligned = sp._apply_ground_plane(result)

    state.ground_plane_enabled = False
    restored = sp._apply_ground_plane(aligned)
    np.testing.assert_allclose(restored.points, result.points, atol=1e-5)
```

- [ ] **Step 2: Run — verify FAIL**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py::test_apply_ground_plane_identity_when_disabled tests/dashboard/test_visualize.py::test_apply_ground_plane_transforms_points tests/dashboard/test_visualize.py::test_apply_ground_plane_invert_when_disabled -v 2>&1 | tail -15
```
Expected: `AttributeError` on `_apply_ground_plane`.

- [ ] **Step 3: Implement `_apply_ground_plane`**

Add to `ScenePanel` in `panes/visualize.py`:

```python
    def _apply_ground_plane(self, result: "FeedforwardResult") -> "FeedforwardResult":
        """Apply (or invert) ground plane transform to points and extrinsics.

        When state.ground_plane_enabled is True, applies R and t.
        When False, applies the inverse transform (R.T, -R.T @ t).
        Returns a new FeedforwardResult via dataclasses.replace — does not mutate input.
        """
        import dataclasses

        R = self._state.ground_plane_R
        t = self._state.ground_plane_t
        if R is None or t is None:
            return result

        if self._state.ground_plane_enabled:
            R_use = R
            t_use = t
        else:
            R_use = R.T
            t_use = -(R.T @ t)

        # Transform points: x_new = R_use @ x + t_use
        pts_new = (R_use @ result.points.T).T + t_use

        # Transform extrinsics (N, 4, 4) world-to-camera: E_new = E @ T_use_inv
        # where T_use = [R_use, t_use; 0, 1], T_use_inv = [R_use.T, -R_use.T@t_use; 0, 1]
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_use.T
        T_inv[:3, 3] = -(R_use.T @ t_use)
        extrinsics_new = result.extrinsics.astype(np.float64) @ T_inv

        return dataclasses.replace(result, points=pts_new.astype(result.points.dtype), extrinsics=extrinsics_new)
```

Also add state watcher for `ground_plane_enabled` in `__init__`:
```python
        state.param.watch(self._on_ground_plane_enabled_change, "ground_plane_enabled")
        state.param.watch(self._on_ground_plane_r_change, "ground_plane_R")
```

And the handlers:
```python
    def _on_ground_plane_enabled_change(self, event: Any) -> None:
        """Re-render when ground plane toggle changes."""
        if self._display_result is None or self.mode not in ("Points", "Mesh", "Similarity"):
            return
        self._plotter.clear()
        self._pcd_actor = None
        self._mesh_actor = None
        self._on_mode_change(self.mode)

    def _on_ground_plane_r_change(self, event: Any) -> None:
        """Re-render when new ground plane R is set (after re-detect)."""
        self._on_ground_plane_enabled_change(event)
```

- [ ] **Step 4: Run — verify PASS**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_visualize.py::test_apply_ground_plane_identity_when_disabled tests/dashboard/test_visualize.py::test_apply_ground_plane_transforms_points tests/dashboard/test_visualize.py::test_apply_ground_plane_invert_when_disabled -v 2>&1 | tail -10
```
Expected: 3 PASSED.

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -15
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(dashboard): ScenePanel ground plane apply/invert via transforms.json R+t"
```

---

### Task 9: SemanticsPane — strip `_method_dd`, watch `state.semantic_extractor`

**Files:**
- Modify: `collab_splats/dashboard/panes/semantics.py`
- Modify: `tests/dashboard/test_semantics.py` (if exists; create otherwise)

- [ ] **Step 1: Find current `_method_dd` usage**

```bash
grep -n "_method_dd\|method_dd" /workspace/collab-splats/collab_splats/dashboard/panes/semantics.py | head -20
```

- [ ] **Step 2: Remove `_method_dd` widget and its watcher**

In `semantics.py`:
- Delete `self._method_dd = pn.widgets.Select(...)` from `__init__`
- Delete any `param.watch` on `_method_dd`
- Add state watcher:

```python
        state.param.watch(self._on_semantic_extractor_change, "semantic_extractor")
```

- [ ] **Step 3: Add `_on_semantic_extractor_change`**

```python
    def _on_semantic_extractor_change(self, event: Any) -> None:
        """Set active extractor when sidebar model selection changes."""
        name = event.new
        if not name:
            return
        # Mirror existing _on_method_change logic — just use name as the method
        # The existing _try_discover_cache already fires on state.output_dir change;
        # here we also trigger it when extractor name changes explicitly.
        if hasattr(self, "_try_discover_cache"):
            self._try_discover_cache()
```

- [ ] **Step 4: Remove `_method_dd` from `panel()` layout**

Find where `_method_dd` appears in the layout (likely in `extractor_row` or top row of SemanticsPane) and remove it. The extractor is now controlled by the sidebar.

- [ ] **Step 5: Smoke test**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import panel as pn
pn.extension('vtk')
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.semantics import SemanticsPane
state = AppState()
sp = SemanticsPane(state=state, op_log=OperationLog())
print('SemanticsPane OK, method_dd removed:', not hasattr(sp, '_method_dd'))
" 2>&1
```
Expected: `SemanticsPane OK, method_dd removed: True`.

- [ ] **Step 6: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -15
```

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/panes/semantics.py
git commit -m "refactor(dashboard): SemanticsPane watches state.semantic_extractor — removes per-pane method_dd"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| `rotation_align_vectors` in `utils/geometry.py` | Task 1 |
| `fit_dominant_plane` in `pointcloud/utils.py` | Task 2 |
| Refactor `align_geometry_floor` | Task 3 |
| AppState 5 new fields | Task 4 |
| Sidebar MODELS: dataset / backend / extractor / Load | Task 5 |
| `_do_load_models` loads FeedforwardResult + sets state | Task 5 |
| transforms.json save/load in load path | Task 6 |
| Sidebar VIEW: ground plane checkbox + re-detect + frustums | Task 6 |
| ScenePanel strips dataset/backend/load/frustum controls | Task 7 |
| Actor caching (PCD + Mesh) | Task 7 |
| Point size without full rebuild | Task 7 |
| Reset camera fix | Task 7 |
| ScenePanel display decimation (150k cap) | Task 7 |
| Ground plane apply/invert via dataclasses.replace | Task 8 |
| SemanticsPane strips `_method_dd` | Task 9 |

All spec items covered. ✓
