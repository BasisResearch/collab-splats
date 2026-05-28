# LocalizePane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `PlaceholderPane("Localize", …)` in the dashboard with a fully functional `LocalizePane` that localizes a query image in a known reconstruction, showing a 50/50 split of correspondence visualization (left) and interactive 3D scene (right).

**Architecture:** Two classes in `collab_splats/dashboard/panes/localize.py`: `LocalizeScenePanel` owns the PyVista VTK 3D viewer; `LocalizePane` owns controls, the matplotlib correspondence PNG pane, and batch mode table. Background threads follow the same pattern as `ReconstructPane`. Data loaded from `feedforward.zarr` via `FeedforwardResult.load_zarr()` or directly from `state.feedforward_result`.

**Tech Stack:** `panel`, `param`, `pyvista`, `matplotlib`, `numpy`, `threading` — all already installed. `CameraLocalizer`, `plot_correspondences` from `collab_splats.pointcloud.localization`. `FeedforwardResult` from `collab_splats.pointcloud.feedforward.base`. `create_camera_frustum_pyvista`, `pointcloud_to_polydata` from `collab_splats.utils.visualization`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/dashboard/panes/localize.py` | **Create** | `LocalizeScenePanel` + `LocalizePane` |
| `collab_splats/dashboard/app.py` | **Modify** | Replace `PlaceholderPane("Localize", …)` |
| `collab_splats/dashboard/__init__.py` | **Modify** | Export `LocalizePane` |
| `tests/dashboard/test_localize_pane.py` | **Create** | Unit tests |

---

### Task 1: `LocalizeScenePanel` — 3D viewer

**Files:**
- Create: `collab_splats/dashboard/panes/localize.py`
- Test: `tests/dashboard/test_localize_pane.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/dashboard/test_localize_pane.py
from unittest.mock import MagicMock, patch
import numpy as np
import panel as pn
import pytest

pn.extension()

def _make_scene_panel():
    from collab_splats.dashboard.panes.localize import LocalizeScenePanel
    pts3d = np.zeros((10, 3), dtype=np.float32)
    extrinsics = np.stack([np.eye(4, dtype=np.float32)] * 4)
    return LocalizeScenePanel(pts3d=pts3d, extrinsics=extrinsics, image_paths=[])

def test_scene_panel_constructs():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
    assert panel is not None

def test_scene_panel_reset_clears_highlight():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel._highlighted_query_idx = 2
        panel._highlighted_ref_idx = 1
        panel.reset()
    assert panel._highlighted_query_idx is None
    assert panel._highlighted_ref_idx is None

def test_scene_panel_highlight_sets_indices():
    with patch("pyvista.Plotter"):
        panel = _make_scene_panel()
        panel.highlight(query_ext=np.eye(4), ref_ext=np.eye(4), query_idx=3, ref_idx=1)
    assert panel._highlighted_query_idx == 3
    assert panel._highlighted_ref_idx == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py -v 2>&1 | head -30
```

Expected: `ImportError` — `localize.py` does not exist yet.

- [ ] **Step 3: Create `localize.py` with `LocalizeScenePanel`**

```python
# collab_splats/dashboard/panes/localize.py
"""LocalizePane — Tab 5 of the unified dashboard.

Camera localization in a known reconstruction: single-image + batch modes.
"""
from __future__ import annotations

import io
import logging
import threading
import traceback
from pathlib import Path
from typing import Any

import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import pyvista as pv

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.state import AppState
from collab_splats.utils.visualization import (
    create_camera_frustum_pyvista,
    pointcloud_to_polydata,
)

logger = logging.getLogger(__name__)

########################################################################
# Colours

_COLOR_DEFAULT = "cornflowerblue"
_COLOR_QUERY   = "tomato"
_COLOR_REF     = "gold"

########################################################################


class LocalizeScenePanel(param.Parameterized):
    """Minimal PyVista 3D viewer for the Localize tab.

    Renders point cloud + camera frustums. Highlights a query/ref pair
    after localization via highlight(); reset() clears the highlight.
    """

    def __init__(
        self,
        pts3d: np.ndarray,
        extrinsics: np.ndarray,
        image_paths: list,
        _off_screen: bool = False,
        **params: Any,
    ):
        """Build scene from pts3d (P,3), extrinsics (N,4,4), image_paths length-N."""
        super().__init__(**params)
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._image_paths = image_paths
        self._highlighted_query_idx: int | None = None
        self._highlighted_ref_idx: int | None = None

        self._plotter = pv.Plotter(off_screen=_off_screen, notebook=False)
        self._vtk_pane = pn.pane.VTK(
            self._plotter.ren_win,
            sizing_mode="stretch_both",
            min_height=300,
        )
        self._build_scene()

    def _build_scene(self) -> None:
        """Render point cloud + all camera frustums at default colour."""
        self._plotter.clear()

        # Point cloud
        if len(self._pts3d) > 0:
            cloud = pointcloud_to_polydata(self._pts3d)
            self._plotter.add_mesh(cloud, color="lightgray", point_size=2, render_points_as_spheres=True)

        # Camera frustums — store actors so we can recolour on highlight
        self._frustum_actors: list[Any] = []
        for ext in self._extrinsics:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(ext))
            actor = self._plotter.add_mesh(frustum, color=_COLOR_DEFAULT, line_width=1)
            self._frustum_actors.append(actor)

        self._plotter.reset_camera()

    def highlight(
        self,
        query_ext: np.ndarray,
        ref_ext: np.ndarray,
        query_idx: int,
        ref_idx: int,
    ) -> None:
        """Highlight query camera (red) and ref camera (yellow); grey out others."""
        self._highlighted_query_idx = query_idx
        self._highlighted_ref_idx = ref_idx

        for i, actor in enumerate(self._frustum_actors):
            if i == query_idx:
                actor.GetProperty().SetColor(*_rgb(_COLOR_QUERY))
                actor.GetProperty().SetLineWidth(3)
            elif i == ref_idx:
                actor.GetProperty().SetColor(*_rgb(_COLOR_REF))
                actor.GetProperty().SetLineWidth(3)
            else:
                actor.GetProperty().SetColor(*_rgb(_COLOR_DEFAULT))
                actor.GetProperty().SetOpacity(0.25)
                actor.GetProperty().SetLineWidth(1)

        self._vtk_pane.param.trigger("object")

    def reset(self) -> None:
        """Clear highlight; restore all cameras to default colour."""
        self._highlighted_query_idx = None
        self._highlighted_ref_idx = None
        for actor in self._frustum_actors:
            actor.GetProperty().SetColor(*_rgb(_COLOR_DEFAULT))
            actor.GetProperty().SetOpacity(1.0)
            actor.GetProperty().SetLineWidth(1)
        self._vtk_pane.param.trigger("object")

    def panel(self) -> pn.pane.VTK:
        """Return the VTK Panel pane."""
        return self._vtk_pane


def _rgb(name: str) -> tuple[float, float, float]:
    """Convert matplotlib colour name to (r, g, b) 0–1 floats for VTK."""
    import matplotlib.colors as mc
    return mc.to_rgb(name)
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py::test_scene_panel_constructs tests/dashboard/test_localize_pane.py::test_scene_panel_reset_clears_highlight tests/dashboard/test_localize_pane.py::test_scene_panel_highlight_sets_indices -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/localize.py tests/dashboard/test_localize_pane.py
git commit -m "feat(dashboard): LocalizeScenePanel — PyVista 3D viewer with highlight/reset

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `LocalizePane` skeleton — controls + layout + AppState wiring

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py` (add `LocalizePane`)
- Modify: `tests/dashboard/test_localize_pane.py` (add pane tests)

- [ ] **Step 1: Write failing tests**

Append to `tests/dashboard/test_localize_pane.py`:

```python
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.panes.localize import LocalizePane
from collab_splats.dashboard.state import AppState

def _make_pane():
    state = AppState()
    op_log = OperationLog()
    return LocalizePane(state=state, op_log=op_log), state, op_log

def test_localize_pane_run_btn_disabled_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._run_btn.disabled is True

def test_localize_pane_run_btn_still_disabled_without_query_image(tmp_path):
    # output_dir set but no query image — run still disabled
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert pane._run_btn.disabled is True

def test_localize_pane_method_dropdown_empty_without_output_dir():
    pane, _, _ = _make_pane()
    assert pane._method_dd.options == []

def test_localize_pane_method_dropdown_populated_when_zarr_found(tmp_path):
    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()
    pane, state, _ = _make_pane()
    state.output_dir = tmp_path
    assert "vggtx" in pane._method_dd.options
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py -k "localize_pane" -v 2>&1 | head -20
```

Expected: `ImportError` for `LocalizePane`.

- [ ] **Step 3: Add `LocalizePane` skeleton to `localize.py`**

Append after `_rgb()` in `localize.py`:

```python
########################################################################


def _scan_recon_methods(output_dir: Path) -> list[str]:
    """Return method names whose feedforward.zarr exists under output_dir."""
    if not output_dir or not output_dir.is_dir():
        return []
    return sorted(
        p.name for p in output_dir.iterdir()
        if p.is_dir() and (p / "feedforward.zarr").exists()
    )


class LocalizePane(param.Parameterized):
    """Camera localization tab — single-image + batch modes.

    Left panel: plot_correspondences() matplotlib PNG.
    Right panel: LocalizeScenePanel PyVista 3D viewer.
    """

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._localizer = None   # CameraLocalizer, built on demand
        self._ff_result = None   # FeedforwardResult currently loaded
        self._scene_panel: LocalizeScenePanel | None = None
        self._loc_thread: threading.Thread | None = None
        self._batch_thread: threading.Thread | None = None

        # Controls
        self._query_input = pn.widgets.TextInput(
            placeholder="Path to query image…",
            width=320,
        )
        self._browse_btn = pn.widgets.Button(name="Browse…", width=80)
        self._method_dd = pn.widgets.Select(
            name="Recon",
            options=[],
            width=120,
        )
        self._extractor_dd = pn.widgets.Select(
            name="Extractor",
            options=["DISK+LightGlue", "XFeat+MNN"],
            value="DISK+LightGlue",
            width=140,
        )
        self._run_btn = pn.widgets.Button(
            name="▶ Localize",
            button_type="success",
            disabled=True,
            width=100,
        )
        self._warp_cb = pn.widgets.Checkbox(name="warp corners", value=True)
        self._status_html = pn.pane.HTML("", width=400)

        # Correspondence display (left panel)
        self._corr_png = pn.pane.PNG(
            object=None,
            sizing_mode="stretch_both",
            min_height=200,
        )
        self._corr_info = pn.pane.HTML(
            "<span style='color:#666;font-size:12px'>No result yet</span>",
        )

        # Batch mode widgets
        self._batch_folder_input = pn.widgets.TextInput(
            placeholder="Path to query image folder…",
            width=280,
        )
        self._batch_run_btn = pn.widgets.Button(
            name="▶ Run batch",
            button_type="primary",
            disabled=True,
            width=100,
        )
        self._batch_export_btn = pn.widgets.Button(
            name="⬇ Export CSV",
            disabled=True,
            width=110,
        )
        self._batch_table = pn.widgets.Tabulator(
            value=_empty_batch_df(),
            show_index=False,
            sizing_mode="stretch_width",
            height=180,
        )

        # Wire callbacks
        self._run_btn.on_click(self._on_run)
        self._batch_run_btn.on_click(self._on_batch_run)
        self._batch_export_btn.on_click(self._on_export_csv)
        self._query_input.param.watch(self._on_query_or_dir_changed, ["value"])
        self._state.param.watch(self._on_output_dir_changed, ["output_dir"])

        # Initial gate state
        self._on_output_dir_changed(None)

    # ------------------------------------------------------------------
    # Gate helpers

    def _on_output_dir_changed(self, event: Any) -> None:
        """Rescan method dropdown; re-evaluate run-button gate."""
        output_dir = self._state.output_dir
        methods = _scan_recon_methods(Path(output_dir)) if output_dir else []
        self._method_dd.options = methods
        if methods:
            self._method_dd.value = methods[0]
        self._batch_run_btn.disabled = not bool(methods)
        self._update_run_btn_gate()

    def _on_query_or_dir_changed(self, event: Any) -> None:
        self._update_run_btn_gate()

    def _update_run_btn_gate(self) -> None:
        """Enable run only when output_dir set, method available, and query path non-empty."""
        has_dir = self._state.output_dir is not None
        has_method = bool(self._method_dd.options)
        has_query = bool(self._query_input.value and self._query_input.value.strip())
        self._run_btn.disabled = not (has_dir and has_method and has_query)

    # ------------------------------------------------------------------
    # Stubs (filled in Task 3 and Task 4)

    def _on_run(self, event: Any) -> None:
        pass

    def _on_batch_run(self, event: Any) -> None:
        pass

    def _on_export_csv(self, event: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Layout

    def panel(self) -> pn.viewable.Viewable:
        """Return the full LocalizePane layout."""
        controls_bar = pn.Row(
            self._query_input,
            self._browse_btn,
            pn.Spacer(sizing_mode="stretch_width"),
            pn.pane.HTML("<b style='color:#8b949e;font-size:12px'>Recon:</b>"),
            self._method_dd,
            self._extractor_dd,
            self._run_btn,
            sizing_mode="stretch_width",
            margin=(4, 0),
        )

        corr_header = pn.Row(
            self._corr_info,
            pn.Spacer(sizing_mode="stretch_width"),
            self._warp_cb,
            margin=(0, 0, 4, 0),
        )

        left_panel = pn.Column(
            corr_header,
            self._corr_png,
            self._status_html,
            sizing_mode="stretch_both",
        )

        # Right panel: placeholder until first localization builds the scene
        right_placeholder = pn.pane.HTML(
            "<div style='display:flex;align-items:center;justify-content:center;"
            "height:100%;color:#555;font-size:13px'>Run localization to see 3D scene</div>",
            sizing_mode="stretch_both",
            min_height=300,
        )
        self._right_col = pn.Column(right_placeholder, sizing_mode="stretch_both")

        split = pn.Row(
            pn.Column(left_panel, sizing_mode="stretch_both", width_policy="max"),
            pn.Column(self._right_col, sizing_mode="stretch_both", width_policy="max"),
            sizing_mode="stretch_width",
            min_height=360,
        )

        batch_section = pn.Card(
            pn.Column(
                pn.Row(
                    self._batch_folder_input,
                    self._batch_run_btn,
                    self._batch_export_btn,
                    margin=(4, 0),
                ),
                self._batch_table,
                sizing_mode="stretch_width",
            ),
            title="Batch mode",
            collapsed=True,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            controls_bar,
            split,
            batch_section,
            sizing_mode="stretch_width",
        )


def _empty_batch_df():
    import pandas as pd
    return pd.DataFrame(columns=["image", "inliers", "status", "t-err (m)", "pose t"])
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py -k "localize_pane" -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/localize.py tests/dashboard/test_localize_pane.py
git commit -m "feat(dashboard): LocalizePane skeleton — controls, layout, AppState wiring

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Single-image localization flow

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py` (fill `_on_run`)
- Modify: `tests/dashboard/test_localize_pane.py`

- [ ] **Step 1: Write failing test**

Append to `tests/dashboard/test_localize_pane.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch

def test_run_localize_calls_localizer_and_updates_corr_png(tmp_path):
    """_run_localize() calls localizer.localize() and updates _corr_png."""
    from collab_splats.pointcloud.localization import LocalizationResult

    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()

    pane, state, op_log = _make_pane()
    state.output_dir = tmp_path
    pane._query_input.value = str(tmp_path / "query.jpg")

    # Create a fake query image file
    import cv2
    fake_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "query.jpg"), fake_img)

    mock_ff = MagicMock()
    mock_ff.points = np.zeros((5, 3), dtype=np.float32)
    mock_ff.extrinsics = np.stack([np.eye(4)] * 3).astype(np.float32)
    mock_ff.intrinsics = np.stack([np.eye(3)] * 3).astype(np.float32)
    mock_ff.image_paths = [tmp_path / f"f{i}.jpg" for i in range(3)]

    loc_result = LocalizationResult(
        pts2d=np.zeros((10, 2), dtype=np.float32),
        pts3d_matched=np.zeros((10, 3), dtype=np.float32),
        inlier_mask=np.ones(10, dtype=bool),
        pose=np.eye(4, dtype=np.float32),
        pts2d_ref=np.zeros((10, 2), dtype=np.float32),
        ref_frame_indices=np.zeros(10, dtype=np.int32),
    )

    mock_localizer = MagicMock()
    mock_localizer.localize.return_value = loc_result

    with patch("collab_splats.dashboard.panes.localize.FeedforwardResult") as MockFF, \
         patch("collab_splats.dashboard.panes.localize.CameraLocalizer") as MockCL, \
         patch("collab_splats.dashboard.panes.localize.LocalizeScenePanel"), \
         patch("collab_splats.dashboard.panes.localize.plot_correspondences"):
        MockFF.load_zarr.return_value = mock_ff
        MockCL.from_feedforward.return_value = mock_localizer
        pane._run_localize(method="vggtx", extractor_name="DISK+LightGlue",
                           query_path=tmp_path / "query.jpg", warp_corners=False)

    mock_localizer.localize.assert_called_once()
```

- [ ] **Step 2: Run to verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py::test_run_localize_calls_localizer_and_updates_corr_png -v 2>&1 | head -20
```

Expected: FAIL — `_run_localize` does not exist.

- [ ] **Step 3: Implement `_on_run` and `_run_localize` in `LocalizePane`**

Replace the stub `_on_run` and add new methods. Edit `localize.py`:

```python
    # ------------------------------------------------------------------
    # Single-image flow

    def _on_run(self, event: Any) -> None:
        """Spawn background localization thread on button click."""
        if self._loc_thread and self._loc_thread.is_alive():
            return
        # Snapshot widget values on main thread
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        query_path = Path(self._query_input.value.strip())
        warp_corners = self._warp_cb.value
        self._run_btn.disabled = True
        self._status_html.object = "<span style='color:#2596be'>⏳ Localizing…</span>"
        self._loc_thread = threading.Thread(
            target=self._run_localize,
            args=(method, extractor_name, query_path, warp_corners),
            daemon=True,
        )
        self._loc_thread.start()

    def _run_localize(
        self,
        method: str,
        extractor_name: str,
        query_path: Path,
        warp_corners: bool,
    ) -> None:
        """Background thread: load result, build localizer, run, update UI."""
        try:
            import cv2
            from collab_splats.pointcloud.feedforward.base import FeedforwardResult
            from collab_splats.pointcloud.localization import (
                CameraLocalizer,
                DiskExtractor,
                XFeatExtractor,
                plot_correspondences,
            )

            output_dir = Path(self._state.output_dir)
            zarr_path = output_dir / method / "feedforward.zarr"
            self._op_log.start_op(f"Localizing in {method}")

            # Load feedforward result from zarr
            ff = FeedforwardResult.load_zarr(zarr_path)
            self._ff_result = ff

            # Build extractor
            extractor = XFeatExtractor() if "XFeat" in extractor_name else DiskExtractor()

            # Build or reuse localizer
            from collab_splats.pointcloud.localization import CameraLocalizer
            localizer = CameraLocalizer.from_feedforward(ff, extractor=extractor)
            self._localizer = localizer

            # Build scene panel if not yet built or method changed
            if self._scene_panel is None:
                self._scene_panel = LocalizeScenePanel(
                    pts3d=ff.points,
                    extrinsics=ff.extrinsics,
                    image_paths=ff.image_paths,
                )
                self._right_col[:] = [self._scene_panel.panel()]

            # Load query image
            bgr = cv2.imread(str(query_path))
            if bgr is None:
                raise FileNotFoundError(f"Cannot read query image: {query_path}")
            query_img = bgr[..., ::-1].copy()  # BGR → RGB

            # Use mean reference intrinsics as query intrinsics (same camera assumed)
            query_intrinsics = ff.intrinsics.mean(axis=0)

            # Run localization
            loc = localizer.localize(query_img, query_intrinsics)

            self._op_log.finish_op()

            if loc.pose is None:
                n_inliers = int(loc.inlier_mask.sum()) if loc.inlier_mask is not None else 0
                self._corr_info.object = (
                    f"<span style='color:#f85149'>✗ Failed — {n_inliers} inliers</span>"
                )
                self._status_html.object = "<span style='color:#e05050'>✗ Localization failed</span>"
                self._op_log.error_op(f"Localization failed: {n_inliers} inliers")
            else:
                # Inlier stats + best ref frame
                inlier_frames = loc.ref_frame_indices[loc.inlier_mask]
                best_ref_idx = int(np.bincount(inlier_frames.astype(np.intp)).argmax())
                n_inliers = int(loc.inlier_mask.sum())
                n_total = int(loc.inlier_mask.shape[0])
                ref_name = Path(ff.image_paths[best_ref_idx]).name

                self._corr_info.object = (
                    f"<span style='color:#3fb950'>● {n_inliers} inliers</span> &nbsp;"
                    f"<span style='color:#f85149'>● {n_total - n_inliers} outliers</span> &nbsp;| "
                    f"&nbsp;best ref: <span style='color:#e3b341'>{ref_name}</span>"
                )

                # Render correspondence PNG
                buf = _render_correspondences_to_png(
                    loc, query_img, ff.image_paths, warp_corners
                )
                if buf is not None:
                    self._corr_png.object = buf

                # Highlight cameras in 3D
                t = loc.pose[:3, 3]
                self._status_html.object = (
                    f"<span style='color:#3fb950'>✓ pose t=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]</span>"
                )
                self._scene_panel.highlight(
                    query_ext=loc.pose,
                    ref_ext=ff.extrinsics[best_ref_idx],
                    query_idx=-1,      # query not in extrinsics array — add at end
                    ref_idx=best_ref_idx,
                )

        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("LocalizePane: localization failed")
            self._op_log.error_op(str(exc))
            self._status_html.object = f"<span style='color:#e05050'>✗ {exc}</span>"
        finally:
            self._run_btn.disabled = False
```

Also add the helper function before `LocalizePane`:

```python
def _render_correspondences_to_png(
    loc: Any,
    query_img: np.ndarray,
    image_paths: list,
    warp_corners: bool,
) -> bytes | None:
    """Call plot_correspondences() and capture matplotlib output as PNG bytes."""
    from collab_splats.pointcloud.localization import plot_correspondences

    buf = io.BytesIO()
    try:
        fig = plt.figure(figsize=(10, 4))
        plot_correspondences(loc, query_img, image_paths, warp_corners=warp_corners)
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close("all")
        buf.seek(0)
        return buf.read()
    except Exception:
        plt.close("all")
        logger.exception("plot_correspondences failed")
        return None
```

Also add to top-level imports in `localize.py` (inside function body to avoid heavy load at import time — note: this matches `reconstruct.py` pattern of importing heavy deps inside the thread function).

> **Note on `query_idx=-1`:** The query camera is not in the reference extrinsics array. `LocalizeScenePanel.highlight()` needs to handle `query_idx == -1` specially: add a new red frustum for `query_ext` instead of recolouring an existing actor. Update `LocalizeScenePanel.highlight()`:

```python
    def highlight(
        self,
        query_ext: np.ndarray,
        ref_ext: np.ndarray,
        query_idx: int,
        ref_idx: int,
    ) -> None:
        """Highlight query camera (red) and ref camera (yellow); grey out others.

        When query_idx == -1, query_ext is not in self._extrinsics — add it
        as a temporary actor instead of recolouring an existing one.
        """
        self._highlighted_query_idx = query_idx
        self._highlighted_ref_idx = ref_idx

        # Remove previous temporary query actor if any
        if hasattr(self, "_query_actor") and self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None

        # Recolour reference frustums
        for i, actor in enumerate(self._frustum_actors):
            if i == ref_idx:
                actor.GetProperty().SetColor(*_rgb(_COLOR_REF))
                actor.GetProperty().SetLineWidth(3)
                actor.GetProperty().SetOpacity(1.0)
            else:
                actor.GetProperty().SetColor(*_rgb(_COLOR_DEFAULT))
                actor.GetProperty().SetOpacity(0.25)
                actor.GetProperty().SetLineWidth(1)

        # Add query camera as temporary red frustum
        if query_idx == -1:
            frustum = create_camera_frustum_pyvista(np.linalg.inv(query_ext))
            self._query_actor = self._plotter.add_mesh(
                frustum, color=_COLOR_QUERY, line_width=3
            )
        else:
            self._frustum_actors[query_idx].GetProperty().SetColor(*_rgb(_COLOR_QUERY))
            self._frustum_actors[query_idx].GetProperty().SetLineWidth(3)
            self._frustum_actors[query_idx].GetProperty().SetOpacity(1.0)
            self._query_actor = None

        self._vtk_pane.param.trigger("object")
```

Also initialise `self._query_actor = None` in `LocalizeScenePanel.__init__`.

- [ ] **Step 4: Run test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py::test_run_localize_calls_localizer_and_updates_corr_png -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/panes/localize.py tests/dashboard/test_localize_pane.py
git commit -m "feat(dashboard): LocalizePane single-image flow — thread, correspondence PNG, 3D highlight

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Batch mode

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py` (fill `_on_batch_run`, `_on_export_csv`)
- Modify: `tests/dashboard/test_localize_pane.py`

- [ ] **Step 1: Write failing test**

Append to `tests/dashboard/test_localize_pane.py`:

```python
def test_batch_run_populates_table(tmp_path):
    """_run_batch() appends one row per image to _batch_table."""
    from collab_splats.pointcloud.localization import LocalizationResult

    (tmp_path / "vggtx").mkdir()
    (tmp_path / "vggtx" / "feedforward.zarr").mkdir()

    # Create fake query images
    import cv2
    for name in ["q1.jpg", "q2.jpg"]:
        cv2.imwrite(str(tmp_path / name), np.zeros((100, 100, 3), dtype=np.uint8))

    pane, state, _ = _make_pane()
    state.output_dir = tmp_path

    mock_ff = MagicMock()
    mock_ff.points = np.zeros((5, 3), dtype=np.float32)
    mock_ff.extrinsics = np.stack([np.eye(4)] * 3).astype(np.float32)
    mock_ff.intrinsics = np.stack([np.eye(3)] * 3).astype(np.float32)
    mock_ff.image_paths = [tmp_path / f"f{i}.jpg" for i in range(3)]

    success_loc = LocalizationResult(
        pts2d=np.zeros((10, 2), dtype=np.float32),
        pts3d_matched=np.zeros((10, 3), dtype=np.float32),
        inlier_mask=np.ones(10, dtype=bool),
        pose=np.eye(4, dtype=np.float32),
        pts2d_ref=np.zeros((10, 2), dtype=np.float32),
        ref_frame_indices=np.zeros(10, dtype=np.int32),
    )

    mock_localizer = MagicMock()
    mock_localizer.localize.return_value = success_loc

    with patch("collab_splats.dashboard.panes.localize.FeedforwardResult") as MockFF, \
         patch("collab_splats.dashboard.panes.localize.CameraLocalizer") as MockCL, \
         patch("collab_splats.dashboard.panes.localize.LocalizeScenePanel"):
        MockFF.load_zarr.return_value = mock_ff
        MockCL.from_feedforward.return_value = mock_localizer
        pane._run_batch(
            method="vggtx",
            extractor_name="DISK+LightGlue",
            folder_path=tmp_path,
        )

    assert len(pane._batch_table.value) == 2
    assert list(pane._batch_table.value["status"]) == ["✓", "✓"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py::test_batch_run_populates_table -v 2>&1 | head -20
```

Expected: FAIL — `_run_batch` not implemented.

- [ ] **Step 3: Implement batch methods in `LocalizePane`**

Replace the `_on_batch_run` and `_on_export_csv` stubs, and add `_run_batch`:

```python
    # ------------------------------------------------------------------
    # Batch flow

    def _on_batch_run(self, event: Any) -> None:
        """Spawn background batch thread."""
        if self._batch_thread and self._batch_thread.is_alive():
            return
        method = self._method_dd.value
        extractor_name = self._extractor_dd.value
        folder_path = Path(self._batch_folder_input.value.strip())
        self._batch_run_btn.disabled = True
        self._batch_export_btn.disabled = True
        import pandas as pd
        self._batch_table.value = _empty_batch_df()
        self._batch_thread = threading.Thread(
            target=self._run_batch,
            args=(method, extractor_name, folder_path),
            daemon=True,
        )
        self._batch_thread.start()

    def _run_batch(
        self,
        method: str,
        extractor_name: str,
        folder_path: Path,
    ) -> None:
        """Background thread: localize every image in folder_path, stream rows."""
        import pandas as pd
        from collab_splats.pointcloud.feedforward.base import FeedforwardResult
        from collab_splats.pointcloud.localization import (
            CameraLocalizer,
            DiskExtractor,
            XFeatExtractor,
        )
        import cv2

        try:
            output_dir = Path(self._state.output_dir)
            zarr_path = output_dir / method / "feedforward.zarr"
            ff = FeedforwardResult.load_zarr(zarr_path)

            extractor = XFeatExtractor() if "XFeat" in extractor_name else DiskExtractor()
            localizer = CameraLocalizer.from_feedforward(ff, extractor=extractor)
            query_intrinsics = ff.intrinsics.mean(axis=0)

            # Collect image paths
            img_exts = {".jpg", ".jpeg", ".png"}
            query_paths = sorted(
                p for p in folder_path.iterdir()
                if p.suffix.lower() in img_exts
            )

            rows: list[dict] = []
            for qp in query_paths:
                bgr = cv2.imread(str(qp))
                if bgr is None:
                    rows.append({
                        "image": qp.name, "inliers": 0, "status": "✗",
                        "t-err (m)": "—", "pose t": "—",
                    })
                    self._batch_table.value = pd.DataFrame(rows)
                    continue

                query_img = bgr[..., ::-1].copy()
                try:
                    loc = localizer.localize(query_img, query_intrinsics)
                except Exception as exc:
                    rows.append({
                        "image": qp.name, "inliers": 0, "status": "✗",
                        "t-err (m)": "—", "pose t": str(exc)[:40],
                    })
                    self._batch_table.value = pd.DataFrame(rows)
                    continue

                if loc.pose is None:
                    n_in = int(loc.inlier_mask.sum()) if loc.inlier_mask is not None else 0
                    rows.append({
                        "image": qp.name, "inliers": n_in, "status": "✗",
                        "t-err (m)": "—", "pose t": "—",
                    })
                else:
                    n_in = int(loc.inlier_mask.sum())
                    t = loc.pose[:3, 3]
                    rows.append({
                        "image": qp.name,
                        "inliers": n_in,
                        "status": "✓",
                        "t-err (m)": "—",   # GT not available unless sidecar found
                        "pose t": f"[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]",
                    })

                self._batch_table.value = pd.DataFrame(rows)

            self._batch_export_btn.disabled = False

        except Exception as exc:
            logger.exception("LocalizePane: batch failed")
            self._op_log.error_op(str(exc))
        finally:
            self._batch_run_btn.disabled = False

    def _on_export_csv(self, event: Any) -> None:
        """Trigger CSV download from Tabulator."""
        self._batch_table.download(filename="localize_batch.csv")
```

- [ ] **Step 4: Run test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py::test_batch_run_populates_table -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_localize_pane.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/panes/localize.py tests/dashboard/test_localize_pane.py
git commit -m "feat(dashboard): LocalizePane batch mode — folder run, streaming table, CSV export

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Wire `LocalizePane` into `App` + export

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/dashboard/__init__.py`

- [ ] **Step 1: Replace `PlaceholderPane` in `app.py`**

In `collab_splats/dashboard/app.py`, add the import:

```python
from collab_splats.dashboard.panes.localize import LocalizePane
```

Then replace:

```python
# old — line 61
"Localize": PlaceholderPane("Localize", "Coming in Phase 5 — camera localization in known scene"),
```

with:

```python
"Localize": LocalizePane(state=self._state, op_log=self._op_log),
```

- [ ] **Step 2: Update `__init__.py`**

In `collab_splats/dashboard/__init__.py`, add:

```python
from collab_splats.dashboard.panes.localize import LocalizePane
```

And add `"LocalizePane"` to `__all__`:

```python
__all__ = ["App", "run_app", "AppState", "OperationLog", "SemanticsPane", "LocalizePane"]
```

- [ ] **Step 3: Run smoke tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_smoke.py tests/dashboard/test_app.py tests/dashboard/test_localize_pane.py -v
```

Expected: all PASS (no import errors, no broken pane construction).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/__init__.py
git commit -m "feat(dashboard): wire LocalizePane into App — replace Phase 5 placeholder

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| 50/50 split with gap | Task 2 `panel()` — `pn.Row` with two equal `width_policy="max"` columns |
| Correspondence PNG left panel | Task 3 `_render_correspondences_to_png` + `pn.pane.PNG` |
| `warp_corners` checkbox in panel header | Task 2 `corr_header` row |
| 3D PyVista VTK right panel | Task 1 `LocalizeScenePanel` |
| Query (red) + ref (yellow) frustums | Task 1 `highlight()` + `_COLOR_QUERY`/`_COLOR_REF` |
| Dashed connector between cameras | Not yet included — `LocalizeScenePanel` draws frustums but no explicit dashed line actor. **Fix:** Add connector in `highlight()` |
| Recon method dropdown (current session + others) | Task 2 `_scan_recon_methods` + `_method_dd` |
| DISK+LightGlue / XFeat+MNN extractor choice | Task 3 `_run_localize` extractor selection |
| Batch mode table: image, inliers, status, t-err, pose t | Task 4 `_run_batch` row dict |
| CSV export | Task 4 `_on_export_csv` |
| AppState integration — pane enables on `output_dir` | Task 2 `_on_output_dir_changed` |
| Error handling: no output_dir | Task 2 — run btn disabled |
| Error handling: no method zarr | Task 2 — `_method_dd.options == []` |
| Error handling: pose=None | Task 3 — shows failure message |
| Error handling: batch per-image failure | Task 4 — row shows ✗, continues |

**Missing: dashed connector line** — add to `LocalizeScenePanel.highlight()` in Task 3 Step 3. After adding the query actor, add:

```python
        # Dashed connector line between query and ref camera centres
        if hasattr(self, "_connector_actor") and self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
        q_pos = np.linalg.inv(query_ext)[:3, 3]
        r_pos = np.linalg.inv(ff_extrinsics_ref)[:3, 3]  # need ref_ext param
        line = pv.Line(q_pos, r_pos)
        self._connector_actor = self._plotter.add_mesh(
            line, color=_COLOR_REF, line_width=2, style="wireframe"
        )
```

Wait — `highlight()` receives `ref_ext` already. Fix the connector implementation to use it:

In `LocalizeScenePanel.highlight()`, after adding the query actor, append:

```python
        # Draw dashed connector between query and ref camera centres
        if hasattr(self, "_connector_actor") and self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
        q_pos = np.linalg.inv(query_ext)[:3, 3]
        r_pos = np.linalg.inv(ref_ext)[:3, 3]
        line = pv.Line(q_pos.tolist(), r_pos.tolist())
        self._connector_actor = self._plotter.add_mesh(
            line, color=_COLOR_REF, line_width=2
        )
```

Also initialise `self._connector_actor = None` in `__init__`. And `reset()` should also remove it:

```python
    def reset(self) -> None:
        if hasattr(self, "_connector_actor") and self._connector_actor is not None:
            self._plotter.remove_actor(self._connector_actor)
            self._connector_actor = None
        if hasattr(self, "_query_actor") and self._query_actor is not None:
            self._plotter.remove_actor(self._query_actor)
            self._query_actor = None
        ...
```

These fixes should be incorporated into Task 1 Step 3 when implementing `LocalizeScenePanel`.

**Placeholder scan:** No TBD/TODO/incomplete sections found in the plan.

**Type consistency:** `LocalizeScenePanel.highlight(query_ext, ref_ext, query_idx, ref_idx)` called consistently across Task 1 and Task 3. `_run_localize` and `_run_batch` use `FeedforwardResult`, `CameraLocalizer`, `DiskExtractor`, `XFeatExtractor` — all from `collab_splats.pointcloud.localization`. ✓
