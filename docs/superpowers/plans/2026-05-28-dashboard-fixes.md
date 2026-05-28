# Dashboard Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four bugs/UX issues in the dashboard — base_dir selector, load-existing video display, semantics single-extractor three-panel, and visualize layout reorganization.

**Architecture:** Each fix is scoped to one or two files. No new files. Fixes 1–2 are in `app.py`/`preprocess.py`; Fix 3 rewrites `panes/semantics.py`; Fix 4 rewrites widget layout in `panes/visualize.py`.

**Tech Stack:** Panel + param (existing). No new deps.

---

## Files modified

| File | Fix |
|---|---|
| `collab_splats/dashboard/app.py` | Fix 1 (dir scanner + Select), Fix 2 (tabs ref + tab switch) |
| `collab_splats/dashboard/panes/preprocess.py` | Fix 2 (`pn.state.execute` wrap) |
| `collab_splats/dashboard/panes/semantics.py` | Fix 3 (single-extractor three-panel rewrite) |
| `collab_splats/dashboard/panes/visualize.py` | Fix 4 (divider, RadioButtonGroup, per-scene query, frustum checkbox) |
| `tests/dashboard/test_app.py` | Fix 1 + 2 tests |
| `tests/dashboard/test_semantics.py` | Fix 3 tests |
| `tests/dashboard/test_visualize.py` | Fix 4 tests (update stale shared-query-bar tests) |

**Test runner:** `python -m pytest tests/dashboard/ -v`
(`python` = `/opt/conda/envs/reconstruction/bin/python`)

---

## Task 1: Fix 1 — base_dir directory scanner + Select widget

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Test: `tests/dashboard/test_app.py`

- [ ] **Step 1.1: Write failing tests for `_scan_output_dirs`**

Add to `tests/dashboard/test_app.py`:

```python
from pathlib import Path
from collab_splats.dashboard.app import _scan_output_dirs


def test_scan_output_dirs_returns_dirs_with_config(tmp_path):
    (tmp_path / "birds_c0043").mkdir()
    (tmp_path / "birds_c0043" / "run_config.yaml").write_text("video_path: /foo.mp4")
    (tmp_path / "empty_dir").mkdir()
    result = _scan_output_dirs(tmp_path)
    assert result == ["birds_c0043"]


def test_scan_output_dirs_sorted(tmp_path):
    for name in ["zoo", "alpha", "beta"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "run_config.yaml").write_text("")
    result = _scan_output_dirs(tmp_path)
    assert result == ["alpha", "beta", "zoo"]


def test_scan_output_dirs_empty_when_no_configs(tmp_path):
    (tmp_path / "no_config").mkdir()
    result = _scan_output_dirs(tmp_path)
    assert result == []


def test_scan_output_dirs_missing_base(tmp_path):
    result = _scan_output_dirs(tmp_path / "nonexistent")
    assert result == []
```

- [ ] **Step 1.2: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_app.py::test_scan_output_dirs_returns_dirs_with_config -v
```

Expected: `ImportError` or `AttributeError` — `_scan_output_dirs` not yet defined.

- [ ] **Step 1.3: Add `_scan_output_dirs` to `app.py`**

Open `collab_splats/dashboard/app.py`. After the existing imports and before the `_HEADER_CSS` constant, add:

```python
def _scan_output_dirs(base_dir: Path) -> list[str]:
    """Return sorted names of subdirs in base_dir that contain run_config.yaml."""
    if not base_dir.is_dir():
        return []
    return sorted(
        p.name for p in base_dir.iterdir()
        if p.is_dir() and (p / "run_config.yaml").exists()
    )
```

- [ ] **Step 1.4: Run scanner tests — expect pass**

```bash
python -m pytest tests/dashboard/test_app.py -k "scan_output_dirs" -v
```

Expected: all 4 pass.

- [ ] **Step 1.5: Write failing test for Select widget in sidebar**

Add to `tests/dashboard/test_app.py`:

```python
import panel as pn
from collab_splats.dashboard.app import App


def test_load_existing_sidebar_uses_select_widget(tmp_path):
    (tmp_path / "scene_01").mkdir()
    (tmp_path / "scene_01" / "run_config.yaml").write_text("")
    app = App(base_dir=str(tmp_path))
    app._build_sidebar()
    assert isinstance(app._output_dir_select, pn.widgets.Select)
    assert "scene_01" in app._output_dir_select.options


def test_load_existing_sidebar_has_refresh_button(tmp_path):
    app = App(base_dir=str(tmp_path))
    app._build_sidebar()
    assert isinstance(app._refresh_dirs_btn, pn.widgets.Button)
```

- [ ] **Step 1.6: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_app.py -k "sidebar_uses_select or has_refresh" -v
```

Expected: `AttributeError: 'App' object has no attribute '_output_dir_select'`

- [ ] **Step 1.7: Replace `_output_dir_input` with `_output_dir_select` in `app.py`**

In `App._build_sidebar`, replace the `_output_dir_input` TextInput definition:

```python
# OLD — delete these two lines:
self._output_dir_input = pn.widgets.TextInput(
    name="Output directory", placeholder="/workspace/outputs/birds_c0043",
    width=280, visible=False,
)
```

Replace with:

```python
self._refresh_dirs_btn = pn.widgets.Button(name="↻", width=40, visible=False)
_dir_options = _scan_output_dirs(self._base_dir)
self._output_dir_select = pn.widgets.Select(
    name="Output directory",
    options=_dir_options if _dir_options else ["(no sessions found)"],
    width=230,
    visible=False,
    disabled=not bool(_dir_options),
)
self._refresh_dirs_btn.on_click(self._on_refresh_dirs)
```

Then update the sidebar Column to include the new widgets (replace the `self._output_dir_input` line):

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
    width=300,
)
```

- [ ] **Step 1.8: Add `_on_load_existing` and `_on_refresh_dirs` methods**

In `App._on_load_existing`, update to show `_output_dir_select` instead of `_output_dir_input`:

```python
def _on_load_existing(self, event: Any) -> None:
    self._video_input.visible = False
    self._output_dir_select.visible = True
    self._refresh_dirs_btn.visible = True
    self._confirm_btn.name = "Load session"
    self._confirm_btn.visible = True
```

Add new `_on_refresh_dirs` method:

```python
def _on_refresh_dirs(self, event: Any) -> None:
    """Re-scan base_dir and refresh the output directory selector options."""
    dirs = _scan_output_dirs(self._base_dir)
    if dirs:
        self._output_dir_select.options = dirs
        self._output_dir_select.disabled = False
    else:
        self._output_dir_select.options = ["(no sessions found)"]
        self._output_dir_select.disabled = True
```

- [ ] **Step 1.9: Update `_on_confirm_session` to read from Select**

In `_on_confirm_session`, the load-existing branch currently reads `self._output_dir_input.value.strip()`. Change to:

```python
# In the else branch (load-existing path):
selected = self._output_dir_select.value
if selected == "(no sessions found)" or not selected:
    self._session_status.object = (
        "<p style='color:#e05050;font-size:12px'>No session selected</p>"
    )
    return
out_dir = self._base_dir / selected
```

Also hide the new widgets at the end of the method (replace `self._output_dir_input.visible = False`):

```python
self._video_input.visible = False
self._output_dir_select.visible = False
self._refresh_dirs_btn.visible = False
self._confirm_btn.visible = False
```

- [ ] **Step 1.10: Run all sidebar tests**

```bash
python -m pytest tests/dashboard/test_app.py -v
```

Expected: all pass.

- [ ] **Step 1.11: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): replace output-dir TextInput with base_dir-scanned Select widget"
```

---

## Task 2: Fix 2 — Tab switch + thread-safe video display

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Test: `tests/dashboard/test_app.py`, `tests/dashboard/test_preprocess.py`

- [ ] **Step 2.1: Write failing test — `App` exposes `_tabs`**

Add to `tests/dashboard/test_app.py`:

```python
def test_app_stores_tabs_reference():
    app = App()
    app.servable()
    assert hasattr(app, "_tabs")
    import panel as pn
    assert isinstance(app._tabs, pn.Tabs)
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_app.py::test_app_stores_tabs_reference -v
```

Expected: `AttributeError` — `_tabs` not yet set.

- [ ] **Step 2.3: Move `tabs` to `self._tabs` in `app.py`**

In `App.servable()`, change:

```python
# OLD:
tabs = pn.Tabs(
    *[(name, self._panes[name].panel()) for name in self._tab_names],
    dynamic=True,
    sizing_mode="stretch_width",
)
visualize_tab_index = list(self._panes.keys()).index("Visualize")
self._panes["Visualize"].wire_tabs(tabs, visualize_tab_index)
main_content = pn.Column(
    tabs,
    self._op_log.panel(),
    sizing_mode="stretch_width",
)
```

```python
# NEW:
self._tabs = pn.Tabs(
    *[(name, self._panes[name].panel()) for name in self._tab_names],
    dynamic=True,
    sizing_mode="stretch_width",
)
visualize_tab_index = list(self._panes.keys()).index("Visualize")
self._panes["Visualize"].wire_tabs(self._tabs, visualize_tab_index)
main_content = pn.Column(
    self._tabs,
    self._op_log.panel(),
    sizing_mode="stretch_width",
)
```

- [ ] **Step 2.4: Run `test_app_stores_tabs_reference` — expect pass**

```bash
python -m pytest tests/dashboard/test_app.py::test_app_stores_tabs_reference -v
```

- [ ] **Step 2.5: Write failing test — confirm session switches to Preprocess tab**

Add to `tests/dashboard/test_app.py`:

```python
import unittest.mock as mock


def test_confirm_load_existing_switches_to_preprocess_tab(tmp_path):
    (tmp_path / "scene_01").mkdir()
    (tmp_path / "scene_01" / "run_config.yaml").write_text("")
    app = App(base_dir=str(tmp_path))
    app.servable()
    # Simulate being on tab 2 (Semantics)
    app._tabs.active = 2

    # Trigger load-existing flow
    app._on_load_existing(None)
    app._output_dir_select.value = "scene_01"

    app._on_confirm_session(None)

    assert app._tabs.active == 0, "Should switch to Preprocess tab (index 0)"
```

- [ ] **Step 2.6: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_app.py::test_confirm_load_existing_switches_to_preprocess_tab -v
```

Expected: `AssertionError` — tab not switched.

- [ ] **Step 2.7: Add tab switch in `_on_confirm_session` (load-existing branch)**

In `App._on_confirm_session`, at the start of the `else` branch (before writing to `self._state`), add:

```python
# Switch to Preprocess tab so the pane is rendered before state updates fire
if hasattr(self, "_tabs"):
    self._tabs.active = 0
```

Place this immediately after `out_dir = self._base_dir / selected` and before `self._state.output_dir = out_dir`.

- [ ] **Step 2.8: Run tab-switch test — expect pass**

```bash
python -m pytest tests/dashboard/test_app.py::test_confirm_load_existing_switches_to_preprocess_tab -v
```

- [ ] **Step 2.9: Write failing test — `_on_video_path_change` uses `pn.state.execute`**

Add to `tests/dashboard/test_preprocess.py`:

```python
import unittest.mock as mock
import panel as pn
from collab_splats.dashboard.panes.preprocess import PreprocessPane
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog


def test_video_path_change_uses_pn_state_execute(tmp_path):
    """Watch callback must use pn.state.execute so update runs on Tornado main thread."""
    state = AppState()
    pane = PreprocessPane(state=state, op_log=OperationLog())

    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"fake")

    execute_calls = []

    class FakeEvent:
        new = str(video_file)

    with mock.patch("collab_splats.dashboard.panes.preprocess.pn.state") as mock_state:
        mock_state.execute = lambda fn: execute_calls.append(fn)
        pane._on_video_path_change(FakeEvent())

    assert len(execute_calls) == 1, "pn.state.execute should be called once"
    assert callable(execute_calls[0])
```

- [ ] **Step 2.10: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_preprocess.py::test_video_path_change_uses_pn_state_execute -v
```

Expected: fail — `_load_video` called directly, not via `pn.state.execute`.

- [ ] **Step 2.11: Wrap `_load_video` call in `pn.state.execute`**

In `collab_splats/dashboard/panes/preprocess.py`, find `_on_video_path_change`:

```python
# OLD:
def _on_video_path_change(self, event: Any) -> None:
    """Auto-load video display and reveal controls when AppState.video_path is set."""
    if event.new and Path(event.new).exists():
        self._load_video(Path(event.new))
        self._controls_card.visible = True
```

```python
# NEW:
def _on_video_path_change(self, event: Any) -> None:
    """Auto-load video display and reveal controls when AppState.video_path is set."""
    if event.new and Path(event.new).exists():
        video_path = Path(event.new)
        pn.state.execute(lambda: self._load_video(video_path))
        self._controls_card.visible = True
```

- [ ] **Step 2.12: Run Fix 2 tests**

```bash
python -m pytest tests/dashboard/test_app.py tests/dashboard/test_preprocess.py -v
```

Expected: all pass.

- [ ] **Step 2.13: Commit**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/panes/preprocess.py \
        tests/dashboard/test_app.py tests/dashboard/test_preprocess.py
git commit -m "fix(dashboard): switch to Preprocess tab on load-existing; use pn.state.execute for video watch"
```

---

## Task 3: Fix 3 — SemanticsPane single-extractor three-panel rewrite

**Files:**
- Modify: `collab_splats/dashboard/panes/semantics.py`
- Test: `tests/dashboard/test_semantics.py`

- [ ] **Step 3.1: Write failing structural tests**

Add to `tests/dashboard/test_semantics.py`:

```python
import panel as pn
from collab_splats.dashboard.panes.semantics import SemanticsPane
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog


def _make_semantics():
    return SemanticsPane(state=AppState(), op_log=OperationLog())


def test_semantics_has_single_method_dropdown():
    pane = _make_semantics()
    assert isinstance(pane._method_dd, pn.widgets.Select)


def test_semantics_has_three_image_panes():
    pane = _make_semantics()
    assert isinstance(pane._original_pane, pn.pane.PNG)
    assert isinstance(pane._pca_pane, pn.pane.PNG)
    assert isinstance(pane._sim_pane, pn.pane.PNG)


def test_semantics_no_extractor_columns():
    pane = _make_semantics()
    assert not hasattr(pane, "_columns"), "ExtractorColumn list should be gone"


def test_semantics_query_btn_disabled_by_default():
    pane = _make_semantics()
    assert pane._query_btn.disabled is True


def test_semantics_panel_returns_column():
    pane = _make_semantics()
    result = pane.panel()
    assert isinstance(result, pn.Column)
```

- [ ] **Step 3.2: Run to confirm failure**

```bash
python -m pytest tests/dashboard/test_semantics.py -k "has_single or has_three or no_extractor or query_btn or panel_returns" -v
```

Expected: `AttributeError` or assertion errors — old layout still in place.

- [ ] **Step 3.3: Rewrite `SemanticsPane` in `semantics.py`**

Open `collab_splats/dashboard/panes/semantics.py`. Keep the module header, imports, `_score_to_rgb`, `_load_frame_rgb` helpers, and all imports unchanged. Delete the `ExtractorColumn` class entirely. Replace the `SemanticsPane` class with the following:

```python
class SemanticsPane(param.Parameterized):
    """Feature extraction pane: single extractor, three-panel display."""

    def __init__(self, state: AppState, op_log: OperationLog, **params: Any):
        super().__init__(**params)
        self._state = state
        self._op_log = op_log
        self._extractor: BaseFeatureExtractor | None = None
        self._feature_zarr_path: Path | None = None
        self._current_frame_idx: int = 0
        self._run_thread: threading.Thread | None = None
        self._query_thread: threading.Thread | None = None

        # Frame selector
        self._frame_slider = pn.widgets.IntSlider(
            name="Frame", start=0, end=0, value=0, width=300
        )
        self._prev_btn = pn.widgets.Button(name="◄", width=40)
        self._next_btn = pn.widgets.Button(name="►", width=40)
        self._frame_count_html = pn.pane.HTML("", width=100)

        # Single extractor selector
        self._method_dd = pn.widgets.Select(
            name="Extractor",
            options=list(BaseFeatureExtractor._registry.keys()),
            width=200,
        )
        self._run_btn = pn.widgets.Button(name="▶ Run", button_type="primary", width=90)
        self._status_html = pn.pane.HTML("", width=400)

        # Three image panels: ground truth | PCA features | cosine similarity
        self._original_pane = pn.pane.PNG(None, width=320, height=240)
        self._pca_pane = pn.pane.PNG(None, width=320, height=240)
        self._sim_pane = pn.pane.PNG(None, width=320, height=240)

        # Query bar
        self._query_input = pn.widgets.TextInput(
            placeholder="Enter text query…", width=300
        )
        self._query_btn = pn.widgets.Button(
            name="Query", button_type="success", width=80, disabled=True
        )

        # Wire callbacks
        self._frame_slider.param.watch(self._on_frame_slider, "value")
        self._prev_btn.on_click(lambda e: self._step_frame(-1))
        self._next_btn.on_click(lambda e: self._step_frame(1))
        self._run_btn.on_click(self._on_run)
        self._query_btn.on_click(self._on_query)
        self._state.param.watch(self._on_frames_zarr_change, "frames_zarr_path")
        self._state.param.watch(self._on_frames_zarr_change, "selected_indices")

    def _on_frames_zarr_change(self, event: Any) -> None:
        """Update frame slider range and load first frame when zarr is ready."""
        if self._state.frames_zarr_path is None:
            return
        import zarr
        z = zarr.open(str(self._state.frames_zarr_path), mode="r")
        n = int(z["frames"].shape[0])
        self._frame_slider.end = max(0, n - 1)
        self._frame_count_html.object = f"<small>/ {n}</small>"
        self._load_original(0)

    def _on_frame_slider(self, event: Any) -> None:
        """Update displays when frame slider moves."""
        idx = event.new
        self._current_frame_idx = idx
        self._load_original(idx)
        if self._feature_zarr_path is not None:
            self._refresh_pca(idx)

    def _step_frame(self, delta: int) -> None:
        """Move frame slider by delta, clamped to valid range."""
        new_val = max(
            self._frame_slider.start,
            min(self._frame_slider.end, self._frame_slider.value + delta),
        )
        self._frame_slider.value = new_val

    def _load_original(self, idx: int) -> None:
        """Load and display the ground-truth frame for idx."""
        from io import BytesIO
        from PIL import Image as PILImage
        if self._state.frames_zarr_path is None:
            return
        frame = _load_frame_rgb(Path(self._state.frames_zarr_path), idx)
        buf = BytesIO()
        PILImage.fromarray(frame).save(buf, format="PNG")
        self._original_pane.object = buf.getvalue()

    def _refresh_pca(self, idx: int) -> None:
        """Render PCA RGB overlay for features at frame idx."""
        from io import BytesIO
        import zarr
        from PIL import Image as PILImage
        if self._feature_zarr_path is None:
            return
        z = zarr.open(str(self._feature_zarr_path), mode="r")
        feat = torch.from_numpy(np.array(z["features"][idx]))  # (D, H_p, W_p)
        rgb = BaseFeatureExtractor.features_to_rgb(feat)
        buf = BytesIO()
        PILImage.fromarray(rgb).save(buf, format="PNG")
        self._pca_pane.object = buf.getvalue()

    def _on_run(self, event: Any) -> None:
        """Start background extraction when Run button clicked."""
        if self._run_thread and self._run_thread.is_alive():
            return
        if self._state.frames_zarr_path is None or self._state.output_dir is None:
            self._status_html.object = (
                "<small style='color:#e05050'>Load frames first (Preprocess tab)</small>"
            )
            return
        method = self._method_dd.value
        self._run_btn.disabled = True
        self._status_html.object = f"<small style='color:#aaa'>Running {method}…</small>"
        self._run_thread = threading.Thread(
            target=self._run_extraction, args=(method,), daemon=True
        )
        self._run_thread.start()

    def _run_extraction(self, method: str) -> None:
        """Background: instantiate extractor, extract all frames, render PCA for current frame."""
        try:
            self._op_log.start_op(f"Extracting {method} features")
            extractor_cls = BaseFeatureExtractor.get(method)
            self._extractor = extractor_cls()
            output_dir = Path(self._state.output_dir)
            cache_dir = output_dir / "features" / method
            zarr_path = self._extractor.extract_and_cache_from_zarr(
                frames_zarr_path=Path(self._state.frames_zarr_path),
                cache_dir=cache_dir,
            )
            self._feature_zarr_path = zarr_path
            self._state.feature_maps_path = zarr_path
            self._op_log.finish_op()
            self._refresh_pca(self._current_frame_idx)
            self._update_query_btn()
            self._status_html.object = f"<small style='color:#50c050'>{method} ready</small>"
        except Exception as exc:
            logger.exception("Extraction failed for %s", method)
            self._status_html.object = f"<small style='color:#e05050'>Error: {exc}</small>"
            self._op_log.error_op(str(exc))
        finally:
            self._run_btn.disabled = False

    def _update_query_btn(self) -> None:
        """Enable query button only for queryable extractors with loaded features."""
        queryable = (
            self._feature_zarr_path is not None
            and isinstance(self._extractor, BaseQueryableExtractor)
        )
        self._query_btn.disabled = not queryable
        if self._feature_zarr_path is not None and not queryable:
            self._status_html.object += (
                "<br><small style='color:#888'>"
                "Run a queryable extractor (e.g. DINOv2 SAM) to enable text queries"
                "</small>"
            )

    def _on_query(self, event: Any) -> None:
        """Start background query thread."""
        if self._query_thread and self._query_thread.is_alive():
            return
        query_text = self._query_input.value.strip()
        if not query_text:
            return
        self._query_btn.disabled = True
        self._query_thread = threading.Thread(
            target=self._run_query,
            args=(self._current_frame_idx, query_text),
            daemon=True,
        )
        self._query_thread.start()

    def _run_query(self, frame_idx: int, query_text: str) -> None:
        """Background: compute cosine similarity heatmap and display in sim pane."""
        from io import BytesIO
        import zarr
        from PIL import Image as PILImage
        try:
            z = zarr.open(str(self._feature_zarr_path), mode="r")
            feat = torch.from_numpy(
                np.array(z["features"][frame_idx]).astype(np.float32)
            )
            score = self._extractor.score_queries(
                feat, positive=[query_text]
            )  # (H_p, W_p)
            rgb = _score_to_rgb(score.cpu().numpy())
            buf = BytesIO()
            PILImage.fromarray(rgb).save(buf, format="PNG")
            self._sim_pane.object = buf.getvalue()
        except Exception as exc:
            logger.exception("Query failed")
            self._op_log.error_op(f"Query error: {exc}")
        finally:
            self._update_query_btn()

    def panel(self) -> pn.Column:
        """Return Panel layout for Tab 2."""
        top_row = pn.Row(
            self._prev_btn,
            self._frame_slider,
            self._next_btn,
            self._frame_count_html,
            pn.HSpacer(),
            pn.pane.HTML("<b style='align-self:center'>Extractor:</b>"),
            self._method_dd,
            self._run_btn,
        )
        image_row = pn.Row(
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Ground truth</b>"),
                self._original_pane,
            ),
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>PCA features</b>"),
                self._pca_pane,
            ),
            pn.Column(
                pn.pane.HTML("<b style='color:#aaa;font-size:12px'>Cosine similarity</b>"),
                self._sim_pane,
            ),
            sizing_mode="stretch_width",
        )
        query_row = pn.Row(
            pn.pane.HTML("<b style='align-self:center'>Query:</b>"),
            self._query_input,
            self._query_btn,
        )
        return pn.Column(
            pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Semantics</h3>"),
            top_row,
            self._status_html,
            image_row,
            query_row,
            sizing_mode="stretch_width",
        )
```

Verify the file still has: `_score_to_rgb`, `_load_frame_rgb`, all imports (`threading`, `torch`, `numpy`, `param`, `panel`, `Path`, `BaseFeatureExtractor`, `BaseQueryableExtractor`, `AppState`, `OperationLog`, `logger`).

- [ ] **Step 3.4: Run all semantics tests**

```bash
python -m pytest tests/dashboard/test_semantics.py -v
```

Expected: all pass (including pre-existing `test_score_to_rgb_*` and `test_load_frame_rgb_*`).

- [ ] **Step 3.5: Commit**

```bash
git add collab_splats/dashboard/panes/semantics.py tests/dashboard/test_semantics.py
git commit -m "feat(semantics): single-extractor three-panel layout — drop ExtractorColumn"
```

---

## Task 4: Fix 4 — VisualizePane layout reorganization

**Files:**
- Modify: `collab_splats/dashboard/panes/visualize.py`
- Test: `tests/dashboard/test_visualize.py`

- [ ] **Step 4.1: Write failing structural tests**

Add to `tests/dashboard/test_visualize.py`:

```python
import panel as pn
import unittest.mock as mock
from collab_splats.dashboard.panes.visualize import ScenePanel, VisualizePane
from collab_splats.dashboard.state import AppState
from collab_splats.dashboard.operation_log import OperationLog


def _make_scene(tmp_path):
    state = AppState()
    return ScenePanel("A", tmp_path, state, OperationLog(), _off_screen=True)


def test_scene_panel_has_radio_button_group(tmp_path):
    sp = _make_scene(tmp_path)
    assert isinstance(sp._mode_selector, pn.widgets.RadioButtonGroup)


def test_scene_panel_default_mode_is_mesh(tmp_path):
    sp = _make_scene(tmp_path)
    assert sp._mode_selector.value == "Mesh"


def test_scene_panel_frustum_is_checkbox(tmp_path):
    sp = _make_scene(tmp_path)
    assert isinstance(sp._frustum_check, pn.widgets.Checkbox)


def test_scene_panel_has_sim_query_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_sim_query_row")
    assert sp._sim_query_row.visible is False
    assert isinstance(sp._extractor_dd, pn.widgets.Select)


def test_scene_panel_has_points_options_row(tmp_path):
    sp = _make_scene(tmp_path)
    assert hasattr(sp, "_points_options_row")
    assert sp._points_options_row.visible is False


def test_visualize_pane_no_shared_query_bar(tmp_path):
    """Shared query bar removed — per-scene query lives on ScenePanel."""
    with mock.patch(
        "collab_splats.dashboard.panes.visualize.ScenePanel",
        lambda *a, **kw: ScenePanel(*a, **{**kw, "_off_screen": True}),
    ):
        vp = VisualizePane(state=AppState(), op_log=OperationLog(), base_dir=tmp_path)
    assert not hasattr(vp, "_query_bar"), "Shared query bar should be removed"
```

- [ ] **Step 4.2: Run to confirm failures**

```bash
python -m pytest tests/dashboard/test_visualize.py -k "radio or default_mode or checkbox or sim_query or points_options or no_shared" -v
```

Expected: `AttributeError` on `_mode_selector`, `_frustum_check`, etc.

- [ ] **Step 4.3: Update `ScenePanel.__init__` in `visualize.py` — replace mode buttons and frustum toggle**

In `ScenePanel.__init__`, find and replace the mode buttons + frustum toggle block:

```python
# OLD — delete these:
self._pcd_btn = pn.widgets.Button(name="PCD", button_type="primary", width=80, disabled=True)
self._mesh_btn = pn.widgets.Button(name="Mesh", width=120, disabled=True)
self._sim_btn = pn.widgets.Button(name="Similarity", width=140, disabled=True)
self._frustum_toggle = pn.widgets.Toggle(name="Show frustums", value=False, width=130)
self._point_size_slider = pn.widgets.IntSlider(name="Point size", value=2, start=1, end=10, width=180)
```

```python
# NEW:
self._mode_selector = pn.widgets.RadioButtonGroup(
    options=["Points", "Mesh", "Similarity"],
    value="Mesh",
    button_type="success",
    width=380,
    disabled=True,
)
self._frustum_check = pn.widgets.Checkbox(name="Show frustums", value=False)
self._point_size_slider = pn.widgets.IntSlider(
    name="Point size", value=2, start=1, end=10, width=180
)
self._points_options_row = pn.Row(
    self._frustum_check,
    self._point_size_slider,
    visible=False,
)
self._extractor_dd = pn.widgets.Select(
    name="Extractor", options=[], width=180
)
self._sim_query_input = pn.widgets.TextInput(
    placeholder="Enter text query…", width=220
)
self._sim_query_btn = pn.widgets.Button(
    name="Query", button_type="success", width=80
)
self._sim_query_row = pn.Row(
    self._extractor_dd,
    self._sim_query_input,
    self._sim_query_btn,
    visible=False,
)
```

- [ ] **Step 4.4: Update callbacks wiring in `ScenePanel.__init__`**

Find the callback wiring block. Replace old button wires:

```python
# OLD — delete these three lines:
self._pcd_btn.on_click(lambda e: self._on_mode_change("PCD"))
self._mesh_btn.on_click(lambda e: self._on_mode_change("Mesh"))
self._sim_btn.on_click(lambda e: self._on_mode_change("Similarity"))
self._frustum_toggle.param.watch(self._on_frustum_toggle, "value")
```

```python
# NEW:
self._mode_selector.param.watch(
    lambda e: self._on_mode_change(e.new), "value"
)
self._frustum_check.param.watch(self._on_frustum_toggle, "value")
self._sim_query_btn.on_click(self._on_sim_query_click)
```

At the end of `__init__`, trigger the initial mode:

```python
# Set initial viewer state to match default "Mesh" selection
self._on_mode_change("Mesh")
```

- [ ] **Step 4.5: Update `_on_mode_change` to use new widgets**

Replace the existing `_on_mode_change` method:

```python
def _on_mode_change(self, new_display_mode: str) -> None:
    """Switch viewer mode; update contextual controls visibility."""
    mode_map = {"Points": "PCD", "Mesh": "Mesh", "Similarity": "Similarity"}
    new_mode = mode_map.get(new_display_mode, new_display_mode)

    if new_mode not in self._available_modes and self._result is not None:
        return
    self.mode = new_mode

    # Show/hide contextual rows
    self._points_options_row.visible = (new_mode == "PCD")
    self._sim_query_row.visible = (new_mode == "Similarity")

    self._plotter.clear()
    if new_mode == "PCD":
        self._rebuild_pcd_viewer()
    elif new_mode == "Mesh":
        self._rebuild_mesh_viewer()
    elif new_mode == "Similarity":
        if self._lifted_normed is None:
            self._load_lifted_features_for_current_extractor()
        self._rebuild_sim_viewer(colors=None)

    self._vtk_pane.synchronize()
```

- [ ] **Step 4.6: Add `_on_sim_query_click` to `ScenePanel`**

The existing query method in `ScenePanel` is `do_query(self, text: str, extractor_name: str)` (line 473 of `visualize.py`). Add a click handler that reads from the per-scene widgets and calls it:

```python
def _on_sim_query_click(self, event: Any) -> None:
    """Fire similarity query from this scene's per-scene query input."""
    text = self._sim_query_input.value.strip()
    extractor_name = self._extractor_dd.value
    if not text or not extractor_name:
        return
    import threading
    threading.Thread(
        target=self.do_query, args=(text, extractor_name), daemon=True
    ).start()
```

Also update `_on_mode_change` in the `Similarity` branch to sync `_extractor_dd.options` with `_available_extractors`:

```python
elif new_mode == "Similarity":
    self._extractor_dd.options = self._available_extractors or []
    if self._available_extractors:
        self._extractor_dd.value = self._available_extractors[0]
    if self._lifted_normed is None:
        self._load_lifted_features_for_current_extractor()
    self._rebuild_sim_viewer(colors=None)
```

- [ ] **Step 4.7: Update `_on_frustum_toggle` to use Checkbox**

Find `_on_frustum_toggle`. The existing method signature uses `event.new` (a bool) which works identically for both `Toggle` and `Checkbox`. No change needed to the method body — just confirm it still reads `event.new`.

- [ ] **Step 4.8: Update `ScenePanel.panel()` layout**

Replace the existing `panel()` method:

```python
def panel(self) -> pn.Column:
    """Return the full scene panel layout."""
    controls_row = pn.Row(
        self._dataset_dd, self._backend_dd, self._load_btn, align="end"
    )
    action_row = pn.Row(self._reset_btn, self._snapshot_btn)
    return pn.Column(
        f"### Scene {self._scene_id}",
        controls_row,
        self._mode_selector,
        self._vtk_pane,
        self._points_options_row,
        self._sim_query_row,
        action_row,
        self._status_html,
        sizing_mode="stretch_both",
    )
```

- [ ] **Step 4.9: Update `VisualizePane` — remove shared query bar, add vertical divider**

In `VisualizePane.__init__`, delete the shared query bar block:

```python
# DELETE all of this:
self._query_input = pn.widgets.TextInput(placeholder="Enter text query…", width=300)
self._a_extractor_dd = pn.widgets.Select(name="Scene A extractor", options=[], width=160)
self._b_extractor_dd = pn.widgets.Select(name="Scene B extractor", options=[], width=160)
self._query_btn = pn.widgets.Button(name="Query", button_type="success", width=90)
self._query_bar = pn.Row(
    self._query_input, self._a_extractor_dd, self._b_extractor_dd, self._query_btn,
    visible=False,
)
self._scene_a.param.watch(self._on_scene_mode_change, "mode")
self._scene_b.param.watch(self._on_scene_mode_change, "mode")
self._scene_a.param.watch(self._update_extractor_dropdowns, "mode")
self._scene_b.param.watch(self._update_extractor_dropdowns, "mode")
self._query_btn.on_click(self._on_query_click)
```

Delete the methods `_on_scene_mode_change`, `_update_extractor_dropdowns`, `_on_query_click` from `VisualizePane`.

Update (or add) `VisualizePane.panel()`:

```python
def panel(self) -> pn.Column:
    """Return Panel layout for Tab 4."""
    divider = pn.pane.HTML(
        "<div style='border-left:1px solid #444;height:100%;margin:0 8px'></div>",
        width=18,
        sizing_mode="stretch_height",
    )
    scenes_row = pn.Row(
        self._scene_a.panel(),
        divider,
        self._scene_b.panel(),
        sizing_mode="stretch_both",
    )
    return pn.Column(
        pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Visualize</h3>"),
        scenes_row,
        sizing_mode="stretch_both",
    )
```

- [ ] **Step 4.10: Update stale test that checks for removed `_query_bar`**

In `tests/dashboard/test_visualize.py`, the existing test `test_visualize_pane_query_bar_visible_on_similarity` references the removed shared query bar. Replace it:

```python
def test_scene_panel_sim_query_row_visible_in_similarity_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"PCD", "Similarity"}
    sp._result = object()  # non-None sentinel
    sp._on_mode_change("Similarity")
    assert sp._sim_query_row.visible is True
    assert sp._points_options_row.visible is False


def test_scene_panel_points_options_visible_in_points_mode(tmp_path):
    sp = _make_scene(tmp_path)
    sp._available_modes = {"PCD"}
    sp._result = object()
    sp._on_mode_change("Points")
    assert sp._points_options_row.visible is True
    assert sp._sim_query_row.visible is False
```

Also remove `test_visualize_pane_query_bar_visible_on_similarity` from the test file — it tests the removed shared query bar.

- [ ] **Step 4.11: Run all visualize tests**

```bash
python -m pytest tests/dashboard/test_visualize.py -v
```

Expected: all pass. If `_query_similarity` method name is wrong (Step 4.6), fix it now.

- [ ] **Step 4.12: Run full dashboard test suite**

```bash
python -m pytest tests/dashboard/ -v
```

Expected: all pass.

- [ ] **Step 4.13: Commit**

```bash
git add collab_splats/dashboard/panes/visualize.py tests/dashboard/test_visualize.py
git commit -m "feat(visualize): RadioButtonGroup modes, per-scene query, frustum checkbox, vertical divider"
```

---

## Final check

- [ ] **Step 5.1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: no regressions outside `tests/dashboard/`.

- [ ] **Step 5.2: Final commit if any stragglers**

If any files were modified but not yet committed:

```bash
git status
git add <files>
git commit -m "fix(dashboard): cleanup after dashboard-fixes implementation"
```
