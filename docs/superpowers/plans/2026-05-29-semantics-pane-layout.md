# Semantics Pane Layout Restructure + Auto-Discover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure SemanticsPane layout (extractor top-left, constant image area, frame slider below images) and auto-load cached features on session load.

**Architecture:** All changes in `collab_splats/dashboard/panes/semantics.py`. Layout restructure is purely in `panel()`. Auto-discover adds `_try_discover_cache()` + `_load_cached_features()` (background thread) plus three new watchers. Tests in `tests/dashboard/test_semantics.py`.

**Tech Stack:** Panel (pn), param, zarr, threading, pytest

---

## File Map

| File | Change |
|---|---|
| `collab_splats/dashboard/panes/semantics.py` | Layout + auto-discover logic |
| `tests/dashboard/test_semantics.py` | New layout + discovery tests |

---

### Task 1: Write failing layout tests

**Files:**
- Modify: `tests/dashboard/test_semantics.py`

- [ ] **Step 1: Add layout structure tests**

Append to `tests/dashboard/test_semantics.py`:

```python
def test_panel_extractor_row_is_first_content():
    """Extractor controls are col.objects[1] (after H3 header)."""
    pane = _make_semantics()
    col = pane.panel()
    extractor_row = col.objects[1]
    assert isinstance(extractor_row, pn.Row)
    assert pane._method_dd in extractor_row.objects


def test_panel_image_area_has_fixed_height():
    """Image area is a Column with height=270 to prevent reflow."""
    pane = _make_semantics()
    col = pane.panel()
    image_area = col.objects[2]
    assert isinstance(image_area, pn.Column)
    assert image_area.height == 270


def test_panel_frame_slider_below_image_area():
    """Frame slider row is col.objects[3], after the image area."""
    pane = _make_semantics()
    col = pane.panel()
    frame_row = col.objects[3]
    assert isinstance(frame_row, pn.Row)
    assert pane._frame_slider in frame_row.objects


def test_panel_query_row_is_last():
    """Query row is col.objects[4] (last)."""
    pane = _make_semantics()
    col = pane.panel()
    query_row = col.objects[4]
    assert isinstance(query_row, pn.Row)
    assert pane._query_input in query_row.objects


def test_panel_status_html_in_extractor_row():
    """status_html lives inside extractor_row, not as a standalone column child."""
    pane = _make_semantics()
    col = pane.panel()
    extractor_row = col.objects[1]
    assert pane._status_html in extractor_row.objects
    # Must NOT be a top-level child of the column
    assert pane._status_html not in col.objects
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py::test_panel_extractor_row_is_first_content tests/dashboard/test_semantics.py::test_panel_image_area_has_fixed_height tests/dashboard/test_semantics.py::test_panel_frame_slider_below_image_area tests/dashboard/test_semantics.py::test_panel_query_row_is_last tests/dashboard/test_semantics.py::test_panel_status_html_in_extractor_row -v
```

Expected: all 5 FAIL (current layout doesn't match).

---

### Task 2: Restructure `panel()` method

**Files:**
- Modify: `collab_splats/dashboard/panes/semantics.py:242-281`

- [ ] **Step 1: Replace the `panel()` method**

Replace the entire `panel()` method (lines 242–281) with:

```python
def panel(self) -> pn.Column:
    """Return Panel layout for Tab 2."""
    # Extractor controls — top left
    extractor_row = pn.Row(
        pn.pane.HTML("<b style='align-self:center'>Extractor:</b>"),
        self._method_dd,
        self._run_btn,
        self._status_html,
    )
    # Fixed-height image area — prevents panel reflow when features load/unload
    image_area = pn.Column(
        pn.Row(
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
        ),
        height=270,
        sizing_mode="fixed",
    )
    # Frame navigation — "scrollbar" below images
    frame_row = pn.Row(
        self._prev_btn,
        self._frame_slider,
        self._next_btn,
        self._frame_count_html,
    )
    query_row = pn.Row(
        pn.pane.HTML("<b style='align-self:center'>Query:</b>"),
        self._query_input,
        self._query_btn,
    )
    return pn.Column(
        pn.pane.HTML("<h3 style='color:#7ec8e3;margin:0 0 8px 0'>Semantics</h3>"),
        extractor_row,
        image_area,
        frame_row,
        query_row,
        sizing_mode="stretch_width",
    )
```

- [ ] **Step 2: Run layout tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py::test_panel_extractor_row_is_first_content tests/dashboard/test_semantics.py::test_panel_image_area_has_fixed_height tests/dashboard/test_semantics.py::test_panel_frame_slider_below_image_area tests/dashboard/test_semantics.py::test_panel_query_row_is_last tests/dashboard/test_semantics.py::test_panel_status_html_in_extractor_row -v
```

Expected: all 5 PASS.

- [ ] **Step 3: Run full test suite to check no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py -v
```

Expected: all existing tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add collab_splats/dashboard/panes/semantics.py tests/dashboard/test_semantics.py
git commit -m "feat(dashboard): restructure semantics pane layout — extractor top-left, fixed image area, frame slider below"
```

---

### Task 3: Write failing auto-discover tests

**Files:**
- Modify: `tests/dashboard/test_semantics.py`

- [ ] **Step 1: Add auto-discover tests**

Append to `tests/dashboard/test_semantics.py`. These tests use `tmp_path` to create a fake feature zarr:

```python
def _write_feature_zarr(path: Path, n_frames: int = 2, feat_dim: int = 4, h: int = 6, w: int = 8):
    """Write a minimal valid feature zarr to path."""
    store = zarr.open(str(path), mode="w")
    store.create_array(
        "features",
        shape=(n_frames, feat_dim, h, w),
        chunks=(1, feat_dim, h, w),
        dtype="float32",
    )
    return path


def _write_frames_zarr(path: Path, n: int = 2, h: int = 48, w: int = 64):
    """Write a minimal valid frames zarr to path."""
    store = zarr.open(str(path), mode="w")
    store.attrs.update({"n_frames": n, "height": h, "width": w})
    arr = store.create_array(
        "frames", shape=(n, h, w, 3), chunks=(1, h, w, 3), dtype="uint8"
    )
    arr[:] = 0
    return path


def test_try_discover_cache_noop_when_no_output_dir():
    """_try_discover_cache does nothing when output_dir is not set."""
    pane = _make_semantics()
    pane._try_discover_cache()  # must not raise
    assert pane._feature_zarr_path is None


def test_try_discover_cache_noop_when_no_frames_zarr(tmp_path):
    """_try_discover_cache does nothing when frames_zarr_path is not set."""
    pane = _make_semantics()
    pane._state.output_dir = str(tmp_path)
    pane._try_discover_cache()  # must not raise
    assert pane._feature_zarr_path is None


def test_try_discover_cache_clears_when_no_cache(tmp_path):
    """_try_discover_cache clears feature state when no zarr exists at candidate path."""
    frames_zarr = _write_frames_zarr(tmp_path / "frames.zarr")
    pane = _make_semantics()
    pane._state.output_dir = str(tmp_path)
    pane._state.frames_zarr_path = str(frames_zarr)
    pane._feature_zarr_path = tmp_path / "stale"  # simulate stale state
    pane._try_discover_cache()
    assert pane._feature_zarr_path is None


def test_try_discover_cache_detects_valid_zarr(tmp_path, monkeypatch):
    """_try_discover_cache calls _load_cached_features when valid cache exists."""
    frames_zarr = _write_frames_zarr(tmp_path / "frames.zarr")
    pane = _make_semantics()
    method = pane._method_dd.value
    cache_dir = tmp_path / "features" / method
    _write_feature_zarr(cache_dir)

    # Stub _load_cached_features to avoid real model instantiation in tests
    calls = []

    def fake_load(m, p):
        calls.append((m, p))
        pane._feature_zarr_path = p

    monkeypatch.setattr(pane, "_load_cached_features", fake_load)

    pane._state.output_dir = str(tmp_path)
    pane._state.frames_zarr_path = str(frames_zarr)
    pane._try_discover_cache()

    if pane._discover_thread and pane._discover_thread.is_alive():
        pane._discover_thread.join(timeout=5.0)

    assert len(calls) == 1
    assert calls[0] == (method, cache_dir)
    assert pane._feature_zarr_path == cache_dir
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py::test_try_discover_cache_noop_when_no_output_dir tests/dashboard/test_semantics.py::test_try_discover_cache_noop_when_no_frames_zarr tests/dashboard/test_semantics.py::test_try_discover_cache_clears_when_no_cache tests/dashboard/test_semantics.py::test_try_discover_cache_detects_valid_zarr -v
```

Expected: all 4 FAIL with `AttributeError: '_try_discover_cache'` or similar.

---

### Task 4: Implement `_try_discover_cache` and `_load_cached_features`

**Files:**
- Modify: `collab_splats/dashboard/panes/semantics.py`

- [ ] **Step 1: Add `_discover_thread` field to `__init__`**

In `__init__`, after `self._query_thread: threading.Thread | None = None` (line 64), add:

```python
        self._discover_thread: threading.Thread | None = None
```

- [ ] **Step 2: Add `_try_discover_cache` and `_load_cached_features` methods**

Add these two methods before `panel()` (i.e., after `_run_query`, around line 241):

```python
    def _try_discover_cache(self) -> None:
        """Check for cached features for selected extractor; load in background if found."""
        if self._state.output_dir is None or self._state.frames_zarr_path is None:
            return
        if self._discover_thread and self._discover_thread.is_alive():
            return
        method = self._method_dd.value
        candidate = Path(self._state.output_dir) / "features" / method
        # Validate zarr synchronously (fast — just opens store metadata)
        try:
            z = zarr.open(str(candidate), mode="r")
            _ = z["features"]
        except Exception:
            # No valid cache — clear stale state
            self._feature_zarr_path = None
            self._pca_pane.object = None
            self._sim_pane.object = None
            self._update_query_btn()
            return
        # Valid cache — instantiate extractor in background (may load model weights)
        self._discover_thread = threading.Thread(
            target=self._load_cached_features,
            args=(method, candidate),
            daemon=True,
        )
        self._discover_thread.start()

    def _load_cached_features(self, method: str, zarr_path: Path) -> None:
        """Background: instantiate extractor and load cached feature zarr."""
        try:
            extractor_cls = BaseFeatureExtractor.get(method)
            self._extractor = extractor_cls()
            self._feature_zarr_path = zarr_path
            self._state.feature_maps_path = zarr_path
            self._refresh_pca(self._current_frame_idx)
            self._update_query_btn()
            self._status_html.object = (
                f"<small style='color:#50c050'>Loaded cached {method}</small>"
            )
        except Exception as exc:
            logger.exception("Cache load failed for %s", method)
            self._status_html.object = (
                f"<small style='color:#e05050'>Cache error: {exc}</small>"
            )
```

- [ ] **Step 3: Run discovery tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py::test_try_discover_cache_noop_when_no_output_dir tests/dashboard/test_semantics.py::test_try_discover_cache_noop_when_no_frames_zarr tests/dashboard/test_semantics.py::test_try_discover_cache_clears_when_no_cache tests/dashboard/test_semantics.py::test_try_discover_cache_detects_valid_zarr -v
```

Expected: all 4 PASS.

---

### Task 5: Wire auto-discover into state watchers

**Files:**
- Modify: `collab_splats/dashboard/panes/semantics.py:96-113`

- [ ] **Step 1: Add new watchers in `__init__`**

In `__init__`, after the existing watcher lines (around line 102–103):

```python
        self._state.param.watch(self._on_frames_zarr_change, "frames_zarr_path")
        self._state.param.watch(self._on_frames_zarr_change, "selected_indices")
```

Add:

```python
        self._state.param.watch(self._on_output_dir_change, "output_dir")
        self._method_dd.param.watch(self._on_method_change, "value")
```

- [ ] **Step 2: Add `_on_output_dir_change` and `_on_method_change` handlers**

Add these two methods after `_on_frames_zarr_change` (after line 113):

```python
    def _on_output_dir_change(self, event: Any) -> None:
        """Try to discover cached features when a session is loaded."""
        self._try_discover_cache()

    def _on_method_change(self, event: Any) -> None:
        """Try to discover cached features when selected extractor changes."""
        self._try_discover_cache()
```

- [ ] **Step 3: Call `_try_discover_cache` at end of `_on_frames_zarr_change`**

`_on_frames_zarr_change` currently ends at line 113:

```python
    def _on_frames_zarr_change(self, event: Any) -> None:
        """Update frame slider range and load first frame when zarr is ready."""
        if self._state.frames_zarr_path is None:
            return
        z = zarr.open(str(self._state.frames_zarr_path), mode="r")
        n = int(z["frames"].shape[0])
        self._frame_slider.end = max(0, n - 1)
        self._frame_count_html.object = f"<small>/ {n}</small>"
        self._load_original(0)
```

Add one line at the end:

```python
        self._try_discover_cache()
```

- [ ] **Step 4: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_semantics.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/semantics.py tests/dashboard/test_semantics.py
git commit -m "feat(dashboard): auto-discover cached features on session load and extractor switch"
```
