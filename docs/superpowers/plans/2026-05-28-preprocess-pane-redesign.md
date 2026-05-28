# PreprocessPane Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine Tab 1 (PreprocessPane) so video shows on left when loaded, frame extraction controls are gated on video load and collapsible, and post-extraction shows interactive Bokeh metric plots + HTML frame strip with tap-to-seek + scroll-to-frame; extracted frames written to zarr instead of held in RAM.

**Architecture:** Three coordinated changes — (1) `AppState` schema drops `frames: list[np.ndarray]`, gains `frames_zarr_path` + `selected_indices`; (2) `PreprocessPane` replaces matplotlib PNG with Bokeh figures and `pn.Row` strip with `pn.pane.HTML`, wraps controls in a collapsible `pn.Card` gated on `video_path`; (3) `app.py` load-existing flow parses `video_path` from `run_config.yaml`. All Panel widget wiring unchanged; only rendering and state schema change.

**Tech Stack:** Panel, param, Bokeh (`ColumnDataSource`, `TapTool`, `Span`), zarr, numcodecs (Blosc/lz4), base64, numpy

**Spec:** `docs/superpowers/specs/2026-05-28-preprocess-pane-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/dashboard/state.py` | Modify | Drop `frames`, add `frames_zarr_path` + `selected_indices` |
| `collab_splats/dashboard/app.py` | Modify | Load-existing: parse `video_path` from `run_config.yaml` |
| `collab_splats/dashboard/panes/preprocess.py` | Modify | Zarr write, Bokeh metrics, HTML frame strip, Card gating, new layout |
| `tests/dashboard/test_state.py` | Modify | Update for new schema |
| `tests/dashboard/test_preprocess.py` | Modify | Drop matplotlib tests, add zarr + HTML strip tests |

---

## Task 1: Update AppState schema

**Files:**
- Modify: `collab_splats/dashboard/state.py`
- Modify: `tests/dashboard/test_state.py`

- [ ] **Step 1: Update `test_state.py` — replace frames tests**

Replace the file content of `tests/dashboard/test_state.py`:

```python
from pathlib import Path

from collab_splats.dashboard.state import AppState


def test_appstate_defaults():
    state = AppState()
    assert state.output_dir is None
    assert state.video_path is None
    assert state.frames_zarr_path is None
    assert state.selected_indices == []
    assert state.feedforward_result is None
    assert state.feature_maps_path is None
    assert state.lifted_features_path is None


def test_appstate_watch_fires_on_output_dir_change():
    state = AppState()
    received = []
    state.param.watch(lambda e: received.append(e.new), "output_dir")
    state.output_dir = Path("/tmp/test_out")
    assert received == [Path("/tmp/test_out")]


def test_appstate_frames_zarr_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/frames.zarr")
    state.frames_zarr_path = p
    assert state.frames_zarr_path == p


def test_appstate_selected_indices_accepts_list():
    state = AppState()
    state.selected_indices = [0, 5, 10, 15]
    assert state.selected_indices == [0, 5, 10, 15]


def test_appstate_feature_maps_path_accepts_path():
    state = AppState()
    p = Path("/workspace/outputs/birds/vggt_omega/features.zarr")
    state.feature_maps_path = p
    assert state.feature_maps_path == p
```

- [ ] **Step 2: Run tests — expect failures**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_state.py -v
```

Expected: `test_appstate_defaults` fails (`frames_zarr_path` not found), `test_appstate_frames_accepts_list_of_arrays` errors.

- [ ] **Step 3: Update `state.py`**

```python
from __future__ import annotations

from pathlib import Path

import param


class AppState(param.Parameterized):
    """Shared data bus passed between all dashboard panes.

    Panes observe fields via param.watch — downstream panes auto-enable
    when upstream data arrives (e.g. output_dir set by PreprocessPane).
    """

    output_dir = param.Parameter(default=None)
    video_path = param.Parameter(default=None)
    frames_zarr_path = param.Parameter(default=None)
    selected_indices = param.List(default=[])
    feedforward_result = param.Parameter(default=None)
    feature_maps_path = param.Parameter(default=None)
    lifted_features_path = param.Parameter(default=None)
```

- [ ] **Step 4: Run tests — all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_state.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/state.py tests/dashboard/test_state.py
git commit -m "feat(dashboard): replace state.frames with frames_zarr_path + selected_indices"
```

---

## Task 2: Fix load-existing bug — parse video_path from run_config.yaml

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `tests/dashboard/test_app.py`

- [ ] **Step 1: Read `tests/dashboard/test_app.py` to understand existing test patterns**

Run:
```bash
cat tests/dashboard/test_app.py
```

- [ ] **Step 2: Add test for load-existing video_path parsing**

Add to `tests/dashboard/test_app.py`:

```python
import tempfile
from pathlib import Path

import yaml

from collab_splats.dashboard.app import App


def test_load_existing_sets_video_path_from_config(tmp_path):
    # Create a fake output_dir with run_config.yaml that has video_path
    video_file = tmp_path / "video.mp4"
    video_file.touch()
    config = {"video_path": str(video_file), "backend": "vggt_omega"}
    config_path = tmp_path / "run_config.yaml"
    config_path.write_text(yaml.dump(config))

    app = App(base_dir=str(tmp_path))
    app._build_sidebar()  # initialise widgets
    app._output_dir_input.value = str(tmp_path)

    class FakeEvent:
        pass

    app._on_confirm_session(FakeEvent())
    assert app._state.video_path == video_file


def test_load_existing_no_video_path_in_config(tmp_path):
    # run_config.yaml exists but has no video_path — state.video_path stays None
    config = {"backend": "vggt_omega"}
    config_path = tmp_path / "run_config.yaml"
    config_path.write_text(yaml.dump(config))

    app = App(base_dir=str(tmp_path))
    app._build_sidebar()
    app._output_dir_input.value = str(tmp_path)

    class FakeEvent:
        pass

    app._on_confirm_session(FakeEvent())
    assert app._state.video_path is None
```

- [ ] **Step 3: Run new tests — expect failure**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_app.py::test_load_existing_sets_video_path_from_config tests/dashboard/test_app.py::test_load_existing_no_video_path_in_config -v
```

Expected: FAIL — `state.video_path` is `None` (not set).

- [ ] **Step 4: Update `_on_confirm_session` in `app.py`**

Replace the `else` branch (load-existing path) in `_on_confirm_session`:

```python
        else:
            out_dir = Path(self._output_dir_input.value.strip())
            config_file = out_dir / "run_config.yaml"
            if not config_file.exists():
                self._session_status.object = (
                    f"<p style='color:#e05050;font-size:12px'>No run_config.yaml in {out_dir}</p>"
                )
                return
            self._state.output_dir = out_dir

            # Parse video_path from config if present and file exists
            import yaml  # noqa: PLC0415
            try:
                config = yaml.safe_load(config_file.read_text())
                raw_vp = config.get("video_path")
                if raw_vp:
                    vp = Path(raw_vp)
                    if vp.exists():
                        self._state.video_path = vp
            except Exception:
                pass  # bad YAML — ignore, video_path stays None

            self._session_status.object = (
                f"<p style='color:#50c050;font-size:12px'>Loaded: {out_dir.name}</p>"
            )
```

- [ ] **Step 5: Run tests — all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_app.py -v
```

Expected: new tests PASS, existing tests PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/app.py tests/dashboard/test_app.py
git commit -m "fix(dashboard): parse video_path from run_config.yaml on load-existing"
```

---

## Task 3: Add zarr write helper + update extraction thread

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `tests/dashboard/test_preprocess.py`

- [ ] **Step 1: Add failing test for `_write_frames_zarr`**

Add to `tests/dashboard/test_preprocess.py`:

```python
import tempfile
from pathlib import Path

import numpy as np
import zarr

from collab_splats.dashboard.panes.preprocess import _write_frames_zarr


def test_write_frames_zarr_creates_store(tmp_path):
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(5)]
    frames[2][60, 80, 0] = 255  # non-zero value to verify data
    out = tmp_path / "frames.zarr"
    _write_frames_zarr(frames, out)
    store = zarr.open(str(out), mode="r")
    arr = store["frames"]
    assert arr.shape == (5, 120, 160, 3)
    assert arr.dtype == np.uint8
    assert arr[2, 60, 80, 0] == 255


def test_write_frames_zarr_chunk_per_frame(tmp_path):
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "frames.zarr"
    _write_frames_zarr(frames, out)
    store = zarr.open(str(out), mode="r")
    arr = store["frames"]
    assert arr.chunks == (1, 120, 160, 3)


def test_write_frames_zarr_overwrites(tmp_path):
    frames_a = [np.ones((10, 10, 3), dtype=np.uint8) * 100 for _ in range(3)]
    frames_b = [np.ones((10, 10, 3), dtype=np.uint8) * 200 for _ in range(2)]
    out = tmp_path / "frames.zarr"
    _write_frames_zarr(frames_a, out)
    _write_frames_zarr(frames_b, out)
    store = zarr.open(str(out), mode="r")
    assert store["frames"].shape == (2, 10, 10, 3)
    assert store["frames"][0, 0, 0, 0] == 200
```

- [ ] **Step 2: Run — expect ImportError/AttributeError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_write_frames_zarr_creates_store tests/dashboard/test_preprocess.py::test_write_frames_zarr_chunk_per_frame tests/dashboard/test_preprocess.py::test_write_frames_zarr_overwrites -v
```

Expected: ImportError `cannot import name '_write_frames_zarr'`.

- [ ] **Step 3: Add `_write_frames_zarr` to `preprocess.py`**

Add after the existing imports at the top of `preprocess.py`:

```python
import base64
import zarr
from numcodecs import Blosc
```

Add after `_frames_to_thumbnails` function:

```python
def _write_frames_zarr(frames: list[np.ndarray], path: Path) -> None:
    """Write frame list to a zarr store at path, one chunk per frame."""
    arr = np.stack(frames)  # (N, H, W, 3) uint8
    store = zarr.open(str(path), mode="w")
    store.create_dataset(
        "frames",
        data=arr,
        chunks=(1, *arr.shape[1:]),
        compressor=Blosc(cname="lz4", clevel=3),
        overwrite=True,
    )
```

- [ ] **Step 4: Run zarr tests — all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_write_frames_zarr_creates_store tests/dashboard/test_preprocess.py::test_write_frames_zarr_chunk_per_frame tests/dashboard/test_preprocess.py::test_write_frames_zarr_overwrites -v
```

Expected: 3 PASS.

- [ ] **Step 5: Update `_run_extraction` to write zarr + set new state fields**

Replace the section in `_run_extraction` from `self._selected_frames = frames` through `self._state.frames = frames`:

```python
            self._selected_frames = frames
            self._selected_indices = list(range(len(frames)))

            # Write frames to zarr; set state fields (no in-memory frame list)
            if self._state.output_dir is None and self._state.video_path is not None:
                self._state.output_dir = Path("/workspace/outputs") / Path(self._state.video_path).stem
            zarr_path = Path(self._state.output_dir) / "frames.zarr"
            _write_frames_zarr(frames, zarr_path)
            self._state.frames_zarr_path = zarr_path
            self._state.selected_indices = self._selected_indices
```

Remove the old block `if self._state.output_dir is None ...` that followed (it's now part of the zarr block above).

- [ ] **Step 6: Update frame strip to load thumbnails from zarr**

Replace the thumbnail generation block in `_run_extraction`:

```python
            # Load thumbnails lazily from zarr (cap at 100 for performance)
            store = zarr.open(str(zarr_path), mode="r")
            n_thumbs = min(100, len(frames))
            thumb_frames = [store["frames"][i] for i in range(n_thumbs)]
            thumbnails = _frames_to_thumbnails(thumb_frames)
            self._active_frame_idx = 0
            self._frame_strip_pane.object = _build_frame_strip_html(thumbnails, active_idx=0)
```

- [ ] **Step 7: Run existing test suite — no regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v
```

Expected: all pass except possibly `test_render_metrics_figure_*` (those will be removed in Task 4).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): write extracted frames to zarr; lazy thumbnail load"
```

---

## Task 4: HTML frame strip with scrollIntoView

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `tests/dashboard/test_preprocess.py`

- [ ] **Step 1: Add failing tests for `_build_frame_strip_html`**

Add to `tests/dashboard/test_preprocess.py`:

```python
from collab_splats.dashboard.panes.preprocess import _build_frame_strip_html


def test_build_frame_strip_html_contains_ids():
    thumbnails = [b"\x89PNG\r\n" + b"x" * 50 for _ in range(3)]
    html = _build_frame_strip_html(thumbnails, active_idx=1)
    assert 'id="frame-0"' in html
    assert 'id="frame-1"' in html
    assert 'id="frame-2"' in html


def test_build_frame_strip_html_active_border():
    thumbnails = [b"\x89PNG\r\n" + b"x" * 50 for _ in range(3)]
    html = _build_frame_strip_html(thumbnails, active_idx=2)
    # active frame gets green border, others get dark border
    assert html.count("#50c050") == 1
    assert html.count("#333") == 2


def test_build_frame_strip_html_base64_src():
    thumbnails = [b"PNGDATA"]
    html = _build_frame_strip_html(thumbnails, active_idx=0)
    import base64
    expected = base64.b64encode(b"PNGDATA").decode()
    assert expected in html


def test_build_frame_strip_html_empty():
    html = _build_frame_strip_html([], active_idx=0)
    assert isinstance(html, str)
```

- [ ] **Step 2: Run — expect ImportError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_build_frame_strip_html_contains_ids tests/dashboard/test_preprocess.py::test_build_frame_strip_html_active_border tests/dashboard/test_preprocess.py::test_build_frame_strip_html_base64_src tests/dashboard/test_preprocess.py::test_build_frame_strip_html_empty -v
```

Expected: ImportError `cannot import name '_build_frame_strip_html'`.

- [ ] **Step 3: Add `_build_frame_strip_html` to `preprocess.py`**

Add after `_write_frames_zarr`:

```python
def _build_frame_strip_html(thumbnails: list[bytes], active_idx: int) -> str:
    """Return HTML string for horizontal frame strip with base64 thumbnails.

    Each thumbnail has id="frame-{i}" for scrollIntoView targeting.
    Active frame gets green border; others get dark border.
    """
    if not thumbnails:
        return "<div style='color:#666;font-size:11px;padding:8px'>No frames extracted yet</div>"

    imgs = []
    for i, png_bytes in enumerate(thumbnails):
        b64 = base64.b64encode(png_bytes).decode()
        border_color = "#50c050" if i == active_idx else "#333"
        imgs.append(
            f'<img id="frame-{i}" src="data:image/png;base64,{b64}" '
            f'style="width:120px;height:90px;cursor:pointer;margin:2px;'
            f'border:2px solid {border_color};border-radius:3px;flex-shrink:0;" />'
        )

    inner = "".join(imgs)
    return (
        f'<div id="frame-strip-container" '
        f'style="display:flex;flex-direction:row;overflow-x:auto;'
        f'padding:4px;background:#0d1117;border-radius:4px;">'
        f"{inner}</div>"
    )
```

- [ ] **Step 4: Run HTML strip tests — all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_build_frame_strip_html_contains_ids tests/dashboard/test_preprocess.py::test_build_frame_strip_html_active_border tests/dashboard/test_preprocess.py::test_build_frame_strip_html_base64_src tests/dashboard/test_preprocess.py::test_build_frame_strip_html_empty -v
```

Expected: 4 PASS.

- [ ] **Step 5: Replace `_frame_strip_row` widget with `pn.pane.HTML` + scroll script pane in `PreprocessPane.__init__`**

In `__init__`, remove these lines:
```python
        self._frame_strip_row = pn.Row(scroll=True, height=150, sizing_mode="stretch_width")
        self._frame_strip_label = pn.pane.HTML("", sizing_mode="stretch_width")
```

Add:
```python
        self._frame_strip_pane = pn.pane.HTML(
            _build_frame_strip_html([], active_idx=0),
            sizing_mode="stretch_width",
            height=120,
        )
        self._scroll_script = pn.pane.HTML("", width=0, height=0)
        self._active_frame_idx: int = 0
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): HTML frame strip with base64 thumbnails and scrollIntoView support"
```

---

## Task 5: Replace matplotlib metrics with Bokeh + TapTool

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `tests/dashboard/test_preprocess.py`

- [ ] **Step 1: Add imports for Bokeh at top of `preprocess.py`**

Add to the imports section:

```python
from bokeh.models import ColumnDataSource, Span, TapTool
from bokeh.plotting import figure as bokeh_figure
```

- [ ] **Step 2: Add failing test for `_build_metrics_sources`**

Add to `tests/dashboard/test_preprocess.py`:

```python
from collab_splats.dashboard.panes.preprocess import _build_metrics_sources


def test_build_metrics_sources_returns_dict():
    scores = {
        "disparity": [1.0, 2.0, 3.0],
        "rotation": [0.1, 0.2, 0.3],
        "hist_similarity": [0.9, 0.8, 0.7],
    }
    sources = _build_metrics_sources(scores)
    assert set(sources.keys()) == {"disparity", "rotation", "hist_similarity"}


def test_build_metrics_sources_data_shape():
    scores = {"disparity": [10.0, 20.0, 30.0], "rotation": [], "hist_similarity": []}
    sources = _build_metrics_sources(scores)
    assert list(sources["disparity"].data["x"]) == [0, 1, 2]
    assert list(sources["disparity"].data["y"]) == [10.0, 20.0, 30.0]


def test_build_metrics_sources_skips_empty():
    scores = {"disparity": [], "rotation": [1.0, 2.0], "hist_similarity": []}
    sources = _build_metrics_sources(scores)
    assert "disparity" not in sources
    assert "hist_similarity" not in sources
    assert "rotation" in sources
```

- [ ] **Step 3: Run — expect ImportError**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_build_metrics_sources_returns_dict tests/dashboard/test_preprocess.py::test_build_metrics_sources_data_shape tests/dashboard/test_preprocess.py::test_build_metrics_sources_skips_empty -v
```

Expected: ImportError `cannot import name '_build_metrics_sources'`.

- [ ] **Step 4: Add `_build_metrics_sources` pure helper to `preprocess.py`**

Add after `_build_frame_strip_html`:

```python
_METRIC_STYLE: dict[str, tuple[str, str]] = {
    "disparity": ("Optical Flow / Disparity", "#7ec8e3"),
    "rotation": ("Rotation (°)", "#f0a500"),
    "hist_similarity": ("Hist. Similarity", "#d090e0"),
}


def _build_metrics_sources(
    frame_scores: dict[str, list[float]],
) -> dict[str, ColumnDataSource]:
    """Build Bokeh ColumnDataSources for each non-empty metric series."""
    sources = {}
    for key in _METRIC_STYLE:
        vals = frame_scores.get(key, [])
        if not vals:
            continue
        sources[key] = ColumnDataSource(
            data={"x": list(range(len(vals))), "y": vals}
        )
    return sources
```

- [ ] **Step 5: Run sources tests — all pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_preprocess.py::test_build_metrics_sources_returns_dict tests/dashboard/test_preprocess.py::test_build_metrics_sources_data_shape tests/dashboard/test_preprocess.py::test_build_metrics_sources_skips_empty -v
```

Expected: 3 PASS.

- [ ] **Step 6: Add `_make_metrics_panel` method + `_seek_to_frame` to `PreprocessPane`**

Add these two methods to `PreprocessPane` (after `_on_extract`):

```python
    def _seek_to_frame(self, frame_idx: int) -> None:
        """Seek video to frame_idx; update active thumbnail in strip; scroll strip."""
        self._active_frame_idx = frame_idx
        # Seek video player
        if self._state.video_path:
            info = get_video_info(str(self._state.video_path))
            fps = info.get("fps", 25.0)
            self._video_pane.time = frame_idx / fps

        # Rebuild strip HTML with new active frame highlighted
        store = zarr.open(str(self._state.frames_zarr_path), mode="r")
        n_thumbs = min(100, store["frames"].shape[0])
        thumb_frames = [store["frames"][i] for i in range(n_thumbs)]
        thumbnails = _frames_to_thumbnails(thumb_frames)
        self._frame_strip_pane.object = _build_frame_strip_html(thumbnails, active_idx=frame_idx)

        # Inject scroll script
        self._scroll_script.object = (
            f"<script>var el=document.getElementById('frame-{frame_idx}');"
            f"if(el)el.scrollIntoView({{behavior:'smooth',inline:'center'}});</script>"
        )

    def _make_metrics_panel(self) -> pn.Column:
        """Build stacked Bokeh metric figures with TapTool; return as pn.Column."""
        sources = _build_metrics_sources(self._frame_scores)
        if not sources:
            return pn.Column(
                pn.pane.HTML(
                    "<p style='color:#666;font-size:11px'>No metrics — run extraction first</p>"
                )
            )

        figs = []
        for key, source in sources.items():
            label, color = _METRIC_STYLE[key]
            p = bokeh_figure(
                height=110,
                sizing_mode="stretch_width",
                toolbar_location=None,
                x_range=(0, max(source.data["x"]) + 1),
            )
            p.background_fill_color = "#111827"
            p.border_fill_color = "#0d1117"
            p.outline_line_color = "#333"
            p.grid.grid_line_color = "#333"
            p.xaxis.axis_label = ""
            p.yaxis.axis_label = label
            p.yaxis.axis_label_text_color = color
            p.yaxis.axis_label_text_font_size = "10px"
            p.xaxis.major_label_text_color = "#555"
            p.yaxis.major_label_text_color = "#555"

            # Line + invisible circles for tap selection
            p.line("x", "y", source=source, color=color, line_width=1.2, alpha=0.9)
            circles = p.circle("x", "y", source=source, size=6, alpha=0, color=color)

            # Green spans at selected frame indices
            for idx in self._selected_indices:
                p.add_layout(Span(location=idx, dimension="height", line_color="#50c050",
                                  line_alpha=0.5, line_width=1.0))

            # TapTool fires Python callback via source selection
            tap = TapTool(renderers=[circles])
            p.add_tools(tap)

            def _on_tap(attr, old, new, src=source):  # noqa: ANN001
                if new:
                    x_val = src.data["x"][new[0]]
                    # Snap to nearest selected frame index
                    if self._selected_indices:
                        nearest = min(self._selected_indices, key=lambda i: abs(i - x_val))
                        self._seek_to_frame(nearest)

            source.selected.on_change("indices", _on_tap)
            figs.append(pn.pane.Bokeh(p, sizing_mode="stretch_width"))

        return pn.Column(*figs, sizing_mode="stretch_width")
```

Also add `self._metrics_col = pn.Column(sizing_mode="stretch_width")` in `__init__` (replace `self._metrics_pane`):

```python
        self._metrics_col = pn.Column(sizing_mode="stretch_width", visible=False)
```

- [ ] **Step 7: Update `_run_extraction` — replace metrics pane update with metrics column rebuild**

Replace the metrics update block in `_run_extraction` (the `metrics_png = ...`, `self._metrics_pane.object = ...`, `self._metrics_pane.visible = True` lines):

```python
            # Rebuild Bokeh metrics panel with new data
            self._metrics_col.objects = [self._make_metrics_panel()]
            self._metrics_col.visible = True
```

Also replace the frame strip update block (old `pn.pane.PNG` row):
```python
            # Frame strip is updated in _seek_to_frame; render initial state here
            store = zarr.open(str(zarr_path), mode="r")
            n_thumbs = min(100, len(frames))
            thumb_frames = [store["frames"][i] for i in range(n_thumbs)]
            thumbnails = _frames_to_thumbnails(thumb_frames)
            self._active_frame_idx = 0
            self._frame_strip_pane.object = _build_frame_strip_html(thumbnails, active_idx=0)
```

- [ ] **Step 8: Remove old matplotlib tests from `test_preprocess.py`**

Remove `test_render_metrics_figure_returns_bytes` and `test_render_metrics_figure_empty_scores` from `tests/dashboard/test_preprocess.py` — `_render_metrics_figure` no longer exists.

Also remove the import of `_render_metrics_figure` from the top of `test_preprocess.py`.

- [ ] **Step 9: Remove `_render_metrics_figure` from `preprocess.py`**

Delete the `_render_metrics_figure` function entirely.

Remove the old `self._metrics_pane` widget from `__init__`.

Remove the `matplotlib` import block from `preprocess.py`:
```python
import matplotlib
if not matplotlib.is_interactive():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- [ ] **Step 10: Run test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v
```

Expected: all PASS.

- [ ] **Step 11: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): replace matplotlib metrics with Bokeh TapTool figures; seek-to-frame on tap"
```

---

## Task 6: Wrap controls in pn.Card; gate visibility on video_path; auto-collapse after extraction

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`

- [ ] **Step 1: In `PreprocessPane.__init__`, build `_controls_card`**

After the existing control widget definitions (just before `# Wire callbacks`), add:

```python
        # Controls wrapped in collapsible Card — hidden until video is loaded
        self._controls_card = pn.Card(
            self._method_dd,
            self._fps_slider,
            self._n_frames_slider,
            self._window_start_slider,
            self._window_end_slider,
            self._min_disparity_slider,
            pn.layout.Divider(),
            self._extract_btn,
            self._frame_count_html,
            title="Frame Extraction",
            collapsed=False,
            visible=False,
            width=400,
        )
```

- [ ] **Step 2: Watch `video_path` to show controls card**

In `__init__`, replace the existing `self._state.param.watch(self._on_video_path_change, "video_path")` with:

```python
        self._state.param.watch(self._on_video_path_change, "video_path")
```

(already exists — no change needed for the watch registration itself)

Update `_on_video_path_change`:

```python
    def _on_video_path_change(self, event: Any) -> None:
        """Auto-load video display and reveal controls when AppState.video_path is set."""
        if event.new and Path(event.new).exists():
            self._load_video(Path(event.new))
            self._controls_card.visible = True
```

- [ ] **Step 3: Auto-collapse card after successful extraction**

At the end of `_run_extraction`, just before `self._op_log.finish_op()`:

```python
            # Auto-collapse controls card now that extraction is done
            self._controls_card.collapsed = True
```

- [ ] **Step 4: Run smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/test_smoke.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py
git commit -m "feat(dashboard): controls in collapsible Card, gated on video_path, auto-collapses post-extraction"
```

---

## Task 7: Update `panel()` layout

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`

- [ ] **Step 1: Rewrite `panel()` method**

Replace the entire `panel()` method:

```python
    def panel(self) -> pn.Row:
        """Return the full PreprocessPane Panel layout."""
        # Left column: video player + info
        video_col = pn.Column(
            self._video_pane,
            self._video_info_html,
            sizing_mode="stretch_height",
            min_width=400,
        )

        # Right column: controls card → metrics → frame strip → scroll script
        right_col = pn.Column(
            self._controls_card,
            pn.layout.Divider(),
            pn.pane.HTML(
                "<h4 style='color:#7ec8e3;margin:4px 0'>Frame Quality Metrics</h4>",
                sizing_mode="stretch_width",
            ),
            self._metrics_col,
            pn.layout.Divider(),
            pn.pane.HTML(
                "<h4 style='color:#7ec8e3;margin:4px 0'>Selected Frames</h4>",
                sizing_mode="stretch_width",
            ),
            self._frame_strip_pane,
            self._scroll_script,
            sizing_mode="stretch_both",
        )

        return pn.Row(video_col, right_col, sizing_mode="stretch_width")
```

- [ ] **Step 2: Run full dashboard test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py
git commit -m "feat(dashboard): updated PreprocessPane layout — video left, card+metrics+strip right"
```

---

## Task 8: Clean up + full test run

**Files:**
- Modify: `tests/dashboard/test_preprocess.py` (remove dead imports)

- [ ] **Step 1: Verify no stale imports in test_preprocess.py**

Check that `_render_metrics_figure` and `_frames_to_thumbnails` imports at top of `test_preprocess.py` are accurate — remove any that no longer exist in `preprocess.py`.

`_frames_to_thumbnails` is still used internally (not exported for test), so remove it from the test import if it was there. `_window_frame_indices` is still present — keep that import.

Updated import block for `test_preprocess.py`:

```python
import base64
import tempfile
from pathlib import Path

import numpy as np
import zarr

from collab_splats.dashboard.panes.preprocess import (
    _build_frame_strip_html,
    _build_metrics_sources,
    _window_frame_indices,
    _write_frames_zarr,
)
```

- [ ] **Step 2: Run complete test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/dashboard/ -v
```

Expected: all PASS, no warnings about missing imports.

- [ ] **Step 3: Final commit**

```bash
git add tests/dashboard/test_preprocess.py
git commit -m "chore(dashboard): clean up stale imports in test_preprocess after matplotlib removal"
```

---

## Self-Review

**Spec coverage:**
- ✅ Video on left when video_path set — Task 6 (`_on_video_path_change` shows video + card)
- ✅ Extraction controls gated on video load — Task 6 (`_controls_card.visible = False` until video)
- ✅ Controls collapsible + re-runnable — Task 6 (`pn.Card`, auto-collapses after extract)
- ✅ Zarr write with Blosc/lz4 + chunk-per-frame — Task 3
- ✅ AppState schema change — Task 1
- ✅ Bokeh metrics with TapTool — Task 5
- ✅ `_seek_to_frame` → seek video + rebuild strip — Task 5
- ✅ `scrollIntoView` on tap — Task 5 (`_scroll_script` pane)
- ✅ Config load bug fix — Task 2
- ✅ HTML frame strip with `id="frame-{i}"` — Task 4
- ✅ Metrics + strip hidden until extraction done — Task 5 (`_metrics_col.visible = False` default)

**Placeholder scan:** No TBDs. All code blocks complete.

**Type consistency:**
- `_write_frames_zarr(frames: list[np.ndarray], path: Path) -> None` — consistent across Tasks 3, 5
- `_build_frame_strip_html(thumbnails: list[bytes], active_idx: int) -> str` — consistent Tasks 4, 5
- `_build_metrics_sources(frame_scores: dict[str, list[float]]) -> dict[str, ColumnDataSource]` — consistent Tasks 5
- `self._metrics_col` (not `self._metrics_pane`) — consistently used Tasks 5, 7
- `self._frame_strip_pane` (not `self._frame_strip_row`) — consistently used Tasks 3, 4, 5, 7
- `self._scroll_script` — consistently used Tasks 4, 5, 7
- `state.frames_zarr_path` / `state.selected_indices` — consistent Tasks 1, 3
