# Method-Coupled Frame Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple frame scoring from extraction so OF metrics are only computed when the OF method is selected, eliminating the double video decode and the misleading per-frame chart in FPS mode.

**Architecture:** `sample_frames_optical_flow` returns `(frames, scores)` instead of just `frames` — scores are computed during the existing OF loop at no extra cost. `_run_extraction` drops `score_all_frames` entirely and dispatches to a new `_render_fps_raster` (FPS mode) or the existing `_make_metrics_panel` (OF mode, fed by the returned scores). All non-dashboard callers unpack `(frames, _)`.

**Tech Stack:** Python 3.11, cv2, numpy, Panel, Bokeh (`ColumnDataSource`, `bokeh_figure`).

**Python interpreter:** `/opt/conda/envs/reconstruction/bin/python`  
**Test runner:** `/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q`  
**Spec:** `docs/superpowers/specs/2026-05-28-method-coupled-frame-scoring-design.md`

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Change `sample_frames_optical_flow` return type; capture scores in OF loop |
| `collab_splats/wrapper/reconstructor.py` | Unpack `(frame_arrays, _)` |
| `collab_splats/wrapper/splatter.py` | Unpack `(sampled_frames, _)` |
| `collab_splats/dashboard/panes/preprocess.py` | Add `_render_fps_raster`; refactor `_run_extraction` |
| `tests/utils/test_frame_sampling.py` | Update OF tests to unpack tuple |
| `tests/dashboard/test_preprocess.py` | Add `_render_fps_raster` test |

---

## Task 1: Change `sample_frames_optical_flow` return type

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py:562-604`
- Test: `tests/utils/test_frame_sampling.py`

- [ ] **Step 1.1: Write failing tests for new return type**

Find the OF tests (around line 34) in `tests/utils/test_frame_sampling.py`. They currently do `frames = sample_frames_optical_flow(...)`. Update all of them to unpack a tuple. Also add a test for the returned scores structure.

Replace the existing OF test block (search for `test_optical_flow_accepts_max_frames` through the end of the OF section):

```python
def test_optical_flow_accepts_max_frames(tiny_video):
    frames, scores = sample_frames_optical_flow(
        tiny_video, max_frames=3, verbose=False
    )
    assert len(frames) <= 3


def test_optical_flow_first_frame_always_included(tiny_video):
    frames, scores = sample_frames_optical_flow(tiny_video, verbose=False)
    assert len(frames) >= 1


def test_optical_flow_frames_are_rgb(tiny_video):
    frames, scores = sample_frames_optical_flow(tiny_video, verbose=False)
    assert len(frames) >= 1
    assert frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8


def test_optical_flow_returns_scores_for_each_frame(tiny_video):
    """scores list has one entry per selected frame with required keys."""
    frames, scores = sample_frames_optical_flow(tiny_video, verbose=False)
    assert len(scores) == len(frames)
    for s in scores:
        assert "disparity" in s
        assert "rotation" in s
        assert "histogram_similarity" in s
        assert "score" in s
        assert s["selected"] is True


def test_optical_flow_calls_on_progress(tiny_video):
    calls = []
    frames, _ = sample_frames_optical_flow(
        tiny_video, on_progress=lambda c, t: calls.append((c, t)), verbose=False
    )
    assert len(calls) > 0
    assert all(n <= t for n, t in calls)


def test_optical_flow_resizes_for_analysis(tiny_video, monkeypatch):
    resize_calls = []
    original_resize = cv2.resize

    def patched_resize(src, dsize, **kwargs):
        resize_calls.append(dsize)
        return original_resize(src, dsize, **kwargs)

    monkeypatch.setattr(cv2, "resize", patched_resize)
    frames, _ = sample_frames_optical_flow(tiny_video, max_frames=3)
    assert len(frames) > 0
```

Also update the ffmpeg backend test near the bottom of the file (find `test_sample_frames_optical_flow_ffmpeg_backend`):

```python
def test_sample_frames_optical_flow_ffmpeg_backend(tiny_video, monkeypatch):
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    frames, scores = fs.sample_frames_optical_flow(tiny_video, verbose=False)
    assert isinstance(frames, list)
    assert isinstance(scores, list)
    assert len(frames) == len(scores)
    for f in frames:
        assert f.shape[2] == 3
        assert f.dtype == np.uint8
```

- [ ] **Step 1.2: Run to confirm failures**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -k "optical_flow" -x -q 2>&1 | tail -10
```

Expected: multiple failures — `cannot unpack non-sequence` / `too many values to unpack`.

- [ ] **Step 1.3: Update `sample_frames_optical_flow` in `frame_sampling.py`**

Find the function at line ~562. Replace the entire body:

```python
def sample_frames_optical_flow(
    video_path: str,
    min_disparity: float = 50.0,
    max_frames: int = 200,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select keyframes using sparse Lucas-Kanade optical flow.

    Combines motion (disparity + rotation) and visual diversity (histogram
    similarity) into a 0–1 score; selects frames scoring >= 0.5.
    OF analysis runs at max 480px wide for speed; selected frames kept full-res.

    Returns (frames, scores) where scores has one dict per selected frame:
        {frame_idx, disparity, rotation, histogram_similarity, score, selected}.
    on_progress: called as on_progress(frames_decoded, total_frames).
    min_disparity: mean pixel displacement threshold. Higher = fewer frames.
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    info = get_video_info(video_path)
    total = info["total_frames"]
    frames: list[np.ndarray] = []
    scores: list[dict] = []
    frames_decoded = 0
    with tqdm(total=total, desc="Optical flow selection", unit="frame", disable=not verbose) as pbar:
        for frame in _iter_decoded_frames(video_path, info["width"], info["height"]):
            if len(frames) >= max_frames:
                break
            frames_decoded += 1
            pbar.update(1)
            if on_progress is not None:
                on_progress(frames_decoded, total)
            # Score at 480px-wide scale for speed; keep full-res copy if selected
            scale = min(1.0, 480.0 / frame.shape[1])
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
            should_select, score, components = selector.should_select_frame(small)
            if should_select:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                scores.append({
                    "frame_idx": len(frames) - 1,
                    "disparity": components.get("disparity", 0.0),
                    "rotation": components.get("rotation", 0.0),
                    "histogram_similarity": components.get("histogram_similarity", 1.0),
                    "score": score,
                    "selected": True,
                })
                selector.accept_frame(small)
    return frames, scores
```

- [ ] **Step 1.4: Run OF tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -k "optical_flow" -x -q 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 1.5: Run full frame_sampling suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -x -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 1.6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): sample_frames_optical_flow returns (frames, scores) tuple"
```

---

## Task 2: Update non-dashboard callers

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:68`
- Modify: `collab_splats/wrapper/splatter.py:311`

- [ ] **Step 2.1: Update `reconstructor.py`**

Find the line (around line 68):
```python
        frame_arrays = sample_frames_optical_flow(
            video_path=str(input_path),
            max_frames=max_frames if max_frames is not None else 200,
        )
```

Replace with:
```python
        frame_arrays, _ = sample_frames_optical_flow(
            video_path=str(input_path),
            max_frames=max_frames if max_frames is not None else 200,
        )
```

- [ ] **Step 2.2: Update `splatter.py`**

Find the line (around line 311):
```python
            sampled_frames = sample_frames_optical_flow(file_path.as_posix(), max_frames=min(n_samples, 200), verbose=False)
```

Replace with:
```python
            sampled_frames, _ = sample_frames_optical_flow(file_path.as_posix(), max_frames=min(n_samples, 200), verbose=False)
```

- [ ] **Step 2.3: Verify imports still work**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.wrapper.reconstructor import _extract_frames; from collab_splats.wrapper.splatter import Splatter; print('ok')"
```

Expected: `ok`

- [ ] **Step 2.4: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py collab_splats/wrapper/splatter.py
git commit -m "fix(wrapper): unpack (frames, _) from sample_frames_optical_flow"
```

---

## Task 3: Add `_render_fps_raster` to `preprocess.py`

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Test: `tests/dashboard/test_preprocess.py`

- [ ] **Step 3.1: Write failing test**

Append to `tests/dashboard/test_preprocess.py`:

```python
def test_render_fps_raster_returns_panel_column():
    """_render_fps_raster returns a pn.Column for non-empty indices."""
    from collab_splats.dashboard.panes.preprocess import _render_fps_raster
    result = _render_fps_raster([0, 15, 30, 45, 60], total_frames=90)
    assert isinstance(result, pn.Column)


def test_render_fps_raster_empty_indices():
    """_render_fps_raster returns a pn.Column even with empty index list."""
    from collab_splats.dashboard.panes.preprocess import _render_fps_raster
    result = _render_fps_raster([], total_frames=90)
    assert isinstance(result, pn.Column)
```

- [ ] **Step 3.2: Run to confirm failures**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py::test_render_fps_raster_returns_panel_column tests/dashboard/test_preprocess.py::test_render_fps_raster_empty_indices -v 2>&1 | tail -8
```

Expected: `ImportError: cannot import name '_render_fps_raster'`

- [ ] **Step 3.3: Add `_render_fps_raster` to `preprocess.py`**

Insert this function inside the `Pure helpers` section, after `_build_metrics_sources` (around line 96), before `_frames_to_thumbnails`:

```python
def _render_fps_raster(
    sampled_indices: list[int],
    total_frames: int,
) -> pn.Column:
    """Build a Bokeh bar chart showing sampled frame positions across the video timeline."""
    if not sampled_indices:
        return pn.Column(
            pn.pane.HTML("<p style='color:#666;font-size:11px'>No frames extracted</p>")
        )
    source = ColumnDataSource(data={
        "left": [i - 0.4 for i in sampled_indices],
        "right": [i + 0.4 for i in sampled_indices],
        "top": [1.0] * len(sampled_indices),
        "bottom": [0.0] * len(sampled_indices),
    })
    p = bokeh_figure(
        height=80,
        sizing_mode="stretch_width",
        toolbar_location=None,
        x_range=(0, max(total_frames, 1)),
        y_range=(0, 1.2),
        title=f"Sampled frame positions ({len(sampled_indices)} frames)",
    )
    p.background_fill_color = "#111827"
    p.border_fill_color = "#0d1117"
    p.outline_line_color = None
    p.title.text_color = "#aaa"
    p.title.text_font_size = "10pt"
    p.quad(
        top="top", bottom="bottom", left="left", right="right",
        source=source, color="#50c050", alpha=0.7,
    )
    p.yaxis.visible = False
    p.xaxis.axis_label = "Frame index"
    p.xaxis.axis_label_text_color = "#aaa"
    p.xaxis.major_label_text_color = "#aaa"
    return pn.Column(pn.pane.Bokeh(p, sizing_mode="stretch_width"), sizing_mode="stretch_width")
```

- [ ] **Step 3.4: Add `_render_fps_raster` to test imports**

In `tests/dashboard/test_preprocess.py`, find the import block and add `_render_fps_raster`:

```python
from collab_splats.dashboard.panes.preprocess import (
    PreprocessPane,
    ThrottledProgress,
    _build_metrics_sources,
    _frames_to_thumbnails,
    _render_fps_raster,
    _window_frame_indices,
    _write_frames_zarr,
)
```

- [ ] **Step 3.5: Run new tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py::test_render_fps_raster_returns_panel_column tests/dashboard/test_preprocess.py::test_render_fps_raster_empty_indices -v 2>&1 | tail -8
```

Expected: both `PASSED`

- [ ] **Step 3.6: Run full preprocess suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py -x -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 3.7: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): add _render_fps_raster Bokeh helper for FPS frame position chart"
```

---

## Task 4: Refactor `_run_extraction` — remove `score_all_frames`, method-coupled metrics

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`

- [ ] **Step 4.1: Remove `score_all_frames` from the import in `preprocess.py`**

Find the import block (around line 27):
```python
from collab_splats.utils.frame_sampling import (
    get_video_info,
    load_video_frames,
    sample_frames_fps,
    sample_frames_optical_flow,
    score_all_frames,
)
```

Replace with:
```python
from collab_splats.utils.frame_sampling import (
    get_video_info,
    load_video_frames,
    sample_frames_fps,
    sample_frames_optical_flow,
)
```

- [ ] **Step 4.2: Replace `_run_extraction` body**

Find `def _run_extraction(self, video_path: Path) -> None:` (around line 300). Replace the entire method body up to (but not including) `def _seek_to_frame`  with:

```python
    def _run_extraction(self, video_path: Path) -> None:
        """Background thread: score frames, extract keyframes, update UI state."""
        def _progress_raw(current: int, total: int, base: int, scale: int) -> None:
            pct = base + int(current / total * scale) if total > 0 else base
            self._op_log.update_progress(pct)
            self._progress_bar.value = pct

        _extract_progress = ThrottledProgress(lambda c, t: _progress_raw(c, t, 0, 100))

        try:
            self._progress_bar.value = 0
            self._progress_bar.visible = True

            # Probe total frame count for FPS raster x-axis
            info = get_video_info(str(video_path))
            total_frames = info["total_frames"]

            # Extract frames using selected method
            self._op_log.start_op("Extracting frames")
            self._progress_label.object = "<p style='font-size:11px;color:#aaa'>Extracting frames…</p>"
            method = self._method_dd.value
            frame_indices: list[int]
            of_scores: list[dict] = []
            if method == "fps":
                frames, frame_indices = sample_frames_fps(
                    str(video_path),
                    fps=self._fps_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=_extract_progress,
                    verbose=False,
                )
            else:
                frames, of_scores = sample_frames_optical_flow(
                    str(video_path),
                    min_disparity=self._min_disparity_slider.value,
                    max_frames=self._n_frames_slider.value,
                    on_progress=_extract_progress,
                    verbose=False,
                )
                frame_indices = list(range(len(frames)))

            # Apply window filter proportionally to extracted set
            if self._window_start_slider.value > 0.0 or self._window_end_slider.value < 1.0:
                n_total = len(frames)
                start_i = int(self._window_start_slider.value * n_total)
                end_i = max(start_i + 1, int(self._window_end_slider.value * n_total))
                frames = frames[start_i:end_i]
                frame_indices = frame_indices[start_i:end_i]
                of_scores = of_scores[start_i:end_i]

            self._selected_frames = frames
            self._selected_indices = frame_indices

            # Ensure output dir exists before writing zarr
            if self._state.output_dir is None and self._state.video_path is not None:
                self._state.output_dir = Path("/workspace/outputs") / Path(self._state.video_path).stem
            zarr_path = Path(self._state.output_dir) / "frames.zarr"
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            self._progress_label.object = "<p style='font-size:11px;color:#aaa'>Writing frames to disk…</p>"
            _write_frames_zarr(frames, zarr_path)
            self._state.frames_zarr_path = zarr_path
            self._state.selected_indices = self._selected_indices
            self._selected_frames = []

            # Rebuild metrics panel based on method
            if method == "fps":
                self._metrics_col.objects = [_render_fps_raster(frame_indices, total_frames)]
            else:
                self._frame_scores = {
                    "disparity": [s["disparity"] for s in of_scores],
                    "rotation": [s["rotation"] for s in of_scores],
                    "hist_similarity": [s["histogram_similarity"] for s in of_scores],
                }
                self._metrics_col.objects = [self._make_metrics_panel()]
            self._metrics_col.visible = True

            # Generate initial thumbnails and render HTML strip
            thumbnails = _frames_to_thumbnails(frames[:100])
            self._cached_thumbnails = thumbnails
            self._active_frame_idx = 0
            self._frame_strip_pane.object = _build_frame_strip_html(thumbnails, active_idx=0)
            self._frame_count_html.object = (
                f"<p style='font-size:11px;color:#50c050'>✓ {len(frames)} frames extracted</p>"
            )

            # Auto-collapse controls card now that extraction is done
            self._controls_card.collapsed = True
            self._op_log.finish_op()
        except Exception as exc:
            logger.exception("Frame extraction failed")
            self._op_log.error_op(str(exc))
            self._frame_count_html.object = (
                f"<p style='color:#e05050'>Extraction failed: {exc}</p>"
            )
        finally:
            self._progress_bar.visible = False
            self._progress_label.object = ""
            self._extract_btn.disabled = False
```

- [ ] **Step 4.3: Run preprocess tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py -x -q 2>&1 | tail -8
```

Expected: all pass.

- [ ] **Step 4.4: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -8
```

Expected: all pass.

- [ ] **Step 4.5: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py
git commit -m "feat(dashboard): method-coupled frame scoring — remove score_all_frames from extraction; FPS raster / OF chart"
```

---

## Self-Review Checklist

- [x] `sample_frames_optical_flow` returns `(frames, scores)` — Task 1
- [x] Scores have all required keys (`disparity`, `rotation`, `histogram_similarity`, `score`, `selected`) — Task 1 step 1.3
- [x] `reconstructor.py` caller updated — Task 2
- [x] `splatter.py` caller updated — Task 2
- [x] `_render_fps_raster` added and tested — Task 3
- [x] `score_all_frames` removed from `_run_extraction` — Task 4
- [x] FPS mode uses raster, OF mode uses `_make_metrics_panel` — Task 4
- [x] `of_scores` trimmed with window filter — Task 4 step 4.2
- [x] Progress flattened to 0→100% (single pass) — Task 4
- [x] `histogram_similarity` (OF key) → `hist_similarity` (frame_scores key) mapped correctly — Task 4 step 4.2
- [x] No TBDs or placeholder steps
- [x] `_render_fps_raster` name consistent across Task 3 and Task 4
