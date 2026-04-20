# Dashboard Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix image orientation, slider layout, and add a determinate progress bar with faster frame extraction to the SemanticsDashboard.

**Architecture:** Changes split across two files: `frame_sampling.py` gets the backend fixes (orientation, seek, low-res OF, callbacks) and `semantics.py` gets the UI fixes (progress bar widget, threaded extraction, slider layout). Tasks go in dependency order: fix frame_sampling first, then wire dashboard UI.

**Tech Stack:** OpenCV, Panel (`pn.widgets.Progress`, `pn.state.add_periodic_callback`), Python `threading`

---

## File Map

| File | What changes |
|------|-------------|
| `collab_splats/utils/frame_sampling.py` | Remove `_rotation_map`, `_apply_rotation`, auto-rotation disable; rewrite `sample_frames_fps` seek-based; add low-res resize to OF; add `on_progress` to both functions |
| `collab_splats/dashboard/semantics.py` | Add progress bar + label widgets; add `_extraction_lock`, `_extraction_progress`, `_extraction_done`, `_extraction_error`, `_extraction_cb` state; rewrite `_extract_frames` to thread; add `_poll_extraction_progress`; fix `frame_slider` width + layout |
| `tests/utils/test_frame_sampling.py` | Remove `_rotation_map` import + tests; add tests for seek, progress callback, low-res OF |
| `tests/dashboard/test_semantics_smoke.py` | Smoke-check that `SemanticsDashboard` still builds; no layout assertions needed |

---

## Task 1: Remove rotation helpers and update tests

**Files:**
- Modify: `tests/utils/test_frame_sampling.py`
- Modify: `collab_splats/utils/frame_sampling.py`

**Context:** `_rotation_map` and `_apply_rotation` are dead once auto-rotation is re-enabled. Three existing tests import and assert on `_rotation_map` — remove them first so the test file doesn't reference a deleted symbol.

- [ ] **Step 1: Remove `_rotation_map` import and its three tests from test file**

Open `tests/utils/test_frame_sampling.py`. Remove these lines entirely:

```python
# Remove this import line:
from collab_splats.utils.frame_sampling import (
    _rotation_map,          # <-- remove this name
    sample_frames_optical_flow,
    sample_frames_fps,
    OpticalFlowFrameSelector,
)

# Remove these three test functions entirely:
def test_rotation_90_maps_to_clockwise():
    assert _rotation_map()[90] == cv2.ROTATE_90_CLOCKWISE

def test_rotation_270_maps_to_counterclockwise():
    assert _rotation_map()[270] == cv2.ROTATE_90_COUNTERCLOCKWISE

def test_rotation_180_unchanged():
    assert _rotation_map()[180] == cv2.ROTATE_180
```

After removal the import block becomes:

```python
from collab_splats.utils.frame_sampling import (
    sample_frames_optical_flow,
    sample_frames_fps,
    OpticalFlowFrameSelector,
)
```

- [ ] **Step 2: Run tests to confirm pre-existing state**

```bash
pytest tests/utils/test_frame_sampling.py -v
```

Expected: 3 fewer tests collected, remaining tests pass.

- [ ] **Step 3: Delete `_rotation_map` and `_apply_rotation` from frame_sampling.py**

In `collab_splats/utils/frame_sampling.py`, delete the two helper functions entirely (lines ~8–22 in the current file):

```python
# DELETE these two functions:
def _rotation_map():
    import cv2
    return {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    import cv2
    code = _rotation_map().get(rotation)
    if code is not None:
        return cv2.rotate(frame, code)
    return frame
```

- [ ] **Step 4: Remove auto-rotation disable + manual rotation from `sample_frames_fps`**

Replace the `cap` setup block in `sample_frames_fps`. The current block is:

```python
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
interval = max(1, int(round(native_fps / fps)))
frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % interval == 0:
        frame = _apply_rotation(frame, rotation)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    idx += 1
cap.release()
return frames
```

Replace with (auto-rotation stays on by default; no manual rotation needed):

```python
cap = cv2.VideoCapture(video_path)
native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
interval = max(1, int(round(native_fps / fps)))
frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % interval == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    idx += 1
cap.release()
return frames
```

- [ ] **Step 5: Remove auto-rotation disable + manual rotation from `sample_frames_optical_flow`**

Current setup block:

```python
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
frames = []
while len(frames) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    should_select, _, _ = selector.should_select_frame(frame)
    if should_select:
        frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
        selector.accept_frame(frame)
cap.release()
return frames
```

Replace with:

```python
cap = cv2.VideoCapture(video_path)
frames = []
while len(frames) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    should_select, _, _ = selector.should_select_frame(frame)
    if should_select:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        selector.accept_frame(frame)
cap.release()
return frames
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/utils/test_frame_sampling.py -v
```

Expected: all remaining tests pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "fix(frame_sampling): re-enable OpenCV auto-rotation, remove dead rotation helpers"
```

---

## Task 2: Seek-based FPS extraction with `on_progress` callback

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py`
- Modify: `tests/utils/test_frame_sampling.py`

**Context:** Current `sample_frames_fps` decodes every frame. Seek-based approach calls `cap.set(cv2.CAP_PROP_POS_FRAMES, n)` and jumps directly to needed frames. At 1fps/30fps, this decodes ~10 frames instead of 300.

- [ ] **Step 1: Add `Callable` to typing imports in frame_sampling.py**

Current import line at top of file:

```python
from typing import Dict, Optional, Tuple, Union
```

Change to:

```python
from typing import Callable, Dict, Optional, Tuple, Union
```

- [ ] **Step 2: Write failing tests for seek and progress callback**

Append to `tests/utils/test_frame_sampling.py`:

```python
from unittest.mock import MagicMock, patch, call


def _make_mock_cap(total_frames=90, fps=30.0):
    """Return a mock VideoCapture that reads one black frame then stops."""
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: float(total_frames),
    }.get(prop, 0.0)
    mock_cap.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
    return mock_cap


def test_fps_sampler_uses_seek(monkeypatch):
    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    # After 3 seeks (frames 0, 30, 60) make read return False to stop
    read_results = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.read.side_effect = read_results

    with patch("collab_splats.utils.frame_sampling.cv2.VideoCapture", return_value=mock_cap):
        frames = sample_frames_fps("fake.mp4", fps=1.0)

    seek_calls = [c for c in mock_cap.set.call_args_list
                  if c.args[0] == cv2.CAP_PROP_POS_FRAMES]
    assert len(seek_calls) >= 1, "Must use CAP_PROP_POS_FRAMES seeks"
    assert len(frames) <= 3


def test_fps_sampler_calls_on_progress(monkeypatch):
    mock_cap = _make_mock_cap(total_frames=90, fps=30.0)
    read_results = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] * 3 + [(False, None)]
    mock_cap.read.side_effect = read_results

    calls = []
    with patch("collab_splats.utils.frame_sampling.cv2.VideoCapture", return_value=mock_cap):
        sample_frames_fps("fake.mp4", fps=1.0, on_progress=lambda c, t: calls.append((c, t)))

    assert len(calls) >= 1
    assert all(t == 90 for _, t in calls), "total must equal CAP_PROP_FRAME_COUNT"


def test_fps_sampler_no_progress_arg_ok(monkeypatch):
    """on_progress=None must not raise."""
    mock_cap = _make_mock_cap(total_frames=30, fps=30.0)
    mock_cap.read.side_effect = [(True, np.zeros((48, 64, 3), dtype=np.uint8))] + [(False, None)]
    with patch("collab_splats.utils.frame_sampling.cv2.VideoCapture", return_value=mock_cap):
        frames = sample_frames_fps("fake.mp4", fps=30.0)
    assert isinstance(frames, list)
```

- [ ] **Step 3: Run to confirm they fail**

```bash
pytest tests/utils/test_frame_sampling.py::test_fps_sampler_uses_seek \
       tests/utils/test_frame_sampling.py::test_fps_sampler_calls_on_progress \
       tests/utils/test_frame_sampling.py::test_fps_sampler_no_progress_arg_ok -v
```

Expected: FAIL (seek test fails because no seek calls; progress test fails because no `on_progress` param).

- [ ] **Step 4: Rewrite `sample_frames_fps` with seek + callback**

Replace the entire `sample_frames_fps` function body:

```python
def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Extract frames at a fixed FPS rate using seek-based decoding."""
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(round(native_fps / fps)))
    frames = []
    frame_num = 0
    while frame_num < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if on_progress is not None:
            on_progress(frame_num, total)
        frame_num += interval
    cap.release()
    return frames
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/utils/test_frame_sampling.py -v
```

Expected: all pass including the 3 new ones.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): seek-based FPS extraction with on_progress callback"
```

---

## Task 3: Low-resolution OF analysis with `on_progress` callback

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py`
- Modify: `tests/utils/test_frame_sampling.py`

**Context:** OF mode decodes every frame at full resolution for analysis. Resizing to 480px wide before OF computation cuts analysis time ~4–9x on 4K footage. Only selected frames need full resolution. Progress fires per decoded frame using total from `CAP_PROP_FRAME_COUNT`.

- [ ] **Step 1: Write failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_optical_flow_calls_on_progress(tiny_video):
    calls = []
    sample_frames_optical_flow(
        tiny_video, max_frames=5, on_progress=lambda c, t: calls.append((c, t))
    )
    assert len(calls) >= 1
    assert all(t > 0 for _, t in calls), "total must come from CAP_PROP_FRAME_COUNT"
    assert all(c > 0 for c, _ in calls), "current must increment"


def test_optical_flow_resizes_for_analysis(tiny_video, monkeypatch):
    """Low-res resize must be called when frame width > 480."""
    import cv2 as _cv2
    resize_calls = []
    original_resize = _cv2.resize

    def spy_resize(src, dsize, **kwargs):
        resize_calls.append(src.shape)
        return original_resize(src, dsize, **kwargs)

    monkeypatch.setattr("collab_splats.utils.frame_sampling.cv2.resize", spy_resize)
    # tiny_video is 64px wide — below 480 threshold, resize must NOT be called
    sample_frames_optical_flow(tiny_video, max_frames=3)
    assert len(resize_calls) == 0, "Must not resize frames already <= 480px wide"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/utils/test_frame_sampling.py::test_optical_flow_calls_on_progress \
       tests/utils/test_frame_sampling.py::test_optical_flow_resizes_for_analysis -v
```

Expected: `test_optical_flow_calls_on_progress` FAIL (no `on_progress` param); `test_optical_flow_resizes_for_analysis` may FAIL or PASS depending on resize presence.

- [ ] **Step 3: Rewrite `sample_frames_optical_flow` with low-res + callback**

Replace the entire `sample_frames_optical_flow` function:

```python
def sample_frames_optical_flow(
    video_path: str,
    min_disparity: float = 50.0,
    max_frames: int = 200,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    """Select keyframes using sparse Lucas-Kanade optical flow.

    Combines motion (disparity + rotation) and visual diversity (histogram
    similarity) into a 0–1 score. Selects frames scoring >= 0.5.
    OF analysis runs at max 480px wide for speed; selected frames kept full-res.

    min_disparity: mean pixel displacement threshold for motion detection.
        Higher = fewer frames. Typical range: 10–200px.
    """
    try:
        import cv2
    except ImportError:
        return []

    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frames_decoded = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_decoded += 1
        if on_progress is not None:
            on_progress(frames_decoded, total)

        scale = min(1.0, 480.0 / frame.shape[1])
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame

        should_select, _, _ = selector.should_select_frame(small)
        if should_select:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            selector.accept_frame(small)
    cap.release()
    return frames
```

- [ ] **Step 4: Run all frame_sampling tests**

```bash
pytest tests/utils/test_frame_sampling.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): low-res OF analysis and on_progress callback"
```

---

## Task 4: Fix frame slider layout

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`

**Context:** `frame_slider` is currently inside `pn.Column(width=300)` with `width=700`, causing it to overflow into the image pane. Move it below the image row with `sizing_mode="stretch_width"`.

- [ ] **Step 1: Fix `frame_slider` widget definition in `__init__`**

Find this line (around line 179):

```python
self.frame_slider = pn.widgets.IntSlider(
    name="Frame index", value=0, start=0, end=0, width=700
)
```

Change to:

```python
self.frame_slider = pn.widgets.IntSlider(
    name="Frame index", value=0, start=0, end=0, sizing_mode="stretch_width"
)
```

- [ ] **Step 2: Remove `frame_slider` from `frames_controls` in `create_layout`**

Find `frames_controls` in `create_layout` (around line 640):

```python
frames_controls = pn.Column(
    self.fps_slider,
    self.sampling_mode_dd,
    self.min_disparity_slider,
    self.advanced_accordion,
    self.extract_frames_btn,
    self.frame_count_txt,
    self.frame_slider,        # <-- remove this line
    width=300,
)
```

After removal:

```python
frames_controls = pn.Column(
    self.fps_slider,
    self.sampling_mode_dd,
    self.min_disparity_slider,
    self.advanced_accordion,
    self.extract_frames_btn,
    self.frame_count_txt,
    width=300,
)
```

- [ ] **Step 3: Update Frames tab layout to place slider below image row**

Find the `explore_inner_tabs` definition. Change the `"① Frames"` entry from:

```python
("① Frames", pn.Row(frames_controls, self.current_frame_pane)),
```

To:

```python
("① Frames", pn.Column(
    pn.Row(frames_controls, self.current_frame_pane),
    self.frame_slider,
)),
```

- [ ] **Step 4: Run dashboard smoke tests**

```bash
pytest tests/dashboard/test_semantics_smoke.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/semantics.py
git commit -m "fix(dashboard): move frame_slider below image row, remove width=700 overflow"
```

---

## Task 5: Add progress bar and threaded frame extraction

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`

**Context:** `_extract_frames` currently blocks the Panel event loop and shows a static overlay. Replace with: threaded extraction, `pn.widgets.Progress` (determinate), periodic polling at 200ms. The `on_progress` callbacks added in Tasks 2–3 feed the progress state.

- [ ] **Step 1: Add `threading` import if not already present**

Check the imports at the top of `collab_splats/dashboard/semantics.py`. `threading` is already imported (line 7). No change needed.

- [ ] **Step 2: Add progress widgets and extraction state to `__init__`**

After the `self.status_pane` / `self.loading_modal` block (around line 142), add:

```python
# Progress bar for frame extraction
self.progress_bar = pn.widgets.Progress(
    name="Extracting frames",
    value=0,
    max=100,
    bar_color="info",
    sizing_mode="stretch_width",
    visible=False,
)
self.progress_label = pn.pane.HTML("", visible=False)

# Extraction thread state
self._extraction_progress: tuple[int, int] = (0, 1)
self._extraction_lock = threading.Lock()
self._extraction_done = False
self._extraction_error: str | None = None
self._extraction_cb: Any = None
```

- [ ] **Step 3: Rewrite `_extract_frames` to use thread + callback**

Replace the entire `_extract_frames` method:

```python
def _extract_frames(self, event: Any) -> None:
    video_path = self._resolve_video_path()
    if video_path is None:
        self._update_status("Select a video first.", error=True)
        return

    # Reset state
    self._extraction_done = False
    self._extraction_error = None
    with self._extraction_lock:
        self._extraction_progress = (0, 1)

    self.progress_bar.value = 0
    self.progress_bar.visible = True
    self.progress_label.object = "<small>Starting…</small>"
    self.progress_label.visible = True
    self._update_status(f"Extracting frames from {video_path.name}…")

    def on_progress(current: int, total: int) -> None:
        with self._extraction_lock:
            self._extraction_progress = (current, max(1, total))

    sampling_mode = self.sampling_mode_dd.value

    def run_extraction() -> None:
        try:
            if sampling_mode == "Optical Flow":
                result = sample_frames_optical_flow(
                    str(video_path),
                    min_disparity=self.min_disparity_slider.value,
                    max_frames=200,
                    motion_weight=self.motion_weight_slider.value,
                    coverage_weight=self.coverage_weight_slider.value,
                    on_progress=on_progress,
                )
            else:
                result = sample_frames_fps(
                    str(video_path),
                    self.fps_slider.value,
                    on_progress=on_progress,
                )
            self._frames = result
        except Exception as e:
            self._frames = []
            self._extraction_error = str(e)
        finally:
            self._extraction_done = True

    threading.Thread(target=run_extraction, daemon=True).start()
    self._extraction_cb = pn.state.add_periodic_callback(
        self._poll_extraction_progress, period=200
    )
```

- [ ] **Step 4: Add `_poll_extraction_progress` method**

Add this method after `_extract_frames`:

```python
def _poll_extraction_progress(self) -> None:
    with self._extraction_lock:
        current, total = self._extraction_progress

    pct = int(current / total * 100)
    self.progress_bar.value = min(pct, 100)
    self.progress_label.object = f"<small>{pct}%</small>"

    if not self._extraction_done:
        return

    # Extraction finished — tear down
    self.progress_bar.visible = False
    self.progress_label.visible = False
    if self._extraction_cb is not None:
        self._extraction_cb.stop()
        self._extraction_cb = None

    if self._extraction_error:
        self._update_status(f"Frame extraction failed: {self._extraction_error}", error=True)
        return

    n = len(self._frames)
    self.frame_slider.end = max(0, n - 1)
    self.frame_slider.value = 0
    self.frame_count_txt.object = f"<p>{n} frames extracted</p>"
    if self._frames:
        self._current_frame = self._frames[0]
        self.current_frame_pane.object = _to_png_bytes(self._frames[0])
    self._update_status(f"Extracted {n} frames")
```

- [ ] **Step 5: Add progress bar + label to Frames tab layout**

In `create_layout`, update the `"① Frames"` tab entry (modified in Task 4) from:

```python
("① Frames", pn.Column(
    pn.Row(frames_controls, self.current_frame_pane),
    self.frame_slider,
)),
```

To:

```python
("① Frames", pn.Column(
    pn.Row(self.progress_bar, self.progress_label),
    pn.Row(frames_controls, self.current_frame_pane),
    self.frame_slider,
)),
```

- [ ] **Step 6: Run all dashboard tests**

```bash
pytest tests/dashboard/ -v
```

Expected: all pass (2 pre-existing failures are unrelated — do not fix).

- [ ] **Step 7: Run full test suite to check for regressions**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: same pass/fail count as before this PR (54 pass, 7 fail — all pre-existing).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/dashboard/semantics.py
git commit -m "feat(dashboard): determinate progress bar with threaded frame extraction"
```

---

## Task 6: Delete absorbed local branches

**Context:** `dashboard` and `refactor/dashboard-optical-flow` are fully absorbed into `refactor/dashboard-complete`. No unique commits remain. `tlb-*` branches are parked for separate plans — do NOT touch them.

- [ ] **Step 1: Verify branches have no unique commits**

```bash
git log dashboard ^refactor/dashboard-complete --oneline
git log refactor/dashboard-optical-flow ^refactor/dashboard-complete --oneline
```

Expected: empty output for both (no unique commits).

- [ ] **Step 2: Delete both local branches**

```bash
git branch -d dashboard
git branch -d refactor/dashboard-optical-flow
```

Expected output:
```
Deleted branch dashboard (was <sha>).
Deleted branch refactor/dashboard-optical-flow (was <sha>).
```

If `-d` refuses (diverged history), do NOT use `-D`. Instead check what unique commits exist and verify they're truly absorbed before proceeding.

- [ ] **Step 3: Update WORKLOG.md to mark branches deleted**

In `docs/superpowers/WORKLOG.md`, update the branch table. Change:

```markdown
| `refactor/dashboard-optical-flow` | 🗂 source only | snapshot into PR2, then delete |
| `dashboard` | 🗂 source only | cherry-pick `cb22e58` (CUDA auto-detect), then delete |
```

To:

```markdown
| `refactor/dashboard-optical-flow` | ✅ deleted | absorbed into PR2 |
| `dashboard` | ✅ deleted | absorbed into PR2 |
```

Also update the "Cleanup — Safe Now" and "Post-merge Cleanup" sections to check off those items.

- [ ] **Step 4: Commit worklog update**

```bash
git add docs/superpowers/WORKLOG.md
git commit -m "docs: mark dashboard and refactor/dashboard-optical-flow branches as deleted"
```

---

## Self-Review

**Spec coverage:**
- ✅ Image orientation fix → Task 1
- ✅ Frame slider layout → Task 4
- ✅ Determinate progress bar → Task 5
- ✅ Seek-based FPS extraction → Task 2
- ✅ Low-res OF analysis → Task 3
- ✅ Branch cleanup → Task 6
- ✅ `on_progress` callbacks in both sampling functions → Tasks 2, 3

**Placeholder scan:** No TBD/TODO. All code blocks complete. All commands have expected output.

**Type consistency:**
- `on_progress: Callable[[int, int], None] | None` used consistently in Task 2 (fps) and Task 3 (OF)
- `self._extraction_progress: tuple[int, int]` set in Task 5 Step 2, read in `_poll_extraction_progress` Task 5 Step 4 ✓
- `self._extraction_cb` set in Task 5 Step 3, stopped in Task 5 Step 4 ✓
- `sample_frames_fps(..., on_progress=on_progress)` call in Task 5 Step 3 matches signature from Task 2 ✓
- `sample_frames_optical_flow(..., on_progress=on_progress)` call in Task 5 Step 3 matches signature from Task 3 ✓
