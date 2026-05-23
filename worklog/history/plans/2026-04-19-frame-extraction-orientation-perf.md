# Frame Extraction — Orientation Fix & Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix sideways frame display in the dashboard and speed up optical flow frame extraction.

**Architecture:** Three commits: (1) fix rotation map direction + disable OpenCV auto-rotation; (2) add `downsample_factor` + switch to mean magnitude + move rotation to save-only; (3) fix `max_frames` kwarg bug, delete dead code, update slider range, reorder tabs.

**Tech Stack:** Python, OpenCV (cv2), NumPy, Panel (pn), pytest

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/semantics/frame_sampling.py` | Rotation map fix, auto-rotation disable, `downsample_factor`, `np.mean`/`np.hypot` |
| `collab_splats/dashboard/semantics.py` | Delete dead `_load_frames_from_video`, swap tab order, update slider range |
| `collab_splats/wrapper/splatter.py` | Fix `n_samples=n_samples` → `max_frames=n_samples` |
| `tests/semantics/__init__.py` | Create (empty) |
| `tests/semantics/test_frame_sampling.py` | Create: rotation map tests, optical flow tests |

---

### Task 1: Fix frame orientation

**Files:**
- Modify: `collab_splats/semantics/frame_sampling.py`
- Create: `tests/semantics/__init__.py`
- Create: `tests/semantics/test_frame_sampling.py`

- [ ] **Step 1: Create failing rotation map tests**

Create `tests/semantics/__init__.py` — empty file.

Create `tests/semantics/test_frame_sampling.py`:

```python
import cv2
import numpy as np
import pytest

from collab_splats.semantics.frame_sampling import _rotation_map, sample_frames_optical_flow


def test_rotation_90_maps_to_clockwise():
    assert _rotation_map()[90] == cv2.ROTATE_90_CLOCKWISE


def test_rotation_270_maps_to_counterclockwise():
    assert _rotation_map()[270] == cv2.ROTATE_90_COUNTERCLOCKWISE


def test_rotation_180_unchanged():
    assert _rotation_map()[180] == cv2.ROTATE_180
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/semantics/test_frame_sampling.py -v
```

Expected: `FAILED test_rotation_90_maps_to_clockwise` and `FAILED test_rotation_270_maps_to_counterclockwise`.

- [ ] **Step 3: Fix rotation map and disable auto-rotation in both samplers**

In `collab_splats/semantics/frame_sampling.py`, replace `_rotation_map()`:

```python
def _rotation_map():
    import cv2

    return {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
```

In `sample_frames_fps`, add one line after `cv2.VideoCapture(video_path)`:

```python
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
```

In `sample_frames_optical_flow`, same addition:

```python
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/semantics/test_frame_sampling.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/frame_sampling.py tests/semantics/__init__.py tests/semantics/test_frame_sampling.py
git commit -m "fix: correct rotation direction and disable cv2 auto-orientation in frame samplers"
```

---

### Task 2: Optimize optical flow extraction

**Files:**
- Modify: `collab_splats/semantics/frame_sampling.py`
- Modify: `tests/semantics/test_frame_sampling.py`

**Note:** `flow_threshold` default changes from `500.0` → `0.5` because we switch from `np.sum` (total-pixel displacement) to `np.mean` (mean per-pixel displacement). Mean is scale-invariant — same threshold works at any `downsample_factor` or input resolution. The dashboard slider range updates in Task 3 to match.

- [ ] **Step 1: Add failing optical flow tests**

Append to `tests/semantics/test_frame_sampling.py` (after existing tests):

```python
@pytest.fixture
def tiny_video(tmp_path):
    path = str(tmp_path / "test.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    for i in range(90):
        frame = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_optical_flow_accepts_downsample_factor(tiny_video):
    frames = sample_frames_optical_flow(
        tiny_video, max_frames=5, downsample_factor=0.5
    )
    assert isinstance(frames, list)
    assert len(frames) >= 1


def test_optical_flow_first_frame_always_included(tiny_video):
    # flow_threshold=9999 means only the forced first frame is emitted
    frames = sample_frames_optical_flow(
        tiny_video, flow_threshold=9999.0, max_frames=10, downsample_factor=0.25
    )
    assert len(frames) >= 1


def test_optical_flow_frames_are_rgb(tiny_video):
    frames = sample_frames_optical_flow(
        tiny_video, max_frames=3, downsample_factor=0.25
    )
    assert frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/semantics/test_frame_sampling.py::test_optical_flow_accepts_downsample_factor -v
```

Expected: `FAILED` — `TypeError: sample_frames_optical_flow() got an unexpected keyword argument 'downsample_factor'`.

- [ ] **Step 3: Rewrite sample_frames_optical_flow**

Replace the entire `sample_frames_optical_flow` function in `collab_splats/semantics/frame_sampling.py`:

```python
def sample_frames_optical_flow(
    video_path: str,
    flow_threshold: float = 0.5,
    max_frames: int = 200,
    downsample_factor: float = 0.25,
) -> list[np.ndarray]:
    """
    Select frames where scene content has changed significantly.

    Accumulates mean per-pixel optical flow magnitude (Farneback) between
    consecutive frames. Emits a frame when cumulative flow exceeds
    flow_threshold, then resets the accumulator. Caps output at max_frames.

    flow_threshold: mean per-pixel displacement to accumulate before saving a
        frame. Scale-invariant — same value works at any downsample_factor.
    downsample_factor: frames are resized by this factor before computing flow.
        Lower = faster. 0.25 (default) gives ~16x fewer pixels than full-res.
    """
    try:
        import cv2
    except ImportError:
        return []

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    frames = []
    prev_gray = None
    accumulated_flow = 0.0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
            continue
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        accumulated_flow += np.mean(np.hypot(flow[..., 0], flow[..., 1]))
        if accumulated_flow >= flow_threshold:
            frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
            accumulated_flow = 0.0
        prev_gray = gray

    cap.release()
    return frames
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
pytest tests/semantics/test_frame_sampling.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/frame_sampling.py tests/semantics/test_frame_sampling.py
git commit -m "perf: add downsample_factor to optical flow sampler, switch to mean magnitude"
```

---

### Task 3: Fix splatter bug, remove dead code, reorder tabs, update slider

**Files:**
- Modify: `collab_splats/wrapper/splatter.py`
- Modify: `collab_splats/dashboard/semantics.py`

- [ ] **Step 1: Fix n_samples → max_frames in splatter.py**

In `collab_splats/wrapper/splatter.py`, find:

```python
            sampled_frames = sample_frames_optical_flow(file_path.as_posix(), n_samples=n_samples)
```

Replace with:

```python
            sampled_frames = sample_frames_optical_flow(file_path.as_posix(), max_frames=n_samples)
```

- [ ] **Step 2: Delete _load_frames_from_video from semantics.py**

In `collab_splats/dashboard/semantics.py`, delete the entire `_load_frames_from_video` function. It starts with:

```python
def _load_frames_from_video(video_path: str, fps: float) -> list[np.ndarray]:
```

and ends after `cap.release()` / `return frames`. Delete the whole block (~25 lines).

- [ ] **Step 3: Update flow_threshold slider range in semantics.py**

In `collab_splats/dashboard/semantics.py`, in `__init__`, find and replace `self.flow_threshold_slider`:

```python
        self.flow_threshold_slider = pn.widgets.FloatSlider(
            name="Flow threshold",
            start=0.05,
            end=5.0,
            value=0.5,
            step=0.05,
            width=200,
            visible=False,
        )
```

- [ ] **Step 4: Swap tab order in create_layout**

In `collab_splats/dashboard/semantics.py`, in `create_layout()`, find the outer `pn.Tabs` call (the one with `Training` and `Explore`, not the inner explore sub-tabs):

```python
                pn.Tabs(
                    ("Training", training_tab),
                    ("Explore", explore_tab),
                ),
```

Replace with:

```python
                pn.Tabs(
                    ("Explore", explore_tab),
                    ("Training", training_tab),
                ),
```

- [ ] **Step 5: Run existing dashboard smoke tests**

```bash
pytest tests/dashboard/ -v
```

Expected: all existing dashboard tests PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/splatter.py collab_splats/dashboard/semantics.py
git commit -m "fix: correct max_frames kwarg, remove dead _load_frames_from_video, Explore tab first, update flow slider range"
```
