# Semantics Dashboard — Fixes, UX Redesign & Optical Flow Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 dashboard bugs/UX issues in PR 1 (`refactor/dashboard`), then add optical flow frame sampling as a new capability in PR 2 (new branch off `refactor/dashboard`).

**Architecture:** PR 1 patches `semantics.py` and `pyproject.toml` directly — no new files. PR 2 introduces `collab_splats/semantics/frame_sampling.py` as a clean module with two public functions, wires it into the dashboard, and integrates it into `Splatter.preprocess()` by pre-extracting frames to a temp dir and switching `ns-process-data` to images mode.

**Tech Stack:** Panel, param, cv2 (already present), PyYAML, nerfstudio's `ns-process-data` CLI.

**Spec:** `worklog/history/specs/2026-04-17-semantics-dashboard-panel.md`

---

## File Map

### PR 1 — on `refactor/dashboard`

| File | Change |
|------|--------|
| `pyproject.toml` | Add `transformers` to `[project.dependencies]` |
| `collab_splats/semantics/features.py` | Add `ImportError` guard with pip hint to each extractor `__init__` |
| `collab_splats/dashboard/semantics.py` | Fix rotation · image size · tab indicator CSS · mode badges · explore sub-tabs · `sampling_mode_dd` (disabled) |
| `tests/dashboard/test_semantics_smoke.py` | Extend with new layout assertions |

### PR 2 — new branch off `refactor/dashboard`

| File | Change |
|------|--------|
| `collab_splats/semantics/frame_sampling.py` | **New** — `sample_frames_fps`, `sample_frames_optical_flow` |
| `collab_splats/semantics/__init__.py` | Export both samplers |
| `collab_splats/dashboard/semantics.py` | Enable `sampling_mode_dd`, add `flow_threshold_slider`, dispatch to sampler |
| `collab_splats/wrapper/splatter.py` | Add `frame_selection` to `SplatterConfig`; dispatch in `preprocess()` |
| `tests/semantics/test_frame_sampling.py` | **New** — unit tests for both samplers |
| `tests/dashboard/test_semantics_smoke.py` | Assert `sampling_mode_dd` is enabled and `flow_threshold_slider` exists |

---

## PR 1 Tasks

---

### Task 1: Add `transformers` dependency and extractor import guards

**Files:**
- Modify: `pyproject.toml`
- Modify: `collab_splats/semantics/features.py`

- [ ] **Step 1: Add `transformers` to main deps**

In `pyproject.toml`, add `"transformers"` to the `dependencies` list (after `"tqdm"`):

```toml
dependencies = [
    "einops",
    "ftfy",
    "gdown",
    "matplotlib",
    "mergedeep",
    "numpy<2.0.0",
    "scipy==1.11.4",
    "lightning",
    "pillow",
    "regex",
    "tqdm",
    "transformers",
    ...
]
```

- [ ] **Step 2: Add import guards to extractor `__init__` methods**

In `collab_splats/semantics/features.py`, wrap the sensitive import in `MaskCLIPExtractor.__init__`:

```python
def __init__(self, model_name: str = "ViT-L/14@336px", cache_dir: str = TORCH_HOME, device: str = "cpu"):
    super().__init__()
    try:
        import maskclip_onnx as _mco
    except ImportError as e:
        raise ImportError(
            "maskclip_onnx is required for MaskCLIPExtractor. "
            "Install via: pip install 'git+https://github.com/RogerQi/maskclip_onnx.git'"
        ) from e
    self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
    ...
```

And in `Talk2DinoExtractor.__init__`:

```python
def __init__(self, hf_model_id: str = "lorebianchi98/Talk2DINOv3-ViTB", device: str = "cpu"):
    super().__init__()
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers is required for Talk2DinoExtractor. "
            "Install via: pip install transformers"
        ) from e
    self._model = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True).to(device).eval()
    ...
```

- [ ] **Step 3: Verify import error is raised correctly**

```bash
python -c "
import sys
sys.modules['transformers'] = None  # simulate missing
from collab_splats.semantics.features import Talk2DinoExtractor
try:
    Talk2DinoExtractor()
except ImportError as e:
    print('OK:', e)
"
```

Expected output contains: `OK: transformers is required`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml collab_splats/semantics/features.py
git commit -m "fix: add transformers dep and import guards to feature extractors"
```

---

### Task 2: Fix upside-down video frames (cv2 rotation metadata)

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_load_frames_applies_rotation(monkeypatch):
    """_load_frames_from_video must rotate frames according to CAP_PROP_ORIENTATION_META."""
    import cv2
    import numpy as np
    from collab_splats.dashboard.semantics import _load_frames_from_video

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[0, 0] = [1, 2, 3]  # mark top-left pixel

    rotate_calls = []

    class _MockCap:
        _reads = 0
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_ORIENTATION_META:
                return 180
            return 0
        def isOpened(self):
            return self._reads < 1
        def read(self):
            self._reads += 1
            return True, frame.copy()
        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: _MockCap())
    monkeypatch.setattr(cv2, "cvtColor", lambda f, _: f)  # bypass BGR→RGB

    _original_rotate = cv2.rotate
    def _spy_rotate(img, code):
        rotate_calls.append(code)
        return _original_rotate(img, code)
    monkeypatch.setattr(cv2, "rotate", _spy_rotate)

    _load_frames_from_video("fake.mp4", fps=30)

    assert cv2.ROTATE_180 in rotate_calls
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_load_frames_applies_rotation -v
```

Expected: FAIL — `AssertionError` (no rotation call).

- [ ] **Step 3: Implement rotation fix**

Replace `_load_frames_from_video` in `collab_splats/dashboard/semantics.py`:

```python
_ROTATION_MAP = {
    90:  "cv2.ROTATE_90_COUNTERCLOCKWISE",
    180: "cv2.ROTATE_180",
    270: "cv2.ROTATE_90_CLOCKWISE",
}

def _load_frames_from_video(video_path: str, fps: float) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as e:
        raise ImportError("pip install opencv-python") from e
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    rotate_code = {
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }.get(rotation)
    step = max(1, int(native_fps / fps))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames
```

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_load_frames_applies_rotation -v
```

Expected: PASS.

- [ ] **Step 5: Run full smoke suite to confirm no regressions**

```bash
pytest tests/dashboard/test_semantics_smoke.py -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "fix: apply cv2 orientation metadata rotation to video frames"
```

---

### Task 3: Constrain image pane size

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_image_panes_have_max_size(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    for pane in [
        dashboard.current_frame_pane,
        dashboard.feature_overlay_pane,
        dashboard.seg_output_pane,
    ]:
        assert pane.max_width == 640
        assert pane.max_height == 480
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_image_panes_have_max_size -v
```

Expected: FAIL — `AttributeError` or assertion error.

- [ ] **Step 3: Update all PNG pane declarations in `__init__`**

In `SemanticsDashboard.__init__`, replace every `pn.pane.PNG(None, width=700)` with:

```python
self.current_frame_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
self.feature_overlay_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
self.seg_output_pane = pn.pane.PNG(None, max_width=640, max_height=480, sizing_mode="scale_both")
```

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_image_panes_have_max_size -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "fix: constrain image panes to max 640x480"
```

---

### Task 4: Active tab indicator — CSS override + mode badges

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_mode_badges_contain_correct_labels(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    layout = dashboard.create_layout()
    # Badges are HTML panes stored as instance attributes
    assert "Training Mode" in dashboard._training_badge.object
    assert "Explore Mode" in dashboard._explore_badge.object
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_mode_badges_contain_correct_labels -v
```

Expected: FAIL — `AttributeError: 'SemanticsDashboard' object has no attribute '_training_badge'`.

- [ ] **Step 3: Add CSS constant, badge factory, and badge attributes**

At module level in `collab_splats/dashboard/semantics.py`, add:

```python
_TAB_CSS = """
.bk-tab.bk-active {
    border-bottom: 4px solid #2596be !important;
    font-weight: 700 !important;
    color: #2596be !important;
}
.bk-tab {
    color: #aaa;
    font-weight: 400;
}
"""

def _mode_badge(label: str) -> "pn.pane.HTML":
    style = (
        "display:inline-flex;align-items:center;gap:6px;"
        "background:#e8f4fd;border:1px solid #2596be;"
        "border-radius:20px;padding:4px 14px;margin-bottom:12px"
    )
    dot = "<span style='width:8px;height:8px;border-radius:50%;background:#2596be;display:inline-block'></span>"
    text_style = "font-size:12px;font-weight:700;color:#2596be;text-transform:uppercase;letter-spacing:0.5px"
    return pn.pane.HTML(f"<div style='{style}'>{dot}<span style='{text_style}'>{label}</span></div>")
```

In `SemanticsDashboard.__init__`, add:

```python
self._training_badge = _mode_badge("Training Mode")
self._explore_badge = _mode_badge("Explore Mode")
```

In `build_app()`, add the CSS:

```python
def build_app(base_dir: str = "/workspace/fieldwork-data") -> pn.template.MaterialTemplate:
    pn.extension(raw_css=[_TAB_CSS])
    dashboard = SemanticsDashboard(base_dir=base_dir)
    return dashboard.create_layout()
```

- [ ] **Step 4: Run test to confirm pass**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_mode_badges_contain_correct_labels -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "feat: add active tab CSS override and Training/Explore mode badges"
```

---

### Task 5: Restructure Explore tab into sub-tabs with two-column layout

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_explore_inner_tabs_exist(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    layout = dashboard.create_layout()
    # sampling_mode_dd must exist and be disabled (optical flow not yet wired)
    assert hasattr(dashboard, "sampling_mode_dd")
    assert dashboard.sampling_mode_dd.disabled is True
    assert dashboard.sampling_mode_dd.options == ["FPS", "Optical Flow"]
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_explore_inner_tabs_exist -v
```

Expected: FAIL — `AttributeError: 'SemanticsDashboard' object has no attribute 'sampling_mode_dd'`.

- [ ] **Step 3: Add `sampling_mode_dd` widget in `__init__`**

In `SemanticsDashboard.__init__`, after the `fps_slider` declaration, add:

```python
self.sampling_mode_dd = pn.widgets.Select(
    name="Sampling mode", options=["FPS", "Optical Flow"], value="FPS", width=200, disabled=True
)
```

- [ ] **Step 4: Rewrite `create_layout` — explore tab section**

Replace the `explore_tab = pn.Column(...)` block in `create_layout` with:

```python
frames_controls = pn.Column(
    self.fps_slider,
    self.sampling_mode_dd,
    self.extract_frames_btn,
    self.frame_count_txt,
    self.frame_slider,
    width=300,
)

features_controls = pn.Column(
    pn.Row(self.extractor_dd, self.device_dd),
    self.extract_features_btn,
    width=300,
)

seg_controls = pn.Column(
    pn.Row(self.seg_strategy_dd, self.seg_device_dd),
    self.seg_btn,
    self.seg_count_txt,
    width=300,
)

query_controls = pn.Column(
    pn.Row(self.hf_model_dd, self.query_device_dd),
    self.text_pairs_input,
    pn.Row(self.method_dd, self.temp_slider),
    self.query_btn,
    width=300,
)

explore_tab = pn.Column(
    self._explore_badge,
    pn.Tabs(
        ("① Frames",       pn.Row(frames_controls, self.current_frame_pane)),
        ("② Features",     pn.Row(features_controls, self.feature_overlay_pane)),
        ("③ Segmentation", pn.Row(seg_controls, self.seg_output_pane)),
        ("④ Query",        pn.Row(query_controls, self.query_gallery)),
        dynamic=True,
    ),
)
```

Also wrap `training_tab` with its badge:

```python
training_tab = pn.Column(
    self._training_badge,
    self.output_path_input,
    pn.Row(self.launch_btn, self.stop_btn),
    self.training_log,
)
```

- [ ] **Step 5: Run test to confirm pass**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_explore_inner_tabs_exist -v
```

Expected: PASS.

- [ ] **Step 6: Run full smoke suite**

```bash
pytest tests/dashboard/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "feat: restructure Explore tab into numbered sub-tabs with two-column layout"
```

---

## PR 2 Tasks

**Start:** create new branch off `refactor/dashboard`:

```bash
git checkout -b feat/optical-flow-sampling
```

---

### Task 6: `frame_sampling.py` — FPS and optical flow samplers

**Files:**
- Create: `collab_splats/semantics/frame_sampling.py`
- Create: `tests/semantics/test_frame_sampling.py`
- Modify: `collab_splats/semantics/__init__.py`

- [ ] **Step 1: Create the test file**

Create `tests/semantics/test_frame_sampling.py`:

```python
"""Unit tests for frame_sampling module."""
import numpy as np
import pytest


def _make_mock_video(tmp_path, n_frames: int = 10, orientation: int = 0):
    """
    Write a minimal valid MP4 using cv2 and return its path.
    Falls back to a skip if cv2 is unavailable.
    """
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "test.mp4")
    h, w = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 20, dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


class TestSampleFramesFps:
    def test_returns_list_of_ndarrays(self, tmp_path):
        from collab_splats.semantics.frame_sampling import sample_frames_fps
        path = _make_mock_video(tmp_path)
        frames = sample_frames_fps(path, fps=5)
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert isinstance(frames[0], np.ndarray)

    def test_fewer_frames_at_lower_fps(self, tmp_path):
        from collab_splats.semantics.frame_sampling import sample_frames_fps
        path = _make_mock_video(tmp_path, n_frames=30)
        frames_5 = sample_frames_fps(path, fps=5)
        frames_15 = sample_frames_fps(path, fps=15)
        assert len(frames_5) <= len(frames_15)

    def test_output_is_rgb_not_bgr(self, tmp_path):
        """cv2 reads BGR; sample_frames_fps must convert to RGB."""
        cv2 = pytest.importorskip("cv2")
        from collab_splats.semantics.frame_sampling import sample_frames_fps
        # Write a red frame in BGR (0, 0, 255)
        path = str(tmp_path / "red.mp4")
        h, w = 32, 32
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        frame_bgr[:, :, 2] = 255  # red channel in BGR
        for _ in range(5):
            out.write(frame_bgr)
        out.release()
        frames = sample_frames_fps(path, fps=30)
        assert frames[0][0, 0, 0] == 255  # R channel in RGB


class TestSampleFramesOpticalFlow:
    def test_returns_list_of_ndarrays(self, tmp_path):
        from collab_splats.semantics.frame_sampling import sample_frames_optical_flow
        path = _make_mock_video(tmp_path, n_frames=20)
        frames = sample_frames_optical_flow(path, flow_threshold=1.0, max_frames=50)
        assert isinstance(frames, list)
        assert isinstance(frames[0], np.ndarray)

    def test_static_video_produces_few_frames(self, tmp_path):
        """A video with no motion should produce very few keyframes."""
        cv2 = pytest.importorskip("cv2")
        from collab_splats.semantics.frame_sampling import sample_frames_optical_flow
        path = str(tmp_path / "static.mp4")
        h, w = 64, 64
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
        for _ in range(30):
            out.write(np.full((h, w, 3), 128, dtype=np.uint8))
        out.release()
        frames = sample_frames_optical_flow(path, flow_threshold=100.0, max_frames=50)
        # Static video: flow never exceeds threshold → only the first frame
        assert len(frames) <= 2

    def test_respects_max_frames(self, tmp_path):
        from collab_splats.semantics.frame_sampling import sample_frames_optical_flow
        path = _make_mock_video(tmp_path, n_frames=30)
        frames = sample_frames_optical_flow(path, flow_threshold=0.1, max_frames=3)
        assert len(frames) <= 3
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/semantics/test_frame_sampling.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.semantics.frame_sampling'`.

- [ ] **Step 3: Create `frame_sampling.py`**

Create `collab_splats/semantics/frame_sampling.py`:

```python
"""
Frame sampling utilities for video preprocessing.

Two strategies:
  - sample_frames_fps: uniform temporal sampling at a fixed FPS rate
  - sample_frames_optical_flow: keyframe selection by cumulative optical flow magnitude
"""

from __future__ import annotations

import numpy as np


def _open_capture(video_path: str):
    try:
        import cv2
    except ImportError as e:
        raise ImportError("pip install opencv-python") from e
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    return cap


def _rotation_code(cap) -> int | None:
    import cv2
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    return {
        90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }.get(rotation)


def sample_frames_fps(video_path: str, fps: float) -> list[np.ndarray]:
    """
    Extract frames at a uniform FPS rate.

    Args:
        video_path: Path to the video file.
        fps: Target frames per second to extract.

    Returns:
        List of RGB uint8 ndarrays, one per selected frame.
    """
    import cv2

    cap = _open_capture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotate_code = _rotation_code(cap)
    step = max(1, int(native_fps / fps))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def sample_frames_optical_flow(
    video_path: str,
    flow_threshold: float = 500.0,
    max_frames: int = 200,
) -> list[np.ndarray]:
    """
    Select keyframes where cumulative optical flow magnitude exceeds flow_threshold.

    Emits a frame each time the accumulated scene motion since the last keyframe
    exceeds flow_threshold, then resets the accumulator. Always emits the first
    frame. Caps output at max_frames.

    Args:
        video_path: Path to the video file.
        flow_threshold: Accumulated flow magnitude (pixels) to trigger a new keyframe.
        max_frames: Maximum number of frames to return.

    Returns:
        List of RGB uint8 ndarrays.
    """
    import cv2

    cap = _open_capture(video_path)
    rotate_code = _rotation_code(cap)

    frames: list[np.ndarray] = []
    prev_gray = None
    accumulated_flow = 0.0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            frames.append(rgb)
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude = np.sum(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
        accumulated_flow += magnitude

        if accumulated_flow >= flow_threshold:
            frames.append(rgb)
            accumulated_flow = 0.0

        prev_gray = gray

    cap.release()
    return frames
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
pytest tests/semantics/test_frame_sampling.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Export from `collab_splats/semantics/__init__.py`**

Add to the existing imports in `collab_splats/semantics/__init__.py`:

```python
from .frame_sampling import sample_frames_fps, sample_frames_optical_flow
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/frame_sampling.py collab_splats/semantics/__init__.py tests/semantics/test_frame_sampling.py
git commit -m "feat: add frame_sampling module with FPS and optical flow samplers"
```

---

### Task 7: Wire optical flow sampler into the dashboard

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_sampling_mode_dd_enabled_and_flow_slider_exists(tmp_path):
    dashboard = SemanticsDashboard(base_dir=str(tmp_path))
    assert dashboard.sampling_mode_dd.disabled is False
    assert hasattr(dashboard, "flow_threshold_slider")
    assert dashboard.flow_threshold_slider.start == 100.0
    assert dashboard.flow_threshold_slider.end == 2000.0
```

- [ ] **Step 2: Run test to confirm failure**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_sampling_mode_dd_enabled_and_flow_slider_exists -v
```

Expected: FAIL — `AssertionError` (disabled is True, no `flow_threshold_slider`).

- [ ] **Step 3: Enable `sampling_mode_dd` and add `flow_threshold_slider`**

In `SemanticsDashboard.__init__`, change:

```python
self.sampling_mode_dd = pn.widgets.Select(
    name="Sampling mode", options=["FPS", "Optical Flow"], value="FPS", width=200, disabled=False
)
self.flow_threshold_slider = pn.widgets.FloatSlider(
    name="Flow threshold", value=500.0, start=100.0, end=2000.0, step=50.0, width=200,
    visible=False,
)
```

Wire the visibility toggle — add to `__init__` after widget declarations:

```python
self.sampling_mode_dd.param.watch(self._on_sampling_mode_change, "value")
```

Add the callback method to the class:

```python
def _on_sampling_mode_change(self, event: Any) -> None:
    self.flow_threshold_slider.visible = (event.new == "Optical Flow")
```

Add `self.flow_threshold_slider` to the `frames_controls` column in `create_layout` (after `self.sampling_mode_dd`):

```python
frames_controls = pn.Column(
    self.fps_slider,
    self.sampling_mode_dd,
    self.flow_threshold_slider,
    self.extract_frames_btn,
    self.frame_count_txt,
    self.frame_slider,
    width=300,
)
```

- [ ] **Step 4: Replace `_extract_frames` dispatch**

Replace `_extract_frames` in `collab_splats/dashboard/semantics.py`:

```python
def _extract_frames(self, event: Any) -> None:
    video_path = self._resolve_video_path()
    if video_path is None:
        self._update_status("Select a video first.", error=True)
        return
    try:
        from collab_splats.semantics.frame_sampling import (
            sample_frames_fps,
            sample_frames_optical_flow,
        )
        self._show_loading(f"Extracting frames from {video_path.name}...")
        if self.sampling_mode_dd.value == "Optical Flow":
            self._frames = sample_frames_optical_flow(
                str(video_path),
                flow_threshold=self.flow_threshold_slider.value,
                max_frames=200,
            )
        else:
            self._frames = sample_frames_fps(str(video_path), self.fps_slider.value)
        n = len(self._frames)
        self.frame_slider.end = max(0, n - 1)
        self.frame_slider.value = 0
        self.frame_count_txt.object = f"<p>{n} frames extracted</p>"
        if self._frames:
            self._current_frame = self._frames[0]
            self.current_frame_pane.object = _to_png_bytes(self._frames[0])
        self._update_status(f"Extracted {n} frames")
    except Exception as e:
        self._update_status(f"Frame extraction failed: {e}", error=True)
    finally:
        self._hide_loading()
```

- [ ] **Step 5: Remove the now-unused module-level `_load_frames_from_video`**

Delete the `_load_frames_from_video` function from `collab_splats/dashboard/semantics.py` (it has been replaced by `sample_frames_fps` in the `frame_sampling` module).

- [ ] **Step 6: Run tests**

```bash
pytest tests/dashboard/ tests/semantics/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "feat: wire optical flow frame sampler into dashboard Explore tab"
```

---

### Task 8: Add `frame_selection` to `SplatterConfig` and `Splatter.preprocess()`

**Files:**
- Modify: `collab_splats/wrapper/splatter.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_grouping.py` or create `tests/test_splatter_config.py`:

```python
"""Tests for SplatterConfig frame_selection field."""
import pytest


def test_splatter_config_accepts_frame_selection():
    from collab_splats.wrapper.splatter import SplatterConfig
    cfg: SplatterConfig = {
        "file_path": "/tmp/fake.mp4",
        "method": "rade-gs",
        "output_path": None,
        "frame_proportion": None,
        "min_frames": None,
        "websocket_port": None,
        "pointcloud_method": None,
        "frame_selection": "optical_flow",
    }
    # TypedDict is just a dict — no runtime enforcement; check key is accepted
    assert cfg["frame_selection"] == "optical_flow"


def test_splatter_config_defaults_fps(tmp_path):
    from collab_splats.wrapper.splatter import Splatter
    cfg = {
        "file_path": tmp_path / "fake.mp4",
        "method": "rade-gs",
        "output_path": tmp_path / "out",
        "frame_proportion": None,
        "min_frames": None,
        "websocket_port": None,
        "pointcloud_method": None,
    }
    splatter = Splatter(cfg)
    assert splatter.config.get("frame_selection", "fps") == "fps"
```

- [ ] **Step 2: Run test to confirm first test passes (TypedDict is structural)**

```bash
pytest tests/test_splatter_config.py::test_splatter_config_accepts_frame_selection -v
```

Expected: PASS (TypedDict doesn't enforce at runtime).

- [ ] **Step 3: Add `frame_selection` to `SplatterConfig`**

In `collab_splats/wrapper/splatter.py`, add to `SplatterConfig`:

```python
from typing import Literal

class SplatterConfig(TypedDict, total=False):
    file_path: Union[str, Path]
    method: str
    output_path: Optional[Union[str, Path]]
    frame_proportion: Optional[float]
    min_frames: Optional[int]
    websocket_port: Optional[int]
    pointcloud_method: Optional[str]
    frame_selection: Literal["fps", "optical_flow"]  # default "fps"
```

The existing `SplatterConfig` already uses `Optional` for all non-required fields. Add the new field the same way:

```python
frame_selection: Optional[Literal["fps", "optical_flow"]]
```

- [ ] **Step 4: Update `Splatter.preprocess()` to dispatch for video inputs**

In `collab_splats/wrapper/splatter.py`, replace the block starting at line 256 (video branch) with:

```python
if ext in [".mp4", ".mov", ".avi"]:
    frame_selection = self.config.get("frame_selection", "fps")
    if frame_selection == "optical_flow":
        # Pre-extract keyframes → save as JPEGs → hand off as images
        from collab_splats.semantics.frame_sampling import sample_frames_optical_flow
        import tempfile
        from PIL import Image as _PIL

        flow_frames_dir = Path(tempfile.mkdtemp(prefix="collab_splats_flow_"))
        frames = sample_frames_optical_flow(str(file_path), flow_threshold=500.0, max_frames=200)
        for i, frame_rgb in enumerate(frames):
            _PIL.fromarray(frame_rgb).save(flow_frames_dir / f"frame_{i:05d}.jpg")
        input_type = "images"
        file_path = flow_frames_dir  # ns-process-data will read from this dir
    else:
        input_type = "video"
```

And keep the rest of `preprocess()` unchanged (the `frame_proportion` block and `cmd` construction).

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v --ignore=tests/dashboard/test_numpy_fix.py -x
```

Expected: all tests pass. (Skip `test_numpy_fix.py` only if it requires torch; otherwise run it too.)

- [ ] **Step 6: Commit**

```bash
git add collab_splats/wrapper/splatter.py tests/test_splatter_config.py
git commit -m "feat: add frame_selection to SplatterConfig; optical flow pre-extracts frames for ns-process-data"
```

---

## Self-Review Notes

- `_load_frames_from_video` is removed in Task 7; the rotation fix applied in Task 2 moves into `frame_sampling._rotation_code()` (shared by both samplers).
- `sampling_mode_dd` is declared with `disabled=True` in Task 5 (PR 1), then changed to `disabled=False` in Task 7 (PR 2). These are on separate branches — no conflict.
- `SplatterConfig` currently uses `total=True` (all keys required). Adding `Optional[Literal["fps", "optical_flow"]]` keeps backwards compatibility since callers not passing `frame_selection` get `None`, which `config.get("frame_selection", "fps")` correctly treats as `"fps"`.
- The `flow_frames_dir` temp dir in Task 8 is not cleaned up after `ns-process-data` runs — add `import shutil; shutil.rmtree(flow_frames_dir, ignore_errors=True)` after the `cmd` block completes if cleanup is desired. This is left as a follow-on since `ns-process-data` may be async.
