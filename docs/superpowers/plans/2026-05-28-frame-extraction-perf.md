# Frame Extraction Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cv2 VideoCapture seeking with an auto-cascade decoder (torchcodec → ffmpeg → cv2) in `frame_sampling.py`, and throttle `on_progress` callbacks in `preprocess.py` to eliminate GIL-induced dashboard browser timeouts and FPS-seeking latency.

**Architecture:** A module-level `_get_decoder_backend()` probe selects the best available backend once at import time. `sample_frames_fps` dispatches to the selected backend; `score_all_frames` and `sample_frames_optical_flow` use a shared `_iter_decoded_frames` generator. The cv2 code paths remain as fallback — all existing tests pass unchanged. `ThrottledProgress` in `preprocess.py` caps `on_progress` at 10 Hz.

**Tech Stack:** Python 3.11, cv2 (opencv-python), torchcodec (optional), ffmpeg subprocess, Panel, numpy.

**Python interpreter:** `/opt/conda/envs/reconstruction/bin/python`  
**Test runner:** `/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q`  
**Spec:** `docs/superpowers/specs/2026-05-28-frame-extraction-perf-design.md`

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Add `import shutil`; extend `get_video_info` (width/height); add `_get_decoder_backend()`, `_iter_decoded_frames()`, `_decode_fps_ffmpeg()`, `_decode_fps_torchcodec()`; refactor `sample_frames_fps`, `score_all_frames`, `sample_frames_optical_flow` |
| `collab_splats/dashboard/panes/preprocess.py` | Add `import time`; add `ThrottledProgress` class; wrap `_progress` lambda in `_run_extraction` |
| `tests/utils/test_frame_sampling.py` | Add tests for `get_video_info` width/height, `_get_decoder_backend`, cascade decode correctness |
| `tests/dashboard/test_preprocess.py` | Add `ThrottledProgress` rate-limit test |

---

## Task 1: Extend `get_video_info` to return `width` and `height`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py:54-68`
- Test: `tests/utils/test_frame_sampling.py`

- [ ] **Step 1.1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py` (after the existing `test_get_video_info_*` tests or at the end of the file):

```python
def test_get_video_info_returns_width_height(tiny_video):
    info = get_video_info(tiny_video)
    assert info["width"] == 64
    assert info["height"] == 48
```

(`tiny_video` fixture creates a 64×48 video — already defined in the file.)

- [ ] **Step 1.2: Run to confirm it fails**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_video_info_returns_width_height -v
```

Expected: `FAILED` — `KeyError: 'width'`

- [ ] **Step 1.3: Implement — update `get_video_info`**

Find `get_video_info` in `collab_splats/utils/frame_sampling.py` (line ~54). Replace the body:

```python
def get_video_info(video_path: str) -> dict:
    """Return basic video metadata without exposing cv2 to callers.

    Keys: total_frames (int), fps (float), duration_s (float), width (int), height (int).
    Returns zeros for all fields if the file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}
```

- [ ] **Step 1.4: Run to confirm it passes**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_video_info_returns_width_height -v
```

Expected: `PASSED`

- [ ] **Step 1.5: Run full frame-sampling suite to check no regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -x -q
```

Expected: all pass.

- [ ] **Step 1.6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): extend get_video_info with width/height"
```

---

## Task 2: Add `_get_decoder_backend()` probe

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (imports + new function after `Helpers` section)
- Test: `tests/utils/test_frame_sampling.py`

- [ ] **Step 2.1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_get_decoder_backend_returns_valid_string():
    """Backend probe returns one of the three valid backends."""
    from collab_splats.utils.frame_sampling import _get_decoder_backend
    result = _get_decoder_backend()
    assert result in ("torchcodec", "ffmpeg", "cv2")


def test_get_decoder_backend_is_cached():
    """Second call returns same value (lru_cache works)."""
    from collab_splats.utils.frame_sampling import _get_decoder_backend
    assert _get_decoder_backend() == _get_decoder_backend()
```

- [ ] **Step 2.2: Run to confirm they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_decoder_backend_returns_valid_string tests/utils/test_frame_sampling.py::test_get_decoder_backend_is_cached -v
```

Expected: `FAILED` — `ImportError: cannot import name '_get_decoder_backend'`

- [ ] **Step 2.3: Add `import shutil` and `from functools import lru_cache` to `frame_sampling.py`**

At the top of `collab_splats/utils/frame_sampling.py`, the imports block currently reads:

```python
from __future__ import annotations

import json
import logging
import subprocess
import cv2
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
```

Replace with:

```python
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import cv2
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional, Tuple, Union
```

- [ ] **Step 2.4: Add `_get_decoder_backend()` after the `Helpers` section header (around line 20)**

After the `########################################################################` Helpers divider and before `_get_rotation_degrees`, insert:

```python
@lru_cache(maxsize=None)
def _get_decoder_backend() -> str:
    """Return best available video decode backend: torchcodec > ffmpeg > cv2.

    Result is cached at module load time — probe runs once per process.
    """
    try:
        import torchcodec  # noqa: F401
        return "torchcodec"
    except ImportError:
        pass
    if shutil.which("ffmpeg") is not None:
        return "ffmpeg"
    return "cv2"
```

- [ ] **Step 2.5: Run to confirm the new tests pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_get_decoder_backend_returns_valid_string tests/utils/test_frame_sampling.py::test_get_decoder_backend_is_cached -v
```

Expected: `PASSED`

- [ ] **Step 2.6: Run full suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -x -q
```

Expected: all pass.

- [ ] **Step 2.7: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add _get_decoder_backend() auto-cascade probe"
```

---

## Task 3: Refactor `sample_frames_fps` with cascade decoder

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (`sample_frames_fps`, new helpers `_decode_fps_ffmpeg`, `_decode_fps_torchcodec`)
- Test: `tests/utils/test_frame_sampling.py`

- [ ] **Step 3.1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_sample_frames_fps_ffmpeg_backend(tmp_path, monkeypatch):
    """sample_frames_fps works correctly when forced to ffmpeg backend."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    from collab_splats.utils import frame_sampling as fs

    # Force ffmpeg backend by patching the cached probe
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")

    # Build a tiny synthetic mp4 (same as tiny_video fixture)
    path = str(tmp_path / "test_ffmpeg.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(90):
        frame = np.full((48, 64, 3), (i * 5) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames, indices = fs.sample_frames_fps(path, fps=5.0, verbose=False)
    assert len(frames) > 0
    assert len(frames) == len(indices)
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_sample_frames_fps_progress_callback(tiny_video):
    """on_progress is called at least once and never with n > total."""
    calls = []
    frames, _ = sample_frames_fps(
        tiny_video, fps=5.0, on_progress=lambda n, t: calls.append((n, t)), verbose=False
    )
    assert len(calls) > 0
    assert all(n <= t for n, t in calls)
```

- [ ] **Step 3.2: Run to confirm they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_sample_frames_fps_ffmpeg_backend tests/utils/test_frame_sampling.py::test_sample_frames_fps_progress_callback -v
```

Expected: `test_sample_frames_fps_ffmpeg_backend` FAILS — backend not yet wired; `test_sample_frames_fps_progress_callback` should PASS (it tests existing behavior — run it to establish baseline).

- [ ] **Step 3.3: Add `_decode_fps_ffmpeg` helper to `frame_sampling.py`**

Insert this new function in the `Frame Selection` section, just before `sample_frames_fps`:

```python
def _decode_fps_ffmpeg(
    video_path: str,
    fps: float,
    n_targets: int,
    width: int,
    height: int,
    native_fps: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract frames at fps via ffmpeg subprocess pipe.

    ffmpeg handles rotation from container metadata automatically.
    Returns (rgb_frames, approximate_source_indices).
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-frames:v", str(n_targets),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-an", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height * 3
    frames: list[np.ndarray] = []
    interval = max(1, int(round(native_fps / fps)))
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
            if on_progress is not None:
                on_progress(len(frames), n_targets)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    indices = [i * interval for i in range(len(frames))]
    return frames, indices
```

- [ ] **Step 3.4: Add `_decode_fps_torchcodec` helper immediately after `_decode_fps_ffmpeg`**

```python
def _decode_fps_torchcodec(
    video_path: str,
    targets: list[int],
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract specific frames by index using torchcodec GPU decoder.

    torchcodec handles rotation from container metadata automatically.
    Returns (rgb_frames, targets) — indices are exact source positions.
    """
    import torch
    from torchcodec.decoders import VideoDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = VideoDecoder(video_path, device=device)
    result = decoder.get_frames_at(indices=targets)
    # result.data: (N, C, H, W) uint8 tensor, RGB
    frames = [
        result.data[i].permute(1, 2, 0).cpu().numpy()
        for i in range(result.data.shape[0])
    ]
    if on_progress is not None:
        for i, _ in enumerate(frames, 1):
            on_progress(i, len(targets))
    return frames, targets[: len(frames)]
```

- [ ] **Step 3.5: Refactor `sample_frames_fps` to dispatch through cascade**

Replace the full `sample_frames_fps` function body (lines ~367-407):

```python
def sample_frames_fps(
    video_path: str,
    fps: float,
    on_progress: Callable[[int, int], None] | None = None,
    max_frames: int | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract frames at a fixed FPS rate using the best available decoder.

    Returns (frames, indices) where indices are the source frame positions.
    on_progress: called as on_progress(n_collected, n_targets) after each frame.
    max_frames: cap the number of extracted frames; None means no cap.
    """
    info = get_video_info(video_path)
    if info["total_frames"] == 0:
        return [], []

    native_fps = info["fps"] or 30.0
    total = info["total_frames"]
    interval = max(1, int(round(native_fps / fps)))
    targets = list(range(0, total, interval))
    if max_frames is not None:
        targets = targets[:max_frames]
    if not targets:
        return [], []

    backend = _get_decoder_backend()
    logger.debug("sample_frames_fps: backend=%s, targets=%d", backend, len(targets))

    if backend == "torchcodec":
        return _decode_fps_torchcodec(video_path, targets, on_progress)

    if backend == "ffmpeg":
        return _decode_fps_ffmpeg(
            video_path, fps, len(targets),
            info["width"], info["height"], native_fps, on_progress,
        )

    # cv2 fallback: seek to each target index
    rotation = _get_rotation_degrees(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    try:
        with tqdm(targets, desc="Sampling frames", unit="frame", disable=not verbose) as pbar:
            for target in pbar:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Could not read frame %d from %s", target, video_path)
                    continue
                frames.append(cv2.cvtColor(_apply_rotation(frame, rotation), cv2.COLOR_BGR2RGB))
                indices.append(target)
                if on_progress is not None:
                    on_progress(len(frames), len(targets))
    finally:
        cap.release()
    return frames, indices
```

- [ ] **Step 3.6: Run new tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_sample_frames_fps_ffmpeg_backend tests/utils/test_frame_sampling.py::test_sample_frames_fps_progress_callback -v
```

Expected: both `PASSED`

- [ ] **Step 3.7: Run full frame-sampling suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -x -q
```

Expected: all pass.

- [ ] **Step 3.8: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): cascade decoder in sample_frames_fps (ffmpeg/torchcodec/cv2)"
```

---

## Task 4: Add `_iter_decoded_frames` + refactor `score_all_frames` and `sample_frames_optical_flow`

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py`
- Test: `tests/utils/test_frame_sampling.py`

- [ ] **Step 4.1: Write the failing test**

Append to `tests/utils/test_frame_sampling.py`:

```python
def test_iter_decoded_frames_yields_bgr_frames(tiny_video):
    """_iter_decoded_frames yields numpy HWC BGR uint8 arrays for every frame."""
    from collab_splats.utils.frame_sampling import _iter_decoded_frames
    info = get_video_info(tiny_video)
    frames = list(_iter_decoded_frames(tiny_video, info["width"], info["height"]))
    assert len(frames) == 90  # tiny_video has 90 frames
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_score_all_frames_ffmpeg_backend(tiny_video, monkeypatch):
    """score_all_frames returns same structure when forced to ffmpeg backend."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    scores = fs.score_all_frames(tiny_video, verbose=False)
    assert isinstance(scores, list)
    assert len(scores) > 0
    assert all("disparity" in s for s in scores)


def test_sample_frames_optical_flow_ffmpeg_backend(tiny_video, monkeypatch):
    """sample_frames_optical_flow returns RGB frames when forced to ffmpeg backend."""
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_decoder_backend", lambda: "ffmpeg")
    frames = fs.sample_frames_optical_flow(tiny_video, verbose=False)
    assert isinstance(frames, list)
    # All frames must be HWC uint8 (RGB)
    for f in frames:
        assert f.shape[2] == 3
        assert f.dtype == np.uint8
```

- [ ] **Step 4.2: Run to confirm they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_iter_decoded_frames_yields_bgr_frames tests/utils/test_frame_sampling.py::test_score_all_frames_ffmpeg_backend tests/utils/test_frame_sampling.py::test_sample_frames_optical_flow_ffmpeg_backend -v
```

Expected: `FAILED` — `cannot import name '_iter_decoded_frames'`

- [ ] **Step 4.3: Add `_iter_decoded_frames` generator to `frame_sampling.py`**

Insert this new function in the `Helpers` section, after `_get_decoder_backend()`:

```python
def _iter_decoded_frames(
    video_path: str,
    width: int,
    height: int,
) -> Iterator[np.ndarray]:
    """Yield BGR uint8 HWC numpy arrays for every frame, rotation already applied.

    Dispatches to torchcodec, ffmpeg, or cv2 based on _get_decoder_backend().
    """
    backend = _get_decoder_backend()

    if backend == "ffmpeg":
        cmd = [
            "ffmpeg", "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-an", "pipe:1",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        frame_size = width * height * 3
        try:
            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                yield np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
        return

    if backend == "torchcodec":
        import torch
        from torchcodec.decoders import VideoDecoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        decoder = VideoDecoder(video_path, device=device)
        for frame_batch in decoder:
            # frame_batch.data: (C, H, W) uint8 RGB tensor
            rgb = frame_batch.data.permute(1, 2, 0).cpu().numpy()
            yield rgb[:, :, ::-1].copy()  # RGB → BGR to match cv2 convention
        return

    # cv2 fallback
    rotation = _get_rotation_degrees(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield _apply_rotation(frame, rotation)
    finally:
        cap.release()
```

- [ ] **Step 4.4: Refactor `score_all_frames` to use `_iter_decoded_frames`**

Replace the `score_all_frames` function body (the `cap = cv2.VideoCapture(...)` block through the `finally: cap.release()`):

```python
def score_all_frames(
    video_path: str,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    stride: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Score decoded frames using OpticalFlowFrameSelector.

    Returns one dict per scored frame with keys:
        frame_idx (int), disparity (float), rotation (float),
        histogram_similarity (float), score (float), selected (bool)

    stride: score every Nth decoded frame; on_progress fires for every frame regardless.
    The first scored frame always has selected=True (score=1.0).
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    info = get_video_info(video_path)
    total = info["total_frames"]
    results = []
    frames_decoded = 0
    with tqdm(total=total, desc="Scoring frames", unit="frame", disable=not verbose) as pbar:
        for frame in _iter_decoded_frames(video_path, info["width"], info["height"]):
            # Score only stride-aligned frames; skip OF analysis on others
            if frames_decoded % stride == 0:
                # Downscale to 480px wide for faster OF computation
                scale = min(1.0, 480.0 / frame.shape[1])
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame
                should_select, score, components = selector.should_select_frame(small)
                if should_select:
                    selector.accept_frame(small)
                results.append({
                    "frame_idx": frames_decoded,
                    "disparity": components.get("disparity", 0.0),
                    "rotation": components.get("rotation", 0.0),
                    "histogram_similarity": components.get("histogram_similarity", 1.0),
                    "score": score,
                    "selected": should_select,
                })
            frames_decoded += 1
            pbar.update(1)
            if on_progress is not None:
                on_progress(frames_decoded, total)
    return results
```

- [ ] **Step 4.5: Refactor `sample_frames_optical_flow` to use `_iter_decoded_frames`**

Replace the body of `sample_frames_optical_flow` (the `cap = cv2.VideoCapture(...)` block through the `finally: cap.release()`):

```python
def sample_frames_optical_flow(
    video_path: str,
    min_disparity: float = 50.0,
    max_frames: int = 200,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Select keyframes using sparse Lucas-Kanade optical flow.

    Combines motion (disparity + rotation) and visual diversity (histogram
    similarity) into a 0–1 score; selects frames scoring >= 0.5.
    OF analysis runs at max 480px wide for speed; selected frames kept full-res.

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
            should_select, _, _ = selector.should_select_frame(small)
            if should_select:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                selector.accept_frame(small)
    return frames
```

- [ ] **Step 4.6: Run new tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py::test_iter_decoded_frames_yields_bgr_frames tests/utils/test_frame_sampling.py::test_score_all_frames_ffmpeg_backend tests/utils/test_frame_sampling.py::test_sample_frames_optical_flow_ffmpeg_backend -v
```

Expected: all `PASSED`

- [ ] **Step 4.7: Run full frame-sampling suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/utils/test_frame_sampling.py -x -q
```

Expected: all pass.

- [ ] **Step 4.8: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): _iter_decoded_frames generator; cascade in score_all_frames + optical_flow"
```

---

## Task 5: Add `ThrottledProgress` to `preprocess.py` + wire into `_run_extraction`

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Test: `tests/dashboard/test_preprocess.py`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/dashboard/test_preprocess.py`:

```python
import time


def test_throttled_progress_rate_limits():
    """ThrottledProgress fires at most max_hz times per second (plus final call)."""
    from collab_splats.dashboard.panes.preprocess import ThrottledProgress

    calls = []
    tp = ThrottledProgress(lambda n, t: calls.append((n, t)), max_hz=10.0)

    # Fire 100 calls in quick succession (< 10ms apart)
    for i in range(1, 101):
        tp(i, 100)

    # First and last must always fire; intermediate calls are throttled
    assert calls[0] == (1, 100)
    assert calls[-1] == (100, 100)
    # With 100ms min interval and near-instant calls, only ~1–3 fire
    assert len(calls) <= 5


def test_throttled_progress_always_fires_final():
    """ThrottledProgress always fires when n == total."""
    from collab_splats.dashboard.panes.preprocess import ThrottledProgress

    calls = []
    tp = ThrottledProgress(lambda n, t: calls.append(n), max_hz=0.01)  # very slow
    tp(50, 100)  # should be throttled after first
    tp(100, 100)  # must fire (n == total)
    assert 100 in calls
```

- [ ] **Step 5.2: Run to confirm they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py::test_throttled_progress_rate_limits tests/dashboard/test_preprocess.py::test_throttled_progress_always_fires_final -v
```

Expected: `FAILED` — `cannot import name 'ThrottledProgress'`

- [ ] **Step 5.3: Add `import time` to `preprocess.py` imports**

In `collab_splats/dashboard/panes/preprocess.py`, find the imports block (around line 7-10). Add `import time` with the other stdlib imports:

```python
import io
import logging
import threading
import time
from pathlib import Path
```

- [ ] **Step 5.4: Add `ThrottledProgress` class to `preprocess.py`**

Insert this class immediately after the `########################################################################\n# Pure helpers` section and before `_window_frame_indices`:

```python
class ThrottledProgress:
    """Wraps an on_progress callback, firing at most max_hz times per second.

    Always fires on the very first call and when n == total.
    """

    def __init__(self, callback: Callable[[int, int], None], max_hz: float = 10.0) -> None:
        self._cb = callback
        self._min_interval = 1.0 / max_hz
        self._last: float = 0.0

    def __call__(self, n: int, total: int) -> None:
        now = time.monotonic()
        if self._last == 0.0 or now - self._last >= self._min_interval or n == total:
            self._last = now
            self._cb(n, total)
```

You'll also need to add `from typing import Callable` if not already imported. Check the imports in `preprocess.py` — if `Callable` isn't there, add it to the `from typing import ...` line.

- [ ] **Step 5.5: Wire `ThrottledProgress` into `_run_extraction`**

In `_run_extraction` (around line 283), the inner `_progress` function is defined as:

```python
def _progress(current: int, total: int, base: int, scale: int) -> None:
    pct = base + int(current / total * scale) if total > 0 else base
    self._op_log.update_progress(pct)
    self._progress_bar.value = pct
```

And passed as:
```python
on_progress=lambda c, t: _progress(c, t, 0, 50),
```

Replace `_run_extraction`'s `_progress` helper and all four `on_progress=` lambda usages with throttled versions. Find and replace the entire inner function + its usages:

```python
def _run_extraction(self, video_path: Path) -> None:
    """Background thread: score frames, extract keyframes, update UI state."""
    def _progress_raw(current: int, total: int, base: int, scale: int) -> None:
        pct = base + int(current / total * scale) if total > 0 else base
        self._op_log.update_progress(pct)
        self._progress_bar.value = pct

    _score_progress = ThrottledProgress(lambda c, t: _progress_raw(c, t, 0, 50))
    _extract_progress = ThrottledProgress(lambda c, t: _progress_raw(c, t, 50, 50))
```

Then update the two `on_progress=` call sites in the same method:

```python
# Replace:
on_progress=lambda c, t: _progress(c, t, 0, 50),
# With:
on_progress=_score_progress,

# Replace:
on_progress=lambda c, t: _progress(c, t, 50, 50),
# With:
on_progress=_extract_progress,
```

(There are two `on_progress=` kwargs in `_run_extraction`: one passed to `score_all_frames`, one passed to `sample_frames_fps` or `sample_frames_optical_flow`.)

- [ ] **Step 5.6: Run new tests**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py::test_throttled_progress_rate_limits tests/dashboard/test_preprocess.py::test_throttled_progress_always_fires_final -v
```

Expected: both `PASSED`

- [ ] **Step 5.7: Run full preprocess test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/dashboard/test_preprocess.py -x -q
```

Expected: all pass.

- [ ] **Step 5.8: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q
```

Expected: all pass.

- [ ] **Step 5.9: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "feat(dashboard): ThrottledProgress — cap on_progress at 10 Hz in PreprocessPane"
```

---

## Self-Review Checklist

- [x] `get_video_info` width/height — Task 1
- [x] `_get_decoder_backend()` auto-cascade — Task 2
- [x] `sample_frames_fps` ffmpeg + torchcodec + cv2 fallback — Task 3
- [x] `score_all_frames` refactored through `_iter_decoded_frames` — Task 4
- [x] `sample_frames_optical_flow` refactored through `_iter_decoded_frames` — Task 4
- [x] `ThrottledProgress` in `preprocess.py` — Task 5
- [x] `on_progress` wired to `ThrottledProgress` in `_run_extraction` — Task 5
- [x] `pyproject.toml` dep (`torchcodec`) — already committed in brainstorm phase
- [x] No TBDs, no placeholder steps
- [x] All type names consistent across tasks (`Callable`, `Iterator`, `ThrottledProgress`)
- [x] `_iter_decoded_frames` added to `__init__.py` exports? No — it's a private helper, not exported.
- [x] `_decode_fps_ffmpeg` / `_decode_fps_torchcodec` consistent names used in Task 3 throughout.
