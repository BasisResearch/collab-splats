# Frame Extraction Performance & Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix garbled frame thumbnails and slow extraction in the dashboard by correcting ffmpeg rotation-aware reshaping and replacing parallel-seek with single-pass decode, plus stream-writing zarr to avoid peak-RAM spikes.

**Architecture:** Three focused changes in two files. `_ffmpeg_output_dims` is a new pure helper in `frame_sampling.py` that computes correct output dims after ffmpeg auto-rotation. `_decode_fps_ffmpeg` is rewritten in-place (same signature) to use a single ffmpeg pass with a `select` filter. `_write_frames_zarr` in `preprocess.py` is rewritten to stream frames one-at-a-time and cap width at 1920px.

**Tech Stack:** Python 3.11, ffmpeg (subprocess), numpy, zarr 3.x, PIL (Pillow)

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Add `_ffmpeg_output_dims`; rewrite `_decode_fps_ffmpeg` as single-pass |
| `collab_splats/dashboard/panes/preprocess.py` | Add `_resize_to_max_width`; rewrite `_write_frames_zarr` as streaming |
| `tests/utils/test_frame_sampling.py` | Add tests for `_ffmpeg_output_dims` and new `_decode_fps_ffmpeg` |
| `tests/dashboard/test_preprocess.py` | Add tests for `_resize_to_max_width` and streaming `_write_frames_zarr` |

---

## Task 1 — `_ffmpeg_output_dims` helper + tests

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (after `_get_rotation_degrees`)
- Modify: `tests/utils/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
# ── _ffmpeg_output_dims ───────────────────────────────────────────────────────


from collab_splats.utils.frame_sampling import _ffmpeg_output_dims


@pytest.mark.parametrize("rotation,expected_wh", [
    (0,   (100, 50)),   # no rotation: native dims unchanged
    (180, (100, 50)),   # 180°: pixel count same, dims unchanged
    (90,  (50, 100)),   # 90° CW: ffmpeg swaps w/h in output
    (270, (50, 100)),   # 270° CW: same swap
])
def test_ffmpeg_output_dims(rotation, expected_wh):
    """Native w=100, h=50. 90/270 swaps to (50, 100) after ffmpeg auto-rotation."""
    assert _ffmpeg_output_dims(100, 50, rotation) == expected_wh
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/utils/test_frame_sampling.py::test_ffmpeg_output_dims -v
```

Expected: `ImportError: cannot import name '_ffmpeg_output_dims'`

- [ ] **Step 3: Implement `_ffmpeg_output_dims` in `frame_sampling.py`**

Add after `_apply_rotation` (around line 127):

```python
def _ffmpeg_output_dims(width: int, height: int, rotation: int) -> tuple[int, int]:
    """Return (out_w, out_h) after ffmpeg auto-rotation.

    ffmpeg rotates 90/270-degree videos by default; native dims are swapped.
    """
    if rotation in (90, 270):
        return height, width
    return width, height
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/utils/test_frame_sampling.py::test_ffmpeg_output_dims -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "feat(frame_sampling): add _ffmpeg_output_dims for rotation-aware reshape"
```

---

## Task 2 — Rewrite `_decode_fps_ffmpeg` as single-pass

**Files:**
- Modify: `collab_splats/utils/frame_sampling.py` (replace `_decode_fps_ffmpeg`)
- Modify: `tests/utils/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/utils/test_frame_sampling.py`:

```python
# ── _decode_fps_ffmpeg (single-pass) ─────────────────────────────────────────


import io as _io
from unittest.mock import patch as _patch, MagicMock as _MagicMock

from collab_splats.utils.frame_sampling import _decode_fps_ffmpeg


def _make_fake_proc(n_frames: int, out_w: int, out_h: int) -> _MagicMock:
    """Return a mock Popen whose stdout yields n_frames raw RGB frames."""
    raw = np.zeros((n_frames * out_h * out_w * 3,), dtype=np.uint8).tobytes()
    bio = _io.BytesIO(raw)
    proc = _MagicMock()
    proc.stdout = bio
    proc.stdout.close = lambda: None
    proc.terminate = lambda: None
    proc.wait = lambda: None
    return proc


def test_decode_fps_ffmpeg_returns_correct_count(monkeypatch):
    """Single-pass: returns exactly len(targets) frames when pipe has enough data."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 0)
    targets = [0, 10, 20]
    proc = _make_fake_proc(3, 64, 48)
    with _patch("subprocess.Popen", return_value=proc):
        frames, indices = _decode_fps_ffmpeg("fake.mp4", targets, 64, 48, 30.0, None)
    assert len(frames) == 3
    assert indices == targets


def test_decode_fps_ffmpeg_frame_shape_no_rotation(monkeypatch):
    """No rotation: output shape matches native (out_h=48, out_w=64)."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 0)
    proc = _make_fake_proc(2, 64, 48)
    with _patch("subprocess.Popen", return_value=proc):
        frames, _ = _decode_fps_ffmpeg("fake.mp4", [0, 10], 64, 48, 30.0, None)
    assert frames[0].shape == (48, 64, 3)
    assert frames[0].dtype == np.uint8


def test_decode_fps_ffmpeg_frame_shape_rotation_90(monkeypatch):
    """90° rotation: ffmpeg swaps dims → output shape (64, 48, 3) not (48, 64, 3)."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 90)
    # native w=64, h=48 → after 90° rotation: out_w=48, out_h=64
    proc = _make_fake_proc(2, out_w=48, out_h=64)
    with _patch("subprocess.Popen", return_value=proc):
        frames, _ = _decode_fps_ffmpeg("fake.mp4", [0, 10], 64, 48, 30.0, None)
    assert frames[0].shape == (64, 48, 3)  # (out_h, out_w, 3)


def test_decode_fps_ffmpeg_progress_fires(monkeypatch):
    """on_progress called once per frame with (n_done, n_targets)."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 0)
    targets = [0, 10, 20]
    proc = _make_fake_proc(3, 64, 48)
    calls = []
    with _patch("subprocess.Popen", return_value=proc):
        _decode_fps_ffmpeg("fake.mp4", targets, 64, 48, 30.0,
                           on_progress=lambda n, t: calls.append((n, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_decode_fps_ffmpeg_empty_targets(monkeypatch):
    """Empty targets list returns ([], []) without calling Popen."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 0)
    with _patch("subprocess.Popen") as mock_popen:
        frames, indices = _decode_fps_ffmpeg("fake.mp4", [], 64, 48, 30.0, None)
    mock_popen.assert_not_called()
    assert frames == []
    assert indices == []


def test_decode_fps_ffmpeg_uses_select_filter(monkeypatch):
    """ffmpeg command must use select filter with correct interval."""
    from collab_splats.utils import frame_sampling as fs
    monkeypatch.setattr(fs, "_get_rotation_degrees", lambda _: 0)
    targets = [0, 5, 10, 15]  # interval=5
    proc = _make_fake_proc(4, 64, 48)
    captured_cmd = []
    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return proc
    with _patch("subprocess.Popen", side_effect=fake_popen):
        _decode_fps_ffmpeg("fake.mp4", targets, 64, 48, 30.0, None)
    cmd_str = " ".join(captured_cmd)
    assert "select=not(mod(n,5))" in cmd_str
    assert "-frames:v" in cmd_str
    assert "4" in cmd_str
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/utils/test_frame_sampling.py -k "decode_fps_ffmpeg" -v
```

Expected: several FAIL (function exists but old parallel-seek implementation doesn't match single-pass contract)

- [ ] **Step 3: Rewrite `_decode_fps_ffmpeg` in `frame_sampling.py`**

Replace the entire `_decode_fps_ffmpeg` function (currently lines 433–479) with:

```python
def _decode_fps_ffmpeg(
    video_path: str,
    targets: list[int],
    width: int,
    height: int,
    native_fps: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract evenly-spaced frames via a single ffmpeg pass with select filter.

    ffmpeg applies rotation from container metadata automatically.
    _ffmpeg_output_dims adjusts reshape dims to match rotated output.
    Returns (rgb_frames, source_indices).
    """
    if not targets:
        return [], []

    rotation = _get_rotation_degrees(video_path)
    out_w, out_h = _ffmpeg_output_dims(width, height, rotation)
    frame_size = out_w * out_h * 3
    n_targets = len(targets)

    # Infer stride from first gap; targets from sample_frames_fps are always evenly spaced
    interval = targets[1] - targets[0] if len(targets) > 1 else 1

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select=not(mod(n,{interval}))",
        "-frames:v", str(n_targets),
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-an", "pipe:1",
    ]
    frames: list[np.ndarray] = []
    indices: list[int] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while len(frames) < n_targets:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(out_h, out_w, 3).copy()
            frames.append(frame)
            indices.append(targets[len(frames) - 1])
            if on_progress is not None:
                on_progress(len(frames), n_targets)
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    return frames, indices
```

Also remove the now-unused `_decode_fps_torchcodec` references to `concurrent.futures` — the old `_decode_fps_ffmpeg` was the only caller of `concurrent.futures`. Remove the `import concurrent.futures` inside the old function body (it was a local import). No module-level change needed since it was always inline.

- [ ] **Step 4: Run new tests to verify PASS**

```
python -m pytest tests/utils/test_frame_sampling.py -k "decode_fps_ffmpeg" -v
```

Expected: all 6 PASSED

- [ ] **Step 5: Run the full frame_sampling test suite to catch regressions**

```
python -m pytest tests/utils/test_frame_sampling.py -v
```

Expected: all existing tests PASS. `test_fps_sampler_uses_seek` monkeypatches `shutil.which` to `None`, forcing cv2 path — still valid, cv2 path unchanged. `test_sample_frames_fps_ffmpeg_backend` runs on a real tiny video via real ffmpeg — still valid.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/utils/frame_sampling.py tests/utils/test_frame_sampling.py
git commit -m "fix(frame_sampling): single-pass ffmpeg + rotation-aware reshape in _decode_fps_ffmpeg

Replaces 200-subprocess parallel-seek approach with one ffmpeg
invocation using select filter. Fixes garbled frames on rotated
(90/270 deg) video by using _ffmpeg_output_dims for reshape dims."
```

---

## Task 3 — Streaming zarr write with 1920px cap

**Files:**
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `tests/dashboard/test_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_preprocess.py`:

```python
from collab_splats.dashboard.panes.preprocess import _resize_to_max_width


def test_resize_to_max_width_no_op_when_under():
    """Frames at or below max_width are returned unchanged (same array)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = _resize_to_max_width(frame, 1920)
    assert result.shape == (480, 640, 3)


def test_resize_to_max_width_caps_wide_frame():
    """Frames wider than max_width are resized; height scaled proportionally."""
    frame = np.zeros((1080, 2000, 3), dtype=np.uint8)
    result = _resize_to_max_width(frame, 1920)
    assert result.shape[1] == 1920
    expected_h = int(round(1080 * 1920 / 2000))
    assert result.shape[0] == expected_h
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


def test_write_frames_zarr_wide_frames_get_resized(tmp_path):
    """Frames wider than 1920px are stored at 1920px; narrow frames unchanged."""
    wide = np.zeros((1080, 2000, 3), dtype=np.uint8)
    zarr_path = tmp_path / "wide.zarr"
    _write_frames_zarr([wide, wide], zarr_path)
    z = zarr.open(str(zarr_path), mode="r")
    assert z["frames"].shape == (2, int(round(1080 * 1920 / 2000)), 1920, 3)


def test_write_frames_zarr_streams_one_at_a_time(tmp_path, monkeypatch):
    """zarr array is written frame-by-frame, not via np.stack."""
    import numpy as _np
    stack_calls = []
    original_stack = _np.stack

    def spy_stack(*args, **kwargs):
        stack_calls.append(args)
        return original_stack(*args, **kwargs)

    monkeypatch.setattr(_np, "stack", spy_stack)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
    zarr_path = tmp_path / "stream.zarr"
    _write_frames_zarr(frames, zarr_path)
    assert len(stack_calls) == 0, "np.stack must not be called — streaming write only"
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/dashboard/test_preprocess.py -k "resize_to_max_width or wide_frames or streams_one" -v
```

Expected: `ImportError: cannot import name '_resize_to_max_width'` and some FAIL

- [ ] **Step 3: Add `_resize_to_max_width` to `preprocess.py`**

Add after `_write_frames_zarr` (around line 165), before `_build_frame_strip_html`:

```python
def _resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize frame to max_width if wider; aspect ratio preserved."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = int(round(h * max_width / w))
    return np.array(Image.fromarray(frame).resize((max_width, new_h), Image.LANCZOS))
```

- [ ] **Step 4: Rewrite `_write_frames_zarr` in `preprocess.py`**

Replace the existing `_write_frames_zarr` function (currently lines 152–165):

```python
def _write_frames_zarr(frames: list[np.ndarray], path: Path, max_width: int = 1920) -> None:
    """Write frame list to zarr, streaming one frame at a time.

    Resizes frames wider than max_width before writing; caps peak memory to one frame.
    """
    if not frames:
        raise ValueError("frames list is empty — nothing to write")
    first = _resize_to_max_width(frames[0], max_width)
    H, W = first.shape[:2]
    N = len(frames)
    store = zarr.open_group(str(path), mode="w")
    store.create_array(
        "frames",
        shape=(N, H, W, 3),
        dtype=np.uint8,
        chunks=(1, H, W, 3),
        compressors=[BloscCodec(cname="lz4", clevel=3)],
    )
    store["frames"][0] = first
    for i in range(1, N):
        store["frames"][i] = _resize_to_max_width(frames[i], max_width)
```

- [ ] **Step 5: Update `test_preprocess.py` import line**

The existing import at line 21 needs `_resize_to_max_width` added:

```python
from collab_splats.dashboard.panes.preprocess import (
    PreprocessPane,
    ThrottledProgress,
    _build_metrics_sources,
    _frames_to_thumbnails,
    _render_fps_raster,
    _resize_to_max_width,
    _window_frame_indices,
    _write_frames_zarr,
)
```

- [ ] **Step 6: Run new tests to verify PASS**

```
python -m pytest tests/dashboard/test_preprocess.py -k "resize_to_max_width or wide_frames or streams_one" -v
```

Expected: 4 PASSED

- [ ] **Step 7: Run full preprocess test suite to catch regressions**

```
python -m pytest tests/dashboard/test_preprocess.py -v
```

Expected: all existing tests PASS. `test_write_frames_zarr_shape` uses 64×48 frames (width < 1920) → no resize → shape still `(5, 48, 64, 3)` ✓. `test_write_frames_zarr_roundtrip` uses 64×48 → no resize → pixel data preserved ✓.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/dashboard/panes/preprocess.py tests/dashboard/test_preprocess.py
git commit -m "fix(dashboard): stream zarr write + 1920px cap in _write_frames_zarr

Replaces np.stack-all-then-write with pre-allocated array + per-frame
writes. Caps width at 1920px before storing; zarr is only used for UI
thumbnails so full 4K resolution is unnecessary. Drops peak RAM from
N*frame_size to 1*frame_size."
```

---

## Task 4 — Full test run

- [ ] **Step 1: Run complete test suite**

```
python -m pytest tests/utils/test_frame_sampling.py tests/dashboard/test_preprocess.py -v
```

Expected: all tests PASS (no regressions from prior tasks).

- [ ] **Step 2: Run broader suite**

```
python -m pytest tests/ -v --ignore=tests/semantics
```

Expected: no new failures outside the two modified modules.
