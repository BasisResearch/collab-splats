# Frame Extraction Performance — Design Spec

**Date:** 2026-05-28  
**Status:** Draft  
**Scope:** `collab_splats/utils/frame_sampling.py`, `collab_splats/dashboard/panes/preprocess.py`

---

## Problem

Two independent issues combine to make frame extraction feel slow and the dashboard unresponsive:

1. **FPS seeking overhead.** `sample_frames_fps` uses `cv2.CAP_PROP_POS_FRAMES` + `cap.read()` for each target frame. On H.264/H.265, each seek requires a keyframe (I-frame) lookup + partial decode forward to the target — O(N_targets × avg_GOP_size). Sparse extraction from long videos is disproportionately slow.

2. **GIL starvation in the dashboard.** The extraction runs in a `threading.Thread`, but `cv2.cap.read()` holds the Python GIL on every call. The Tornado IOLoop (Panel's server) can't process any callbacks while the thread holds the GIL → browser clicks time out.

3. **`on_progress` flooding.** Called every decoded frame. For a 3600-frame video that's 3600 Panel reactive updates on the IOLoop — worsening the starvation.

The optical flow method has no seeking penalty (sequential decode is fine), but still suffers from GIL starvation via the same cv2 loop. The LK flow computation itself is *not* addressed here — user confirmed extraction is the bottleneck, evaluate after this lands.

---

## Design

### 1. Auto-cascade decoder backend (`frame_sampling.py`)

Add a module-level probe that selects the best available decoder once at import time:

```
torchcodec  →  ffmpeg subprocess  →  cv2  (fallback)
```

**`_get_decoder_backend() -> Literal["torchcodec", "ffmpeg", "cv2"]`**

- Tries `import torchcodec` — if available, returns `"torchcodec"`.
- Tries `shutil.which("ffmpeg")` — if found, returns `"ffmpeg"`.
- Falls back to `"cv2"`.

Result cached in a module-level variable (probe runs once).

---

### 2. `sample_frames_fps` — backend dispatch

Current: seek loop with `CAP_PROP_POS_FRAMES`.  
New: dispatch to backend after computing target indices (target computation unchanged).

**torchcodec path:**
```python
from torchcodec.decoders import VideoDecoder
decoder = VideoDecoder(video_path, device="cuda" if torch.cuda.is_available() else "cpu")
frames_tensor = decoder.get_frames_at(indices=targets)  # (N, H, W, 3) uint8
frames = [t.numpy() for t in frames_tensor.data]
```
Container-level seeking — no GOP decode cost. GPU decode when CUDA available.

**ffmpeg path:**
```python
# Build select filter: eq(n,0)+eq(n,15)+...
select_expr = "+".join(f"eq(n\\,{i})" for i in targets)
cmd = ["ffmpeg", "-i", video_path,
       "-vf", f"select={select_expr}", "-vsync", "0",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
frame_size = width * height * 3
frames = []
for _ in targets:
    raw = proc.stdout.read(frame_size)
    if len(raw) < frame_size:
        break
    frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
proc.wait()
```
Out-of-process — no GIL. Container-level seek. `width`/`height` probed via ffprobe (already called for rotation) or a single `cv2.VideoCapture` metadata read.

**cv2 fallback:** existing seek loop, unchanged.

---

### 3. `sample_frames_optical_flow` — decode layer replacement

The sequential `cap.read()` loop is replaced with the cascade decoder's sequential iterator. LK flow (`cv2.calcOpticalFlowPyrLK`, `cv2.goodFeaturesToTrack`, RANSAC) stays unchanged — it operates on numpy arrays from either path.

**torchcodec path:** iterate `VideoDecoder` frame-by-frame (already supports sequential access).  
**ffmpeg path:** pipe all frames (`ffmpeg -i video -f rawvideo -pix_fmt rgb24 pipe:1`), read in chunks of `frame_size`.  
**cv2 fallback:** existing `cap.read()` loop.

The key gain: ffmpeg subprocess and torchcodec both release (or avoid) the GIL during decode, eliminating IOLoop starvation.

---

### 4. Dashboard UI fixes (`preprocess.py`)

**Throttled progress callbacks.**  
Wrap `on_progress` with a `ThrottledProgress` helper:

```python
class ThrottledProgress:
    """Wraps an on_progress callback, firing at most max_hz times per second."""
    def __init__(self, callback, max_hz: float = 10.0):
        self._cb = callback
        self._min_interval = 1.0 / max_hz
        self._last = 0.0

    def __call__(self, n: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last >= self._min_interval or n == total:
            self._last = now
            self._cb(n, total)
```

Always fires on final update (`n == total`) so the UI reaches 100%.

**IOLoop callback audit.**  
Any cv2 reads in Panel event callbacks (e.g. video frame click → seek preview) must be wrapped in `pn.state.execute(partial(run_in_executor, fn), schedule=True)` or moved to the extraction thread. Specifically: if any `_on_frame_click` or similar handler calls `cv2.VideoCapture` synchronously, switch to `asyncio.get_event_loop().run_in_executor(None, fn)`.

---

### 5. `pyproject.toml`

`torchcodec` added as a runtime dependency (already committed). CPU-only builds install without error; GPU path activates automatically when CUDA is available.

---

## Files Changed

| File | Change |
|------|--------|
| `collab_splats/utils/frame_sampling.py` | Add `_get_decoder_backend()`, `ThrottledProgress`; refactor `sample_frames_fps` and `sample_frames_optical_flow` to dispatch through cascade |
| `collab_splats/dashboard/panes/preprocess.py` | Wrap `on_progress` with `ThrottledProgress`; audit IOLoop callbacks for cv2 reads |
| `pyproject.toml` | Add `torchcodec` dep (done) |

---

## Non-Goals

- OF LK/RANSAC GPU acceleration — evaluate extraction speedup first.
- Explicit backend selector param exposed to callers — auto-cascade covers both notebook and dashboard transparently.
- Changing frame-writing logic in `_extract_frames` (feedforward pipeline helper) — out of scope; uses same `sample_frames_fps` / `sample_frames_optical_flow`, benefits automatically.

---

## Testing

- Existing `tests/` unit tests for `sample_frames_fps` and `sample_frames_optical_flow` must pass with all three backends (mock torchcodec/ffmpeg absent to force cv2 fallback in CI).
- Dashboard: manual verify — click on video frame during active extraction no longer times out browser.
- Benchmark (informal): time `sample_frames_fps` on a 2-min 1080p H.264 video before/after; expect >5× speedup on ffmpeg path vs cv2 seek.
