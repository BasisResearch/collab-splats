# preproc cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Split `collab_splats/preproc` into measure-then-select, delete the overengineering it accreted, and cut the docstrings to something a human reads.

**Architecture:** `qa.compute_video_quality` measures the whole video once into `video_quality_report.json`; the three samplers (`sample_fps`, `sample_uniform`, `sample_optical_flow`) read that report through one predicate, `filter_frame_quality`, instead of running an inline quality gate. `video.py` collapses 8 functions to 5. `qa` measures, `sampling` decides, nothing writes a verdict into the report.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), ffmpeg/ffprobe, OpenCV, numpy, zarr 3.1.5, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-preproc-cleanup-design.md`

---

## Read this before Task 1

**Environment.** Every command in this plan uses `/opt/venv/reconstruction/bin/python` (py3.11). The bare `python` on PATH may be 3.13 and is the wrong interpreter for this project.

**Formatting.** `black` in the venv is newer than the repo's formatting. **Never run `black .` or `isort .` across the repo.** Format only the files you touched:

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 <files you edited>
```

`isort` inherits black's default line length of 88, not the repo's 120 — write long import lists parenthesized from the start rather than fixing them up afterwards.

**Committing.** Other sessions share this git index. Before every commit:

```bash
ls .git/sequencer 2>/dev/null && echo "REBASE IN PROGRESS — stop and ask" || echo ok
```

Commit with `git commit --only <paths>` and never bare `git commit -a`. `docs/superpowers/` is gitignored, so plan/spec edits need `git add -f`.

**Suite.** `/opt/venv/reconstruction/bin/python -m pytest tests/ -q` is green at the end of every task. If a task leaves it red, the task is not done.

### Code style — binding on every code block in this plan

Two rules this cleanup exists to enforce. The code blocks below are already written in this shape; copy them as they are, and hold any code you write yourself to the same standard.

**1. Docstrings open and close on their own lines.** The summary starts on the line after `"""`, never on the same line:

```python
# NO — summary jammed against the quotes
def f():
    """Raise if ffmpeg is missing."""

# YES — quotes alone, summary its own paragraph
def f():
    """
    Raise if ffmpeg is missing.
    """
```

Multi-line docstrings put a blank line between the summary and the bullets that follow. Bullets, not prose blocks. Module docstrings follow the same rule.

**2. Blank lines between logical blocks.** Every block comment gets a blank line above it, and a run of statements that does one thing is separated from the next run. Code that arrives as an undifferentiated wall is not human-readable, and this pass exists partly to fix that:

```python
# NO
hist = cv2.calcHist(...)
n = hist.sum()
levels = np.arange(256.0)
mean = float((hist * levels).sum() / n)
std = float(np.sqrt(...))

# YES
hist = cv2.calcHist(...)
n = hist.sum()
levels = np.arange(256.0)

# Mean and median together: they separate when a bright region drags the mean
mean = float((hist * levels).sum() / n)

# Contrast. A low std is a flat, textureless frame regardless of brightness.
std = float(np.sqrt(...))
```

---

## File Structure

| file | disposition |
|---|---|
| `collab_splats/utils/progress.py` | **create** — one `progress()` generator for tqdm-or-callback |
| `collab_splats/preproc/video.py` | 8 functions → 5; `iter_frames` gains ranges and yields `(idx, frame)` |
| `collab_splats/preproc/qa.py` | delete the gate; histogram exposure; ORB detect/match split; `workers=`; `load_video_quality` |
| `collab_splats/preproc/sampling.py` | 3 public samplers + `filter_frame_quality`; dispatcher and 6 helpers deleted |
| `collab_splats/preproc/frame_store.py` | delete `is_stale`, `_STALENESS_KEYS`, `records()`; blank-line the blocks |
| `collab_splats/preproc/__init__.py` | exports pruned to the live surface |
| `collab_splats/preproc/viz.py` | reads report rows; loses the `_combine_scores` import |
| `collab_splats/wrapper/reconstructor.py` | `preprocessing` → `preproc`; three-sampler switch; report wiring |
| `collab_splats/dashboard/pipeline.py` | three-sampler switch |
| `collab_splats/wrapper/splatter.py` | `sample_frames` → `sample_optical_flow` |
| `evals/datasets.py` | bug fix: `_load_video` currently raises unconditionally |
| `configs/base.yaml`, `configs/loop_closure.yaml`, `docs/source/tutorials/03_splats/configs/base.yaml` | section rename + `n_workers: 4` |
| `configs/README.md`, `CLAUDE.md` | doc rows |
| `tests/preproc/*`, `tests/wrapper/*`, `tests/dashboard/test_pipeline.py`, `tests/evals/test_datasets.py`, `tests/remote/test_rerun.py`, `tests/pointcloud/test_loger_creator.py` | follow their subjects |

---

### Task 1: Capture the parity baseline

Nothing in this task changes production code. It records what today's sampler returns, so Task 8's rewrite has something to be checked against. **It must run before any sampler edit** — once `sample_frames` is gone the baseline cannot be recaptured.

**Files:**
- Create: `tests/preproc/data/parity_baseline.json`
- Create: `tests/preproc/test_sampling_parity.py`

- [x] **Step 1: Capture the baseline from the current code**

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import json, pathlib
from collab_splats.preproc.sampling import sample_frames

VID = "data/tutorial/tutorial_example-video.mp4"
out = {"video": VID, "cases": {}}

# uniform: 30 frames over the video. spacing 82 -> radius 3, so every window is
# full width and the substitution logic is genuinely exercised.
_, recs = sample_frames(VID, method="uniform", max_frames=30)
out["cases"]["uniform_max30"] = [int(r["frame_idx"]) for r in recs]

# fps: 0.5 samples/s over a 23.976 fps source -> stride 48, 50 targets.
_, recs = sample_frames(VID, method="fps", fps=0.5)
out["cases"]["fps_0.5"] = [int(r["frame_idx"]) for r in recs]

p = pathlib.Path("tests/preproc/data/parity_baseline.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2))
for k, v in out["cases"].items():
    print(k, len(v), v[:6], "...")
PY
```

Expected: two lines, `uniform_max30 30 [...]` and `fps_0.5 50 [...]`.

- [x] **Step 2: Write the parity test (passing today, and it must still pass after Task 8)**

```python
"""
Sampler parity: the report-driven rewrite must select the frames the inline gate did.

The baseline in data/parity_baseline.json was captured from the pre-cleanup
sample_frames on the tutorial video (plan Task 1). Sharpness is bit-identical
across the rewrite; exposure moves from the 480-wide analysis gray to the native
gray, so a diff here is either that row or a real regression — see the design
doc, section 3.3. Do not relax this test to make it pass.
"""

import json
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent / "data" / "parity_baseline.json"
VIDEO = Path("data/tutorial/tutorial_example-video.mp4")

pytestmark = pytest.mark.skipif(not VIDEO.exists(), reason="tutorial video not present")


@pytest.fixture(scope="module")
def baseline():
    return json.loads(BASELINE.read_text())


def test_uniform_matches_baseline(baseline):
    from collab_splats.preproc.sampling import sample_frames

    _, records = sample_frames(str(VIDEO), method="uniform", max_frames=30)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["uniform_max30"]


def test_fps_matches_baseline(baseline):
    from collab_splats.preproc.sampling import sample_frames

    _, records = sample_frames(str(VIDEO), method="fps", fps=0.5)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["fps_0.5"]
```

- [x] **Step 3: Run it — it must pass against the unchanged code**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling_parity.py -q
```

Expected: `2 passed`. A failure here means the capture and the test disagree about the call — fix before continuing, because every later task trusts this file.

- [x] **Step 4: Commit**

```bash
git commit --only tests/preproc/data/parity_baseline.json tests/preproc/test_sampling_parity.py \
  -m "test(preproc): record sampler parity baseline before the cleanup rewrite"
```

---

### Task 2: `collab_splats/utils/progress.py`

**Files:**
- Create: `collab_splats/utils/progress.py`
- Test: `tests/utils/test_progress.py`

- [x] **Step 1: Write the failing test**

```python
"""
One progress helper for tqdm bars and dashboard callbacks alike.
"""

from collab_splats.utils.progress import progress


def test_progress_yields_every_item():
    assert list(progress(range(5), total=5)) == [0, 1, 2, 3, 4]


def test_progress_forwards_done_and_total_to_callback():
    seen = []

    list(progress(range(3), total=3, on_progress=lambda d, t: seen.append((d, t))))

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_progress_infers_total_from_a_sized_iterable():
    seen = []

    list(progress([7, 8], on_progress=lambda d, t: seen.append((d, t))))

    assert seen == [(1, 2), (2, 2)]


def test_progress_handles_an_unsized_iterable():
    seen = []

    out = list(progress(iter([1, 2]), on_progress=lambda d, t: seen.append((d, t))))

    assert out == [1, 2]
    # No length available, so total stays 0 rather than guessing
    assert seen == [(1, 0), (2, 0)]
```

- [x] **Step 2: Run it to confirm it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/utils/test_progress.py -q
```

Expected: FAIL, `ModuleNotFoundError: No module named 'collab_splats.utils.progress'`.

- [x] **Step 3: Implement**

```python
"""
Progress reporting: one tqdm-or-callback wrapper shared across modules.
"""

import logging
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> Iterator[Any]:
    """
    Yield from an iterable, driving a tqdm bar or an on_progress(done, total) callback.

    A caller with a UI passes on_progress and gets no terminal bar; a caller
    without one gets the bar. Exactly one of the two runs, never both.
    """
    # len() when the caller did not say — an unsized iterable reports total 0,
    # which tqdm renders as an open-ended bar rather than a wrong denominator.
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = 0

    # Callback path: no bar at all, so a dashboard worker writes nothing to stdout
    if on_progress is not None:
        for done, item in enumerate(iterable, start=1):
            yield item
            on_progress(done, total)
        return

    # Terminal path: tqdm owns the counting, and closes even if the consumer breaks
    bar = tqdm(total=total or None, desc=desc, unit="frame")

    try:
        for item in iterable:
            yield item
            bar.update(1)
    finally:
        bar.close()
```

- [x] **Step 4: Run the test**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/utils/test_progress.py -q
```

Expected: `4 passed`.

- [x] **Step 5: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/utils/progress.py tests/utils/test_progress.py
git commit --only collab_splats/utils/progress.py tests/utils/test_progress.py \
  -m "feat(utils): progress() — one tqdm-or-callback wrapper for every module"
```

---

### Task 3: `video.py` — 8 functions to 5

**Files:**
- Modify: `collab_splats/preproc/video.py` (whole file)
- Modify: `collab_splats/preproc/qa.py:18` (import), `collab_splats/preproc/qa.py:347-354` (loop)
- Modify: `collab_splats/preproc/sampling.py:24-28` (import), `:320` (`_iter_selected_frames` call), `:443` (`_iter_frames` call)
- Modify: `collab_splats/preproc/__init__.py`
- Test: `tests/preproc/test_video.py`

`iter_frames` yields `(frame_idx, frame_bgr)` — every consumer needs the source index, and today each one re-derives it from `enumerate` or a `zip` against a target list.

- [x] **Step 1: Write the failing tests**

Change the import block at the top of `tests/preproc/test_video.py` (lines 9-10) to the public names:

```python
import numpy as np
import pytest

from collab_splats.preproc.video import (
    extract_frame,
    get_video_info,
    iter_frames,
)
```

Delete `test_probe_dims_matches_full_info` (lines 34-45) — `_probe_dims` is gone, and the first test below replaces it. Rewrite `test_iter_frames_yields_all_frames_bgr` (line 47) to unpack tuples, or delete it as superseded. Then append:

```python
def test_get_video_info_without_count_frames_matches_dims(tiny_video):
    full = get_video_info(tiny_video)
    cheap = get_video_info(tiny_video, count_frames=False)

    assert (cheap["width"], cheap["height"]) == (full["width"], full["height"])
    assert cheap["fps"] == full["fps"]


def test_iter_frames_yields_index_and_bgr(tiny_video):
    out = list(iter_frames(tiny_video))
    total = get_video_info(tiny_video)["total_frames"]

    assert len(out) == total
    assert [i for i, _ in out] == list(range(total))
    assert out[0][1].ndim == 3 and out[0][1].dtype == np.uint8


def test_iter_frames_indices_yields_only_those(tiny_video):
    wanted = [0, 3, 7]

    out = list(iter_frames(tiny_video, indices=wanted))

    assert [i for i, _ in out] == wanted


def test_iter_frames_indices_matches_full_decode(tiny_video):
    everything = {i: f for i, f in iter_frames(tiny_video)}

    for idx, frame in iter_frames(tiny_video, indices=[1, 4]):
        assert np.array_equal(frame, everything[idx])


def test_iter_frames_range_matches_full_decode(tiny_video):
    everything = {i: f for i, f in iter_frames(tiny_video)}

    out = list(iter_frames(tiny_video, start=2, count=3))

    assert [i for i, _ in out] == [2, 3, 4]
    for idx, frame in out:
        assert np.array_equal(frame, everything[idx])


def test_iter_frames_rejects_indices_with_a_range(tiny_video):
    with pytest.raises(ValueError, match="indices"):
        list(iter_frames(tiny_video, indices=[1], start=2))


def test_extract_frame_accepts_a_preprobed_info(tiny_video):
    info = get_video_info(tiny_video)

    assert np.array_equal(extract_frame(tiny_video, 2, info=info), extract_frame(tiny_video, 2))
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -q
```

Expected: FAIL with `ImportError: cannot import name 'iter_frames'`.

- [x] **Step 3: Rewrite `video.py`**

Replace the whole file:

```python
"""
Video decode and probe: the only module that shells out to ffmpeg/ffprobe.

Holds no measurement and no selection logic, so both preproc.qa and
preproc.sampling can depend on it without a cycle.

Colour convention: iter_frames yields BGR (what cv2 wants), extract_frame
returns RGB (what its consumers store). Both are uint8 HWC.
"""

import itertools
import json
import logging
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


########################################################################
# Probe
########################################################################


def _require_ffmpeg() -> None:
    """
    Raise if ffmpeg/ffprobe are missing — the only supported decode backend.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH; install ffmpeg (e.g. `apt install ffmpeg`)")


def _rotation_degrees(stream: dict) -> int:
    """
    CW display rotation from an ffprobe stream dict.

    Two metadata locations: legacy tags.rotate (CW), and Display Matrix side
    data (modern GoPro/iPhone; ffprobe reports CCW, convert with (-rot) % 360).
    """
    rotate = stream.get("tags", {}).get("rotate")
    if rotate:
        return int(rotate) % 360

    for sd in stream.get("side_data_list", []):
        if sd.get("side_data_type") == "Display Matrix" and sd.get("rotation") is not None:
            return int(-sd["rotation"]) % 360

    return 0


def get_video_info(video_path: str | Path, *, count_frames: bool = True) -> dict:
    """
    Video metadata via ffprobe.

    - Keys: total_frames, fps, duration_s, width, height. All zeros if unprobeable.
    - Width/height are DISPLAY dims (rotation applied), matching what decode yields.
    - count_frames=False skips the -count_packets full demux, which is the whole
      cost of this call on a long video. total_frames then comes from the
      container's nb_frames and is 0 when the container does not carry it.
    """
    _require_ffmpeg()
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}

    # -count_packets demuxes the whole file for a reliable count when nb_frames
    # is absent; -select_streams v:0 keeps the cheap path to one stream.
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0", "-show_streams"]
    if count_frames:
        cmd.append("-count_packets")
    cmd.append(str(video_path))

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe failed for %s", video_path, exc_info=True)
        return zeros

    for s in streams:
        if s.get("codec_type") != "video":
            continue

        # Frame rate arrives as a ratio string, e.g. "30000/1001"
        num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0

        total = int(s.get("nb_frames") or s.get("nb_read_packets") or 0)
        width, height = int(s.get("width") or 0), int(s.get("height") or 0)

        # ffmpeg auto-rotates its output, so 90/270 swaps the display W/H
        if _rotation_degrees(s) in (90, 270):
            width, height = height, width

        duration_s = total / fps if fps > 0 else 0.0
        return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}

    return zeros


########################################################################
# Decode
########################################################################


def iter_frames(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    start: int = 0,
    count: int | None = None,
    info: dict | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, BGR uint8 HWC) from one ffmpeg rawvideo pipe.

    Three modes, one pipe and one cleanup path:

    - no arguments — every frame, in order.
    - indices=[...] — only those source frames, via a `select` filter. ffmpeg
      still demuxes from frame 0, so this is cheap in Python but not in IO.
    - start=/count= — a contiguous range via an INPUT seek, so the demuxer skips
      everything before it. This is the mode range-parallel workers use; the
      `select` filter cannot serve them because it would have every worker demux
      the whole file.

    indices and start/count are mutually exclusive. Pass info= to reuse a probe.
    """
    if indices is not None and (start or count is not None):
        raise ValueError("iter_frames: pass either indices= or start=/count=, not both")

    _require_ffmpeg()

    # count_frames=False: decoding never needs the total, and the full demux it
    # costs is the single most expensive thing this module does.
    info = info if info is not None else get_video_info(video_path, count_frames=False)
    w, h = info["width"], info["height"]
    if w == 0 or h == 0:
        return

    cmd = ["ffmpeg", "-v", "error"]

    if indices is not None:
        if len(indices) == 0:
            return

        # select='eq(n\,i)+eq(n\,j)+...' emits only these frame numbers; -vsync 0
        # keeps them 1:1 (no constant-frame-rate resampling or duplication).
        ordered = sorted({int(i) for i in indices})
        expr = "+".join(f"eq(n\\,{i})" for i in ordered)
        cmd += ["-i", str(video_path), "-vf", f"select={expr}", "-vsync", "0"]
        index_source: Iterator[int] = iter(ordered)
    else:
        # Seek to the frame midpoint, not its start: PTS float rounding can
        # otherwise land the demuxer past the target and start at frame N+1.
        if start:
            fps = info["fps"]
            if not fps:
                raise ValueError(f"iter_frames: cannot seek {video_path} without fps")
            cmd += ["-ss", f"{max(start - 0.5, 0) / fps:.6f}"]

        cmd += ["-i", str(video_path)]
        if count is not None:
            cmd += ["-frames:v", str(int(count))]
        index_source = itertools.count(start)

    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3

    try:
        # Read fixed-size frames until the pipe runs dry
        for frame_idx in index_source:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield frame_idx, np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def extract_frame(video_path: str | Path, frame_idx: int, *, info: dict | None = None) -> np.ndarray:
    """
    Decode one frame via ffmpeg input-seek; returns (H, W, 3) uint8 RGB.

    - Exact on constant-frame-rate video; may land one frame off near keyframes
      on VFR sources.
    - Pass info= (a get_video_info dict) to hoist the probe out of a loop —
      probing per call is ~8x the cost of the decode itself.
    """
    _require_ffmpeg()
    info = info if info is not None else get_video_info(video_path)
    fps, w, h, total = info["fps"], info["width"], info["height"], info["total_frames"]

    if not fps or not w or not h:
        raise ValueError(f"cannot seek {video_path}: missing fps/width/height")
    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"extract_frame: frame {frame_idx} out of range for {video_path}")

    # Seek to the frame midpoint — same PTS-rounding guard as iter_frames.
    # -ss before -i is an input seek (demuxer-level); the rawvideo pipe avoids a temp file.
    seek_s = max(frame_idx - 0.5, 0) / fps
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{seek_s:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    raw = proc.stdout
    if len(raw) < w * h * 3:
        err = proc.stderr.decode(errors="replace")[-500:]
        raise ValueError(f"extract_frame: frame {frame_idx} not found in {video_path}: {err}")

    return np.frombuffer(raw[: w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()
```

- [x] **Step 4: Update the in-repo callers**

`collab_splats/preproc/qa.py` line 18:

```python
from collab_splats.preproc.video import get_video_info, iter_frames
```

`collab_splats/preproc/qa.py` — the loop at 347-354 becomes this; the `enumerate` goes away because `iter_frames` supplies the index:

```python
    for idx, bgr in tqdm(
        iter_frames(video_path),
        total=info["total_frames"],
        desc="measuring frames",
        unit="frame",
    ):
```

`collab_splats/preproc/sampling.py` lines 24-28:

```python
from collab_splats.preproc.video import get_video_info, iter_frames
```

`collab_splats/preproc/sampling.py` line 320 — `_iter_selected_frames` took `(path, indices, w, h)` and yielded bare frames; `iter_frames` probes for itself and yields pairs:

```python
    frame_by_idx = dict(iter_frames(video_path, indices=wanted))
```

`collab_splats/preproc/sampling.py` line 443, inside `_iter_scored_frames`:

```python
        for idx, frame in iter_frames(video_path):
```

`_sample_positions` and `_sample_uniform`/`_sample_fps` still pass `w=`/`h=`; leave those parameters in place for now — Task 10 deletes `_sample_positions` entirely. They are simply unused by the new call.

`collab_splats/preproc/__init__.py` — add `iter_frames` to the import from `.video` and to `__all__`.

`tests/preproc/test_qa.py` line 20 and line 319:

```python
from collab_splats.preproc.video import iter_frames
```

```python
    grays = [_analysis_gray(bgr) for _, bgr in iter_frames(tiny_video)]
```

`tests/preproc/test_sampling.py` line 62:

```python
from collab_splats.preproc.video import iter_frames
```

Lines 134-147 monkeypatched `_iter_selected_frames` to prove uniform sampling does not full-decode. One function now serves both modes, so assert on the call shape instead — replace that test with:

```python
def test_uniform_decodes_in_one_select_pass(tiny_video, monkeypatch):
    """
    Uniform sampling must use the select filter, in exactly one ffmpeg call.
    """
    import collab_splats.preproc.sampling as s

    real = s.iter_frames
    calls = []

    def counting(path, **kwargs):
        calls.append(kwargs.get("indices"))
        return real(path, **kwargs)

    monkeypatch.setattr(s, "iter_frames", counting)

    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=4)

    assert len(frames) == 4
    # One call, and it named the frames it wanted rather than decoding everything
    assert len(calls) == 1 and calls[0] is not None
```

`tests/preproc/test_sampling.py` line 155:

```python
    bgr = [f for _, f in iter_frames(tiny_video)]
```

`tests/pointcloud/test_loger_creator.py` line 22 — `_seek_frame` is gone:

```python
from collab_splats.preproc.video import extract_frame, get_video_info
```

The call site at 735-745 becomes `extract_frame(video_path, idx, info=info)`. Read lines 730-750 before editing; keep the surrounding loop and the hoisted `info = get_video_info(...)` exactly as they are — the point of that test is that the probe stays hoisted, and `info=` is how it now stays hoisted.

- [x] **Step 5: Run the preproc and loger tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/test_loger_creator.py -q
```

Expected: all pass. `tests/preproc/test_sampling_parity.py` passing here is the real signal — the decode rewrite must not move a single selected index.

- [x] **Step 6: Full suite, format, commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/video.py collab_splats/preproc/qa.py collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_video.py tests/preproc/test_qa.py tests/preproc/test_sampling.py tests/pointcloud/test_loger_creator.py
git commit --only collab_splats/preproc/video.py collab_splats/preproc/qa.py collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_video.py tests/preproc/test_qa.py tests/preproc/test_sampling.py tests/pointcloud/test_loger_creator.py \
  -m "refactor(preproc): video.py 8 functions to 5 — iter_frames covers all three decode modes"
```

---

### Task 4: Rename `preprocessing:` to `preproc:` and add `n_workers`

Mechanical, isolated, and it fails loudly on a stale config rather than silently substituting defaults.

**Files:**
- Modify: `configs/base.yaml:16`, `configs/loop_closure.yaml:19`, `docs/source/tutorials/03_splats/configs/base.yaml:5`
- Modify: `collab_splats/wrapper/reconstructor.py` (`validate_config`, `preprocess`, `build_pointcloud`, `_extract_frames` message)
- Modify: `configs/README.md:287-288,319-322`
- Test: `tests/wrapper/test_reconstructor.py`, `tests/wrapper/test_reconstructor_preprocess.py`, `tests/remote/test_rerun.py`

**`configs/base.yaml` is already modified in the working tree by concurrent work — run `git diff configs/base.yaml` first and edit around what is there. Do not overwrite the file.**

- [x] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py`:

```python
def test_stale_preprocessing_key_is_refused(tmp_path):
    """
    A pre-2026-08-22 config carries `preprocessing:`; it must raise, not be ignored.
    """
    from collab_splats.wrapper.reconstructor import Reconstructor

    config = {
        "input_path": str(tmp_path / "v.mp4"),
        "output_path": str(tmp_path / "out"),
        "pointcloud": {"method": "feedforward", "backend": "vggt_omega", "loop_closure": False},
        "preprocessing": {"frame_selection": "fps", "fps": 1.0},
    }

    with pytest.raises(ValueError, match="renamed to 'preproc'"):
        Reconstructor.validate_config(config)
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_stale_preprocessing_key_is_refused -q
```

Expected: FAIL — `DID NOT RAISE`.

- [x] **Step 3: Rename the config section**

`configs/base.yaml` — change line 16 from `preprocessing:` to `preproc:` and add the new key after `max_frames`, keeping the existing comment-column alignment:

```yaml
  n_workers: 4                # quality report only: decode+measure this many frame ranges in
                              # parallel. 1 = serial. Measured on a 2388-frame 1080x1920 video:
                              # 2 -> 1.90x, 4 -> 3.04x (best), 6 -> 2.95x, 12 -> 2.06x. NOT
                              # auto-derived — os.cpu_count() reports HOST cores inside a
                              # container and ignores the cgroup quota. Set to 1 while a GPU
                              # eval run is using the machine (CLAUDE.md: OOM risk).
```

`configs/loop_closure.yaml:19` and `docs/source/tutorials/03_splats/configs/base.yaml:5` — change `preprocessing:` to `preproc:`. No `n_workers` in either; both deep-merge over base.

- [x] **Step 4: Update `reconstructor.py`**

In `validate_config`, after the `input_path`/`output_path` loop and before the `pointcloud` block:

```python
        # `preprocessing` was renamed to `preproc` (2026-08-22) to match the module
        # and the stage name. Refuse a stale block rather than silently ignoring it
        # and substituting base.yaml defaults — every run_config.yaml already under
        # environments-processed/ carries the old name.
        if "preprocessing" in config:
            raise ValueError(
                "config key 'preprocessing' was renamed to 'preproc' (2026-08-22); "
                "rename the section in your config"
            )
```

`preprocess()` line 656:

```python
        pre_cfg = self.config["preproc"]
```

`build_pointcloud` line 705:

```python
            max_frames=self.config["preproc"]["max_frames"],
```

`_extract_frames` line 141:

```python
            f"preproc.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}"
```

Then sweep the remaining prose references:

```bash
rtk proxy grep -n "preprocessing" collab_splats/wrapper/reconstructor.py
```

Rewrite each hit that names the config key. Leave hits that mean the English word ("model preprocessing").

- [x] **Step 5: Update the config docs and fixtures**

`configs/README.md` — replace `preprocessing.` with `preproc.` on lines 287, 288, 319, 320, 321, 322, and add a row after the `max_frames` row:

```markdown
| `preproc.n_workers` | int | `4` | Quality-report parallelism: decode+measure this many frame ranges at once. `1` = serial. Not auto-derived (`os.cpu_count()` reports host cores in a container). Set to `1` during a GPU eval run. |
```

Fixtures — `preprocessing` → `preproc` at `tests/remote/test_rerun.py:19,76,191`, `tests/wrapper/test_reconstructor.py:34,104,222,224,225`, `tests/wrapper/test_reconstructor_preprocess.py:33`. The new test from Step 1 keeps the old key deliberately; do not sweep it.

```bash
rtk proxy grep -rn '"preprocessing"' tests/ --include=*.py
```

- [x] **Step 6: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
```

Expected: all pass, including the new refusal test.

- [x] **Step 7: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit --only configs/base.yaml configs/loop_closure.yaml docs/source/tutorials/03_splats/configs/base.yaml configs/README.md collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py tests/remote/test_rerun.py \
  -m "refactor(configs)!: rename preprocessing: to preproc:, add preproc.n_workers

Old configs raise rather than silently falling back to defaults."
```

---

### Task 5: `qa.py` — histogram exposure

**Files:**
- Modify: `collab_splats/preproc/qa.py:123-136`
- Test: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Append to `tests/preproc/test_qa.py`:

```python
@pytest.mark.parametrize("kind", ["uniform", "constant", "bimodal", "narrow"])
def test_compute_exposure_matches_the_numpy_path(kind):
    """
    Histogram exposure must agree with the numpy definitions it replaced.
    """
    rng = np.random.default_rng(0)
    shapes = [(2, 2), (3, 5), (64, 64), (17, 31)]

    for shape in shapes:
        if kind == "uniform":
            gray = rng.integers(0, 256, shape, dtype=np.uint8)
        elif kind == "constant":
            gray = np.full(shape, 137, np.uint8)
        elif kind == "bimodal":
            gray = rng.integers(0, 2, shape, dtype=np.uint8) * 255
        else:
            gray = rng.integers(100, 140, shape, dtype=np.uint8)

        out = compute_exposure(gray)

        # Exact: mean, median and both clipping fractions. The float64 cast on the
        # histogram and the two-order-statistic median are what make them exact —
        # see the design doc, section 5.1.
        assert out["exposure_mean"] == float(gray.mean())
        assert out["exposure_median"] == float(np.median(gray))
        assert out["clipped_low_frac"] == float((gray == 0).mean())
        assert out["clipped_high_frac"] == float((gray == 255).mean())

        # std differs only in summation order; measured max deviation 4.3e-14
        assert out["exposure_std"] == pytest.approx(float(gray.std()), abs=1e-9)
```

Confirm `compute_exposure`, `np` and `pytest` are imported at the top of the file; add `compute_exposure` to the `collab_splats.preproc.qa` import list if absent.

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k matches_the_numpy_path -q
```

Expected: PASS. The current numpy implementation trivially satisfies its own definitions — this test is a **pin**, not a red-green driver. Its job is to fail in Step 4 if the histogram rewrite is wrong. Note that and continue.

- [x] **Step 3: Replace `compute_exposure`**

```python
def compute_exposure(gray: np.ndarray) -> dict:
    """
    Brightness distribution plus the fraction of pixels pinned at either end.

    - One 256-bin histogram serves all five numbers: ~98x faster than the numpy
      path it replaces (0.34 ms vs 33.5 ms on 1920x1080), and equal to it.
    """
    # float64: cv2.calcHist returns float32, and a 1080x1920 frame's 2.07M counts
    # do not survive it — the clipping fractions come out ~3e-8 off.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    n = hist.sum()
    levels = np.arange(256.0)
    cumulative = np.cumsum(hist)

    # Mean and median together: they separate when a small bright region
    # (a window, a lamp) drags the mean while most of the scene stays dark.
    exposure_mean = float((hist * levels).sum() / n)

    # np.median averages the two central order statistics on an even pixel count.
    # searchsorted(cumulative, n/2) alone returns only the lower one — measured
    # 127.5 off on a two-pixel frame — so take both and average.
    k_lo, k_hi = (int(n) - 1) // 2, int(n) // 2
    exposure_median = float(
        (np.searchsorted(cumulative, k_lo, side="right") + np.searchsorted(cumulative, k_hi, side="right")) / 2
    )

    # Contrast. A low std is a flat, textureless frame regardless of brightness.
    exposure_std = float(np.sqrt((hist * (levels - exposure_mean) ** 2).sum() / n))

    # Clipped pixels are destroyed data, not merely dark or bright data: 0 and 255
    # are the two values where the sensor recorded nothing recoverable.
    clipped_low_frac = float(hist[0] / n)
    clipped_high_frac = float(hist[255] / n)

    return {
        "exposure_mean": exposure_mean,
        "exposure_median": exposure_median,
        "exposure_std": exposure_std,
        "clipped_low_frac": clipped_low_frac,
        "clipped_high_frac": clipped_high_frac,
    }
```

- [x] **Step 4: Run the test against the new implementation**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -q
```

Expected: all pass. A failure on `exposure_median` means the two-order-statistic form was dropped; on a clipping fraction, the `float64` cast.

- [x] **Step 5: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "perf(preproc): 256-bin histogram exposure — 98x faster, equal to the numpy path"
```

---

### Task 6: `qa.py` — ORB detected once per frame, not twice

`match_orb(a, b)` runs `detectAndCompute` on both members of every pair, so every frame's ORB runs twice across a video. Split detect from match; `compute_video_quality` already holds a `pending` dict to hang the descriptors on.

**Files:**
- Modify: `collab_splats/preproc/qa.py:172-206` (`match_orb`), `:331,361-370` (the pending loop)
- Test: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

```python
def test_detect_orb_and_match_descriptors_reproduce_match_orb(noise_gray):
    """
    The split path must return exactly what the one-shot wrapper returns.
    """
    from collab_splats.preproc.qa import detect_orb, match_descriptors, match_orb

    other = np.roll(noise_gray, 5, axis=1)

    expected_a, expected_b = match_orb(noise_gray, other)
    got_a, got_b = match_descriptors(detect_orb(noise_gray), detect_orb(other))

    assert np.array_equal(got_a, expected_a)
    assert np.array_equal(got_b, expected_b)


def test_detect_orb_on_a_featureless_frame_returns_no_descriptors():
    from collab_splats.preproc.qa import detect_orb

    kp, desc = detect_orb(np.zeros((64, 64), np.uint8))

    assert desc is None or len(kp) == 0
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k detect_orb -q
```

Expected: FAIL, `ImportError: cannot import name 'detect_orb'`.

- [x] **Step 3: Split `match_orb`**

Replace lines 172-206 of `qa.py`:

```python
def detect_orb(gray: np.ndarray, *, n_features: int = 1000) -> tuple[tuple, np.ndarray | None]:
    """
    ORB keypoints and descriptors for one grayscale frame.

    - Split out of match_orb so a video run detects each frame ONCE: every frame
      is the partner of one pair and the current frame of the next, so the
      combined call ran ORB twice per frame. Measured 1.38x on the pair loop.
    - desc is None on a frame with no detectable features; that is a fact about
      the frame, not an error.
    """
    return cv2.ORB_create(nfeatures=n_features).detectAndCompute(gray, None)


def match_descriptors(feat_a: tuple, feat_b: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Mutually-matched keypoint coordinates between two detect_orb results, as Nx2 float32.

    - crossCheck makes both sides injective, which is what the downstream RANSAC
      wants, but it bounds nothing about whether the frames show the same scene:
      mutual-best still returns a full set of matches on unrelated frames. See
      docs/superpowers/specs/2026-08-20-video-quality-report-measured.md.
    """
    kp_a, desc_a = feat_a
    kp_b, desc_b = feat_b

    # A featureless frame yields no descriptors at all. Return empty rather than
    # raise: zero matches is a fact about the video, not an error.
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if desc_a is None or desc_b is None:
        return empty

    # ORB descriptors are binary, hence Hamming. crossCheck keeps only mutual
    # best matches, which removes the need for a Lowe ratio test.
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return empty

    # Pull the pixel coordinates behind each match into two aligned Nx2 arrays
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)

    return pts_a, pts_b


def match_orb(gray_a: np.ndarray, gray_b: np.ndarray, *, n_features: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    ORB matches between two grayscale frames — detect both, then match.
    """
    return match_descriptors(detect_orb(gray_a, n_features=n_features), detect_orb(gray_b, n_features=n_features))
```

- [x] **Step 4: Use the cache in `compute_video_quality`**

Line 331 — `pending` now holds the features, not the gray:

```python
    # Hold only the ORB features still owed a partner: stride + 1 frames at a
    # time, so memory does not track video length. Features, not grays: each
    # frame's ORB is computed once here and replayed for its pair.
    pending: dict[int, tuple] = {}
```

Lines 361-370:

```python
        # Motion against the frame one stride back, once one exists
        pending[idx] = detect_orb(_analysis_gray(bgr))
        partner = idx - stride

        if partner in pending:
            pts_a, pts_b = match_descriptors(pending[partner], pending[idx])

            frame_idx_a.append(partner)
            frame_idx_b.append(idx)
            n_matches.append(int(len(pts_a)))
            translation_px.append(compute_translation(pts_a, pts_b))
            parallax.append(compute_parallax(pts_a, pts_b))

            del pending[partner]
```

`_analysis_gray` is renamed to `analysis_gray` in Task 11. Use the private name here and let Task 11's sweep pick it up.

- [x] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -q
```

Expected: all pass, including the existing `compute_video_quality` tests — the report values must be unchanged, because the descriptors are the same descriptors.

- [x] **Step 6: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "perf(preproc): detect ORB once per frame, not once per pair member (1.38x)"
```

---

### Task 7: `qa.py` — `workers=` and `load_video_quality`

**Files:**
- Modify: `collab_splats/preproc/qa.py` (`compute_video_quality`, plus a new `_measure_range`, `_merge_ranges` and `load_video_quality`)
- Test: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing tests**

```python
def test_video_quality_workers_produce_an_identical_report(tiny_video):
    """
    Range-parallel measurement must not change a single number.
    """
    serial = compute_video_quality(tiny_video, motion_stride=2)
    parallel = compute_video_quality(tiny_video, motion_stride=2, workers=3)

    assert parallel["frames"] == serial["frames"]
    assert parallel["pairs"] == serial["pairs"]


def test_video_quality_rejects_a_bad_worker_count(tiny_video):
    with pytest.raises(ValueError, match="workers"):
        compute_video_quality(tiny_video, workers=0)


def test_load_video_quality_writes_then_reuses(tiny_video, tmp_path):
    from collab_splats.preproc.qa import load_video_quality

    report_path = tmp_path / "video_quality_report.json"

    first = load_video_quality(tiny_video, report_path, motion_stride=2)

    assert report_path.exists() and first["available"]

    # Second call must read the file, not re-measure it
    stamp = report_path.stat().st_mtime_ns
    second = load_video_quality(tiny_video, report_path, motion_stride=2)

    assert report_path.stat().st_mtime_ns == stamp
    assert second["frames"] == first["frames"]
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k "workers or load_video_quality" -q
```

Expected: FAIL — `compute_video_quality() got an unexpected keyword argument 'workers'`.

- [x] **Step 3: Add the range worker and the merge**

Add to the imports at the top of `qa.py`:

```python
import os
from concurrent.futures import ProcessPoolExecutor
```

Add above `compute_video_quality`:

```python
def _measure_range(args: tuple) -> tuple[list[dict], list[dict]]:
    """
    Measure one contiguous frame range in its own process; returns (frame rows, pair rows).

    Runs at module scope because ProcessPoolExecutor pickles by qualified name.
    """
    video_path, start, count, stride, info = args

    # THE THREAD PIN IS LOAD-BEARING. cv2 and numpy each fan out over every core,
    # so the "serial" baseline is already parallel and naive process fan-out
    # oversubscribes: unpinned, this measured 0.67x — SLOWER than serial.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(1)

    frame_rows: list[dict] = []
    pair_rows: list[dict] = []
    pending: dict[int, tuple] = {}

    for idx, bgr in iter_frames(video_path, start=start, count=count, info=info):
        frame_rows.append({"frame_idx": idx, **compute_frame_quality(bgr)})

        # Motion against the frame one stride back, once one exists
        pending[idx] = detect_orb(_analysis_gray(bgr))
        partner = idx - stride

        if partner in pending:
            pts_a, pts_b = match_descriptors(pending[partner], pending[idx])

            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    "n_matches": int(len(pts_a)),
                    "translation_px": compute_translation(pts_a, pts_b),
                    "parallax": compute_parallax(pts_a, pts_b),
                }
            )

            del pending[partner]

    return frame_rows, pair_rows


def _merge_ranges(results: list[tuple[list[dict], list[dict]]]) -> tuple[list[dict], list[dict]]:
    """
    Merge worker outputs, dropping the deliberate boundary overlap.

    Adjacent ranges overlap by `stride` frames so boundary pairs are still
    measured. The overlapping rows are measured twice, from two independently
    seeked ffmpeg pipes, and MUST come back identical — if they do not, input
    seek landed on different frames and every index in the report is suspect.
    """
    frames: dict[int, dict] = {}
    pairs: dict[tuple[int, int], dict] = {}

    for frame_rows, pair_rows in results:
        # Photometry: the overlap is the guard, so disagreement is fatal
        for row in frame_rows:
            seen = frames.get(row["frame_idx"])
            if seen is not None and seen != row:
                raise ValueError(
                    f"video quality: workers disagree on frame {row['frame_idx']} "
                    f"({seen} vs {row}) — input seek is unreliable on this file. "
                    "Re-run with n_workers: 1."
                )
            frames[row["frame_idx"]] = row

        # Motion: keyed by the pair, so a boundary pair measured twice collapses
        for row in pair_rows:
            pairs[(row["frame_idx_a"], row["frame_idx_b"])] = row

    return [frames[k] for k in sorted(frames)], [pairs[k] for k in sorted(pairs)]
```

- [x] **Step 4: Rewrite `compute_video_quality` around them**

The signature gains `workers`:

```python
def compute_video_quality(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    motion_stride: int | None = None,
    workers: int = 1,
) -> dict:
    """
    Measure per-frame photometry and per-pair motion across a whole video.

    Report-only: no thresholds, no verdicts. Selection policy lives in
    preproc.sampling.filter_frame_quality.

    Args:
        video_path: source video to decode; every frame is measured.
        output_path: where to write video_quality_report.json. None returns the
            report without touching disk.
        motion_stride: frames between the two members of each measured pair.
            None means round(fps) — one second of video. Must be >= 1.
        workers: decode+measure this many contiguous frame ranges in parallel.
            1 = serial. Measured 3.04x at 4 on a 2388-frame 1080x1920 video,
            declining past 6. NOT auto-derived — see the design doc, section 5.2.
    """
```

Add after the `motion_stride` validation:

```python
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
```

Replace the measurement loop — the `for idx, bgr in tqdm(...)` block and the `frames`/`pending` scaffolding above it — with the range split:

```python
    # One range per worker, each overlapping its predecessor by `stride` so the
    # pairs that straddle a boundary are still measured. Overlap dropped on merge.
    total = info["total_frames"]

    if workers == 1 or total <= stride * 2:
        ranges = [(str(video_path), 0, None, stride, info)]
    else:
        per = total // workers
        ranges = []
        for k in range(workers):
            start = max(k * per - stride, 0) if k else 0
            end = total if k == workers - 1 else (k + 1) * per
            ranges.append((str(video_path), start, end - start, stride, info))

    started = time.perf_counter()

    if len(ranges) == 1:
        results = [_measure_range(ranges[0])]
    else:
        with ProcessPoolExecutor(len(ranges)) as pool:
            results = list(pool.map(_measure_range, ranges))

    frame_rows, pair_rows = _merge_ranges(results)
```

Then assemble the columnar report from the merged rows:

```python
    frame_keys = (
        "frame_idx",
        "blur",
        "laplacian",
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    )
    frames = {k: [row[k] for row in frame_rows] for k in frame_keys}
```

and the pairs block becomes:

```python
            "pairs": {
                "frame_idx_a": [r["frame_idx_a"] for r in pair_rows],
                "frame_idx_b": [r["frame_idx_b"] for r in pair_rows],
                "n_matches": [r["n_matches"] for r in pair_rows],
                # nan -> null on the only two columns that can be non-finite.
                # json.dumps writes a bare NaN no strict parser accepts, and
                # np.nan_to_num is not the fix: its 0.0 fill would read as "no
                # motion", the opposite of "this pair failed to match".
                "translation_px": [None if np.isnan(r["translation_px"]) else r["translation_px"] for r in pair_rows],
                "parallax": [None if np.isnan(r["parallax"]) else r["parallax"] for r in pair_rows],
            },
```

Keep the existing `logger.info` announcement, the empty-report branch, the throughput log and the `output_path` write exactly as they are; only `len(frame_idx_a)` becomes `len(pair_rows)`.

- [x] **Step 5: Add `load_video_quality`**

At the end of the module:

```python
def load_video_quality(
    video_path: str | Path,
    report_path: str | Path,
    *,
    workers: int = 1,
    motion_stride: int | None = None,
) -> dict:
    """
    The quality report at report_path, measuring and writing it first if absent.

    Reuse is by existence, the same rule frames.zarr follows — so a re-run of a
    scene never re-measures, and nothing needs a staleness check.
    """
    report_path = Path(report_path)

    if report_path.exists():
        logger.info("video quality: reusing %s", report_path)
        return json.loads(report_path.read_text())

    return compute_video_quality(
        video_path,
        output_path=report_path,
        motion_stride=motion_stride,
        workers=workers,
    )
```

- [x] **Step 6: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -q
```

Expected: all pass. `test_video_quality_workers_produce_an_identical_report` is the one that matters — it is the boundary-merge guard's own test.

- [x] **Step 7: Full suite and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): compute_video_quality(workers=) range parallelism + load_video_quality"
```

---

### Task 8: `sampling.py` — `filter_frame_quality` and the three samplers

Additive. `sample_frames` stays until Task 10, so the suite and every caller stay green through this task and the next.

**Files:**
- Modify: `collab_splats/preproc/sampling.py`
- Test: `tests/preproc/test_sampling.py`

- [x] **Step 1: Write the failing tests**

```python
def _synthetic_report(n=20, bad=()):
    """
    A report shaped like compute_video_quality's, with `bad` frames failing the gate.
    """
    frames = {
        "frame_idx": list(range(n)),
        "laplacian": [200.0] * n,
        "exposure_mean": [128.0] * n,
        "exposure_std": [40.0] * n,
        "blur": [0.3] * n,
    }
    for i in bad:
        frames["laplacian"][i] = 1.0

    return {"available": True, "frames": frames}


def test_filter_frame_quality_flags_soft_frames():
    from collab_splats.preproc.sampling import filter_frame_quality

    mask = filter_frame_quality(_synthetic_report(10, bad=(3, 7)))

    assert mask.tolist() == [True, True, True, False, True, True, True, False, True, True]


def test_filter_frame_quality_flags_bad_exposure():
    from collab_splats.preproc.sampling import filter_frame_quality

    report = _synthetic_report(4)
    report["frames"]["exposure_mean"][1] = 250.0  # blown
    report["frames"]["exposure_std"][2] = 2.0  # no contrast

    assert filter_frame_quality(report).tolist() == [True, False, False, True]


def test_filter_frame_quality_blur_max_is_off_by_default():
    from collab_splats.preproc.sampling import filter_frame_quality

    report = _synthetic_report(3)
    report["frames"]["blur"] = [1.0, 1.0, 1.0]  # Crete-Roffet saturated

    assert filter_frame_quality(report).all()
    assert not filter_frame_quality(report, blur_max=0.5).any()


def test_sample_uniform_takes_the_report(tiny_video):
    from collab_splats.preproc.qa import compute_video_quality
    from collab_splats.preproc.sampling import sample_uniform

    report = compute_video_quality(tiny_video, motion_stride=2)

    frames, records = sample_uniform(tiny_video, max_frames=4, report=report)

    assert len(frames) == len(records) == 4
    assert [r["frame_idx"] for r in records] == sorted(r["frame_idx"] for r in records)


def test_sample_uniform_substitutes_a_neighbour_for_a_bad_frame(tiny_video):
    """
    A frame the report condemns must be replaced from within its window, keeping the count.
    """
    from collab_splats.preproc.qa import compute_video_quality
    from collab_splats.preproc.sampling import sample_uniform

    report = compute_video_quality(tiny_video, motion_stride=2)
    clean = [r["frame_idx"] for r in sample_uniform(tiny_video, max_frames=4, report=report)[1]]

    # Condemn every frame that clean selection picked; the count must survive
    for i in clean:
        report["frames"]["laplacian"][i] = 0.0

    frames, records = sample_uniform(tiny_video, max_frames=4, report=report)

    assert len(frames) == 4
    assert [r["frame_idx"] for r in records] != clean


def test_sample_fps_respects_the_frame_band(tiny_video):
    from collab_splats.preproc.qa import compute_video_quality
    from collab_splats.preproc.sampling import sample_fps

    report = compute_video_quality(tiny_video, motion_stride=2)

    frames, _ = sample_fps(tiny_video, fps=30.0, max_frames=5, report=report)

    assert len(frames) <= 5


def test_sample_fps_rejects_a_missing_rate(tiny_video):
    from collab_splats.preproc.qa import compute_video_quality
    from collab_splats.preproc.sampling import sample_fps

    report = compute_video_quality(tiny_video, motion_stride=2)

    with pytest.raises(ValueError, match="positive fps"):
        sample_fps(tiny_video, fps=None, report=report)


def test_sample_optical_flow_skips_report_rejected_frames(tiny_video):
    from collab_splats.preproc.qa import compute_video_quality
    from collab_splats.preproc.sampling import sample_optical_flow

    report = compute_video_quality(tiny_video, motion_stride=2)
    for i in range(len(report["frames"]["frame_idx"])):
        report["frames"]["laplacian"][i] = 0.0

    frames, records = sample_optical_flow(tiny_video, report=report)

    assert frames == [] and records == []


def test_samplers_require_a_report(tiny_video):
    from collab_splats.preproc.sampling import sample_uniform

    with pytest.raises(TypeError):
        sample_uniform(tiny_video, max_frames=4)
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "filter_frame_quality or sample_uniform or sample_fps or sample_optical_flow or require_a_report" -q
```

Expected: FAIL, `ImportError: cannot import name 'filter_frame_quality'`.

- [x] **Step 3: Add the predicate**

Add to `sampling.py`'s imports:

```python
from collab_splats.preproc.qa import _analysis_gray
from collab_splats.utils.progress import progress
```

Add a new section after the constants block:

```python
########################################################################
# Quality filter — the one place a threshold meets the report.
# qa measures; sampling decides. Nothing writes a verdict into the report.
########################################################################


def filter_frame_quality(
    report: dict,
    *,
    laplacian_min: float = 50.0,
    exposure_mean_range: tuple[float, float] = (20.0, 235.0),
    exposure_min_std: float = 10.0,
    blur_max: float | None = None,
) -> np.ndarray:
    """
    Per-frame usability mask over a quality report's photometry columns.

    - Returns one bool per frame, True = usable. Positive polarity, matching the
      name: a filter_* reports what survives.
    - Indexed by frame index directly — compute_video_quality enumerates every
      frame, so frame_idx is contiguous 0..N-1.
    - blur_max is OFF by default. Crete-Roffet `blur` saturates at 1.0 on any
      low-detail frame (a flat field and a single sharp edge both score 1.0), so
      thresholding it discards sharp frames of plain surfaces. `laplacian` is the
      sharpness gate; `blur` is there for a caller who knows the trap.

    Args:
        laplacian_min: Laplacian variance below which a frame is soft.
        exposure_mean_range: brightness band; outside is crushed or blown.
        exposure_min_std: contrast floor; below it the frame is flat.
        blur_max: optional Crete-Roffet ceiling, higher = blurrier.
    """
    f = report["frames"]
    lap = np.asarray(f["laplacian"], dtype=float)
    mean = np.asarray(f["exposure_mean"], dtype=float)
    std = np.asarray(f["exposure_std"], dtype=float)
    lo, hi = exposure_mean_range

    # Sharp enough: Laplacian variance, not Crete-Roffet blur (which saturates)
    sharp = lap >= laplacian_min

    # Exposed usably: inside the brightness band AND carrying some contrast
    exposed = (mean >= lo) & (mean <= hi) & (std >= exposure_min_std)

    usable = sharp & exposed

    # Optional perceptual-blur ceiling, off unless the caller asks for it
    if blur_max is not None:
        usable &= np.asarray(f["blur"], dtype=float) <= blur_max

    return usable
```

- [x] **Step 4: Add `_take_frames`, the shared body of the two target-list samplers**

```python
def _take_frames(
    video_path: str,
    targets: list[int],
    *,
    total: int,
    usable: np.ndarray,
    laplacian: np.ndarray,
    search_radius: int,
    on_progress,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Pick one frame per target from its window, then decode exactly those.

    The report supplies sharpness for every frame, so the winner is chosen
    BEFORE any decode: this touches len(targets) frames where the pre-report
    code decoded the whole union of windows, ~(2*search_radius + 1) times more.
    """
    if not targets:
        return [], []

    # Window radius: half the target spacing, capped, and kept under spacing/2 so
    # neighbouring windows never overlap (the index map stays deterministic).
    spacing = targets[1] - targets[0] if len(targets) > 1 else total
    radius = min(max((spacing - 1) // 2, 0), search_radius)

    # Per target, prefer a usable frame and break ties on sharpness. max() over an
    # ascending range returns the FIRST maximal element, matching the old strict
    # `key > best` comparison — the tie-break is parity-critical.
    chosen: list[int] = []
    for t in targets:
        window = sorted({min(max(t + o, 0), total - 1) for o in range(-radius, radius + 1)})
        chosen.append(max(window, key=lambda i: (bool(usable[i]), float(laplacian[i]))))

    # One ffmpeg select pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=chosen))

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx in progress(chosen, total=len(chosen), desc=desc, on_progress=on_progress):
        bgr = decoded.get(idx)
        if bgr is None:
            continue  # ffmpeg dropped the frame (should not happen)

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # blur_score comes from the report, not a recompute — same measurement,
        # and it is the column the record has always carried.
        records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx])})

    return frames, records
```

- [x] **Step 5: Add the three public samplers**

```python
def sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    report: dict,
    search_radius: int = 3,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Exactly max_frames evenly-spaced frames spanning the whole video.

    - The COUNT is the contract; spacing falls out of the video length.
    - report is required — a quality report from qa.compute_video_quality or
      qa.load_video_quality. quality= overrides filter_frame_quality's thresholds.
    """
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0 or max_frames <= 0:
        return [], []

    # N evenly-spaced indices spanning the video, endpoint-anchored. Cap at the
    # source length then dedup: rounding collides as max_frames approaches total.
    targets = np.unique(np.linspace(0, total - 1, min(max_frames, total)).round().astype(int)).tolist()

    usable = filter_frame_quality(report, **(quality or {}))
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    return _take_frames(
        video_path,
        targets,
        total=total,
        usable=usable,
        laplacian=laplacian,
        search_radius=search_radius,
        on_progress=on_progress,
        desc="Uniform sampling",
    )


def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    search_radius: int = 3,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds; re-spread if the count falls outside the band.

    - The SPACING is the contract, so the baseline between consecutive frames is
      fixed regardless of video length and the count floats.
    - Outside [min_frames, max_frames] the targets are re-spread evenly over the
      WHOLE video, never truncated — truncation would hand the reconstructor half
      a scene.
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []

    # Stride floors at 1 — a rate above the source rate cannot sample sub-frame
    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / fps)))
    targets = list(range(0, total, step))

    # Clamp the floating count into the band by re-spreading, never by truncating
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = np.unique(np.linspace(0, total - 1, min(bounded, total)).round().astype(int)).tolist()
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            native_fps * len(targets) / total,
        )

    usable = filter_frame_quality(report, **(quality or {}))
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    return _take_frames(
        video_path,
        targets,
        total=total,
        usable=usable,
        laplacian=laplacian,
        search_radius=search_radius,
        on_progress=on_progress,
        desc="fps sampling",
    )


def sample_optical_flow(
    video_path: str,
    *,
    report: dict,
    max_frames: int | None = None,
    min_disparity: float = 50.0,
    select_threshold: float = 0.5,
    rotation_threshold_deg: float = 5.0,
    lk_params: dict | None = None,
    feature_params: dict | None = None,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Keyframes by motion (LK disparity + rotation) and coverage (histogram diversity).

    - Runs its own decode pass: LK disparity is measured against a MOVING keyframe
      reference, which the report's fixed-stride pairs cannot supply. It does not
      re-measure blur or exposure — those come from the report.
    - A frame the report condemns is skipped as it arrives: the selector never
      scores it and never adopts it as its reference, so it moves to the next.
    - max_frames caps the result; frames scoring >= select_threshold are selected.
    """
    info = get_video_info(str(video_path))
    usable = filter_frame_quality(report, **(quality or {}))
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        rotation_threshold_deg=rotation_threshold_deg,
        lk_params=lk_params,
        feature_params=feature_params,
    )

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx, bgr in progress(
        iter_frames(video_path),
        total=info["total_frames"],
        desc="Optical flow selection",
        on_progress=on_progress,
    ):
        # Report gate first: an unusable frame never reaches the selector, so it
        # cannot become the reference the next frames are scored against.
        if idx >= len(usable) or not usable[idx]:
            continue

        score, components = selector.score_frame(_analysis_gray(bgr))
        if score < select_threshold:
            continue

        selector.accept_frame(_analysis_gray(bgr))
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # frame_idx is the SOURCE video index
        records.append(
            {
                "frame_idx": int(idx),
                "blur_score": float(laplacian[idx]),
                "score": score,
                "selected": True,
                **components,
            }
        )

        if max_frames is not None and len(frames) >= max_frames:
            break

    return frames, records
```

- [x] **Step 6: De-constant `OpticalFlowFrameSelector` and fold in `_combine_scores`**

`_combine_scores` is `viz.plot_disparity_sensitivity`'s only reason to exist as a separate function; it becomes a method so the selector owns its own formula. Replace the class:

```python
class OpticalFlowFrameSelector:
    """
    Streaming keyframe selector: motion (LK flow + rotation) and coverage scoring.

    Holds the reference keyframe between calls — score each candidate with
    score_frame(), promote selected frames with accept_frame(). Construct fresh
    per video.

    No cv2/kornia/open3d equivalent exists for the streaming "is this frame
    different enough from the last one I kept" policy; the pieces it is built
    from (goodFeaturesToTrack, calcOpticalFlowPyrLK, estimateAffinePartial2D,
    calcHist/compareHist) are all cv2's.
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        *,
        rotation_threshold_deg: float = 5.0,
        lk_params: dict | None = None,
        feature_params: dict | None = None,
    ):
        self.min_disparity = min_disparity
        self.rotation_threshold_deg = rotation_threshold_deg

        # Built here, not as defaults: a mutable dict default is shared across
        # every instance in the process, which is a Python trap, not a style call.
        # Lucas-Kanade sparse flow.
        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        # Shi-Tomasi corners, the seed points that flow tracks.
        self.feature_params = feature_params or dict(maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)

        # Reference keyframe state, seeded on the first scored frame
        self.last_keyframe_gray: np.ndarray | None = None
        self.last_keyframe_pts: np.ndarray | None = None

    def score_frame(self, gray: np.ndarray) -> tuple[float, dict]:
        """
        Score a grayscale frame against the current keyframe.

        Returns (score in [0, 1], components). The first frame scores 1.0 and
        seeds the keyframe state. Components:

            disparity            median LK pixel motion since last kept frame
            rotation             in-plane rotation vs last kept frame, degrees
            histogram_similarity intensity-histogram correlation with last kept frame
        """
        if self.last_keyframe_gray is None:
            self.accept_frame(gray)
            return 1.0, {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}

        # Motion signals from LK flow of keyframe corners into this frame
        disparity, rotation = 0.0, 0.0
        prev_pts, curr_pts = self._compute_flow(gray)
        if prev_pts is not None:
            disparity = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))
            rotation = self._estimate_rotation(prev_pts, curr_pts)

        # Coverage signal: histogram correlation vs the keyframe
        hist_similarity = self._hist_similarity(gray)

        components = {
            "disparity": disparity,
            "rotation": rotation,
            "histogram_similarity": hist_similarity,
        }

        return self.combine(disparity, hist_similarity, rotation=rotation), components

    def combine(self, disparity: float, histogram_similarity: float, *, rotation: float = 0.0) -> float:
        """
        Weighted motion + coverage score in [0, 1]; >= select_threshold selects.
        """
        # Fixed motion/coverage weighting
        motion_weight, coverage_weight = 0.6, 0.4

        # Motion: max of the normalised translation and rotation components
        translation_score = min(disparity / max(self.min_disparity, 1e-6), 1.0)
        rotation_score = min(rotation / self.rotation_threshold_deg, 1.0)
        motion_score = max(translation_score, rotation_score)

        # Coverage: inverse histogram correlation vs the last keyframe
        coverage_score = 1.0 - histogram_similarity

        return (motion_weight * motion_score + coverage_weight * coverage_score) / (motion_weight + coverage_weight)

    def accept_frame(self, gray: np.ndarray) -> None:
        """
        Make the given grayscale frame the new reference keyframe.
        """
        self.last_keyframe_gray = gray.copy()
        self.last_keyframe_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)

    def _compute_flow(self, gray: np.ndarray):
        """
        LK flow from keyframe corners; (None, None) if under 10 inliers survive.
        """
        if self.last_keyframe_pts is None or len(self.last_keyframe_pts) == 0:
            return None, None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_keyframe_gray, gray, self.last_keyframe_pts, None, **self.lk_params
        )
        if curr_pts is None:
            return None, None

        good_prev = self.last_keyframe_pts[status == 1]
        good_curr = curr_pts[status == 1]

        # Too few inliers -> tracking unreliable for motion estimation
        if len(good_prev) < 10:
            return None, None

        return good_prev, good_curr

    def _estimate_rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """
        Camera rotation angle (degrees) via RANSAC partial-affine fit.
        """
        if len(prev_pts) < 4:
            return 0.0

        try:
            M, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
            if M is None:
                return 0.0
            return float(np.abs(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        except Exception:
            logger.debug("Rotation estimation failed", exc_info=True)
            return 0.0

    def _hist_similarity(self, gray: np.ndarray, bins: int = 64) -> float:
        """
        Histogram correlation vs the keyframe, clamped to [0, 1].
        """
        h1 = cv2.calcHist([self.last_keyframe_gray], [0], None, [bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [bins], [0, 256])

        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()

        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))
```

Keep `_combine_scores` for now as a shim so `viz.py` and the old dispatcher keep working through Task 9; Task 10 deletes it:

```python
def _combine_scores(disparity, histogram_similarity, min_disparity, rotation=0.0):
    """
    Deprecated shim — deleted in the same pass; use OpticalFlowFrameSelector.combine.
    """
    return OpticalFlowFrameSelector(min_disparity=min_disparity).combine(
        disparity, histogram_similarity, rotation=rotation
    )
```

`_LK_PARAMS`, `_FEATURE_PARAMS`, `_SELECT_THRESHOLD` and `_ROTATION_THRESHOLD_DEG` are now unreferenced by the class but still referenced by `_iter_scored_frames`; leave the constants in place until Task 10.

- [x] **Step 7: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q
```

Expected: all pass, including `test_sampling_parity.py` (still exercising the old `sample_frames`) and the existing `test_combine_scores_monotonic_in_disparity`.

- [x] **Step 8: Full suite and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py \
  -m "feat(preproc): three report-driven samplers + filter_frame_quality"
```

---

### Task 9: Switch every caller to the three samplers

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:82-164` (`_extract_frames`), `:646-670` (`preprocess`)
- Modify: `collab_splats/dashboard/pipeline.py:26,339-380,422`
- Modify: `collab_splats/wrapper/splatter.py:296,311`
- Modify: `evals/datasets.py:246-262`
- Test: `tests/wrapper/test_reconstructor.py:145-150,186`, `tests/dashboard/test_pipeline.py:62,108`, `tests/evals/test_datasets.py:502-519`

- [x] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor_preprocess.py`:

```python
def test_preprocess_writes_a_quality_report_beside_frames_zarr(tmp_path, tiny_video):
    """
    The report is a first-class scene artefact, reused by existence like frames.zarr.
    """
    from collab_splats.wrapper.reconstructor import Reconstructor

    out = tmp_path / "out"
    rec = Reconstructor(
        {
            "input_path": str(tiny_video),
            "output_path": str(out),
            "preproc": {"frame_selection": "uniform", "fps": None, "min_frames": None, "max_frames": 4, "n_workers": 1},
        }
    )

    rec.preprocess()

    assert (out / "video_quality_report.json").exists()
    assert (out / "frames.zarr").exists()
```

Match the fixture style of the file's existing tests — read the top of `tests/wrapper/test_reconstructor_preprocess.py` and reuse its config fixture rather than hand-rolling one if it has a helper.

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py -k quality_report -q
```

Expected: FAIL — no `video_quality_report.json`.

- [x] **Step 3: `reconstructor.py`**

Replace `sample_frames` in the imports with the three samplers plus `load_video_quality`:

```python
from collab_splats.preproc import (
    FrameStore,
    get_video_info,
    load_video_quality,
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
```

`_extract_frames` gains a parameter and does the measure step:

```python
def _extract_frames(
    input_path: Path,
    frames_zarr: Path,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    n_workers: int = 1,
) -> int:
    """
    Extract frames from video or image dir into frames.zarr (sole persistent store).

    Two steps for video input: measure the whole video into
    video_quality_report.json, then select from it. An image directory takes
    every image and needs no report.
    """
```

The image-dir branch is unchanged. The video branch becomes:

```python
    # Measure before selecting. The report lands beside frames.zarr and is reused
    # by existence, so a re-run never re-measures. Report-only: it carries no
    # verdicts — filter_frame_quality applies the thresholds inside the samplers.
    report = load_video_quality(
        input_path,
        frames_zarr.parent / "video_quality_report.json",
        workers=n_workers,
    )

    # 'fps' samples at a constant wall-clock rate (band-bounded), 'uniform' spreads
    # exactly max_frames over the whole video, 'optical_flow' picks high-motion frames.
    if frame_selection == "fps":
        frame_arrays, records = sample_fps(
            str(input_path), fps=fps, min_frames=min_frames, max_frames=max_frames, report=report
        )
    elif frame_selection == "uniform":
        frame_arrays, records = sample_uniform(str(input_path), max_frames=max_frames, report=report)
    elif frame_selection == "optical_flow":
        frame_arrays, records = sample_optical_flow(str(input_path), max_frames=max_frames, report=report)
    else:
        raise ValueError(
            f"preproc.frame_selection must be 'fps', 'uniform' or 'optical_flow', got {frame_selection!r}"
        )

    method = frame_selection
```

The "0 frames selected" message drops its reference to the deleted gate:

```python
    if not frame_arrays:
        raise ValueError(
            f"0 frames selected from {input_path} ({total_frames} decoded) with "
            f"frame_selection={frame_selection!r}. Every frame failed the quality filter — "
            f"check {frames_zarr.parent / 'video_quality_report.json'} for the measurements."
        )
```

`preprocess()` passes the new key:

```python
        pre_cfg = self.config["preproc"]

        n_frames = _extract_frames(
            input_path=Path(self.config["input_path"]),
            frames_zarr=self.frames_zarr,
            frame_selection=pre_cfg["frame_selection"],
            fps=pre_cfg["fps"],
            min_frames=pre_cfg["min_frames"],
            max_frames=pre_cfg["max_frames"],
            n_workers=pre_cfg["n_workers"],
        )
```

- [x] **Step 4: `dashboard/pipeline.py`**

Line 26:

```python
from collab_splats.preproc import extract_frame, load_video_quality, sample_fps, sample_optical_flow, sample_uniform
```

`_sample` gains an output directory so it can place the report, and switches on the method:

```python
def _sample(video_path: Path, config: RunConfig, out_dir: Path, op_log: OperationLog):
    """
    Sample frames per the configured method; return (frames, records).
    """

    # Live label shows images processed / total; log=False so per-frame pings don't flood the log.
    # Throttle to ~100 writes total (every 1% of frames) — the UI polls at 300ms regardless.
    def on_progress(done: int, total: int) -> None:
        step = max(1, total // 100)
        if done % step and done != total:
            return
        op_log.update_progress(
            int(5 + 15 * done / max(total, 1)),
            f"sampling: frame {done}/{total}",
            log=False,
        )

    # Measure first, select second. Reused by existence, so a re-run is free.
    op_log.update_progress(4, "sampling: measuring video quality")
    report = load_video_quality(video_path, out_dir / "video_quality_report.json")

    op_log.update_progress(5, f"sampling: {config.sampling_method}")
    method = config.sampling_method

    if method == "fps":
        return sample_fps(
            str(video_path),
            fps=config.fps,
            max_frames=config.max_frames,
            report=report,
            on_progress=on_progress,
        )

    if method == "optical_flow":
        return sample_optical_flow(
            str(video_path),
            min_disparity=config.min_disparity,
            max_frames=config.max_frames,
            report=report,
            on_progress=on_progress,
        )

    return sample_uniform(
        str(video_path),
        max_frames=config.max_frames,
        report=report,
        on_progress=on_progress,
    )
```

Line 422 — the caller already has `out_dir` in scope:

```python
            frames, records = _sample(Path(video_path), config, out_dir, op_log)
```

- [x] **Step 5: `wrapper/splatter.py`**

Line 296 — move the import to the top of the module per the repo's import rule (`preproc` is a light import; it pulls no torch):

```python
from collab_splats.preproc import load_video_quality, sample_optical_flow
```

Line 311:

```python
            report = load_video_quality(file_path, tmp_dir / "video_quality_report.json")

            sampled_frames, _ = sample_optical_flow(
                file_path.as_posix(), max_frames=min(n_samples, 200), report=report
            )
```

- [x] **Step 6: `evals/datasets.py` — the bug fix**

`_load_video` currently calls `sample_frames(..., method="uniform", fps=fps)`, and `uniform` rejects `fps`, so it raises `ValueError` on every call. Drop the dead parameter:

```python
def _load_video(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    """
    Sample max_frames evenly over a video, writing to a sidecar _frames/ dir.
    """
    # Output frames into <stem>_frames/ sibling directory; created if absent
    frames_dir = seq_dir.parent / (seq_dir.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Measure once, select once; the report is reused across re-runs of the eval
    report = load_video_quality(seq_dir, frames_dir / "video_quality_report.json")

    # Decode video once: sample_uniform returns in-memory frames + records
    frames, records = sample_uniform(str(seq_dir), max_frames=max_frames, report=report)

    store = FrameStore.create(
        frames_dir / "frames.zarr",
        frames,
        records,
        provenance={"video_path": str(seq_dir), "method": "uniform", "max_frames": max_frames},
    )

    # Write JPEGs ONCE from the in-memory frames, no re-decode
    images = store.export(frames_dir)

    # GT poses not available for raw video; zeros placeholder
    return EvalDataset(images=images, gt_poses=np.zeros((len(images), 4, 4), dtype=np.float32))
```

Update the import at the top of `evals/datasets.py` from `sample_frames` to `load_video_quality, sample_uniform`, then check whether any caller passes `fps=` to `_load_video`:

```bash
rtk proxy grep -rn "_load_video" evals/ | grep -v results
```

- [x] **Step 7: Update the test doubles**

`tests/wrapper/test_reconstructor.py:145-150` patches `R.sample_frames`; it must now patch the sampler the config selects, and stub the report:

```python
    def fake_sample_uniform(path, **kwargs):
        return [np.zeros((4, 4, 3), np.uint8)], [{"frame_idx": 0, "blur_score": 1.0}]

    monkeypatch.setattr(R, "sample_uniform", fake_sample_uniform)
    monkeypatch.setattr(R, "load_video_quality", lambda *a, **k: {"available": True, "frames": {}})
```

Read lines 140-190 before editing — line 186 asserts on a list of patched names and must gain the new ones. Apply the same two-patch shape at `tests/dashboard/test_pipeline.py:62` and `:108` (`pl.sample_uniform` / `pl.load_video_quality`, matching each test's configured method) and at `tests/evals/test_datasets.py:509` (`datasets.sample_uniform`, `datasets.load_video_quality`).

- [x] **Step 8: Run the suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [x] **Step 9: Dashboard smoke gate — mandatory before committing a dashboard change**

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: prints `SMOKE PASS`.

- [x] **Step 10: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/wrapper/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/wrapper/splatter.py evals/datasets.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py tests/dashboard/test_pipeline.py tests/evals/test_datasets.py
git commit --only collab_splats/wrapper/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/wrapper/splatter.py evals/datasets.py tests/wrapper/test_reconstructor.py tests/wrapper/test_reconstructor_preprocess.py tests/dashboard/test_pipeline.py tests/evals/test_datasets.py \
  -m "refactor(preproc): callers measure then select; fix evals._load_video raising unconditionally"
```

---

### Task 10: Delete the dispatcher and the dead helpers

**Files:**
- Modify: `collab_splats/preproc/sampling.py`
- Modify: `collab_splats/preproc/__init__.py`, `collab_splats/preproc/viz.py`
- Test: `tests/preproc/test_sampling.py`, `tests/preproc/test_sampling_parity.py`, `tests/preproc/test_viz.py`

- [x] **Step 1: Re-point the parity test at the new samplers**

This is the moment the parity claim is settled. The baseline JSON is unchanged; only the calls move. Replace the body of `tests/preproc/test_sampling_parity.py` below the fixtures:

```python
@pytest.fixture(scope="module")
def report():
    from collab_splats.preproc.qa import compute_video_quality

    return compute_video_quality(str(VIDEO), workers=4)


def test_uniform_matches_baseline(baseline, report):
    from collab_splats.preproc.sampling import sample_uniform

    _, records = sample_uniform(str(VIDEO), max_frames=30, report=report)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["uniform_max30"]


def test_fps_matches_baseline(baseline, report):
    from collab_splats.preproc.sampling import sample_fps

    _, records = sample_fps(str(VIDEO), fps=0.5, report=report)

    assert [int(r["frame_idx"]) for r in records] == baseline["cases"]["fps_0.5"]
```

- [x] **Step 2: Run it and read the result honestly**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling_parity.py -q
```

Expected: `2 passed`.

**If it fails, do not relax the test.** Diff the two index lists and check each differing frame against the report: `laplacian` is bit-identical across the rewrite, so a diff can only come from the exposure row moving from the 480-wide analysis gray to the native gray (design doc §3.3). Confirm that by checking whether the differing frames sit near `exposure_std == 10.0`. If they do, update the baseline JSON **and record the change in the design doc's §3.3 table** as the measured outcome. If they do not, it is a real regression in `_take_frames` — most likely the tie-break, which must be first-maximal.

- [x] **Step 3: Delete**

From `sampling.py`, remove: `_combine_scores`, `_progress_reporter`, `_uniform_targets`, `_fps_targets`, `sample_frames`, `_sample_positions`, `_sample_uniform`, `_sample_fps`, `_iter_scored_frames`, `_sample_optical_flow`, `score_frames`, and the constants `_VALID_PROBE_MAX`, `_SELECT_THRESHOLD`, `_ROTATION_THRESHOLD_DEG`, `_LK_PARAMS`, `_FEATURE_PARAMS`. Drop the now-unused `Iterator` and `tqdm` imports, and the `check_frame_quality` / `compute_blur_score` / `_DEFAULT_BLUR_THRESHOLD` names from the `qa` import — only `_analysis_gray` remains.

The module docstring becomes:

```python
"""
Keyframe selection: which frames to keep, and the three methods that pick them.

sample_fps, sample_uniform and sample_optical_flow each take a quality report
from preproc.qa and filter it through filter_frame_quality — the one place a
threshold meets the report. qa measures; this module decides. Decoding and
video metadata live in preproc.video.
"""
```

Verify nothing else references the deleted names:

```bash
rtk proxy grep -rn "sample_frames\|score_frames\|_combine_scores\|_progress_reporter\|_sample_positions" \
  collab_splats/ evals/ tests/ docs/source/ --include=*.py --include=*.ipynb
```

The only surviving hits should be in `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`, which is out of scope (Task 14 records it).

- [x] **Step 4: `preproc/__init__.py`**

```python
"""
Video preprocessing: decode (video), capture quality (qa), frame selection (sampling).

Two steps: qa.compute_video_quality measures the whole video into a report, then
the samplers select from it. Plots live in collab_splats.preproc.viz and are
deliberately not re-exported (keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import _analysis_gray as analysis_gray
from collab_splats.preproc.qa import compute_video_quality, load_video_quality
from collab_splats.preproc.sampling import (
    filter_frame_quality,
    sample_fps,
    sample_optical_flow,
    sample_uniform,
)
from collab_splats.preproc.video import extract_frame, get_video_info, iter_frames

__all__ = [
    "FrameStore",
    "analysis_gray",
    "compute_video_quality",
    "extract_frame",
    "filter_frame_quality",
    "get_video_info",
    "iter_frames",
    "load_video_quality",
    "sample_fps",
    "sample_optical_flow",
    "sample_uniform",
]
```

The `_analysis_gray as analysis_gray` alias keeps this task's commit green on its own; Task 11 renames the function itself and drops the alias.

- [x] **Step 5: `viz.py`**

Delete the `_combine_scores` import (line 14). `plot_disparity_sensitivity` re-scores through the selector instead:

```python
def plot_disparity_sensitivity(frame_scores: list, disparity_values: list) -> None:
    """
    Approximate selected-frame count vs the min_disparity threshold.

    Re-thresholds precomputed records through the selector's own formula, so the
    plot cannot drift from selection. Approximate: it ignores the stateful
    keyframe updates a true re-run would perform.
    """
    from collab_splats.preproc.sampling import OpticalFlowFrameSelector

    counts = []
    for threshold in disparity_values:
        selector = OpticalFlowFrameSelector(min_disparity=threshold)
        counts.append(
            sum(1 for d in frame_scores if selector.combine(d["disparity"], d["histogram_similarity"]) >= 0.5)
        )
```

Keep the plotting body below unchanged.

`plot_frame_scores` still works unchanged on `sample_optical_flow`'s records (they carry `disparity`, `rotation`, `histogram_similarity`, `selected`) — change its docstring line "Takes score_frames() records" to "Takes sample_optical_flow() records" and reformat it to the new docstring shape.

`plot_quality_examples` categorised on `reject_reason`, which no longer exists. It now takes a report and derives the categories from `filter_frame_quality`:

```python
def plot_quality_examples(store: FrameStore, report: dict, n_examples: int = 4) -> None:
    """
    Example frames per quality-filter outcome: usable / soft / badly exposed.

    Takes a qa.compute_video_quality report and reads only the displayed frames
    from frames.zarr (no re-decode). Empty categories are dropped from the grid;
    counts still show in the title.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    f = report["frames"]
    lap = np.asarray(f["laplacian"], float)
    mean = np.asarray(f["exposure_mean"], float)
    std = np.asarray(f["exposure_std"], float)
    usable = filter_frame_quality(report)

    def rows_for(mask):
        return [
            {"frame_idx": i, "blur_score": lap[i], "exposure_mean": mean[i]}
            for i in np.flatnonzero(mask)
            if store.has_frame_idx(i)
        ]

    # Reason per rejected frame: sharpness is checked first, matching the filter
    soft = ~usable & (lap < 50.0)

    categories = [
        ("Usable", rows_for(usable)),
        ("Rejected: soft", rows_for(soft)),
        ("Rejected: exposure", rows_for(~usable & ~soft)),
    ]
```

Keep the existing body from the `counts = " · ".join(...)` line down, unchanged. `has_frame_idx` is added in Task 12 — **run Task 12 before this line executes, or add the method as part of this task**; the plan orders Task 12 after, so add the three-line method now and let Task 12 be a no-op on it.

- [x] **Step 6: Update the export-list test**

`tests/preproc/test_sampling.py:365-390` asserts on `collab_splats.preproc.__all__`. Replace the expected list with the one from Step 4 and rewrite the trailing comment to explain the current surface, not the old one.

- [x] **Step 7: Run everything**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: all pass, `SMOKE PASS`. Delete any test in `tests/preproc/test_sampling.py` whose subject was deleted in Step 3 — `test_combine_scores_monotonic_in_disparity`, every `test_sample_frames_*` knob-rejection test, every `test_score_frames_*`. The knob-rejection tests are obsolete by construction: three separate signatures cannot be passed another method's knob.

- [x] **Step 8: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py collab_splats/preproc/viz.py collab_splats/preproc/frame_store.py tests/preproc/test_sampling.py tests/preproc/test_sampling_parity.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py collab_splats/preproc/viz.py collab_splats/preproc/frame_store.py tests/preproc/test_sampling.py tests/preproc/test_sampling_parity.py tests/preproc/test_viz.py \
  -m "refactor(preproc): delete sample_frames dispatcher, score_frames and five dead helpers"
```

---

### Task 11: `qa.py` — delete the gate, publish `analysis_gray`, cut the prose

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `collab_splats/preproc/sampling.py` (import), `collab_splats/preproc/__init__.py` (drop the alias)
- Test: `tests/preproc/test_qa.py`

- [x] **Step 1: Delete the gate and its constants**

Remove from `qa.py`: `check_frame_quality`, `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD`, `compute_blur_score`, and the "Frame quality gate" section divider. `compute_blur_score` was one `cv2.Laplacian` line; fold it into `compute_blur`'s return:

```python
    # Laplacian variance: unbounded, higher = sharper. Runs in the opposite
    # direction to `blur` on purpose — see the docstring.
    return {"blur": perceptual, "laplacian": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
```

- [x] **Step 2: Publish `analysis_gray` and de-constant it**

```python
def analysis_gray(frame_bgr: np.ndarray, *, width: int = 480) -> np.ndarray:
    """
    Grayscale copy downscaled to `width` for scoring.

    - Stays cv2: this is a resize and a colour convert on a numpy BGR frame.
      Routing it through kornia would mean a torch tensor, a layout change and a
      device transfer to replace two cv2 calls.
    - Bounds LK flow and Laplacian cost regardless of source resolution.
    """
    scale = min(1.0, width / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr

    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
```

Delete `_ANALYSIS_WIDTH`. Sweep the renames:

```bash
rtk proxy grep -rn "_analysis_gray\|compute_blur_score\|check_frame_quality\|_ANALYSIS_WIDTH" collab_splats/ tests/ evals/ --include=*.py
```

Every hit becomes `analysis_gray`, the folded Laplacian, or a deletion. In `preproc/__init__.py`, collapse the two `qa` import lines back into one plain import.

- [x] **Step 3: Add the measurement-family header and trim the docstrings**

Replace `qa.py`'s module docstring:

```python
"""
Video capture quality: how good is the source footage, per frame and per pair.

Two measurement families:

    Photometry — per frame: is this frame sharp and correctly exposed?
                 blur, laplacian, exposure_{mean,median,std}, clipped_{low,high}_frac
    Motion     — per pair:  how far did the camera move between two frames?
                 n_matches, translation_px, parallax

REPORT-ONLY. Nothing here selects, rejects, ranks or scores a frame against a
threshold. Selection policy lives in preproc.sampling.filter_frame_quality.
Measured evidence for every column is in
docs/superpowers/specs/2026-08-20-video-quality-report-measured.md.
"""
```

Then cut each `compute_*` docstring to a one-line summary plus bullets, in the plan's docstring shape. The prose currently carrying measured numbers — `compute_blur`'s saturation paragraph, `compute_frame_quality`'s 0.2128-vs-0.2772 comparison and its 300.4 ms/39.4 ms timing, `match_descriptors`' 373-matches-on-noise result, `compute_translation`'s 2.37/3.28 ratios, `compute_parallax`'s Spearman 0.708 — moves to the measured report, each leaving a one-line pointer. Two worked examples:

```python
def compute_blur(gray: np.ndarray, *, h_size: int = 11) -> dict:
    """
    Blur two ways: Crete-Roffet perceptual blur and Laplacian variance.

    - `blur` is [0, 1], higher = blurrier. `laplacian` is unbounded, higher =
      sharper. Opposite directions on purpose.
    - `blur` SATURATES at 1.0 on any low-detail frame, so a high blur beside a
      high laplacian means detail is sparse, not that the frame is soft. That is
      why laplacian ships next to it, not instead.
    - h_size: width of the re-blur kernel Crete-Roffet compares against. Larger
      reports less blur; 11 is skimage's default.
    - Measured comparability caveats: see the 2026-08-20 measured report.
    """
```

```python
def compute_frame_quality(bgr: np.ndarray, *, analysis_width: int = 480, blur_h_size: int = 11) -> dict:
    """
    Photometry for one BGR frame: blur and exposure together.

    - Exposure reads the NATIVE gray. Downscaling averages scattered saturated
      pixels out of existence, so clipping fractions from a resized frame read
      0.0 no matter how blown out the capture was.
    - Blur reads the `analysis_width` gray, which buys ~7.6x throughput and pays
      for it in cross-video comparability — the blur column is not comparable
      across videos whose source width straddles 480. laplacian and exposure are
      unaffected.
    """
```

`compute_frame_quality`'s two new parameters pass straight through to `analysis_gray` and `compute_blur`. `_measure_range` and `compute_video_quality` forward them unchanged — defaults only, no new config keys.

- [x] **Step 4: Update the tests**

Delete every `test_check_frame_quality_*` (lines 33-68) and `test_compute_blur_score_sharp_exceeds_blurred` (line 27) from `tests/preproc/test_qa.py` — their subjects are gone. Line 102's `compute_blur(...)["laplacian"] == compute_blur_score(...)` becomes a direct assertion:

```python
def test_compute_blur_laplacian_is_the_variance_of_the_laplacian(noise_gray):
    import cv2

    assert compute_blur(noise_gray)["laplacian"] == float(cv2.Laplacian(noise_gray, cv2.CV_64F).var())
```

Rename `_analysis_gray` to `analysis_gray` throughout the file (lines 9, 215-224, 319).

- [x] **Step 5: Run and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/qa.py collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py tests/preproc/test_qa.py \
  -m "refactor(preproc): qa is measurement only — gate deleted, analysis_gray public, docstrings cut"
```

---

### Task 12: `frame_store.py`

**Files:**
- Modify: `collab_splats/preproc/frame_store.py`
- Test: `tests/preproc/test_frame_store.py:43-49`, `tests/wrapper/test_reconstructor.py:200`

- [x] **Step 1: Write the failing test**

```python
def test_has_frame_idx(tmp_path):
    """
    Public membership check — viz needs it, and it replaces a private reach-in.
    """
    store = FrameStore.create(
        tmp_path / "frames.zarr",
        [np.zeros((4, 4, 3), np.uint8)],
        [{"frame_idx": 7, "blur_score": 1.0}],
        provenance={"video_path": "v.mp4"},
    )

    assert store.has_frame_idx(7)
    assert not store.has_frame_idx(8)


def test_is_stale_is_gone(tmp_path):
    """
    Reuse is by existence; a staleness check was unreachable and is deleted.
    """
    store = FrameStore.create(
        tmp_path / "frames.zarr",
        [np.zeros((4, 4, 3), np.uint8)],
        [{"frame_idx": 0}],
        provenance={"video_path": "v.mp4"},
    )

    assert not hasattr(store, "is_stale")
    assert not hasattr(store, "records")
```

- [x] **Step 2: Run to confirm failure**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frame_store.py -k "has_frame_idx or is_gone" -q
```

Expected: FAIL — `is_stale` and `records` still present. (`has_frame_idx` may already exist from Task 10 Step 5.)

- [x] **Step 3: Edit `frame_store.py`**

Delete lines 21-22 (`_STALENESS_KEYS` and its comment), `records()` (89-91) and `is_stale()` (97-100). Add after `image_by_frame_idx`, if Task 10 did not already:

```python
    def has_frame_idx(self, frame_idx: int) -> bool:
        """
        True if that SOURCE video index is among the selected frames.
        """
        return int(frame_idx) in self._idx_to_row
```

Put a blank line between every logical block in `create` — the columnar-records loop, the attrs writes and the reopen are currently one undifferentiated run:

```python
        path = Path(path)
        imgs = np.stack(frames).astype(np.uint8)  # (N, H, W, 3) RGB
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(path), mode="w")

        # Chunk one frame per chunk so a consumer reads a single keyframe alone
        store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]), compressors=[lz4])

        # Columnar records: every key present on any record becomes an array
        keys = sorted({k for r in records for k in r})
        for k in keys:
            col = np.array([r.get(k, np.nan) for r in records])
            store.create_array(k, data=col, chunks=col.shape, compressors=[lz4])

        # Provenance is descriptive only — reuse is by existence, never by comparison
        store.attrs["record_keys"] = keys
        store.attrs["provenance"] = {k: provenance.get(k) for k in provenance}
        store.attrs["schema_version"] = 1

        return cls(path, zarr.open(str(path), mode="r"))
```

Reformat every docstring in the file to the plan's shape, and update the module docstring's last line if it still promises a staleness contract.

- [x] **Step 4: Delete the orphaned tests**

Delete `tests/preproc/test_frame_store.py::test_is_stale` (43-49) and the `store.is_stale(...)` assertion at `tests/wrapper/test_reconstructor.py:200` — read 190-205 first; if the surrounding test exists only to exercise staleness, delete the whole test, and if it also asserts on provenance contents, keep those lines.

- [x] **Step 5: Run and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m black --line-length 120 collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py tests/wrapper/test_reconstructor.py
git commit --only collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py tests/wrapper/test_reconstructor.py \
  -m "refactor(preproc): drop FrameStore.is_stale/records, add has_frame_idx, space the blocks"
```

---

### Task 13: Measure the real thing

Every timing in the design doc after `330 s` came from a standalone harness, not from `compute_video_quality`. The spec says the definitive number is a run of the real function after implementation. Take it.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-22-preproc-cleanup-design.md` (§5.4)

- [x] **Step 1: Run serial and 4-worker on the tutorial video**

Run in tmux, not inline — this takes minutes and CLAUDE.md forbids parallel work during heavy runs.

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import logging, time
logging.basicConfig(level=logging.INFO)
from collab_splats.preproc.qa import compute_video_quality

VID = "data/tutorial/tutorial_example-video.mp4"
for workers in (1, 4):
    t = time.perf_counter()
    r = compute_video_quality(VID, workers=workers)
    print(f"workers={workers}: {time.perf_counter() - t:.1f}s  "
          f"{len(r['frames']['frame_idx'])} frames, {len(r['pairs']['frame_idx_a'])} pairs")
PY
```

- [x] **Step 2: Confirm the two reports agree**

The worker-invariance test covers a tiny fixture; this covers the real video, where input seek actually has to land correctly.

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
from collab_splats.preproc.qa import compute_video_quality
VID = "data/tutorial/tutorial_example-video.mp4"
a = compute_video_quality(VID, workers=1)
b = compute_video_quality(VID, workers=4)
assert a["frames"] == b["frames"], "photometry differs between worker counts"
assert a["pairs"] == b["pairs"], "motion differs between worker counts"
print("identical:", len(a["frames"]["frame_idx"]), "frames,", len(a["pairs"]["frame_idx_a"]), "pairs")
PY
```

Expected: `identical: 2388 frames, 2364 pairs`. **A failure here is the boundary guard doing its job** — report it rather than working around it.

- [x] **Step 3: Time the end-to-end preproc stage**

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import shutil, time
from pathlib import Path
from collab_splats.wrapper.reconstructor import _extract_frames

out = Path("/tmp/claude-0/preproc-timing"); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
t = time.perf_counter()
n = _extract_frames(Path("data/tutorial/tutorial_example-video.mp4"), out / "frames.zarr",
                    "fps", 1.0, None, 300, n_workers=4)
first = time.perf_counter() - t
t = time.perf_counter()
_extract_frames(Path("data/tutorial/tutorial_example-video.mp4"), out / "frames.zarr",
                "fps", 1.0, None, 300, n_workers=4)
print(f"{n} frames — first run {first:.1f}s, re-run {time.perf_counter() - t:.1f}s")
PY
```

The re-run number should be dominated by the sampler alone, since `load_video_quality` reuses the report. (`_extract_frames` rewrites `frames.zarr`; `Reconstructor.preprocess` is what short-circuits on existence.)

- [x] **Step 4: Replace §5.4 of the design doc with the measured table**

Rewrite the section with the real numbers, keeping the harness figures as the prediction they were and stating the delta. Replace the paragraph beginning "All post-`330 s` figures come from a standalone harness" — it has served its purpose and is now superseded by measurement.

- [x] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-22-preproc-cleanup-design.md
git commit --only docs/superpowers/specs/2026-08-22-preproc-cleanup-design.md \
  -m "docs(specs): preproc cleanup — measured timings from the real compute_video_quality"
```

---

### Task 14: Docs and the final gate

**Files:**
- Modify: `CLAUDE.md`, `configs/README.md`, `docs/known-test-failures.md`
- Modify: `docs/superpowers/plans/2026-08-22-preproc-cleanup.md` (tick the boxes)

- [x] **Step 1: `CLAUDE.md`**

Line 87's architecture entry no longer describes the module:

```
  preproc/                 # video preprocessing: measure (qa) then select (sampling)
    video.py               # ffmpeg/ffprobe decode: get_video_info, iter_frames, extract_frame
    qa.py                  # report-only capture quality: compute_video_quality, load_video_quality
    sampling.py            # sample_fps | sample_uniform | sample_optical_flow + filter_frame_quality
    frame_store.py         # frames.zarr: the decode-once keyframe store
```

The Code Style section's docstring bullet gains the format this cleanup enforces:

```markdown
- **Docstrings:** every public function and class gets a one-line summary docstring. The
  `"""` open and close on their own lines — summary starts on the line after the opening
  quotes, never on the same line. Multi-line docstrings put a blank line between the summary
  and the bullets that follow. Bullets, not prose blocks. No restating the function name.
- **Blank lines between blocks:** a run of statements that does one thing is separated from
  the next run, and every block comment gets a blank line above it. Walls of undifferentiated
  code are not human-readable.
```

Add a "Recently completed" entry at the top of the list, following the existing entries' shape: what changed, the config rename and its loud failure, the measured numbers from Task 13, and the traps — the thread pin being load-bearing (unpinned workers measured 0.67×, slower than serial), the exposure-resolution parity row, and the `float64`/two-order-statistic corrections that make histogram exposure equal to numpy.

- [x] **Step 2: `configs/README.md`**

Add the two-step flow to the preproc section: `video_quality_report.json` is written beside `frames.zarr`, is reused by existence, and carries no verdicts. Note that `preproc.n_workers` affects only the report.

- [x] **Step 3: `docs/known-test-failures.md`**

The tutorial notebook entry at line 87 already records `keyframe_extraction.ipynb` as broken. Extend it: it now also uses `score_frames` and the deleted quality gate, and its rebuild is a separate pass (design doc §7).

- [x] **Step 4: Final gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
graphify update .
```

Expected: full suite green, `SMOKE PASS`. Compare the pass count against the pre-Task-1 baseline and account for every difference: deleted tests are expected, an unexplained drop is not.

- [x] **Step 5: Commit**

```bash
git add -f docs/superpowers/plans/2026-08-22-preproc-cleanup.md
git commit --only CLAUDE.md configs/README.md docs/known-test-failures.md docs/superpowers/plans/2026-08-22-preproc-cleanup.md graphify-out \
  -m "docs(preproc): record the measure-then-select cleanup and its traps"
```

---

## Owed after this plan

Recorded so they are not mistaken for done:

- **Tutorial rebuild** — `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` uses `score_frames` and the deleted gate. Rebuilt in a separate pass where it can demonstrate the report properly rather than being patched to compile (design doc §7).
- **BFMatcher `nfeatures` tuning** — after the ORB cache, `crossCheck` is ~20 ms of the ~32 ms pair cost, and `nfeatures=1000` under crossCheck is ~10⁶ Hamming comparisons per pair. Lowering it is the next real win, but it changes the values the report publishes, which a cleanup pass must not do (design doc §5.3).
- **A second video for the parity baseline** — the current baseline is one tutorial video whose frames sit nowhere near the `exposure_std >= 10` boundary, so it cannot exercise the one row where the rewrite is not bit-identical (design doc §3.3).
