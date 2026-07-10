# Preproc Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `utils/frame_sampling.py` (869 lines, 3 decode backends) with a top-level `collab_splats/preproc/` package: ffmpeg-only decode, blur+exposure quality gate, unified `sample_frames` dispatcher, matplotlib isolated in `viz.py`.

**Architecture:** Two real modules behind a re-exporting `__init__.py`. `sampling.py` owns ffprobe metadata, one streaming ffmpeg rawvideo decode path, the quality gate, `OpticalFlowFrameSelector`, the `sample_frames`/`score_frames` public functions, and frame I/O. `viz.py` owns the 4 matplotlib plots and is deliberately NOT re-exported (keeps matplotlib out of the pipeline import path). Old module + `semantics/frame_sampling.py` re-export are hard-deleted; 7 production call sites and 4 test files migrate.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), ffmpeg/ffprobe (subprocess), cv2 (image ops only), numpy, tqdm, matplotlib (viz only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-preproc-package-design.md`

**Conventions that apply to every task:**
- Test runner: `/opt/venv/reconstruction/bin/python -m pytest`
- Flat test functions, no test classes.
- Every public function: one-line docstring. Every logical block: a short what/why comment.
- Internal module-level functions carry a leading underscore. `OpticalFlowFrameSelector` is unprefixed but internal-by-omission from `__init__`.
- Commit after every task with conventional-commit messages.
- One deviation from the spec signature, agreed during planning: `fps` defaults to `None` (derive the window from `max_frames`, falling back to 2.0 fps) — this deletes the `target_fps = max_frames / duration` derivation currently duplicated in webapp, dashboard, and reconstructor.

---

### Task 1: Package skeleton, ffprobe metadata, ffmpeg streaming decode

**Files:**
- Create: `collab_splats/preproc/__init__.py` (placeholder, finalized in Task 7)
- Create: `collab_splats/preproc/sampling.py`
- Create: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Create package dirs and placeholder init**

```bash
mkdir -p /workspace/collab-splats/collab_splats/preproc /workspace/collab-splats/tests/preproc
```

Write `collab_splats/preproc/__init__.py` containing exactly:

```python
"""Video preprocessing: frame sampling, quality gating, frame I/O."""
```

- [ ] **Step 2: Write the failing tests**

Write `tests/preproc/test_sampling.py`:

```python
import numpy as np
import pytest
import cv2

from collab_splats.preproc.sampling import (
    _iter_frames,
    _iter_frames_at,
    _require_ffmpeg,
    get_video_info,
)


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """Synthesize a 60-frame 320x240 mp4: static noise texture + moving square.

    Noise gives LK flow corners to track; the moving square creates motion.
    """
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(0)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(60):
        frame = noise.copy()
        x = 10 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_get_video_info_keys(tiny_video):
    info = get_video_info(tiny_video)
    assert set(info) == {"total_frames", "fps", "duration_s", "width", "height"}


def test_get_video_info_values(tiny_video):
    info = get_video_info(tiny_video)
    assert info["total_frames"] == 60
    assert info["fps"] == pytest.approx(30.0)
    assert (info["width"], info["height"]) == (320, 240)
    assert info["duration_s"] == pytest.approx(2.0)


def test_get_video_info_missing_file():
    info = get_video_info("/nonexistent/video.mp4")
    assert info["total_frames"] == 0 and info["fps"] == 0.0


def test_require_ffmpeg_raises_without_binary(monkeypatch):
    # Simulate ffmpeg absent from PATH — the only decode backend must hard-fail
    monkeypatch.setattr("collab_splats.preproc.sampling.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        _require_ffmpeg()


def test_iter_frames_yields_all_frames_bgr(tiny_video):
    frames = list(_iter_frames(tiny_video))
    assert len(frames) == 60
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_iter_frames_at_yields_requested_indices(tiny_video):
    got = list(_iter_frames_at(tiny_video, [5, 20, 20, 3]))
    # Deduplicated, in stream order
    assert [idx for idx, _ in got] == [3, 5, 20]
    assert all(f.shape == (240, 320, 3) for _, f in got)


def test_iter_frames_at_empty_indices(tiny_video):
    assert list(_iter_frames_at(tiny_video, [])) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.preproc.sampling'`

- [ ] **Step 4: Write `sampling.py` (metadata + decode sections)**

Write `collab_splats/preproc/sampling.py`:

```python
"""Video preprocessing: metadata, frame decoding, quality gating, keyframe sampling.

All decoding goes through ffmpeg/ffprobe — the only supported backend. cv2 is
used for in-memory image operations only (grayscale, resize, Laplacian, LK
flow, JPEG write), never for decode.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


########################################################################
# Constants
########################################################################

# Analysis frames are downscaled to this width before scoring — bounds LK flow
# and Laplacian cost regardless of source resolution.
_ANALYSIS_WIDTH = 480

# Quality gate: Laplacian variance below this = blurred. Sharp indoor video
# sits well above 100; heavy motion blur drops below 50.
_DEFAULT_BLUR_THRESHOLD = 50.0
# Exposure bounds: mean outside this range = blown out; std below = no contrast.
_EXPOSURE_MEAN_RANGE = (20.0, 235.0)
_EXPOSURE_MIN_STD = 10.0

# Optical-flow selection: combined score at or above this selects the frame.
_SELECT_THRESHOLD = 0.5
# Rotation (degrees) that saturates the motion score.
_ROTATION_THRESHOLD_DEG = 5.0

# Lucas-Kanade sparse flow parameters.
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# Shi-Tomasi corner detection parameters for flow seed points.
_FEATURE_PARAMS = dict(maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=7)


########################################################################
# Video metadata / decoding (ffmpeg + ffprobe only)
########################################################################


def _require_ffmpeg() -> None:
    """Raise if ffmpeg/ffprobe are missing — the only supported decode backend."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH; install ffmpeg (e.g. `apt install ffmpeg`)"
        )


def _rotation_degrees(stream: dict) -> int:
    """CW display rotation from an ffprobe stream dict.

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


def get_video_info(video_path: str) -> dict:
    """Return video metadata via ffprobe.

    Keys: total_frames (int), fps (float), duration_s (float), width (int),
    height (int). Width/height are display dims (rotation applied), matching
    the frames the decode functions yield. All zeros if the file can't be probed.
    """
    _require_ffmpeg()
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    # -count_packets gives a reliable frame count when nb_frames is absent
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
             "-count_packets", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
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
        # Report display dims: ffmpeg auto-rotates output, so 90/270 swaps W/H
        if _rotation_degrees(s) in (90, 270):
            width, height = height, width
        duration_s = total / fps if fps > 0 else 0.0
        return {"total_frames": total, "fps": fps, "duration_s": duration_s,
                "width": width, "height": height}
    return zeros


def _iter_frames(video_path: str) -> Iterator[np.ndarray]:
    """Yield every frame as BGR uint8 HWC via one ffmpeg rawvideo pipe.

    ffmpeg applies rotation metadata itself, so yielded dims always match
    get_video_info's display dims.
    """
    info = get_video_info(str(video_path))
    w, h = info["width"], info["height"]
    if w == 0 or h == 0:
        return
    cmd = ["ffmpeg", "-i", str(video_path), "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-an", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    try:
        # Read fixed-size frames until the pipe runs dry
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            yield np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _iter_frames_at(video_path: str, frame_indices: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for the requested indices via one streaming pass.

    Indices are deduplicated and yielded in stream order; the decode stops
    after the last requested index. Streaming beats per-index seeking: exact
    for every codec, one process, no approximate-seek issues.
    """
    wanted = set(frame_indices)
    if not wanted:
        return
    last = max(wanted)
    for idx, frame in enumerate(_iter_frames(video_path)):
        if idx in wanted:
            yield idx, frame
        if idx >= last:
            break
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: 8 PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/preproc tests/preproc
git commit -m "feat(preproc): package skeleton — ffprobe metadata + ffmpeg-only streaming decode"
```

---

### Task 2: Quality gate

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (append section)
- Modify: `tests/preproc/test_sampling.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/preproc/test_sampling.py`)

```python
from collab_splats.preproc.sampling import _check_frame_quality, compute_blur_score


def _sharp_gray():
    """High-frequency noise — very high Laplacian variance."""
    rng = np.random.default_rng(1)
    return (rng.random((240, 320)) * 255).astype(np.uint8)


def test_compute_blur_score_sharp_exceeds_blurred():
    sharp = _sharp_gray()
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    assert compute_blur_score(sharp) > compute_blur_score(blurred) * 10


def test_check_frame_quality_accepts_sharp_frame():
    assert _check_frame_quality(_sharp_gray()) is True


def test_check_frame_quality_rejects_blurred_frame():
    sharp = _sharp_gray()
    blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
    # Threshold between the two measured scores makes the test threshold-robust
    threshold = (compute_blur_score(sharp) + compute_blur_score(blurred)) / 2
    assert _check_frame_quality(blurred, blur_threshold=threshold) is False
    assert _check_frame_quality(sharp, blur_threshold=threshold) is True


def test_check_frame_quality_rejects_bad_exposure():
    # Near-black and near-white frames fail regardless of sharpness
    dark = np.zeros((240, 320), dtype=np.uint8)
    bright = np.full((240, 320), 255, dtype=np.uint8)
    assert _check_frame_quality(dark, blur_threshold=0.0) is False
    assert _check_frame_quality(bright, blur_threshold=0.0) is False


def test_check_frame_quality_uses_precomputed_blur_score():
    # Passing blur_score short-circuits the Laplacian recompute
    gray = _sharp_gray()
    assert _check_frame_quality(gray, blur_threshold=100.0, blur_score=50.0) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v -k quality or blur`
Expected: FAIL — `ImportError: cannot import name '_check_frame_quality'`

- [ ] **Step 3: Implement** (append to `sampling.py` after the decode section)

```python
########################################################################
# Frame quality
########################################################################


def compute_blur_score(gray: np.ndarray) -> float:
    """Sharpness as Laplacian variance — higher is sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _check_frame_quality(
    gray: np.ndarray,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    blur_score: float | None = None,
) -> bool:
    """True if the frame is usable: sharp enough and reasonably exposed.

    blur_score: pass a precomputed value to skip the Laplacian recompute.
    """
    # Reject motion blur / defocus
    if blur_score is None:
        blur_score = compute_blur_score(gray)
    if blur_score < blur_threshold:
        return False
    # Reject over/under-exposure and contrast-free frames
    lo, hi = _EXPOSURE_MEAN_RANGE
    return lo <= float(gray.mean()) <= hi and float(gray.std()) >= _EXPOSURE_MIN_STD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: 13 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): blur+exposure quality gate (compute_blur_score, _check_frame_quality)"
```

---

### Task 3: OpticalFlowFrameSelector + shared scoring math

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (append section)
- Modify: `tests/preproc/test_sampling.py` (append tests)

Simplifications vs the old class (all verified unused in production): no
`reset()`, no homography branch, no `return_components` flag, no
`max_features`/`rotation_threshold` ctor params (module constants now), no
`last_keyframe_hist` state (histograms were always recomputed from grays
anyway). `score_frame` takes a **grayscale** frame — callers convert once.

- [ ] **Step 1: Write the failing tests** (append)

```python
from collab_splats.preproc.sampling import OpticalFlowFrameSelector, _combine_scores


def test_selector_first_frame_scores_one():
    selector = OpticalFlowFrameSelector()
    score, components = selector.score_frame(_sharp_gray())
    assert score == 1.0
    assert components == {"disparity": 0.0, "rotation": 0.0, "histogram_similarity": 1.0}


def test_selector_scores_are_normalized():
    selector = OpticalFlowFrameSelector()
    rng = np.random.default_rng(2)
    for _ in range(5):
        gray = (rng.random((240, 320)) * 255).astype(np.uint8)
        score, _ = selector.score_frame(gray)
        assert 0.0 <= score <= 1.0


def test_selector_identical_frame_scores_low():
    selector = OpticalFlowFrameSelector()
    gray = _sharp_gray()
    selector.score_frame(gray)  # seeds keyframe
    score, components = selector.score_frame(gray)
    # No motion, near-identical histogram → low combined score
    assert score < 0.3
    assert components["disparity"] < 1.0


def test_selector_rejects_invalid_weights():
    with pytest.raises(ValueError):
        OpticalFlowFrameSelector(motion_weight=0.0, coverage_weight=0.0)
    with pytest.raises(ValueError):
        OpticalFlowFrameSelector(motion_weight=1.5)


def test_combine_scores_monotonic_in_disparity():
    lo = _combine_scores(10.0, 0.5, min_disparity=50.0)
    hi = _combine_scores(60.0, 0.5, min_disparity=50.0)
    assert hi > lo
    assert 0.0 <= lo <= hi <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v -k selector or combine`
Expected: FAIL — `ImportError: cannot import name 'OpticalFlowFrameSelector'`

- [ ] **Step 3: Implement** (append to `sampling.py`)

```python
########################################################################
# Optical-flow selector
########################################################################


def _combine_scores(
    disparity: float,
    histogram_similarity: float,
    min_disparity: float,
    rotation: float = 0.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
) -> float:
    """Weighted motion+coverage score in [0, 1] from raw per-frame signals.

    Single home for the selection formula — the selector and
    viz.plot_disparity_sensitivity both call this, so they can't drift apart.
    """
    # Motion: max of normalised translation and rotation components
    translation_score = min(disparity / max(min_disparity, 1e-6), 1.0)
    rotation_score = min(rotation / _ROTATION_THRESHOLD_DEG, 1.0)
    motion_score = max(translation_score, rotation_score)
    # Coverage: inverse histogram correlation vs the last keyframe
    coverage_score = 1.0 - histogram_similarity
    total = motion_weight + coverage_weight
    return (motion_weight * motion_score + coverage_weight * coverage_score) / total


class OpticalFlowFrameSelector:
    """Streaming keyframe selector: motion (LK flow + rotation) and coverage scoring.

    Holds the reference keyframe between calls — score each candidate frame
    with score_frame(); promote selected frames with accept_frame(). Raw
    signals accumulate in .stats for viz. Construct fresh per video.
    """

    def __init__(
        self,
        min_disparity: float = 50.0,
        motion_weight: float = 0.6,
        coverage_weight: float = 0.4,
    ):
        # Validate weights; normalisation happens in _combine_scores
        if not (0 <= motion_weight <= 1 and 0 <= coverage_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if motion_weight + coverage_weight == 0:
            raise ValueError("At least one weight must be > 0")
        self.min_disparity = min_disparity
        self.motion_weight = motion_weight
        self.coverage_weight = coverage_weight
        # Reference keyframe state, seeded on the first scored frame
        self.last_keyframe_gray: np.ndarray | None = None
        self.last_keyframe_pts: np.ndarray | None = None
        # Accumulated raw signals for viz / sensitivity analysis
        self.stats: dict[str, list] = {
            "disparities": [], "rotations": [], "histogram_similarities": [],
        }

    def score_frame(self, gray: np.ndarray) -> tuple[float, dict]:
        """Score a grayscale frame against the current keyframe.

        Returns (score in [0, 1], components dict with raw disparity /
        rotation / histogram_similarity). The first frame scores 1.0 and
        seeds the keyframe state.
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
        # Accumulate raw signals for downstream plots
        self.stats["disparities"].append(disparity)
        self.stats["rotations"].append(rotation)
        self.stats["histogram_similarities"].append(hist_similarity)
        score = _combine_scores(
            disparity, hist_similarity, self.min_disparity, rotation=rotation,
            motion_weight=self.motion_weight, coverage_weight=self.coverage_weight,
        )
        return score, {
            "disparity": disparity, "rotation": rotation,
            "histogram_similarity": hist_similarity,
        }

    def accept_frame(self, gray: np.ndarray) -> None:
        """Make the given grayscale frame the new reference keyframe."""
        self.last_keyframe_gray = gray.copy()
        self.last_keyframe_pts = cv2.goodFeaturesToTrack(gray, **_FEATURE_PARAMS)

    def _compute_flow(self, gray: np.ndarray):
        """LK flow from keyframe corners; (None, None) if under 10 inliers survive."""
        if self.last_keyframe_pts is None or len(self.last_keyframe_pts) == 0:
            return None, None
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_keyframe_gray, gray, self.last_keyframe_pts, None, **_LK_PARAMS
        )
        if curr_pts is None:
            return None, None
        good_prev = self.last_keyframe_pts[status == 1]
        good_curr = curr_pts[status == 1]
        # Too few inliers → tracking unreliable for motion estimation
        if len(good_prev) < 10:
            return None, None
        return good_prev, good_curr

    def _estimate_rotation(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> float:
        """Camera rotation angle (degrees) via RANSAC partial-affine fit."""
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
        """Histogram correlation vs the keyframe, clamped to [0, 1]."""
        h1 = cv2.calcHist([self.last_keyframe_gray], [0], None, [bins], [0, 256])
        h2 = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        h1 = cv2.normalize(h1, h1).flatten()
        h2 = cv2.normalize(h2, h2).flatten()
        return float(max(0.0, min(1.0, cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: 18 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): OpticalFlowFrameSelector + _combine_scores (dead knobs removed)"
```

---

### Task 4: `sample_frames` dispatcher, uniform + optical-flow methods, `score_frames`

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (append section)
- Modify: `tests/preproc/test_sampling.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append)

```python
from collab_splats.preproc.sampling import sample_frames, score_frames


@pytest.fixture(scope="module")
def blur_pattern_video(tmp_path_factory):
    """40-frame video where every even frame is heavily blurred.

    Lets tests verify sharpest-in-window picks odd (sharp) frames.
    """
    path = tmp_path_factory.mktemp("vid") / "blurry.mp4"
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(3)
    noise = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(40):
        frame = noise.copy()
        cv2.rectangle(frame, (10 + i * 4, 60), (70 + i * 4, 140), (0, 255, 0), -1)
        if i % 2 == 0:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)
        writer.write(frame)
    writer.release()
    return str(path)


def test_sample_frames_unknown_method_raises(tiny_video):
    with pytest.raises(ValueError, match="Unknown method"):
        sample_frames(tiny_video, method="dso")


def test_sample_frames_params_are_keyword_only(tiny_video):
    with pytest.raises(TypeError):
        sample_frames(tiny_video, "uniform")  # positional method must fail


def test_uniform_returns_frames_and_records(tiny_video):
    frames, records = sample_frames(tiny_video, method="uniform", fps=10.0)
    # 60 frames @30fps sampled at 10fps → one per 3-frame window = 20
    assert len(frames) == len(records) == 20
    assert frames[0].shape == (240, 320, 3)
    assert set(records[0]) == {"frame_idx", "blur_score"}


def test_uniform_respects_max_frames(tiny_video):
    frames, records = sample_frames(tiny_video, method="uniform", fps=10.0, max_frames=5)
    assert len(frames) == len(records) == 5


def test_uniform_derives_window_from_max_frames(tiny_video):
    # fps omitted: 60 frames / max_frames 6 → interval 10 → 6 windows
    frames, _ = sample_frames(tiny_video, method="uniform", max_frames=6)
    assert len(frames) == 6


def test_uniform_picks_sharpest_in_window(blur_pattern_video):
    # Even frames blurred → the sharpest frame in every window is odd
    _, records = sample_frames(blur_pattern_video, method="uniform", fps=7.5, blur_threshold=0.0)
    assert len(records) > 0
    assert all(r["frame_idx"] % 2 == 1 for r in records)


def test_uniform_frames_are_rgb(tiny_video):
    frames, _ = sample_frames(tiny_video, method="uniform", fps=10.0)
    bgr = list(_iter_frames(tiny_video))
    # RGB return means channel order is reversed vs the BGR decode
    first_idx = sample_frames(tiny_video, method="uniform", fps=10.0)[1][0]["frame_idx"]
    np.testing.assert_array_equal(frames[0], bgr[first_idx][:, :, ::-1])


def test_optical_flow_first_frame_selected(tiny_video):
    frames, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=0.0)
    assert len(frames) >= 1
    assert records[0]["frame_idx"] == 0


def test_optical_flow_records_have_source_indices(tiny_video):
    _, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=0.0)
    idxs = [r["frame_idx"] for r in records]
    # Source video indices: strictly increasing, within range — NOT list positions
    assert idxs == sorted(idxs) and idxs[-1] < 60
    assert set(records[0]) == {
        "frame_idx", "blur_score", "score", "selected",
        "disparity", "rotation", "histogram_similarity",
    }


def test_optical_flow_respects_max_frames(tiny_video):
    frames, _ = sample_frames(tiny_video, method="optical_flow", max_frames=2, blur_threshold=0.0)
    assert len(frames) <= 2


def test_optical_flow_gate_rejects_all_blurred(tiny_video):
    # Threshold above any real Laplacian variance → every frame gated out
    frames, records = sample_frames(tiny_video, method="optical_flow", blur_threshold=1e12)
    assert frames == [] and records == []


def test_sample_frames_on_progress_called(tiny_video):
    calls = []
    sample_frames(tiny_video, method="uniform", fps=10.0,
                  on_progress=lambda done, total: calls.append((done, total)))
    assert calls and calls[-1][0] == 60 and calls[-1][1] == 60


def test_score_frames_one_record_per_frame(tiny_video):
    records = score_frames(tiny_video, blur_threshold=0.0)
    assert len(records) == 60
    assert set(records[0]) == {
        "frame_idx", "blur_score", "score", "selected",
        "disparity", "rotation", "histogram_similarity",
    }
    assert records[0]["selected"] is True  # first usable frame always selected
    assert all(0.0 <= r["score"] <= 1.0 for r in records)


def test_sample_frames_missing_file_returns_empty():
    frames, records = sample_frames("/nonexistent/video.mp4", method="uniform")
    assert frames == [] and records == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v -k "sample_frames or uniform or optical_flow or score_frames"`
Expected: FAIL — `ImportError: cannot import name 'sample_frames'`

- [ ] **Step 3: Implement** (append to `sampling.py`)

```python
########################################################################
# Sampling
########################################################################


def _analysis_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """Grayscale copy downscaled to _ANALYSIS_WIDTH for scoring."""
    scale = min(1.0, _ANALYSIS_WIDTH / frame_bgr.shape[1])
    small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale) if scale < 1.0 else frame_bgr
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _progress_reporter(
    total: int,
    desc: str,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[Callable[[int], None], Callable[[], None]]:
    """Unified progress: forward to on_progress when given, else an internal tqdm bar.

    Returns (report(done), close()) — the single progress mechanism for all loops.
    """
    if on_progress is not None:
        return (lambda done: on_progress(done, total)), (lambda: None)
    bar = tqdm(total=total or None, desc=desc, unit="frame")
    return (lambda _done: bar.update(1)), bar.close


def sample_frames(
    video_path: str,
    *,
    method: str = "uniform",
    max_frames: int | None = None,
    fps: float | None = None,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Select frames from a video for reconstruction.

    Methods:
        "uniform": one frame per fixed window — the sharpest usable frame in
            each. Window size comes from `fps` (samples/second), or is derived
            from `max_frames` when fps is None (falls back to 2.0 fps).
        "optical_flow": motion (LK disparity + rotation) + coverage (histogram
            diversity) scoring; frames scoring >= 0.5 are selected.
            Uses min_disparity / motion_weight / coverage_weight.

    Both methods apply the quality gate (blur + exposure); `blur_threshold`
    tunes it (0.0 disables the blur check).

    Returns (frames, records): RGB arrays and one dict per selected frame with
    at least frame_idx (SOURCE video index) and blur_score; optical_flow adds
    disparity, rotation, histogram_similarity, score, selected.
    """
    if method == "uniform":
        return _sample_uniform(
            video_path, fps=fps, max_frames=max_frames,
            blur_threshold=blur_threshold, on_progress=on_progress,
        )
    if method == "optical_flow":
        return _sample_optical_flow(
            video_path, max_frames=max_frames, min_disparity=min_disparity,
            motion_weight=motion_weight, coverage_weight=coverage_weight,
            blur_threshold=blur_threshold, on_progress=on_progress,
        )
    raise ValueError(f"Unknown method: {method!r} (expected 'uniform' or 'optical_flow')")


def _sample_uniform(
    video_path: str,
    *,
    fps: float | None,
    max_frames: int | None,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Uniform windows over the video; keep the sharpest usable frame per window."""
    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []
    # Window size: from fps if given, else spread max_frames over the video
    native_fps = info["fps"] or 30.0
    if fps is not None:
        interval = max(1, int(round(native_fps / fps)))
    elif max_frames:
        interval = max(1, total // max_frames)
    else:
        interval = max(1, int(round(native_fps / 2.0)))
    report, close = _progress_reporter(total, "Uniform sampling", on_progress)
    frames: list[np.ndarray] = []
    records: list[dict] = []
    best: tuple[float, int, np.ndarray] | None = None  # (blur, idx, frame)
    try:
        for idx, frame in enumerate(_iter_frames(video_path)):
            report(idx + 1)
            # Track the sharpest gate-passing frame within the current window
            gray = _analysis_gray(frame)
            blur = compute_blur_score(gray)
            usable = _check_frame_quality(gray, blur_threshold, blur_score=blur)
            if usable and (best is None or blur > best[0]):
                best = (blur, idx, frame)
            # Window boundary: flush the best frame and start the next window
            if (idx + 1) % interval == 0:
                if best is not None:
                    frames.append(cv2.cvtColor(best[2], cv2.COLOR_BGR2RGB))
                    records.append({"frame_idx": best[1], "blur_score": best[0]})
                best = None
                if max_frames is not None and len(frames) >= max_frames:
                    return frames, records
        # Final partial window
        if best is not None and (max_frames is None or len(frames) < max_frames):
            frames.append(cv2.cvtColor(best[2], cv2.COLOR_BGR2RGB))
            records.append({"frame_idx": best[1], "blur_score": best[0]})
    finally:
        close()
    return frames, records


def _iter_scored_frames(
    video_path: str,
    selector: OpticalFlowFrameSelector,
    *,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
    desc: str,
) -> Iterator[tuple[int, np.ndarray, bool, float, float, dict]]:
    """Yield (frame_idx, frame_bgr, selected, blur_score, score, components) per frame.

    Single scoring loop shared by _sample_optical_flow and score_frames.
    Gate-rejected frames yield selected=False with score 0.0 and never reach
    the selector. Selected frames (score >= _SELECT_THRESHOLD) become the
    selector's new reference keyframe.
    """
    info = get_video_info(str(video_path))
    report, close = _progress_reporter(info["total_frames"], desc, on_progress)
    try:
        for idx, frame in enumerate(_iter_frames(video_path)):
            report(idx + 1)
            gray = _analysis_gray(frame)
            blur = compute_blur_score(gray)
            # Quality gate first: unusable frames never reach the selector
            if not _check_frame_quality(gray, blur_threshold, blur_score=blur):
                yield idx, frame, False, blur, 0.0, {}
                continue
            score, components = selector.score_frame(gray)
            selected = score >= _SELECT_THRESHOLD
            if selected:
                selector.accept_frame(gray)
            yield idx, frame, selected, blur, score, components
    finally:
        close()


def _sample_optical_flow(
    video_path: str,
    *,
    max_frames: int | None,
    min_disparity: float,
    motion_weight: float,
    coverage_weight: float,
    blur_threshold: float,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Optical-flow keyframe selection; keeps selected frames full-res RGB."""
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    frames: list[np.ndarray] = []
    records: list[dict] = []
    for idx, frame, selected, blur, score, comp in _iter_scored_frames(
        video_path, selector, blur_threshold=blur_threshold,
        on_progress=on_progress, desc="Optical flow selection",
    ):
        if not selected:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # frame_idx is the SOURCE video index (fixes old positional-index bug)
        records.append({"frame_idx": idx, "blur_score": blur, "score": score,
                        "selected": True, **comp})
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames, records


def score_frames(
    video_path: str,
    *,
    min_disparity: float = 50.0,
    motion_weight: float = 0.6,
    coverage_weight: float = 0.4,
    blur_threshold: float = _DEFAULT_BLUR_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Score every frame without keeping pixel data — analysis/viz workflow.

    Returns one record per frame: frame_idx, blur_score, disparity, rotation,
    histogram_similarity, score, selected.
    """
    selector = OpticalFlowFrameSelector(
        min_disparity=min_disparity,
        motion_weight=motion_weight,
        coverage_weight=coverage_weight,
    )
    records: list[dict] = []
    for idx, _frame, selected, blur, score, comp in _iter_scored_frames(
        video_path, selector, blur_threshold=blur_threshold,
        on_progress=on_progress, desc="Scoring frames",
    ):
        records.append({
            "frame_idx": idx, "blur_score": blur, "score": score, "selected": selected,
            "disparity": comp.get("disparity", 0.0),
            "rotation": comp.get("rotation", 0.0),
            "histogram_similarity": comp.get("histogram_similarity", 1.0),
        })
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: 33 PASS. If `test_uniform_picks_sharpest_in_window` is flaky due to codec noise, raise the GaussianBlur kernel in the fixture to (51, 51) — do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): sample_frames dispatcher (uniform sharpest-in-window + optical_flow) and score_frames"
```

---

### Task 5: Frame I/O

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (append section)
- Modify: `tests/preproc/test_sampling.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append)

```python
from collab_splats.preproc.sampling import extract_frames, load_frames


def test_load_frames_returns_rgb_arrays(tiny_video):
    frames = load_frames(tiny_video, [0, 10, 30])
    assert len(frames) == 3
    bgr = list(_iter_frames(tiny_video))
    np.testing.assert_array_equal(frames[1], bgr[10][:, :, ::-1])


def test_extract_frames_writes_named_jpegs(tiny_video, tmp_path):
    paths = extract_frames(tiny_video, [4, 2, 4], tmp_path / "out")
    # Deduplicated, sorted, zero-padded names
    assert [p.name for p in paths] == ["frame_000002.jpg", "frame_000004.jpg"]
    assert all(p.exists() for p in paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v -k "load_frames or extract_frames"`
Expected: FAIL — `ImportError: cannot import name 'load_frames'`

- [ ] **Step 3: Implement** (append to `sampling.py`)

```python
########################################################################
# Frame I/O
########################################################################


def load_frames(video_path: str, frame_indices: list[int]) -> list[np.ndarray]:
    """Read specific frames by index; returns RGB arrays in index order."""
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for _, f in _iter_frames_at(video_path, frame_indices)]


def extract_frames(video_path: str, frame_indices: list[int], output_dir) -> list[Path]:
    """Save specific frames as frame_NNNNNN.jpg in output_dir; returns saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, frame in _iter_frames_at(video_path, frame_indices):
        out_path = output_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    return saved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v`
Expected: 35 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): load_frames/extract_frames via shared streaming decode"
```

---

### Task 6: viz.py

**Files:**
- Create: `collab_splats/preproc/viz.py`
- Create: `tests/preproc/test_viz.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/preproc/test_viz.py`:

```python
import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from collab_splats.preproc.viz import (
    plot_disparity_sensitivity,
    plot_frame_grid,
    plot_frame_scores,
    plot_selection,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fake_records(n=30):
    """Score records shaped like score_frames output."""
    rng = np.random.default_rng(0)
    return [
        {
            "frame_idx": i,
            "blur_score": 200.0,
            "disparity": float(rng.random() * 80),
            "rotation": float(rng.random() * 3),
            "histogram_similarity": float(rng.random()),
            "score": float(rng.random()),
            "selected": i % 5 == 0,
        }
        for i in range(n)
    ]


def test_plot_frame_grid_smoke():
    frames = [np.zeros((24, 32, 3), dtype=np.uint8)] * 4
    plot_frame_grid(frames, "grid")
    assert plt.gcf() is not None


def test_plot_selection_both_sets():
    plot_selection(100, fps_indices=[0, 10, 20], of_indices=[0, 5, 30])
    assert len(plt.gcf().axes) == 2


def test_plot_frame_scores_smoke():
    plot_frame_scores(_fake_records())
    assert len(plt.gcf().axes) == 3


def test_plot_frame_scores_empty_input():
    plot_frame_scores([])  # must not raise


def test_plot_disparity_sensitivity_monotonic():
    # Higher disparity threshold → same or fewer frames selected
    records = _fake_records(60)
    plot_disparity_sensitivity(records, [10.0, 50.0, 200.0])
    ax = plt.gcf().axes[0]
    counts = ax.lines[0].get_ydata()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.preproc.viz'`

- [ ] **Step 3: Implement**

Write `collab_splats/preproc/viz.py`:

```python
"""Matplotlib plots for frame sampling analysis — notebook use only.

Deliberately NOT re-exported from collab_splats.preproc.__init__ so pipeline
consumers never import matplotlib. Import explicitly:
`from collab_splats.preproc.viz import plot_frame_scores`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from collab_splats.preproc.sampling import _combine_scores


def plot_frame_grid(frames: list, title: str, n_cols: int = 6) -> None:
    """Display a grid of RGB frames."""
    n = len(frames)
    n_cols = min(n_cols, n)
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2), squeeze=False)
    axes = np.array(axes).flatten()
    # Fill the grid; blank any unused trailing cells
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(frames[i])
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_selection(
    total_frames: int,
    fps_indices: list | None = None,
    of_indices: list | None = None,
) -> None:
    """Vertical-line timeline of selected frame indices; two panels when both sets given."""
    sets = [
        (fps_indices, "Uniform", "steelblue"),
        (of_indices, "Optical Flow", "darkorange"),
    ]
    active = [(idx, label, color) for idx, label, color in sets if idx is not None]
    fig, axes = plt.subplots(len(active), 1, figsize=(12, 2 * len(active)), squeeze=False)
    for ax, (indices, label, color) in zip(axes[:, 0], active):
        if indices:
            ax.vlines(indices, 0, 1, colors=color, linewidth=1.5, alpha=0.8)
        ax.set_xlim(0, total_frames)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_xlabel("Frame index")
        ax.set_title(f"{label}  (n={len(indices) if indices else 0})", fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_frame_scores(frame_scores: list) -> None:
    """3-panel timeseries of per-frame signals: disparity / rotation / histogram similarity.

    Takes score_frames() records; selected frames marked with vertical grey lines.
    """
    if not frame_scores:
        plt.subplots(3, 1, figsize=(12, 6))
        plt.show()
        return
    idxs = [d["frame_idx"] for d in frame_scores]
    selected_idxs = [d["frame_idx"] for d in frame_scores if d["selected"]]
    panels = [
        ([d["disparity"] for d in frame_scores], "Disparity (px)", "steelblue"),
        ([d["rotation"] for d in frame_scores], "Rotation (deg)", "seagreen"),
        ([d["histogram_similarity"] for d in frame_scores], "Histogram similarity", "tomato"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for ax, (values, ylabel, color) in zip(axes, panels):
        ax.plot(idxs, values, color=color, linewidth=0.8)
        # Faint vertical line at each selected frame
        for x in selected_idxs:
            ax.axvline(x, color="gray", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Per-frame optical flow scores  (grey lines = selected frames)", fontsize=11)
    fig.tight_layout()
    plt.show()


def plot_disparity_sensitivity(frame_scores: list, disparity_values: list) -> None:
    """Approximate selected-frame count vs min_disparity threshold.

    Re-thresholds precomputed score_frames() records via the real scoring
    formula (_combine_scores) — no video re-decode, no formula drift.
    Approximate: ignores the stateful keyframe updates of a true re-run.
    """
    # Re-score each threshold from the recorded raw signals
    counts = []
    for threshold in disparity_values:
        n = sum(
            1
            for d in frame_scores
            if _combine_scores(d["disparity"], d["histogram_similarity"], threshold) >= 0.5
        )
        counts.append(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(disparity_values, counts, marker="o", color="steelblue", linewidth=1.5)
    ax.set_xlabel("min_disparity threshold (px)")
    ax.set_ylabel("Frames selected (approx.)")
    ax.set_title("Frame count vs disparity threshold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v`
Expected: 41 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit -m "feat(preproc): viz module — 4 plots, matplotlib isolated from pipeline imports"
```

---

### Task 7: Public API in `__init__.py`

**Files:**
- Modify: `collab_splats/preproc/__init__.py`
- Modify: `tests/preproc/test_sampling.py` (append test)

- [ ] **Step 1: Write the failing test** (append to `tests/preproc/test_sampling.py`)

```python
def test_public_api_surface():
    import collab_splats.preproc as preproc

    # Exactly the 6 public names — viz is opt-in and must NOT be re-exported
    assert set(preproc.__all__) == {
        "sample_frames", "score_frames", "get_video_info",
        "load_frames", "extract_frames", "compute_blur_score",
    }
    assert not hasattr(preproc, "plot_frame_scores")


def test_importing_preproc_does_not_import_matplotlib():
    # Fresh subprocess: pipeline-level import must not pull matplotlib
    import subprocess as sp
    import sys

    code = (
        "import sys; import collab_splats.preproc; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    result = sp.run([sys.executable, "-c", code])
    assert result.returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -v -k public_api`
Expected: FAIL — `AttributeError: module 'collab_splats.preproc' has no attribute '__all__'`

- [ ] **Step 3: Implement**

Overwrite `collab_splats/preproc/__init__.py`:

```python
"""Video preprocessing: frame sampling, quality gating, frame I/O.

Plots live in collab_splats.preproc.viz and are deliberately not re-exported
(keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.sampling import (
    compute_blur_score,
    extract_frames,
    get_video_info,
    load_frames,
    sample_frames,
    score_frames,
)

__all__ = [
    "sample_frames",
    "score_frames",
    "get_video_info",
    "load_frames",
    "extract_frames",
    "compute_blur_score",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v`
Expected: 43 PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/__init__.py tests/preproc/test_sampling.py
git commit -m "feat(preproc): public API — 6 names, viz excluded from pipeline imports"
```

---

### Task 8: Migrate webapp preprocess router

**Files:**
- Modify: `collab_splats/webapp/routers/preprocess.py:14` (import) and the `_extract_sse` `run()` body

- [ ] **Step 1: Replace the import** (line 14)

```python
# OLD
from collab_splats.utils.frame_sampling import get_video_info, sample_frames_fps, sample_frames_optical_flow
# NEW
from collab_splats.preproc import get_video_info, sample_frames
```

Note: `get_video_info` stays imported — the router's info endpoint uses it.

- [ ] **Step 2: Replace the sampling branch inside `run()` in `_extract_sse`**

```python
# OLD
if method == "optical_flow":
    frames, _ = sample_frames_optical_flow(
        str(s.video_path), max_frames=max_frames,
        min_disparity=min_disparity, on_progress=progress, verbose=False,
    )
else:
    # Balanced FPS: derive target fps from desired frame count and duration
    info = get_video_info(str(s.video_path))
    dur = info.get("duration_s") or (info["total_frames"] / (info.get("fps") or 30.0))
    target_fps = max_frames / max(dur, 1.0)
    frames, _ = sample_frames_fps(
        str(s.video_path), fps=target_fps, max_frames=max_frames,
        on_progress=progress, verbose=False,
    )

# NEW — dispatch and window derivation live in the library now
sampling_method = "optical_flow" if method == "optical_flow" else "uniform"
frames, _ = sample_frames(
    str(s.video_path), method=sampling_method, max_frames=max_frames,
    min_disparity=min_disparity, on_progress=progress,
)
```

- [ ] **Step 3: Verify no old names remain in the file**

Run: `grep -n "frame_sampling\|sample_frames_fps\|sample_frames_optical_flow\|verbose=" collab_splats/webapp/routers/preprocess.py`
Expected: no output (`verbose` was a sampler-only param).

- [ ] **Step 4: Run any webapp tests + import check**

Run: `/opt/venv/reconstruction/bin/python -c "import collab_splats.webapp.routers.preprocess" && /opt/venv/reconstruction/bin/python -m pytest tests/ -v -k webapp`
Expected: import succeeds; any webapp tests pass (or "no tests ran" if none match).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/webapp/routers/preprocess.py
git commit -m "refactor(webapp): preprocess router uses preproc.sample_frames dispatcher"
```

---

### Task 9: Migrate dashboard (pipeline.py, app.py, test_pipeline.py)

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py:27-31` (import) and the sampling branch (~line 160-185)
- Modify: `collab_splats/dashboard/app.py:324` (inline import)
- Modify: `tests/dashboard/test_pipeline.py:41-42,89-90` (patch targets)

- [ ] **Step 1: Replace the pipeline import block**

```python
# OLD
from collab_splats.utils.frame_sampling import (
    get_video_info,
    sample_frames_fps,
    sample_frames_optical_flow,
)
# NEW
from collab_splats.preproc import sample_frames
```

If `get_video_info` is used elsewhere in pipeline.py (check with `grep -n get_video_info collab_splats/dashboard/pipeline.py`), keep it in the new import: `from collab_splats.preproc import get_video_info, sample_frames`.

- [ ] **Step 2: Replace the sampling branch** (the block starting `if config.sampling_method == "optical_flow":`)

```python
# OLD
if config.sampling_method == "optical_flow":
    frames, _ = sample_frames_optical_flow(
        str(video_path),
        min_disparity=config.min_disparity,
        max_frames=config.max_frames,
        on_progress=on_progress,
        verbose=False,
    )
    # optical-flow sampler returns score dicts, not source frame numbers; indices are positional
    indices = list(range(len(frames)))
else:
    duration_s = info.get("duration_s") or (info["total_frames"] / (info.get("fps") or 30.0))
    target_fps = config.max_frames / max(duration_s, 1.0)
    frames, indices = sample_frames_fps(
        str(video_path),
        fps=target_fps,
        max_frames=config.max_frames,
        on_progress=on_progress,
        verbose=False,
    )
return frames, indices

# NEW — one call; records carry true source indices for both methods
method = "optical_flow" if config.sampling_method == "optical_flow" else "uniform"
frames, records = sample_frames(
    str(video_path),
    method=method,
    min_disparity=config.min_disparity,
    max_frames=config.max_frames,
    on_progress=on_progress,
)
return frames, [r["frame_idx"] for r in records]
```

If the surrounding function computed `info = get_video_info(...)` only for the deleted `duration_s` derivation, delete that line too.

- [ ] **Step 3: Update app.py inline import** (line ~324, inside `_apply_max_frames_bound`)

```python
# OLD
from collab_splats.utils.frame_sampling import get_video_info
# NEW
from collab_splats.preproc import get_video_info
```

(Keep it as a function-local import — the surrounding code lazy-imports deliberately inside a try block.)

- [ ] **Step 4: Update test patch targets** in `tests/dashboard/test_pipeline.py`

At both occurrence sites (~lines 41-42 and 89-90):

```python
# OLD
patch.object(pl, "sample_frames_fps", return_value=_fake_frames()),
patch.object(pl, "get_video_info", return_value={"duration_s": 3.0, "fps": 30}),
# NEW
patch.object(pl, "sample_frames", return_value=_fake_frames()),
```

Drop the `get_video_info` patch line only if Step 2 removed pipeline's `get_video_info` usage; otherwise keep it with the same fake dict. Then update `_fake_frames()` (top of the test file) so its second element is records, not bare indices:

```python
# OLD (second tuple element, whatever shape it currently has, e.g. list(range(n)))
# NEW
def _fake_frames(n=3):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(n)]
    records = [{"frame_idx": i, "blur_score": 200.0} for i in range(n)]
    return frames, records
```

(Match the existing `_fake_frames` frame shapes — only change the second element to record dicts.)

- [ ] **Step 5: Run dashboard tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/pipeline.py collab_splats/dashboard/app.py tests/dashboard/test_pipeline.py
git commit -m "refactor(dashboard): migrate to preproc.sample_frames; true source indices for optical_flow"
```

---

### Task 10: Migrate wrapper (reconstructor.py, splatter.py)

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py:53` region
- Modify: `collab_splats/wrapper/splatter.py:296` region

- [ ] **Step 1: reconstructor.py — replace import and both sampler calls**

Replace the function-local import:

```python
# OLD
import cv2
from collab_splats.utils.frame_sampling import sample_frames_fps, sample_frames_optical_flow
# NEW
import cv2
from collab_splats.preproc import get_video_info, sample_frames
```

(cv2 stays — the function writes JPEGs below.) Replace the video sampling branch:

```python
# OLD
if frame_selection == "optical_flow":
    frame_arrays, _ = sample_frames_optical_flow(
        video_path=str(input_path),
        max_frames=max_frames if max_frames is not None else 200,
    )
else:
    # Probe video metadata; fall back to 30 fps if CAP_PROP_FPS is unavailable
    cap = cv2.VideoCapture(str(input_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Derive target count from proportion, clamped to [min_frames, max_frames]
    target_count = max(min_frames, int(total_frames * frame_proportion))
    if max_frames is not None:
        target_count = min(target_count, max_frames)
    target_fps = native_fps * target_count / max(total_frames, 1)
    frame_arrays, _ = sample_frames_fps(
        video_path=str(input_path),
        fps=target_fps,
        max_frames=max_frames,
    )

# NEW
if frame_selection == "optical_flow":
    frame_arrays, _ = sample_frames(
        str(input_path), method="optical_flow",
        max_frames=max_frames if max_frames is not None else 200,
    )
else:
    # Derive target count from proportion, clamped to [min_frames, max_frames];
    # the uniform sampler spreads that count over the video itself (fps=None)
    total_frames = get_video_info(str(input_path))["total_frames"]
    target_count = max(min_frames, int(total_frames * frame_proportion))
    if max_frames is not None:
        target_count = min(target_count, max_frames)
    frame_arrays, _ = sample_frames(
        str(input_path), method="uniform", max_frames=target_count,
    )
```

- [ ] **Step 2: splatter.py — replace import and call**

```python
# OLD
from collab_splats.utils.frame_sampling import sample_frames_optical_flow
...
sampled_frames, _ = sample_frames_optical_flow(file_path.as_posix(), max_frames=min(n_samples, 200), verbose=False)
# NEW
from collab_splats.preproc import sample_frames
...
sampled_frames, _ = sample_frames(file_path.as_posix(), method="optical_flow", max_frames=min(n_samples, 200))
```

(Keep the import function-local as it is today — splatter lazy-imports inside the branch deliberately.)

- [ ] **Step 3: Verify and run wrapper tests**

Run: `grep -rn "frame_sampling\|sample_frames_fps\|sample_frames_optical_flow" collab_splats/wrapper/` — expected: no output.
Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -v -k "wrapper or splatter or reconstructor"`
Expected: PASS (or no tests matched).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py collab_splats/wrapper/splatter.py
git commit -m "refactor(wrapper): migrate reconstructor/splatter to preproc.sample_frames"
```

---

### Task 11: Migrate evals, utils `__init__`, cu121 import gate

**Files:**
- Modify: `evals/datasets.py:9` + call sites
- Modify: `collab_splats/utils/__init__.py`
- Modify: `tests/test_cu121_migration.py:133`

- [ ] **Step 1: evals/datasets.py**

```python
# OLD (line 9)
from collab_splats.utils.frame_sampling import extract_video_frames, sample_frames_fps
# NEW
from collab_splats.preproc import extract_frames, sample_frames
```

Then run `grep -n "extract_video_frames\|sample_frames_fps" evals/datasets.py` and apply at each call site:
- `extract_video_frames(` → `extract_frames(` (same positional args: video_path, frame_indices, output_dir)
- `sample_frames_fps(<path>, fps=<x>, ...)` → `sample_frames(<path>, method="uniform", fps=<x>, ...)`, dropping any `verbose=` kwarg. Return shape note: second element is now record dicts — if datasets.py uses the indices, convert with `[r["frame_idx"] for r in records]`.

- [ ] **Step 2: utils/__init__.py — remove the frame_sampling re-exports**

Delete line 1 (`from .frame_sampling import OpticalFlowFrameSelector, sample_frames_fps, sample_frames_optical_flow`) and remove `"OpticalFlowFrameSelector"`, `"sample_frames_fps"`, `"sample_frames_optical_flow"` from `__all__`.

- [ ] **Step 3: tests/test_cu121_migration.py — update the module list** (line ~133)

```python
# OLD
"collab_splats.utils.frame_sampling",
# NEW
"collab_splats.preproc",
"collab_splats.preproc.sampling",
```

- [ ] **Step 4: Run the touched tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/test_cu121_migration.py -v && /opt/venv/reconstruction/bin/python -c "import evals.datasets; import collab_splats.utils"`
Expected: PASS, imports succeed.

- [ ] **Step 5: Commit**

```bash
git add evals/datasets.py collab_splats/utils/__init__.py tests/test_cu121_migration.py
git commit -m "refactor: migrate evals + utils exports + cu121 import gate to preproc"
```

---

### Task 12: Delete old modules and old tests; full-suite gate

**Files:**
- Delete: `collab_splats/utils/frame_sampling.py`
- Delete: `collab_splats/semantics/frame_sampling.py`
- Delete: `tests/utils/test_frame_sampling.py`

- [ ] **Step 1: Confirm zero remaining references, then delete**

```bash
grep -rn "frame_sampling" collab_splats tests evals --include='*.py' | grep -v preproc
```
Expected: no output. If anything appears, fix it first (same rename table as Task 11). Then:

```bash
git rm collab_splats/utils/frame_sampling.py collab_splats/semantics/frame_sampling.py tests/utils/test_frame_sampling.py
```

Also check `collab_splats/semantics/__init__.py` for a `frame_sampling` re-export line (`grep -n frame_sampling collab_splats/semantics/__init__.py`) and remove it if present.

- [ ] **Step 2: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -x -q`
Expected: green modulo `docs/known-test-failures.md`. Fix any failure caused by this refactor before proceeding — do not xfail around it.

- [ ] **Step 3: Format**

```bash
black collab_splats/preproc tests/preproc && isort collab_splats/preproc tests/preproc
```

Re-run `pytest tests/preproc -q` after formatting. Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(preproc)!: delete utils/frame_sampling + semantics re-export (hard cut)"
```

---

### Task 13: Update tutorial notebook

**Files:**
- Modify: `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`

- [ ] **Step 1: Apply the rename table across all code cells** (use NotebookEdit per cell)

| Old | New |
|---|---|
| `from collab_splats.utils.frame_sampling import ...` | `from collab_splats.preproc import ...` (sampling names) + `from collab_splats.preproc.viz import ...` (plot names) |
| `sample_frames_fps(path, fps=X, ...)` | `sample_frames(path, method="uniform", fps=X, ...)` |
| `sample_frames_optical_flow(path, ...)` | `sample_frames(path, method="optical_flow", ...)` |
| `score_all_frames(` | `score_frames(` |
| `load_video_frames(` | `load_frames(` |
| `extract_video_frames(` | `extract_frames(` |
| `save_frame_scores(scores, p)` | `Path(p).write_text(json.dumps(scores))` (add `import json` / `from pathlib import Path` to the cell) |
| `load_frame_scores(p)` | `json.loads(Path(p).read_text())` |
| any `verbose=False/True` kwarg to samplers | delete the kwarg |

Where the notebook uses the second return value of the old fps sampler as indices, replace with `[r["frame_idx"] for r in records]`.

- [ ] **Step 2: Verify no old names remain**

```bash
grep -c "frame_sampling\|sample_frames_fps\|sample_frames_optical_flow\|score_all_frames\|load_video_frames\|extract_video_frames\|save_frame_scores\|load_frame_scores" docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
```
Expected: `0`

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb
git commit -m "docs(tutorials): keyframe_extraction migrated to preproc API"
```

---

### Task 14: Project docs + graph + final gate

**Files:**
- Modify: `CLAUDE.md` (architecture tree)

- [ ] **Step 1: Update the CLAUDE.md architecture tree**

In the `collab_splats/` tree: add under the top level:

```
  preproc/                 # video preprocessing: frame sampling + quality gate
    sampling.py            # ffmpeg decode, blur/exposure gate, sample_frames (uniform | optical_flow)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
```

Remove the two stale lines: `frame_sampling.py      # re-exported here; canonical at utils/frame_sampling.py` (under semantics/) and `frame_sampling.py      # optical-flow + FPS keyframe selection` (under utils/).

- [ ] **Step 2: Full suite + graph refresh**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
graphify update .
```
Expected: suite green modulo known failures; graph updated.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md graphify-out
git commit -m "docs: CLAUDE.md architecture tree — preproc package"
```

---

## Self-Review (done during planning)

- **Spec coverage:** package structure (T1-T7), ffmpeg-only decode (T1), quality gate (T2), selector simplifications (T3), dispatcher + sharpest-in-window + fps=None derivation (T4), frame I/O (T5), viz + `_combine_scores` dedup (T6), public API + matplotlib isolation test (T7), all 7 caller migrations + 3 test-file updates (T8-T11), hard cut (T12), notebook incl. json-direct score I/O (T13), docs (T14). `frame_idx` source-index bug fixed and tested (T4).
- **Type consistency:** `sample_frames` → `tuple[list[np.ndarray], list[dict]]` everywhere; records always contain `frame_idx`+`blur_score`; `score_frames` → `list[dict]` with the 7 documented keys; `_iter_scored_frames` 6-tuple consistent between T4 definition and both consumers.
- **Known judgment calls for the implementer:** synthetic-video blur tests measure thresholds relative to actual decoded scores (not absolute constants) to survive codec variation; if `mp4v` fourcc is unavailable in the environment, switch the fixtures to `avc1`.
