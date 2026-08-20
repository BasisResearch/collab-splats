# Video Quality Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure per-frame photometric quality and per-pair camera motion across a source video, and write the raw per-frame and per-pair rows to `video_quality_report.json` — a report, not a verdict.

**Architecture:** One new module, `collab_splats/preproc/qa.py`, holding seven free functions in three layers: per-frame (`compute_blur`, `compute_exposure`, `compute_frame_quality`), per-pair (`match_orb`, `compute_translation`, `compute_parallax`), and whole-video (`compute_video_quality`). The whole-video function decodes once through the existing ffmpeg pipe and emits a columnar JSON payload. `collab_splats/preproc/sampling.py` is **not modified** — `qa.py` imports `_iter_frames`, `_analysis_gray`, `get_video_info`, and `compute_blur_score` from it and adds nothing to the frame-selection path.

**Tech Stack:** OpenCV 4.13.0 (ORB, BFMatcher, USAC_MAGSAC), scikit-image 0.26.0 (`blur_effect`, Crete-Roffet), SciPy 1.17.1 (`spearmanr`), NumPy. No new dependencies. Python: `/opt/venv/reconstruction/bin/python`.

---

## Three corrections to the spec, all measured

The spec (`docs/superpowers/specs/2026-08-20-video-quality-report-design.md`) was written before exact code existed. Writing it surfaced three defects. **This plan is authoritative where they disagree**; Task 7 amends the spec.

**1. `compute_blur_score` stays; `check_frame_quality` is not thinned.** The spec had `check_frame_quality` call `compute_frame_quality` so the measurement existed once. Measured at the gate's own resolution (480×270):

```
laplacian:    1.27 ms/frame
blur_effect: 13.76 ms/frame
```

Routing the gate through `compute_frame_quality` makes every gated frame pay `blur_effect` — 11× the gate's current cost, on `_iter_scored_frames`, which runs on *every* frame of the video. `qa.compute_blur` calls `compute_blur_score` instead, so the Laplacian still has exactly one implementation and the gate is byte-identical. **`sampling.py`, `preproc/__init__.py`, and `tests/preproc/test_sampling.py` are all untouched by this plan.**

**2. `qa.py` is NOT re-exported from `preproc/__init__.py`.** Measured cold import cost:

```
collab_splats.preproc: 1287 ms
scipy.stats:           1160 ms
skimage.measure:         12 ms
```

Re-exporting would put `scipy.stats` in the `collab_splats.preproc` import chain and roughly double it — and `collab_splats.remote` / the dashboard fast-bind path depend on `preproc` staying light. `preproc/viz.py` is already excluded from `__init__.py` for the same reason (matplotlib). Callers use `from collab_splats.preproc.qa import compute_video_quality`.

**3. No `clean_for_json` import from `collab_splats.geometry.verification`.** That module imports `pycolmap` at module scope, which would drag pycolmap into `preproc`. The payload here is flat columnar lists, so a four-line local `_json_safe` replaces the recursive walker — and it preserves `int` columns (`n_matches`), which `clean_for_json` would not have been asked to do.

## New trap, not in the spec

**A planar scene shot while translating reads `parallax == 0.0` — identical to pure rotation.** Measured:

```
3D scene + translation   parallax=0.807  translation=51.8 px
3D scene + 5° rotation   parallax=0.000  translation=45.1 px
planar scene + translation parallax=0.000 translation=50.0 px
```

A homography explains a planar scene exactly regardless of camera motion. `translation_px` is what disambiguates: high translation + zero parallax = flat scene or a pan; low translation + zero parallax = the camera did not move. Neither column means anything alone. Task 5's test asserts all three cases so a future simplification pass cannot delete one and keep the illusion that `parallax` is self-sufficient.

**MAGSAC is randomized.** Repeated runs on identical input gave `parallax` 0.793 and 0.807. Every parallax assertion in this plan is an inequality with a wide margin, never an equality.

---

## File Structure

- **Create `collab_splats/preproc/qa.py`** — the whole feature. Seven public functions, two private helpers, zero module constants. Tuning values are keyword arguments with defaults.
- **Create `tests/preproc/test_qa.py`** — all tests. `tests/preproc/` has no `conftest.py`, so this file carries its own fixtures (the `tiny_video` fixture in `test_sampling.py` is module-scoped and not importable across files).
- **Create `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`** — the measured companion, mirroring `2026-08-20-scene-error-report-measured.md`.
- **Modify `docs/superpowers/specs/2026-08-20-video-quality-report-design.md`** — amend for the three corrections and the planar trap.
- **Modify `CLAUDE.md`** — in-flight entry.

**Before Task 1, check `git status` for an untracked `collab_splats/preproc/qa.py`.** Planning ran a scratch copy of the module in place to measure the numbers quoted throughout this plan, then deleted it. If it reappears, delete it so Task 1 starts from nothing — a pre-existing file makes Step 2's "verify it fails" pass silently.

---

### Task 1: `compute_blur`

**Files:**
- Create: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Create `tests/preproc/test_qa.py`:

```python
import json

import cv2
import numpy as np
import pytest

from collab_splats.preproc.qa import compute_blur


@pytest.fixture(scope="module")
def noise_gray():
    """240x320 uniform noise — maximum high-frequency content, the sharp end of the ladder."""
    rng = np.random.default_rng(0)
    return (rng.random((240, 320)) * 255).astype(np.uint8)


########################################################################
# Per frame
########################################################################


def test_compute_blur_keys(noise_gray):
    assert set(compute_blur(noise_gray)) == {"blur", "laplacian"}


def test_compute_blur_moves_in_opposite_directions(noise_gray):
    # blur is Crete-Roffet (high = blurrier); laplacian is variance (high = sharper).
    # Progressive Gaussian blur must raise one and lower the other, monotonically.
    ladder = [compute_blur(cv2.GaussianBlur(noise_gray, (0, 0), s) if s else noise_gray) for s in (0, 1, 3, 6)]
    blur = [r["blur"] for r in ladder]
    laplacian = [r["laplacian"] for r in ladder]
    assert blur == sorted(blur), blur
    assert laplacian == sorted(laplacian, reverse=True), laplacian


def test_compute_blur_measured_values(noise_gray):
    # Pinned to measured values so a library swap that silently rescales either
    # metric fails loudly rather than shifting every report on disk.
    sharp = compute_blur(noise_gray)
    blurred = compute_blur(cv2.GaussianBlur(noise_gray, (0, 0), 3))
    assert sharp["blur"] == pytest.approx(0.1202, abs=0.01)
    assert sharp["laplacian"] == pytest.approx(108108.3, rel=0.05)
    assert blurred["blur"] == pytest.approx(0.4798, abs=0.01)
    assert blurred["laplacian"] == pytest.approx(3.6, rel=0.2)


def test_compute_blur_is_bounded(noise_gray):
    assert 0.0 <= compute_blur(noise_gray)["blur"] <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.preproc.qa'`

- [ ] **Step 3: Write the minimal implementation**

Create `collab_splats/preproc/qa.py`:

```python
"""Per-frame and per-pair video capture quality measurements — report only."""

import logging

import numpy as np
from skimage.measure import blur_effect

from collab_splats.preproc.sampling import compute_blur_score

logger = logging.getLogger(__name__)


########################################################################
# Per frame
########################################################################


def compute_blur(gray: np.ndarray) -> dict:
    """Blur measured two ways: Crete-Roffet perceptual blur and Laplacian variance.

    blur is [0, 1] and higher means blurrier; laplacian is unbounded and higher
    means sharper. They run in opposite directions on purpose — where the two
    disagree, the frame is textureless rather than blurred.
    """
    return {"blur": float(blur_effect(gray)), "laplacian": compute_blur_score(gray)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_blur — perceptual blur alongside Laplacian variance"
```

---

### Task 2: `compute_exposure`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `tests/preproc/test_qa.py` to:

```python
from collab_splats.preproc.qa import compute_blur, compute_exposure
```

Append to `tests/preproc/test_qa.py`:

```python
def test_compute_exposure_keys():
    keys = set(compute_exposure(np.full((10, 10), 128, np.uint8)))
    assert keys == {
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    }


def test_compute_exposure_flat_image():
    result = compute_exposure(np.full((10, 10), 128, np.uint8))
    assert result["exposure_mean"] == pytest.approx(128.0)
    assert result["exposure_median"] == pytest.approx(128.0)
    assert result["exposure_std"] == pytest.approx(0.0)
    assert result["clipped_low_frac"] == 0.0
    assert result["clipped_high_frac"] == 0.0


def test_compute_exposure_counts_clipping_at_both_ends():
    # 5 of 100 pixels crushed to black, 5 of 100 blown to white
    gray = np.full((10, 10), 128, np.uint8)
    gray[0, :5] = 0
    gray[1, :5] = 255
    result = compute_exposure(gray)
    assert result["clipped_low_frac"] == pytest.approx(0.05)
    assert result["clipped_high_frac"] == pytest.approx(0.05)


def test_compute_exposure_clipping_rises_only_at_saturation():
    # Scale a uniform mid-bright frame up and down: the mean tracks the scale,
    # but the clipping fractions stay 0 until pixels actually reach 255 or 0.
    base = np.full((10, 10), 200, np.uint8)
    brighter = [compute_exposure(np.clip(base * f, 0, 255).astype(np.uint8)) for f in (1.0, 1.2, 1.3)]
    assert [r["exposure_mean"] for r in brighter] == pytest.approx([200.0, 240.0, 255.0])
    assert [r["clipped_high_frac"] for r in brighter] == [0.0, 0.0, 1.0]
    darker = [compute_exposure((base * f).astype(np.uint8)) for f in (0.1, 0.0)]
    assert [r["exposure_mean"] for r in darker] == pytest.approx([20.0, 0.0])
    assert [r["clipped_low_frac"] for r in darker] == [0.0, 1.0]


def test_compute_exposure_median_separates_from_mean():
    # A dark scene with a bright window: the mean is dragged up, the median is not.
    gray = np.full((100, 100), 30, np.uint8)
    gray[:10, :] = 250
    result = compute_exposure(gray)
    assert result["exposure_median"] == pytest.approx(30.0)
    assert result["exposure_mean"] > 50.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_exposure'`

- [ ] **Step 3: Write the minimal implementation**

Append to the "Per frame" section of `collab_splats/preproc/qa.py`:

```python
def compute_exposure(gray: np.ndarray) -> dict:
    """Brightness distribution plus the fraction of pixels pinned at either end."""
    return {
        "exposure_mean": float(gray.mean()),
        "exposure_median": float(np.median(gray)),
        "exposure_std": float(gray.std()),
        "clipped_low_frac": float((gray == 0).mean()),
        "clipped_high_frac": float((gray == 255).mean()),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_exposure — brightness distribution and clipping fractions"
```

---

### Task 3: `compute_frame_quality`

The merge of Tasks 1 and 2, and the one place a resolution decision is made: **exposure reads native-resolution grayscale, blur reads the 480 px analysis grayscale.** This is not a stylistic split. Downscaling averages scattered saturated pixels out of existence, so clipping fractions measured on a resized frame are wrong — measured, 300 scattered white pixels in a 480×640 frame give `clipped_high_frac` 0.000977 natively and **exactly 0.0** after `_analysis_gray`. Blur goes the other way: `blur_effect` costs 62 ms at 1024 px and 13.8 ms at 480 px while the score barely moves (0.1659 → 0.1671 across that range).

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `tests/preproc/test_qa.py` to:

```python
from collab_splats.preproc.qa import compute_blur, compute_exposure, compute_frame_quality
```

and add below it:

```python
from collab_splats.preproc.sampling import _analysis_gray
```

Append to `tests/preproc/test_qa.py`:

```python
@pytest.fixture(scope="module")
def clipped_bgr():
    """640x480 mid-grey BGR with 300 scattered saturated pixels.

    Scattered, not a block: a saturated block survives downscaling because the
    interpolation window is entirely white, so it would not exercise the bug.
    """
    rng = np.random.default_rng(0)
    bgr = rng.integers(64, 192, (480, 640, 3)).astype(np.uint8)
    ys, xs = rng.integers(0, 480, 300), rng.integers(0, 640, 300)
    bgr[ys, xs] = 255
    return bgr


def test_compute_frame_quality_merges_both_measurements(clipped_bgr):
    assert set(compute_frame_quality(clipped_bgr)) == set(compute_blur(np.zeros((8, 8), np.uint8))) | set(
        compute_exposure(np.zeros((8, 8), np.uint8))
    )


def test_compute_frame_quality_reads_exposure_at_native_resolution(clipped_bgr):
    # The contract that keeps clipping measurable: exposure must NOT go through
    # _analysis_gray, which erases scattered saturated pixels completely.
    native = compute_frame_quality(clipped_bgr)
    downscaled = compute_exposure(_analysis_gray(clipped_bgr))
    assert native["clipped_high_frac"] == pytest.approx(300 / (480 * 640), rel=0.05)
    assert downscaled["clipped_high_frac"] == 0.0


def test_compute_frame_quality_reads_blur_at_analysis_resolution(clipped_bgr):
    # Blur must go through _analysis_gray; assert by equality with the explicit path.
    assert compute_frame_quality(clipped_bgr)["blur"] == pytest.approx(compute_blur(_analysis_gray(clipped_bgr))["blur"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_frame_quality'`

- [ ] **Step 3: Write the minimal implementation**

Add `cv2` to the imports at the top of `collab_splats/preproc/qa.py` and extend the `sampling` import so the block reads:

```python
import cv2
import numpy as np
from skimage.measure import blur_effect

from collab_splats.preproc.sampling import _analysis_gray, compute_blur_score
```

Append to the "Per frame" section:

```python
def compute_frame_quality(bgr: np.ndarray) -> dict:
    """Photometric measurements for one BGR frame.

    Exposure reads native-resolution grayscale because downscaling averages
    saturated pixels away entirely; blur reads the downscaled analysis
    grayscale because blur_effect costs ~14 ms there and its score is
    near-invariant to the downscale.
    """
    return {**compute_blur(_analysis_gray(bgr)), **compute_exposure(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_frame_quality — native exposure, analysis-res blur"
```

---

### Task 4: `match_orb` and `compute_translation`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `tests/preproc/test_qa.py` to:

```python
from collab_splats.preproc.qa import (
    compute_blur,
    compute_exposure,
    compute_frame_quality,
    compute_translation,
    match_orb,
)
```

Append to `tests/preproc/test_qa.py`:

```python
########################################################################
# Per pair
########################################################################


def test_match_orb_returns_paired_float32_arrays(noise_gray):
    pts_a, pts_b = match_orb(noise_gray, np.roll(noise_gray, 17, axis=1))
    assert pts_a.shape == pts_b.shape
    assert pts_a.shape[1] == 2
    assert pts_a.dtype == np.float32
    assert len(pts_a) > 200


def test_match_orb_respects_n_features(noise_gray):
    few, _ = match_orb(noise_gray, np.roll(noise_gray, 5, axis=1), n_features=50)
    many, _ = match_orb(noise_gray, np.roll(noise_gray, 5, axis=1), n_features=1000)
    assert len(few) < len(many)


def test_match_orb_on_featureless_frames_returns_empty():
    # A flat image has no corners: ORB returns no descriptors at all
    flat = np.zeros((50, 50), np.uint8)
    pts_a, pts_b = match_orb(flat, flat)
    assert len(pts_a) == 0 and len(pts_b) == 0
    assert pts_a.shape == (0, 2)


def test_compute_translation_recovers_known_shift(noise_gray):
    # Roll the image 17 px right; the median match displacement must be 17 px
    pts_a, pts_b = match_orb(noise_gray, np.roll(noise_gray, 17, axis=1))
    assert compute_translation(pts_a, pts_b) == pytest.approx(17.0, abs=1.0)


def test_compute_translation_is_nan_without_matches():
    empty = np.empty((0, 2), np.float32)
    assert np.isnan(compute_translation(empty, empty))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_translation'`

- [ ] **Step 3: Write the minimal implementation**

Append to `collab_splats/preproc/qa.py`:

```python
########################################################################
# Per pair
########################################################################


def match_orb(gray_a: np.ndarray, gray_b: np.ndarray, *, n_features: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """ORB keypoints matched mutually between two grayscale frames as Nx2 float32 arrays."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if desc_a is None or desc_b is None:
        return empty
    # crossCheck keeps only mutual best matches, which removes the need for a ratio test
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return empty
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)
    return pts_a, pts_b


def compute_translation(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Median match displacement in pixels — how far image content moved between the pair."""
    if len(pts_a) == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add match_orb and compute_translation for per-pair motion"
```

---

### Task 5: `compute_parallax`

The three-case test below is the whole point of this task. Delete any one case and the remaining two make `parallax` look like a self-sufficient "is there depth here" number, which it is not.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_parallax` to the `from collab_splats.preproc.qa import (...)` list at the top of `tests/preproc/test_qa.py`.

Append to `tests/preproc/test_qa.py`:

```python
def _project(points_3d):
    """Pinhole-project Nx3 world points with fx=fy=500, cx=320, cy=240."""
    x = 500.0 * points_3d[:, 0] / points_3d[:, 2] + 320.0
    y = 500.0 * points_3d[:, 1] / points_3d[:, 2] + 240.0
    return np.stack([x, y], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_scenes():
    """A depth-varying point cloud and a planar one, both 300 points."""
    rng = np.random.default_rng(3)
    volume = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-3, 3, 300), rng.uniform(4, 12, 300)], axis=1)
    plane = np.stack([rng.uniform(-3, 3, 300), rng.uniform(-3, 3, 300), np.full(300, 8.0)], axis=1)
    return volume, plane


def test_compute_parallax_high_when_depth_varies(synthetic_scenes):
    # Translation across a scene with real depth spread: a homography cannot
    # explain the pair, so most H inliers are lost relative to F.
    volume, _ = synthetic_scenes
    pts_a, pts_b = _project(volume), _project(volume - np.array([0.8, 0.0, 0.0]))
    assert compute_parallax(pts_a, pts_b) > 0.5


def test_compute_parallax_zero_for_rotation_only(synthetic_scenes):
    # A pure rotation is exactly a homography no matter how much depth exists
    volume, _ = synthetic_scenes
    theta = np.deg2rad(5.0)
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    pts_a, pts_b = _project(volume), _project(volume @ rot.T)
    assert compute_parallax(pts_a, pts_b) < 0.1
    # ...and the camera really did move the image content, so translation alone
    # cannot tell this case apart from the planar one below.
    assert compute_translation(pts_a, pts_b) > 10.0


def test_compute_parallax_zero_for_translating_over_a_plane(synthetic_scenes):
    # THE TRAP: a flat scene reads parallax 0.0 even under real translation,
    # because a plane is also exactly a homography. parallax alone cannot
    # distinguish "camera did not move" from "scene has no depth".
    _, plane = synthetic_scenes
    pts_a, pts_b = _project(plane), _project(plane - np.array([0.8, 0.0, 0.0]))
    assert compute_parallax(pts_a, pts_b) < 0.1
    assert compute_translation(pts_a, pts_b) > 10.0


def test_compute_parallax_is_nan_below_eight_matches():
    # Eight is the fundamental matrix minimum; fewer is not a small sample, it is undefined
    pts = np.random.default_rng(0).random((7, 2)).astype(np.float32) * 100
    assert np.isnan(compute_parallax(pts, pts + 1.0))


def test_compute_parallax_is_bounded(synthetic_scenes):
    volume, _ = synthetic_scenes
    value = compute_parallax(_project(volume), _project(volume - np.array([0.8, 0.0, 0.0])))
    assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_parallax'`

- [ ] **Step 3: Write the minimal implementation**

Append to the "Per pair" section of `collab_splats/preproc/qa.py`:

```python
def compute_parallax(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """One minus the homography/fundamental inlier ratio — how far the pair departs from a plane.

    A homography explains rotation-only motion and planar scenes exactly, so a
    ratio near 1 (parallax near 0) means the pair carries no depth information.
    Read it alongside translation: a flat scene under real translation also
    reads 0. nan below 8 matches, the fundamental matrix minimum.
    """
    if len(pts_a) < 8:
        return float("nan")
    _, h_inliers = cv2.findHomography(pts_a, pts_b, cv2.USAC_MAGSAC, 3.0)
    _, f_inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, 3.0)
    n_h = int(h_inliers.sum()) if h_inliers is not None else 0
    n_f = int(f_inliers.sum()) if f_inliers is not None else 0
    if n_f == 0:
        return float("nan")
    return float(1.0 - min(1.0, n_h / n_f))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 22 passed

Then run the parallax tests three more times — MAGSAC is randomized and the margins must hold across draws (`pytest-repeat` is not installed, so loop in the shell):

```bash
for i in 1 2 3; do /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k parallax -q; done
```

Expected: 5 passed on every iteration

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_parallax — H/F inlier ratio as a depth-information probe"
```

---

### Task 6: `compute_video_quality`

Decodes the video once, measures every frame, and matches each frame against its partner `motion_stride` frames back. `motion_stride` defaults to `round(fps)` — one second, the pair spacing a reconstruction actually sees under the shipping `fps: 1.0` sampling rate.

The payload is **columnar** (a dict of lists), not a list of row dicts. Measured on the 2388-frame tutorial video: 632,651 bytes, 264 bytes/frame, so a 13k-frame video lands near 3.4 MB where row-of-dicts would be roughly 3× that.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Test: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_video_quality` to the `from collab_splats.preproc.qa import (...)` list at the top of `tests/preproc/test_qa.py`.

Append to `tests/preproc/test_qa.py`:

```python
########################################################################
# Whole video
########################################################################


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """Synthesize a 60-frame 320x240 30fps mp4: static noise texture + moving square."""
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    width, height = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    rng = np.random.default_rng(0)
    noise = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    for i in range(60):
        frame = noise.copy()
        x = 10 + i * 4
        cv2.rectangle(frame, (x, 60), (x + 60, 140), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_compute_video_quality_top_level_keys(tiny_video):
    report = compute_video_quality(tiny_video)
    assert set(report) == {"available", "video", "params", "frames", "pairs", "correlations"}
    assert report["available"] is True


def test_compute_video_quality_video_block(tiny_video):
    video = compute_video_quality(tiny_video)["video"]
    assert set(video) == {"path", "mtime", "total_frames", "fps", "duration_s", "width", "height"}
    assert video["total_frames"] == 60
    assert (video["width"], video["height"]) == (320, 240)


def test_compute_video_quality_frame_columns_are_equal_length(tiny_video):
    frames = compute_video_quality(tiny_video)["frames"]
    assert set(frames) == {
        "frame_idx",
        "blur",
        "laplacian",
        "exposure_mean",
        "exposure_median",
        "exposure_std",
        "clipped_low_frac",
        "clipped_high_frac",
    }
    assert {len(v) for v in frames.values()} == {60}
    assert frames["frame_idx"] == list(range(60))


def test_compute_video_quality_pairs_use_the_default_stride(tiny_video):
    report = compute_video_quality(tiny_video)
    pairs = report["pairs"]
    assert set(pairs) == {"frame_idx_a", "frame_idx_b", "translation_px", "parallax", "n_matches"}
    # 30 fps rounds to a stride of 30, leaving 60 - 30 = 30 pairs
    assert report["params"]["motion_stride"] == 30
    assert {len(v) for v in pairs.values()} == {30}
    assert pairs["frame_idx_a"][:3] == [0, 1, 2]
    assert pairs["frame_idx_b"][:3] == [30, 31, 32]


def test_compute_video_quality_honours_motion_stride(tiny_video):
    report = compute_video_quality(tiny_video, motion_stride=5)
    assert report["params"]["motion_stride"] == 5
    assert len(report["pairs"]["frame_idx_a"]) == 55
    assert report["pairs"]["frame_idx_b"][0] - report["pairs"]["frame_idx_a"][0] == 5


def test_compute_video_quality_keeps_n_matches_integral(tiny_video):
    n_matches = compute_video_quality(tiny_video, motion_stride=5)["pairs"]["n_matches"]
    assert all(isinstance(v, int) for v in n_matches)
    assert min(n_matches) > 0


def test_compute_video_quality_correlations_shape(tiny_video):
    correlations = compute_video_quality(tiny_video)["correlations"]
    assert set(correlations) == {"blur_vs_laplacian", "translation_vs_blur"}
    for entry in correlations.values():
        assert set(entry) == {"rho", "n"}
        assert entry["rho"] is None or -1.0 <= entry["rho"] <= 1.0
    # Every frame contributes to the photometric correlation
    assert correlations["blur_vs_laplacian"]["n"] == 60


def test_compute_video_quality_writes_json_with_no_nan(tiny_video, tmp_path):
    # Bare nan is not valid JSON; every non-finite value must serialise as null
    out = tmp_path / "nested" / "video_quality_report.json"
    report = compute_video_quality(tiny_video, motion_stride=5, output_path=out)
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == report


def test_compute_video_quality_serialises_unmatched_pairs_as_null(tmp_path):
    # A featureless video is the case that produces nan: ORB finds no corners,
    # so every pair has 0 matches and nan translation/parallax. nan is not valid
    # JSON, and it is also the interesting measurement — it must survive as null.
    path = tmp_path / "flat.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for _ in range(20):
        writer.write(np.zeros((240, 320, 3), np.uint8))
    writer.release()

    out = tmp_path / "flat.json"
    report = compute_video_quality(path, motion_stride=5, output_path=out)
    assert report["pairs"]["n_matches"] == [0] * 15
    assert report["pairs"]["translation_px"] == [None] * 15
    assert report["pairs"]["parallax"] == [None] * 15
    # A frame of pure black is fully clipped low, and rho is undefined on constant columns
    assert report["frames"]["clipped_low_frac"][0] == 1.0
    assert report["correlations"]["translation_vs_blur"] == {"rho": None, "n": 0}
    assert "NaN" not in out.read_text()
    assert json.loads(out.read_text()) == report


def test_compute_video_quality_reports_unavailable_for_an_undecodable_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"")
    report = compute_video_quality(broken)
    assert report["available"] is False
    assert "broken.mp4" in report["reason"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_video_quality'`

- [ ] **Step 3: Write the minimal implementation**

Extend the imports at the top of `collab_splats/preproc/qa.py` so the block reads:

```python
"""Per-frame and per-pair video capture quality measurements — report only."""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from scipy import stats
from skimage.measure import blur_effect

from collab_splats.preproc.sampling import (
    _analysis_gray,
    _iter_frames,
    compute_blur_score,
    get_video_info,
)

logger = logging.getLogger(__name__)
```

Append to `collab_splats/preproc/qa.py`:

```python
########################################################################
# Whole video
########################################################################


def _json_safe(values: list) -> list:
    """Replace nan with None so a numeric column round-trips through JSON."""
    return [None if isinstance(v, float) and not np.isfinite(v) else v for v in values]


def _spearman(x: list, y: list) -> dict:
    """Spearman rho over the entries where both columns are finite."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if keep.sum() < 2:
        return {"rho": None, "n": int(keep.sum())}
    rho = stats.spearmanr(xa[keep], ya[keep]).statistic
    return {"rho": None if not np.isfinite(rho) else float(rho), "n": int(keep.sum())}


def compute_video_quality(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    motion_stride: int | None = None,
    n_features: int = 1000,
) -> dict:
    """Measure per-frame photometry and per-pair motion across a whole video."""
    video_path = Path(video_path)
    info = get_video_info(str(video_path))
    # Default the pair spacing to one second, matching the shipping fps: 1.0 sampling rate
    stride = int(motion_stride) if motion_stride else max(1, round(info["fps"] or 1))

    frames = {
        k: []
        for k in (
            "frame_idx",
            "blur",
            "laplacian",
            "exposure_mean",
            "exposure_median",
            "exposure_std",
            "clipped_low_frac",
            "clipped_high_frac",
        )
    }
    pairs = {k: [] for k in ("frame_idx_a", "frame_idx_b", "translation_px", "parallax", "n_matches")}
    # Hold only the grays still owed a partner: stride + 1 frames at a time
    pending: dict[int, np.ndarray] = {}

    for idx, bgr in enumerate(_iter_frames(str(video_path))):
        frames["frame_idx"].append(idx)
        for key, value in compute_frame_quality(bgr).items():
            frames[key].append(value)
        pending[idx] = _analysis_gray(bgr)
        partner = idx - stride
        if partner in pending:
            pts_a, pts_b = match_orb(pending[partner], pending[idx], n_features=n_features)
            pairs["frame_idx_a"].append(partner)
            pairs["frame_idx_b"].append(idx)
            pairs["n_matches"].append(int(len(pts_a)))
            pairs["translation_px"].append(compute_translation(pts_a, pts_b))
            pairs["parallax"].append(compute_parallax(pts_a, pts_b))
            del pending[partner]

    if not frames["frame_idx"]:
        return _write_report({"available": False, "reason": f"no frames decoded from {video_path}"}, output_path)

    # Join each pair back to the blur of its first frame for the motion/blur correlation
    blur_by_idx = dict(zip(frames["frame_idx"], frames["blur"]))
    report = {
        "available": True,
        "video": {"path": str(video_path), "mtime": video_path.stat().st_mtime, **info},
        "params": {"motion_stride": stride, "n_features": int(n_features)},
        "frames": {k: _json_safe(v) for k, v in frames.items()},
        "pairs": {k: _json_safe(v) for k, v in pairs.items()},
        "correlations": {
            "blur_vs_laplacian": _spearman(frames["blur"], frames["laplacian"]),
            "translation_vs_blur": _spearman(pairs["translation_px"], [blur_by_idx[i] for i in pairs["frame_idx_a"]]),
        },
    }
    logger.info(
        "video quality: %d frames, %d pairs, stride %d",
        len(frames["frame_idx"]),
        len(pairs["frame_idx_a"]),
        stride,
    )
    return _write_report(report, output_path)


def _write_report(report: dict, output_path: str | Path | None) -> dict:
    """Write the report to disk when a path was given, and return it either way."""
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 32 passed. `scipy` emits `ConstantInputWarning` on the synthetic clip — the static noise texture makes `translation_px` constant, so `translation_vs_blur` correctly yields `rho: None`. The warning is expected and `pyproject.toml` sets no `filterwarnings = error`.

- [ ] **Step 5: Run the whole preproc suite to confirm nothing else moved**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q`
Expected: all pass, with the pre-existing `test_sampling.py` count unchanged

- [ ] **Step 6: Format**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/qa.py tests/preproc/test_qa.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/qa.py tests/preproc/test_qa.py
```

Do **not** run `black .` — the venv's black is newer than the repo's formatting and would reformat unrelated files.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_video_quality — columnar per-frame and per-pair report"
```

---

### Task 7: Confirm the frame-selection path is untouched, and amend the spec

No production behaviour may change. This task proves it and records why the spec's original plan was dropped.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-video-quality-report-design.md`

- [ ] **Step 1: Prove `sampling.py` and the exports did not move**

```bash
git diff --stat main -- collab_splats/preproc/sampling.py collab_splats/preproc/__init__.py
```

Expected: **empty output**. If either file appears, revert it — `qa.py` reads from `sampling.py` and writes nothing back.

- [ ] **Step 2: Prove `collab_splats.preproc` did not get heavier to import**

```bash
/opt/venv/reconstruction/bin/python -c "
import time, importlib
t = time.perf_counter(); importlib.import_module('collab_splats.preproc')
print(f'preproc import: {(time.perf_counter()-t)*1000:.0f} ms')
import sys; print('scipy.stats loaded:', 'scipy.stats' in sys.modules)"
```

Expected: around 1300 ms and `scipy.stats loaded: False`. A `True` here means someone re-exported `qa` from `preproc/__init__.py`; remove it.

- [ ] **Step 3: Prove frame selection is byte-identical**

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.preproc.sampling import sample_frames
import hashlib, numpy as np
frames, idx = sample_frames('data/tutorial/tutorial_example-video.mp4', max_frames=20, method='uniform')
print('indices:', list(idx))
print('digest:', hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()[:16])"
```

Run this once on `main` (`git stash`) and once with the branch applied. Expected: identical indices and identical digest. Record both in the measured report in Task 8.

- [ ] **Step 4: Amend the spec**

Edit `docs/superpowers/specs/2026-08-20-video-quality-report-design.md`:

1. In the reuse table, change the `compute_blur_score` row from "deleted, absorbed into `compute_blur`" to "**kept** — `compute_blur` calls it; the gate must not pay `blur_effect` (1.27 ms vs 13.76 ms per frame at 480×270, on a path that runs on every frame)".
2. In the same section, replace any statement that `check_frame_quality` is thinned with: "`sampling.py` is not modified at all."
3. Add a line stating `qa.py` is **not** re-exported from `preproc/__init__.py`, with the measured reason: `scipy.stats` costs 1160 ms to import and `collab_splats.preproc` currently costs 1287 ms; `preproc/viz.py` is excluded for the same reason.
4. Replace the `clean_for_json` reuse row with a local `_json_safe`, noting that `geometry/verification.py` imports `pycolmap` at module scope.
5. Add to the Traps section:

```markdown
**A planar scene under real translation reads `parallax == 0.0`, exactly like a
pure rotation.** A homography explains a plane regardless of camera motion.
`translation_px` is the disambiguator: high translation with zero parallax means
a flat scene or a pan; low translation with zero parallax means the camera did
not move. Neither column is interpretable alone. Measured: 3D scene +
translation → 0.807; 3D scene + 5° rotation → 0.000; planar scene + translation
→ 0.000.

**MAGSAC is randomized.** The same input gave `parallax` 0.793 and 0.807 on
consecutive runs. Never assert an exact parallax value.
```

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-design.md
git commit -m "docs(specs): kept compute_blur_score and the light preproc import; planar parallax trap"
```

---

### Task 8: Measured run on real footage, measured report, CLAUDE.md entry

**Files:**
- Create: `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the report against the committed tutorial video**

```bash
/opt/venv/reconstruction/bin/python -c "
import time, numpy as np
from collab_splats.preproc.qa import compute_video_quality
t = time.perf_counter()
r = compute_video_quality('data/tutorial/tutorial_example-video.mp4', output_path='/tmp/tutorial_vqr.json')
elapsed = time.perf_counter() - t
n = len(r['frames']['frame_idx'])
tr = np.array([v for v in r['pairs']['translation_px'] if v is not None])
px = np.array([v for v in r['pairs']['parallax'] if v is not None])
print(f'{elapsed:.1f}s / {n} frames = {elapsed/n*1000:.0f} ms/frame; {len(r[\"pairs\"][\"frame_idx_a\"])} pairs')
print('video      ', r['video'], 'params', r['params'])
print('blur       p05/p50/p95', np.percentile(r['frames']['blur'], [5, 50, 95]).round(4))
print('laplacian  p05/p50/p95', np.percentile(r['frames']['laplacian'], [5, 50, 95]).round(1))
print('exposure   p05/p50/p95', np.percentile(r['frames']['exposure_mean'], [5, 50, 95]).round(1))
print('clip hi/lo max', max(r['frames']['clipped_high_frac']), max(r['frames']['clipped_low_frac']))
print('translation p05/p50/p95', np.percentile(tr, [5, 50, 95]).round(1))
print('parallax    p05/p50/p95', np.percentile(px, [5, 50, 95]).round(3))
print('n_matches   p05/p50', np.percentile(r['pairs']['n_matches'], [5, 50]).round(0))
print('correlations', r['correlations'])"
ls -l /tmp/tutorial_vqr.json
```

Run it in tmux, not inline — it took **304.2 s** during planning. These are the reference values that run produced; the implementation is wrong if the shape does not match (exact floats will differ slightly, since MAGSAC is randomized):

```
ELAPSED 304.2s  frames 2388  127 ms/frame  pairs 2364
video   1080x1920, fps 23.976, duration_s 99.60, total_frames 2388
params  motion_stride 24, n_features 1000
blur          p05/p50/p95   0.1998   0.2155   0.2626
laplacian     p05/p50/p95   1927.4   4680.6   6097.8
exposure_mean p05/p50/p95     73.6     82.2     94.2
clipped_high_frac max 0.0563   clipped_low_frac max 0.0043
translation   p05/p50/p95     48.6    118.3    196.1
parallax      p05/p50/p95    0.221    0.676    0.842
n_matches     p05/p50           272      310
correlations  blur_vs_laplacian   rho -0.615  n 2388
              translation_vs_blur rho +0.366  n 2364
json 632,651 bytes = 264 bytes/frame
```

Two things to note when writing the report in Step 2, both descriptive and neither a verdict: `blur` and `laplacian` anti-correlate at only −0.615 across 2388 frames, so they are not redundant measurements of one quantity; and `translation_vs_blur` at +0.366 is the expected physical coupling — frames captured while the camera moves faster are blurrier.

- [ ] **Step 2: Write the measured report**

Create `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`, mirroring the structure of `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`. It must contain, as literal numbers from Step 1:

- The command, the video, its frame count, fps, and resolution.
- Wall-clock total and ms/frame, plus the JSON size in bytes and bytes/frame.
- The p05/p50/p95 table for every one of the eight frame columns and every one of the three pair value columns.
- Both correlation entries with their `rho` and `n`.
- The frame-selection parity digests from Task 7 Step 3, both sides, stated as identical.
- One paragraph of **description without verdict**: what the distributions look like, no threshold, no advice, no pass/fail. If a column is degenerate on this clip (for example every `clipped_low_frac` is 0.0), say so as an observation about this video, not as a property of the metric.

- [ ] **Step 3: Add the CLAUDE.md in-flight entry**

Add to the `## In-Flight Work` list in `CLAUDE.md`:

```markdown
- **video-quality-report** — per-frame photometric + per-pair motion survey of source video ([spec](docs/superpowers/specs/2026-08-20-video-quality-report-design.md) · [plan](docs/superpowers/plans/2026-08-20-video-quality-report.md) · [measured](docs/superpowers/specs/2026-08-20-video-quality-report-measured.md))
```

- [ ] **Step 4: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: no new failures against the baseline in `docs/known-test-failures.md`

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
git add CLAUDE.md
git commit -m "docs(specs): measured video quality report on the tutorial video"
```

---

## Deferred, deliberately

- **Visualization.** The user asked for a plotting rework as a separate spec. Nothing in this plan draws anything; the raw columns exist so that spec has something to plot.
- **Pipeline wiring.** No stage, no config key, no `Reconstructor` change. `compute_video_quality` is called directly.
- **A second video.** One measured video establishes the shape; the distributions across a corpus are a follow-on.
- **Loop pairs.** Only sequential `(i, i - stride)` pairs are measured.
- **Switching the gate's metric to `blur`.** The spec lists this as a follow-on, and it needs a threshold calibrated from measured footage. It is also the one change here that *would* invalidate every `frames.zarr` on disk — which is exactly why it is not in this plan.
