# Video Quality Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure per-frame photometric quality and per-pair camera motion across a source video, write the raw rows to `video_quality_report.json` — a report, not a verdict — and split `preproc/` so each module has one responsibility.

**Architecture:** `preproc/sampling.py` currently holds four jobs; its own section dividers admit it ("Video metadata / decoding", "Frame quality", "Optical-flow selector", "Sampling", "Frame I/O"). Tasks 1–2 split it into three modules that import strictly downward, then Tasks 3–8 build the new measurements on top:

```
preproc/video.py     decode + probe          imports nothing from preproc
preproc/qa.py        measure a frame/pair    imports video
preproc/sampling.py  select frames           imports video + qa
```

**Tech Stack:** OpenCV 4.13.0 (ORB, BFMatcher, USAC_MAGSAC), scikit-image 0.26.0 (`blur_effect`, Crete-Roffet), NumPy. No new dependencies, and **no SciPy** — see "Correlations dropped" below. Python: `/opt/venv/reconstruction/bin/python`.

---

## Decisions that changed after the spec was written

The spec (`docs/superpowers/specs/2026-08-20-video-quality-report-design.md`) predates exact code. **This plan is authoritative where they disagree**; Task 9 amends the spec.

**1. `sampling.py` is split, not merely reused.** The spec had `qa.py` import `_analysis_gray`, `compute_blur_score`, `_iter_frames`, and `get_video_info` from `sampling.py`. That leaves quality measurement living inside a frame-selection module and points the dependency the wrong way — selection uses measurement, not the reverse. Tasks 1–2 move each symbol to the layer that owns it. Blast radius is small: every production caller imports from the `collab_splats.preproc` package, not from `preproc.sampling`, so as long as `__init__.py` re-exports the same names, **no production file changes**. Only `tests/preproc/test_sampling.py` and `tests/pointcloud/test_loger_creator.py` import by module path.

**2. Correlations are dropped, and with them `_spearman` and SciPy.** The spec shipped `rho(blur, laplacian)` and `rho(translation_px, blur)`. Both inputs ship raw and in full, so both rho values are one line for a reader to compute — which is exactly the rule that killed `QUANTILE_GRID` ("once every column ships raw, its quantiles are convenience a reader can compute"). Keeping them also meant a `_spearman` wrapper, the kind of `describe()`/`_distribution` helper this design already refused. Dropping them removes `scipy.stats`, which costs **1160 ms** to import against `collab_splats.preproc`'s **1287 ms** — decisive once `sampling.py` imports `qa.py`, because that cost would land on the dashboard fast-bind path. Reference values from the planning run, for the measured report only:

```
rho(blur, laplacian)      -0.615   n=2388
rho(translation_px, blur) +0.366   n=2364
```

**3. `compute_blur_score` is kept, not absorbed.** Measured at the gate's own resolution (480×270): `laplacian` 1.27 ms/frame, `blur_effect` 13.76 ms/frame. `check_frame_quality` runs on every frame via `_iter_scored_frames`, so routing it through `compute_frame_quality` would cost 11×. `compute_blur` calls `compute_blur_score`, keeping one implementation of the Laplacian and leaving the gate's cost unchanged.

**4. No `clean_for_json`, and no helper at all.** `clean_for_json` (`geometry/verification.py:376`) is a recursive nan → None walker; `json.dumps` writes a bare `NaN` that no strict parser accepts, which is why it exists. Two reasons not to reuse it: `verification.py` does `import pycolmap` at module scope (line 20), and it tests `np.isnan`, so an inf would slip through. `np.nan_to_num` is not a substitute either — it fills with `0.0`, and for `translation_px` that reads as "the camera held perfectly still", the opposite of "the pair failed to match". Only two columns can be non-finite, so the conversion is inlined at those two sites and no helper exists.

**5. `n_features` is off `compute_video_quality`.** It was a pass-through that is always default in real use; the only test that varies it varies it on `match_orb`, where it is a genuine knob. Signature is now `compute_video_quality(video_path, *, output_path=None, motion_stride=None)`.

**6. JSON stays; Parquet was measured and rejected.** On the real 2388-frame report: JSON 632,651 B, gzipped JSON 150,930 B, Parquet+zstd 172,387 B across **two** files. Parquet loses to gzipped JSON at this row count, splits `frames` and `pairs` into separate files, has nowhere natural for the `video`/`params` metadata, is not greppable, and adds `pyarrow` as a hard dependency plus a third serialization format beside JSON and zarr. It becomes the right answer only for cross-video queries over a corpus, which is a follow-on.

## New trap, not in the spec

**A planar scene shot while translating reads `parallax == 0.0` — identical to pure rotation.** Measured:

```
3D scene + translation      parallax=0.807  translation=51.8 px
3D scene + 5° rotation      parallax=0.000  translation=45.1 px
planar scene + translation  parallax=0.000  translation=50.0 px
```

A homography explains a plane exactly regardless of camera motion. `translation_px` disambiguates: high translation with zero parallax means a flat scene or a pan; low translation with zero parallax means the camera did not move. Neither column means anything alone. Task 7 asserts all three cases so a future pass cannot delete one and keep the illusion that `parallax` is self-sufficient.

**MAGSAC is randomized.** Identical input gave `parallax` 0.793 and 0.807 on consecutive runs. Every parallax assertion here is a wide inequality, never an equality.

---

## File Structure

**Create `collab_splats/preproc/video.py`** — decode and probe. Nothing in it measures or selects anything. Receives, unchanged, from `sampling.py`:

| Symbol | Current line |
|---|---|
| `_require_ffmpeg` | 62 |
| `_rotation_degrees` | 68 |
| `get_video_info` | 83 |
| `_probe_dims` | 120 |
| `_iter_frames` | 155 |
| `_iter_selected_frames` | 461 |
| `_seek_frame` | 752 |
| `extract_frame` | 799 |

**Create `collab_splats/preproc/qa.py`** — measure one frame, one pair, or one video. Receives, unchanged, from `sampling.py`:

| Symbol | Current line |
|---|---|
| `_ANALYSIS_WIDTH` | 30 |
| `_DEFAULT_BLUR_THRESHOLD` | 34 |
| `_EXPOSURE_MEAN_RANGE` | 36 |
| `_EXPOSURE_MIN_STD` | 37 |
| `compute_blur_score` | 185 |
| `check_frame_quality` | 190 |
| `_analysis_gray` | 335 |

Then gains the seven new functions: `compute_blur`, `compute_exposure`, `compute_frame_quality`, `match_orb`, `compute_translation`, `compute_parallax`, `compute_video_quality`. The new functions declare **zero constants** — tuning values are keyword arguments with defaults. The four constants above are pre-existing gate values and keep their names and values exactly.

**Modify `collab_splats/preproc/sampling.py`** — selection only, ~813 → ~430 LOC. Keeps `_VALID_PROBE_MAX`, `_SELECT_THRESHOLD`, `_ROTATION_THRESHOLD_DEG`, `_LK_PARAMS`, `_FEATURE_PARAMS`, `_combine_scores`, `OpticalFlowFrameSelector`, `_progress_reporter`, `_uniform_targets`, `_fps_targets`, `sample_frames`, `_sample_positions`, `_sample_uniform`, `_sample_fps`, `_iter_scored_frames`, `_sample_optical_flow`, `score_frames`.

**Modify `collab_splats/preproc/__init__.py`** — same seven exported names, sourced from their new modules. This is what keeps production untouched.

**Modify `tests/preproc/test_sampling.py`** — move the decode tests to `test_video.py` and the gate tests to `test_qa.py`; fix the remaining import paths.
**Create `tests/preproc/test_video.py`**, **create `tests/preproc/test_qa.py`**.
**Modify `tests/pointcloud/test_loger_creator.py:22`** — the one non-preproc test importing by module path.
**Create `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`**, **modify** the design spec and `CLAUDE.md`.

**Before Task 1, run `git status`.** Planning ran a scratch `collab_splats/preproc/qa.py` in place to measure the numbers quoted throughout this plan, then deleted it. If it reappears, delete it — a pre-existing file makes every "verify it fails" step pass silently.

---

### Task 1: Extract `preproc/video.py`

A pure move. No line of moved code changes. The proof is that the existing suite passes with only import paths edited.

**Files:**
- Create: `collab_splats/preproc/video.py`
- Modify: `collab_splats/preproc/sampling.py`, `collab_splats/preproc/__init__.py`
- Create: `tests/preproc/test_video.py`
- Modify: `tests/preproc/test_sampling.py`, `tests/pointcloud/test_loger_creator.py`

- [ ] **Step 1: Record the baseline**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/test_loger_creator.py -q 2>&1 | tail -3
```

Write down the pass count. Every later step in this task must reproduce it exactly.

- [ ] **Step 2: Create `collab_splats/preproc/video.py`**

Move lines 57–177 (the `Video metadata / decoding` divider, `_require_ffmpeg`, `_rotation_degrees`, `get_video_info`, `_probe_dims`, `_iter_frames`), 461–505 (`_iter_selected_frames`), and 747–813 (the `Frame I/O` divider, `_seek_frame`, `extract_frame`) out of `sampling.py` and into a new file with this header. **Cut, do not copy** — the originals must be gone from `sampling.py`.

```python
"""Video decode and probe: the only module that shells out to ffmpeg/ffprobe.

Holds no measurement and no selection logic, so both preproc.qa and
preproc.sampling can depend on it without a cycle.
"""

import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
```

Keep the two `########` section dividers that already wrap this code ("Video metadata / decoding (ffmpeg + ffprobe only)" and "Frame I/O").

- [ ] **Step 3: Point `sampling.py` at the new module**

Add to `sampling.py`'s import block:

```python
from collab_splats.preproc.video import (
    _iter_frames,
    _iter_selected_frames,
    _probe_dims,
    _require_ffmpeg,
    _seek_frame,
    get_video_info,
)
```

Then delete any now-unused imports from `sampling.py` (`json`, `shutil`, `subprocess`, and `Iterator` if nothing left uses them — let `isort`/manual inspection decide, and confirm with the run in Step 6).

- [ ] **Step 4: Keep `preproc/__init__.py` exporting the same names**

```python
"""Video preprocessing: decode (video), capture quality (qa), frame selection (sampling).

Plots live in collab_splats.preproc.viz and are deliberately not re-exported
(keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.sampling import (
    check_frame_quality,
    compute_blur_score,
    sample_frames,
    score_frames,
)
from collab_splats.preproc.video import extract_frame, get_video_info

__all__ = [
    "FrameStore",
    "sample_frames",
    "score_frames",
    "get_video_info",
    "extract_frame",
    "compute_blur_score",
    "check_frame_quality",
]
```

`check_frame_quality` and `compute_blur_score` still come from `sampling.py` at this point; Task 2 moves them.

- [ ] **Step 5: Move the decode tests and fix the two direct importers**

`tests/preproc/test_sampling.py` places its imports **mid-file, under each section divider** (lines 7, 79, 135, 182) rather than all at the top. Each section is therefore a clean cut-block including its own import line — keep that layout in the new files rather than normalizing it.

Create `tests/preproc/test_video.py` by moving lines 37–73 from `tests/preproc/test_sampling.py` — `test_get_video_info_keys` (37), `test_get_video_info_values` (42), `test_get_video_info_missing_file` (50), `test_probe_dims_matches_full_info` (55), `test_require_ffmpeg_raises_without_binary` (61), `test_iter_frames_yields_all_frames_bgr` (68) — and **copying** the `tiny_video` fixture (17–34). Copy, not move: `test_sampling.py`'s own tests take `tiny_video` as an argument, and `tests/preproc/` has no `conftest.py` to share it from. Header:

```python
import cv2
import numpy as np
import pytest

from collab_splats.preproc.video import (
    _iter_frames,
    _probe_dims,
    _require_ffmpeg,
    get_video_info,
)
```

The monkeypatch target inside `test_require_ffmpeg_raises_without_binary` must change:

```python
    monkeypatch.setattr("collab_splats.preproc.video.shutil.which", lambda _: None)
```

`tests/preproc/test_sampling.py`'s top import (lines 7–14) drops `_iter_frames`, `_probe_dims`, and `_require_ffmpeg`, leaving:

```python
from collab_splats.preproc.sampling import _fps_targets, _uniform_targets
from collab_splats.preproc.video import get_video_info
```

In `tests/pointcloud/test_loger_creator.py:22`, change:

```python
from collab_splats.preproc.video import _seek_frame, get_video_info
```

- [ ] **Step 6: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/test_loger_creator.py -q 2>&1 | tail -3
```

Expected: the same pass count as Step 1, redistributed across `test_video.py` and `test_sampling.py`.

- [ ] **Step 7: Prove frame selection is byte-identical**

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.preproc import sample_frames
import hashlib, numpy as np
frames, idx = sample_frames('data/tutorial/tutorial_example-video.mp4', max_frames=20, method='uniform')
print('indices:', list(idx))
print('digest:', hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()[:16])"
```

Run this on `main` (via `git stash`) and again with the change applied. Both digests and both index lists must match. Save the output — Task 10 records it.

- [ ] **Step 8: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/ tests/preproc/
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/ tests/preproc/
git add collab_splats/preproc/ tests/preproc/ tests/pointcloud/test_loger_creator.py
git commit -m "refactor(preproc): extract video.py — decode and probe leave the sampler"
```

Do **not** run `black .`; the venv's black is newer than the repo's formatting and would reformat unrelated files.

---

### Task 2: Extract `preproc/qa.py`

Also a pure move. After this task `sampling.py` contains selection and nothing else.

**Files:**
- Create: `collab_splats/preproc/qa.py`
- Modify: `collab_splats/preproc/sampling.py`, `collab_splats/preproc/__init__.py`
- Create: `tests/preproc/test_qa.py`
- Modify: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Create `collab_splats/preproc/qa.py`**

Cut from `sampling.py`: `_ANALYSIS_WIDTH` (line 30, with its comment), `_DEFAULT_BLUR_THRESHOLD` (34), `_EXPOSURE_MEAN_RANGE` (36), `_EXPOSURE_MIN_STD` (37), `compute_blur_score` (185), `check_frame_quality` (190), and `_analysis_gray` (335). Bodies unchanged. Header:

```python
"""Video capture quality: how good is the source footage, per frame and per pair.

Measurement only — nothing here selects, rejects, or ranks frames on its own.
check_frame_quality is the one exception, and it is a gate the sampler calls,
not a decision this module makes.
"""

import logging

import cv2
import numpy as np
from skimage.measure import blur_effect

logger = logging.getLogger(__name__)


########################################################################
# Constants — pre-existing gate values, unchanged
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
```

`blur_effect` is imported now but unused until Task 3; add it in Task 3 instead if a linter objects.

- [ ] **Step 2: Point `sampling.py` at `qa.py`**

Add to `sampling.py`'s imports:

```python
from collab_splats.preproc.qa import _analysis_gray, _DEFAULT_BLUR_THRESHOLD, check_frame_quality, compute_blur_score
```

`_DEFAULT_BLUR_THRESHOLD` is imported because two public signatures default to it — `sample_frames` (line 387) and `score_frames` (line 712). It stays private and stays in `qa.py`: it is a property of the quality gate, not of the sampler. Confirm both call sites still resolve:

```bash
grep -n '_DEFAULT_BLUR_THRESHOLD\|_ANALYSIS_WIDTH\|_EXPOSURE_' collab_splats/preproc/sampling.py
```

Expected: only the two signature defaults and the import line — every `_EXPOSURE_*` and `_ANALYSIS_WIDTH` reference must have left with `check_frame_quality` and `_analysis_gray`.

- [ ] **Step 3: Update `preproc/__init__.py`**

```python
"""Video preprocessing: decode (video), capture quality (qa), frame selection (sampling).

Plots live in collab_splats.preproc.viz and are deliberately not re-exported
(keeps matplotlib out of pipeline imports).
"""

from collab_splats.preproc.frame_store import FrameStore
from collab_splats.preproc.qa import check_frame_quality, compute_blur_score
from collab_splats.preproc.sampling import sample_frames, score_frames
from collab_splats.preproc.video import extract_frame, get_video_info

__all__ = [
    "FrameStore",
    "sample_frames",
    "score_frames",
    "get_video_info",
    "extract_frame",
    "compute_blur_score",
    "check_frame_quality",
]
```

- [ ] **Step 4: Move the gate tests**

Create `tests/preproc/test_qa.py` by moving the whole `Quality gate` block from `tests/preproc/test_sampling.py` — lines 75–128, i.e. the divider, the import at 79, the `_sharp_gray()` helper (82), `test_compute_blur_score_sharp_exceeds_blurred` (88), `test_check_frame_quality_accepts_sharp_frame` (94), `test_check_frame_quality_rejects_blurred_frame` (100), `test_check_frame_quality_rejects_bad_exposure` (111), `test_check_frame_quality_metrics_fields` (120), `test_check_frame_quality_uses_precomputed_blur_score` (125). Header:

```python
import cv2
import numpy as np
import pytest

from collab_splats.preproc.qa import check_frame_quality, compute_blur_score
```

`tests/preproc/test_sampling.py` drops its `check_frame_quality, compute_blur_score` import (line 79) and the whole moved block.

- [ ] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/test_loger_creator.py -q 2>&1 | tail -3
```

Expected: the same pass count as Task 1 Step 1.

- [ ] **Step 6: Prove the layering holds and nothing got heavier**

```bash
/opt/venv/reconstruction/bin/python -c "
import time, importlib, sys
t = time.perf_counter(); importlib.import_module('collab_splats.preproc')
print(f'preproc import: {(time.perf_counter()-t)*1000:.0f} ms')
print('scipy.stats loaded:', 'scipy.stats' in sys.modules)
import collab_splats.preproc.video as v
print('video imports qa:', 'collab_splats.preproc.qa' in [m for m in sys.modules if m in getattr(v, '__dict__', {})] or hasattr(v, 'qa'))"
grep -n 'from collab_splats' collab_splats/preproc/video.py
```

Expected: around 1300 ms, `scipy.stats loaded: False`, and the `grep` printing **nothing** — `video.py` must import no sibling. Any sibling import there is a cycle waiting to happen.

- [ ] **Step 7: Re-run the selection parity check from Task 1 Step 7**

Expected: identical indices and digest again.

- [ ] **Step 8: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/ tests/preproc/
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/ tests/preproc/
git add collab_splats/preproc/ tests/preproc/
git commit -m "refactor(preproc): extract qa.py — sampling.py is now selection only"
```

---

### Task 3: `compute_blur`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Change the import at the top of `tests/preproc/test_qa.py` to:

```python
from collab_splats.preproc.qa import check_frame_quality, compute_blur, compute_blur_score
```

Append to `tests/preproc/test_qa.py`:

```python
########################################################################
# Per frame
########################################################################


@pytest.fixture(scope="module")
def noise_gray():
    """240x320 uniform noise — maximum high-frequency content, the sharp end of the ladder."""
    rng = np.random.default_rng(0)
    return (rng.random((240, 320)) * 255).astype(np.uint8)


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
    # metric fails loudly rather than shifting every report already on disk.
    sharp = compute_blur(noise_gray)
    blurred = compute_blur(cv2.GaussianBlur(noise_gray, (0, 0), 3))
    assert sharp["blur"] == pytest.approx(0.1202, abs=0.01)
    assert sharp["laplacian"] == pytest.approx(108108.3, rel=0.05)
    assert blurred["blur"] == pytest.approx(0.4798, abs=0.01)
    assert blurred["laplacian"] == pytest.approx(3.6, rel=0.2)


def test_compute_blur_reuses_the_gate_metric(noise_gray):
    # One implementation of the Laplacian in the repo, not two
    assert compute_blur(noise_gray)["laplacian"] == compute_blur_score(noise_gray)


def test_compute_blur_is_bounded(noise_gray):
    assert 0.0 <= compute_blur(noise_gray)["blur"] <= 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_blur'`

- [ ] **Step 3: Write the minimal implementation**

Append to `collab_splats/preproc/qa.py`, under a new `# Per frame` divider:

```python
########################################################################
# Per frame
########################################################################


def compute_blur(gray: np.ndarray) -> dict:
    """Blur measured two ways: Crete-Roffet perceptual blur and Laplacian variance.

    blur is [0, 1] and higher means blurrier; laplacian is unbounded and higher
    means sharper. They run in opposite directions on purpose — where the two
    disagree, the frame is textureless rather than blurred.
    """
    # Crete-Roffet re-blurs the image and measures how little changes. A frame
    # that is already blurred barely moves, so its score rises toward 1.
    perceptual = float(blur_effect(gray))

    # Laplacian variance reuses the frame-selection gate's own metric verbatim,
    # so sharpness has exactly one implementation in the repo.
    return {"blur": perceptual, "laplacian": compute_blur_score(gray)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 new tests pass alongside the gate tests moved in Task 2

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_blur — perceptual blur alongside Laplacian variance"
```

---

### Task 4: `compute_exposure`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_exposure` to the `from collab_splats.preproc.qa import ...` line. Append to `tests/preproc/test_qa.py`:

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
    # A dark scene with a bright window: the mean is dragged up, the median is not
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

Append to the `# Per frame` section of `collab_splats/preproc/qa.py`:

```python
def compute_exposure(gray: np.ndarray) -> dict:
    """Brightness distribution plus the fraction of pixels pinned at either end."""
    return {
        # Mean and median together: they separate when a small bright region
        # (a window, a lamp) drags the mean while most of the scene stays dark.
        "exposure_mean": float(gray.mean()),
        "exposure_median": float(np.median(gray)),
        # Contrast. A low std is a flat, textureless frame regardless of brightness.
        "exposure_std": float(gray.std()),
        # Clipped pixels are destroyed data, not merely dark or bright data:
        # 0 and 255 are the two values where the sensor recorded nothing recoverable.
        "clipped_low_frac": float((gray == 0).mean()),
        "clipped_high_frac": float((gray == 255).mean()),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_exposure — brightness distribution and clipping fractions"
```

---

### Task 5: `compute_frame_quality`

The one place a resolution decision is made, and it goes two different ways on purpose. Downscaling averages scattered saturated pixels out of existence — measured, 300 scattered white pixels in a 480×640 frame give `clipped_high_frac` 0.000977 natively and **exactly 0.0** after `_analysis_gray`. Blur goes the other way: `blur_effect` costs 62 ms at 1024 px against 13.8 ms at 480 px while the score barely moves (0.1659 → 0.1671).

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_frame_quality` and `_analysis_gray` to the `from collab_splats.preproc.qa import ...` line. Append to `tests/preproc/test_qa.py`:

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
    blank = np.zeros((8, 8), np.uint8)
    assert set(compute_frame_quality(clipped_bgr)) == set(compute_blur(blank)) | set(compute_exposure(blank))


def test_compute_frame_quality_reads_exposure_at_native_resolution(clipped_bgr):
    # The contract that keeps clipping measurable: exposure must NOT go through
    # _analysis_gray, which erases scattered saturated pixels completely.
    native = compute_frame_quality(clipped_bgr)
    downscaled = compute_exposure(_analysis_gray(clipped_bgr))
    assert native["clipped_high_frac"] == pytest.approx(300 / (480 * 640), rel=0.05)
    assert downscaled["clipped_high_frac"] == 0.0


def test_compute_frame_quality_reads_blur_at_analysis_resolution(clipped_bgr):
    # Blur must go through _analysis_gray; assert by equality with the explicit path
    expected = compute_blur(_analysis_gray(clipped_bgr))["blur"]
    assert compute_frame_quality(clipped_bgr)["blur"] == pytest.approx(expected)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_frame_quality'`

- [ ] **Step 3: Write the minimal implementation**

Append to the `# Per frame` section of `collab_splats/preproc/qa.py`:

```python
def compute_frame_quality(bgr: np.ndarray) -> dict:
    """Photometric measurements for one BGR frame: blur and exposure together."""
    # Exposure reads the NATIVE-resolution gray. Downscaling averages scattered
    # saturated pixels out of existence, so clipping fractions taken from a
    # resized frame read 0.0 no matter how blown out the capture actually was.
    native_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    exposure = compute_exposure(native_gray)

    # Blur reads the 480 px analysis gray. blur_effect costs 62 ms at 1024 px
    # against 13.8 ms at 480 px, and the score barely moves across that range
    # (0.1659 -> 0.1671), so the downscale is close to free.
    blur = compute_blur(_analysis_gray(bgr))

    return {**blur, **exposure}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 3 more tests pass

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_frame_quality — native exposure, analysis-res blur"
```

---

### Task 6: `match_orb` and `compute_translation`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `match_orb` and `compute_translation` to the `from collab_splats.preproc.qa import ...` line. Append to `tests/preproc/test_qa.py`:

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
    # A flat image has no corners, so ORB returns no descriptors at all
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
Expected: collection error, `ImportError: cannot import name 'match_orb'`

- [ ] **Step 3: Write the minimal implementation**

Append to `collab_splats/preproc/qa.py`:

```python
########################################################################
# Per pair
########################################################################


def match_orb(gray_a: np.ndarray, gray_b: np.ndarray, *, n_features: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """ORB keypoints matched mutually between two grayscale frames as Nx2 float32 arrays."""
    # Detect and describe each frame independently — no shared state, so the
    # measurement never depends on which frames were selected before this pair.
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)

    # A featureless frame yields no descriptors at all. Return empty rather than
    # raise: zero matches is a fact about the video, not an error.
    empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
    if desc_a is None or desc_b is None:
        return empty

    # ORB descriptors are binary, hence Hamming distance. crossCheck keeps only
    # mutual best matches, which removes the need for a Lowe ratio test.
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
    if not matches:
        return empty

    # Pull the pixel coordinates behind each match into two aligned Nx2 arrays
    pts_a = np.array([kp_a[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 2)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 2)
    return pts_a, pts_b


def compute_translation(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Median match displacement in pixels — how far image content moved between the pair."""
    # nan, not 0.0: with no matches the displacement is unknown, and 0.0 would
    # read as "the camera held perfectly still", the opposite conclusion.
    if len(pts_a) == 0:
        return float("nan")

    # Median over per-match displacement, so a handful of bad matches cannot
    # drag the number the way a mean would.
    return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

- [ ] **Step 5: Commit**

```bash
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add match_orb and compute_translation for per-pair motion"
```

---

### Task 7: `compute_parallax`

The three-case test below is the whole task. Delete any one case and the remaining two make `parallax` look like a self-sufficient "is there depth here" number, which it is not.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_parallax` to the `from collab_splats.preproc.qa import ...` line. Append to `tests/preproc/test_qa.py`:

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
    # ...and the image content really did move, so translation alone cannot tell
    # this case apart from the planar one below.
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
    pts = (np.random.default_rng(0).random((7, 2)) * 100).astype(np.float32)
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

Append to the `# Per pair` section of `collab_splats/preproc/qa.py`:

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

    # Fit both models to the same correspondences. H can only explain a plane or
    # a pure rotation; F can additionally explain translation through depth, so
    # the gap between their inlier counts IS the depth information in the pair.
    _, h_inliers = cv2.findHomography(pts_a, pts_b, cv2.USAC_MAGSAC, 3.0)
    _, f_inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, 3.0)
    n_h = int(h_inliers.sum()) if h_inliers is not None else 0
    n_f = int(f_inliers.sum()) if f_inliers is not None else 0

    # No F inliers means the pair is unexplained by any two-view geometry
    if n_f == 0:
        return float("nan")

    # min() guards the case where H outfits F on a degenerate pair, which would
    # otherwise push the complement negative.
    return float(1.0 - min(1.0, n_h / n_f))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

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

### Task 8: `compute_video_quality`

Decodes once and matches each frame against its partner `motion_stride` frames back. `motion_stride` defaults to `round(fps)` — one second, the pair spacing a reconstruction actually sees under the shipping `fps: 1.0` sampling rate.

The payload is **columnar** (a dict of lists), not a list of row dicts. Measured on the 2388-frame tutorial video: 632,651 bytes, 264 bytes/frame.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [ ] **Step 1: Write the failing test**

Add `compute_video_quality` to the `from collab_splats.preproc.qa import ...` line and add `import json` at the top of the file. Append to `tests/preproc/test_qa.py`:

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
    assert set(report) == {"available", "video", "params", "frames", "pairs"}
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
    # frame_idx is the source video index, not a row position
    assert frames["frame_idx"] == list(range(60))


def test_compute_video_quality_pairs_use_the_default_stride(tiny_video):
    report = compute_video_quality(tiny_video)
    pairs = report["pairs"]
    assert set(pairs) == {"frame_idx_a", "frame_idx_b", "translation_px", "parallax", "n_matches"}
    # 30 fps rounds to a stride of 30, leaving 60 - 30 = 30 pairs
    assert report["params"] == {"motion_stride": 30}
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


def test_compute_video_quality_writes_json(tiny_video, tmp_path):
    out = tmp_path / "nested" / "video_quality_report.json"
    report = compute_video_quality(tiny_video, motion_stride=5, output_path=out)
    assert json.loads(out.read_text()) == report


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
    # A frame of pure black is fully clipped low
    assert report["frames"]["clipped_low_frac"][0] == 1.0
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

Extend `qa.py`'s imports — this is where `qa.py` first depends on `video.py`:

```python
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import blur_effect

from collab_splats.preproc.video import _iter_frames, get_video_info
```

Append to `collab_splats/preproc/qa.py`:

```python
########################################################################
# Whole video
########################################################################


def compute_video_quality(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    motion_stride: int | None = None,
) -> dict:
    """Measure per-frame photometry and per-pair motion across a whole video.

    Args:
        video_path: source video to decode; every frame is measured.
        output_path: where to write video_quality_report.json. None returns the
            report without touching disk.
        motion_stride: frames between the two members of each measured pair.
            None means round(fps) — one second of video, the pair spacing a
            reconstruction sees under the shipping fps: 1.0 sampling rate.
    """
    video_path = Path(video_path)
    info = get_video_info(str(video_path))
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
    frame_idx_a, frame_idx_b, translation_px, parallax, n_matches = [], [], [], [], []

    # Hold only the grays still owed a partner: stride + 1 frames at a time,
    # so memory does not track video length.
    pending: dict[int, np.ndarray] = {}

    for idx, bgr in enumerate(_iter_frames(str(video_path))):
        # Photometry for every frame, no stride
        frames["frame_idx"].append(idx)
        for key, value in compute_frame_quality(bgr).items():
            frames[key].append(value)

        # Motion against the frame one stride back, once one exists
        pending[idx] = _analysis_gray(bgr)
        partner = idx - stride
        if partner in pending:
            pts_a, pts_b = match_orb(pending[partner], pending[idx])
            frame_idx_a.append(partner)
            frame_idx_b.append(idx)
            n_matches.append(int(len(pts_a)))
            translation_px.append(compute_translation(pts_a, pts_b))
            parallax.append(compute_parallax(pts_a, pts_b))
            del pending[partner]

    if not frames["frame_idx"]:
        report = {"available": False, "reason": f"no frames decoded from {video_path}"}
    else:
        report = {
            "available": True,
            "video": {"path": str(video_path), "mtime": video_path.stat().st_mtime, **info},
            "params": {"motion_stride": stride},
            "frames": frames,
            "pairs": {
                "frame_idx_a": frame_idx_a,
                "frame_idx_b": frame_idx_b,
                "n_matches": n_matches,
                # nan -> null on the only two columns that can be non-finite.
                # json.dumps writes a bare NaN that no strict parser accepts, and
                # np.nan_to_num is not the fix: its 0.0 fill would read as "no
                # motion", the opposite of "this pair failed to match".
                "translation_px": [None if np.isnan(v) else v for v in translation_px],
                "parallax": [None if np.isnan(v) else v for v in parallax],
            },
        }
        logger.info("video quality: %d frames, %d pairs, stride %d", len(frames["frame_idx"]), len(frame_idx_a), stride)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 9 more tests pass

- [ ] **Step 5: Run the whole preproc suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/test_loger_creator.py -q`
Expected: the Task 1 Step 1 baseline plus the new `test_qa.py` tests

- [ ] **Step 6: Confirm the layering still holds**

```bash
/opt/venv/reconstruction/bin/python -c "
import sys, collab_splats.preproc
print('scipy.stats loaded:', 'scipy.stats' in sys.modules)"
grep -n 'from collab_splats' collab_splats/preproc/video.py
```

Expected: `False`, and `grep` printing nothing.

- [ ] **Step 7: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/qa.py tests/preproc/test_qa.py
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/qa.py tests/preproc/test_qa.py
git add collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit -m "feat(preproc): add compute_video_quality — columnar per-frame and per-pair report"
```

---

### Task 9: Amend the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-video-quality-report-design.md`

- [ ] **Step 1: Rewrite the reuse section for the three-module split**

Replace the "What is reused vs new" table's premise. It currently describes `qa.py` importing four names from `sampling.py`. The truth is now a split: `video.py` (decode), `qa.py` (measure), `sampling.py` (select), importing strictly downward. State the two facts that make it safe: production imports the `collab_splats.preproc` package rather than the module, and `video.py` imports no sibling.

- [ ] **Step 2: Delete the "Correlations — the only derived numbers that ship" section**

Replace it with a short paragraph under "Why raw, not binned": correlations are derivable from the shipped columns, which is the same rule that removed `QUANTILE_GRID`, so they do not ship. Note that this is what keeps SciPy out of `preproc` — 1160 ms against the package's 1287 ms, on the dashboard fast-bind path. Record the measured values for reference:

```markdown
Measured on the tutorial video, for reference only — these do not ship:
rho(blur, laplacian) = -0.615 (n=2388), rho(translation_px, blur) = +0.366 (n=2364).
```

- [ ] **Step 3: Correct the reuse rows**

1. `compute_blur_score`: **kept**, not deleted — `compute_blur` calls it, and the gate must not pay `blur_effect` (1.27 ms vs 13.76 ms per frame at 480×270, on a path that runs on every frame).
2. `check_frame_quality`: unchanged in behaviour and signature; it moves file, nothing more.
3. `clean_for_json`: not used. `verification.py` imports `pycolmap` at module scope, and it guards on `np.isnan` so an inf would pass. Two inline comprehensions replace it. `np.nan_to_num` is not an alternative — a `0.0` fill on `translation_px` asserts the camera held still.
4. `n_features` is not a `compute_video_quality` parameter; it lives on `match_orb`.

- [ ] **Step 4: Record the format decision**

Add under "Report shape":

```markdown
JSON, not Parquet. Measured on the 2388-frame tutorial report: JSON 632,651 B,
gzipped JSON 150,930 B, Parquet+zstd 172,387 B across two files. Parquet loses
to gzipped JSON at this row count, splits frames and pairs into separate files,
has nowhere natural for the video/params metadata, is not greppable, and adds
pyarrow plus a third serialization format beside JSON and zarr. It becomes the
right answer only for cross-video queries over a corpus.
```

- [ ] **Step 5: Add the two new traps**

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

- [ ] **Step 6: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-design.md
git commit -m "docs(specs): three-module preproc split; correlations and clean_for_json dropped"
```

---

### Task 10: Measured run, measured report, CLAUDE.md entry

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
print('n_matches   p05/p50', np.percentile(r['pairs']['n_matches'], [5, 50]).round(0))"
ls -l /tmp/tutorial_vqr.json
```

Run it in tmux, not inline — it took **304.2 s** during planning. Those reference values, which the implementation should reproduce in shape (exact floats drift, MAGSAC is randomized):

```
ELAPSED 304.2s  frames 2388  127 ms/frame  pairs 2364
video   1080x1920, fps 23.976, duration_s 99.60, total_frames 2388
params  motion_stride 24
blur          p05/p50/p95   0.1998   0.2155   0.2626
laplacian     p05/p50/p95   1927.4   4680.6   6097.8
exposure_mean p05/p50/p95     73.6     82.2     94.2
clipped_high_frac max 0.0563   clipped_low_frac max 0.0043
translation   p05/p50/p95     48.6    118.3    196.1
parallax      p05/p50/p95    0.221    0.676    0.842
n_matches     p05/p50           272      310
json 632,651 bytes = 264 bytes/frame
```

- [ ] **Step 2: Write the measured report**

Create `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`, mirroring the structure of `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`. It must contain, as literal numbers:

- The command, the video, its frame count, fps, and resolution.
- Wall-clock total and ms/frame, plus JSON size in bytes and bytes/frame.
- The p05/p50/p95 table for all eight frame columns and all three pair value columns.
- The frame-selection parity digests from Task 1 Step 7 and Task 2 Step 7, both sides, stated as identical — this is the evidence that a 380-line refactor of `sampling.py` changed no output.
- The `preproc` import timing before and after the split, and `scipy.stats loaded: False`.
- One paragraph of **description without verdict**: what the distributions look like, no threshold, no advice, no pass/fail. If a column is degenerate on this clip, say so as an observation about this video, not as a property of the metric.

- [ ] **Step 3: Add the CLAUDE.md in-flight entry**

Add to the `## In-Flight Work` list in `CLAUDE.md`:

```markdown
- **video-quality-report** — per-frame photometric + per-pair motion survey of source video, plus the preproc split into video/qa/sampling ([spec](docs/superpowers/specs/2026-08-20-video-quality-report-design.md) · [plan](docs/superpowers/plans/2026-08-20-video-quality-report.md) · [measured](docs/superpowers/specs/2026-08-20-video-quality-report-measured.md))
```

- [ ] **Step 4: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: no new failures against the baseline in `docs/known-test-failures.md`. The split touched `preproc`, which most of the pipeline imports, so this run is the real gate — do not skip it.

- [ ] **Step 5: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`. The dashboard's fast-bind path depends on `collab_splats.preproc` staying light, which this plan changed the import graph of.

- [ ] **Step 6: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
git add CLAUDE.md
git commit -m "docs(specs): measured video quality report on the tutorial video"
```

---

## Deferred, deliberately

- **Visualization.** A plotting rework is its own spec. Nothing here draws anything; the raw columns exist so that spec has something to plot.
- **Pipeline wiring.** No stage, no config key, no `Reconstructor` change. `compute_video_quality` is called directly.
- **A second video.** One measured video establishes the shape; distributions across a corpus are a follow-on — and that corpus is where Parquet becomes the right format.
- **Loop pairs.** Only sequential `(i, i - stride)` pairs are measured.
- **`OpticalFlowFrameSelector` is still a class** in a codebase that prefers functions. It is selection logic and stays in `sampling.py`; converting it is unrelated to this work.
- **Switching the gate's metric to `blur`.** Needs a threshold calibrated from measured footage, and it is the one change here that *would* invalidate every `frames.zarr` on disk.
