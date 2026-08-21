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

**7. Every run reports its progress on the console.** A 2388-frame video is a multi-minute decode; a silent run is indistinguishable from a hung one. `compute_video_quality` logs what it is about to do, shows a `tqdm` bar over the decode pass, and logs elapsed wall time, throughput, and the written file size.

The bar goes on the **orchestrator**, not the primitives. `compute_blur`, `compute_exposure`, `compute_frame_quality`, `match_orb`, `compute_translation`, and `compute_parallax` stay silent: they run once per frame or per pair, so a log line inside any of them is 2388 lines of console spam, and a nested bar per frame is worse. The single pass in `compute_video_quality` is the only loop, so it is the only thing that draws. This is the same split `sampling.py` already uses — `tqdm.auto` in the sampler, nothing in the scorers.

`tqdm` is already a `preproc` dependency (`sampling.py` imports `tqdm.auto`), so this adds no new package. Note that `logger.info` only reaches the console once a handler exists — a caller running this as a script needs `logging.basicConfig(level=logging.INFO)`; the `tqdm` bar writes to stderr regardless.

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

- [x] **Step 1: The baseline, already measured**

Measured on `da73454d`, before any change in this plan:

```
tests/preproc/                        71 passed   (248 s)
tests/pointcloud/test_loger_creator.py 65 passed
```

Every later step must reproduce **71** for `tests/preproc/` plus whatever new tests that step adds. Run the two suites **separately** — running them in one process OOM-killed the container (exit 137) even though the tests themselves passed, because the loger test loads a model on top of the preproc fixtures.

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q -p no:randomly 2>&1 | tail -3
```

- [x] **Step 2: Create `collab_splats/preproc/video.py`**

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

Keep the two `########` section dividers that already wrap this code ("Video metadata / decoding (ffmpeg + ffprobe only)" and "Frame I/O"). **Measured: no moved function uses `cv2`, so drop that import** — the other eight header lines are all live.

- [x] **Step 3: Point `sampling.py` at the new module**

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

- [x] **Step 4: Keep `preproc/__init__.py` exporting the same names**

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

- [x] **Step 5: Move the decode tests and fix the two direct importers**

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

`_iter_frames` does not disappear from `test_sampling.py` entirely: `test_fps_frames_are_rgb`, which stays, calls it directly. Add it to the **Samplers** section's own mid-file import block rather than the top one, matching the file's per-section convention:

```python
from collab_splats.preproc.sampling import sample_frames, score_frames
from collab_splats.preproc.video import _iter_frames
```

In `tests/pointcloud/test_loger_creator.py:22`, change:

```python
from collab_splats.preproc.video import _seek_frame, get_video_info
```

- [x] **Step 6: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q -p no:randomly 2>&1 | tail -3
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -q -p no:randomly 2>&1 | tail -3
```

Expected: the same pass count as Step 1, redistributed across `test_video.py` and `test_sampling.py`.

- [x] **Step 7: Prove frame selection is byte-identical**

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.preproc import sample_frames
import hashlib, numpy as np
frames, idx = sample_frames('data/tutorial/tutorial_example-video.mp4', max_frames=20, method='uniform')
print('indices:', list(idx))
print('digest:', hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()[:16])"
```

The pre-change values, measured on `da73454d`, are already recorded — **do not `git stash` to re-derive them**, the working tree carries unrelated changes from concurrent sessions:

```
n_frames: 20
indices:  [0, 129, 250, 379, 500, 631, 752, 878, 1002, 1134,
           1259, 1383, 1511, 1636, 1762, 1883, 2013, 2139, 2264, 2387]
digest:   b8d8bc70f766c466
```

Both the index list and the digest must come back identical after the move. A digest mismatch means the refactor changed decoded pixels; an index mismatch means it changed selection. Either one fails the task.

Note the snippet's second return value is `records`, not bare indices — `list(idx)` prints twenty `{"frame_idx": ..., "blur_score": ...}` dicts. Compare the `frame_idx` fields against the list above; the `blur_score` values ride along and are not part of the contract.

- [x] **Step 8: Commit**

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

- [x] **Step 1: Create `collab_splats/preproc/qa.py`**

Cut these seven symbols from `sampling.py`, bodies unchanged: `_ANALYSIS_WIDTH` (with its comment), `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD`, `compute_blur_score`, `check_frame_quality`, `_analysis_gray`.

**Locate them by grep, not by the line numbers in this plan** — Task 1's cleanup commit shifted every number below its import block:

```bash
grep -n '^_ANALYSIS_WIDTH\|^_DEFAULT_BLUR_THRESHOLD\|^_EXPOSURE_\|^def compute_blur_score\|^def check_frame_quality\|^def _analysis_gray' collab_splats/preproc/sampling.py
```

**Leave `_LK_PARAMS` and `_FEATURE_PARAMS` behind.** They sit interleaved with the gate constants in the same `Constants` block, but they are optical-flow *selection* parameters — they belong to `OpticalFlowFrameSelector`, which stays. So `sampling.py` keeps its `Constants` divider holding just those two, and `qa.py` gets its own.

Header:

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

`blur_effect` is imported now but unused until Task 3; add it in Task 3 instead if a linter objects. **It objects** — `F401 imported but unused` is on by flake8's default select, so Task 2 shipped `qa.py` without the `skimage` import and **Task 3 must add `from skimage.measure import blur_effect` itself**.

Not because of `pyproject.toml`. Measured: `[tool.flake8]` there is **inert** — flake8 7.3.0 does not read `pyproject.toml` without the `flake8_pyproject` plugin, which is not installed, and the repo has no `.flake8`, `setup.cfg`, or `tox.ini`. Proof: `E501` fires at **79** characters on these files despite `max-line-length = 120` and `extend-ignore = ["E203", "E501"]`. Pre-existing repo issue, out of scope here; the `F401` conclusion holds regardless because it is a default, not a configured rule.

- [x] **Step 2: Point `sampling.py` at `qa.py`**

Add to `sampling.py`'s imports:

```python
from collab_splats.preproc.qa import _analysis_gray, _DEFAULT_BLUR_THRESHOLD, check_frame_quality, compute_blur_score
```

`_DEFAULT_BLUR_THRESHOLD` is imported because two public signatures default to it — `sample_frames` and `score_frames` (grep for them; Task 1's cleanup shifted the numbers). It stays private and stays in `qa.py`: it is a property of the quality gate, not of the sampler.

A reviewer proposed promoting it to a public `DEFAULT_BLUR_THRESHOLD` on the grounds that `sampling` importing a private name from a sibling is a smell. **Rejected:** a leading underscore marks a name package-internal, not module-internal, and Sphinx renders a default as its *value* (`50.0`), so nothing private leaks into the public signature. Promoting it would grow the documented API surface for no caller.

Confirm both call sites still resolve:

```bash
grep -n '_DEFAULT_BLUR_THRESHOLD\|_ANALYSIS_WIDTH\|_EXPOSURE_' collab_splats/preproc/sampling.py
```

Expected: only the two signature defaults and the import line — every `_EXPOSURE_*` and `_ANALYSIS_WIDTH` reference must have left with `check_frame_quality` and `_analysis_gray`.

- [x] **Step 3: Update `preproc/__init__.py`**

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

- [x] **Step 4: Move the gate tests**

Create `tests/preproc/test_qa.py` by moving the whole `Quality gate` block out of `tests/preproc/test_sampling.py`: the `#### Quality gate ####` divider, its `# isort: split` barrier and the `check_frame_quality, compute_blur_score` import beneath it, the `_sharp_gray()` helper (Step 9 replaces it with a shared fixture), and six tests — `test_compute_blur_score_sharp_exceeds_blurred`, `test_check_frame_quality_accepts_sharp_frame`, `test_check_frame_quality_rejects_blurred_frame`, `test_check_frame_quality_rejects_bad_exposure`, `test_check_frame_quality_metrics_fields`, `test_check_frame_quality_uses_precomputed_blur_score`.

**Locate the block by grep, not by line number** — `62a0352c` moved the `tiny_video` fixture to `conftest.py` and added the `# isort: split` barrier, shifting everything:

```bash
grep -n '^####\|^# isort: split\|^def _sharp_gray\|^def test_' tests/preproc/test_sampling.py
```

The block runs from the `Quality gate` divider to the line before the next `####` divider. Header for the new file:

**Two corrections found while executing this step:**

**`_sharp_gray` has callers on both sides of the cut.** `test_selector_first_frame_scores_one` and `test_selector_identical_frame_scores_low` in the **Selector** section also call it, so moving it leaves two `F821 undefined name` failures behind. This step shipped it as a *copy* in both files, justified as "the same reason Task 1 copied the `tiny_video` fixture" — **that precedent does not exist**: Task 1 *moved* `tiny_video` into `conftest.py` and converted it to a session fixture, leaving zero duplication. Step 9 corrects the copy the same way.

**The `# isort: split` barrier stays in `test_sampling.py`.** It is not a guard on the one import beneath it — `isort` treats it as a whole-file split. Measured by piping the file through `isort` with and without it: removing the barrier hoists the **Selector** section's import into the top block. The Samplers imports are *not* hoisted — real code between them and the top block already blocks it. So the barrier protects one import, not three, but it is still load-bearing and must stay. Move the divider label, keep the barrier: after the cut, `# isort: split` sits directly above the Selector section's import. `test_qa.py` needs no barrier — its one import is already in the top block.

**`pytest` is not in `test_qa.py`'s header** — no moved test uses it, and flake8's default `F401` flags the unused import. Task 3 adds it.

```python
import cv2
import numpy as np

from collab_splats.preproc.qa import check_frame_quality, compute_blur_score
```

`tests/preproc/test_sampling.py` drops its `check_frame_quality, compute_blur_score` import (line 79) and the whole moved block.

- [x] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q -p no:randomly 2>&1 | tail -3
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -q -p no:randomly 2>&1 | tail -3
```

Expected: the same pass count as Task 1 Step 1.

- [x] **Step 6: Prove the layering holds and nothing got heavier**

```bash
/opt/venv/reconstruction/bin/python -c "
import time, importlib, sys
t = time.perf_counter(); importlib.import_module('collab_splats.preproc')
print(f'preproc import: {(time.perf_counter()-t)*1000:.0f} ms')
print('scipy.stats loaded:', 'scipy.stats' in sys.modules)"
grep -n 'from collab_splats' collab_splats/preproc/video.py
```

Expected: around 1300 ms, `scipy.stats loaded: False`, and the `grep` printing **nothing** — `video.py` must import no sibling. Any sibling import there is a cycle waiting to happen.

- [x] **Step 7: Re-run the selection parity check from Task 1 Step 7**

Expected: identical indices and digest again.

- [x] **Step 8: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/ tests/preproc/
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/ tests/preproc/
git add collab_splats/preproc/ tests/preproc/
git commit -m "refactor(preproc): extract qa.py — sampling.py is now selection only"
```

- [x] **Step 9: Apply the code-quality review findings**

Three defects the spec-compliance pass could not see, all follow-ons to the move rather than errors in it.

**`qa.py` was documented nowhere.** Sphinx `automodule` filters members on `__module__`, and neither module defines `__all__`, so once `check_frame_quality.__module__` became `collab_splats.preproc.qa`, the `sampling` page stopped rendering it and no `qa` page existed to pick it up. `compute_blur_score` and `check_frame_quality` fell off the docs entirely. Task 1 set the precedent — `62a0352c` added the `video` block *in the same commit as the move*. Add to `docs/source/api/preproc.rst`, between the `video` and `sampling` blocks so the file reads in import order:

```rst
.. automodule:: collab_splats.preproc.qa
   :members:
   :show-inheritance:
```

and correct the intro prose, which still said `sampling` did the quality gating. **There is no docs-build test** (`tests/docs/` is notebook-only), so nothing catches this class of break — every later task that adds a module must add its block by hand.

**`_sharp_gray` becomes one session fixture.** Move it to `tests/preproc/conftest.py` as `noise_gray` and delete both copies. The "conftest hosts fixtures, not plain helpers" objection dissolves once it *is* a fixture. Seed changes `default_rng(1)` → `default_rng(0)` so Task 3's blur ladder can reuse it; the gate tests are threshold-robust (they derive thresholds from the data, or assert a ratio) so nothing re-pins — measured sharp/blurred ratio 41205× on seed 0 vs 39983× on seed 1, against an assertion of >10×. No caller mutates the array, so session scope is safe; the docstring says so.

**`_ANALYSIS_WIDTH` is the gate's, not the report's.** The standing constraint is zero module constants in the new report code. Retitle the divider to `# Constants — gate-only. Report functions take tuning as keyword args.` so Tasks 3-8 do not reach for it.

Also: `tests/test_cu121_migration.py` gains `"collab_splats.preproc.qa"` (it listed `video` and `sampling` but not `qa`), and `sampling.py`'s docstring drops its claim that cv2 is used there for "grayscale, resize" — both left with `_analysis_gray`; the surviving `cvtColor` calls are BGR→RGB output.

**Rejected: removing `qa.py`'s unused `logging` import and `logger`.** Flagged as inconsistent with dropping `blur_effect` for being unused, but `logger = logging.getLogger(__name__)` is per-module boilerplate, not a function-specific import — `frame_store.py` already carries one with zero call sites. Task 8's `compute_video_quality` uses it. F401 does not fire because `logger` consumes the import.

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/test_cu121_migration.py -q -p no:randomly 2>&1 | tail -3
git add collab_splats/preproc/ tests/preproc/ tests/test_cu121_migration.py docs/source/api/preproc.rst
git commit -m "docs(preproc): document qa.py and fold _sharp_gray into a shared fixture"
```

---

### Task 3: `compute_blur`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Change the imports at the top of `tests/preproc/test_qa.py` to:

```python
import pytest

from collab_splats.preproc.qa import (
    check_frame_quality,
    compute_blur,
    compute_blur_score,
)
```

**Write the import parenthesized — isort's line limit here is 88, not 120.** `[tool.isort]` sets `profile = "black"` with no `line_length` override, and that profile carries black's *default* 88; the repo's `[tool.black] line-length = 120` does not reach isort. Measured: `Config(settings_file="pyproject.toml").line_length == 88`, and the one-line form (89 chars) gets split. Black then keeps the split because of the magic trailing comma. **Tasks 4-8 each add another name to this same line — keep it parenthesized and the problem never recurs.**

`import pytest` is added here, not inherited: Task 2 shipped `test_qa.py` without it because none of the six moved gate tests used it, and flake8 flags unused imports (`F401` is on by flake8's default select — note `pyproject.toml`'s `[tool.flake8]` block is inert, see Task 2's notes). This step is the first to need `@pytest.fixture` and `pytest.approx`. `cv2` and `numpy` are already in the file's header from Task 2.

Append to `tests/preproc/test_qa.py`:

**Do not define a `noise_gray` fixture here.** Task 2's review-fix commit put a session-scoped `noise_gray` in `tests/preproc/conftest.py` — 240x320 uniform noise, `default_rng(0)` — and both `test_qa.py` and `test_sampling.py` already take it as a fixture argument. Just take it as an argument too. It is shared across the session, so never write into it; blur a copy instead (`cv2.GaussianBlur` returns a new array, so the tests below are already safe).

```python
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

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_blur'`

- [x] **Step 3: Write the minimal implementation**

**First add the import.** Task 2 deliberately shipped `qa.py` *without* `from skimage.measure import blur_effect` — `F401 imported but unused` is on by flake8's default select, so an unused import fails the lint. This step is where it becomes used, so add it now:

```python
from skimage.measure import blur_effect
```

Then append to `collab_splats/preproc/qa.py`, under a new `# Per frame` divider:

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

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 new tests pass alongside the gate tests moved in Task 2

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): add compute_blur — perceptual blur alongside Laplacian variance"
```

- [x] **Step 6: Apply the code review findings**

Four changes to the signature, the docstring and the tests. None change what `compute_blur` measures on the path Task 5 feeds it.

**`compute_blur` refuses a colour frame.** `blur_effect` has `channel_axis=None` by default, so a 3-channel array has its channel axis read as *spatial*: the slice loop is empty, `M1 == 0`, and every axis divides by zero. Measured on the noise fixture: `blur_effect(bgr)` → `nan` while `cv2.Laplacian(bgr).var()` → `108108.29`, bit-identical to the gray value. The row is then half-valid, and per the spec's Trap 5 (`nan` *is* a measurement) it reads as a genuinely failed capture rather than a bad call. numpy warns once and the default `once` filter silences the next 2387 frames. So: `if gray.ndim != 2: raise ValueError(...)`.

**`h_size` becomes a keyword-only argument, default 11.** It is this metric's only tuning value — the implementation principle is that tuning values are keyword arguments, and the deferred resolution-sensitivity study has no lever without it. 11 is skimage's own default, so behaviour is unchanged. Measured range on the noise fixture: `h_size=3` → 0.4132, `11` → 0.1202, `21` → 0.0651.

**The docstring was wrong, not merely thin.** It said that where `blur` and `laplacian` disagree "the frame is textureless rather than blurred". Measured: flat field `blur=1.0 laplacian=0.00`, smooth gradient `blur=1.0 laplacian=0.41`, and a single perfectly sharp edge `blur=1.0 laplacian=406.41`. Crete-Roffet saturates whenever there is little high-frequency content to *destroy*, sharp or not — a maximally sharp edge scores maximally blurry. Corrected to: a high `blur` beside a high `laplacian` means detail is **sparse**, which is exactly why `laplacian` ships next to it rather than instead of it.

**`test_compute_blur_is_bounded` cannot fail, and was replaced.** `blur_effect` returns `|M1 - M2| / M1` with `0 <= M2 <= M1` by construction, and the `0.1202` pin already covers the value. Replaced by `test_compute_blur_saturates_on_sparse_detail`, which pins the top of the range and the sharp-edge case the corrected docstring now claims. Two more tests added: `test_compute_blur_rejects_colour_input` and `test_compute_blur_h_size_is_tunable`.

**Not changed: the tolerances.** `rel=0.2` on `laplacian=3.6` looks loose and is not — it admits only sigma 2.80-3.25, and the band has to exist because uint8 quantization makes the Laplacian locally non-monotonic there (`sigma 2.9` → 3.53, `3.0` → 3.62, `3.1` → 3.35).

`tests/preproc/test_qa.py`: 16 → 18 passed (one test removed, three added).

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "fix(preproc): compute_blur refuses colour input and exposes h_size"
```

---

### Task 4: `compute_exposure`

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_exposure'`

- [x] **Step 3: Write the minimal implementation**

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

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): add compute_exposure — brightness distribution and clipping fractions"
```

---

### Task 5: `compute_frame_quality`

The one place a resolution decision is made, and it goes two different ways on purpose. Downscaling averages scattered saturated pixels out of existence — measured, 300 scattered white pixels in a 480×640 frame give `clipped_high_frac` 0.000977 natively and **exactly 0.0** after `_analysis_gray`. Blur goes the other way: `blur_effect` costs 62 ms at 1024 px against 13.8 ms at 480 px while the score barely moves (0.1659 → 0.1671).

**Trap this creates: `blur` is not comparable across videos of different widths.** `_analysis_gray` scales by `min(1.0, 480 / W)`, so a 1080p source is measured at 480 px while a 320 px source is measured natively at 320. Measured by downscaling `data/tutorial/tutorial_example-video.mp4` (1080 wide, 5 frames) to each width:

| width | 320 | 480 | 640 | 1024 | 1080 (native) |
|---|---|---|---|---|---|
| `blur` | 0.2042 | 0.2128 | 0.2272 | 0.2772 | 0.2719 |

480 → 1024 is **+30.3%** on identical frames. Wider sources are downscaled further and read *sharper*, so the bias grows with source width. Every video wider than 480 px is mutually comparable; anything narrower is measured natively and is not. `laplacian` and every exposure column are unaffected, being native-resolution. Record it, do not fix it — normalising would mean inventing a resolution the report was built to observe.

**The spec's "score barely moves (0.1659 → 0.1671)" is wrong and Task 9 must correct it.** Do not try to reproduce the effect by *upscaling* a small fixture instead: interpolating a 320 px noise image up to 1280 invents smooth content and the ladder comes out non-monotonic and interpolation-dependent (two independent runs got 0.1844/0.3002 and 0.1651/0.1560 at 640/1280). Downscaling real footage is the only valid direction, because it is the operation `_analysis_gray` actually performs.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Add `compute_frame_quality` and `_analysis_gray` **inside the parentheses** of the `from collab_splats.preproc.qa import (...)` block, keeping the trailing comma. It has been a multi-line block since Task 3 — isort's limit here is 88 (see Task 3 Step 1), so it will never go back to one line. Append to `tests/preproc/test_qa.py`:

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

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_frame_quality'`

- [x] **Step 3: Write the minimal implementation**

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

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 3 more tests pass

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): add compute_frame_quality — native exposure, analysis-res blur"
```

- [x] **Step 6: Apply the code review findings**

**The blur comment claimed the opposite of the trap above** — "the score barely moves across that range (0.1659 → 0.1671), so the downscale is close to free". Not reproducible; see the measured table. Rewritten to say what the downscale actually buys (7.6x throughput) and what it costs (cross-width comparability), with the trap lifted into the docstring, since the code is where a future reader looks rather than this plan.

**Both section dividers above `_analysis_gray` had gone false.** The review caught that "Frame quality gate — legacy policy, **not part of the report**" now covers a function the report calls. It also covers `compute_blur_score`, which `compute_blur` returns as its `laplacian` column — so the divider was wrong twice, not once. `_ANALYSIS_WIDTH`, `compute_blur_score` and `_analysis_gray` move into a new `# Shared by the gate and the report` section above the gate. Pure move, no logic change; it also makes the "Constants — gate-only" claim true again now that only the gate thresholds sit under it.

**`test_compute_frame_quality_merges_both_measurements` could not detect a key collision.** A set union is identical whether or not `{**blur, **exposure}` overwrites a key, so `assert len(merged) == 7` joins it. No collision exists today.

Also verified, no action: the resolution split is not vacuously tested — a deliberately swapped implementation fails both assertions (exposure `0.0` vs `0.0009765625`; blur `0.120520` vs `0.125305` against `pytest.approx`'s 1e-6 default). All seven values are Python `float`, not numpy scalars, so the Task 8 JSON payload is sound. `blur` can only go `nan` at 1-2 px frame dimensions, unreachable from any decoder.

`tests/preproc`: **86 passed**.

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "fix(preproc): correct the blur resolution claim and the section dividers"
```

---

### Task 6: `match_orb` and `compute_translation`

**Pre-verified against the current tree (2026-08-21), every assertion below holds with margin:** on `noise_gray` vs `np.roll(noise_gray, 17, axis=1)`, `match_orb` returns **539** float32 `(N, 2)` pairs (the test asserts > 200) and `compute_translation` returns **exactly 17.0** (asserted to ±1.0 — the roll's wrap-around seam matches are a small enough minority that the median is untouched). `n_features` 50 → **26** matches against 1000 → **533**. A flat 50x50 frame gives `desc is None` on both sides, so the empty `(0, 2)` return path is genuinely exercised.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Add `match_orb` and `compute_translation` **inside the parentheses** of the `from collab_splats.preproc.qa import (...)` block, keeping the trailing comma and alphabetical order. Append to `tests/preproc/test_qa.py`:

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

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_translation'` — the import block is alphabetical, so `compute_translation` resolves before `match_orb` and raises first. Right failure, different symbol.

- [x] **Step 3: Write the minimal implementation**

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

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): add match_orb and compute_translation for per-pair motion"
```

---

### Task 7: `compute_parallax`

The three-case test below is the whole task. Delete any one case and the remaining two make `parallax` look like a self-sufficient "is there depth here" number, which it is not.

**Pre-verified over 40 MAGSAC draws (2026-08-21) — zero variance, because the synthetic correspondences are noise-free:**

| case | `parallax` (min = max) | asserted | `translation` |
|---|---|---|---|
| volume + translate | **0.8067** | > 0.5 | 51.83 |
| rotation only | **0.0000** | < 0.1 | 45.07 |
| plane + translate | **0.0000** | < 0.1 | 50.00 |

The Step 4 three-times loop is still worth running — MAGSAC *is* randomized, and it is real footage in Task 10 where the margins narrow. n = 7 returns `nan`.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Add `compute_parallax` **inside the parentheses** of the `from collab_splats.preproc.qa import (...)` block, keeping the trailing comma and alphabetical order. Append to `tests/preproc/test_qa.py`:

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

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_parallax'`

- [x] **Step 3: Write the minimal implementation**

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

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 5 more tests pass

Then run the parallax tests three more times — MAGSAC is randomized and the margins must hold across draws (`pytest-repeat` is not installed, so loop in the shell):

```bash
for i in 1 2 3; do /opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k parallax -q; done
```

Expected: 5 passed on every iteration

- [x] **Step 5: Commit**

```bash
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "feat(preproc): add compute_parallax — H/F inlier ratio as a depth-information probe"
```

---

### Task 8: `compute_video_quality`

Decodes once and matches each frame against its partner `motion_stride` frames back. `motion_stride` defaults to `round(fps)` — one second, the pair spacing a reconstruction actually sees under the shipping `fps: 1.0` sampling rate.

The payload is **columnar** (a dict of lists), not a list of row dicts. Measured on the 2388-frame tutorial video: 632,651 bytes, 264 bytes/frame.

**Pre-verified against the current tree (2026-08-21):**
- `get_video_info` **never raises** — it returns `{"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}` on an unprobeable file, and `_iter_frames` returns immediately when either dim is 0. So the `available: False` branch is reached by falling through an empty loop, not by catching anything. Its keys are exactly the five above, which is why the `video` block is those five plus `path` and `mtime`.
- `tiny_video`: 60 frames @ 30.00 fps, so `round(fps)` is 30 and the default stride leaves 30 pairs; `motion_stride=5` leaves 55. Minimum ORB match count is **427** at stride 30 and **581** at stride 5, so `min(n_matches) > 0` holds with room. Note the fixture returns a **`str`**, not a `Path` — `Path(video_path)` in the implementation absorbs that.
- The flat mp4v video decodes to exactly 20 frames and its gray is `min == max == 0`, giving `clipped_low_frac == 1.0`. mp4v's YUV round-trip does *not* lift black off zero, so that assertion is not fragile.
- `tqdm` 4.67.3 is installed and `tqdm.auto` imports. `import collab_splats.preproc` does not pull in `scipy.stats`, so the Step 6 layering gate is already green before the task starts.

**Files:**
- Modify: `collab_splats/preproc/qa.py`
- Modify: `tests/preproc/test_qa.py`

- [x] **Step 1: Write the failing test**

Add `compute_video_quality` **inside the parentheses** of the `from collab_splats.preproc.qa import (...)` block, keeping the trailing comma and alphabetical order, and add `import json` and `import logging` to the test file's own imports. Append to `tests/preproc/test_qa.py`:

**Do not define a `tiny_video` fixture here.** Task 1's cleanup commit (`62a0352c`) created `tests/preproc/conftest.py` holding a session-scoped `tiny_video` with exactly the properties these tests need — 60 frames, 320×240, 30 fps, noise texture plus a moving square. Redefining it in this file would shadow the shared one and re-encode the mp4 a third time. Just take it as a fixture argument.

```python
########################################################################
# Whole video
########################################################################


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


def test_compute_video_quality_logs_before_and_after_the_decode(tiny_video, caplog):
    # A silent multi-minute run is indistinguishable from a hung one, so the
    # announce-then-summarise pair is a contract, not a nicety.
    with caplog.at_level(logging.INFO, logger="collab_splats.preproc.qa"):
        compute_video_quality(tiny_video, motion_stride=5)
    messages = [r.getMessage() for r in caplog.records]
    assert any("60 frames @" in m for m in messages), "no line logged before the decode"
    assert any("frames/s" in m for m in messages), "no elapsed/throughput line logged after"
```

- [x] **Step 2: Run the test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: collection error, `ImportError: cannot import name 'compute_video_quality'`

- [x] **Step 3: Write the minimal implementation**

Extend `qa.py`'s imports — this is where `qa.py` first depends on `video.py`:

```python
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import blur_effect
from tqdm.auto import tqdm

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

    # Announce the work before the first decode — a multi-minute silent run is
    # indistinguishable from a hung one. %s on the ints so a probe that came
    # back with None does not crash the log line itself.
    logger.info(
        "video quality: %s — %s frames @ %.2f fps, %sx%s, stride %d",
        video_path.name,
        info["total_frames"],
        info["fps"] or 0.0,
        info["width"],
        info["height"],
        stride,
    )
    started = time.perf_counter()

    for idx, bgr in enumerate(
        tqdm(
            _iter_frames(str(video_path)),
            total=info["total_frames"],
            desc="measuring frames",
            unit="frame",
        )
    ):
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
        # Throughput, not just a count: it is the number that tells a reader
        # whether a long run is progressing or degrading.
        elapsed = max(time.perf_counter() - started, 1e-9)
        logger.info(
            "video quality: %d frames, %d pairs, stride %d — %.1fs (%.1f frames/s)",
            len(frames["frame_idx"]),
            len(frame_idx_a),
            stride,
            elapsed,
            len(frames["frame_idx"]) / elapsed,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        logger.info("video quality: wrote %s (%.1f kB)", output_path, output_path.stat().st_size / 1000)
    return report
```

- [x] **Step 4: Run the test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v`
Expected: 10 more tests pass

- [x] **Step 5: Run the whole preproc suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -q -p no:randomly 2>&1 | tail -3
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loger_creator.py -q -p no:randomly 2>&1 | tail -3`
Expected: the Task 1 Step 1 baseline plus the new `test_qa.py` tests

- [x] **Step 6: Confirm the layering still holds**

```bash
/opt/venv/reconstruction/bin/python -c "
import sys, collab_splats.preproc
print('scipy.stats loaded:', 'scipy.stats' in sys.modules)"
grep -n 'from collab_splats' collab_splats/preproc/video.py
```

Expected: `False`, and `grep` printing nothing.

- [x] **Step 7: Export `compute_video_quality`, and only it**

The report is the deliverable; the six primitives exist to build it. Add one name to `collab_splats/preproc/__init__.py` — the import line and `__all__` — taking the package surface from seven to eight:

```python
from collab_splats.preproc.qa import check_frame_quality, compute_blur_score, compute_video_quality
```

**`compute_blur`, `compute_exposure`, `compute_frame_quality`, `match_orb`, `compute_translation` and `compute_parallax` stay unexported.** They remain importable as `collab_splats.preproc.qa.compute_blur` and Sphinx renders them all from the `qa` automodule block, so nothing is hidden — but re-exporting six building blocks would grow the package surface by 86% for callers who only ever want the report. `compute_blur_score` and `check_frame_quality` keep their exports because they already had them; that is legacy, not a precedent to extend.

**This also breaks `tests/preproc/test_sampling.py::test_public_api_surface`, which the plan missed.** It pins `__all__` to the exact 7-name set and fails the moment the export lands. Update it to 8 and add the other half of the decision while you are there — assert the six primitives are *not* attributes of the package, so the surface choice is pinned rather than merely counted. Also note `isort` reflows the new `from collab_splats.preproc.qa import ...` line: three names cross the 88-char effective limit, so it becomes a parenthesized block.

- [x] **Step 8: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m black collab_splats/preproc/ tests/preproc/
/opt/venv/reconstruction/bin/python -m isort collab_splats/preproc/ tests/preproc/
git commit --only collab_splats/preproc/qa.py collab_splats/preproc/__init__.py tests/preproc/test_qa.py \
  -m "feat(preproc): add compute_video_quality — columnar per-frame and per-pair report"
```

**`git commit --only`, never `git add` + `git commit`.** The git index is shared across concurrent sessions in this worktree — a plain commit sweeps up whatever another session has staged. Measured once already: `91a4f6df` swallowed a subagent's staged `git rm`.

---

### Task 9: Amend the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-video-quality-report-design.md`

The two sites that carry stale measurements are the **Runtime** section (the `blur_effect` cost table and the near-invariance paragraph beneath it) and **Traps**. Grep for `0.1659` and `near-invariant` rather than trusting line numbers.

- [x] **Step 1: Rewrite the reuse section for the three-module split**

Replace the "What is reused vs new" table's premise. It currently describes `qa.py` importing four names from `sampling.py`. The truth is now a split: `video.py` (decode), `qa.py` (measure), `sampling.py` (select), importing strictly downward. State the two facts that make it safe: production imports the `collab_splats.preproc` package rather than the module, and `video.py` imports no sibling.

- [x] **Step 2: Delete the "Correlations — the only derived numbers that ship" section**

Replace it with a short paragraph under "Why raw, not binned": correlations are derivable from the shipped columns, which is the same rule that removed `QUANTILE_GRID`, so they do not ship. Note that this is what keeps SciPy out of `preproc` — 1160 ms against the package's 1287 ms, on the dashboard fast-bind path. Record the measured values for reference:

```markdown
Measured on the tutorial video, for reference only — these do not ship:
rho(blur, laplacian) = -0.615 (n=2388), rho(translation_px, blur) = +0.366 (n=2364).
```

- [x] **Step 3: Correct the reuse rows**

1. `compute_blur_score`: **kept**, not deleted — `compute_blur` calls it, and the gate must not pay `blur_effect` (1.27 ms vs 13.76 ms per frame at 480×270, on a path that runs on every frame).
2. `check_frame_quality`: unchanged in behaviour and signature; it moves file, nothing more.
3. `clean_for_json`: not used. `verification.py` imports `pycolmap` at module scope, and it guards on `np.isnan` so an inf would pass. Two inline comprehensions replace it. `np.nan_to_num` is not an alternative — a `0.0` fill on `translation_px` asserts the camera held still.
4. `n_features` is not a `compute_video_quality` parameter; it lives on `match_orb`.
5. **`blur` saturates at 1.0 on sparse detail, not only on blur.** A flat field, a smooth gradient and a single perfectly sharp edge all measure exactly 1.0 (laplacian 0.00 / 0.41 / 406.41). Reading `blur` alone as softness inverts on any low-texture frame. Belongs in Traps beside Trap 1.
6. **`blur` is width-dependent, and the spec's current "barely moves (0.1659 → 0.1671)" claim is false — the spec predicted this itself.** Its Runtime section already warns the figure is "weak evidence from one synthetic input" and calls re-measuring on real footage "a plan task, not a settled fact". The re-measurement is done and the answer is no.

   **Root cause, measured — the synthetic input was the problem, not the sample size.** Uniform noise has a *flat* power spectrum, so every downscale still carries maximum high-frequency content relative to its own Nyquist and Crete-Roffet barely moves. Real footage is roughly 1/f, so downscaling discards exactly the fine detail the metric reads. Same 1024 → 768 → 512 → 384 ladder, two inputs:

   | input | 1024 | 768 | 512 | 384 | spread |
   |---|---|---|---|---|---|
   | uniform noise | 0.1200 | 0.1250 | 0.1204 | 0.1207 | **4.2%** |
   | real footage | 0.2524 | 0.2322 | 0.2106 | 0.1990 | **26.8%** |

   The noise row reproduces the spec's near-invariance; the footage row is monotonic and 6x wider. Rewrite the Runtime paragraph around this, do not merely swap the numbers — the reason a synthetic probe cannot answer this question is the durable part. Measured by downscaling the tutorial video: 0.2042 / 0.2128 / 0.2272 / 0.2772 at 320 / 480 / 640 / 1024 px — **+30.3%** across 480 → 1024. `_analysis_gray` caps at 480, so videos wider than 480 are mutually comparable and narrower ones are not. Replace the claim and add the trap. The downscale still earns its place on time alone: 300.4 ms native (1920x1080) against 39.4 ms at 853x480, 7.6x, or 12 minutes against 94 seconds over the tutorial video's 2388 frames.
7. **`compute_blur` raises on a colour frame.** `blur_effect` defaults to `channel_axis=None` and returns `nan` on a 3-channel array while `cv2.Laplacian` returns a plausible number — a half-valid row indistinguishable from a real failed capture. The spec's `blur_effect(gray, h_size=11)` contract line should note the 2-D requirement.
8. **`compute_video_quality` is the only new name exported from `collab_splats/preproc/__init__.py`** (surface goes 7 → 8). The six primitives stay at `collab_splats.preproc.qa.*`, where Sphinx's `qa` automodule block still documents them. The report is the deliverable; re-exporting its building blocks would grow the package surface 86% for callers who only want the report.

9. **`parallax` has three nan causes, and the spec names only one.** The spec (grep `too few matches`) says nan means too few matches and that `n_matches` explains it. `compute_parallax` also returns nan when the fundamental fit keeps zero inliers, which is reachable with a full match set — hit 39 times in a 5,977-trial sweep, on pairs carrying duplicated keypoint locations. A 300-match pair reporting nan would read as a contradiction in `report.json`. State all three: `len(pts_a) < 8`, the `cv2.error` of item 11, and `n_f == 0`. (Corrected during Task 9 — this item originally said "two" and "state both causes", counting item 11's cause as the same one.)
10. **`compute_parallax` gained `ransac_thresh_px` (default 3.0).** The spec's contract line still shows a bare `(pts_a, pts_b)`. It is a first-order lever, not a detail: over 99 tutorial pairs at stride 24, 1.0 against 3.0 moves parallax 0.19 on average (max 0.49) and *reorders* the pairs, Spearman **0.708** — where 3.0 against 5.0 is 0.053 and 0.945. MAGSAC re-run spread at a fixed threshold is 0.0000, so all of that movement is the threshold. It carries a unit, analysis-grid pixels, the same grid as `translation_px`.

11. **`compute_parallax` returns nan when OpenCV cannot fit, and this is a production crash the spec must record.** USAC asserts rather than returning an empty model on configurations it cannot estimate, at *any* correspondence count — not only below the 8-point floor. Measured on `tiny_video` frames 40/41: 720 matches, 97.5% zero-displacement, `findHomography` fine, `findFundamentalMat` raises at `estimator.cpp:353`. Unhandled it discarded all 60 frames of photometry and the other 57 pairs. Any tripod or paused segment reaches it. The nan is the measurement, same rationale as `compute_translation`'s.
12. **`motion_stride` must be >= 1, and the check is `is not None`, not truthiness.** `0` fell through to the fps default and silently measured stride 30. A negative stride made `pending` grow with the video rather than staying at `stride + 1` frames — `idx - stride` runs forward, so nothing was ever retired.
13. **The plan's "`get_video_info` never raises" pre-verification is wrong.** With ffmpeg off `PATH`, `_require_ffmpeg` raises `RuntimeError` and it propagates out of `compute_video_quality`. That is the right behaviour — a broken environment is not an unavailable video — but the note as written is false.
14. **`available: False` names the actual condition.** A missing path reported "no frames decoded", sending a reader after a codec problem instead of a typo. Now "file does not exist: <path>".

- [x] **Step 4: Record the format decision**

Add under "Report shape":

```markdown
JSON, not Parquet. Measured on the 2388-frame tutorial report: JSON 632,651 B,
gzipped JSON 150,930 B, Parquet+zstd 172,387 B across two files. Parquet loses
to gzipped JSON at this row count, splits frames and pairs into separate files,
has nowhere natural for the video/params metadata, is not greppable, and adds
pyarrow plus a third serialization format beside JSON and zarr. It becomes the
right answer only for cross-video queries over a corpus.
```

- [x] **Step 5: Add the two new traps**

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

**`translation_px` is in analysis-grid pixels and does not rescale to source
pixels.** `match_orb` reports in whatever grid it was handed, and the report
hands it `_analysis_gray` output, so the column is `_ANALYSIS_WIDTH` pixels.
Multiplying by the resize factor does not recover the native number, because ORB
detects a different keypoint set at each resolution: measured on the tutorial
video (1920x1080, factor 2.25), native-over-analysis is 2.37 on one pair and
3.28 on another. Comparable within a report, not across videos of differing
width — the same shape of caveat as Trap 1's blur.

**A scene cut reads as large confident motion, not as a failure.** `crossCheck`
makes the match sets mutually injective, which is what RANSAC wants, but it
bounds nothing about whether the two frames show the same scene. Two independent
noise images yield 373 mutual matches at a median displacement of 92 px, against
539 at 17 px for a true 17 px shift — so neither `n_matches` nor a `nan` flags
it, and `parallax` comes back high and stable across MAGSAC draws. The column
that would separate them is descriptor distance: median Hamming 80 against 32.
It is not in the shipped schema. Adding it is a follow-on, not a fix — it widens
`match_orb`'s return past the 2-tuple that `compute_translation` and
`compute_parallax` both consume — but the limitation is real and belongs here.
```

- [x] **Step 6: Record the progress-logging contract**

Add under "Report shape":

```markdown
**Every run is visible on the console.** `compute_video_quality` logs the video,
frame count, fps, resolution and stride before the first decode, draws a tqdm bar
over the decode pass, and logs elapsed seconds, frames/s and the written file
size after. A 2388-frame video takes ~5 minutes; a silent run of that length is
indistinguishable from a hung one.

The bar belongs to the orchestrator alone. `compute_blur`, `compute_exposure`,
`compute_frame_quality`, `match_orb`, `compute_translation` and
`compute_parallax` log nothing — each runs once per frame or per pair, so a line
inside any of them is thousands of lines of spam and a nested bar is worse. This
mirrors `sampling.py`, which puts tqdm in the sampler and nothing in the scorers.
tqdm is already a preproc dependency, so this adds no package. `logger.info`
reaches the console only once a handler exists; a script caller needs
`logging.basicConfig(level=logging.INFO)`, while the bar writes to stderr anyway.
```

- [ ] **Step 7: Commit**

```bash
# docs/superpowers/ is gitignored, so force-add first; --only can only name a
# path git already knows. Then --only, so a concurrent session's staged work
# does not ride along.
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-design.md
git commit --only docs/superpowers/specs/2026-08-20-video-quality-report-design.md \
  -m "docs(specs): three-module preproc split; correlations and clean_for_json dropped"
```

---

### Task 10: Measured run, measured report, CLAUDE.md entry

**Files:**
- Create: `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`
- Modify: `CLAUDE.md`

- [x] **Step 1: Run the report against the committed tutorial video**

```bash
/opt/venv/reconstruction/bin/python -c "
import logging, time, numpy as np
from collab_splats.preproc.qa import compute_video_quality
# Without a handler the announce/summary lines go nowhere; the tqdm bar is on
# stderr regardless. This is the run that proves the logging contract works.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
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

- [x] **Step 2: Write the measured report**

Create `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`, mirroring the structure of `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`. It must contain, as literal numbers:

- The command, the video, its frame count, fps, and resolution.
- Wall-clock total and ms/frame, plus JSON size in bytes and bytes/frame.
- The p05/p50/p95 table for all eight frame columns and all three pair value columns.
- The frame-selection parity digests from Task 1 Step 7 and Task 2 Step 7, both sides, stated as identical — this is the evidence that a 380-line refactor of `sampling.py` changed no output.
- The `preproc` import timing before and after the split, and `scipy.stats loaded: False`.
- One paragraph of **description without verdict**: what the distributions look like, no threshold, no advice, no pass/fail. If a column is degenerate on this clip, say so as an observation about this video, not as a property of the metric.

- [x] **Step 3: Add the CLAUDE.md in-flight entry**

Add to the `## In-Flight Work` list in `CLAUDE.md`:

```markdown
- **video-quality-report** — per-frame photometric + per-pair motion survey of source video, plus the preproc split into video/qa/sampling ([spec](docs/superpowers/specs/2026-08-20-video-quality-report-design.md) · [plan](docs/superpowers/plans/2026-08-20-video-quality-report.md) · [measured](docs/superpowers/specs/2026-08-20-video-quality-report-measured.md))
```

- [x] **Step 4: Run the full test suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q`
Expected: no new failures against the baseline in `docs/known-test-failures.md`. The split touched `preproc`, which most of the pipeline imports, so this run is the real gate — do not skip it.

- [x] **Step 5: Run the dashboard smoke gate**

Run: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: `SMOKE PASS`. The dashboard's fast-bind path depends on `collab_splats.preproc` staying light, which this plan changed the import graph of.

- [x] **Step 6: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-video-quality-report-measured.md
git commit --only docs/superpowers/specs/2026-08-20-video-quality-report-measured.md CLAUDE.md \
  -m "docs(specs): measured video quality report on the tutorial video"
```

---

## Deferred, deliberately

- **Visualization.** A plotting rework is its own spec. Nothing here draws anything; the raw columns exist so that spec has something to plot.
- **Pipeline wiring.** No stage, no config key, no `Reconstructor` change. `compute_video_quality` is called directly.
- **A second video.** One measured video establishes the shape; distributions across a corpus are a follow-on — and that corpus is where Parquet becomes the right format.
- **Loop pairs.** Only sequential `(i, i - stride)` pairs are measured.
- **`OpticalFlowFrameSelector` is still a class** in a codebase that prefers functions. It is selection logic and stays in `sampling.py`; converting it is unrelated to this work.
- **Switching the gate's metric to `blur`.** Needs a threshold calibrated from measured footage, and it is the one change here that *would* invalidate every `frames.zarr` on disk.
- **Promoting `_seek_frame` to public `seek_frame`.** Raised in Task 1's code review: it is the only batched-seek entry point (`extract_frame` re-probes per call) and it is imported from outside the package. Measured: that one outside importer is `tests/pointcloud/test_loger_creator.py:22`, a test-local performance trick — **zero production callers**. Promoting it would add an eighth name to a public surface the split was meant to keep at seven. Revisit if a production caller ever needs batched seeking.
