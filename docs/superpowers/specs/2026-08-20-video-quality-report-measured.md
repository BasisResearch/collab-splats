# Video quality report — measured numbers

Companion to [the design](2026-08-20-video-quality-report-design.md) and [the plan](2026-08-20-video-quality-report.md).
Every number below was measured on 2026-08-21 against the committed tutorial clip, on the
`refactor/cu121-uv-migration` branch with `collab_splats/preproc/` at `7175c6b8`.

This document reports. It does not grade the video, and no number here is a threshold.

---

## The run

### Command

```bash
/opt/venv/reconstruction/bin/python -c "
import logging, time
from collab_splats.preproc.qa import compute_video_quality
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
t = time.perf_counter()
r = compute_video_quality('/workspace/collab-splats/data/tutorial/tutorial_example-video.mp4',
                          output_path='/tmp/video_quality_report.json')
print(f'{time.perf_counter() - t:.1f}s')"
```

`motion_stride` was left at its default, so the run picked it up from the video's frame rate:
`round(23.976) = 24`.

### The video

| | |
|---|---|
| path | `data/tutorial/tutorial_example-video.mp4` |
| resolution | 1080 x 1920 (portrait) |
| fps | 23.976023976023978 |
| duration | 99.59949999999999 s |
| total frames | 2388 |

### Cost

| | |
|---|---|
| wall clock | 332.6 s |
| per frame | 139 ms |
| throughput | 7.2 frames/s |
| frames measured | 2388 (every frame, `frame_idx` contiguous `0..2387`) |
| pairs measured | 2364 — first `(0, 24)`, last `(2363, 2387)` |
| JSON on disk | 632,460 bytes |
| per frame | 264.8 bytes |

Planning measured 304.2 s for the same run; this one shared the machine with another agent.
The JSON differs from the planning-time 632,651 bytes only in the length of the `path` and
`mtime` strings — the columns are identical.

The three `logger.info` lines the run emitted, which are the whole of the logging contract
outside the tqdm bar:

```
09:32:12 INFO video quality: tutorial_example-video.mp4 — 2388 frames @ 23.98 fps, 1080x1920, stride 24
09:37:42 INFO video quality: 2388 frames, 2364 pairs, stride 24 — 330.4s (7.2 frames/s)
09:37:42 INFO video quality: wrote /tmp/video_quality_report.json (632.5 kB)
```

### Distributions

Seven per-frame value columns (the eighth key, `frame_idx`, is the index):

| column | min | p05 | p50 | p95 | max | null |
|---|---|---|---|---|---|---|
| `blur` | 0.1929 | 0.1998 | 0.2155 | 0.2626 | 0.3428 | 0 |
| `laplacian` | 844.22 | 1927.36 | 4680.55 | 6097.83 | 7431.99 | 0 |
| `exposure_mean` | 56.150 | 73.558 | 82.165 | 94.202 | 106.809 | 0 |
| `exposure_median` | 32.0 | 46.0 | 58.0 | 79.0 | 114.0 | 0 |
| `exposure_std` | 44.645 | 53.139 | 64.866 | 72.073 | 74.881 | 0 |
| `clipped_low_frac` | 0.0000 | 0.0002 | 0.0012 | 0.0027 | 0.0043 | 0 |
| `clipped_high_frac` | 0.0000 | 0.0006 | 0.0080 | 0.0356 | 0.0563 | 0 |

Three per-pair value columns (the other two keys, `frame_idx_a` and `frame_idx_b`, are indices):

| column | min | p05 | p50 | p95 | max | null |
|---|---|---|---|---|---|---|
| `translation_px` | 10.354 | 48.558 | 118.268 | 196.098 | 389.074 | 0 |
| `parallax` | 0.0000 | 0.2215 | 0.6761 | 0.8421 | 0.8909 | 0 |
| `n_matches` | 234 | 272 | 310 | 388 | 594 | 0 |

`translation_px` and the `ransac_thresh_px` behind `parallax` are both in the analysis grid
(`_ANALYSIS_WIDTH = 480`), not source pixels — see `compute_translation`'s docstring for why
the two do not convert by the resize factor.

Serialization checks: the file contains no bare `NaN` token, and `json.load` of it compares
equal to the in-memory report.

### What the distributions look like

Every one of the 2388 frames and 2364 pairs produced a number; nothing on this clip was
unmeasurable, so there are no nulls anywhere. `blur` occupies a narrow band — the p05-to-p95
span is 0.063 on a `[0, 1]` scale — while `laplacian`, which is meant to track the same
property in the opposite direction, spans a factor of 3.2 over the same frames; the two
disagree about how much this clip varies. Exposure sits well below the mid-grey point, with
`exposure_median` (p50 58) consistently under `exposure_mean` (p50 82), the ordering a
right-skewed brightness histogram produces. Clipping is present at both ends, and the high end
is the larger and more variable of the two: `clipped_high_frac` reaches 0.0563 where
`clipped_low_frac` tops out at 0.0043. Motion is continuous rather than bursty — at a
24-frame stride the median pair moves 118 px of the 480 px analysis grid, and the full range
runs from 10 px to 389 px with no gap. `parallax` covers most of its available range, median
0.676, with 514 of 2364 pairs at 0.8 or above and exactly two pairs at 0.0. Those two zeros
are the only degenerate entries in the run, and they are a statement about those two pairs of
this clip, not about the metric.

---

## Refactor parity

The report code arrived alongside a split of one 813-line `sampling.py` into three modules:

| module | lines | role |
|---|---|---|
| `video.py` | 254 | decode and probe |
| `qa.py` | 412 | measure |
| `sampling.py` | 527 | select |

`sampling.py` is smaller than the file it came from but is not merely the remainder — it kept
its selection logic and lost the decode and measurement halves.

### Frame selection is byte-identical

Fifteen commits touched `collab_splats/preproc/`. The check below was run on `da73454d`
before any of them, again after the `video.py` extraction (plan Task 1 Step 7), again after
the `qa.py` extraction (Task 2 Step 7), and again at the end:

```bash
/opt/venv/reconstruction/bin/python -c "
from collab_splats.preproc import sample_frames
import hashlib, numpy as np
frames, idx = sample_frames('data/tutorial/tutorial_example-video.mp4', max_frames=20, method='uniform')
print('indices:', [r['frame_idx'] for r in idx])
print('digest:', hashlib.sha256(np.asarray(frames).tobytes()).hexdigest()[:16])"
```

All four runs returned the same thing:

```
n_frames: 20
indices:  [0, 129, 250, 379, 500, 631, 752, 878, 1002, 1134,
           1259, 1383, 1511, 1636, 1762, 1883, 2013, 2139, 2264, 2387]
digest:   b8d8bc70f766c466
```

Identical indices means selection did not change; an identical SHA-256 over the decoded pixel
buffer means decoding did not either.

### Import graph

`video.py` imports no sibling — `grep -n 'from collab_splats' collab_splats/preproc/video.py`
prints nothing, so the one-way `sampling.py → qa.py → video.py` layering has no cycle in it.

`scipy.stats` is **not** loaded by `import collab_splats.preproc`, before or after. That was
the specific gate, and it holds.

Import cost did move. Seven runs each, interleaved pre/HEAD/pre/HEAD so that drifting machine
load cannot land on one side, and with the editable-install finder removed from `sys.meta_path`
so that `sys.path` decides which tree is imported. With the finder in place `PYTHONPATH` is
ignored and both measurements silently import HEAD, which is worth stating because it is how
this measurement first came out flat; the provenance of every run below was asserted via
`collab_splats.__file__`.

| | min | median | max |
|---|---|---|---|
| `da73454d`, pre-split | 729 | 877 | 3016 |
| HEAD | 1164 | 1245 | 1424 |

Around +370 ms at the medians. Timing on this box is noisy — the pre-split maximum is a cold
first run, and repeated single measurements elsewhere in this session ranged over a full second
— so the load-bearing comparison is not the medians but the fact that HEAD's *fastest* run,
1164 ms, is slower than every warm pre-split run, the slowest of which was 990 ms. The two
distributions do not overlap.

The cause is visible in the module set: importing `collab_splats.preproc`
now pulls `skimage`, `scipy` and `lazy_loader`, and pre-split it pulled none of the three.
Pre-split `sampling.py` imported only `cv2`, `numpy` and `tqdm` for its image work; the new
`blur` column calls `skimage.measure.blur_effect`, `qa.py` imports it at module level, and
`preproc/__init__.py` re-exports `compute_video_quality` from `qa.py`, so the cost is paid by
every importer of the package whether or not they ever ask for a report. `scipy` arrives as
a dependency of `skimage`, which is why the base package appears while `scipy.stats` still
does not. Deferring that one import into `compute_blur` would recover the difference; it is
recorded here rather than done, because the design did not call for it and
`collab_splats/CLAUDE.md` asks for imports at the top of the file.

### Suite

`7 failed, 1886 passed, 2 skipped in 692.57s`. None of the seven are in `tests/preproc/`, and
none is a regression from the split — the attribution was measured, not inferred:

- Five fail only in this working tree, which carries another session's uncommitted
  `configs/base.yaml` (the whole `loger:` block deleted, `fps 1.0→2.0`, four mesh values
  changed). Re-run in a clean worktree at HEAD, they pass: `2 failed, 5 passed`.
- `tests/wrapper/test_reconstructor.py::test_build_localization_db_runs_when_missing` asserts
  `build.assert_called_once_with(ff, "loma", rec.frames_zarr, top_k=8)`. It passes at
  `da73454d` and was broken by `3b3f1e20 fix(wrapper): propagate overwrite through
  _build_localization_db`.
- `tests/localization/test_local_matcher.py::test_real_xfeat_general_path_matches_pairwise`
  fails on `assert (969 == 967)`. It does not exist at `da73454d` — `d5c836ce` added it after
  the baseline.

The last two are concurrent work on `wrapper/` and `localization/`: one passed at the pre-split
baseline and broke on a later commit that is not ours, the other did not exist at the baseline
at all. `python -m collab_splats.dashboard --smoke` prints `SMOKE PASS`.

---

## Reproduction

```bash
# the report
/opt/venv/reconstruction/bin/python -c "
import logging
from collab_splats.preproc.qa import compute_video_quality
logging.basicConfig(level=logging.INFO)
compute_video_quality('data/tutorial/tutorial_example-video.mp4', output_path='/tmp/vqr.json')"

# the percentile tables
/opt/venv/reconstruction/bin/python -c "
import json, numpy as np
r = json.load(open('/tmp/vqr.json'))
for block, skip in (('frames', {'frame_idx'}), ('pairs', {'frame_idx_a', 'frame_idx_b'})):
    for k, v in r[block].items():
        if k in skip:
            continue
        a = np.array([x for x in v if x is not None], float)
        print(k, np.percentile(a, [0, 5, 50, 95, 100]).round(4), 'null', sum(x is None for x in v))"

# import cost, either tree, provenance asserted
/opt/venv/reconstruction/bin/python -c "
import sys, time, importlib
sys.meta_path = [f for f in sys.meta_path if not type(f).__module__.startswith('__editable__')]
sys.path.insert(0, '/workspace/collab-splats')
import collab_splats
assert collab_splats.__file__.startswith('/workspace/collab-splats'), collab_splats.__file__
t = time.perf_counter(); importlib.import_module('collab_splats.preproc')
print(f'{(time.perf_counter() - t) * 1000:.0f} ms', 'scipy.stats' in sys.modules)"
```
