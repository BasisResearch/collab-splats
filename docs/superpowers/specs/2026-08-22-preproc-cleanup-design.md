# preproc cleanup — design

**Date:** 2026-08-22
**Scope:** `collab_splats/preproc/` and its callers
**Status:** design, awaiting approval

First of a planned series of per-module cleanup passes. Goal: remove
overengineering, remove duplicated implementation, cut prose docstrings down to
something a human reads, and push constants out of module scope into function
parameters.

---

## 1. Motivation

`preproc` has accreted four overlapping problems:

- **Two dispatch layers** pick a sampler: `Reconstructor._extract_frames`
  branches on a string, then `sample_frames` branches on the same string again.
- **Duplicated decode**: `video._iter_frames` and `video._iter_selected_frames`
  are the same rawvideo-pipe body with one filter difference; `video._probe_dims`
  is a subset of `video.get_video_info`; `video._seek_frame` and
  `video.extract_frame` differ only in who probes.
- **Module-scope constants** encode policy where a caller cannot reach it:
  `_ANALYSIS_WIDTH`, `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`,
  `_EXPOSURE_MIN_STD`, `_VALID_PROBE_MAX`, `_SELECT_THRESHOLD`,
  `_ROTATION_THRESHOLD_DEG`, `_LK_PARAMS`, `_FEATURE_PARAMS`.
- **Dead code**: `FrameStore.is_stale` / `_STALENESS_KEYS` / `records()`,
  `sampling.score_frames`, `sampling._progress_reporter`,
  `sampling._combine_scores`.

Confirmed dead by call-site audit (no production caller anywhere in the repo):
`FrameStore.is_stale`, `FrameStore._STALENESS_KEYS`, `FrameStore.records()`,
`video._seek_frame` (one test call site). `Reconstructor` reuses `frames.zarr`
by **existence only** (`reconstructor.py:650`), which is what makes `is_stale`
unreachable rather than merely unused.

---

## 2. The organising decision: measurement and selection separate

Today the quality gate is inline in the sampler: every candidate frame is
decoded, greyed, blur-scored, and accepted or rejected on the spot
(`sampling.py:335`, `sampling.py:447` call `qa.check_frame_quality`). The
sampler both measures and decides.

`qa.compute_video_quality` already measures the whole video properly —
per-frame photometry plus per-pair motion — and is **report-only** (no
thresholds, no verdicts, per the binding rule set when it was built). It is
currently wired into nothing.

**After this pass, selection reads the report.** The preproc stage becomes two
steps:

1. **Measure** — `compute_video_quality` runs over the source video and writes
   `video_quality_report.json` into the scene directory. Mandatory. Reused by
   existence, like `frames.zarr`.
2. **Select** — the sampler takes that report and uses it to filter frames.
   It never re-measures.

This deletes the inline gate (`check_frame_quality` and its three threshold
constants) and gives the user one artefact that explains why every frame was
kept or dropped.

**The report-only rule survives.** The quality-filter predicate lives in
`sampling.py`, not `qa.py`. `qa` measures; `sampling` decides. Nothing writes a
verdict into the report JSON.

### 2.1 How each sampler uses the report

- `sample_fps` / `sample_uniform` — compute target index `t`, then take the
  window `t ± search_radius` (default 3, previously `_VALID_PROBE_MAX`) and pick
  its winner by the existing key `(usable, laplacian)`: prefer a
  qualifying frame, break ties on sharpness, and fall back to the sharpest frame
  in the window when none qualifies, so the count stays exact. The target index
  is **not** privileged — today's code already maximises over the whole window
  including `t`, and parity requires keeping that.

  Because the report supplies `laplacian` for every frame, the winner is now
  chosen **before** any decode. Today the sampler decodes the union of all
  windows — roughly `search_radius * 2 + 1` times more frames than it keeps —
  and greys each one. After this pass it decodes exactly the frames it returns,
  in one `select=` pass, and greys nothing. Records take `blur_score` from the
  report's `laplacian` column instead of recomputing it.
- `sample_optical_flow` — streams the video with its stateful LK selector.
  Frames the filter rejects are skipped as they arrive: the selector never scores
  them and never adopts one as its keyframe reference, so it moves on to the
  next candidate. This is the user's requirement verbatim:

  > "we can eliminate frames from what optical flow is selecting while running
  > (e.g., the frames are already disqualified, and therefore optical flow
  > should consider a different frame if its already below threshold)"

  Optical flow still runs its own decode pass, because its LK disparity is
  measured against a *moving* keyframe reference, which the report's fixed-stride
  pairs cannot supply. It does **not** recompute blur or exposure — those come
  from the report.

### 2.2 Where the report lives, and when it does not exist

`video_quality_report.json` is written beside `frames.zarr`
(`frames_zarr.parent`), so it travels with the scene and is reused by existence
exactly like `frames.zarr` — the same rule that made `FrameStore.is_stale`
unreachable.

**Image-directory input has no report and needs none.** That branch of
`_extract_frames` never samples: it reads every image in the directory and
writes them straight to `frames.zarr`. No sampler runs, so no report is
required. Unchanged by this pass.

### 2.3 Cost of making the report mandatory

Measured on `data/tutorial/tutorial_example-video.mp4` (1080×1920, 2388 frames).
Defaults are **serial** — see §5.2 for why `workers` does not auto-scale.

| stage | before | after (default, `workers=1`) | after (`workers=4`) |
|---|---|---|---|
| `compute_video_quality` | 330 s (measured, unwired) | **≈163 s** | **≈53 s** |
| `sample_uniform` (100 frames) | ~20 s (measured) | lower — decodes ~7× fewer frames (§2.1), not measured | same |
| preproc stage, first run | ~20 s | **≈183 s** | **≈73 s** |
| preproc stage, re-run | ~0 s | ~0 s (both artefacts reused) | ~0 s |

**First-run preproc gets ~9× slower by default.** That is the honest price of
a mandatory report. It is paid once per scene and buys a per-frame explanation
of every selection decision plus the removal of the inline gate — but it is a
large enough number to deserve a deliberate decision rather than an assumption.

**Resolved:** accept the serial cost, and surface the worker count in config so
a machine that has cores can opt into them. The report keeps exactly one shape —
splitting it photometry-only to save ≈66 s once per scene buys less than it
costs in artefact complexity.

`configs/base.yaml`, under `preproc:` (renamed — §4.1):

```yaml
  n_workers: 4                # quality report only: decode+measure this many frame
                              # ranges in parallel. 1 = serial. Measured on a 2388-frame
                              # 1080x1920 video: 2 -> 1.90x, 4 -> 3.04x (best), 6 -> 2.95x,
                              # 12 -> 2.06x. NOT auto-derived — os.cpu_count() reports host
                              # cores inside a container and ignores the cgroup quota.
                              # Set to 1 while a GPU eval run is using the machine.
```

This is the **only** new config key in the pass. It reaches
`compute_video_quality(workers=...)` through `Reconstructor._extract_frames`;
nothing else in preproc reads it.

**The shipped default and the function default differ deliberately.**
`compute_video_quality(workers: int = 1)` stays serial, because a library
function that silently spawns processes is a trap for every non-pipeline caller
(notebooks, evals, the dashboard worker). `configs/base.yaml` ships `4` because
the pipeline is the one caller that knows it owns the machine for the duration
of a reconstruction. Opting *down* is a one-line config edit the comment
documents.

---

## 3. Module-by-module design

### 3.1 `video.py` — decode only, three public functions

| before (8) | after (5) |
|---|---|
| `_require_ffmpeg` | `_require_ffmpeg` (unchanged) |
| `_rotation_degrees` | `_rotation_degrees` (unchanged) |
| `get_video_info`, `_probe_dims` | `get_video_info(path, *, count_frames=True)` |
| `_iter_frames`, `_iter_selected_frames` | `iter_frames(path, *, indices=None, start=0, count=None)` |
| `_seek_frame`, `extract_frame` | `extract_frame(path, frame_idx, *, info=None)` |

- `get_video_info(..., count_frames=False)` skips the `-count_packets` full
  demux. That is the entire reason `_probe_dims` existed.
- `iter_frames` — one rawvideo pipe, one body, one `finally`, three modes:

  | call | ffmpeg |
  |---|---|
  | `iter_frames(p)` | full decode, no filter |
  | `iter_frames(p, indices=[...])` | `-vf select='eq(n\,i)+...' -vsync 0` |
  | `iter_frames(p, start=s, count=n)` | input-seek `-ss (s-0.5)/fps` before `-i`, then `-frames:v n` |

  `indices` and `start`/`count` are **mutually exclusive** — passing both raises
  `ValueError`. `start`/`count` exist for range-parallel decode (§5.2): a worker
  cannot use `indices` because the `select=` filter still demuxes the whole file
  from frame 0, which is precisely the cost parallelism is meant to divide.
  Each yielded item is `(frame_idx, frame)` so a range worker knows its absolute
  index without arithmetic at the call site.
- `extract_frame(..., info=None)` — pass a pre-probed info dict to hoist the
  probe out of a loop. Kills the private/public split whose only purpose was
  letting `tests/pointcloud/test_loger_creator.py:739` do exactly that; that
  test updates to the public call.

### 3.2 `qa.py` — measurement only, named families

Top of file gets a short section header naming what is measured, one line each:

```
Photometry — per frame: is this frame sharp and correctly exposed?
Motion     — per pair:  how far did the camera move between two frames?
```

Changes:

- **Delete** `check_frame_quality`, `_DEFAULT_BLUR_THRESHOLD`,
  `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD`. The gate is gone (§2).
- **Delete** `compute_blur_score` as a separate function — one `cv2.Laplacian`
  line, folded into `compute_blur` and dropped from the package exports.
- `_analysis_gray` → **public** `analysis_gray(bgr, *, width=480)`. Stays cv2:
  it is a resize and a colour convert on a numpy BGR frame. Routing that
  through kornia would mean a torch tensor, a layout change, and a device
  transfer to replace two cv2 calls — cv2 *is* the library here. Only the
  constant moves out.
- `compute_exposure` reimplemented on a 256-bin `cv2.calcHist` (§5.1) —
  bit-exact, 2.5× faster.
- Every `compute_*` builds its return dict from **named locals with a block
  comment above each**, so a reader sees how each number is produced:

  ```python
  # Mean vs median: mean tracks overall brightness, median resists a blown sky
  exposure_mean = float((hist * levels).sum() / n)
  exposure_median = float(np.searchsorted(np.cumsum(hist), n / 2))
  ```

- Docstrings cut to a one-line summary plus bullets. The measured evidence
  currently carried in prose (blur 0.2128@480 vs 0.2772@1024; ORB 373 matches
  @92px on noise; parallax 1.0-vs-3.0 Spearman 0.708) moves to
  `docs/superpowers/specs/2026-08-20-video-quality-report-measured.md`, which is
  where measured results belong. A one-line pointer stays in the docstring.
- All thresholds and sizes become keyword parameters with defaults:
  `analysis_width`, `blur_h_size`, `n_features`, `ransac_thresh_px`.
- Blank line after every `"""` before the summary text starts.

`compute_video_quality` gains `workers: int = 1` — serial by default, explicit
opt-in for range parallelism (§5.2).

**New:** `load_video_quality(video_path, report_path, *, workers=1, motion_stride=None)`.
Returns the report at `report_path`, computing and writing it first if the file
is absent. Four callers need "a report, reused by existence" —
`Reconstructor._extract_frames`, `dashboard/pipeline._sample`,
`wrapper/splatter`, `evals/datasets._load_video` — and reuse-by-existence is the
rule that makes `is_stale` unnecessary (§2.2). It lives in one function rather
than four `if path.exists()` blocks that can drift apart.

### 3.3 `sampling.py` — three public samplers, no dispatcher

Public surface becomes exactly three functions, one per method:

```python
sample_fps(video_path, *, fps, min_frames=None, max_frames=None, report, ...)
sample_uniform(video_path, *, max_frames, report, ...)
sample_optical_flow(video_path, *, max_frames=None, report, ...)
```

`report` is a **required** keyword on all three. No `None` fallback and no
"measure it myself if absent" branch — that branch is exactly the coupling this
pass removes. A caller without a report calls `compute_video_quality` first.

**Deleted:**

| symbol | why |
|---|---|
| `sample_frames` | second dispatch layer; `_extract_frames` calls the three directly |
| `score_frames` | superseded by the report |
| `_combine_scores` | opaque weighted sum; folded into `OpticalFlowFrameSelector.score_frame` where its two inputs are already in scope |
| `_progress_reporter` | replaced by `collab_splats/utils/progress.py` (§3.5) |
| `_uniform_targets`, `_fps_targets` | two lines of arithmetic each; inlined into their callers |
| `_sample_positions` | shrinks to `_take_frames(path, targets, report, search_radius)` once the inline gate is gone — decode the targets, substitute from the report, build records |
| `_VALID_PROBE_MAX`, `_SELECT_THRESHOLD`, `_ROTATION_THRESHOLD_DEG`, `_LK_PARAMS`, `_FEATURE_PARAMS` | become parameters |

**New:**

```python
def filter_frame_quality(
    report,
    *,
    laplacian_min=50.0,                 # Laplacian variance; below this = soft
    exposure_mean_range=(20.0, 235.0),  # outside = crushed or blown
    exposure_min_std=10.0,              # below = no contrast
    blur_max=None,                      # Crete-Roffet; off by default, see below
) -> np.ndarray:                        # bool per frame, True = usable
    """Per-frame usability mask over the report's photometry columns."""
    f = report["frames"]
    lap, mean, std = (np.asarray(f[k]) for k in ("laplacian", "exposure_mean", "exposure_std"))
    lo, hi = exposure_mean_range

    # Sharp enough: Laplacian variance, not Crete-Roffet blur (which saturates)
    sharp = lap >= laplacian_min
    # Exposed usably: inside the brightness band AND carrying some contrast
    exposed = (mean >= lo) & (mean <= hi) & (std >= exposure_min_std)

    return sharp & exposed
```

The single place a threshold is applied to the report. Called once per sampling
run; every sampler consumes the mask. Indexed by frame index directly —
`compute_video_quality` enumerates every frame, so `frame_idx` is contiguous
`0..N-1`.

**Polarity is positive** (`True` = keep), matching the name: a `filter_*`
function reports what survives, not what is thrown away.

The defaults are the deleted `check_frame_quality` gate's three tests with the
same numbers, applied to the same three measurements.

**One measurement is not identical, and the difference is real.** The old gate
ran on the 480-wide analysis gray: `check_frame_quality` receives
`_analysis_gray(bgr)` and computes `mean`/`std` on it (`sampling.py:335`,
`sampling.py:447`). The report computes **exposure on the native-resolution
gray** and only blur on the 480-wide one (`qa.compute_frame_quality`). So:

| measurement | old gate | report | identical? |
|---|---|---|---|
| `laplacian` | 480-wide | 480-wide | **yes, bit-for-bit** |
| `exposure_mean` | 480-wide | native | no — downscale averages pixels |
| `exposure_std` | 480-wide | native | no — downscale *reduces* std |

Downscaling is a local average, so it pulls values toward the frame mean:
`exposure_std` measured at 480 is systematically **lower** than at native
resolution. A frame sitting near `std >= 10.0` can therefore pass the new filter
having failed the old one. `exposure_mean` moves far less (an average of
averages), and the `(20, 235)` band is wide, so the mean test is effectively
unchanged.

This is not worth "fixing" by re-measuring exposure at 480 — native is the more
correct measurement and it is what the published report already contains. It is
recorded because the parity test (§6) must be allowed to show it rather than be
tuned until it hides it. **Expected outcome: identical indices on the tutorial
video** (its frames are not near the std boundary); if a real video does differ,
the diff must be attributable to this row and nothing else.

**`blur_max` defaults to `None` (disabled) on purpose.** The report carries two
blur measurements running in opposite directions: `laplacian` (higher =
sharper) and `blur` (Crete-Roffet, higher = blurrier). Crete-Roffet **saturates
at 1.0 on any low-detail frame** — a flat field, a smooth gradient and a single
perfectly sharp edge all score 1.0 — so thresholding it alone would discard
sharp frames of plain surfaces. `laplacian` is the sharpness gate; `blur` is
available for a caller who wants it and knows the trap.

`OpticalFlowFrameSelector` **stays**. Audit of cv2/kornia/open3d found no
drop-in: cv2 supplies the pieces it already uses (`goodFeaturesToTrack`,
`calcOpticalFlowPyrLK`, `estimateAffinePartial2D`, `calcHist`/`compareHist`),
but the streaming stateful "is this frame different enough from the last one I
kept" policy is ours and has no upstream equivalent. Changes are de-constanting
and documentation:

- `lk_params` and `feature_params` become dict parameters, defaulting to `None`
  and built inline in the function body with a block comment (a mutable dict
  default is a Python trap, so `None` is the default and the dict is visible at
  the point of use).
- `min_disparity`, `select_threshold`, `rotation_threshold_deg` become
  parameters.
- Every value in the score dict gets a 5–10 word description:

  ```
  disparity            median LK pixel motion since last kept frame
  rotation             in-plane rotation vs last kept frame, degrees
  histogram_similarity intensity-histogram correlation with last kept frame
  score                weighted motion + coverage, >= select_threshold selects
  ```

### 3.4 `frame_store.py`

- **Delete** `_STALENESS_KEYS` and `is_stale` (no production caller; reuse is by
  existence).
- **Delete** `records()` (no caller; `record(i)` is the live accessor).
- Blank lines between logical blocks throughout — the file is currently lumped
  with no visual separation.
- Everything else (`create`, `open`, `frame_idx_from_path`, `__len__`, `image`,
  `image_by_frame_idx`, `images`, `record`, `frame_indices`, `export`) is live
  and stays.

### 3.5 `collab_splats/utils/progress.py` — new

One generator, used across modules, replacing `_progress_reporter` and the ad
hoc `tqdm` wrapping in `qa.py`:

```python
def progress(iterable, *, total=None, desc="", on_progress=None):
    """Wrap an iterable in tqdm, and forward (done, total) to an optional callback."""
```

`on_progress` exists because the dashboard needs a callback, not a terminal
bar. One implementation serves both.

---

## 4. Call-site changes

| file | change |
|---|---|
| `wrapper/reconstructor.py` | `_extract_frames` calls `sample_fps`/`sample_uniform`/`sample_optical_flow` directly; runs `compute_video_quality` first and passes the report |
| `dashboard/pipeline.py` | same three-function switch |
| `wrapper/splatter.py` | `sample_frames` → `sample_uniform` |
| `evals/datasets.py:252` | **bug fix** — `sample_frames(..., method="uniform", fps=fps)` raises `ValueError` unconditionally (`fps` defaults to `1.0`, and `uniform` rejects `fps`). `_load_video` is currently broken. Becomes `sample_uniform(..., max_frames=max_frames)`; the unused `fps` parameter is dropped |
| `preproc/__init__.py` | exports pruned to the live surface: `FrameStore`, `sample_fps`, `sample_uniform`, `sample_optical_flow`, `filter_frame_quality`, `get_video_info`, `iter_frames`, `extract_frame`, `compute_video_quality`, `analysis_gray` |
| `preproc/viz.py` | `plot_frame_scores`, `plot_disparity_sensitivity`, `plot_quality_examples` take report rows instead of `score_frames` output; `plot_disparity_sensitivity` loses its `_combine_scores` import |
| `tests/preproc/*` | rewritten to the new surface |
| `tests/pointcloud/test_loger_creator.py:739` | `_seek_frame(...)` → `extract_frame(..., info=info)` |
| `tests/preproc/test_frame_store.py:43-49`, `tests/wrapper/test_reconstructor.py:200` | two `is_stale` call sites, deleted with their subject |
| `docs/` | preproc user docs updated to the three-sampler surface and the two-step flow |

### 4.1 Config surface

Two changes, both mechanical:

1. **The section is renamed `preprocessing:` → `preproc:`**, matching the module
   it configures (`collab_splats/preproc/`) and the pipeline stage name
   (`Reconstructor._STAGE_DEPS["preproc"]`). `preprocessing` was the only name in
   the config that did not match its code.
2. **One key is added**, `preproc.n_workers` (§2.3).

`frame_selection`, `fps`, `min_frames` and `max_frames` keep their meaning and
their values. Quality-filter thresholds stay signature defaults, not YAML keys —
the report plus `filter_frame_quality`'s arguments already contain that scope.

Rename call sites (historical specs/plans under `docs/superpowers/` are frozen
records and are **not** rewritten):

| file | sites |
|---|---|
| `configs/base.yaml` | section header |
| `configs/loop_closure.yaml` | section header |
| `docs/source/tutorials/03_splats/configs/base.yaml` | section header |
| `configs/README.md` | 6 rows in the config-key tables |
| `wrapper/reconstructor.py` | 2 reads (`config["preprocessing"]` at 656, 705) + 5 references in messages/docstrings |
| `tests/remote/test_rerun.py`, `tests/wrapper/test_reconstructor.py`, `tests/wrapper/test_reconstructor_preprocess.py` | 9 fixture/assertion sites |

**Old configs fail loudly, not silently.** A `run_config.yaml` written before
this pass — including every one already sitting under `environments-processed/`
— still carries a `preprocessing:` block. Deep-merging it over the new base
would leave that block inert and silently substitute defaults, which is exactly
the silent-wrong-value class this repo keeps getting bitten by. So
`Reconstructor.validate_config` raises on a top-level `preprocessing` key:

```python
# `preprocessing` was renamed to `preproc` (2026-08-22) — refuse rather than
# silently ignore the stale block and substitute defaults.
if "preprocessing" in config:
    raise ValueError("config key 'preprocessing' was renamed to 'preproc'")
```

(`validate_config` is a `@classmethod` taking `config`, not `self` —
`reconstructor.py:579`.)

No compatibility shim: the repo takes no legacy fallbacks, and the fix is a
one-word edit the error names.

---

## 5. Speedups

Both were required by the user to be measured before being asserted. They were.

### 5.1 Histogram exposure — 330 s → 192 s

`compute_exposure` was 33.9 ms/frame, of which `np.median` on a 480×854 uint8
array dominated. A 256-bin `cv2.calcHist` gives mean, median, std and both
clipping fractions in one O(N) pass:

```python
# float64: cv2.calcHist returns float32, and a 1080x1920 frame's 2.07M counts
# do not survive it — clipping fractions come out ~3e-8 off.
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
n, levels, cum = hist.sum(), np.arange(256.0), np.cumsum(hist)
mean = float((hist * levels).sum() / n)
# np.median averages the two central order statistics on even N. The naive
# searchsorted(cum, n/2) returns only the lower one — measured 127.5 off on a
# two-pixel frame. Take both.
k_lo, k_hi = (int(n) - 1) // 2, int(n) // 2
median = float((np.searchsorted(cum, k_lo, side="right") + np.searchsorted(cum, k_hi, side="right")) / 2)
std = float(np.sqrt((hist * (levels - mean) ** 2).sum() / n))
clipped_low_frac, clipped_high_frac = float(hist[0] / n), float(hist[255] / n)
```

**Exactness, measured over 400 random uint8 frames** (uniform, constant,
bimodal 0/255, narrow-band), against the numpy path:

| field | agreement |
|---|---|
| `exposure_mean` | exact, 0/400 differ |
| `exposure_median` | exact, 0/400 differ |
| `clipped_low_frac`, `clipped_high_frac` | exact, 0/400 differ |
| `exposure_std` | 66/400 differ, **max 4.3e-14** — summation order only |

So the test pins four fields with `==` and `exposure_std` with `atol=1e-9`.
Asserting `==` on `std` would be a flaky test dressed up as a strict one.

Both correction terms above are load-bearing: without the `float64` cast the
clipping fractions drift ~3e-8, and without the two-order-statistic median an
even-pixel-count frame reports the lower central value instead of the average.
An earlier draft of this spec asserted bit-exactness for all five fields on the
naive form; it was wrong on three of them.

Standalone timing, 1920×1080 uint8: **0.34 ms/frame against 33.5 ms** (98×).

Per-component cost that identified it, on the tutorial video: decode 23.2 ms/f,
`compute_exposure` 33.9 ms/f, `blur_effect`@480 34.3 ms/f, `match_orb`
36.3 ms/f, `compute_parallax` 3.6 ms/f.

### 5.2 Range-parallel decode — opt-in, measured 1.92× at 4 workers (§5.4)

Split the video into contiguous frame ranges, one `ProcessPoolExecutor` worker
per range, each opening its own ffmpeg with an input seek to its range start.
Ranges overlap by `motion_stride` frames so boundary pairs are still measured;
overlapping rows are dropped on merge.

**The thread pin is load-bearing.** cv2 and numpy fan out over every core by
default, so the "serial" baseline is already parallel and naive process fan-out
oversubscribes. Unpinned, workers measured **0.67×** — *slower than serial*.
Each worker must set `cv2.setNumThreads(1)`, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` before doing any work. This goes in the code as a
comment, not just here.

Harness sweep, 2388 frames, stride 24, 96-core host, threads pinned. These
figures set the worker default; the shipped function's own numbers are in §5.4
and are lower:

| workers | wall | speedup | redundant rows |
|---|---|---|---|
| 1 | 191.5 s | 1.00× | 0 |
| 2 | 100.7 s | 1.90× | 24 |
| 3 | 78.2 s | 2.45× | 48 |
| **4** | **62.9 s** | **3.04×** | 72 |
| 6 | 65.1 s | 2.95× | 120 |
| 12 | 92.9 s | 2.06× | 264 |

Throughput saturates at 4 and declines past 6: each worker runs its own
multithreaded ffmpeg, and the per-boundary overlap grows linearly with worker
count. Note the first step is nearly free — 2 workers already return 1.90×, so
even a two-core machine gets most of the available win.

**Function default: `workers = 1`. Shipped pipeline config: `4`. No
auto-derivation in either case.**

The obvious default — scale off `os.cpu_count()` — is wrong here, and not
merely conservative-wrong:

- `os.cpu_count()` reports **host** cores inside a container and ignores the
  cgroup CPU quota. This repo runs in exactly such a container. A derived
  default would read 96 on a box whose quota is 4.
- `os.sched_getaffinity(0)` respects a cpuset but still not a CFS quota, so it
  is not a fix either.
- CLAUDE.md already states the operating rule: *"don't run parallel processes
  during heavy eval runs (OOM risk)"*. A default that silently spawns workers
  violates it.

No probe is trustworthy, so the design does not probe. `workers` is an explicit
parameter, surfaced as `preproc.n_workers` in `configs/base.yaml` (§2.3), and
the table above is the tuning reference. The 2-worker row is what makes this
cheap: even a modest machine gets most of the available win.

**Boundary guard.** Adjacent ranges overlap by `motion_stride` frames, so the
overlapping photometry rows are measured twice — once by each worker, from two
independently seeked ffmpeg pipes. On merge those duplicate rows must compare
**exactly equal**; if they do not, input-seek landed on different frames in the
two workers and every frame index in the report is suspect. Raise on mismatch
with a message naming `n_workers: 1` as the fix. Silent acceptance here would
mis-index the whole report, which is the failure mode the worker-invariance test
(§6) exists to catch.

### 5.3 ORB descriptors computed once per frame, not twice

`match_orb(gray_a, gray_b)` runs `detectAndCompute` on **both** members of every
pair. Every frame is the partner of one pair and the current frame of another,
so ORB runs **twice per frame** across a whole video. `compute_video_quality`
already holds a `pending` dict of grays awaiting a partner; storing
`(keypoints, descriptors)` beside the gray computes each frame's ORB once and
replays the pair half from the stored tables.

Same shape as the `matcher-feature-cache` pass (2026-08-21): split detect from
match, persist the detect half. No new processes, no thread pinning, ~5 lines.

Measured, 400 frames / 376 pairs, stride 24, pinned, decode hoisted out of the
timed region, best of 3. Both paths return identical match totals (115,058):

| | wall | ms/pair |
|---|---|---|
| re-detect (before) | 14.454 s | 38.4 |
| cached descriptors | **10.765 s** | **28.6** (1.34×) |

≈ **22 s** saved over 2388 frames — 12% of the serial run. Modest, but it costs
nothing in complexity, so it is in scope.

It also relocates the bottleneck: detection is ~12 ms of the pair cost and
BFMatcher `crossCheck` is now ~20 ms. `nfeatures=1000` under crossCheck is ~10⁶
Hamming comparisons per pair, and lowering it would be the next real win —
**but it changes the values the report publishes**, which a cleanup pass must
not do. Recorded as owed, not done here.

### 5.4 Where that leaves the numbers

| | serial (default) | 4 workers (opt-in) |
|---|---|---|
| before, unwired | 330 s | — |
| harness prediction (§5.1 + §5.3) | ≈163 s | ≈53 s (3.04×) |
| **measured, shipped function** | **180.1 s** | **93.9 s (1.92×)** |

The measured row is `compute_video_quality` itself on
`data/tutorial/tutorial_example-video.mp4` — 2388 frames, stride 24, 96-core
host. The two reports are byte-identical across worker counts: 2388 frame rows,
2364 pair rows, same values.

Serial landed within 11% of the harness prediction. The 4-worker figure did not
— **1.92× against a predicted 3.04×**. Leading explanation: the harness pins
cv2 and numpy per worker, but nothing pins **ffmpeg's** decode threads, so the
serial baseline is already partly parallel and the process fan-out competes with
it instead of adding to it. That is a hypothesis. Nobody has run the
counterfactual with ffmpeg held to one thread, and the code says so where it
states the number.

End to end through `Reconstructor.extract_frames` on the same video — 100 frames
at `fps: 1.0`, 4 workers — **98.3 s cold, 28.7 s** once
`video_quality_report.json` exists. The report is essentially the entire cost of
preproc; selecting frames out of it is not.

---

## 6. Testing

- **Parity**: `sample_uniform` and `sample_fps` return the same frame indices as
  today's `sample_frames(method=...)` on the tutorial video — recorded against
  the pre-change output, not against a re-run of the new code. The sharpness
  term is bit-identical (§3.3), so any diff is either the exposure-resolution
  row or a genuine regression, and the test's job is to force that distinction
  rather than to be relaxed until it passes.
- **Exposure exactness**: histogram implementation vs the numpy path on random
  uint8 arrays across four distribution shapes — exact equality on `mean`,
  `median` and both clipping fractions; `atol=1e-9` on `std` (§5.1).
- **Worker invariance**: `compute_video_quality(workers=1)` and `workers=3`
  produce identical reports on a short fixture. Catches boundary-overlap merge
  bugs.
- **Quality filter**: a synthetic report with known-bad rows, asserting each
  sampler's response — substitution within radius for fps/uniform, skip-and-
  continue for optical_flow.
- **Deletions**: tests for `is_stale`, `score_frames` and `check_frame_quality`
  are removed with their subjects.
- Full suite green before commit.

---

## 7. Scope

**In:** `collab_splats/preproc/*`, `collab_splats/utils/progress.py` (new),
`wrapper/reconstructor.py`, `dashboard/pipeline.py`, `wrapper/splatter.py`,
`evals/datasets.py`, `configs/*.yaml` + `configs/README.md`, `docs/` preproc
pages, and the tests that touch any of the above: `tests/preproc/*`,
`tests/wrapper/test_reconstructor.py`,
`tests/wrapper/test_reconstructor_preprocess.py`,
`tests/dashboard/test_pipeline.py`, `tests/evals/test_datasets.py`,
`tests/remote/test_rerun.py`, `tests/pointcloud/test_loger_creator.py`.

**Out, recorded as owed:** the tutorial rebuild
(`docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`). It uses
`score_frames` and the old gate, so it breaks with this pass and is rebuilt in a
separate pass — where it can demonstrate the report properly instead of being
patched to compile.

**Not in scope:** any change to reconstruction quality, pose estimation, or the
meaning of an existing config key. This is a cleanup; sampler output on a clean
video is unchanged.

---

## 8. Implementation principles

- Reuse before adding: `iter_frames`, `analysis_gray` and `progress` each become
  the single implementation for what were two or three.
- Delete what this pass obsoletes, in the same pass — no deprecation shims, no
  compatibility aliases for `sample_frames` or `score_frames`.
- No constant in module scope that a caller might reasonably want to change.
- Docstring is one line plus bullets; measured evidence lives in the measured
  report, not the docstring.
- Blank line after `"""` before the summary text.
