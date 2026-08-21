# Video quality report — measuring the capture before the reconstruction sees it

Status: implemented; amended 2026-08-21 so every contract below matches the shipped code in
`collab_splats/preproc/qa.py`. Where this document and the code disagree, the code is right.
Parallel to [scene error report](2026-08-20-scene-error-report-design.md), which measures error *after*
reconstruction. This one measures the source video *before* it, on every frame, and joins to that report
by source frame index.

## Goal

Answer one question with numbers: **what did the camera actually record, frame by frame?**

Blur, exposure, clipping, inter-frame translation, and whether that translation carried parallax. One
scalar per frame per channel, shipped raw, keyed by source video index so it joins to `frames.zarr`
records and to `report.json`'s per-frame rows.

The report exists because three questions are currently unanswerable:

1. **Why did the blur gate never fire?** `_DEFAULT_BLUR_THRESHOLD = 50.0` (`preproc/qa.py`) rejects
   almost nothing on real footage. Three candidate causes — Laplacian variance is resolution-dependent
   (it is computed on a 480 px-wide gray, so a fixed `50.0` means different things per source), it
   confounds texture with sharpness (a sharp white wall scores low), and sensor noise inflates it (a dark
   grainy frame scores *sharper*). The report emits both blur metrics on identical frames, so the scatter
   distinguishes them.
2. **Was a sharper frame available nearby?** The sampler picks the sharpest gate-passing frame in a
   window and records only the winner. Scoring every frame makes the sampler auditable against the
   frames it declined.
3. **Does capture quality predict reconstruction error?** Only a join answers this. Neither report can
   on its own.

### What each channel uniquely covers

Channels earn their place by covering a distinct reconstruction failure. This table is the argument for
the module's shape; anything not in it was cut.

| Reconstruction failure | Covered by |
|---|---|
| Motion blur | `blur` — measured directly |
| Bad exposure, crushed blacks, blown highlights | `exposure_*`, `clipped_*_frac` |
| **Insufficient baseline between the frames actually used** | `translation_px` at reconstruction stride |
| **Pure rotation — no parallax, geometry undetermined** | `parallax` |
| **Low overlap — matching will fail** | `n_matches` at reconstruction stride |

Blur and exposure are generic image quality; any video grader measures them. The bottom three rows are
what make this a *reconstruction* report, and they are the failures worth catching before spending an
hour on a scene — a video shot by panning instead of walking looks fine and cannot be triangulated.

## Non-goals

- **No verdict.** No `good`/`usable_with_caveats`/`reshoot`, no threshold table, no advice strings, no
  pass/fail counts. The report ships numbers and their sample sizes. The reader draws the line.
  This extends to sample size: there is no "too few frames to correlate" guard, because that is a verdict.
- **No CLI.** No `main()`, no argparse, no exit codes, no console renderer. Plain functions, callable from
  `Reconstructor`, a script, or a notebook.
- **No change to frame selection.** The quality gate, its metric, its three thresholds, and every
  `frames.zarr` on disk are untouched. The gate and its constants *move file* (into `qa.py`) with no
  edit to a line of their code, and selection is proved byte-identical after each move. See "What the
  gate does *not* become".
- **No automatic wiring into the preproc stage.** Running QA on every frame costs a second full decode
  (~156 s on a 13k-frame GoPro), which would destroy the single-ffmpeg-pass fast path that `uniform`
  and `fps` sampling depend on (14fdc42). The report is called explicitly. See deferred follow-ons.
- **No new plotting.** Every plot in this report's inventory is a matplotlib one-liner over a raw column.
  `preproc/viz.py` gets a separate rework spec, after this report measures real distributions — bin
  choices and axis limits should come from data, not guesses.
- **No histograms, quantile grids, percentile ranks, or stored bin edges.** See "Why raw, not binned".

## Design

### Three modules, seven functions, zero report constants

The report is not bolted onto the frame sampler. `preproc/sampling.py` held four jobs — its own section
dividers admitted it — so building the report first split it into three modules that import strictly
downward:

```
preproc/video.py     decode + probe          imports nothing from preproc
preproc/qa.py        measure a frame/pair    imports video
preproc/sampling.py  select frames           imports video + qa
```

Selection uses measurement, never the reverse, and both use decode. Two facts make the split safe to
land without touching production:

- **Every production caller imports the `collab_splats.preproc` package, not a module inside it.** As
  long as `__init__.py` re-exports the same names, no production file changes. Only two tests imported
  by module path (`tests/preproc/test_sampling.py`, `tests/pointcloud/test_loger_creator.py`).
- **`video.py` imports no sibling.** That is what keeps the layering acyclic rather than merely tidy.

The seven functions live in `collab_splats/preproc/qa.py`:

```python
# per frame
compute_blur(gray, *, h_size=11) -> dict    # blur, laplacian
compute_exposure(gray) -> dict              # exposure_mean, exposure_median, exposure_std,
                                            # clipped_low_frac, clipped_high_frac
compute_frame_quality(bgr) -> dict          # merges both, from one BGR frame

# per pair
match_orb(gray_a, gray_b, *, n_features=1000) -> tuple[np.ndarray, np.ndarray]
compute_translation(pts_a, pts_b) -> float
compute_parallax(pts_a, pts_b, *, ransac_thresh_px=3.0) -> float

# whole video
compute_video_quality(video_path, *, output_path=None, motion_stride=None) -> dict
```

Every tuning value is a keyword argument with a default, and the seven report functions declare **no
module-level constants of their own** — the same discipline the scene error report's plan landed on for
`geometry/metrics.py`. `qa.py` is not constant-free, and saying it was would be wrong: the split moved
the four pre-existing gate values into it, with the gate that owns them. `_ANALYSIS_WIDTH`,
`_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE` and `_EXPOSURE_MIN_STD` keep their names and values
exactly, and sit under a divider that reads *gate-only, report functions take tuning as keyword args* so
a later pass does not reach for them from the report. `_ANALYSIS_WIDTH`, `compute_blur_score` and
`_analysis_gray` are shared by gate and report and sit under their own divider above it.

**`compute_video_quality` is the only new name exported from `collab_splats/preproc/__init__.py`**,
taking the package surface from 7 to 8: `FrameStore`, `sample_frames`, `score_frames`, `get_video_info`,
`extract_frame`, `compute_blur_score`, `check_frame_quality`, `compute_video_quality`. The six
primitives stay at `collab_splats.preproc.qa.*`, where the `qa` automodule block in
`docs/source/api/preproc.rst` documents them. The report is the deliverable; re-exporting its building
blocks would grow the package surface 86% for callers who only want the report.

### Channel 1 — photometric, every frame

One pass of `_iter_frames(video_path)` (`preproc/video.py`), the repo's only decoder. Per frame:

**`compute_blur(gray, *, h_size=11)` returns both blur metrics, deliberately.**

- `blur` — `skimage.measure.blur_effect(gray, h_size=h_size)`, Crete-Roffet perceptual blur, normalized
  to `[0, 1]`. **Higher means more blurred.** `h_size` is the width of the re-blur kernel Crete-Roffet
  compares against; 11 is skimage's own default, and larger values report less blur (measured on the
  noise fixture: `h_size=3` → 0.4132, `11` → 0.1202, `21` → 0.0651). It exists so the deferred
  resolution-sensitivity study has a lever.
- `laplacian` — `cv2.Laplacian(gray, cv2.CV_64F).var()`, the incumbent, via the surviving
  `compute_blur_score`. **Higher means sharper.**

**`gray` must be 2-D; a colour frame raises `ValueError`.** This is not defensive padding. `blur_effect`
defaults to `channel_axis=None`, so it reads a 3-channel array's channel axis as spatial, the slice loop
comes out empty and every axis divides by zero — measured, `blur_effect(bgr)` → `nan` while
`cv2.Laplacian(bgr).var()` → 108108.29, bit-identical to the gray value. The row would then be half
valid, and since nan *is* a measurement here (trap 6), it would read as a genuinely failed capture
rather than as a bad call. numpy warns once and its default `once` filter swallows the next 2387.

The two run in *opposite directions*. Both are computed on the same downscaled gray, so their scatter is
a controlled comparison and is the direct evidence for why the gate failed. See trap 1.

**`compute_exposure(gray)` returns mean, median, std, and both clipping fractions.**

Clipping is `mean(gray == 0)` and `mean(gray == 255)`. **Measured at native resolution, before any
downscale** — bilinear averaging pulls saturated pixels off exactly 0 and 255, so a resized frame
systematically under-reports clipping. This is the one metric in the report that does not share the
downscaled gray.

`compute_frame_quality(bgr)` is the merge: native-resolution gray for exposure, `_analysis_gray`
(`preproc/qa.py`, 480 px) for blur, one flat dict of seven Python floats out. The split is measured in
both directions — 300 scattered saturated pixels in a 480×640 frame give `clipped_high_frac` 0.000977
natively and **exactly 0.0** through `_analysis_gray`, while `blur_effect` costs 300.4 ms on a
1920×1080 frame against 39.4 ms at 853×480. What the downscale costs is stated in Runtime and trap 8;
it is not free.

### Channel 2 — motion, strided pairs

Photometric metrics say nothing about whether the camera moved usefully. Two frames `k` apart:

- `match_orb` — ORB (`n_features=1000`) + `BFMatcher(NORM_HAMMING, crossCheck=True)`, returning two
  `(N, 2)` pixel arrays. Named for its method, not its role, because the choice of matcher is the
  contested decision here and the reader should not have to open it to learn which one this is.
  **The seam is the point-pair interface, not this function**: `compute_translation` and
  `compute_parallax` take arrays, so `LocalMatcher.match_images` (`localization/extractors.py`)
  substitutes from a notebook with no change to this module. The returned pair matches
  `MatchResult.query_px` / `.ref_px` — both `(K, 2)` float32 — for exactly that reason.

  It stays in `qa.py` rather than moving to `localization/`: that package imports `torch` and `vismatch`
  at module scope, `preproc/` imports neither, and QA is the pre-reconstruction path that must run
  without a GPU. Nor does it earn a shared home — one caller does not justify inventing a module.
- `compute_translation` — the **median of the per-match displacement norms**, `median(‖pts_b − pts_a‖)`,
  in pixels. No model is fitted: a median over the raw correspondences is robust to a handful of bad
  matches without a second RANSAC, and a fitted similarity transform would report the motion the model
  *chose to explain* rather than the motion the matches show. The pixels are whatever grid `match_orb`
  was handed, which for the report is the analysis grid — see trap 9.
- `compute_parallax(pts_a, pts_b, *, ransac_thresh_px=3.0)` —
  `1 − (homography inliers ÷ fundamental inliers)`, both fit with `USAC_MAGSAC`. A homography explains
  pure rotation and planar scenes exactly; a translating camera viewing 3D structure breaks it. **The
  complement is taken so the name and the number agree: higher means more parallax, 0.0 means the motion
  carried none.** Reporting the raw H/F ratio would invert the reading of every plot drawn from the
  column.

  `ransac_thresh_px` is a **first-order lever, not a detail**, which is why it is on the signature.
  Measured over 99 tutorial pairs at stride 24: 1.0 against 3.0 moves `parallax` by 0.19 on average
  (max 0.49) and *reorders* the pairs, Spearman **0.708** — where 3.0 against 5.0 barely does, 0.053 and
  0.945. Loosening it lets a homography explain more, so `parallax` falls monotonically. MAGSAC's own
  re-run spread at a fixed threshold is 0.0000, so all of that movement is the threshold, not the
  solver. It carries a unit: analysis-grid pixels, the same grid as `translation_px`.

`motion_stride` defaults to `round(fps)` — roughly one second, the separation the reconstruction
actually works at — and is recorded in the report as the resolved integer, never as `None`.

**`motion_stride` must be `>= 1`, and the default is selected on `is not None`, not on truthiness.**
Both halves were bugs found in implementation. `0` is an explicit value: under a truthiness check it
fell through to the fps default and silently measured a stride of 30, so a caller asking for adjacent
frames got one-second pairs and nothing said so. A negative stride is worse than wrong — the partner
index `idx - stride` runs *forward*, so no entry is ever retired from `pending` and the resident frame
buffer grows with the video instead of holding `stride + 1` grays. Both now raise `ValueError` before
the first decode.

**Every stride-dependent quantity is reported in
source-frame and second units, never in "analyzed frames".** The pasted `capture-qa` spec got this wrong
three separate ways: its `exposure_stability`, its `trans_px > 1.0` test, and its reshoot-triggering
`k_gap = 15` were all in analyzed-frame units but compared against fixed limits, so a longer video
silently changed its own verdict by changing its stride.

#### Why ORB, and not the two alternatives already in the repo

**Not the learned matchers.** Measured: loma via `verify()` costs **1346.7 ms/pair** on GPU
([measured report](2026-08-20-scene-error-report-measured.md)); ORB costs ~30 ms/pair on CPU. That is
**45×**, and it is the smaller reason. The larger one: QA runs *before* reconstruction, on a machine that
may have no GPU and no weights — and a learned matcher would *succeed* on exactly the degraded frames
whose degradation this report exists to detect. ORB failing is signal. LoMa not failing is not.

**Not `OpticalFlowFrameSelector`.** It already computes LK flow per frame and looks like free reuse. It
is not, because it measures flow from the **last accepted keyframe**, and `accept_frame` resets that
baseline on every selection:

1. *Sawtooth, not signal.* Disparity climbs from ~0 after each acceptance, then resets. The value at
   frame *k* encodes time-since-last-keyframe, not camera speed.
2. *Self-referential.* The selector accepts *because* disparity crossed `min_disparity`, which resets
   disparity — so the distribution is bounded by the selector's own threshold and describes the selector,
   not the capture.
3. *Tuning-dependent.* Change `min_disparity` and every value changes.

**Not bare LK at stride 1 either.** Consecutive-frame displacement is the *cause* of motion blur, and
blur is already measured directly — so it adds a covariate of an existing column while covering none of
the three reconstruction failures in the goal table. At a 1 s baseline, pyramidal LK
(`winSize=21, maxLevel=3`, `_LK_PARAMS` in `preproc/sampling.py`) tops out around ~170 px displacement
at 480 px width and fails *silently* on exactly the fast-motion frames that matter most. ORB's failure
surfaces as
`n_matches → 0`, which is a reading.

### Why raw, not binned

The scene error report keeps exactly one histogram, and only because its per-pixel residuals are
`N_pairs · H · W` values that cannot be shipped. **This report has one scalar per frame per channel.** A
13k-frame GoPro is 13k floats per column; the raw data *is* shippable, so shipping it is strictly more
capable than shipping counts:

| Plot | Derivation |
|---|---|
| Histogram of any metric, **any bins, chosen at plot time** | `plt.hist(blur, bins=50)` |
| ECDF | `plt.plot(np.sort(v), np.linspace(0, 1, len(v)))` |
| Any metric vs wall-clock time | `frame_idx / fps` on x |
| Exposure mean ± std band | `exposure_mean`, `exposure_std` |
| Clipping over time, stacked | `clipped_low_frac`, `clipped_high_frac` |
| **Blur vs Laplacian scatter** — the gate diagnostic | both columns, same frames |
| Camera path-length proxy | `np.cumsum(translation_px)` |
| Pure-rotation segments | `parallax → 0.0` |
| **Sampler audit** — selected vs declined frame quality | join `frame_idx` ↔ `FrameStore.frame_indices()` |
| **Blur vs depth error** | join `frame_idx` ↔ `report.json` per-frame rows |

A stored histogram supports the first row only, at bins fixed months earlier, and supports neither join.
Quantile grids, percentile ranks, and cumulative curves are cut for the same reason: all are one numpy
call over a column the reader already has.

**Correlations are cut by that same rule, and an earlier draft of this spec got it wrong.** It shipped
two Spearman coefficients — `rho(blur, laplacian)` and `rho(translation_px, blur)` — on the grounds that
they were findings rather than conveniences. They are not: both inputs ship raw and in full, so each rho
is one line for the reader to compute, which is exactly the argument that removed the quantile grid.
Keeping them also meant a `_spearman` wrapper, the shape of `describe()`/`_distribution` helper this
design already refused. Dropping them is what keeps SciPy out of `preproc`: `import scipy.stats` costs
**1160 ms** against the whole package's **1287 ms**, and once `sampling.py` imports `qa.py` that cost
lands on the dashboard's fast-bind path. Measured on the tutorial video, for reference only — these do
**not** ship:

```
rho(blur, laplacian)       -0.615   n=2388
rho(translation_px, blur)  +0.366   n=2364
```

The honest cost is size: ~1.1 MB of JSON for a 13k-frame video. Acceptable, and the reason the payload
is **columnar** (`{"frames": {"blur": [...], ...}}`) rather than a list of dicts — row-of-dicts repeats
eight key names 13,000 times and triples the file.

### Report shape

`video_quality_report.json`, written when `output_path` is given. `compute_video_quality` returns the
same dict whether or not a path was passed. Two branches, and these are the complete key sets:

```
available: true
  video:   path, mtime, total_frames, fps, duration_s, width, height
  params:  motion_stride                                          (the resolved int, never None)
  frames:  frame_idx, blur, laplacian,
           exposure_mean, exposure_median, exposure_std,
           clipped_low_frac, clipped_high_frac                    (columnar, length N)
  pairs:   frame_idx_a, frame_idx_b, n_matches,
           translation_px, parallax                               (columnar, length ~N/stride)

available: false
  reason                                                          (and nothing else)
```

`params` holds `motion_stride` alone. **`n_features` is not a `compute_video_quality` parameter** — it
lives on `match_orb`, where it is a genuine knob, and on the orchestrator it was a pass-through that is
always default in real use. There is no `correlations` block; see "Why raw, not binned".

Column names carry units where a quantity has one (`translation_px`, `*_frac`); the functions that
produce them keep the short names above.

**nan → null is inlined at the two sites that need it, and `clean_for_json` is deliberately not
reused.** Only `translation_px` and `parallax` can be non-finite, so the conversion is two list
comprehensions in the payload literal. `geometry.verification.clean_for_json` was the obvious reuse and
is the wrong one twice over: `verification.py` does `import pycolmap` at module scope, which would drag
a heavy optional dependency onto the pre-reconstruction path, and it tests `np.isnan`, so an inf would
slip through unconverted. `np.nan_to_num` is not a substitute either — its `0.0` fill on
`translation_px` asserts that the camera held perfectly still, the exact opposite of "this pair failed
to match". A bare `NaN` from `json.dumps` is not an option: no strict parser accepts it.

**Both nan columns have more than one cause, and a reader must be able to tell them apart.**

`translation_px` is nan exactly when `match_orb` returned zero correspondences, which happens two ways:
either frame produced no ORB descriptors at all (a flat, textureless frame — `detectAndCompute` returns
`None`), or descriptors existed on both sides but `crossCheck` left no mutual match. `n_matches == 0`
distinguishes neither from the other, but it does confirm the cause is the match set.

`parallax` is nan for **three** distinct reasons, and only the first is visible in `n_matches`:

1. **Fewer than 8 correspondences** — the linear 8-point algorithm's minimum. (MAGSAC's 7-point solver
   does return an F at exactly 7, and OpenCV raises below that, but 8 is the floor this reports
   against.) Always accompanied by a small `n_matches`.
2. **The fundamental fit kept zero inliers.** Reachable with a *full* match set — hit 39 times in a
   5,977-trial sweep, on pairs carrying duplicated keypoint locations. A 300-match pair reporting
   `parallax: null` looks like a contradiction in the file and is not.
3. **OpenCV could not fit at all.** USAC *asserts* on configurations it cannot estimate rather than
   returning an empty model, and it does so at **any** correspondence count, not only below the
   8-point floor. Measured on `tiny_video` frames 40/41: 720 matches, 97.5% of them zero-displacement,
   `findHomography` fine, `findFundamentalMat` raising at `estimator.cpp:353`. Unhandled, that one pair
   discarded all 60 frames of photometry and the other 57 pairs — so the `cv2.error` is caught and
   reported as nan. Any tripod shot or paused segment reaches it.

In all five cases **the failure is itself the measurement**, not an error, and null preserves it. Trap 6
is the rule; this is its full enumeration.

If the video yields zero frames, the report is `{"available": false, "reason": "..."}` — same contract as
the scene error report's channels, and the only two keys present on that branch. **The reason names the
actual condition**: a path that does not exist reports `file does not exist: <path>`, not
`no frames decoded`, which sent a reader hunting a codec problem when the real fault was a typo.

Nothing raises *for an unavailable video*. A broken **environment** is a different thing and does raise:
with ffmpeg off `PATH`, `_require_ffmpeg` raises `RuntimeError` out of `get_video_info` and straight out
of `compute_video_quality`. That is correct — a missing decoder is not a property of the video — but it
means "nothing raises" is not true unqualified, and `motion_stride < 1` raises `ValueError` as well.

JSON, not Parquet. Measured on the 2388-frame tutorial report: JSON 632,651 B, gzipped JSON 150,930 B,
Parquet+zstd 172,387 B across **two** files. Parquet loses to gzipped JSON at this row count, splits
`frames` and `pairs` into separate files, has nowhere natural for the `video`/`params` metadata, is not
greppable, and adds `pyarrow` plus a third serialization format beside JSON and zarr. It becomes the
right answer only for cross-video queries over a corpus, which is a follow-on.

**Every run is visible on the console.** `compute_video_quality` logs the video, frame count, fps,
resolution and stride before the first decode, draws a `tqdm` bar over the decode pass, and logs elapsed
seconds, frames/s and the written file size after. A 2388-frame video takes ~5 minutes; a silent run of
that length is indistinguishable from a hung one.

The bar belongs to the orchestrator alone. `compute_blur`, `compute_exposure`, `compute_frame_quality`,
`match_orb`, `compute_translation` and `compute_parallax` log nothing — each runs once per frame or per
pair, so a line inside any of them is thousands of lines of spam and a nested bar is worse. This mirrors
`sampling.py`, which puts `tqdm` in the sampler and nothing in the scorers. `tqdm` is already a
`preproc` dependency, so this adds no package. `logger.info` reaches the console only once a handler
exists; a script caller needs `logging.basicConfig(level=logging.INFO)`, while the bar writes to stderr
regardless.

### What the gate does *not* become

An earlier draft moved the quality gate onto the perceptual blur metric and split judgment out of
`check_frame_quality`. Both are dropped. The reason is sequencing, not taste: switching the gate needs a
threshold on a `[0, 1]` perceptual scale, and no such number exists yet — `50.0` is a Laplacian-variance
value and is meaningless there. **Measuring the distribution is this spec's entire job; setting a
threshold from it is the follow-on.** Changing the gate first would be inventing the number this report
was built to supply.

So:

- `check_frame_quality` **survives, byte-for-byte**: same signature, same behaviour, same
  `reject_reason`. An earlier draft had its body thinned to call `qa.compute_frame_quality` so the
  exposure and Laplacian math existed once; that was dropped. It moves file, from `sampling.py` to
  `qa.py`, and nothing more. Thinning it would have routed the gate through `blur_effect`, which is the
  cost `compute_blur_score` exists to avoid, and it would have put a behaviour change inside a refactor
  whose entire proof is that frame selection is byte-identical.
- `compute_blur_score` is **kept**, not deleted, and an earlier draft of this spec was wrong to absorb
  it. `compute_blur` *calls* it, so the Laplacian still has exactly one implementation — but the gate
  must not pay for `blur_effect`. Measured at the gate's own resolution (480×270): `laplacian`
  1.27 ms/frame against `blur_effect` 13.76 ms/frame, on a path `_iter_scored_frames` runs for **every**
  frame. Absorbing it would have made the sampler 11× more expensive to buy nothing. Its export from
  `collab_splats.preproc` and its tests stay too.
- `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD` and `_ANALYSIS_WIDTH` all
  **stay**, with their names and values unchanged. They *move file* — into `qa.py`, with
  `check_frame_quality` and `_analysis_gray` — so the claim that no constant anywhere is added, moved or
  deleted is not true of this design as shipped. They are the gate's policy, and inlining them into
  signatures would not be simpler. `_LK_PARAMS` and `_FEATURE_PARAMS` deliberately did **not** move:
  they sat in the same constants block but belong to `OpticalFlowFrameSelector`, which is selection.
- **No `frames.zarr` is invalidated. No staleness key is added. Frame selection is byte-identical.**
- `viz.py` is untouched: `plot_quality_examples` (`viz.py:109`) filters on `reject_reason` from
  `score_frames`, which still produces it.

## What is reused vs new

An earlier draft of this table had `qa.py` importing `_analysis_gray`, `compute_blur_score`,
`_iter_frames` and `get_video_info` *from `sampling.py`*. That premise is gone. It left quality
measurement living inside a frame-selection module and pointed the dependency the wrong way — selection
uses measurement, not the reverse — so the split described under "Three modules" moved each symbol to
the layer that owns it. The reuse is unchanged in substance; only the address is.

| Need | Supplied by |
|---|---|
| Decode every frame, rotation-corrected | `preproc.video._iter_frames` — the repo's only decoder |
| fps, frame count, dimensions | `preproc.video.get_video_info` |
| ffmpeg presence check | `preproc.video._require_ffmpeg` |
| Downscaled analysis gray | `preproc.qa._analysis_gray`, **unchanged**, at 480 px |
| Laplacian variance | `preproc.qa.compute_blur_score`, **kept and called**, not absorbed |
| Perceptual blur | `skimage.measure.blur_effect` (scikit-image 0.26.0, installed) |
| Robust H and F | `cv2.findHomography` / `cv2.findFundamentalMat`, `USAC_MAGSAC` (opencv 4.13.0, installed) |
| ORB + mutual matching | `cv2.ORB_create`, `cv2.BFMatcher(NORM_HAMMING, crossCheck=True)` |
| Inter-frame displacement | `np.median(np.linalg.norm(...))` — **no** `cv2.estimateAffinePartial2D`, no second model fit |
| Progress bar | `tqdm.auto` — already a `preproc` dependency via `sampling.py` |
| nan → null | two inline comprehensions — **not** `geometry.verification.clean_for_json`, see "Report shape" |
| Rank correlation | **nothing** — correlations do not ship, and SciPy stays out of `preproc` |
| Quantiles, histograms, ECDFs, cumulative sums | **the reader's `numpy`** — not shipped |

No new dependencies, and one fewer than the earlier draft assumed: SciPy is not imported anywhere on
this path. Everything above is present in `/opt/venv/reconstruction`.

**Overlap to hold, not resolve here:** `preproc.sampling.score_frames` already walks every
frame and emits `blur_score`, `exposure_mean`, `exposure_std`, `reject_reason`, plus selector
`disparity`/`rotation`/`histogram_similarity`. It is not absorbed: its purpose is tuning the optical-flow
selector, and its motion signal has the moving-baseline defect described above. Whether the two converge
belongs to the viz rework spec.

## Runtime

Measured per-frame steady state on this machine:

| Operation | Cost |
|---|---|
| `blur_effect` @ 1024 px | 62.3 ms |
| `blur_effect` @ 768 px | 31.7 ms |
| `blur_effect` @ 512 px | 11.6 ms |
| `blur_effect` @ 384 px | 6.2 ms |
| ORB detect, `n_features=1000` | 22.9 ms |
| ORB `BFMatcher` crossCheck, n=1000 | 6.6 ms |

### `blur` is not resolution-invariant — the correction, and why the first measurement lied

An earlier version of this section reported scores of 0.1659/0.1657/0.1671/0.1662 at 1024/768/512/384
and concluded `blur` was **near-invariant**, so that reusing `_analysis_gray` at 480 px "costs nothing".
It flagged itself as weak evidence from one synthetic input and made re-measuring on real footage a plan
task. The re-measurement is done and **the answer is no**.

**The synthetic input was the problem, not the sample size** — and that is the durable part, worth more
than the corrected number. The probe was uniform noise, which has a *flat* power spectrum: every
downscale still carries maximum high-frequency content relative to its own Nyquist, so Crete-Roffet,
which reads exactly that content, barely moves. Real footage is roughly 1/f, so downscaling discards
precisely the fine detail the metric measures. The same ladder over the two inputs:

| input | 1024 | 768 | 512 | 384 | spread |
|---|---|---|---|---|---|
| uniform noise | 0.1200 | 0.1250 | 0.1204 | 0.1207 | **4.2%** |
| real footage | 0.2524 | 0.2322 | 0.2106 | 0.1990 | **26.8%** |

The noise row *reproduces* the original near-invariance finding — it was not a mistake in arithmetic,
it was a mistake about what the probe could answer. The footage row is monotonic and 6× wider. The
generalisation: **a synthetic probe cannot answer a question about spectral content, because choosing
the probe already chooses the spectrum.** Any future sensitivity study on this metric must downscale
real footage, which is also the operation `_analysis_gray` actually performs. Do not try to substitute
*upscaling* a small fixture: interpolating a 320 px noise image up to 1280 invents smooth content, and
the ladder comes back non-monotonic and interpolation-dependent — two independent runs gave 0.1844/
0.3002 and 0.1651/0.1560 at 640/1280.

Measured the right way, by downscaling `data/tutorial/tutorial_example-video.mp4` (1080 wide, 5 frames)
to each width:

| width | 320 | 480 | 640 | 1024 | 1080 (native) |
|---|---|---|---|---|---|
| `blur` | 0.2042 | 0.2128 | 0.2272 | 0.2772 | 0.2719 |

**+30.3% across 480 → 1024 on identical frames**, and the direction is the trap: wider sources are
downscaled further and therefore read *sharper*. See trap 8.

The downscale still earns its place, on time alone rather than on invariance: 300.4 ms native
(1920×1080) against 39.4 ms at 853×480 is **7.6×**, or 12 minutes against 94 seconds over the tutorial
video's 2388 frames. Cheap in time, not free in value — which is the opposite of what "costs nothing"
claimed.

### Whole-video cost

600 frames ≈ 67 s plus decode. 13k frames photometric-only ≈ 156 s; ~6.5 min if ORB ran on every frame,
which is why `motion_stride` exists. Measured end to end on the tutorial video: 2388 frames in 304 s,
127 ms/frame.

The pasted spec's "≤ 60 s" target is not achievable on long footage and is not adopted.

## Validation — a negative control per channel

Each channel must be shown to move when, and only when, the thing it measures moves.

- **Blur** — Gaussian-blur a known-sharp frame at increasing σ. `blur` must increase monotonically;
  `laplacian` must decrease. A metric that does not respond to synthetic blur will not detect real blur.
- **Exposure** — scale a frame toward 0 and toward 255. `exposure_mean` tracks; `clipped_high_frac` rises
  only once pixels actually reach 255. **Run this control at native resolution and again on a resized
  copy** — the resized copy must under-report clipping, which is the evidence for the native-resolution
  rule above.
- **Translation** — shift a frame by a known pixel offset. `compute_translation` must recover it.
- **Parallax** — the discriminating control, and the one that can silently pass. It needs **three**
  cases, not two: a 3D scene under translation must rise well above 0 (measured 0.807), a pure rotation
  of that same scene must sit at 0.0, and a **planar** scene under real translation must *also* sit at
  0.0 (measured 0.000, with `translation_px` 50.0). A control testing only the first two cannot
  distinguish "correct" from "always returns 0.0", and a control omitting the third leaves the illusion
  that `parallax` is self-sufficient. Assert all three so a later pass cannot delete one.
  **Every parallax assertion is a wide inequality, never an equality** — MAGSAC is randomized, and
  identical input gave 0.793 and 0.807 on consecutive runs.
- **Report** — a video whose frames are all identical must produce `n_matches` collapsing and nan
  translations serialized as null, not an exception. So must a paused or tripod segment, which reaches
  the USAC assert with a full match set.
- **Selection parity** — `sample_frames` on `data/tutorial/` must return identical frame indices and a
  byte-identical pixel digest before and after the three-module split. This is the guard on the "no
  change to frame selection" non-goal, and it is what licenses moving ~380 lines out of `sampling.py`:
  the refactor is only safe if it is provably invisible. Run it after **each** extraction, not once at
  the end — an index mismatch means selection changed, a digest mismatch means decoded pixels did.

## Implementation principles

- Functions, not classes. No state to carry.
- Existing packages over new code: skimage, opencv, numpy, tqdm. Nothing here is a reimplementation.
  (SciPy was on this list and is not any more — see "Why raw, not binned".)
- The seven report functions declare zero module constants; tuning values are keyword arguments with
  defaults (`h_size`, `n_features`, `ransac_thresh_px`, `motion_stride`). The four pre-existing gate
  constants move file with the gate and keep their values.
- `compute_` prefix, and names state quantity plus unit (`translation_px`, `clipped_low_frac`).
- No wrapper layers: no `describe()`, no `_distribution`, no `_cumulative`, no `_spearman`, no
  `SCHEMA_VERSION`, no quantile grid, no small-sample guard.
- Delete what is obsoleted — but verify it *is* obsoleted first. `compute_blur_score` was slated for
  deletion on this principle and survives, because measurement showed the gate cannot afford its
  replacement. The principle is "delete dead code", not "delete code that resembles new code".
- Ship raw; derive nothing the reader can derive.

## Traps

1. **`blur` and `laplacian` point in opposite directions.** Higher `blur` is worse; higher `laplacian` is
   better. Any threshold, sort, or `argmax` written against the wrong one inverts silently — a sampler
   using `max(blur)` would confidently select the blurriest frames.
2. **`blur` saturates at 1.0 on sparse detail, not only on blur.** Crete-Roffet works by re-blurring and
   measuring how little changes, so it maxes out whenever there is little high-frequency content left to
   *destroy* — sharp or not. Measured: a flat field scores `blur=1.0` at `laplacian=0.00`, a smooth
   gradient `1.0` at `0.41`, and a **single perfectly sharp edge** `1.0` at `406.41`. A maximally sharp
   image can read maximally blurry. Reading `blur` alone as softness inverts on any low-texture frame,
   which is exactly why `laplacian` ships beside it rather than instead of it: a high `blur` next to a
   high `laplacian` means detail is *sparse*, not that the frame is soft.
3. **`parallax` is the complement of the H/F ratio, not the ratio.** Higher means more parallax. Anyone
   "simplifying" the function to return the raw ratio inverts every plot drawn from the column.
4. **Clipping must be measured before downscaling.** Any refactor that hoists one shared gray to the top
   of the loop for speed breaks this, and no test catches it unless the resized-copy control exists.
5. **`frame_idx` is the source video index**, zero-padded to 6 in `frame_XXXXXX` names, and is the join
   key to `frames.zarr` and `report.json`. It is not a row position. Positional indexing into the
   report's columns and into `FrameStore` rows are different things.
6. **nan is a measurement.** A nan `translation_px` means the pair produced no correspondences; a nan
   `parallax` means one of three things, only one of which is visible in `n_matches`. Both are the
   interesting cases, and dropping nan rows before plotting silently deletes the worst pairs.
   "Report shape" enumerates every cause — read it before writing any filter over these two columns,
   because the intuitive one ("nan means too few matches") is true of `translation_px` and false of
   `parallax`.
7. **`n_matches → 0` is ambiguous without `translation_px`.** Large inter-frame motion and unusable
   frames both produce it. Ship `n_matches` on every pair row so the two stay separable.
8. **`blur` is not comparable across videos of differing source width.** `_analysis_gray` scales by
   `min(1.0, 480 / W)`, so a 1080p source is measured at 480 px while a 320 px source is measured
   natively at 320 — and Crete-Roffet is *not* resolution-invariant on real footage, however invariant
   it looks on synthetic noise (see Runtime). Wider sources are downscaled further and read **sharper**:
   +30.3% from 480 to 1024 px on identical frames. Every video wider than 480 px is mutually
   comparable; anything narrower is measured natively and is not. `laplacian` and every exposure column
   are unaffected, being native-resolution. Record it, do not "fix" it — normalising would mean
   inventing a resolution the report exists to observe.
9. **`translation_px` is in analysis-grid pixels and does not rescale to source pixels.** `match_orb`
   reports in whatever grid it was handed, and the report hands it `_analysis_gray` output, so the
   column is `_ANALYSIS_WIDTH` pixels. Multiplying by the resize factor does **not** recover the native
   number, because ORB detects a different keypoint set at each resolution: measured on the tutorial
   video (1080×1920 portrait, factor 2.25), native-over-analysis is 2.37 on one pair and 3.28 on
   another. Comparable within a report, not across videos of differing width — the same shape of caveat
   as trap 8's blur, for the same reason.
10. **A planar scene under real translation reads `parallax == 0.0`, exactly like a pure rotation.** A
    homography explains a plane regardless of camera motion. `translation_px` is the disambiguator:
    high translation with zero parallax means a flat scene or a pan; low translation with zero parallax
    means the camera did not move. **Neither column is interpretable alone.** Measured: 3D scene +
    translation → 0.807 (translation 51.8 px); 3D scene + 5° rotation → 0.000 (45.1 px); planar scene +
    translation → 0.000 (50.0 px).
11. **MAGSAC is randomized.** The same input gave `parallax` 0.793 and 0.807 on consecutive runs. Never
    assert an exact parallax value — every assertion is a wide inequality.
12. **A scene cut reads as large confident motion, not as a failure.** `crossCheck` makes the match sets
    mutually injective, which is what RANSAC wants, but it bounds nothing about whether the two frames
    show the same scene. Two independent noise images yield 373 mutual matches at a median displacement
    of 92 px, against 539 at 17 px for a true 17 px shift — so neither `n_matches` nor a nan flags it,
    and `parallax` comes back high and stable across MAGSAC draws. The column that would separate them
    is descriptor distance: median Hamming 80 against 32. It is **not** in the shipped schema. Adding it
    is a follow-on, not a fix — it widens `match_orb`'s return past the 2-tuple that
    `compute_translation` and `compute_parallax` both consume — but the limitation is real and belongs
    here.
13. **`OpticalFlowFrameSelector` looks like free motion reuse and is not.** Its baseline moves with
    selection; see "Why ORB". A future simplification pass will find this and should read that section
    first.

## Deferred follow-ons

- **Set a gate threshold on `blur`** from measured footage, then switch `check_frame_quality`'s metric.
  That change *does* invalidate every `frames.zarr` — deliberately out of scope here.
- **`preproc/viz.py` rework** — consistent axes and units, report-driven, replacing the four ad-hoc
  plotters; possibly absorbing `score_frames`. Own spec, written after this report measures real
  distributions.
- **Wire into the preproc stage** if the second decode pass can be avoided — most likely by folding QA
  into the optical-flow path, which already decodes every frame, while leaving the `uniform`/`fps` fast
  path alone.
- **Join study**: `blur` and `parallax` against `report.json` depth error, to test whether capture
  quality predicts reconstruction error at all. This is the question that motivates both reports and
  neither answers alone.
- **Descriptor distance on the pair rows**, so a scene cut separates from fast motion (trap 12). It
  widens `match_orb`'s return past the 2-tuple both consumers take, so it is a schema change, not a fix.
- **A resolution-sensitivity study for `blur`** using `h_size` and real footage at several widths, which
  is the lever trap 8 would need before any cross-width normalisation could be justified.
- **Parquet, for a corpus.** Rejected for one video (see "Report shape") and the right answer only once
  the question becomes cross-video queries over many reports.
- **Loop pairs.** Only sequential `(i, i − stride)` pairs are measured.
