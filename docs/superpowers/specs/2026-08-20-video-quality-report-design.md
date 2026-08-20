# Video quality report — measuring the capture before the reconstruction sees it

Status: design, unimplemented.
Parallel to [scene error report](2026-08-20-scene-error-report-design.md), which measures error *after*
reconstruction. This one measures the source video *before* it, on every frame, and joins to that report
by source frame index.

## Goal

Answer one question with numbers: **what did the camera actually record, frame by frame?**

Blur, exposure, clipping, inter-frame translation, and whether that translation carried parallax. One
scalar per frame per channel, shipped raw, keyed by source video index so it joins to `frames.zarr`
records and to `report.json`'s per-frame rows.

The report exists because three questions are currently unanswerable:

1. **Why did the blur gate never fire?** `_DEFAULT_BLUR_THRESHOLD = 50.0` (`sampling.py:34`) rejects
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
- **No change to frame selection.** The quality gate, its metric, its three constants, and every
  `frames.zarr` on disk are untouched. See "What the gate does *not* become".
- **No automatic wiring into the preproc stage.** Running QA on every frame costs a second full decode
  (~156 s on a 13k-frame GoPro), which would destroy the single-ffmpeg-pass fast path that `uniform`
  and `fps` sampling depend on (14fdc42). The report is called explicitly. See deferred follow-ons.
- **No new plotting.** Every plot in this report's inventory is a matplotlib one-liner over a raw column.
  `preproc/viz.py` gets a separate rework spec, after this report measures real distributions — bin
  choices and axis limits should come from data, not guesses.
- **No histograms, quantile grids, percentile ranks, or stored bin edges.** See "Why raw, not binned".

## Design

### One module, seven functions, zero constants

`collab_splats/preproc/qa.py`.

```python
# per frame
compute_blur(gray) -> dict          # blur, laplacian
compute_exposure(gray) -> dict      # mean, median, std, clipped_low_frac, clipped_high_frac
compute_frame_quality(bgr) -> dict  # merges both, from one BGR frame

# per pair
match_orb(gray_a, gray_b, *, n_features=1000) -> tuple[np.ndarray, np.ndarray]
compute_translation(pts_a, pts_b) -> float
compute_parallax(pts_a, pts_b) -> float

# whole video
compute_video_quality(video_path, *, output_path=None, motion_stride=None, n_features=1000) -> dict
```

Every tuning value is a keyword argument with a default. `qa.py` declares **no module-level constants** —
the same discipline the scene error report's plan landed on for `geometry/metrics.py`. It adds, moves,
and deletes no constant elsewhere either: `sampling.py`'s four stay as they are, and `_analysis_gray` is
reused unchanged.

### Channel 1 — photometric, every frame

One pass of `_iter_frames(video_path)` (`sampling.py:155`), the repo's only decoder. Per frame:

**`compute_blur(gray)` returns both blur metrics, deliberately.**

- `blur` — `skimage.measure.blur_effect(gray, h_size=11)`, Crete-Roffet perceptual blur, normalized to
  `[0, 1]`. **Higher means more blurred.**
- `laplacian` — `cv2.Laplacian(gray, cv2.CV_64F).var()`, the incumbent. **Higher means sharper.**

The two run in *opposite directions*. Both are computed on the same downscaled gray, so their scatter is
a controlled comparison and is the direct evidence for why the gate failed. See trap 1.

**`compute_exposure(gray)` returns mean, median, std, and both clipping fractions.**

Clipping is `mean(gray == 0)` and `mean(gray == 255)`. **Measured at native resolution, before any
downscale** — bilinear averaging pulls saturated pixels off exactly 0 and 255, so a resized frame
systematically under-reports clipping. This is the one metric in the report that does not share the
downscaled gray.

`compute_frame_quality(bgr)` is the merge: native-resolution gray for exposure, `_analysis_gray`
(`sampling.py:335`, 480 px) for blur, one flat dict out.

### Channel 2 — motion, strided pairs

Photometric metrics say nothing about whether the camera moved usefully. Two frames `k` apart:

- `match_orb` — ORB (`n_features=1000`) + `BFMatcher(NORM_HAMMING, crossCheck=True)`, returning two
  `(N, 2)` pixel arrays. Named for its method, not its role, because the choice of matcher is the
  contested decision here and the reader should not have to open it to learn which one this is.
  **The seam is the point-pair interface, not this function**: `compute_translation` and
  `compute_parallax` take arrays, so `LomaMatcher.match_pair` (`extractors.py:179`) substitutes from a
  notebook with no change to this module. Shape matches `MatchResult` for exactly that reason.

  It stays in `qa.py` rather than moving to `localization/`: that package imports `torch` and `vismatch`
  at module scope, `preproc/` imports neither, and QA is the pre-reconstruction path that must run
  without a GPU. Nor does it earn a shared home — one caller does not justify inventing a module.
- `compute_translation` — median displacement from `cv2.estimateAffinePartial2D`, in pixels.
- `compute_parallax` — `1 − (homography inliers ÷ fundamental inliers)`, both fit with `USAC_MAGSAC`.
  A homography explains pure rotation and planar scenes exactly; a translating camera viewing 3D
  structure breaks it. **The complement is taken so the name and the number agree: higher means more
  parallax, 0.0 means the motion carried none.** Reporting the raw H/F ratio would invert the reading of
  every plot drawn from the column.

`motion_stride` defaults to `round(fps)` — roughly one second, the separation the reconstruction
actually works at — and is recorded in the report. **Every stride-dependent quantity is reported in
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
(`winSize=21, maxLevel=3`, `sampling.py:48`) tops out around ~170 px displacement at 480 px width and
fails *silently* on exactly the fast-motion frames that matter most. ORB's failure surfaces as
`n_matches → 0`, which is a reading.

### Correlations — the only derived numbers that ship

Everything else is a raw measurement. Two Spearman coefficients ship because they are findings, not
conveniences:

- `rho(blur, laplacian)` — the gate-failure diagnostic. If the incumbent metric ranked frames the same
  way perception does, it would be near −1.
- `rho(translation_px, blur)` — does camera speed predict softness. Pairs are joined to their first
  frame's row for this.

Each ships beside its own `n`. Called as `scipy.stats.spearmanr(a, b).statistic`, directly — no wrapper
function, no small-sample guard.

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

The honest cost is size: ~1.1 MB of JSON for a 13k-frame video. Acceptable, and the reason the payload
is **columnar** (`{"frames": {"blur": [...], ...}}`) rather than a list of dicts — row-of-dicts repeats
eight key names 13,000 times and triples the file.

### Report shape

`video_quality_report.json`, written when `output_path` is given:

```
video:        path, mtime, fps, total_frames, width, height, duration_s
params:       motion_stride, n_features
frames:       frame_idx, blur, laplacian,
              exposure_mean, exposure_median, exposure_std,
              clipped_low_frac, clipped_high_frac              (columnar, length N)
pairs:        frame_idx_a, frame_idx_b, translation_px, parallax, n_matches
                                                               (columnar, length ~N/stride)
correlations: {name: {rho, n}}
```

Column names carry units where a quantity has one (`translation_px`, `*_frac`); the functions that
produce them keep the short names above.

`clean_for_json` (`geometry/verification.py:376`, already public) converts nan → null. nan is the
expected value from `compute_translation` and `compute_parallax` when a pair has too few matches to fit
a model — **that failure is itself the measurement**, not an error, and null preserves it while the row's
`n_matches` explains it.

If the video yields zero frames, the report is `{"available": false, "reason": "..."}` — same contract as
the scene error report's channels. Nothing raises.

### What the gate does *not* become

An earlier draft moved the quality gate onto the perceptual blur metric and split judgment out of
`check_frame_quality`. Both are dropped. The reason is sequencing, not taste: switching the gate needs a
threshold on a `[0, 1]` perceptual scale, and no such number exists yet — `50.0` is a Laplacian-variance
value and is meaningless there. **Measuring the distribution is this spec's entire job; setting a
threshold from it is the follow-on.** Changing the gate first would be inventing the number this report
was built to supply.

So:

- `check_frame_quality` **survives**, with its signature, its behaviour, and its `reject_reason`
  unchanged. Its body is thinned to call `qa.compute_frame_quality` for the measurement, so the
  exposure and Laplacian math exists once. Policy stays where the decision is made.
- `compute_blur_score` is **deleted** — absorbed into `compute_blur` — along with its export
  (`preproc/__init__.py:10, 23`) and its tests (`tests/preproc/test_sampling.py:88-91, 496`).
- `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD`, `_ANALYSIS_WIDTH`
  (`sampling.py:30-38`) all **stay**. They are the gate's and the selector's policy, and inlining them
  into signatures would not be simpler.
- **No `frames.zarr` is invalidated. No staleness key is added. Frame selection is byte-identical.**
- `viz.py` is untouched: `plot_quality_examples` (`viz.py:109`) filters on `reject_reason` from
  `score_frames`, which still produces it.

## What is reused vs new

| Need | Supplied by |
|---|---|
| Decode every frame, rotation-corrected | `sampling._iter_frames` (`sampling.py:155`) — the repo's only decoder |
| fps, frame count, dimensions | `sampling.get_video_info` (`sampling.py:83`) |
| ffmpeg presence check | `sampling._require_ffmpeg` (`sampling.py:62`) |
| Downscaled analysis gray | `sampling._analysis_gray` (`sampling.py:335`), **unchanged**, at 480 px |
| Perceptual blur | `skimage.measure.blur_effect` (scikit-image 0.26.0, installed) |
| Laplacian variance | `cv2.Laplacian(...).var()` — the body of the deleted `compute_blur_score` |
| Robust H and F | `cv2.findHomography` / `cv2.findFundamentalMat`, `USAC_MAGSAC` (opencv 4.13.0, installed) |
| Similarity transform | `cv2.estimateAffinePartial2D` |
| Rank correlation | `scipy.stats.spearmanr(...).statistic` (scipy 1.17.1, installed) |
| nan → null | `geometry.verification.clean_for_json` (`verification.py:376`) |
| Quantiles, histograms, ECDFs, cumulative sums | **the reader's `numpy`** — not shipped |

No new dependencies. Everything above is present in `/opt/venv/reconstruction`.

**Overlap to hold, not resolve here:** `sampling.score_frames` (`sampling.py:708`) already walks every
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

Scores at 1024/768/512/384 were 0.1659/0.1657/0.1671/0.1662 — near-invariant, which is why reusing
`_analysis_gray` at 480 px costs nothing. That is weak evidence from one synthetic input; re-measuring
resolution sensitivity on real footage is a plan task, not a settled fact.

Whole-video estimates: 600 frames ≈ 67 s plus decode. 13k frames photometric-only ≈ 156 s; ~6.5 min if
ORB ran on every frame, which is why `motion_stride` exists.

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
- **Parallax** — the discriminating control, and the one that can silently pass. Synthesize a pure
  in-plane rotation/zoom of a single image: `parallax` must sit near 0.0. Then use a real pair with known
  camera translation over non-planar structure: it must rise well above. A control testing only the first
  case cannot distinguish "correct" from "always returns 0.0".
- **Report** — a video whose frames are all identical must produce `n_matches` collapsing and nan
  translations serialized as null, not an exception.
- **Selection parity** — `sample_frames` on `data/tutorial/` must produce byte-identical `frames.zarr`
  before and after the `check_frame_quality` thinning. This is the guard on the "no change to frame
  selection" non-goal.

## Implementation principles

- Functions, not classes. No state to carry.
- Existing packages over new code: skimage, opencv, scipy, numpy. Nothing here is a reimplementation.
- `qa.py` declares zero constants; tuning values are keyword arguments with defaults. No constant
  elsewhere in `preproc/` is added, moved, or deleted.
- `compute_` prefix, matching the repo's 8 existing uses (`calculate_` appears zero times).
- Column names carry units (`translation_px`, `clipped_low_frac`).
- Delete what is obsoleted: `compute_blur_score` goes in the same change that absorbs it, with its
  export and tests.
- Ship raw; derive nothing the reader can derive.

## Traps

1. **`blur` and `laplacian` point in opposite directions.** Higher `blur` is worse; higher `laplacian` is
   better. Any threshold, sort, or `argmax` written against the wrong one inverts silently — a sampler
   using `max(blur)` would confidently select the blurriest frames.
2. **`parallax` is the complement of the H/F ratio, not the ratio.** Higher means more parallax. Anyone
   "simplifying" the function to return the raw ratio inverts every plot drawn from the column.
3. **Clipping must be measured before downscaling.** Any refactor that hoists one shared gray to the top
   of the loop for speed breaks this, and no test catches it unless the resized-copy control exists.
4. **`frame_idx` is the source video index**, zero-padded to 6 in `frame_XXXXXX` names, and is the join
   key to `frames.zarr` and `report.json`. It is not a row position. Positional indexing into the
   report's columns and into `FrameStore` rows are different things.
5. **nan is a measurement.** `compute_translation` returning nan means the pair failed to match, which is
   the interesting case. Dropping nan rows before plotting silently deletes the worst frames.
6. **`n_matches → 0` is ambiguous without `translation_px`.** Large inter-frame motion and unusable
   frames both produce it. Ship `n_matches` on every pair row so the two stay separable.
7. **`OpticalFlowFrameSelector` looks like free motion reuse and is not.** Its baseline moves with
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
