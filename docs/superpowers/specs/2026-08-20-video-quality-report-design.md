# Video quality report — measuring the capture before the reconstruction sees it

Status: design, unimplemented.
Parallel to [scene error report](2026-08-20-scene-error-report-design.md), which measures error *after*
reconstruction. This one measures the source video *before* it, on every frame, and joins to that report
by source frame index.

## Goal

Answer one question with numbers: **what did the camera actually record, frame by frame?**

Blur, exposure, clipping, inter-frame motion, and whether that motion carried parallax. One scalar per
frame per channel, shipped raw, keyed by source video index so it joins to `frames.zarr` records and to
`report.json`'s per-frame rows.

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

## Non-goals

- **No verdict.** No `good`/`usable_with_caveats`/`reshoot`, no threshold table, no advice strings, no
  pass/fail counts. The report ships numbers and their sample sizes. The reader draws the line.
  This extends to sample size: there is no "too few frames to correlate" guard, because that is a verdict.
- **No CLI.** No `main()`, no argparse, no exit codes, no console renderer. Plain functions, callable from
  `Reconstructor`, a script, or a notebook.
- **No automatic wiring into the preproc stage.** Running QA on every frame costs a second full decode
  (~156 s on a 13k-frame GoPro), which would destroy the single-ffmpeg-pass fast path that `uniform`
  and `fps` sampling depend on (14fdc42). The report is called explicitly. See deferred follow-ons.
- **No new plotting.** Every plot in this report's inventory is a matplotlib one-liner over a raw column.
  `preproc/viz.py` gets a separate rework spec, after this report measures real distributions — bin
  choices and axis limits should come from data, not guesses.
- **No histograms, quantile grids, or stored bin edges.** See "Why raw, not binned" below.
- **No staleness key.** The gate's decision metric changes, so every `frames.zarr` on disk is stale by
  construction. Detecting that is pointless when the answer is always "yes". Everything re-runs once.

## Design

### One module, seven functions, zero constants

`collab_splats/preproc/qa.py`.

```python
# per frame
compute_blur(gray, *, long_side=512) -> dict          # blur_strength, laplacian_variance
compute_exposure(gray) -> dict                        # mean/median/std/clipped_low_frac/clipped_high_frac
compute_frame_quality(bgr, *, long_side=512) -> dict  # merges both, from one BGR frame

# per pair
match_frames(gray_a, gray_b, *, n_features=1000) -> tuple[np.ndarray, np.ndarray]
compute_translation_px(pts_a, pts_b) -> float
compute_homography_inlier_ratio(pts_a, pts_b) -> float

# whole video
compute_video_quality(video_path, *, output_path=None, motion_stride=None,
                      long_side=512, n_features=1000) -> dict
```

Every tuning value is a keyword argument with a default. `qa.py` has **no module-level constants** — the
same discipline the scene error report's plan landed on for `geometry/metrics.py`. (`sampling.py` keeps
`_ANALYSIS_WIDTH`, which is the optical-flow selector's tuning, not this report's.)

### Channel 1 — photometric, every frame

One pass of `_iter_frames(video_path)` (`sampling.py:155`), the repo's only decoder. Per frame:

**`compute_blur(gray)` returns both blur metrics, deliberately.**

- `blur_strength` — `skimage.measure.blur_effect(gray, h_size=11)`, Crete-Roffet perceptual blur,
  normalized to `[0, 1]`. **Higher means more blurred.**
- `laplacian_variance` — `cv2.Laplacian(gray, cv2.CV_64F).var()`, the incumbent. **Higher means sharper.**

The two run in *opposite directions* and the names say so, because a column called `blur` that goes up
with sharpness is the kind of thing that survives a code review and then poisons a plot. Both are
computed on the same downscaled gray, so their scatter is a controlled comparison and is the direct
evidence for why the gate failed.

**`compute_exposure(gray)` returns mean, median, std, and both clipping fractions.**

Clipping is `mean(gray == 0)` and `mean(gray == 255)`. **Measured at native resolution, before any
downscale** — bilinear averaging pulls saturated pixels off exactly 0 and 255, so a resized frame
systematically under-reports clipping. This is the one metric in the report that does not share the
downscaled gray.

`compute_frame_quality(bgr)` is the merge: it does the native-resolution gray for exposure, the
downscaled gray for blur, and returns one flat dict. It **replaces `check_frame_quality` and
`compute_blur_score`**, which are deleted.

### Channel 2 — motion, strided pairs

Photometric metrics say nothing about whether the camera moved usefully. Two frames `k` apart:

- `match_frames` — ORB (`n_features=1000`) + `BFMatcher(NORM_HAMMING, crossCheck=True)`, returning two
  `(N, 2)` pixel arrays. Shape deliberately matches `localization.extractors.MatchResult`, so
  `LomaMatcher.match_pair` (`extractors.py:179`) can be substituted from a notebook without touching
  this module.
- `compute_translation_px` — median displacement from `cv2.estimateAffinePartial2D`, in pixels.
- `compute_homography_inlier_ratio` — `USAC_MAGSAC` homography inliers ÷ fundamental inliers. A
  homography explains pure rotation and planar scenes exactly; a translating camera viewing 3D structure
  breaks it. **Ratio → 1.0 flags motion that carried no parallax** — the failure mode that produces a
  plausible-looking video and a degenerate reconstruction.

`motion_stride` defaults to `round(fps)`, i.e. roughly one second of separation, and is recorded in the
report. **Every stride-dependent quantity is reported in source-frame and second units, never in
"analyzed frames".** The pasted `capture-qa` spec got this wrong three separate ways — its
`exposure_stability`, its `trans_px > 1.0` test, and its reshoot-triggering `k_gap = 15` were all in
analyzed-frame units but compared against fixed limits, so a longer video silently changed its own
verdict by changing its stride.

#### Why ORB and not the localization matchers

Measured: loma via `verify()` costs **1346.7 ms/pair** on GPU
([measured report](2026-08-20-scene-error-report-measured.md)); ORB costs ~30 ms/pair on CPU. That is
**45×**, and it is the smaller reason. The larger one: QA runs *before* reconstruction, on a machine that
may have no GPU and no weights — and a learned matcher would *succeed* on exactly the degraded frames
whose degradation this report exists to detect. ORB failing is signal. LoMa not failing is not.

### Correlations — the only derived numbers that ship

Everything else in the report is a raw measurement. Two Spearman coefficients ship because they are
findings, not conveniences:

- `rho(blur_strength, laplacian_variance)` — the gate-failure diagnostic. If the incumbent metric ranked
  frames the same way perception does, it would be near −1.
- `rho(translation_px, blur_strength)` — does camera speed predict softness.

Each ships beside its own `n`. Called as `scipy.stats.spearmanr(a, b).statistic`, directly — no wrapper
function, no small-sample guard.

### Why raw, not binned

The scene error report keeps exactly one histogram, and only because its per-pixel residuals are
`N_pairs · H · W` values that cannot be shipped. **This report has one scalar per frame per channel.** A
13k-frame GoPro is 13k floats per column; the raw data *is* shippable, so shipping it is strictly more
capable than shipping counts:

| Plot | Derivation |
|---|---|
| Histogram of any metric, **any bins, chosen at plot time** | `plt.hist(blur_strength, bins=50)` |
| ECDF | `plt.plot(np.sort(v), np.linspace(0, 1, len(v)))` |
| Any metric vs wall-clock time | `frame_idx / fps` on x |
| Exposure mean ± std band | `exposure_mean`, `exposure_std` |
| Clipping over time, stacked | `clipped_low_frac`, `clipped_high_frac` |
| **Blur vs Laplacian scatter** — the gate diagnostic | both columns, same frames |
| Camera path-length proxy | `np.cumsum(translation_px)` |
| Pure-rotation segments | `homography_inlier_ratio → 1.0` |
| **Sampler audit** — selected vs declined frame quality | join `frame_idx` ↔ `FrameStore.frame_indices()` |
| **Blur vs depth error** | join `frame_idx` ↔ `report.json` per-frame rows |

A stored histogram supports the first row only, at bins fixed months earlier, and supports neither join.
Quantile grids, percentile ranks, and cumulative curves are cut for the same reason: all are one numpy
call over a column the reader already has.

The honest cost is size: ~1.1 MB of JSON for a 13k-frame video. Acceptable, and the reason the payload
is **columnar** (`{"frames": {"blur_strength": [...], ...}}`) rather than a list of dicts — row-of-dicts
repeats eight key names 13,000 times and triples the file.

### Report shape

`video_quality_report.json`, written when `output_path` is given:

```
video:        path, mtime, fps, total_frames, width, height, duration_s
params:       motion_stride, long_side, n_features
frames:       frame_idx, blur_strength, laplacian_variance,
              exposure_mean, exposure_median, exposure_std,
              clipped_low_frac, clipped_high_frac          (columnar, length N)
pairs:        frame_idx_a, frame_idx_b, translation_px,
              homography_inlier_ratio, n_matches            (columnar, length ~N/stride)
correlations: {name: {rho, n}}
```

`clean_for_json` (`geometry/verification.py:376`, already public) converts nan → null. nan is the
expected value from `compute_translation_px` and `compute_homography_inlier_ratio` when a pair has too
few matches to fit a model — **that failure is itself the measurement**, not an error, and null preserves
it while the row's `n_matches` explains it.

If the video yields zero frames, the report is `{"available": false, "reason": "..."}` — same contract as
the scene error report's channels. Nothing raises.

### What happens to the quality gate

The report passes no judgment, but the *sampler* must: selecting frames is deciding. So measurement and
judgment are split — which is the refactor `check_frame_quality` was always owed. It did both, and that
fusion is why the measurement was only ever as good as the gate needed it to be.

- **Measure:** `qa.compute_frame_quality(bgr)` — returns metrics only. **No `reject_reason` key.**
- **Judge:** one private helper stays in `sampling.py`:

  ```python
  def _gate_reason(quality, *, blur_threshold, exposure_range, min_exposure_std) -> str | None:
      """First failing gate check named, or None — measures nothing."""
  ```

  It takes an already-measured dict and returns `None | "blur" | "exposure"`. Both call sites —
  `_sample_positions` (`sampling.py:551-552`) and `_iter_scored_frames` (`sampling.py:664`) — call
  `compute_frame_quality` then `_gate_reason`, so the gate's policy still lives in exactly one place.

- `check_frame_quality` and `compute_blur_score` are **deleted**, with their `preproc/__init__.py`
  exports (lines 9-10, 23-24) and their tests (`tests/preproc/test_sampling.py:79-127`, and the export
  assertions at 496-497).
- The three gate constants `_DEFAULT_BLUR_THRESHOLD`, `_EXPOSURE_MEAN_RANGE`, `_EXPOSURE_MIN_STD`
  (`sampling.py:34-38`) are deleted as constants and become **defaults on the two public entry points**
  that already expose `blur_threshold` — `sample_frames` (`sampling.py:379`) and `score_frames`
  (`sampling.py:708`) — which thread them to `_gate_reason`. `_ANALYSIS_WIDTH = 480` **stays**; see
  trap 7.
- The gate judges on **`blur_strength`**, not `laplacian_variance`. This changes which frames are
  selected, so **every `frames.zarr` on disk must be rebuilt.** No staleness key is added; everything
  re-runs once.
- `reject_reason` **survives** in the sampler path and in `score_frames`'s per-frame records, now
  produced by `_gate_reason`. It is absent only from the report. Consequently
  `viz.plot_quality_examples` (`viz.py:109`), which filters on `reject_reason` from `score_frames`,
  keeps working unchanged.

The threshold *value* on `blur_strength` is unmeasured and **this spec does not invent one** — the
existing `50.0` is a Laplacian-variance number and is meaningless on a `[0, 1]` perceptual scale.
Sequencing: build the report, read the blur distribution off real footage, then set the gate as a plan
task. Same "blocked on measured distributions" posture the scene error report takes. Until then the
blur gate is inert by construction, which is the status quo — it was inert at `50.0` too, just
unknowingly.

## What is reused vs new

| Need | Supplied by |
|---|---|
| Decode every frame, rotation-corrected | `sampling._iter_frames` (`sampling.py:155`) — the repo's only decoder |
| fps, frame count, dimensions | `sampling.get_video_info` (`sampling.py:83`) |
| ffmpeg presence check | `sampling._require_ffmpeg` (`sampling.py:62`) |
| Downscaled analysis gray | `sampling._analysis_gray` (`sampling.py:335`), generalized to take `long_side` instead of the module constant `_ANALYSIS_WIDTH` |
| Perceptual blur | `skimage.measure.blur_effect` (scikit-image 0.26.0, installed) |
| Laplacian variance | `cv2.Laplacian(...).var()` — the body of the deleted `compute_blur_score` |
| Robust H and F | `cv2.findHomography` / `cv2.findFundamentalMat`, `USAC_MAGSAC` (opencv 4.13.0, installed) |
| Similarity transform | `cv2.estimateAffinePartial2D` |
| Rank correlation | `scipy.stats.spearmanr(...).statistic` (scipy 1.17.1, installed) |
| nan → null | `geometry.verification.clean_for_json` (`verification.py:376`) |
| Quantiles, histograms, ECDFs, cumulative sums | **the reader's `numpy`** — not shipped |

No new dependencies. Everything above is present in `/opt/venv/reconstruction`.

**Overlap to hold, not resolve here:** `sampling.score_frames` (`sampling.py:708`) already walks every
frame and already emits `blur_score`, `exposure_mean`, `exposure_std`, `reject_reason`, plus LK optical
flow `disparity`/`rotation`/`histogram_similarity`. It is not absorbed, because its motion signal is
measured against the *last accepted keyframe* — a moving, selector-dependent baseline, which is exactly
the stride-dependence this report refuses — and because its purpose is tuning the optical-flow selector,
not describing the capture. Its quality dict changes shape when `check_frame_quality` dies. Whether the
two converge belongs to the viz rework spec.

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

`long_side=512` is the default because it is the knee. Scores at 1024/768/512/384 were
0.1659/0.1657/0.1671/0.1662 on synthetic noise — near-invariant, but that is weak evidence from one
synthetic input, and re-measuring resolution sensitivity on real footage is a plan task, not a settled
fact.

Whole-video estimates: 600 frames ≈ 67 s plus decode. 13k frames photometric-only ≈ 156 s at 512 px;
~6.5 min if ORB ran on every frame, which is why `motion_stride` exists.

The pasted spec's "≤ 60 s" target is not achievable on long footage and is not adopted.

## Validation — a negative control per channel

Each channel must be shown to move when, and only when, the thing it measures moves.

- **Blur** — Gaussian-blur a known-sharp frame at increasing σ. `blur_strength` must increase
  monotonically; `laplacian_variance` must decrease. A metric that does not respond to synthetic blur
  will not detect real blur.
- **Exposure** — scale a frame toward 0 and toward 255. `exposure_mean` tracks; `clipped_high_frac` rises
  only once pixels actually reach 255. **Run this control at native resolution and again on a resized
  copy** — the resized copy must under-report clipping, which is the evidence for the native-resolution
  rule above.
- **Translation** — shift a frame by a known pixel offset. `compute_translation_px` must recover it.
- **Homography ratio** — the discriminating control, and the one that can silently pass. Synthesize a
  pure in-plane rotation/zoom of a single image: the ratio must sit near 1.0. Then use a real pair with
  known camera translation over non-planar structure: the ratio must fall well below it. A control that
  only tests the first case cannot distinguish "correct" from "always returns 1.0".
- **Report** — a video whose frames are all identical must produce `n_matches` collapsing and nan
  translations serialized as null, not an exception.

## Implementation principles

- Functions, not classes. No state to carry.
- Existing packages over new code: skimage, opencv, scipy, numpy. Nothing here is a reimplementation.
- Zero module constants. Tuning values are keyword arguments with defaults.
- `compute_` prefix, matching the repo's 8 existing uses (`calculate_` appears zero times).
- A name states the quantity and its unit, not its shape: `translation_px`, not `trans`;
  `homography_inlier_ratio`, not `parallax`.
- Delete what is obsoleted. `check_frame_quality` and `compute_blur_score` go in the same change that
  replaces them, with their tests and exports.
- Ship raw; derive nothing the reader can derive.

## Traps

1. **`blur_strength` and `laplacian_variance` point in opposite directions.** Higher `blur_strength` is
   worse; higher `laplacian_variance` is better. Any threshold, sort, or `argmax` written against the
   wrong one inverts the gate silently — the sampler will confidently select the blurriest frames.
2. **Clipping must be measured before downscaling.** Any refactor that hoists a single shared gray to the
   top of the loop for speed will break this without any test failing unless the resized-copy control
   from the Validation section exists.
3. **The gate change invalidates every `frames.zarr` on disk**, local and under `environments-processed/`.
   Nothing detects this — by decision. Anything comparing against a pre-change baseline is comparing
   against different frames.
4. **`frame_idx` is the source video index**, zero-padded to 6 in `frame_XXXXXX` names, and is the join
   key to `frames.zarr` and `report.json`. It is not a row position. Positional indexing into the report's
   columns and into `FrameStore` rows are different things.
5. **nan is a measurement.** `compute_translation_px` returning nan means the pair failed to match, which
   is the interesting case. Dropping nan rows before plotting silently deletes the worst frames.
6. **ORB is scale- and rotation-limited by design here.** Large inter-frame motion will drop
   `n_matches` toward zero. That is signal, but it is indistinguishable from "the frames are unusable"
   without also reading `translation_px`. Ship `n_matches` on every pair row so the two are separable.
7. **`_analysis_gray` currently reads `_ANALYSIS_WIDTH` directly** and is called from the optical-flow
   selector, whose LK cost is tuned to 480 px. Adding a `long_side` parameter must keep 480 as the
   selector's effective value, or frame selection changes for a second, unrelated reason.

## Deferred follow-ons

- **`preproc/viz.py` rework** — consistent axes and units, report-driven, replacing the four ad-hoc
  plotters; possibly absorbing `score_frames`. Own spec, written after this report measures real
  distributions.
- **Set the gate threshold on `blur_strength`** from measured footage.
- **Wire into the preproc stage** if the second decode pass can be avoided — most likely by folding QA
  into the optical-flow path, which already decodes every frame, while leaving the `uniform`/`fps` fast
  path alone.
- **Join study**: `blur_strength` and `homography_inlier_ratio` against `report.json` depth error, to
  test whether capture quality predicts reconstruction error at all. This is the question that motivates
  both reports and neither answers alone.
