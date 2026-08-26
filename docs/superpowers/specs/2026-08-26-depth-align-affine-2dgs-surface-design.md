# Affine depth alignment, contiguous VDA context, and RaDe-GS median normals

**Date:** 2026-08-26
**Scene:** `GH010229_undist_r7` (300 keyframes, InstantSfM backend, undistorted)
**Goal:** raise 2DGS mesh fidelity — near-field detail up, far-field sloppiness down. PSNR secondary.

## Problem

The 2DGS mesh from the InstantSfM path looks sloppy at distance and loses small detail up
close. Depth supervision is the only channel through which geometry priors reach the splats
(`splats.losses.depth`, weight 0.01), so the depth targets are the lever.

Three candidate changes were measured before any training:

- **A** — run VDA over a contiguous 8 FPS stream instead of 300 keyframes sitting 44 video
  frames apart. A temporal model handed a sparse subsample is out of distribution.
- **B** — replace the per-frame scalar depth alignment with an affine fit in disparity.
- **C** — blend RaDe-GS median-depth normals into the normal-consistency loss.

## Measurements

All numbers hold the reconstruction fixed (existing `GH010229_undist_r7` COLMAP model) and
vary only the depth. Correspondences are track observations, exactly the pairs
`align_depth_to_reconstruction` already uses. Scripts live in the session scratchpad
(`vda_vid8.py`, `affine_score.py`, `disp_floor.py`, `disp_control.py`, `median_smoke.py`).

### The metric that matters is disparity, not depth ratio

`gsplat.losses.depth_l1_loss` is **L1 in disparity space**. The depth-ratio spread we had
been grading on understates the effect of alignment by ~5x. The right number is the
irreducible disparity L1 a perfect renderer would still pay against the aligned target.

| depth source | align | disparity L1 | vs baseline |
|---|---|---|---|
| 2 FPS keyframes (shipped) | scale | 0.00449 | — |
| 2 FPS keyframes (shipped) | affine | 0.00364 | **-18.9%** |
| 8.56 FPS contiguous | scale | 0.00436 | -2.9% |
| 8.56 FPS contiguous | affine | 0.00353 | **-21.4%** |

Affine dominates; the two changes are near-additive.

### Affine gains are uniform across depth, and largest near

| band | scale | affine | delta |
|---|---|---|---|
| pooled | 0.00449 | 0.00364 | -18.9% |
| Q1 3.3-12.1m | 0.00629 | 0.00448 | **-28.8%** |
| Q2 12.1-16.5m | 0.00386 | 0.00344 | -10.8% |
| Q3 16.5-26.1m | 0.00385 | 0.00336 | -12.7% |
| Q4 26.1-770m | 0.00397 | 0.00329 | -17.3% |

**The far-field hypothesis was wrong.** Affine was expected to fix distance error by
correcting a disparity offset; the gain is uniform instead. Disparity already suppresses the
far field — an 86% relative error at 30m costs less gradient than a 19% error at 5m — and Q4
carries only 22% of the loss budget. Masking far depth is therefore **not** pursued.

### The win is the offset term, not the space

A one-parameter fit in disparity scores 0.00449, identical to the shipped depth-space scale.
This is algebraic (`1/(s*d) = (1/s)*q`), so the control mostly confirms the robust estimator
is irrelevant. All 18.9% comes from the second parameter `b`.

### 8 FPS contiguous stream

Stride 7 of 59.94 (the stride `sample_fps` picks for `fps: 8.0`) -> 8.56 FPS, 2130 frames.

| | shipped 2 FPS kf | 8.56 FPS contiguous |
|---|---|---|
| per-frame CV | 0.4089 | 0.4008 |
| spread per-frame | 1.331 | 1.316 |
| co-visible sep-50 | 34.5% | 25.3% |

Consistent with the earlier 10 FPS probe (0.4012 / 1.317 / 25.9%) — the effect saturates by
8 FPS. In loss space this is only -2.9%. It may still pay off through the SfM depth prior,
which this metric cannot observe.

### Median depth is usable

`rasterization_2dgs`'s 6th return is `(1,H,W,1)` — identical shape to expected depth — carries
`grad_fn`, is a distinct signal (mean `|expected - median|` = 0.554 on a synthetic scene), and
its gradient reaches the Gaussian means. Sparse by construction: only the median Gaussian on
each ray receives gradient.

### Extrapolation is a masking signal, but only on the far side

`mesh.conf_percentile: 20` is configured, but InstantSfM carries no confidence, so
`reconstructor.py:1603` logs "no confidence in zarr — using unmasked depth" and every
positive-depth pixel becomes a target. The only evidence-based reliability signal available on
this path is extrapolation: the affine fit is constrained solely over the disparity range the
track observations span.

Held-out test (fit on half of each frame's tracks, score the other half): out-of-range
observations are **2.2x worse** (0.00782 vs 0.00355 mean disparity residual). Real signal.

But the extrapolated pixels are not where the naive reading suggests:

| side | meaning | mean | p90 |
|---|---|---|---|
| `q < q_min` | farther than the furthest track | 0.58% | 1.33% |
| `q > q_max` | nearer than the nearest track | **4.22%** | 11.34% |

Median depth of the near-side pixels is 2.58m against a nearest-track median of 8.67m. SIFT
tracks do not cover close surfaces, so a two-sided extrapolation mask would delete the closest
~4% of every frame — precisely the near-field geometry this work exists to sharpen, and where
VDA is most reliable (near = large disparity). **The mask must be one-sided.**

## Components

### A — VDA context stream (`preproc/video.py`, `pointcloud/sfm.py`, `wrapper/reconstructor.py`)

Today `generate_vda_depth` is handed the 300 staged keyframes. Instead, decode the source
video at a configured rate, run VDA over that contiguous stream, and keep only the keyframe
rows.

- `context_indices(video_path, *, target_fps, info=None)` — the frame grid, using the same
  `step = round(native_fps / target_fps)` rule as `sample_fps` so the two agree.
- `decode_context(video_path, indices, *, profile, roi, out_size)` — chunked at 64 frames.
  Undistorts at native 1920x1080 **before** downscaling, or `K_new` stops matching. Frames
  are emitted at short-side 518, the model's native grid: `Resize(lower_bound)` upscales
  anything smaller, so decoding below that loses detail without saving GPU.
- `generate_vda_depth(..., keep_rows=None)` — writes only the requested rows.
- `reconstructor.py` builds the stream when `vda_context_fps` is set and
  `provenance["video_path"]` resolves; otherwise it warns and falls back to today's keyframe
  path. The fallback is load-bearing: rerun-from-processed scenes have no source video.

**Keyframes are selected from the context grid**, so they are a subset by construction — no
union, no index matching. This is a preproc change, not only a VDA one: `_sample_by_quality`
gains a `candidates` index set, and `sample_fps`/`sample_uniform` pass the context grid through
so both targets and blur/exposure substitutes are drawn from grid members only.

Cost, accepted: substitution narrows from 15 candidates per slot to 7. Today 300 keyframes over
13115 source frames spaces targets 43.7 apart, so `search_radius: 7` binds and gives 15
candidates. Over the 2130-frame grid, spacing is 7.1, so the half-spacing rule caps the radius
at 3 grid frames and gives 7.

### B — affine depth alignment (`pointcloud/sfm.py`)

- Extract `_depth_correspondences(reconstruction, image_names, depth)` from the existing loop.
  `align_depth_to_reconstruction` keeps its signature and behaviour on top of it.
- `align_depth_affine(...)` fits `1/d_colmap ~= a*(1/d_vda) + b` per frame by least squares
  with two rounds of MAD-3sigma rejection.
- Applied in depth form, `d_new = d_vda / (a + b*d_vda)`, which avoids dividing by zero depth.
  Pixels where the denominator collapses are written as 0 (no target, dropped from
  `world_points`).
- Falls back to scale-only when: fewer than 50 observations, `a <= 0`, or `a*q_lo + b <= eps`
  where **`q_lo = 1/percentile(depth, 99)`**. Using the map maximum rejects 78/300 frames on a
  single sky pixel; p99 rejects 14/300. No frame fails for a bad fit — `a` min is 0.096.
- Saturation cost measured: 0.06% of pixels invalidated on average, 0.98% on the worst frame,
  median horizon 51m.
- New attrs `depth_align_model` and `depth_affine_ab`. `depth_scale: "colmap"` is retained so
  the legacy guards at `reconstructor.py:1388` and `:1572` and `tests/wrapper/test_splats_stage.py`
  are untouched.

Alignment runs **after** SfM, so affine needs no new reconstruction.

### C — RaDe-GS median normals (`splats/rendering.py`, `losses.py`, `trainer.py`, `outputs.py`)

RaDe-GS blends two normal-consistency **losses** (0.4 expected / 0.6 median). This is not the
upstream-2DGS `depth_ratio`, which blends the two depths into one `surf_depth`. We take the
RaDe-GS semantics.

- `rendering.py` keeps the currently-discarded `_median_depth` and adds
  `render["depth_normal_median"]`.
- All six `OPTIONAL_LOSSES` functions take `spec` as a uniform 5th argument; `compute_losses`
  passes it. This preserves the module's stated "same signature" invariant.
- `normal_consistency_loss` returns `(1-r)*cos(n, dn_expected) + r*cos(n, dn_median)`.
- `trainer.py` allow-list gains `depth_ratio`, validated `0 <= r <= 1`, `normal_consistency`
  only, and must be 0 for 3dgs — mirroring the existing 2dgs-only distortion guard.
- **Default `depth_ratio: 0.0`**, so unchanged configs reproduce today's behaviour exactly.
- `outputs.py` writes `median_depth` alongside `depth`.

### D — reproducible SfM (`pointcloud/sfm.py`)

`InitializeRandomPositions` uses unseeded `np.random.uniform(-1, 1)` for camera translations
and track xyzs, making reconstructions non-deterministic. `random_seed` is a supported
upstream RUNTIME_OPTION (`controllers/global_mapper.py:25`, seeds numpy/random/torch) that
neither we nor upstream's CLI sets. One line beside the existing `use_depths`.

### F — honest sfm depth-target masking (`pointcloud/sfm.py`, `wrapper/reconstructor.py`)

Two parts, both following from the measurement above.

- **Far-side extrapolation bound.** `align_depth_affine` already zeroes pixels past its
  saturation horizon; extend the same "no supervision without evidence" rule to pixels whose
  disparity falls below the fitted range (`q < q_min` over that frame's track observations).
  One-sided by design — the near side is left fully supervised. Combined cost 0.58% + 0.06%
  of pixels.
- **Stop the silent no-op.** `reconstructor.py` currently accepts `conf_percentile` on the sfm
  path and quietly ignores it at info level. Alignment-based masking is now the sfm channel, so
  the log must say that rather than imply the percentile was applied.

### E — mesh depth source (`mesh/`)

`mesh.splat_depth: expected | median` selects which rendered depth TSDF fuses. RaDe-GS itself
never did this — its `mesh_adapter.py:14` hardcoded `depth_name: "depth"`, so median never
reached its mesh.

## Config surface

```yaml
preproc:
  vda_context_fps: 8.0        # null = today's keyframe-only VDA
pointcloud:
  instantsfm:
    depth_align: affine       # scale | affine
    random_seed: 0
mesh:
  splat_depth: expected       # expected | median
splats:
  losses:
    normal_consistency: {weight: 0.05, start: 7000, depth_ratio: 0.0}
```

## Grid

2dgs, 300 frames, 12k steps (40 visits/view), sequential in tmux, one GPU job at a time.
Every cell: `semantics: {enabled: false}`, `appearance_opt: true`, `pose_opt: true`,
`grow_grad2d: 2e-4`. One-change-per-step ladder:

| cell | recon | align | depth_ratio | isolates |
|---|---|---|---|---|
| 1 | A (existing) | scale | 0.0 | baseline = recorded 2dgs 20.80 |
| 2 | A | affine | 0.0 | affine |
| 3 | A | affine | 0.6 | median normals |
| 4 | B (8 FPS, seeded) | affine | 0.6 | contiguous VDA |

Cells 1-3 reuse the existing reconstruction, so only **one** new SfM run is needed. Recon A is
frozen on disk and unseeded; recon B is seeded. This asymmetry is acceptable because A is never
re-rolled.

## Grading

Primary (mesh): main-component vertex fraction, speckle component count, vertex count after
`clean_repair`, at TSDF voxel 0.1 — 0.05 OOMs on 2dgs. Renders from scene cameras, not vertex
counts alone: vertex counts have previously hidden truncation.

Secondary: PSNR/SSIM. Expect these to read below the recorded 20.80, which was trained at 30k
(100 visits/view) — 12k is 40. Cross-cell deltas remain valid.

## Non-goals

- Far-field depth masking — measured as low-value above.
- 400-frame scenes — deferred; 400 at 12k is 30 visits/view, below the 34 that cost -0.9 dB in
  the 875-frame run. Frame count gets its own axis later.
- The `upstream max_res=1280` cap: measured, exact null. `Resize(lower_bound, ensure_multiple_of=14)`
  forces the short side to 518 regardless.

## Risks

- Cell 4 is the least isolated cell. Selecting keyframes from the context grid means recon B
  differs from recon A in **which frames were chosen**, and the depth prior change means it
  differs in poses and points too — not only in depth targets. Its delta is attributable to
  "contiguous VDA and everything downstream of it", not to the depth loss specifically.
- Median-depth gradients are sparse — only the median Gaussian per ray. If `depth_ratio: 0.6`
  destabilises training, fall back to 0.4 (RaDe-GS's expected-normal weight) rather than
  abandoning the component.
- Affine's saturation mask interacts with mesh fusion as well as depth targets. Measured cost is
  <1% of pixels, but it is a behaviour change to both consumers.
