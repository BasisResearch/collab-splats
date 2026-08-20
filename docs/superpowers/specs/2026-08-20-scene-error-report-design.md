# Scene error report — a `report` stage that localizes error in space and time

**Date:** 2026-08-20
**Branch:** `refactor/cu121-uv-migration`
**Status:** design for review.
**Supersedes:** `docs/superpowers/handoffs/2026-08-20-scene-error-report-handoff.md` for
implementation. Its reuse map and traps remain accurate; its "attribution verdict" and
"artifact shape" proposals are revised below.

## Goal

Understand **where reconstruction error comes from** on scenes that have no ground truth:
whether pose, depth, or appearance is responsible, where in the trajectory it occurs, and
whether disagreement accumulates.

The report **describes distributions and how they vary**. It does not grade a scene, does
not label a cause, and does not gate anything. Every quantity is a number the reader
interprets.

## Non-goals

- **No verdicts.** No "pose error" / "depth error" labels, no pass/fail, no scene score.
  Thresholds that would justify a verdict are exactly what this stage exists to inform, so
  inventing them now would be a guess dressed as a finding.
- **No feedback into the reconstruction.** Report-only, same contract as `verify`. Nothing
  written here is read by pointcloud, mesh, semantics, localize, or refine.
- **No per-pixel arrays.** `report.json` only (see *Artifact*).
- **No new backbones, matchers, or config knobs.** Zero config surface (see *Stage wiring*).
- **No retrieval/loop pairs for the epipolar channel** — deferred follow-on.

## Design

### Stage wiring

New leaf stage `report`:

```python
_STAGE_DEPS["report"] = ["pointcloud"]
```

`LEAF_STAGES` derives it automatically, so `--stages report` re-runs against a scene pulled
from `environments-processed` with no `rerun.py` change. `Reconstructor.report(overwrite)`
mirrors `verify` (`reconstructor.py:1095-1159`) closely enough to copy: resolve the result,
reuse the shared feature cache, write one artifact, log the path.

**Always on, no config boolean.** Every other diagnostic in the repo ships behind a
default-`false` flag (`use_multiview_confidence`, `geometric_verification`,
`bundle_adjustment`). This one does not, because a report nobody runs answers nothing and
the measured cost is bounded (see *Runtime*). The one boolean it would have had is the
boolean that would keep it off.

**Never fails a reconstruction.** Any channel that cannot run records
`{"available": false, "reason": "..."}` and the rest still emit. The known cases: no
index-stable matcher (`verify_reconstruction` hard-refuses XFeatStar), no `confidence` array
in an older zarr store, no `frames.zarr` for the original-res grid.

### The four channels

Attribution is possible only because the channels have **different dependencies**. This is
the entire reason there are four and not one blended number.

| channel | depends on | grid | source |
|---|---|---|---|
| **epipolar** — rot error, t-direction error | **poses only**, never touches depth | original | `verification.json` |
| **reprojection** — per-frame px, track survival | poses + independent tracks | original | `verification.json` |
| **depth cross-view** — signed relative residual, scale | poses **+ depth** | model | refactored mv loop |
| **photometric** — normalised warp residual | poses + depth **+ appearance** | original | new |
| **confidence** — model self-report, validated | — | model | `feedforward.zarr` |

Epipolar and reprojection are one channel with two readouts (same source, same dependency
set); depth, photometric, and confidence are the other three.

Confidence is **not** an error channel. It is an input to be validated: the interesting
quantity is how it co-varies with the measured disagreement — does the model know when it
is wrong? Nothing in the repo has ever checked this.

`report` **consumes** `verification.json`, running `verify` when absent. It never re-runs
the epipolar pass itself — two stages calling `verify_matches` would be waste.

### Resolution contract — per channel, not one global switch

`_rescale_reconstruction_to_original_dimensions` scales **K and image dimensions only; R and
t are untouched**. So resolution affects only pixel-space quantities, and the channels
differ:

| quantity | grid | why |
|---|---|---|
| rot error, t-direction error | original (already) | angles from poses — identical either way; verify already estimates from original-res keypoints |
| reprojection error | original (already) | genuinely resolution-dependent; reported in px **and** % of image width |
| depth residual, scale | **model** | see below |
| photometric | **original** | RGB detail exists only at original res; genuine `frames.zarr` data, only depth upsampled |
| coverage | original | structurally invisible at model res |

**Why depth is evaluated at model resolution.** Depth values are identical under nearest
upsampling — a model pixel and its original-res counterpart hold the same number — so
original-res evaluation returns the same answer. Worse, it would sample a
*guided-filtered* depth map on a finer grid, which reports **lower** disagreement than model
res. That improvement is `guided_upsample_depth`'s edge-sharpening, not the model's: it
would flatter the model with a smoother the model did not produce.

Measured on 7-Scenes (`original_coords[0] = [0, 0, 640, 480, 640, 480]`, model grid
592×448), original is only **1.16× the model area** — so this is not a cost decision. It is
a correctness decision. The factor is dataset-dependent and much larger elsewhere (a
1920×1080 source against a ~518-wide model is ~13× area).

**Every JSON block stamps `grid` and `resolution`.** Trap: model-res depth paired with
original-res K caused the 2026-08-11 mesh collapse;
`compute_multiview_depth_confidence` has a hard guard (principal point strictly inside the
depth grid) that exists to refuse it.

### Units — scale-free or normalised, everywhere

1 recon unit is not 1 metre (measured 2.596 m on chess/seq-01) and the factor differs per
scene **and** per backbone. Backbones do not even share a model resolution: VGGTX is fixed
518-wide with a 518 centre-crop (`vggtx.py:45,237`), MapAnything is 512-or-518 across three
resize modes (`mapanything.py:185`), the measured omega store is 592×448. **A raw distance
threshold or a bare pixel count in this report is a bug.**

- depth residual, scale ratio — relative, already unitless
- spatial distance — `‖C_i − C_j‖ / camera_extent`
- reprojection — px **and** fraction of image width
- photometric — normalised (zero-mean, unit-variance per patch), which also absorbs the
  `[0, 255]` (VGGT family) vs `[0, 1]` (MapAnything) image-scale split and makes the
  measure invariant to exposure change, which would otherwise swamp geometry
- depth stratification bins — quantiles of the scene's own depth, never absolute distance

### Signed residual: scale separated from noise

`compute_multiview_depth_confidence` computes `expected_d` (frame *i*'s point pushed into
*j* through the pose) and `sampled_d` (*j*'s own predicted depth there), then thresholds
`|expected − sampled|` and **discards the residual**. The refactor returns it signed:

```
r = (sampled_d − expected_d) / expected_d
```

Per pair this splits into two independent numbers:

- **`median(r)` = scale bias** — *j*'s depth is systematically larger or smaller than *i*'s
  propagated through the pose
- **spread of `r − median(r)` = geometric noise**

A pure scale error has a large median and a small spread; a pose error has ~zero median and
a large spread. That is a third attribution axis, obtained for free from a loop that already
computes both quantities.

`median(r)` against `|i − j|` is the **reference-free scale-drift** signal — no GT.

Refactor the existing function rather than writing a second projection loop; two loops
drift apart. Prior art recording the same residual:
`evals/scripts/depth_disagreement.py` (107 lines, already durable).

Scale invariance holds **only** with `abs_thresh=0.0`, and the input must be **Z-depth** —
not ray length, not disparity (`base.py:504-509`).

### Depth stratification

Every residual is additionally binned by `expected_d` quantile. This answers "do things
disagree more at greater depth" directly, and it has a **predicted shape to test against**:
triangulation uncertainty σ_Z ∝ Z²/(f·B), so a *relative* residual should grow roughly
linearly in Z. Growing faster indicates something beyond geometry (far-field extrapolation,
confidence miscalibration); flat indicates depth normalised in a way that hides error.

### The parallax bridge — putting pixel and depth channels on one axis

Far pixels have less parallax, so they disagree **less** in pixel terms while disagreeing
**more** in depth terms. This looks like the two channels contradicting each other. They do
not: the relation between them is exact to first order and computable per pixel.

For a pair with perpendicular baseline `B`, a point at depth `Z` has disparity
`d = f·B/Z`. A depth error `δZ` moves it in the image by `δd = f·B·δZ/Z²`. Substituting the
relative residual `r = δZ/Z` the focal and baseline collapse out:

```
δd = r · d
```

**Pixel disagreement = relative depth disagreement × disparity.** The `1/Z` inside `d` is
the whole apparent contradiction. Two derived quantities follow, both already computable
from arrays the projection loop holds.

**1. Equivalent pixel error** — `δd_equiv = r · d`. The depth residual expressed in pixel
units *through the pair's actual parallax*. This is the legitimate way to difference the
channels: convert first, then compare. It also converts to the interpretable question "how
many pixels of matching error would produce the depth disagreement we measured", which is
the natural unit for judging whether a residual is large.

**2. Explained fraction** — `ρ = measured_pixel_residual / δd_equiv`.

| ρ | reading |
|---|---|
| ≈ 1 | pixel disagreement fully accounted for by the depth disagreement — one underlying error, seen twice |
| ≫ 1 | pixel error exceeds what any depth error explains → the excess is **pose** (pose error moves pixels while leaving depths mutually consistent) or appearance |
| ≪ 1 | depth disagrees more than pixels do → the depth error lies along the ray, where this pair's baseline cannot see it — a low-observability configuration, not necessarily a bad depth |

ρ is unitless and parallax-normalised, so **it is comparable across depth bins**, which the
raw channels are not. That makes it the metric that actually answers the question: plot
ρ against depth quantile. **Flat means the depth trend is pure geometry** — the channels
were never in conflict. **Rising means a genuine far-field problem** beyond what parallax
explains. Same plot against `|i − j|` and against spatial distance gives the second-order
version.

A third, cheaper readout: the theory predicts `corr(r, δd)` should have **slope `d`**.
Measured slope against predicted slope is a direct check on the bridge itself, and a
per-depth-bin noise floor `r_floor = 1/d` states the relative depth precision a 1-px
matching error implies — measured `r` below `r_floor` means the depth channel is reading
matching noise, not model error.

**Compute the parallax angle directly from the two ray directions, not from `f·B/Z`.** The
small-angle pinhole form needs a focal length, and focal is exactly what is *not* comparable
across backbones (measured 11% fx spread on omega alone, which is why `shared_camera=True`
landed in BA). The angular form is scale-free, so report ρ in angular terms as the
backbone-comparable number and the pixel form alongside it for interpretability.

**Degenerate case, and it is not hypothetical.** `B` is the baseline component *perpendicular
to the viewing ray*, not `‖C_i − C_j‖`. Forward camera motion drives it to ~zero near the
epipole, so `d → 0` and ρ blows up. This is the same root cause as the recorded AUC@5
ill-conditioning on 10–20 mm indoor baselines (trap 10). ρ is therefore **undefined below a
parallax-angle floor**, and the report emits the *fraction of pixels in that regime* rather
than an infinity — a scene that is mostly below the floor has no usable ρ, and saying so is
the finding.

### The pair table — the second-order engine

Every pair contributes one row:

| field | meaning |
|---|---|
| `i`, `j` | frame indices |
| `temporal_separation` | `|i − j|` |
| `spatial_distance` | `‖C_i − C_j‖ / camera_extent` |
| `frustum_overlap` | did the AABB gate pass |
| `parallax_angle_deg` | median triangulation angle — the pair's depth **observability** |
| `below_parallax_floor_frac` | fraction of pixels where ρ is undefined |
| per-channel residuals | epipolar (when the pair is in the epipolar set), depth `median(r)` and spread, photometric |
| `equivalent_pixel_error`, `explained_fraction` | the parallax bridge, per pair |

Every second-order question is then a groupby on this table rather than a separate metric:

- **"how much error does a similar position give?"** → large `temporal_separation`, small
  `spatial_distance`. These are **revisits**, and their disagreement *is* the reference-free
  drift estimate.
- **"does error build over the trajectory?"** → residual vs frame index **at fixed
  separation**. Frame index alone is a confounded axis (scene content, motion speed,
  exposure all correlate with it), so a rise against index is not by itself evidence of
  accumulation. Two axes, two different stories.

**Revisit detection is free.** `_frustum_world_aabbs` + `_aabbs_overlap` (`base.py:394,428`)
gate pairs on whether their depth frusta intersect in world space, and the gate is
conservative — "a pair that truly overlaps can never be gated out". So an all-pairs run with
the gate on **is** a geometric revisit detector: the surviving pairs are exactly those
observing the same volume, temporal neighbours *plus* spatially-near/temporally-far
revisits. No retrieval pass, no DinoSalad.

**Asymmetry to state plainly:** the dense channels get this for free; the **epipolar channel
does not**, because its pairs come from `SequentialPairGenerator`. `quadratic_overlap`
(available at `verification.py:184`, currently unused) is enabled to add power-of-two-spaced
long-baseline pairs, but genuine revisits for the epipolar channel remain a deferred
follow-on (retrieval pairs via `localization/retrieval.py`).

**Limitation, stated rather than papered over:** a trajectory that never revisits anywhere
has no reference-free drift signal at all. The report says so — `"revisit_pairs": 0,
"drift": "unmeasurable without revisits or GT"` — rather than emitting a number.

### Distributions and cumulative error

Instead of three summary points, each channel emits:

1. **A quantile grid** (empirical CDF) — the distribution's *shape*, not median/p90/p99
   alone. `_distribution` (`verification.py:273`) is extended, not replaced.
2. **A cumulative curve along the trajectory** — sequential-pair residual accumulated
   against frame index. This is the direct "does disagreement build over the course of the
   reconstruction" readout.

*Assumption stated explicitly:* "cumulative error" is implemented in both senses — the
running accumulation along the trajectory and the empirical CDF — because both are cheap and
both are defensible readings of the request. If only one was meant, the other costs nothing
to ignore.

### Histograms — arbitrary-threshold queries must be exact

Queries of the form *"what percentage of frames / pairs / pixels fall within X error"* are
answerable from the file alone, at any X the reader picks:

- **Frame- and pair-level queries are already exact.** Both tables are row-per-entity, so
  "% of frames under X" is a count, not an estimate.
- **Pixel-level queries need a histogram, not just quantiles.** A quantile grid gives the
  CDF at fixed *probabilities*; inverting it to "fraction below arbitrary X" means
  interpolating between quantiles, and that interpolation is worst in the tail — the part
  that matters. So every pixel-level channel emits a **fixed-bin histogram with explicit bin
  edges** alongside its quantile grid. Per-bin counts make arbitrary-X queries exact and let
  a reader re-bin, or plot a PDF or CDF, without the raw pixels.
  `evals/scripts/depth_disagreement.py` already does this (2000 bins over [-0.5, 0.5]) —
  reuse its binning rather than inventing one.
- **Bin edges are stored, never assumed.** They differ per channel and per units convention.
- **`schema_version`** is stamped at the top level so a reader can handle older files
  instead of crashing on them.

Histograms are also what make ρ tractable: it is a per-pixel ratio with a long tail and an
undefined regime, so a quantile grid alone would hide both.

### Per-frame ranks, not calls

For each frame and channel, record the frame's **percentile rank within that scene's own
distribution** — a number in [0, 1]. Nothing else. The reader sees that frame 47 sits at
p99 in depth and p60 in epipolar; the report does not say what that means, does not name a
cause, and does not flag it.

This needs no absolute threshold, which sidesteps both the units problem and cross-backbone
incomparability. It is descriptive by construction.

### Coverage

From `original_coords` (`[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`): the fraction of each
original frame the model crop actually reconstructed. On the measured 7-Scenes store this is
100% (crop `0,0 → 640,480`), but VGGTX resizes width to 518 and **centre-crops height to
518**, so on a 16:9 source a large band of every frame has no depth at all. Model-res
evaluation is structurally blind to this, because the model-res grid *is* the crop.

### Ground truth — an optional block, not a second code path

An optional TUM trajectory path adds a `gt:` key with `ate` / `rpe` / `auc` from
`loop_closure/eval.py` (`ate_translation:139`, `rpe:175`, `auc_at_threshold:206`), reusing
`_write_tum` (`evals/scripts/eval.py:195`) for the interface. Everything else computes
identically with or without it. **The stage never forks on GT presence.**

The JSON states which alignment was used: `loop_closure/eval.py` aligns Umeyama **Sim(3)**
while `evals/metrics.py` defaults to **SE(3)**, so their numbers are not comparable.

AUC@5 is ill-conditioned on short-baseline indoor sequences (10–20 mm inter-frame baselines
swing the bearing term past the 1° bins); prefer ATE/RPE indoors and do not read a bad AUC
as "poses ruined".

### Artifact

**`report.json` only**, at `<backend_dir>/report.json`. Per-frame rows, per-pair rows,
per-channel quantile grids, cumulative curves, scene metadata. Small, diffable, plots
directly.

Per-pixel arrays for the follow-on viewer are **not** written. If the viewer later needs
per-pixel colouring, re-running with an array flag is a one-line follow-on, and deferring it
keeps the pixel-subsampling option open (see *Runtime*).

Added to the processed-scene output contract in `configs/README.md`.

## What is reused vs new

**Reused as-is:** `verification.json` (epipolar + reprojection), `_distribution`,
`confidence_mask`, `guided_upsample_depth`, `FeedforwardResult.load_zarr`, `FrameStore`,
`ate_translation` / `rpe` / `auc_at_threshold`, `_write_tum`, `build_localization_db` +
`load_reconstruction_features`, `_STAGE_DEPS` / `LEAF_STAGES` / `_stage_output_exists`,
`_frustum_world_aabbs` / `_aabbs_overlap`, `SequentialPairingOptions.quadratic_overlap`.

**New, and only this:**
1. Signed-residual **and parallax-angle** return from a refactored
   `compute_multiview_depth_confidence`. Both are already implicit in that loop — it
   unprojects the rays and computes `expected_d`, then keeps only a boolean. The bridge
   (`δd_equiv`, ρ) is arithmetic on those two arrays, not a new pass.
2. Normalised photometric warp, sharing that projection loop.
3. The `report.json` aggregator and its schema.

Roughly 70% of the stage is wiring calls that already exist.

## Runtime

**Measured:** dense depth pass, 60 frames @ 592×448, all N² pairs with the gate on = **3.9 s**
(1.09 ms/pair, `judged=60/60`).

| frames | depth + scale | + photometric | total (ceiling) |
|---|---|---|---|
| 60 | 3.9 s | ~8 s | ~12 s |
| 100 | 11 s | ~22 s | ~33 s |
| 300 (`max_frames`) | 1.6 min | ~3.3 min | **~5 min** |

These are **ceilings**: the frustum gate prunes non-overlapping pairs, and it prunes harder
the longer the trajectory. The 60-frame measurement is near full N² only because that scene
is a single room where nearly every pair genuinely overlaps.

**Epipolar is unmeasured.** No `verification.json` exists anywhere in the repo, so `verify`
has never been run to completion here. Structure: ~N·(10 + log₂N) pairs — **O(N log N), not
N²** — so ~5,400 at 300 frames, dominated by pairwise matching. Feature extraction is free
when `localize` has run (shared zarr cache).

**No runtime levers in v1.** Dense keeps all-pairs + gate, because that gate *is* the
revisit detector and cutting it would delete the drift signal. Pixel subsampling exists as
an **off-by-default module constant** (mv-confidence precedent: tuning knobs stay out of
config) — it is pure speed with no information cost at `report.json` granularity, available
if a specific scene proves slow.

**Task 1 of the plan measures the matcher**, because it is the only unmeasured component
and, after the above, the dominant one. Every remaining optimisation decision depends on it.

## Validation — a negative control per channel

A metric that does not move under an injected fault is decoration. Each channel gets a fault
whose magnitude and location are known:

| injected fault | must move | must **not** move |
|---|---|---|
| +2° on one pose | that camera's epipolar pairs (~2°), its track survival | clean pairs (<0.2°) |
| ×1.1 on one frame's depth | that frame's `median(r)` → ≈0.1 | **epipolar** |
| exposure shift on one frame | that frame's photometric residual | depth, epipolar |

The depth-scale control is the load-bearing one: it is the test that proves attribution
actually *separates* rather than three channels moving together.

**The bridge gets a quantitative control, not just a directional one.** `r = 0.1` predicts
`δd_equiv = 0.1 · d` in closed form, so the injected-scale test asserts a *number* — the
measured equivalent pixel error must match `0.1·d` per depth bin, and ρ must stay ≈1
(the pixel disagreement is fully explained by the injected depth error). The +2° pose
injection is the complement: ρ must go **≫ 1** there, since pixels move while depths stay
mutually consistent. Two injections, opposite ρ signatures, from one formula — that is what
makes ρ a measurement rather than a plausible-looking ratio.

**Rank control:** run on mapanything and vggt_omega on chess/seq-01, which differ 1.6× in
ATE. If the report cannot order those two, it will not separate anything.

**Sanity target:** reproduce the measured `evals/results/mv_vggt_omega` numbers — median
|rel| 0.37%, p90 2.27%, p99 25.67%, tightening to p90 0.92% at conf>p20 — or explain the
difference.

## Implementation principles

- Reuse existing functions; refactor rather than duplicate (one projection loop, not two).
- Delete what this obsoletes. `evals/scripts/depth_disagreement.py` is prior art and is
  retired once its residual lives in the refactored function.
- No premature abstraction, no config knobs beyond zero, no dead branches.
- Every public function gets a one-line docstring; block-level inline comments; `########`
  section dividers; `logging`, never `print()`.

## Traps carried from measurement

1. **Recon units are not metres** — 2.596 m/unit measured on chess/seq-01, varies per scene
   and backbone. Scale-free or normalised metrics only.
2. **Two ATE implementations disagree** — Sim(3) vs SE(3). State which.
3. **Model-res vs original-res K** — caused the 2026-08-11 mesh collapse. Every projection
   states its grid.
4. **Z-depth required**, and scale invariance holds only at `abs_thresh=0.0`.
5. **`triangulate_points` mutates its argument in place** and returns the same object —
   copy via `pycolmap.Reconstruction(recon)` first.
6. **`ba/colmap/sparse/0` has `num_observations = 0`** — BA's model cannot supply
   reprojection residuals.
7. **`_filter_observations` is self-confirming** — it selects observations by agreement with
   the model's own poses. This is why the report uses `verify`'s model-agnostic localization
   features and **not** VGGSfM tracks, which would flatter the model.
8. **Confidence is logits on some backbones** (LoGeR). Percentiles, never absolute
   thresholds, and never compared raw across backbones.
9. **`multiview_mask` clamps `min_views` per pixel** — a high K silently degrades to "all
   partners agree" rather than emptying the scene.
10. **AUC@5 is ill-conditioned indoors** — prefer ATE/RPE.

## Deferred follow-ons

- Retrieval (DinoSalad) revisit pairs for the **epipolar** channel — the dense channels
  already get revisits from the frustum gate.
- Per-pixel arrays (`report.zarr` or `mv_*`-style optional fields) for the summary viewer.
- Cross-scene / cross-backbone rollup — currently just reading several `report.json` files.
- Absolute, calibrated thresholds and any scene-level grade — blocked on the measured
  distributions this stage produces.
