# PSNR levers: per-image appearance, depth-loss decay, sharper frame pick

**Date:** 2026-08-26
**Status:** approved design
**Baseline:** GH010229 (300 fr, InstantSfM, undistort + normalize_scene + dense COLMAP-aligned VDA
targets), 3dgs MCMC + coarse-to-fine, 12k steps: **21.02 dB / 0.708 SSIM** (see
`2026-08-25-splatfacto-parity-design.md` §Measured).

Three independent levers, one spec, evaluated one at a time against the baseline.

## 1. Per-image appearance (affine, "A")

GoPro auto-exposure / white balance drifts across the sequence; the splats absorb it as colour
noise. Splatfacto answers with a bilateral grid; we start with the cheapest model and only
vendor the grid if this one is weak.

- `collab_splats/splats/appearance.py` — `AppearanceModule(n_images)`: `nn.Embedding(n, 6)`
  zero-initialised, split into per-channel `gain = 1 + g` and `bias = b`.
  `forward(rgb, camera_ids)` returns `rgb * gain + bias` (broadcast over H, W).
- Applied to `render["rgb"]` **before** the random-background blend in the train loop and in
  `render_all_views` (the background is not part of the photo's exposure).
- Regulariser: `appearance_reg` becomes an entry of `OPTIONAL_LOSSES` — mean squared
  embedding weight, pulling towards identity — so it rides the existing `losses:` schedule
  machinery. Value is `None` when the module is off.
- Optimiser: its own Adam at `appearance_lr` with the same exponential 0.01x decay the pose
  refiner uses (`make_pose_refiner` pattern; factor it as `make_appearance_module`).
- Config (`splats:`): `appearance_opt: false`, `appearance_lr: 1.0e-3`,
  `losses.appearance_reg: {weight: 1.0e-3}`. `SplatsConfig` mirrors both scalars.
- Checkpoint: `ckpt.pt["appearance"]` = state_dict or `None`; `write_splat_outputs` /
  `render_all_views` take `appearance: AppearanceModule | None`. Old checkpoints (no key) load
  as `None`.
- Held-out / novel views have no embedding and render with identity — the module is a training
  aid for the geometry, not a render-time feature. Spec'd, not solved.
- Follow-on "B" (vendored gsplat `lib_bilagrid.py`) only if A gains < +0.2 dB.

## 2. Depth-loss decay

Dense depth targets are what lifted 19.38 → 20.27, but they are metric-noisy VDA predictions:
strong early (they place geometry), a liability late (they fight photometric detail). Splatfacto
has no depth loss at all.

- Schedule entries grow two optional keys: `{weight, start, end, end_weight}`.
- New `loss_weight(step, spec) -> float` in `losses.py`:
  - `spec is None` or `step < start` → 0
  - no `end` → `weight`
  - `start <= step <= end` → log-linear interpolation `weight → end_weight`
  - `step > end` → `end_weight`
- `loss_active(step, spec)` becomes `loss_weight(step, spec) > 0`; `compute_losses` uses the
  interpolated weight.
- Validation at trainer start (`_validate_loss_schedule`): when `end` is present, `end > start`
  and `weight > 0` and `end_weight > 0` (log-linear needs positive endpoints), else `ValueError`.
- The 2dgs distortion `/scale` rescaling divides `end_weight` too when present.
- `base.yaml` unchanged (no decay by default until measured). Eval override:
  `depth: {weight: 0.01, end: 12000, end_weight: 0.001}`.

## 3. `preproc.search_radius`

`_sample_by_quality` already picks the sharpest usable frame within ±`search_radius` of each
target, but the radius is hard-coded to 3 in `sample_fps` / `sample_uniform`. At 2 fps on 30 fps
video the spacing is 15 frames, so the window could be ±7 without overlap; the existing cap
`min((spacing - 1) // 2, search_radius)` guarantees that.

- `configs/base.yaml` `preproc.search_radius: 7` with a comment on the spacing cap.
- `Reconstructor` reads it and forwards to `sample_fps` / `sample_uniform` (optical_flow has no
  window; it is not touched).
- Changing the pick changes the frame set, so the eval run rebuilds frames + SfM + zarr.

## Testing

- `tests/splats/test_losses.py`: `loss_weight` constant / before start / mid-decay (geometric
  midpoint) / after end / validation errors; `compute_losses` uses the decayed weight.
- `tests/splats/test_appearance.py`: identity at init (rgb unchanged), gradients reach the
  embedding, only the addressed camera's row changes, checkpoint round-trip through
  `write_splat_outputs`.
- `tests/preproc/test_sampling.py`: radius capped by spacing; radius 7 picks the sharpest frame in
  the wider window.
- `tests/wrapper/test_reconstructor.py`: `search_radius` forwarded to the sampler.

## Evaluation

Base config = `gopro_3dgs_c2f_overrides.yaml` (21.02 / 0.708). Sequential tmux runs, 12k each,
`semantics: {enabled: false}`:

1. `+appearance` (`appearance_opt: true`)
2. `+decay` (depth `end: 12000, end_weight: 0.001`)
3. `+radius7` (rebuild frames/SfM/zarr; also reports SfM stats to separate frame-set effects)
4. combined, if at least two win

Each run also meshes (source: splats) and archives as `instantsfm/splats_<tag>/`. Results go into
this spec's §Measured; defaults flip on evidence only.

## Measured — 2026-08-26 (GH010229, 300 fr, InstantSfM + retriangulation, undistort, dense aligned VDA targets)

Base = `gopro_3dgs_c2f_overrides.yaml`: 3dgs MCMC + c2f (`num_downscales 2`,
`resolution_schedule 3000`), `normalize_scene`, `pose_opt`, 12k steps, cap_max 1M.

| run | PSNR | SSIM | gaussians | train s | Δ PSNR |
|---|---|---|---|---|---|
| base 3dgs MCMC + c2f | 21.020 | 0.7076 | 1 000 000 | 596 | — |
| **+ appearance (`appearance_opt: true`)** | **21.606** | **0.7169** | 1 000 000 | 601 | **+0.59** |
| + depth decay (`end: 12000, end_weight: 0.001`) | 21.117 | 0.7094 | 1 000 000 | 787 | +0.10 |
| + `preproc.search_radius: 7` (fresh scene) | 21.07 | 0.708 | 1 000 000 | 589 | +0.05 |

Same-target loss trace (appearance and decay share the base's 300 images; radius7 does not):

| step | base | appearance | decay | radius7 |
|---|---|---|---|---|
| 3000 | 0.2332 | 0.2230 | 0.2265 | 0.2337 |
| 6000 | 0.1896 | 0.1791 | 0.1850 | 0.1859 |
| 9000 | 0.1051 | 0.0995 | 0.1013 | 0.1039 |

### Verdicts

- **Appearance: WIN, +0.59 dB for +5 s.** Clears the 0.2 dB bar set for option A, so the
  bilateral grid (option B) is not needed. Caveat: PSNR is measured on train views with the
  per-image correction applied, so part of the gain is a free 6-param affine fit per image —
  `$SP/appearance_identity_eval.py` decomposes it (geometry gain vs fit) and is still owed.
  Learned params are smooth, not per-frame noise: neighbour |Δ| median 0.0081 vs overall std
  0.0508 (ratio 0.16), luminance gain p5 0.917 / p95 1.077, per-view WB spread median 0.0181.
- **Depth decay: MARGINAL, +0.10 dB for +32% wall-clock (596 → 787 s).** Not worth defaulting on.
- **`search_radius: 7`: NO PSNR EFFECT, +0.05 dB** — and measured on a different frame set, so
  not even a clean A/B. The sampler did what it claims: Laplacian variance p10 233.9 → 252.5
  (+8.0%), mean 379.3 → 387.3, min 88.6 → 93.6; only 100/300 slots kept the same frame (median
  shift 3, max 10). Cost is spacing regularity: 37–50 → 29–58, which cost 0.5% of the points
  (104 436 → 103 871 points3D, 536 522 → 533 094 obs) at unchanged track quality (mean track
  5.14 → 5.13, 300/300 registered). Keep for mesh/localization sharpness, not for splat PSNR.

The plan's step 4 (combined run) required at least two winners; only one lever won, so it is
not run by default.

### Incidental fix

`fix(preproc): cap SIFT threads in undistort self-calibration` (17381b58). The fresh-scene
rebuild SIGKILLed during `estimate_camera_distortion`: the pycolmap wheel here is CPU-only
(`has_cuda False`), so the default `num_threads = -1` spawned one SIFT thread per host core
(96) on 1080p frames and blew past the 46.6 GB cgroup cap. Now capped at 8, mirroring
`pointcloud/sfm.py::_SIFT_NUM_THREADS`. Any fresh undistorted scene would have hit this.

## Combined runs (user-requested, both primitives)

Run anyway on request, on the r7 scene (`GH010229_undist_r7`, already built), all three levers
stacked: `appearance_opt: true` + depth decay `{0.01 → 0.001 @ 12k}` + the radius-7 frame set.
Configs `$SP/gopro_lever_combined.yaml` (3dgs) and `$SP/gopro_lever_combined_2dgs.yaml` (2dgs).

| run | primitive | PSNR | SSIM | gaussians | train s | mesh verts / tris | main-frac |
|---|---|---|---|---|---|---|---|
| base MCMC + c2f | 3dgs | 21.020 | 0.7076 | 1M | 596 | 2 840 472 / 4 461 422 | 0.41 |
| appearance only | 3dgs | **21.606** | **0.7169** | 1M | 601 | 2 981 732 / 4 662 060 | — |
| combined (all 3) | 3dgs | 21.400 | 0.7140 | 1M | 612 | 3 453 612 / 5 420 513 | 0.457 |
| dense_best | 2dgs | 20.270 | 0.6630 | 1.87M | — | 2 100 000 (approx) | 0.60 |
| combined (all 3) | 2dgs | **20.800** | **0.6780** | 1 875 613 | 803 | 1 967 540 / 3 310 479 | 0.634 |

**The levers are anti-additive on 3dgs.** Summing the isolated deltas predicts 21.76; the
combined run gives 21.400, i.e. −0.21 dB below appearance alone. Decay's late depth
down-weight and the r7 frame set both pull against the appearance fit. Appearance alone
remains the best 3dgs configuration.

**2dgs gains +0.53 dB** (20.270 → 20.800) at effectively unchanged gaussian count, tracking
the +0.59 dB appearance-alone gain measured on 3dgs — the appearance model is the whole effect
on this primitive too, and it is primitive-agnostic by construction (`trainer.py:557` applies
the affine to `render["rgb"]` after rasterization). This is the new 2dgs high-water mark.
Coarse-to-fine stayed pinned off for 2dgs; DefaultStrategy still OOMs with it.

**Mesh.** 2dgs remains the better surface source: main-component fraction 0.634 vs 0.457, at
43% fewer triangles. The 3dgs combined mesh is the largest of the five runs (3.45M verts) but
its main-fraction barely moves, so the extra geometry is fragments, not surface.

**Verdict unchanged:** ship `appearance_opt` on its own. Do not stack decay or radius-7 for PSNR.
