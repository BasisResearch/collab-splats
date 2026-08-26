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
