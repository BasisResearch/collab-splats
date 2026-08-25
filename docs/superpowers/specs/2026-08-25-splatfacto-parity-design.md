# Splatfacto Parity — Undistortion Preproc + Trainer Alignment

**Date:** 2026-08-25
**Status:** Approved (design), plan to follow
**Branch:** refactor/cu121-uv-migration

## Problem

The retired nerfstudio/rade-gs pipeline produced visibly better splats than the
minimal `collab_splats/splats/` trainer on the same class of footage. Audit
against nerfstudio main @ 50e0e3c (1.1.5) and the retired `rade_gs.py` method
config (git `145230ff~1`) shows the optimizer core was ported faithfully
(identical lrs, loss mix, SH schedule) — the divergence is machinery around it:

| Piece | rade-gs / splatfacto | Our trainer today |
|---|---|---|
| Undistortion | cv2.undistort at load, alpha=0, ROI crop, K rewritten | none — GoPro Linear leaves 3–4 px corner residual (measured k1≈+0.006, k2≈−0.003) |
| View sampling | seeded shuffled permutation, pop until empty, reshuffle | `torch.randint` with replacement (visit spread ±17% at 30k/875) |
| Coarse-to-fine | start ÷4 res, double every `resolution_schedule=3000`, `num_downscales=2` | native res from step 0 |
| Densification (DefaultStrategy) | `densify_grad_thresh 8e-4` **paired with `absgrad=True`**, `prune_opa 0.1`, `prune_scale3d 0.5`, `refine_scale2d_stop_iter 4000`, `pause_refine_after_reset n_views+100` | grow 2e-4 with absgrad=False; gsplat defaults elsewhere |
| Pose opt | camera_optimizer mode="off" | on (+2.2 dB measured GoPro — keep) |
| Depth loss | none | 0.01 sfm/feedforward targets (+0.13–0.17 dB measured — keep) |
| Exposure | none (bilateral grid ships off) | none |

Key provenance finding: base.yaml's original `grow_grad2d: 8e-4` came from
splatfacto, where it is calibrated for `absgrad=True`. Our port kept the
threshold but not absgrad — the mismatched pair starved 2dgs densification
(160k gaussians, −1.26 dB). The correct restoration is the PAIR, not 2e-4.

## Decision (option A: parity + keep our wins)

Adopt splatfacto's undistortion, sampling, coarse-to-fine, and densification
settings. KEEP our measured wins: depth loss, pose_opt default-on, MCMC for
3dgs, eval on pose-corrected cameras.

Out of scope: exposure/appearance module (splatfacto ships none; follow-on if
A/B still trails), strict-parity eval convention, densification sweeps beyond
splatfacto's values, any 875-frame rerun.

## Design

### 1. Undistortion — general preproc step at the `frames.zarr` boundary

New `collab_splats/preproc/undistort.py`:

- `estimate_camera_distortion(frames) -> DistortionProfile` — pycolmap
  OPENCV-model self-calibration on ≤60 evenly spaced already-selected frames,
  shared camera (the method already validated when k1/k2 were measured).
  Profile: k1 k2 p1 p2 + calibrated K + input dims.
- `undistort_frames(frames, profile) -> (frames, K_new)` — splatfacto-exact
  path: `cv2.getOptimalNewCameraMatrix(alpha=0)` → `cv2.undistort` → ROI crop
  → rewritten K.
- Wired into `Reconstructor.extract_frames` between frame selection and
  `FrameStore.create`. Profile + K_new + ROI stored in frames.zarr provenance.
  `undistort` joins `FrameStore._STALENESS_KEYS` so toggling the flag rebuilds
  the store.
- Rationale for preproc placement: every consumer (feedforward backbones,
  SIFT/InstantSfM, splat trainer, localization DB export) assumes pinhole;
  undistorting once fixes all of them and keeps a single camera contract.
  Nothing downstream changes.
- Config surface: ONE boolean `preproc.undistort`, default **false** until the
  A/B below measures the win (GoPro run overrides set true; flip the default
  on evidence).
- Documented limitation: localization QUERY images are not undistorted (query
  path reads arbitrary user images, not frames.zarr).

### 2. Trainer — permutation sampling + coarse-to-fine

- Sampler: seeded shuffled index list, pop until empty, reshuffle on empty
  (splatfacto's exact scheme). Guarantees each view trains `max_steps/n_views`
  ± 1 times. No config surface.
- Coarse-to-fine: new `SplatsConfig` fields `num_downscales: 2` and
  `resolution_schedule: 3000` (splatfacto defaults). Effective downscale at a
  step is `2^max(0, num_downscales - step // resolution_schedule)`; target
  image, K, and depth target are downscaled accordingly (image bilinear, depth
  nearest). `num_downscales: 0` disables.

### 3. Densification — splatfacto arg set on DefaultStrategy (2dgs)

- Pass splatfacto's non-default args explicitly: `prune_opa=0.1`,
  `prune_scale3d=0.5`, `refine_scale2d_stop_iter=4000`,
  `pause_refine_after_reset=n_views+100`.
- absgrad/8e-4 pair gated on a verification task: splatfacto's pair was
  calibrated on 3dgs `rasterization`; whether `rasterization_2dgs` produces
  `means2d.absgrad` is unverified. If supported → defaults become
  `absgrad: true` + `grow_grad2d: 8e-4` (new `absgrad` config field). If not →
  2dgs keeps the measured-good `(False, 2e-4)` and only the other args land.
- 3dgs stays MCMC (`cap_max`), untouched by this section.

### 4. Measurement (after implementation)

A/B on GH010229, 300-frame instantsfm 2dgs, 30k steps, against baselines
19.21 (sparse targets) / 19.38 (dense targets, insfm-exec):

1. baseline + `preproc.undistort: true` only;
2. run 1 + all trainer changes (sampling, coarse-to-fine, densification).

Two runs, ~1–1.5 h GPU each. Report deltas per lever. No 875-frame runs.

## Testing

- `tests/preproc/test_undistort.py`: synthetic-distortion roundtrip (apply
  known k1/k2 → estimate → undistort → residual < tolerance); alpha=0 crop
  keeps dims even and K consistent; profile provenance round-trips through
  FrameStore; staleness key triggers rebuild.
- `tests/splats/test_trainer.py` additions: permutation covers every view
  exactly once per epoch and reshuffles across epochs (seeded, deterministic);
  coarse-to-fine returns the right resolution/K at boundary steps; config
  rejects unknown keys unchanged; densification args reach the strategy
  (constructor introspection).
- absgrad-on-2dgs verification is a plan task with a recorded verdict, not a
  unit test (CUDA-dependent).

## Public API surface

- `preproc.undistort` (bool, base.yaml, default false)
- `splats.num_downscales` (int, default 2), `splats.resolution_schedule`
  (int, default 3000), `splats.absgrad` (bool, default set by the
  verification task)
- `collab_splats.preproc.undistort.{estimate_camera_distortion, undistort_frames}`
