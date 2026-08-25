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
  **Plan-time correction:** `FrameStore._STALENESS_KEYS` no longer exists —
  preproc-cleanup made store reuse existence-only ("reuse is by existence,
  never by comparison", frame_store.py). Toggling `preproc.undistort` on an
  existing scene therefore requires `preprocess(overwrite=True)`; documented
  beside the flag in base.yaml.
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
- absgrad/8e-4 pair — **verification VERDICT (recorded at plan time): NOT
  usable on 2dgs.** `rasterization_2dgs` accepts `absgrad`, but the backward
  sets `.absgrad` only on `means2d` (gsplat `cuda/_wrapper.py`,
  `_RasterizeToPixels2DGS.backward`), while our `DefaultStrategy` reads
  `info["gradient_2dgs"].absgrad` (`key_for_gradient="gradient_2dgs"`,
  `strategy/default.py:245`) — the `densify` tensor never gets the attribute,
  so `absgrad=True` raises AttributeError at the first refine step. Switching
  `key_for_gradient` to `means2d` is not parity either: splatfacto's 8e-4 is
  calibrated on the 3dgs means2d gradient, and gsplat added the dedicated
  `gradient_2dgs` densify tensor precisely because 2dgs means2d gradients are
  not the right densification signal. 2dgs keeps the measured-good
  `(absgrad=False, grow_grad2d=2e-4)`; only the other splatfacto args land.
  No `splats.absgrad` config field.
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
  FrameStore.
- `tests/splats/test_trainer.py` additions: permutation covers every view
  exactly once per epoch and reshuffles across epochs (seeded, deterministic);
  coarse-to-fine returns the right resolution/K at boundary steps; config
  rejects unknown keys unchanged; densification args reach the strategy
  (constructor introspection).
- absgrad-on-2dgs verification: done at plan time by source inspection
  (verdict above); no unit test.

## Public API surface

- `preproc.undistort` (bool, base.yaml, default false)
- `splats.num_downscales` (int, default 2), `splats.resolution_schedule`
  (int, default 3000)
- `collab_splats.preproc.undistort.{estimate_camera_distortion, undistort_frames}`

## Amendment 2026-08-25 — scene normalisation (`splats.normalize_scene`)

Implementation-time finding: gsplat's `scene_scale` is NOT equivalent to
splatfacto's scene normalisation. Splatfacto centres poses on the camera-position
mean and scales so max |camera coord| = 1 (L-inf), then trains with
`scene_scale = 1.0`; we kept world units and multiplied lrs/thresholds by
`1.1 x max L2 camera spread`. The two agree (to ~1.1–1.9x) for `means_lr` and
the 3D densification thresholds, but not for the depth loss (disparity L1 x
scene_scale), the 2dgs distortion loss (depth units), `pose_lr`, or the kNN
scale init. Measured consequence: the §3 splatfacto strategy args
(`prune_scale3d=0.5`, calibrated for the unit cube) applied in world units
multiply to a threshold no Gaussian exceeds — run 1 (undistort + strategy args,
no normalisation) grew to 3.1M Gaussians by step 10.5k and OOM'd (44 GB).

- New `SplatsConfig.normalize_scene` (bool, default false). On: cameras, seed
  points and depth targets are Sim3-normalised before training
  (`scene_normalization`), `scene_scale = 1.0`; Gaussians, cameras and
  pose-refiner translation deltas are mapped back to world units
  (`denormalize_outputs`) before `write_splat_outputs`, so ply/ckpt/zarr/mesh
  consumers are unchanged. The "up" re-orientation is skipped (no loss or lr
  depends on world rotation).
- Measurement amended: run 1 as specified is an incoherent config and is
  dropped. The A/B is now **full stack** (undistort + sampler + strategy args +
  coarse-to-fine + normalize_scene) vs the 19.21/0.621 baseline; per-lever
  attribution runs only if the full stack wins and the user asks.

## Measured 2026-08-25 — GH010229, 300 fr, InstantSfM 2dgs, pose_opt, 12k steps

| # | Config (cumulative) | PSNR | SSIM | Gaussians |
|---|---|---|---|---|
| 0 | baseline 30k, sparse targets, world units | 19.21 | 0.621 | 1.40M |
| 1 | 12k + permutation sampler | 18.97 | 0.615 | 2.04M |
| 2 | + `preproc.undistort` | 19.23 | 0.624 | 1.84M |
| 3 | + `normalize_scene`, pose lr shared 1e-5 | exploded (>2x growth by 4k) | — | — |
| 4 | + normalize, pose lr shared 7.9e-4 | 17.62 | 0.492 | 927k |
| 5 | + normalize, split pose lr (9fff4573) | 19.38 | 0.634 | 1.93M |
| 6 | #5 + dense COLMAP-aligned VDA targets | **20.27** | **0.663** | 1.87M |
| 7 | #6 + coarse-to-fine (`num_downscales 2`) | OOM @ 10.9k | — | — |

- Undistort +0.26 dB; normalisation +0.15 dB but ONLY with the pose-lr split
  (rotation lr x world extent, translation lr x training-frame scene_scale;
  shared lr in either frame loses); dense targets +0.89 dB on top (compound
  with normalisation — +0.17 measured unnormalised). Sampler neutral.
- Coarse-to-fine rejected: three OOMs at ~10.5k steps (last with the fixed
  pose lr) — over-densification at low resolution on this scene.
- Mesh from #6 splats (voxel 0.2 world ~ 6 cm): 2.10M verts, parity with the
  2.09M sfm reference, render-checked.
- Defaults on evidence: `preproc.undistort: true`, `splats.normalize_scene:
  true`, `num_downscales: 0`. Owed: 30k confirmation, holdout eval.
