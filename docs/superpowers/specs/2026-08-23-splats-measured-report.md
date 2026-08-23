# Splats measured report — 3DGS-MCMC vs 2DGS-Default on `data/tutorial/`

Date: 2026-08-23. Companion to
[2026-08-22-splats-module-design.md](2026-08-22-splats-module-design.md) and the follow-ups plan
[2026-08-23-splats-followups.md](../plans/2026-08-23-splats-followups.md). Every number below comes
from a run under `evals/results/splats/` (gitignored); the producing scripts are
`evals/scripts/eval_splats.py` (train) and `evals/scripts/analyze_splats.py` (normals + mesh).

## Setup

- Scene: `data/tutorial/` → `data/outputs/feedforward.zarr`, vggt_omega, 30 frames, model res
  688×384, native res 1920×1080 (`frames.zarr`). Points masked at `conf_percentile=20`.
- Trainer: `collab_splats/splats/` on gsplat `d2f5c0f`, 30 000 steps, every view is a train view
  (no held-out split — PSNR/SSIM are **train-view** fit, not novel-view quality).
- 3DGS: `rasterization(packed=False, rasterize_mode="antialiased")`, MCMC `cap_max=1e6`,
  `opacity_reg`/`scale_reg` 0.01. 2DGS: `rasterization_2dgs`, DefaultStrategy
  `grow_grad2d=8e-4`, `distortion` loss. Both: depth 0.01, `normal_consistency` 0.05 from step
  7000, pose_opt off unless stated.
- Hardware: one GPU, fused-ssim built with `/usr/local/cuda/bin/nvcc` (CUDA 12.1).
- Native-res inputs rescale K from model res by `(orig_w/model_w, orig_h/model_h)`, parity with
  `_rescale_reconstruction_to_original_dimensions`.

## Table 1 — training

| run | primitive | res | psnr | ssim | gaussians | seconds | ms/step | notes |
|---|---|---|---|---|---|---|---|---|
| modelres | 3dgs | 688×384 | 21.74 | 0.790 | 1 000 000 | 1382 | 46.1 | before fused-ssim + normals gate |
| modelres | 2dgs | 688×384 | 13.51 | 0.234 | 138 030 | 1055 | 35.1 | **broken**: distortion weight 100 |
| native | 3dgs | 1920×1080 | **19.74** | **0.707** | 1 000 000 | 1709 | 57.0 | shipped defaults |
| native v1 | 2dgs | 1920×1080 | 13.92 | 0.350 | 214 940 | 1474 | 49.1 | **broken**: distortion weight 100 |
| native + pose_opt | 3dgs | 1920×1080 | 20.18 | 0.746 | 1 000 000 | 1797 | 59.9 | `--pose-opt` |
| native v2 | 2dgs | 1920×1080 | **19.96** | **0.707** | 1 170 273 | 1796 | 59.9 | distortion weight 0.01 (shipped) |

Final per-loss terms (last logged step):

| run | l1 | ssim | depth | normal_consistency | distortion / opacity_reg |
|---|---|---|---|---|---|
| native 3dgs | 0.033 | 0.139 | 0.186 | 0.304 | opacity_reg 0.365, scale_reg 0.002 |
| native 3dgs + pose_opt | 0.047 | 0.230 | 0.143 | 0.152 | opacity_reg 0.330 |
| native 2dgs v2 | 0.031 | 0.144 | 0.315 | 0.394 | distortion 0.035 |
| native 2dgs v1 (broken) | 0.123 | 0.624 | 49.5 | 0.012 | distortion 0.000 |

Native 3DGS per-frame PSNR spread: min 17.34, median 19.46, max 24.58 — the scene is
blur/exposure-limited on some frames; the logged single-step loss is one view, so it swings
±0.04 between neighbouring log lines.

Model-res and native PSNR are **not comparable** (different pixel grids, the model-res run fits
a 5× smaller image). Compare within a resolution only.

## Perf (measured before / after the two leaks)

- `fused_ssim` was absent from the venv → `gsplat.losses.ssim_loss` torch fallback. 3DGS normals
  were rendered every step though `normal_consistency` starts at 7000. Step-500 rate
  20.5 → 37.5 steps/s (1.8×) after installing fused-ssim (`pyproject` git pin
  `a7c48d6`) and gating `render_normals` on `loss_active(step, normal_spec)`.
- Once normals switch on at 7000: 23.8 → 15.5 steps/s (−35%) — `extra_signals` through the
  3DGS kernel is the single most expensive optional term.
- 3DGS native 57 ms/step vs 2DGS 60 ms/step at 30k: parity, once 2DGS stops growing (~16k).

## Table 2 — normals sanity (`analyze_splats.py`, alpha > 0.5 pixels, 30 views, 62.0M px)

| primitive | unit-norm frac | unit-norm frac (/alpha) | angle vs depth-normal mean | median | folded mean | folded median |
|---|---|---|---|---|---|---|
| 3dgs | 1.0000 | 0.9998 | 30.84° | 21.11° | 27.79° | 21.08° |
| 2dgs | 0.7058 | 0.7119 | 31.30° | 21.01° | 28.38° | 20.97° |

- 3DGS `extra_signals` normals are unit length with no sign flips (folded ≈ raw) — the
  shortest-axis definition is consistent across views.
- 2DGS normals are alpha-weighted by upstream (non-unit on 29% of opaque pixels even after
  dividing by alpha — semi-transparent stacking), same median angle.
- Both sit ~21° median off the finite-difference depth normal. That is the depth map's noise as
  much as the normals': the depth prior is nearest-upsampled 688×384 → 1920×1080, so the
  finite-difference reference is blocky. Usable as a consistency signal, not as a
  ground-truth-grade surface normal.

## Table 3 — TSDF mesh (`mesh.source`; open3d_tsdf, voxel 0.0025, sdf_trunc 0.01, depth_trunc 1.5, clean_repair, color_map 0)

| source | vertices | triangles | components | largest comp. frac | seconds |
|---|---|---|---|---|---|
| feedforward | 1 526 729 | 2 812 219 | 50 612 | 0.434 | 68.4 |
| 3dgs | 2 870 379 | 5 088 310 | 134 413 | 0.488 | 102.3 |
| 2dgs | 2 296 571 | 4 027 384 | 117 988 | 0.475 | 75.0 |

- Splat-rendered depth fuses into 1.5–1.9× more vertices and 2.3–2.7× more components than the
  feedforward depth: rendered depth is speckled at 2.5 mm voxels and `clean_repair` removes
  only the out-of-bounds components (302 for 3dgs, 1750 for 2dgs).
- 2DGS mesh is cleaner than 3DGS (−20% vertices, −12% components) as expected from
  surface-aligned primitives, but not the step change the 2DGS paper shows — the distortion
  loss at 0.01 barely binds on this scene (final term 0.035).
- Largest-component fraction is a few points *higher* for splats: the main surface is denser,
  the extra components are small speckle.

## Verdict

1. **3DGS stays the default.** Same PSNR/SSIM as 2DGS at native res (19.74/0.707 vs
   19.96/0.707), fixed 1.0M budget, 5% faster per step, cleaner normals (unit, no flips).
2. **2DGS works after the distortion fix** (13.9 → 19.96 PSNR). The shipped weight 100.0 was a
   transcription error — upstream `simple_trainer_2dgs.py` uses `dist_lambda = 1e-2` (and has
   it off by default). At 100 the loss pruned 500k → 140–215k gaussians and the photometric
   terms never recovered.
3. **`mesh.source: splats` is built and wired but not better**: both splat meshes carry 2.3–2.7×
   more speckle components than `feedforward`. Keep `mesh.source: feedforward` as default.
4. **pose_opt is now the default** (`splats.pose_opt: true`, flipped 2026-08-23). On the
   tutorial it is +0.44 dB / +0.04 SSIM for +5% time; on the 300-frame GoPro scene below it is
   +2.2 dB (3DGS) / +1.9 dB (2DGS) for +3% — feedforward poses, not the depth prior, are the
   ceiling on long walking captures.
5. **3DGS `extra_signals` normals are usable** (unit, consistent) but ~21° median off the depth
   normal; don't treat them as measured surface normals.

## Table 4 — GoPro scene `2026_07_15-Goprosplat-GH010229` (300 frames @ 2 fps, 1920×1080, vggt_omega, 30k steps)

Full pipeline rerun 2026-08-23 (`/workspace/outputs/rerun_2026_08_23/`): quality report +
`sample_fps` (photometric gate never binding — laplacian medians 8–12k vs gate 50),
vggt_omega, talk2dino, splats, native-res TSDF mesh. Per-segment columns are mean PSNR over
selected-frame index ranges; 130–200 is t = 238–364 s, the stretch with 40% faster camera
motion (55 px/pair vs 40) and 5× more dark-clipped pixels.

| run | gaussians | PSNR | SSIM | 0–130 | 130–200 | 200–300 | time (s) |
|---|---|---|---|---|---|---|---|
| 3dgs | 1 000 000 | 16.81 | 0.464 | 17.81 | 14.61 | 17.05 | 2035 |
| 2dgs, grow_grad2d 4e-4 | 391 522 | 16.54 | 0.441 | 17.43 | 14.64 | 16.70 | 1882 |
| 3dgs + pose_opt | 1 000 000 | **18.97** | **0.596** | **21.41** | 15.57 | 18.17 | 2061 |
| 2dgs + pose_opt, grow_grad2d 4e-4 | 723 543 | 18.48 | 0.564 | 20.82 | 15.33 | 17.64 | 2100 |

- pose_opt is the lever on both primitives; primitive choice is worth 0.3–0.5 dB in 3DGS's
  favour at every segment.
- Halving `grow_grad2d` did NOT add 2DGS gaussians (0.39M, below the 3DGS cap); with pose_opt
  the same threshold gave 0.72M — `DefaultStrategy` densification follows gradient
  consistency, i.e. pose quality, not the threshold.
- Segment 130–200 stays at ~15.5 dB in all four runs. Local pose refinement cannot recover
  it; the upstream feedforward poses/depth are wrong there. Next levers are
  `pointcloud.bundle_adjustment: true` or `frame_selection: optical_flow` (denser frames on
  the fast stretch), not splat knobs.
- Mesh stage OOM'd (exit 137, cgroup 46.6 GB) in `color_map_optimization` at 300 native
  frames × 300 iterations; rerun with `color_map_iterations: 0`. Memory scales with
  frames × native resolution — cap at ~100 frames or disable.

## Traps hit

- `distortion: {weight: 100.0}` shipped in base.yaml, trainer default and spec; now 0.01
  everywhere. Symptom is silent: training completes, PSNR ~13.5, gaussians collapse.
- `fused_ssim` silently absent → torch SSIM fallback, 1.8× slower. `setup.sh` smoke import now
  includes it; it needs nvcc (`no-build-isolation-package`).
- Eval v1 ran at model res by accident — native (frames.zarr + rescaled K) is the rule; PSNR
  across resolutions is not comparable.
- `CUDA_VISIBLE_DEVICES=""` breaks importing `collab_splats.mesh` (vggt `layers/mlp.py` CUDA
  warm-up at import) — run `analyze_splats.py` and `tests/evals` with the GPU visible.
- `analyze_splats.py` logged tables list primitives present in `<results>/<prim>/splats.zarr`;
  the 2DGS v2 run was trained into `tutorial_2dgs_v2/` and copied over `tutorial/2dgs/`
  (v1 kept as `tutorial/2dgs_v1_broken/`).

## Not measured

- Novel-view quality (no held-out split) and ground-truth depth/normal error (7-Scenes).
- `cap_max` / `grow_grad2d` sweeps; more than 30 frames (pipeline default is 300).
- Whether 2DGS with the distortion loss at upstream's `dist_lambda` but *on from step 3000*
  beats upstream's default (off) — only one setting was run.
- Docker image rebuild against the new lock: no `docker` binary in this container.
- `05_semantics` notebooks: `FeatureAutoencoder.load(path)` one-arg calls are stale
  (signature is `(path, extractor)`) — pre-existing, not fixed here.
