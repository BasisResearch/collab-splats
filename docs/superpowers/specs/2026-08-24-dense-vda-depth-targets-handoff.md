# Handoff: spec "dense scale-aligned VDA depth targets" (per-frame least-squares to projected points3D)

**To the receiving model:** this is a handoff brief, not the spec. Run the repo's brainstorming/spec
process (CLAUDE.md: spec → `docs/superpowers/specs/`, plan → `docs/superpowers/plans/`). Everything
below is measured context from the instantsfm GoPro fidelity effort (2026-08-24, session on branch
`insfm-exec`, merged trunk `refactor/cu121-uv-migration`).

## Problem

InstantSfM scenes (`pointcloud.method: sfm`) carry **two inconsistent scales**:

- COLMAP reconstruction (poses, points3D): instantsfm world scale.
- `pointcloud.zarr` `depth` / `world_points`: Video-Depth-Anything **metric** scale.

Measured on GH010229 (300 frames, fps 2.0): VDA/COLMAP depth ratio **median 0.3092, ±35% p10–p90
per-frame jitter** (n = 19,572 track observations). InstantSfM's `use_depths` inv-depth prior
(weight 0.1) is too soft to pin the scales together — the drift is upstream slack, verified engaged.

Consequences today:

1. **Splat depth targets are sparse.** First training run with dense VDA targets collapsed
   (PSNR 6.15 — depth loss fought photometric for 30k steps). Fix `45077365` switched the sfm
   branch to `points3d_depth_maps` (sparse per-view rasterization of the reconstruction's own
   points3D, upstream-faithful port of vggt `vis/utils/colmap.py:361-382`). Result: splats
   PSNR **18.38 / SSIM 0.574** vs vggt_omega reference **18.97 / 0.596** on identical frames.
   Sparse supervision is the leading suspect for the gap; splat depth noise is also the floor
   for `mesh.source: splats` quality.
2. **Known limitation (documented, not yet fixed):** zarr `depth`/`world_points` stay VDA-metric
   vs COLMAP poses — breaks localization depth lookup and dashboard feature lifting for sfm scenes;
   `Reconstructor.mesh()` refuses `mesh.source: feedforward` on sfm scenes for the same reason.

## Proposed feature (lever #1 from the session's ranking)

Per-frame least-squares alignment of dense VDA depth to the depths of that frame's projected
points3D → **dense, COLMAP-scale depth targets**. Per-frame (not global median) because it also
removes the ±35% per-frame jitter, not just the 3.2× global offset.

## Anchor points in the code

- Dispatch site: `Reconstructor.splats()` sfm branch, `collab_splats/wrapper/reconstructor.py`
  (~line 1505): `if depth_on and self.config["pointcloud"]["method"] == "sfm":` →
  `points3d_depth_maps(result.reconstruction, [p.name for p in result.image_paths], height, width)`.
- `points3d_depth_maps` in `collab_splats/pointcloud/utils.py` — reuse its projection machinery
  (nearest-pixel rasterize, collision keeps nearer). Correspondences for the fit can come from
  points2D↔points3D tracks (exact pixels) rather than the rasterized map.
- VDA depth sources: per-frame `<scene>/instantsfm/depth_vda/images/npy/<stem>.npy` (518-wide
  model res) or `pointcloud.zarr` `depth` (same values). Image names are stems `frame_NNNNNN`.
- Trainer contract: `depth_targets` model-res, aligned to `image_paths` rows, **0 = no target**,
  nearest-resized per view in `collab_splats/splats/trainer.py`; loss is L1 disparity, weight 0.01.
- sfm scenes have **no `confidence` array — absent, never zeros** (repo-wide contract).

## Decisions the spec must make

1. **Fit model:** scale-only `s·d_vda`, or affine `s·d + t`, in depth or inverse-depth space.
   Robustness required (occlusions, bad tracks): median-of-ratios, IRLS, or RANSAC — justify choice.
2. **Scope:** (a) trainer depth targets only (minimal), vs (b) also rewrite `pointcloud.zarr`
   `depth`/`world_points` to COLMAP scale at `_run_sfm` time — (b) fixes the localization/dashboard
   known limitation and could lift the `mesh.source: feedforward` refusal, but changes the stored
   artifact contract (provenance attrs, existing scenes stale). Recommend deciding (b) explicitly,
   not by accident.
3. **Sparse-frame fallback:** frames with too few points3D observations (min-count guard) —
   fall back to sparse targets, neighbor-frame scale, or global median? The fps-1.0 probe
   registered only 60/100 frames; poorly-tracked frames are real.
4. **Where the fit lives:** `_run_sfm` (persisted once) vs `splats()` (recomputed). Persisting
   suggests stamping the per-frame scales into zarr attrs for provenance.

## Validation plan (already-measured baselines to beat)

- GH010229, 300 frames, 3dgs pose_opt 30k: splats PSNR 18.38 / SSIM 0.574 (sparse targets),
  omega reference 18.97 / 0.596. Report `splats_quality_report.json`.
- After splats: re-fuse mesh (`source: splats`, voxel 0.2 / sdf_trunc 0.8 / depth_trunc 100,
  `conf_percentile: 0`) and **render from scene cameras** (images 0/100/200) — vertex counts hid a
  catastrophic truncation once; renders are the only trusted mesh diagnostic.
- Cheap unit-level check: post-alignment residuals at track pixels should collapse the ±35%
  spread; report before/after ratio distribution.

## Traps / environment

- Scene lives at `/workspace/outputs/rerun_2026_08_23_instantsfm/GH010229`; `splats.zarr` is a
  symlink to `/tmp/claude-0/splats_local_GH010229` (**/workspace has an ~8.7 GB write quota
  invisible to df**).
- TSDF params for splat-source meshes on sfm scenes are world-scale-derived (~78× the omega
  reference params, NOT 3.2×) — see memory `project_instantsfm_backend.md` and CLAUDE.md.
- Heavy runs: tmux only, 46.6 GB cgroup cap, no parallel GPU work; venv
  `/opt/venv/reconstruction/bin/python`.
- `test_no_inline_defaults_in_source` forbids `.get(k, default)` in reconstructor.py.
- Related parked alternative: PAGaS-style init from splats.zarr depth renders (do not fold in).
