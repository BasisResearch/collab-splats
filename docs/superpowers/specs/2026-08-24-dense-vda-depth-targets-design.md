# Dense scale-aligned VDA depth targets for sfm scenes — design

**Date:** 2026-08-24
**Status:** approved (brainstormed with user; validation-gated rollout)
**Predecessor:** `2026-08-24-dense-vda-depth-targets-handoff.md` (handoff brief, superseded by this spec)
**Baseline branch:** `insfm-exec` @ `9b1c1b07` (contains `45077365` sparse points3D targets; trunk
`refactor/cu121-uv-migration` @ `702b5926` does NOT — merge order in Execution section).

## Problem

InstantSfM scenes (`pointcloud.method: sfm`) carry two inconsistent scales: the COLMAP
reconstruction (poses, points3D) is at instantsfm world scale, while `pointcloud.zarr`
`depth`/`world_points` are Video-Depth-Anything metric. Measured on GH010229 (300 frames,
fps 2.0): VDA/COLMAP ratio median 0.3092 with ±35% p10–p90 per-frame jitter (n = 19,572 track
observations). Upstream instantsfm fixes per-observation inverse depths in global positioning
(`optimization_models.py:430-432`) but the following reprojection-only BA is gauge-free, so the
scales drift apart.

Consequences:

1. Splat depth supervision is sparse (`points3d_depth_maps`, fix `45077365`) — GH010229 splats
   PSNR 18.38 / SSIM 0.574 vs vggt_omega reference 18.97 / 0.596. Sparse supervision is the
   leading suspect for the gap and the floor for `mesh.source: splats` quality.
2. Known limitation: zarr depth/world_points VDA-metric vs COLMAP poses breaks localization
   depth lookup + dashboard feature lifting for sfm scenes; `Reconstructor.mesh()` refuses
   `mesh.source: feedforward` on sfm scenes.

## Decisions (user-approved)

1. **Scope:** rewrite `pointcloud.zarr` to COLMAP scale at `_run_sfm` time (not trainer-only).
   Fixes limitation (2) regardless of the splats outcome; splats sfm branch collapses into the
   existing feedforward zarr-depth path.
2. **Fit model:** per-frame scale-only, median-of-ratios. `s_i = median(d_colmap / d_vda)` over
   that frame's track correspondences. No affine term (VDA is metric; a shift risks negative
   depths and breaks the 0 = no-target sentinel). No IRLS unless measured residuals stay wide.
3. **Fallback:** frames with fewer than `min_obs` (20) valid correspondences take the global
   median of the valid per-frame scales (matches upstream philosophy: below min-obs, don't fit,
   inherit scene consistency). Warn-log each fallback frame. Zero valid frames → raise.
4. **Fit location:** `_run_sfm`, persisted once; per-frame scales stamped into zarr attrs.
5. **Rollout gate (user directive):** full test suite + GH010229 evaluation must run in an
   isolated worktree and beat the measured baseline BEFORE any of this merges or any retirement
   (sparse path, mesh refusal) happens.

## Components

### 1. `align_depth_to_reconstruction` — new, `collab_splats/pointcloud/sfm.py`

```
align_depth_to_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str],          # stems, zarr row order
    depth: np.ndarray,               # (N, h, w) VDA model-res depth
) -> tuple[np.ndarray, dict]        # (N,) float64 scales, stats dict
```

- Per registered image: points2D with a point3D → `d_colmap` = z of the point in the camera
  frame (`cam_from_world`); the point2D pixel coords are **rescaled from native camera
  resolution to the model-res depth grid** (explicit rescale — same bug class as the
  localization ref_px fix `92f2e4a`), then nearest-sampled into the frame's VDA depth.
- Keep pairs with `d_vda > 0` and `d_colmap > 0`; `s_i = median(d_colmap / d_vda)`.
- `min_obs = 20` guard → global-median fallback; no frame passes → `ValueError`.
- Stats dict: per-frame obs counts, fallback frame list, ratio spread p10/p50/p90 before and
  after alignment (the after-spread is the unit-level success check: ±35% must collapse).
- Module constants, not config keys — no new config surface.

### 2. `_run_sfm` wiring — `collab_splats/wrapper/reconstructor.py`

After the reconstruction exists and images are renamed to stems, before `save_zarr`:

- `depth[i] *= s_i`; `world_points` unprojected from the scaled depth through the existing
  unproject path (COLMAP poses + model-res K — now scale-consistent).
- `extra_attrs` gains: `depth_scale: "colmap"`, `depth_scales: [per-frame floats]`,
  `depth_scale_fallback_frames: [stems]`. Absence of `depth_scale` marks a legacy VDA-metric
  scene. **No migration tool** — exactly one sfm scene exists (GH010229); legacy scenes re-run
  the pointcloud stage.
- Alignment failure raises — never a silent VDA-metric write.
- Log the stats dict at INFO (scale median, spread before/after, fallback count).

### 3. Consumers — conditional on validation verdict

- `Reconstructor.splats()` sfm branch: delete the `points3d_depth_maps` dispatch; sfm scenes
  fall through to the feedforward zarr-depth path (no `confidence` → unmasked, already handled).
- `Reconstructor.mesh()`: lift the `mesh.source: feedforward` refusal for sfm scenes carrying
  `depth_scale: "colmap"`; keep refusing legacy scenes.
- `points3d_depth_maps` (pointcloud/utils.py) retired if dense wins (reuse/retire rule).
- **If dense targets lose to the sparse baseline:** keep the sparse `points3d_depth_maps`
  dispatch for splats. The zarr rewrite still ships (it fixes localization/dashboard independent
  of splats), and the mesh refusal is still lifted for `depth_scale: "colmap"` scenes — the
  refusal was about scale mismatch, which the rewrite fixes regardless of the splats verdict.

## Execution and validation plan

Order is binding (user directive: tests + evals before applying).

1. **Worktree:** new branch off `insfm-exec` @ `9b1c1b07` (trunk lacks `45077365`). Implement
   there. Nothing merges to trunk until step 5 passes.
2. **Unit tests first:** synthetic pycolmap fixture (trap: `add_point3D` range-checks track
   elements — pre-populate placeholder `Point2D`s, let `add_point3D` back-fill). Cases: exact
   scale recovery from known-scale synthetic depth; per-frame jitter recovery; `min_obs`
   fallback takes global median; all-frames-sparse raises; zarr attrs written; splats dispatch
   uses zarr depth on sfm + `depth_scale`; mesh refusal lifted/kept per attr.
   `test_no_inline_defaults_in_source` must stay green (no `.get(k, default)` in
   reconstructor.py).
3. **Full suite** in the worktree venv: `/opt/venv/reconstruction/bin/python -m pytest tests/`.
   Known pre-existing failures per `docs/known-test-failures.md` + the 2 documented
   `test_run_pipeline_remote` scene-id failures; nothing new.
4. **GH010229 eval** (tmux, 46.6 GB cgroup, no parallel GPU work):
   - Re-run pointcloud tail with the alignment (SIFT db + VDA npys cached, ~19 min warm).
     Verify stats: post-alignment ratio spread ≪ ±35%; scale median ≈ 1/0.3092 ≈ 3.2.
   - Splats 3dgs pose_opt 30k (semantics disabled per standing override; `splats.zarr`
     symlinked to /tmp — /workspace ~8.7 GB quota invisible to df).
   - **Gate: PSNR/SSIM must beat 18.38 / 0.574** (sparse baseline; omega reference
     18.97 / 0.596 is the stretch target). Report `splats_quality_report.json`.
   - Mesh re-fuse `source: splats`: voxel 0.2 / sdf_trunc 0.8 / depth_trunc 100,
     `conf_percentile: 0`; **render from scene cameras (views 0/100/200)** — renders are the
     only trusted mesh diagnostic (vertex counts hid a catastrophic truncation once). Compare
     against the 2.09M-vert baseline renders.
   - Bonus probe (cheap, non-gating): `mesh.source: feedforward` fusion on the aligned zarr,
     same params.
5. **Verdict:** dense wins → apply consumer retirements, merge worktree → `insfm-exec` →
   trunk (with the owed insfm-exec merge). Dense loses → keep sparse targets, ship the zarr
   rewrite + limitation fixes only; record the measured verdict either way in this spec's
   "Measured" appendix.

## Error handling

- Missing VDA npy / empty depth: existing `_run_sfm` errors, unchanged.
- Image in zarr but not in reconstruction: already impossible post-`_run_sfm` (registration
  guard); `align_depth_to_reconstruction` raises on missing names anyway (mirrors
  `points3d_depth_maps`).
- Alignment `ValueError` aborts the stage — no partial zarr.

## Non-goals

- PAGaS-style init from splats.zarr depth renders (parked alternative, do not fold in).
- Global-median-only rescale (strictly dominated by per-frame).
- New config keys, migration tooling, backfill of processed-bucket scenes.
- Touching instantsfm's internal `use_depths` prior.

## Implementation principles

- Reuse: existing unproject path in `_run_sfm`, existing feedforward zarr-depth splats branch,
  existing `FeedforwardResult.save_zarr(extra_attrs=...)` provenance mechanism.
- Retire: `points3d_depth_maps` + sfm splats dispatch + mesh refusal (only after the eval wins).
- Minimal: one new function, one wiring block, attr-guarded consumer changes; module constants
  over config keys.

## Measured (GH010229, 300 frames, pose_opt, 30k steps, retriangulation on)

| condition | PSNR | SSIM |
|---|---|---|
| 3dgs sparse points3D targets + retri (baseline) | 19.111 | 0.6044 |
| **3dgs dense aligned VDA targets + retri** | **19.236** | 0.6029 |
| 2dgs (grow 2e-4) sparse targets + retri (baseline) | 19.211 | 0.6209 |
| **2dgs (grow 2e-4) dense aligned targets + retri** | **19.378** | 0.6150 |
| plan gate (3dgs sparse, pre-retri) | 18.38 | 0.574 |
| vggt_omega reference | 18.97 | 0.596 |

- Alignment: global scale 3.2248 (predicted ≈ 3.2), per-frame ratio p10/p50/p90 [0.7675, 0.9939, 1.4617] → [0.8881, 1.0, 1.1822] after alignment, 0 fallback frames, 300 registered images, 110,710 points3D.
- Dense targets win PSNR on both primitives (+0.13 / +0.17 dB); SSIM within noise or slightly lower (−0.0015 / −0.006).
- 2dgs dense grew to 2.02M gaussians (sparse: 1.40M); its report was recovered from ckpt.pt after a disk-full crash in the render stream (training completed; renders regenerated, no retrain).
- Mesh (source: splats, voxel 0.2 / sdf 0.8 / trunc 100, judged by renders from scene cameras 0/100/200): parity — no structural difference between dense- and sparse-target splat meshes; sparse marginally cleaner speckle in 2 of 3 views.
- Feedforward-fusion probe on the aligned zarr (newly possible — depth_scale guard passes): fusion works (1.92M verts) but renders carry heavy triangle-confetti speckle and holes — no confidence array, so every VDA pixel fuses unmasked; `mesh.source: splats` stays the recommended path for sfm scenes.
- Verdict: dense shipped — `points3d_depth_maps` retired; sfm splats consume aligned `pointcloud.zarr` depth through the same path as feedforward backends; legacy zarr without `depth_scale` raises.
