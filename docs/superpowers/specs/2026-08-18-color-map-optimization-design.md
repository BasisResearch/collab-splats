# Mesh Color Map Optimization — Design

**Date:** 2026-08-18
**Status:** Approved (user: rigid only, wire directly)
**Scope:** `collab_splats/mesh/utils.py` (new `optimize_color_map` + `pointcloud_to_mesh` hook), `collab_splats/wrapper/reconstructor.py` (mesh stage wiring), `configs/base.yaml`

## Problem

TSDF vertex colors are an average over every frame that sees a voxel. Cross-view
misalignment — pose error, model-res depth error, non-metric scale wobble — makes each frame
project slightly different image content onto the same voxel, and the average of misaligned
sharp samples is blur. Native-resolution fusion (2026-08-18 mesh-native-res-fusion) sharpened
the per-sample RGB but cannot fix the cross-view registration itself; the user confirmed
residual blur on the reference scene with `native_resolution: true` + `conf_percentile: 20`.

## Design

Zhou–Koltun rigid color map optimization ("Color Map Optimization for 3D Reconstruction with
Consumer Depth Cameras", SIGGRAPH 2014) as an opt-in post-fuse step, via Open3D's existing
implementation — `o3d.pipelines.color_map.run_rigid_optimizer` (verified present in the
venv's open3d 0.19.0). It refines each camera's 6-DoF pose to maximize photo-consistency of
the projected images, then reassigns vertex colors from the refined poses. Rigid only — the
non-rigid variant (per-image warp fields) is out of scope until rigid is measured.

### Config

One key:

```yaml
mesh:
  color_map_iterations: 0  # rigid color-map optimization iterations (0 = off)
```

Default `0` = off — shipping output byte-identical. The iteration count is the only knob and
doubles as the runtime throttle (optimizer is CPU-bound; upstream default is 300).

### `optimize_color_map` (mesh/utils.py)

```
optimize_color_map(mesh_path, depths, rgbs, c2w, intrinsics, iterations, depth_trunc) -> None
```

- Loads mesh.ply, runs the optimizer, overwrites mesh.ply in place (same in-place contract as
  `clean_repair_mesh`).
- Builds the RGBD list from the SAME `(depths, rgbs, c2w, intrinsics)` arrays
  `_feedforward_to_tsdf_inputs` returned for fusion — model-res and native paths both work,
  and mesh/image resolution consistency is guaranteed by construction (the mesh was fused
  from these exact arrays). uint8 RGB used as-is; float [0,1] RGB (model-res path) converted
  `(rgb * 255).astype(uint8)` at the boundary — mirror of the tsdf.py loop.
- Depth images: float32, `depth_scale=1.0` (depths are already in world units),
  `convert_rgb_to_intensity=False`. Depth 0 = no observation, consistent with fusion.
- Camera trajectory: `PinholeCameraTrajectory` of per-frame `PinholeCameraParameters`
  (intrinsic from each frame's K, extrinsic = w2c, i.e. `invert(c2w)`).
- `RigidOptimizerOption(maximum_iteration=iterations, maximum_allowable_depth=depth_trunc)` —
  the option's 2.5 default is a metric-depth assumption; our depth is non-metric, so the
  visibility cutoff must follow `depth_trunc` or frames would see "background" the fusion
  itself truncated.
- **Poses are report-only.** The optimizer refines a private trajectory used solely for color
  assignment; nothing is written back to COLMAP or the zarr (COLMAP stays the pose authority —
  same contract as geometric-verification).

### Placement and wiring

`pointcloud_to_mesh` gains `color_map_iterations: int = 0` and calls `optimize_color_map`
**after** `mesher.create(...)` returns — i.e. after fusion AND after clean_repair, so
iterations color the final geometry instead of speckle about to be deleted. It reads
`depth_trunc` from the mesher (`getattr(mesher, "depth_trunc")`; open3d_tsdf always has it).
Only the "open3d_tsdf" method path supports it today; a non-zero value with another method
raises `ValueError` (loud, like every other option misuse in this adapter).

`_run_tsdf_mesh` gains `color_map_iterations: int = 0`, forwarded from
`mesh_cfg["color_map_iterations"]` in `mesh()` — same plumbing as `conf_percentile`.

### Memory / runtime

RGBD list resident: ~14.5 MB/frame at 1080p (uint8 color + float32 depth) → ~4.4 GB at 300
frames, on top of the input arrays already held — fits the 46.6 GB cap alongside the measured
16.8 GB native-fusion peak. Runtime is CPU-bound and unknown until measured; the verification
step records wall-clock on the reference scene, and `iterations` caps it.

## Verification

Reference scene (`/workspace/outputs/2026_07_15-Goprosplat-GH010229`, vggt_omega), driven by
the existing scratchpad harness pattern:

1. **Regression:** `color_map_iterations: 0` → mesh.ply byte-identical to the current output.
2. **A/B:** native + p20 fuse, then `color_map_iterations ∈ {30, 100}` (bounded first probe;
   escalate toward upstream's 300 only if quality still improving and runtime tolerable) —
   wall-clock, peak RSS, before/after renders from the Task-5 render harness for user eyeball.

## Testing

- Unit: tiny synthetic scene (reuse `_tiny_ff_result` scale) — `optimize_color_map` with
  `iterations=5` runs without error and leaves a valid colored mesh on disk.
- `pointcloud_to_mesh(color_map_iterations=0)` output byte-identical to before (existing
  defaults-regression style).
- Non-TSDF method + `color_map_iterations>0` raises ValueError.
- base.yaml declares `color_map_iterations: 0` (extend the fidelity-keys test).

## Alternatives considered

- **Non-rigid optimizer** — stronger against depth error (our dominant term), but slower,
  more fragile, more knobs. Deferred until rigid is measured; config stays a single int so a
  future `color_map_mode` can slot in without breaking anything.
- **Best-view / winner-take-all vertex coloring** — cheaper, no pose refinement; kills
  averaging blur but keeps misprojection and adds view seams. Not built; revisit if optimizer
  runtime is prohibitive.
- **BA with dense tracks** — the root fix for pose error; parked, much bigger.

## Implementation principles

- Reuse: the adapter's existing arrays, the in-place mesh.ply contract, Open3D's optimizer —
  zero new dependencies, no persisted intermediates.
- Config surface: one int key, default off, byte-identical shipping output.
- Poses report-only; COLMAP remains the single pose authority.
