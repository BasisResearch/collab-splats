# Mesh TSDF adapter convergence — design

Date: 2026-08-11
Status: approved, awaiting plan

## Context

Mesh quality dropped sharply on the config-driven pipeline. Investigation found the
cause is not `clean_repair` and not `depth_trunc`: `Reconstructor.mesh()` fuses
**model-resolution depth against original-resolution COLMAP intrinsics**, and divides
already-`[0,1]` RGB by 255 a second time.

There are two mesh paths in the repo that do the same job and disagree. This spec
collapses them into one.

## Evidence

Measured on `data/outputs/` (30 frames, model 384×688, original 1080×1920, scale 2.81×)
via a standalone TSDF harness. `K_orig[0]` is `fx=1545, cx=540, cy=960` applied to a
384×688 depth image — the principal point sits outside the image.

| case | verts | tris | bbox | mean vertex colour |
|---|---|---|---|---|
| A correct (`K_model`, no `/255`, trunc 20) | 5,060,194 | 6,904,081 | 9.29×3.15×6.31 | 0.527 |
| C `K_orig` alone | 1,408,474 | 1,051,072 | — | 0.503 |
| D `/255` alone | 5,060,194 | 6,904,081 | — | 0.0001 (black) |
| E `depth_trunc=1.0` alone | 136,877 | 231,761 | 2.38×0.86×1.64 | — |
| B all three — what `Reconstructor` does today | 75,215 | 74,704 | 2.3×0.64×1.18 | 0.000 |

`K_orig` alone costs 72% of vertices and 85% of triangles. `/255` alone makes the mesh black.

## Root cause

`_run_tsdf_mesh` (`collab_splats/wrapper/reconstructor.py:317`) takes a
**`PointcloudResult`**, which wraps a pycolmap `Reconstruction`. `build_colmap`
(`collab_splats/pointcloud/feedforward/base.py:891`) runs
`_rescale_reconstruction_to_original_dimensions` before writing, so COLMAP's camera is
original-resolution from that point on. Depth and RGB come from `feedforward.zarr` at
model resolution.

The trap: both result types expose `.intrinsics`, meaning different things.

| type | `.intrinsics` | RGB |
|---|---|---|
| `FeedforwardResult` | model-res (384×688) | `images`, already `[0,1]` |
| `PointcloudResult` | original-res (1080×1920) | — |

`_run_tsdf_mesh` already held a `PointcloudResult` for poses and reached for
`.intrinsics` on it. Nothing types or asserts the resolution.

## Timeline

Both defective lines date to **6bc9c81 (2026-05-26)**, the commit that introduced
`Reconstructor`. They were invisible because `mesh.enabled` was `false` in committed
`base.yaml` and meshes were produced through the dashboard/notebook path instead.
`run_pipeline.py` (2026-07-20) and the remote driver (2026-07-30) made the config path
the one actually used.

## Ruled out

- **`clean_repair`** — `clean_repair: true` has never appeared in `configs/` history.
  Always `false`. Not the cause; the meshlib rework is not implicated.
- **`depth_trunc`** — real and separately costly (case E), but not the core defect.
  Stays at the current `2.0` by decision.
- **Loop closure** — pose-only; the K rescale never touches pose.

## Design

One adapter. `_feedforward_to_tsdf_inputs` (`collab_splats/mesh/utils.py:411`) becomes
the single source of TSDF inputs for both callers, reading `result.images` and
`result.depth` directly.

```python
# collab_splats/mesh/utils.py
def _feedforward_to_tsdf_inputs(result: FeedforwardResult):
    depths = result.depth                        # was: project world_points → camera Z
    rgbs   = result.images → (N, H, W, 3)        # was: PIL re-read + crop + resize + /255
    c2w    = invert_poses(result.extrinsics)
    return depths, rgbs, c2w, result.intrinsics  # model-res, always
```

```python
# collab_splats/wrapper/reconstructor.py — _run_tsdf_mesh
ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)
ff.extrinsics = result.extrinsics    # COLMAP stays the pose authority (BA/LC land there)
pointcloud_to_mesh(ff, output_dir, method="open3d_tsdf", **mesh_kwargs)
```

`_run_tsdf_mesh` keeps its signature and its call site in `Reconstructor.mesh()`. It stops
building TSDF inputs itself.

**Guards** (one line each, both cheap):

- In `_run_tsdf_mesh`: raise `ValueError` naming both artifacts when
  `result.extrinsics.shape[0] != ff.depth.shape[0]`. The stage-rerun path
  (`_resolve_result()` reads COLMAP off disk) can pair a reconstruction with a
  `feedforward.zarr` from a different run.
- In `Open3DTSDFFusion.create`: raise `ValueError` when `rgbs.max() > 1.5`. `create`
  already documents `rgbs` as `[0, 1]`; this turns a silently black mesh into a failure.

**Not forwarded:** `mesh_cfg["enabled"]` must be stripped before `**mesh_kwargs` reaches
the creator constructor.

## Deleted

Justified by measurement on `data/outputs/`:

- `world_points → camera Z` projection — `result.depth` is identical to it
  (max |Δ| 2e-6, relative 4e-8) and all three backends populate `depth`.
- `PILImage.open(result.image_paths)` re-read, `original_coords` crop branch, and
  `/255`. `result.images` is the tensor the model actually saw, so it is pixel-aligned
  with `depth`; the PIL re-resize differs only by resample filter (max |Δ| 0.133, mean
  0.0092, both in `[0,1]`) and can be sub-pixel misaligned against the depth grid. Crop
  is already baked into `result.images` by the preprocessor, so crop-mode stays correct
  by construction.
- The disk dependency itself. `images/` was removed in the FrameStore refactor
  (ab11b37); the PIL read only still resolves here because
  `tutorial_cache/omega_frames/*.jpg` are symlinks.
- `tests/mesh/test_adapter.py:87` (asserts the crop branch) — removed with the branch.

## Tests

Flat functions, mirroring the existing layout.

`tests/mesh/test_adapter.py`
- `_feedforward_to_tsdf_inputs` returns `result.intrinsics` unchanged and `result.depth`
  unchanged.
- RGB comes back `(N, H, W, 3)` in `[0, 1]`, not rescaled.

`tests/wrapper/test_reconstructor.py`
- The K reaching `Open3DTSDFFusion.create` equals the zarr K, **not** a 2×-rescaled
  COLMAP camera. This is the regression test for the whole bug.
- Poses reaching `create` come from `result.extrinsics`, not the zarr's.
- Frame-count mismatch between COLMAP and the zarr raises `ValueError` naming both paths.

`tests/mesh/` (or alongside `tsdf.py`)
- `Open3DTSDFFusion.create` rejects `rgbs` in `[0, 255]`.

## Verification

1. `Reconstructor.mesh(overwrite=True)` on `data/outputs/` at `depth_trunc=20`.
   Expect ≈5.06M verts, ≈6.90M tris, bbox ≈9.29×3.15×6.31, mean vertex colour ≈0.53 —
   case A above. Anything near case B means the fix did not land.
2. Repeat at the shipping `depth_trunc: 2.0` and record the numbers as the new baseline.
3. Dashboard mesh output must be unchanged apart from the RGB resample difference —
   it was already correct, so this is a no-regression check.
4. `python -m collab_splats.dashboard --smoke` must print `SMOKE PASS`.
5. `/opt/venv/reconstruction/bin/python -m pytest tests/ -p no:randomly`.

### Measured after the fix (2026-08-11)

`data/outputs/` has no `colmap/` dir, so verification ran the adapter directly and then
`_run_tsdf_mesh` against a `PointcloudResult` double carrying the 2.81×-rescaled COLMAP K.
Both paths agree exactly — the original-resolution K no longer reaches the fusion.

| config | verts | tris | bbox | mean vertex colour |
|---|---|---|---|---|
| `depth_trunc=20` (matches case A) | 5,060,214 | 6,904,130 | 9.29×3.15×6.31 | 0.5271 |
| `depth_trunc=2.0` (shipping baseline) | 711,079 | 915,039 | 3.24×1.48×2.35 | 0.4509 |

Case A reproduces to 0.0004% (5,060,214 vs the 5,060,194 measured by the standalone harness) —
the adapter uses `invert_poses` where the harness used `np.linalg.inv`, a float32 rounding
difference.

**`depth_trunc: 2.0` still costs 86% of vertices** (711k vs 5.06M). It is not the bug, but on
this scene it is the dominant remaining limit on mesh extent — the mesh stops at a 3.24 m box.
Depth here is non-metric, so `2.0` is not "2 metres". Revisiting it is separate work.

## Consequences

Every `mesh.ply` on disk — local and under `environments-processed/` — was fused with
original-resolution K and is wrong. All need `mesh(overwrite=True)`. Nothing is deleted
remotely; the re-run overwrites in place per the existing stage-rerun contract.

## Implementation principles

- Reuse `_feedforward_to_tsdf_inputs`, `pointcloud_to_mesh`, `invert_poses`. No new
  helpers.
- Delete what the change obsoletes — the world_points projection, the PIL/crop branch,
  and their test.
- No fallback branches for the deleted derivations. All three backends populate
  `images` and `depth`; a fallback would be dead code.
- Config knobs unchanged: `depth_trunc: 2.0`, `clean_repair: false`, `voxel_size`,
  `sdf_trunc` all stay as they are.

## Open risks

- `ff.extrinsics = result.extrinsics` assumes the zarr frame order matches
  `PointcloudResult.image_paths` order. Both derive from the same FrameStore order, but
  only the count is checked. If ordering ever diverges, the mesh degrades silently —
  worth a follow-up that checks names, not just `N`.
- The RGB resample difference (mean 0.0092) will make new meshes' vertex colours
  differ slightly from dashboard-produced ones. Cosmetic, expected, documented here.
