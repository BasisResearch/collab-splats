# Pointcloud reorg — design

- **Status:** SUPERSEDED 2026-09-27 by `2026-09-27-pointcloud-unify-design.md` — contradicts the contract in 19 places; reference only
- **Branch:** `clean/pointcloud-release`, worktree `.worktrees/pointcloud-release`, tip `4520452a` at design time
- **Inputs:** `/workspace/scratch/pc-release/REORG_PROPOSAL_single_result.md` (D1-D12),
  `HANDOFF_sfm_ff_reorg.md` (survey F1-F7), the user's 12 review points
- **Reverses** `2026-09-26-pointcloud-release-cleanup-design.md:23` "no shared SfM base class" (user-approved)

## Goal

One implementation per operation, reused across the package.

- one result type every creator returns; COLMAP is interop, not a second result type
- every geometric operation lives in `geometry/`; no function reimplements one the package already has
- shared joints across modules generalized once (see Joint owners), not patched per copy
- imports at module top; no dead parameters, no single-use wrappers

## Scope

- `collab_splats/pointcloud/` is redesigned
- callers are updated, not redesigned: `wrapper/`, `geometry/`, `semantics/`, `splats/pgsr.py`, `dashboard/`, `evals/`
- notebooks untouched; breaks listed in a hand-off note for tutorial-rework
- no merge, no push until every row here has landed

## Branch order

Three branches touch the same files (`geometry/transforms.py`, `feedforward/base.py`, `wrapper/reconstructor.py`).

- written against the TARGET APIs: `rescale_intrinsics` (clean/consistency phase 1),
  `utils/io.write_json` and depth-only `geometry/metrics` (clean/geometry-round3)
- implementation starts after round3 and consistency phase 1 land in `clean/final`;
  pointcloud-release rebases onto that tip first
- G′ and parity are re-recorded at the rebased tip before the first reorg commit

## Joint owners

Joints this branch generalizes. clean/consistency phases 2-3 drop the rows marked "supersedes".

| Joint | Copies today | One implementation | Supersedes |
|---|---|---|---|
| K model ↔ original pixels | `scale_intrinsics_to_original`, depth_align inline `*= sx`, ff `_rescale_reconstruction_to_original_dimensions`, `eval_splats` | `geometry/transforms.rescale_intrinsics(K, crop_box, model_hw, *, to_original)` | round3 `intrinsics_to_original` = the `to_original=True` case |
| crop box in original pixels | vggtx, omega, mapanything `_crop_box`, `full_frame_coords` | `geometry/transforms.center_crop_box(orig_wh, resized_wh, crop_wh)` | — |
| unproject | `splats/pgsr.unproject`, ff `_raw_to_world_points` / `unproject_and_filter_points`, vggt `unproject_depth_map_to_point_map` (depth_align, refine, `_verify_geometry`, LC) | `geometry/projection.unproject` | — |
| project | `splats/pgsr.project`, `pointcloud/utils.reproject_pixels`, inline in lifting + multiview | `geometry/projection.project` | — |
| cross-view depth test | multiview confidence loop, `lift_features` | `geometry/projection.depth_residual` | — |
| COLMAP IO | ff `write_colmap`, `sfm/common.write_sfm_model` + `rename_images_to_stems`, `PointcloudResult.from_colmap`, `sparse/0` hardcoded ×4 | `utils/colmap.py` | consistency phase 2 `geometry/colmap.py` |
| sfm skeleton | instantsfm, colmap, hloc | `sfm/base.BaseSfmCreator` | consistency phase 3 "SfM base" |
| JSON writes | — | round3 `utils/io.write_json` | — |

Out of scope, noted for other branches:

- BA `_reproject_per_camera` / `_reproject_shared` stay in BA (round3b #2)
- `mesh/texture._project_kernel` (GPU raster kernel)
- `geometry/loop_closure/wrapper.reconstruct` copy of the creator template → geometry round
- single kind-tagged backend registry (`SFM_CREATORS` vs `get_creator`) → consistency phase 4
- InstantSfM upstream fixes (3 remaining patches) → upstream PRs or a fork, vismatch-fork pattern

## 1. Result type

### `PointcloudResult` — `pointcloud/base.py`

`FeedforwardResult` renamed and moved; the old pycolmap-wrapping `PointcloudResult` is deleted.
Every creator, feedforward and sfm, returns it.

| Field | Shape | Meaning |
|---|---|---|
| `extrinsics` | (N, 4, 4) | w2c, OpenCV |
| `intrinsics` | (N, 3, 3) | K on the model grid (`model_hw`) |
| `image_names` | list[str] | frame stems, `frame_NNNNNN` |
| `crop_box` | (N, 6) | `[tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`, original pixels (was `original_coords`) |
| `model_hw` | (int, int) | model grid height, width |
| `camera_model` | str | pycolmap model name: `PINHOLE` (ff), the mapper's model (sfm) |
| `camera_params` | (N, P) | non-pinhole params only (SIMPLE_RADIAL: `k1`); focal + principal point live in `intrinsics` |
| `points`, `colors` | (P, 3) | final sparse set: conf mask → clean → `subsample_points` |
| `pixel_indices` | (P, 3) | `(frame, y, x)` source pixel of each sparse point |
| optional `depth`, `world_points`, `confidence`, `images` | (N, H, W[, 3]) | dense per-frame arrays |
| optional `mv_ratio`, `mv_inlier_count`, `mv_valid_count` | (N, H, W) | multiview confidence, ff only, as today |

Methods, nothing else:

- `save_zarr(path)` / `load_zarr(path)` — the pipeline's format
  - clean-break schema: the new field names above; old names are not read
  - an old-schema zarr raises "stale pointcloud.zarr; delete and re-run" (the `load_video_quality` pattern)
- `from_colmap(root)` — sparse fields only (poses, K at original res, points, colors,
  `crop_box` full-frame, `model_hw` = camera size); no dense fields
- `to_colmap() -> pycolmap.Reconstruction`
  - always original resolution: `rescale_intrinsics(..., to_original=True)`, no flag
  - cameras built from `camera_model` + `intrinsics` + `camera_params`; the model is recorded, never re-derived
  - replaces ff `write_colmap`, `build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`
- `reproject()` — via `geometry/projection.project`

`BasePointcloudCreator` (same file) is rewritten: `create(images_dir, out_dir) -> PointcloudResult`,
implemented by `BaseFeedforwardCreator` and `BaseSfmCreator`.

### `utils/colmap.py`

Minimal; every function has a caller outside this file.

| Function | Job |
|---|---|
| `SPARSE_SUBDIR`, `sparse_dir(root)` | the one place `sparse/0` is spelled |
| `is_complete(root)` | all three `.bin` files exist; replaces the `cameras.bin`-only done-check (`reconstructor.py:1638`) |
| `read_colmap(root)` | `pycolmap.Reconstruction(sparse_dir(root))` with a clear missing-model error |
| `write_colmap(recon, root)` | image names → stems, atomic write (temp dir → rename) |

- pose / K one-liners stay inline at the call site (consistency phase-2 rule)

### Sparse set: clean, then sample

Order changes from sample → clean to:

```
ff:  dense world_points ─ conf mask ─ clean (SOR) ─ subsample_points ─> points
sfm: mapper points3D ─────────────── clean (SOR) ─ (no cap) ─────────> points
```

- stored once in the zarr; COLMAP export, PLY, splat seed and lifting all read the same set
- `subsample_points(mask, max_points, seed=0) -> mask` in `pointcloud/utils.py`
  - the `_limit_trues` body: private seeded rng over the True positions
  - replaces `_limit_trues` and the array-based `subsample_points`
- SOR cost on the full conf-masked set is measured in P2; fallback if too slow: clean a 4× `max_points` sample, then cap
- PLY written once from the cleaned set (today 3×: `ff/base:1271`, `reconstructor.py:1123`, refine)

### Lifting

- `lift_features` → `semantics/lifting.py` (every caller lifts semantic features)
- reads `PointcloudResult` arrays + depth; visibility via `geometry/projection.depth_residual`
- the pipeline always lifts from the zarr, which has depth for both methods; no COLMAP-track lifting

## 2. sfm flow

### Skeleton — `sfm/base.py`

```python
class BaseSfmCreator(BasePointcloudCreator):
    def create(self, images_dir, out_dir) -> PointcloudResult:
        depths = estimate_depth(images_dir, names, out_dir)       # depth.py; instantsfm reads out_dir/depth/ as priors
        recon = self.reconstruct(images_dir, out_dir / "colmap", names)   # backend-specific
        write_colmap(recon, out_dir / "colmap")                   # mapper model as-is, tracks kept
        return self.densify(recon, depths, images)
```

- backends implement `reconstruct` only: images in, largest `pycolmap.Reconstruction` out
- `densify(recon, depths, images, min_obs=20)`: per-frame median of COLMAP track depth / predicted depth,
  scaled depth → `geometry/projection.unproject`; K via `rescale_intrinsics(to_original=False)`, full-frame box
  - subsets to registered frames (today `reconstructor._registered_rows`)
  - one-caller rule: a method here, not a module
  - known limitation: ignores the SIMPLE_RADIAL `k1` term
- `sift_database(image_dir, db_path, names, pairing, ...)`: pycolmap extract + match, used by colmap and instantsfm
  - reuse iff the DB's image names equal `names` AND a `collab_params` sqlite row equals the params; the sidecar JSON goes
- `Reconstructor._run_sfm` shrinks to `creator.create()` + `save_zarr`
- `evals/scripts/eval.py` `_run_instantsfm` calls `creator.create()` instead of re-running the stages
- `_SFM_BLOCK_KEYS` / `_validate_sfm_block` bounds checks → creators' `__post_init__`

### `pointcloud/depth.py`

Plain functions; no base class until a second depth model exists.

```python
def load_vda_model(device: str) -> torch.nn.Module
def estimate_depth(images_dir: Path, names: list[str], out_dir: Path) -> np.ndarray
```

- cached at `out_dir/depth/*.npy`; decodes frames only on a cache miss; a miss wipes and regenerates
- `vda_depth_complete` deleted (its two jobs are inside `estimate_depth`)
- contract in the docstring: up-to-scale depth, not disparity (`densify` fits scale only, no shift)
- `depth/` not `depth_vda/`: upstream InstantSfM accepts both (`data_reader.py:40`)

### Backends

| File | Keeps | Changes |
|---|---|---|
| `sfm/colmap.py` | `reconstruct`, vocab tree fetch (sha-checked; pycolmap does not auto-fetch) | largest-model pick inline |
| `sfm/hloc.py` | `reconstruct`, private `_sequential_pairs` (hloc needs a pairs file; pycolmap's sequential matcher is DB-bound) | hloc via uv git source at `c13273b`; `setup/hloc.sh`, editable path source, `HLOC_PIN` deleted |
| `sfm/instantsfm.py` | `reconstruct`, 3 patches (track ids, pypose target, bae PCG) | writer replaced by in-memory conversion → `pycolmap.Reconstruction`; `_patch_instantsfm_colmap_write` + scratch read-back deleted |
| `sfm/__init__.py` | `SFM_CREATORS` | importlib by name on request, so every backend imports its deps at module top |

- InstantSfM upstream (pin `d3e599e`, `setup.sh:125`): no fix for the 3 patched bugs found upstream; hardcodes `PCG(tol=1e-5)`, no CuDSS
- hloc: not on PyPI; tag v1.4 predates pycolmap 4 `cam_from_world()`, hence the commit pin
- check at implementation: uv fetches hloc's git submodules AND superpoint/superglue still import
  - pyproject.toml:147-148: editable was chosen because they `sys.path`-append `../../third_party`
  - if a git install breaks that, keep the editable `setup/hloc.sh` clone and delete only `HLOC_PIN`

### Deleted

`sfm/common.py`, `sfm/sift_db.py`, `pointcloud/depth_align.py`, `pointcloud/vda.py`, every `provenance()`,
the `depth_scale` attr gate in mesh (`reconstructor.py:1354`) and splats (`:1530`) → field-presence check,
refine's `get_creator(backend)(**block).camera_model` (`:1226`) → stored `camera_model`.

## 3. Feedforward

`feedforward/base.py` keeps the creator template and multiview confidence (ff-only, private).

| Job today | Goes to |
|---|---|
| result type + zarr IO | `pointcloud/base.PointcloudResult` |
| COLMAP export | `PointcloudResult.to_colmap` + `utils/colmap.py` |
| multiview confidence | stays; inner depth test → `geometry/projection.depth_residual`; `collect=` path unchanged |
| `_limit_trues`, `_mask_to_points` | `pointcloud/utils.subsample_points` |
| `_raw_to_world_points`, `_verify_geometry` unprojection | `geometry/projection.unproject` |
| 4 crop-box functions | `geometry/transforms.center_crop_box`; each model file keeps only its ported sizing lines + citation |
| template method | stays |

Per-creator boilerplate collapsed into the base (survey F4):

- `extract_intermediate_features` near-twins (vggtx / omega / mapanything) → one base method, per-model layer hook
- synthetic `image_paths` ×4, `loader_names` + `frames_as_pil_source` ×3, mapanything c2w→w2c ×3
- `cross_frame_attention_ratio`, `mean_top_quarter` → `geometry/loop_closure` (only consumer)
- `utils.reproject_pixels` deleted (tests only)

### `geometry/projection.py` (new, torch)

Works in the input dtype; numpy callers pass `torch.from_numpy`.

```python
def unproject(depth, w2c, K) -> Tensor                        # from pgsr
def project(points, w2c, K, *, min_depth) -> (pixels, points_cam)   # from pgsr
def depth_residual(points, depth_j, K_j, w2c_j) -> (residual, in_frame)
```

- `depth_residual` docstring states what it computes:
  - project world points into camera j; sample j's depth map at the projected pixel
  - residual = (sampled − projected z) / projected z, signed relative depth error
  - `in_frame` = pixel inside the image and z > 0; callers threshold the residual
- multiview confidence counts `|residual| < abs/z + rel` and bins it into `collect`
- lifting thresholds on its tolerance
- `splats/pgsr.py` imports `project` / `unproject`; keeps its `(1, 4, 4)` squeeze at the call site
- `geometry/transforms.py` stays numpy (poses, K, crop box, umeyama)

## 4. Gates

Every commit is a refactor (outputs equal) or a parity change (own commit, measured). Refactors land first;
parity changes last, each measured against the refactored baseline.

### Refactors — outputs equal

| Item | Gate |
|---|---|
| `PointcloudResult`, `utils/colmap.py`, sfm skeleton, `depth.py` | G′ + `pointcloud.zarr` arrays equal on a fixture scene, ff and sfm |
| `rescale_intrinsics` for the full-frame copies | K equal at atol 1e-9 |
| `center_crop_box` | equal to the 4 old functions on a size grid (VGGT-X: see P5) |
| `project` / `unproject` replacing pgsr + copies | pgsr tests unchanged; float64 in → equal out |
| InstantSfM in-memory conversion (test first) | poses, points, track observations equal to the patched writer's; pycolmap loads it |
| hloc uv git source | lock resolves `c13273b`; model equal on a fixture |
| `lift_features` move | AST-equal body |

### Parity changes — one commit each

| # | Change | Measure |
|---|---|---|
| P1 | unprojection in the input dtype, not forced float64 | max point drift, ff zarr |
| P2 | conf mask → clean → `subsample_points` | point count, SOR runtime + peak RSS (300-frame scene, 46.6 GB cap), splat PSNR |
| P3 | `subsample_points` in mask space (LC sample changes) | LC parity `rotation_only`, chess seq-01 ATE |
| P4 | one tolerance `abs + rel·z` for multiview and lifting | lifted-feature cosine vs old, mv inlier counts |
| P5 | VGGT-X box y uses per-axis scale (`vggtx.py:51` uses `518/orig_w`) | box Δpx, TSDF mesh on one cropped scene; dropped if upstream resizes y by `518/orig_w` |
| P6 | ff `to_colmap` writes `PINHOLE` from the stored K (drops mean-vs-max) | K per camera on the exported model |
| P7 | zarr schema clean break | old zarr raises the stale error |
| P8 | SIFT DB reuse keyed on DB contents | rerun reuses; first run after upgrade rebuilds once |

Scenes: chess seq-01 (poses, LC, has GT), C0043 (pointcloud, mesh, splats).

### Process

- G′ + parity (`$SP/parity.py --check`) after every commit, from the worktree with a printed `collab_splats.__file__` line
- net-negative code per refactor commit
- `git commit --only <paths>`; never amend, rebase or reset while agents hold dirty state
- tests first for every changed behavior, on fixtures that can see it (non-identity poses, cropped boxes)
- disclosures appended to `/workspace/scratch/pc-release/final_report_notes.md`

## Carried reorg inputs (F-Bfix1 items 5-7)

- item 5: `_limit_trues` never-passed `seed` → dissolved by `subsample_points`
- item 6: `tests/pointcloud/test_mv_conf.py:53` identity extrinsics → non-identity when `depth_residual` lands
- item 7: `tests/pointcloud/sfm/test_common.py:64-66` → deleted with `common.py`
- omega `_CropProbe` proposal → superseded by `center_crop_box` + size-grid equality

## Resulting layout

```
pointcloud/
  base.py          PointcloudResult, BasePointcloudCreator
  depth.py         load_vda_model, estimate_depth
  utils.py         clean_pointcloud, confidence_mask, subsample_points
  feedforward/     base.py (template + multiview), vggtx, vggt_omega, mapanything, loger
  sfm/             base.py (skeleton, densify, sift_database), colmap, hloc, instantsfm, __init__
geometry/
  transforms.py    + rescale_intrinsics, center_crop_box
  projection.py    new: unproject, project, depth_residual
semantics/
  lifting.py       lift_features
utils/
  colmap.py        new: sparse_dir, is_complete, read_colmap, write_colmap
```
