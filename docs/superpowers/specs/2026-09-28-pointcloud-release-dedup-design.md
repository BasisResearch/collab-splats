# Pointcloud release dedup — design

Branch `clean/pointcloud-release` (tip `1ec9a5a5`). Follows the unify work
([2026-09-27-pointcloud-unify-design.md](2026-09-27-pointcloud-unify-design.md)) and the
review handoff `/workspace/scratch/pc-release/HANDOFF_unify_review.md`.

Goal: delete the duplicates and overengineering left in `pointcloud/` after unify, share the
steps feedforward and sfm have in common, and bring comments/docstrings to geometry style.
Every change is net-negative in lines. No new wrapper layers, no K-undo helper, no layout
constants.

## Rules

- parity gate after every behavior commit: `WT=<wt> bash /workspace/scratch/pc-release/unify_gate.sh <tag>`
  - rc is 0 on failure — read the output lines
  - baseline chain `parity_baseline_unify` only; `parity_baseline/` never touched
- a baseline move is reported to the user before it is accepted
- the style pass (Section 5.7) is last and behavior-free: AST-equal after stripping docstrings
- no merge, no push, no history rewrite; `git commit --only`

## 1. Multiview depth confidence

One function replaces the dataclass, mask helper, result fields and frustum gate.

- new `geometry/projection.py::multiview_depth_confidence(depth, intrinsics, extrinsics, rel_thresh)`
  - next to `depth_residual`, built on it
  - returns per-pixel `(agree, seen)` counts, both `(N, H, W)` int
  - `seen`: other views where the pixel projects in front and in bounds
  - `agree`: of those, views whose depth matches within `rel_thresh`
  - keeps the principal-point guard; port of `mapanything/utils/multiview_confidence.py:125`
- filter in feedforward `_postprocess`: `keep = agree >= np.minimum(self.min_views, seen)`
  - `seen == 0` keeps the pixel (nothing to disagree with)
  - config: `min_views: int = 0` (0 = off), `mv_rel_thresh: float = 0.01`
  - removed config: `use_multiview_confidence`, `mv_conf_abs_thresh`, `mv_conf_rel_thresh`
- report stage keeps its pair stats in `geometry/metrics.py`, adds per-frame `multiview_agreement`
- deleted from `feedforward/base.py`:
  - `MultiviewConfidence`, `multiview_mask`, `_mv_result_fields`, `compute_multiview_depth_confidence`
  - `_frustum_world_aabbs`, `_aabbs_overlap` — measured no pruning:
    - omega 196 frames: 97.9 s gated vs 94.4 s ungated, 37,093 pairs both
    - colmap 135 frames: 57.3 s vs 52.2 s, 18,040 pairs both
- deleted from `pointcloud/base.py`: `PointcloudResult.mv_*` fields and their zarr arrays

## 2. Creator structure

Shared template in `BasePointcloudCreator`; subclasses only reconstruct.

- `create` → `create_pointcloud(images_dir, out_dir, model_dir)` on the base:
  ```python
  paths = frames.frame_paths(images_dir)
  result = self._reconstruct(paths, out_dir)
  if self.postprocess:
      result = clean_pointcloud(result, remove_outliers=self.clean, max_points=self.max_points)
  write_colmap_reconstruction(result.to_colmap(), model_dir)
  ```
  - sfm overrides the export when postprocess is off, so its model keeps tracks
  - `run()` folds in via `model_dir=None` if it differs only by the write
- `_preprocess(paths)` takes real image paths; the loaders read them directly
  - deleted: `frames_as_pil_source`, `_source_frame_idxs`, `_decode_dir_to_frames`,
    `_STORE_FRAME_NAME`, `_decode_source`
  - `center_crop_coords` stays, at the bottom of `feedforward/base.py`
- MapAnything `_forward` stacks its per-view list into the base raw dict → one base `_postprocess`
- file order in `feedforward/base.py` and `sfm/base.py`: classes top, helpers bottom

### Postprocess naming

- SOR mask renamed `clean_pointcloud` → `outlier_mask(points) -> bool mask`, beside `confidence_mask`
- freed name: `pointcloud/utils.py::clean_pointcloud(result, *, remove_outliers, max_points) -> PointcloudResult`
  ```python
  keep = outlier_mask(result.points) if remove_outliers else np.ones(len(result.points), bool)
  keep = subsample_points(keep, max_points)
  return result.select_points(keep)
  ```
- callers: `create_pointcloud`, `refine_poses`; replaces 3 inline SOR + cap copies

### Opt-in sfm postprocess

- base field `postprocess: bool`; feedforward default `True`, sfm default `False`
- deleted: sfm SOR on `recon.points3D` (`sfm/base.py:121-131`)
- behavior change, sfm only:
  - old: SOR before `align_depth`, the VDA scale fit saw the cleaned sparse points
  - new: default off → fit sees unfiltered sparse points; on → clean runs on the dense result
  - `align_depth` is a per-frame median ratio, robust to outliers; sfm baseline re-recorded
- export: off → `write_colmap_reconstruction(recon)` keeps tracks; on → `to_colmap()`, no tracks
  - nothing in `collab_splats` reads tracks from the written model (verified)

## 3. Unify duplicates

- `semantics/lifting.py:167-174`: `project` then `depth_residual` project the same points twice
  - `depth_residual` also returns the projected `uv`; lifting drops its `project` call
- `refine_poses` (`wrapper/reconstructor.py:1149-1215`) becomes:
  ```python
  refined = dataclasses.replace(result, extrinsics=...).reproject()
  refined = clean_pointcloud(refined, remove_outliers=..., max_points=...)
  write_colmap_reconstruction(refined.to_colmap(), model_dir)
  ```
- `geometry/metrics.py:236-243` repeats the `PointcloudResult.__post_init__` K lift
  - report stage passes the already-lifted `result.intrinsics`; lift block deleted
  - `original_coords` argument dropped if nothing else in the report reads it
- `pointcloud/depth.py` pixel mapping: `int` truncation vs `floor` → one rule, `np.floor` then clip
- `pointcloud/depth.py`: `_depth_correspondences` + `_fit_depth_scales` folded into `align_depth`
  - per-frame median ratio with global fallback, ~20 inline lines
- parity: the lifting, refine and metrics items must reproduce the unify baseline exactly;
  the pixel-mapping change is checked by the gate

## 4. VDA loader

Keep Video-Depth-Anything; clean the edges only. Results unchanged.

- `sys.path` hack → `uv` git dependency at the pinned SHA `4f5ae23`, if upstream installs
  - otherwise the pinned clone stays and only the comments are trimmed
- checkpoint via `hf_hub_download`
- `Metric-Video-Depth-Anything-Large` is cc-by-nc-4.0: stated in the `vda.py` docstring and `docs/`
  - Metric-Small (Apache-2.0) and DA-V2 metric (per-frame) considered and not taken

## 5. Small items and style

1. inlines
   - `mean_top_quarter` → `r[r >= np.percentile(r, 75)].mean()` at `feedforward/base.py` and
     `evals/scripts/eval_similarity_calibration.py:228`
   - `full_frame_coords`, `_mask_to_points`, `_raw_to_world_points` (dead `subsample=8` path)
2. `unproject_and_filter_points` → `confidence_mask` + `unproject`
   - parity move: `confidence_mask` is strict `>` with an all-True fallback; ties may differ
   - gate decides; a move is reported before accepting
3. LC `_verify_geometry` deleted; LC calls `unproject` directly
4. real file names
   - `image_names` are real basenames from `frame_paths(images_dir)`
   - `reconstructor.py:799` hard-coded `frame_{:06d}.png` ids → `p.name`
   - 7-Scenes names become the real `.color.png`; check `clean/evals-release` GT matching for
     string keys before landing
5. `write_colmap_model` on the feedforward creator deleted; base calls `write_colmap_reconstruction`
6. report: per-frame `multiview_agreement` = share of seen pixels with `agree >= 1`
7. style pass over `pointcloud/`, geometry style
   - one-line summaries, fragment bullets, no provenance essays
   - one upstream cite per file at the top
   - classes top, helpers bottom; blank lines around blocks
   - `tests/test_docstring_contract.py` passes

## Order

1. Section 1 (multiview) — gate
2. Section 2 (creator structure, postprocess, sfm opt-in) — gate; sfm re-baseline reported
3. Section 3 (unify dups, depth.py) — gate
4. Section 4 (VDA loader) — gate
5. Section 5.1–5.6 — gate
6. Section 5.7 style pass — AST-equality proof

## Out of scope

- LC `extract_extrinsics` / `dedup_overlap`, LC vggt unproject (`loop_closure/wrapper.py:346`),
  reconstructor cv2 reads → consistency phase 3
- dashboard `_rotation_align` (dup of `transforms.rotation_align_vectors`) → own owner
- instantsfm smoke re-run, CHANGELOG + CLAUDE.md row, final whole-branch review → handoff items

## Testing

- `tests/geometry`: `multiview_depth_confidence` on a two-view synthetic scene
  (agree, occluded, out-of-bounds, `seen == 0`)
- `tests/pointcloud`: `clean_pointcloud(result)` cap + outlier paths; `outlier_mask` rename
- `tests/pointcloud/sfm`: postprocess off keeps tracks in the written model
- deleted helpers' tests deleted with them
- full `tests/pointcloud tests/geometry tests/semantics tests/wrapper` gate before each commit
