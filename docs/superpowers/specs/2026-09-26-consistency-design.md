# Package consistency — design

- **Status:** phase 1 designed; phases 2-7 scoped, each gets its own spec section before its plan
- **Branch:** `clean/consistency`, worktree `.worktrees/consistency`, forked from the `clean/final` tip at worktree creation
- **Audit report:** https://claude.ai/artifact/2tLJ8KpwcJ4c21yjfDiBw8 (9 audits, 2026-09-26)

## Goal

One implementation per convention across the package.

- the audit found ~2,300 LOC of repeated code in `collab_splats/` + `evals/`, ~600 in tests
- the duplication that matters is copies that disagree on a convention: w2c vs c2w,
  crop-aware vs crop-unaware intrinsics, frame image formats, NaN handling
- every verified bug in the audit is one of those disagreements

## Branch and integration

One long-lived worktree; every phase lands on it in order.

- forked from the `clean/final` tip at worktree creation (recorded in the plan), never merges a sibling in
- rebases onto `clean/final` before integration; merge into `clean/final` is the user's call
- known overlap at fork time:
  - `clean/r4-lc` rewrites `geometry/metrics.py`, `geometry/verification.py`, loop closure, `evals/scripts/eval.py`
  - `clean/pointcloud-release` (sfm-backends) rewrites `wrapper/reconstructor.py` and the pointcloud registry
  - `clean/tutorials` rewrites the localization notebook
- phase 1 keeps each fix to the smallest hunk at the defect so rebase conflicts stay local
- test runs: `cd .worktrees/consistency && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest ...`,
  gated on a printed `collab_splats.__file__` line; `third_party/*` symlinked in so guarded tests do not skip
- no full-suite runs while the GPU is shared; per-package gates, compared against a control on the fork point

## Phases

Each phase lands as its own commit group on the branch.

| # | Phase | Scope |
|---|---|---|
| 1 | Convention bugs | this spec, below |
| 2 | Shared low-level helpers | lazy `utils/__init__`; one `utils/io.py` (below); `geometry/colmap.py`, `camera_centers`, `to_numpy`; adopt `get_device` / `pytorch_gc` |
| 3 | Backend hoisting | feedforward `_postprocess` / `_reproject` / QKV hook, LC pose extraction, SfM base |
| 4 | Pipeline and registry | single kind-tagged backend registry, `SceneLayout`, config helpers, `localization/store.py`, dashboard calls Reconstructor stages |
| 5 | API renames | `conf_percentile` / `min_conf`, `poses_w2c` / `poses_c2w`, `out_dir` / `images_dir` |
| 6 | Test consolidation | shared builders with non-identity defaults, conftest cleanup, autouse seeding |
| 7 | Dependencies | drop list, open3d to core, declare rich / scikit-image |

### Phase 2 IO module

All file IO helpers live in one `collab_splats/utils/io.py`, split by `########` sections.

- **images:** `read_image` (one decoder, RGB HWC uint8), `iter_images`, `to_uint8_hwc`
  (explicit range, no guessing); replaces ~9 ad-hoc cv2 / PIL readers and 8 uint8 converters
- **json:** `to_jsonable` (NaN / ±inf → null, numpy scalars, ndarray, tuple, Path) and
  `write_json(path, obj, *, indent=2, atomic=True)`; replaces `clean_for_json` and the
  9 JSON sites
- **zarr:** `LZ4` codec, `UNREADABLE_STORE`, `open_valid`, `stamp_valid` (completion marker
  written after the arrays); replaces the 4 codec copies and 2 open-else-rebuild paths
- import-light: numpy, json, zarr, PIL / cv2 only, never torch, so `remote` and the dashboard
  fast path can import it
- stays out:
  - the `images/` + `frames.json` keyframe store stays in `preproc/frames.py` and calls `io.read_image`
  - PIL-exact feedforward preprocessing stays where it matches an upstream loader
  - `to_numpy` stays in `torch_utils`

Dead code the audit found goes into `2026-09-09-clean-final-dead-code-design.md`, not here:
`ConfigLoader`, `infer_batch_size`, `OPENGL_TO_OPENCV`, `FeedforwardResult.save/load`, `load_hf_weights`.

## Phase 1 — convention bugs

Every fix is test-first on a fixture that can see the convention.

- non-identity rotations, off-centre principal point, non-square and cropped images
- a test that passes on the unfixed code is rejected, not kept
- suspected bugs stay in scope only if their failing test reproduces them

### Out of scope — fixed on a sibling

| Bug | Fixed on | Check after rebase |
|---|---|---|
| LC gates raw confidence against `conf_threshold=25` | `clean/r4-lc` (percentile via `prev_submap.conf_threshold`) | `graph.py` compares against the submap percentile |
| localization notebook inverts extrinsics before `create_camera_frustum_pyvista` | `clean/tutorials` (`06_localization`) | no `np.linalg.inv` feeding the frustum call |
| `instantsfm` absent from `pointcloud._REGISTRY` | not a bug on `clean/pointcloud-release`: sfm backends leave the registry | folded into phase 4 registry work |

### 1. RPE frame convention

`evals/trajectory_metrics.rpe` applies the c2w relative-pose formula to w2c input, unaligned.

- caller: `evals/scripts/eval.py` passes `pred` (w2c) and `dataset.gt_poses` (w2c)
- effect: wrong-frame errors, reported in model scale as if metres
- contract: `rpe(poses_w2c, gt_w2c, delta=1)`
  - invert both to c2w
  - Sim3-align predicted centres to gt centres (`umeyama_sim3`), apply to the c2w poses
  - relative error as today on the aligned c2w poses
- test: non-identity gt trajectory; pred = gt under a known Sim3 (scale 2.5, rotated) → RPE ≈ 0;
  pred with one perturbed frame → nonzero at that pair only

### 2. `cam.params` unpacking

`evals/scripts/eval_verification.py:62` (`_recon_to_arrays`) unpacks `fx, fy, cx, cy = cam.params`.

- effect: ValueError on SIMPLE_PINHOLE / SIMPLE_RADIAL, silent misassignment on longer param vectors
- contract: K from `cam.calibration_matrix()`
- test: SIMPLE_PINHOLE and PINHOLE cameras produce the same K for equal focal lengths

### 3. JSON NaN leak

`clean_for_json` only replaces builtin `float` NaN in dicts and lists.

- reproduced: `{'a': np.float32('nan'), 'b': (nan,), 'c': np.float64('nan')}` → `{"a": NaN, "b": [NaN], "c": null}`
- `geometry/metrics.py:~698` already calls `clean_for_json` with `default=lambda o: o.item()`;
  a float32 NaN passes `clean_for_json` untouched, then `.item()` turns it into a bare `NaN`
- contract: `clean_for_json` handles `np.floating`, `np.integer`, tuples, ndarrays;
  NaN and ±inf → `None`; the `default=` fallback in `metrics.py` is then dead and removed
- fixed in place only: no `utils/jsonio.py`, no lazy `utils/__init__`, no migration of the
  other JSON sites — those are phase 2, which replaces `clean_for_json` with `to_jsonable`
- test: `json.dumps(clean_for_json(payload), allow_nan=False)` succeeds for float32 / float64 NaN,
  inf, a NaN inside a tuple and inside an ndarray

### 4. Crop-aware intrinsics

`feedforward/base.py:879` `_rescale_reconstruction_to_original_dimensions` is wrong for any cropped box.

- crop box: `original_coords = [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`, all in original pixels
  (as `vggtx.py:51` and `vggt_omega.py:54` build it)
- base.py defects on a cropped box:
  - scale is `orig / model`; should be `crop / model`
  - cx, cy get no `+ tl`
  - point2D shift is `(xy - tl) * scale`; tl is original-pixel, so it must be added after scaling
  - SIMPLE_PINHOLE takes `max(scale_x, scale_y)`; pycolmap params rescaled as `params[-2:] *= scale`,
    assuming cx, cy are last
- callers: `base.py:1277` (pointcloud stage) and `reconstructor.py:1195` (refine rewrite)
- crop-aware version exists: `geometry/metrics.py:239` `_scale_intrinsics_to_original` (`K / s + tl`,
  `s = model / crop`)
- copy of the same rule: `evals/scripts/eval_splats.py:73` `_native_images_and_intrinsics`
  (its docstring: "scale = orig / model ... no crop offset")
- not a separate site: the mesh arm (`reconstructor.py:~664`) uses `PointcloudResult.intrinsics`,
  which base.py produced; fixed by the base.py fix
- existing `tests/pointcloud/test_feedforward_intrinsics.py` covers full-frame boxes only and must
  keep passing unchanged
- contract: promote to `geometry/transforms.rescale_intrinsics(K, crop_box, model_hw, *, to_original: bool)`;
  both directions; base.py, eval_splats.py and `metrics.py` call it; pycolmap cameras rebuilt from
  the new K via `calibration_matrix()`-compatible params, and point2D mapped as `xy / s + tl`
- test: portrait image, crop with nonzero tl, off-centre principal point; a 3D point projected
  with model-res K then mapped through the crop equals its projection with original-res K;
  round-trip `to_original` then back is identity

### 5. Suspected crop and convention bugs

Each stays only if its failing test reproduces it; otherwise it is dropped with a note in the plan.

- **localizer ref-pixel lookup** (`localization/localizer.py:931-942`): rescales reference pixels
  onto the `world_points` grid by size ratio only, no crop offset; test: cropped reference, known 3D point,
  lookup returns that point
- **MapAnything crop box**: `mapanything.py:226` writes `[0, 0, model_w, model_h, orig_w, orig_h]`,
  cr in model pixels; `loger.py:349` and `depth_align.py:303` write `[0, 0, orig_w, orig_h, ...]`;
  `metrics.py:344-346` derives `s = model / crop` from `[:4]`, so MapAnything gets `s = 1`;
  test: MapAnything box through `metrics.py` and `rescale_intrinsics` gives the same original-res K
  as the equivalent full-frame box
- dropped from the audit: viewer FOV (`viewer.py:102`) reads `h` before the stride and pairs it with
  the unstrided K, so it is already correct

### 6. Localization DB provenance (suspected)

`dashboard/pipeline.py:499` `_stamp_db_provenance` reads `run_config.yaml` through the dashboard
`RunConfig.from_yaml`, which drops every key it does not define.

- `wrapper/batch.py:116` writes `run_config.yaml` as the Reconstructor config; the dashboard
  localizes those processed scenes, so `backbone`, `frame_indices` and `video_ref` are all stamped
  from `RunConfig` defaults
- contract: stamp all three from whichever schema the file holds; the file-name collision
  itself is phase 4
- test: Reconstructor-written `run_config.yaml` naming a non-default backbone → stamped attrs
  name that backbone
- dropped from the audit: the `.jpg` vs `.png` frame-id split no longer exists on `clean/final`
  (`reconstructor.py:777`, `dashboard/pipeline.py:694` both `.jpg`)

### 7. PNG for every frame image

The keyframe store writes `.png`; frame images elsewhere still use `.jpg`.

- `dashboard/pipeline.py:694`: localized query frames saved as `.jpg` via PIL at default
  quality 75, then fed back into the localization DB — lossy; switch to `.png`
- `wrapper/reconstructor.py:777`: localization ids `frame_NNNNNN.jpg` are labels, not files;
  switch to `.png`
  - blocker: `localizer.py:831-834` (`from_feedforward`) staleness check compares whole strings, so every DB on
    disk would warn stale
  - fix: compare `Path(x).stem` lists; old `.jpg` DBs and new `.png` ids both match, no migration
  - delete the comment at `reconstructor.py:770-775` justifying `.jpg`
- `pointcloud/sfm/instantsfm.py:292`: fallback name `f"{idx}.jpg"` → `.png`
- kept as JPEG: `mesh/io.py:317` texture atlas (8192², display-only output; PNG is 10-20x
  larger); the call site gets a comment saying so
- test: a DB whose attrs hold `.jpg` ids, loaded with `.png` ids → no stale warning; a DB with
  different stems → warning; dashboard-appended query frame on disk is PNG and decodes
  byte-identical to the array written

## Error handling

- `rescale_intrinsics` raises `ValueError` on a crop box that is not 6 values or has zero size
- `rpe` keeps its `ValueError` for `delta >= N`

## Testing

- tests mirror the package: `tests/evals/`, `tests/geometry/`, `tests/pointcloud/`, `tests/localization/`
- control: the same per-package gates on the fork point, recorded before the first fix
- a fix is done when its new test fails on the fork point and passes on the branch
