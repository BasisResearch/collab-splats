# Package consistency — design

- **Status:** phase 1 designed; phases 2-7 scoped, each gets its own spec section before its plan
- **Branch:** `clean/consistency`, worktree `.worktrees/consistency`, forked from `clean/final` @ `6343a0fb`
- **Audit report:** https://claude.ai/artifact/2tLJ8KpwcJ4c21yjfDiBw8 (9 audits, 2026-09-26)

## Goal

One implementation per convention across the package.

- the audit found ~2,300 LOC of repeated code in `collab_splats/` + `evals/`, ~600 in tests
- the duplication that matters is copies that disagree on a convention: w2c vs c2w,
  crop-aware vs crop-unaware intrinsics, `.jpg` vs `.png` frame ids, NaN handling
- every verified bug in the audit is one of those disagreements

## Branch and integration

One long-lived worktree; every phase lands on it in order.

- forked from `clean/final` @ `6343a0fb`, never merges a sibling in
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
| 2 | Shared low-level helpers | lazy `utils/__init__`; `jsonio`, `zarr_io`, `geometry/colmap.py`, `camera_centers`, `to_uint8_hwc`, `to_numpy`; adopt `get_device` / `pytorch_gc` |
| 3 | Backend hoisting | feedforward `_postprocess` / `_reproject` / QKV hook, LC pose extraction, SfM base |
| 4 | Pipeline and registry | single kind-tagged backend registry, `SceneLayout`, config helpers, `localization/store.py`, dashboard calls Reconstructor stages |
| 5 | API renames | `conf_percentile` / `min_conf`, `poses_w2c` / `poses_c2w`, `out_dir` / `images_dir` |
| 6 | Test consolidation | shared builders with non-identity defaults, conftest cleanup, autouse seeding |
| 7 | Dependencies | drop list, open3d to core, declare rich / scikit-image |

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

`evals/scripts/eval_verification.py:52` unpacks `fx, fy, cx, cy = cam.params`.

- effect: ValueError on SIMPLE_PINHOLE / SIMPLE_RADIAL, silent misassignment on longer param vectors
- contract: K from `cam.calibration_matrix()`
- test: SIMPLE_PINHOLE and PINHOLE cameras produce the same K for equal focal lengths

### 3. JSON NaN leak

`clean_for_json` only replaces builtin `float` NaN in dicts and lists.

- reproduced: `{'a': np.float32('nan'), 'b': (nan,), 'c': np.float64('nan')}` → `{"a": NaN, "b": [NaN], "c": null}`
- `geometry/metrics.py:696-699` writes its report with no conversion at all
- contract: `clean_for_json` handles `np.floating`, `np.integer`, tuples, ndarrays;
  NaN and ±inf → `None`; `metrics.py` routes through it
- fixed in place only: no `utils/jsonio.py`, no lazy `utils/__init__`, no migration of the
  other JSON sites — those are phase 2, which replaces `clean_for_json` with `to_jsonable`
- test: `json.dumps(clean_for_json(payload), allow_nan=False)` succeeds for float32 / float64 NaN,
  inf, a NaN inside a tuple and inside an ndarray

### 4. Crop-aware intrinsics

`feedforward/base.py:879` `_rescale_reconstruction_to_original_dimensions` scales K by the size
ratio and drops the crop top-left.

- crop box: `original_coords = [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]`
- crop-aware version exists: `geometry/metrics.py:239` `_scale_intrinsics_to_original` (`K / s + tl`)
- same defect suspected at `evals/scripts/eval_splats.py:73` and the mesh arm of
  `wrapper/reconstructor.py:663` (crop-aware depth upsample, crop-unaware K)
- also: pycolmap params are rescaled as `params[-2:] *= scale`, assuming cx, cy are last
- contract: promote to `geometry/transforms.rescale_intrinsics(K, crop_box, model_hw, *, to_original: bool)`;
  both directions; all three sites and `metrics.py` call it; pycolmap cameras rebuilt from the new K
- test: portrait image, crop with nonzero tl, off-centre principal point; a 3D point projected
  with model-res K then mapped through the crop equals its projection with original-res K;
  round-trip `to_original` then back is identity

### 5. Suspected crop and convention bugs

Each stays only if its failing test reproduces it; otherwise it is dropped with a note in the plan.

- **localizer ref-pixel lookup** (`localization/localizer.py:931-942`): rescales reference pixels
  onto the `world_points` grid by size ratio only, no crop offset; test: cropped reference, known 3D point,
  lookup returns that point
- **`original_coords` full-frame convention**: `mapanything.py:226` builds it from the model size,
  `loger.py:349` and `depth_align.py:303` from the original size; test: uncropped input, each
  backend's box round-trips K through `rescale_intrinsics` unchanged
- **viewer FOV** (`collab_splats/viewer.py`, line to be pinned in the plan): FOV computed before striding; test: strided and unstrided
  views give the same FOV

### 6. Localization DB provenance (suspected)

`dashboard/pipeline.py:499` `_stamp_db_provenance` reads `run_config.yaml` through the dashboard
`RunConfig.from_yaml`, which drops every key it does not define.

- a scene built by the Reconstructor has a `run_config.yaml` in the Reconstructor schema, so
  `env_model` falls back to its default and the DB is stamped with the wrong backbone
- contract: stamp `backbone` from whichever schema the file holds; the file-name collision
  itself is phase 4
- test: Reconstructor-written `run_config.yaml` naming a non-default backbone → stamped attrs
  name that backbone
- dropped from the audit: the `.jpg` vs `.png` frame-id split no longer exists on `clean/final`
  (`reconstructor.py:777`, `dashboard/pipeline.py:694` both `.jpg`)

## Error handling

- `rescale_intrinsics` raises `ValueError` on a crop box that is not 6 values or has zero size
- `rpe` keeps its `ValueError` for `delta >= N`

## Testing

- tests mirror the package: `tests/evals/`, `tests/geometry/`, `tests/pointcloud/`, `tests/localization/`
- control: the same per-package gates on the fork point, recorded before the first fix
- a fix is done when its new test fails on the fork point and passes on the branch
