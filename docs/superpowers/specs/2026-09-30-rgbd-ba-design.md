# RGB-D bundle adjustment for feedforward poses

Branch `feat/rgbd-ba` off `clean/reconstructor-release@971f499b`, worktree `.worktrees/rgbd-ba`.

## Goal

Measure whether depth or photometric residuals in our BA improve VGGT-Omega poses. Report-only:
BA stays off by default; the decision follows the numbers.

## Background

- We pin bae `0.2.4` (`90909b33`); the installed copy is byte-identical to the tag.
- bae branch `rgbd` (`c2c1c34`, 2026-09-14) adds **no library code**. Its three own commits live in
  `examples/rgbd_pose_refine/`; the rest of its diff is release history up to 0.2.5.
- The example is dense photometric pose refinement on Depth Anything 3 output, not RGB-D BA:
  - residual: linearized intensity `I_i − (I_j + ∇I_j·Δuv)` over depth-induced warps
  - optimizes poses only, view 0 fixed; depth is held fixed and only unprojects pixels
  - needs `pypose.autograd.psjac` (unreleased pypose); we have pypose 0.7.5
  - every validation run sets `BAE_USE_PYPOSE_AMBIENT_GRAD=1`
- Their results (`docs/validation/RESULTS.md`, 50 ScanNet++ scenes, AUC@5):
  - DA3 29.04 → stage D raw 36.83 → D+E raw 36.89: stage E adds +0.07 pp
  - guards reject 41/50 improvements: guarded D 29.46
  - self-calibration runs on every arm, baseline included: input prep, not part of the gain
- Omega's `pointcloud.zarr` already holds what their method consumes: model-grid images, depth,
  confidence, K and world-to-cam poses from one forward pass.
- Our 2026-08-19 diagnosis: the track objective is not minimized at true poses; depth priors and
  observation quality are the remaining levers.

## Design

### Residual terms

One `BundleAdjustment`, one LM. Each term is switched by a config flag; `_BAModel.forward`
concatenates the active blocks.

- **reprojection** (today): track pixel error, σ = 1 px.
- **photometric**: rgbd stage D, ported whole.
  - kept: 3-level pyramid with (3, 2, 1) relinearizations × 5 LM steps, gradient-weighted pixel
    sampling (2048 per view), pose-kNN covisibility (16), 3× weight on consecutive pairs, IRLS Huber
    with δ = 1.5 × median residual, occlusion check on depth
  - dropped, with the evidence above: stage E, guards, SIFT self-calibration
  - `psjac` becomes `@map_transform`; c2w becomes our w2c
  - pixel lift and projection reuse `geometry/projection.unproject` / `project`
  - lives in `geometry/photometric.py`; the solve stays in `bundle_adjustment.py`
  - attribution at file top: `pypose/bae @ c2c1c34, examples/rgbd_pose_refine/photometric.py`
- **depth (C1)**: `√c_i(u) · (z_i(X_p) − D_i(u)) / (D_i(u) · depth_sigma)` per track observation.
  - `z_i(X_p)`: from the existing `_reproject_*`, which now also return camera z
  - `D_i(u)`, `c_i(u)`: Omega depth and confidence indexed nearest-neighbor at the observed track
    pixel, inline, once before the solve
  - confidence normalized to median 1 per scene
  - rows with `D_i(u) ≤ 0` dropped; reprojection rows kept
- **depth scale (C2)**: C1 with `s_i · D_i(u)`, `s_i = exp(σ_i)`, one free `(N, 1)` tensor.

Parameters registered per active term: poses always; landmarks and focal only with reprojection;
`σ` only with depth scale.

### LM

- one setup for all terms: `reject=10`, CuDSS, `TrustRegion(up=2, down=1/16)`
- deviation from rgbd: `reject` 30 → 10, PCG → CuDSS; neither moves the optimum
- photometric on: the pyramid schedule sets the step count (30); otherwise `lm_steps` (40)
- pyramid loop and IRLS reweight inline in the existing `_optimize` step loop

### Gauge fix

The residuals cannot see some whole-scene motions:

| Terms | Unobservable | Scale set by |
|---|---|---|
| reprojection | Sim(3), 7 DoF | nothing |
| reprojection + depth | SE(3), 6 DoF | Omega depth |
| reprojection + depth scale | Sim(3), 7 DoF | nothing |
| photometric | SE(3), 6 DoF | Omega depth |
| reprojection + depth + photometric | SE(3), 6 DoF | Omega depth |

- after every solve, `umeyama_sim3` of refined onto input camera centers (active frames), applied
  to all refined poses; scale also multiplies `s_i`
- scale fitted only when nothing sets it; else scale is forced to 1, translation recomputed inline
- no residual changes; the output lands in the input gauge
- fixes a current bug: today's scale drift misplaces points reprojected from fixed Omega depth
- done by editing `_carry_dropped_frames` in place (it already calls `umeyama_sim3` on centers):
  apply the alignment to active frames too; dropped frames keep their input poses
- fitted scale kept as `ba.gauge_scale`, written to `refine.json` (for reprojection-only, the
  first measure of today's drift)

### Ambient-grad patch

- bae 0.2.4 installs it at `import bae` (`bae/__init__.py:3`) and reads the env var per call
  (`bae/utils/parameter.py:26,36`)
- `bundle_adjustment.py`, after its imports: set `BAE_USE_PYPOSE_AMBIENT_GRAD=1`, call
  `install_pypose_ambient_grad_monkeypatch()`
- process-wide: it changes today's reprojection BA too, so every grid cell runs under it

### Config

```yaml
bundle_adjustment:
  enabled: false
  use_reprojection: true
  use_photometric: false   # not with increment_size > 0
  use_depth: false         # needs use_reprojection
  fit_depth_scale: false   # needs use_depth
  depth_sigma: 0.01        # relative depth error equal to 1 px
```

- replaces the bool `pointcloud.bundle_adjustment`; existing checks (refused with loop closure,
  refused for sfm) read `enabled`
- `BundleAdjustmentConfig` gains the same five fields
- invalid combinations (including no term) raise `ValueError` in the existing Reconstructor
  validation block beside the LC / sfm checks; no `__post_init__`

### Data flow

- `Reconstructor.refine()` builds the config from the block and passes `ff.depth`,
  `ff.confidence` to `ba.refine()`
- `ba.refine()` return signature unchanged; `depth_scales` exposed as an attribute, like
  `loss_history`, ones unless `fit_depth_scale`
- `refine()` multiplies `ff.depth` by `depth_scales` before `reproject()`; the zarr is rewritten
  whole, so depth, poses and points on disk agree
- tracks extracted only when `use_reprojection`
- `refine.json` adds `losses` (final loss per active term, sliced from the stacked residual
  in `_optimize`), `gauge_scale`, `depth_scales`; all three are `ba` attributes beside
  `loss_history`, so the existing `write_json` call only gains keys
- only caller is `reconstructor.py:621`; the BA tutorial notebook is already broken and stays
  tutorial-rework's job

## Code change budget

New functions: two, both private, both in the new `geometry/photometric.py`.

- `_photometric_residual`: `@map_transform` linearized residual (rgbd `psjac`)
  - must be a function: bae differentiates `@map_transform` functions, as with `_reproject_*`
  - private: its only caller is `_BAModel.forward`; nothing in preproc or metrics warps
    differentiably (`preproc/qa.py` is per-frame blur/exposure; `metrics.compute_photometric_ncc`
    is a numpy, nearest-read scoring warp with no gradients)
- `_photometric_rows`: sampled pixels, depths, image gradients and covisible pairs for one
  pyramid level
  - a function because `_optimize` calls it once per level (3 levels) and it holds the ported
    rgbd sampling code under the file-top attribution
  - covis kNN (`cdist` + `topk`), gradient-weighted sampling, gradients and occlusion check inline

Edited in place, no new functions:

| File | Change |
|---|---|
| `geometry/bundle_adjustment.py` | config fields; `refine` gains `depth`, `confidence`; `_optimize` builds term rows, pyramid + IRLS in its loop; `_BAModel` gains optional `σ` and photometric/depth rows in `forward`; `_reproject_*` also return camera z; `_carry_dropped_frames` aligns all frames; ambient-grad patch after imports |
| `reconstructor.py` | validation block reads `["enabled"]` + combination checks; stage map line 432 → `["enabled"]`; `refine()` builds config from block, passes depth/confidence, applies `depth_scales` |
| `configs/base.yaml` | bool → block |
| `configs/README.md` | line 44 |
| `evals/configs/rgbd_ba.yaml` | new grid |
| `tests/geometry/` | tests below |

No new classes, helpers or constants.

## Tests

Run before any grid cell:

1. finite-difference Jacobian check for every residual, today's reprojection included, with and
   without the ambient-grad patch
2. synthetic scene (6 frames, 200 points, non-collinear baseline): stacked Jacobian's near-zero
   singular values equal 7 / 6 / 7 / 6 / 6 for the five term sets above
3. gauge fix: residuals identical before and after; a dropped frame keeps its input pose
4. config validation: each invalid combination raises
5. smoke: `omega_joint` on 200 chess frames, peak RSS logged (cgroup cap 46.6 GB); if it
   overflows, lower pixels per view and record it

## Evaluation

`evals/configs/rgbd_ba.yaml`, base `semantics` and `mesh` off, no loop closure.

- datasets: 7-Scenes chess, fire, office, seq-01, `max_frames: 200`
- 6 conditions × 3 scenes = 18 cells

| Condition | Terms |
|---|---|
| `omega` | no BA |
| `omega_reproj` | reprojection |
| `omega_photometric` | photometric |
| `omega_reproj_depth` | reprojection + depth |
| `omega_reproj_depth_scale` | reprojection + depth scale |
| `omega_joint` | reprojection + photometric + depth |

- readouts: ATE, RPE (GT); photometric NCC, cross-view depth agreement (reference-free)
- results written up in a decision doc

## Risks

- 7-Scenes Kinect RGB auto-exposure breaks brightness constancy; no gain/offset term (parity).
  The NCC readout normalizes gain away and will not show it. First suspect if photometric loses.
- `s_i` is weakly determined for frames with few tracks or near-pure rotation.
- short baselines weaken focal and photometric translation along the optical axis.

## Out of scope

- bae pin bump (0.2.5+: hard `warp` import, cuDSS < 0.9, BSR removal)
- stage E, guards, self-calibration, brightness terms
- weight sweeps; other backbones
- using BA with loop closure
