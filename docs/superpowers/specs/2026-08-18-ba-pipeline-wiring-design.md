# Bundle Adjustment Pipeline Wiring — Design

**Date:** 2026-08-18
**Status:** Draft — awaiting user review
**Goal:** `pointcloud.bundle_adjustment: true` refines camera poses in the pipeline; `--stages refine` re-runs BA against a processed scene from disk. One implementation, two triggers.

## Implementation principles

- **Reuse first.** The optimizer (`BundleAdjustment.refine`), track extraction + zarr cache, deterministic reprojection (`FeedforwardResult.reproject`), zarr loading (`FeedforwardResult.load_zarr`), COLMAP writing (`build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`), PLY export (`Reconstructor._export_pointcloud_ply`), transforms export (`Reconstructor._write_transforms_json`), and stage machinery (`_STAGE_DEPS` / `LEAF_STAGES` derivation, `prepare_scene`) all exist. This design adds **no new optimizer code, no new persistence formats, no new stage-plumbing code** — only one stage method, one config guard, and one intrinsics-resolution guard.
- **No unnecessary functions.** No `_persist_refined` wrapper layer: persistence is inline calls to the existing free functions inside the one new stage method.
- **No overengineering.** Config surface stays ONE boolean. `BundleAdjustmentConfig` knobs stay code-level defaults. No pluggable track-source abstraction in this pass (COLMAP `database.db` matches as an alternative track source is a noted follow-on, not built).
- **Inline block comments** per repo style — each logical block gets a short `# what this does` comment.

## Context

- `collab_splats/geometry/bundle_adjustment.py` is complete and validated: `BundleAdjustment.refine(result)` (VGGSfM track extraction with zarr cache, LM optimize, allonce + incremental), method-agnostic over `FeedforwardResult`.
- `evals/scripts/eval.py` already runs the `ba` condition end-to-end: `BundleAdjustment(cfg).refine(creator.outputs).reproject()`.
- `Reconstructor.build_pointcloud` (reconstructor.py:643) raises `NotImplementedError` when `pointcloud.bundle_adjustment: true` — the only gap is wiring.
- `feedforward.zarr` persists `images`, `depth`, `world_points`, `confidence`, `pixel_indices` — everything track extraction and reprojection need, so BA runs from disk with **no live creator**.
- `result.reproject()` (depth + `pixel_indices`, all model-res) re-derives points deterministically without creator state.
- Known trap: VGGT-Omega's zarr stores **model-res** intrinsics while `_scale_intrinsics_to_model` assumes original-res K → double-scale risk (prior finding: "localization OK, BA not").

## Design

### 1. New stage: `refine`

- Add `"refine"` to `_STAGE_ORDER` (after `pointcloud`) and `_STAGE_DEPS["refine"] = ["pointcloud"]`.
- `refine` is a leaf by the existing `LEAF_STAGES` derivation → `--stages refine` pulls the scene from `environments-processed` via the existing `prepare_scene` machinery with **zero changes to `remote/rerun.py`**.
- Deliberately NOT a dependency of `mesh`/`semantics`/`localize` — that would demote them from the leaf set and break their disk re-run path. Staleness is a documented contract instead (§4).

### 2. One stage method, two triggers

New `Reconstructor.refine_poses(overwrite: bool = False)`:

1. Resolve input: use in-memory `FeedforwardResult` when the pointcloud stage just ran; else `FeedforwardResult.load_zarr(backend_dir / "feedforward.zarr")`.
2. Refuse when already refined and not `overwrite` (same named-stage refusal semantics as other stages; marker in §3).
3. `result = BundleAdjustment(BundleAdjustmentConfig(tracks_cache_dir=backend_dir)).refine(result).reproject()` — the exact call chain eval.py already exercises; `reproject()` is the creator-free zarr path.
4. Persist (§3).

Triggers:

- **Inline:** `build_pointcloud` replaces the `NotImplementedError` with a call to `refine_poses()` after the feedforward result exists (before mesh/semantics/localize run, so downstream stages in the same run see refined poses).
- **Disk:** `--stages refine` dispatches to the same method through the existing stage runner.

Config validation: `pointcloud.bundle_adjustment: true` **and** `pointcloud.loop_closure: true` → `ValueError` at config load ("BA over LC submaps is not supported; disable one"). LC submaps do not carry the tensors track extraction needs; per-submap BA is out of scope.

### 3. Persistence — reuse the creator's exact write path

Inside `refine_poses`, with the refined `FeedforwardResult`:

- COLMAP: `build_pycolmap_reconstruction(...)` at model res, then `_rescale_reconstruction_to_original_dimensions(...)`, then `recon.write_binary(colmap/sparse/0)` — the same three calls `BaseFeedforwardCreator.build_colmap` makes, invoked directly (they are free functions; no creator needed).
- `transforms.json`: existing `Reconstructor._write_transforms_json`.
- `sparse_pc.ply`: existing `Reconstructor._export_pointcloud_ply`.
- `feedforward.zarr`: write back the pose-derived arrays — `extrinsics`, `intrinsics`, `points`; recompute `world_points` from stored `depth` under refined poses using the existing unprojection helper (same math as `result.reproject`, full grid). Zarr and COLMAP must not disagree — localization reads `world_points`.
- Marker + provenance: one small `colmap/refine.json` (BA config used, LM loss history from `BundleAdjustment._last_loss_history`, timestamp). Doubles as the `_stage_output_exists("refine")` marker — no separate marker file.

Nothing new is invented: same artifacts, same writers, plus one provenance JSON.

### 4. Staleness contract

`refine` run from disk does **not** auto-invalidate `mesh/`, lifted semantics, or the localization DB built under old poses. Documented in `configs/README.md` (same section as the stage re-run contract): after `--stages refine`, re-run dependents with `overwrite` if pose-sensitive outputs matter. Inline-trigger runs never hit this (refine runs before dependents).

### 5. Omega intrinsics guard

In `_scale_intrinsics_to_model`: detect K already at model resolution (principal point consistent with `model_width/2, model_height/2` rather than original dims from `original_coords`) → skip scaling, `logger.warning` once. Prevents silent double-scaling on VGGT-Omega zarr loads. Loger's stored K verified the same way during implementation.

### 6. Config surface

- `pointcloud.bundle_adjustment: false` — key unchanged, comment updated ("refines camera poses via LM bundle adjustment; incompatible with loop_closure").
- All knobs remain `BundleAdjustmentConfig` defaults (`increment_size=0` = global BA). No nested yaml block.

## Backends

All four (vggtx, mapanything, omega, loger). BA is method-agnostic over `FeedforwardResult`; omega needs only the §5 guard; loger is exercised in validation (§8).

## Error handling

- LC + BA both enabled → `ValueError` at config validation (fail loud, never silent skip).
- `refine` named without `overwrite` when `colmap/refine.json` exists → refusal, consistent with other named stages.
- `load_zarr` missing required arrays (pre-BA-era store) → the loader's existing error propagates; no fallback path.

## Testing

- Config validation: LC+BA → `ValueError`.
- Inline trigger: `build_pointcloud` with `bundle_adjustment: true` calls `refine_poses` (BA mocked).
- Disk path: synthetic `feedforward.zarr` fixture → `refine_poses` refines + persists; COLMAP poses, `transforms.json`, zarr arrays all updated and mutually consistent.
- Refusal: second `refine_poses()` without `overwrite` refuses.
- Omega K guard: model-res K input → no double-scale, warning emitted; original-res K → scaled exactly as before (regression).
- Existing eval.py `ba` condition untouched by construction (imports only, no signature changes).

## Validation (compute, tmux, human-gated)

ATE on 7-Scenes chess/seq-01, all 4 backends, baseline vs `bundle_adjustment: true`, via existing `evals/scripts/eval.py` conditions. Reference-free control run alongside (ATE-vs-GT noise-floor lesson from LoGeR bench). Expected outcomes documented, including the known BA no-op on tiny-baseline scenes (C0043). Numbers append to this spec.

### Measured results (2026-08-18/19, chess/seq-01, 100 frames, single-pass, `--conditions baseline ba`)

Single-pass only — the windowed (`--submap_size`) BA path stays broken by the known LC
`_assemble_result` images gap and was not run. Results dir:
`evals/results/ba_wiring_chess/` (gitignored); logs per backbone alongside.

| backbone | ATE base | ATE ba | RPE-t base→ba | RPE-rot base→ba | AUC@5 base→ba | time base→ba |
|---|---|---|---|---|---|---|
| vggtx | 0.0082 | **0.0141** | 0.0085→0.0087 | 0.186→0.198° | 20.4→13.1 | 143s→129s |
| vggt_omega | 0.0080 | **0.0109** | 0.0087→0.0086 | 0.183→0.189° | 55.2→9.4 | 93s→128s |
| mapanything | 0.0133 | **0.0120** | 0.0188→0.0141 | 0.213→0.201° | 8.3→19.2 | 167s→189s |
| loger | 0.0073 | **0.0094** | 0.0095→0.0095 | 1.082→0.553° | 23.7→13.7 | 220s→264s |

Read: BA **helps only the weakest baseline** (mapanything: every metric improves — ATE −10%,
RPE-t −25%, AUC@5 +11pt) and **hurts the three sub-cm baselines** (vggtx/omega/loger ATE +28–72%,
AUC@5 drops; omega worst, 55.2→9.4). One genuine bright spot elsewhere: BA halves loger's
rotational drift (RPE-rot 1.08°→0.55°) even while ATE worsens. Consistent with the C0043
tiny-baseline no-op and the CO3Dv2 findings (BA helped there, where baselines were far worse):
at ~8mm ATE the feedforward poses sit near the 7-Scenes GT noise floor and the shipped BA
knobs (SIMPLE_PINHOLE shared-per-frame focal, VGGSfM tracks at model res, max_reproj 4.0)
inject more track noise than they remove. **Default stays `bundle_adjustment: false`**; BA is
the right tool when the baseline is visibly bad, not a free upgrade on good scenes.

The run also flushed out and fixed two latent production bugs (first live BA since the
bae/pypose env upgrade): bae 0.2.4 `LM.step` vs pypose 0.7.5 `RobustModel.forward(target)`
TypeError — BA optimize was dead in production, the 3 test xfails tracked it (fix `6e8ab15`,
xfails removed); and LoGeR's float32 se3-refined rotations failing `mat2SE3`'s orthogonality
check — fixed by SVD projection to the nearest proper rotation before conversion.

## Out of scope

- BA over LC submaps (images gap; separate effort if ever).
- Pluggable track sources (COLMAP `database.db` matches from geometric-verification — natural follow-on, zero code now).
- Closing the in-flight **bae-vggt-parity** spec (separate validation effort).
- Auto-invalidation of downstream stages after disk-path refine.
