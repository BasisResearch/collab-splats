# Handoff — BA pipeline wiring + ATE validation (2026-08-19)

**Branch:** `refactor/cu121-uv-migration` · **Status: COMPLETE** — implementation, validation, docs, memory all landed. Nothing in flight from this effort.

## What this effort delivered

Bundle adjustment is wired into the pipeline as leaf stage `refine`, validated live on all 4 backends, and two latent production bugs found by the first live BA run since the env upgrade are fixed.

### Commits (chronological)

| commit | what |
|---|---|
| `49e01ed` | spec: `docs/superpowers/specs/2026-08-18-ba-pipeline-wiring-design.md` |
| `244eab5` | plan: `docs/superpowers/plans/2026-08-18-ba-pipeline-wiring.md` (7 tasks, all done) |
| `a7e918a` | K-space guard in `_scale_intrinsics_to_model` (model-res K detection) |
| `c8d2423` | LC×BA mutual-exclusion ValueError; `build_pointcloud` NotImplementedError dropped |
| `ba3541e` | `Reconstructor.refine_poses` — BA refine + reproject + COLMAP/zarr/transforms rewrite |
| `2fcb6e0` | `refine` stage registration (inline bool + `--stages refine`) |
| `9852a78` | configs/README contract + base.yaml comment |
| `6c5bb13` | BA progress logging (per-LM-step loss, both optimizer paths) + `loger` in eval.py CLI |
| `6e8ab15` | **fix: bae/pypose target shim** — BA optimize was dead in production |
| `4e73fe8` | docs: 3 BA xfails marked resolved in known-test-failures.md |
| `173e0f5` | **fix: SVD-orthogonalize rotations before `mat2SE3`** (LoGeR crash) |
| `f6e13a9` | spec: measured ATE results + CLAUDE.md Recently-completed entry |

## Design recap (contracts the next person must know)

- **One impl, two triggers.** `Reconstructor.refine_poses(overwrite)` runs for both `pointcloud.bundle_adjustment: true` (inline, appended after pointcloud) and `--stages refine` (from `environments-processed`, zero rerun.py changes). Always loads `FeedforwardResult.load_zarr(zarr_path, load_images=True)` — never a live creator.
- **Persist path:** `build_pycolmap_reconstruction` (model-res) → `_rescale_reconstruction_to_original_dimensions` → `write_binary(colmap/sparse/0)`; zarr `r+` write-back of extrinsics/intrinsics/points + `world_points` recomputed via `unproject_depth_map_to_point_map`; then `_load_pointcloud_from_disk` → `_export_pointcloud_ply` + `_write_transforms_json`. Marker + provenance: `colmap/refine.json` (BA config asdict, loss history, n_frames).
- **LC×BA mutually exclusive** — `ValueError` in `validate_config`.
- **Staleness contract:** refine does NOT invalidate mesh/semantics/localize outputs — user re-runs dependents with `overwrite`. Deliberate: adding a dep would demote them from `LEAF_STAGES`. See `configs/README.md`.
- **K-space guard:** `_scale_intrinsics_to_model` compares `2·cx` vs `W_model` vs crop centre. Model-res K (what every creator now emits) → identity; legacy original-res K → old crop-aware scaling. Without it BA double-applied the crop transform on ALL backends.

## Bugs fixed during validation (both would bite anyone touching BA)

1. **bae 0.2.4 × pypose 0.7.5 incompatibility (`6e8ab15`).** bae's `LM.step` calls `self.model(input)` with no `target`; pypose ≥0.7 `RobustModel.forward(input, target)` has no default → every LM step raised TypeError. BA optimize was dead in production since the env upgrade; the 3 xfails in `tests/geometry/test_bundle_adjustment.py` tracked exactly this and are now removed (file is 25 green). Fix: bind `target=None` on the wrapped model instance via `functools.partial` right after constructing `LM`. If bae or pypose is ever upgraded, re-check this seam first.
2. **LoGeR rotations fail `mat2SE3` (`173e0f5`).** se3-refined float32 extrinsics aren't strictly orthogonal → pypose raises "Input rotation matrices are not all orthogonal matrix". Fix: SVD-project to nearest proper rotation (det>0 enforced) before conversion; exactly-orthogonal inputs pass through unchanged.

## Measured results (chess/seq-01, 100 frames, single-pass, baseline vs `ba`)

Full table + interpretation in the spec (§Validation). Headline:

- **mapanything**: BA improves everything (ATE 0.0133→0.0120, RPE-t −25%, AUC@5 8.3→19.2).
- **vggtx / vggt_omega / loger**: BA HURTS ATE (+28–72%); omega's AUC@5 collapses 55.2→9.4. loger's RPE-rot halves (1.08°→0.55°) even as ATE worsens.
- **Verdict: default stays `bundle_adjustment: false`.** BA is for visibly-bad baselines, not a free upgrade — sub-cm feedforward poses sit near the 7-Scenes GT noise floor and the shipped knobs (SIMPLE_PINHOLE shared-per-frame focal, VGGSfM model-res tracks, max_reproj 4.0) inject more track noise than they remove.

Raw results: `evals/results/ba_wiring_chess/<backbone>/{ate.json, metrics.json, *.tum}` + per-backbone logs (gitignored, on this machine only).

### Reproduce

```bash
/opt/venv/reconstruction/bin/python evals/scripts/eval.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --output_dir evals/results/ba_wiring_chess/<bb> \
  --max_frames 100 --backbone <bb> \
  --conditions baseline ba \
  --output_ate evals/results/ba_wiring_chess/<bb>/ate.json
```

Backbones: `vggtx vggt_omega mapanything loger` (loger CLI support added in `6c5bb13`). Run serially in tmux (cgroup 46.6 GB). **No `--submap_size`** — windowed BA is broken (below).

## Known gaps / follow-ups (none blocking)

- **Windowed BA (`--submap_size` + `ba`) still broken** — LC `_assemble_result` doesn't populate `FeedforwardResult.images` → `_extract_tracks_vggsfm` AttributeError. Known, scoped out (docs/known-test-failures.md 2026-07-21). Single-pass caps eval at ~100–300 frames.
- **BA knobs unexplored** for the hurts-good-baselines result: per-camera focal, tighter `max_reproj_error`, track density. Only worth touching with a scene whose baseline is visibly bad.
- **Alternative track source**: COLMAP `database.db` matches from geometric-verification — natural follow-on, zero code committed.
- bae LM's internal `print("Loss: ...")` spam is upstream, not ours; our INFO logging is the supported surface.

## Environment notes

- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Heavy runs: tmux only, serial.
- Concurrent-session trap: another session keeps uncommitted `configs/base.yaml` mods (causes 5 wrapper-test "failures" that are NOT ours). Partial-stage with `git hash-object -w` + `git update-index --cacheinfo` if you must commit a hunk of a dirtied file.
- graphify graph current as of f6e13a9.
