# Handoff: LC parity integration (1b7bc84) — code-cleanup pass

**Date:** 2026-07-10 · **Branch:** `refactor/cu121-uv-migration` @ `1b7bc84` · **Audience:** cleanup agent (no feature work — hygiene, consolidation, doc drift only)

## What landed (squash commit `1b7bc84`, 154 files, +7973/−165)

Loop closure now applies loops correctly on all four backbones (vggt_spark, vggtx, vggt_omega, mapanything). Full narrative + numbers: `docs/superpowers/specs/2026-07-09-lc-parity-probe-results.md` (read top-to-bottom — it is chronological: pre-fix baseline → post-fix → threshold recalibration addendum → matrix closing section). Plan with all design facts: `docs/superpowers/plans/2026-07-09-lc-fix-loop-application.md`. Original spec: `docs/superpowers/specs/2026-07-08-lc-parity-validation-design.md`.

Headline changes by area:
- `collab_splats/pointcloud/feedforward/`: verify contract returns `lc_data={"poses","world_points","conf"}` on all backbones; shared `_decode_verify_geometry` helper in base.py; per-model classvars `_lc_layer_index` / `default_verify_match_ratio` (spark 0.95 native, vggtx L10/1.17, omega L13/1.55, mapanything L4/1.46 — all calibration comments cite method/date/seed); mapanything `_lc_collate_outputs` emits depth/intrinsics_downsampled/depth_conf.
- `collab_splats/pointcloud/loop_closure/closure.py`: VGGT-SLAM-parity 3-edge scaled loop chain (`_lc_anchor_scale`, `_cam_local_points`, `_loop_chain_relatives`); `LoopClosureConfig.verify_match_ratio: float|None=None` resolved at wrapper init; dead `PoseGraph.add_loop_edge` removed.
- `collab_splats/pointcloud/wrappers.py`: per-model threshold resolution; LC Submaps carry world_points/conf; `_ablate_loops()` per-loop Δ-ATE (inspection attr `_lc_ablation_extrinsics`).
- `evals/`: driver pins subprocess PYTHONPATH to its own checkout; TUM SLAM ref restricted to GT-filtered frames (`--image_list`); `lc_loop_pr.py` (GT loop precision/recall table columns); `visualize_lc_correction.py` (3-trajectory HTML); eval_gt writes `loop_ablation.json`.
- Tests: +~250; suite on this branch after integration: **1056 passed, 0 failed, 2 skipped, 3 xfailed** (residue matches `docs/known-test-failures.md`).

## Cleanup targets (collected from the review trail — none blocking, all verified observations)

1. **Formatting:** `mapanything.py`, `vggtx.py`, `closure.py` are black/isort-dirty (pre-existing; grew during this work). One dedicated `style:` commit — do not mix with logic.
2. **`closure.py`:** unused `normalize_to_sl4` import (~line 20); `_lc_anchor_scale` (def ~386) calls `_cam_local_points` defined below (~491) — reorder for read order; `subsample: int = 8` param never passed by any caller (wire from config or drop — CLAUDE.md always-default rule); deprecated `lc_threshold`/`lc_cosine_threshold` config fields — `dataclasses.replace` in wrappers re-runs `__post_init__`, so a config using a deprecated field warns twice (drop the fields or suppress the second warning).
3. **`wrappers.py`:** two inline `logging.getLogger(__name__)` call-sites (~194, ~287) — promote to a module-level logger.
4. **`feedforward/base.py` ~869/877:** `_verify_loop_candidate` signature default `0.85` + "matches VGGT-SPARK calibration" docstring is drift — the wrapper always passes the resolved per-model value now; make the docstring describe the resolution chain instead.
5. **`vggt_omega.py` ~126:** check for a stale comment claiming the class "inherits default_verify_match_ratio=0.85" — superseded by the L13/1.55 calibration (b0aaf3f-era); reconcile comment with classvar.
6. **Import provenance:** spark/omega import `unproject_depth_map_to_point_map` from the installed `vggt` (VGGT-X tree), not their vendored trees — byte-identical today, silent-drift risk. Either import from each vendored tree or leave one comment acknowledging the shared dependency.
7. **`evals/eval_gt.py`** has grown (subprocess mode, keyframe guard, ablation writer) — candidate for splitting the LC-stats/ablation block into a helper module if you touch it anyway; judgement call.
8. **Test duplication:** fake-model stubs duplicated between `tests/pointcloud/feedforward/test_verify_lc_data.py` and `test_mapanything_creator.py` (deliberate at the time — concurrent editing); a shared conftest fixture would be cleaner.
9. **se3-manifold note:** sequential+loop scale folds into `Pose3` with a non-orthonormal Rot3 on the `manifold="se3"` path (pre-existing pattern; sl4 is the parity default). Flagged twice in review — document the limitation at the config field or fix properly; do not silently change behavior.

## Deferred work (documented in the results doc — NOT for the cleanup agent)

Dense d5 long-scene arms (office/redkitchen/fr3), KITTI generalization tier, production-sampler eval arm, jump-guard ablation (deprioritized: loop precision is 1.00 everywhere measured; the only harmful loop observed was GT-true — a correction-quality case, n=1, redkitchen/mapanything).

## Environment notes

- **Editable-install gotcha:** `import collab_splats` resolves to THIS checkout; any git-worktree run needs `PYTHONPATH=<worktree>:<worktree>/third_party/xfeat` (the parity driver pins this itself; regression test in `tests/evals/test_run_lc_parity.py`).
- **Raw eval outputs** (trajectories/pointclouds too heavy for git) archived at `evals/baselines/lc_parity_d5_postfix/` and `evals/baselines/lc_parity_matrix/` (gitignored). Metrics/decisions/tables are committed.
- Heavy runs: tmux only, serial (46 GB cgroup cap); `/opt/venv/reconstruction/bin/python`.
- The feature branch `feat/lc-parity-validation` and its worktree were deleted after this squash; granular history exists only in reflog until GC.
