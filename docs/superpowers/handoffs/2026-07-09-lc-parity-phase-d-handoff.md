# Handoff: LC Parity — Phase D (fix loop application), mid-flight

**Date:** 2026-07-09 · **Worktree:** `/workspace/collab-splats/.claude/worktrees/lc-parity` · **Branch:** `feat/lc-parity-validation` @ `46953ae`
**Read first:** plan `docs/superpowers/plans/2026-07-09-lc-fix-loop-application.md` (Tasks A/B/C, all design facts with file:line refs) · results `docs/superpowers/specs/2026-07-09-lc-parity-probe-results.md` · spec `docs/superpowers/specs/2026-07-08-lc-parity-validation-design.md`.

## Where things stand

**Done and validated (don't redo):**
- Parity harness complete (Tasks 1–13 of `docs/superpowers/plans/2026-07-08-lc-parity-validation.md`): driver `evals/runners/run_lc_parity.py` (multi-backbone, `--min_disparity`, resumable), gate table `build_parity_table.py`, helpers `lc_parity_common.py`, Level-2 `compare_loop_edges.py`, TUM support, per-candidate LC decision logs. 51 parity tests green (`tests/evals/test_lc_parity_common.py test_run_lc_parity.py test_build_parity_table.py test_lc_decisions.py test_ate_utils_tum.py test_compare_loop_edges.py`).
- **Chess d5 probe (pre-fix baseline, committed @ `8b2141c`):** SLAM ref 384 kf / 45 submaps / **21 loops** / ATE 0.0455. All 3 backbones retrieve exactly 21 candidates (retrieval+verify at upstream parity). spark: base 0.0442 PASS, lc 0/21 applied (all `no_joint_poses`) → no-op. omega: base 0.0300, lc 20/21 applied → **0.6255 HARMFUL**. mapanything: base 0.1927, lc no-op. Artifacts: `evals/baselines/lc_parity_d5/` (metrics/decisions committed; heavy files untracked — leave them).
- **Task D-A committed @ `46953ae`** (`fix(lc): verify paths return poses (+points where available); remove no_joint_poses drop`): verify contract widened to `(accepted, lc_data)` with `lc_data={"poses","world_points","conf"}`; spark extracts poses+depth-unprojected points from its native forward; mapanything derives poses via postprocess; the `wrappers.py` poses-None drop removed; LC Submap now carries world_points/conf. Implementer was cut off right AFTER committing ("136 passed, final gate" then commit landed) — tree is clean.

**⚠️ D-A is committed but UNREVIEWED and its GPU sanity check is unverified.** The implementer's planned end-to-end 2-frame spark verify (print lc_data shapes, poses[0]≈I) may not have run.

## Remaining work (in order)

1. **Review D-A** (`git show 46953ae`): verify per plan Task A — contract change consistent across base/spark/mapanything/wrappers; LC Submap reshape (2,H,W,3)→(2,H*W,3) matches `submap.py` conventions; reject_reason behavior preserved; no logic change beyond plan scope. Run the GPU sanity check from plan step A-sanity (2-frame spark `_verify_loop_candidate`, ~2 min) if GPU free.
2. **Task D-B — scaled 3-edge loop chain** (plan Task B, not started): `closure.py:551-566` — replace direct `add_loop_edge(nid_q, nid_d, inv(P0)@P1)` (inverted + unscaled = omega's catastrophe) with upstream-parity chain: 2 LC graph nodes + 3 edges (query→LC0 scaled anchor, LC0→LC1 inner `P0@inv(P1)`, LC1→detected scaled anchor), scale via `estimate_scale_pairwise` on pixel-aligned identical-image pointclouds with the conf fallback chain (mirror sequential-edge machinery closure.py:452-511 and upstream `solver.py:118-170,262-295`). Write numeric gtsam tests FIRST (plan B1 specifies fixtures: zero-error consistent graph, drift-correction, 2×-scale recovery). Key convention (verified numerically 2026-07-08): graph's between-factor form is `H_inner = P_prev @ inv(P_next)`; the old edge was its exact inverse.
3. **Task D-C — probe re-run:** same command as pre-fix probe, `--out_root evals/baselines/lc_parity_d5_postfix` (tmux, ~3 h, serial — 46 GB cap). Gates: spark lc **21/21 loops + ATE ≤ ~0.0478**; omega lc **HARMLESS** (≤ max(1.05×base, base+5mm)); mapanything **loops_applied ≥ 1**. Then `build_parity_table.py --root evals/baselines/lc_parity_d5_postfix`, commit metrics+table, append post-fix section to the results doc.
4. **After probe green (user decision):** multi-scene paper-config matrix — `run_lc_parity.py` full run; scene data state: 7-Scenes chess/fire/heads/office downloaded to `/workspace/collab-splats/evals/data/7scenes/`; pumpkin/redkitchen/stairs + all 4 TUM likely NOT downloaded (downloader died mid-run; rerun `bash /workspace/collab-splats/evals/data/download_parity_scenes.sh` in tmux — idempotent, skips existing).

## Environment gotchas (will bite you)

- **Worktree needs symlinks:** `third_party/*` subdirs are symlinked individually into the worktree (bare dir symlink breaks); `evals/data` is NOT symlinked — always pass `--data_root /workspace/collab-splats/evals/data` to the driver.
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). Heavy runs in tmux only, serial (memory cap).
- `import collab_splats.pointcloud` fails in this env (`No module named 'modules'` from localization.py, pre-existing) — 17 known env test failures in tests/evals + tests/pointcloud predate all Phase-D work; compare against stash baseline before blaming a change.
- `docs/superpowers/` and `evals/baselines/lc_parity*/metrics.json` need `git add -f` (gitignored but tracked-by-precedent).
- Parallel session on main branch promoted `collab_splats/localization/` to top-level and DELETED `pointcloud.localization` — merge conflict guaranteed at integration time; do not "fix" imports to the old path.
- tmux sessions `lc_parity` and `scene_dl` exist; probe log convention: `tee /tmp/<name>.log; echo <TAG>_EXIT_$?` + a Monitor grepping terminal signals only (per-candidate loop lines flood context — filter them out).

## Working method (this branch)

Subagent-driven development: fresh implementer per task with full task text + verified design facts inline, then spec review, then quality review (superpowers:code-reviewer). Commit messages: conventional, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

> **STATUS UPDATE 2026-07-09 (end of session):** Phase D+E complete on this branch — all chess-d5 gates green (spark 0.0421, omega 0.0186, mapanything 0.0555, all with real anchor scale, zero fallbacks). See plan 2026-07-09-lc-fix-loop-application.md (Tasks A-E ticked) and the post-fix section of specs/2026-07-09-lc-parity-probe-results.md. Remaining: vggtx verify-gate calibration (attention ratio anti-discriminative at layer 20 — sweep in flight), 6 stale-mock test fixes, 4-scene matrix.
