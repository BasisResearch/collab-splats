# cu121 Test Baseline — Handoff for fixing failing tests

**Date:** 2026-05-31
**Branch:** `refactor/cu121`
**Why:** Blocks the [uv branch integration](2026-05-31-uv-branch-integration-design.md). That work
paused at its Phase 0 gate — the conda env baseline is far worse than documented, so we fix cu121
first, then resume the uv rebase.

## Goal

Get `refactor/cu121`'s test suite back to a clean baseline in the conda env `reconstruction`:
failures reduced to the documented known set in `docs/known-test-failures.md` (or that doc updated
to reflect reality). The key regression is **collection loss**, not new logic bugs.

## Current state

- Env: only one conda env exists — `reconstruction` (py3.11).
  `/opt/conda/envs/reconstruction/bin/python`. `torch 2.5.1+cu121` and `nerfstudio` import fine.
- Baseline run saved to `evals/results/baseline-conda-tests.txt` (gitignored).
  Command used:
  ```bash
  /opt/conda/envs/reconstruction/bin/python -m pytest tests/ -m 'not slow' \
      --ignore=tests/test_cu121_migration.py --continue-on-collection-errors -q -rfE
  ```
  Result: **73 failed, 346 passed, 4 skipped, 1 xpassed, 16 errors — 423 collected.**
- Documented baseline (`docs/known-test-failures.md`, 2026-05-26): 39 failed, 600 passed,
  2 collection errors, ~644 collected.
- **~220 tests no longer collect.** Whole modules error on import → they vanish from the run, which
  also explains the passed-count drop (600 → 346).

## Root cause: missing deps in `reconstruction` env

Import probes (`python -c "import X"`) in the env:

| Module | Result | Used by |
|--------|--------|---------|
| `torch` | OK (2.5.1+cu121) | core |
| `nerfstudio` | OK | core |
| `vggt` | **ModuleNotFoundError** | core (VGGT-X) — `tests/mesh/test_adapter.py`, feedforward |
| `evo` | **ModuleNotFoundError** | eval — `tests/evals/test_metrics_auc.py` |
| `panel` | **ModuleNotFoundError** | dashboard — all `tests/dashboard/*` |
| `param` | **ModuleNotFoundError** | dashboard — all `tests/dashboard/*` |

Looks like an **incomplete env build**, not the wrong env. Fixes likely belong in `setup.sh` /
`setup/feedforward.sh` (vggt) and dashboard/eval extras — verify each dep is actually meant to be
installed in this env, then install + re-run.

## Failures grouped by module (from `-rfE` summary)

**ERRORS (collection / import — fix these first, they unlock the most tests):**
```
tests/dashboard/test_app.py            ModuleNotFoundError: panel
tests/dashboard/test_localize_pane.py  (panel/param)
tests/dashboard/test_operation_log.py
tests/dashboard/test_preprocess.py
tests/dashboard/test_reconstruct_pane.py
tests/dashboard/test_semantics.py
tests/dashboard/test_smoke.py
tests/dashboard/test_state.py
tests/dashboard/test_video_server.py
tests/dashboard/test_visualize.py
tests/evals/test_metrics_auc.py        ModuleNotFoundError: evo
tests/mesh/test_adapter.py             ModuleNotFoundError: vggt
tests/test_feedforward_logging.py      ImportError
tests/test_models.py                   ModuleNotFoundError: nerfstudio.data
tests/wrapper/test_splatter_query.py   ModuleNotFoundError: nerfstudio.utils
tests/pointcloud (1)                   (collection)
```

**FAILED by module:**
```
14  tests/evals/test_eval_gt_helpers.py
14  tests/nerfstudio/test_datamanager_config.py   ← Group 6 namespace shadow (known)
 8  tests/dashboard/test_localize.py
 7  tests/scripts/test_reconstruct.py
 6  tests/wrapper/test_splatter_mesh.py            ← Group 6 namespace shadow (known)
 5  tests/examples/test_run_c0043_pipeline.py
 4  tests/nerfstudio/test_imports.py               ← Group 6 namespace shadow (known)
 3  tests/integration/test_pipeline_cu121.py
 3  tests/webapp/test_session.py
 2  tests/webapp/test_preprocess.py
 2  tests/webapp/test_reconstruct.py
 2  tests/webapp/test_visualize.py
 1  tests/mesh/test_tsdf.py
 1  tests/test_bae_smoke.py
 1  tests/wrapper/test_reconstructor.py
```

## Suggested order

1. **Install missing deps** (`vggt`, `evo`, `panel`, `param`) — biggest win; unlocks the 16 errors
   and likely many of the dashboard/eval/webapp failures. Confirm each belongs in the env and add to
   the relevant `setup/*.sh` so the fix survives a rebuild.
2. **Re-run baseline.** Re-measure collected count + failures.
3. **Triage the remainder** against `docs/known-test-failures.md` — the `nerfstudio.*` namespace
   shadow (Group 6) and the LC/VGGTX test-API groups are pre-known and have documented fixes.
4. **Update `docs/known-test-failures.md`** to the new reality (date + counts + groups).

## Resume protocol (uv integration)

Once cu121 is green: re-run Phase 0 to refresh `evals/results/baseline-conda-tests.txt`, then
proceed to Phase 1 (rebase uv onto cu121) per the
[integration spec](2026-05-31-uv-branch-integration-design.md). The uv worktree is untouched at
`51e5a93`; no rebase has occurred.
