# cu121 Test Fixes — Design

**Date:** 2026-06-01
**Branch:** `refactor/cu121`
**Predecessor:** [cu121 test baseline handoff](2026-05-31-cu121-test-baseline-handoff.md)
**Unblocks:** [uv branch integration](2026-05-31-uv-branch-integration-design.md) Phase 0 gate

## Problem

The `reconstruction` conda env (only env present; py3.11) runs the suite far below the
documented baseline:

- **Now:** 422 collected · 16 collection errors · ~73 failed.
- **Documented** (`docs/known-test-failures.md`, 2026-05-26): ~644 collected · 2 collection
  errors · 39 failed.
- ~220 tests no longer collect — whole modules error on import and vanish, which also drops
  the passed count (600 → 346).

Investigation found **two independent root causes** plus a residual band of genuine
test-vs-code drift. The handoff's single "missing deps" hypothesis is correct but incomplete.

## Root causes (verified)

### Cause 1 — incomplete env build (14 of 16 collection errors)

Import probes in `reconstruction`: `torch, nerfstudio, pypose, bae` OK;
`vggt, evo, panel, param, mapanything, gtsam, salad` **missing**.

| Collection errors | Missing dep | Declared in | Installed by | Why missing |
|---|---|---|---|---|
| 10 × `tests/dashboard/*` | panel, param | pyproject `[dashboard]` | **nothing** | `[dashboard]` extra is orphaned — no setup script installs it |
| `evals/test_metrics_auc` | evo | pyproject `[feedforward]` + `setup/feedforward.sh` | `setup/feedforward.sh` | script never run in this env |
| `mesh/test_adapter`, `pointcloud` (agg), `test_feedforward_logging` | vggt | `third_party/VGGT-X` submodule + `setup/feedforward.sh` | `setup/feedforward.sh` | same |

Two distinct gaps:

- **vggt / evo / mapanything / gtsam / salad** — `setup/feedforward.sh` installs them, but
  `setup.sh` (default install) never calls it. CLAUDE.md treats feedforward as a separate
  manual step. This env skipped it.
- **panel / param** — declared only as the pyproject `[dashboard]` extra; **no setup script
  installs them at all**. A perfect clean rebuild still leaves them missing.

Beyond the 14 collection errors, this is the dominant effect: the ~220 vanished tests are
mostly downstream of these missing imports. **Most of the ~73 reported failures cannot be
triaged until deps are restored**, because installing them changes the collected set.

### Cause 2 — `tests/nerfstudio/` namespace shadow (2 collection errors + ~20 downstream failures)

`tests/test_models.py` and `tests/wrapper/test_splatter_query.py` **collect clean in
isolation** but error in the full run with `ModuleNotFoundError: No module named
'nerfstudio.data' / 'nerfstudio.utils'`. Dep-independent (nerfstudio is installed and imports
fine).

Mechanism — sharper than `known-test-failures.md` Group 6 states. Config already sets
`--import-mode=importlib` and `tests/nerfstudio/__init__.py` exists (added 2026-05-27). That
`__init__.py` is the **culprit, not a fix**: in importlib mode pytest derives a module's
top-level package by walking up through `__init__.py` files. `tests/nerfstudio/` has one,
`tests/` does not → the derived name for `tests/nerfstudio/test_datamanager_config.py` becomes
`nerfstudio.test_datamanager_config`. Pytest inserts `tests/` on `sys.path` and registers
`tests/nerfstudio` as `nerfstudio` in `sys.modules`, shadowing the real site-package for the
rest of the session. Every later `import nerfstudio.<submodule>` then fails.

Confirmed: running `tests/nerfstudio/ tests/test_models.py tests/wrapper/test_splatter_query.py`
together reproduces 2 collection errors; running the two victims alone passes collection.

### Residual — test-vs-code API drift (~50 failures, count unconfirmed)

In modules that **do** collect: `known-test-failures.md` Groups 2,3,4,5,7,8 (LC verifier API,
VGGTX `model_name` default, depth tensor shape, `_mapanything` attribute, `pose_convention`
abstract stubs, reconstruct-smoke `nerfstudio.process_data` import) plus handoff-reported
`evals/test_eval_gt_helpers` (14), `scripts/test_reconstruct` (7), `examples` (5),
`integration/test_pipeline_cu121` (3), `webapp/*` (9). Genuine drift between tests and current
code. **Counts are provisional** — installing deps (Cause 1) changes which modules collect, so
the residual set must be re-measured before triage.

## Solution — staged

Ordered so each stage's effect is measurable before the next.

### Stage 1 — make the env build complete by default

Per user decision: feedforward + dashboard deps install by default.

- **`setup.sh`**: after Step 3, add a step that runs `setup/feedforward.sh`, and a step that
  installs dashboard deps (`pip install -e '.[dashboard]'`). Keep `setup/feedforward.sh` itself
  callable standalone (no behavior change to it).
- Result: `vggt, evo, mapanything, gtsam, salad, panel, param` all present after a default
  `bash setup.sh`. Unlocks 14 collection errors and restores the ~220 vanished tests.
- Out of scope (deferred per user): restructuring the extras / `[dashboard]` vs `[feedforward]`
  split, and the uv migration. This stage only closes the install gaps in place.

### Stage 2 — kill the namespace shadow

- Rename `tests/nerfstudio/` → `tests/nerfstudio_methods/` (or similar non-`nerfstudio` name)
  and remove the misleading `__init__.py` so pytest cannot derive a top-level `nerfstudio`.
- Update any intra-suite references to the old path.
- Unlocks 2 collection errors + the ~20 Group-6 downstream failures.
- Correct the `known-test-failures.md` Group-6 note: root cause is the `__init__.py`-driven
  top-level derivation under importlib mode, not merely the directory name.

### Stage 3 — re-baseline

- Re-run `pytest tests/ -m 'not slow' --ignore=tests/test_cu121_migration.py
  --continue-on-collection-errors -q -rfE` in the fixed env.
- Record new collected / passed / failed / errors. Confirm collection errors → 0 (or only
  intentional).

### Stage 4 — triage residual drift

- Diff the re-baselined failures against `known-test-failures.md` Groups 2,3,4,5,7,8.
- Apply the documented per-group test fixes (mostly updating tests to current API/signatures).
- Investigate handoff-reported bands (`eval_gt_helpers`, `scripts/test_reconstruct`,
  `examples`, `integration`, `webapp`) that are not yet grouped; classify each as test drift vs
  real regression.

### Stage 5 — update `known-test-failures.md`

- Rewrite to the new reality: date, counts, surviving groups, and the corrected Group-6 cause.
- This becomes the refreshed Phase-0 baseline the uv integration resumes from.

## Success criteria

- Default `bash setup.sh` yields an env where `vggt, evo, panel, param` all import.
- 0 collection errors (or only deliberately ignored modules).
- Collected count back near the documented ~644 (±, accounting for tests added/removed since).
- Remaining failures are an explained set, all recorded in `known-test-failures.md`.

## Out of scope

- uv package-manager migration (resumes after this work; the uv worktree at `51e5a93` is
  untouched).
- Extras/dependency-structure refactor (user will address separately).
- Fixing genuine product regressions surfaced in Stage 4 beyond classifying them — those, if
  any, become their own tasks.
