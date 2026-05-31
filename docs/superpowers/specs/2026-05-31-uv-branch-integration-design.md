# uv Branch Integration + Staged Env Debug — Design

**Date:** 2026-05-31
**Status:** PAUSED at Phase 0 gate — cu121 test baseline must be fixed first (see Handoff below)
**Author:** Tommy

## Handoff (2026-05-31) — Phase 0 gate tripped

Phase 0 ran but **failed the env-health gate**. Do NOT rebase (Phase 1) until cu121 is green.

- **Baseline run** (conda env `reconstruction`, command per Phase 0):
  `73 failed, 346 passed, 4 skipped, 1 xpassed, 16 errors`, **423 tests collected**.
  Saved to `evals/results/baseline-conda-tests.txt`.
- **Documented baseline** (`docs/known-test-failures.md`, 2026-05-26): `39 failed, 600 passed,
  2 collection errors`, ~644 collected. So ~220 tests now **fail to collect** — whole modules
  error out on import, not new logic failures.
- **Root cause:** missing deps in the only conda env (`reconstruction`) —
  `vggt` (core / VGGT-X), `evo` (eval), `panel` + `param` (dashboard) all `ModuleNotFoundError`.
  Plus the pre-known `nerfstudio.*` namespace shadow (Group 6). `torch 2.5.1+cu121` and
  `nerfstudio` import fine. Likely an **incomplete env build**, not a wrong env — fixes belong in
  `setup.sh` / `setup/feedforward.sh` (vggt) and dashboard/eval extras.
- **Decision:** a separate agent will investigate and fix the failing tests on `refactor/cu121`
  first. This integration resumes only once cu121's baseline is clean (failures == documented
  known set).

**Resume protocol:** when cu121 is green, re-run Phase 0 to refresh
`evals/results/baseline-conda-tests.txt`, then proceed to Phase 1 (rebase). The uv worktree is
untouched at `51e5a93`; no rebase has occurred.

## Goal

Integrate the latest `refactor/cu121` work (code, eval, loop-closure bug fixes) into the
`refactor/cu121-uv-migration` branch, then switch the working environment to the uv-based
Docker image and drive out any environment regressions test-by-test.

## Context

Two branches diverged from common base `585f016`:

- **`refactor/cu121`** (current working branch) — 7 commits since base: eval harness, cross-model
  LC benchmark, loop-closure pose-extraction fixes, docs. No infrastructure changes.
- **`refactor/cu121-uv-migration`** — 10 commits since base: replaces Miniconda with `uv` in the
  `Dockerfile` and all `setup/*.sh` scripts. New env path `/opt/venv/reconstruction` replaces
  `/opt/conda/envs/reconstruction`. `pyproject.toml` is **not** touched (dependencies live in the
  setup scripts / Dockerfile, not in pyproject).

**Zero file overlap** between the two branch diffs — integration is conflict-free.

Branch facts that shape the plan:

- uv branch is **not pushed** to any remote → rebase is safe, no force-push coordination needed.
- uv branch is checked out in worktree `/workspace/collab-splats/.worktrees/uv-migration` → the
  rebase must run inside that worktree (cannot check the branch out twice).
- Active env in the current (conda) container is `reconstruction` (py3.11):
  `/opt/conda/envs/reconstruction/bin/python`.

## Decisions

- **Integration method:** rebase uv onto cu121 (not merge). Conflict-free given zero overlap;
  yields linear history with uv infrastructure sitting on top of the latest code.
- **Baseline:** capture a fresh full-pytest pass/fail list in the current conda env before any
  change, rather than trusting the possibly-stale `docs/known-test-failures.md`.
- **Env-health gate:** the same baseline run doubles as a health check on the freshly-built conda
  env — validate its failures against `docs/known-test-failures.md` before proceeding.

## Phases

### Phase 0 — Confirm conda env healthy + capture baseline (gate)

One pytest run, two purposes:

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -q
```

- **Health check:** diff the failure set against `docs/known-test-failures.md`. Any failure *not*
  listed means the freshly-built conda env is itself broken → stop and fix the env before touching
  the uv branch.
- **Baseline:** save the full pass/fail list (e.g. `evals/results/baseline-conda-tests.txt`,
  gitignored, or a tmp path) as the reference for Phase 3.

**Gate:** proceed only if there are no unexpected failures.

### Phase 1 — Rebase uv onto cu121

```bash
cd /workspace/collab-splats/.worktrees/uv-migration
git rebase refactor/cu121
```

Replays the 10 uv commits on top of the current `refactor/cu121` tip. Expected to apply cleanly
(zero overlap).

**Verify:** `git diff --stat refactor/cu121 HEAD` lists only the uv-infrastructure files
(`Dockerfile`, `setup.sh`, `setup/feedforward.sh`, `setup/hloc.sh`, `setup/vggt_slam.sh`,
`CLAUDE.md`, `evals/download_co3dv2.sh`, `evals/eval_suite.sh`, and the uv-migration plan doc).

### Phase 2 — Switch + rebuild (user-driven)

User builds the new Docker image from the rebased uv branch and restarts the container. The live
environment becomes the uv venv at `/opt/venv/reconstruction`, replacing the conda env.

### Phase 3 — Debug loop (systematic-debugging)

Iterate until no new failures remain:

1. Run `pytest tests/ -q` in the new uv env.
2. Diff the failure set against the Phase-0 baseline. Failures present in the baseline are
   pre-existing and out of scope; **new** failures are uv-caused.
3. For each new failure: root-cause (missing dependency, path drift, torch/CUDA mismatch, etc.),
   fix in the `Dockerfile` or the relevant `setup/*.sh` script, and note whether the fix requires a
   rebuild (Dockerfile change) or is testable in place.
4. Repeat.

**Exit criteria:** the new-failure set is empty (uv env matches the conda baseline modulo known
failures).

## Follow-ups (post-debug)

- Confirm uv commit `6a143de` updated `CLAUDE.md` env paths to `/opt/venv/reconstruction`; correct
  any lingering `nerfstudio`/conda references.
- Update `docs/known-test-failures.md` if any failure persists under uv and is accepted.
- Update auto-memory `feedback_python_env_split` once the uv venv is the live environment.

## Out of Scope

- Touching `pyproject.toml` or restructuring dependency declaration.
- Pushing the uv branch to remote or any PR mechanics.
- Refactoring beyond what is needed to make tests pass under uv.
