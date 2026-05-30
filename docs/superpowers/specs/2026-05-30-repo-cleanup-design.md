# Repo Cleanup: Remove worklog, tools, examples

**Date:** 2026-05-30  
**Status:** Approved

## Goal

Remove three directories (`worklog/`, `tools/`, `examples/`) that no longer serve active purposes, relocate the content that still has value, and update `CLAUDE.md` to reflect the new layout.

## What Goes Where

| Source | Destination | Reason |
|---|---|---|
| `worklog/decisions/` | `docs/decisions/` | 13 ADRs — permanent architectural record |
| `worklog/known-test-failures.md` | `docs/known-test-failures.md` | Living doc; keep as standalone file |
| `worklog/STATE.md` | deleted | Superseded by CLAUDE.md In-Flight + memory |
| `worklog/WORKLOG.md` | deleted | Superseded by git log + memory |
| `worklog/ROADMAP.md` | deleted | Stale; not maintained |
| `worklog/README.md` | deleted | Describes deleted structure |
| `worklog/history/` | deleted | Archived plans; preserved in git history |
| `tools/patch_tutorial_configs.py` | deleted | One-off script; no imports; in git |
| `examples/` (4 scripts + `__pycache__`) | deleted | Scene-specific, no imports, unmaintained |

## CLAUDE.md Changes

- Remove session-start instruction: "read worklog/STATE.md, then WORKLOG.md, then decisions/"
- Update `Known test failures:` path → `docs/known-test-failures.md`
- Remove `docs(worklog):` from commit convention example scope list
- Keep everything else unchanged

## What Is Not Changing

- `docs/superpowers/specs/` and `docs/superpowers/plans/` — active specs/plans stay
- `docs/` module docs — untouched
- All source code, tests, evals — untouched

## Success Criteria

- `worklog/`, `tools/`, `examples/` do not exist in the working tree
- `docs/decisions/` contains all 13 ADRs
- `docs/known-test-failures.md` exists with original content
- CLAUDE.md references no paths under `worklog/`
- `pytest` passes (no regressions from file moves)
