# Worklog Split — Design Spec

**Date:** 2026-05-20
**Branch:** `refactor/core-modules`
**Status:** Implemented (same-day, single commit)
**Supersedes:** [2026-05-20-docs-reorg-design.md](2026-05-20-docs-reorg-design.md)

---

## Goal

Split repo documentation (`docs/`, user-facing) from internal progress tracking (`worklog/`, new top-level). Consolidate scattered READMEs/notes into the proper tree. After this PR:

- `docs/` only contains user-facing module documentation, mirrored to code paths.
- `worklog/` holds STATE, WORKLOG, ROADMAP, specs/, plans/, notes/, decisions/, history/, with a top-level `README.md` describing each subfolder.
- No stray markdown outside those two trees (root `README.md` + `CLAUDE.md` excepted).

---

## Why this supersedes the earlier same-day spec

The earlier spec (`2026-05-20-docs-reorg-design.md`) kept progress tracking under `docs/superpowers/`, `docs/decisions/`, `docs/history/`. User feedback: `docs/` should describe **how to use the repo**; team-facing progress files belong in their own top-level directory. New spec adopts a `worklog/` root.

Parts of the earlier spec that remain valid and were already executed pre-supersession:
- ADR template + ADRs 006–012 written.
- `history/` archive created with renamed slugs.
- WORKLOG entry-template trim.

This PR reuses those artifacts; only the *containing directory* changes.

---

## Target Tree

```
<repo root>
├── docs/                                # user-facing repo documentation
│   ├── README.md                        # index of module docs + how to navigate
│   ├── semantics.md                     # existing top-level write-up
│   ├── nerfstudio/README.md             # from collab_splats/nerfstudio/README.md
│   ├── evals/baselines/vggt_slam/README.md  # from evals/baselines/vggt_slam/README.md
│   ├── pointcloud/                      # existing notebook dir
│   ├── semantics/                       # existing notebook dir
│   ├── splats/                          # existing notebook dir
│   └── data/                            # existing
│
├── worklog/                             # internal progress tracking (NEW)
│   ├── README.md
│   ├── STATE.md
│   ├── WORKLOG.md
│   ├── ROADMAP.md
│   ├── known-test-failures.md
│   ├── specs/    (active design specs)
│   ├── plans/    (active implementation plans)
│   ├── notes/    (investigation notes, dated)
│   ├── decisions/  (ADRs, sequential)
│   └── history/    (specs/, plans/, prs/ — archive)
```

Layout principle for `docs/`: mirror code tree. `<source-path>/<module>/` → `docs/<module>/`. Drop `collab_splats/` prefix (main package implied). Keep top-level dir names (`evals/`).

---

## Inventory + Disposition

| Source | Destination | Notes |
|---|---|---|
| `docs/superpowers/STATE.md` | `worklog/STATE.md` | `git mv` |
| `docs/superpowers/WORKLOG.md` | `worklog/WORKLOG.md` | `git mv` |
| `docs/superpowers/known-test-failures.md` | `worklog/known-test-failures.md` | `git mv` |
| `docs/ROADMAP.md` | `worklog/ROADMAP.md` | `git mv` |
| `docs/superpowers/specs/<active-trio>` (3 files) | `worklog/specs/` | `git mv` |
| `docs/superpowers/plans/<active-trio>` (3 files) | `worklog/plans/` | `git mv` |
| `docs/superpowers/notes/*` (3 files) | `worklog/notes/` | `git mv` |
| `docs/superpowers/specs/2026-05-20-eval-reorganization-design.md` | `worklog/history/specs/` | completed today |
| `docs/superpowers/plans/2026-05-20-eval-reorganization.md` | `worklog/history/plans/` | completed today |
| `docs/superpowers/specs/2026-05-20-docs-reorg-design.md` | `worklog/history/specs/` | superseded by this spec |
| `docs/superpowers/plans/2026-05-20-docs-reorg.md` | `worklog/history/plans/` | superseded |
| `docs/decisions/*` (12 files, ADRs 001–012) | `worklog/decisions/` | `git mv` (whole dir) |
| `docs/history/*` (101 files) | `worklog/history/` | `git mv` (whole dir) |
| `EVAL_NOTES.md` (repo root) | `worklog/notes/2026-05-08-co3dv2-eval-notes.md` | dated rename |
| `archive/REFACTOR.md` (untracked) | `worklog/history/plans/2026-04-16-pointcloud-submodule-refactor.md` | plain `mv`; new addition to git |
| `archive/` (now empty) | (deleted) | `rmdir` |
| `collab_splats/nerfstudio/README.md` | `docs/nerfstudio/README.md` | `git mv` |
| `evals/baselines/vggt_slam/README.md` | `docs/evals/baselines/vggt_slam/README.md` | `git mv` |
| `stage/NOTEBOOK_IMPROVEMENTS.md` (untracked) | (deleted) | `rm` (scratch) |
| `PROGRESS.md`, `REFACTOR.md` (deleted-unstaged at root) | (staged for deletion) | pre-existing deletions folded into this commit |

Blame preservation: every move uses `git mv` for tracked files; untracked files were never in git so blame is N/A.

---

## Cross-Reference Updates

Single mechanical sweep across all `.md` files (excluding `.git`, `.worktrees/`, `node_modules/`, `.pytest_cache/`, `third_party/`):

```
docs/superpowers/STATE.md            → worklog/STATE.md
docs/superpowers/WORKLOG.md          → worklog/WORKLOG.md
docs/superpowers/known-test-failures.md → worklog/known-test-failures.md
docs/superpowers/notes/              → worklog/notes/
docs/superpowers/specs/              → worklog/history/specs/   (most are archived; active-3 patched below)
docs/superpowers/plans/              → worklog/history/plans/   (most archived; active-3 patched)
docs/superpowers/prs/                → worklog/history/prs/
docs/decisions/                      → worklog/decisions/
docs/history/                        → worklog/history/
docs/ROADMAP.md                      → worklog/ROADMAP.md
EVAL_NOTES.md                        → worklog/notes/2026-05-08-co3dv2-eval-notes.md
```

Active-3 patch (after sweep): rewrite `worklog/history/specs/<X>-design.md` → `worklog/specs/<X>-design.md` for the three active topics (`gt-eval-harness`, `feedforward-import-cleanup`, `feedforward-mesh`); same for plans.

Sibling relative-link fix in `worklog/STATE.md` + `worklog/WORKLOG.md`:
- `../decisions/` → `decisions/`
- `../history/` → `history/`
- `../ROADMAP.md` → `ROADMAP.md`

Archived files inside `worklog/history/` retain whatever relative paths they had — they are immutable historical record. Path strings that survived the sweep still resolve (the depth shift from `docs/superpowers/specs/` → `worklog/history/specs/` is the same number of levels up).

---

## `CLAUDE.md` Update

Replace lines 7–9 to point at `worklog/` and add a `docs/` note:

```
- Before anything, read worklog/STATE.md (current state), then latest entries in worklog/WORKLOG.md, then worklog/decisions/NNN-*.md as referenced.
- Active spec/plan for in-flight work lives in worklog/{specs,plans}/. Completed work is archived to worklog/history/.
- Architecture decisions: worklog/decisions/NNN-slug.md (sequential numbering).
- User-facing module docs live in docs/ (mirrors code tree).
```

---

## Verification

```bash
# 1. No dangling refs to old paths in non-archive files.
grep -rn -E '(docs/superpowers/|docs/decisions/|docs/history/|docs/ROADMAP\.md|EVAL_NOTES\.md|archive/REFACTOR\.md|stage/NOTEBOOK_IMPROVEMENTS\.md)' \
  --include='*.md' --include='*.py' --include='*.toml' \
  --exclude-dir=.git --exclude-dir=.worktrees --exclude-dir=node_modules \
  --exclude-dir=.pytest_cache --exclude-dir=third_party --exclude-dir=worklog/history .

# 2. Tree present.
test -f worklog/README.md && test -f worklog/STATE.md && test -f worklog/WORKLOG.md \
  && test -f worklog/ROADMAP.md && test -d worklog/decisions && test -d worklog/history \
  && test -f docs/README.md && test -f docs/nerfstudio/README.md \
  && test -f docs/evals/baselines/vggt_slam/README.md

# 3. Stray markdown gone.
test ! -e EVAL_NOTES.md && test ! -e archive && test ! -e stage/NOTEBOOK_IMPROVEMENTS.md \
  && test ! -e collab_splats/nerfstudio/README.md \
  && test ! -e evals/baselines/vggt_slam/README.md

# 4. Blame preserved on 3 sample files.
git log --follow --oneline worklog/history/specs/2026-04-16-pointcloud-submodule-design.md | head -3
git log --follow --oneline worklog/decisions/002-defer-sl4.md | head -3
git log --follow --oneline worklog/notes/2026-05-08-co3dv2-eval-gap-investigation.md | head -3
```

---

## Out of Scope

- Reorg of `docs/{pointcloud,semantics,splats,data}/` notebook trees.
- Reorg of `evals/`, `eval_results/`, `tests/`.
- New ADRs.
- Editing past WORKLOG entries or archived specs (beyond this supersession header on the predecessor).
- Code changes.
- `.pytest_cache/README.md` (auto-generated).
- Worktree contents under `.worktrees/`.
