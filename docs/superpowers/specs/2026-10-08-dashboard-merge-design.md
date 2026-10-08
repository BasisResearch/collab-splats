# Dashboard merge — land `clean/dashboard` on `clean/final`

Date: 2026-10-08 · Status: approved · Branch: `clean/dashboard` → `clean/final`

## Problem

- dashboard-release ([spec](2026-10-01-dashboard-release-design.md) · [plan](../plans/2026-10-01-dashboard-release.md))
  is built but not on `clean/final`
- two branches carry it:
  - `clean/dashboard-release` `5236786b`: stale; based on `32021ba6`, 70 behind; its ~30 non-dashboard
    commits (semantics store cleanup, ocr-lens viewer, mesh prepare) already reached `clean/final`
    squashed — merging it re-applies them
  - `clean/dashboard` `ce378958`: the 15 dashboard commits replayed onto `clean/final` `4781e72c`
    plus one fix-up; 16 ahead, 23 behind — **the merge source**
- `clean/final` `da6d9d58` gained semantics-storage, decision 023 and the scene viewer since `4781e72c`

## Trial merge (2026-10-08, scratch worktree, discarded)

- textual conflicts: none; only `CLAUDE.md` touched both sides, auto-merged
- dashboard diff confined to `collab_splats/dashboard/`, `tests/dashboard/`, its spec/plan, one
  `README.md` hunk, one notebook line, `docs/source/api/preproc.rst`
- gate on merged tree: `tests/dashboard tests/test_viewer.py tests/reconstructor
  tests/test_import_style.py tests/test_docstring_contract.py` → 1659 passed, 0 failed
- `python -m collab_splats.dashboard --smoke` → `SMOKE PASS`
- APIs the dashboard calls survive: `read_point_features` (new `name=` defaulted),
  `transfer_features`, `Reconstructor.done/outputs`, `SceneSource`
- stale semantics handled: `rec.done("semantics")` is false when `mesh_sha256` differs, so the
  dashboard gets no lifted store and says "run the semantics stage"

## Known gaps, accepted

- dashboard ignores decision 023's stored `vertex_features`; `dashboard/viewer.py` re-derives
  vertex features via `transfer_features` on first query — duplicate work, works; follow-up,
  recorded in the changelog
- two viewers coexist: `collab_splats/viewer.py` (viser, scene viewer) and
  `collab_splats/dashboard/viewer.py` (pyvista, dashboard pane); noted in the changelog

## Decisions (from brainstorm)

- merge source `clean/dashboard`; `clean/dashboard-release` is retired, never merged
- dashboard lands as-is; no vertex-feature rework before the merge
- one squash commit on `clean/final`, matching the other cleanup releases
- GH010229 manual acceptance runs **before** the squash, on the synced branch; fixes fold into the squash
- the main checkout's uncommitted work (`README.md`, `2026-09-09-clean-final-dead-code-design.md`)
  is never stashed, committed or reverted; the squash is built in a scratch worktree
- no push; pushing waits for the user
- stray untracked `docs/source/tutorials/07_localization/ref_image.jpg` removed (not tutorial
  input; the notebook's query is `data/tutorial/tutorial_example-frame.jpg` via `QUERY_IMAGE`)

## Steps

1. **Backups:** `refs/backup/dashboard-release/pre-squash` → `ce378958`;
   `refs/backup/clean-final/pre-dashboard` → `clean/final` tip, recorded as `$BASE`.
2. **Sync:** in `.worktrees/dashboard`, `git merge clean/final` into `clean/dashboard`.
3. **Gate:** in that worktree, `cd <wt> && PYTHONPATH=<wt>` (print `collab_splats.__file__`;
   `third_party/*` symlinked), full `pytest tests/`, exit code checked directly (no `| tail`);
   failures compared against `docs/known-test-failures.md`; dashboard `--smoke`.
4. **GH010229 acceptance** (manual, user drives the browser): load scene, switch backend, run
   one leaf stage, text query in pointcloud and mesh views. Fixes are `fix(dashboard):` commits
   on `clean/dashboard`.
5. **Bookkeeping commit on the branch:** CHANGELOG entry for dashboard-release (gaps above as
   follow-ups); `CLAUDE.md` drops dashboard-release from In-Flight, adds it to Recently Completed.
6. **Squash:** `git commit-tree <clean/dashboard>^{tree} -p $BASE` with message
   `feat(dashboard): run the dashboard over Reconstructor`. Gate: the squash tree equals the
   synced branch tree.
7. **Move `clean/final`** in the main checkout, one bash command, abort on any mismatch:
   - tip check: `clean/final` still `$BASE`; no merge/rebase/cherry-pick in progress
   - `git update-ref refs/heads/clean/final <squash> $BASE`, then `git read-tree HEAD` (index
     = new tree, working files untouched)
   - working files: `git checkout -- <squash paths except README.md>`; README gets only the
     dashboard hunk: `git diff $BASE <squash> -- README.md | git apply`
   - gate: `git status` shows exactly the user's prior uncommitted changes, nothing else
8. **Retire:** remove worktrees `.worktrees/dashboard-release` and `.worktrees/dashboard`, delete
   branches `clean/dashboard-release` and `clean/dashboard`; backup refs kept.

## Out of scope

- dashboard reading stored `vertex_features` (follow-up)
- merging the two viewers
- pushing `clean/final`; `clean/final` → `main`
