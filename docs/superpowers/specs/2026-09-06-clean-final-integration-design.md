# clean/final — integration branch for the cleanup efforts

**Date:** 2026-09-06
**Status:** design approved, plan pending
**Branch under construction:** `clean/final` (worktree `.worktrees/clean-final`)

## Problem

Five cleanup efforts ran in parallel on their own branches. Each is reviewable in
isolation; none of them is integrated. There is no branch that holds all of the
cleanup work at once, and trunk (`refactor/cu121-uv-migration`) holds a
41-commit interleaving of four of the five efforts that the extraction pass was
meant to untangle.

`clean/final` is the branch where the five land. It is built once, in a fixed
order, one squashed commit per effort.

## Coverage audit

Before designing the merge, every candidate source of work was checked against
the five branches. **Nothing on trunk is uncovered.** This table exists so the
question is not re-litigated.

| Source | Verdict |
|---|---|
| trunk's 41 commits above `6f060dfa` | **41/41 covered.** Every subject has a copy on a `clean/*` branch, or is an original already inside `clean/preproc`. |
| trunk below `6f060dfa` | Shared ancestry — present in all five branches. |
| `preproc/phase-c-staging` (+98) | **Ancestor of `clean/preproc`.** Fully absorbed. |
| `insfm-exec` (+23) | **Superseded.** Its dense-VDA / `depth_scale` chain is already at `6f060dfa` (identical `git grep -c` counts across `collab_splats/`). `trunk → branch` is 2073 insertions / 4315 deletions — the branch is behind trunk. |
| `dense-vda-align` (+10) | Strict subset of `insfm-exec` (`rev-list --left-right --count` = 13/0). Same verdict. |
| `feat/splats-module` (+50) | Behind trunk: 2012 insertions / 5747 deletions. |
| `feat/windowed-streaming` (+43) | Behind trunk: 9414 insertions / 13645 deletions. Last touched 2026-07-22. |
| `feat/mesh-texture-bake` (+24) | Not an ancestor of `clean/mesh`, but `clean/mesh` carries a leaner reimplementation of `texture.py` (−392/+285) plus the nvdiffrast atlas rasterizer. Superseded prototype. |

The only work genuinely outside the five branches is the 11 uncommitted paths in
the main tree (`collab_data[data-dashboard,analysis-dashboard]` extras in
`pyproject.toml`/`uv.lock`, `docs/examples/run_pipeline_remote.py` and its test,
`wrapper/reconstructor.py`). That belongs to another live session and is out of
scope.

## Base

```
clean/final := 5d480452
```

`5d480452` is the trunk commit that `clean/preproc` merged. It carries the three
infra commits and the originals of the 26 pointcloud / 8 semantics / 4 splats
commits.

The choice is load-bearing. `merge-base(5d480452, clean/preproc) == 5d480452`,
so the preproc squash diff is **exactly** preproc's own work — no foreign
originals land inside a commit labelled preproc. Every later squash then nets to
only that effort's new commits, because the base already holds the originals its
branch re-applied as copies.

The two rejected bases:

- `6f060dfa` (the shared fork point) — the preproc squash would also carry the
  32 trunk originals, producing a mislabelled mega-commit.
- `90af13d9` (trunk tip) — its 9 newest commits are pointcloud work whose copies
  live on `clean/pointcloud`. Basing there forces the VDA conflict twice: once
  at the preproc squash, again at the pointcloud squash.

## Merge order

```
5d480452
  ├── squash clean/preproc      ─┐ now
  ├── squash clean/semantics    ─┘
  │   ─────── WAIT for the user ───────
  ├── squash clean/pointcloud   ─┐ on request
  ├── squash clean/splats       ─┘
  └── clean/mesh                   last
```

`clean/mesh` is last because its own plan depends on the siblings. `clean/splats`
and `clean/pointcloud` are still being worked on and are not frozen.

## Disposing of the existing WIP

`clean/final` already exists at `90af13d9` with an abandoned merge of
`clean/preproc` in `.worktrees/clean-final`: 87 dirty paths, `MERGE_HEAD` =
`d46b6fdd`, four conflicts of which two (`pointcloud/sfm.py`,
`wrapper/reconstructor.py`) were hand-resolved and two
(`tests/pointcloud/test_instantsfm.py`, `tests/wrapper/test_sfm_stage.py`) still
carry markers.

Disposal, in order:

1. Copy the two hand-resolved files to the session scratchpad — the same VDA
   collision returns at the pointcloud squash, and the resolutions are a useful
   reference there.
2. `git add -A` and commit the conflicted state as-is, markers included. This
   produces a real merge commit with both parents, reachable and inspectable —
   strictly better than a loose patch file.
3. `git update-ref refs/backup/clean-final-wip-<timestamp> HEAD`.
4. `git reset --hard 5d480452`.

Those resolutions were made against a base being dropped, so they are reference
material, not work to preserve in the history.

## Per-effort recipe

Applies unchanged to all five efforts.

1. **Freeze.** `EXPECTED=$(git rev-parse <branch>)`. Re-check immediately before
   the commit; abort on drift. Branches move: during the design conversation
   alone, `clean/splats` went `385e5235` → `02d9a693`, `clean/pointcloud`
   `7bf784a1` → `5c16ec0b`, `clean/mesh` `c0e34e2c` → `30326769`.
2. **Squash.** Derive the merge base first — it is **the last re-applied copy
   on the branch, not the fork point**. Each extracted branch opens with copies
   of trunk commits `5d480452` already carries; merging from `6f060dfa`
   re-merges every copy against its original. Match by subject (the copies were
   re-applied, so SHAs differ), then use the plumbing, which leaves `HEAD` alone
   exactly like a squash:

   ```bash
   git merge-recursive <last-copy-sha> -- HEAD <branch-tip-sha>
   ```

   Measured on `clean/semantics`: base `4697626e` (8 copies, then 9 branch-only
   commits) gives **2 conflicts**; the plain `git merge --squash` form gives
   **18**, of which 16 are spurious original-vs-copy noise.
3. **Tree proof.** For preproc, `git write-tree` must equal `d46b6fdd^{tree}`
   bit-exact — the merge-base identity guarantees it, so any difference is a bug
   in the procedure. For the others, prove it per-file against the **branch-only
   footprint** `diff(<last-copy>, <branch-tip>)`: (A) the squash changed nothing
   outside that footprint, (B) footprint files the previous efforts never touched
   are bit-identical to the branch, (C) every path outside the footprint still
   equals the previous commit's blob. Do **not** state the proof against
   `diff(5d480452, <branch>)` — the branch forked at `6f060dfa` and so lacks the
   sibling efforts' trunk originals, which appear in that diff as removals;
   enforcing equality there would delete sibling work.
4. **Commit.** The message states what the effort changed and why, drawn from
   that branch's `docs/superpowers/specs/` and `plans/` documents, and ends:

   ```
   Squashed from <branch> @ <SHA> (N commits).
   History: refs/backup/<branch-backup-ref>
   ```

5. **Scoped gate.** Scope is derived, never guessed:
   - every `tests/` path appearing in `git diff --name-only <base>..HEAD`;
   - **plus** callers — grep the changed public symbols across `tests/` and
     `evals/` and add the files that hit. A change to a return contract breaks
     callers the changed module's own tests never touch.

   Run it as:

   ```bash
   cd /workspace/collab-splats/.worktrees/clean-final && \
   PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
   /opt/venv/reconstruction/bin/python -u -c \
     'import collab_splats; print(collab_splats.__file__)' && \
   PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
   /opt/venv/reconstruction/bin/python -u -m pytest <scope>; echo "PYTEST_RC=$?"
   ```

   Compare the failure **set** — not the count — against the same scope run on
   the branch tip. The bar is `NEW failures (0)`, with the skip count also
   matching.

## Landmines

- **`docs/known-test-failures.md` diverged** between the originals and the
  extraction's re-applied copies (`trunk → clean/semantics` is −30 lines;
  `trunk → clean/pointcloud` is +22/−29). It will conflict on the semantics,
  pointcloud and splats squashes. Resolve toward the branch copy, which is the
  version its gates were measured against.
- **A squash records no merge ancestry.** Each branch must be squashed exactly
  once, when frozen. A second squash from the same branch re-diffs from
  `6f060dfa` and double-applies everything.
- **The base already holds each branch's first N commits as trunk originals.**
  This is the whole reason base `5d480452` was chosen, but git cannot see it:
  `merge-base` returns `6f060dfa` because the copies were re-applied with new
  SHAs. The base must be supplied by hand on every squash after preproc.
- **A clean merge does not mean a working tree.** Semantics'
  `extract_feature_cache` called `FrameStore.open()` after preproc deleted
  `FrameStore`. The preproc side already carried the port to
  `images_dir`/`frame_paths`, so the merge resolved it silently and correctly —
  but nothing in the merge would have complained had it not. Grep the merged
  tree for symbols the other effort retired.
- **Worktree PYTHONPATH trap.** A bare `pytest` inside a worktree imports the
  *main* tree's `collab_splats` and reports a false green. `PYTHONPATH` alone is
  insufficient — `sys.path[0]` is the cwd, and it resets between shell
  invocations. Only `cd <worktree> && PYTHONPATH=<worktree> python ...` works,
  and every run must print `collab_splats.__file__` as proof.
- **Fresh-worktree skip inflation.** `third_party/*` is gitignored, so guarded
  tests silently *skip* instead of failing. Symlink the vendored trees in and
  diff the skip count against the branch-tip control, not only the failure set.
- **Stale baselines.** The venv now holds the pinned upstream `gsplat 1.5.3 @
  d2f5c0f` and `gsplat.losses` imports. Every failure baseline recorded before
  2026-09-06 is void — controls must be measured fresh.
- **`clean/mesh` plans to rebase onto `clean/final`, not be squashed into it.**
  That is compatible with this shape (mesh has no copies of trunk originals, so
  its 30 commits replay cleanly onto the squashed tip), but it means mesh
  arrives as a rebase and fast-forward. Settle the choice when mesh is frozen.

## Out of scope

- Replacing trunk. `refactor/cu121-uv-migration` is left alone; `clean/final`
  becoming trunk gets its own spec once all five efforts have landed.
  `trunk-rewrite-candidate` (`6d828023`) stays in place, unused.
- The 11 uncommitted main-tree paths owned by another live session.
- Retiring the ~20 stale `preproc-t*` / `semantics-t*` task branches and the six
  superseded side branches.

## Success criteria

- `clean/final` is based at `5d480452` and carries exactly two commits: the
  preproc squash and the semantics squash.
- The preproc squash's tree equals `d46b6fdd^{tree}` bit-exact.
- The semantics squash changes exactly the files in its branch-only footprint,
  with every semantics-only file bit-identical to the branch tip and every
  preproc path outside the footprint bit-identical to the preproc squash.
- Both scoped gates report `NEW failures (0)` against a freshly measured
  branch-tip control, with matching skip counts and a printed
  `collab_splats.__file__` proof line.
- The abandoned WIP is reachable at `refs/backup/clean-final-wip-<timestamp>`.
- The recipe above is written down well enough that the remaining three efforts
  need no new design work.
