# Dashboard Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the dashboard-release work (`clean/dashboard`) on `clean/final` as one squash commit, leaving the main checkout's uncommitted work untouched.

**Architecture:** Sync `clean/dashboard` with `clean/final`, gate it (full pytest, smoke, manual GH010229), add bookkeeping, then build the squash with `git commit-tree` from the branch tree and move `clean/final` by a tip-checked `update-ref` + `read-tree` that never touches the user's dirty files. Spec: [2026-10-08-dashboard-merge-design.md](../specs/2026-10-08-dashboard-merge-design.md).

**Tech Stack:** git 2.34 (no `merge-tree --write-tree`), pytest in `/opt/venv/reconstruction`, Panel dashboard.

---

## Ground rules (read before any task)

- Main checkout: `/workspace/collab-splats`, branch `clean/final`. It has the **user's uncommitted
  changes** (`README.md`, `docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md`).
  Never `stash`, `checkout`, `reset --hard`, `add -A` or commit there except where a task says so.
- Dashboard worktree: `/workspace/collab-splats/.worktrees/dashboard`, branch `clean/dashboard`.
- Other sessions may commit concurrently. Commit with `git commit --only <paths>`; never `--amend`
  a commit you did not just make.
- Tests in a worktree only test the worktree when run as `cd <wt> && PYTHONPATH=<wt> python ...`;
  print `collab_splats.__file__` to prove it.
- Never pipe pytest into `tail`/`head` when the exit code matters; write to a log, check `$?`.
- Scratch dir: `SP=/tmp/claude-0/-workspace-collab-splats/853dbffe-6f2b-47b1-a775-29c728120cc6/scratchpad`
  (use the current session's scratchpad if different).
- Stop and report to the user on any unexpected conflict, test failure or gate mismatch.

---

### Task 1: Record base and create backup refs

**Files:** none (refs only)

- [ ] **Step 1: Record `clean/final` tip as BASE and back up both tips**

```bash
cd /workspace/collab-splats
BASE=$(git rev-parse clean/final); echo "$BASE" > $SP/BASE
DASH=$(git rev-parse clean/dashboard); echo "$DASH" > $SP/DASH_PRE
git update-ref refs/backup/clean-final/pre-dashboard "$BASE"
git update-ref refs/backup/dashboard-release/pre-squash "$DASH"
git for-each-ref refs/backup/clean-final refs/backup/dashboard-release
```

Expected: `pre-dashboard` = BASE (at plan time `e2e31963` or later), `pre-squash` = `ce378958`.
The pre-existing `refs/backup/dashboard-release/pre-cf-rebase` is also listed; keep it.

### Task 2: Sync `clean/dashboard` with `clean/final`

**Files:** merge commit on `clean/dashboard`; expected auto-merged: `CLAUDE.md`

- [ ] **Step 1: Confirm the worktree is clean**

```bash
cd /workspace/collab-splats/.worktrees/dashboard && git status --short && git rev-parse HEAD
```

Expected: no output from status; HEAD = `$(cat $SP/DASH_PRE)`.

- [ ] **Step 2: Merge**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
git merge --no-edit "$(cat $SP/BASE)" -m "merge: sync clean/dashboard with clean/final"
git diff --name-only --diff-filter=U
```

Expected: `Auto-merging CLAUDE.md`, no conflicted files (trial merge on 2026-10-08 had none).
On any conflict: `git merge --abort` and stop.

- [ ] **Step 3: Check CLAUDE.md kept both sides**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
grep -n "dashboard-release\|scene-viewer" CLAUDE.md
```

Expected: the In-Flight `dashboard-release` line and `- **scene-viewer** (2026-10-08)` both present.

### Task 3: Full gate on the synced branch

**Files:** none

- [ ] **Step 1: Prove the worktree package is under test**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
ls -l third_party/   # LoGeR, Video-Depth-Anything, hloc symlinks present
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats, gsplat; print(collab_splats.__file__, gsplat.__version__)"
```

Expected: path under `.worktrees/dashboard/`; note the gsplat version (shared venv flips 1.4.0/1.5.3 — if it changed from the last session, see memory `project_shared_venv_gsplat_downgrade` before trusting splats results).

- [ ] **Step 2: Full pytest (long; run with run_in_background or in tmux)**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
PYTHONPATH=$PWD timeout 3600 /opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:cacheprovider > $SP/gate_full.log 2>&1; echo "exit=$?" >> $SP/gate_full.log
tail -5 $SP/gate_full.log
grep -E "^(FAILED|ERROR)" $SP/gate_full.log
```

Expected: `exit=0`, or every FAILED/ERROR line listed as unresolved in `docs/known-test-failures.md`.
Any other failure: stop and report.

- [ ] **Step 3: Dashboard smoke**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
PYTHONPATH=$PWD timeout 300 /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke --port 7871; echo "exit=$?"
```

Expected: `SMOKE PASS: page (... B) + bokeh.min.js (... B) served`, `exit=0`.

### Task 4: GH010229 manual acceptance (user drives)

**Files:** none unless a fix is needed (then `collab_splats/dashboard/*`, `tests/dashboard/*`)

- [ ] **Step 1: Launch the dashboard from the worktree in tmux**

```bash
tmux new-session -d -s dash-accept "cd /workspace/collab-splats/.worktrees/dashboard && PYTHONPATH=\$PWD /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --port 7860 --base-dir /workspace/outputs 2>&1 | tee $SP/dash_accept.log"
sleep 20; grep -i "error\|traceback" $SP/dash_accept.log; echo "check http://<host>:7860"
```

- [ ] **Step 2: Ask the user to run the checklist and report each item pass/fail**

1. scene GH010229 loads; viewer shows the pointcloud
2. backend dropdown switch reloads the other backend's outputs (or says "not run — press Run")
3. Run with one leaf stage selected (e.g. `mesh`) completes; operation log shows the step
4. text query in pointcloud view colors points
5. switch to mesh view; same query colors the mesh

- [ ] **Step 3: On failure — fix on the branch**

Reproduce in a test under `tests/dashboard/` first (failing), fix in `collab_splats/dashboard/`,
re-run `tests/dashboard` + smoke (Task 3 Step 3), commit:

```bash
cd /workspace/collab-splats/.worktrees/dashboard
git add <files> && git commit -m "fix(dashboard): <what>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Repeat Step 2 for the failed item. Report-before-change rule applies: describe the bug and the
proposed fix to the user and wait for go-ahead before editing.

- [ ] **Step 4: Stop the dashboard**

```bash
tmux kill-session -t dash-accept
```

### Task 5: Bookkeeping commit on the branch

**Files:**
- Modify: `.worktrees/dashboard/docs/superpowers/CHANGELOG.md` (new entry after line 4)
- Modify: `.worktrees/dashboard/CLAUDE.md` (In-Flight + Recently Completed)

- [ ] **Step 1: Add the changelog entry** — insert as the first entry (line 6, after the intro paragraph and blank line):

```markdown
Recently completed (2026-10-08): **dashboard-release** — single-page dashboard run through `Reconstructor` (backend dropdown over feedforward + sfm creators, stage selection, YAML overrides recorded into `<backend>/run_config.yaml`, force re-run); `dashboard/pipeline.py`, `config.py`, `shell.py`, `localize.py` and the localization tab deleted; viewer reads the lifted store via `semantics.store.read_point_features`. Squashed onto `clean/final` from `clean/dashboard` (replay of `clean/dashboard-release` onto `4781e72c`, tip `<DASH_TIP>`; backups `refs/backup/dashboard-release/pre-squash`, `pre-cf-rebase`); `clean/dashboard-release` retired unmerged ([spec](specs/2026-10-01-dashboard-release-design.md) · [plan](plans/2026-10-01-dashboard-release.md) · [merge spec](specs/2026-10-08-dashboard-merge-design.md) · [merge plan](plans/2026-10-08-dashboard-merge.md)). **Gates.** Full pytest `<N passed / known failures>`; `--smoke` PASS; GH010229 manual acceptance `<result>`. **Follow-ups.** Dashboard ignores decision 023's stored `vertex_features` and re-derives vertex features with `transfer_features` on first query (`dashboard/viewer.py`) — switch to reading the store. Two viewers coexist: `collab_splats/viewer.py` (viser scene viewer) and `collab_splats/dashboard/viewer.py` (pyvista dashboard pane).
```

Fill `<DASH_TIP>` with `git rev-parse --short HEAD` of the worktree before this commit, and the
gate placeholders with the real Task 3/4 results (these are run-time values, not plan gaps).

- [ ] **Step 2: Update CLAUDE.md** — delete the In-Flight line starting `- **dashboard-release** —`;
in Recently Completed replace the five-newest list with:

```markdown
- **dashboard-release** (2026-10-08)
- **scene-viewer** (2026-10-08)
- **semantics-storage** (2026-10-07)
- **mesh-query-heat** (2026-10-07)
- **rgbd-ba + lc-window-ba** (2026-10-06)
```

- [ ] **Step 3: Verify and commit**

```bash
cd /workspace/collab-splats/.worktrees/dashboard
grep -c "dashboard-release" CLAUDE.md          # expect 1 (Recently Completed only)
wc -c CLAUDE.md                                # expect well under 40000
git add -f docs/superpowers/CHANGELOG.md CLAUDE.md
git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md -m "docs(changelog): dashboard-release

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(The merge spec and this plan are already committed on `clean/final`, so they reach the branch via
the Task 2 sync.)

### Task 6: Build the squash commit

**Files:** none (object only)

- [ ] **Step 1: Re-check `clean/final` has not moved since Task 1**

```bash
cd /workspace/collab-splats
[ "$(git rev-parse clean/final)" = "$(cat $SP/BASE)" ] && echo SAME || echo MOVED
```

If MOVED: in the dashboard worktree `git merge --no-edit clean/final`, rerun Task 3 Step 3 (smoke)
and `tests/dashboard`, write the new tip to `$SP/BASE`, and continue.

- [ ] **Step 2: commit-tree the squash**

```bash
cd /workspace/collab-splats
BASE=$(cat $SP/BASE); TIP=$(git rev-parse clean/dashboard)
cat > $SP/squash_msg <<'EOF'
feat(dashboard): run the dashboard over Reconstructor

- one page: scene + backend dropdown (feedforward and sfm creators), stage selection,
  YAML overrides recorded into <backend>/run_config.yaml, force re-run
- loading reads Reconstructor outputs; pulls scoped to the chosen backend
- viewer reads the lifted store via semantics.store.read_point_features
- deleted: dashboard pipeline/config/shell/localize modules and the localization tab
- squashed from clean/dashboard (backup refs/backup/dashboard-release/pre-squash)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
SQ=$(git commit-tree "$TIP^{tree}" -p "$BASE" -F $SP/squash_msg); echo "$SQ" > $SP/SQUASH
[ "$(git rev-parse $SQ^{tree})" = "$(git rev-parse $TIP^{tree})" ] && echo TREE-OK || echo TREE-MISMATCH
git diff --stat "$BASE" "$SQ" | tail -3
```

Expected: `TREE-OK`; stat touches only dashboard code/tests, its spec/plan, CHANGELOG, CLAUDE.md,
README.md, `docs/source/api/preproc.rst`, the keyframe notebook.

- [ ] **Step 3: Confirm the squash does not touch the user's dirty spec file**

```bash
git diff --name-only "$(cat $SP/BASE)" "$(cat $SP/SQUASH)" | grep -x "docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md" && echo CLASH || echo NO-CLASH
```

Expected: `NO-CLASH`. On CLASH: stop and ask the user.

### Task 7: Move `clean/final` without touching uncommitted work

**Files:** main checkout working tree (only squash paths; `README.md` by patch)

- [ ] **Step 1: Snapshot the user's uncommitted state**

```bash
cd /workspace/collab-splats
git status --short > $SP/status_before
git diff > $SP/user_wip.patch
sha256sum README.md docs/superpowers/specs/2026-09-09-clean-final-dead-code-design.md > $SP/wip_sha_before
cat $SP/status_before
```

- [ ] **Step 2: Move the ref, update index and files — ONE command, aborts on any check**

```bash
cd /workspace/collab-splats
BASE=$(cat $SP/BASE); SQ=$(cat $SP/SQUASH); G=$(git rev-parse --git-dir)
set -e
[ "$(git symbolic-ref HEAD)" = "refs/heads/clean/final" ]
[ "$(git rev-parse HEAD)" = "$BASE" ]
[ ! -e "$G/MERGE_HEAD" ] && [ ! -e "$G/CHERRY_PICK_HEAD" ] && [ ! -d "$G/rebase-merge" ] && [ ! -d "$G/rebase-apply" ]
git update-ref refs/heads/clean/final "$SQ" "$BASE"
git read-tree HEAD
git diff --name-only "$BASE" "$SQ" | grep -vx README.md > $SP/sq_paths
git diff --name-only --diff-filter=D "$BASE" "$SQ" > $SP/sq_deleted
xargs -a $SP/sq_deleted -r rm -f --
grep -vxF -f $SP/sq_deleted $SP/sq_paths | xargs -r git checkout --
git diff "$BASE" "$SQ" -- README.md | git apply
set +e; echo MOVED-OK
```

Notes: `update-ref` with old value refuses if another session moved the branch; `read-tree HEAD`
sets the index to the squash without touching files; deleted dashboard files are removed by hand
because `checkout --` cannot restore a path absent from the index; README gets only the dashboard
hunk (user's hunk is at the top, dashboard's at ~line 148 — trial showed no overlap).

If `git apply` fails: run `git diff $BASE $SQ -- README.md > $SP/readme_dash.patch` and stop — tell
the user; their README WIP is untouched and the patch can be applied after they commit.

- [ ] **Step 3: Gate — only the user's prior changes remain**

```bash
cd /workspace/collab-splats
git log --oneline -2
git status --short > $SP/status_after; diff $SP/status_before $SP/status_after && echo STATUS-SAME
sha256sum -c $SP/wip_sha_before 2>&1 | grep dead-code
git diff -- README.md | head -40
```

Expected: HEAD = squash, parent BASE; `STATUS-SAME`; dead-code spec `OK`; README diff shows only
the user's top-of-file hunk (README.md checksum changes because the dashboard hunk was applied —
that is expected; it must be the only difference from `git show HEAD:README.md` besides the user hunk).

- [ ] **Step 4: Post-move smoke from the main checkout**

```bash
cd /workspace/collab-splats
/opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
timeout 300 /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke --port 7872; echo "exit=$?"
/opt/venv/reconstruction/bin/python -m pytest tests/dashboard -q -p no:cacheprovider > $SP/post.log 2>&1; echo "exit=$?"; tail -2 $SP/post.log
```

Expected: `SMOKE PASS`, pytest `exit=0`.

### Task 8: Retire the old branches and worktrees

**Files:** none (worktrees, branches)

- [ ] **Step 1: Confirm both worktrees are clean and backups exist**

```bash
cd /workspace/collab-splats
git -C .worktrees/dashboard status --short; git -C .worktrees/dashboard-release status --short
git for-each-ref refs/backup/dashboard-release
```

Expected: no status output; `pre-squash` and `pre-cf-rebase` listed. Also back up the final
dashboard tip if Task 4/5 added commits: `git update-ref refs/backup/dashboard-release/final $(git rev-parse clean/dashboard)`.

- [ ] **Step 2: Ask the user before deleting** (deleting branches/worktrees is hard to undo beyond the backup refs). On yes:

```bash
cd /workspace/collab-splats
git worktree remove .worktrees/dashboard-release
git worktree remove .worktrees/dashboard
git branch -D clean/dashboard-release clean/dashboard
git worktree list; git branch --list 'clean/dashboard*'
```

Expected: neither worktree nor branch listed.

- [ ] **Step 3: Update memory** — rewrite `project_dashboard_release.md`: landed on `clean/final`
as `<SQUASH short sha>` 2026-10-08, not pushed, branches retired, backups
`refs/backup/dashboard-release/{pre-squash,pre-cf-rebase,final}`, follow-up = read stored
`vertex_features`. Update its `MEMORY.md` line.

- [ ] **Step 4: Run `graphify update .`** in the main checkout (CLAUDE.md rule after code changes).

---

## Not in this plan

- pushing `clean/final` (waits for the user)
- dashboard reading stored `vertex_features`; merging the two viewers
