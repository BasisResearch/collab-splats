# clean/final Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `clean/final` at base `5d480452` carrying one squashed commit per finished cleanup effort — `clean/preproc` then `clean/semantics` — with a bit-exact tree proof and a `NEW failures (0)` gate on each.

**Architecture:** `clean/final` is reset to `5d480452` (the trunk commit `clean/preproc` merged, so `merge-base(base, clean/preproc) == base` and the preproc squash diff is exactly preproc's own work). Each effort lands via `git merge --squash`, proved by comparing blobs against the source branch rather than by a scratch re-merge, and gated by a fixed pytest scope derived from the union of both branches' blast radii.

**Tech Stack:** git 2.34.1 (no `merge-tree --write-tree`), pytest, `/opt/venv/reconstruction/bin/python` (py3.11).

**Spec:** `docs/superpowers/specs/2026-09-06-clean-final-integration-design.md`

---

## Constants used throughout

```bash
WT=/workspace/collab-splats/.worktrees/clean-final
MAIN=/workspace/collab-splats
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
BASE=5d480452
PREPROC_SHA=d46b6fdd      # clean/preproc, frozen
SEM_SHA=a2002db0          # clean/semantics, frozen
```

**The fixed pytest scope** (used identically for the control run and the gate run — it must not change between them):

```
tests/dashboard tests/docs tests/evals tests/geometry tests/integration
tests/mesh tests/pointcloud tests/preproc tests/remote tests/semantics
tests/utils tests/wrapper
```

Excluded: `tests/examples`, `tests/localization`, `tests/nerfstudio_methods`,
`tests/scripts`, `tests/splats` — neither branch touches any file under them.
The only `tests/conftest.py` change is purely additive (a new `tiny_video`
session fixture plus a blank line), so it cannot perturb the excluded dirs.

---

## File Structure

No source files are created or modified by this plan. The artifacts are git refs
and two commits:

| Artifact | Responsibility |
|---|---|
| `refs/backup/clean-final-wip-<ts>` | Reachable archive of the abandoned merge |
| `$SCRATCH/vda-resolution-ref/` | The two hand-resolved files, reference material for the later pointcloud squash |
| `clean/final` commit 1 | `clean/preproc` squashed |
| `clean/final` commit 2 | `clean/semantics` squashed |
| `$SCRATCH/control-preproc.txt` | Failure set + skip count after commit 1 — the control for commit 2 |
| `$SCRATCH/gate-semantics.txt` | Failure set + skip count after commit 2 |

---

## Task 1: Archive the abandoned WIP and reset clean/final

**Files:**
- Modify: `/workspace/collab-splats/.worktrees/clean-final` (working tree, hard reset)
- Create: `$SCRATCH/vda-resolution-ref/sfm.py`, `$SCRATCH/vda-resolution-ref/reconstructor.py`

- [ ] **Step 1: Verify the preconditions**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse HEAD
git rev-parse --short clean/preproc
cat "$(git rev-parse --git-dir)/MERGE_HEAD"
git status --porcelain | grep -cE '^(UU|AA|DD|AU|UA|DU|UD)'
```

Expected exactly:
- `HEAD` → `90af13d9...`
- `clean/preproc` → `d46b6fdd`
- `MERGE_HEAD` → `d46b6fddae07c88432abd6f5232dc6f548c186c6`
- conflict count → `4`

If any differs, STOP and report. Do not reset over a state you did not verify.

- [ ] **Step 2: Copy the two hand-resolved files to the scratchpad**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
mkdir -p $SCRATCH/vda-resolution-ref
cd /workspace/collab-splats/.worktrees/clean-final
cp collab_splats/pointcloud/sfm.py        $SCRATCH/vda-resolution-ref/sfm.py
cp collab_splats/wrapper/reconstructor.py $SCRATCH/vda-resolution-ref/reconstructor.py
grep -c '^<<<<<<< ' $SCRATCH/vda-resolution-ref/*.py
```

Expected: `0` for both files (they were hand-resolved; the markers are gone).

Write `$SCRATCH/vda-resolution-ref/README.md` with this exact content:

```markdown
# VDA merge resolution reference

Salvaged 2026-09-06 from the abandoned `clean/final <- clean/preproc` merge that
was made against base `90af13d9` (trunk tip). That base was dropped in favour of
`5d480452`, so these resolutions are NOT applied to clean/final.

They resolve the collision between `clean/preproc`'s deletion of the VDA context
stream and the `vda.py` extraction. The same collision returns when
`clean/pointcloud` is squashed (its copies are `6259adf1`, `dd1be9f2`,
`172f9e17`). Consult these files then, but re-derive the resolution against the
tree that actually exists at that point.

Full abandoned state: refs/backup/clean-final-wip-<timestamp>
```

- [ ] **Step 3: Commit the conflicted state as-is onto a backup ref**

Committing the markers is intentional — it produces a real merge commit with
both parents, reachable and inspectable, which a loose patch file is not.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git add -A
git commit --no-verify -m "wip: abandoned clean/final <- clean/preproc merge (archived)

Merge attempted against base 90af13d9 (trunk tip), a base later rejected in
favour of 5d480452. Two of four conflicts were hand-resolved
(collab_splats/pointcloud/sfm.py, collab_splats/wrapper/reconstructor.py);
tests/pointcloud/test_instantsfm.py and tests/wrapper/test_sfm_stage.py still
carry conflict markers. Archived rather than discarded so the VDA resolutions
stay reachable for the clean/pointcloud squash.

Not for merging. Conflict markers are committed on purpose."
TS=$(date +%Y%m%d-%H%M%S)
git update-ref refs/backup/clean-final-wip-$TS HEAD
git for-each-ref refs/backup/clean-final-wip-$TS --format='%(refname) -> %(objectname:short)'
```

Expected: one line printing the new ref and a short SHA.

- [ ] **Step 4: Reset clean/final to the base**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git reset --hard 5d480452
git rev-parse HEAD
git status --porcelain
```

Expected: `HEAD` → `5d480452...`; `git status --porcelain` prints nothing.

- [ ] **Step 5: Verify the archive survived the reset**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git log --oneline -1 $(git for-each-ref refs/backup/ --format='%(refname)' | grep clean-final-wip | head -1)
```

Expected: the `wip: abandoned clean/final <- clean/preproc merge (archived)` subject.

- [ ] **Step 6: Commit** — nothing to commit; Task 1 produces refs, not tracked
      changes. Confirm with `git status --porcelain` printing nothing.

---

## Task 2: Restore third_party in the worktree so the gate cannot false-green

`third_party/*` is gitignored, so a fresh worktree has none of the vendored
trees. Guarded tests then **skip** instead of failing, and the gate reports a
green that is really a smaller test run.

**Files:**
- Create: 9 symlinks + 1 file link under `/workspace/collab-splats/.worktrees/clean-final/third_party/`

- [ ] **Step 1: Confirm the gap**

```bash
ls /workspace/collab-splats/third_party/
ls -a /workspace/collab-splats/.worktrees/clean-final/third_party/
```

Expected: main tree lists `LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc
vggt-omega vggt_spark xfeat` plus `.vda_fetch_done`; the worktree lists only
`README.md` (tracked).

- [ ] **Step 2: Symlink each vendored tree in**

```bash
cd /workspace/collab-splats/.worktrees/clean-final/third_party
for d in LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat .vda_fetch_done; do
  [ -e "$d" ] || ln -s "/workspace/collab-splats/third_party/$d" "$d"
done
ls -la
```

Expected: ten symlinks pointing into `/workspace/collab-splats/third_party/`,
plus the tracked `README.md`.

- [ ] **Step 3: Verify the symlinks did not dirty the tree**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git status --porcelain
```

Expected: prints nothing. If a symlink shows as untracked, it is not covered by
`.gitignore` — stop and report rather than committing it.

- [ ] **Step 4: Prove the interpreter resolves to the worktree**

```bash
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -c 'import collab_splats; print(collab_splats.__file__)'
```

Expected: a path under `/workspace/collab-splats/.worktrees/clean-final/`.
If it prints `/workspace/collab-splats/collab_splats/...` the venv's editable
finder won the race — STOP, every gate from here would test the wrong tree.

- [ ] **Step 5: Commit** — nothing to commit (all links are ignored). Confirm
      `git status --porcelain` prints nothing.

---

## Task 3: Squash clean/preproc into clean/final

**Files:**
- Modify: `clean/final` (one new commit)

- [ ] **Step 1: Freeze the source branch**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/preproc
git rev-parse HEAD
git merge-base 5d480452 clean/preproc
```

Expected:
- `clean/preproc` → `d46b6fddae07c88432abd6f5232dc6f548c186c6`
- `HEAD` → `5d480452...`
- `merge-base` → `5d480452...` — **this identity is what makes the squash exact.**

If `clean/preproc` has moved off `d46b6fdd`, STOP and report; the plan's tree
proof is written against that SHA.

- [ ] **Step 2: Squash-merge**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git merge --squash clean/preproc
git status --porcelain | grep -cE '^(UU|AA|DD|AU|UA|DU|UD)' || true
```

Expected: conflict count `0`. Because the merge base *is* HEAD, this is a
fast-forward-shaped squash and cannot conflict. Any conflict means the
preconditions in Step 1 were not actually met — STOP.

- [ ] **Step 3: Tree proof — bit-exact**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
echo "squashed: $(git write-tree)"
echo "branch:   $(git rev-parse clean/preproc^{tree})"
test "$(git write-tree)" = "$(git rev-parse clean/preproc^{tree})" && echo "TREE PROOF: PASS" || echo "TREE PROOF: FAIL"
```

Expected: `TREE PROOF: PASS`, with both hashes printed and identical.
A `FAIL` here means the squash did not reproduce preproc's tree — STOP, do not
commit, report the differing paths via
`git diff --name-only clean/preproc -- .`

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git commit --no-verify -m "refactor(preproc): centralize video preprocessing into measure, filter, sample

Squashes the preproc centralization effort. What changed and why:

- preproc is now three ordered concerns: measure (qa.py) reports capture
  quality, filter drops bad frames, sample selects keyframes. The old
  interleaving made it impossible to say whether a frame was dropped for
  quality or simply not selected.
- frames.zarr is retired; the keyframe store is images/ + frames.json again.
  The zarr store bought nothing over a directory of JPEGs and cost a decode
  round-trip. No backfill — existing scenes re-decode.
- Blur filtering moved from a static threshold to a robust per-video MAD
  z-score. The static threshold cut exactly zero frames on real footage; the
  z-score cuts 5.11% of GH010229 and 14.53% of the tutorial video.
- The VDA context stream and decode_context are deleted. Both were dead paths
  kept for a streaming design that was never built.
- Samplers no longer take candidates=; callers pass the frame list directly.
- undistort_frames returns a LARGER image and a NEW camera. Callers must use
  the returned camera, not the one they passed in.
- Decode is PyAV. Seek-per-index measured SLOWER than a linear scan
  (97.4s vs 76.2s), so decode walks the stream once.
- Docstring lint anchors on ^\\s*param: rather than matching a substring, which
  was flagging prose.

Squashed from clean/preproc @ d46b6fdd (122 commits).
History: refs/backup/clean-preproc-20260906-013531"
git log --oneline -2 | cat
```

Expected: the new commit on top of `5d480452`.

- [ ] **Step 5: Re-verify the frozen SHA after committing**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/preproc
test "$(git rev-parse clean/preproc)" = "d46b6fddae07c88432abd6f5232dc6f548c186c6" \
  && echo "FREEZE HELD" || echo "FREEZE BROKEN — branch moved during the squash"
```

Expected: `FREEZE HELD`. If broken, the commit describes a SHA that is no longer
the tip — amend the message with the new SHA and re-run the tree proof.

---

## Task 4: Establish the control baseline

The preproc squash needs no A/B comparison — its tree is bit-identical to
`clean/preproc`, which was already gated on its own branch. This run exists to
produce the control that the **semantics** squash is measured against.

**Files:**
- Create: `$SCRATCH/control-preproc.txt`

- [ ] **Step 1: Run the fixed scope in the background**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -c 'import collab_splats; print("IMPORT:", collab_splats.__file__)' \
  > $SCRATCH/control-preproc.txt 2>&1 && \
echo "TREE: $(git rev-parse HEAD)" >> $SCRATCH/control-preproc.txt && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -m pytest -q \
  tests/dashboard tests/docs tests/evals tests/geometry tests/integration \
  tests/mesh tests/pointcloud tests/preproc tests/remote tests/semantics \
  tests/utils tests/wrapper \
  >> $SCRATCH/control-preproc.txt 2>&1; \
echo "PYTEST_RC=$?" >> $SCRATCH/control-preproc.txt
```

Run this with `run_in_background: true`. The trailing `echo` is what makes the
exit code recoverable — a background task's notification reports the exit code
of the LAST command, not pytest's.

- [ ] **Step 2: Wait for it, then confirm it actually ran**

Poll by file growth, not by `ps` or `stat`:

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
wc -c $SCRATCH/control-preproc.txt; tail -3 $SCRATCH/control-preproc.txt
```

Expected once finished: the last lines carry a pytest summary and `PYTEST_RC=<n>`.
A 0-byte file is not proof of a dead writer — `-u` is already set, so growth is
the signal.

- [ ] **Step 3: Verify the import proof line**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
head -2 $SCRATCH/control-preproc.txt
```

Expected: `IMPORT: /workspace/collab-splats/.worktrees/clean-final/collab_splats/__init__.py`
and a `TREE:` line matching the Task 3 commit. If `IMPORT:` points at the main
tree, this control is worthless — fix the invocation and re-run.

- [ ] **Step 4: Extract the failure set and skip count**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
grep -E '^(FAILED|ERROR) ' $SCRATCH/control-preproc.txt | sort > $SCRATCH/control-preproc.set
wc -l < $SCRATCH/control-preproc.set
grep -oE '[0-9]+ skipped' $SCRATCH/control-preproc.txt | tail -1
tail -1 $SCRATCH/control-preproc.txt
```

Record all three numbers in the plan as the control. The **set** is what
matters; the counts are a cross-check.

- [ ] **Step 5: Commit** — nothing to commit; the artifacts live in the
      scratchpad. Confirm `git status --porcelain` prints nothing.

---

## Task 5: Squash clean/semantics into clean/final

> **CORRECTED DURING EXECUTION.** This task originally named a 15-file overlap
> computed as `diff(5d480452, clean/preproc)` ∩ `diff(5d480452, clean/semantics)`.
> That list does not predict anything, because **`5d480452` is not the merge base
> git uses.** `clean/semantics` was rebased onto `6f060dfa` during the extraction,
> so `5d480452` is not its ancestor and `merge-base(HEAD, clean/semantics)`
> resolves to `6f060dfa`. A plain `git merge --squash clean/semantics` therefore
> re-merges the eight re-applied **copies** against the trunk **originals** the
> base already holds: measured, **18 conflicts**, 13 of them on paths outside the
> predicted list. The section below is the corrected procedure. It applies
> unchanged to `clean/pointcloud` and `clean/splats`, which were extracted the
> same way.

### The base is the last re-applied copy, not the fork point

Each extracted branch begins with copies of trunk commits the base already
carries, followed by the commits that are genuinely branch-only. Merging from the
fork point double-counts the copies. Merging from **the last copy** nets exactly
the branch-only work.

Find the boundary by subject (the copies are re-applied, so SHAs differ):

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git log --format='%s' 6f060dfa..5d480452 | sort -u > /tmp/trunk-subjects.txt
git log --format='%h %s' --reverse 6f060dfa..<branch> | while read -r h rest; do
  grep -qxF "$rest" /tmp/trunk-subjects.txt \
    && echo "COPY-OF-TRUNK  $h $rest" || echo "BRANCH-ONLY    $h $rest"
done
```

The last `COPY-OF-TRUNK` line is the merge base. For `clean/semantics` that is
`4697626e` — 8 copies, then 9 branch-only commits (`ed3faf8d..a2002db0`).

Merge with that base explicitly. `git merge --squash` cannot take one, so use the
plumbing, which leaves `HEAD` alone exactly like a squash and writes no
`MERGE_HEAD`:

```bash
git merge-recursive <last-copy-sha> -- HEAD <branch-tip-sha>
```

Measured for semantics: **2 conflicts instead of 18** — `docs/superpowers/CHANGELOG.md`
and `tests/dashboard/test_pipeline.py`.

Before trusting the boundary, confirm the copies really are faithful:

```bash
git diff --name-only 6f060dfa <branch-tip> | sort > /tmp/B.txt
while read -r f; do
  git diff --quiet <last-copy-sha> 5d480452 -- "$f" || echo "DRIFT: $f"
done < /tmp/B.txt
```

Any `DRIFT` path is one trunk changed *after* the originals landed — that is
HEAD-side work to keep, not branch drift. For semantics there were three
(`wrapper/reconstructor.py`, `evals/scripts/eval_similarity_calibration.py`,
`tests/wrapper/test_reconstructor.py`), all pointcloud/splats work.

**Files:**
- Modify: `clean/final` (one new commit)

- [ ] **Step 1: Freeze the source branch**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/semantics
git merge-base 5d480452 clean/semantics
```

Expected: `clean/semantics` → `a2002db0...`; merge-base → `6f060dfa...`
(semantics forked at the pristine base, which is an ancestor of `5d480452`).

If `clean/semantics` has moved off `a2002db0`, STOP and report.

- [ ] **Step 2: Squash-merge**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git merge-recursive 4697626e -- HEAD a2002db0
git status --porcelain | grep -E '^(UU|AA|DD|AU|UA|DU|UD)' || echo "NO CONFLICTS"
```

Expected (measured): exactly two conflicts —
`docs/superpowers/CHANGELOG.md` and `tests/dashboard/test_pipeline.py`.

A conflict count near 18 means the plain `git merge --squash` form was used and
the base fell back to `6f060dfa`. Reset and redo with the plumbing.

- [ ] **Step 3: Resolve `docs/known-test-failures.md` toward the branch copy**

This file provably diverged during the extraction (`trunk → clean/semantics` is
−30 lines). The branch copy is the version semantics' own gates were measured
against, so it wins.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git checkout --theirs docs/known-test-failures.md 2>/dev/null || \
  git show clean/semantics:docs/known-test-failures.md > docs/known-test-failures.md
git add docs/known-test-failures.md
grep -c '^<<<<<<< ' docs/known-test-failures.md || true
```

Expected: `0` markers.

Note: under `merge --squash` there is no `MERGE_HEAD`, so `--theirs` may not be
available; the `git show` fallback is the reliable path.

- [ ] **Step 4: Resolve any remaining conflicts among the other 14 files**

For each remaining conflicted path, take the semantics side where the hunk is
semantics-only work, and keep the preproc side where the hunk is preproc-only
work. `collab_splats/semantics/utils.py` is the one to read carefully: trunk
moved `ae_path`, `lifted_store_path` and `find_lifted_extractor` out of
`compression.py` into it, and in the analogous preproc merge **neither side of
the import block was takeable whole** — the block had to be reassembled by hand.

Verify no markers survive anywhere:

```bash
cd /workspace/collab-splats/.worktrees/clean-final
grep -rl '^<<<<<<< ' --include='*.py' --include='*.md' --include='*.yaml' --include='*.ipynb' . || echo "NO MARKERS"
```

Expected: `NO MARKERS`.

- [ ] **Step 5: Tree proof — three checks against the corrected base**

The proof must be stated against the branch-only footprint, not against
`diff(5d480452, <branch>)`. That second diff is polluted: the branch forked at
`6f060dfa` and therefore *lacks* the pointcloud and splats trunk originals, which
show up in it as removals. Demanding equality on those paths would delete
sibling work.

**Capture with `rtk proxy`.** A `PreToolUse` hook rewrites bare `git` through
`rtk`, which decorates its output — a redirected `git diff --name-only` writes
two blank lines and a `--- Changes ---` banner into the file, inflating every
`comm`/`wc` count by three.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad

rtk proxy git diff --name-only 4697626e a2002db0      | sort > $SCRATCH/new9.txt
rtk proxy git diff --name-only 5d480452 clean/preproc | sort > $SCRATCH/P-preproc-own.txt
git write-tree > /dev/null

# A. the squash must touch nothing outside the branch-only footprint
rtk proxy git diff --cached --name-only HEAD | sort > $SCRATCH/actually-changed.txt
comm -23 $SCRATCH/actually-changed.txt $SCRATCH/new9.txt

bad=0
# B. files the branch touched and preproc did not must be bit-identical to the branch
comm -13 $SCRATCH/P-preproc-own.txt $SCRATCH/new9.txt > $SCRATCH/N-not-P.txt
while read -r f; do
  git diff --cached --quiet a2002db0 -- "$f" || { echo "SEM-ONLY MISMATCH: $f"; bad=1; }
done < $SCRATCH/N-not-P.txt

# C. every preproc path outside the footprint must still equal clean/preproc
while read -r f; do
  git diff --cached --quiet clean/preproc -- "$f" || { echo "PREPROC DRIFT: $f"; bad=1; }
done < <(comm -23 $SCRATCH/P-preproc-own.txt $SCRATCH/new9.txt)

[ $bad -eq 0 ] && echo "TREE PROOF: PASS" || echo "TREE PROOF: FAIL"
echo "=== decision surface (differs from both, by construction) ==="
comm -12 $SCRATCH/P-preproc-own.txt $SCRATCH/new9.txt
```

Measured for semantics: **22 files changed, check A empty, 19 semantics-only
files bit-identical, `TREE PROOF: PASS`**, decision surface of three —
`collab_splats/semantics/utils.py`, `docs/superpowers/CHANGELOG.md`,
`tests/dashboard/test_pipeline.py`.

Check C is the load-bearing one: it proves the squash did not disturb the
preproc commit underneath it.

- [ ] **Step 6: Commit** — nothing to commit; Task 1 produces refs, not tracked
      changes. Confirm with `git status --porcelain` printing nothing.

---

## Task 2: Restore third_party in the worktree so the gate cannot false-green

`third_party/*` is gitignored, so a fresh worktree has none of the vendored
trees. Guarded tests then **skip** instead of failing, and the gate reports a
green that is really a smaller test run.

**Files:**
- Create: 9 symlinks + 1 file link under `/workspace/collab-splats/.worktrees/clean-final/third_party/`

- [ ] **Step 1: Confirm the gap**

```bash
ls /workspace/collab-splats/third_party/
ls -a /workspace/collab-splats/.worktrees/clean-final/third_party/
```

Expected: main tree lists `LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc
vggt-omega vggt_spark xfeat` plus `.vda_fetch_done`; the worktree lists only
`README.md` (tracked).

- [ ] **Step 2: Symlink each vendored tree in**

```bash
cd /workspace/collab-splats/.worktrees/clean-final/third_party
for d in LoGeR VGGT-SLAM VGGT-X Video-Depth-Anything bae hloc vggt-omega vggt_spark xfeat .vda_fetch_done; do
  [ -e "$d" ] || ln -s "/workspace/collab-splats/third_party/$d" "$d"
done
ls -la
```

Expected: ten symlinks pointing into `/workspace/collab-splats/third_party/`,
plus the tracked `README.md`.

- [ ] **Step 3: Verify the symlinks did not dirty the tree**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git status --porcelain
```

Expected: prints nothing. If a symlink shows as untracked, it is not covered by
`.gitignore` — stop and report rather than committing it.

- [ ] **Step 4: Prove the interpreter resolves to the worktree**

```bash
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -c 'import collab_splats; print(collab_splats.__file__)'
```

Expected: a path under `/workspace/collab-splats/.worktrees/clean-final/`.
If it prints `/workspace/collab-splats/collab_splats/...` the venv's editable
finder won the race — STOP, every gate from here would test the wrong tree.

- [ ] **Step 5: Commit** — nothing to commit (all links are ignored). Confirm
      `git status --porcelain` prints nothing.

---

## Task 3: Squash clean/preproc into clean/final

**Files:**
- Modify: `clean/final` (one new commit)

- [ ] **Step 1: Freeze the source branch**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/preproc
git rev-parse HEAD
git merge-base 5d480452 clean/preproc
```

Expected:
- `clean/preproc` → `d46b6fddae07c88432abd6f5232dc6f548c186c6`
- `HEAD` → `5d480452...`
- `merge-base` → `5d480452...` — **this identity is what makes the squash exact.**

If `clean/preproc` has moved off `d46b6fdd`, STOP and report; the plan's tree
proof is written against that SHA.

- [ ] **Step 2: Squash-merge**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git merge --squash clean/preproc
git status --porcelain | grep -cE '^(UU|AA|DD|AU|UA|DU|UD)' || true
```

Expected: conflict count `0`. Because the merge base *is* HEAD, this is a
fast-forward-shaped squash and cannot conflict. Any conflict means the
preconditions in Step 1 were not actually met — STOP.

- [ ] **Step 3: Tree proof — bit-exact**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
echo "squashed: $(git write-tree)"
echo "branch:   $(git rev-parse clean/preproc^{tree})"
test "$(git write-tree)" = "$(git rev-parse clean/preproc^{tree})" && echo "TREE PROOF: PASS" || echo "TREE PROOF: FAIL"
```

Expected: `TREE PROOF: PASS`, with both hashes printed and identical.
A `FAIL` here means the squash did not reproduce preproc's tree — STOP, do not
commit, report the differing paths via
`git diff --name-only clean/preproc -- .`

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git commit --no-verify -m "refactor(preproc): centralize video preprocessing into measure, filter, sample

Squashes the preproc centralization effort. What changed and why:

- preproc is now three ordered concerns: measure (qa.py) reports capture
  quality, filter drops bad frames, sample selects keyframes. The old
  interleaving made it impossible to say whether a frame was dropped for
  quality or simply not selected.
- frames.zarr is retired; the keyframe store is images/ + frames.json again.
  The zarr store bought nothing over a directory of JPEGs and cost a decode
  round-trip. No backfill — existing scenes re-decode.
- Blur filtering moved from a static threshold to a robust per-video MAD
  z-score. The static threshold cut exactly zero frames on real footage; the
  z-score cuts 5.11% of GH010229 and 14.53% of the tutorial video.
- The VDA context stream and decode_context are deleted. Both were dead paths
  kept for a streaming design that was never built.
- Samplers no longer take candidates=; callers pass the frame list directly.
- undistort_frames returns a LARGER image and a NEW camera. Callers must use
  the returned camera, not the one they passed in.
- Decode is PyAV. Seek-per-index measured SLOWER than a linear scan
  (97.4s vs 76.2s), so decode walks the stream once.
- Docstring lint anchors on ^\\s*param: rather than matching a substring, which
  was flagging prose.

Squashed from clean/preproc @ d46b6fdd (122 commits).
History: refs/backup/clean-preproc-20260906-013531"
git log --oneline -2 | cat
```

Expected: the new commit on top of `5d480452`.

- [ ] **Step 5: Re-verify the frozen SHA after committing**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/preproc
test "$(git rev-parse clean/preproc)" = "d46b6fddae07c88432abd6f5232dc6f548c186c6" \
  && echo "FREEZE HELD" || echo "FREEZE BROKEN — branch moved during the squash"
```

Expected: `FREEZE HELD`. If broken, the commit describes a SHA that is no longer
the tip — amend the message with the new SHA and re-run the tree proof.

---

## Task 4: Establish the control baseline

The preproc squash needs no A/B comparison — its tree is bit-identical to
`clean/preproc`, which was already gated on its own branch. This run exists to
produce the control that the **semantics** squash is measured against.

**Files:**
- Create: `$SCRATCH/control-preproc.txt`

- [ ] **Step 1: Run the fixed scope in the background**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -c 'import collab_splats; print("IMPORT:", collab_splats.__file__)' \
  > $SCRATCH/control-preproc.txt 2>&1 && \
echo "TREE: $(git rev-parse HEAD)" >> $SCRATCH/control-preproc.txt && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -m pytest -q \
  tests/dashboard tests/docs tests/evals tests/geometry tests/integration \
  tests/mesh tests/pointcloud tests/preproc tests/remote tests/semantics \
  tests/utils tests/wrapper \
  >> $SCRATCH/control-preproc.txt 2>&1; \
echo "PYTEST_RC=$?" >> $SCRATCH/control-preproc.txt
```

Run this with `run_in_background: true`. The trailing `echo` is what makes the
exit code recoverable — a background task's notification reports the exit code
of the LAST command, not pytest's.

- [ ] **Step 2: Wait for it, then confirm it actually ran**

Poll by file growth, not by `ps` or `stat`:

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
wc -c $SCRATCH/control-preproc.txt; tail -3 $SCRATCH/control-preproc.txt
```

Expected once finished: the last lines carry a pytest summary and `PYTEST_RC=<n>`.
A 0-byte file is not proof of a dead writer — `-u` is already set, so growth is
the signal.

- [ ] **Step 3: Verify the import proof line**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
head -2 $SCRATCH/control-preproc.txt
```

Expected: `IMPORT: /workspace/collab-splats/.worktrees/clean-final/collab_splats/__init__.py`
and a `TREE:` line matching the Task 3 commit. If `IMPORT:` points at the main
tree, this control is worthless — fix the invocation and re-run.

- [ ] **Step 4: Extract the failure set and skip count**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
grep -E '^(FAILED|ERROR) ' $SCRATCH/control-preproc.txt | sort > $SCRATCH/control-preproc.set
wc -l < $SCRATCH/control-preproc.set
grep -oE '[0-9]+ skipped' $SCRATCH/control-preproc.txt | tail -1
tail -1 $SCRATCH/control-preproc.txt
```

Record all three numbers in the plan as the control. The **set** is what
matters; the counts are a cross-check.

- [ ] **Step 5: Commit** — nothing to commit; the artifacts live in the
      scratchpad. Confirm `git status --porcelain` prints nothing.

---

## Task 5: Squash clean/semantics into clean/final

> **CORRECTED DURING EXECUTION.** This task originally named a 15-file overlap
> computed as `diff(5d480452, clean/preproc)` ∩ `diff(5d480452, clean/semantics)`.
> That list does not predict anything, because **`5d480452` is not the merge base
> git uses.** `clean/semantics` was rebased onto `6f060dfa` during the extraction,
> so `5d480452` is not its ancestor and `merge-base(HEAD, clean/semantics)`
> resolves to `6f060dfa`. A plain `git merge --squash clean/semantics` therefore
> re-merges the eight re-applied **copies** against the trunk **originals** the
> base already holds: measured, **18 conflicts**, 13 of them on paths outside the
> predicted list. The section below is the corrected procedure. It applies
> unchanged to `clean/pointcloud` and `clean/splats`, which were extracted the
> same way.

### The base is the last re-applied copy, not the fork point

Each extracted branch begins with copies of trunk commits the base already
carries, followed by the commits that are genuinely branch-only. Merging from the
fork point double-counts the copies. Merging from **the last copy** nets exactly
the branch-only work.

Find the boundary by subject (the copies are re-applied, so SHAs differ):

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git log --format='%s' 6f060dfa..5d480452 | sort -u > /tmp/trunk-subjects.txt
git log --format='%h %s' --reverse 6f060dfa..<branch> | while read -r h rest; do
  grep -qxF "$rest" /tmp/trunk-subjects.txt \
    && echo "COPY-OF-TRUNK  $h $rest" || echo "BRANCH-ONLY    $h $rest"
done
```

The last `COPY-OF-TRUNK` line is the merge base. For `clean/semantics` that is
`4697626e` — 8 copies, then 9 branch-only commits (`ed3faf8d..a2002db0`).

Merge with that base explicitly. `git merge --squash` cannot take one, so use the
plumbing, which leaves `HEAD` alone exactly like a squash and writes no
`MERGE_HEAD`:

```bash
git merge-recursive <last-copy-sha> -- HEAD <branch-tip-sha>
```

Measured for semantics: **2 conflicts instead of 18** — `docs/superpowers/CHANGELOG.md`
and `tests/dashboard/test_pipeline.py`.

Before trusting the boundary, confirm the copies really are faithful:

```bash
git diff --name-only 6f060dfa <branch-tip> | sort > /tmp/B.txt
while read -r f; do
  git diff --quiet <last-copy-sha> 5d480452 -- "$f" || echo "DRIFT: $f"
done < /tmp/B.txt
```

Any `DRIFT` path is one trunk changed *after* the originals landed — that is
HEAD-side work to keep, not branch drift. For semantics there were three
(`wrapper/reconstructor.py`, `evals/scripts/eval_similarity_calibration.py`,
`tests/wrapper/test_reconstructor.py`), all pointcloud/splats work.

**Files:**
- Modify: `clean/final` (one new commit)

- [ ] **Step 1: Freeze the source branch**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git rev-parse clean/semantics
git merge-base 5d480452 clean/semantics
```

Expected: `clean/semantics` → `a2002db0...`; merge-base → `6f060dfa...`
(semantics forked at the pristine base, which is an ancestor of `5d480452`).

If `clean/semantics` has moved off `a2002db0`, STOP and report.

- [ ] **Step 2: Squash-merge**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git merge-recursive 4697626e -- HEAD a2002db0
git status --porcelain | grep -E '^(UU|AA|DD|AU|UA|DU|UD)' || echo "NO CONFLICTS"
```

Expected (measured): exactly two conflicts —
`docs/superpowers/CHANGELOG.md` and `tests/dashboard/test_pipeline.py`.

A conflict count near 18 means the plain `git merge --squash` form was used and
the base fell back to `6f060dfa`. Reset and redo with the plumbing.

- [ ] **Step 3: Resolve `docs/known-test-failures.md` toward the branch copy**

This file provably diverged during the extraction (`trunk → clean/semantics` is
−30 lines). The branch copy is the version semantics' own gates were measured
against, so it wins.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git checkout --theirs docs/known-test-failures.md 2>/dev/null || \
  git show clean/semantics:docs/known-test-failures.md > docs/known-test-failures.md
git add docs/known-test-failures.md
grep -c '^<<<<<<< ' docs/known-test-failures.md || true
```

Expected: `0` markers.

Note: under `merge --squash` there is no `MERGE_HEAD`, so `--theirs` may not be
available; the `git show` fallback is the reliable path.

- [ ] **Step 4: Resolve any remaining conflicts among the other 14 files**

For each remaining conflicted path, take the semantics side where the hunk is
semantics-only work, and keep the preproc side where the hunk is preproc-only
work. `collab_splats/semantics/utils.py` is the one to read carefully: trunk
moved `ae_path`, `lifted_store_path` and `find_lifted_extractor` out of
`compression.py` into it, and in the analogous preproc merge **neither side of
the import block was takeable whole** — the block had to be reassembled by hand.

Verify no markers survive anywhere:

```bash
cd /workspace/collab-splats/.worktrees/clean-final
grep -rl '^<<<<<<< ' --include='*.py' --include='*.md' --include='*.yaml' --include='*.ipynb' . || echo "NO MARKERS"
```

Expected: `NO MARKERS`.

- [ ] **Step 5: Tree proof — per-file, against both sources**

No scratch re-merge is needed (and git 2.34.1 has no `merge-tree --write-tree`).
Prove the result directly: every file changed by only one side must equal that
side's blob, and only the 15 overlap files may differ from both.

**Capture with `rtk proxy`.** A `PreToolUse` hook rewrites bare `git` through
`rtk`, which decorates its output — a redirected `git diff --name-only` writes
two blank lines and a `--- Changes ---` banner into the file, inflating every
`comm`/`wc` count by three. `rtk proxy` bypasses the filter.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
rtk proxy git diff --name-only 5d480452 clean/preproc   | sort > $SCRATCH/p.txt
rtk proxy git diff --name-only 5d480452 clean/semantics | sort > $SCRATCH/s.txt
comm -12 $SCRATCH/p.txt $SCRATCH/s.txt > $SCRATCH/overlap.txt
comm -13 $SCRATCH/p.txt $SCRATCH/s.txt > $SCRATCH/sem-only.txt
comm -23 $SCRATCH/p.txt $SCRATCH/s.txt > $SCRATCH/pre-only.txt

git write-tree > /dev/null   # materialise the index

bad=0
while read -r f; do
  [ -e "$f" ] || continue
  git diff --quiet clean/semantics -- "$f" || { echo "SEM-ONLY MISMATCH: $f"; bad=1; }
done < $SCRATCH/sem-only.txt
while read -r f; do
  [ -e "$f" ] || continue
  git diff --quiet clean/preproc -- "$f" || { echo "PRE-ONLY MISMATCH: $f"; bad=1; }
done < $SCRATCH/pre-only.txt
[ $bad -eq 0 ] && echo "TREE PROOF: PASS" || echo "TREE PROOF: FAIL"
wc -l $SCRATCH/overlap.txt $SCRATCH/sem-only.txt $SCRATCH/pre-only.txt
```

Expected: `TREE PROOF: PASS`, and the counts `15` overlap, `42` semantics-only,
`72` preproc-only (from 87 preproc-changed and 57 semantics-changed paths).
Any `MISMATCH` line names a file the merge got wrong — fix it and re-run before
committing. If the counts come out three higher than these, the `rtk proxy`
prefix was dropped and the banner lines are in the files.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git commit --no-verify -m "refactor(semantics): fold artifact I/O into utils, trim the autoencoder, fix mask compositing

Squashes the semantics cleanup effort. What changed and why:

- All semantics artifact I/O lives in semantics/utils.py. It was spread across
  compression.py and the extractors, so no single place knew the on-disk
  layout.
- FeatureAutoencoder drops the regression head, the lr_scheduler and
  hidden_dim, and moves to path-based save/load. None of the three had a
  caller; the checkpoint format is pinned so legacy files still load.
- preprocess is hoisted into BaseFeatureExtractor and the image constants move
  to utils.image. Every extractor had reimplemented the same resize/normalize.
- create_composite_mask had two off-by-one indices: a rotate-by-one made the
  WRONG segment win every overlap. Mask IDs are now uint16 so scenes with more
  than 255 masks stop wrapping.
- segment() has an honest contract: a nullable return type matching what the
  overrides actually return, instead of an abstract signature none of them met.
- The feature-cache validity marker is written LAST, so an interrupted write
  can no longer leave a cache that reads as valid.
- TORCH_HOME is inlined at its single call site; dead code, a dead raise and a
  compatibility shim are removed; imports are hoisted to module top.
- Docstrings converted to Args/Returns, and the coverage claim corrected.

Squashed from clean/semantics @ a2002db0 (17 commits).
History: refs/backup/clean-semantics-20260906-013531"
git log --oneline -3 | cat
```

Expected: three commits — semantics squash, preproc squash, `5d480452`.

- [ ] **Step 7: Re-verify the frozen SHA**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
test "$(git rev-parse clean/semantics)" = "$(git rev-parse a2002db0)" \
  && echo "FREEZE HELD" || echo "FREEZE BROKEN"
```

Expected: `FREEZE HELD`.

---

## Task 6: Gate the semantics squash

**Files:**
- Create: `$SCRATCH/gate-semantics.txt`, `$SCRATCH/gate-semantics.set`

- [ ] **Step 1: Run the identical scope**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -c 'import collab_splats; print("IMPORT:", collab_splats.__file__)' \
  > $SCRATCH/gate-semantics.txt 2>&1 && \
echo "TREE: $(git rev-parse HEAD)" >> $SCRATCH/gate-semantics.txt && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -m pytest -q \
  tests/dashboard tests/docs tests/evals tests/geometry tests/integration \
  tests/mesh tests/pointcloud tests/preproc tests/remote tests/semantics \
  tests/utils tests/wrapper \
  >> $SCRATCH/gate-semantics.txt 2>&1; \
echo "PYTEST_RC=$?" >> $SCRATCH/gate-semantics.txt
```

Run with `run_in_background: true`. The scope string must be byte-identical to
Task 4 Step 1 — a different scope makes the comparison meaningless.

- [ ] **Step 2: Verify the import proof and the tree line**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
head -2 $SCRATCH/gate-semantics.txt
```

Expected: `IMPORT:` under `.worktrees/clean-final/`, and a `TREE:` SHA that
differs from the control's (it must be the semantics squash commit).

- [ ] **Step 3: Diff the failure sets**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
grep -E '^(FAILED|ERROR) ' $SCRATCH/gate-semantics.txt | sort > $SCRATCH/gate-semantics.set
echo "--- NEW failures (in gate, not in control) ---"
comm -13 $SCRATCH/control-preproc.set $SCRATCH/gate-semantics.set
echo "--- FIXED (in control, not in gate) ---"
comm -23 $SCRATCH/control-preproc.set $SCRATCH/gate-semantics.set
echo "NEW count: $(comm -13 $SCRATCH/control-preproc.set $SCRATCH/gate-semantics.set | wc -l)"
```

Expected: `NEW count: 0`. Anything listed under NEW is a regression the squash
introduced — investigate before proceeding.

A non-empty FIXED list is **not** self-evidently good: a test that stopped
running looks identical to a test that started passing. For each FIXED entry,
confirm it still collects:

```bash
cd /workspace/collab-splats/.worktrees/clean-final && \
PYTHONPATH=/workspace/collab-splats/.worktrees/clean-final \
/opt/venv/reconstruction/bin/python -u -m pytest --collect-only -q <that::test::id>
```

- [ ] **Step 4: Diff the skip counts**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/4f312e18-0fbb-4f7c-a2a3-20eb149a798c/scratchpad
echo "control: $(grep -oE '[0-9]+ skipped' $SCRATCH/control-preproc.txt | tail -1)"
echo "gate:    $(grep -oE '[0-9]+ skipped' $SCRATCH/gate-semantics.txt | tail -1)"
```

Expected: identical. A rise in skips is the `third_party` trap re-appearing —
re-check Task 2.

- [ ] **Step 5: Commit** — nothing to commit; artifacts live in the scratchpad.

---

## Task 7: Carry the spec and plan onto clean/final, and record the hand-off

`docs/superpowers/` is gitignored, so these files exist only in the main tree's
working copy and must be force-added on `clean/final`.

**Files:**
- Create on `clean/final`: `docs/superpowers/specs/2026-09-06-clean-final-integration-design.md`
- Create on `clean/final`: `docs/superpowers/plans/2026-09-06-clean-final-integration.md`
- Modify on `clean/final`: `CLAUDE.md` (In-Flight Work list)

- [ ] **Step 1: Copy the two documents into the worktree**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp /workspace/collab-splats/docs/superpowers/specs/2026-09-06-clean-final-integration-design.md \
   docs/superpowers/specs/
cp /workspace/collab-splats/docs/superpowers/plans/2026-09-06-clean-final-integration.md \
   docs/superpowers/plans/
ls docs/superpowers/specs/2026-09-06-* docs/superpowers/plans/2026-09-06-*
```

Expected: both paths listed.

- [ ] **Step 2: Add the In-Flight entry to CLAUDE.md**

`CLAUDE.md` is clean on `clean/final` (the main tree's copy is dirty with
another session's work — do not touch that one). Insert this line into the
`## In-Flight Work` list, immediately after the `scaffold-gs` bullet:

```markdown
- **clean-final** — integration branch for the five cleanup efforts; preproc and semantics landed, pointcloud/splats/mesh pending ([spec](docs/superpowers/specs/2026-09-06-clean-final-integration-design.md) · [plan](docs/superpowers/plans/2026-09-06-clean-final-integration.md))
```

- [ ] **Step 3: Verify CLAUDE.md stayed under the size limit**

A `PreToolUse` hook enforces a ceiling; a truncated `CLAUDE.md` silently drops
the rules below the cut.

```bash
cd /workspace/collab-splats/.worktrees/clean-final
wc -c CLAUDE.md
```

Expected: well under `40000`.

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
git add -f docs/superpowers/specs/2026-09-06-clean-final-integration-design.md \
           docs/superpowers/plans/2026-09-06-clean-final-integration.md
git add CLAUDE.md
git commit --no-verify -m "docs(specs,plans): clean/final integration design and plan

Carries the design and the implementation plan onto the integration branch
itself, and records clean-final in the In-Flight Work list. preproc and
semantics have landed; pointcloud, splats and mesh follow on request.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
git log --oneline -4 | cat
```

Expected: four commits — docs, semantics squash, preproc squash, `5d480452`.

- [ ] **Step 5: Report the final state**

```bash
cd /workspace/collab-splats/.worktrees/clean-final
echo "clean/final tip: $(git rev-parse --short HEAD)"
echo "base:            $(git rev-parse --short 5d480452)"
git log --oneline 5d480452..HEAD | cat
git status --porcelain
git for-each-ref refs/backup/ --format='%(refname) -> %(objectname:short)' | grep clean-final-wip
```

Expected: three commits above the base, a clean status, and the WIP backup ref.

---

## Hand-off: the remaining three efforts

Do **not** start these without the user saying go — `clean/splats` and
`clean/pointcloud` are still being worked on and moved three times during the
design conversation alone.

**"In code review" is not frozen.** `clean/pointcloud` entered review on
2026-09-06 and its tip had already moved again by the time the question was
asked (`5c16ec0b` → `1d7c4734`, the fourth reading). Review produces commits by
construction, and rule 2 below allows exactly one squash per branch. Wait for
the branch to stop moving. The read-only reconnaissance below can be — and was —
done ahead of the freeze; it does not touch the branch and costs nothing if the
tip moves again.

When one is frozen, run Tasks 3, 5 and 6 with these substitutions:

| | `clean/pointcloud` | `clean/splats` | `clean/mesh` |
|---|---|---|---|
| Frozen SHA | re-read at freeze time (was `1d7c4734`, in review) | re-read at freeze time (was `02d9a693`) | re-read at freeze time (was `30326769`) |
| Backup ref | `refs/backup/clean-pointcloud-20260906-013531` | `refs/backup/clean-splats-20260906-013531` | none yet — create one before squashing |
| Merge base | measured `2ef1c7bb` @ `1d7c4734` (17 copies / 20 branch-only) — **re-derive at freeze** | derive it — last `COPY-OF-TRUNK` (Task 5) | n/a if rebased |
| Expected conflict | the VDA collision; consult `$SCRATCH/vda-resolution-ref/`. 34-file footprint, **15 of them already touched by `clean/final`** — see below | `docs/known-test-failures.md` | to be measured |
| Scope | recompute as `rtk proxy git diff --name-only <last-copy> <branch> -- tests/`, union with the existing scope | same | same |

Four standing rules:

1. **Derive the merge base; never assume the fork point.** `clean/pointcloud`
   and `clean/splats` were extracted exactly like `clean/semantics`, so both
   open with re-applied copies of trunk commits `5d480452` already holds.
   Run the boundary script in Task 5, take the last `COPY-OF-TRUNK` commit, and
   merge with `git merge-recursive <that> -- HEAD <branch-tip>`. Using
   `git merge --squash` instead silently falls back to `6f060dfa` and re-merges
   every copy against its original — for semantics that was 18 conflicts rather
   than 2, and the extra 16 are all spurious.
2. **Squash each branch exactly once.** A squash records no merge ancestry, so a
   second squash re-diffs from the fork point and double-applies everything.
3. **Re-measure the control after every squash.** The control for effort N+1 is
   the gate run of effort N, not an older baseline.
4. **`clean/mesh` intends to rebase onto `clean/final` rather than be squashed
   into it.** That is compatible with this shape — mesh carries no copies of
   trunk originals, so its commits replay cleanly onto the squashed tip — but
   confirm the choice with the user when mesh is frozen rather than assuming
   either path.

**Cross-effort breaks are real and the merge will not flag them.** Semantics'
`extract_feature_cache` still called `FrameStore.open()`, which preproc had
deleted; the port to `images_dir`/`frame_paths` already existed on the preproc
side, so the auto-merge picked it up silently. Expect the same shape from
pointcloud (VDA context stream) and grep the merged tree for symbols the other
effort retired before trusting a clean merge.

### `clean/pointcloud` reconnaissance (measured 2026-09-06, tip `1d7c4734`)

Read-only, done while the branch was still in review. Re-run it at freeze time —
these numbers describe a tip that is expected to move.

- Boundary: **17 copies / 20 branch-only**, so the merge base is `2ef1c7bb`
  (`fix(sfm): prune depth_vda on a gate miss; cover the VDA cache gate`).
- Branch-only footprint: 34 files, **15 of which `clean/final` has already
  changed**. That overlap is the whole conflict surface.

**The one that is not mechanical: `pointcloud/sfm.py` is a modify/delete.**
`clean/pointcloud` explodes the 62 KB module into a package and splits two more
out of it:

```
D  collab_splats/pointcloud/sfm.py
A  collab_splats/pointcloud/sfm/__init__.py
A  collab_splats/pointcloud/sfm/colmap.py
A  collab_splats/pointcloud/sfm/hloc.py
A  collab_splats/pointcloud/sfm/instantsfm.py
A  collab_splats/pointcloud/vda.py
A  collab_splats/pointcloud/depth_align.py
```

`clean/final` meanwhile carries **+62/−13 in seven hunks** of the file pointcloud
deletes — `generate_vda_depth`, `_nudge_edge_keypoints`, `_sift_database_valid`,
and three inside `InstantSfMCreator`. Git will report this as
deleted-by-them/modified-by-us and offer no useful resolution: taking either side
loses work. Each of the seven hunks has to be hand-routed into whichever new
module now owns that function (`vda.py` for `generate_vda_depth`,
`sfm/instantsfm.py` for the rest — confirm, don't assume). Budget real time for
this one; it is the only conflict in the whole integration that requires reading
both implementations rather than picking a side.

**The rest of the overlap is ordinary.** `preproc/undistort.py` is the shape to
expect: `clean/final` rewrote it (+105/−161) while `clean/pointcloud` changed one
comment line, repointing `_SIFT_NUM_THREADS`'s cross-reference from
`pointcloud/sfm.py` to `pointcloud/sfm/instantsfm.py::_generate_sift_database`.
The symbol survives the rewrite at `undistort.py:37`, so the conflict is context
only — take pointcloud's text. Same for `docs/known-test-failures.md`,
`docs/superpowers/CHANGELOG.md` and `CLAUDE.md`: resolve toward keeping both
efforts' entries, as was done for semantics.

Trunk replacement is out of scope for this plan and gets its own spec once all
five efforts have landed.
