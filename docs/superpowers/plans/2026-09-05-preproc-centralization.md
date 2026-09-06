# preproc Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `frames.zarr` with a COLMAP-style `images/` directory, make the quality filter actually filter, and replace three hand-rolled subsystems (undistortion framing, ffmpeg decode, MAD) with library calls — cutting `collab_splats/preproc/` from 2268 to ~1485 lines.

**Architecture:** Five sequential phases. A swaps the storage format behind a flat function API with byte-identical frame selection. B changes which frames are selected and is gated on an A/B measurement. C and D replace undistortion and decode with pycolmap and PyAV. E deletes dead code and rewrites docstrings. Each phase leaves the test suite green and the dashboard smoke-passing.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), OpenCV, pycolmap 4.0.4, PyAV 17.0.1, scipy, numpy, pytest.

**Spec:** [2026-09-05-preproc-centralization-design.md](../specs/2026-09-05-preproc-centralization-design.md)

---

## Ground Rules

Read these before Task 1. They apply to every task.

- **Python is `/opt/venv/reconstruction/bin/python`.** The base shell `python` is 3.13 and wrong for this project. Every command in this plan spells the interpreter out.
- **Format before every commit:** `black . && isort .` — but never repo-wide `black` on unrelated files; stage only what the task touched.
- **`docs/superpowers/` is gitignored.** Committing anything under it needs `git add -f`.
- **Commit with `git commit --only <paths>`**, never bare `git commit -a`. Other sessions share this working tree and a bare commit sweeps their staged work.
- **The dashboard smoke gate is mandatory** before any commit that touches `collab_splats/dashboard/`: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`. A pass prints `SMOKE PASS: page (N B) + bokeh.min.js (N B) served; bind took Ns`. **Do not spell it `-m collab_splats.dashboard.serve`** — `serve.py` has no `__main__` block, so that spelling prints nothing and exits 0 however broken the page is (measured 2026-09-05). No output means the gate did not run.
- **Two hard gates block phases.** Phase B does not land until Task 14 (the GH010229 sampling A/B) passes and the user has read it. Phase D does not land until Task 19 (the rotated-video fixture) exists and is green against the current ffmpeg code. Do not skip them.
- **Colour convention:** every function in `preproc/frames.py` takes and returns **RGB**. `cv2` reads and writes BGR, so conversions happen inside `frames.py` and nowhere else.

---

## Formatting — read before running black or isort

**This repo is not `black`-clean at HEAD.** The venv's black is 26.5.1, newer
than whatever last formatted the tree, so `black <directory>` rewrites files you
never touched. Measured on Task 7: a directory-wide run produced **373
insertions / 273 deletions** against ~90 lines of real semantic change, and all
nine files it touched fail `black --check` at HEAD. `isort` on the same files is
likewise pure churn — it re-wraps pre-existing imports and reorders module
constants.

So every `black <dir>` / `isort <dir>` command written into the tasks below is
wrong, including the ones that name directories like `tests/wrapper/`. Instead:

- Run `black` only on the individual files you edited, then read the diff.
- If it reformats lines outside your change, revert it entirely and simply keep
  your own added lines under 120 characters.
- `line-length = 120` and flake8 ignores E501, so there is no reason to hand-wrap
  at 88. `isort`'s `profile="black"` wraps *imports* at 88; that part is real.
- Verify with `flake8 --max-line-length=120 <files>` and confirm `pyflakes`
  output is unchanged, rather than trusting a formatter to prove correctness.

## Parallel Execution Schedule

The 28 tasks are not a straight line. Their file sets are disjoint in two big
places, and exploiting that collapses the critical path from 28 slots to ~18.

**Every wave runs in its own git worktree, one per task.** The main checkout at
`/workspace/collab-splats` is shared with other sessions and its git index is
shared with every worktree — two agents committing there at once sweep each
other's staged AND unstaged work. A worktree gives each agent its own index, so
broad `git commit --only <dir>` paths become safe again.

### Worktree protocol

```bash
# integration branch, forked once per wave from the previous wave's merge
git worktree add /workspace/collab-splats/.worktrees/preproc-t<N> -b preproc/t<N> <base-sha>
```

Inside a worktree, **`PYTHONPATH` is mandatory**. The venv installs
`collab_splats` editable through a finder that hardcodes
`/workspace/collab-splats`, so a bare `pytest` in a worktree silently tests the
MAIN tree's code and reports success on work that was never applied:

```bash
cd /workspace/collab-splats/.worktrees/preproc-t<N>
PYTHONPATH=/workspace/collab-splats/.worktrees/preproc-t<N>:/workspace/collab-data \
  /opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Every `pytest`, every `python -c`, every dashboard smoke run inside a worktree
carries that prefix. A task that reports green without it has verified nothing.

**The second path entry is not optional either.** `collab_splats/remote/sources.py`
imports `collab_data.data_dashboard.rclone_client`. `collab_data` is a declared
dependency (`pyproject.toml:98`, `file:///workspace/collab-data`) that is **not
installed in this venv** — env drift, not a code bug. Without
`/workspace/collab-data` on the path, `tests/remote/test_sources.py` and
`tests/remote/test_rerun.py` fail at collection and the dashboard smoke gate exits
1 before it binds. With it, the smoke gate passes.

**Collection errors abort the entire pytest run.** This is the single most
misleading thing about testing this repo right now: one un-importable module
means pytest reports `Interrupted: N errors during collection` and runs *zero*
tests. A task that sees that and reads it as "the suite is broken" or as a hang is
misreading it. Check `docs/known-test-failures.md` and pass `--ignore` for each
known-blocked path.

### Merge protocol — take every commit, then re-read the merged file

A task branch is not one commit. Task 4 landed two (`d88814d7` and `8545a3a1`);
the wave-1 merge took the first and dropped the second, and nothing reported it.
The result compiled and imported fine, and two tests in
`tests/preproc/test_undistort.py` raised `TypeError` on a keyword the rename had
removed. **Enumerate `git log <base>..<branch>` before merging and confirm every
SHA is an ancestor of the integration tip afterwards:**

```bash
git log --oneline <base>..preproc-t<N>              # what the branch actually holds
git merge-base --is-ancestor <sha> HEAD && echo OK  # per SHA, after the merge
```

**A clean merge is not a correct merge.** Two branches that both add the same
method to the same class merge without a conflict and leave the class holding it
twice — Python silently takes the second. That happened to
`Reconstructor.images_dir` in wave 1. After merging any two branches that touched
one file, grep the merged result for duplicated `def`s:

A `git grep -c "def <name>"` catches it only if you already suspect the name. Parse
instead — this is cheap, covers the whole tree, and knows that `@x.setter`,
`@overload` and `@f.register` reuse a name on purpose:

```bash
PYTHONPATH=$PWD:/workspace/collab-data /opt/venv/reconstruction/bin/python - <<'EOF'
import ast, pathlib

OK = {"setter", "getter", "deleter", "register", "overload"}

def legal(node):
    for d in getattr(node, "decorator_list", []):
        d = d.func if isinstance(d, ast.Call) else d
        if isinstance(d, ast.Attribute) and d.attr in OK:
            return True
        if isinstance(d, ast.Name) and d.id in OK:
            return True
    return False

bad = 0
for root in map(pathlib.Path, ("collab_splats", "evals", "scripts", "tests")):
    for f in sorted(root.rglob("*.py")) if root.exists() else []:
        def scan(body, scope):
            global bad
            seen = {}
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if n.name in seen and not legal(n):
                        print(f"DUP {f}:{seen[n.name]},{n.lineno} {scope}{n.name}")
                        bad += 1
                    seen[n.name] = n.lineno
                if isinstance(n, ast.ClassDef):
                    scan(n.body, f"{scope}{n.name}.")
        scan(ast.parse(f.read_text()).body, "")
print(f"real duplicates: {bad}")
EOF
```

Expected: `real duplicates: 0`. It read 0 across the whole tree after wave 1's
`images_dir` duplicate was removed, so any non-zero result is new.

**Renames hide behind positional arguments.** `f(a, b)` into a callee whose first
parameter was renamed raises nothing and changes what the argument means. After a
parameter rename lands, grep the *callers* — not the signature.

**Import-smoke the merged tree before running the suite.** A collection error costs a
full suite run to discover and tells you only that *something* failed to import. This
answers the same question in about a minute and names every module:

```bash
PYTHONPATH=$PWD:/workspace/collab-data /opt/venv/reconstruction/bin/python - <<'EOF'
import importlib, pkgutil
import collab_splats

bad = []
for m in pkgutil.walk_packages(collab_splats.__path__, "collab_splats."):
    try:
        importlib.import_module(m.name)
    except Exception as e:
        bad.append((m.name, f"{type(e).__name__}: {e}"))
for name, err in bad:
    print(f"FAIL {name}: {err}")
print(f"failed {len(bad)}")
EOF
```

Expected on a healthy merge: `failed 1`, that one being
`collab_splats.splats: ImportError: cannot import name 'losses' from 'gsplat'` — the
container's broken gsplat, not your merge. Anything else is yours.

**`git log --oneline` lies here.** The repo-wide `rtk` wrapper filters merge commits
out of `git log --oneline` and `--graph`, so an integration branch built from merges
renders as a flat list of the merged-in commits and reads like a rebase. That is how
`preproc-t8` — actually the wave-0 integration line, three merge commits deep — looked
like a plain task branch. Verify ancestry with commands whose output is a fact rather
than a rendering, and pull raw history through the proxy:

```bash
git merge-base --is-ancestor <sha> HEAD    # exit status, not text
rtk proxy git log --format='%h %p %s' -10 <ref>   # %p exposes the second parent
```

### Coverage audit — the defect this plan keeps producing

Thirteen times now the same defect: a file that references the thing being retired,
named in no task's Files list. It is invisible during execution because each task
only greps its own scope, and it surfaces as a runtime break one wave later.

Found so far: `preproc/viz.py` (fixed by moving Task 25 to wave 0), six orphaned test
files (Task 8b), five `reconstructor.py` production call sites (Task 4b),
`docs/source/tutorials/tutorial_config.py` and four extra notebooks (folded into
Task 8), `evals/scripts/eval.py` (Task 8), `configs/README.md` (Task 9),
`tests/preproc/test_sampling.py` (Task 9), `tests/wrapper/_stubs.py` (Task 8b),
`tests/remote/test_sources.py` (Task 9) and `tests/docs/test_tutorial_config.py`
(Task 8, sent to the agent mid-flight), `tests/preproc/test_undistort.py` (Task 8b) and
two dead imports in `tests/wrapper/test_splats_stage.py` (fixed directly).

The `test_splats_stage.py` one is worth naming because of who caused it. Commit
`9c09585b` moved `_stub_reconstructor` out of that file but left behind the
`FrameStore` and `Reconstructor` imports it had needed, both now unused — pyflakes
catches it in a second, and Task 9's grep would have tripped on it a wave later.
**Run `pyflakes` on any file you remove code from**, in the same commit.

**Both of the last two are the test file of a production file a task already owned.**
Task 9's Files list named `collab_splats/remote/sources.py` but not its test; Task 8's
named `tutorial_config.py` but listed only `test_notebook_utils.py` beside it. When
adding a production file to a task, add its test in the same edit — the audit grep
finds these, reading the Files list does not.

The last two are worth separating from the rest, because they are a different shape:

- `tests/preproc/test_sampling.py::test_public_api_surface` asserts `preproc.__all__`
  equals a literal name set containing `"FrameStore"`. It never imports the class and
  never touches a store, so it survives every import-based check and every "does this
  module use FrameStore" reading — and then fails the instant Task 9 edits `__init__.py`.
  **A string-literal assertion on a public surface is a call site.** Grep for the name as
  text, not as an import.
- `tests/wrapper/_stubs.py` is a call site *this plan created*. Commit `9c09585b` split
  it out of `test_splats_stage.py` to unblock `_run_sfm` coverage, and carried the old
  `FrameStore.create` across verbatim. Mid-plan refactors inherit the old API by default;
  re-run the audit after any of them, not only before each wave.

Re-run this audit before dispatching each wave and diff it against the union of every
task's Files list. Note the two greps disagree — a file can reach the store through a
constant without ever naming `FrameStore`, which is exactly how three tutorial
notebooks hid from Task 8's own grep:

```bash
rtk proxy git grep -l -E 'FrameStore|frames_zarr|frames\.zarr' -- \
  'collab_splats/*' 'evals/*' 'scripts/*' 'tests/*' 'configs/*' \
  'docs/source/*' | sort
```

`docs/superpowers/` is excluded on purpose: those are historical specs and plans that
record what was true when written. Do not rewrite them.

The "diff it against the Files lists" half was being done by eye, which is how thirteen
gaps got past it. Do it mechanically instead — this is the whole audit, grep plus
ownership check, and it runs in a second:

```bash
/opt/venv/reconstruction/bin/python - <<'PY'
import subprocess
from pathlib import Path

plan = Path("docs/superpowers/plans/2026-09-05-preproc-centralization.md").read_text()
files = subprocess.run(
    ["git", "grep", "-l", r"FrameStore\|frames_zarr\|frames\.zarr", "--",
     "collab_splats/", "tests/", "evals/", "scripts/", "configs/", "docs/source/"],
    capture_output=True, text=True).stdout.split()

# The plan cites paths in abbreviated form ("pointcloud/feedforward/loger.py:295"),
# so match any suffix of the path, not the full path
for f in sorted(files):
    parts = f.split("/")
    if not any("/".join(parts[i:]) in plan for i in range(len(parts))):
        print("ORPHAN", f)
PY
```

**Run this after every merge, not only before every wave.** Two of the thirteen were
created *by* the plan mid-flight, so a pre-wave-only cadence cannot catch them.

> Clean as of the Task 4b merge (`5b6b171e`, 2026-09-05): **zero orphans** across 25
> matching files.
>
> Re-run after the Task 8b and Task 10 merges (`d75817d3`, 2026-09-05): **zero orphans**
> across **22** matching files — three fewer because Task 8b converted
> `tests/wrapper/test_vda_context.py`, `tests/wrapper/_stubs.py` and
> `tests/preproc/test_undistort.py` off the store. Every remaining reference belongs to
> Task 9 or to a later task named in Task 9's ownership table.

**Eight hits are prose, not code**: `pointcloud/feedforward/loger.py:295`,
`pointcloud/sfm.py:297`, `preproc/qa.py:478`, `preproc/undistort.py:2` and `:49`,
`preproc/video.py:273`, `preproc/viz.py:103`, and the comment at
`tests/pointcloud/feedforward/test_preprocess_frames.py:29`. Each names `frames.zarr` in a
docstring or block comment that Phase A makes false — `undistort.py:2` is the module
docstring, "Camera undistortion at the frames.zarr boundary". They break nothing at
runtime, so do not let them block a wave.

> **Correction, 2026-09-05, after the Task 8b and Task 10 merges.** This paragraph used to
> say the eight belong to Task 27 and that they "will each match Task 9's Step 1 grep ... so
> fix them before Task 9's gate or that gate cannot pass." **Both halves are wrong, and the
> second is wrong in the direction that wastes a wave.**
>
> Task 9's Step 1 greps `frame_store\|FrameStore`. Not one of these eight matches it — they
> spell it `frames.zarr`, with a dot, and none names the class. Measured on `d75817d3`:
> Step 1's grep returns 21 lines and **none** of them is in this list. Task 9's gate does not
> need them fixed and never did.
>
> Task 27 does not own them either. Its Files list is "every public function in
> `collab_splats/preproc/`" and its deliverable is a docstring-contract lint —
> `loger.py` and `sfm.py` are outside that scope entirely, and a block comment at
> `viz.py:103` is not a public function's docstring.
>
> The real ownership is in the table under **"Do not chase the `frames.zarr` prose here"** in
> Task 9's section. Use that table, not this paragraph. It assigns four of the eight
> (`viz.py`, `qa.py`, `loger.py`, `sfm.py`) to Task 9's commit as orphans, leaves
> `undistort.py` to Tasks 15-18 and `video.py` to Tasks 20-24, which rewrite those headers
> anyway, and names three further hits that are **correct as written** and must not be
> converted (`preproc/frames.py:188`, `scripts/migrate_frames_zarr.py`,
> `tests/preproc/test_frames.py:139-145` — all three legitimately describe the old store).
>
> The general lesson is the one Task 8b's errors 2 and 3 already paid for, restated at the
> plan level: **a claim that "grep X will catch these" is a claim about a pattern, and it has
> to be run, not reasoned about.** Two spellings of the same store, one dot apart, made this
> paragraph confidently wrong for four days.

### A monkeypatched stub is a reader, and its contract is a partial dict

The grep-plus-ownership check above finds files that *name* an identifier. It cannot find
the readers that matter most for Tasks 20-24, because they do not name the thing they
depend on — they replace it.

`get_video_info` returns a dict. Six tests replace it with a lambda returning a **partial**
one, measured on the tip:

| site | stub |
|---|---|
| `tests/wrapper/test_reconstructor.py:163`, `:213`, `:233` | `lambda path: {"total_frames": 100}` |
| `tests/wrapper/test_reconstructor.py:1578` | `lambda path: {"total_frames": n}` |
| `tests/dashboard/test_app.py:525` | `lambda p: {"total_frames": 777}` |
| `tests/preproc/test_viz.py:159` | `lambda path: {"fps": 10.0}` |

Each stub silently asserts two things the function's signature does not: **which keys its
caller reads**, and **that the caller passes exactly one positional argument.** The real
signature is `get_video_info(video_path, *, count_frames: bool = True)`, and `video.py:140`
already passes `count_frames=False` — so a rewrite that pushes that keyword to one more
call site turns six stubs into `TypeError: <lambda>() got an unexpected keyword argument`.
Adding one key a caller reads turns them into `KeyError`. Neither is visible from
`collab_splats/preproc/video.py`, from `tests/preproc/test_video.py`, or from pyflakes.

A seventh site asserts the *absence* of a call: `tests/wrapper/test_vda_context.py:230`
patches it with `side_effect=AssertionError("video was probed")`. A backend that opens the
container earlier fires it.

**The rule for Tasks 20-24: before changing any function in `preproc/video.py`, grep
`monkeypatch.setattr(.*<name>` and `patch(.*<name>` across all of `tests/`, and state in
the report that the returned key set and the call signature are unchanged — or name every
stub you had to update.** A green `tests/preproc/test_video.py` is not evidence about any
of them; none of those files is in Task 20's Files list.

### The gate that reports nothing is not a gate that passed

The Bash tool's default `timeout` is **120000 ms**, and this repo's scoped suites run
longer. A run that exceeds it is SIGTERMed: exit **143**, and an output file of **0
bytes** — no summary line, no traceback, no FAILED list. It looks exactly like a run
still in flight, and nothing about it reads as a failure.

Measured 2026-09-05: three consecutive `tests/wrapper tests/pointcloud tests/preproc`
runs died this way, and two task agents blocked waiting on runs that no longer existed.

**Pass `timeout: 600000` on every pytest invocation** — foreground or background,
yours or a subagent's. And before reading any result, confirm a summary line
(`N passed`, `N failed in Xs`) is actually present.

> **Retracted 2026-09-05 — an earlier revision of this box claimed a foreground call
> is SIGTERMed at the 600 s ceiling and therefore "cannot outlive `tests/preproc`".
> That is false, and the claim was the coordinator's, not this plan's.** Task 21
> disputed it with a completed `181 passed ... in 605.50s` run, and a direct experiment
> settled it: a Bash call with `timeout: 60000` running `sleep 90` returns
> `Command did not complete within its 60s timeout and was moved to the background
> (ID: ...)` and then notifies on completion. **Exceeding `timeout` promotes the call to
> background; it does not kill it.** A long suite run this way still produces a valid
> summary line.
>
> What survives: `timeout: 600000` is still right, the summary-line check is still
> mandatory, and `run_in_background: true` is still *preferable* for anything over ~10
> minutes — because the promotion costs a wasted foreground turn and, more importantly,
> because it is what keeps you from writing the poll loop described below. But the
> truncated files this session saw were **not** caused by the ceiling. Their cause is
> unproven; contention or an OOM kill are the open candidates. Do not repeat the
> SIGTERM explanation as if it were measured.

**An empty test result is never a pass.** This is the third false-green shape this plan
has hit, after the `.serve --smoke` spelling that runs nothing and the bare `pytest` in a
worktree that tests the main tree. All three report success by producing nothing.

#### The fourth shape: `| tail -N` throws pytest's exit code away

Measured 2026-09-05 on the post-8b/post-T10 integration tip:

```
pytest tests/wrapper tests/pointcloud tests/preproc -q ... 2>&1 | tail -40
```

reported **`[exited with code 0]`** over an output that stops at `[ 27%]` with no summary
line. The run did not pass — it died a quarter of the way in, with 31 concurrent pytest
processes on the mount from three task agents. The `0` is `tail`'s exit code. Confirmed
directly:

```
$ python -c "import sys; sys.exit(7)" 2>&1 | tail -3 ; echo $?
0
$ python -c "import sys; sys.exit(7)" ; echo $?
7
```

Bash reports the **last** command in a pipeline, and `tail` always succeeds. So a piped
gate cannot fail. This is worse than the SIGTERM shape, because a 0-byte file at least
looks wrong — this one prints a plausible progress bar and an exit code that says pass.

**Redirect, do not pipe.** `... > /tmp/.../gate.txt 2>&1` then read the file, or use
`${PIPESTATUS[0]}` if you must pipe. And apply the same rule as above regardless of what
the exit code says: **no `N passed` / `N failed in Xs` line means the run did not finish.**
The summary line is the only evidence that counts; every exit code in this environment has
now been observed lying at least once (143 with an empty file, 0 through a pipe).

A corollary on scheduling: with several task agents running suites at once this mount
carries 30+ pytest processes and runs stretch from ~2 minutes to 19-26. Do not add an
integration-tip gate on top of a full wave — run it when the agents are done, or expect it
to be the one that dies.

#### The fifth shape: a valid summary line proves the run finished, not that it is current

The four shapes above are all about a run that did not finish. This one is about a run that
finished perfectly and is still worthless. Measured 2026-09-05, two full-`tests/` runs both
returned real summary lines:

```
bo0lszkap   33 failed, 742 passed, 15 skipped, 24 warnings in 3323.98s (0:55:23)
bjh8bxbw6   30 failed, 745 passed, 15 skipped, 24 warnings in 3475.57s (0:57:55)
```

Both are stale. `bjh8bxbw6` finished at 11:23:07 after 3475 s, so it *started* at ~10:25 —
and the Task 8b and Task 10 merges landed at 10:48 (`b85df5b9`, `c65788ff`). Neither run
ever saw them. The tell is in their own FAILED lists: both name
`test_run_sfm_falls_back_when_the_video_changed_since_frames_zarr`, and on the tip that
test is `..._since_extraction` (`tests/wrapper/test_vda_context.py:477`) — Task 8b renamed
it. A failure naming a test that no longer exists is proof of a stale tree, and it is the
only cheap check available after the fact.

The two runs also disagree with each other on `tests/preproc`, which is the second tell:
`bjh8bxbw6` reports `test_undistort.py::test_estimate_camera_distortion_tutorial_smoke`
FAILED and `bo0lszkap` does not; `bo0lszkap` reports three `test_video.py` failures and
`bjh8bxbw6` one. Neither set matches Task 10's verified post-merge
`175 passed` on `tests/preproc`. Two stale runs at different points on the branch are two
different trees, so their numbers cannot be reconciled and neither is a baseline.

**On a 55-minute suite, the tree moves underneath the run.** Anything at that duration must
record the tip SHA it started from (`git rev-parse HEAD` in the same command, before the
pytest call) and be discarded if that SHA is no longer an ancestor of the branch tip. Cost
of the check is one line; cost of skipping it is an hour of wall clock and, worse, a
plausible-looking failure list that sends the next task chasing bugs that were already
merged away.

#### The sixth shape: the gate that never returns at all

The five shapes above all produce a *result* that misleads. This one produces nothing,
forever, and it is the most expensive of the six. Measured 2026-09-05 on this plan's own
task wave: **nine poll loops from one session stuck between 2.9 and 8.0 hours, roughly 28
agent-hours, three task agents producing nothing** — one of them sitting on 15 files of
finished but **uncommitted** work.

Two causes compound:

**(a) `pgrep -f` matches the polling shell itself.** The idiom

```bash
until ! pgrep -f "pytest tests/preproc" > /dev/null; do sleep 10; done
```

can never terminate. The poller's own `/proc/<pid>/cmdline` contains the literal string
`pytest tests/preproc`, so `pgrep -f` finds *it*, the condition stays true, and the loop
runs until killed. One instance ran 8 hours. The `kill -0 $(pgrep -f ...)` variant has the
identical bug. If a pgrep is unavoidable, bracket the first character —
`pgrep -f "[p]ytest ..."` — which no longer matches the literal in the poller's own
cmdline.

**(b) Waiting on a summary line that never arrives.** Something ended the pytest early,
leaving a file frozen mid-progress-bar with no summary; a loop then greps that file for
`passed|failed` forever. Two of the nine loops were polling the *same* truncated file,
`after2.txt`, frozen at `[ 66%]`. **Note the cause of the truncation is unproven** — it is
*not* the Bash timeout, which promotes to background rather than killing (see the retraction
above). Contention and an OOM kill are the open candidates. The deadlock does not depend on
the cause: any loop that waits on a summary line hangs forever if the writer stops for any
reason, which is the whole argument against writing one.

**The rule: never write a poll loop for a pytest run.** Use `run_in_background: true` and
let the harness's completion notification wake you. Polling is not a fallback for a slow
gate — it is how a slow gate becomes an infinite one.

Two notes for whoever has to diagnose this from the coordinator seat. `ps` is trustworthy
here (confirm by planting a `sleep` and finding it), and a stuck shell's real command is
readable with `tr '\0' ' ' < /proc/<pid>/cmdline` — the `until` body is visible in it. And
a truncated output file gaining a byte or two is **not** evidence the run is alive; check
it against the process table before believing it.

The recovery is `SendMessage` to the agent, not `kill`. `kill` may be refused by the
permission classifier, and it is not needed: the 600 s Bash ceiling means a livelocked
agent still reaches a tool round between retries, so a message lands.

### The env breaks that are not this plan's fault

Both are recorded in `docs/known-test-failures.md`; neither may be worked around
in source.

1. **gsplat.** Installed is the retired `gsplat-rade` fork at 1.4.0; the repo pins
   upstream `d2f5c0f` (v1.5.3), which is where `gsplat.losses` lives. This blocks
   `tests/splats/` entirely, plus `tests/wrapper/test_splats_stage.py`,
   `tests/wrapper/test_vda_context.py`, `tests/evals/test_eval_splats.py` and
   `tests/mesh/test_absent_confidence.py`. **It cannot be repaired in this
   container**: the build needs `nvcc`, and `/usr/local/cuda-12.1` here is
   runtime-only (no `bin/`), with no nvcc anywhere on the image. Restoring the pin
   needs a container with the CUDA build toolkit.

   `test_vda_context.py` **is no longer one of them.** It never needed gsplat: it
   inherited the dependency by importing `_stub_reconstructor` from
   `test_splats_stage.py`, which imports `SplatsConfig` at module level. Commit
   `9c09585b` moved that helper to `tests/wrapper/_stubs.py`, which imports only
   `FrameStore` and `Reconstructor`. The file now collects 33 tests. **Drop
   `--ignore=tests/wrapper/test_vda_context.py` from every command below.**

   It is the only end-to-end exercise of `Reconstructor._run_sfm`, and the tests that
   reach that path still fail — but on **`instantsfm`, not gsplat**:
   `importlib.metadata.PackageNotFoundError: No package metadata was found for
   instantsfm`, raised at `collab_splats/wrapper/reconstructor.py:1210`. Same env
   root cause (the pruning `uv sync`), different missing package, and unlike gsplat
   `instantsfm` needs no CUDA toolkit — it is a pip install away, blocked only by the
   shared-venv constraint. So the file is now *collectible and diffable*: an
   `--collect-only` run and the `TypeError`/`PackageNotFoundError` split are real
   signal about Task 5's rewrite, where before there was none.

2. **collab_data.** Not installed; fixed by the path entry above.

3. **vismatch.** `vismatch==1.3.1` is declared at `pyproject.toml:70` and absent from
   the venv, with no source on disk to put on `PYTHONPATH`. Fails 16 tests in
   `tests/localization/test_local_matcher.py`. **Do not pip-install it** — the venv is
   shared with several concurrent sessions and `vismatch` hard-pins `uniception==0.1.1`
   and `lightning==2.3.3`, which `pyproject.toml:236-240` deliberately overrides.

### Wave 1 merge gate — measured 2026-09-05

The first run with both `PYTHONPATH` entries and the full `--ignore` list, and so the
first to produce a count rather than `Interrupted: N errors during collection`:

```
29 failed, 1969 passed, 15 skipped, 1453 warnings in 1532.33s (0:25:32)
```

All 29 attributed, **none to wave-1 work**:

| n | tests | cause | status |
|---|---|---|---|
| 16 | `tests/localization/test_local_matcher.py` | `vismatch` not installed | env, break 3 above |
| 3 | `tests/test_cu121_migration.py` | gsplat 1.4.0 vs the v1.5.3 pin | env, break 1 above |
| 3 | `tests/wrapper/test_reconstructor.py` | `configs/base.yaml` drift (`preproc.fps` is 2.0, the test asserts 1.0) | pre-existing, documented |
| 2 | `tests/evals/test_run_vggt_slam.py` | gitignored `third_party/VGGT-SLAM` absent in a worktree | env, documented |
| 2 | `tests/examples/test_run_pipeline_remote.py` | fix lives in a concurrent session's uncommitted edit | pre-existing, documented |
| 2 | `tests/preproc/test_undistort.py` | the dropped `8545a3a1` | **fixed**; the run predates the cherry-pick |

Proof that the 16 are not ours: `git log 9d3dcfcf..HEAD -- collab_splats/localization
tests/localization` is **empty** — no commit in this plan has touched that package.

Two supporting gates on the same tip: the import smoke reports `failed 1`
(`collab_splats.splats`, gsplat), and the duplicate-def scan reports `real duplicates: 0`.

**What this gate does not cover.** `tests/splats/`, `test_splats_stage.py`,
`test_vda_context.py`, `test_eval_splats.py` and `test_absent_confidence.py` were all
`--ignore`d for gsplat when this run was measured.

`test_vda_context.py` has since come off that list (commit `9c09585b`, above): its
gsplat dependency was an accident of a test-helper import, not a real one. Post-fix,
`tests/wrapper/` minus the two genuinely-blocked files is `3 failed, 165 passed` — all
three the documented `configs/base.yaml` drift, so no regression. Within
`test_vda_context.py` itself: `22 failed, 11 passed`, splitting into 3 ×
`TypeError: extract_frames() got an unexpected keyword argument 'frames_zarr'` (Task
8b's scope, and now finally *measurable*) and the rest on `instantsfm`.

So `_run_sfm` coverage is blocked by **instantsfm**, not gsplat. Task 5's rewrite is
still not proven green in this container, but it is now diffable at collection and its
failures have a single named cause instead of an uncollectible module.

### The waves

| wave | tasks | notes |
|---|---|---|
| **0** | **25, 26, 19, 1, 24a** | fully independent; nothing here depends on anything else |
| **1** | **2, 4, 4b, 5, 6, 7, 8** | all need Task 1's `frames.py`. **Task 4b runs strictly after Task 4** (same file) and after 6 and 7 (it calls their new signatures). **Tasks 4 and 5 both edit `wrapper/reconstructor.py`** (this task's premise that the staging block is in `sfm.py` is wrong) and both add a `Reconstructor.images_dir` property — merge them adjacently and expect a conflict. The rest are disjoint. |
| 2 | 3 (reference, no code), **8b**, 9 | 8b needs Task 4 merged and must precede 9; 9 needs all of 4-8b merged |
| 3 | 10 -> 11 -> 12 -> 13 -> 14 | serial: one file, each builds on the last's helpers |
| 4 | 15 -> 16 -> 17 -> 18 | serial: `undistort.py` then `reconstructor.py` |
| 5 | 20 -> 21 -> 22 -> 23 -> 24b | serial: one file, cumulative |
| 6 | 27 -> 28 | 27 sweeps every module the earlier waves touched |

**The wave numbers order dependencies, not calendar time.** Wave 3 opened while waves 1
and 2 were still in flight, because Task 10's only files are
`collab_splats/preproc/sampling.py` and `tests/preproc/test_sampling.py`, and nothing
running touches either. Before starting a later wave early, check the *files*, not the
number:

- Task 13 is the one that genuinely cannot move — it edits `wrapper/reconstructor.py`
  and `configs/base.yaml`, which Tasks 4b and 9 own.
- Task 10 and Task 9 both end up in `tests/preproc/test_sampling.py`, at opposite ends of
  the file (Task 10 appends filter tests; Task 9 edits the `__all__` set around line 361).
  That is a merge to read, not a conflict to avoid.

**Task 8b is new, and covers the same class of gap as Task 25 below.**
Tasks 4-8 each name the production file they convert and the test file that
covers it. Six test modules reach for `FrameStore` without being any task's test
file: `test_vda_context.py`, `test_localization_db_overwrite.py`,
`test_loger_creator.py`, `test_splats_stage.py`, `test_reconstructor_mv_config.py`
and `test_sources.py`. Five of the six patch `FrameStore` in
`reconstructor.py`'s namespace, so Task 8b runs after Task 4 and before Task 9.

**Task 25 moves to wave 0, and this is not only a scheduling choice.**
`preproc/viz.py` imports `FrameStore` and `plot_quality_examples` is its only
consumer. No Phase A task converts `viz.py` — Task 7 is semantics/geometry/mesh,
Task 8 is dashboard/evals/notebooks — so Task 9's step-1 verification grep would
return `collab_splats/preproc/viz.py` and block. Deleting the function first
closes the gap for free. `plot_selection` and `plot_frame_extremes` never touch
the store.

**Task 24 splits.** 24a declares `av` in `pyproject.toml` and runs at wave 0 —
it already resolves transitively, so declaring it early breaks nothing and makes
wave 5's imports legal. 24b sweeps the dead `info=` and needs Tasks 21 and 22.

### What cannot be parallelised, and why

Four single-file chains, each task consuming the previous one's helpers:
`sampling.py` (10-13), `video.py` (20-23), `undistort.py` (15-16).
`reconstructor.py` is the cross-lane lock — Tasks 4, 13, 17 and 18 all edit it
across three different waves, so those four never overlap.

Tasks 15 and 16 have no file overlap with the wave-3 chain and could run beside
it. They stay after Task 14 anyway: the gate exists so a human approves the
selection change before more breaking work lands, and jumping it buys one slot.

### Merging a wave

Each task's branch merges back into the wave's integration branch, in task
order, then the full suite and the dashboard smoke run **once** on the merged
result before the next wave forks. Individual task branches are green in
isolation; only the merge proves they are green together.

Serialise the heavy ones even within a wave: Task 8 re-runs six notebooks and
binds the dashboard smoke port, and Task 15's tests run real SIFT plus
incremental mapping. The container cap is 46.6 GB.

---

## File Structure

**Created:**

| file | responsibility |
|---|---|
| `collab_splats/preproc/frames.py` | The `images/` + `frames.json` store: write, read, list paths, read manifest. Replaces `frame_store.py`. |
| `scripts/migrate_frames_zarr.py` | One-shot converter: existing `frames.zarr` -> `images/` + `frames.json`, no video decode. |
| `tests/preproc/test_frames.py` | Directory-store tests. Replaces `test_frame_store.py`. |
| `tests/preproc/data/make_rotated_fixture.py` | Generates `tests/preproc/data/rotated_90.mp4`, the fixture phase D is gated on. |
| `tests/preproc/test_docstrings.py` | Lints the Args/Returns contract across `preproc.__all__`. |

**Deleted:**

| file | reason |
|---|---|
| `collab_splats/preproc/frame_store.py` | Replaced by `frames.py`. |
| `tests/preproc/test_frame_store.py` | Replaced by `test_frames.py`. |
| `tests/preproc/test_sampling_parity.py` | Asserts the window-argmax substitution the eligible pool replaces; its premise is deleted. **Partly done:** the uniform case was retired when Task 11 merged (`706c2d38`); the fps case stays green until Task 12 retires the fps window, and only then does the file go. |

**Heavily modified:**

| file | change |
|---|---|
| `collab_splats/preproc/sampling.py` | `filter_frame_quality` rewritten; `_sample_by_quality` deleted; three samplers take an eligible pool; gains `context_indices`. |
| `collab_splats/preproc/undistort.py` | `DistortionProfile` deleted; `calibrate_camera` + `undistort_frames` over `pycolmap.Camera`. |
| `collab_splats/preproc/video.py` | ffmpeg decode subprocesses replaced by PyAV; `context_indices` and `decode_context` move out. |
| `collab_splats/preproc/qa.py` | Three pair-motion functions collapse into `compute_pair_motion`. |
| `collab_splats/preproc/viz.py` | Two dead plots deleted. |
| `collab_splats/wrapper/reconstructor.py` | `extract_frames` writes `images/`, then splits by source; `_apply_undistortion` rewritten. |
| `collab_splats/pointcloud/sfm.py` | Gains `decode_context`; `image_path` points at `<scene>/images`. |

---

# Phase A — Storage

**Invariant for this whole phase: frame selection does not change.** Every frame chosen before the phase is chosen after it. Any behavioural difference is a bug introduced here, not an intended change.

---

### Task 1: `preproc/frames.py` — the directory store

**Files:**
- Create: `collab_splats/preproc/frames.py`
- Create: `tests/preproc/test_frames.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/preproc/test_frames.py`:

```python
"""
Directory-backed keyframe store: images/frame_NNNNNN.png + frames.json.
"""

import json

import numpy as np
import pytest

from collab_splats.preproc import frames as fr


def _frames(n=3, h=8, w=12):
    """
    n deterministic RGB frames, each a different flat colour.
    """
    return [np.full((h, w, 3), i * 40 + 5, dtype=np.uint8) for i in range(n)]


def _records(idxs):
    return [{"frame_idx": int(i), "blur_score": float(i) * 1.5} for i in idxs]


def test_frame_idx_from_path_reads_the_padded_stem():
    assert fr.frame_idx_from_path("images/frame_000042.png") == 42


def test_write_then_read_round_trips_rgb(tmp_path):
    images = tmp_path / "images"
    written = fr.write_frames(images, _frames(3), _records([0, 5, 11]), {"method": "uniform"})

    assert [p.name for p in written] == ["frame_000000.png", "frame_000005.png", "frame_000011.png"]

    out = fr.read_frames(images)
    assert out.shape == (3, 8, 12, 3)
    assert out.dtype == np.uint8

    # PNG is lossless and the store is RGB at both boundaries
    np.testing.assert_array_equal(out, np.stack(_frames(3)))


def test_read_frames_selects_by_frame_idx_not_position(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    out = fr.read_frames(images, idxs=[11, 0])
    np.testing.assert_array_equal(out[0], _frames(3)[2])
    np.testing.assert_array_equal(out[1], _frames(3)[0])


def test_read_frames_raises_on_an_index_the_directory_does_not_hold(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})

    with pytest.raises(KeyError, match="7"):
        fr.read_frames(images, idxs=[7])


def test_frame_paths_is_sorted_and_extension_filtered(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([11, 0, 5]), {})
    (images / "notes.txt").write_text("ignore me")

    assert [p.name for p in fr.frame_paths(images)] == [
        "frame_000000.png",
        "frame_000005.png",
        "frame_000011.png",
    ]


def test_frame_paths_on_a_missing_directory_is_empty(tmp_path):
    assert fr.frame_paths(tmp_path / "nope") == []


def test_manifest_carries_records_and_provenance(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(2), _records([0, 5]), {"method": "fps", "fps": 2.0})

    manifest = fr.read_manifest(images)
    assert manifest["schema_version"] == 2
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5]

    # frames.json sits beside images/, not inside it
    assert (tmp_path / "frames.json").exists()
    assert not (images / "frames.json").exists()


def test_manifest_converts_nan_to_null(tmp_path):
    images = tmp_path / "images"
    records = [{"frame_idx": 0, "blur_score": float("nan")}]
    fr.write_frames(images, _frames(1), records, {})

    raw = (tmp_path / "frames.json").read_text()
    assert "NaN" not in raw
    assert json.loads(raw)["frames"][0]["blur_score"] is None


def test_write_frames_clears_a_previous_longer_run(tmp_path):
    images = tmp_path / "images"
    fr.write_frames(images, _frames(3), _records([0, 5, 11]), {})
    fr.write_frames(images, _frames(2), _records([0, 5]), {})

    assert [p.name for p in fr.frame_paths(images)] == ["frame_000000.png", "frame_000005.png"]


def test_write_frames_rejects_records_without_frame_idx(tmp_path):
    with pytest.raises(ValueError, match="frame_idx"):
        fr.write_frames(tmp_path / "images", _frames(1), [{"blur_score": 1.0}], {})


def test_write_frames_rejects_a_length_mismatch(tmp_path):
    with pytest.raises(ValueError, match="against"):
        fr.write_frames(tmp_path / "images", _frames(3), _records([0]), {})


def test_read_manifest_names_the_migration_script_when_absent(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="migrate_frames_zarr"):
        fr.read_manifest(tmp_path / "images")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'collab_splats.preproc.frames'`.

- [ ] **Step 3: Write the implementation**

Create `collab_splats/preproc/frames.py`:

```python
"""
Canonical keyframe store: a COLMAP-style images/ directory plus frames.json.

The preprocess stage decodes a video once and writes images/frame_NNNNNN.png
(lossless, PNG compression 1) beside frames.json, which holds the selection
records and provenance COLMAP has no slot for. Every pixel consumer reads the
directory; path-locked consumers take the directory itself, so nothing stages
a second copy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# The repo's single image-extension listing — reconstructor and feedforward both defer here
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

_MANIFEST_NAME = "frames.json"

# OpenCV's default. Level 9 costs 10x the time for 11% of the size (measured, spec 2.2).
_PNG_COMPRESSION = 1


def _manifest_path(dir) -> Path:
    """
    frames.json, which sits beside the images directory rather than inside it.
    """
    return Path(dir).parent / _MANIFEST_NAME


def _jsonable(value):
    """
    numpy scalar or NaN -> a plain JSON value (NaN becomes null).
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if np.isnan(value) else value
    return value


def frame_idx_from_path(path) -> int:
    """
    Source frame index encoded in a frame_{idx:06d}.<ext> filename.

    Args:
        path: path whose stem ends in the zero-padded source index.

    Returns:
        The source video frame index.
    """
    return int(Path(path).stem.split("_")[-1])


def frame_paths(dir) -> list[Path]:
    """
    Image paths in a frame directory, in filename order.

    Args:
        dir: directory holding frame_NNNNNN.<ext> images.

    Returns:
        Sorted image paths; empty when the directory is missing or holds none.
    """
    dir = Path(dir)
    if not dir.is_dir():
        return []
    return sorted(p for p in dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def write_frames(dir, frames, records, provenance) -> list[Path]:
    """
    Write frames as PNGs and the manifest beside them.

    Args:
        dir: images directory to create; stale frame images in it are removed first.
        frames: RGB uint8 (H, W, 3) frames, one per record.
        records: selection records, each carrying an int 'frame_idx' (source index).
        provenance: descriptive dict stamped into frames.json.

    Returns:
        Written image paths, in record order.
    """
    dir = Path(dir)
    if len(frames) != len(records):
        raise ValueError(f"write_frames: {len(frames)} frames against {len(records)} records")
    if not records or "frame_idx" not in records[0]:
        raise ValueError("write_frames: every record must contain 'frame_idx' (source video index)")

    dir.mkdir(parents=True, exist_ok=True)

    # A re-run selecting fewer frames must not leave the previous run's extras behind,
    # where frame_paths would serve them as if they were this run's selection
    for stale in frame_paths(dir):
        stale.unlink()

    # Store is RGB at the boundary; cv2 writes BGR
    paths: list[Path] = []
    for frame, record in zip(frames, records):
        path = dir / f"frame_{int(record['frame_idx']):06d}.png"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION],
        )
        paths.append(path)

    # Row-oriented: a reader wants one frame's record, not one column
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "frames": [{k: _jsonable(v) for k, v in record.items()} for record in records],
    }
    _manifest_path(dir).write_text(json.dumps(manifest, indent=2))

    logger.info("frames: wrote %d PNGs to %s", len(paths), dir)
    return paths


def read_frames(dir, idxs=None) -> np.ndarray:
    """
    Read frames from an images directory as one RGB stack.

    Args:
        dir: images directory holding frame_NNNNNN.<ext>.
        idxs: SOURCE frame indices to read, in the order given; None reads every
            frame in filename order.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    paths = frame_paths(dir)
    if not paths:
        raise FileNotFoundError(f"read_frames: no frame images in {dir}")

    # idxs select by source frame_idx, never by row position — a caller holding a
    # frame_idx from a record must not have to know where it landed in the directory
    if idxs is not None:
        by_idx = {frame_idx_from_path(p): p for p in paths}
        missing = [int(i) for i in idxs if int(i) not in by_idx]
        if missing:
            raise KeyError(f"read_frames: frame_idx {missing[:5]} not in {dir}")
        paths = [by_idx[int(i)] for i in idxs]

    return np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])


def read_manifest(dir) -> dict:
    """
    Selection records and provenance written beside an images directory.

    Args:
        dir: images directory; frames.json sits in its parent.

    Returns:
        {'schema_version', 'provenance', 'frames'}.
    """
    path = _manifest_path(dir)
    if not path.exists():
        raise FileNotFoundError(
            f"read_manifest: {path} not found. A scene written before this format holds "
            "frames.zarr — convert it with scripts/migrate_frames_zarr.py."
        )
    return json.loads(path.read_text())
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/preproc/frames.py tests/preproc/test_frames.py
isort collab_splats/preproc/frames.py tests/preproc/test_frames.py
git commit --only collab_splats/preproc/frames.py tests/preproc/test_frames.py \
  -m "feat(preproc): images/ + frames.json store to replace FrameStore"
```

---

### Task 2: `scripts/migrate_frames_zarr.py`

> **LANDED** as `30977222`, with one real defect corrected.
>
> **This task's test import cannot resolve as written.**
> `from scripts.migrate_frames_zarr import migrate_scene` raises
> `ModuleNotFoundError` even with the repo root on `sys.path`. The `uniception`
> dependency (MapAnything) installs a **regular** top-level package named
> `scripts` into site-packages, and under PEP 420 a regular package anywhere on
> `sys.path` beats a namespace directory earlier on it — so `import scripts`
> resolved to site-packages, not the repo.
>
> **That diagnosis was wrong, and the fix it produced only worked in isolation.**
> The shadow is not `uniception`. It is in-repo: `tests/evals/test_datasets.py:10`
> runs `sys.path.insert(0, <repo>/evals)` at import time, which puts
> `evals/scripts/` — a real package with its own `__init__.py` — ahead of the
> repo's `scripts/` for the rest of the pytest session. Collect
> `tests/preproc/test_frames.py` on its own and it passes; collect it after
> `tests/evals` and `import scripts` is already bound to `evals/scripts`, so
> `scripts.migrate_frames_zarr` does not exist. No `__init__.py` wins that,
> because both contenders are regular packages and the loser is whichever runs
> second.
>
> Landed fix: `scripts/__init__.py` is **deleted**, and `tests/preproc/test_frames.py`
> loads the script through `importlib.util.spec_from_file_location` under the private
> name `_migrate_frames_zarr`. The script is a CLI, not a library — the error message in
> `frames.py` already names it by path — so it needs no importable package and the
> collision surface goes to zero. Any later task importing a repo script from a test
> should do the same rather than re-adding the package.
>
> (Superseded) Original fix: adding `scripts/__init__.py` (docstring only), making the repo's
> `scripts/` a regular package that wins on path order. Verified first that
> nothing in `uniception`/`mapanything` imports `scripts.*` at runtime and that
> the repo had no other `import scripts`, so the shadowing is one-directional
> and inert. `[tool.setuptools.packages] find` includes only `collab_splats`, so
> nothing new is installed. **This is a third file this task's inventory does not
> list**, and any later task importing a repo script from a test inherits the fix.
>
> Style: this task puts `import zarr` and the `migrate_scene` import inside the
> test function. CLAUDE.md forbids inline imports; both were hoisted.

**Files:**
- Create: `scripts/migrate_frames_zarr.py`
- Test: `tests/preproc/test_frames.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/preproc/test_frames.py`:

```python
def test_migrate_converts_a_zarr_store_without_decoding(tmp_path):
    """
    The migration script reads frames.zarr and writes images/ + frames.json.
    """
    import zarr

    from scripts.migrate_frames_zarr import migrate_scene

    # Build a minimal frames.zarr by hand — the same shape FrameStore.create wrote
    scene = tmp_path / "scene"
    scene.mkdir()
    imgs = np.stack(_frames(3))
    store = zarr.open(str(scene / "frames.zarr"), mode="w")
    store.create_array("images", data=imgs, chunks=(1, *imgs.shape[1:]))
    store.create_array("frame_idx", data=np.array([0, 5, 11]))
    store.create_array("blur_score", data=np.array([1.0, 2.0, 3.0]))
    store.attrs["record_keys"] = ["blur_score", "frame_idx"]
    store.attrs["provenance"] = {"method": "fps", "fps": 2.0}
    store.attrs["schema_version"] = 1

    migrate_scene(scene)

    out = fr.read_frames(scene / "images")
    np.testing.assert_array_equal(out, imgs)

    manifest = fr.read_manifest(scene / "images")
    assert manifest["provenance"] == {"method": "fps", "fps": 2.0}
    assert [r["frame_idx"] for r in manifest["frames"]] == [0, 5, 11]
    assert manifest["frames"][1]["blur_score"] == 2.0
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py::test_migrate_converts_a_zarr_store_without_decoding -v
```

Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.migrate_frames_zarr'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/migrate_frames_zarr.py`:

```python
"""
Convert a scene's frames.zarr into images/ + frames.json with no video decode.

Existing processed scenes hold frames.zarr and no images/. Re-running preproc
would re-decode the source video (98 s cold per scene); this reads the store
instead. The old store is left in place — delete it once the scene reads back.

Usage:
    python scripts/migrate_frames_zarr.py <scene_dir> [<scene_dir> ...]
    python scripts/migrate_frames_zarr.py --all <processed_root>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr

from collab_splats.preproc import frames as fr

logger = logging.getLogger(__name__)


def migrate_scene(scene_dir: Path) -> int:
    """
    Write images/ + frames.json from a scene's frames.zarr.

    Args:
        scene_dir: directory holding frames.zarr.

    Returns:
        Number of frames written.
    """
    scene_dir = Path(scene_dir)
    store_path = scene_dir / "frames.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"no frames.zarr in {scene_dir}")

    store = zarr.open(str(store_path), mode="r")
    images = store["images"][:]
    keys = list(store.attrs.get("record_keys", ["frame_idx"]))

    # Columnar arrays back into row dicts, one per selected frame
    columns = {k: store[k][:] for k in keys}
    records = [{k: columns[k][row] for k in keys} for row in range(images.shape[0])]

    provenance = dict(store.attrs.get("provenance", {}))
    fr.write_frames(scene_dir / "images", [np.asarray(f) for f in images], records, provenance)
    return int(images.shape[0])


def main(argv=None) -> int:
    """
    CLI entry point.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path, help="scene directories, or a root with --all")
    parser.add_argument("--all", action="store_true", help="treat each argument as a root of scene directories")
    args = parser.parse_args(argv)

    # --all expands each root into the scenes under it that still hold a store
    targets: list[Path] = []
    for arg in args.scenes:
        if args.all:
            targets += sorted(p.parent for p in Path(arg).glob("*/frames.zarr"))
        else:
            targets.append(Path(arg))

    for scene in targets:
        n = migrate_scene(scene)
        logger.info("migrated %s (%d frames)", scene, n)

    logger.info("migrated %d scene(s)", len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_frames.py -v
```

Expected: 13 passed.

- [ ] **Step 5: Format and commit**

```bash
black scripts/migrate_frames_zarr.py tests/preproc/test_frames.py
isort scripts/migrate_frames_zarr.py tests/preproc/test_frames.py
git commit --only scripts/migrate_frames_zarr.py tests/preproc/test_frames.py \
  -m "feat(preproc): frames.zarr -> images/ migration script"
```

---

### Task 3: The call-site translation table

**Files:**
- Read only: this task produces no code. It is the reference Tasks 4-9 apply.

Every `FrameStore` use in the repo is one of these shapes. Apply the right-hand side mechanically. `fr` is `from collab_splats.preproc import frames as fr`, and `images_dir` is `<scene>/images`.

| old | new |
|---|---|
| `FrameStore.create(z, frames, records, provenance=prov)` | `fr.write_frames(images_dir, frames, records, prov)` |
| `FrameStore.open(z)` | *(delete — there is no handle any more)* |
| `store.images()` | `fr.read_frames(images_dir)` |
| `store.images(idxs)` | `fr.read_frames(images_dir)[idxs]` — **positions**, so index the stack |
| `store.image(i)` | `fr.read_frames(images_dir)[i]`, or `cv2.imread` on `fr.frame_paths(images_dir)[i]` in a loop |
| `store.image_by_frame_idx(fi)` | `fr.read_frames(images_dir, idxs=[fi])[0]` |
| `store.has_frame_idx(fi)` | `fi in {fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)}` |
| `store.frame_indices()` | `np.array([fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)])` |
| `store.record(i)` | `fr.read_manifest(images_dir)["frames"][i]` |
| `store.provenance()` | `fr.read_manifest(images_dir)["provenance"]` |
| `len(store)` | `len(fr.frame_paths(images_dir))` |
| `store.export(tmp)` / `store.export(tmp, ext="jpg")` | *(delete the call and the tmpdir; pass `images_dir` itself)* |
| `FrameStore.frame_idx_from_path(p)` | `fr.frame_idx_from_path(p)` |
| a `FrameStore \| Path` parameter | `Path` |
| `sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})` | `fr.frame_paths(d)` |

**Two traps:**

1. `store.images(idxs)` took **row positions**; `fr.read_frames(dir, idxs=...)` takes **source frame indices**. They are the same list only when every frame was selected. Translate positional calls to `fr.read_frames(dir)[idxs]`, never to the `idxs=` keyword.
2. `read_frames` reads the whole directory each call. A loop that called `store.image(i)` per iteration must hoist one `read_frames` above the loop, or iterate `fr.frame_paths` and `cv2.imread` one path at a time — not call `read_frames` per iteration.

- [x] **Step 1: Confirm the surface before starting**

```bash
git grep -ln 'FrameStore' -- 'collab_splats/*' 'evals/*' 'scripts/*'
```

Expected: 11 files — `dashboard/localize.py`, `dashboard/pipeline.py`, `geometry/loop_closure/wrapper.py`, `geometry/metrics.py`, `mesh/utils.py`, `pointcloud/feedforward/base.py`, `preproc/__init__.py`, `preproc/frame_store.py`, `preproc/viz.py`, `semantics/features/base.py`, `wrapper/reconstructor.py`, plus `evals/datasets.py`, `evals/scripts/eval_splats.py`, `evals/scripts/eval_verification.py`.

If the list differs, a concurrent session has moved things — reconcile before continuing.

> **Measured on the integration tip, 2026-09-05 — eight files, not fourteen.**
>
> ```
> collab_splats/dashboard/localize.py        collab_splats/preproc/frame_store.py
> collab_splats/dashboard/pipeline.py        collab_splats/wrapper/reconstructor.py
> collab_splats/preproc/__init__.py          evals/datasets.py
> evals/scripts/eval_splats.py               evals/scripts/eval_verification.py
> ```
>
> The six that dropped off were converted by wave 1, each attributable to a commit:
> `geometry/loop_closure/wrapper.py`, `geometry/metrics.py`, `mesh/utils.py` and
> `semantics/features/base.py` by `bab3a47b`; `pointcloud/feedforward/base.py` by
> `8047f9fe`; `preproc/viz.py` by `a93da04e`, which deleted the two plot functions
> outright rather than converting them.
>
> **Every one of the remaining eight is already owned**, which is the real result here —
> nothing was orphaned by the wave-1 merge:
>
> | files | owner | state |
> |---|---|---|
> | `wrapper/reconstructor.py` | Task 4b | in flight |
> | the two `dashboard/` + the three `evals/` | Task 8 | in flight |
> | `preproc/frame_store.py`, `preproc/__init__.py` | Task 9 | deletes them; must run last |
>
> So the wave-2 ordering in the table above (8b before 9, 9 after everything) is
> confirmed by measurement rather than by reading. Re-run this grep before dispatching
> Task 9: it must show exactly the two `preproc/` files, and if it shows any other, that
> file's owning task did not land.

---

### Task 4: `reconstructor.extract_frames` writes `images/`

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`extract_frames`, `_apply_undistortion`, `_LazyFrames`, and the `frames_zarr` parameter name)
- Test: `tests/wrapper/test_reconstructor_preprocess.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor_preprocess.py`:

```python
def test_extract_frames_writes_an_images_dir_and_manifest(tmp_path, monkeypatch):
    """
    Image-directory input lands as images/frame_NNNNNN.png + frames.json.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc import frames as fr
    from collab_splats.wrapper.reconstructor import extract_frames

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"img_{i}.png"), np.full((8, 12, 3), i * 40 + 5, np.uint8))

    scene = tmp_path / "scene"
    scene.mkdir()

    n = extract_frames(src, scene / "images", "uniform", None, None, 10)

    assert n == 3
    assert [p.name for p in fr.frame_paths(scene / "images")] == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]
    assert fr.read_manifest(scene / "images")["provenance"]["method"] == "dir"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py::test_extract_frames_writes_an_images_dir_and_manifest -v
```

Expected: FAIL — `extract_frames` still writes a zarr store, so `frame_paths` returns `[]`.

- [ ] **Step 3: Rename the parameter and swap the writer**

In `collab_splats/wrapper/reconstructor.py`:

1. Rename the `frames_zarr: Path` parameter of `extract_frames` to `images_dir: Path` throughout the function, and at every call site inside the module. The sibling paths it derives change with it:

```python
# was: frames_zarr.parent / "video_quality_report.json"
report_path = images_dir.parent / "video_quality_report.json"
```

2. Replace both `FrameStore.create(...)` calls (the directory branch and the video branch) with:

```python
frames.write_frames(images_dir, frame_arrays, records, prov)
```

3. Replace the directory-branch extension listing:

```python
# was: exts = {".jpg", ".jpeg", ".png"}
#      frames = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
source_paths = frames.frame_paths(input_path)
if not source_paths:
    raise ValueError(f"No images ({list(frames.IMAGE_EXTS)}) found in directory {input_path}")
frame_arrays = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in source_paths]
```

4. Update the imports at the top of the module:

```python
from collab_splats.preproc import frames
```

and delete `from collab_splats.preproc.frame_store import FrameStore`.

5. Replace `_LazyFrames` with a cached reader. Find the class and substitute:

```python
@lru_cache(maxsize=1)
def _scene_frames(images_dir: Path) -> np.ndarray:
    """
    Every frame of a scene as one RGB stack, cached per directory.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        (N, H, W, 3) uint8 RGB.
    """
    return frames.read_frames(images_dir)
```

with `from functools import lru_cache` at the top. Replace each `_LazyFrames(...)` construction with `_scene_frames(images_dir)` and each `lazy[i]` subscript with `_scene_frames(images_dir)[i]`.

- [ ] **Step 4: Run the wrapper suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
```

Expected: the new test passes. Other tests in this file that assert on `frames.zarr` will fail — fix them in the same task by applying the Task 3 table to their assertions.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/
isort collab_splats/wrapper/reconstructor.py tests/wrapper/
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/ \
  -m "refactor(wrapper): extract_frames writes images/ + frames.json"
```

**Outcome (executed).** Landed as two commits on `preproc-t4`: `d88814d7` (the rename) and
`8545a3a1` (callers outside the plan's inventory). Suite at the time:
`4 failed, 332 passed` over `tests/wrapper/ tests/preproc/`, all four failures pre-existing
and recorded in `docs/known-test-failures.md`.

**Only `d88814d7` reached `preproc/integration` in the wave-1 merge. `8545a3a1` was dropped**
and re-applied afterwards as `c91a2526`. Until then, `tests/preproc/test_undistort.py` still
passed `frames_zarr=` into an `extract_frames` that no longer accepted it — a `TypeError` in
two tests that the merge introduced and nothing in the merge reported. Cherry-pick when a
task's branch carries more than one commit; a merge of the branch tip is not the same thing.

Four plan errors, all in Task 4's own steps:

1. > "and delete `from collab_splats.preproc.frame_store import FrameStore`."

   Wrong — that breaks module import. About 15 other `FrameStore` uses remain in
   `reconstructor.py`. The import was kept with a transitional comment; it also keeps
   `patch.object(R, "FrameStore")` alive in two test files. **Task 4b is the task that
   actually removes it.**

2. > "each `lazy[i]` subscript with `_scene_frames(images_dir)[i]`"

   No `lazy[i]` subscript exists anywhere in the module. `_LazyFrames` had exactly one
   construction site and was handed whole to `verify_reconstruction`.

3. The Files list is wrong in both directions. `_apply_undistortion` needed **no** change
   (it takes frame arrays, never a store path), and the test inventory is short by four
   files: `tests/wrapper/test_reconstructor.py`, `tests/wrapper/test_verify_stage.py`,
   `tests/preproc/test_undistort.py`, `tests/wrapper/test_vda_context.py` (left broken —
   uncollectible behind the gsplat error, see "The env breaks that are not this plan's
   fault"; it is Task 8b's).

4. > `black collab_splats/wrapper/reconstructor.py tests/wrapper/`

   Neither black nor isort was run. A `black --line-length 120 --diff` dry run on
   `reconstructor.py` showed 117 changed lines, **none** of them lines this task touched —
   pure pre-existing drift against the venv's newer black. `git commit --only tests/wrapper/`
   would also have swept other sessions' unstaged edits in that directory; explicit file
   paths were committed instead. Both hazards are standing repo rules the plan's boilerplate
   steps contradict.

**Outbound keyword arguments — a gap the task steps never mention.** Renaming the parameter
means every call *out* of `reconstructor.py` into a module Task 7 renamed must move too.
Three of five moved here: `pointcloud_to_mesh(images_dir=...)`, `_run_tsdf_mesh` (its own
param renamed, caller passes `self.images_dir`), and
`build_reconstruction_quality_report(images_dir=...)`. `_run_feedforward` is Task 6's;
`extract_frames` was already correct. The native-resolution branch no longer opens a store
at all — it forwards the directory, since `_feedforward_to_tsdf_inputs` does its own
`frame_paths` / `read_frames`.

**Phase A invariant — differentially measured, not asserted.** The pre-change package was
materialised read-only with `git archive HEAD collab_splats | tar -x` (no checkout, no stash,
no branch switch), then one probe script ran against each tree over a synthetic 90-frame video
with a deliberately blurred band at frames 40-49, across four configs: image directory,
`uniform`, `fps 2.0`, `optical_flow`. Each run dumped `n`, selected `frame_idx`, `blur_score`,
array shape, and a SHA-256 of the decoded RGB stack. `diff` → **byte-identical**, pixels
included: indices matched exactly (including the blur-substituted frame 42), and the PNG
round-trip proved lossless against the bytes the zarr held.

---

### Task 4b: the rest of `reconstructor.py` — five call sites Task 4 left open

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Test: `tests/wrapper/test_reconstructor.py`

**Why this task exists.** Task 4's scope is `extract_frames`, `_apply_undistortion`,
`_LazyFrames` and the `frames_zarr` parameter name. It landed exactly that and kept
`Reconstructor.frames_zarr` as a transitional read-only property for "the call sites plan
tasks 5-8 still own". They do not own them: Tasks 5-8 convert `sfm.py`, `feedforward/base.py`,
`semantics`/`geometry`/`mesh`, and `dashboard`/`evals`/notebooks. **Five `FrameStore.open`
call sites live inside `reconstructor.py` itself and belong to no task.** After Task 4 merges,
`preprocess()` writes `images/` and nothing writes `frames.zarr`, so every one of them opens a
store that does not exist. Run this immediately after Task 4.

**One of the five is worse than a missing file.** `_extract_2d_features` forwards its path
positionally into `BaseFeatureExtractor.extract_and_cache_from_zarr`, whose first parameter
Task 7 renamed from `frames_zarr` to `images_dir`. Positional, so there is no `TypeError` — the
semantics stage would read `<scene>/frames.zarr` as an image directory, find no image files,
and cache zero features. Silent.

- [ ] **Step 1: Confirm the five sites**

```bash
git grep -n 'FrameStore' -- collab_splats/wrapper/reconstructor.py
```

Expected: the module-level import, a function-local import, and opens inside
`_run_feedforward`, `_build_localization_db`, `_run_sfm` (two) and `splats`, plus one
`FrameStore.frame_idx_from_path` static call.

- [ ] **Step 2: Convert each site**

`fr` is `from collab_splats.preproc import frames as fr`. `_scene_frames(images_dir)` already
exists in this module (Task 4 added it) — an `lru_cache`'d whole-directory read. Prefer it over
a bare `fr.read_frames` wherever the same directory is read more than once per run.

| site | old | new |
|---|---|---|
| `_run_feedforward` | `store = FrameStore.open(frames_zarr)` | delete the open; rename the parameter to `images_dir` |
| `_run_feedforward` | `n_frames = len(store)` | `n_frames = len(fr.frame_paths(images_dir))` |
| `_run_feedforward` | `creator.reconstruct(store, output_dir)` | `creator.reconstruct(images_dir, output_dir)` — Task 6 made `setup_inference(source: Path)` take the directory |
| `_extract_2d_features` | `extract_and_cache_from_zarr(frames_zarr, cache_dir)` | rename the parameter to `images_dir` and pass it; the callee's parameter is already `images_dir` |
| `_build_localization_db` | `store.frame_indices()` | `[fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)]` |
| `_build_localization_db` | `(store.image_by_frame_idx(fi) for fi in ...)` | `(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in fr.frame_paths(images_dir))` — keep it a genexpr, the cache-hit path must stay zero-read |
| `_run_sfm` | `FrameStore.open(self.frames_zarr).frame_indices()` | the same list comprehension over `fr.frame_paths(self.images_dir)` |
| `_run_sfm` | `store: FrameStore` parameters on `_ensure_vda_depth` and the COLMAP-name helper | `images_dir: Path` |
| `splats` | `store.image_by_frame_idx(...)` per frame | `_scene_frames(self.images_dir)` indexed by position, after mapping `result.image_paths` through `fr.frame_idx_from_path` |
| module | `FrameStore.frame_idx_from_path(img_path)` | `fr.frame_idx_from_path(img_path)` |

- [ ] **Step 3: The `.jpg` ids in `_build_localization_db` are a join key — do not change them blind**

```python
ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
```

These become the localization DB's image names. The store now writes `.png`, so the extension
no longer matches the file on disk. Before touching it, find what reads these ids back:

```bash
git grep -n 'local_features' -- collab_splats/ | grep -i 'name\|id'
git grep -n 'frame_idx_from_path\|\.jpg' -- collab_splats/localization/
```

If nothing parses the extension — if the ids are opaque labels joined only by their
`frame_NNNNNN` stem — **leave the string exactly as it is** and add a comment saying so.
Changing it silently invalidates every localization DB already on disk and under
`environments-processed/`, which is a migration, not a rename. If something does parse the
extension, that is a finding: stop and report it rather than deciding alone.

- [ ] **Step 4: Delete the transitional property and both imports**

Remove `Reconstructor.frames_zarr`, the module-level
`from collab_splats.preproc.frame_store import FrameStore`, and the function-local one in
`_build_localization_db`. Then:

```bash
git grep -n 'FrameStore\|frames_zarr' -- collab_splats/wrapper/reconstructor.py
```

Expected: no matches. Drop the `rec.frames_zarr` assertion from
`test_reconstructor_images_dir` and convert the `rec.frames_zarr` uses elsewhere in
`tests/wrapper/test_reconstructor.py` to `rec.images_dir`.

- [ ] **Step 5: Run**

```bash
PYTHONPATH=$(pwd):/workspace/collab-data /opt/venv/reconstruction/bin/python -m pytest \
  tests/wrapper tests/pointcloud -q --ignore=tests/wrapper/test_splats_stage.py
```

The one ignore is the gsplat env break recorded in `docs/known-test-failures.md`, not
anything this task caused. **Collection errors abort the entire pytest run**, so without the
ignore you get zero tests, not a partial result.

`test_vda_context.py` used to be ignored here too. It is not any more — commit `9c09585b`
moved `_stub_reconstructor` out of `test_splats_stage.py` and into `tests/wrapper/_stubs.py`,
which imports no splats code, so the file collects. Its `_run_sfm` tests still fail, on
`instantsfm` rather than gsplat; that is expected in this container and is not a signal
about this task. The second `PYTHONPATH` entry supplies `collab_data`, declared in
`pyproject.toml` but absent from the venv.

- [ ] **Step 6: Commit**

```bash
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py \
  -m "refactor(wrapper): the reconstructor call sites Task 4 left on frames.zarr"
```


> ## Task 4b outcomes — landed `f6b90efd`, merged `5b6b171e` 2026-09-05
>
> Verified post-merge on `preproc/integration`. `grep 'FrameStore\|frames_zarr\|frames\.zarr'
> collab_splats/wrapper/reconstructor.py` → **no matches**. Six plan errors, one real bug the
> plan flagged only in passing, and one deliberate deviation.
>
> ### Six plan errors, all confirmed against `git show 58bc38b4:collab_splats/wrapper/reconstructor.py`
>
> | # | plan said | ground truth | consequence if followed |
> |---|---|---|---|
> | 1 | Step 2 row `_run_sfm` — `FrameStore.open(self.frames_zarr).frame_indices()` | that line is **1046**, inside `_load_pointcloud_from_disk` (def at 1033), not `_run_sfm` | you convert the wrong function and leave the real one open |
> | 2 | Step 2 module row — "one `FrameStore.frame_idx_from_path` static call" | **three**: 1135, 1782, 1815 | two survive the conversion and raise `NameError` after Step 4 deletes the import |
> | 3 | Step 2 table has no row for the stage callers | `self.frames_zarr` is passed into helpers at **1550** (`extract_semantics`) and **1669** (`build_localization_db`) | both are property *reads*, so pyflakes sees nothing; deleting the property in Step 4 turns them into `AttributeError` at runtime only |
> | 4 | "the COLMAP-name helper" | `_sfm_result_from_reconstruction` | unnameable step; the executor has to guess which helper |
> | 5 | Step 5's ignore list | insufficient as originally written (two ignores) | *already corrected in this plan before 4b ran — recorded here only so the sixth is not read as isolated* |
> | 6 | Step 6 commit line names **two** files | the change needs **six**: `reconstructor.py`, `test_reconstructor.py`, `test_loger_creator.py`, `test_localization_db_overwrite.py`, `test_reconstructor_mv_config.py`, `test_sfm_result.py` | a `--only` commit with the plan's two paths leaves four files uncommitted and the branch un-mergeable |
>
> Error 3 is the dangerous shape and worth naming: **a plan step that says "delete this property"
> must inventory its readers, not its importers.** `grep FrameStore` finds none of them, pyflakes
> finds none of them, and a passing test suite finds none of them unless a test happens to reach
> that line.
>
> ### The silent bug — `_extract_2d_features` would have cached zero features
>
> The plan mentioned this in prose but gave it no step. It is real, and verified on the merged tree:
>
> - `semantics/features/base.py:177` — `extract_and_cache_from_zarr(self, images_dir: Path, cache_dir: Path)`. Task 7 renamed the first parameter from `frames_zarr`.
> - `reconstructor.py:533` and `dashboard/pipeline.py:113` both forward **positionally**.
>
> So there is no `TypeError`. The semantics stage would have passed `<scene>/frames.zarr` as an
> image directory; `frame_paths` returns `[]` for a missing directory rather than raising; the
> cache is written empty and the stage reports success. Fixed here.
>
> The method is still *named* `extract_and_cache_from_zarr` while reading a directory of PNGs.
> That is not a gap in this plan: `docs/superpowers/specs/2026-09-05-semantics-cleanup-design.md`
> §3.3 already retires it to `utils.extract_feature_cache`. Leave it alone here — renaming a
> public surface from two plans at once is how the call sites get lost.
>
> ### The `.jpg` localization ids stay — with evidence
>
> Step 3 asked for a decision rather than prescribing one. The search it specifies found no consumer
> that parses the extension: every reader joins on the `frame_NNNNNN` stem. So the ids keep `.jpg`
> even though `write_frames` now writes `.png`, and a comment at the site says why. Rewriting them
> would invalidate every localization DB already on disk and under `environments-processed/` — a
> migration, not a rename.
>
> ### Deliberate deviation — `_ensure_vda_depth` does **not** use `_scene_frames`
>
> Step 2 says to prefer the `lru_cache`'d `_scene_frames(images_dir)` wherever a directory is read
> more than once. `_ensure_vda_depth` reads straight through `frames.read_frames` instead
> (`reconstructor.py:1307-1313`), because the `del keyframes` two lines later has to actually
> release the stack — ~1.9 GB at 300×1080p — before InstantSfM's global mapping runs, and an
> `lru_cache` would pin it for the life of the process. Container cap is 46.6 GB. The comment at
> the site records this so a later reader does not "fix" it back.
>
> `_scene_frames` **is** used where the plan intends it: `reconstructor.py:1742` and `:1793`, both
> on the `verify()` path where a second call in the same session must not re-decode.
>
> ### A name-shadowing class this task had to clear first
>
> `reconstructor.py` imports `from collab_splats.preproc import frames, get_video_info` at line 35.
> Three local variables were named `frames` and shadowed the module inside functions that now need
> it. Renamed: `frame_entries` (1116-1135) and `keyframes` (1312, 1373). This is a general hazard
> for every remaining task that introduces the `frames` module into a file — check for a local
> `frames` before adding the import, because the shadow only breaks at the first module-attribute
> access, which may be in a branch no test takes.
>
> ### One vacuous assertion, faithfully renamed rather than fixed
>
> `test_reconstructor.py:1360` was `rec.frames_zarr.mkdir(parents=True, exist_ok=True)` under the
> comment "Both upstream stages already complete on disk". It is now `rec.images_dir.mkdir(...)`.
> The `mkdir` proves nothing either way: `rec.preprocess` is stubbed to append unconditionally and
> the assertion is `calls == ["preproc", "pointcloud", "localize"]`, so the directory's existence
> never enters the test. Left as-is — renaming it was in scope, repairing it was not. Flagged for
> whoever owns `test_reconstructor.py` next.
>
> ### Equality check
>
> The branch and a clean `git archive` of its base `58bc38b4` both give
> **29 failed, 1956 passed, 15 skipped** with byte-identical FAILED lists. The four pyflakes
> findings in `test_reconstructor.py` (`shutil`, `subprocess`, `warnings` unused; `result` assigned
> never used) are pre-existing — running pyflakes on `git show 58bc38b4:tests/wrapper/test_reconstructor.py`
> produces the identical four. This task added no lint.
---

### Task 5: `pointcloud/sfm.py` drops the JPEG staging step

> **LANDED** as `a31ec6fa`, with four corrections — the most consequential of
> any wave-1 task.
>
> 1. **Wrong file, and the wave schedule is wrong because of it.** This task
>    says the staging block is in `collab_splats/pointcloud/sfm.py`. It is not;
>    `sfm.py` had no staging code and no `FrameStore` reference at all. The block
>    lives in `wrapper/reconstructor.py::_run_sfm`. **Tasks 4 and 5 are therefore
>    NOT file-disjoint**, contrary to the wave-1 row of the Parallel Execution
>    Schedule above. Both also add a `Reconstructor.images_dir` property. Expect
>    a merge conflict and resolve it by keeping one definition.
>
> 2. **Deleting the staging block verbatim silently corrupts re-runs.** That
>    block was the SIFT database's ONLY cache invalidation:
>    ```python
>    if staged != names:
>        shutil.rmtree(image_dir, ignore_errors=True)
>        (backend_dir / "colmap" / "instantsfm.db").unlink(missing_ok=True)
>    ```
>    Without it, a re-run with a different frame selection reuses SIFT features
>    extracted from frames no longer in the scene. Landed replacement:
>    `_sift_database_valid` takes `image_names` and compares them against the
>    DB's own `images` table — stricter than the old check, since it keys on what
>    was actually extracted rather than on a staged copy. Four tests cover it,
>    and it restores parity with the eval path, which already had the equivalent.
>
> 3. **The replacement code has nowhere to go as specified.** This task says to
>    "pass `_sfm_image_dir(images_dir)` where the staged directory was passed",
>    but `InstantSfMCreator.reconstruct(data_dir)` derives its image directory
>    from `data_dir` through upstream `ReadData`, and `data_dir` is
>    `<scene>/instantsfm` while the images are at `<scene>/images`. Landed as an
>    optional `images_dir` parameter that overrides `path_info.image_path` after
>    `ReadData`, beside the existing `database_path`/`output_path` overrides.
>    `evals/scripts/eval.py:367` keeps its own staging; `images_dir=None` leaves
>    it byte-identical.
>
> 4. `_sfm_image_dir` as specified is a bare identity function, which CLAUDE.md
>    forbids. It absorbed the existence guard that was inline in `reconstruct`.
>
> **Frame content reaching COLMAP changes, and this plan never says so.** Staging
> re-encoded every frame to JPEG; COLMAP now reads the store's lossless PNGs.
> SIFT keypoints and every downstream model differ slightly from any
> reconstruction built before this commit. Phase A's invariant holds in the sense
> it states — the same frames are selected — but a gate that diffs SfM *output*
> rather than the selection will show a difference by design.
>
> **`_run_sfm` landed without integration coverage.** `tests/wrapper/test_vda_context.py`
> is the only suite exercising it end-to-end, and it cannot even be collected in
> this venv: installed `gsplat` is **1.4.0**, while `collab_splats/splats/losses.py`
> does `from gsplat import losses`, which exists only in the repo's pinned
> `d2f5c0f` (v1.5.3). Same cause blocks `tests/wrapper/test_splats_stage.py` and
> `tests/mesh/test_absent_confidence.py`. Fix the environment before trusting any
> green run over the splats or VDA paths.
>
> **Correction, 2026-09-05 — the mechanism above is wrong for `test_vda_context.py`.**
> That file does not import gsplat. It imported `_stub_reconstructor` from
> `test_splats_stage.py`, and *that* file imports `SplatsConfig`; the dependency was
> inherited through a test helper. Commit `9c09585b` moved the helper to
> `tests/wrapper/_stubs.py` and the file collects again (33 tests). Its `_run_sfm`
> tests still fail, but on `importlib.metadata.PackageNotFoundError: No package
> metadata was found for instantsfm` at `wrapper/reconstructor.py:1210`. So `_run_sfm`
> coverage is blocked by **instantsfm**, not gsplat — a pip install rather than a
> container rebuild, held only by the shared-venv constraint.

**Files:**
- Modify: `collab_splats/pointcloud/sfm.py`
- Test: `tests/wrapper/test_sfm_result.py`

- [ ] **Step 1: Find the staging block**

```bash
git grep -n 'export\|instantsfm/images\|image_path' -- collab_splats/pointcloud/sfm.py
```

The block writes `<scene>/instantsfm/images/frame_NNNNNN.jpg` from the store and then hands that directory to InstantSfM as `image_path`.

- [ ] **Step 2: Write the failing test**

Append to `tests/wrapper/test_sfm_result.py`:

```python
def test_sfm_points_at_the_scene_images_dir_and_stages_nothing(tmp_path):
    """
    InstantSfM reads <scene>/images directly; no instantsfm/images/ copy is written.
    """
    from collab_splats.pointcloud.sfm import _sfm_image_dir

    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)

    assert _sfm_image_dir(scene / "images") == scene / "images"
    assert not (scene / "instantsfm" / "images").exists()
```

- [ ] **Step 3: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_result.py::test_sfm_points_at_the_scene_images_dir_and_stages_nothing -v
```

Expected: FAIL, `ImportError: cannot import name '_sfm_image_dir'`.

- [ ] **Step 4: Delete the staging loop**

In `collab_splats/pointcloud/sfm.py`, delete the loop that writes `instantsfm/images/*.jpg` and the `FrameStore.open(...)` above it, and replace the whole block with:

```python
def _sfm_image_dir(images_dir: Path) -> Path:
    """
    Image directory InstantSfM reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        The same directory — the store IS the COLMAP image layout, so nothing is staged.
    """
    return Path(images_dir)
```

Then pass `_sfm_image_dir(images_dir)` where the staged directory was passed. VDA's depth naming (`depth_vda/images/npy/<stem>.npy`) is stem-keyed and needs no change, because the stems are still `frame_NNNNNN`.

- [ ] **Step 5: Run the test and the sfm suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_sfm_result.py tests/pointcloud/ -v
```

Expected: PASS.

- [ ] **Step 6: Format and commit**

```bash
black collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py
isort collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py
git commit --only collab_splats/pointcloud/sfm.py tests/wrapper/test_sfm_result.py \
  -m "refactor(sfm): read <scene>/images directly, drop JPEG staging"
```

---

### Task 6: `feedforward/base.py` drops the transient export

> **LANDED** as `8047f9fe`, with three corrections.
>
> 1. **The transient export this task is named after does not exist.** There is
>    no `tempfile`, no `TemporaryDirectory` and no `.export(` anywhere in
>    `collab_splats/pointcloud/`; the earlier in-memory `frames_as_pil_source`
>    refactor already removed path staging. Steps 3.2 and 3.3 are moot. The real
>    work was the store-handle removal plus the extension listing.
>
> 2. **Read literally, this task silently loses frame labels.** `_decode_source`
>    had two branches: a `FrameStore` supplied *source* frame indices
>    (`store.frame_indices()`), while an image dir got sort-order labels
>    `0..N-1`. Collapsing to "an images dir" degrades a real scene's labels to
>    row positions — and those labels become the COLMAP image names that
>    `reconstructor.py:1776` and `evals/scripts/eval_splats.py:84` parse back
>    with `frame_idx_from_path` to join poses to `frames.json`. After quality
>    filtering the indices are gappy, so the result is a silent **misjoin**, not
>    an error. The landed code adds `_source_frame_idxs(paths)`: parse the source
>    index when every stem matches `^frame_\d+$`, else fall back to sort order.
>    The regex gate is deliberate — 7-Scenes eval dirs are
>    `frame-000012.color.png` and `frame_idx_from_path` guesses on any numeric
>    tail, the same trap `geometry/metrics.py:675` already guards.
>
> 3. `from collab_splats.preproc import frames` is unusable in this module —
>    `frames` is a ubiquitous local name in `base.py`, including inside
>    `_decode_dir_to_frames`, where it would make `frames.frame_paths(...)` an
>    `UnboundLocalError`. Use Task 3's alias `frames as fr`.
>
> Behavioural note for merge: the scene path used to hand `_preprocess` a stacked
> `(N, H, W, 3)` array; it now hands a list of per-image arrays, like the legacy
> dir path always did. Every `_preprocess` already accepts both.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (`_decode_source`, and the listing at ~L968)
- Test: `tests/pointcloud/feedforward/test_preprocess_frames.py`, `tests/pointcloud/test_feedforward_preprocess_store.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/pointcloud/feedforward/test_preprocess_frames.py`:

```python
def test_decode_source_takes_an_images_dir_and_makes_no_tempdir(tmp_path, monkeypatch):
    """
    Path-locked model preprocessing reads the scene's images/ directly.
    """
    import tempfile

    import cv2
    import numpy as np

    from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator

    images = tmp_path / "images"
    images.mkdir()
    for i in (0, 5):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((8, 12, 3), i + 5, np.uint8))

    # Any TemporaryDirectory here means a copy is still being staged
    monkeypatch.setattr(
        tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("staged a temporary copy")
    )

    paths = BaseFeedforwardCreator._source_paths(images)
    assert [p.name for p in paths] == ["frame_000000.png", "frame_000005.png"]
```

Add `import pytest` at the top of the file if it is not already there.

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_preprocess_frames.py::test_decode_source_takes_an_images_dir_and_makes_no_tempdir -v
```

Expected: FAIL, `AttributeError: type object 'BaseFeedforwardCreator' has no attribute '_source_paths'`.

- [ ] **Step 3: Replace the export with a direct read**

In `collab_splats/pointcloud/feedforward/base.py`:

1. Add the static helper, replacing the duplicated extension listing at ~L968:

```python
@staticmethod
def _source_paths(images_dir: Path) -> list[Path]:
    """
    Frame image paths a path-locked model preprocessor reads.

    Args:
        images_dir: the scene's images/ directory.

    Returns:
        Image paths in filename order.
    """
    return frames.frame_paths(images_dir)
```

with `from collab_splats.preproc import frames` at the top.

2. In `_decode_source`, change the parameter from a `FrameStore` to `images_dir: Path`, delete the `with tempfile.TemporaryDirectory() as tmp:` block and the `store.export(tmp)` inside it, and use `self._source_paths(images_dir)` where the exported paths were used. Un-indent the body that was inside the `with`.

3. Delete the now-unused `import tempfile` if nothing else in the module uses it.

- [ ] **Step 4: Run the feedforward suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/ -v
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/pointcloud/feedforward/base.py tests/pointcloud/
isort collab_splats/pointcloud/feedforward/base.py tests/pointcloud/
git commit --only collab_splats/pointcloud/feedforward/base.py tests/pointcloud/ \
  -m "refactor(feedforward): read images/ directly, drop transient export"
```

---

### Task 7: `semantics`, `geometry` and `mesh` call sites

> **LANDED** as `bab3a47b`, with four corrections.
>
> 1. **`geometry/loop_closure/wrapper.py` has no `isinstance` branch.** This
>    task says to "delete the `isinstance` branch that handled the store"; both
>    `reconstruct` and `run` simply forward `source` to `setup_inference`. The
>    isinstance dispatch being described lives in `pointcloud/feedforward/base.py`,
>    which is Task 6's file.
>
> 2. **This task's file list omits `tests/mesh/test_tsdf.py`.** It defines its
>    own `FakeStore` duck-type and calls `_feedforward_to_tsdf_inputs(ff,
>    frame_store=FakeStore(), ...)`, a hard `TypeError` after the rename. Landed
>    converted to the shared `_write_images_dir` helper.
>
> 3. **This task's `black`/`isort` commands are harmful.** See the Formatting
>    section above — they were measured here and produced 373/273 lines of churn.
>
> 4. **Two outbound kwargs in `reconstructor.py` break and no task owned them:**
>    line 704 `pointcloud_to_mesh(..., frame_store=...)` and line 1888
>    `build_reconstruction_quality_report(..., frames_zarr=...)` must become
>    `images_dir=`. Lines 951, 992 and 1630 pass `frames_zarr=` too and need
>    checking against their callees. Folded into Task 4.
>
> Every converted read took **row positions**, not source frame indices, so
> Phase A's no-selection-change invariant holds by construction. The one that
> mattered was `semantics/features/base.py`'s `frames.image(i)` inside a loop —
> Task 3's hoisting trap; converting it per-iteration would have been both wrong
> and quadratic. Landed as a hoisted `frame_paths` list plus `Image.open`.
>
> Pre-existing failure, unrelated and not in `docs/known-test-failures.md`:
> `tests/mesh/test_absent_confidence.py::test_splats_depth_targets_skip_masking_when_confidence_absent`
> fails at HEAD too — `from gsplat import losses` raises `ImportError`.

**Files:**
- Modify: `collab_splats/semantics/features/base.py`, `collab_splats/geometry/metrics.py`, `collab_splats/geometry/loop_closure/wrapper.py`, `collab_splats/mesh/utils.py`
- Test: `tests/semantics/features/test_extract_from_zarr.py`, `tests/geometry/test_metrics.py`, `tests/mesh/test_utils.py`, `tests/mesh/test_absent_confidence.py`

- [ ] **Step 1: Find every use**

```bash
git grep -n 'FrameStore' -- collab_splats/semantics collab_splats/geometry collab_splats/mesh
```

- [ ] **Step 2: Apply the Task 3 table**

Mechanical, one file at a time. Two specifics:

- `geometry/loop_closure/wrapper.py` has a `FrameStore | Path` union parameter. It collapses to `Path`; delete the `isinstance` branch that handled the store and keep the path branch.
- `geometry/metrics.py` uses `FrameStore.frame_idx_from_path` for its source-frame join. That becomes `frames.frame_idx_from_path`, imported from `collab_splats.preproc.frames`.

- [ ] **Step 3: Run the affected suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/semantics/ tests/geometry/ tests/mesh/ -v
```

Expected: PASS. Test files that construct a `FrameStore` fixture switch to `fr.write_frames(tmp_path / "images", ...)`.

- [ ] **Step 4: Format and commit**

```bash
black collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh
isort collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh
git commit --only collab_splats/semantics collab_splats/geometry collab_splats/mesh tests/semantics tests/geometry tests/mesh \
  -m "refactor(semantics,geometry,mesh): read images/ in place of FrameStore"
```

---

### Task 8: `dashboard`, `evals` and the notebooks

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py`, `collab_splats/dashboard/localize.py`, `evals/datasets.py`, `evals/scripts/eval.py`, `evals/scripts/eval_splats.py`, `evals/scripts/eval_verification.py`, `docs/source/tutorials/notebook_utils.py`, `docs/source/tutorials/tutorial_config.py`
- Test: `tests/dashboard/`, `tests/evals/`, `tests/docs/test_notebook_utils.py`, `tests/docs/test_tutorial_config.py`

`tests/docs/test_tutorial_config.py:18` asserts `ns["FRAMES_ZARR"] == REPO_ROOT / "data/outputs/frames.zarr"` — both the constant name and the path this task changes. It was missing from this list; sent to the running agent 2026-09-05.

- [ ] **Step 1: Find every use**

```bash
git grep -n 'FrameStore' -- collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py
```

- [ ] **Step 2: Apply the Task 3 table**

One extra edit — **and the original instruction here was wrong; this is the corrected one.**
`_local_ref_paths` lives at `collab_splats/dashboard/pipeline.py:651`, not in `localize.py`, and
it writes nothing: it is a pure path remapper over `zip(localizer.image_paths,
localizer.frame_sources)`. Do **not** replace it with `frames.frame_paths(images_dir)` — that
returns reconstruction frames only, while the list is indexed by `ref` against
`localizer.image_paths`, which also carries `localized` entries living in `localized_frames/`.
Substituting it shifts every index past the first localized frame and renders the wrong
reference image, raising nothing. Keep the two-source remap; change only its
`"frames"` subdirectory to `"images"`, and inline it at its single call site
(`pipeline.py:794`) since it has exactly one caller.

- [ ] **Step 3: Update the notebooks**

**Convert `tutorial_config.py` first.** It defines

```python
FRAMES_ZARR = OUTPUT_DIR / "frames.zarr"          # canonical keyframes (nb 01 writes)
```

and every notebook loads it with `%run ../tutorial_config.py`, then uses the bare name.
Convert the notebooks without renaming this and they still point at `frames.zarr` — the
edits take no effect. It becomes `IMAGES_DIR = OUTPUT_DIR / "images"`.
`tests/docs/test_tutorial_config.py` asserts on the old name.

**Ten notebooks need editing, not six, and `git grep -ln 'FrameStore'` finds only seven
of them.** Seven name `FrameStore`:

```
01_preprocessing/keyframe_extraction.ipynb
04_semantics/feature_extraction.ipynb
04_semantics/maskclip_vs_talk2dino.ipynb
04_semantics/segmentation.ipynb
05_lifting/semantic_lifting.ipynb
06_mesh/splats_mesh.ipynb
07_localization/localization.ipynb
```

Three reach the store only through the constant, so the `FrameStore` grep misses them
entirely:

```
02_pointcloud/bundle_adjustment.ipynb
02_pointcloud/feedforward_methods.ipynb
03_splats/train_splats.ipynb
```

Grep for the union instead:

```bash
git grep -ln -E 'FrameStore|FRAMES_ZARR|frames\.zarr' -- 'docs/source/tutorials/**/*.ipynb'
```

Those three call `load_keyframe_paths(FRAMES_ZARR, TUTORIAL_CACHE / "_frames_export")`.
That helper exists to export the zarr to a transient jpg directory because the creator API
is path-based — `images/` holds real files, so `frames.frame_paths(images_dir)` returns the
paths directly and the bridge has no reason to exist. Delete it if nothing else uses it.

Then apply the Task 3 table to each notebook's source cells:

Edit each with the NotebookEdit tool, not by hand-editing JSON. Re-run each edited notebook top to bottom and commit the executed output — a notebook with a stale traceback is how `plot_quality_examples` stayed broken.

- [ ] **Step 4: Run the suites and the mandatory smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/dashboard/ tests/evals/ tests/docs/ -v
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: tests PASS; smoke exits 0.

- [ ] **Step 5: Format and commit**

```bash
black collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py tests/dashboard tests/evals tests/docs
isort collab_splats/dashboard evals docs/source/tutorials/notebook_utils.py tests/dashboard tests/evals tests/docs
git commit --only collab_splats/dashboard evals docs/source/tutorials tests/dashboard tests/evals tests/docs \
  -m "refactor(dashboard,evals,docs): read images/ in place of FrameStore"
```

---

> ## Task 8 outcomes — landed `a250a7c7`, merged 2026-09-05
>
> One commit, 25 files, +178/−209. Post-merge on the integration tip:
> `2 failed, 344 passed in 89.12s` over `tests/dashboard tests/evals tests/docs`
> (`--ignore=tests/evals/test_eval_splats.py`, gsplat). Both failures are
> `test_run_vggt_slam.py`, whose file is byte-identical to base and which asserts
> `VGGTSLAM_DIR.is_dir()` on a gitignored clone that exists only in the main checkout —
> `docs/known-test-failures.md:15` already names them. Dashboard smoke gate: **PASS**.
>
> ### The plan error that would have shipped a silent bug
>
> Task 8 told the engineer: *"`dashboard/localize.py`'s `_local_ref_paths` writes a
> thumbnail directory that duplicates what `images/` now holds. Delete the function and
> point its caller at `frames.frame_paths(images_dir)`."* Three things wrong, compounding:
>
> 1. The function is at **`dashboard/pipeline.py:651`**, not `localize.py`.
> 2. It writes **nothing**. It is a pure path remapper — `sub = "frames" if src ==
>    "reconstruction" else "localized_frames"` — with zero IO.
> 3. **The prescribed replacement is wrong and fails silently.** `frame_paths(images_dir)`
>    returns reconstruction frames only, but the list it replaces is indexed by `ref`
>    against `localizer.image_paths` / `frame_sources`, which also carry `localized`
>    entries living in `localized_frames/`. Substituting `frame_paths` shortens the list
>    and shifts every index past the first localized frame — the localization UI would
>    render the wrong reference image beside each match, with no exception anywhere.
>
> The agent kept the two-source remap and inlined it at its single call site
> (`pipeline.py:794`), with `"frames"` → `"images"`. That is the correct reading.
> **A plan step that says "delete X and call Y instead" is an assertion that X and Y
> return the same thing. Check that before writing the step.**
>
> ### The gate that never ran
>
> Step 4's smoke command was `python -m collab_splats.dashboard.serve --smoke`.
> Measured on this tip:
>
> ```
> $ python -m collab_splats.dashboard.serve --smoke   # no output, exit 0
> $ python -m collab_splats.dashboard --smoke
> SMOKE PASS: page (8358 B) + bokeh.min.js (1264808 B) served; bind took 3s
> ```
>
> `serve.py` contains zero `__main__` occurrences, so `-m` on it imports the module,
> runs nothing and exits 0 **no matter what is broken**. The dashboard smoke gate is a
> mandatory pre-commit check, and the plan's spelling of it was a silent false green.
> The correct target is the package, `collab_splats.dashboard`, whose `__main__.py`
> holds the entry point. A gate that prints nothing did not run.
>
> ### A real convention split, already handled
>
> The dashboard's localizer ids now come from `[p.name for p in fr.frame_paths(...)]`,
> so they carry `.png` — while Task 4b deliberately keeps `.jpg` labels in
> `_build_localization_db` ("opaque labels, every consumer joins on the `frame_NNNNNN`
> stem"). A dashboard reading a reconstructor-built DB therefore sees ids whose
> extension does not match the files on disk. `localize.py` resolves this by looking up
> the **source frame index** rather than the path — `fr.read_frames(images_dir,
> idxs=[fr.frame_idx_from_path(...)])` — and says so in a comment. Leave it that way;
> do not "fix" one side to match the other.
>
> ### Also found
>
> - **`notebook_utils.load_keyframe_paths` is dead and was deleted.** Its only purpose
>   was exporting the zarr to a transient jpg dir for the path-based creator API;
>   `images/` holds real files. Its test went with it — that is the entire −1 from the
>   345-pass baseline.
> - **Ten notebooks edited, none executed.** `NotebookEdit` cleared `outputs` and
>   `execution_count` on every cell it touched, collapsed `source` line-lists to single
>   strings, and float→int-ified `metadata.widgets` on cells that were never edited. The
>   agent rebuilt all ten from their base blobs, transplanting only changed `source`
>   values. Verified on the merged tree: diffs are 2–9 changed lines per notebook and
>   `git diff -U0 | grep -cE '"(execution_count|output_type|model_id|outputs)"'` is **0**.
>   **Do not use `NotebookEdit` on this repo's tutorials without checking that count.**
> - `01_preprocessing/keyframe_extraction.ipynb:459` still reads `frames.zarr` — it is a
>   **stored output string**, not code, and clears on the next execution. Leave it.
> - Lint on the 15 edited `.py` files reproduces byte-for-byte on the base blobs: zero
>   new lint. No directory-wide `black` was run, per the standing constraint.

### Task 8b: The test modules no task owned

**Files:**
- Modify: `tests/wrapper/test_vda_context.py`, `tests/wrapper/_stubs.py`, `tests/preproc/test_undistort.py`
- Test: the three files above are the tests

**Runs strictly after Task 4b, and this is not a scheduling preference.**
`test_vda_context.py:514` calls `recon._ensure_vda_depth(recon.backend_dir, store, names)`.
Task 4b changes that signature to `(self, backend_dir: Path, images_dir: Path, names)` —
`reconstructor.py:1209` on `preproc-t4b`. Converting the call before 4b lands writes it
against a signature that does not exist yet; converting it after is a one-token change.
Python raises nothing on a positional mismatch here, so a premature edit fails silently.

**Read Task 3's translation table first.** Every edit below is one of its rows.

- [ ] **Step 1: Confirm the surface**

```bash
git grep -n 'FrameStore\|frames_zarr' -- tests/wrapper/test_vda_context.py tests/wrapper/_stubs.py
```

Expected: 12 hits in `test_vda_context.py`, 2 in `_stubs.py`. Fewer means Task 4b or a
concurrent session already took some — re-read the file before editing.

- [ ] **Step 2: Convert `tests/wrapper/_stubs.py`**

This file is new (commit `9c09585b`) and was written against the old API, so it is a
call site the plan created rather than inherited. Two edits:

```python
# line 16 — replace
from collab_splats.preproc.frame_store import FrameStore
# with
from collab_splats.preproc import frames as fr
```

```python
# in _stub_reconstructor — replace
FrameStore.create(recon.frames_zarr, frames, records, provenance={"video_path": "v"})
# with
fr.write_frames(recon.images_dir, frames, records, {"video_path": "v"})
```

`write_frames` takes `provenance` positionally (`preproc/frames.py:97`), not as a keyword.

- [ ] **Step 3: Convert `tests/wrapper/test_vda_context.py`**

| line | old | new |
|---|---|---|
| 21 | `from collab_splats.preproc.frame_store import FrameStore` | `from collab_splats.preproc import frames as fr` |
| 104-105 | `FrameStore.create(recon.frames_zarr, frames, records, provenance={...})` | `fr.write_frames(recon.images_dir, frames, records, {...})` |
| 230 | `frames_zarr=tmp_path / "frames.zarr",` | `images_dir=tmp_path / "images",` |
| 241 | `frames_zarr = tmp_path / "frames.zarr"` | `images_dir = tmp_path / "images"` |
| 246 | `frames_zarr=frames_zarr,` | `images_dir=images_dir,` |
| 255 | `store = FrameStore.open(frames_zarr)` | *(delete — there is no handle)* |
| 266 | `frames_zarr = tmp_path / "frames.zarr"` | `images_dir = tmp_path / "images"` |
| 271 | `frames_zarr=frames_zarr,` | `images_dir=images_dir,` |
| 279 | `FrameStore.open(frames_zarr).provenance()["vda_context_fps"] is None` | `fr.read_manifest(images_dir)["provenance"]["vda_context_fps"] is None` |
| 514 | `store = FrameStore.open(recon.frames_zarr)` | *(delete)* |

Two sites need more than a substitution:

At 255-260, `store.frame_indices()` feeds `chosen`. Replace the handle with:

```python
    manifest = fr.read_manifest(images_dir)
    grid = context_indices(tiny_video, target_fps=10.0)
    chosen = [fr.frame_idx_from_path(p) for p in fr.frame_paths(images_dir)]

    assert set(chosen) <= set(grid)
    assert _context_keep_rows(grid, chosen) is not None
    assert manifest["provenance"]["vda_context_fps"] == 10.0
```

At 514-517, `names` is built from `store.frame_indices()` and `store` is then passed into
`_ensure_vda_depth`. Both go:

```python
    names = [
        f"frame_{fr.frame_idx_from_path(p):06d}.jpg"
        for p in fr.frame_paths(recon.images_dir)
    ]
    ...
        recon._ensure_vda_depth(recon.backend_dir, recon.images_dir, names)
```

The `.jpg` suffix stays even though `write_frames` writes `.png` — these are opaque
labels joined on the `frame_NNNNNN` stem, the same convention `_build_localization_db`
documents on `preproc-t4b`.

- [ ] **Step 3b: Convert `tests/preproc/test_undistort.py`**

Task 4 converted this file's `extract_frames` calls but left one test behind, and no other
task claims it. `test_provenance_roundtrip_through_frame_store` (line ~152) proves the
undistort payload survives a serialisation round-trip; the store it round-trips through is
the one Task 9 deletes. Rename it and repoint it at the manifest:

```python
def test_provenance_roundtrip_through_the_manifest(tmp_path):
    # The undistort provenance payload written by extract_frames must survive the
    # frames.json round-trip and rebuild an identical profile.
    ...
    images_dir = tmp_path / "images"
    fr.write_frames(images_dir, out, [{"frame_idx": 0}], prov)

    stored = fr.read_manifest(images_dir)["provenance"]["undistort"]
    assert DistortionProfile.from_dict(stored["profile"]) == profile
    assert stored["roi"] == list(roi)
    assert np.allclose(np.array(stored["K_new"]), K_new)
```

The body above the `fr.write_frames` line is unchanged. Drop the
`from collab_splats.preproc.frame_store import FrameStore` at line 13 and add
`from collab_splats.preproc import frames as fr`. The test is still worth keeping: it is
the only check that `DistortionProfile` survives JSON, and `frames.json` is a stricter
round-trip than zarr attrs were — a numpy scalar that zarr accepted will now raise in
`_jsonable`, which is the point.

Gate: `pytest tests/preproc/test_undistort.py -q` must stay at **10 passed**.

- [ ] **Step 4: Run**

```bash
PYTHONPATH=$(pwd):/workspace/collab-data /opt/venv/reconstruction/bin/python -m pytest \
  tests/wrapper/test_vda_context.py -q
```

Measured on `5b6b171e` (the Task 4b merge), which is this task's base:

```
23 failed, 10 passed, 9 warnings in 39.45s

  20  AttributeError: 'Reconstructor' object has no attribute 'frames_zarr'
   3  TypeError: extract_frames() got an unexpected keyword argument 'frames_zarr'
```

**All 23 are yours.** An earlier draft of this step said `22 failed, 11 passed` with only
three owned failures and an `instantsfm` residue — that was measured before Task 4b, which
deleted the `Reconstructor.frames_zarr` property. Those 20 tests now die on attribute
access *before* they reach `_run_sfm`, which is where `instantsfm` would have been raised.

The 20 are property **reads**. `git grep FrameStore` does not find them and pyflakes does
not either — grep `frames_zarr` as bare text.

After the conversion the tests that do reach `_run_sfm` will start failing on
`importlib.metadata.PackageNotFoundError: No package metadata was found for instantsfm`
(`reconstructor.py:1210`), the environment break documented in
`docs/known-test-failures.md`. That residue is expected and is not yours — do not chase it
and do not stub around it. The success shape is: **no failure anywhere names
`frames_zarr`**, and every remaining one names `instantsfm`. Report the after-count and the
cause split, not just the count.

Then confirm nothing else regressed:

```bash
PYTHONPATH=$(pwd):/workspace/collab-data /opt/venv/reconstruction/bin/python -m pytest \
  tests/wrapper -q --ignore=tests/wrapper/test_splats_stage.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/wrapper/test_vda_context.py tests/wrapper/_stubs.py
git commit -m "test(wrapper): read images/ in the VDA-context tests"
```

---

> ## Task 8b outcomes — landed `8efde39b`, merged 2026-09-05
>
> Verified post-merge on `preproc/integration`, not taken from the report. Gate 4 grep is clean
> for all three spellings — `FrameStore`, `frames_zarr` **and** `frames.zarr` (exit 1 on each);
> pyflakes exit 0 on all three files. `tests/wrapper/test_vda_context.py` went
> `23 failed, 10 passed` → **`19 failed, 14 passed`**, split 18 instantsfm + 1 gsplat, and no
> remaining failure names `frames_zarr`.
>
> ### Four plan errors
>
> | # | plan said | ground truth | consequence if followed |
> |---|---|---|---|
> | 1 | Step 4 baseline `22 failed, 11 passed`, residue = instantsfm | `23 failed, 10 passed`: **20 × `AttributeError: 'Reconstructor' object has no attribute 'frames_zarr'`** + **3 × `TypeError: extract_frames() got an unexpected keyword argument 'frames_zarr'`**, **zero instantsfm** — Task 4b deleted the property, so those tests die before reaching `_run_sfm` | the executor clears 3 failures, sees 20 left, and concludes the env is broken instead of finishing the task *(corrected in this plan at `a31db4fc`, before 8b ran)* |
> | 2 | Step 3 table has no row for `test_vda_context.py:473` | the **test name itself** is `test_run_sfm_falls_back_when_the_video_changed_since_frames_zarr` — one of the 12 Step-1 hits | *"Left as-is, `git grep frames_zarr` never goes clean."* Gate 4 cannot pass. Renamed to `..._since_extraction` |
> | 3 | Step 3 table has no row for `test_vda_context.py:487` | `assert "was modified since frames.zarr was written" in caplog.text`; Task 4b reworded that log — `reconstructor.py:175` now emits `"%s was modified since the images/ store was written (mtime %s -> %s)"` | *"The Step-1 grep pattern cannot catch this (`frames.zarr`, dot not underscore)"* — the test converts cleanly, then fails on the assertion |
> | 4 | `docs/known-test-failures.md:318` and this plan cite `reconstructor.py:1210` for the instantsfm raise | the line is **1197** | a line citation that sends the next reader to the wrong function *(fixed in `a3d2afbd`)* |
>
> Errors 2 and 3 are the same shape as Task 4b's error 3 and worth naming as a class: **a
> grep-defined work surface only covers what the grep pattern spells.** A test *name* containing
> the dead identifier is a gate failure the Files table never lists; a log message containing
> `frames.zarr` is invisible to `FrameStore\|frames_zarr` because of one dot.
>
> ### Deliberate deviation
>
> Plan Step 3b: `names = [f"frame_{fr.frame_idx_from_path(p):06d}.jpg" for p in fr.frame_paths(recon.images_dir)]`.
> Landed instead:
>
> ```python
> names = [p.name for p in fr.frame_paths(recon.images_dir)]
> ```
>
> — byte-for-byte how the production call site builds it (`reconstructor.py:1163`), so the test
> feeds `_ensure_vda_depth` the same input shape `_run_sfm` does rather than a `.jpg` re-spelling
> of it. `_ensure_vda_depth` uses `names` only for `len(names)` and npy stems, so both satisfy the
> assertions. Also: Step 3b's "add `from collab_splats.preproc import frames as fr`" was already
> done at line 12 by `89534e01` — only the `FrameStore` import needed dropping.
>
> ### The one thing 8b could not prove
>
> `tests/wrapper/test_splats_stage.py:27` imports `_stub_reconstructor` from `_stubs.py` and calls
> it 13 times. It is a second consumer of a file 8b edits and **cannot be run** — the module is
> uncollectible under the gsplat break, which is why Gate 3 ignores it. Verified by inspection only.
>
> `docs/known-test-failures.md` was rewritten afterwards (`a3d2afbd`) for the three claims this
> commit made stale: the `_stubs.py` import description, the `:1210` line number, and the
> `22 failed, 11 passed` split.

---

---

> ## Coordinator merge ledger — the branch topology, as of 2026-09-05
>
> Read this first after any context loss. It is the only place the *state* of the wave lives;
> the task bodies below describe intent, not what landed.
>
> **`preproc/integration`** is the trunk this wave merges into. Every merge is `--no-ff` with a
> `merge(preproc): Task N — ...` subject, so `git log --first-parent` reads as the ledger.
> Merged and gated: Tasks 1, 2, 4, 4b, 5, 6, 7, 8, 8b, 9, 10, 11, 19, 20, 21, 25, 26.
>
> | pin | scope | result |
> |---|---|---|
> | `706c2d38` (post-T11) | `tests/preproc` | `184 passed` in 878.72 s |
> | `26b8cbcd` (post-T21) | `tests/preproc` | `191 passed` in 290.82 s |
> | `c9c75812` (post-T9) | `tests/preproc tests/remote tests/pointcloud/feedforward/test_preprocess_frames.py` | `336 passed` in 565.81 s |
>
> The `336` decomposes as 185 preproc + 151 remote/feedforward, and 185 is Task 9's own
> branch figure of 169 plus Task 11's 9 and Task 21's 7. Every pin was re-checked as an
> ancestor of the tip *after* its run finished.
>
> **`preproc/phase-c-staging`** exists because Task 15 cannot merge into integration alone.
> Task 15 deletes `DistortionProfile` and `estimate_camera_distortion`, which
> `collab_splats/wrapper/reconstructor.py:43-46` still imports at module scope — that is a
> **collection** error, so every test module that transitively imports the reconstructor,
> `tests/preproc/test_undistort.py` included, runs zero tests. Task 17 is the fix. So Task 15
> merged to staging (`2f97931a`), Tasks 16-18 execute on top, and the whole chain merges to
> integration as one green unit. **A knowingly-red trunk is as corrosive as a false green:
> nothing merges to integration that leaves it red pending a later task.**
>
> **`preproc-t4` is superseded, not lost.** It shows one unmerged commit
> (`8545a3a1 test(preproc): point the remaining extract_frames callers at images/`), but
> integration already carries every one of its changes by a different route, and its version of
> `tests/preproc/test_undistort.py` still imports the now-deleted `FrameStore`. `git diff
> preproc-t4 HEAD` on its two files shows integration strictly ahead. Do not merge it.
>
> **`git merge-base --is-ancestor` gives false "already merged" positives** for a branch with
> zero new commits. Use `git rev-list --count HEAD..<branch>` to decide what is outstanding.
> The installed git also rejects `git merge-tree --write-tree`; the conflict preview here is
> the legacy three-argument form, `git merge-tree $(git merge-base A B) A B`.

---

> ## Coordinator merge note — Tasks 9 and 15 collide on `__all__`
>
> **Both tasks rewrite the same two literals and neither one's arithmetic is right after the
> other lands.** `collab_splats/preproc/__init__.py.__all__` and the exact-set assertion in
> `tests/preproc/test_sampling.py::test_public_api_surface` (including the `Exactly the N public
> names` comment above it) are edited by Task 9 *and* Task 15, computed independently.
>
> **Corrected 2026-09-05 against the landed branches — an earlier version of this note said
> the answer was 12 and it was wrong.** That figure was read off this plan's own Task 9 prose
> ("correct the comment to 13"), not off the branch. The plan text is the error: the same
> Task 9 step that drops `FrameStore` also *adds* the five `frames.py` names, so 14 − 1 + 5 =
> **18**, not 13. Task 9's executor caught this, landed 18, and flagged it.
>
> Measured on the landed branches with `git show <sha>:collab_splats/preproc/__init__.py`:
>
> | branch | `__all__` | delta from integration's 14 |
> |---|---|---|
> | integration `26b8cbcd` | 14 | — |
> | Task 9 `fe43ddc9` | **18** | −`FrameStore`; +`frame_idx_from_path`, `frame_paths`, `read_frames`, `read_manifest`, `write_frames` |
> | Task 15 `preproc-t15` | **13** | −`DistortionProfile`, −`estimate_camera_distortion`; +`calibrate_camera` |
>
> The two deltas are disjoint, so together the answer is 14 − 1 − 2 + 5 + 1 = **17**. A textual
> merge of two set literals will either conflict or silently keep a deleted name, so the
> coordinator reconciles by hand at the second merge. The target set is exactly:
>
> ```
> analysis_gray, calibrate_camera, compute_video_quality, extract_frame,
> filter_frame_quality, frame_idx_from_path, frame_paths, get_video_info,
> iter_frames, load_video_quality, read_frames, read_manifest, sample_fps,
> sample_optical_flow, sample_uniform, undistort_frames, write_frames
> ```
>
> Seventeen names, and the comment must read `Exactly the 17 public names`. The gate is not the
> count — it is that `set(preproc.__all__)` equals that literal and `tests/preproc` is green.
>
> **The lesson generalises past `__all__`:** a merge-reconciliation figure computed from plan
> prose is a prediction, not a measurement. Read every colliding literal off the branch blob
> before writing the target down.

---

### Task 9: Delete `frame_store.py` and close out Phase A

**Files:**
- Delete: `collab_splats/preproc/frame_store.py`, `tests/preproc/test_frame_store.py`
- Modify: `collab_splats/preproc/__init__.py`, `collab_splats/remote/sources.py`, `configs/base.yaml`, `configs/README.md`, `docs/source/api/preproc.rst`, `tests/preproc/test_sampling.py`, `tests/pointcloud/feedforward/test_preprocess_frames.py`, `tests/remote/test_sources.py`

`tests/preproc/test_sampling.py` is a **hard failure**, not a tidy-up.
`test_public_api_surface` (line ~361) asserts `set(preproc.__all__)` equals a literal
14-name set containing `"FrameStore"`. Removing the name from `__init__.py` breaks that
equality, and no other task's Files list names the file. Drop `"FrameStore"` from the set
and correct the "Exactly the 14 public names" comment above it to 13.
**WRONG — do not follow this figure.** Step 2 below also *adds* the five `frames.py`
names, so the correct count is 14 − 1 + 5 = **18**. Landed as 18 in `fe43ddc9`; see the
coordinator merge note above for the post-Task-15 reconciliation to 17.

`tests/pointcloud/feedforward/test_preprocess_frames.py:29` is only a comment
(`# frames = the exact decoded pixels a FrameStore would hold`), but Step 1's grep matches
text, not imports, so it will show up as an unconverted call site and block the gate.
Reword it to name `images/`.

`tests/remote/test_sources.py` is the test file for `remote/sources.py`, which this task
already modifies, and it hard-codes the retired name five times: line 532 parametrizes
over `("frames.zarr/zarr.json", "frames.zarr/images/c/0/0/0", "frames.zarr")`, line 537
asserts `"frames.zarr/**" not in PUSH_EXCLUDES`, and lines 557/562 pass
`excludes=("frames.zarr/**",)`. Line 537 is a **regression guard**, not decoration — a
past bug excluded the keyframe store from the push and made pulled scenes unlocalizable
(`sources.py:52` records it). Keep the guard and repoint it at `images/**`; do not delete
it.

> **One real hazard when you repoint these.** `PULL_EXCLUDES` already contains
> `pointcloud.zarr/images/**`, and Phase A adds a *top-level* `images/` to the same tree.
> The two do not collide — the pattern has two path components, so it needs a
> `pointcloud.zarr` parent — but `sources.py` documents (measured, rclone v1.53.3-DEV)
> that an unanchored pattern matches at **any** depth. So a bare `images/**` added to
> either tuple would also swallow `pointcloud.zarr/images/**` and every `<backend>/images`.
> Anchor anything you add with a leading slash, exactly as the `/semantics/**` entry above
> it does, and say in the commit message which tuple you touched.

`configs/README.md` was in no task's Files list and states things Phase A makes false: line 51 says "there is no images/ dir", and line 497 describes `images/frame_NNNNNN.jpg` as "frames.zarr staged for the path-locked COLMAP/InstantSfM tools" — the staging step Task 5 deleted. Also check the `preproc.undistort` and `splats.enabled` rows, which describe the store by name.

Three more lines in the same file, added to Task 9's surface after the Task 10 merge:

- **line 289** — "writes `video_quality_report.json` beside `frames.zarr`". It lands beside
  `images/` (`reconstructor.py`: `images_dir.parent / "video_quality_report.json"`).
- **line 300** — "The sampler reads that report through `filter_frame_quality`, **which is where
  every threshold lives**." Task 10 deleted the thresholds. The rule is now a robust MAD cut on
  `log(laplacian)` (`sharpness_k`, a z-score cut, not an absolute) plus an absolute ceiling on
  `clipped_low_frac + clipped_high_frac` (`max_clipped_frac`).
- **line 353** — the `preproc.vda_context_fps` row says "blur substitution", which Task 11 renames.
  Re-read it after Task 11 lands rather than editing it here.

`docs/source/_static/preproc/README.md` was also routed here by Task 10's audit and needs **no**
change — it is a plot inventory, and every row names a plotter, not the filter.

### Do not chase the `frames.zarr` prose here — Step 1's grep does not spell it

Step 1 greps `frame_store\|FrameStore`. Nine more files still say `frames.zarr` in prose, and
**none of them is Task 9's**. Measured on the tip after the Task 10 merge:

| hit | owner |
|---|---|
| `collab_splats/preproc/undistort.py:2`, `:49` | Task 15/16 (undistort rewrite) |
| `collab_splats/preproc/video.py:273` | Tasks 20-24 (the PyAV chain) |
| `collab_splats/preproc/viz.py:103` | **no task owns it** — Task 25 already merged and only deleted two plotters |
| `collab_splats/preproc/qa.py:478` | **no task owns it** — Task 26 already merged and only collapsed the pair-motion four |
| `collab_splats/preproc/frames.py:188` | correct as written — it is the migration error message |
| `scripts/migrate_frames_zarr.py` (7 hits) | correct as written — it reads the old store by definition |
| `collab_splats/remote/sources.py:52`, `:392`, `:473` | **Task 9, Step 2** — already in the step |
| `collab_splats/pointcloud/feedforward/loger.py:295` | **no task owns it.** One comment: "frames.zarr is ordered" |
| `collab_splats/pointcloud/sfm.py:297` | **no task owns it.** One docstring line: "(frames.zarr order)" |

Four rows have no owner: `viz.py:103` and `qa.py:478` (their tasks merged without touching the
prose, which neither task's Files list covered), plus `loger.py:295` and `sfm.py:297`, which are
outside `preproc/` and outside every Files list in this plan. Fix all four in Task 9's commit —
two words each, and leaving them makes `git grep frames.zarr` permanently noisy — but add them to
the `--only` path list, and say so in the commit message. `viz.py` and `qa.py` are already inside
Step 4's `collab_splats/preproc` path; `loger.py` and `sfm.py` are not.

Two more hits are **test names and assertions**, the class Task 8b's errors 2 and 3 named:
`tests/remote/test_sources.py:532-537` parametrises over the literal strings `"frames.zarr"`,
`"frames.zarr/zarr.json"`, `"frames.zarr/images/c/0/0/0"` and asserts `"frames.zarr/**" not in
PUSH_EXCLUDES`, inside a test *named* `test_push_no_longer_excludes_the_keyframe_store` — which is
itself a `frame_store` grep hit. That file is in Step 2's list; the strings and the name have to
move with the excludes, or Step 2's rewrite lands and the test asserts about a pattern nothing
uses. Likewise `tests/preproc/test_frames.py:139-145`, which builds a `frames.zarr` by hand to feed
the migration script — **correct as written**, do not convert it, but its line 141 comment names
`FrameStore.create` and so survives Step 1's grep after the class is gone.

- [ ] **Step 1: Verify nothing imports it**

```bash
git grep -n 'frame_store\|FrameStore' -- 'collab_splats/*' 'evals/*' 'scripts/*' 'tests/*'
```

Expected: only `collab_splats/preproc/frame_store.py`, `collab_splats/preproc/__init__.py` and `tests/preproc/test_frame_store.py`. Anything else is an unconverted call site — go back and convert it.

If `collab_splats/preproc/viz.py` appears, Task 25 has not run. It is the only `FrameStore` consumer no Phase A task converts; run Task 25 rather than writing a conversion for a function that is about to be deleted.

- [ ] **Step 2: Delete and rewire**

```bash
git rm collab_splats/preproc/frame_store.py tests/preproc/test_frame_store.py
```

In `collab_splats/preproc/__init__.py`, replace the `FrameStore` import with:

```python
from collab_splats.preproc.frames import (
    frame_idx_from_path,
    frame_paths,
    read_frames,
    read_manifest,
    write_frames,
)
```

and update `__all__` — remove `"FrameStore"`, add `"frame_idx_from_path"`, `"frame_paths"`, `"read_frames"`, `"read_manifest"`, `"write_frames"`, keeping the list alphabetically sorted. Update the module docstring's `frame_store.py` mention to `frames.py`.

In `collab_splats/remote/sources.py`, rewrite `PUSH_EXCLUDES` / `PULL_EXCLUDES` and their comment block: every `frames.zarr/**` pattern becomes `images/**`, and the comment explaining why the store is excluded now says "the images/ directory is regenerable from the source video, and is the largest thing in a scene".

In `configs/base.yaml`, rewrite the `preproc:` block's leading comment:

```yaml
preproc:
  # Output: <output_path>/images/frame_NNNNNN.png (COLMAP-style image directory,
  # lossless PNG) + <output_path>/frames.json (selection records + provenance).
```

and change the `undistort:` comment's `frames.zarr is written` to `images/ is written`, and its `frames.zarr reuse is by EXISTENCE` to `images/ reuse is by EXISTENCE`.

In `docs/source/api/preproc.rst`, replace the `frame_store` automodule entry with `frames`.

- [ ] **Step 3: Run the whole suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the suite is green apart from anything already listed in `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 4: Update the graph and commit**

```bash
graphify update .
black collab_splats/preproc configs
isort collab_splats/preproc
git commit --only collab_splats/preproc collab_splats/remote/sources.py configs/base.yaml docs/source/api/preproc.rst \
  -m "refactor(preproc)!: delete FrameStore, images/ is the store

BREAKING: a scene holding only frames.zarr raises. Convert it with
scripts/migrate_frames_zarr.py."
```

**Phase A gate:** the suite is green, the dashboard smoke passes, and frame selection is byte-identical to before the phase. If a selection changed, it is a bug in Phase A — find it before starting Phase B.

---

## Phase B — the quality filter fires, and sampling reads from the pool it makes

Invariant for this phase: **selection changes on purpose**, and Task 14 is the gate that proves the change is the intended one. Nothing in Phase B may land on a branch that has not passed Task 14.

---

### Task 10: `filter_frame_quality` — robust MAD on log(laplacian)

**Files:**
- Modify: `collab_splats/preproc/sampling.py:32-76`
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def _report(laplacian, *, clipped_low=None, clipped_high=None):
    """
    Minimal quality report carrying only the columns the filter reads.
    """
    n = len(laplacian)
    return {
        "frames": {
            "laplacian": list(laplacian),
            "clipped_low_frac": list(clipped_low if clipped_low is not None else [0.0] * n),
            "clipped_high_frac": list(clipped_high if clipped_high is not None else [0.0] * n),
        }
    }


def test_filter_cuts_the_soft_frame_in_an_otherwise_sharp_run():
    """
    One frame two orders of magnitude softer than its neighbours is cut.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    mask = filter_frame_quality(_report([400.0] * 20 + [3.0] + [400.0] * 20))

    assert mask[20] == False  # noqa: E712 — the soft frame
    assert mask.sum() == 40


def test_filter_is_scale_free():
    """
    Multiplying every laplacian by a constant cannot change the mask —
    that is the whole point of a robust z-score on the log.
    """
    import numpy as np

    from collab_splats.preproc.sampling import filter_frame_quality

    lap = [400.0, 380.0, 410.0, 3.0, 395.0, 405.0] * 8
    a = filter_frame_quality(_report(lap))
    b = filter_frame_quality(_report([x * 1000.0 for x in lap]))

    assert np.array_equal(a, b)


def test_filter_keeps_everything_when_sharpness_is_uniform():
    """
    Zero MAD must not divide-by-zero into an all-False mask.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    assert filter_frame_quality(_report([250.0] * 30)).all()


def test_filter_cuts_a_clipped_frame():
    """
    Clipping is an absolute rule: >25% destroyed pixels is out regardless of sharpness.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    lap = [400.0] * 10
    low = [0.0] * 9 + [0.30]
    mask = filter_frame_quality(_report(lap, clipped_low=low))

    assert mask[9] == False  # noqa: E712
    assert mask[:9].all()


def test_filter_clipping_is_the_sum_of_both_tails():
    """
    0.15 crushed + 0.15 blown is 0.30 destroyed, over the 0.25 ceiling.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    mask = filter_frame_quality(
        _report([400.0] * 4, clipped_low=[0.0, 0.0, 0.0, 0.15], clipped_high=[0.0, 0.0, 0.0, 0.15])
    )

    assert mask[3] == False  # noqa: E712


def test_filter_handles_an_empty_report():
    """
    A report with no rows returns an empty mask, not an exception.
    """
    from collab_splats.preproc.sampling import filter_frame_quality

    assert filter_frame_quality(_report([])).shape == (0,)


def test_filter_no_longer_takes_the_deleted_thresholds():
    """
    laplacian_min et al are gone; passing one is a TypeError, not a silent no-op.
    """
    import pytest

    from collab_splats.preproc.sampling import filter_frame_quality

    for dead in ("laplacian_min", "exposure_mean_range", "exposure_min_std", "blur_max"):
        with pytest.raises(TypeError):
            filter_frame_quality(_report([400.0] * 5), **{dead: 1})
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k filter -v
```

Expected: the scale-free, soft-frame, clipping and dead-kwarg tests FAIL — the current filter uses a fixed `laplacian_min=50` and reads no clipping columns.

- [ ] **Step 3: Replace the filter**

In `collab_splats/preproc/sampling.py`, replace the whole of `filter_frame_quality` (lines 32-76) with:

```python
def filter_frame_quality(
    report: dict,
    *,
    sharpness_k: float = 2.0,
    max_clipped_frac: float = 0.25,
) -> np.ndarray:
    """
    Per-frame usability mask over a quality report's photometry columns.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        sharpness_k: robust z-score cut on log(laplacian); larger keeps more.
        max_clipped_frac: ceiling on clipped_low_frac + clipped_high_frac.

    Returns:
        (N,) bool, True = keep, indexed by source frame index.
    """
    f = report["frames"]
    lap = np.asarray(f["laplacian"], dtype=float)
    if lap.size == 0:
        return np.zeros(0, dtype=bool)

    # Sharpness is relative: laplacian variance scales with resolution, texture and
    # content, so the cut is a robust z-score on the log rather than an absolute value.
    log_lap = np.log(np.clip(lap, 1e-6, None))
    centre = np.median(log_lap)
    spread = float(median_abs_deviation(log_lap, scale="normal"))

    # A zero MAD means every frame is equally sharp — nothing to cut, and the
    # z-score would be a division by zero.
    sharp = np.ones_like(lap, dtype=bool) if spread == 0.0 else log_lap >= centre - sharpness_k * spread

    # Clipping is absolute: a pixel at 0 or 255 recorded nothing recoverable.
    clipped = np.asarray(f["clipped_low_frac"], dtype=float) + np.asarray(f["clipped_high_frac"], dtype=float)

    return sharp & (clipped <= max_clipped_frac)
```

Add the import at the top of the module:

```python
from scipy.stats import median_abs_deviation
```

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k filter -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py \
  -m "feat(preproc)!: robust MAD sharpness cut + absolute clipping cut

BREAKING: filter_frame_quality drops laplacian_min, exposure_mean_range,
exposure_min_std and blur_max. It now cuts frames; the old defaults never did."
```

The rest of the suite will now fail wherever `_sample_by_quality` consumed the old mask. That is expected — Tasks 11-13 fix it.

---

> ## Task 10 outcomes — landed `c65788ff`, merged 2026-09-05
>
> Gate `tests/preproc -q -p no:randomly`: **`171 passed` → `175 passed`**, zero failures, zero
> skips (−3 threshold-era filter tests, +7 new). `test_public_api_surface` still passes with its
> 14-name set untouched. Two files, +29/−32 and +80/−15.
>
> ### The plan's implementation fails the plan's own first test
>
> Step 3 shipped this guard:
>
> ```python
> # A zero MAD means every frame is equally sharp — nothing to cut, and the
> # z-score would be a division by zero.
> sharp = np.ones_like(lap, dtype=bool) if spread == 0.0 else log_lap >= centre - sharpness_k * spread
> ```
>
> Step 1's own `test_filter_cuts_the_soft_frame_in_an_otherwise_sharp_run` feeds
> `_report([400.0] * 20 + [3.0] + [400.0] * 20)` and asserts `mask[20] is False`, `mask.sum() == 40`.
> Measured on that fixture: median `5.991464547107982`,
> `median_abs_deviation(log_lap, scale="normal")` = **exactly `0.0`** — 40 of 41 rows are identical.
> The guard fires, the mask is all-True, and the test fails `assert np.True_ == False`.
> Reproduced independently by the coordinator before merge.
>
> **The guard's premise is wrong: `MAD == 0` means over half the values sit exactly on the median,
> not that the column is constant.** Landed guard:
>
> ```python
> if log_lap.min() == log_lap.max():
>     sharp = np.ones_like(lap, dtype=bool)
> else:
>     sharp = log_lap >= centre - sharpness_k * spread
> ```
>
> With `MAD == 0` on a non-constant column the cut lands on the median itself — cuts the outlier,
> divides by nothing.
>
> ### Two edits Task 10 forces that no step mentions
>
> - `tests/preproc/test_sampling.py:24-38` — `_synthetic_report` had no `clipped_low_frac` /
>   `clipped_high_frac`. The new filter reads both unconditionally, so **every** test in the file
>   would `KeyError`. Added as zeros.
> - `tests/preproc/test_sampling.py:262-271` and `:400-407` — the two "report condemns everything"
>   tests condemned by zeroing `laplacian`. Under a scale-free rule a zeroed column is *constant*
>   and condemns nothing. Re-expressed through the absolute clipping rule
>   (`clipped_low_frac[i] = 1.0`), assertions and intent preserved.
>
> ### The "ships red" concession was not true
>
> Task 10's closing line said: *"The rest of the suite will now fail wherever `_sample_by_quality`
> consumed the old mask. That is expected — Tasks 11-13 fix it."* It does not, and the green gate
> above is not luck. `_sample_by_quality` picks
> `max(window, key=lambda i: (bool(usable[i]), float(laplacian[i])))`, and **any**
> threshold-on-laplacian filter makes `usable` monotone in `laplacian`, so the argmax cannot move.
> `test_sampling_parity.py` (byte-parity against `tests/preproc/data/parity_baseline.json`) passes
> untouched. Only `sample_optical_flow` is mask-sensitive, because its gate is a hard skip
> (`if idx >= len(usable) or not usable[idx]: continue`) — that is the one test that had to be
> re-expressed.
>
> ### The merge-order landmine the two waves made together
>
> `_clean_report(n=60)` in `tests/wrapper/test_vda_context.py:59-70` builds `laplacian` but neither
> clipped column, so the new filter raises `KeyError: 'clipped_low_frac'` against it. Task 10's
> audit found it and correctly judged it masked — both consumers were failing earlier at
> `TypeError: extract_frames() got an unexpected keyword argument 'frames_zarr'`. **Task 8b cleared
> exactly those TypeErrors, in the same wave.** Neither branch is wrong alone; the merge of both is.
> Fixed on `preproc/integration` between the two merges; that fixture was the only one lacking the
> columns (`tests/wrapper/test_reconstructor.py:1560-1570` already carries both).
>
> ### Routing owed
>
> - `collab_splats/wrapper/reconstructor.py:267` — comment still says "filter_frame_quality applies
>   the thresholds inside the samplers"; those thresholds are gone.
> - `configs/README.md`, `docs/source/_static/preproc/README.md` — prose about the filter.
> - `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` — already in
>   `docs/known-test-failures.md`.

---

### Task 11: the eligible pool and `sample_uniform`

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (add `_eligible` and `_decode_selection`, rewrite `sample_uniform`)
- Test: `tests/preproc/test_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def test_eligible_intersects_the_mask_with_the_candidate_grid():
    """
    The quality mask and the VDA context grid are one restriction, not two.
    """
    import numpy as np

    from collab_splats.preproc.sampling import _eligible

    report = _report([400.0] * 20 + [3.0] + [400.0] * 9)
    pool = _eligible(report, quality=None, candidates=[0, 10, 20, 25])

    # 20 is condemned by the mask, so the grid loses it
    assert np.array_equal(pool, np.array([0, 10, 25]))


def test_eligible_raises_when_the_pool_is_empty():
    """
    An empty pool is a config error, not an empty scene written silently.
    """
    import pytest

    from collab_splats.preproc.sampling import _eligible

    with pytest.raises(ValueError, match="no eligible frames"):
        _eligible(_report([400.0] * 10), quality=None, candidates=[])


def test_sample_uniform_spans_the_eligible_pool(monkeypatch, tmp_path):
    """
    Picks are evenly spaced in POOL index, and never land on a condemned frame.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    # 30 frames, the middle 10 blurred out
    report = _report([400.0] * 10 + [2.0] * 10 + [400.0] * 10)

    monkeypatch.setattr(sampling, "get_video_info", lambda p: {"total_frames": 30, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=4, report=report)

    picked = [r["frame_idx"] for r in records]
    assert len(frames) == 4
    assert all(i < 10 or i >= 20 for i in picked), picked
    assert picked == sorted(picked)


def test_sample_uniform_returns_the_whole_pool_when_it_is_short(monkeypatch, tmp_path, caplog):
    """
    A pool smaller than max_frames returns the pool and logs the shortfall.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    report = _report([400.0] * 3)
    monkeypatch.setattr(sampling, "get_video_info", lambda p: {"total_frames": 3, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_uniform(str(tmp_path / "v.mp4"), max_frames=10, report=report)

    assert [r["frame_idx"] for r in records] == [0, 1, 2]
    assert "eligible" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "eligible or uniform" -v
```

Expected: FAIL, `ImportError: cannot import name '_eligible'`.

- [ ] **Step 3: Add the two helpers**

In `collab_splats/preproc/sampling.py`, delete `_sample_by_quality` in full (lines 238-336) and put these in its place:

```python
def _eligible(report: dict, *, quality: dict | None, candidates: Sequence[int] | None) -> np.ndarray:
    """
    Source frame indices a sampler may select from.

    Args:
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.
        candidates: optional index grid every pick must be a member of (the VDA context grid).

    Returns:
        (M,) int64, ascending.
    """
    pool = np.flatnonzero(filter_frame_quality(report, **(quality or {})))

    # The mask and the context grid are the same kind of restriction, so they intersect
    if candidates is not None:
        pool = np.intersect1d(pool, np.asarray(sorted({int(c) for c in candidates}), dtype=np.int64))

    if pool.size == 0:
        raise ValueError(
            "no eligible frames: the quality filter and the candidate grid have no index in common"
        )

    return pool


def _decode_selection(
    video_path: str,
    chosen: Sequence[int],
    *,
    report: dict,
    on_progress,
    desc: str,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Decode exactly the selected frames, in one pass, as RGB.

    Args:
        video_path: source video.
        chosen: ascending source frame indices.
        report: the quality report the blur_score column is read from.
        on_progress: optional (done, total) callback.
        desc: progress-bar label.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    laplacian = np.asarray(report["frames"]["laplacian"], dtype=float)

    # One ffmpeg select pass over exactly the frames we keep
    decoded = dict(iter_frames(video_path, indices=list(chosen)))

    frames: list[np.ndarray] = []
    records: list[dict] = []

    for idx in progress(chosen, total=len(chosen), desc=desc, on_progress=on_progress):
        bgr = decoded.get(idx)
        if bgr is None:
            continue  # ffmpeg dropped the frame (should not happen)

        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # blur_score comes from the report, not a recompute — same measurement,
        # and it is the column the record has always carried.
        records.append({"frame_idx": int(idx), "blur_score": float(laplacian[idx])})

    return frames, records
```

- [ ] **Step 4: Rewrite `sample_uniform`**

Replace `sample_uniform` (through its `return _sample_by_quality(...)`) with:

```python
def sample_uniform(
    video_path: str,
    *,
    max_frames: int,
    report: dict,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    Exactly max_frames evenly-spaced picks from the eligible pool.

    Args:
        video_path: source video.
        max_frames: how many frames to keep — the COUNT is the contract.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.
        candidates: optional index grid every pick must be a member of.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    if max_frames <= 0:
        return [], []

    pool = _eligible(report, quality=quality, candidates=candidates)

    # Spacing is even in POOL index, not in time: budget is not spent inside
    # footage the filter just condemned.
    if pool.size <= max_frames:
        logger.warning(
            "max_frames=%d but only %d eligible frames; keeping the whole pool", max_frames, pool.size
        )
        chosen = pool.tolist()
    else:
        chosen = pool[np.linspace(0, pool.size - 1, max_frames).round().astype(int)].tolist()

    return _decode_selection(
        video_path, chosen, report=report, on_progress=on_progress, desc="Uniform sampling"
    )
```

Note the deleted `get_video_info` call: the pool comes from the report, whose length already is the video's frame count, so uniform sampling no longer probes the video at all.

- [ ] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "eligible or uniform" -v
```

Expected: PASS. `test_sample_uniform_spans_the_eligible_pool`'s monkeypatched `get_video_info` is now unused for `sample_uniform` — leave it, `sample_fps` in the next task needs it.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
isort collab_splats/preproc/sampling.py tests/preproc/test_sampling.py
git commit --only collab_splats/preproc/sampling.py tests/preproc/test_sampling.py \
  -m "refactor(preproc)!: sample_uniform picks from the eligible pool

BREAKING: deletes _sample_by_quality and sample_uniform's search_radius."
```

---

> ## Task 11 outcomes — landed `04e08388`, merged `eb6f4a47` 2026-09-05
>
> Five files, +225/−107. Gates on the task's own base `d75817d3`: `tests/preproc`
> **`175 passed` → `1 failed, 178 passed`**, the one red being the deliberate parity break below
> (175 + 4 new = 179; nothing else changed state, proven by `comm -3` of sorted FAILED lists
> rather than by comparing counts). pyflakes on the five touched files: 4 findings, all proven
> pre-existing by running it against the `HEAD:` blob and getting byte-identical output.
> `isort --check-only` exit 0. `black --check` flags two files that were **already** dirty at
> base — and the task *reduced* `reconstructor.py`'s black debt 350 → 334 lines by writing its
> new call one-arg-per-line. Wrapper regression set was proven identical, not merely
> equal-in-count.
>
> **Integration baseline re-measured on the merged tip.** SHA-pinned `706c2d38` (the merge
> `eb6f4a47` plus the parity retirement below), run alone, no agent wave:
> **`184 passed, 9 warnings in 878.72s (0:14:38)`**, `pytest exit: 0`, and the pin re-checked as
> an ancestor of the tip *after* the run. That satisfies the only arithmetic it could:
> 181 carried in from Task 20, plus the 4 this task adds, minus the 1 parity assertion retired
> below. **`184 passed` is the number every later Phase B/C/D task must reproduce on
> `tests/preproc` before it starts editing.**
>
> ### The one that would have shipped: a `TypeError` in production
>
> **Task 11's Files list omits `collab_splats/wrapper/reconstructor.py`**, which forwarded
> `search_radius=search_radius` into `sample_uniform`. Task 11 removes that parameter. Following
> the Files list literally would have raised `TypeError: sample_uniform() got an unexpected
> keyword argument 'search_radius'` on **every `frame_selection: uniform` run** — production, not
> just tests. The task fixed it despite it nominally belonging to Task 13, which is what the
> plan's own "Coverage audit" preamble tells tasks to do.
>
> Two more reader classes the Files list missed, both invisible to a grep for `sample_uniform`:
> `tests/wrapper/test_reconstructor.py` asserted `"search_radius": 3` inside a **dict** of
> expected kwargs, and `tests/wrapper/test_reconstructor_preprocess.py` **monkeypatched**
> `sample_uniform` wholesale. Nine further call sites in `tests/preproc/test_sampling.py` passed
> the kwarg positionally-by-name. **That is the fourth task in a row where the Files list was
> the wrong work surface.**
>
> ### Other plan errors
>
> **`:2755` — "delete `_sample_by_quality` in full (lines 238-336)" is false and would have
> broken `sample_fps`.** `sample_fps` still calls it, and `sample_fps` belongs to **Task 12**.
> Obeying the line numbers yields `NameError` across most of `tests/preproc` and every
> `frame_selection: fps` production run. Landed instead: `_sample_by_quality` kept, the two new
> helpers added beside it, and only its decode tail delegated to `_decode_selection`.
> **This is the third time a line-numbered "delete lines N-M" instruction in this plan has been
> wrong.** Later tasks: anchor on content, never on line numbers.
>
> **Task 11's Step 1 test snippets put `import numpy as np` / `import pytest` inside the test
> functions**, which `CLAUDE.md` § Code Style forbids outright ("no inline imports inside
> functions"). Hoisted to the module header. The plan's own code samples are not exempt from the
> repo's style rules, and several later tasks copy this same shape.
>
> ### A deliberate red, retired at the merge — and why it was not left standing
>
> `test_sampling_parity.py::test_uniform_matches_baseline` went red exactly as intended:
> `assert [0, 70, 144, ...] == [0, 83, 167, ...]`, `At index 1 diff: 70 != 83`. Reproduced by the
> coordinator on the merged tip before acting on it (`1 failed, 1 passed in 306.64s`) — a task's
> own report is not sufficient grounds to delete a test.
>
> `cases["uniform_max30"]` encodes the window-argmax substitution **this task deleted**, so the
> assertion had become an assertion of the bug. It was **retired in `706c2d38`, not rebaselined**
> — regenerating the baseline would have made the test a tautology that passes against whatever
> the code happens to do.
>
> It was also not left red for Task 13 to collect, even though Task 13 nominally owns the file.
> Carrying a knowingly-false failure across four pending tasks means every later brief has to
> contain the sentence "one failure is expected" — and that sentence is the exact mechanism by
> which a real regression gets waved through. **The integration branch stays green; a red is
> either fixed or explained away in the same session that creates it.**
>
> `test_fps_matches_baseline` is untouched and still green: `sample_fps` keeps its window until
> Task 12. Task 13's text has been corrected to delete the file only *after* Task 12 lands.
>
> ### An error in the coordinator's own brief
>
> The dispatch brief told the task that **Task 12** owned `test_sampling_parity.py`. It does not
> — `:3386`, inside **Task 13**, does. No consequence here (the task touched neither the file nor
> its baseline), but it is the same defect class the plan keeps producing, committed by the
> coordinator rather than the plan: **routing by recollection instead of by grep.** Verify
> ownership against the task text before putting it in a brief.
>
> ### Stale prose routed to Task 13
>
> | Location | What is now false |
> |---|---|
> | `configs/base.yaml:38-40` | "fps/uniform: each target is replaced by the sharpest usable frame within +-search_radius" — true for fps only |
> | `collab_splats/wrapper/reconstructor.py:206-208` | "vda_context_fps restricts every selected frame (target and blur substitute)" — uniform has no substitute now |
>
> Task 13 deletes the knob and owns `extract_frames`, so both land there. Verified against Task
> 13's Files list, which does name `configs/base.yaml` and `reconstructor.py` — unlike the Task 28
> routing earlier in this plan, this one is covered.

---

### Task 12: `sample_fps` snaps to the pool, and `context_indices` moves here

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`sample_fps`), `collab_splats/preproc/video.py:189-211` (delete `context_indices`)
- Test: `tests/preproc/test_sampling.py`, `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_sampling.py`:

```python
def test_sample_fps_snaps_targets_to_the_nearest_eligible_frame(monkeypatch, tmp_path):
    """
    Constant-rate targets land on the closest eligible index, never on a condemned one.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    # 60 frames at 30 fps; frames 10-14 blurred out. fps=3 targets 0,10,20,...
    report = _report([400.0] * 10 + [2.0] * 5 + [400.0] * 45)

    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    frames, records = sampling.sample_fps(str(tmp_path / "v.mp4"), fps=3.0, report=report)
    picked = [r["frame_idx"] for r in records]

    # target 10 is condemned; 9 and 15 are equidistant-ish, 9 is nearer
    assert 10 not in picked
    assert 9 in picked
    assert picked == sorted(set(picked))


def test_sample_fps_respreads_outside_the_band(monkeypatch, tmp_path, caplog):
    """
    A count outside [min_frames, max_frames] re-spreads over the whole pool, never truncates.
    """
    import numpy as np

    from collab_splats.preproc import sampling

    report = _report([400.0] * 60)
    monkeypatch.setattr(sampling, "get_video_info", lambda p, **k: {"total_frames": 60, "fps": 30.0})
    monkeypatch.setattr(
        sampling,
        "iter_frames",
        lambda p, indices=None: [(i, np.full((4, 4, 3), i % 251, np.uint8)) for i in indices],
    )

    with caplog.at_level("WARNING"):
        frames, records = sampling.sample_fps(
            str(tmp_path / "v.mp4"), fps=15.0, report=report, max_frames=6
        )

    picked = [r["frame_idx"] for r in records]
    assert len(picked) == 6
    assert picked[-1] >= 55, "re-spread must still span the video, not truncate at frame 6"
    assert "re-spread" in caplog.text


def test_context_indices_lives_in_sampling():
    """
    It computes a selection grid, so it belongs to this module.
    """
    from collab_splats.preproc.sampling import context_indices

    assert context_indices("x.mp4", target_fps=2.0, info={"total_frames": 10, "fps": 10.0}) == [0, 5]
```

And in `tests/preproc/test_video.py`, delete every `context_indices` test and add:

```python
def test_video_no_longer_exports_context_indices():
    """
    context_indices moved to sampling; video.py decodes, it does not select.
    """
    from collab_splats.preproc import video

    assert not hasattr(video, "context_indices")
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py -k "fps or context" tests/preproc/test_video.py -k context -v
```

Expected: FAIL, `ImportError: cannot import name 'context_indices' from 'collab_splats.preproc.sampling'`.

- [ ] **Step 3: Move `context_indices`**

Cut `context_indices` out of `collab_splats/preproc/video.py` (lines 189-211) and paste it into `collab_splats/preproc/sampling.py` above `_eligible`, with the docstring rewritten to house style:

```python
def context_indices(video_path: str | Path, *, target_fps: float, info: dict | None = None) -> list[int]:
    """
    Source frame indices on a constant-rate grid at target_fps.

    Args:
        video_path: source video.
        target_fps: grid rate; must be positive.
        info: a get_video_info dict, to hoist the probe out of a loop.

    Returns:
        Ascending source frame indices. Stride floors at 1 — a rate above the
        source rate cannot sample sub-frame.
    """
    # target_fps is the contract here, so an absent one is a config error, not a default
    if target_fps is None or target_fps <= 0:
        raise ValueError(f"context_indices needs a positive target_fps, got {target_fps!r}")

    # Reuse a caller's probe when given — a fresh one costs a container parse
    info = info if info is not None else get_video_info(video_path)
    total = info["total_frames"]
    if total == 0:
        return []

    native_fps = info["fps"] or 30.0
    step = max(1, int(round(native_fps / target_fps)))
    return list(range(0, total, step))
```

Add `from pathlib import Path` to `sampling.py` if absent. Then update the imports:

- `sampling.py`: `from collab_splats.preproc.video import get_video_info, iter_frames` (drop `context_indices`).
- Anything importing `context_indices` from `video`:

```bash
git grep -n 'context_indices'
```

Expected callers: `collab_splats/wrapper/reconstructor.py` (VDA context grid) and `collab_splats/preproc/__init__.py` if it lists it. Repoint them at `collab_splats.preproc.sampling`.

- [ ] **Step 4: Rewrite `sample_fps`**

Replace the body of `sample_fps` from `info = get_video_info(...)` to its `return`, and drop `search_radius` from the signature:

```python
def sample_fps(
    video_path: str,
    *,
    fps: float,
    report: dict,
    min_frames: int | None = None,
    max_frames: int | None = None,
    quality: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    candidates: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """
    One frame every 1/fps seconds, each snapped to the nearest eligible frame.

    Args:
        video_path: source video.
        fps: target rate — the SPACING is the contract, the count floats.
        report: a quality report from qa.compute_video_quality or qa.load_video_quality.
        min_frames: floor; a count below it re-spreads over the whole video.
        max_frames: ceiling; a count above it re-spreads over the whole video.
        quality: overrides for filter_frame_quality's thresholds.
        on_progress: optional (done, total) callback.
        candidates: optional index grid every pick must be a member of.

    Returns:
        (frames, records) — (H, W, 3) uint8 RGB arrays and their {frame_idx, blur_score} rows.
    """
    # fps is the contract here, so an absent one is a config error, not a default
    if fps is None or fps <= 0:
        raise ValueError(f"sample_fps needs a positive fps, got {fps!r}")

    info = get_video_info(str(video_path))
    total = info["total_frames"]
    if total == 0:
        return [], []

    pool = _eligible(report, quality=quality, candidates=candidates)

    # One source of truth for the stride: a context grid built at this same rate
    # contains these targets by construction, not by coincidence
    targets = context_indices(video_path, target_fps=fps, info=info)

    # Clamp the floating count into the band by re-spreading over the pool, never by
    # truncating — truncation would hand the reconstructor half a scene
    requested = len(targets)
    bounded = requested
    if max_frames is not None:
        bounded = min(bounded, max_frames)
    if min_frames is not None:
        bounded = max(bounded, min(min_frames, total))

    if bounded != requested:
        targets = pool[np.linspace(0, pool.size - 1, min(bounded, pool.size)).round().astype(int)].tolist()
        logger.warning(
            "fps=%.3f wanted %d frames, outside [min_frames=%s, max_frames=%s]; re-spread to "
            "%d frames over the whole video (effective %.3f fps)",
            fps,
            requested,
            min_frames,
            max_frames,
            len(targets),
            (info["fps"] or 30.0) * len(targets) / total,
        )

    # Snap each target to the nearest eligible frame, then dedup: two targets either
    # side of an excised stretch can snap to the same survivor.
    pos = np.clip(np.searchsorted(pool, targets), 1, pool.size - 1)
    left, right = pool[pos - 1], pool[pos]
    snapped = np.where(np.abs(targets - left) <= np.abs(right - targets), left, right)
    chosen = sorted(set(snapped.tolist()))

    return _decode_selection(video_path, chosen, report=report, on_progress=on_progress, desc="fps sampling")
```

The `np.clip(..., 1, pool.size - 1)` is deliberate: it makes `pos - 1` and `pos` both valid for a single-element pool, where they collapse to the same index and the `where` picks it either way.

- [ ] **Step 5: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py tests/preproc/test_video.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc tests/preproc
isort collab_splats/preproc tests/preproc
git commit --only collab_splats/preproc tests/preproc \
  -m "refactor(preproc)!: sample_fps snaps to the eligible pool; context_indices moves to sampling

BREAKING: preproc.video.context_indices is now preproc.sampling.context_indices."
```

---

> ## Task 13 pre-flight — the grep is too wide, the Files list too narrow
>
> Measured on the integration tip, 2026-09-05. Task 13's Step 1 runs `git grep -n 'search_radius'`
> and predicts the hits. The prediction is wrong in both directions: it names a category that
> does not exist, misses two test modules, and its unanchored pattern sweeps in four lines of
> unrelated Open3D.
>
> | Hit | What it is | Owner |
> |---|---|---|
> | `tests/wrapper/test_vda_context.py:255,280` | passes `search_radius=7` into the sampler call | **Task 13** — in neither the Files block nor Step 1's expected list |
> | `tests/wrapper/test_reconstructor_preprocess.py:116-130` | the whole test is `test_preprocess_forwards_search_radius`, and its `fake_uniform` stub signature carries `search_radius=3` | **Task 13** — glob-covered by Step 1, absent from the Files block |
> | `docs/superpowers/plans/scaffold-runs/gopro_scaffold_2dgs_500f_50k.yaml:28` | a run config that *sets* the key | **Task 13** — see "goes inert" below |
> | `collab_splats/pointcloud/utils.py:273,369,478` | Open3D `tree.search_radius_vector_3d(...)` | **correct as written — do not touch** |
> | `collab_splats/mesh/utils.py:553` | same Open3D call | **correct as written — do not touch** |
>
> **Step 1's pattern is unanchored.** `search_radius` is a prefix of Open3D's
> `search_radius_vector_3d`, so the literal command in the plan returns four hits in
> `pointcloud/utils.py` and `mesh/utils.py` that have nothing to do with frame sampling.
> Use `git grep -nw 'search_radius'` instead, or the task starts by editing the KD-tree
> queries. This is the mirror of the Task 9 finding: there the grep was too narrow to see
> `frames.zarr`; here it is too wide.
>
> **Step 1 predicts notebooks. There are none.** `git grep -n 'search_radius' -- '*.ipynb'`
> returns nothing. Drop that clause rather than hunting for a file that does not exist.
>
> **`configs/base.yaml` and `reconstructor.py:959` must move in ONE commit.** The read is a
> bare dict lookup with no default:
>
> ```python
> search_radius=pre_cfg["search_radius"],
> ```
>
> and nothing whitelists the `preproc` block — `validate_config` only renames the old
> `preprocessing` key (`reconstructor.py:828-834`). So deleting the base.yaml line while
> `:959` still reads it raises `KeyError: 'search_radius'` on **every** video preprocess,
> and the tests that stub the sampler will not catch it because they never reach the config
> read.
>
> **The stray run config goes inert, not red.** Because there is no unknown-key validation,
> `gopro_scaffold_2dgs_500f_50k.yaml:28` keeps deep-merging `search_radius: 7` into the
> preproc block after Task 13, and nobody reads it. No error, no warning — the knob simply
> stops doing anything. That file's own comment is `# the "_r7" the 2dgs baseline scene is
> named for`, so the scene name would go on advertising a setting that no longer exists.
> Delete the line and leave a one-line note that the r7 baseline predates this plan, rather
> than leaving a config that silently lies.
>
> **A test name is a gate failure.** Step 2's gate is `git grep` going clean, and
> `test_preprocess_forwards_search_radius` keeps it dirty even after every call site is
> fixed. Rename it with the change — exactly the trap Task 8b hit with
> `test_run_sfm_falls_back_when_the_video_changed_since_frames_zarr`. And note the stub
> hazard from the Phase D section applies here too: `fake_uniform`'s signature is a
> **contract**, so dropping the parameter from `sample_uniform` without dropping it from the
> stub leaves a test that passes while asserting nothing about the real signature.

### Task 13: retire `search_radius` everywhere

**Files:**
- Modify: `collab_splats/preproc/sampling.py` (`sample_optical_flow`), `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml`
- Test: `tests/preproc/test_sampling.py`, `tests/wrapper/test_reconstructor.py`, `tests/preproc/test_sampling_parity.py`

- [ ] **Step 1: Find every mention**

```bash
git grep -n 'search_radius'
```

Expected: `collab_splats/preproc/sampling.py`, `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml` (where it says `7`), `tests/preproc/test_sampling.py`, `tests/preproc/test_sampling_parity.py`, `tests/wrapper/test_reconstructor*.py`, and the notebooks.

- [ ] **Step 2: Write the failing test**

Append to `tests/preproc/test_sampling.py`:

```python
def test_samplers_no_longer_take_search_radius():
    """
    The window search is gone; the pool replaced it.
    """
    import inspect

    from collab_splats.preproc import sampling

    for fn in (sampling.sample_uniform, sampling.sample_fps, sampling.sample_optical_flow):
        assert "search_radius" not in inspect.signature(fn).parameters, fn.__name__
```

- [ ] **Step 3: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_sampling.py::test_samplers_no_longer_take_search_radius -v
```

Expected: FAIL — `sample_optical_flow` never had it, but the reconstructor still passes it and `base.yaml` still sets it.

- [ ] **Step 4: Delete it**

1. `collab_splats/preproc/sampling.py` — `sample_optical_flow` keeps its signature but swaps its gate to the shared pool. Replace its `usable = filter_frame_quality(...)` line and the loop's gate with:

```python
    pool = set(_eligible(report, quality=quality, candidates=None).tolist())
```

and in the loop:

```python
        # Pool gate first: an ineligible frame never reaches the selector, so it
        # cannot become the reference the next frames are scored against.
        if idx not in pool:
            continue
```

Rewrite its docstring to the `Args:`/`Returns:` house form while you are in there.

2. `collab_splats/wrapper/reconstructor.py` — delete the `search_radius: int = 3` parameter from `extract_frames`, every `search_radius=search_radius` it forwards to a sampler, and the `cfg.preproc.search_radius` read at the call site.

3. `configs/base.yaml` — delete the `search_radius: 7` line and its comment.

4. Delete `search_radius` from every test and notebook the grep found.

- [ ] **Step 5: Run the preproc and wrapper suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/wrapper/ -v
git grep -n 'search_radius'
```

Expected: tests PASS; the grep returns nothing.

`tests/preproc/test_sampling_parity.py` asserted the old window-argmax behaviour in two cases. Do not rewrite either into a test of the new behaviour — Tasks 11-12 already cover that.

> **Half of this is already done — read before you delete.** Merging Task 11 turned
> `test_uniform_matches_baseline` red (`70 != 83` at index 1), because Task 11 is precisely what
> deleted its premise. It was **retired at the merge** (`706c2d38`), not left for you: carrying a
> knowingly-false red across four pending tasks would have forced every later brief to say "one
> failure is expected", and that sentence is how a real regression gets waved through.
>
> What is left for Task 13 is the **fps** half. `test_fps_matches_baseline` is still green and
> still meaningful, because `sample_fps` keeps its window until **Task 12** retires it. So:
> **if Task 12 has landed, delete the file and `data/parity_baseline.json` with it. If it has
> not, delete nothing here** — a green fps parity case is real coverage of a rule that still
> exists, and deleting it early removes the only check on Task 12's behaviour change.
>
> `cases["uniform_max30"]` is deliberately still in the JSON as the captured record of the old
> behaviour. It is unused, and that is intended until the file goes.

- [ ] **Step 6: Commit**

```bash
black collab_splats tests configs
isort collab_splats tests
git commit --only collab_splats/preproc collab_splats/wrapper/reconstructor.py configs/base.yaml tests/preproc tests/wrapper \
  -m "refactor(preproc)!: retire search_radius

BREAKING: preproc.search_radius leaves base.yaml and the samplers. It said 7 in
config and defaulted to 3 in code — nothing depended on either."
```

---

> ## Task 14 pre-flight — measured on the integration tip, 2026-09-05
>
> Run before Task 11 landed, so these are the *filter's* numbers, not the A/B's. Two things
> the executor needs and one bug that would have stopped Step 1 on its first line.
>
> ### The inputs exist
>
> | video | report |
> |---|---|
> | GH010229, 13,115 frames | `/workspace/outputs/rerun_2026_08_23/GH010229/video_quality_report.json` |
> | tutorial, 2,388 frames | `/workspace/collab-splats/data/tutorial/video_quality/video_quality_report.json` |
>
> Both carry every column the filter reads (`laplacian`, `clipped_low_frac`,
> `clipped_high_frac`). No re-measure needed — do not decode either video.
>
> ### The landed filter reproduces the numbers this plan was written against
>
> ```
> GH010229   n=13115  kept=12445  cut=5.11%   log_lap 7.585..9.900  median 9.049  MAD 0.3320
> tutorial   n=2388   kept=2041   cut=14.53%  log_lap 6.738..8.914  median 8.451  MAD 0.2397
> ```
>
> The plan's motivating figures were **5.1%** and **14.5%**. Task 10 shipped with a guard
> different from the one the plan specified (`log_lap.min() == log_lap.max()` instead of
> `spread == 0.0`, see its outcomes block), so this is the check that the deviation changed
> only the degenerate case and not the design. It did.
>
> **The entire cut is the sharpness rule.** Frames cut by clipping alone: **0 on both
> videos** — no frame on either has `clipped_low_frac + clipped_high_frac > 0.25`. So
> `max_clipped_frac` is inert on this data: it is a correctness guard against a blown
> capture, not an active part of the 5.11%. Do not tune it against these two videos; they
> cannot move it. Say this in the A/B write-up, or the reader will assume both rules fired.
>
> ### Step 1's harness does not run as written
>
> ```python
> report = load_video_quality(Path(report_path))
> ```
>
> `qa.py:468` is `load_video_quality(video_path, report_path, *, workers=1, motion_stride=None)`
> — two required positionals. As written this raises
> `TypeError: load_video_quality() missing 1 required positional argument: 'report_path'`
> on the harness's first statement. The reports already exist on disk and the function
> short-circuits on `report_path.exists()`, so the fix is to read them directly:
>
> ```python
> report = json.loads(Path(report_path).read_text())
> ```
>
> which also removes the risk that a typo'd path silently triggers a 13,115-frame decode
> instead of failing. `json` is already imported in the harness.
>
> Verified correct as written, same run: `context_indices("", target_fps=fps, info={...})`
> — the signature is `(video_path, *, target_fps, info=None)` and a supplied `info`
> short-circuits the probe, so the empty `video_path` is never used.

### Task 14: **HARD GATE** — the GH010229 A/B

> **Coordinator note — both reports are on disk; the A/B decodes nothing.** Verified
> 2026-09-05, columns and row counts read directly:
>
> | video | report | frames |
> |---|---|---|
> | GH010229 | `/workspace/outputs/rerun_2026_08_23/GH010229/video_quality_report.json` | 13,115 |
> | tutorial | `/workspace/collab-splats/data/tutorial/video_quality/video_quality_report.json` | 2,388 |
>
> Both match the counts this plan was written against exactly. The GH010229 report names its
> source video at `/workspace/outputs/2026_07_15-Goprosplat-GH010229/GH010229.mp4`, 59.94 fps,
> 218.8 s, 1920x1080 — **do not open it**; the harness needs only the JSON.
>
> The schema is columnar, not a list of per-frame records: `report["frames"]` is a dict of
> eight equal-length lists (`frame_idx`, `blur`, `laplacian`, `exposure_mean`,
> `exposure_median`, `exposure_std`, `clipped_low_frac`, `clipped_high_frac`), with `pairs`,
> `params`, `video` and `available` beside it. A harness that iterates `report["frames"]`
> expecting dicts will silently iterate the eight column *names*.
>
> Because nothing is decoded, this gate is cheap and can run under a live agent wave — it is
> pure JSON and numpy, so the container's "no parallel processes during heavy eval" rule does
> not apply to it. It is still human-read before Phase B ships.


**Files:**
- Create: `/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad/ab_sampling.py`

Nothing in Phase B ships until this runs and a human reads it. Phase B changes which frames a scene is built from — that is the point — and this is what turns "the filter fires" into a number someone signed off on.

- [ ] **Step 1: Write the harness**

```python
"""
A/B the old window-argmax selection against the new eligible-pool selection.

Run against GH010229 (13,115 frames, the video whose 5.1% cut motivated the change)
and against the tutorial video (14.5% cut).
"""

import json
import sys
from pathlib import Path

import numpy as np

from collab_splats.preproc.qa import load_video_quality
from collab_splats.preproc.sampling import _eligible, context_indices, filter_frame_quality


def main(report_path: str, fps: float, max_frames: int) -> None:
    report = load_video_quality(Path(report_path))
    lap = np.asarray(report["frames"]["laplacian"], dtype=float)
    n = lap.size

    mask = filter_frame_quality(report)
    pool = _eligible(report, quality=None, candidates=None)

    # New fps selection: constant-rate targets snapped to the pool
    targets = np.asarray(context_indices("", target_fps=fps, info={"total_frames": n, "fps": 30.0}))
    pos = np.clip(np.searchsorted(pool, targets), 1, pool.size - 1)
    left, right = pool[pos - 1], pool[pos]
    new = np.unique(np.where(np.abs(targets - left) <= np.abs(right - targets), left, right))

    # Old fps selection: the same targets, unfiltered (the old mask never fired)
    old = np.unique(targets)

    print(f"frames               {n}")
    print(f"cut by filter        {100 * (1 - mask.mean()):.1f}%")
    print(f"old picks            {old.size}")
    print(f"new picks            {new.size}")
    print(f"unchanged            {np.intersect1d(old, new).size}")
    print(f"max index shift      {int(np.abs(new - old[: new.size]).max()) if new.size else 0}")
    print(f"old mean laplacian   {lap[old].mean():.1f}")
    print(f"new mean laplacian   {lap[new].mean():.1f}")
    print(f"old min laplacian    {lap[old].min():.1f}")
    print(f"new min laplacian    {lap[new].min():.1f}")

    # Largest gap in source frames — the bunching the spec predicts
    print(f"old max gap          {int(np.diff(old).max()) if old.size > 1 else 0}")
    print(f"new max gap          {int(np.diff(new).max()) if new.size > 1 else 0}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]))
```

- [ ] **Step 2: Run it on both videos**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad
/opt/venv/reconstruction/bin/python $SCRATCH/ab_sampling.py <gh010229 scene>/video_quality_report.json 2.0 300
/opt/venv/reconstruction/bin/python $SCRATCH/ab_sampling.py <tutorial scene>/video_quality_report.json 2.0 300
```

If no report exists for GH010229, generate one first:

```bash
/opt/venv/reconstruction/bin/python -c "
from pathlib import Path
from collab_splats.preproc.qa import compute_video_quality
compute_video_quality('<path to GH010229.MP4>', output_path=Path('$SCRATCH/gh_report.json'))
"
```

- [ ] **Step 3: Read the numbers against these expectations**

| line | expected | what a miss means |
|---|---|---|
| `cut by filter` GH010229 | ~5.1% | a different number means the filter is not the one measured in the spec |
| `cut by filter` tutorial | ~14.5% | same |
| `new min laplacian` | strictly above `old min laplacian` | the filter is not removing the soft frames it exists to remove |
| `new max gap` | larger than `old max gap` | the bunching the spec predicts; a gap of hundreds of frames means a long excised stretch — check it really is blurry footage |
| `unchanged` | most picks | a low overlap means the snap is moving frames it should not |

- [ ] **Step 4: Present to the user and stop**

Paste both tables. State plainly which frames moved and by how much. **Do not start Phase C until the user reads this and says go.** If `new max gap` is large, spot-check the excised stretch by writing those frames out and looking at them — a filter that condemns a correctly-exposed static shot is a bug, not a feature.


> ## Task 14 outcomes — **HARD GATE RUN AND PASSED**, coordinator, 2026-09-05
>
> Run on `preproc-t12` pinned at `05bf3094` (Task 12 + Task 13 both landed; re-verified as the
> branch tip after the run). Pure JSON + numpy — **neither video was decoded**, so this ran
> safely under a live three-agent wave.
>
> ### Three corrections the harness needed before it produced a number
>
> 1. `load_video_quality(Path(p))` — two required positionals, as the pre-flight said. Read the
>    JSON directly.
> 2. **The plan hardcodes `info={"total_frames": n, "fps": 30.0}`.** GH010229 is **59.94 fps**.
>    A hardcoded 30 halves the stride: the grid picks ~2x the frames and covers only the first
>    half of the video, so every downstream figure would have been wrong in a way that still
>    looked plausible. Read the true rate off `report["video"]["fps"]`.
> 3. `max_frames` was accepted and never used. Dropped from the signature.
>
> Also fixed: `max index shift` compared `new` against `old[:new.size]`, which pairs by position
> across arrays of different length. Replaced with the per-target shift `|snapped - target|`,
> which is well defined.
>
> ### The numbers
>
> | line | GH010229 | tutorial | expected | verdict |
> |---|---|---|---|---|
> | frames | 13,115 @ 59.94 fps | 2,388 @ 23.98 fps | — | — |
> | cut by filter | **5.11%** (670) | **14.53%** (347) | ~5.1% / ~14.5% | **PASS** |
> | eligible pool | 12,445 | 2,041 | — | — |
> | old picks | 438 | 199 | — | — |
> | new picks | 429 (9 collisions) | 193 (6 collisions) | — | — |
> | unchanged | 414 / 438 = **94.5%** | 169 / 199 = **84.9%** | most picks | **PASS** |
> | targets that moved | 24 / 438 | 30 / 199 | — | — |
> | max index shift | 153 | 17 | — | — |
> | mean index shift | 2.15 | 0.94 | — | — |
> | old min laplacian | 2,140.3 | 946.0 | — | — |
> | new min laplacian | **4,389.1** | **2,910.3** | strictly above old | **PASS** |
> | old mean laplacian | 9,038.7 | 4,389.9 | — | — |
> | new mean laplacian | 9,200.2 | 4,605.2 | — | — |
> | old max gap | 30 | 12 | — | — |
> | new max gap | **324** | **40** | larger than old | **PASS** |
>
> All five of Step 3's expectations hold on both videos.
>
> ### Step 4's spot-check of the excision — it is not a bug
>
> The largest gap on GH010229 spans **frames 4899 -> 5223**, 324 source frames = **5.4 s**.
> Step 4 requires this be confirmed as genuinely soft footage rather than a correctly-exposed
> static shot the filter wrongly condemned. Checked from the report alone, no decode:
>
> | | excised stretch 4899-5223 | whole video |
> |---|---|---|
> | laplacian min / median / max | 2,052 / **2,983** / 4,507 | 1,968 / **8,514** / 19,923 |
> | exposure_mean median | 84.2 | 106.1 |
> | frames cut by clipping | **0** | 0 |
> | kept by the filter | **2 / 325** | 12,445 / 13,115 |
>
> The stretch is **2.9x softer than the video's median, and its sharpest frame (4,507) is still
> below the cut line** — with `sharpness_k=2.0`, median log-lap 9.0495 and scaled MAD 0.3319 put
> the threshold at 8.386, while the stretch's log-lap range is 7.626..8.413. It is **one
> contiguous rejected run**, frames 4900-5222 (323 frames), not scattered rejections — the
> signature of a real blurry/dark passage, not of a threshold nibbling at a good shot. It is
> also darker than the video median. **Verdict: correct excision.**
>
> ### The clipping rule is inert on both videos
>
> Confirmed again here: **zero frames on either video are cut by clipping.** The entire 5.11%
> and 14.53% is the sharpness rule. `max_clipped_frac=0.25` is a correctness guard against a
> blown capture, not an active part of these numbers — do not tune it against this data, it
> cannot move.
>
> ### One config consequence worth stating
>
> `filter_frame_quality`'s `sharpness_k` and `max_clipped_frac` are **code-only**. They are
> reachable solely through the samplers' `quality=` kwarg, which `extract_frames` does not
> expose, and there is no `preproc.quality` config block. So the A/B cannot vary them from
> config, and no scene config can currently loosen or tighten the 5.11%.
>
> Harness: `scratchpad/ab_sampling.py`, spot-check: `scratchpad/gapcheck.py`.

---

## Phase C — undistortion: pycolmap picks the camera, cv2 moves the pixels

`undistort` defaults to `false`, so nothing published depends on this phase. That is exactly why the framing change (COLMAP's fixed-focal expanding canvas, not cv2's fixed-canvas shrinking focal) can land without a toggle.

---

> ## Task 15 pre-flight — four readers of `DistortionProfile` no Files list names
>
> Measured on the integration tip, 2026-09-05, by inventorying **readers** rather than
> importers — the rule Task 8b and Task 10 both had to learn the expensive way. Task 15's
> Files block is two entries (`collab_splats/preproc/undistort.py`, `tests/preproc/test_undistort.py`)
> and Task 16's and Task 17's add one module each. `git grep -n 'DistortionProfile\|estimate_camera_distortion'`
> returns four production/test readers outside all three lists. Two of them do not fail a
> test — they fail **collection**, which aborts the entire run.
>
> | Reader | What it does | Owner |
> |---|---|---|
> | `tests/preproc/test_sampling.py:426-448` | `test_public_api_surface` asserts **exact set equality** on `preproc.__all__` | **Task 15** — must edit in the same commit |
> | `collab_splats/preproc/__init__.py:22,23,29,33` | imports and re-exports both deleted names | **Task 15** — named in its Step text, missing from its Files block |
> | `tests/wrapper/test_vda_context.py:22,336` | `from ...undistort import DistortionProfile`, then constructs one | **Task 17** — add to its Files list |
> | `tests/preproc/test_video.py:10,284` | same import, constructs at 320x240, passes `profile=` to `decode_context` | **Task 17** — add to its Files list |
> | `collab_splats/preproc/video.py:273` | docstring line naming `DistortionProfile` | **Task 17** — ships with its `profile:` -> `camera:` signature change |
>
> **The collection hazard.** `tests/wrapper/test_vda_context.py` and `tests/preproc/test_video.py`
> import `DistortionProfile` at module scope. The moment Task 15 deletes the class those two
> modules raise `ImportError` while pytest is still collecting, and pytest reports
> `Interrupted: N errors during collection` — **zero tests run**. So the symptom is not
> "two tests went red at the end of a green run"; it is a suite that produces no results at
> all, in two directories Task 15 never touches. `test_video.py:284` breaks twice over: it
> names the parameter (`decode_context(tiny_video, [0, 3], profile=profile, ...)`), so
> Task 17's rename to `camera=` is a second, independent break on the same line.
>
> **The `__all__` set is a three-way merge landmine.** `test_public_api_surface` asserts
> equality against a literal set, and three separate tasks edit that one literal:
>
> * today it is **14** names, and the comment above it says so in prose — `# Exactly the 14 public names`;
> * **Task 9** removes `FrameStore` and adds five `frames` names;
> * **Task 15** removes `DistortionProfile` and `estimate_camera_distortion` and adds `calibrate_camera`.
>
> 14 - 3 + 1 + 5 = **17**, which is what Step 5 of Task 15 already predicts and what the
> spec's section 13 lists. Whichever of the two merges second will hit a conflict in that
> set *and* in the prose count one line above it. Re-read the file at merge time; do not
> reconstruct the set from either task's diff alone. This is the same shape as the
> `_clean_report` landmine recorded in the Task 10 outcomes.
>
> **Do not chase the historical plans.** `git grep` also hits
> `docs/superpowers/plans/2026-08-25-splatfacto-parity.md` (which *built* `DistortionProfile`),
> `docs/superpowers/plans/2026-08-26-depth-align-affine-2dgs-surface.md`, and the design spec
> for this plan. All are **correct as written** — they are records of what was true when they
> shipped, not live references. Editing them would falsify the history. Only the five rows in
> the table above are orphans.

### Task 15: `calibrate_camera` returns a `pycolmap.Camera`

**Files:**
- Rewrite: `collab_splats/preproc/undistort.py` (delete `DistortionProfile` and `estimate_camera_distortion`)
- Test: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing tests**

Replace the `DistortionProfile` and `estimate_camera_distortion` tests in `tests/preproc/test_undistort.py` with:

```python
def test_calibrate_camera_returns_a_pycolmap_camera(tmp_path):
    """
    Calibration's output type IS pycolmap's, so nothing round-trips through a dataclass.
    """
    import pycolmap

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    cam = calibrate_camera(images, max_frames=12)

    assert isinstance(cam, pycolmap.Camera)
    assert cam.model.name == "OPENCV"
    assert (cam.width, cam.height) == (320, 240)


def test_calibrate_camera_stages_no_image_copies(tmp_path, monkeypatch):
    """
    pycolmap.extract_features(image_names=...) reads the scene's images/ in place.
    """
    import cv2
    import pytest

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    _write_textured_sequence(images, n=12, width=320, height=240)

    # The old code staged JPEG copies into a tempdir; any write here means it still does.
    # The COLMAP database still gets a tempdir — that is scratch, not a copy of the images.
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: pytest.fail("staged a temporary image copy"))
    calibrate_camera(images, max_frames=12)


def test_calibrate_camera_raises_when_registration_is_thin(tmp_path):
    """
    A featureless sequence cannot calibrate, and says so rather than returning nonsense.
    """
    import cv2
    import numpy as np
    import pytest

    from collab_splats.preproc.undistort import calibrate_camera

    images = tmp_path / "images"
    images.mkdir()
    for i in range(12):
        cv2.imwrite(str(images / f"frame_{i:06d}.png"), np.full((240, 320, 3), 128, np.uint8))

    with pytest.raises(RuntimeError, match="registered"):
        calibrate_camera(images, max_frames=12)
```

`_write_textured_sequence` is the existing helper in this file that renders a moving textured pattern; keep it. If it does not exist, add it:

```python
def _write_textured_sequence(out_dir, *, n, width, height):
    """
    n frames of a high-frequency pattern translating a few pixels per frame.
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (height + 4 * n, width + 4 * n, 3), dtype=np.uint8)
    for i in range(n):
        crop = canvas[2 * i : 2 * i + height, 2 * i : 2 * i + width]
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), crop)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k calibrate -v
```

Expected: FAIL, `ImportError: cannot import name 'calibrate_camera'`.

- [ ] **Step 3: Rewrite the calibration half of `undistort.py`**

Delete `DistortionProfile` (the whole frozen dataclass, `to_dict`, `from_dict`, `K`, `dist_coeffs`) and `estimate_camera_distortion`. Put this in their place:

```python
"""
Camera calibration and undistortion.

pycolmap estimates the camera and picks the undistorted framing; cv2 moves the
pixels. The camera IS a pycolmap.Camera — there is no local dataclass mirroring it.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

logger = logging.getLogger(__name__)

# SIFT reads the cgroup's core count, not the container's cap: 96 host cores on a
# 46.6 GB cgroup OOMs. See project_splatfacto_parity.
_SIFT_NUM_THREADS = 8


def calibrate_camera(images_dir: Path, *, max_frames: int = 60) -> pycolmap.Camera:
    """
    Estimate one shared OPENCV camera from a scene's images.

    Args:
        images_dir: the scene's images/ directory.
        max_frames: how many evenly-spaced images to calibrate from.

    Returns:
        A pycolmap.Camera (model OPENCV) carrying f, pp and k1 k2 p1 p2.
    """
    paths = frames.frame_paths(images_dir)
    if len(paths) < 8:
        raise ValueError(f"calibrate_camera needs at least 8 images, found {len(paths)} in {images_dir}")

    # Evenly-spaced subset: calibration wants baseline, not every frame
    idxs = np.linspace(0, len(paths) - 1, min(max_frames, len(paths))).round().astype(int)
    names = [paths[i].name for i in np.unique(idxs)]

    # The database is scratch; the images are read from images_dir in place
    with tempfile.TemporaryDirectory(prefix="calib_db_") as tmp:
        database = Path(tmp) / "database.db"

        pycolmap.extract_features(
            database,
            images_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=pycolmap.ImageReaderOptions(camera_model="OPENCV"),
            extraction_options=pycolmap.FeatureExtractionOptions(num_threads=_SIFT_NUM_THREADS),
        )
        pycolmap.match_exhaustive(database)

        recons = pycolmap.incremental_mapping(database, images_dir, Path(tmp) / "sparse")

    if not recons:
        raise RuntimeError(f"calibration failed: no reconstruction from {len(names)} images in {images_dir}")

    recon = recons[0]

    # A reconstruction over a minority of the input has not seen the lens
    if len(recon.images) < 0.6 * len(names):
        raise RuntimeError(
            f"calibration too thin: {len(recon.images)} of {len(names)} images registered "
            f"(need 60%); the footage may be featureless or the motion degenerate"
        )

    camera = next(iter(recon.cameras.values()))
    logger.info("calibrated %s from %d/%d images: %s", images_dir, len(recon.images), len(names), camera)
    return camera
```

The `TemporaryDirectory` that survives holds the COLMAP database and sparse output — scratch, not copies of the images. What is deleted is the old `cv2.imwrite` JPEG staging loop that wrote every calibration frame out a second time; `image_names=` makes it unnecessary, and that is what the test above pins.

`from collab_splats.preproc import frames` is already in the import block shown above.

- [ ] **Step 3b: Rewire the package exports**

`collab_splats/preproc/__init__.py` still imports the two deleted names. Replace:

```python
from collab_splats.preproc.undistort import DistortionProfile, estimate_camera_distortion, undistort_frames
```

with:

```python
from collab_splats.preproc.undistort import calibrate_camera, undistort_frames
```

and in `__all__`, drop `"DistortionProfile"` and `"estimate_camera_distortion"`, add `"calibrate_camera"`, keeping the list sorted. With Task 9's five `frames` names this brings `__all__` to the 17 entries the spec's section 13 lists.

Then check nothing else imports them:

```bash
git grep -n 'DistortionProfile\|estimate_camera_distortion'
```

Expected after this task: nothing outside `docs/`. `collab_splats/wrapper/reconstructor.py` still has hits at this point — Task 17 clears them, and the tree is red between the two. Do Tasks 15, 16 and 17 back to back.

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k calibrate -v
```

Expected: PASS. These run real SIFT + mapping on 12 tiny images; budget ~20 s.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
isort collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
git commit --only collab_splats/preproc/undistort.py tests/preproc/test_undistort.py \
  -m "refactor(preproc)!: calibrate_camera returns a pycolmap.Camera

BREAKING: DistortionProfile and estimate_camera_distortion are deleted."
```

> ## Task 15 outcomes — landed `129069cb` + `3a4a7828`, not yet merged 2026-09-05
>
> Gates, each stamped with the SHA it ran against. `tests/preproc` **`181 passed, 9 warnings in
> 423.52s`** @ `24d11cbb` → **`181 passed, 1 xfailed, 9 warnings in 898.93s`** @ `129069cb`
> (net-zero test count: 10 old tests in `test_undistort.py` out, 10 + 1 xfail in; the 2.1×
> wall-clock is real SIFT + mapper now running in that directory). `pytest tests/ --collect-only
> -q` @ `3a4a7828`: **`1698 tests collected, 27 errors`**. `tests/wrapper tests/pointcloud
> --ignore=.../test_splats_stage.py` **`22 failed, 582 passed, 15 skipped in 442.00s`** @
> `24d11cbb` → **`Interrupted: 14 errors during collection`** @ `3a4a7828` — see (b). pyflakes 0,
> black and isort clean on all five touched files. `__all__` landed at **13**, not 17 — see (a).
>
> ### Nine things this task's own text got wrong
>
> **(a) `:3595` — "14 - 3 + 1 + 5 = 17" is not the number Task 15 can land.** The `-3 +5`
> half is Task 9's, and Task 9 has **not landed on this task's base**: `git show
> 24d11cbb:collab_splats/preproc/__init__.py` still imports `FrameStore` and `__all__` is still
> 14 names. Task 15 alone is `14 - 2 + 1 = ` **13**, which is what shipped, with the prose
> comment at `tests/preproc/test_sampling.py:428` moved from `14` to `13`. There are now
> **three different numbers in play** — the pre-flight's predicted **17**, this task's landed
> **13**, and the coordinator's stated reconciled target of **12** after Task 9 — which is the
> proof that the set cannot be computed from any one task's diff. **Re-read
> `collab_splats/preproc/__init__.py` at merge time and copy the set; do not do the arithmetic.**
>
> **(b) `:3579` — "two modules import `DistortionProfile` at module scope" undercounts by 9×.**
> The pre-flight names `tests/wrapper/test_vda_context.py` and `tests/preproc/test_video.py`.
> Measured after the delete: **18** modules error during collection, because
> `collab_splats/wrapper/reconstructor.py:44-45` imports both deleted names at **module scope**,
> so every test module that imports the reconstructor dies with it — `tests/examples/
> test_run_pipeline.py`, `tests/geometry/test_metrics.py`, `tests/mesh/test_absent_confidence.py`,
> `tests/pointcloud/test_loger_creator.py`, `tests/remote/test_rerun.py`, and thirteen of
> `tests/wrapper/`. The pre-flight table routes `reconstructor.py` to Task 17 but never states
> that its import is the collection hazard. (The other 9 of the 27 collection errors are
> pre-existing and unrelated: `ImportError: cannot import name 'losses' from 'gsplat'`.)
> **This is why `tests/wrapper tests/pointcloud` cannot be run as a gate between Tasks 15 and 17
> at all** — not "runs with more failures", but *zero tests executed*.
>
> **(c) `:3816` — Step 4's "Expected: PASS" is unreachable as written.** Task 15's own Files
> list names `tests/preproc/test_undistort.py`, and that file's base version imports the
> reconstructor at module scope (`24d11cbb:tests/preproc/test_undistort.py:11`, `import
> collab_splats.wrapper.reconstructor as recon_mod`). After Step 3 the file cannot be collected,
> so Step 4 returns a collection error, not a pass. Landed fix: the one test that needs the
> reconstructor (`test_extract_frames_dir_no_undistort_no_payload`) defers the import into the
> body and carries `@pytest.mark.xfail(strict=True)`. **`strict=True` is deliberate: when Task 17
> repairs `reconstructor.py`, this test starts passing and `strict` turns that into a FAILURE,
> so Task 17 cannot forget to delete the marker.**
>
> **(d) `:3784` — "`from collab_splats.preproc import frames` is already in the import block
> shown above" is false.** The block at `:3713-3722` is `logging`, `tempfile`, `pathlib`, `cv2`,
> `numpy`, `pycolmap` and nothing else, while the body at `:3741` calls `frames.frame_paths(...)`.
> Copied literally, `calibrate_camera` raises `NameError: name 'frames' is not defined` on its
> first line. Landed as `from collab_splats.preproc.frames import frame_paths`.
>
> **(e) `:3672` + `:3766` — the task's own test cannot pass against the task's own code.**
> Step 1's featureless-sequence test asserts `pytest.raises(RuntimeError, match="registered")`.
> Measured on exactly that fixture (12 constant-gray 320x240 frames): `incremental_mapping`
> returns `{}`, so the branch that fires is `:3766`, whose message is `"calibration failed: no
> reconstruction from 12 images in ..."` — **no substring `registered`**. Only the *other*
> branch (`:3773`) contains the word. Both landed messages now carry it.
>
> **(f) `:3768` — `recons[0]` is a dict-key lookup, not "the first model".**
> `pycolmap.incremental_mapping` returns `dict[int, Reconstruction]` keyed by model id (measured:
> `type(recons) is dict`, `keys == [0]` on a 12-image solve, and `recons[0]` raises `KeyError: 0`
> on the empty dict). On a single-model solve it works by coincidence; on a multi-model solve
> key `0` is neither guaranteed to exist nor to be the largest model. Landed as
> `max(recons.values(), key=lambda r: r.num_reg_images())`. Relatedly `:3771`'s `len(recon.images)`
> is not the registered count — `num_reg_images()` is; they happened to agree (12 == 12) on this
> fixture, which is exactly why the wrong one survives review.
>
> **(g) `:3761` — Step 3 silently drops a thread cap the base code had, and Step 1 deletes the
> only test that would have caught it.** Base `undistort.py:134-137` passed
> `matching_options=FeatureMatchingOptions(num_threads=_SIFT_NUM_THREADS)`; the plan's
> replacement is a bare `pycolmap.match_exhaustive(database)`. Measured on this box:
> `pycolmap.FeatureMatchingOptions().num_threads == -1`, i.e. one matcher thread per **host**
> core (96) under a 46.6 GB cgroup — the OOM recorded in `project_splatfacto_parity`, which
> `_SIFT_NUM_THREADS` exists to prevent. Step 1 says to replace "the `DistortionProfile` and
> `estimate_camera_distortion` tests", which deletes base
> `test_estimate_camera_distortion_caps_sift_threads` — the only assertion of
> `{"extract": 8, "match": 8}`. Both cap and test are kept, renamed
> `test_calibrate_camera_caps_sift_threads`.
>
> **(h) `:3623-3690` — every prescribed test imports inside the test body**, against
> `CLAUDE.md`'s "Imports at top: no inline imports inside functions or methods". Landed at
> module scope. (The one deliberate exception is (c)'s deferred reconstructor import, which is
> commented as such.)
>
> **(i) `:3823` — Step 5's commit command omits half the task's own edits.** It commits
> `undistort.py` and `test_undistort.py` only, while Step 3b edits
> `collab_splats/preproc/__init__.py` and the pre-flight's own first table row makes
> `tests/preproc/test_sampling.py` Task 15's ("must edit in the same commit"). Run verbatim, it
> leaves the export rewire and the `__all__` set uncommitted — and the `__all__` set is the one
> thing three tasks collide on.
>
> ### Three stale line numbers in the pre-flight table
>
> Measured against `24d11cbb`, the tip the pre-flight says it was measured on:
> `collab_splats/preproc/video.py:273` → **`:263`**; `tests/preproc/test_video.py:10,284` →
> **`:14`** (import) and **`:304,307`** (construction and the `profile=` kwarg).
> `collab_splats/preproc/__init__.py:22,23,29,33` and `tests/preproc/test_sampling.py:426-448`
> are correct.
>
> ### `tests/preproc/test_video.py` had to be repaired here, not in Task 17
>
> The pre-flight routes it to Task 17 — but Task 17 is a separate commit, and between the two
> `tests/preproc` collects **zero tests**, which destroys Task 15's *and* Task 16's only gate.
> `3a4a7828` swaps its one `DistortionProfile` construction for the equivalent
> `pycolmap.Camera(model="OPENCV", width=320, height=240, params=[300.0, 300.0, 160.0, 120.0,
> -0.2, 0.0, 0.0, 0.0])`. `decode_context` forwards `profile` straight to `undistort_frames`
> (`video.py:292`), so the camera flows through unchanged and the test passes. **Task 17 still
> owns the `profile:` → `camera:` rename on that call, and `video.py:263`'s docstring.**
>
> ### Debt handed to Task 17, with the measurement that will bite it
>
> Two tests in Task 15's *own* Files-listed file were readers of `to_dict`/`from_dict` and had to
> be deleted; neither is named anywhere in the plan, and **Task 17 owes their replacement**:
> `test_provenance_roundtrip_through_the_manifest` (asserted
> `DistortionProfile.from_dict(manifest["provenance"]["undistort"]["profile"]) == profile` after a
> real `frames.write_frames` round-trip) and `test_extract_frames_dir_undistorts` (the only test
> of the `undistort=True` branch of `extract_frames`).
>
> **`camera.todict()` is not a drop-in provenance payload.** Measured on `pycolmap 4.0.4`:
>
> ```
> todict()["model"] = CameraModelId.OPENCV      # == "OPENCV" -> False
> json.dumps(camera.todict())
>   TypeError: Object of type CameraModelId is not JSON serializable
> ```
>
> `params` comes back as an `ndarray` for the same reason. `frames.write_frames` `json.dumps`
> the provenance dict verbatim, so Task 17 must serialise `model.name` and `params.tolist()`
> itself; `pycolmap.Camera(**todict())` does round-trip in memory, which is what makes this look
> safe until the manifest write. So **two lines of Task 17 are already known-false**:
> `:4192`'s `prov["undistort"]["camera"]["model"] == "OPENCV"` and `:4228`'s `camera.todict()`
> as a `json.dumps`-able payload. `:4246`'s escape hatch — "if `todict()` is absent, use
> `{"model": camera.model.name, ..., "params": list(camera.params)}`" — is the correct code,
> but its condition never fires, because `todict()` **is** present in 4.0.4. Task 17 should
> take that fallback unconditionally.
>
> Two more Task 17 shapes, both the ones Task 15 hit: `:4186`'s
> `monkeypatch.setattr(reconstructor, "calibrate_camera", lambda d, **k: cam)` is a
> signature-blind stub that cannot fail; and Task 17's Files block (`:4165-4167`) names only
> `reconstructor.py` and `tests/wrapper/test_reconstructor.py` while its Step 3b edits
> `collab_splats/preproc/video.py` and the pre-flight assigns it
> `tests/wrapper/test_vda_context.py:22,336` — so its Step 5 `git commit --only` at `:4291`
> would leave the `video.py` signature change uncommitted.
>
> The one signature-blind stub in the deleted tests was
> `monkeypatch.setattr(recon_mod, "estimate_camera_distortion", lambda frames, **kw: profile)` —
> a bare lambda, not a `patch(..., autospec=True)`, so it could not have caught a signature
> change. Its replacement should be spec'd against `calibrate_camera(images_dir, *, max_frames)`,
> whose first argument is now a **directory**, not a frame list.
>
> ### `-k calibrate` is no longer a 20-second command
>
> `:3816` budgets "~20 s" and that was right for the three tests it prescribes (measured
> **3.69 s + 2.58 s + 0.28 s = 6.6 s**). But the preserved real-footage smoke test was renamed
> `estimate_camera_distortion_tutorial_smoke` → `calibrate_camera_tutorial_smoke`, so it now
> matches `-k calibrate` and costs **165.69 s**. `pytest tests/preproc/test_undistort.py -k
> calibrate` measures **`6 passed, 5 deselected in 177.78s`**. Use
> `-k "calibrate and not tutorial"` for the fast loop.

---

### Task 16: `undistort_frames` uses `pycolmap.undistort_camera`

**Files:**
- Modify: `collab_splats/preproc/undistort.py` (`undistort_frames`)
- Test: `tests/preproc/test_undistort.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/preproc/test_undistort.py`:

```python
def _distorted_camera(width=1920, height=1080):
    """
    A barrel-distorted OPENCV camera, k1=-0.25.
    """
    import pycolmap

    return pycolmap.Camera(
        model="OPENCV",
        width=width,
        height=height,
        params=[1190.4, 1190.4, width / 2, height / 2, -0.25, 0.05, 0.0, 0.0],
    )


def test_undistort_frames_keeps_the_focal_and_grows_the_canvas():
    """
    COLMAP's framing: focal is preserved, the canvas expands to hold the corners.
    """
    import numpy as np

    from collab_splats.preproc.undistort import undistort_frames

    cam = _distorted_camera()
    frames_in = np.zeros((2, 1080, 1920, 3), np.uint8)

    out, new_cam = undistort_frames(frames_in, cam)

    assert new_cam.model.name == "PINHOLE"
    assert new_cam.focal_length_x == cam.focal_length_x
    assert (new_cam.width, new_cam.height) > (cam.width, cam.height)
    assert out.shape == (2, new_cam.height, new_cam.width, 3)


def test_undistort_frames_straightens_a_line():
    """
    A row of dots bowed by barrel distortion comes back collinear.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc.undistort import undistort_frames

    cam = _distorted_camera(width=640, height=480)

    # Project a straight world line THROUGH the distortion, so undistorting must straighten it
    xs = np.linspace(-0.35, 0.35, 9)
    img = np.zeros((480, 640, 3), np.uint8)
    for x in xs:
        u, v = cam.img_from_cam(np.array([[x, -0.2, 1.0]]))[0]
        cv2.circle(img, (int(round(u)), int(round(v))), 3, (255, 255, 255), -1)

    out, _ = undistort_frames(img[None], cam)

    # Centroid of each blob in the output; a straight line has near-zero y spread
    gray = cv2.cvtColor(out[0], cv2.COLOR_RGB2GRAY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats((gray > 128).astype(np.uint8))
    ys = sorted(c[1] for c in centroids[1:])

    assert n - 1 >= 7, "lost blobs — the warp is dropping content"
    assert max(ys) - min(ys) < 2.0, f"line still bowed: y spread {max(ys) - min(ys):.2f} px"


def test_undistort_frames_returns_a_pinhole_camera_with_no_distortion():
    """
    The returned camera has no distortion params left to apply twice.
    """
    from collab_splats.preproc.undistort import undistort_frames

    import numpy as np

    _, new_cam = undistort_frames(np.zeros((1, 1080, 1920, 3), np.uint8), _distorted_camera())

    assert list(new_cam.params[4:]) == []
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -k undistort_frames -v
```

Expected: FAIL — the current `undistort_frames` takes a `DistortionProfile` and returns `(frames, K, roi)`.

- [ ] **Step 3: Rewrite `undistort_frames`**

Replace the whole function with:

```python
def undistort_frames(
    frames_in: np.ndarray, camera: pycolmap.Camera
) -> tuple[np.ndarray, pycolmap.Camera]:
    """
    Undistort a stack of frames onto COLMAP's undistorted framing.

    Args:
        frames_in: (N, H, W, 3) uint8; H, W must match the camera.
        camera: a distorted pycolmap.Camera from calibrate_camera.

    Returns:
        ((N, H', W', 3) uint8, PINHOLE camera). The focal length is preserved and
        the canvas grows to hold the corners — the centre stays 1:1, nothing is
        resampled down to fit the original frame size.
    """
    frames_in = np.asarray(frames_in)
    if frames_in.ndim != 4 or frames_in.shape[1:3] != (camera.height, camera.width):
        raise ValueError(
            f"undistort_frames: frames are {frames_in.shape[1:3]}, camera is "
            f"{(camera.height, camera.width)}"
        )

    # COLMAP picks the framing: focal fixed, canvas sized to hold the corners
    new_cam = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), camera)

    # cv2 moves the pixels: one dst->src map, built once, reused for every frame
    map1, map2 = cv2.initUndistortRectifyMap(
        camera.calibration_matrix(),
        np.asarray(camera.params[4:], dtype=np.float64),
        None,
        new_cam.calibration_matrix(),
        (new_cam.width, new_cam.height),
        cv2.CV_32FC1,
    )
    out = np.stack([cv2.remap(f, map1, map2, cv2.INTER_LINEAR) for f in frames_in])

    logger.info(
        "undistorted %d frames: %dx%d -> %dx%d, f=%.1f preserved",
        len(out),
        camera.width,
        camera.height,
        new_cam.width,
        new_cam.height,
        new_cam.focal_length_x,
    )
    return out, new_cam
```

Deleted with it: `getOptimalNewCameraMatrix`, the even-ROI rounding (`w -= w % 2`), the crop, the `K_out[0,2] -= x` principal-point shift, and the `roi` return value. The principal-point shift is the classic silent bug in this pattern — a crop that moves the optical centre without updating `cx, cy` produces poses that are subtly wrong everywhere. It cannot happen now because there is no crop.

- [ ] **Step 4: Run the tests**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_undistort.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
isort collab_splats/preproc/undistort.py tests/preproc/test_undistort.py
git commit --only collab_splats/preproc/undistort.py tests/preproc/test_undistort.py \
  -m "refactor(preproc)!: undistort_camera picks the framing, cv2.remap moves the pixels

BREAKING: undistort_frames takes a pycolmap.Camera and returns (frames, camera);
the roi return is gone. Output is now larger than the input at native focal
length, where it used to be input-sized at ~0.81x focal."
```

---

### Task 17: the reconstructor's undistort call site and provenance

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_apply_undistortion`)
- Test: `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/wrapper/test_reconstructor.py`:

```python
def test_undistort_provenance_records_the_camera_not_a_profile(tmp_path, monkeypatch):
    """
    Provenance carries pycolmap's camera dicts, and no roi.
    """
    import numpy as np
    import pycolmap

    from collab_splats.wrapper import reconstructor

    cam = pycolmap.Camera(
        model="OPENCV", width=64, height=48, params=[60.0, 60.0, 32.0, 24.0, -0.2, 0.0, 0.0, 0.0]
    )
    monkeypatch.setattr(reconstructor, "calibrate_camera", lambda d, **k: cam)

    prov = {}
    out = reconstructor._apply_undistortion(np.zeros((3, 48, 64, 3), np.uint8), tmp_path, prov)

    assert "roi" not in prov["undistort"]
    assert prov["undistort"]["camera"]["model"] == "OPENCV"
    assert prov["undistort"]["undistorted_camera"]["model"] == "PINHOLE"
    assert out.shape[1:3] == (
        prov["undistort"]["undistorted_camera"]["height"],
        prov["undistort"]["undistorted_camera"]["width"],
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py::test_undistort_provenance_records_the_camera_not_a_profile -v
```

Expected: FAIL — `_apply_undistortion` still calls `estimate_camera_distortion` and stamps `profile`/`K_new`/`roi`.

- [ ] **Step 3: Rewrite `_apply_undistortion`**

```python
def _apply_undistortion(frame_arrays: np.ndarray, images_dir: Path, prov: dict) -> np.ndarray:
    """
    Calibrate from the extracted frames and undistort them in place.

    Args:
        frame_arrays: (N, H, W, 3) uint8 RGB, as selected.
        images_dir: where the frames were written — calibration reads them from here.
        prov: provenance dict, stamped with both cameras.

    Returns:
        (N, H', W', 3) uint8 RGB on the undistorted framing.
    """
    camera = calibrate_camera(images_dir)
    undistorted, new_camera = undistort_frames(frame_arrays, camera)

    # Both cameras, as COLMAP writes them — no local mirror of the same numbers
    prov["undistort"] = {
        "camera": camera.todict(),
        "undistorted_camera": new_camera.todict(),
    }
    return undistorted
```

Update the import line to `from collab_splats.preproc.undistort import calibrate_camera, undistort_frames`, and update the call site: `_apply_undistortion` now runs **after** the frames are written (calibration reads `images_dir`), so the sequence in `extract_frames` becomes write, calibrate-and-undistort, rewrite:

```python
    # Write once so calibration has images to read, then rewrite the undistorted stack
    frames.write_frames(images_dir, frame_arrays, records, prov)
    if undistort:
        frame_arrays = _apply_undistortion(frame_arrays, images_dir, prov)
        frames.write_frames(images_dir, frame_arrays, records, prov)
```

The double write is deliberate and cheap relative to SIFT + mapping. It is also the only ordering that lets `calibrate_camera` take a directory rather than an array.

If `pycolmap.Camera.todict()` is absent in 4.0.4, use `{"model": camera.model.name, "width": camera.width, "height": camera.height, "params": list(camera.params)}` — check first:

```bash
/opt/venv/reconstruction/bin/python -c "import pycolmap; print(hasattr(pycolmap.Camera, 'todict'))"
```

- [ ] **Step 3b: Repoint the VDA context decode at the camera**

`DistortionProfile` is gone, so `decode_context` and its one caller change with it. In `collab_splats/preproc/video.py`, `decode_context`'s `profile: DistortionProfile | None = None` becomes `camera: pycolmap.Camera | None = None`, and the undistort call inside its chunk loop becomes:

```python
            if camera is not None:
                chunk, _ = undistort_frames(chunk, camera)
```

The lazy import at the top of that function becomes `from collab_splats.preproc.undistort import undistort_frames`.

In `collab_splats/wrapper/reconstructor.py`, `_ensure_vda_depth` reads the camera back out of provenance instead of rebuilding a profile:

```python
                        # Decode with the same camera images/ was written with, or the
                        # context frames and the keyframes disagree on the framing
                        camera = None
                        if provenance.get("undistort"):
                            camera = pycolmap.Camera(**provenance["undistort"]["camera"])

                        # (the existing logger.info about the context stream stays here)
                        context_frames = decode_context(video_path, grid, camera=camera)
```

with `import pycolmap` at the top and `DistortionProfile` dropped from the import line. Note it reads `["camera"]` — the DISTORTED camera, because `decode_context` undistorts raw decoded frames, exactly as it did with the distorted profile before.

- [ ] **Step 4: Run the test and the wrapper suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
isort collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/test_reconstructor.py \
  -m "refactor(wrapper)!: undistort provenance carries both pycolmap cameras

BREAKING: prov['undistort'] loses 'profile', 'K_new' and 'roi'."
```

---

> ## Task 18 pre-flight — nine positional call sites and three mocks that cannot fail
>
> Measured on the integration tip, 2026-09-05. Task 18's Files block is two entries
> (`collab_splats/wrapper/reconstructor.py`, `tests/wrapper/test_reconstructor_preprocess.py`).
> `extract_frames` has **fourteen** call sites across four test modules. The signature today
> is six required positionals followed by four keyword-defaulted params:
>
> ```python
> def extract_frames(
>     input_path, images_dir, frame_selection, fps, min_frames, max_frames,
>     n_workers=1, undistort=False, search_radius=3, vda_context_fps=None,
> ) -> int:
> ```
>
> | Module | Call sites | How they bind | Owner |
> |---|---|---|---|
> | `tests/wrapper/test_reconstructor.py` | `:184 :196 :206 :218 :251 :1584 :1596` | **all six args positional** | **Task 18** — unlisted |
> | `tests/wrapper/test_reconstructor.py` | `:363 :374 :387` | `patch("...reconstructor.extract_frames")` | **Task 18** — unlisted, and see below |
> | `tests/wrapper/test_vda_context.py` | `:232 :248 :273` (+ import `:27`) | keyword | **Task 18** — unlisted |
> | `tests/preproc/test_undistort.py` | `:190 :214` | keyword | **Task 18** — unlisted |
> | `tests/wrapper/test_reconstructor_preprocess.py` | `:151` | six positional | Task 18 — listed |
> | `tests/evals/test_datasets.py:497,522` | prose in a docstring/comment | **correct as written** |
>
> **Positional binding is the hazard, and it is silent.** Seven calls in
> `test_reconstructor.py` look like `R.extract_frames(video, out / "a" / "images", "fps", 2.0, 5, 50)`.
> Reorder or drop any of the first six parameters and those calls do not raise — they bind
> the wrong value to the wrong name and the test asserts against a scenario nobody wrote.
> This is the failure mode the plan already recorded as *"property reads and positional
> forwarding are invisible to grep, pyflakes, and a green suite"*; here it is nine lines of
> it in one file the task never opens.
>
> **The three `patch(...)` mocks are signature-blind.** `patch("collab_splats.wrapper.reconstructor.extract_frames")`
> installs a `MagicMock`, which accepts **any** arguments. Those three tests stay green
> through every possible rewrite of the signature while asserting nothing about it — the
> stub-as-reader hazard from the Phase D section, in its most permissive form. If Task 18's
> new shape matters to those call sites, one of them needs `autospec=True` or the split is
> untested where it is used.
>
> **Task 13 and Task 18 collide in `tests/wrapper/test_vda_context.py`.** Two of the three
> `extract_frames` calls there pass `search_radius=7` (`:255`, `:280`) — Task 13 must delete
> that argument and Task 18 must reshape the call around it, and **neither task's Files list
> names the module**. Whichever lands second re-reads the file; do not reconstruct the calls
> from the other task's diff.
>
> **The good news, measured:** every call outside `test_reconstructor.py` and
> `test_reconstructor_preprocess.py` binds by keyword, and the six positional args map
> exactly onto the six required parameters. So **Task 13 alone is safe on positional
> binding** — it removes `search_radius`, the ninth parameter, which no caller passes
> positionally. The positional exposure is created by Task 18's split, not inherited.

### Task 18: split `extract_frames` and drop the `method` alias

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`extract_frames`)
- Test: `tests/wrapper/test_reconstructor_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
def test_extract_frames_has_no_method_alias():
    """
    `method = frame_selection` was a no-op rename inside the function.
    """
    import inspect

    from collab_splats.wrapper import reconstructor

    src = inspect.getsource(reconstructor.extract_frames)
    assert "method = frame_selection" not in src
    assert hasattr(reconstructor, "_frames_from_dir")
    assert hasattr(reconstructor, "_frames_from_video")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor_preprocess.py::test_extract_frames_has_no_method_alias -v
```

Expected: FAIL, `AssertionError` on the alias.

- [ ] **Step 3: Split the function**

Lift the two branches out of `extract_frames` into module-level helpers with the same bodies:

```python
def _frames_from_dir(input_path: Path) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Every image in a directory, in filename order.

    Args:
        input_path: directory of images.

    Returns:
        (frames, records, provenance).
    """
    # Body is the directory branch of extract_frames, moved unchanged: the
    # frames.frame_paths listing, the empty-directory ValueError, the imread/cvtColor
    # comprehension, the records built from enumerate, and the {"method": "dir", ...}
    # provenance dict.


def _frames_from_video(
    input_path: Path,
    *,
    frame_selection: str,
    fps: float | None,
    min_frames: int | None,
    max_frames: int | None,
    report: dict,
    n_workers: int,
    vda_context_fps: float | None,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """
    Frames selected from a video by one of the three sampling methods.

    Args:
        input_path: source video.
        frame_selection: "fps" | "uniform" | "optical_flow".
        fps: target rate for frame_selection="fps".
        min_frames: floor for the fps re-spread band.
        max_frames: cap; the contract for frame_selection="uniform".
        report: quality report from qa.compute_video_quality.
        n_workers: worker count for the quality pass.
        vda_context_fps: rate for the VDA context grid every keyframe must be a member of.

    Returns:
        (frames, records, provenance).
    """
    # Body is the video branch of extract_frames, moved unchanged: the report load or
    # compute, the vda_context_fps grid, the three-way frame_selection dispatch to
    # sample_fps / sample_uniform / sample_optical_flow, and the provenance dict.
```

`extract_frames` then reduces to: pick the branch, call it, write, optionally undistort and rewrite, plot. Delete the `method = frame_selection` line and use `frame_selection` throughout.

- [ ] **Step 4: Run the wrapper suite and the smoke gate**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: tests PASS; smoke exits 0.

- [ ] **Step 5: Commit**

```bash
black collab_splats/wrapper/reconstructor.py tests/wrapper/
isort collab_splats/wrapper/reconstructor.py tests/wrapper/
git commit --only collab_splats/wrapper/reconstructor.py tests/wrapper/ \
  -m "refactor(wrapper): split extract_frames into _frames_from_dir/_frames_from_video"
```

---

## Phase D — PyAV replaces the ffmpeg subprocess

Measured on GH010229 (13,115 frames): full linear scan **76.2 s** against the ffmpeg pipe's **89.2 s** (1.17×), scattered index reads 2.19×, single-frame seek 10×, output pixel-identical (maxdiff 0). PyAV is never slower than what it replaces. The one strategy *inside* PyAV that is slower — seek-per-index over a scattered list, 97.4 s, because every seek lands on a keyframe and re-decodes forward — is not used; scattered reads go through the linear scan.

**CORRECTED 2026-09-05 after Task 20 measured it — the original claim below was false.**
~~`ffprobe` stays, for rotation only. PyAV 17 cannot read a container display matrix.~~
It can. On `av 17.0.1` a **decoded frame** carries `Type.DISPLAYMATRIX`, and `av.VideoFrame.rotation`
returns the angle directly (`rotated_90.mp4` → `-90`; a `rotate=270` remux → `+90`). Only the
*stream-level* accessor is missing, which is what the original probe looked at — `av.sidedata`
exposing `['encparams', 'motionvectors', 'sidedata']` is a fact about the module, not about the
library's capability. **ffprobe is therefore gone from `video.py` entirely as of Task 20**, and no
task in Phase D may reintroduce it. The prescribed ffprobe fallback was also un-runnable on this
box (`No match for section 'stream_side_data'` on ffprobe 4.4.2), so following the original text
would have returned 0 rotation for every video. Measured speed is **1804 ms → 113 ms (~16×)**, not
3.5 s → 110 ms; ffprobe costs 1771 ms even with `count_frames=False`, so the container open
dominates and `-count_packets` was never "the whole cost". Full detail: the Task 20 outcomes block.

---

### Task 19: **HARD GATE** — synthesize a rotated-video fixture

> **LANDED** as `a5c36112`. Three things in this task were written against a
> newer ffmpeg than this box has (4.4.2-0ubuntu0.22.04.1), and the committed
> generator differs from the listing below:
>
> 1. `-display_rotation` is ffmpeg 6+. It does not exist here, so the fallback
>    this task names is unusable.
> 2. A single pass that both transcodes (`-c:v libx264`) and sets
>    `-metadata:s:v:0 rotate=90` silently drops the tag — the output carries no
>    `tags.rotate` and no display matrix. The generator therefore encodes a
>    landscape clip to a temporary file, then **remuxes** with `-c copy
>    -metadata:s:v:0 rotate=...`; the rotation lands on the remux.
> 3. ffmpeg 4.4 **inverts** the rotate tag across a mux/demux round trip.
>    Writing `rotate=90` reads back as `tags.rotate: "270"`. The generator
>    writes `MUX_ROTATE = 270` so that `rotated_90.mp4` reports the 90 degrees
>    its name claims. Both metadata sources agree with each other either way,
>    so `_rotation_degrees` was never inconsistent — only the name/value
>    pairing needed fixing.
>
> Also: this task's premise, "no fixture in the repo carries a rotation", is
> wrong. `tests/preproc/test_video.py:138` already builds one at test time. The
> committed fixture still earns its place — fixed bytes, no ffmpeg needed to
> produce it — but Phase D was not entirely unguarded.

**Files:**
- Create: `tests/preproc/data/make_rotated_fixture.py`, `tests/preproc/data/rotated_90.mp4`

No fixture in the repo carries a rotation. The tutorial video is natively portrait (1080×1920) with `tags.rotate: None` and empty `side_data_list`, so it exercises nothing. Phase D changes the code path that reads rotation. Without this fixture that change is untested, and a rotated GoPro clip would silently decode transposed.

- [ ] **Step 1: Write the generator**

```python
"""
Generate tests/preproc/data/rotated_90.mp4 — a landscape clip carrying a 90-degree
display matrix, so its display dimensions are portrait.

Run once, commit the .mp4. Regenerate with:
    /opt/venv/reconstruction/bin/python tests/preproc/data/make_rotated_fixture.py
"""

import subprocess
from pathlib import Path

OUT = Path(__file__).parent / "rotated_90.mp4"


def main() -> None:
    # A 320x180 landscape clip with a moving bar, so a transposed decode is obvious
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "testsrc=size=320x180:rate=10:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-metadata:s:v:0", "rotate=90",
            str(OUT),
        ],
        check=True,
        capture_output=True,
    )

    # Verify the display matrix actually landed — -metadata rotate= is silently
    # ignored by some muxers, in which case the fixture tests nothing
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
         "-show_streams", str(OUT)],
        capture_output=True, text=True, check=True,
    )
    if "Display Matrix" not in probe.stdout and '"rotate"' not in probe.stdout:
        raise SystemExit(
            f"{OUT} carries no rotation metadata — the muxer dropped it. Try "
            f"`-display_rotation 90` on the input instead."
        )

    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
/opt/venv/reconstruction/bin/python tests/preproc/data/make_rotated_fixture.py
```

Expected: `wrote .../rotated_90.mp4 (<some bytes>)`. If it exits with the muxer complaint, retry with `-display_rotation 90` before `-i` (ffmpeg 6+), which writes a real display matrix rather than a tag.

- [ ] **Step 3: Write the fixture's own test**

Append to `tests/preproc/test_video.py`:

```python
ROTATED = Path(__file__).parent / "data" / "rotated_90.mp4"


def test_rotated_fixture_reports_display_dimensions():
    """
    The fixture is 320x180 stored, 180x320 displayed. get_video_info reports displayed.
    """
    from collab_splats.preproc.video import get_video_info

    info = get_video_info(ROTATED)

    assert (info["width"], info["height"]) == (180, 320)


def test_rotated_fixture_decodes_upright():
    """
    ffmpeg auto-rotates, and whatever replaces it must too: decoded frames match
    the reported display dimensions.
    """
    from collab_splats.preproc.video import get_video_info, iter_frames

    info = get_video_info(ROTATED)
    _, frame = next(iter(iter_frames(ROTATED)))

    assert frame.shape[:2] == (info["height"], info["width"]) == (320, 180)
```

- [ ] **Step 4: Run them against the CURRENT code**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k rotated -v
```

Expected: **PASS**. This is the point of the gate — the tests must pass on today's ffmpeg implementation, so that when they fail after Task 20 or 21, the failure is unambiguously the new code.

If they fail here, the fixture is wrong, not the code. Fix the fixture.

- [ ] **Step 5: Commit and report to the user**

```bash
git add tests/preproc/data/make_rotated_fixture.py tests/preproc/data/rotated_90.mp4
git commit --only tests/preproc/data/make_rotated_fixture.py tests/preproc/data/rotated_90.mp4 tests/preproc/test_video.py \
  -m "test(preproc): rotated-video fixture, green against the ffmpeg implementation"
```

**Tell the user the fixture exists and passes on current code before starting Task 20.** Phase D has no other guard against a transposed decode.

---

### Task 20: `get_video_info` reads metadata through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py:56-100` (`get_video_info`), `_rotation_degrees`
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_get_video_info_frame_count_is_exact(tiny_video):
    """
    PyAV reads stream.frames from the container — no full demux, no packet count.
    """
    from collab_splats.preproc.video import get_video_info

    info = get_video_info(tiny_video)

    assert info["total_frames"] > 0
    assert info["fps"] > 0
    assert abs(info["duration_s"] - info["total_frames"] / info["fps"]) < 0.05


def test_get_video_info_has_no_count_frames_flag():
    """
    The -count_packets full demux is gone; there is no cheap/expensive split left.
    """
    import inspect

    from collab_splats.preproc.video import get_video_info

    assert "count_frames" not in inspect.signature(get_video_info).parameters


def test_get_video_info_returns_zeros_for_a_non_video(tmp_path):
    """
    An unprobeable file returns the zeros dict rather than raising.
    """
    from collab_splats.preproc.video import get_video_info

    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nope")

    assert get_video_info(bad)["total_frames"] == 0
```

Delete any existing test that passes `count_frames=`.

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k get_video_info -v
```

Expected: the `count_frames` test FAILS; the others may pass already.

- [ ] **Step 3: Rewrite the probe**

Replace `get_video_info` with:

```python
def get_video_info(video_path: str | Path) -> dict:
    """
    Video metadata: PyAV for everything except rotation.

    Args:
        video_path: source video.

    Returns:
        {total_frames, fps, duration_s, width, height}, all zeros if unprobeable.
        Width/height are DISPLAY dims (rotation applied), matching what decode yields.
    """
    zeros = {"total_frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}

    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]

            fps = float(stream.average_rate) if stream.average_rate else 0.0
            width, height = stream.codec_context.width, stream.codec_context.height

            # stream.frames is the container's own count; fall back to duration x rate
            total = int(stream.frames)
            if total == 0 and stream.duration and stream.time_base:
                total = int(round(float(stream.duration * stream.time_base) * fps))
    except Exception:
        logger.debug("PyAV could not open %s", video_path, exc_info=True)
        return zeros

    # PyAV 17 cannot see the container display matrix, so rotation alone still
    # costs one ffprobe. Everything else above came from a 110 ms container parse.
    if _rotation_degrees(video_path) in (90, 270):
        width, height = height, width

    duration_s = total / fps if fps > 0 else 0.0
    return {"total_frames": total, "fps": fps, "duration_s": duration_s, "width": width, "height": height}
```

and rewrite `_rotation_degrees` to take a path and do its own probe:

```python
def _rotation_degrees(video_path: str | Path) -> int:
    """
    CW display rotation, via ffprobe.

    Args:
        video_path: source video.

    Returns:
        0, 90, 180 or 270. Two metadata locations: legacy tags.rotate (CW) and
        Display Matrix side data (modern GoPro/iPhone; ffprobe reports CCW).

    PyAV 17 exposes neither, which is the only reason ffprobe survives in this module.
    """
    if shutil.which("ffprobe") is None:
        logger.debug("ffprobe missing; assuming no rotation for %s", video_path)
        return 0

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate:stream_side_data=rotation", str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        logger.debug("ffprobe rotation probe failed for %s", video_path, exc_info=True)
        return 0

    for s in streams:
        rotate = s.get("tags", {}).get("rotate")
        if rotate:
            return int(rotate) % 360

        for sd in s.get("side_data_list", []):
            if sd.get("rotation") is not None:
                return int(-sd["rotation"]) % 360

    return 0
```

Add `import av` at the top, delete `_require_ffmpeg` (nothing needs the ffmpeg *binary* any more; the rotation probe checks `shutil.which("ffprobe")` itself), and update the module docstring — it currently opens "the only module that shells out to ffmpeg/ffprobe", which is still true but for a different reason. Say so:

```python
"""
Video decode and probe.

PyAV decodes and reads metadata. ffprobe survives for exactly one field —
container display rotation, which PyAV 17 does not expose — and is skipped
entirely when it is not on PATH.

Colour convention: iter_frames yields BGR (what cv2 wants), extract_frame
returns RGB (what its consumers store). Both are uint8 HWC.
"""
```

- [ ] **Step 4: Fix the `count_frames=` callers**

```bash
git grep -n 'count_frames'
```

Every hit is a caller passing `count_frames=False` for the cheap path. Delete the kwarg — the cheap path is now the only path.

- [ ] **Step 5: Run the video suite, rotated fixture included**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -v
```

Expected: PASS, **including `test_rotated_fixture_reports_display_dimensions`**. If that one fails, the rotation probe is broken — do not proceed.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc)!: get_video_info reads metadata via PyAV (3.5s -> 110ms)

BREAKING: count_frames= is gone; the count is exact and free."
```

---

> ## Task 20 outcomes — landed `dfcd296d`, merged `d977d397` 2026-09-05
>
> Gates. `tests/preproc/test_video.py` **`30 passed` → `36 passed`**. `tests/preproc`
> `171 passed` → `177 passed` on the task's own base. Re-measured by the coordinator on the
> merged tip `d977d397`: **`181 passed, 0 failed in 682.61s`** — 175 carried in from Tasks 8b/10
> plus the 6 this task adds, which is the arithmetic the number had to satisfy. SHA-pinned, and
> the pin re-checked as an ancestor of the tip *after* the run finished. **`181 passed` is the
> baseline every later Phase D task must reproduce before it starts editing.** `tests/wrapper tests/pointcloud --ignore=.../test_splats_stage.py` held at
> `26 failed, 578 passed, 15 skipped` with the FAILED sets proven identical by `diff` of the
> sorted lists — not by eyeballing the counts. pyflakes went `1` → `0`
>
> **Disputed 2026-09-05 — this wrapper/pointcloud figure is probably 4 failures too high.**
> Task 15's executor re-measured the same scope (`tests/wrapper tests/pointcloud
> --ignore=.../test_splats_stage.py`) on `24d11cbb` and got **`22 failed, 582 passed, 15
> skipped in 442.00s`**. The totals agree exactly — 619 either way — so nothing was added or
> lost; four tests simply flipped side between the two runs. That is the signature of the
> false-RED contention shape, not of a regression, and the higher number is the one measured
> under a live agent wave. Neither reading has been confirmed alone on a quiet box, so
> **do not use either as a pass/fail gate.** The gate that holds regardless is the one this
> block already used: `diff` the sorted FAILED name lists before and after, and require them
> identical. Re-measure the count when the wave is done, scoped and alone.
> (the base had `test_video.py:109: redefinition of unused 'extract_frame'`); black and isort clean.
> Two files, +124/−64.
>
> ### Six things this task's own text got wrong
>
> **(a) `:3715` and `:3974` — "PyAV 17 cannot read a container display matrix" is false.**
> On `av 17.0.1` a *decoded frame* carries `Type.DISPLAYMATRIX` and `av.VideoFrame.rotation`
> returns the angle: `tests/preproc/data/rotated_90.mp4` → `-90`, a `rotate=270` remux → `+90`.
> Only the *stream-level* half of the claim is true, and the plan generalised it to the whole
> library. `_rotation_degrees` now takes the already-open container and reads
> `-frame.rotation % 360`; **ffprobe is gone from the module entirely.** Verified independently
> before merge — `hasattr(av.VideoFrame, "rotation")` is `True` on this box's `av 17.0.1`.
>
> **(b) `:3982` — the prescribed ffprobe fallback is not a runnable command here.**
> `-show_entries "stream_tags=rotate:stream_side_data=rotation"` on this box's ffprobe 4.4.2:
>
> ```
> No match for section 'stream_side_data'
> Failed to set value 'stream_tags=rotate:stream_side_data=rotation' for option 'show_entries': Invalid argument
> ```
>
> Following the plan literally would have returned **0 rotation for every video**, silently, with
> no test able to see it — because (a) means the fallback would also have been the only path.
> Reproduced independently before merge.
>
> **(c) `:4003` — "delete `_require_ffmpeg`" is false *at Task 20*.** `iter_frames`
> (`video.py:127`) and `extract_frame` (`video.py:213`) still shell out; they are Tasks 21 and 22.
> Deleting it here breaks both. It was kept and narrowed from `ffmpeg or ffprobe` to `ffmpeg`
> only. **Task 21 and Task 22 own the deletion, and neither may delete it until both have landed.**
>
> **(d) `:4009` — the prescribed replacement module docstring is false in both halves** once
> Task 20 lands and before 21/22 do. Rewritten to say metadata is a PyAV parse while decode
> still shells out.
>
> **(e) `:3946` — `stream.duration * stream.time_base` is not the robust duration fallback.**
> For `.mkv` `stream.duration is None` while `container.duration` is populated (`2000000` µs)
> for mkv/ts/mp4 alike. Landed as `container.duration / av.time_base * fps`.
>
> **(f) `:4040` — "3.5s -> 110ms" is wrong in its numerator and in its premise.** Measured on
> the 2388-frame tutorial video: ffprobe **1804 ms**, PyAV **113 ms** → **~16×**, not ~32×.
> The premise fails too: ffprobe with `count_frames=False` still costs **1771 ms**, so
> `-count_packets` is *not* "the whole cost" — the container open dominates. The commit message
> carries the measured numbers, not the plan's.
>
> ### `count_frames=` is gone from the public signature
>
> `(video_path, *, count_frames: bool = True) -> dict` → `(video_path) -> dict`. The count is
> exact and free from `stream.frames`, so the flag had one reachable value.
>
> **The returned key set is unchanged** — `['duration_s', 'fps', 'height', 'total_frames',
> 'width']`, verified side by side on three inputs. That is what makes the change safe for the
> six partial-dict stubs that fabricate a `get_video_info` return
> (`tests/wrapper/test_reconstructor.py:163,213,233,1578`, `tests/dashboard/test_app.py:525`,
> `tests/preproc/test_viz.py:159`): they cannot `KeyError`, and the one-arg lambdas cannot
> `TypeError` now that the keyword is gone. `tests/wrapper/test_vda_context.py:230`'s
> probe-absence guard is untouched.
>
> ### Eleven call sites the Files list does not name
>
> All eleven consume the returned dict and none was listed. They survive only because the key set
> held; had Task 20 renamed one key they would all have broken, and only two are in `tests/preproc`:
>
> `preproc/qa.py:354` · `preproc/sampling.py:359,409,475` · `preproc/viz.py:293` ·
> `preproc/video.py:203,224` · `wrapper/reconstructor.py:253` · `dashboard/app.py:390` ·
> `dashboard/localize.py:479` · `tests/pointcloud/test_loger_creator.py:737`
>
> **Rule for Tasks 21/22, which change decode and not just metadata:** a Files list that names
> only the module being edited is not a work surface. Grep the callers of every function whose
> signature or return moves, and record the count before editing.
>
> ### Stale ffprobe prose this task uncovered — routed
>
> | Location | Owner |
> |---|---|
> | `preproc/video.py:192` ("a fresh one costs an ffprobe subprocess") | **Task 12** |
> | `preproc/video.py:220` | **Task 22** |
> | `dashboard/app.py:381` | **Task 28, Step 3b** |
> | `wrapper/reconstructor.py:244` | **Task 28, Step 3b** |
> | `tests/pointcloud/test_loger_creator.py:734` | **Task 28, Step 3b** |
> | `CLAUDE.md:97` — still calls `video.py` "ffmpeg/ffprobe decode" | **Task 28, Step 3** (already covered) |
>
> Routing these to Task 28 was initially wrong for three of the four: Step 3 rewrites the
> `CLAUDE.md` architecture tree and so already covered `CLAUDE.md:97`, but Task 28's Files list
> named only `CLAUDE.md` and `CHANGELOG.md` — the three code comments had **no owner at all**.
> **Step 3b was added to Task 28 to take them**, and its exit condition is `grep -rn ffprobe`
> returning zero hits repo-wide. Routing a finding to a task is not the same as that task's text
> covering it; check the Files list, not the task's title.
>
> `CLAUDE.md:97` is the one that matters: it is loaded into every session in this repo, so it
> keeps re-teaching the wrong architecture until Task 28 fixes it.
>
> ### The full-suite baseline, and two flakes that are not regressions
>
> The first non-stale full-`tests/` run landed here: **`25 failed, 754 passed, 15 skipped in
> 3714.10s (1:01:54)`**, started ~10:54 — after the T8b/T10 merges at 10:48 — and its FAILED list
> names the post-rename `..._since_extraction`, which is the tell that separates it from the two
> worthless 55-minute runs (see "The fifth shape").
>
> Two of its 25 are **`tests/preproc`**, which no scoped gate had ever shown:
> `test_sampling.py::test_fps_uses_constant_stride` and
> `test_sampling.py::test_sample_uniform_decodes_in_one_select_pass`. Both are **contention
> flakes, not regressions**, on three independent measurements at the same code: the two in
> isolation `2 passed`; the whole file `52 passed`; and the whole directory `175 passed` in a
> separate SHA-pinned run. Only docs commits and Task 20 (`video.py` alone) landed after that run
> started, so nothing could have "fixed" a sampling test in between. The mechanism is the decode
> hazard `CLAUDE.md` already names — input seek (`-ss` before `-i`) landing off-by-N — surfacing
> under 30+ concurrent pytest processes on the MooseFS mount.
>
> The wall clock corroborates this independently of any pass/fail result: the same
> `tests/preproc` directory took **2278.27s during the wave and 682.61s after it drained —
> 3.3×**, on trees differing by one module. Timing that far apart on identical work is the
> contention, measured rather than assumed.
>
> **Do not chase these two, and do not add them to `docs/known-test-failures.md`.** A scoped
> `tests/preproc` gate run alone is the authority; a full-suite run under a live wave is not.

---

### Task 21: `iter_frames` decodes through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py` (`iter_frames`)
- Test: `tests/preproc/test_video.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_iter_frames_yields_bgr(tiny_video):
    """
    Colour convention is unchanged: iter_frames is BGR, extract_frame is RGB.
    """
    import numpy as np

    from collab_splats.preproc.video import extract_frame, iter_frames

    _, bgr = next(iter(iter_frames(tiny_video)))
    rgb = extract_frame(tiny_video, 0)

    assert np.array_equal(bgr[..., ::-1], rgb) or np.abs(
        bgr[..., ::-1].astype(int) - rgb.astype(int)
    ).max() <= 2


def test_iter_frames_indices_yields_exactly_those_indices(tiny_video):
    """
    Scattered reads come back in ascending order, once each.
    """
    from collab_splats.preproc.video import iter_frames

    got = [i for i, _ in iter_frames(tiny_video, indices=[7, 2, 2, 5])]

    assert got == [2, 5, 7]


def test_iter_frames_start_count_window(tiny_video):
    """
    start/count is a contiguous window in source frame indices.
    """
    from collab_splats.preproc.video import iter_frames

    got = [i for i, _ in iter_frames(tiny_video, start=4, count=3)]

    assert got == [4, 5, 6]


def test_iter_frames_stops_early_without_decoding_the_tail(tiny_video):
    """
    The last requested index ends the scan; the rest of the file is not decoded.
    """
    from collab_splats.preproc.video import get_video_info, iter_frames

    total = get_video_info(tiny_video)["total_frames"]
    assert total > 20

    got = [i for i, _ in iter_frames(tiny_video, indices=[1, 3])]
    assert got == [1, 3]
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k iter_frames -v
```

Expected: PASS on the current implementation — these are the behaviours PyAV must preserve, written down before the rewrite so the rewrite has something to violate. Add `test_iter_frames_uses_no_subprocess`:

```python
def test_iter_frames_uses_no_subprocess(tiny_video, monkeypatch):
    """
    Decode is in-process; no ffmpeg pipe.
    """
    import subprocess

    import pytest

    from collab_splats.preproc.video import iter_frames

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawned an ffmpeg pipe"))

    assert len(list(iter_frames(tiny_video, indices=[0, 2]))) == 2
```

That one FAILS now.

- [ ] **Step 3: Rewrite `iter_frames` as a linear scan**

```python
def iter_frames(
    video_path: str | Path,
    *,
    indices: Sequence[int] | None = None,
    start: int = 0,
    count: int | None = None,
    info: dict | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Yield (frame_idx, BGR uint8 HWC) in one decode pass.

    Args:
        video_path: source video.
        indices: yield only these source frame indices, ascending and deduped.
        start: first index of a contiguous window.
        count: length of that window; None runs to the end.
        info: accepted and ignored — kept so callers need not change.

    Returns:
        An iterator of (source frame index, BGR frame).

    One linear scan, always. Seeking to each requested index is SLOWER on a
    scattered list (97.4 s against 76.2 s on a 13k-frame clip) because every seek
    lands on a keyframe and re-decodes forward from it.
    """
    wanted = sorted({int(i) for i in indices}) if indices is not None else None
    if wanted is not None and not wanted:
        return

    # The scan stops at the last frame anyone asked for
    if wanted is not None:
        last = wanted[-1]
    elif count is not None:
        last = start + count - 1
    else:
        last = None

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]

        # Let PyAV use every core it is allowed; decode is the whole cost here
        stream.thread_type = "AUTO"

        cursor = 0
        pending = iter(wanted) if wanted is not None else None
        target = next(pending, None) if pending is not None else None

        for frame in container.decode(stream):
            if last is not None and cursor > last:
                break

            if wanted is not None:
                if target is None:
                    break
                take = cursor == target
                if take:
                    target = next(pending, None)
            else:
                take = cursor >= start and (count is None or cursor < start + count)

            if take:
                # to_ndarray already applies the container's rotation
                yield cursor, frame.to_ndarray(format="bgr24")

            cursor += 1
```

`info=` is kept and ignored on purpose: it exists in a dozen call sites purely to hoist an ffprobe that no longer happens. Deleting the parameter would be a wider change than this task, and Task 24 sweeps it.

- [ ] **Step 4: Run the video suite and the rotated fixture**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Expected: PASS, **including `test_rotated_fixture_decodes_upright`**. If that fails, `to_ndarray` is not applying the display matrix and the decode needs an explicit rotate — do not paper over it by transposing in the caller.

- [ ] **Step 5: Confirm the speed claim on real footage**

```bash
SCRATCH=/tmp/claude-0/-workspace-collab-splats/ada04594-773b-4b10-b55a-fa414346cead/scratchpad
/opt/venv/reconstruction/bin/python -c "
import time
from collab_splats.preproc.video import iter_frames
t = time.perf_counter()
n = sum(1 for _ in iter_frames('<path to GH010229.MP4>'))
print(f'{n} frames in {time.perf_counter() - t:.1f} s')
"
```

Expected: ~76 s for 13,115 frames. Materially slower means `thread_type = "AUTO"` did not take effect.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc): iter_frames decodes in-process via PyAV (1.17x scan, 2.19x scattered)"
```

---

### Task 22: `extract_frame` seeks through PyAV

**Files:**
- Modify: `collab_splats/preproc/video.py` (`extract_frame`, `_require_ffmpeg`, module docstring)
- Test: `tests/preproc/test_video.py`

> **Task 22 owns the `_require_ffmpeg` deletion — no earlier task does.**
> Task 20's text told it to delete the guard; that was wrong and it was kept (outcomes block,
> error (c)), because `iter_frames` and `extract_frame` both still shelled out. Task 21 removes
> the first caller and **its text never mentions the guard at all**, so after 21 lands the guard
> has exactly one caller left — this task's. Deleting it here is the last step that makes
> `video.py` subprocess-free.
>
> Before deleting, confirm both callers are actually gone:
>
> ```bash
> grep -n "_require_ffmpeg\|subprocess\|\"ffmpeg\"\|ffprobe" collab_splats/preproc/video.py
> ```
>
> Expected after this task: **no matches outside the module docstring.** Then drop the
> `import subprocess` and `import shutil` lines if nothing else uses them (pyflakes will say so),
> and rewrite the module docstring — Task 20 left it saying decode still shells out, which stops
> being true here.

- [ ] **Step 1: Write the failing test**

```python
def test_extract_frame_matches_the_scan(tiny_video):
    """
    Seeking to frame N returns the same pixels the scan yields at N.
    """
    import numpy as np

    from collab_splats.preproc.video import extract_frame, iter_frames

    scanned = dict(iter_frames(tiny_video, indices=[12]))
    seeked = extract_frame(tiny_video, 12)

    assert np.abs(scanned[12][..., ::-1].astype(int) - seeked.astype(int)).max() <= 2


def test_extract_frame_rejects_an_out_of_range_index(tiny_video):
    """
    Past the end is an error, not a silently-clamped last frame.
    """
    import pytest

    from collab_splats.preproc.video import extract_frame, get_video_info

    total = get_video_info(tiny_video)["total_frames"]
    with pytest.raises(ValueError, match="out of range"):
        extract_frame(tiny_video, total + 5)
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_video.py -k extract_frame -v
```

Expected: the match test may pass; the range test's message may differ. Adjust the `match=` to the message you actually write, not the other way round.

- [ ] **Step 3: Rewrite `extract_frame`**

```python
def extract_frame(video_path: str | Path, frame_idx: int, *, info: dict | None = None) -> np.ndarray:
    """
    Decode one frame by seeking to it.

    Args:
        video_path: source video.
        frame_idx: source frame index.
        info: a get_video_info dict, to skip the range check's own probe.

    Returns:
        (H, W, 3) uint8 RGB.

    Seek-then-scan-forward, which is 10x a full scan for ONE frame and the reason
    this is a separate function from iter_frames. It is exact on CFR video and may
    land one frame off near a keyframe on a VFR source.
    """
    info = info if info is not None else get_video_info(video_path)
    total, fps = info["total_frames"], info["fps"]

    if frame_idx < 0 or (total and frame_idx >= total):
        raise ValueError(f"frame {frame_idx} out of range for {video_path} ({total} frames)")
    if not fps:
        raise ValueError(f"cannot seek {video_path}: missing fps")

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek lands on the keyframe at or before the target; decode forward from there
        target_pts = int(frame_idx / fps / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)

        for frame in container.decode(stream):
            if frame.pts is None or frame.pts >= target_pts:
                return frame.to_ndarray(format="rgb24")

    raise ValueError(f"decode of {video_path} ended before reaching frame {frame_idx}")
```

- [ ] **Step 4: Run the video suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats/preproc/video.py tests/preproc/test_video.py
isort collab_splats/preproc/video.py tests/preproc/test_video.py
git commit --only collab_splats/preproc/video.py tests/preproc/test_video.py \
  -m "perf(preproc): extract_frame seeks in-process via PyAV (10x)"
```

---

### Task 23: move `decode_context` to the module that uses it

**Files:**
- Modify: `collab_splats/preproc/video.py` (delete `decode_context`), `collab_splats/pointcloud/sfm.py` (add it), `collab_splats/wrapper/reconstructor.py` (import)
- Test: move the `decode_context` tests from `tests/preproc/test_video.py` to `tests/pointcloud/test_sfm.py`

- [ ] **Step 1: Write the failing test**

```python
def test_decode_context_lives_with_vda():
    """
    It exists to feed generate_vda_depth; it is not general video decode.
    """
    from collab_splats.pointcloud import sfm
    from collab_splats.preproc import video

    assert hasattr(sfm, "decode_context")
    assert not hasattr(video, "decode_context")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_sfm.py::test_decode_context_lives_with_vda -v
```

Expected: FAIL, `AttributeError`.

- [ ] **Step 3: Move it**

Cut `decode_context` (with the `camera=` signature from Task 17's Step 3b) out of `collab_splats/preproc/video.py` and paste it into `collab_splats/pointcloud/sfm.py`, above `generate_vda_depth`. Its imports move with it:

```python
from collab_splats.preproc.undistort import undistort_frames
from collab_splats.preproc.video import get_video_info, iter_frames
```

The lazy import inside the function becomes a top-level one — the cycle it was avoiding (`video` → `undistort`) does not exist from `sfm`.

Update `collab_splats/wrapper/reconstructor.py`:

```python
from collab_splats.pointcloud.sfm import decode_context
from collab_splats.preproc.sampling import context_indices
```

Move the `decode_context` tests from `tests/preproc/test_video.py` into `tests/pointcloud/test_sfm.py`, adjusting only their import line.

- [ ] **Step 4: Run both suites**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/ tests/pointcloud/ tests/wrapper/ -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
black collab_splats tests
isort collab_splats tests
git commit --only collab_splats/preproc/video.py collab_splats/pointcloud/sfm.py collab_splats/wrapper/reconstructor.py tests/preproc tests/pointcloud \
  -m "refactor(preproc)!: decode_context moves to pointcloud.sfm, next to its only caller

BREAKING: preproc.video.decode_context is now pointcloud.sfm.decode_context."
```

---

### Task 24: promote `av` to a direct dependency and sweep the dead `info=`

> **Split into 24a and 24b; 24a has LANDED** as `9d3dcfcf`. `av>=17.0` is
> declared in `pyproject.toml` next to `scipy`. That half was safe to run at
> wave 0 because PyAV already resolved transitively, so declaring it changed
> nothing at install time. What remains here is 24b: the dead `info=` sweep,
> which needs Tasks 21 and 22.

**Files:**
- Modify: `pyproject.toml`, `collab_splats/preproc/video.py`, every `info=` call site

- [ ] **Step 1: Confirm av's provenance**

```bash
/opt/venv/reconstruction/bin/python -c "import av; print(av.__version__)"
grep -n '^ *"av' uv.lock | head
```

Expected: `17.0.1`, arriving transitively. A module we now import directly must be declared directly.

- [ ] **Step 2: Declare it**

In `pyproject.toml`, add to `dependencies` (alphabetical position, before `black` or wherever the list orders it):

```toml
    "av>=17.0",
```

- [ ] **Step 3: Sync and verify**

```bash
uv sync
/opt/venv/reconstruction/bin/python -c "import av; print(av.__version__)"
```

Expected: still `17.0.1`, now pinned by us. **Do not run a bare `uv sync` that prunes extras** — this project's extras carry VGGT-X and MapAnything; use the same invocation `setup.sh` uses if `uv sync` alone drops them.

- [ ] **Step 4: Delete the dead `info=` parameter**

`iter_frames` ignores it, and `extract_frame` uses it only to skip a now-cheap probe.

```bash
git grep -n 'info=info\|info=\(get_video_info\|_info\)'
```

Remove `info=` from `iter_frames` entirely (signature and every call site). Keep it on `extract_frame` — it still saves a probe per call in a loop, and the probe is 110 ms, not free.

- [ ] **Step 5: Run everything**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: green apart from `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 6: Commit**

```bash
black collab_splats tests
isort collab_splats tests
git commit --only pyproject.toml uv.lock collab_splats tests \
  -m "build: declare av as a direct dependency; drop iter_frames' dead info="
```


> ## Coordinator blocker — `uv.lock` does not contain `av`, and it is not being worked around
>
> Measured 2026-09-05 on `preproc-t22`, then independently re-verified by the coordinator:
>
> | source | says |
> |---|---|
> | `pyproject.toml:35` | `"av>=17.0"` |
> | `uv.lock` (1,719,835 B) | `grep -c '^name = "av"'` -> **0**. No near-miss name either. |
> | the venv | `av 17.0.1`, installed by uv |
>
> So the declared dependency is real, the installed package is real, and the lock file knows
> about neither. Task 24's Step 1 assumed `av` would arrive in the lock transitively; it does
> not. Anyone running `uv sync --locked` from a clean checkout gets an environment without
> `av` — and `av` is now load-bearing: Tasks 21 and 22 route `iter_frames` and `extract_frame`
> through PyAV, so `collab_splats.preproc.video` does not import without it.
>
> **Not fixed here, deliberately.** Reconciling requires a resolver run, and the venv at
> `/opt/venv/reconstruction` is shared with three other live sessions (`mesh-texture`,
> `insfm-exec`, `docs-site`). Running `uv sync` to fix a lock file would mutate their
> interpreter underneath them. That trade is not this plan's to make.
>
> **Owed, as a separate piece of work, by someone who can take the venv offline:** re-resolve
> the lock (`uv lock`) and confirm `av>=17.0` lands in it, then `uv sync` and re-run
> `tests/preproc`. Until then Task 24's Step 3 (`uv sync`) and Step 6 (`uv.lock` in the commit
> path) stay skipped, and `uv.lock` is untouched on every branch in this plan.

---

## Phase E — delete the dead, collapse the redundant, fix the docstrings

---

### Task 25: delete `plot_disparity_sensitivity` and `plot_quality_examples`

> **Runs in wave 0, before Task 9.** `plot_quality_examples` is the only `FrameStore`
> consumer in `viz.py`, and no Phase A task converts that file. Deleting it here is
> what lets Task 9's verification grep come back clean.

**Files:**
- Modify: `collab_splats/preproc/viz.py:102-188`, `tests/preproc/test_viz.py`, `docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`, `docs/known-test-failures.md`

`plot_quality_examples` has been broken since the FrameStore rework — the tutorial notebook carries its traceback in committed output, and `docs/known-test-failures.md` lists it. Its "Rejected: soft" split hardcodes `lap < 50.0`, the threshold Task 10 deleted. `plot_disparity_sensitivity` re-thresholds records through a formula that ignores the selector's stateful keyframe updates, so its counts are approximate by its own docstring and nothing consumes them.

- [ ] **Step 1: Write the failing test**

Replace the `plot_disparity_sensitivity` and `plot_quality_examples` tests in `tests/preproc/test_viz.py` with:

```python
def test_dead_plots_are_gone():
    """
    Both were broken or approximate, and nothing outside the notebook called them.
    """
    from collab_splats.preproc import viz

    assert not hasattr(viz, "plot_disparity_sensitivity")
    assert not hasattr(viz, "plot_quality_examples")
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py::test_dead_plots_are_gone -v
```

Expected: FAIL, `AssertionError`.

- [ ] **Step 3: Delete**

Remove both functions from `collab_splats/preproc/viz.py` (lines 102-188) and the imports they alone needed — `OpticalFlowFrameSelector` and `filter_frame_quality`, if nothing else in the module uses them:

```bash
git grep -n 'OpticalFlowFrameSelector\|filter_frame_quality' -- collab_splats/preproc/viz.py
```

- [ ] **Step 4: Fix the notebook**

`docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb` has three call cells (`plot_quality_examples(video_lookup, frame_scores)`, `plot_quality_examples(video_lookup, demo_scores)`, `plot_disparity_sensitivity(frame_scores, disparity_values)`), their imports, the `_VideoFrameLookup` shim that existed only to fake a store for them, the `disparity_values` definition, and the surrounding markdown. Delete all of it with the NotebookEdit tool, then re-run the notebook end to end and commit the executed output.

The notebook keeps `plot_selection` and `plot_frame_extremes`, which work.

- [ ] **Step 5: Drop the known-failure entry**

Remove the `plot_quality_examples` row from `docs/known-test-failures.md`. Deleting the function is the fix.

- [ ] **Step 6: Run and commit**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_viz.py -v
black collab_splats/preproc/viz.py tests/preproc/test_viz.py
isort collab_splats/preproc/viz.py tests/preproc/test_viz.py
git commit --only collab_splats/preproc/viz.py tests/preproc/test_viz.py docs/known-test-failures.md docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb \
  -m "refactor(preproc)!: delete plot_disparity_sensitivity and plot_quality_examples

BREAKING: both leave preproc.viz. plot_quality_examples had been raising since
the frame-store rework; plot_disparity_sensitivity's counts were approximate."
```

---

### Task 26: collapse the qa pair-motion four into one

**Files:**
- Modify: `collab_splats/preproc/qa.py` (`detect_orb`, `match_descriptors`, `compute_translation`, `compute_parallax` → `compute_pair_motion`)
- Test: `tests/preproc/test_qa.py`

The four are called in exactly one place, always in the same order, always on the same two frames. `detect_orb` stays separate — the worker caches its result to detect once per frame rather than twice per pair, a measured 1.34× win. The other three collapse.

- [ ] **Step 1: Write the failing test**

```python
def test_compute_pair_motion_returns_all_three_measures():
    """
    One call per pair, replacing match + translation + parallax.
    """
    import cv2
    import numpy as np

    from collab_splats.preproc.qa import compute_pair_motion, detect_orb

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (300, 400), dtype=np.uint8)
    a, b = canvas[:240, :320], canvas[10:250, 8:328]

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert set(row) == {"n_matches", "translation_px", "parallax"}
    assert row["n_matches"] > 20
    assert 8.0 < row["translation_px"] < 20.0


def test_compute_pair_motion_on_an_unmatchable_pair():
    """
    Two unrelated frames give zero matches and zeroed measures, not an exception.
    """
    import numpy as np

    from collab_splats.preproc.qa import compute_pair_motion, detect_orb

    a = np.zeros((240, 320), np.uint8)
    b = np.full((240, 320), 255, np.uint8)

    row = compute_pair_motion(detect_orb(a), detect_orb(b))

    assert row["n_matches"] == 0
    assert row["translation_px"] == 0.0
    assert row["parallax"] == 0.0


def test_the_three_collapsed_helpers_are_gone():
    """
    match_descriptors/compute_translation/compute_parallax had one caller between them.
    """
    from collab_splats.preproc import qa

    for dead in ("match_descriptors", "compute_translation", "compute_parallax"):
        assert not hasattr(qa, dead), dead
```

- [ ] **Step 2: Run them to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -k pair_motion -v
```

Expected: FAIL, `ImportError: cannot import name 'compute_pair_motion'`.

- [ ] **Step 3: Collapse**

In `collab_splats/preproc/qa.py`, replace `match_descriptors`, `compute_translation` and `compute_parallax` with one function whose body is their three bodies in sequence:

```python
def compute_pair_motion(feat_a: tuple, feat_b: tuple, *, ransac_thresh_px: float = 3.0) -> dict:
    """
    Match two frames' ORB features and measure the motion between them.

    Args:
        feat_a: (keypoints, descriptors) from detect_orb, the earlier frame.
        feat_b: (keypoints, descriptors) from detect_orb, the later frame.
        ransac_thresh_px: homography inlier threshold for the parallax estimate.

    Returns:
        {n_matches, translation_px, parallax}. All zero when the pair does not match.
    """
    # match_descriptors' body: BFMatcher(NORM_HAMMING, crossCheck=True), the empty-descriptor
    # guard, and the two (N, 2) float32 point arrays it returns.
    # then compute_translation's body: median L2 norm of (pts_b - pts_a).
    # then compute_parallax's body: findHomography(RANSAC, ransac_thresh_px) and the median
    # residual of pts_b against the homography-warped pts_a.
```

The three bodies move in unchanged, and their three separate empty-match guards collapse into one at the top:

```python
    if feat_a[1] is None or feat_b[1] is None or len(feat_a[1]) == 0 or len(feat_b[1]) == 0:
        return {"n_matches": 0, "translation_px": 0.0, "parallax": 0.0}
```

This is a call-shape change, not a behaviour change. Keep `detect_orb` exactly as it is.

- [ ] **Step 4: Update the worker call site**

In `collab_splats/preproc/qa.py` (~lines 305-335):

```python
        pending[idx] = detect_orb(gray_small)
        partner = idx - stride
        if partner in pending:
            pair_rows.append(
                {
                    "frame_idx_a": partner,
                    "frame_idx_b": idx,
                    **compute_pair_motion(pending[partner], pending[idx]),
                }
            )
            del pending[partner]
```

- [ ] **Step 5: Run the qa suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_qa.py -v
```

Expected: PASS. Delete the tests of the three collapsed helpers — their behaviour is now covered by `compute_pair_motion`'s.

- [ ] **Step 6: Commit**

```bash
black collab_splats/preproc/qa.py tests/preproc/test_qa.py
isort collab_splats/preproc/qa.py tests/preproc/test_qa.py
git commit --only collab_splats/preproc/qa.py tests/preproc/test_qa.py \
  -m "refactor(preproc)!: collapse pair motion into compute_pair_motion

BREAKING: match_descriptors, compute_translation and compute_parallax leave
preproc.qa. detect_orb stays — the worker caches it per frame."
```

---

> ## Task 27 pre-flight — the lint run, measured on today's tip
>
> Task 27's Step 2 says the lint's failures "are this task's checklist" but does not say how
> long it is. It was run against the integration tip on 2026-09-05, exactly as written, and
> the answer is:
>
> ```
> 11 failed, 13 passed in 19.99s
> ```
>
> `preproc.__all__` is 14 names, of which 12 are functions (`DistortionProfile` and
> `FrameStore` are classes and `inspect.isfunction` drops them), so the run is 24 tests.
> **Eleven of the twelve public functions fail, and they all fail the same assertion** —
> `Args:` is absent:
>
> | Function | Params it documents none of |
> |---|---|
> | `analysis_gray` | `frame_bgr`, `width` |
> | `compute_video_quality` | `video_path`, `output_path`, `motion_stride`, `workers` |
> | `estimate_camera_distortion` | `frames`, `max_frames` |
> | `extract_frame` | `video_path`, `frame_idx`, `info` |
> | `get_video_info` | `video_path`, `count_frames` |
> | `iter_frames` | `video_path`, `indices`, `start`, `count`, `info` |
> | `load_video_quality` | `video_path`, `report_path`, `workers`, `motion_stride` |
> | `sample_fps` | 9 params |
> | `sample_optical_flow` | 10 params |
> | `sample_uniform` | 7 params |
> | `undistort_frames` | `frames`, `profile` |
>
> **The one that passes is `filter_frame_quality`** — the function Task 10 rewrote. That is
> the useful signal in this run: the claim in Task 27's preamble that "Tasks 10-26 already
> wrote the new and rewritten functions in this form" is true so far, and holds for exactly
> as many functions as those tasks have touched. Task 27's size shrinks by one for every
> function an earlier task rewrites, and four of the eleven above
> (`estimate_camera_distortion`, `undistort_frames`, `get_video_info`, `iter_frames`) are
> already owned by Tasks 15, 16, 20 and 21.
>
> **`test_summary_is_one_line` passes 12/12 today.** Every existing summary is already a
> single line under 100 chars that does not restate its own name. That half of the lint
> costs nothing to adopt and is pure regression protection.
>
> **The `Returns:` deficit is unmeasured.** In `test_public_function_documents_its_inputs_and_outputs`
> the `Args:` assertion fires before the `Returns:` one, so eleven functions never reached
> the second check. Expect a **second wave** of failures after the Args blocks land. Do not
> read "11 failures" as "11 docstrings to fix once".
>
> **One hardening the lint needs.** The per-parameter check is a bare substring test:
>
> ```python
> assert f"{p}:" in doc, f"{name} does not document '{p}'"
> ```
>
> `"frames:" in "max_frames: the budget"` is `True`, and so is `"fps:" in "vda_context_fps: ..."`
> and `"report:" in "report_path: ..."`. A parameter whose name is a suffix of another
> documented name is scored as documented without anyone writing a line for it. No signature
> in `preproc` has such a pair *today*, so the lint is sound as written — but Tasks 12, 15
> and 20-24 all reshape these signatures, and `fps`/`vda_context_fps` and
> `report`/`report_path` are both live name pairs in this module. Anchor the match to the
> start of a line instead:
>
> ```python
> assert re.search(rf"^\s*{re.escape(p)}:", doc, re.M), f"{name} does not document '{p}'"
> ```
>
> which also enforces that the parameter is documented *as an entry*, not mentioned in
> passing in the summary.
>
> ### Re-measured on `4bf82f03` (integration tip, after Tasks 9, 11, 21) — supersedes the numbers above
>
> The block above was measured before Task 9 landed. Two of its figures are now wrong and one
> of its predictions is refuted. Re-run with the **hardened** `re.search(rf"^\s*{p}:", ...)`
> assertion, so these counts are already the strict ones:
>
> | figure | pre-flight above | today |
> |---|---|---|
> | `__all__` | 14 | **18** |
> | public functions | 12 | **17** (`DistortionProfile` is the only non-function) |
> | lint tests | 24 | **34** |
> | `Args:` incomplete | 11 / 12 | **10 / 17** |
> | `Returns:` absent | unmeasured | **10 / 17 — the same ten** |
> | summary-line check | 12 / 12 pass | **17 / 17 pass** |
>
> **The predicted "second wave" does not exist.** The pre-flight assumed the `Args:` assertion
> was masking an unknown number of `Returns:` failures. Measured with the two checks run
> independently, the sets are *identical*: every function that documents its Args also has a
> `Returns:` block, and every function missing one is missing both. Task 27 is one pass over ten
> docstrings, not two.
>
> **The ten, and who owns them:**
>
> | function | undocumented params | owner |
> |---|---|---|
> | `analysis_gray` | `frame_bgr`, `width` | Task 27 |
> | `compute_video_quality` | `video_path`, `output_path`, `motion_stride`, `workers` | Task 27 |
> | `extract_frame` | `video_path`, `frame_idx`, `info` | Task 27 (Task 22 reshaped the body, not the doc) |
> | `get_video_info` | `video_path` | Task 27 |
> | `iter_frames` | `video_path`, `indices`, `start`, `count`, `info` | Task 21 landed; doc still unfixed → Task 27 |
> | `load_video_quality` | `video_path`, `report_path`, `workers`, `motion_stride` | Task 27 |
> | `sample_fps` | 9 params | Task 12 |
> | `sample_optical_flow` | 10 params | Task 13 |
> | `estimate_camera_distortion` | `frames`, `max_frames` | **deleted by Task 15** — drops off the list |
> | `undistort_frames` | `frames`, `profile` | Task 16 rewrites it |
>
> **Seven already pass**, and they are exactly the functions Tasks 1 and 10 wrote or rewrote:
> `filter_frame_quality`, `sample_uniform`, and the five `frames.py` names (`frame_idx_from_path`,
> `frame_paths`, `read_frames`, `read_manifest`, `write_frames`). The pre-flight's claim that
> "Tasks 10-26 already wrote the new functions in this form" is holding — the list shrinks by one
> per rewritten function exactly as predicted.
>
> **Projected size when Task 27 starts** (after Phase C and Tasks 12/13 land): `estimate_camera_distortion`
> is gone, `undistort_frames`/`calibrate_camera`/`sample_fps`/`sample_optical_flow` are rewritten in
> the new form by their own tasks, leaving **six** docstrings — `analysis_gray`, `compute_video_quality`,
> `extract_frame`, `get_video_info`, `iter_frames`, `load_video_quality`. Do not size the task off the
> figure `11`; size it off `6`, and re-run the lint at the top of the task to get the real list.
>
> Reproduce with `inspect.signature` + `inspect.getdoc` over `preproc.__all__`; it needs no fixtures
> and no video, and it runs in ~1 s outside pytest.

### Task 27: the docstring pass, and the rule that keeps it

**Files:**
- Modify: every public function in `collab_splats/preproc/`, `CLAUDE.md`
- Test: `tests/preproc/test_docstrings.py` (create)

The brief's third item: docstrings state what a function does, what its inputs are, and what its outputs are — not paragraphs. Tasks 10-26 already wrote the new and rewritten functions in this form. This task sweeps whatever they did not touch and pins the rule so it does not erode.

- [ ] **Step 1: Write the enforcing test**

Create `tests/preproc/test_docstrings.py`:

```python
"""
The docstring contract for preproc's public surface.

Every public function documents its inputs and its outputs. This is a lint, not
a behaviour test — it exists because the alternative is a slow drift back to
paragraphs that describe neither.
"""

import inspect

import pytest

from collab_splats import preproc

PUBLIC = [
    (name, getattr(preproc, name))
    for name in preproc.__all__
    if inspect.isfunction(getattr(preproc, name))
]


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_public_function_documents_its_inputs_and_outputs(name, fn):
    doc = inspect.getdoc(fn)
    assert doc, f"{name} has no docstring"

    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    if params:
        assert "Args:" in doc, f"{name} takes {params} and documents none of them"
        for p in params:
            assert f"{p}:" in doc, f"{name} does not document '{p}'"

    if "-> None" not in str(inspect.signature(fn)):
        assert "Returns:" in doc, f"{name} returns something and documents nothing"


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_summary_is_one_line(name, fn):
    doc = inspect.getdoc(fn)
    summary = doc.split("\n\n")[0]

    assert "\n" not in summary, f"{name}'s summary spans multiple lines"
    assert len(summary) <= 100, f"{name}'s summary is {len(summary)} chars"
    assert not summary.startswith(name), f"{name}'s summary restates its own name"
```

- [ ] **Step 2: Run it and read the failures**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_docstrings.py -v
```

Expected: a list of exactly which functions still need work. That list is this task's checklist.

- [ ] **Step 3: Fix each one**

For every failure, rewrite the docstring in the house form:

```python
def f(a, b):
    """
    One line saying what it does.

    Args:
        a: what a is.
        b: what b is.

    Returns:
        What comes back.
    """
```

Deleting is as valid as rewriting: a bullet that restates the signature, explains a parameter that no longer exists, or narrates history goes.

- [ ] **Step 4: Run it clean**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/preproc/test_docstrings.py -v
```

Expected: PASS, every parametrized case.

- [ ] **Step 5: Pin the rule in `CLAUDE.md`**

In the **Code Style** section, replace the existing `**Docstrings:**` bullet with:

```markdown
- **Docstrings:** every public function and class gets a one-line summary docstring. The `"""` open and close on their own lines — summary starts on the line after the opening quotes, never on the same line. Multi-line docstrings put a blank line between the summary and what follows. A function with parameters documents every one under `Args:`; a function that returns something documents it under `Returns:`. Bullets or Args/Returns, not prose blocks. No restating the function name. No padding. `tests/preproc/test_docstrings.py` enforces this for `preproc`; extend it as other modules are cleaned up.
```

- [ ] **Step 6: Run everything, one last time**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
graphify update .
```

Expected: green apart from `docs/known-test-failures.md`; smoke exits 0.

- [ ] **Step 7: Commit**

```bash
black collab_splats/preproc tests/preproc
isort collab_splats/preproc tests/preproc
git commit --only collab_splats/preproc tests/preproc CLAUDE.md \
  -m "docs(preproc): Args/Returns on every public function, enforced by test"
```

---

### Task 28: close the work out

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/CHANGELOG.md`
- Modify: `collab_splats/dashboard/app.py:381`, `collab_splats/wrapper/reconstructor.py:244`, `tests/pointcloud/test_loger_creator.py:734` (stale ffprobe comments — see Step 3b)

- [ ] **Step 1: Count what happened**

```bash
/opt/venv/reconstruction/bin/python -c "
from pathlib import Path
total = 0
for p in sorted(Path('collab_splats/preproc').glob('*.py')):
    n = len(p.read_text().splitlines())
    total += n
    print(f'{n:5d}  {p}')
print(f'{total:5d}  TOTAL  (was 2268)')
"
```

Expected: ~1485. A number materially above that means something the spec said to delete is still there — find it before writing the changelog.

- [ ] **Step 2: Write the changelog entry**

Append to `docs/superpowers/CHANGELOG.md`, following the format of the entries above it: what changed per module, what broke, and the measured numbers — the filter's 5.1%/14.5% cuts from Task 14, PyAV's 1.17×/2.19×/10×, undistortion's framing change, and the line count.

- [ ] **Step 3: Update `CLAUDE.md`**

Remove `preproc-centralization` from **In-Flight Work** if it was listed. Update the `preproc/` block of the architecture tree:

```
  preproc/                 # video preprocessing: measure (qa) then select (sampling)
    video.py               # PyAV decode: get_video_info, iter_frames, extract_frame
    qa.py                  # report-only capture quality: compute_video_quality, load_video_quality
    sampling.py            # context_indices + sample_fps | sample_uniform | sample_optical_flow, all from filter_frame_quality's eligible pool
    frames.py              # images/frame_NNNNNN.png + frames.json: the COLMAP-style keyframe store
    undistort.py           # calibrate_camera (pycolmap) + undistort_frames (pycolmap framing, cv2 pixels)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
```

- [ ] **Step 3b: Kill the last three ffprobe comments outside `preproc/`**

Task 20 removed ffprobe from `video.py`; Tasks 21 and 22 removed the ffmpeg binary. Three
comments elsewhere in the repo still describe the old architecture and no other task touches
their files. They are comments only — no behaviour changes.

```bash
rtk proxy grep -rn "ffprobe" collab_splats/ tests/ CLAUDE.md
```

Expected before this step: exactly three hits outside `preproc/` —
`collab_splats/dashboard/app.py:381`, `collab_splats/wrapper/reconstructor.py:244`,
`tests/pointcloud/test_loger_creator.py:734`. Reword each to say the probe is a PyAV container
parse. **Expected after: zero hits for `ffprobe` anywhere in the repo, `CLAUDE.md` included.**
If the grep still finds one in `collab_splats/preproc/`, a Phase D task did not finish —
stop and report rather than editing it here.

- [ ] **Step 4: Commit**

```bash
git commit --only CLAUDE.md docs/superpowers/CHANGELOG.md \
  -m "docs: preproc centralization changelog"
```

- [ ] **Step 5: Tell the user what shipped**

Report the line count, the two gate results, and the four breaking changes a caller outside this repo would hit: `FrameStore` is gone, `filter_frame_quality`'s kwargs changed, `undistort_frames` takes a camera and returns a larger image, and `preproc.video.context_indices` / `decode_context` moved.
