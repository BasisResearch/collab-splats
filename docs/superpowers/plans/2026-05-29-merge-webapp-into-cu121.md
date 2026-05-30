# Merge feat/webapp → refactor/cu121 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge `feat/webapp` (FastAPI app, semantics viz, mesh gen, localize) into `refactor/cu121` (frame_sampling overhaul, VGGTOmega, SL(4) LC, semantics refactor), making `refactor/cu121` the single integration branch.

**Architecture:** `git merge feat/webapp --no-ff` on `refactor/cu121`. Conflicts resolved manually per zone using cu121 as ground truth for backend/pipeline and webapp for net-new UI additions. Merge commit preserves full history from both branches.

**Tech Stack:** git, Python 3.11 (`/opt/conda/envs/reconstruction/bin/python`), pytest

---

## Conflict Map

Files modified by BOTH branches (will conflict):

**dashboard/**
- `collab_splats/dashboard/app.py`
- `collab_splats/dashboard/state.py`
- `collab_splats/dashboard/panes/localize.py`
- `collab_splats/dashboard/panes/preprocess.py`
- `collab_splats/dashboard/panes/reconstruct.py`
- `collab_splats/dashboard/panes/semantics.py`
- `collab_splats/dashboard/panes/visualize.py`
- `collab_splats/dashboard/operation_log.py`
- `collab_splats/dashboard/video_server.py`

**semantics/**
- `collab_splats/semantics/__init__.py`
- `collab_splats/semantics/features/base.py`
- `collab_splats/semantics/features/dino.py`
- `collab_splats/semantics/features/maskclip.py`
- `collab_splats/semantics/features/talk2dino.py`
- `collab_splats/semantics/compression.py`
- `collab_splats/semantics/utils.py`
- All `tests/semantics/` files

**infra/setup**
- `setup.sh`, `setup/feedforward.sh`, `setup/hloc.sh`, `setup/vggt_slam.sh`
- `requirements.txt`
- `Dockerfile`, `Dockerfile.cu121`
- `CLAUDE.md`, `README.md`, `Makefile`

**Net-new from webapp (no conflict expected):**
- `collab_splats/webapp/` — entire FastAPI app
- `.superpowers/brainstorm/` — brainstorm artifacts

---

### Task 1: Checkout refactor/cu121 and initiate merge

**Files:** none modified yet

- [ ] **Step 1: Switch to target branch**

```bash
git checkout refactor/cu121
git status  # confirm clean working tree
```

- [ ] **Step 2: Run merge**

```bash
git merge feat/webapp --no-ff --no-commit
```

`--no-commit` prevents auto-commit on clean merge so you can review before committing.

- [ ] **Step 3: Triage conflicts**

```bash
git status | grep "both modified"
git diff --name-only --diff-filter=U
```

Record the list of conflicted files. Compare against the Conflict Map above — any unexpected conflicts warrant investigation before resolving.

---

### Task 2: Resolve dashboard/app.py and dashboard/state.py conflicts

**Files:**
- Modify: `collab_splats/dashboard/app.py`
- Modify: `collab_splats/dashboard/state.py`

**Context:**
- cu121 added: `_try_discover_cache`, `_load_cached_features`, auto-discover wiring to `output_dir` + method watchers, zarr stream write, 1920px cap
- webapp added: localize extractor fields in `AppState`, localize method field, progress callback wiring

**Resolution strategy:** Take cu121 as base. Graft webapp additions that are absent in cu121.

- [ ] **Step 1: View each side of the conflict**

```bash
git show feat/webapp:collab_splats/dashboard/state.py | grep -n "localize_extractor\|localize_method\|progress_callback" | head -30
git show refactor/cu121:collab_splats/dashboard/state.py | grep -n "localize_extractor\|localize_method\|progress_callback" | head -30
```

- [ ] **Step 2: Resolve state.py**

Open `collab_splats/dashboard/state.py`. For each `<<<<<<` marker:
- Keep cu121's version (`=======` to `>>>>>>>` is webapp's)
- Exception: if cu121 is missing `localize_method`, `localize_extractor`, or `progress_callback` fields that webapp added, add them to the cu121 version

- [ ] **Step 3: Resolve app.py using same strategy**

Open `collab_splats/dashboard/app.py`. For each conflict:
- Keep cu121's auto-discover wiring
- Keep webapp's any net-new route handlers or wiring that cu121 doesn't have

- [ ] **Step 4: Mark resolved**

```bash
git add collab_splats/dashboard/app.py collab_splats/dashboard/state.py
```

---

### Task 3: Resolve dashboard/panes/ conflicts

**Files:**
- Modify: `collab_splats/dashboard/panes/localize.py`
- Modify: `collab_splats/dashboard/panes/preprocess.py`
- Modify: `collab_splats/dashboard/panes/reconstruct.py`
- Modify: `collab_splats/dashboard/panes/semantics.py`
- Modify: `collab_splats/dashboard/panes/visualize.py`

**Context per pane:**
- `localize.py`: cu121 added zarr cache, batch mode, streaming table, CSV export; webapp added extractor selector, progress bar
- `preprocess.py`: cu121 added ffmpeg rotation-aware decode, `_write_frames_zarr`; webapp added video player, frame strip UI
- `reconstruct.py`: cu121 has creator config tweaks; webapp added SSE log streaming
- `semantics.py`: cu121 has semantics refactor (new extractor structure); webapp adds 3-panel viz
- `visualize.py`: cu121 has auto-discover, VTK defer; webapp adds mesh toggle, ground plane, frustums

**Resolution strategy:**
- `localize.py`: cu121 base + graft webapp's extractor/progress additions
- `preprocess.py`: cu121 base (ffmpeg is correct); check if webapp's video player UI is in dashboard pane or webapp only
- `semantics.py`: cu121 base (structural refactor must be preserved); graft webapp's 3-panel viz rendering
- `visualize.py`: cu121 base + graft webapp's mesh/frustum/ground-plane additions

- [ ] **Step 1: Resolve each pane file**

For each file in `collab_splats/dashboard/panes/`, open it, resolve each conflict marker:
```
<<<<<<< HEAD           ← cu121 version (keep as base)
...
=======
...
>>>>>>> feat/webapp    ← webapp additions (graft if net-new)
```

- [ ] **Step 2: Spot-check semantics.py carefully**

cu121's semantics refactor restructured the whole module. webapp's semantics pane changes must use the new extractor API from cu121, not the old one. Verify any webapp code that calls `BaseFeatureExtractor` uses the cu121 signature:

```bash
git show refactor/cu121:collab_splats/semantics/features/base.py | grep -n "^class\|^    def " | head -30
```

Compare against how webapp's semantics pane calls the extractor. Update calls to match cu121 API if needed.

- [ ] **Step 3: Mark resolved**

```bash
git add collab_splats/dashboard/panes/
```

---

### Task 4: Resolve semantics/ module conflicts

**Files:**
- Modify: `collab_splats/semantics/__init__.py`
- Modify: `collab_splats/semantics/features/base.py`
- Modify: `collab_splats/semantics/features/dino.py`
- Modify: `collab_splats/semantics/features/maskclip.py`
- Modify: `collab_splats/semantics/features/talk2dino.py`
- Modify: `collab_splats/semantics/compression.py`
- Modify: `collab_splats/semantics/utils.py`

**Context:**
- cu121's `b3ea4fe` squash included a "semantics refactor" — likely restructured extractors into `features/` + `segmentation/` subpackages
- webapp's semantics changes added query API (`get()` method on extractors, text→feature→similarity scoring)

**Resolution strategy:** cu121 structure wins entirely. Graft only webapp's query/similarity additions that are absent.

- [ ] **Step 1: Identify what webapp added to semantics**

```bash
git log feat/webapp --oneline -- collab_splats/semantics/ | head -15
git show feat/webapp:collab_splats/semantics/features/base.py | grep -n "def get\|def query\|similarity" | head -20
```

- [ ] **Step 2: Check if cu121 already has query API**

```bash
git show refactor/cu121:collab_splats/semantics/features/base.py | grep -n "def get\|def query\|similarity" | head -20
```

If cu121 already has it (from the squash), accept cu121 for all semantics conflicts — no grafting needed.
If cu121 is missing it, graft webapp's `get()` / query method onto cu121's class structure.

- [ ] **Step 3: Resolve each conflicted semantics file**

For each file: take cu121 as base, graft any webapp additions that are genuinely absent.

- [ ] **Step 4: Mark resolved**

```bash
git add collab_splats/semantics/
```

---

### Task 5: Resolve tests/semantics/ conflicts

**Files:**
- Modify: `tests/semantics/` (all conflicted test files)

**Context:**
- cu121 added tests for new semantics structure (segmentation, compression, positional debiasing, query API)
- webapp may have added tests for query/similarity

**Resolution strategy:** Keep ALL tests from both branches. If both added a `test_query_api.py`, merge the test functions — no test should be dropped.

- [ ] **Step 1: Resolve test conflicts**

For each conflicted test file: keep ALL test functions from both sides. Never drop a test.
If two functions have the same name, rename the duplicate (e.g. `test_query_basic` vs `test_query_basic_webapp`), then consolidate after verifying they test different things.

- [ ] **Step 2: Mark resolved**

```bash
git add tests/semantics/
```

---

### Task 6: Resolve setup/infra conflicts

**Files:**
- Modify: `setup.sh`, `setup/feedforward.sh`, `setup/hloc.sh`, `setup/vggt_slam.sh`
- Modify: `requirements.txt`
- Modify: `Dockerfile`, `Dockerfile.cu121`
- Modify: `CLAUDE.md`, `README.md`, `Makefile`

**Context:**
- `setup.sh`: cu121 pip installs for new deps (VGGTOmega, cuDSS split); webapp pip installs for FastAPI, uvicorn, etc.
- `requirements.txt`: union of both — keep all packages

**Resolution strategy:** Union all deps. For setup scripts, keep all install steps from both branches.

- [ ] **Step 1: Resolve requirements.txt**

For each conflict in `requirements.txt`: keep packages from BOTH sides. No package should be dropped.

- [ ] **Step 2: Resolve setup.sh**

Keep all `pip install` lines from both sides. Check for duplicate installs of the same package — keep the cu121 version (likely more up to date).

- [ ] **Step 3: Resolve Dockerfiles**

For `Dockerfile.cu121`: keep cu121's CUDA/env changes as base; add webapp's additional package installs.

- [ ] **Step 4: Resolve docs (CLAUDE.md, README.md)**

For `CLAUDE.md`: keep cu121's version (most recent project-wide guidance); check if webapp added webapp-specific sections and graft them.
For `README.md`: combine — keep cu121 as base, add any webapp-specific usage instructions from webapp's version.

- [ ] **Step 5: Mark resolved**

```bash
git add setup.sh setup/feedforward.sh setup/hloc.sh setup/vggt_slam.sh requirements.txt Dockerfile Dockerfile.cu118 Dockerfile.cu121 CLAUDE.md README.md Makefile
```

---

### Task 7: Resolve remaining conflicts

**Files:** Any files from `git diff --name-only --diff-filter=U` not yet resolved

- [ ] **Step 1: Find remaining conflicts**

```bash
git diff --name-only --diff-filter=U
```

- [ ] **Step 2: Resolve each remaining file**

For each file: open it, resolve `<<<<<<`/`=======`/`>>>>>>>` markers.
Default rule: cu121 wins for backend/pipeline; webapp wins for net-new UI/API additions; combine for docs/configs.

- [ ] **Step 3: Mark resolved**

```bash
git add <each resolved file>
```

- [ ] **Step 4: Verify no conflict markers remain**

```bash
grep -r "<<<<<<\|=======\|>>>>>>>" collab_splats/ tests/ setup.sh requirements.txt CLAUDE.md --include="*.py" --include="*.sh" --include="*.txt" --include="*.md" -l
```

Expected: no output. If any files listed, open and fix them.

---

### Task 8: Run tests and fix regressions

**Files:** any that need fixes to pass tests

- [ ] **Step 1: Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -x -q 2>&1 | tail -40
```

- [ ] **Step 2: Fix import errors first**

If tests fail with `ImportError` or `ModuleNotFoundError`, the module restructuring from cu121's semantics refactor likely broke webapp's imports. Find and fix:

```bash
grep -rn "from collab_splats.semantics" collab_splats/webapp/ collab_splats/dashboard/ --include="*.py"
```

Update any import paths to match cu121's new module structure.

- [ ] **Step 3: Fix test failures**

For each failing test, read the traceback and fix the underlying issue. Do not skip or delete tests.

- [ ] **Step 4: Run tests again to confirm green**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -q 2>&1 | tail -20
```

Expected: all tests pass (or same failures as pre-merge baseline).

---

### Task 9: Commit merge

**Files:** no new changes

- [ ] **Step 1: Verify staging is complete**

```bash
git status
```

Expected: "All conflicts fixed but you are still merging." with no untracked/unstaged changes that belong in the merge.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
merge(cu121): integrate feat/webapp — FastAPI app, semantics viz, mesh gen, localize

Merges feat/webapp into refactor/cu121. Conflict resolution:
- dashboard/: cu121 backend (auto-discover, ffmpeg, zarr) + webapp UI additions
- semantics/: cu121 refactor structure preserved; query API grafted from webapp
- setup/deps: union of all packages from both branches

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

```bash
git log --oneline -3
git status  # should be clean
```

---

### Task 10: Post-merge smoke check

- [ ] **Step 1: Verify FastAPI app can be imported**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.webapp.app import app; print('webapp OK')"
```

Expected: `webapp OK`

- [ ] **Step 2: Verify dashboard can be imported**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.dashboard.app import CollabSplatsApp; print('dashboard OK')"
```

Expected: `dashboard OK`

- [ ] **Step 3: Verify semantics module**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.semantics.features.base import BaseFeatureExtractor; print('semantics OK')"
```

Expected: `semantics OK`
