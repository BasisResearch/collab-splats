# Phase 1 PR 1 — Core Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean `refactor/core-modules` branch from main containing semantics, pointcloud, nerfstudio restructure, utils shims, and wrapper — with no dashboard files and two bug fixes applied.

**Architecture:** Snapshot non-dashboard files from `refactor/dashboard-optical-flow` onto a fresh branch from main, rebase `refactor/nerfstudio-submodule` on top, then apply two targeted bug fixes. No behavior changes — pure structural refactor.

**Tech Stack:** Python, pytest, git, GitHub CLI (`gh`)

---

## File Map

| File | Action |
|------|--------|
| `collab_splats/semantics/features.py` | Modify: guard `maskclip_onnx` import + fix `pytorch_gc` |
| `collab_splats/semantics/` (all others) | Bring from HEAD snapshot |
| `collab_splats/pointcloud/` | Bring from HEAD snapshot |
| `collab_splats/nerfstudio/` | Bring from `refactor/nerfstudio-submodule` after rebase |
| `collab_splats/utils/features.py` | Bring from HEAD snapshot (shim only) |
| `collab_splats/utils/segmentation.py` | Bring from HEAD snapshot (shim only) |
| `collab_splats/utils/pointcloud.py` | Bring from HEAD snapshot (shim only) |
| `collab_splats/wrapper/config.py` | Bring from HEAD snapshot |
| `collab_splats/wrapper/splatter.py` | Bring from HEAD snapshot |
| `collab_splats/wrapper/__init__.py` | Bring from HEAD snapshot |
| `collab_splats/__init__.py` | Bring from HEAD snapshot |
| `pyproject.toml` | Bring from HEAD snapshot |
| `REFACTOR.md` | Bring from HEAD snapshot |
| `docs/semantics.md` | Bring from HEAD snapshot |
| `docs/splats/configs/` | Bring from HEAD snapshot |
| `docs/superpowers/` | Bring from HEAD snapshot |
| `tests/semantics/` | Bring from HEAD snapshot |
| `tests/pointcloud/` | Bring from HEAD snapshot |
| `tests/semantics/test_features_guards.py` | Create: new tests for bug fixes |

---

## Task 1: Branch Cleanup

**Files:** none (git only)

- [ ] **Step 1: Delete absorbed ancestor branches**

```bash
git branch -d refactor/pointcloud refactor/semantics refactor/dashboard
```
Expected: 3 branches deleted (all fully absorbed by `refactor/dashboard-optical-flow`)

- [ ] **Step 2: Drop superseded stashes**

```bash
git stash drop stash@{1}
git stash drop stash@{0}
```
Expected: `Dropped stash@{1}` then `Dropped stash@{0}`. Note: `stash@{2}` (WIP meshing on main) — keep for Phase 3.

- [ ] **Step 3: Delete merged branch**

```bash
git branch -d tlb-repo-config 2>/dev/null && echo "deleted" || echo "already gone"
```

- [ ] **Step 4: Verify branch list**

```bash
git branch
```
Expected output includes: `dashboard`, `main`, `refactor/dashboard-optical-flow`, `refactor/nerfstudio-submodule`, `refactor/core-modules` (not yet), `tlb-grouping-segmentation`, `tlb-improve-mesh`, `tlb-improve-splatter`.

---

## Task 2: Create `refactor/core-modules` Branch

**Files:** All core files brought from `refactor/dashboard-optical-flow`

- [ ] **Step 1: Create branch from main**

```bash
git checkout main -b refactor/core-modules
```
Expected: `Switched to a new branch 'refactor/core-modules'`

- [ ] **Step 2: Snapshot core files from HEAD**

```bash
git checkout refactor/dashboard-optical-flow -- \
  collab_splats/__init__.py \
  collab_splats/semantics/ \
  collab_splats/pointcloud/ \
  collab_splats/utils/features.py \
  collab_splats/utils/segmentation.py \
  collab_splats/utils/pointcloud.py \
  collab_splats/wrapper/__init__.py \
  collab_splats/wrapper/config.py \
  collab_splats/wrapper/splatter.py \
  tests/semantics/ \
  tests/pointcloud/ \
  pyproject.toml \
  REFACTOR.md \
  docs/semantics.md \
  docs/splats/configs/ \
  docs/superpowers/ \
  .gitignore
```
Expected: many files staged, no errors.

- [ ] **Step 3: Verify no dashboard files are present**

```bash
ls collab_splats/dashboard/ 2>/dev/null && echo "ERROR: dashboard present" || echo "OK: no dashboard"
```
Expected: `OK: no dashboard`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: snapshot core modules from stacked branch — semantics, pointcloud, wrapper, utils shims"
```

---

## Task 3: Fix `maskclip_onnx` Bare Import

**Files:**
- Modify: `collab_splats/semantics/features.py:24`
- Create: `tests/semantics/test_features_guards.py`

- [ ] **Step 1: Write the failing test**

Create `tests/semantics/test_features_guards.py`:

```python
import pytest
import collab_splats.semantics.features as feat_mod


def test_maskclip_extractor_raises_without_package(monkeypatch):
    """MaskCLIPExtractor raises ImportError when maskclip_onnx unavailable."""
    orig = feat_mod._MASKCLIP_AVAILABLE
    try:
        feat_mod._MASKCLIP_AVAILABLE = False
        with pytest.raises(ImportError, match="maskclip_onnx"):
            feat_mod.MaskCLIPExtractor()
    finally:
        feat_mod._MASKCLIP_AVAILABLE = orig
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/semantics/test_features_guards.py::test_maskclip_extractor_raises_without_package -v
```
Expected: `FAILED` — `AttributeError: module has no attribute '_MASKCLIP_AVAILABLE'`

- [ ] **Step 3: Apply fix to `features.py`**

Replace line 24 (`import maskclip_onnx`) with:

```python
try:
    import maskclip_onnx
    _MASKCLIP_AVAILABLE = True
except ImportError:
    maskclip_onnx = None  # type: ignore[assignment]
    _MASKCLIP_AVAILABLE = False
```

Then find `MaskCLIPExtractor.__init__` (first `def __init__` after line 147) and add as its first statement:

```python
if not _MASKCLIP_AVAILABLE:
    raise ImportError(
        "maskclip_onnx is not installed. Install with: pip install maskclip_onnx"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/semantics/test_features_guards.py::test_maskclip_extractor_raises_without_package -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_features_guards.py
git commit -m "fix: guard maskclip_onnx import behind try/except with pip hint"
```

---

## Task 4: Fix `pytorch_gc` CPU Crash

**Files:**
- Modify: `collab_splats/semantics/features.py:43-46`
- Modify: `tests/semantics/test_features_guards.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/semantics/test_features_guards.py`:

```python
def test_pytorch_gc_safe_on_cpu(monkeypatch):
    """pytorch_gc must not raise RuntimeError on CPU-only systems."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import importlib
    import collab_splats.semantics.features as mod
    importlib.reload(mod)
    mod.pytorch_gc()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/semantics/test_features_guards.py::test_pytorch_gc_safe_on_cpu -v
```
Expected: `FAILED` with `RuntimeError: No CUDA GPUs are available` on CPU-only systems. On GPU systems may pass — apply fix regardless.

- [ ] **Step 3: Replace `pytorch_gc` body in `features.py`**

Replace lines 43–46:
```python
def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
```
With:
```python
def pytorch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
```

- [ ] **Step 4: Run both guard tests**

```bash
pytest tests/semantics/test_features_guards.py -v
```
Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_features_guards.py
git commit -m "fix: guard torch.cuda calls in pytorch_gc behind is_available() check"
```

---

## Task 5: Rebase and Integrate `refactor/nerfstudio-submodule`

**Files:**
- Create: `collab_splats/nerfstudio/` (all files from nerfstudio-submodule branch)

- [ ] **Step 1: List worktrees**

```bash
git worktree list
```
Expected: main worktree at `/workspace/collab-splats` and nerfstudio worktree at `/workspace/collab-splats/.worktrees/nerfstudio-submodule`

- [ ] **Step 2: Rebase nerfstudio-submodule onto refactor/core-modules**

```bash
cd .worktrees/nerfstudio-submodule
git rebase refactor/core-modules
```
Expected: `Successfully rebased` or conflict markers if `pyproject.toml` / `collab_splats/__init__.py` diverged.

- [ ] **Step 3: Resolve conflicts if any**

For `pyproject.toml` conflicts: keep nerfstudio-submodule's entry point paths (they point to `collab_splats.nerfstudio.*`), keep refactor/core-modules' dep additions.

For `collab_splats/__init__.py` conflicts: accept nerfstudio-submodule's version (it removes nerfstudio-specific exports from the top-level init).

```bash
# For each conflicted file:
git add <file>
git rebase --continue
```

- [ ] **Step 4: Merge into refactor/core-modules**

```bash
cd /workspace/collab-splats
git checkout refactor/core-modules
git merge --ff-only refactor/nerfstudio-submodule 2>/dev/null || \
  git merge refactor/nerfstudio-submodule -m "refactor: integrate nerfstudio submodule restructure"
```

- [ ] **Step 5: Verify nerfstudio subdir exists**

```bash
ls collab_splats/nerfstudio/
```
Expected: `__init__.py  models/  configs/  datamanagers/` (and possibly `trainer_config.py`, `model_loading.py`)

- [ ] **Step 6: Verify nerfstudio entry points resolve**

```bash
python -c "
import importlib
for ep in ['rade-gs', 'rade-features']:
    print(ep, 'OK')
" 2>/dev/null || python -m pytest tests/ -k "nerfstudio" -v --co -q 2>/dev/null | head -10
```
If no nerfstudio tests exist, verify entry points in `pyproject.toml` reference `collab_splats.nerfstudio.*` paths.

---

## Task 6: Full Test Suite Verification

**Files:** none (verification only)

- [ ] **Step 1: Run all non-dashboard tests**

```bash
pytest tests/semantics/ tests/pointcloud/ -v --tb=short
```
Expected: all pass, zero failures, zero import errors.

- [ ] **Step 2: Verify import chain**

```bash
python -c "from collab_splats.semantics import features, segmentation, protocols, frame_sampling; print('semantics OK')"
python -c "from collab_splats.pointcloud import base, feedforward, sfm; print('pointcloud OK')"
python -c "from collab_splats.wrapper import Splatter, SplatterConfig; print('wrapper OK')"
python -c "from collab_splats.wrapper.config import ConfigLoader; print('config OK')"
```
Expected: each line prints its OK message.

- [ ] **Step 3: Verify utils shims still export**

```bash
python -c "
from collab_splats.utils.features import BaseFeatureExtractor
from collab_splats.utils.segmentation import Segmentor
print('shims OK')
"
```
Expected: `shims OK` (shims re-export from `collab_splats.semantics.*`)

---

## Task 7: Submit PR 1

- [ ] **Step 1: Push branch**

```bash
git push -u origin refactor/core-modules
```

- [ ] **Step 2: Create PR**

```bash
gh pr create \
  --title "refactor: extract core modules — semantics, pointcloud, nerfstudio" \
  --base main \
  --body "$(cat <<'EOF'
## Summary
- Extracts `collab_splats.semantics` submodule (features, segmentation, protocols, frame_sampling)
- Extracts `collab_splats.pointcloud` submodule (base, feedforward, sfm, utils, registry)
- Restructures nerfstudio-specific code into `collab_splats/nerfstudio/` subdir, updates pyproject.toml entry points
- Reduces `collab_splats/utils/` to backward-compat shims (no behavior change for existing callers)
- Fixes: `maskclip_onnx` bare import crash on machines without the package
- Fixes: `pytorch_gc()` `RuntimeError` on CPU-only systems

## No behavior changes
All model behavior unchanged. Entry points updated to new paths.

## Test plan
- [ ] `pytest tests/semantics/ tests/pointcloud/ -v` — all pass
- [ ] `python -c "from collab_splats.semantics import features"` — no crash without maskclip_onnx installed
- [ ] `python -c "from collab_splats.utils.features import BaseFeatureExtractor"` — shim still works
EOF
)"
```

Expected: PR URL printed. **Gate: PR 2 cannot start until this PR is merged.**
