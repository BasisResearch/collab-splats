# Phase 1 PR 2 — Dashboard Complete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `refactor/dashboard-complete` branch containing the Panel dashboard, ConfigPanel, video_discovery, optical flow frame sampling wired into UI, and CUDA auto-detect for all device dropdowns.

**Architecture:** Branch from main after PR 1 merges. Snapshot dashboard files from `refactor/dashboard-optical-flow`, then manually apply the CUDA auto-detect change (commit `cb22e58` from `dashboard` branch, not yet in HEAD).

**Tech Stack:** Python, Panel, pytest, git, GitHub CLI (`gh`)

**Gate: PR 1 (`refactor/core-modules`) must be merged to main before starting this plan.**

---

## File Map

| File | Action |
|------|--------|
| `collab_splats/dashboard/__init__.py` | Bring from `refactor/dashboard-optical-flow` |
| `collab_splats/dashboard/__main__.py` | Bring from `refactor/dashboard-optical-flow` |
| `collab_splats/dashboard/semantics.py` | Bring from `refactor/dashboard-optical-flow` + apply CUDA auto-detect |
| `collab_splats/dashboard/config_panel.py` | Bring from `refactor/dashboard-optical-flow` |
| `collab_splats/dashboard/video_discovery.py` | Bring from `refactor/dashboard-optical-flow` |
| `tests/dashboard/__init__.py` | Bring from `refactor/dashboard-optical-flow` |
| `tests/dashboard/test_semantics_smoke.py` | Bring + add CUDA auto-detect test |
| `tests/dashboard/test_config_panel.py` | Bring from `refactor/dashboard-optical-flow` |
| `tests/dashboard/test_numpy_fix.py` | Bring from `refactor/dashboard-optical-flow` |
| `tests/dashboard/test_video_discovery.py` | Bring from `refactor/dashboard-optical-flow` |

---

## Task 1: Create `refactor/dashboard-complete` Branch

**Files:** dashboard files brought from `refactor/dashboard-optical-flow`

- [ ] **Step 1: Verify PR 1 is merged**

```bash
gh pr list --state merged --search "refactor/core-modules"
```
Expected: shows the PR 1 entry as merged. If not merged, stop — do not proceed.

- [ ] **Step 2: Update main**

```bash
git fetch origin
git checkout main
git pull
```
Expected: main is up to date with PR 1 changes.

- [ ] **Step 3: Create branch**

```bash
git checkout -b refactor/dashboard-complete
```
Expected: `Switched to a new branch 'refactor/dashboard-complete'`

- [ ] **Step 4: Verify core modules are present (from PR 1)**

```bash
python -c "from collab_splats.semantics import features; print('core OK')"
```
Expected: `core OK`

- [ ] **Step 5: Snapshot dashboard files from stacked branch**

```bash
git checkout refactor/dashboard-optical-flow -- \
  collab_splats/dashboard/ \
  tests/dashboard/
```
Expected: `collab_splats/dashboard/` present, `tests/dashboard/` present.

- [ ] **Step 6: Verify snapshot**

```bash
ls collab_splats/dashboard/
```
Expected: `__init__.py  __main__.py  config_panel.py  semantics.py  video_discovery.py`

- [ ] **Step 7: Initial commit**

```bash
git add collab_splats/dashboard/ tests/dashboard/
git commit -m "refactor: add Panel dashboard — SemanticsDashboard, ConfigPanel, video_discovery, optical flow UI"
```

---

## Task 2: Apply CUDA Auto-Detect to Dashboard

Commit `cb22e58` from the `dashboard` branch is not in `refactor/dashboard-optical-flow`. Apply manually.

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`
- Modify: `tests/dashboard/test_semantics_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/dashboard/test_semantics_smoke.py`:

```python
def test_default_device_matches_cuda_availability():
    """_DEFAULT_DEVICE equals 'cuda' when CUDA available, else 'cpu'."""
    import torch
    from collab_splats.dashboard import semantics as mod
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert mod._DEFAULT_DEVICE == expected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_default_device_matches_cuda_availability -v
```
Expected: `FAILED` — `AttributeError: module 'collab_splats.dashboard.semantics' has no attribute '_DEFAULT_DEVICE'`

- [ ] **Step 3: Add `torch` import to `semantics.py`**

In `collab_splats/dashboard/semantics.py`, find the import block (around line 16–18 where `import numpy as np` appears). Add `import torch` before `import numpy as np`:

```python
import torch
import numpy as np
```

- [ ] **Step 4: Add `_DEFAULT_DEVICE` constant**

In `collab_splats/dashboard/semantics.py`, after the `CONFIGS_DIR = ...` line, add:

```python
_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 5: Set `value=_DEFAULT_DEVICE` on all three device dropdowns**

In `collab_splats/dashboard/semantics.py`, find each `pn.widgets.Select(name="Device", ...)` (there are exactly three — `device_dd`, `seg_device_dd`, `query_device_dd`). Add `value=_DEFAULT_DEVICE` to each:

```python
self.device_dd = pn.widgets.Select(
    name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
)
```

```python
self.seg_device_dd = pn.widgets.Select(
    name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
)
```

```python
self.query_device_dd = pn.widgets.Select(
    name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100
)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/dashboard/test_semantics_smoke.py::test_default_device_matches_cuda_availability -v
```
Expected: `PASSED`

- [ ] **Step 7: Commit**

```bash
git add collab_splats/dashboard/semantics.py tests/dashboard/test_semantics_smoke.py
git commit -m "feat: auto-detect CUDA and default all device dropdowns accordingly"
```

---

## Task 3: Full Dashboard Test Suite Verification

**Files:** none (verification only)

- [ ] **Step 1: Run all dashboard tests**

```bash
pytest tests/dashboard/ -v --tb=short
```
Expected: all pass, zero failures.

- [ ] **Step 2: Verify dashboard import chain**

```bash
python -c "from collab_splats.dashboard.semantics import SemanticsDashboard; print('SemanticsDashboard OK')"
python -c "from collab_splats.dashboard.config_panel import ConfigPanel; print('ConfigPanel OK')"
python -c "from collab_splats.dashboard.video_discovery import discover_videos; print('video_discovery OK')"
```
Expected: each prints its OK message.

- [ ] **Step 3: Verify `__main__` entry point is importable**

```bash
python -c "import collab_splats.dashboard.__main__; print('launcher OK')"
```
Expected: `launcher OK` (should not launch a server, just import cleanly)

- [ ] **Step 4: Run full test suite (regression check)**

```bash
pytest tests/ -v --tb=short --ignore=tests/dashboard/
```
Wait — this runs tests from PR 1's modules. All should still pass.
Expected: all pass.

---

## Task 4: Submit PR 2

- [ ] **Step 1: Push branch**

```bash
git push -u origin refactor/dashboard-complete
```

- [ ] **Step 2: Create PR**

```bash
gh pr create \
  --title "refactor: replace Gradio dashboard with Panel SemanticsDashboard" \
  --base main \
  --body "$(cat <<'EOF'
## Summary
- Replaces Gradio semantics dashboard with Panel `SemanticsDashboard`
- Adds `ConfigPanel` for per-video YAML config editing via the UI
- Adds `video_discovery` module for fieldwork filesystem scanning
- Wires optical flow frame sampling into dashboard UI (flow_threshold slider, Explore tab)
- Auto-detects CUDA availability and defaults all three device dropdowns accordingly

## Depends on
`refactor/core-modules` (must be merged first — dashboard imports from `collab_splats.semantics.*`)

## Test plan
- [ ] `pytest tests/dashboard/ -v` — all pass
- [ ] On a GPU machine: launch dashboard, verify all device dropdowns default to `cuda`
- [ ] On CPU-only machine: launch dashboard, verify all device dropdowns default to `cpu`
- [ ] Upload a video, run optical flow sampling, verify frame count respects flow_threshold slider
EOF
)"
```

Expected: PR URL printed.

---

## Post-PR Cleanup

After both PRs are merged:

- [ ] Delete `refactor/dashboard-optical-flow` (fully absorbed)

```bash
git branch -d refactor/dashboard-optical-flow
git push origin --delete refactor/dashboard-optical-flow
```

- [ ] Delete `dashboard` branch (CUDA fix now in PR 2)

```bash
git branch -d dashboard
git push origin --delete dashboard 2>/dev/null || true
```

- [ ] Delete `refactor/nerfstudio-submodule` branch and worktree (absorbed by PR 1)

```bash
git worktree remove .worktrees/nerfstudio-submodule
git branch -d refactor/nerfstudio-submodule
```
