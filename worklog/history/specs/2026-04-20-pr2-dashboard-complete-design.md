# PR2 Dashboard Complete — Design Spec

**Date:** 2026-04-20  
**Branch:** `refactor/dashboard-complete` (base: `refactor/core-modules`)  
**Status:** Approved, ready for implementation planning

---

## Context

PR1 (`refactor/core-modules`) consolidates core modules: semantics, nerfstudio, utils, pointcloud. Core implementation (Tasks 1–4) is done and 25 tests pass, but PR1 has uncommitted frame_sampling-move changes and is not yet finalized for merge. It is far enough along to serve as the stable base for PR2.

PR2 brings in the Panel-based dashboard (`SemanticsDashboard`, `ConfigPanel`, `video_discovery`) from the `refactor/dashboard-optical-flow` snapshot branch, wires it to the updated module layout on PR1, and adds CUDA auto-detect UX. Development is iterative: dashboard ships first, MapAnything and other integrations follow in subsequent iterations (Phase 3+).

PRs are stacked: PR2 branches from `refactor/core-modules`, not `main`. Changes flow core → dashboard via rebase.

---

## Scope

**In:**
- `collab_splats/dashboard/` — SemanticsDashboard, ConfigPanel, video_discovery, `__init__.py`, `__main__.py`
- `tests/dashboard/` — smoke tests (18 tests, no CUDA/browser required)
- Import path update: `collab_splats.semantics.frame_sampling` → `collab_splats.utils.frame_sampling`
- CUDA auto-detect: `_DEFAULT_DEVICE` default on 3 device dropdowns
- PR description files for PR1 and PR2

**Out:**
- MapAnything UI (Phase 3)
- Dashboard UX iteration (Phase 2)
- Opening GitHub PRs (deferred — .md files hold the content)

---

## Architecture

Dashboard sits entirely in `collab_splats/dashboard/`. No core module changes. Dependency direction: dashboard → semantics, utils, pointcloud (read-only).

```
collab_splats/
  dashboard/
    __init__.py        # build_app(), run_app()
    __main__.py        # CLI entry point
    semantics.py       # SemanticsDashboard (Panel/MaterialTemplate)
    config_panel.py    # ConfigPanel
    video_discovery.py # filesystem scanner
tests/
  dashboard/
    test_semantics_smoke.py   # 18 smoke tests
    test_config_panel.py
    test_video_discovery.py
    test_numpy_fix.py
worklog/history/prs/
    pr1-core-modules.md
    pr2-dashboard-complete.md
```

---

## Implementation Phases

### Phase 0 — Prerequisite commit on `refactor/core-modules`
Commit the 8 pending frame_sampling-move files before branching. Files:
- `CLAUDE.md`
- `collab_splats/semantics/__init__.py`
- `collab_splats/utils/__init__.py`
- `collab_splats/utils/frame_sampling.py` (rename from semantics/)
- `collab_splats/wrapper/splatter.py`
- `worklog/WORKLOG.md`
- `tests/utils/test_frame_sampling.py` (rename from semantics/)
- `tests/utils/__init__.py`

### Phase 1 — Create branch + bring in dashboard files
```bash
git checkout -b refactor/dashboard-complete refactor/core-modules
git checkout refactor/dashboard-optical-flow -- collab_splats/dashboard/ tests/dashboard/
```
Commit: `feat(dashboard): bring in SemanticsDashboard, ConfigPanel, video_discovery`

### Phase 2 — Update frame_sampling imports
All dashboard files that import `collab_splats.semantics.frame_sampling` → `collab_splats.utils.frame_sampling`.

Known sites:
- `collab_splats/dashboard/semantics.py`
- `tests/dashboard/test_semantics_smoke.py`

Search for any others: `grep -r "semantics.frame_sampling" collab_splats/dashboard/ tests/dashboard/`

Commit: `fix(dashboard): update frame_sampling imports to canonical utils path`

### Phase 3 — CUDA auto-detect
Cherry-pick `cb22e58` from the `dashboard` branch.

Adds:
```python
import torch
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
Applied to 3 dropdowns: `device_dd`, `seg_device_dd`, `query_device_dd`.

Commit already exists — cherry-pick is atomic.

### Phase 4 — Verify tests
```bash
pytest tests/dashboard/ -v
```
All 18+ smoke tests must pass. No CUDA or browser required.

Verify import chain:
```bash
python -c "from collab_splats.dashboard import SemanticsDashboard, build_app; print('ok')"
```

### Phase 5 — WORKLOG + PR description files
Update `worklog/WORKLOG.md`:
- Check off PR1 Tasks 1–4 (already implemented)
- Check off "Update PR1 description" item
- Mark `refactor/dashboard-complete` as in-progress
- Add session log entry

Create `worklog/history/prs/pr1-core-modules.md` — PR title, base branch (`main`), body.
Create `worklog/history/prs/pr2-dashboard-complete.md` — PR title, base branch (`refactor/core-modules`), body (from WORKLOG PR body section).

---

## Decisions

| Question | Decision | Reason |
|----------|----------|--------|
| CUDA auto-detect location | Dashboard only | Pure dropdown UX default; models handle device placement via PyTorch natively |
| MapAnything stub | Skip | Class exists in core-modules but full UI design is Phase 3; no dead UI |
| frame_sampling imports | Update to canonical path | Removes reliance on semantics compat re-export; cleaner dependency |
| Open GitHub PRs now? | No | Write .md files; open PRs when ready for review |

---

## Verification

- `pytest tests/dashboard/ -v` — all pass (no GPU/browser)
- `pytest tests/ -v` — 25 pass, 1 skip, same 6 pre-existing env failures as before
- `python -c "from collab_splats.dashboard import SemanticsDashboard, build_app"` — no import errors
