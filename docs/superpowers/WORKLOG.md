# collab-splats Refactor Worklog

Living cross-session log. Update at start/end of each session. Replaces `PROGRESS.md` and `plans/2026-04-19-phase1-pr2-dashboard-complete.md`.

---

## Current State

| Branch | Status | Notes |
|--------|--------|-------|
| `refactor/core-modules` | ⏳ awaiting merge (PR1) | 25 pass, 1 skip (batching absorbed) |
| `refactor/dashboard-complete` | 🔒 not started (PR2) | design complete — ready to create |
| `refactor/dashboard-optical-flow` | 🗂 source only | snapshot into PR2, then delete |
| `dashboard` | 🗂 source only | cherry-pick `cb22e58` (CUDA auto-detect), then delete |
| `refactor/semantics` | ✅ deleted | absorbed into PR1 via cherry-pick |
| `tlb-grouping-segmentation` | 🧊 parked | separate plan TBD |
| `tlb-improve-splatter` | 🧊 parked | separate plan TBD |

---

## Architecture Decisions (2026-04-20)

### Parallel stacked PR development
- PR2 branches from `refactor/core-modules`, not `main`
- Changes flow one direction: core → dashboard via rebase
- Rebase PR2 onto PR1 when `collab_splats.semantics.*` or `collab_splats.pointcloud.*` API changes

### Dashboard = integration test harness
- Unit tests verify API contracts; dashboard verifies full pipeline feels right
- Build dashboard with semantics first (already working in PR1), validate visually
- Add MapAnything to both branches together when core implementation lands

### MapAnything stub pattern
```python
try:
    from collab_splats.pointcloud import MapAnythingCreator
    _MAPANYTHING_AVAILABLE = True
except ImportError:
    _MAPANYTHING_AVAILABLE = False
```
- MapAnything tab/widget exists in UI but disabled when `_MAPANYTHING_AVAILABLE = False`
- Stub is the development seam — unlocks automatically when PR1 lands `MapAnythingCreator`
- Tests use `pytest.importorskip` for MapAnything-dependent tests

---

## PR1 — Core Modules (`refactor/core-modules`)

### Done
- `collab_splats/semantics/` — features, segmentation, protocols, frame_sampling
- `collab_splats/nerfstudio/` — models, method_configs, datamanagers (moved from top-level)
- `collab_splats/utils/` — backward-compat shims only
- `collab_splats/wrapper/config.py` — ConfigLoader
- Bug fix: `maskclip_onnx` bare import guard
- Bug fix: `pytorch_gc()` CPU crash guard

### Remaining
- [ ] **Task 1:** `pointcloud/` skeleton — `base.py`, `utils.py`, `__init__.py`, shim, tests
- [ ] **Task 2:** `NerfstudioSfmCreator` (`pointcloud/sfm.py`) — hloc + pycolmap backends
- [ ] **Task 3:** `MapAnythingCreator` (`pointcloud/feedforward.py`) — feedforward reconstruction
- [ ] **Task 4:** Registry + Splatter integration — `get_creator()`, `pointcloud_method` config key

Full task details: `plans/2026-04-19-phase1-pr1-core-modules.md` (REFACTOR.md at repo root)

### Pending: Absorb `refactor/semantics` into PR1

Decision: collapse Phase 2 batching work into PR1 so we ship two clean PRs (core + dashboard) instead of a phased plan. Dashboard gets `forward_batch`/`infer_batch_size` on day 1.

`refactor/semantics` has 10 unique commits:
- `infer_batch_size()` VRAM-aware batch size utility
- `forward_batch`/`reshape_batch` protocol on `BaseFeatureExtractor` + all extractors (MaskCLIP, Talk2DINO, DINO)
- Batch regularization in datamanager
- Optical flow test tightening
- Semantics module usage guide

Steps:
- [ ] **Validate rebase:** `git checkout refactor/semantics && git rebase refactor/core-modules`
  - Clean → proceed. Conflicts → assess cost; if large, keep separate.
- [ ] **If clean:** fast-forward or cherry-pick commits onto `refactor/core-modules`
- [ ] **Run full test suite:** `pytest tests/ -v` — all pass
- [x] **Validate rebase** — full rebase conflicted (both branches created pointcloud/); used cherry-pick of batching-only commits instead
- [x] **Cherry-pick applied** — 6 of 10 commits landed (4 skipped: already in core-modules or superseded)
- [x] **Tests pass** — 25 pass, 1 skip; 6 pre-existing failures (nerfstudio env + GPU smoke test)
- [x] **Delete `refactor/semantics`** branch (absorbed)
- [ ] **Update PR1 description** to include batching protocol

---

## PR2 — Dashboard (`refactor/dashboard-complete`)

### Scope
- `collab_splats/dashboard/` — Panel `SemanticsDashboard`, `ConfigPanel`, `video_discovery`
- Optical flow frame sampling wired into UI
- CUDA auto-detect for all device dropdowns
- MapAnything tab (stub until PR1 Task 3 lands)

### Setup steps (do after PR1 has MapAnythingCreator or when ready to stub)
- [ ] `git checkout -b refactor/dashboard-complete` from `refactor/core-modules`
- [ ] `git checkout refactor/dashboard-optical-flow -- collab_splats/dashboard/ tests/dashboard/`
- [ ] Verify imports: dashboard already uses `collab_splats.semantics.frame_sampling` (correct)
- [ ] Apply CUDA auto-detect: `import torch`, `_DEFAULT_DEVICE`, set `value=_DEFAULT_DEVICE` on 3 dropdowns
- [ ] Add MapAnything stub tab with try/except import guard
- [ ] `pytest tests/dashboard/ -v` — all pass
- [ ] Verify import chain for `SemanticsDashboard`, `ConfigPanel`, `video_discovery`
- [ ] Push + open PR against `refactor/core-modules` base

### PR body (when ready)
```
Replaces Gradio dashboard with Panel SemanticsDashboard
Adds ConfigPanel for per-video YAML config editing
Adds video_discovery for fieldwork filesystem scanning
Wires optical flow frame sampling into dashboard UI
Auto-detects CUDA, defaults all device dropdowns accordingly
MapAnything tab stubbed — activates when pointcloud module lands
```

---

## Cleanup — Safe Now

- [ ] Delete `refactor/nerfstudio-submodule` — 0 unique commits vs `core-modules`, fully absorbed
  ```bash
  git worktree remove .worktrees/nerfstudio-submodule
  git branch -d refactor/nerfstudio-submodule
  ```

## Post-merge Cleanup (after both PRs merged to main)

- [ ] Delete `refactor/dashboard-optical-flow` (fully absorbed into PR2)
- [ ] Delete `dashboard` branch (CUDA fix absorbed into PR2)
- [ ] Review `stash@{0}` — WIP meshing on main before Phase 3

---

## Phase 2 — Dashboard Iteration (not started)

After Phase 1 merged + field-tested:
- UX iteration from field feedback
- ~~`infer_batch_size()`, `forward_batch`/`reshape_batch`~~ → absorbed into PR1

## Phase 3 — Feature Extension (not started)

After Phase 2 stable. All work plugs in via registry:
- MapAnything full integration
- Meshing improvements
- Visualization Panel pane
- Splat training integration

Source: `tlb-improve-mesh` branch (WIP MapAnything + feedforward meshing)

---

## Session Log

### 2026-04-20
- Brainstormed PR2 approach
- Decided: parallel stacked PRs, PR2 branches from `refactor/core-modules`
- Decided: dashboard = integration test harness, MapAnything stub pattern
- Retired `PROGRESS.md` + `plans/2026-04-19-phase1-pr2-dashboard-complete.md` → this doc
- Decided: absorb `refactor/semantics` (batching work) into PR1 — two big PRs only
- Decided: `refactor/nerfstudio-submodule` safe to delete now (0 unique commits)
- Deleted `refactor/nerfstudio-submodule` (worktree + branch)
- Cherry-picked batching commits from `refactor/semantics` → 25 pass, 1 skip, 0 new failures
- Deleted `refactor/semantics` (fully absorbed)
- **Next:** Start PR1 remaining tasks (pointcloud skeleton → NerfstudioSfmCreator → MapAnythingCreator → registry), then create PR2 branch

---

## Deferred / Parked

| Item | Reason | Where |
|------|--------|-------|
| `Segmentor` shim in `utils/segmentation.py` | Out of scope PR1 | `collab_splats/utils/segmentation.py` |
| `refactor/semantics` branch | Absorbing into PR1 — see rebase task above | local branch |
| `tlb-grouping-segmentation`, `tlb-improve-splatter` | Separate plan TBD | local branches |
| `stash@{0}` | WIP meshing on main — review before Phase 3 | `git stash list` |
