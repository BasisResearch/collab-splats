# collab-splats Refactor Worklog

Living cross-session log. Update at start/end of each session. Replaces `PROGRESS.md` and `plans/2026-04-19-phase1-pr2-dashboard-complete.md`.

---

## Current State

| Branch | Status | Notes |
|--------|--------|-------|
| `refactor/core-modules` | ⏳ awaiting merge (PR1) | 23 pass, 2 skip |
| `refactor/dashboard-complete` | 🔒 not started (PR2) | blocked on design — see below |
| `refactor/dashboard-optical-flow` | 🗂 source only | snapshot into PR2, then delete |
| `dashboard` | 🗂 source only | cherry-pick `cb22e58` (CUDA auto-detect), then delete |
| `refactor/semantics` | 🧊 parked | Phase 2 batching work |
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

## Post-merge Cleanup (after both PRs merged to main)

- [ ] Delete `refactor/dashboard-optical-flow` (fully absorbed)
- [ ] Delete `dashboard` branch (CUDA fix absorbed)
- [ ] Delete `refactor/nerfstudio-submodule` worktree + branch (absorbed by PR1)
- [ ] Review `stash@{0}` — WIP meshing on main before Phase 3

---

## Phase 2 — Dashboard Iteration (not started)

After Phase 1 merged + field-tested:
- `infer_batch_size()` VRAM-aware batching wired in
- `forward_batch` / `reshape_batch` for all extractors (MaskCLIP, Talk2DINO)
- UX iteration from field feedback

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
- **Next:** Start PR1 Task 1 (pointcloud skeleton) or create PR2 branch + snapshot dashboard files

---

## Deferred / Parked

| Item | Reason | Where |
|------|--------|-------|
| `Segmentor` shim in `utils/segmentation.py` | Out of scope PR1 | `collab_splats/utils/segmentation.py` |
| `refactor/semantics` branch (9 commits) | Phase 2 batching work | local branch |
| `tlb-grouping-segmentation`, `tlb-improve-splatter` | Separate plan TBD | local branches |
| `stash@{0}` | WIP meshing on main — review before Phase 3 | `git stash list` |
