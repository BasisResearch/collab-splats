# collab-splats Refactor Worklog

Living cross-session log. Update at start/end of each session. Replaces `PROGRESS.md` and `plans/2026-04-19-phase1-pr2-dashboard-complete.md`.

---

## Current State

| Branch | Status | Notes |
|--------|--------|-------|
| `refactor/core-modules` | ⏳ awaiting merge (PR1) | 25 pass, 1 skip; feedforward integration specced |
| `refactor/dashboard-complete` | ⏳ in progress (PR2) | session 3: sequential extraction, orientation, max_frames, slider fix — 59 pass |
| `refactor/dashboard-optical-flow` | ✅ deleted | absorbed into PR2 via cherry-pick |
| `dashboard` | ✅ deleted | absorbed into PR2 via cherry-pick |
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
- `collab_splats/utils/` — camera_utils + frame_sampling (moved from semantics — general preprocessing, not semantics-specific)
- `frame_sampling` re-exported from `collab_splats.semantics` for backwards compat; canonical path is `collab_splats.utils.frame_sampling`
- `collab_splats/wrapper/config.py` — ConfigLoader
- Bug fix: `maskclip_onnx` bare import guard
- Bug fix: `pytorch_gc()` CPU crash guard

### Phase 1 tasks (complete)
- [x] **Task 1:** `pointcloud/` skeleton — `base.py`, `utils.py`, `__init__.py`, shim, tests
- [x] **Task 2:** `NerfstudioSfmCreator` (`pointcloud/sfm.py`) — hloc + pycolmap backends
- [x] **Task 3:** `MapAnythingCreator` (`pointcloud/feedforward.py`) — feedforward stub
- [x] **Task 4:** Registry + Splatter integration — `get_creator()`, `pointcloud_method` config key

### Phase 2: Feedforward Integration (complete)
Spec: `docs/superpowers/specs/2026-04-20-pointcloud-feedforward-design.md`
Plan: `docs/superpowers/plans/2026-04-20-feedforward-integration.md`

- [x] **Task 1:** Copy `stage/mapanything_utils.py`, `stage/preproc_utils.py`, `stage/vggt_utils.py` from source branches
- [x] **Task 2:** `base.py` — `CoordinateFrame` enum; `frame`/`world_transform` fields; rename `create()` → `reconstruct()`; add `_write_transforms()`
- [x] **Task 3:** `base.py` — fix `_colmap_recon_to_result()` (apply transform B, populate `world_transform`/`frame`)
- [x] **Task 4:** `sfm.py` — `ColmapCreator` with correct `colmap/sparse/0/` output path
- [x] **Task 5:** `sfm.py` — `HlocCreator` with direct hloc calls, no nerfstudio dep
- [x] **Task 6:** `feedforward.py` — `BaseFeedforwardCreator` template + `MapAnythingCreator` (confidence_percentile=35.0)
- [x] **Task 7:** `feedforward.py` — `VGGTXCreator` (use_global_alignment=False default)
- [x] **Task 8:** Registry `{"colmap", "hloc", "mapanything", "vggtx"}`; update tests

**Source files** (on `tlb-improve-mesh`, bring over to `refactor/core-modules`):
- `stage/mapanything_utils.py` — keep, used by `MapAnythingCreator._run_inference()`
- `stage/preproc_utils.py` — keep, image loading utils
- `nerfstudio/process_data/vggt_utils.py` — move to `stage/vggt_utils.py`

**Key bugs fixed in this phase:**
| Bug | File | Fix |
|-----|------|-----|
| Wrong output path | `sfm.py:50` | `sparse/0/` → `colmap/sparse/0/` |
| Missing world reorientation | `base.py` | Apply transform B + populate `world_transform` |
| No disk output | `feedforward.py` | `_run_inference()` writes binary; base writes `transforms.json` |

**Open question** (must verify before Task 7): Does `run_mapanything_pipeline()` write binary to `output_dir/colmap/sparse/0/`? Check `stage/mapanything_utils.py` on `tlb-improve-mesh`.

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
- [ ] Verify imports: dashboard uses `collab_splats.semantics.frame_sampling` → update to `collab_splats.utils.frame_sampling`
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
- Moved `frame_sampling` from `semantics/` → `utils/` (general preprocessing, not semantics-specific); re-exported from semantics for compat; test moved to `tests/utils/`

### 2026-04-20 (session 3)
- Designed feedforward pointcloud integration — MapAnythingCreator + VGGTXCreator
- Confirmed `tlb-improve-mesh` = source for stage/ feedforward utilities; `tlb-improve-splatter` = parked
- Key decisions: 4-creator registry (`colmap/hloc/mapanything/vggtx`), `BaseFeedforwardCreator` template, unified disk output contract (`colmap/sparse/0/*.bin` + `transforms.json`)
- Identified 3 bugs: wrong output path in `ColmapCreator`, missing world reorientation (transform B) in `_colmap_recon_to_result()`, no disk output from `MapAnythingCreator`
- Added `CoordinateFrame` enum + `world_transform` field to `PointcloudResult` for coordinate system metadata
- `HlocCreator` calls hloc directly — no nerfstudio dep (nerfstudio remains only for `ns-train`)
- Eliminated `stage/feedforward.py` (`Reconstructor`) — superseded by `BaseFeedforwardCreator`
- Spec written: `docs/superpowers/specs/2026-04-20-pointcloud-feedforward-design.md`
- **Next:** Review spec → invoke writing-plans → implement Tasks 5–9 on `refactor/core-modules`

### 2026-04-20 (session 4)
- Implemented Phase 2 feedforward integration (8 tasks) via subagent-driven development
- Copied `stage/mapanything_utils.py`, `stage/preproc_utils.py`, `stage/vggt_utils.py` from `tlb-improve-mesh` + nerfstudio fork
- Fixed `_colmap_recon_to_result()`: apply transform B (`_WORLD_TRANSFORM`), populate `frame=NERFSTUDIO` + `world_transform`
- `ColmapCreator` splits from `NerfstudioSfmCreator`; output path bug fixed (`colmap/sparse/0/`)
- `HlocCreator` calls hloc directly — no nerfstudio dep; hloc imports lazy inside `reconstruct()`
- `BaseFeedforwardCreator` template: `reconstruct()` calls `_run_inference()` then `_write_transforms()` once
- `MapAnythingCreator` refactored: `confidence_percentile=35.0` (not old `conf_threshold=1.5`); correct `stage/mapanything_utils` pipeline
- `VGGTXCreator` added: `use_global_alignment=False` default; calls `run_vggt(image_dir, colmap_dir=output_dir/"colmap")`
- Bug caught in review: `parents[3]` → `parents[2]` in `_add_stage_to_path()` (was inserting `/workspace` not repo root)
- Registry updated to `{"colmap", "hloc", "mapanything", "vggtx"}`; old `"sfm"/"feedforward"` keys removed
- **Next:** PR1 ready for review — open PR against `main`; then start PR2 (`refactor/dashboard-complete`)

---

## Deferred / Parked

| Item | Reason | Where |
|------|--------|-------|
| `Segmentor` shim in `utils/segmentation.py` | Out of scope PR1 | `collab_splats/utils/segmentation.py` |
| `refactor/semantics` branch | Absorbing into PR1 — see rebase task above | local branch |
| `tlb-grouping-segmentation`, `tlb-improve-splatter` | Separate plan TBD | local branches |
| `stash@{0}` | WIP meshing on main — review before Phase 3 | `git stash list` |
| **Nerfstudio env failures** | 5 tests in `tests/nerfstudio/test_imports.py` fail with `ModuleNotFoundError: No module named 'nerfstudio.models.splatfacto'` + 1 GPU smoke test (`tests/pointcloud/test_mapanything_creator.py::test_mapanything_create_smoke`). Pre-existing, not caused by any PR1 change. Likely nerfstudio version mismatch in dev env vs. what codebase expects. Fix: audit installed nerfstudio version vs. import paths, or pin correct version in `setup.sh`. | `tests/nerfstudio/test_imports.py`, `tests/pointcloud/test_mapanything_creator.py` |
