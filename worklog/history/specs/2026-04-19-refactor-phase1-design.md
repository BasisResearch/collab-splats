---
title: collab-splats Refactor Roadmap — Phases 1–3
date: 2026-04-19
status: approved
---

# collab-splats Refactor Roadmap

## Context

Current state: `refactor/dashboard-optical-flow` (HEAD) contains a stacked PR train of
structural refactoring work across semantics, pointcloud, dashboard, and nerfstudio. Several
branches are stale ancestors of HEAD; one (`dashboard`) has a small unreleased fix; one
(`refactor/nerfstudio-submodule`) has independent module-restructure work in a worktree.

Goal: land the refactor cleanly, then build Phase 2 (dashboard iteration) and Phase 3
(MapAnything, meshing, visualization) on top of stable module interfaces.

---

## Architecture Principle

All extension points follow a **registry/protocol pattern** established in Phase 1:

- New extractor → implement `BaseFeatureExtractor` protocol → register → appears in dashboard automatically
- New pointcloud creator → implement `BasePointcloudCreator` → register → wired via `Splatter.pointcloud_method`
- New dashboard component → add Panel pane → rest of layout untouched

`protocols.py` is the contract. Once PR 1 lands it is frozen unless a breaking change is
explicitly versioned.

---

## Branch Cleanup (before PR work)

| Branch | Action |
|--------|--------|
| `refactor/pointcloud` | Delete — fully absorbed by HEAD |
| `refactor/semantics` | Delete — fully absorbed by HEAD |
| `refactor/dashboard` | Delete — fully absorbed by HEAD |
| `dashboard` | Cherry-pick cb22e58 (CUDA auto-detect) into PR 2, then delete |
| `refactor/nerfstudio-submodule` | Rebase onto HEAD, fold into PR 1 |
| `tlb-grouping-segmentation` | Archive/delete — stale, behind main |
| `tlb-improve-splatter` | Archive/delete — stale |
| `tlb-repo-config` | Delete — merged into main |
| `tlb-improve-mesh` | Keep — raw material for Phase 3 (has WIP MapAnything + feedforward meshing) |
| stash@{0}, stash@{1} | Drop both — superseded by HEAD |

---

## Phase 1 — Structural Refactor (no behavior changes)

### PR 1 — Core Modules

**Branch:** `refactor/core-modules` (squash from current HEAD minus dashboard commits)

**Scope:**

```
collab_splats/
  semantics/         features, segmentation, protocols, frame_sampling
  pointcloud/        base, feedforward, sfm, utils, registry
  nerfstudio/        models/, configs/, datamanagers/ — moved in, entry points updated
  utils/             features.py, segmentation.py, pointcloud.py — backward-compat shims only
  wrapper/           config.py (ConfigLoader), splatter.py (pointcloud_method wiring)
```

**Bug fixes included:**
1. `semantics/features.py` — guard `import maskclip_onnx` behind try/except with pip hint
2. `semantics/features.py` `pytorch_gc()` — guard `torch.cuda.*` calls with `if torch.cuda.is_available()`

**Explicitly excluded:**
- No changes to `rade_gs_model.py` or `rade_features_model.py` internals
- No `infer_batch_size` / VRAM-aware batching (Phase 2)
- No new extractors or creators (Phase 3)

**Tests:** existing `tests/semantics/`, `tests/pointcloud/` pass unchanged.

---

### PR 2 — Dashboard Refactor

**Branch:** `refactor/dashboard-complete` (squash dashboard commits from HEAD + cherry-pick cb22e58)

**Scope:**

```
collab_splats/
  dashboard/         semantics.py (Panel), config_panel.py, video_discovery.py, __main__.py
```

**Includes:**
- Panel `SemanticsDashboard` replacing Gradio
- `ConfigPanel` for per-video YAML editing
- `video_discovery` for fieldwork filesystem scanning
- Optical flow frame sampling wired into dashboard UI (flow_threshold slider)
- CUDA auto-detect: `_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — defaults all 3 device dropdowns

**Depends on:** PR 1 merged (imports from `collab_splats.semantics.*`)

**Tests:** existing `tests/dashboard/` pass unchanged.

---

## Phase 2 — Dashboard Iteration

**Starts after:** both Phase 1 PRs merged and dashboard tested in the field.

**Scope (not yet fully designed):**
- `infer_batch_size()` VRAM-aware batch size utility wired into extraction pipeline
- `forward_batch` / `reshape_batch` batching for all extractors
- UX iteration based on field testing
- Any dashboard layout rework

**Constraint:** no changes to `protocols.py` interfaces without explicit versioning.

---

## Phase 3 — Feature Extension

**Starts after:** Phase 2 stable.

**Extension points (all via registry, no existing code touched):**

| Work item | Where it lands | Source material |
|-----------|---------------|-----------------|
| MapAnything full integration | `semantics/features.py` (extractor) + `pointcloud/feedforward.py` (creator) | `tlb-improve-mesh` WIP |
| Meshing improvements | `pointcloud/` new creator | `tlb-improve-mesh` WIP |
| Visualization | `dashboard/` new Panel pane | new work |
| Splat training integration | `nerfstudio/` models | new work |

**Dashboard wiring:** extractor dropdown and pointcloud method selector read from registries —
new Phase 3 classes appear automatically without dashboard changes.

---

## Key Risks

| Risk | Mitigation |
|------|-----------|
| `protocols.py` needs to change post-PR1 | Lock signature in PR 1 review; version if change needed |
| `refactor/nerfstudio-submodule` rebase conflicts | Rebase before PR 1 branch cut |
| `tlb-improve-mesh` divergence grows | Don't merge yet — use as reference only until Phase 3 |
| `maskclip_onnx` hard dep at import time | Fix in PR 1 (guard import) |
