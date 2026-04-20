# collab-splats Refactor Progress

Quick-reference status tracker. Full details in `docs/superpowers/`.

---

## Phase 1 — Structural Refactor

### PR 1 — Core Modules (`refactor/core-modules`) ⏳ awaiting merge

**Contains:**
- `collab_splats/semantics/` — features, segmentation, protocols, frame_sampling
- `collab_splats/pointcloud/` — base, feedforward, sfm, utils, registry
- `collab_splats/nerfstudio/` — models, method_configs, datamanagers (moved from top-level; entry points updated)
- `collab_splats/utils/` — backward-compat shims only
- `collab_splats/wrapper/` — config.py (ConfigLoader), splatter.py (pointcloud_method wiring)
- Bug fix: `maskclip_onnx` bare import guard
- Bug fix: `pytorch_gc()` CPU crash guard

**Tests:** 23 pass, 2 skip (mapanything + sfm require optional deps)

**Plan:** `docs/superpowers/plans/2026-04-19-phase1-pr1-core-modules.md`

---

### PR 2 — Dashboard (`refactor/dashboard-complete`) 🔒 blocked on PR 1

**Contains:**
- `collab_splats/dashboard/` — Panel SemanticsDashboard, ConfigPanel, video_discovery
- Optical flow frame sampling wired into UI
- CUDA auto-detect for all device dropdowns

**Plan:** `docs/superpowers/plans/2026-04-19-phase1-pr2-dashboard-complete.md`

---

## Phase 2 — Dashboard Iteration (not started)

After Phase 1 merged + field-tested. Key items:
- `infer_batch_size()` VRAM-aware batching wired in
- `forward_batch` / `reshape_batch` for all extractors (MaskCLIP, Talk2DINO)
- UX iteration from field feedback

---

## Phase 3 — Feature Extension (not started)

After Phase 2 stable. All work plugs in via registry — no existing code touched:
- MapAnything full integration (`semantics/` extractor + `pointcloud/` creator)
- Meshing improvements
- Visualization Panel pane
- Splat training integration (`nerfstudio/` models)

Source material: `tlb-improve-mesh` branch (WIP MapAnything + feedforward meshing)

---

## Reference

| Doc | Purpose |
|-----|---------|
| `docs/superpowers/specs/2026-04-19-refactor-phase1-design.md` | Full roadmap, branch cleanup table, architecture principles |
| `docs/superpowers/plans/2026-04-19-phase1-pr1-core-modules.md` | PR 1 step-by-step implementation plan |
| `docs/superpowers/plans/2026-04-19-phase1-pr2-dashboard-complete.md` | PR 2 step-by-step implementation plan |
| `REFACTOR.md` | Detailed module-level refactor notes |

## Deferred / Parked

| Item | Reason | Where |
|------|--------|-------|
| `Segmentor` shim in `utils/segmentation.py` | Out of scope for PR 1 — port later | `collab_splats/utils/segmentation.py` |
| `refactor/semantics` branch (9 commits) | Phase 2 batching work — keep branch | `refactor/semantics` local branch |
| `tlb-grouping-segmentation`, `tlb-improve-splatter` | Separate plan TBD | local branches |
| stash@{0} | WIP meshing on main — review before Phase 3 | `git stash list` |
