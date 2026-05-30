# Repo Cleanup Design

**Date:** 2026-05-29  
**Status:** approved

## Goal

Three coordinated cleanup tasks: consolidate split spec/plan locations, produce public README + enriched agent context, and prune stale notebook structure.

---

## Part 1 — Consolidate specs/plans under `docs/superpowers/`

### Problem

Specs and plans live in two places:
- `worklog/specs/` — 10 files (older, manually created)
- `docs/superpowers/specs/` — 35+ files (brainstorming skill default)
- `worklog/plans/` — 18 files
- `docs/superpowers/plans/` — 40+ files

### Solution

Move `worklog/specs/*` and `worklog/plans/*` into `docs/superpowers/specs/` and `docs/superpowers/plans/` respectively. `docs/superpowers/` becomes the single canonical location for all AI-generated planning artifacts.

`worklog/` retains only human-readable tracking files:
- `STATE.md`, `WORKLOG.md`, `ROADMAP.md`, `README.md`
- `decisions/` (ADRs 001–013)
- `notes/`, `history/`, `known-test-failures.md`

### CLAUDE.md changes

- Remove references to `worklog/specs/` and `worklog/plans/` as spec/plan locations
- Add: "Active spec/plan for in-flight work lives in `docs/superpowers/specs/` and `docs/superpowers/plans/`"
- Keep: "Before anything, read `worklog/STATE.md`, then `worklog/WORKLOG.md`, then `worklog/decisions/`"

### Cross-reference fixes

`worklog/STATE.md` In-Flight Work section links to `specs/...` and `plans/...` (relative paths). After move, update these links to repo-root-relative paths: `docs/superpowers/specs/...` and `docs/superpowers/plans/...`.

Similarly update any links in `worklog/WORKLOG.md` and `worklog/ROADMAP.md`.

---

## Part 2 — Public README + agent context

### Public `README.md` (repo root)

Scope: what it is, how to install, where to start. Keep short — this is a research codebase, not a product.

Sections:
1. **What it does** — one-paragraph summary: video/images → pointcloud (VGGT-X or MapAnything) → optional BA → optional loop closure → mesh + semantic features
2. **Key capabilities** — bullet list: feedforward reconstruction, bundle adjustment, SL(4) loop closure, semantic feature lifting, TSDF/Poisson meshing, camera localization
3. **Install** — two commands: `bash setup.sh` then `bash setup_feedforward.sh`; note Python env (`/opt/conda/envs/reconstruction/bin/python`)
4. **Getting started** — link to `docs/source/tutorials/` numbered sequence (01 preprocessing → 07 localization)
5. **Evaluation** — one line pointing to `evals/eval_gt.py` and `docs/` for visualization notebooks

Do NOT include: architecture diagrams, module internals, worklog contents, or anything that belongs in tutorials.

### Agent context additions to `CLAUDE.md`

Add a section covering things not derivable from code:

**In-flight work** (agent must not assume complete):
- `gt-eval-harness` — ground-truth ATE evaluation harness
- `feedforward-import-cleanup` — import hygiene pass
- `feedforward-mesh` — feedforward → TSDF mesh pipeline
- `docs-site` — Sphinx site setup
- `bae-vggt-parity` — verify BA output matches upstream `zitongzhan/vggt --implementation bae`

**Module interaction chain:**
```
video/images
  → frame_sampling (keyframe selection)
  → feedforward/ (VGGTXCreator | MapAnythingCreator) → PointcloudResult
  → bundle_adjustment.py (optional LM refinement)
  → loop_closure/ (optional SL(4) pose graph)
  → mesh/ (TSDF | Poisson)
  → semantics/ (DINOv2 | SAM feature extraction → lift_features)
  → localization.py (SALAD retrieval → hloc matching)
```

**Spec/plan location:** `docs/superpowers/specs/` and `docs/superpowers/plans/`

**Known issues pointer:** `worklog/known-test-failures.md`

---

## Part 3 — Notebook audit

### Current live structure (`docs/source/tutorials/`)

All notebooks registered in `index.rst` are retained as-is:
- `01_preprocessing/keyframe_extraction.ipynb`
- `02_pointcloud/feedforward_methods.ipynb`, `bundle_adjustment.ipynb`, `slam_loop_closure.ipynb`, `feedforward_mesh.ipynb`, `colmap_sfm.ipynb`
- `03_splats/derive_splats.ipynb`, `visualization.ipynb`
- `04_semantics/feature_extraction.ipynb`, `segmentation.ipynb`, `maskclip_vs_talk2dino.ipynb`
- `05_lifting/semantic_lifting.ipynb`
- `06_mesh/create_mesh.ipynb`
- `07_localization/localization.ipynb`
- `evals/ground_truth_evals.ipynb`

### Delete

- `docs/source/tutorials/02_pointcloud/stage/` — draft copies superseded by live notebooks
- `docs/source/tutorials/06_mesh/stage/` — draft copies; `mesh_method_comparison.ipynb` here is not mature enough to promote (no index entry, incomplete)
- `docs/source/tutorials/06_mesh/feedforward_mesh.ipynb` — duplicate of `02_pointcloud/feedforward_mesh.ipynb`, not in index

### No gaps identified

Old `_build` structure notebooks are all covered by the new 01–07 numbering. No tutorial content is missing.

---

## Out of scope

- Content of existing notebooks (no rewrites)
- Adding new tutorials
- Changes to `evals/` scripts
- Sphinx config changes
