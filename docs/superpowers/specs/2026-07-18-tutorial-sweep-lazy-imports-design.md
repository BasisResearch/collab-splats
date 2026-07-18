# Tutorial Notebook Sweep + Library Lazy-Import Pass — Design

**Date:** 2026-07-18
**Status:** Approved — REVISED 2026-07-18: Phase 1 (library lazy-import pass) DEFERRED
by user decision; scope is notebook simplification only. Import-time cost stays
until a future pass picks Phase 1 up; the design below is kept as its reference.
**Branch:** refactor/cu121-uv-migration
**Predecessor:** 2026-07-16-tutorial-data-format-keyframe-design.md (§Deferred is this spec's backlog)

## Context

The keyframe rework (2026-07-17) landed the new tutorial data layout (`FRAMES`,
`TUTORIAL_CACHE`, `DATASET = "2024_02_06/C0043"`) and reworked tutorial 01, but
deferred the rest of the notebook suite. Current state:

- 6 notebooks broken by the `IMAGES` → `FRAMES` rename:
  `02_pointcloud/{feedforward_methods,slam_loop_closure,bundle_adjustment}`,
  `04_semantics/{segmentation,maskclip_vs_talk2dino,feature_extraction}`.
- `03_splats`/`06_mesh` still override `BASE_DIR` to `/workspace/fieldwork-data/`.
- `07_localization` predates the layout and showcases XFeat, while the dashboard
  default is now `loma` (commit 6b507f8).
- First import of the reconstruction stack in a notebook takes ~55 s. Measured
  with `-X importtime`: `collab_splats` top-level is already lazy (~11 ms), but
  `from collab_splats.pointcloud.feedforward import VGGTXCreator` alone is
  ~14.9 s — `pointcloud.utils` eagerly imports `semantics.features` (mobile_sam
  → timm, ~3.0 s), `geometry.bundle_adjustment` adds ~4.9 s, `vggt.layers.mlp`
  ~3.6 s self-time, and `pointcloud/__init__` eagerly imports every backend
  (sfm/hloc, vggtx, mapanything, try-omega, try-spark). torch/open3d/pyvista
  make up the rest and are unavoidable for inference notebooks.

## Decisions

1. **One spec, three-phase plan.** Library lazy-imports first (notebook
   re-execution then happens once, with final import behavior), notebook sweep
   second, data-migration tail third.
2. **VGGT-Omega is the only backend that executes** in `feedforward_methods`.
   VGGT-X, MapAnything, and VGGT-SPARK are demoted to a markdown section:
   availability, one-line swap via `make_creator("<name>")`, API-docs link.
   Downstream notebooks read the omega zarr cache only.
3. **LoMa is the primary local matcher** in `07_localization` (matches dashboard
   default). XFeat/DISK become markdown alternatives.
4. **Library-level lazy-import pass** (not notebook-level workarounds): PEP 562
   `__getattr__` in `pointcloud/__init__`, deferred `semantics` import inside
   `lift_features`, deferred BA import at call site. Benefits dashboard,
   tutorials, and tests alike.
5. Full deferred-sweep scope from the predecessor spec is in scope, including
   `03_splats`/`06_mesh` migration, evals results path, and `CACHE_DIR`
   retirement.

## Design

### Phase 1 — Library lazy-import pass

**`collab_splats/pointcloud/__init__.py`**

- Replace eager creator imports with PEP 562 `__getattr__` (same pattern as
  `collab_splats/dashboard/__init__.py`).
- `_REGISTRY` becomes `name -> (module_path, class_name)` string pairs;
  `get_creator` imports the module on demand. Omega/SPARK availability is
  probed at resolve time: `get_creator("vggt_omega")` raises a clear
  `ImportError` mentioning `setup/feedforward.sh` if the submodule is absent.
- `make_creator` unchanged in signature and behavior.
- Test-patchable symbols (`unproject_and_filter_points` re-export etc.) keep
  resolving through `__getattr__`; any test that patches
  `collab_splats.pointcloud.X` must still find `X` after one attribute access.

**`collab_splats/pointcloud/utils.py`**

- `semantics.features` import moves inside `lift_features` (the only consumer).
  This is the documented optional-heavy-dep exception to the imports-at-top
  rule; add the clear-`ImportError` guard. Removes the mobile_sam/timm ~3 s tax
  from every pointcloud import.

**`collab_splats/pointcloud/feedforward/base.py`** (and `pointcloud/base.py` if
implicated)

- `geometry.bundle_adjustment` import deferred to the method that runs BA
  (~4.9 s saved on plain feedforward imports).

**Non-goals:** `vggt.layers.mlp` self-time (external package) is deferred, not
removed. torch import cost stays.

**Measurement:** `-X importtime` before/after for
`import collab_splats.pointcloud` and
`from collab_splats.pointcloud.feedforward import VGGTOmegaCreator`; numbers
recorded in the plan's completion notes. Target: notebook first-import
~55 s → ~25–30 s.

### Phase 2 — Notebook sweep (pipeline order, each executed vs C0043)

Common changes for every touched notebook: `%run ../tutorial_config.py` layout
(`FRAMES` read-only input, `TUTORIAL_CACHE` for all notebook writes, `CACHE_DIR`
usages folded into `OUTPUT_DIR`), imports at top, block comments per code style.

1. **`02_pointcloud/feedforward_methods.ipynb`** — omega-only execution:
   load frames from `FRAMES`, run `VGGTOmegaCreator`, save zarr to the canonical
   cache location downstream notebooks read. Markdown section for the other
   backends (decision 2). Prerequisite text updated (`frames/`, not `images/`).
   *Caution:* this file and `feedforward_mesh.ipynb` carry uncommitted edits
   from the in-flight feedforward-mesh work — reconcile the working-tree state
   before rewriting (shared-branch concurrent-session history; see memory note
   on 572e970).
2. **`02_pointcloud/bundle_adjustment.ipynb`**, **`slam_loop_closure.ipynb`** —
   switch primary backbone to omega (LC calibration: `target_layer=13`,
   `threshold=1.55`); other backbones mentioned in markdown.
3. **`04_semantics/{segmentation,maskclip_vs_talk2dino,feature_extraction}.ipynb`**
   — `FRAMES` for images, omega zarr where a reconstruction is needed.
4. **`05_lifting/semantic_lifting.ipynb`** — omega cache + layout alignment.
5. **`07_localization/localization.ipynb`** — rework: load omega zarr via
   tutorial_config paths; LoMa primary matcher end-to-end (retrieval →
   LoMa match → PnP+RANSAC → 3D view); XFeat/DISK as markdown alternatives
   with the one-line extractor swap.
6. **`02_pointcloud/colmap_sfm.ipynb`**, **`02_pointcloud/feedforward_mesh.ipynb`**
   — layout alignment only (no method changes; feedforward_mesh is owned by the
   in-flight feedforward-mesh effort — touch paths only, not structure).

Execution: headless via nbconvert in tmux (46.6 GB cgroup cap; no parallel
heavy runs), `MAX_FRAMES = 30`. A notebook is done only when it executes
end-to-end against C0043.

### Phase 3 — Data-migration tail

- **`03_splats/{derive_splats,visualization}.ipynb`**, **`06_mesh/create_mesh.ipynb`**:
  migrate `BASE_DIR` override from `/workspace/fieldwork-data/` to the standard
  layout + `2024_02_06/C0043`. Gate: verify splat training data exists for
  C0043 first; if absent, keep the override with a dated TODO markdown note and
  record the blocker in the plan.
- **`evals/ground_truth_evals.ipynb`**: replace hardcoded results path with a
  configurable one relative to `evals/results/`.
- **`tutorial_config.py`**: retire `CACHE_DIR` alias once no notebook reads it;
  drop the dead first candidate in `_infer_video_path` (rclone-relative
  `video_ref` never exists as an absolute path).

## Error handling

- `get_creator` on a missing optional backend → `ImportError` naming the setup
  script, not a bare `KeyError`.
- Notebook cells that load the omega zarr fail with a clear message pointing at
  `feedforward_methods.ipynb` as the producer (existing pattern, kept).

## Verification

- Full pytest suite green (`/opt/venv/reconstruction/bin/python -m pytest tests/`),
  attention on tests patching `collab_splats.pointcloud.*` symbols.
- Dashboard smoke gate after Phase 1 (`python -m collab_splats.dashboard --smoke`
  must print SMOKE PASS) — dashboard imports `get_creator`/`make_creator`.
- `-X importtime` before/after numbers recorded (Phase 1).
- Every swept notebook executed end-to-end vs C0043 in tmux.
- `black . && isort .` on touched modules.
