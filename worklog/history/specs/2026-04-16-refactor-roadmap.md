---
title: collab-splats Refactor Roadmap
date: 2026-04-16
status: active
---

# collab-splats Refactor Roadmap

## Architectural Vision

The codebase is undergoing a systematic transformation from a flat-utility structure to a domain-submodule structure.

**From:**
```
collab_splats/utils/    ← catch-all for everything
stage/                  ← scripts that collab_splats/ imports from (architectural debt)
```

**To:**
```
collab_splats/
  semantics/            ← feature extractors, segmentation
  pointcloud/           ← SfM + feedforward reconstruction
  dashboard/            ← Panel-based UI
  camera/               ← camera math, NDC conversions, backprojection
  mesh/                 ← mesh exporters, repair, alignment
  preprocessing/        ← frame extraction, optical flow, video utils
  grouping/             ← Gaussian grouping and segmentation
  nerfstudio/           ← nerfstudio integration layer
  utils/                ← only truly generic helpers remain
stage/                  ← notebooks and exploratory scripts ONLY (no package imports)
```

**Established submodule pattern** (from `semantics/` and `pointcloud/`):
- `submodule/protocols.py` — Protocol/ABC interfaces
- `submodule/base.py` — dataclasses + base class + internal helpers
- `submodule/impl.py` — concrete implementations
- `submodule/__init__.py` — clean public `__all__`
- `tests/submodule/` — unit tests per submodule
- `collab_splats/utils/old_module.py` → shim re-exporting from new location until all callers migrated

---

## Active Branch Inventory

### Stacked PR Train (in-flight, merge in order)

The original combined working branch (`refactor/semantics-pointcloud`) has been split into a clean stacked PR sequence. Each branch is a slice of the full work; each merges to main independently, and the next rebases on top.

```
main
 └─ refactor/semantics          PR 1 — semantics submodule
     └─ refactor/pointcloud     PR 2 — pointcloud submodule (stacked on PR 1)
         └─ refactor/dashboard  PR 3 — dashboard Panel redesign (stacked on PR 2)
             └─ refactor/nerfstudio-submodule  PR 4 — nerfstudio/ submodule (stacked on PR 3)
```

| Branch | PR # | Owns | Status |
|---|---|---|---|
| `refactor/semantics` | 1 | `collab_splats/semantics/`, `utils/features.py` + `segmentation.py` shims | Ready to merge |
| `refactor/pointcloud` | 2 | `collab_splats/pointcloud/`, `utils/pointcloud.py` shim, `stage/feedforward.py`, `stage/mapanything_utils.py` | Ready to merge |
| `refactor/dashboard` | 3 | `collab_splats/dashboard/` (Panel redesign), `wrapper/config.py`, `wrapper/splatter.py`, `utils/mesh.py` minor fix, `utils/model_loading.py` | Ready to merge |
| `refactor/nerfstudio-submodule` | 4 | `collab_splats/nerfstudio/` — models, datamanagers, trainer_config, method_configs | In progress |

`refactor/semantics-pointcloud` — original combined working branch. Superseded by the stacked splits above. Close after all 4 PRs merge.

### Old WIP Branches (Superseded — Close After PR Train Merges)

| Branch | Reason to close |
|---|---|
| `tlb-improve-mesh` | All valuable content absorbed into stacked PR train. VGGT-X experimentation (branches commits: "worse alternative") not worth porting. `stage/optical_flow.py` + `stage/preproc_utils.py` will move to `preprocessing/` in Tier 2. |
| `refactor/pointcloud-submodule` | Identical history to `tlb-improve-mesh`. |
| `tlb-improve-splatter` | Fully merged into `tlb-improve-mesh`, itself superseded. |
| `tlb-semantics-refactor` | Superseded by `refactor/semantics`. |
| `tlb-repo-config` | Empty (0 commits vs main). |

**Before closing `tlb-improve-mesh`:** Cherry-pick `docs/splats/configs/` dataset YAML files into a clean PR on top of main.

### Zone C: `tlb-grouping-segmentation` (Stalled)

`datamanagers/grouping_datamanager.py` is 132 lines of entirely commented-out code. Needs design decision before proceeding (see Tier 3).

**Independent bugfix to cherry-pick regardless of grouping decision:** `datamanagers/features_datamanager.py` has a cache path normalization fix (Path vs str comparison on cache filenames) that is valid and should land in its own small PR after the main PR train merges.

### Inherited Architectural Debt (Priority Fix in Tier 2)

`collab_splats/pointcloud/feedforward.py` contains:
```python
from stage.feedforward import Reconstructor
from stage.mapanything_utils import build_colmap_reconstruction
```

An installed package must not import from a scripts directory. This is fixed by the `preprocessing/` submodule refactor.

---

## Refactor Queue

### Tier 0 — Stacked PR Train (in review / in progress)

| Refactor | Branch | Files | State |
|---|---|---|---|
| `semantics/` | `refactor/semantics` | `utils/features.py` + `segmentation.py` → `collab_splats/semantics/` with shims | ✅ Ready — PR 1 |
| `pointcloud/` | `refactor/pointcloud` | `stage/feedforward.py` + `mapanything_utils.py` → `collab_splats/pointcloud/` | ✅ Ready — PR 2 |
| `dashboard/` | `refactor/dashboard` | Gradio → Panel (`SemanticsDashboard`, `ConfigPanel`, `video_discovery`) | ✅ Ready — PR 3 |
| `nerfstudio/` | `refactor/nerfstudio-submodule` | models, datamanagers, trainer_config → `collab_splats/nerfstudio/`; update pyproject.toml entry points | 🔧 In progress — PR 4 |

---

### Tier 1 — Unblocked (no conflict with any active zone)

**Target files: `utils/camera_utils.py` (511 lines), `utils/visualization.py` (193 lines)**

#### `collab_splats/camera/` submodule

Move `utils/camera_utils.py` to `collab_splats/camera/` following the established pattern. Shim `utils/camera_utils.py` for backward compatibility. Absorb `utils/visualization.py` here (only 2 functions: `visualize_splat`, `create_camera_frustum_pyvista` — both camera-domain).

Proposed layout:
```
collab_splats/camera/
  __init__.py             # exports ColmapCamera, convert_to_colmap_camera, all math fns
  base.py                 # ColmapCamera dataclass + coordinate transforms
  rendering.py            # projection, NDC, backprojection, ray generation
  visualization.py        # visualize_splat, create_camera_frustum_pyvista (from utils/visualization.py)
tests/camera/
  __init__.py
  test_camera.py
```

Safe because: `camera_utils.py` and `visualization.py` are not touched by any active branch. `models/` (which imports `camera_utils`) is protected — shim handles it.

#### `utils/utils.py` audit

81 lines of generic helpers. Audit, move any domain-specific helpers to appropriate submodules, then either dissolve or keep as `utils/utils.py` for truly generic items.

---

### Tier 2 — After PR train (PRs 1–4) merges to main

#### `collab_splats/preprocessing/` submodule

**Motivation:** Fixes the architectural debt. After this refactor, `stage/` contains only notebooks — no Python modules that the package imports from.

Move:
- `stage/feedforward.py` → `collab_splats/preprocessing/feedforward.py` (`Reconstructor`)
- `stage/mapanything_utils.py` → `collab_splats/preprocessing/mapanything.py`
- `stage/optical_flow.py` → `collab_splats/preprocessing/optical_flow.py` (`OpticalFlowFrameSelector`)
- `stage/preproc_utils.py` → `collab_splats/preprocessing/video.py`

Update `collab_splats/pointcloud/feedforward.py`:
```python
# Before (debt):
from stage.feedforward import Reconstructor
# After (clean):
from collab_splats.preprocessing.feedforward import Reconstructor
```

Add shims in `stage/*.py` re-exporting from new locations to avoid breaking any user scripts.

#### `collab_splats/mesh/` submodule

`utils/mesh.py` is 1764 lines with a clear `GSMeshExporter` class hierarchy (6 concrete exporters: `GaussiansToPoisson`, `DepthAndNormalMapsPoisson`, `LevelSetExtractor`, `MarchingCubesMesh`, `TSDFFusion`, `Open3DTSDFFusion`). Extract to:

```
collab_splats/mesh/
  __init__.py
  base.py           # GSMeshExporter ABC + PointcloudResult-equivalent MeshResult
  exporters.py      # all 6 concrete exporter classes
  repair.py         # clean_repair_mesh, align_geometry_floor, get_floor_plane
  utils.py          # pick_indices_at_random, find_depth_edges, normals2vertex, features2vertex, mesh_clustering
  registry.py       # registry pattern (same as pointcloud/registry.py)
tests/mesh/
  __init__.py
  test_base.py
  test_exporters.py
```

Add `Splatter.mesh()` → select exporter via registry instead of current hardcoded logic.

#### `wrapper/` audit

`wrapper/splatter.py` is growing (~700 lines). After Tier 2 submodules exist, audit whether `Splatter` needs decomposition. Current methods span: preprocess, train, mesh, query, visualize, load_model, inspect_data — these may warrant a pipeline/stage abstraction.

---

### Tier 3 — After Tier 2 + Zone C design decision

#### `collab_splats/grouping/` submodule

**Blocked by design decision:** `tlb-grouping-segmentation` is stalled with `grouping_datamanager.py` entirely commented out. Two competing approaches in that branch's history:
1. **Original:** SAM segmentation → 3D Gaussian lift (`GroupingClassifier`)
2. **Newer direction:** Feature-based clustering / "semantic dictionary approach"

Before designing this submodule, decide which approach to pursue. Once decided, extract `utils/grouping.py` (`GroupingClassifier`, 490 lines) + new `grouping_datamanager.py` into `collab_splats/grouping/`.

---

### (no Tier 4 — `nerfstudio/` moved to Tier 0 PR 4)

---

## Merge Order / Sequencing

```
PR 1: refactor/semantics           →  main
PR 2: refactor/pointcloud          →  main  (rebase onto main after PR 1)
PR 3: refactor/dashboard           →  main  (rebase after PR 2)
PR 4: refactor/nerfstudio-submodule→  main  (rebase after PR 3)

[close tlb-improve-mesh, refactor/pointcloud-submodule,
 tlb-improve-splatter, tlb-semantics-refactor, tlb-repo-config,
 refactor/semantics-pointcloud]

[cherry-pick docs/splats/configs/ dataset YAMLs → clean PR]
[cherry-pick features_datamanager.py cache path fix from tlb-grouping-segmentation → clean PR]

Tier 1 (parallel — no file overlap with PR train):
  camera/ submodule branch     →  main

Tier 2 (after PR train fully merged):
  preprocessing/ submodule     →  main   (fixes stage import debt)
  mesh/ submodule              →  main
  wrapper/ audit               →  main

Tier 3 (after Tier 2 + grouping design decision):
  grouping/ submodule          →  main
```

`camera/` branch has zero file overlap with the PR train — can be opened and merged in parallel.

---

## Files That Remain in `utils/` Long-Term

| File | Lines | Disposition |
|---|---|---|
| `utils/trainer_config.py` | 40 | → `nerfstudio/` in Tier 0 PR 4 (shim remains) |
| `utils/model_loading.py` | 35 | → `nerfstudio/` in Tier 0 PR 4 (shim remains) |
| `utils/utils.py` | 81 | Audit in Tier 1; dissolve or keep for generic helpers |
| `utils/camera_utils.py` | 511 | → `camera/` in Tier 1 (shim remains) |
| `utils/visualization.py` | 193 | → `camera/visualization.py` in Tier 1 (shim remains) |
| `utils/mesh.py` | 1764 | → `mesh/` in Tier 2 (shim remains) |
| `utils/grouping.py` | 490 | → `grouping/` in Tier 3 (shim remains) |
| `utils/features.py` | 113 | shim → `semantics/` — Tier 0 PR 1 |
| `utils/segmentation.py` | 32 | shim → `semantics/` — Tier 0 PR 1 |
| `utils/pointcloud.py` | 6 | shim → `pointcloud/` — Tier 0 PR 2 |
