# Feedforward Import Cleanup

**Date:** 2026-05-08  
**Branch:** refactor/core-modules

## Context

The feedforward refactor (`vggtx.py`, `mapanything.py`, `closure.py`) left imports scattered inside methods — some lazy for optional-dependency reasons, some try/except ImportError guards, some circular-import workarounds. The codebase assumes correct package setup, so all these guards are dead weight. This cleanup hoists all imports to file top, removes ImportError guards, and fixes stale test patch paths introduced during the refactor.

## What Does NOT Change

- `_preprocess`/`_forward`/`_postprocess`/`_load_model` stay underscored — these are Template Method hooks (abstract, called by base class, not external callers). Correct as-is.
- Nested functions inside `_patch_vggtx_compute_similarity` stay nested — they close over each other for monkey-patching. Not general utilities.

## Files Changed

### `collab_splats/pointcloud/feedforward/vggtx.py`

**Add to top-level imports:**
```python
import torch
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
```

**Remove:**
- `try/except ImportError` block at module top (`_HAS_VGGT` flag)
- `if not _HAS_VGGT: raise ImportError(...)` guard in `unproject_and_filter_points`
- `import numpy as np`, `from vggt.models.aggregator import Aggregator`, `from vggt.models.vggt import VGGT` at top of `_patch_vggtx_compute_similarity` (now at file top)
- `import torch` + VGGT imports + ImportError guards inside `_load_model`, `_preprocess`, `_forward`, `_verify_loop_candidate`
- `import collab_splats.pointcloud.feedforward as _ff` circular workaround in `_postprocess` and `_reproject_after_ba`
- `import torch as _torch` in `_postprocess`

**Change:**
- `_ff.unproject_and_filter_points(...)` → `unproject_and_filter_points(...)` (direct call, same file)
- `_torch.from_numpy(...)` / `_torch.stack(...)` → `torch.from_numpy(...)` / `torch.stack(...)`

---

### `collab_splats/pointcloud/feedforward/mapanything.py`

**Add to top-level imports:**
```python
import sys
import torch
import torch.nn.functional as F
import open3d as o3d
from PIL import Image as PILImage
import mapanything.utils.wai.intersection_check as ic
from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import load_images
```

**Remove:**
- `import mapanything.utils.wai.intersection_check as ic` and `import sys` inside `_patch_mapanything_torch_compat`
- All try/except ImportError guards (in `_load_model`, `_preprocess`, `collect_pts3d_from_outputs`, `_reproject_mapanything`)
- All lazy imports in `_load_model`, `_preprocess`, `run_mapanything`, `_postprocess`, `_verify_loop_candidate`, `collect_pts3d_from_outputs`, `_reproject_mapanything`

**Change:**
- `import torch as _torch` + `_torch.stack(...)` → `torch.stack(...)`

---

### `collab_splats/pointcloud/loop_closure/closure.py`

**Root cause of lazy imports:** `loop_closure/__init__.py` imports `closure.py`, so `from collab_splats.pointcloud.loop_closure import PoseGraph` inside `closure.py` is circular. Fix: use relative sibling imports instead.

**Add to top-level imports:**
```python
import torch
import pypose as pp
import pypose.optim as ppopt
from scipy.spatial.transform import Rotation as ScipyR
from .pose_graph import PoseGraph, Sim3PoseGraph
from .alignment import dedup_overlap, overlap_region_align_sim3
```

**Remove (all now at top):**
- 4× `from scipy.spatial.transform import Rotation as ScipyR` across `_se3_to_sim3_data`, `_sim3_data_from_sRt`, `_apply_sim3_nodes_to_frames`, `merge_submap_outputs`
- `import torch`, `import pypose as pp`, `import pypose.optim as ppopt` from `run_sim3_pose_graph_optimization`
- `from collab_splats.pointcloud.loop_closure import PoseGraph` from `build_pose_graph` (→ replaced by `from .pose_graph import PoseGraph` at top)
- `from collab_splats.pointcloud.loop_closure.alignment import dedup_overlap` from `run_pose_graph_optimization`
- `from collab_splats.pointcloud.loop_closure.alignment import dedup_overlap, overlap_region_align_sim3` and `from collab_splats.pointcloud.loop_closure.pose_graph import Sim3PoseGraph` from `run_sim3_pose_graph_optimization`

---

### `tests/pointcloud/test_vggtx_creator.py`

Fix stale patch paths (introduced when the circular `_ff` workaround was added, now direct call removes the indirection):

| Old (stale) | New (correct) |
|---|---|
| `"collab_splats.pointcloud._vggt.unproject_and_filter_points"` | `"collab_splats.pointcloud.feedforward.vggtx.unproject_and_filter_points"` |
| `"collab_splats.pointcloud.feedforward.run_global_alignment"` | `"collab_splats.pointcloud.feedforward.vggtx.run_global_alignment"` |

## Verification

```bash
# Run the mock-based tests (no GPU needed)
pytest tests/pointcloud/test_vggtx_creator.py -v -m "not gpu"
pytest tests/pointcloud/test_sim3_pose_graph.py -v
pytest tests/pointcloud/test_loop_closure_eval.py -v
```

GPU smoke test (`test_vggtx_reconstruct_smoke`) is excluded unless a GPU environment is available.
