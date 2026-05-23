# Feedforward Import Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoist all lazy/scattered imports to file top in `vggtx.py`, `mapanything.py`, and `closure.py`; remove ImportError guards; fix circular import workaround; update stale test patch paths.

**Architecture:** Pure refactor — no behavior change. Each file gets its imports consolidated at the top. The circular `import collab_splats.pointcloud.feedforward as _ff` in `vggtx.py` is replaced by a direct call (same-file function). Intra-package lazy imports in `closure.py` become relative imports at the top (safe: sibling modules don't import `closure.py`).

**Tech Stack:** Python 3.10, pytest, vggt, mapanything, scipy, torch, pypose, open3d

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/feedforward/vggtx.py` | Hoist 7 imports, remove `_HAS_VGGT`, remove circular `_ff`, fix `_torch` alias |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Hoist 9 imports, remove all try/except ImportError guards |
| `collab_splats/pointcloud/loop_closure/closure.py` | Hoist 6 imports (use relative paths to avoid circular), remove 4× duplicated scipy import |
| `tests/pointcloud/test_vggtx_creator.py` | Fix 2 stale `patch()` targets that reference nonexistent module paths |

---

## Task 1: Clean up `closure.py`

Cleanest starting point — no test changes needed. Pure import hoist.

**Files:**
- Modify: `collab_splats/pointcloud/loop_closure/closure.py`

**Background:** `closure.py` lives inside `loop_closure/`, whose `__init__.py` imports `closure.py`. So `from collab_splats.pointcloud.loop_closure import PoseGraph` inside `closure.py` would be circular. The fix: use relative sibling imports (`from .pose_graph import PoseGraph`). Scipy/torch/pypose have no circularity risk.

- [ ] **Step 1: Add top-level imports to `closure.py`**

In `collab_splats/pointcloud/loop_closure/closure.py`, replace the current top-level block:

```python
# BEFORE (lines 1–10):
"""Loop closure orchestration utilities.

Houses pre-add gates, pose-graph build/dedup, and submap-output merging.
Pure numpy — no torch. (gtsam crosses through PoseGraph, imported lazily.)
"""
from __future__ import annotations

import numpy as np

from .submap import Submap
```

```python
# AFTER:
"""Loop closure orchestration utilities.

Houses pre-add gates, pose-graph build/dedup, and submap-output merging.
"""
from __future__ import annotations

import numpy as np
import torch
import pypose as pp
import pypose.optim as ppopt
from scipy.spatial.transform import Rotation as ScipyR

from .alignment import dedup_overlap, overlap_region_align_sim3
from .pose_graph import PoseGraph, Sim3PoseGraph
from .submap import Submap
```

- [ ] **Step 2: Remove lazy import from `build_pose_graph`**

```python
# BEFORE:
def build_pose_graph(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    overlap_frames: int,
) -> "PoseGraph":
    from collab_splats.pointcloud.loop_closure import PoseGraph
    pg = PoseGraph()
```

```python
# AFTER:
def build_pose_graph(
    submaps: list[Submap],
    lc_submaps: list[Submap],
    overlap_frames: int,
) -> "PoseGraph":
    pg = PoseGraph()
```

Also update the return type annotation (no longer needs quotes since `PoseGraph` is now imported at top):

```python
) -> PoseGraph:
```

- [ ] **Step 3: Remove lazy import from `run_pose_graph_optimization`**

```python
# BEFORE:
def run_pose_graph_optimization(...) -> np.ndarray:
    from collab_splats.pointcloud.loop_closure.alignment import dedup_overlap
    pg = build_pose_graph(submaps, lc_submaps, overlap_frames)
```

```python
# AFTER:
def run_pose_graph_optimization(...) -> np.ndarray:
    pg = build_pose_graph(submaps, lc_submaps, overlap_frames)
```

- [ ] **Step 4: Remove lazy scipy import from `merge_submap_outputs`**

```python
# BEFORE (inside the wp_chunks loop):
        if sim3_nodes is not None:
            from scipy.spatial.transform import Rotation as ScipyR
            node = sim3_nodes[i]
```

```python
# AFTER:
        if sim3_nodes is not None:
            node = sim3_nodes[i]
```

- [ ] **Step 5: Remove lazy scipy imports from helpers**

In `_se3_to_sim3_data`:
```python
# BEFORE:
def _se3_to_sim3_data(mat44: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as ScipyR
    R = mat44[:3, :3].astype(np.float64)
```

```python
# AFTER:
def _se3_to_sim3_data(mat44: np.ndarray) -> np.ndarray:
    R = mat44[:3, :3].astype(np.float64)
```

In `_sim3_data_from_sRt`:
```python
# BEFORE:
def _sim3_data_from_sRt(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as ScipyR
    q = ScipyR.from_matrix(R.astype(np.float64)).as_quat().astype(np.float32)
```

```python
# AFTER:
def _sim3_data_from_sRt(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    q = ScipyR.from_matrix(R.astype(np.float64)).as_quat().astype(np.float32)
```

In `_apply_sim3_nodes_to_frames`:
```python
# BEFORE:
def _apply_sim3_nodes_to_frames(...):
    from scipy.spatial.transform import Rotation as ScipyR
    result: dict[int, np.ndarray] = {}
```

```python
# AFTER:
def _apply_sim3_nodes_to_frames(...):
    result: dict[int, np.ndarray] = {}
```

- [ ] **Step 6: Remove lazy imports from `run_sim3_pose_graph_optimization`**

```python
# BEFORE:
def run_sim3_pose_graph_optimization(...):
    import torch
    import pypose as pp
    import pypose.optim as ppopt
    from collab_splats.pointcloud.loop_closure.alignment import (
        dedup_overlap,
        overlap_region_align_sim3,
    )
    from collab_splats.pointcloud.loop_closure.pose_graph import Sim3PoseGraph

    N = len(submaps)
```

```python
# AFTER:
def run_sim3_pose_graph_optimization(...):
    N = len(submaps)
```

- [ ] **Step 7: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_sim3_pose_graph.py tests/pointcloud/test_loop_closure_eval.py -v
```

Expected: all pass (no behavior change, only import locations changed).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/loop_closure/closure.py
git commit -m "refactor(closure): hoist all lazy imports to file top"
```

---

## Task 2: Clean up `mapanything.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`

- [ ] **Step 1: Add top-level imports**

Replace the current top-level block (after `from .base import ...`):

```python
# BEFORE (current top of file, after existing imports):
from ..utils import voxel_downsample
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, console
```

```python
# AFTER:
import sys

import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

import mapanything.utils.wai.intersection_check as ic
from mapanything.models import MapAnything
from mapanything.utils.geometry import closed_form_pose_inverse
from mapanything.utils.image import load_images

from ..utils import voxel_downsample
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, console
```

- [ ] **Step 2: Remove lazy imports from `_patch_mapanything_torch_compat`**

```python
# BEFORE (first ~3 lines inside the function body):
def _patch_mapanything_torch_compat() -> None:
    ...
    import mapanything.utils.wai.intersection_check as ic

    target = ic.frustum_intersection_check
```

```python
# AFTER:
def _patch_mapanything_torch_compat() -> None:
    ...
    target = ic.frustum_intersection_check
```

Also remove `import sys` near the bottom of that function:

```python
# BEFORE:
    import sys
    for mod in list(sys.modules.values()):
```

```python
# AFTER:
    for mod in list(sys.modules.values()):
```

- [ ] **Step 3: Remove try/except + lazy import from `_load_model`**

```python
# BEFORE:
    def _load_model(self, device: str) -> Any:
        try:
            from mapanything.models import MapAnything
        except ImportError as e:
            raise ImportError(
                "MapAnything required. "
                "pip install git+https://github.com/facebookresearch/map-anything.git"
            ) from e
        # In-tree workaround: see _patch_mapanything_torch_compat (torch<2.4).
        _patch_mapanything_torch_compat()
```

```python
# AFTER:
    def _load_model(self, device: str) -> Any:
        _patch_mapanything_torch_compat()
```

- [ ] **Step 4: Remove lazy imports from `_preprocess`**

```python
# BEFORE:
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        from PIL import Image as PILImage
        try:
            from mapanything.utils.image import load_images
        except ImportError as e:
            raise ImportError(
                "MapAnything required. "
                "pip install git+https://github.com/facebookresearch/map-anything.git"
            ) from e

        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
```

```python
# AFTER:
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
```

- [ ] **Step 5: Remove lazy import from `run_mapanything`**

```python
# BEFORE:
def run_mapanything(model, views, *, confidence_percentile, minibatch_size, use_multiview_confidence):
    import torch

    with torch.no_grad():
```

```python
# AFTER:
def run_mapanything(model, views, *, confidence_percentile, minibatch_size, use_multiview_confidence):
    with torch.no_grad():
```

- [ ] **Step 6: Remove lazy imports from `_postprocess`**

```python
# BEFORE:
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        import open3d as o3d

        pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(raw_outputs)

        model_h: int = self.views[0]["img"].shape[-2]
        model_w: int = self.views[0]["img"].shape[-1]

        # Populate BA fields for BundleAdjustment wrapper
        import torch as _torch
        _images = _torch.stack([p["img_no_norm"][0].cpu().permute(2, 0, 1) for p in raw_outputs])
        if raw_outputs[0].get("conf") is not None:
            conf_list = [p["conf"][0] for p in raw_outputs]
            _conf = _torch.stack([c[0] if c.ndim == 3 else c for c in conf_list])
```

```python
# AFTER:
    def _postprocess(self, raw_outputs: Any, **kwargs: Any) -> FeedforwardResult:
        pts3d, colors, extrinsics, intrinsics = collect_pts3d_from_outputs(raw_outputs)

        model_h: int = self.views[0]["img"].shape[-2]
        model_w: int = self.views[0]["img"].shape[-1]

        _images = torch.stack([p["img_no_norm"][0].cpu().permute(2, 0, 1) for p in raw_outputs])
        if raw_outputs[0].get("conf") is not None:
            conf_list = [p["conf"][0] for p in raw_outputs]
            _conf = torch.stack([c[0] if c.ndim == 3 else c for c in conf_list])
```

- [ ] **Step 7: Remove lazy imports from `_verify_loop_candidate`**

```python
# BEFORE:
    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        import torch
        import torch.nn.functional as F
        retrieval = getattr(self, "_lc_retrieval", None)
```

```python
# AFTER:
    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        retrieval = getattr(self, "_lc_retrieval", None)
```

- [ ] **Step 8: Remove try/except + lazy imports from `collect_pts3d_from_outputs`**

```python
# BEFORE:
def collect_pts3d_from_outputs(outputs):
    try:
        from mapanything.utils.geometry import closed_form_pose_inverse
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    all_points: list[np.ndarray] = []
```

```python
# AFTER:
def collect_pts3d_from_outputs(outputs):
    all_points: list[np.ndarray] = []
```

- [ ] **Step 9: Remove try/except + lazy imports from `_reproject_mapanything`**

```python
# BEFORE:
def _reproject_mapanything(raw_outputs, refined_extrinsics):
    try:
        from mapanything.utils.geometry import closed_form_pose_inverse
    except ImportError as e:
        raise ImportError(
            "MapAnything required. "
            "pip install git+https://github.com/facebookresearch/map-anything.git"
        ) from e

    all_pts: list[np.ndarray] = []
```

```python
# AFTER:
def _reproject_mapanything(raw_outputs, refined_extrinsics):
    all_pts: list[np.ndarray] = []
```

- [ ] **Step 10: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py
git commit -m "refactor(mapanything): hoist all lazy imports to file top"
```

---

## Task 3: Clean up `vggtx.py` + fix test patch paths

This task removes the `_HAS_VGGT` guard, the circular `_ff` workaround, and the `_torch` alias. The test patch paths must be updated in lockstep because removing the `_ff` circular import changes where `unproject_and_filter_points` is resolved.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Modify: `tests/pointcloud/test_vggtx_creator.py`

- [ ] **Step 1: Update test patch paths first**

In `tests/pointcloud/test_vggtx_creator.py`, fix both stale `patch()` targets:

```python
# BEFORE (test_vggtx_postprocess_calls_global_alignment, lines ~54-57):
    with patch("collab_splats.pointcloud.feedforward.run_global_alignment",
               return_value=(refined_ext, refined_int)) as mock_ga, \
         patch("collab_splats.pointcloud._vggt.unproject_and_filter_points",
               return_value=(pts, colors)):
```

```python
# AFTER:
    with patch("collab_splats.pointcloud.feedforward.vggtx.run_global_alignment",
               return_value=(refined_ext, refined_int)) as mock_ga, \
         patch("collab_splats.pointcloud.feedforward.vggtx.unproject_and_filter_points",
               return_value=(pts, colors)):
```

```python
# BEFORE (test_vggtx_no_global_alignment_when_disabled, lines ~76-79):
    with patch("collab_splats.pointcloud.feedforward.run_global_alignment") as mock_ga, \
         patch("collab_splats.pointcloud._vggt.unproject_and_filter_points",
               return_value=(pts, colors)):
```

```python
# AFTER:
    with patch("collab_splats.pointcloud.feedforward.vggtx.run_global_alignment") as mock_ga, \
         patch("collab_splats.pointcloud.feedforward.vggtx.unproject_and_filter_points",
               return_value=(pts, colors)):
```

- [ ] **Step 2: Add top-level imports to `vggtx.py`**

Replace the current top-level block. The file currently starts with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..postproc import run_global_alignment
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, _raw_to_world_points, console


# ── Constants ─────────────...

VGGTX_IMG_LOAD_RESOLUTION: int = 518


# ── VGGT-X compat patch ────...

def _patch_vggtx_compute_similarity() ...


# ── Inference utilities ─────────────────────────────────────────────────────────

try:
    import torch as _torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import randomly_limit_trues
    _HAS_VGGT = True
except ImportError:
    _HAS_VGGT = False
```

Replace only the `try/except` block (keep everything else) with:

```python
import torch
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
```

Add these after the existing `import numpy as np` line and before `from ..postproc import run_global_alignment`. Full new imports block:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_ratio
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from ..postproc import run_global_alignment
from .base import BaseFeedforwardCreator, FeedforwardResult, _extrinsics_3x4_to_4x4, _raw_to_world_points, console
```

- [ ] **Step 3: Remove `_HAS_VGGT` guard from `unproject_and_filter_points`**

```python
# BEFORE (first lines of unproject_and_filter_points body):
    if not _HAS_VGGT:
        raise ImportError(
            "VGGT-X not installed. "
            "pip install git+https://github.com/Linketic/VGGT-X.git"
        )

    points3d = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
```

```python
# AFTER:
    points3d = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
```

- [ ] **Step 4: Remove lazy imports from `_patch_vggtx_compute_similarity`**

```python
# BEFORE (first 3 lines of the function body):
def _patch_vggtx_compute_similarity() -> None:
    ...
    import numpy as np
    from vggt.models.aggregator import Aggregator
    from vggt.models.vggt import VGGT

    if getattr(Aggregator, "_vggtx_similarity_patched", False):
```

```python
# AFTER:
def _patch_vggtx_compute_similarity() -> None:
    ...
    if getattr(Aggregator, "_vggtx_similarity_patched", False):
```

(The `numpy`, `Aggregator`, and `VGGT` names are now available from the file-top imports.)

- [ ] **Step 5: Remove try/except + lazy imports from `_load_model`**

```python
# BEFORE:
    def _load_model(self, device: str) -> Any:
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as e:
            raise ImportError(
                "VGGT-X not installed. "
                "pip install git+https://github.com/Linketic/VGGT-X.git"
            ) from e

        dtype = (
```

```python
# AFTER:
    def _load_model(self, device: str) -> Any:
        dtype = (
```

- [ ] **Step 6: Remove try/except + lazy import from `_preprocess`**

```python
# BEFORE:
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        try:
            from vggt.utils.load_fn import load_and_preprocess_images_ratio
        except ImportError as e:
            raise ImportError(
                "VGGT-X not installed. "
                "pip install git+https://github.com/Linketic/VGGT-X.git"
            ) from e

        image_dir = Path(image_dir)
```

```python
# AFTER:
    def _preprocess(self, image_dir: Path) -> tuple[Any, list[Path], np.ndarray]:
        image_dir = Path(image_dir)
```

- [ ] **Step 7: Remove lazy imports from `_forward`**

```python
# BEFORE:
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = views
```

```python
# AFTER:
    def _forward(self, model: Any, views: Any, **kwargs: Any) -> dict:
        images = views
```

- [ ] **Step 8: Remove circular import + `_torch` alias from `_postprocess`**

```python
# BEFORE (inside _postprocess, after the global_alignment block):
        pts3d, colors = _ff.unproject_and_filter_points(
            ...
        )
        ...
        # Populate BA fields: subsampled world-point grid for track extraction.
        import torch as _torch
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
        ...
        _conf = _torch.from_numpy(raw_outputs["depth_conf"])   # (N, H, W)
```

Full replacement — find these two blocks in `_postprocess`:

```python
        # Import via package root so tests can patch
        # collab_splats.pointcloud.feedforward.unproject_and_filter_points.
        import collab_splats.pointcloud.feedforward as _ff

        extrinsic = raw_outputs["extrinsic"]
```

→ Remove the two-line `import` + comment:

```python
        extrinsic = raw_outputs["extrinsic"]
```

Then find:

```python
        pts3d, colors = _ff.unproject_and_filter_points(
```

→ Change to:

```python
        pts3d, colors = unproject_and_filter_points(
```

Then find:

```python
        # Populate BA fields: subsampled world-point grid for track extraction.
        import torch as _torch
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
```

→ Remove `import torch as _torch`:

```python
        # Populate BA fields: subsampled world-point grid for track extraction.
        world_pts_flat, _ = _raw_to_world_points(raw_outputs, subsample=1)
```

Then find:

```python
        _conf = _torch.from_numpy(raw_outputs["depth_conf"])   # (N, H, W)
```

→ Change to:

```python
        _conf = torch.from_numpy(raw_outputs["depth_conf"])   # (N, H, W)
```

- [ ] **Step 9: Remove circular import from `_reproject_after_ba`**

```python
# BEFORE:
    def _reproject_after_ba(self, raw_outputs, extrinsics_3x4, intrinsics):
        # Import via package root to respect any test patches on unproject_and_filter_points.
        import collab_splats.pointcloud.feedforward as _ff
        return _ff.unproject_and_filter_points(
```

```python
# AFTER:
    def _reproject_after_ba(self, raw_outputs, extrinsics_3x4, intrinsics):
        return unproject_and_filter_points(
```

- [ ] **Step 10: Remove lazy imports from `_verify_loop_candidate`**

```python
# BEFORE:
    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        ...
        import torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        device = next(self.model.parameters()).device
```

```python
# AFTER:
    def _verify_loop_candidate(self, frame1, frame2, verify_match_ratio=0.85):
        ...
        device = next(self.model.parameters()).device
```

- [ ] **Step 11: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py -v -m "not gpu"
```

Expected: `test_vggtx_defaults`, `test_vggtx_is_feedforward_creator`, `test_vggtx_missing_image_dir_raises`, `test_vggtx_postprocess_calls_global_alignment`, `test_vggtx_no_global_alignment_when_disabled` — all PASS.

- [ ] **Step 12: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_creator.py
git commit -m "refactor(vggtx): hoist imports, remove _HAS_VGGT guard and circular _ff import"
```

---

## Task 4: Final verification

- [ ] **Step 1: Run full pointcloud test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v -m "not gpu"
```

Expected: all non-GPU tests pass.

- [ ] **Step 2: Verify no stray lazy imports remain**

```bash
grep -n "^    import \|^    from \|^        import \|^        from " \
  collab_splats/pointcloud/feedforward/vggtx.py \
  collab_splats/pointcloud/feedforward/mapanything.py \
  collab_splats/pointcloud/loop_closure/closure.py
```

Expected output: only lines inside the nested functions of `_patch_vggtx_compute_similarity` (`import torch` and `import numpy as np` inside `_get_similarity` — user-approved to leave these nested).
