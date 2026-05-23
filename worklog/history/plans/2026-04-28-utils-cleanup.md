# Utils Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove 5 files from `collab_splats/utils/` by deleting dead shims, dispersing `utils.py` functions to their proper homes, inlining `TwoLayerMLP`, and parking grouping/metrics in `stage/`.

**Architecture:** Each task is independent and safe to commit individually. No behavioral changes — pure structural cleanup. After each deletion the import smoke test must pass before moving on.

**Tech Stack:** Python, pytest. Use `/opt/conda/envs/nerfstudio/bin/python` for all Python commands (nerfstudio env is py3.10).

---

## File Map

**Deleted:**
- `collab_splats/utils/pointcloud.py` — dead shim
- `collab_splats/utils/segmentation.py` — shim after caller migrated
- `collab_splats/utils/features.py` — shim + TwoLayerMLP after both dispersed
- `collab_splats/utils/utils.py` — after all functions dispersed
- `collab_splats/utils/grouping.py` — moved to stage

**Modified:**
- `collab_splats/utils/__init__.py` — remove `get_device` re-export
- `collab_splats/nerfstudio/datamanagers/features.py` — swap segmentation import
- `collab_splats/nerfstudio/models/rade_features.py` — inline TwoLayerMLP, fix BaseFeatureExtractor import
- `collab_splats/semantics/utils.py` — append `get_device`
- `collab_splats/semantics/features.py` — swap `get_device` import source

**Created:**
- `stage/metrics.py` — eval metric functions from `utils/utils.py`

---

### Task 1: Delete dead pointcloud shim

**Files:**
- Delete: `collab_splats/utils/pointcloud.py`

- [ ] **Step 1: Verify 0 production callers**

```bash
grep -r "utils.pointcloud\|utils/pointcloud" /workspace/collab-splats/collab_splats --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 2: Delete the file**

```bash
rm /workspace/collab-splats/collab_splats/utils/pointcloud.py
```

- [ ] **Step 3: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git -C /workspace/collab-splats add -u
git -C /workspace/collab-splats commit -m "refactor(utils): delete dead pointcloud shim (0 callers)"
```

---

### Task 2: Migrate segmentation caller, delete shim

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py` (import line ~22)
- Delete: `collab_splats/utils/segmentation.py`

- [ ] **Step 1: Find the import line**

```bash
grep -n "utils.segmentation" /workspace/collab-splats/collab_splats/nerfstudio/datamanagers/features.py
```

Expected output like: `22:from collab_splats.utils.segmentation import Segmentation, aggregate_masked_features`

- [ ] **Step 2: Replace the import**

In `collab_splats/nerfstudio/datamanagers/features.py`, change:

```python
from collab_splats.utils.segmentation import Segmentation, aggregate_masked_features
```

to:

```python
from collab_splats.semantics.segmentation import Segmentation, aggregate_masked_features
```

- [ ] **Step 3: Smoke test before deleting shim**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Verify no other callers of the shim**

```bash
grep -r "utils.segmentation\|utils/segmentation" /workspace/collab-splats/collab_splats --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 5: Delete the shim**

```bash
rm /workspace/collab-splats/collab_splats/utils/segmentation.py
```

- [ ] **Step 6: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git -C /workspace/collab-splats add -u
git -C /workspace/collab-splats commit -m "refactor(utils): delete segmentation shim; update nerfstudio datamanager to semantics.segmentation"
```

---

### Task 3: Inline TwoLayerMLP, fix features imports, delete utils/features.py

**Files:**
- Modify: `collab_splats/nerfstudio/models/rade_features.py` (lines 25-27)
- Modify: `stage/feedforward.py` (import line)
- Delete: `collab_splats/utils/features.py`

- [ ] **Step 1: Update rade_features.py imports**

In `collab_splats/nerfstudio/models/rade_features.py`, replace:

```python
from collab_splats.utils.features import TwoLayerMLP, BaseFeatureExtractor
```

with:

```python
from collab_splats.semantics.features import BaseFeatureExtractor
```

Then add the `TwoLayerMLP` class definition directly in `rade_features.py`, after the imports and before the `@dataclass` for `RadegsFeaturesModelConfig`. Paste this verbatim:

```python
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TwoLayerMLP(nn.Module):
    """Two-layer MLP via 1x1 convolutions for reconstructing feature maps."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        features_dim_dict: Dict[str, Tuple[int, int, int]],
    ):
        super().__init__()
        self.hidden_conv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.feature_branch_dict = nn.ModuleDict(
            {
                model: nn.Conv2d(hidden_dim, feat_shape[0], kernel_size=1)
                for model, feat_shape in features_dim_dict.items()
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.relu(self.hidden_conv(x))
        return {model: conv(x) for model, conv in self.feature_branch_dict.items()}

    @torch.no_grad()
    def per_gaussian_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        w_hidden = self.hidden_conv.weight.view(self.hidden_conv.out_channels, -1)
        x = F.relu(F.linear(x, w_hidden, self.hidden_conv.bias))
        outputs = {}
        for model, conv in self.feature_branch_dict.items():
            w_out = conv.weight.view(conv.out_channels, -1)
            outputs[model] = F.linear(x, w_out, conv.bias)
        return outputs
```

Note: `rade_features.py` already imports `torch`, `F`, and `Parameter` but NOT `import torch.nn as nn`. Add `import torch.nn as nn` at the top if not present. Also add `from typing import Dict, Tuple` if not already there (check with `grep -n "^from typing" rade_features.py` first).

- [ ] **Step 2: Smoke test rade_features**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.models.rade_features import RadegsFeaturesModel; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Update stage/feedforward.py**

In `stage/feedforward.py`, find:

```bash
grep -n "utils.features" /workspace/collab-splats/stage/feedforward.py
```

Replace any line like:

```python
from collab_splats.utils.features import BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor
```

with:

```python
from collab_splats.semantics.features import BaseFeatureExtractor, MaskCLIPExtractor, DINOFeatureExtractor
```

- [ ] **Step 4: Verify no remaining callers of utils.features**

```bash
grep -r "utils.features\|utils/features" /workspace/collab-splats --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 5: Delete the file**

```bash
rm /workspace/collab-splats/collab_splats/utils/features.py
```

- [ ] **Step 6: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git -C /workspace/collab-splats add -u
git -C /workspace/collab-splats commit -m "refactor(utils): inline TwoLayerMLP into rade_features; delete utils/features.py shim"
```

---

### Task 4: Disperse utils/utils.py, delete it

**Files:**
- Modify: `collab_splats/semantics/utils.py` (append `get_device`)
- Modify: `collab_splats/semantics/features.py` (swap `get_device` import)
- Modify: `collab_splats/utils/__init__.py` (remove `get_device` re-export)
- Create: `stage/metrics.py`
- Delete: `collab_splats/utils/utils.py`

- [ ] **Step 1: Add get_device to semantics/utils.py**

Append to the end of `collab_splats/semantics/utils.py`:

```python


def get_device() -> str:
    """Return 'cuda' if a CUDA device is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
```

`torch` is already imported in that file — verify with `grep -n "^import torch" collab_splats/semantics/utils.py` before adding.

- [ ] **Step 2: Update semantics/features.py**

In `collab_splats/semantics/features.py`, replace:

```python
from collab_splats.utils import get_device
```

with:

```python
from collab_splats.semantics.utils import get_device
```

- [ ] **Step 3: Smoke test semantics**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.semantics.features import BaseFeatureExtractor; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Create stage/metrics.py**

Create `/workspace/collab-splats/stage/metrics.py` with this content:

```python
"""Evaluation metrics (accuracy, completeness, normal error). Not part of production path."""

import numpy as np
import torch
from scipy.spatial import cKDTree


def calculate_accuracy(reconstructed_points, reference_points, percentile=90):
    """How far away percentile% of reconstructed points are from reference."""
    tree = cKDTree(reference_points)
    distances, _ = tree.query(reconstructed_points)
    return np.percentile(distances, percentile)


def calculate_completeness(reconstructed_points, reference_points, threshold=0.05):
    """Percentage of reference points within threshold of reconstructed cloud."""
    tree = cKDTree(reconstructed_points)
    distances, _ = tree.query(reference_points)
    within_threshold = np.sum(distances < threshold) / len(distances)
    return within_threshold * 100


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean angular error between predicted and reference normals (B, C, H, W)."""
    dot_products = torch.sum(gt * pred, dim=1)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    return torch.acos(dot_products)
```

- [ ] **Step 5: Update utils/__init__.py**

In `collab_splats/utils/__init__.py`, remove:

```python
from .utils import get_device
```

and remove `"get_device"` from `__all__`. The file should become:

```python
from .camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal
from .frame_sampling import OpticalFlowFrameSelector, sample_frames_fps, sample_frames_optical_flow
from .image import open_image, resize_image

__all__ = [
    "ColmapCamera",
    "convert_to_colmap_camera",
    "depth_double_to_normal",
    "OpticalFlowFrameSelector",
    "sample_frames_fps",
    "sample_frames_optical_flow",
    "open_image",
    "resize_image",
]
```

- [ ] **Step 6: Verify no remaining callers of utils/utils.py or get_device via utils**

```bash
grep -r "from collab_splats.utils import get_device\|from collab_splats.utils.utils\|utils\.utils\." /workspace/collab-splats --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 7: Delete utils/utils.py**

```bash
rm /workspace/collab-splats/collab_splats/utils/utils.py
```

- [ ] **Step 8: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('OK')"
```

Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git -C /workspace/collab-splats add -A stage/metrics.py
git -C /workspace/collab-splats add -u
git -C /workspace/collab-splats commit -m "refactor(utils): move get_device to semantics/utils; move eval metrics to stage/; delete utils/utils.py"
```

---

### Task 5: Move grouping.py to stage/

**Files:**
- Move: `collab_splats/utils/grouping.py` → `stage/grouping.py`

- [ ] **Step 1: Verify no production callers**

```bash
grep -r "utils.grouping\|utils/grouping" /workspace/collab-splats/collab_splats --include="*.py" | grep -v __pycache__
```

Expected: no output.

- [ ] **Step 2: Move the file**

```bash
git -C /workspace/collab-splats mv collab_splats/utils/grouping.py stage/grouping.py
```

- [ ] **Step 3: Fix import inside stage/grouping.py**

`grouping.py` imports from `collab_splats.utils.segmentation` and `collab_splats.utils.utils`. After the move, update those imports:

Find current imports:
```bash
grep -n "^from\|^import" /workspace/collab-splats/stage/grouping.py | head -20
```

Replace:
```python
from collab_splats.utils.segmentation import (
```
→
```python
from collab_splats.semantics.segmentation import (
```

Replace:
```python
from collab_splats.utils.utils import project_gaussians
```
→ inline `project_gaussians` directly into `stage/grouping.py` (copy the function body from `utils/utils.py` — it was already noted above). Or define it at the top of `stage/grouping.py`:

```python
import torch
from typing import Dict


def project_gaussians(meta: dict) -> Dict[str, torch.Tensor]:
    """Projects Gaussians into 2D image space. Returns lookup tensors indexed by global Gaussian ID."""
    W, H = meta["width"], meta["height"]
    radii = meta["radii"].squeeze()
    valid_mask = (radii > 1.0).sum(dim=1) > 0
    gaussian_ids = valid_mask.nonzero(as_tuple=False).squeeze()
    xy_rounded = torch.round(meta["means2d"]).squeeze().long()
    x = torch.clamp(xy_rounded[:, 0], 0, W - 1)
    y = torch.clamp(xy_rounded[:, 1], 0, H - 1)
    projected_flattened = x + y * W
    return {
        "proj_flattened": projected_flattened.detach().cpu(),
        "proj_depths": meta["depths"].squeeze().detach().cpu(),
        "valid_mask": valid_mask.detach().cpu(),
        "gaussian_ids": gaussian_ids.detach().cpu(),
    }
```

Remove the old `from collab_splats.utils.utils import project_gaussians` line.

- [ ] **Step 4: Smoke test**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import collab_splats; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git -C /workspace/collab-splats add -u
git -C /workspace/collab-splats add stage/grouping.py
git -C /workspace/collab-splats commit -m "refactor(utils): move grouping.py to stage/; inline project_gaussians"
```

---

### Task 6: Final verification

- [ ] **Step 1: Confirm utils/ end state**

```bash
ls /workspace/collab-splats/collab_splats/utils/
```

Expected: `__init__.py  camera_utils.py  frame_sampling.py  image.py  visualization.py` (plus `__pycache__`)

- [ ] **Step 2: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest /workspace/collab-splats/tests/ -x -q 2>&1 | tail -20
```

All tests pass (or pre-existing failures only — do not introduce new failures).

- [ ] **Step 3: Full import check**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.nerfstudio.models.rade_features import RadegsFeaturesModel
from collab_splats.nerfstudio.models.rade_gs import RadegsModel
from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager
from collab_splats.semantics.features import BaseFeatureExtractor
from collab_splats.semantics.utils import get_device
from collab_splats.utils import convert_to_colmap_camera, depth_double_to_normal
print('All imports OK')
"
```

Expected: `All imports OK`
