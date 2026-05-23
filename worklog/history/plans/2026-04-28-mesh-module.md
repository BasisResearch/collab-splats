# Mesh Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `collab_splats/utils/mesh.py` (1764-line dn-splatter port with hard nerfstudio imports) with a clean `collab_splats/mesh/` module whose core has zero nerfstudio dependency.

**Architecture:** Array-in/mesh-out interface (`BaseMeshCreator.create(depths, rgbs, c2w, intrinsics)`) isolates geometry logic from model loading. Nerfstudio data extraction lives in `collab_splats/nerfstudio/mesh_adapter.py`. Only `Open3DTSDFFusion` ported; three dead exporters dropped; two Poisson exporters kept as stubs.

**Tech Stack:** numpy, open3d, meshlib (optional, guarded), scipy, torch (tsdf only for type compat)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `collab_splats/mesh/__init__.py` | `get_mesh_creator()` registry, public re-exports |
| Create | `collab_splats/mesh/base.py` | `BaseMeshCreator`, `MeshResult` — zero deps |
| Create | `collab_splats/mesh/tsdf.py` | `Open3DTSDFFusion` — array-in, no eval_setup |
| Create | `collab_splats/mesh/poisson.py` | `DepthNormalPoisson`, `GaussiansPoisson` stubs |
| Create | `collab_splats/mesh/utils.py` | standalone geometry helpers (moved, not rewritten) |
| Create | `collab_splats/nerfstudio/mesh_adapter.py` | `extract_mesh_inputs(load_config)` — all nerfstudio imports here |
| Create | `tests/mesh/__init__.py` | empty |
| Create | `tests/mesh/test_utils.py` | tests for geometry helpers |
| Create | `tests/mesh/test_tsdf.py` | tests for Open3DTSDFFusion with synthetic data |
| Create | `tests/mesh/test_registry.py` | tests for get_mesh_creator |
| Modify | `collab_splats/wrapper/splatter.py` | update `create_mesh()` ~line 435 |
| Delete | `collab_splats/utils/mesh.py` | replaced by mesh/ module |

---

## Task 1: `collab_splats/mesh/base.py` + `MeshResult`

**Files:**
- Create: `collab_splats/mesh/__init__.py` (empty placeholder)
- Create: `collab_splats/mesh/base.py`

- [ ] **Step 1: Create empty `__init__.py`**

```bash
mkdir -p /workspace/collab-splats/collab_splats/mesh
touch /workspace/collab-splats/collab_splats/mesh/__init__.py
```

- [ ] **Step 2: Write `base.py`**

Create `collab_splats/mesh/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshResult:
    mesh_path: Path
    pcd_path: Path | None = None


@dataclass
class BaseMeshCreator:
    output_dir: Path

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError(
            f"{type(self).__name__}.create() is not implemented"
        )
```

- [ ] **Step 3: Verify import**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.mesh.base import BaseMeshCreator, MeshResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/mesh/__init__.py collab_splats/mesh/base.py
git commit -m "feat(mesh): add mesh/ module skeleton with BaseMeshCreator + MeshResult"
```

---

## Task 2: `collab_splats/mesh/utils.py`

Move standalone geometry helpers from `collab_splats/utils/mesh.py`. No logic changes — pure move. meshlib import guarded.

**Files:**
- Create: `collab_splats/mesh/utils.py`
- Create: `tests/mesh/__init__.py`
- Create: `tests/mesh/test_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/mesh/__init__.py` (empty).

Create `tests/mesh/test_utils.py`:

```python
import numpy as np
import pytest


def test_find_depth_edges_shape():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.random.rand(64, 64).astype(np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=1)
    assert edges.shape == (64, 64)
    assert edges.dtype == bool


def test_find_depth_edges_constant_depth_no_edges():
    from collab_splats.mesh.utils import find_depth_edges

    depth = np.ones((32, 32), dtype=np.float32)
    edges = find_depth_edges(depth, threshold=0.01, dilation_itr=0)
    assert not edges.any()


def test_normals2vertex_output_shape():
    from collab_splats.mesh.utils import normals2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    normals = rng.random((200, 3)).astype(np.float32)
    result = normals2vertex(mesh_vertices, points, normals, k=5)
    assert result.shape == (50, 3)


def test_features2vertex_output_shape():
    from collab_splats.mesh.utils import features2vertex

    rng = np.random.default_rng(0)
    mesh_vertices = rng.random((50, 3)).astype(np.float32)
    points = rng.random((200, 3)).astype(np.float32)
    features = rng.random((200, 16)).astype(np.float32)
    result = features2vertex(mesh_vertices, points, features, k=5)
    assert result.shape == (50, 16)
```

- [ ] **Step 2: Run tests — expect ImportError (module not yet created)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_utils.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'find_depth_edges' from 'collab_splats.mesh.utils'`

- [ ] **Step 3: Create `collab_splats/mesh/utils.py`**

Copy these functions verbatim from `collab_splats/utils/mesh.py` (do NOT copy the class definitions or nerfstudio imports):

Functions to copy: `find_depth_edges`, `normals2vertex`, `features2vertex`, `clean_repair_mesh`, `_filter_mesh_components`, `align_geometry_floor`, `get_floor_plane`, `mesh_clustering`, `pick_indices_at_random`.

The file header should be:

```python
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm, trange

try:
    import meshlib.mrmeshpy as mm
    _MM_AVAILABLE = True
except ImportError:
    mm = None
    _MM_AVAILABLE = False
```

Then paste each function body from `collab_splats/utils/mesh.py`. The functions start at these approximate line numbers:
- `pick_indices_at_random` — line ~95
- `find_depth_edges` — line ~103
- `normals2vertex` — line ~143
- `features2vertex` — line ~157
- `clean_repair_mesh` — line ~227 (includes `_filter_mesh_components` helper above it — check actual file)
- `align_geometry_floor` — line ~312
- `get_floor_plane` — line ~403
- `mesh_clustering` — line ~425

One fix required: add `n_removed = 0` at the top of `clean_repair_mesh` (before the if/else block) to prevent `UnboundLocalError` when `use_largest=True`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_utils.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/mesh/utils.py tests/mesh/__init__.py tests/mesh/test_utils.py
git commit -m "feat(mesh): add mesh/utils.py — geometry helpers moved from utils/mesh.py"
```

---

## Task 3: `collab_splats/mesh/poisson.py`

Stubs only — both raise `NotImplementedError`.

**Files:**
- Create: `collab_splats/mesh/poisson.py`

- [ ] **Step 1: Create `collab_splats/mesh/poisson.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collab_splats.mesh.base import BaseMeshCreator, MeshResult


@dataclass
class DepthNormalPoisson(BaseMeshCreator):
    """Backproject rendered depth+normals → Poisson reconstruction. Not yet implemented."""

    poisson_depth: int = 9

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        normals: np.ndarray | None = None,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError("DepthNormalPoisson not yet implemented")


@dataclass
class GaussiansPoisson(BaseMeshCreator):
    """Gaussian means+normals → Poisson reconstruction. Not yet implemented."""

    poisson_depth: int = 9

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        means: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        **kwargs,
    ) -> MeshResult:
        raise NotImplementedError("GaussiansPoisson not yet implemented")
```

- [ ] **Step 2: Verify import**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.mesh.poisson import DepthNormalPoisson, GaussiansPoisson; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/mesh/poisson.py
git commit -m "feat(mesh): add Poisson stub exporters (DepthNormalPoisson, GaussiansPoisson)"
```

---

## Task 4: `collab_splats/mesh/tsdf.py` — `Open3DTSDFFusion`

**Files:**
- Create: `collab_splats/mesh/tsdf.py`
- Create: `tests/mesh/test_tsdf.py`

- [ ] **Step 1: Write failing tests**

Create `tests/mesh/test_tsdf.py`:

```python
import numpy as np
import pytest
from pathlib import Path


def _synthetic_frames(N=3, H=32, W=32):
    """Synthetic data: flat depth plane at 1m, random colors, identity poses."""
    depths = np.ones((N, H, W), dtype=np.float32)
    rgbs = (np.random.rand(N, H, W, 3) * 0.5 + 0.25).astype(np.float32)
    # Cam-to-world: identity + small x-translations
    c2w = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)
    for i in range(N):
        c2w[i, 0, 3] = i * 0.05
    # Intrinsics: ~45 deg fov
    f = float(W)
    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = f
    intrinsics[:, 1, 1] = f
    intrinsics[:, 0, 2] = W / 2.0
    intrinsics[:, 1, 2] = H / 2.0
    return depths, rgbs, c2w, intrinsics


def test_open3d_tsdf_returns_mesh_result(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion
    from collab_splats.mesh.base import MeshResult

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    result = creator.create(depths, rgbs, c2w, intrinsics)
    assert isinstance(result, MeshResult)


def test_open3d_tsdf_writes_ply(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    result = creator.create(depths, rgbs, c2w, intrinsics)
    assert result.mesh_path.exists()
    assert result.mesh_path.suffix == ".ply"


def test_open3d_tsdf_clean_repair_skips_without_meshlib(tmp_path, monkeypatch):
    import collab_splats.mesh.tsdf as tsdf_mod

    monkeypatch.setattr(tsdf_mod, "_MM_AVAILABLE", False)
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = Open3DTSDFFusion(output_dir=tmp_path, clean_repair=True)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    # Should not crash even without meshlib
    result = creator.create(depths, rgbs, c2w, intrinsics)
    assert result.mesh_path.exists()


def test_open3d_tsdf_creates_output_dir(tmp_path):
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    nested = tmp_path / "a" / "b" / "c"
    creator = Open3DTSDFFusion(output_dir=nested, clean_repair=False)
    depths, rgbs, c2w, intrinsics = _synthetic_frames()
    creator.create(depths, rgbs, c2w, intrinsics)
    assert nested.exists()
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_tsdf.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'Open3DTSDFFusion' from 'collab_splats.mesh.tsdf'`

- [ ] **Step 3: Create `collab_splats/mesh/tsdf.py`**

```python
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

try:
    import meshlib.mrmeshpy as mm
    _MM_AVAILABLE = True
except ImportError:
    mm = None
    _MM_AVAILABLE = False

from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.utils import clean_repair_mesh

logger = logging.getLogger(__name__)


@dataclass
class Open3DTSDFFusion(BaseMeshCreator):
    """TSDF fusion via Open3D ScalableTSDFVolume.

    Accepts rendered depth + RGB frames as numpy arrays — no nerfstudio dependency.
    """

    output_dir: Path
    voxel_size: float = 0.01
    sdf_trunc: float = 0.04
    depth_trunc: float = 20.0
    depth_scale: float = 1.0
    clean_repair: bool = True
    clean_max_hole_size: float = 3.0
    clean_max_edge_splits: int = 10000
    clean_use_largest: bool = False

    def create(
        self,
        depths: np.ndarray,
        rgbs: np.ndarray,
        c2w: np.ndarray,
        intrinsics: np.ndarray,
        **kwargs,
    ) -> MeshResult:
        """Fuse depth+RGB frames into a mesh via TSDF.

        Args:
            depths:     (N, H, W) float32, metres
            rgbs:       (N, H, W, 3) float32, [0, 1]
            c2w:        (N, 4, 4) float32, cam-to-world OpenCV
            intrinsics: (N, 3, 3) float32
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        N, H, W = depths.shape

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i in tqdm(range(N), desc="TSDF integration"):
            rgb_u8 = (np.ascontiguousarray(rgbs[i]) * 255).astype(np.uint8)
            depth_f32 = np.ascontiguousarray(depths[i] / self.depth_scale).astype(np.float32)

            rgb_o3d = o3d.geometry.Image(rgb_u8)
            depth_o3d = o3d.geometry.Image(depth_f32)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=self.depth_trunc,
                convert_rgb_to_intensity=False,
            )

            fx = float(intrinsics[i, 0, 0])
            fy = float(intrinsics[i, 1, 1])
            cx = float(intrinsics[i, 0, 2])
            cy = float(intrinsics[i, 1, 2])
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            extrinsic = np.linalg.inv(c2w[i])

            volume.integrate(rgbd, intrinsic=intrinsic_o3d, extrinsic=extrinsic)

        mesh = volume.extract_triangle_mesh()

        raw_path = self.output_dir / "mesh_tsdf.ply"
        o3d.io.write_triangle_mesh(str(raw_path), mesh)
        final_path = raw_path

        if self.clean_repair:
            if _MM_AVAILABLE:
                temp_path = self.output_dir / "mesh_tsdf_temp.ply"
                shutil.copy(raw_path, temp_path)
                cleaned = clean_repair_mesh(
                    str(temp_path),
                    max_hole_size=self.clean_max_hole_size,
                    max_edge_splits=self.clean_max_edge_splits,
                    use_largest=self.clean_use_largest,
                )
                clean_path = self.output_dir / "mesh_tsdf_clean.ply"
                mm.saveMesh(cleaned, str(clean_path))
                temp_path.unlink(missing_ok=True)
                final_path = clean_path
            else:
                logger.warning(
                    "clean_repair=True but meshlib not installed; skipping. "
                    "Install with: pip install meshlib"
                )

        return MeshResult(mesh_path=final_path)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_tsdf.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py
git commit -m "feat(mesh): add Open3DTSDFFusion — array-in/mesh-out, no nerfstudio dep"
```

---

## Task 5: `collab_splats/mesh/__init__.py` — registry

**Files:**
- Modify: `collab_splats/mesh/__init__.py`
- Create: `tests/mesh/test_registry.py`

- [ ] **Step 1: Write failing tests**

Create `tests/mesh/test_registry.py`:

```python
import pytest
from pathlib import Path


def test_get_mesh_creator_open3d_tsdf(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    creator = get_mesh_creator("open3d_tsdf", output_dir=tmp_path)
    assert isinstance(creator, Open3DTSDFFusion)
    assert creator.output_dir == tmp_path


def test_get_mesh_creator_depth_normal_poisson(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.poisson import DepthNormalPoisson

    creator = get_mesh_creator("depth_normal_poisson", output_dir=tmp_path)
    assert isinstance(creator, DepthNormalPoisson)


def test_get_mesh_creator_gaussians_poisson(tmp_path):
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.mesh.poisson import GaussiansPoisson

    creator = get_mesh_creator("gaussians_poisson", output_dir=tmp_path)
    assert isinstance(creator, GaussiansPoisson)


def test_get_mesh_creator_unknown_raises():
    from collab_splats.mesh import get_mesh_creator

    with pytest.raises(ValueError, match="Unknown mesh method"):
        get_mesh_creator("nonexistent", output_dir=Path("/tmp"))


def test_get_mesh_creator_passes_kwargs(tmp_path):
    from collab_splats.mesh import get_mesh_creator

    creator = get_mesh_creator("open3d_tsdf", output_dir=tmp_path, voxel_size=0.05)
    assert creator.voxel_size == 0.05
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_registry.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'get_mesh_creator' from 'collab_splats.mesh'`

- [ ] **Step 3: Write `collab_splats/mesh/__init__.py`**

```python
from __future__ import annotations

from pathlib import Path

from collab_splats.mesh.base import BaseMeshCreator, MeshResult
from collab_splats.mesh.poisson import DepthNormalPoisson, GaussiansPoisson
from collab_splats.mesh.tsdf import Open3DTSDFFusion

REGISTRY: dict[str, type[BaseMeshCreator]] = {
    "open3d_tsdf": Open3DTSDFFusion,
    "depth_normal_poisson": DepthNormalPoisson,
    "gaussians_poisson": GaussiansPoisson,
}


def get_mesh_creator(method: str, output_dir: Path, **kwargs) -> BaseMeshCreator:
    if method not in REGISTRY:
        raise ValueError(
            f"Unknown mesh method {method!r}. Choose from: {sorted(REGISTRY)}"
        )
    return REGISTRY[method](output_dir=Path(output_dir), **kwargs)


__all__ = [
    "get_mesh_creator",
    "BaseMeshCreator",
    "MeshResult",
    "Open3DTSDFFusion",
    "DepthNormalPoisson",
    "GaussiansPoisson",
    "REGISTRY",
]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/ -v
```

Expected: all 9 tests pass (4 utils + 4 tsdf + 5 registry — wait: 4+4+5=13).

- [ ] **Step 5: Run full suite — confirm no regressions**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --ignore=tests/mesh -x 2>&1 | tail -20
```

Expected: same pass/fail counts as before this task.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/mesh/__init__.py tests/mesh/test_registry.py
git commit -m "feat(mesh): add get_mesh_creator registry + __init__.py"
```

---

## Task 6: `collab_splats/nerfstudio/mesh_adapter.py`

Extract `(depths, rgbs, c2w, intrinsics)` from a trained nerfstudio pipeline. All nerfstudio imports live here.

**Files:**
- Create: `collab_splats/nerfstudio/mesh_adapter.py`

- [ ] **Step 1: Create `collab_splats/nerfstudio/mesh_adapter.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def extract_mesh_inputs(
    load_config: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-frame (depths, rgbs, c2w, intrinsics) from a trained nerfstudio pipeline.

    Args:
        load_config: Path to nerfstudio config yaml (e.g. outputs/.../config.yml)

    Returns:
        depths:     (N, H, W) float32, metres
        rgbs:       (N, H, W, 3) float32, [0, 1]
        c2w:        (N, 4, 4) float32, cam-to-world OpenCV convention
        intrinsics: (N, 3, 3) float32
    """
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(Path(load_config))

    cameras = pipeline.datamanager.train_dataset.cameras
    N = len(pipeline.datamanager.train_dataset)

    all_depths: list[np.ndarray] = []
    all_rgbs: list[np.ndarray] = []
    all_c2w: list[np.ndarray] = []
    all_intrinsics: list[np.ndarray] = []

    with torch.no_grad():
        for image_idx, data in enumerate(pipeline.datamanager.train_dataset):
            camera = cameras[image_idx : image_idx + 1]
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)

            rgb = outputs["rgb"].cpu().numpy().astype(np.float32)          # (H, W, 3)
            depth = outputs["depth"].squeeze(-1).cpu().numpy().astype(np.float32)  # (H, W)

            # nerfstudio camera_to_worlds is (3, 4); pad to (4, 4)
            c2w_34 = camera.camera_to_worlds[0].cpu().numpy()
            c2w_44 = np.eye(4, dtype=np.float32)
            c2w_44[:3] = c2w_34

            K = np.eye(3, dtype=np.float32)
            K[0, 0] = camera.fx.item()
            K[1, 1] = camera.fy.item()
            K[0, 2] = camera.cx.item()
            K[1, 2] = camera.cy.item()

            all_rgbs.append(rgb)
            all_depths.append(depth)
            all_c2w.append(c2w_44)
            all_intrinsics.append(K)

    return (
        np.stack(all_depths),
        np.stack(all_rgbs),
        np.stack(all_c2w),
        np.stack(all_intrinsics),
    )
```

- [ ] **Step 2: Verify import (no nerfstudio needed to import — all imports are lazy)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.mesh_adapter import extract_mesh_inputs; print('OK')"
```

Expected: `OK` (nerfstudio import only happens when `extract_mesh_inputs()` is called)

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/nerfstudio/mesh_adapter.py
git commit -m "feat(mesh): add nerfstudio/mesh_adapter.py — extract_mesh_inputs"
```

---

## Task 7: Update `splatter.py` + delete `utils/mesh.py`

**Files:**
- Modify: `collab_splats/wrapper/splatter.py` ~line 435
- Delete: `collab_splats/utils/mesh.py`

- [ ] **Step 1: Read current `create_mesh` method in splatter.py**

Read `collab_splats/wrapper/splatter.py` around line 430. Confirm exact block to replace.

The current block looks like:

```python
if not mesh_dir.exists() or overwrite:
    from collab_splats.utils import mesh

    print(f"Initializing mesher {mesher_type}")

    mesher = getattr(mesh, mesher_type)(
        load_config=Path(self.config["model_config_path"]),
        output_dir=mesh_dir,
        **mesher_kwargs,
    )

    self.config["mesh_info"] = mesher.main()
```

- [ ] **Step 2: Replace the mesher block in `splatter.py`**

Replace the block above with:

```python
if not mesh_dir.exists() or overwrite:
    from collab_splats.mesh import get_mesh_creator
    from collab_splats.nerfstudio.mesh_adapter import extract_mesh_inputs

    print(f"Initializing mesher {mesher_type}")

    depths, rgbs, c2w, intrinsics = extract_mesh_inputs(
        load_config=Path(self.config["model_config_path"])
    )
    creator = get_mesh_creator(mesher_type, output_dir=mesh_dir, **mesher_kwargs)
    result = creator.create(depths, rgbs, c2w, intrinsics)
    self.config["mesh_info"] = {"mesh": result.mesh_path}
```

- [ ] **Step 3: Delete `collab_splats/utils/mesh.py`**

```bash
cd /workspace/collab-splats && git rm collab_splats/utils/mesh.py
```

- [ ] **Step 4: Verify nothing imports old path**

```bash
grep -rn "from collab_splats.utils import mesh\|from collab_splats.utils.mesh\|utils\.mesh" /workspace/collab-splats/collab_splats/ /workspace/collab-splats/tests/ --include="*.py"
```

Expected: no output (zero remaining references).

- [ ] **Step 5: Verify full test suite still passes**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: mesh tests still pass; no regressions vs pre-task run.

- [ ] **Step 6: Verify `collab_splats.mesh` imports cleanly without nerfstudio side effects**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -c "
import collab_splats.mesh as m
print('registry:', sorted(m.REGISTRY.keys()))
print('no nerfstudio imported:', 'nerfstudio' not in dir())
"
```

Expected:
```
registry: ['depth_normal_poisson', 'gaussians_poisson', 'open3d_tsdf']
no nerfstudio imported: True
```

- [ ] **Step 7: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/wrapper/splatter.py
git commit -m "refactor(mesh): wire splatter.py to new mesh module; delete utils/mesh.py"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `collab_splats/mesh/` module created | Task 1 |
| `BaseMeshCreator` + `MeshResult` | Task 1 |
| `Open3DTSDFFusion` — array-in, no eval_setup | Task 4 |
| `DepthNormalPoisson` + `GaussiansPoisson` stubs | Task 3 |
| `utils.py` — standalone geometry helpers | Task 2 |
| `get_mesh_creator()` registry | Task 5 |
| `nerfstudio/mesh_adapter.py` | Task 6 |
| `utils/mesh.py` deleted | Task 7 |
| `splatter.py` updated | Task 7 |
| Tests: tsdf, utils, registry | Tasks 2, 4, 5 |
| meshlib guard (`_MM_AVAILABLE`) | Tasks 2, 4 |

**Placeholder scan:** No TBDs, TODOs, or missing code blocks.

**Type consistency check:**
- `MeshResult` defined in Task 1, used in Tasks 4, 5 ✓
- `BaseMeshCreator` defined in Task 1, inherited in Tasks 3, 4 ✓
- `get_mesh_creator(method, output_dir, **kwargs)` defined in Task 5, called in Task 7 ✓
- `extract_mesh_inputs(load_config)` defined in Task 6, called in Task 7 ✓
- `clean_repair_mesh(path, max_hole_size, max_edge_splits, use_largest)` used in Task 4 — comes from `utils.py` (Task 2) ✓
- `mm.saveMesh(cleaned, str(path))` in tsdf.py — `mm` is the `meshlib.mrmeshpy` import; guarded by `_MM_AVAILABLE` ✓
