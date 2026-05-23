# Feedforward → Direct Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pointcloud_result_to_tsdf_inputs()` adapter + `feedforward_mesh.ipynb` notebook so MapAnything pointclouds can be meshed directly without nerfstudio.

**Architecture:** New `collab_splats/mesh/adapter.py` extracts Z-depth from `PointcloudResult.world_points` by projecting each frame's per-pixel world points into camera space via extrinsics, loads RGB from `image_paths`, inverts w2c → c2w. Returns `(depths, rgbs, c2w, intrinsics)` ready for existing `Open3DTSDFFusion.create()`. Notebook in `docs/pointcloud/` demonstrates the full pipeline end-to-end on 7scenes chess.

**Tech Stack:** numpy, PIL, Open3D, `collab_splats.mesh`, `collab_splats.pointcloud.feedforward`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `collab_splats/mesh/adapter.py` | `pointcloud_result_to_tsdf_inputs()` — world_points → depth, image_paths → rgb, inv(extrinsics) → c2w |
| Modify | `collab_splats/mesh/__init__.py` | export `pointcloud_result_to_tsdf_inputs` |
| Create | `tests/mesh/test_adapter.py` | unit tests for adapter |
| Create | `docs/pointcloud/feedforward_mesh.ipynb` | end-to-end example notebook |

---

### Task 1: Write failing tests for the adapter

**Files:**
- Create: `tests/mesh/test_adapter.py`

- [ ] **Step 1: Create test file**

```python
# tests/mesh/test_adapter.py
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from collab_splats.mesh.adapter import pointcloud_result_to_tsdf_inputs
from collab_splats.pointcloud.feedforward.base import PointcloudResult


def _make_result(N=2, H=4, W=6, world_points_z=1.5, with_world_points=True):
    """Build a minimal PointcloudResult with real image files in a temp dir."""
    rng = np.random.default_rng(42)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity w2c

    world_points = None
    if with_world_points:
        world_points = rng.random((N, H, W, 3)).astype(np.float32)
        world_points[..., 2] = world_points_z  # Z in cam frame = world_points_z when identity extrinsics

    tmpdir = tempfile.mkdtemp()
    image_paths = []
    for i in range(N):
        img_arr = (rng.random((H * 4, W * 4, 3)) * 255).astype(np.uint8)
        p = Path(tmpdir) / f"frame_{i:04d}.png"
        PILImage.fromarray(img_arr).save(p)
        image_paths.append(p)

    return PointcloudResult(
        pts3d=rng.random((10, 3)).astype(np.float32),
        colors=(rng.random((10, 3)) * 255).astype(np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.eye(3, dtype=np.float32)[None].repeat(N, axis=0),
        image_paths=image_paths,
        world_points=world_points,
    )


def test_output_shapes():
    N, H, W = 2, 4, 6
    result = _make_result(N=N, H=H, W=W)
    depths, rgbs, c2w, intrinsics = pointcloud_result_to_tsdf_inputs(result)
    assert depths.shape == (N, H, W)
    assert rgbs.shape == (N, H, W, 3)
    assert c2w.shape == (N, 4, 4)
    assert intrinsics.shape == (N, 3, 3)


def test_output_dtypes():
    result = _make_result()
    depths, rgbs, c2w, intrinsics = pointcloud_result_to_tsdf_inputs(result)
    assert depths.dtype == np.float32
    assert rgbs.dtype == np.float32
    assert c2w.dtype == np.float32
    assert intrinsics.dtype == np.float32


def test_depth_values_identity_extrinsics():
    # With identity w2c, cam_pts = world_points, so depth = world_points[..., 2]
    result = _make_result(world_points_z=1.5)
    depths, _, _, _ = pointcloud_result_to_tsdf_inputs(result)
    np.testing.assert_allclose(depths, 1.5, atol=1e-5)


def test_c2w_is_inverse_of_extrinsics():
    result = _make_result()
    _, _, c2w, _ = pointcloud_result_to_tsdf_inputs(result)
    # extrinsics is identity, so c2w should also be identity
    np.testing.assert_allclose(c2w, np.eye(4), atol=1e-5)


def test_rgb_range():
    result = _make_result()
    _, rgbs, _, _ = pointcloud_result_to_tsdf_inputs(result)
    assert rgbs.min() >= 0.0
    assert rgbs.max() <= 1.0


def test_raises_on_none_world_points():
    result = _make_result(with_world_points=False)
    with pytest.raises(ValueError, match="world_points"):
        pointcloud_result_to_tsdf_inputs(result)
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist yet)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_adapter.py -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'pointcloud_result_to_tsdf_inputs' from 'collab_splats.mesh.adapter'`

---

### Task 2: Implement the adapter

**Files:**
- Create: `collab_splats/mesh/adapter.py`

- [ ] **Step 1: Write `collab_splats/mesh/adapter.py`**

```python
# collab_splats/mesh/adapter.py
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from collab_splats.pointcloud.feedforward.base import PointcloudResult


def pointcloud_result_to_tsdf_inputs(
    result: PointcloudResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert PointcloudResult to inputs for Open3DTSDFFusion.create().

    Requires result.world_points (always populated by MapAnythingCreator and VGGTXCreator).
    Depth is derived by projecting world_points into each camera frame via extrinsics (w2c).
    RGBs are loaded from result.image_paths and resized to match model resolution (H, W).
    c2w = inv(result.extrinsics) — no coordinate-convention flip needed (MapAnything is OpenCV).

    Args:
        result: PointcloudResult from a feedforward creator with world_points populated.

    Returns:
        depths:     (N, H, W) float32, metres, Z in camera frame
        rgbs:       (N, H, W, 3) float32, [0, 1]
        c2w:        (N, 4, 4) float32, cam-to-world OpenCV convention
        intrinsics: (N, 3, 3) float32, camera intrinsics at model resolution

    Raises:
        ValueError: if result.world_points is None.
    """
    if result.world_points is None:
        raise ValueError(
            "result.world_points is None. Run inference with MapAnythingCreator or "
            "VGGTXCreator — both populate world_points during _postprocess()."
        )

    world_points = result.world_points          # (N, H, W, 3)
    N, H, W, _ = world_points.shape

    depths = np.empty((N, H, W), dtype=np.float32)
    for i in range(N):
        R = result.extrinsics[i, :3, :3]        # (3, 3) world-to-cam rotation
        t = result.extrinsics[i, :3, 3]         # (3,) world-to-cam translation
        cam_pts = world_points[i] @ R.T + t     # (H, W, 3)
        depths[i] = cam_pts[..., 2].clip(0)     # Z >= 0; negatives are boundary artefacts

    rgbs = np.empty((N, H, W, 3), dtype=np.float32)
    for i, path in enumerate(result.image_paths):
        img = PILImage.open(path).convert("RGB").resize((W, H), PILImage.BILINEAR)
        rgbs[i] = np.asarray(img, dtype=np.float32) / 255.0

    c2w = np.linalg.inv(result.extrinsics).astype(np.float32)

    return depths, rgbs, c2w, result.intrinsics.copy()
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_adapter.py -v 2>&1 | tail -20
```

Expected: 6 tests PASSED

- [ ] **Step 3: Commit**

```bash
git add collab_splats/mesh/adapter.py tests/mesh/test_adapter.py
git commit -m "feat(mesh): add pointcloud_result_to_tsdf_inputs adapter"
```

---

### Task 3: Export from mesh `__init__.py`

**Files:**
- Modify: `collab_splats/mesh/__init__.py`

- [ ] **Step 1: Add import and `__all__` entry**

In `collab_splats/mesh/__init__.py`, add after the existing imports:

```python
from collab_splats.mesh.adapter import pointcloud_result_to_tsdf_inputs
```

And add `"pointcloud_result_to_tsdf_inputs"` to `__all__`:

```python
__all__ = [
    "get_mesh_creator",
    "pointcloud_result_to_tsdf_inputs",
    "BaseMeshCreator",
    "MeshResult",
    "Open3DTSDFFusion",
    "DepthNormalPoisson",
    "GaussiansPoisson",
    "REGISTRY",
]
```

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.mesh import pointcloud_result_to_tsdf_inputs; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Re-run tests to confirm nothing broken**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/mesh/test_adapter.py -v 2>&1 | tail -10
```

Expected: 6 PASSED

- [ ] **Step 4: Commit**

```bash
git add collab_splats/mesh/__init__.py
git commit -m "feat(mesh): export pointcloud_result_to_tsdf_inputs from package"
```

---

### Task 4: Write the example notebook

**Files:**
- Create: `docs/pointcloud/feedforward_mesh.ipynb`

7scenes chess frames live at `data/7scenes/chess/chess/seq-01/` with filenames `frame-XXXXXX.color.png`.
Use 10 frames (indices 0–9) — enough to get a recognisable mesh, small enough to run quickly.

- [ ] **Step 1: Create the notebook**

Write `docs/pointcloud/feedforward_mesh.ipynb` with the following cells (use `nbformat` or write JSON directly):

**Cell 1 — Setup**
```python
from pathlib import Path
import numpy as np
import open3d as o3d

from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator
from collab_splats.mesh import Open3DTSDFFusion, pointcloud_result_to_tsdf_inputs

# Adjust IMAGE_DIR to your local data
IMAGE_DIR = Path("../../data/7scenes/chess/chess/seq-01")
OUTPUT_DIR = Path("/tmp/feedforward_mesh_demo/pointcloud")
MESH_DIR   = Path("/tmp/feedforward_mesh_demo/mesh")

# Use 10 frames for a quick demo
image_paths = sorted(IMAGE_DIR.glob("*.color.png"))[:10]
print(f"{len(image_paths)} images  →  {image_paths[0].name} … {image_paths[-1].name}")
```

**Cell 2 — Run MapAnything inference**
```python
creator = MapAnythingCreator(confidence_percentile=35.0)
result = creator.reconstruct(image_paths, OUTPUT_DIR)

print(f"pts3d:        {result.pts3d.shape}")
print(f"world_points: {result.world_points.shape}")   # (N, H, W, 3)
print(f"extrinsics:   {result.extrinsics.shape}")     # (N, 4, 4)  w2c
print(f"intrinsics:   {result.intrinsics.shape}")     # (N, 3, 3)
```

**Cell 3 — Inspect one depth frame**
```python
import matplotlib.pyplot as plt

depths, rgbs, c2w, intrinsics = pointcloud_result_to_tsdf_inputs(result)

print(f"depths: {depths.shape}  range [{depths.min():.2f}, {depths.max():.2f}] m")
print(f"rgbs:   {rgbs.shape}")
print(f"c2w:    {c2w.shape}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(rgbs[0])
axes[0].set_title("RGB frame 0")
axes[1].imshow(depths[0], cmap="plasma")
axes[1].set_title("Depth frame 0 (metres)")
plt.tight_layout()
plt.show()
```

**Cell 4 — TSDF fusion**
```python
mesher = Open3DTSDFFusion(
    output_dir=MESH_DIR,
    voxel_size=0.02,    # 2 cm voxels — coarser than splat-derived (0.01) due to noisier depth
    sdf_trunc=0.08,
    depth_trunc=10.0,
    clean_repair=True,
)
mesh_result = mesher.create(depths, rgbs, c2w, intrinsics)
print(f"Mesh saved → {mesh_result.mesh_path}")
```

**Cell 5 — Visualise**
```python
mesh = o3d.io.read_triangle_mesh(str(mesh_result.mesh_path))
print(f"Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")
o3d.visualization.draw_geometries([mesh])
```

- [ ] **Step 2: Verify notebook is valid JSON**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "import json; json.load(open('docs/pointcloud/feedforward_mesh.ipynb')); print('valid')"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add docs/pointcloud/feedforward_mesh.ipynb
git commit -m "docs(pointcloud): add feedforward_mesh notebook — MapAnything → TSDF direct mesh"
```

---

## Self-Review

**Spec coverage:**
- ✓ `pointcloud_result_to_tsdf_inputs` in `collab_splats/mesh/adapter.py`
- ✓ Exported from `collab_splats/mesh/__init__.py`
- ✓ Depth from world_points via extrinsics projection
- ✓ RGB from image_paths, resized to model (H, W)
- ✓ c2w = inv(extrinsics), no convention flip
- ✓ raises ValueError on world_points=None
- ✓ Notebook: 5 cells covering setup → inference → inspect → mesh → visualise
- ✓ Existing nerfstudio mesh path untouched
- ✓ No changes to PointcloudResult, Open3DTSDFFusion, BaseMeshCreator

**Placeholder scan:** none found.

**Type consistency:**
- `pointcloud_result_to_tsdf_inputs` defined Task 2, imported Task 3, used Task 4 — consistent.
- `PointcloudResult` from `collab_splats.pointcloud.feedforward.base` — matches actual module path.
- `Open3DTSDFFusion` from `collab_splats.mesh` — matches __init__.py.
