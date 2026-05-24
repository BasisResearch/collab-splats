# Geometry Utilities Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `collab_splats/utils/geometry.py` with `extrinsics_to_homogeneous`, `invert_poses`, `extract_intrinsics`, and `OPENGL_TO_OPENCV`, then replace all 13+ inline callsites across the codebase.

**Architecture:** New pure-numpy module in `utils/`. All callsites import from there. `_extrinsics_3x4_to_4x4` private function in `feedforward/base.py` is deleted. `OPENGL_TO_OPENCV` moves from `camera_utils.py` to `geometry.py` with no re-export.

**Tech Stack:** numpy, pytest. Python env: `/opt/conda/envs/nerfstudio/bin/python`

---

## File Map

| Action | File | Change |
|---|---|---|
| Create | `collab_splats/utils/geometry.py` | New module |
| Create | `tests/utils/test_geometry.py` | Unit tests |
| Modify | `collab_splats/utils/__init__.py` | Export new symbols |
| Modify | `collab_splats/utils/camera_utils.py` | Remove `OPENGL_TO_OPENCV` |
| Modify | `collab_splats/nerfstudio/utils/mesh_adapter.py` | New import + inline 3x4→4x4 |
| Modify | `collab_splats/pointcloud/feedforward/base.py` | Delete private fn, use public |
| Modify | `collab_splats/pointcloud/feedforward/vggt_omega.py` | Update import |
| Modify | `collab_splats/pointcloud/feedforward/mapanything.py` | Replace `closed_form_pose_inverse` |
| Modify | `collab_splats/pointcloud/wrappers.py` | Replace 2 callsites |
| Modify | `collab_splats/pointcloud/utils.py` | Replace 3 callsites |
| Modify | `collab_splats/pointcloud/base.py` | Replace vstack + linalg.inv |
| Modify | `collab_splats/wrapper/splatter.py` | Replace inline 3x4→4x4 |
| Modify | `collab_splats/mesh/utils.py` | Replace linalg.inv |
| Modify | `collab_splats/mesh/tsdf.py` | Replace linalg.inv + fx/fy/cx/cy |

---

## Task 1: Create geometry.py with tests (TDD)

**Files:**
- Create: `collab_splats/utils/geometry.py`
- Create: `tests/utils/test_geometry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/utils/test_geometry.py
import numpy as np
import pytest
from collab_splats.utils.geometry import (
    OPENGL_TO_OPENCV,
    OPENCV_TO_OPENGL,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
)


class TestExtrinsicsToHomogeneous:
    def test_batched(self):
        ext = np.random.rand(4, 3, 4).astype(np.float32)
        out = extrinsics_to_homogeneous(ext)
        assert out.shape == (4, 4, 4)
        np.testing.assert_array_equal(out[:, 3, :], [[0, 0, 0, 1]] * 4)
        np.testing.assert_array_equal(out[:, :3, :], ext)

    def test_single(self):
        ext = np.random.rand(3, 4).astype(np.float32)
        out = extrinsics_to_homogeneous(ext)
        assert out.shape == (4, 4)
        np.testing.assert_array_equal(out[3, :], [0, 0, 0, 1])
        np.testing.assert_array_equal(out[:3, :], ext)

    def test_dtype_preserved(self):
        ext = np.random.rand(2, 3, 4).astype(np.float64)
        out = extrinsics_to_homogeneous(ext)
        assert out.dtype == np.float64


class TestInvertPoses:
    def _random_rigid(self, *shape):
        # Build valid rotation via QR decomposition
        A = np.random.randn(*shape, 3, 3)
        Q, _ = np.linalg.qr(A)
        t = np.random.randn(*shape, 3, 1)
        poses = np.zeros(shape + (4, 4))
        poses[..., :3, :3] = Q
        poses[..., :3, 3:] = t
        poses[..., 3, 3] = 1.0
        return poses.astype(np.float64)

    def test_single_roundtrip(self):
        T = self._random_rigid()
        np.testing.assert_allclose(invert_poses(T) @ T, np.eye(4), atol=1e-10)

    def test_batched_roundtrip(self):
        T = self._random_rigid(5)
        result = invert_poses(T) @ T
        np.testing.assert_allclose(result, np.eye(4)[None].repeat(5, 0), atol=1e-10)

    def test_arbitrary_batch_shape(self):
        T = self._random_rigid(3, 7)
        result = invert_poses(T) @ T
        eye = np.eye(4)[None, None].repeat(3, 0).repeat(7, 1)
        np.testing.assert_allclose(result, eye, atol=1e-10)

    def test_dtype_preserved(self):
        T = self._random_rigid().astype(np.float32)
        assert invert_poses(T).dtype == np.float32


class TestExtractIntrinsics:
    def test_basic(self):
        K = np.array([[500.0, 0, 320.0], [0, 480.0, 240.0], [0, 0, 1.0]])
        fx, fy, cx, cy = extract_intrinsics(K)
        assert fx == 500.0
        assert fy == 480.0
        assert cx == 320.0
        assert cy == 240.0

    def test_returns_floats(self):
        K = np.eye(3, dtype=np.float32)
        fx, fy, cx, cy = extract_intrinsics(K)
        assert isinstance(fx, float)


class TestConstants:
    def test_opengl_to_opencv_shape(self):
        assert OPENGL_TO_OPENCV.shape == (4, 4)

    def test_self_inverse(self):
        np.testing.assert_array_equal(OPENGL_TO_OPENCV, OPENCV_TO_OPENGL)

    def test_flips_yz(self):
        # diag(1, -1, -1, 1)
        expected = np.diag([1, -1, -1, 1]).astype(np.float64)
        np.testing.assert_array_equal(OPENGL_TO_OPENCV, expected)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_geometry.py -v 2>&1 | tail -5
```
Expected: `ImportError` or `ModuleNotFoundError` — `geometry` doesn't exist yet.

- [ ] **Step 3: Create `collab_splats/utils/geometry.py`**

```python
"""Pure-numpy camera geometry utilities shared across the pipeline.

Conventions:
  OpenCV camera axes:  X right, Y down,  Z forward  (COLMAP, VGGT-X, BA)
  OpenGL camera axes:  X right, Y up,    Z backward  (nerfstudio, splats)
"""
from __future__ import annotations

import numpy as np

########################################################################
########## Constants ###################################################
########################################################################

# Camera axis convention flip (OpenCV ↔ OpenGL). Self-inverse: applying
# twice returns to original. diag(1, -1, -1, 1).
OPENGL_TO_OPENCV: np.ndarray = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)
OPENCV_TO_OPENGL: np.ndarray = OPENGL_TO_OPENCV  # same matrix (self-inverse)

########################################################################
########## Geometry helpers ############################################
########################################################################


def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    """Append [0,0,0,1] row to convert (N,3,4)→(N,4,4) or (3,4)→(4,4).

    Output dtype matches input dtype.
    """
    single = extrinsics.ndim == 2  # (3,4) → (4,4)
    if single:
        extrinsics = extrinsics[np.newaxis]  # (1,3,4)
    n = extrinsics.shape[0]
    bottom = np.tile(
        np.array([[0, 0, 0, 1]], dtype=extrinsics.dtype), (n, 1, 1)
    )  # (N,1,4)
    out = np.concatenate([extrinsics, bottom], axis=1)  # (N,4,4)
    return out[0] if single else out


def invert_poses(poses: np.ndarray) -> np.ndarray:
    """Closed-form SE3 inverse: (...,4,4) → (...,4,4).

    Works on any leading batch shape: (4,4), (N,4,4), (B,N,4,4).
    Uses R^T, -R^T@t — numerically exact for valid rotation matrices and
    faster than np.linalg.inv. Assumes poses are valid rigid-body transforms.
    """
    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -(R_inv @ t)            # -R^T t
    out = np.zeros_like(poses)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = t_inv
    out[..., 3, 3] = 1.0
    return out


def extract_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a (3,3) camera intrinsics matrix."""
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/utils/test_geometry.py -v 2>&1 | tail -10
```
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/geometry.py tests/utils/test_geometry.py
git commit -m "feat(utils): add geometry.py with extrinsics_to_homogeneous, invert_poses, extract_intrinsics"
```

---

## Task 2: Export from utils `__init__.py`

**Files:**
- Modify: `collab_splats/utils/__init__.py`

- [ ] **Step 1: Add imports**

Current top of file:
```python
from .camera_utils import ColmapCamera, convert_to_colmap_camera, depth_double_to_normal
```

Add after the existing imports (before `__all__`):
```python
from .geometry import (
    OPENGL_TO_OPENCV,
    OPENCV_TO_OPENGL,
    extrinsics_to_homogeneous,
    invert_poses,
    extract_intrinsics,
)
```

Add to `__all__`:
```python
    "OPENGL_TO_OPENCV",
    "OPENCV_TO_OPENGL",
    "extrinsics_to_homogeneous",
    "invert_poses",
    "extract_intrinsics",
```

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.utils import extrinsics_to_homogeneous, invert_poses, extract_intrinsics, OPENGL_TO_OPENCV; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add collab_splats/utils/__init__.py
git commit -m "feat(utils): export geometry helpers from utils __init__"
```

---

## Task 3: Move `OPENGL_TO_OPENCV` — remove from `camera_utils.py`, update `mesh_adapter.py`

**Files:**
- Modify: `collab_splats/utils/camera_utils.py` (line 287)
- Modify: `collab_splats/nerfstudio/utils/mesh_adapter.py` (line 10)

- [ ] **Step 1: Remove from `camera_utils.py`**

Delete this line (line 287):
```python
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
```

- [ ] **Step 2: Update `mesh_adapter.py` import**

Change:
```python
from collab_splats.utils.camera_utils import OPENGL_TO_OPENCV
```
To:
```python
from collab_splats.utils.geometry import OPENGL_TO_OPENCV
```

- [ ] **Step 3: Verify**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.nerfstudio.utils.mesh_adapter import extract_mesh_inputs; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Run existing tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```
Expected: same pass/fail count as before this task.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/utils/camera_utils.py collab_splats/nerfstudio/utils/mesh_adapter.py
git commit -m "refactor(utils): move OPENGL_TO_OPENCV to geometry.py"
```

---

## Task 4: Update `feedforward/base.py` — delete private fn, use public

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 1: Add import at top of file**

Add to imports section:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
```

- [ ] **Step 2: Delete `_extrinsics_3x4_to_4x4` function**

Delete the entire function (around lines 231–242):
```python
def _extrinsics_3x4_to_4x4(extrinsics_3x4: np.ndarray) -> np.ndarray:
    """Append a [0, 0, 0, 1] bottom row to convert (N, 3, 4) → (N, 4, 4).
    ...
    """
    n = extrinsics_3x4.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (n, 1, 1))  # (N, 1, 4)
    return np.concatenate([extrinsics_3x4, bottom], axis=1)                   # (N, 4, 4)
```

Also remove `_extrinsics_3x4_to_4x4` from the module docstring at the top of the file.

- [ ] **Step 3: Replace callsite in `_raw_to_world_points` (lines 293–294)**

Change:
```python
    extr_4x4 = _extrinsics_3x4_to_4x4(extr_3x4)
    cam2world = np.linalg.inv(extr_4x4.astype(np.float64)).astype(np.float32)
```
To:
```python
    extr_4x4 = extrinsics_to_homogeneous(extr_3x4)
    cam2world = invert_poses(extr_4x4.astype(np.float64)).astype(np.float32)
```

- [ ] **Step 4: Find and replace remaining `_extrinsics_3x4_to_4x4` calls in this file**

```bash
grep -n '_extrinsics_3x4_to_4x4' /workspace/collab-splats/collab_splats/pointcloud/feedforward/base.py
```
Replace any remaining calls with `extrinsics_to_homogeneous(...)`.

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py
git commit -m "refactor(feedforward): replace private _extrinsics_3x4_to_4x4 with geometry.extrinsics_to_homogeneous"
```

---

## Task 5: Update `feedforward/vggt_omega.py` — remove private import

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`

- [ ] **Step 1: Update import**

Change the import block that includes `_extrinsics_3x4_to_4x4`:
```python
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _extrinsics_3x4_to_4x4,
    _raw_to_world_points,
)
```
To:
```python
from .base import (
    BaseFeedforwardCreator,
    FeedforwardResult,
    _raw_to_world_points,
)
from collab_splats.utils.geometry import extrinsics_to_homogeneous
```

- [ ] **Step 2: Replace any `_extrinsics_3x4_to_4x4` calls in this file**

```bash
grep -n '_extrinsics_3x4_to_4x4' /workspace/collab-splats/collab_splats/pointcloud/feedforward/vggt_omega.py
```
Replace each call `_extrinsics_3x4_to_4x4(x)` with `extrinsics_to_homogeneous(x)`.

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggt_omega.py
git commit -m "refactor(feedforward): update vggt_omega to use geometry.extrinsics_to_homogeneous"
```

---

## Task 6: Update `feedforward/mapanything.py` — replace `closed_form_pose_inverse`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`

- [ ] **Step 1: Add import, remove `closed_form_pose_inverse` import**

Remove:
```python
from mapanything.utils.geometry import closed_form_pose_inverse
```
Add:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
```

- [ ] **Step 2: Replace callsite at line 65–67**

Change:
```python
        ext_4x4 = np.concatenate([extrinsics_3x4[i], [[0, 0, 0, 1]]], axis=0)  # (4, 4)
        cam2world = closed_form_pose_inverse(ext_4x4[None])[0]                   # (4, 4)
```
To:
```python
        ext_4x4 = extrinsics_to_homogeneous(extrinsics_3x4[i])   # (4, 4)
        cam2world = invert_poses(ext_4x4)                          # (4, 4)
```

- [ ] **Step 3: Replace callsite at line 215–216**

Change:
```python
            cam2world = pred["camera_poses"][0].cpu().numpy()
            extrinsics_list.append(closed_form_pose_inverse(cam2world[None])[0][:3, :4])
```
To:
```python
            cam2world = pred["camera_poses"][0].cpu().numpy()
            extrinsics_list.append(invert_poses(cam2world)[:3, :4])
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py
git commit -m "refactor(feedforward): replace closed_form_pose_inverse with geometry.invert_poses"
```

---

## Task 7: Update `pointcloud/wrappers.py` — 2 callsites

**Files:**
- Modify: `collab_splats/pointcloud/wrappers.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
```

- [ ] **Step 2: Replace line 181 (np.tile bottom row)**

Find the block around line 181:
```python
                bottom = np.tile([0, 0, 0, 1], (k, 1)).reshape(k, 1, 4).astype(np.float32)
```
This is part of a 3x4→4x4 construction. Replace the full construction with `extrinsics_to_homogeneous`. First find context:
```bash
grep -n -B3 -A3 'np.tile.*0, 0, 0, 1' /workspace/collab-splats/collab_splats/pointcloud/wrappers.py
```
Replace the multi-line 3x4→4x4 block with a single `extrinsics_to_homogeneous(...)` call.

- [ ] **Step 3: Replace line 234 (linalg.inv)**

Change:
```python
                            np.linalg.inv(lc_poses[1].astype(np.float64))
```
To:
```python
                            invert_poses(lc_poses[1].astype(np.float64))
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py
git commit -m "refactor(pointcloud): use geometry helpers in wrappers.py"
```

---

## Task 8: Update `pointcloud/utils.py` — 3 callsites

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
```

- [ ] **Step 2: Replace line 856 (np.tile bottom row)**

Find context:
```bash
grep -n -B3 -A3 'np.tile.*0, 0, 0, 1' /workspace/collab-splats/collab_splats/pointcloud/utils.py
```
Replace the inline `np.tile([0,0,0,1]...)` 3x4→4x4 construction with `extrinsics_to_homogeneous(...)`.

- [ ] **Step 3: Replace line 956 (linalg.inv)**

Change:
```python
    cam2world = np.linalg.inv(w2c)   # (N, 4, 4)
```
To:
```python
    cam2world = invert_poses(w2c)    # (N, 4, 4)
```

- [ ] **Step 4: Replace `extrinsics_to_c2w` body (line 1034)**

Change:
```python
    return [np.linalg.inv(ext) for ext in extrinsics]
```
To:
```python
    return list(invert_poses(np.asarray(extrinsics)))
```

- [ ] **Step 5: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/utils.py
git commit -m "refactor(pointcloud): use geometry helpers in utils.py"
```

---

## Task 9: Update `pointcloud/base.py` — vstack + linalg.inv

**Files:**
- Modify: `collab_splats/pointcloud/base.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses
```

- [ ] **Step 2: Replace lines 89–90 in `_colmap_recon_to_result`**

Change:
```python
        w2c = np.vstack([w2c_34, [0.0, 0.0, 0.0, 1.0]])  # (4, 4)
        c2w = np.linalg.inv(w2c)
```
To:
```python
        w2c = extrinsics_to_homogeneous(w2c_34)  # (4, 4)
        c2w = invert_poses(w2c)
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/base.py
git commit -m "refactor(pointcloud): use geometry helpers in base.py"
```

---

## Task 10: Update `wrapper/splatter.py` — inline 3x4→4x4

**Files:**
- Modify: `collab_splats/wrapper/splatter.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import extrinsics_to_homogeneous
```

- [ ] **Step 2: Replace line 726**

Change:
```python
        transform = np.concatenate([transform, np.array([0, 0, 0, 1])[np.newaxis]], axis=0)
```
To:
```python
        transform = extrinsics_to_homogeneous(transform)
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/wrapper/splatter.py
git commit -m "refactor(wrapper): use extrinsics_to_homogeneous in splatter.py"
```

---

## Task 11: Update `mesh/utils.py` — linalg.inv

**Files:**
- Modify: `collab_splats/mesh/utils.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import invert_poses
```

- [ ] **Step 2: Replace line 466**

Change:
```python
    c2w = np.linalg.inv(result.extrinsics).astype(np.float32)
```
To:
```python
    c2w = invert_poses(result.extrinsics).astype(np.float32)
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/mesh/utils.py
git commit -m "refactor(mesh): use invert_poses in mesh/utils.py"
```

---

## Task 12: Update `mesh/tsdf.py` — linalg.inv + extract_intrinsics

**Files:**
- Modify: `collab_splats/mesh/tsdf.py`

- [ ] **Step 1: Add import**

Add at top of file:
```python
from collab_splats.utils.geometry import extract_intrinsics, invert_poses
```

- [ ] **Step 2: Precompute batch inverse before the loop**

Find the loop that calls `np.linalg.inv(c2w[i])` (around line 80). Add this line before the `for i in range(N):` loop:
```python
        w2c = invert_poses(c2w)  # (N, 4, 4) — precompute all at once
```

- [ ] **Step 3: Replace per-frame linalg.inv and fx/fy/cx/cy extraction inside loop**

Change (lines 83–88 inside the loop):
```python
            fx = float(intrinsics[i, 0, 0])
            fy = float(intrinsics[i, 1, 1])
            cx = float(intrinsics[i, 0, 2])
            cy = float(intrinsics[i, 1, 2])
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            extrinsic = np.linalg.inv(c2w[i])
```
To:
```python
            fx, fy, cx, cy = extract_intrinsics(intrinsics[i])
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            extrinsic = w2c[i]
```

- [ ] **Step 4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add collab_splats/mesh/tsdf.py
git commit -m "refactor(mesh): use geometry helpers in tsdf.py"
```

---

## Task 13: Update `mesh_adapter.py` — inline 3x4→4x4

**Files:**
- Modify: `collab_splats/nerfstudio/utils/mesh_adapter.py`

- [ ] **Step 1: Add `extrinsics_to_homogeneous` to existing geometry import**

Change:
```python
from collab_splats.utils.geometry import OPENGL_TO_OPENCV
```
To:
```python
from collab_splats.utils.geometry import OPENGL_TO_OPENCV, extrinsics_to_homogeneous
```

- [ ] **Step 2: Replace inline 3x4→4x4 construction**

Change:
```python
            c2w_34 = camera.camera_to_worlds[0].cpu().numpy()
            c2w_44 = np.eye(4, dtype=np.float32)
            c2w_44[:3] = c2w_34
            c2w_44 = c2w_44 @ OPENGL_TO_OPENCV.astype(np.float32)
```
To:
```python
            c2w_34 = camera.camera_to_worlds[0].cpu().numpy()
            c2w_44 = extrinsics_to_homogeneous(c2w_34)
            c2w_44 = (c2w_44 @ OPENGL_TO_OPENCV).astype(np.float32)
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add collab_splats/nerfstudio/utils/mesh_adapter.py
git commit -m "refactor(nerfstudio): use geometry helpers in mesh_adapter.py"
```

---

## Task 14: Final verification

- [ ] **Step 1: Confirm no remaining inline patterns**

```bash
grep -rn '_extrinsics_3x4_to_4x4\|closed_form_pose_inverse\|np\.tile.*0, 0, 0, 1\|np\.vstack.*0\.0, 0\.0, 0\.0, 1' \
  /workspace/collab-splats/collab_splats --include='*.py' | grep -v __pycache__
```
Expected: no output (all replaced).

- [ ] **Step 2: Confirm no stray `np.linalg.inv` on pose matrices**

```bash
grep -rn 'linalg\.inv' /workspace/collab-splats/collab_splats --include='*.py' | grep -v __pycache__
```
Review output. Remaining `linalg.inv` calls should only be on non-pose matrices (e.g. homography `H_inv`, intrinsics matrix inversion for unprojection).

- [ ] **Step 3: Full test run**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -q 2>&1 | tail -15
```
Expected: all tests pass (same count as before).

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "refactor(utils): geometry centralization complete — all callsites updated"
```
