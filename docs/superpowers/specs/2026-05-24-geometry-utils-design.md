# Geometry Utilities Centralization

**Date:** 2026-05-24
**Status:** Approved

## Problem

Repeated inline implementations of the same geometric operations scattered across the codebase:

- `extrinsics_to_homogeneous` (3x4→4x4): 6+ callsites, each with a different inline variant (`np.tile`, `np.vstack`, `np.eye(4)`, `np.concatenate`)
- `invert_poses` (w2c↔c2w): 7+ `np.linalg.inv` callsites, plus a list-comprehension wrapper `extrinsics_to_c2w` in `pointcloud/utils.py`
- `extract_intrinsics` (K→fx,fy,cx,cy): inline `K[0,0], K[1,1], K[0,2], K[1,2]` pattern at multiple sites
- `OPENGL_TO_OPENCV` constant: defined in `camera_utils.py`, imported by `mesh_adapter.py`, with ad-hoc alternatives elsewhere

## Solution

New module `collab_splats/utils/geometry.py` — pure numpy, no heavy deps, no nerfstudio coupling.

## Public API

### Constants

```python
# Camera axis convention flip (OpenCV ↔ OpenGL). Self-inverse: same matrix both directions.
# OpenCV:  X right, Y down,  Z forward (COLMAP, VGGT-X, BA)
# OpenGL:  X right, Y up,    Z backward (nerfstudio, splats)
OPENGL_TO_OPENCV: np.ndarray  # diag(1, -1, -1, 1), float64
OPENCV_TO_OPENGL: np.ndarray  # same matrix (self-inverse alias)
```

### Functions

```python
def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    """Append [0,0,0,1] row to convert (N,3,4)→(N,4,4) or (3,4)→(4,4).

    Handles both batched and single-frame inputs. Output dtype matches input.
    """

def invert_poses(poses: np.ndarray) -> np.ndarray:
    """Closed-form SE3 inverse: (...,4,4) → (...,4,4).

    Works on any leading batch shape: (4,4), (N,4,4), (B,N,4,4).
    Uses R^T, -R^T@t — numerically exact for valid rotation matrices,
    faster than np.linalg.inv (no LAPACK solver, no pivoting).
    Assumes poses are valid rigid-body transforms (orthogonal R block).
    """

def extract_intrinsics(K: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) from a (3,3) camera intrinsics matrix."""
```

## Callsite Map

### `extrinsics_to_homogeneous` (6 callsites)

| File | Current pattern | Action |
|---|---|---|
| `pointcloud/feedforward/base.py:240` | `_extrinsics_3x4_to_4x4` (private) | replace + delete private fn |
| `pointcloud/feedforward/mapanything.py:66` | `np.concatenate([..., [[0,0,0,1]]])` in loop | replace |
| `pointcloud/wrappers.py:181` | `np.tile([0,0,0,1], ...)` | replace |
| `pointcloud/utils.py:856` | `np.tile(np.array([[0,0,0,1]]))` | replace |
| `wrapper/splatter.py:726` | `np.concatenate([..., np.array([0,0,0,1])[np.newaxis]])` | replace |
| `nerfstudio/utils/mesh_adapter.py` | `np.eye(4); c2w_44[:3] = c2w_34` | replace |

### `invert_poses` (7 callsites)

| File | Current pattern | Action |
|---|---|---|
| `pointcloud/base.py:90` | `np.linalg.inv(w2c)` | replace |
| `pointcloud/feedforward/base.py:277` | `np.linalg.inv(extr_4x4.astype(np.float64)).astype(np.float32)` | replace |
| `pointcloud/utils.py:956` | `np.linalg.inv(w2c)` | replace |
| `pointcloud/utils.py:1034` (`extrinsics_to_c2w`) | `[np.linalg.inv(ext) for ext in extrinsics]` | replace body with `invert_poses`; keep public signature (callers exist) |
| `pointcloud/wrappers.py:234` | `np.linalg.inv(lc_poses[1].astype(np.float64))` | replace |
| `mesh/utils.py:466` | `np.linalg.inv(result.extrinsics).astype(np.float32)` | replace |
| `mesh/tsdf.py:88` | `np.linalg.inv(c2w[i])` (per-frame loop) | replace with batched call before loop |

### `extract_intrinsics` (2 callsites)

| File | Current pattern | Action |
|---|---|---|
| `mesh/tsdf.py:83-87` | `fx=K[0,0]; fy=K[1,1]; cx=K[0,2]; cy=K[1,2]` per-frame in loop | replace |
| `pointcloud/feedforward/base.py` | same inline pattern | replace where present |

### `OPENGL_TO_OPENCV` (1 callsite)

| File | Action |
|---|---|
| `utils/camera_utils.py:287` | remove definition |
| `nerfstudio/utils/mesh_adapter.py:10` | update import to `collab_splats.utils.geometry` |

No re-export from `camera_utils.py`.

## What Is NOT Changing

- `camera_utils.py` structure — nerfstudio-coupled helpers (`ColmapCamera`, `convert_to_colmap_camera`, `depth_double_to_normal`, `build_rotation`, torch projection functions) stay as-is
- `intrinsics_to_o3d` — not added; o3d dep stays local to `mesh/`
- `convert_camera_convention` / `convert_world_frame` — not added; too few callsites to justify
- `CoordinateFrame` enum in `pointcloud/base.py` — stays there, load-bearing

## Implementation Notes

### `extrinsics_to_homogeneous` shape handling

```python
def extrinsics_to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    single = extrinsics.ndim == 2          # (3,4) → (4,4)
    if single:
        extrinsics = extrinsics[np.newaxis]  # (1,3,4)
    n = extrinsics.shape[0]
    bottom = np.tile(np.array([[0, 0, 0, 1]], dtype=extrinsics.dtype), (n, 1, 1))
    out = np.concatenate([extrinsics, bottom], axis=1)  # (N,4,4)
    return out[0] if single else out
```

### `invert_poses` — batched closed-form

```python
def invert_poses(poses: np.ndarray) -> np.ndarray:
    R = poses[..., :3, :3]
    t = poses[..., :3, 3:]
    R_inv = np.swapaxes(R, -1, -2)
    t_inv = -(R_inv @ t)
    out = np.zeros_like(poses)
    out[..., :3, :3] = R_inv
    out[..., :3, 3:] = t_inv
    out[..., 3, 3]   = 1.0
    return out
```

## `utils/__init__.py` Exports

Add `extrinsics_to_homogeneous`, `invert_poses`, `extract_intrinsics`, `OPENGL_TO_OPENCV`, `OPENCV_TO_OPENGL` to `collab_splats/utils/__init__.py`.

## Testing

- Unit tests in `tests/utils/test_geometry.py`
- `extrinsics_to_homogeneous`: single + batched, dtype preservation
- `invert_poses`: `T @ invert_poses(T) ≈ eye(4)` for random rigid transforms; (4,4), (N,4,4), (B,N,4,4) shapes
- `extract_intrinsics`: round-trip against known K matrix
- No integration tests needed — existing tests cover callsite behavior

## Files Created / Modified

**New:** `collab_splats/utils/geometry.py`, `tests/utils/test_geometry.py`

**Modified:** `pointcloud/feedforward/base.py`, `pointcloud/feedforward/mapanything.py`, `pointcloud/wrappers.py`, `pointcloud/utils.py`, `pointcloud/base.py`, `wrapper/splatter.py`, `nerfstudio/utils/mesh_adapter.py`, `mesh/utils.py`, `mesh/tsdf.py`, `utils/camera_utils.py`, `utils/__init__.py`
