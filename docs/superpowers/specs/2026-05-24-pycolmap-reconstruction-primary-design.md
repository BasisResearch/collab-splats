# Design: pycolmap.Reconstruction as Primary Storage in PointcloudResult

**Date:** 2026-05-24  
**Status:** Draft  
**Scope:** `collab_splats/pointcloud/` — `PointcloudResult`, `FeedforwardResult`, `CameraLocalizer`, conversion utilities

---

## Problem

`PointcloudResult` currently stores camera data twice:

```python
@dataclass
class PointcloudResult:
    camera_poses: np.ndarray | None        # (M, 4, 4) c2w — extracted from pycolmap
    camera_intrinsics: np.ndarray | None   # (M, 3, 3) K — extracted from pycolmap
    colmap_reconstruction: Any | None      # pycolmap.Reconstruction — the source of truth
```

- `colmap_reconstruction` is typed `Any | None` — no type safety
- Not always populated (only after BA / COLMAP paths)
- `camera_poses` and `camera_intrinsics` are derived from `colmap_reconstruction` and then stored redundantly
- `_colmap_recon_to_result` and `colmap_reconstruction_to_result` (~100 lines combined) exist solely to unpack `pycolmap.Reconstruction` into numpy and repack it
- `colmap_reconstruction` stores in COLMAP frame; `camera_poses` stores c2w in nerfstudio frame — different conventions in the same dataclass
- `camera_intrinsics` silently encodes only the PINHOLE/linear part of the camera model; distortion parameters are dropped
- `CameraLocalizer` rebuilds a `pycolmap.Camera` from the K matrix on every query, discarding model info that was never stored

Additionally, `FeedforwardResult` and `PointcloudResult` use divergent terminology for the same concepts (`pts3d` vs `points`, `conf` vs `confidence`, `extrinsics` vs `camera_poses`, `intrinsics` vs `camera_intrinsics`), preventing shared tooling and making the API inconsistent.

---

## Goals

1. Make `pycolmap.Reconstruction` the primary and always-present camera store in `PointcloudResult`
2. Unify field terminology across `FeedforwardResult` and `PointcloudResult`
3. Eliminate conversion boilerplate (`_colmap_recon_to_result`, `colmap_reconstruction_to_result`)
4. Preserve full camera model info (distortion, non-PINHOLE models)
5. Fix `CameraLocalizer` to use stored Camera objects rather than rebuilding from K

---

## Non-Goals

- Migrating `FeedforwardResult` to pycolmap types (wrong domain — dense arrays have no pycolmap equivalent)
- Adopting `pycolmap.bundle_adjustment()` (separate decision; requires full track population)
- Adding a `HasCameras` Protocol (deferred — implement when a second tool needs it beyond CameraLocalizer)

---

## Coordinate Convention

All extrinsics in both types use **world-to-camera (w2c)** convention:

```
x_cam = E @ x_world        (homogeneous, x_world = [X, Y, Z, 1]^T)
```

- Shape: `(N, 4, 4) float32`
- Top-left `(3, 3)`: rotation matrix R (axes of camera frame in world)
- Top-right `(3, 1)`: translation t, where `t = -R @ cam_origin_in_world`
- Bottom row: `[0, 0, 0, 1]`
- Camera axes: OpenCV convention — X right, Y down, Z into scene
- `pycolmap.Rigid3d` stores the same w2c convention natively

`PointcloudResult.frame` indicates the coordinate frame of the **world origin** (not the camera axes):
- `CoordinateFrame.COLMAP` — COLMAP world (Y-down), as produced by pycolmap / feedforward models
- `CoordinateFrame.NERFSTUDIO` — nerfstudio world (+Z up), after the COLMAP→nerfstudio axis swap

The stored `pycolmap.Reconstruction` is in whichever frame `PointcloudResult.frame` declares. The `extrinsics` property reads poses directly from the Reconstruction — no silent transform is applied. Callers that need a specific frame must check `result.frame`.

---

## Intrinsics Convention

```
intrinsics[i] = K_i = [[fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]]
```

- Shape: `(N, 3, 3) float32`
- Derived from `pycolmap.Camera.params` — always the **linear/PINHOLE part only**
- **Distortion parameters are NOT captured in this matrix.** For `SIMPLE_RADIAL`, `OPENCV`, fisheye, or any non-PINHOLE model, the full `pycolmap.Camera` object in `reconstruction.cameras[camera_id]` contains the complete parameterization
- Callers needing distortion-correct projection must use `camera.cam_from_img()` directly, not the `intrinsics` property

---

## Terminology Unification

Both types expose the same names for shared concepts. The direction of change:

| Concept | FeedforwardResult (rename) | PointcloudResult (rename) |
|---|---|---|
| 3D world points | `pts3d` → **`points`** | `points` ✓ (property) |
| Point colors | `colors` ✓ | `colors` ✓ (property) |
| Per-point confidence | `conf` → **`confidence`** | `confidence` ✓ |
| Camera w2c transforms | `extrinsics` ✓ | `camera_poses` → **`extrinsics`** (property) |
| Camera K matrices | `intrinsics` ✓ | `camera_intrinsics` → **`intrinsics`** (property) |
| Image file references | `image_paths` ✓ | *(none)* → **`image_paths: list[Path]`** (stored field) |

`FeedforwardResult` renames (`pts3d→points`, `conf→confidence`) are mechanical — no logic changes. All internal uses and tests update in the same PR.

---

## Proposed `PointcloudResult`

```python
from functools import cached_property

@dataclass
class PointcloudResult:
    """Sparse reconstruction output: pycolmap.Reconstruction + scene metadata.

    reconstruction is the primary store for cameras, images, and 3D points.
    frame declares the coordinate system of the world origin in reconstruction.
    image_paths defines the canonical frame ordering for extrinsics/intrinsics.

    The four cached_property fields (points, colors, extrinsics, intrinsics) are
    computed once on first access and stored on the instance. PointcloudResult is
    treated as immutable after construction — do not mutate reconstruction afterward
    or the cache will be stale.
    """

    reconstruction: pycolmap.Reconstruction          # primary — always set
    frame: CoordinateFrame                           # coord system of world origin
    image_paths: list[Path]                          # canonical frame ordering (N entries)
    confidence: np.ndarray | None = None             # (P,) float32 — feedforward per-point
    world_transform: np.ndarray | None = None        # (3, 4) applied COLMAP→nerfstudio axis swap

    @cached_property
    def points(self) -> np.ndarray:
        """(P, 3) float32 world XYZ of the tracked sparse point set, ordered by point3D_id.

        P is the filtered sparse set — smaller than FeedforwardResult.points which
        contains all feedforward model output including untracked points.
        Computed once and cached; do not mutate reconstruction.points3D after first access.
        """
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([p.xyz for p in pts3d.values()], dtype=np.float32)

    @cached_property
    def colors(self) -> np.ndarray:
        """(P, 3) uint8 RGB, same order as points. Cached — see points docstring."""
        pts3d = self.reconstruction.points3D
        if not pts3d:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([p.color for p in pts3d.values()], dtype=np.uint8)

    @cached_property
    def extrinsics(self) -> np.ndarray:
        """(N, 4, 4) float32 w2c transforms, ordered by image_paths. Cached."""
        images = self.reconstruction.images
        name_to_image = {img.name: img for img in images.values()}
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = t
            result.append(E)
        return np.stack(result) if result else np.zeros((0, 4, 4), dtype=np.float32)

    @cached_property
    def intrinsics(self) -> np.ndarray:
        """(N, 3, 3) float32 K matrices (PINHOLE/linear part only), ordered by image_paths.

        Distortion params are NOT captured here. Access reconstruction.cameras[id]
        directly for the full pycolmap.Camera when distortion-correct projection is needed.
        Cached — see points docstring.
        """
        images = self.reconstruction.images
        cameras = self.reconstruction.cameras
        name_to_image = {img.name: img for img in images.values()}
        result = []
        for path in self.image_paths:
            img = name_to_image[path.name]
            cam = cameras[img.camera_id]
            params = cam.params   # [fx, fy, cx, cy, ...] — first 4 always fx/fy/cx/cy
            K = np.array([[params[0], 0, params[2]],
                          [0, params[1], params[3]],
                          [0,         0,          1]], dtype=np.float32)
            result.append(K)
        return np.stack(result) if result else np.zeros((0, 3, 3), dtype=np.float32)
```

All four derived properties use `cached_property` — computed once on first access, stored on the instance dict. `dataclasses.replace(result, ...)` produces a new instance with a cold cache, which is correct.

Note: `extrinsics` and `intrinsics` build a `{name: Image}` index inline rather than calling a helper — avoids rebuilding the index twice when both are accessed.

---

## What Gets Deleted

| File | Deleted |
|---|---|
| `base.py` | `_colmap_recon_to_result()` (~50 lines), `camera_poses` + `camera_intrinsics` fields |
| `utils.py` | `colmap_reconstruction_to_result()` (~55 lines) |
| `feedforward/base.py` | Import of `colmap_reconstruction_to_result`; `build_colmap()` now constructs `PointcloudResult` directly |
| `localization.py` | Per-query `pycolmap.Camera(...)` rebuild in `_localize()` |

---

## `build_colmap()` — New Flow

Currently: `build_pycolmap_reconstruction(...)` → `colmap_reconstruction_to_result(recon)` → PointcloudResult with unpacked numpy arrays.

After: `build_pycolmap_reconstruction(...)` → `PointcloudResult(reconstruction=recon, frame=..., image_paths=...)`. The Reconstruction IS the result. One step, no unpacking.

---

## `CameraLocalizer` Fix

`_localize()` currently builds:
```python
camera = pycolmap.Camera(model="PINHOLE", width=w, height=h,
                         params=[fx, fy, cx, cy])
```
from the K matrix on every query call. After this change, `CameraLocalizer` receives a `PointcloudResult` and accesses `result.reconstruction.cameras[id]` directly — the Camera object is already there, with its full model and params.

---

## Frame Ordering

`image_paths: list[Path]` is the canonical ordering. All N-length arrays produced by `extrinsics` and `intrinsics` properties are ordered to match `image_paths[i]`.

`build_pycolmap_reconstruction` already adds images in frame order; `image_paths` captures that ordering explicitly at construction time. On disk-load (via `reconstruction.read()`), `image_paths` is restored from the stored list alongside the COLMAP binary files.

---

## What Does NOT Change

- `FeedforwardResult` stores remain numpy/zarr — dense arrays (depth, conf, world_points, features) have no pycolmap equivalent
- `BundleAdjustment.refine()` input/output stays `FeedforwardResult` — BA operates in numpy/torch domain
- `build_pycolmap_reconstruction()` function stays — still the numpy→pycolmap conversion point, called inside `build_colmap()`
- `CoordinateFrame` enum unchanged
- `world_transform` unchanged (records applied axis swap for nerfstudio compatibility)

---

## Migration Surface

| Location | Change |
|---|---|
| `base.py` | Rewrite `PointcloudResult`; delete `_colmap_recon_to_result` |
| `utils.py` | Delete `colmap_reconstruction_to_result`; update imports |
| `feedforward/base.py` | Rename `pts3d→points`, `conf→confidence`; update `build_colmap()` |
| `feedforward/vggtx.py` | Rename `pts3d→points`, `conf→confidence` |
| `feedforward/vggt_omega.py` | Same renames |
| `pointcloud/__init__.py` | Update exports; remove `_colmap_recon_to_result` export |
| `localization.py` | Fix `_localize()` to use stored Camera; update constructor |
| `bundle_adjustment.py` | Rename field references (`conf→confidence`) |
| `tests/` | Update field names; update `PointcloudResult` construction in fixtures |
| `docs/` notebooks | `camera_poses.shape[0]` → `len(result.reconstruction.images)` |

---

## Risks

1. **`intrinsics` property vs non-PINHOLE models** — for any camera model with distortion, the `intrinsics` property silently drops distortion. This is the same behavior as today (storing K matrix). Risk: callers that think they're doing correct projection but aren't. Mitigation: docstring on the property is explicit; the full Camera is always available via `reconstruction.cameras`.

2. **`points`/`colors`/`extrinsics`/`intrinsics` cost** — all four use `cached_property`: O(P) or O(N) work on first access only, O(1) thereafter. `dataclasses.replace()` produces a new instance with cold cache — callers that replace fields and immediately re-access will pay the cost once.

3. **`confidence` alignment** — `confidence` is `(P,)` indexed by position in `image_paths` ordering, but `points`/`colors` are indexed by `point3D_id` ordering. These are not the same ordering. This misalignment pre-exists and is out of scope — tracked separately.
