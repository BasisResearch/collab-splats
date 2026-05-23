# Spec: PyPose BAE Bundle Adjustment

**Date:** 2026-05-06
**Branch:** `refactor/core-modules`
**Status:** Ready for implementation

---

## Context

After VGGT-X or MapAnything forward pass, poses and 3D structure are good but not jointly
optimised — each model predicts per-frame depth and poses independently (or with limited
cross-frame coupling). Bundle adjustment minimises reprojection error across all frames
jointly to refine extrinsics, intrinsics, and 3D structure.

PyPose BAE (`pip install bae`) provides PyTorch-native sparse BA with LM + GPU sparse
linear algebra — 18–23× faster than GTSAM/g2o/Ceres. It has an existing VGGT integration
pattern (`prepare_bae` in the VGGT fork) that we adapt here.

**Track source:** `predict_tracks` from `vggt.dependency.track_predict` — VGGSfM tracker
using ALIKED+SP keypoints. Independent from both backends' pose predictions. Works for
VGGT-X (images in `raw_outputs["images"]`) and MapAnything (images in `pred["img_no_norm"]`
per frame). HLOC callers provide tracks directly.

**World points re-derivation:** after BA refines extrinsics + intrinsics, world_points are
re-derived from camera-frame geometry (pose-independent) at the same density as today:
- VGGT-X: re-run `unproject_and_filter_points` with refined poses
- MapAnything: apply refined `cam2world[i]` to per-frame `pts3d_cam`, re-apply mask, voxel downsample

---

## Design

### New: `bundle_adjustment.py`

New file: `collab_splats/pointcloud/bundle_adjustment.py`

#### `extract_tracks_vggsfm`

```python
def extract_tracks_vggsfm(
    images: torch.Tensor,          # (N, 3, H, W) float — normalised RGB
    conf: torch.Tensor | None,     # (N, H, W) confidence scores
    world_points: np.ndarray | None,  # (N, H, W, 3) for confidence-guided sampling
    *,
    max_query_pts: int = 2048,
    query_frame_num: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict cross-frame 2D tracks via VGGSfM tracker.

    Returns:
        tracks:     (N, P, 2) float32 — 2D pixel coords per frame per point
        vis_scores: (N, P)    float32 — visibility score ∈ [0, 1]
        pts3d:      (P, 3)    float32 — world-space 3D positions at keypoints
    """
```

Wraps `vggt.dependency.track_predict.predict_tracks(images, conf, points_3d, ...)`.
Returns numpy arrays. `pts3d` is sampled from `world_points` at keypoint pixel locations
using `world_points_conf > 1.2` gate (matching upstream behaviour).

#### `run_bundle_adjustment`

```python
def run_bundle_adjustment(
    points3d: np.ndarray,     # (P, 3)   initial 3D point positions
    extrinsics: np.ndarray,   # (N, 3, 4) world2cam [R|t]
    intrinsics: np.ndarray,   # (N, 3, 3) camera intrinsics K
    tracks: np.ndarray,       # (N, P, 2) 2D pixel observations
    vis_mask: np.ndarray,     # (N, P)    bool visibility mask
    image_size: tuple[int, int],  # (H, W)
    *,
    max_reproj_error: float | None = 4.0,
    lm_steps: int = 40,
    shared_camera: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BAE LM bundle adjustment.

    Returns:
        refined_points3d:  (P, 3)    float64
        refined_extrinsics: (N, 3, 4) float32
        refined_intrinsics: (N, 3, 3) float32
    """
```

Internal implementation follows the VGGT BAE fork pattern:

1. Filter by `max_reproj_error` (drop observations with initial reprojection > threshold)
2. Deduplicate landmarks + keyframes (`unique_landmark`, `unique_keyframe`)
3. Build `ReprojNonBatched(cameras, points3d)` — `nn.Module` with `pp.Parameter` for poses and `nn.Parameter` for points
4. Build sparse index tensors `point_indices`, `camera_indices`, `observations`
5. `strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)`
6. `optimizer = LM(model, strategy=strategy, solver=PCG(), reject=10)`
7. `scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=lm_steps, patience=3, decreasing=1e-3, verbose=False)`
8. `scheduler.optimize(input=input)`
9. Extract refined poses → `(N, 3, 4)` and refined 3D points → `(P, 3)`

Camera parameterisation: `pp.mat2SE3(extrinsics_tensor)` concatenated with focal length
`f = (K[0,0] + K[1,1]) / 2`. Shared intrinsics when `shared_camera=True` (weighted mean init).

---

### Changes to `feedforward.py`

#### `VGGTXCreator`

Add config field:
```python
use_ba: bool = False
```

In `_postprocess`, after `unproject_and_filter_points` produces initial world_points:

```python
if self.use_ba:
    from .bundle_adjustment import extract_tracks_vggsfm, run_bundle_adjustment

    # images: (N, 3, H, W) already in raw_outputs["images"]
    # conf:   (N, H, W) depth_conf
    # world_points_per_frame: (N, H, W, 3) — unmasked dense unprojection of raw depth
    #   derived via _raw_to_world_points(raw) before conf threshold masking
    tracks, vis, pts3d_kp = extract_tracks_vggsfm(
        raw_outputs["images"],
        raw_outputs["depth_conf"],
        world_points_per_frame,
    )
    pts3d_kp, extrinsic, intrinsic = run_bundle_adjustment(
        pts3d_kp, extrinsic, intrinsic, tracks, vis,
        image_size=(model_h, model_w),
    )
    # Re-derive world_points with refined poses
    pts3d, colors = unproject_and_filter_points(
        depth=raw_outputs["depth"],
        depth_conf=raw_outputs["depth_conf"],
        images=raw_outputs["images"],
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        conf_threshold=self.conf_threshold,
    )
```

#### `MapAnythingCreator`

Add config field:
```python
use_ba: bool = False
```

In `_postprocess`, after initial `collect_pts3d_from_outputs` + voxel downsample:

```python
if self.use_ba:
    from .bundle_adjustment import extract_tracks_vggsfm, run_bundle_adjustment

    # Reconstruct (N, 3, H, W) images tensor from per-frame img_no_norm
    images = torch.stack([
        torch.from_numpy(p["img_no_norm"][0]).permute(2, 0, 1)
        for p in raw_outputs
    ])  # (N, 3, H, W)
    conf = torch.stack([p["conf"][0] for p in raw_outputs])  # (N, H, W)
    world_points_per_frame = np.stack([
        p["pts3d"][0].cpu().numpy() for p in raw_outputs
    ])  # (N, H, W, 3) — approx, used only for track sampling

    tracks, vis, pts3d_kp = extract_tracks_vggsfm(images, conf, world_points_per_frame)
    pts3d_kp, extrinsics, intrinsics = run_bundle_adjustment(
        pts3d_kp, extrinsics, intrinsics, tracks, vis,
        image_size=images.shape[-2:],
    )
    # Re-derive world_points: apply refined cam2world to per-frame pts3d_cam
    pts3d, colors = _reproject_mapanything(raw_outputs, extrinsics)
    # voxel downsample pts3d + colors (same density as today)
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pts3d)
    _pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    _pcd, _ = voxel_downsample(_pcd)
    pts3d = np.asarray(_pcd.points, dtype=np.float32)
    colors = (np.asarray(_pcd.colors) * 255).astype(np.uint8)
```

New private helper `_reproject_mapanything(raw_outputs, refined_extrinsics)` in `_mapanything.py`:
- For each frame `i`: `pts3d_cam[i]` (camera-frame points from `pred["pts3d_cam"][0]`)
- Compute `refined_cam2world[i] = inv(refined_extrinsics[i])` via `closed_form_pose_inverse`
- Apply mask (`pred["mask"][0]` + `depth_z > 0`)
- Transform `pts3d_cam[i][mask]` → world frame
- Concatenate all frames → `(P, 3)` + `colors (P, 3)`

---

### Changes to `setup.sh`

```bash
# After existing pip installs in nerfstudio env block:
pip install bae
```

---

### Changes to `__init__.py`

```python
from .bundle_adjustment import run_bundle_adjustment
```

---

## Files changed

| File | Change |
|------|--------|
| `collab_splats/pointcloud/bundle_adjustment.py` | New — `extract_tracks_vggsfm`, `run_bundle_adjustment` |
| `collab_splats/pointcloud/_mapanything.py` | Add `_reproject_mapanything` helper |
| `collab_splats/pointcloud/feedforward.py` | `VGGTXCreator` + `MapAnythingCreator`: add `use_ba: bool = False`; call extract→BA→re-derive |
| `collab_splats/pointcloud/__init__.py` | Export `run_bundle_adjustment` |
| `setup.sh` | `pip install bae` |

---

## Unit tests

| Test | What it checks |
|------|----------------|
| `test_run_bundle_adjustment_reduces_reproj_error` | Construct synthetic scene with known poses + noise; verify reprojection error decreases |
| `test_extract_tracks_vggsfm_shape` | Mock `predict_tracks`; verify output shapes `(N,P,2)`, `(N,P)`, `(P,3)` |
| `test_reproject_mapanything_uses_refined_extrinsics` | Verify re-projected pts3d differ from original when extrinsics change |
| `test_vggtxcreator_use_ba_updates_world_points` | Integration: run `VGGTXCreator._postprocess` with `use_ba=True` on dummy raw_outputs; check world_points and extrinsics changed |

---

## Out of scope

- HLOC track source (caller provides tracks directly to `run_bundle_adjustment`)
- `TrackSource` enum / configurable track extraction method (single VGGSfM source for now)
- Iterative BA (single-pass LM only)
- Radial distortion support (SIMPLE_PINHOLE only)
- MapAnything world_points density change post-BA (same voxel params as pre-BA)
- Re-running MapAnything model forward pass after BA

---

## Future investigation

**Unify point cloud derivation across backends.**
`VGGTXCreator` uses `unproject_and_filter_points` (depth + conf → masked pts3d) and
`MapAnythingCreator` uses `collect_pts3d_from_outputs` (iterate per-frame dicts → pts3d).
Both do: apply mask, extract colors, invert pose, concatenate frames, voxel downsample.
The BA re-derivation step (`_reproject_mapanything` + re-running `unproject_and_filter_points`)
makes this duplication more visible. A unified `derive_world_points(frames, extrinsics, intrinsics, ...)`
abstraction could serve both backends and simplify the BA integration point.
Investigate before the next major refactor of `feedforward.py`.
