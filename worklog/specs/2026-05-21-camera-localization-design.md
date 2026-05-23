# Camera Localization Design

**Date:** 2026-05-21
**Status:** ready-for-implementation

## Goal

Complete `collab_splats/pointcloud/localization.py` so it:
1. Meets project code standards (logging, section dividers, docstrings)
2. Implements Stage 2 — local feature extraction + matching
3. Implements Stage 3 — PnP pose estimation
4. Exposes `CameraLocalizer`: given a reconstructed scene + a query image → camera pose

## Context

`localization.py` already has Stage 1 (`BaseRetrievalExtractor` + `DinoSaladExtractor`). Stages 2
and 3 are empty comment stubs. The file is otherwise structurally correct but missing `logging`.

The pipeline is a post-reconstruction utility — called after `reconstruct()` to answer "where is
this new camera in the scene?" It is not part of the reconstruction hot path.

## Non-Goals

- H5 / persistent feature database (future spec)
- Semantic feature descriptors (DINOv2 patches) for matching — not precise enough geometrically
- Dependency on hloc for any stage
- Modifications to `FeedforwardResult`, `BaseFeedforwardCreator`, or loop closure

## Dependencies

All zero new installs. Already in environment or vendored:

| Component | Source | Purpose |
|---|---|---|
| `kornia.feature.DISK` | kornia 0.8.1 (installed) | Local feature extraction |
| `kornia.feature.LightGlue` | kornia 0.8.1 (installed) | Learned feature matching |
| `modules.xfeat.XFeat` | vendor/xfeat/ (clone) | Fast alternative extractor |
| `kornia.feature.match_mnn` | kornia 0.8.1 (installed) | MNN matching for XFeat |
| `cv2.solvePnPRansac` | OpenCV 4.6.0 (installed) | PnP pose solver |
| `vendor/salad/` | existing | Stage 1 retrieval (unchanged) |

## Architecture

### Code quality fixes (Stage 1 — existing)

- Add `import logging` + `logger = logging.getLogger(__name__)`
- Move `T.Compose` construction out of `DinoSaladExtractor.forward()` — build once in `__init__`
- Add `logger.debug` / `logger.info` calls at key steps

### Stage 2 — Local feature extraction + matching

Two extractors. Each owns its full detect→match logic so pairing is never ambiguous.

**`DiskExtractor`**

Wraps `kornia.feature.DISK.from_pretrained('depth')` + `kornia.feature.LightGlue(features='disk')`.

```
extract(image: np.ndarray) → (keypoints: Tensor (N,2), descriptors: Tensor (N,128))
match(feats_query, feats_db, image_size) → matches: Tensor (M,2)   # index pairs
```

Weights download once to torch hub cache on first use (~4 MB).

**`XFeatExtractor`**

Wraps vendored `vendor/xfeat/` (`accelerated_features.XFeat`) + `kornia.feature.match_mnn`.

```
extract(image: np.ndarray) → (keypoints: Tensor (N,2), descriptors: Tensor (N,64))
match(feats_query, feats_db, image_size=None) → matches: Tensor (M,2)
```

Vendor path injected via `sys.path.insert` guard (same pattern as SALAD).

### 2D→3D assignment builder (new — key logic)

At `CameraLocalizer.__init__`, for each reference frame `i`:

1. Project `pts3d` into frame `i`:
   `p2d = K_i @ (R_i @ pts3d.T + t_i)` → normalize → `(P, 2)` pixel coords
2. Filter points behind camera (`p2d[:, 2] > 0`) and outside image bounds
3. Extract DISK keypoints for frame `i` from `image_paths[i]`
4. For each keypoint, find nearest projected 3D point within `radius` pixels (`torch.cdist` NN search)
5. Store `kpt_idx → pt3d_idx` assignment map for frame `i`

This gives the lookup needed at query time: matched db keypoint → 3D world point.

### Stage 3 — PnP

```python
cv2.solvePnPRansac(
    objectPoints=pts3d_matched,   # (K, 3) float32
    imagePoints=kpts_query,       # (K, 2) float32
    cameraMatrix=intrinsics,      # (3, 3)
    distCoeffs=None,
    reprojectionError=8.0,
    confidence=0.999,
    iterationsCount=1000,
)
→ rvec, tvec  →  4×4 world-to-camera  (same convention as FeedforwardResult.extrinsics)
```

Returns `None` on failure (insufficient inliers or PnP error).

### `CameraLocalizer`

```python
CameraLocalizer(
    pts3d:       np.ndarray,   # (P, 3) world-space points
    extrinsics:  np.ndarray,   # (N, 4, 4) world-to-camera
    intrinsics:  np.ndarray,   # (N, 3, 3) K per frame
    image_paths: list[Path],   # length N — source images
    extractor=None,            # DiskExtractor() default
)
```

No global retrieval step. `DinoSaladExtractor` is intentionally excluded: retrieval could silently
exclude frames with strong keypoint overlap if their global appearance differs from the query
(viewpoint shift, lighting change). For the scene sizes this pipeline targets (50–300 frames),
matching against all frames via LightGlue is fast enough (~200ms) and more robust.

**`__init__`** (expensive — run once per scene):
- Load extractor (DISK+LightGlue by default)
- Extract local features for all N frames
- Build 2D→3D assignment maps (one per frame)

**`.localize(query_image, query_intrinsics) → np.ndarray | None`** (fast — run per query):
1. Extract local features from query image
2. For each of N reference frames: LightGlue match → collect 2D(query)↔3D(world) pairs
3. If same 3D point matched from multiple frames, keep highest LightGlue confidence score
4. `cv2.solvePnPRansac` on all collected pairs → `(4, 4)` world-to-camera, or `None` on failure

**`.from_feedforward(result: FeedforwardResult, **kwargs) → CameraLocalizer`** classmethod.

## File structure

Everything stays in `localization.py`. No new files. Section layout:

```
########  Stage 1: Global retrieval          (existing — minor fixes)
########  Stage 2: Local feature extraction  (new: DiskExtractor, XFeatExtractor)
########  Stage 3: Pose estimation           (new: _build_2d3d_assignments, CameraLocalizer)
```

## Vendor step

```bash
git clone https://github.com/verlab/accelerated_features vendor/xfeat
```

Update `vendor/README.md` with XFeat entry.

## Testing

`tests/pointcloud/test_localization.py` (new file, flat functions):

- `test_disk_extractor_returns_keypoints_and_descriptors` — smoke test with random image tensor (`@pytest.mark.slow` — downloads weights ~4 MB)
- `test_xfeat_extractor_returns_keypoints_and_descriptors` — same (`@pytest.mark.slow`)
- `test_camera_localizer_registry` — `BaseRetrievalExtractor.get("dino-salad")` unchanged
- `test_2d3d_assignment_projects_correctly` — synthetic scene, verify assignments with known geometry
- `test_localize_recovers_known_pose` — synthetic scene: build localizer from N frames, query with
  frame N+1 (known pose), verify returned pose within tolerance

Synthetic scene (no GPU, no downloads): random `pts3d`, analytically computed extrinsics/intrinsics,
rendered keypoints at known projected positions. Avoids pretrained model weight downloads in CI.

## Open questions

None. Design is settled. Proceed to implementation.
