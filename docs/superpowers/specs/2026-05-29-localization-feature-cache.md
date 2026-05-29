# Localization Feature Cache — Spec

**Date:** 2026-05-29  
**Status:** ready-for-planning  
**Predecessor:** `2026-05-29-localize-tab-redesign.md` (Approach C, deferred)

## Problem

`CameraLocalizer.__init__` reads every reference frame from disk and runs local feature extraction (XFeat/DISK) for each one. This is O(N) GPU inference at query time — ~30 s for 20 frames, several minutes for 100+ frames. It re-runs on every dashboard session start.

Two sub-problems:
1. **Cold start**: features re-extracted every time, even though images haven't changed.
2. **Incremental update**: adding new cameras to an existing scene requires re-indexing all frames from scratch.

## Goal

Cache extracted local features (keypoints + descriptors) in `feedforward.zarr` alongside existing reconstruction data. Subsequent `CameraLocalizer` builds load from zarr (~1 s) instead of running GPU inference. When new frames are added, only extract features for the new frames.

## Non-Goals

- Caching `_assignments` (kpt→3D maps) — these depend on pts3d/extrinsics/intrinsics which change after BA/LC. Rebuild assignments from cached features + current geometry. Cost is negligible (pure numpy).
- Caching global retrieval descriptors (DinoSalad, PECLIP) — not used in exhaustive matching path.
- Automatic cache invalidation when images change — caller responsibility; document clearly.

---

## Design

### Storage format

Under `feedforward.zarr`, new group `local_features/{extractor_name}/`:

```
feedforward.zarr/
  local_features/
    disk/                       ← extractor name key
      frame_offsets  (N+1,) int64   ← CSR-style: frame i → rows [offsets[i]:offsets[i+1]]
      keypoints      (M, 2) float32 ← all keypoints concatenated
      descriptors    (M, D) float32 ← all descriptors concatenated
      scores         (M,) float32   ← XFeat only; absent or all-zeros for DISK
      image_paths    (N,) str       ← absolute paths at cache-build time (for staleness check)
      hw             (2,) int32     ← [H, W] of first frame (CameraLocalizer._image_hw)
    xfeat/
      ...
```

CSR layout avoids ragged arrays. Reconstruction over extractor name means different extractors don't conflict and old caches auto-ignored when extractor changes.

### API additions to `CameraLocalizer`

```python
def save_index(self, zarr_path: str | Path, extractor_name: str) -> None:
    """Persist extracted frame features to feedforward.zarr for fast reload."""

@classmethod
def load_index(
    cls,
    zarr_path: str | Path,
    extractor_name: str,
    pts3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    config: dict | None = None,
) -> "CameraLocalizer":
    """Load features from zarr; rebuild kpt→3D assignments from current geometry.

    Raises KeyError if extractor_name cache not found (caller falls back to build).
    """

@classmethod
def from_feedforward(cls, result, extractor=None, progress_callback=None, **kwargs) -> "CameraLocalizer":
    """Existing classmethod — extended to try zarr cache first.

    New kwarg: zarr_path (Path | None). If provided and cache hit:
      → load_index (fast). If miss: build + save_index.
    """

def update_index(
    self,
    new_image_paths: list,
    zarr_path: str | Path,
    extractor_name: str,
    progress_callback=None,
) -> None:
    """Extract features for new frames only; append to zarr cache.

    Caller is responsible for updating pts3d/extrinsics/intrinsics on self
    and calling _build_frame_assignments() to rebuild assignment maps.
    """
```

### `from_feedforward` cache-or-build flow

```
from_feedforward(result, extractor, zarr_path=result._zarr_path)
  │
  ├─ zarr_path provided AND local_features/{extractor_name} exists?
  │     └─ YES → load_index (keypoints/descriptors from zarr; rebuild assignments)
  │              → return (fast path, ~1 s)
  │
  └─ NO → existing build path (GPU inference, O(N) frames)
           → save_index() after build
           → return
```

### `FeedforwardResult` change

Add `_zarr_path: Path | None = None` to `FeedforwardResult` — set in `load_zarr()` so callers don't need to pass it explicitly. `from_feedforward` reads `result._zarr_path` if `zarr_path` kwarg not given.

### Incremental update flow (new cameras)

```python
# New frames appended to a scene after initial indexing:
localizer.update_index(
    new_image_paths=[path_to_frame_100, path_to_frame_101, ...],
    zarr_path=zarr_path,
    extractor_name="disk",
)
# Caller must also:
#   localizer._pts3d = updated_pts3d
#   localizer._extrinsics = updated_extrinsics
#   localizer._intrinsics = updated_intrinsics
#   localizer._assignments = _build_frame_assignments(...)
```

`update_index` appends to existing `frame_offsets`, `keypoints`, `descriptors`, `scores`, `image_paths` arrays in zarr. Uses zarr `resize` + slice assignment.

---

## Files changed

| File | Change |
|---|---|
| `collab_splats/pointcloud/localization.py` | `save_index`, `load_index`, `update_index` on `CameraLocalizer`; extend `from_feedforward` |
| `collab_splats/pointcloud/feedforward/base.py` | `FeedforwardResult.load_zarr` sets `_zarr_path` |
| `collab_splats/dashboard/panes/localize.py` | `_build_localizer` passes `zarr_path` to `from_feedforward` |
| `tests/pointcloud/test_localization.py` | Tests for `save_index`, `load_index`, `update_index`, cache-or-build round-trip |

---

## Cache invalidation

The cache is **not** automatically invalidated when source images change. Document this clearly in `save_index` docstring. Staleness check: compare `image_paths` array in zarr against current `result.image_paths` — log a warning if they differ. Do not raise; let caller decide.

Geometry changes (BA/LC update pts3d/extrinsics) do **not** invalidate the feature cache — only `_assignments` need rebuilding, which `load_index` does from current geometry.

---

## Open questions for implementer

1. **zarr compression for features**: features are float32 dense arrays — Blosc+LZ4 or raw? Recommend Blosc+LZ4 (fast decompress, ~2× compression ratio).
2. **scores absent for DISK**: store as zeros or skip the array entirely? Recommend skip (check for key existence in load).
3. **`FeedforwardResult._zarr_path`**: private attr on dataclass — add to `__post_init__` or set externally in `load_zarr`? Recommend set in `load_zarr` after construction (avoids dataclass field pollution).
4. **Thread safety for `save_index`**: dashboard calls `_build_localizer` in a background thread. zarr writes are not thread-safe across processes but single-thread write is fine. Document.

---

## Testing checklist

- [ ] `save_index` + `load_index` round-trip: loaded `CameraLocalizer` produces identical `localize()` results as built-from-scratch
- [ ] `from_feedforward` with cache hit: no GPU inference (mock extractor, assert `extract` not called)
- [ ] `from_feedforward` with cache miss: build + save, second call is cache hit
- [ ] `update_index`: new frames appended; re-localization finds query in new frame region
- [ ] Staleness warning logged when `image_paths` mismatch
- [ ] `load_index` raises `KeyError` on missing extractor_name (triggers fallback)
