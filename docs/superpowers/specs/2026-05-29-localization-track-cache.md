# Localization Track Cache — Spec

**Date:** 2026-05-29
**Status:** ready-for-planning
**Predecessor:** `2026-05-29-localization-feature-cache.md` (feature cache spec — storage layout amended here)

---

## Problem

`CameraLocalizer` matches each new query against a fixed set of reconstruction frames. As more query images are localized successfully, their coverage is discarded. Later queries can't benefit from denser frame coverage accumulated in prior localizations. Additionally, session restarts lose all localized poses.

## Goal

1. After successful localization, add query frame (features + estimated pose) to the in-memory reference set so future queries can match against it.
2. Persist localized frames to `feedforward.zarr` so they survive session restarts.
3. Track provenance per frame (`"reconstruction"` vs `"localized"`) for dashboard display and selective reset.

## Non-Goals

- Triangulating new 3D points from localized frames — only existing pts3d are reprojected into new frame poses.
- Automatic invalidation when BA/LC changes reconstruction geometry — caller clears localized frames and rebuilds.
- Merging localized frame poses back into the feedforward reconstruction.

---

## Storage Layout Amendment

The existing spec (`localization-feature-cache`) stored reconstruction arrays flat under `local_features/{extractor}/`. This spec moves them into a `reconstruction/` subgroup and adds a `localized/` sibling:

```
feedforward.zarr/
  local_features/
    disk/
      reconstruction/              ← existing arrays (moved from flat layout)
        frame_offsets  (N+1,) int64
        keypoints      (M, 2) float32
        descriptors    (M, D) float32
        scores         (M,) float32    ← XFeat only; absent for DISK
        image_paths    (N,) str
        hw             (2,) int32
      localized/                   ← NEW: appended after each successful localize()
        frame_offsets  (N_loc+1,) int64
        keypoints      (M_loc, 2) float32
        descriptors    (M_loc, D) float32
        scores         (M_loc,) float32    ← XFeat only; absent for DISK
        image_paths    (N_loc,) str
        extrinsics     (N_loc, 4, 4) float32   ← stored here; not in FeedforwardResult
        intrinsics     (N_loc, 3, 3) float32
    xfeat/
      reconstruction/
        ...
      localized/
        ...
```

`hw` is not duplicated in `localized/` — localized frames must match the resolution of reconstruction frames. The shared `hw` lives in `reconstruction/`.

---

## Data Model Changes

### `LocalizationResult`

Add one field (zero extra cost — features are computed inside `localize()` regardless):

```python
@dataclass
class LocalizationResult:
    ...
    query_features: LocalFeatures | None = None
    # Always populated by localize(). Caller passes to add_localized_frame().
```

### `CameraLocalizer` in-memory state

New parallel lists alongside `_frame_features`:

```python
self._frame_sources: list[str]   # "reconstruction" or "localized"
self._image_paths: list[Path]    # source path per frame — used for duplicate guard
# Both same length as _frame_features
```

`_image_paths` is not currently stored by `CameraLocalizer.__init__` — add it during construction (populate from the `image_paths` parameter).

---

## API

### `localize()` — change

Populates `LocalizationResult.query_features` before returning. No other behaviour change.

### `add_localized_frame()`

```python
def add_localized_frame(
    self,
    image_path: str | Path,
    pose: np.ndarray,          # (4, 4) world-to-camera from LocalizationResult.pose
    intrinsics: np.ndarray,    # (3, 3) query intrinsics
    features: LocalFeatures,   # from LocalizationResult.query_features
    zarr_path: str | Path | None = None,
    extractor_name: str | None = None,
) -> None:
    """Add a successfully localized frame to the in-memory reference set.

    Rebuilds kpt→3D assignments for the new frame from existing pts3d + given pose.
    If zarr_path and extractor_name are provided, appends to localized/ group in zarr
    (resize + slice-assign). Single-writer assumption — not thread-safe across
    concurrent callers.

    Skips silently with a warning if image_path already exists in _image_paths
    (duplicate guard). Skips with a warning if features resolution mismatches hw.
    """
```

### `clear_localized_frames()`

```python
@staticmethod
def clear_localized_frames(zarr_path: str | Path, extractor_name: str) -> None:
    """Delete the localized/ group for extractor_name from feedforward.zarr.

    Reconstruction data is untouched. Call this after BA/LC updates that
    invalidate previously estimated localized poses, then reload via load_index().
    """
```

### `load_index()` — extended

Loads `reconstruction/` and `localized/` groups. Merges into single in-memory lists:

```
_frame_features = rec_features + loc_features
_frame_sources  = ["reconstruction"] * N_rec + ["localized"] * N_loc
_assignments    = rec_assignments + loc_assignments
```

Reconstruction frame assignments use FeedforwardResult geometry (pts3d, extrinsics, intrinsics) as before. Localized frame assignments use stored extrinsics/intrinsics from `localized/` group.

If `localized/` group is absent, load proceeds normally (no error).

### `save_index()` — change

Writes to `reconstruction/` subgroup instead of flat layout. No other behaviour change.

---

## `from_feedforward` cache-or-build flow (updated)

```
from_feedforward(result, extractor, zarr_path=result._zarr_path)
  │
  ├─ zarr_path AND reconstruction/ exists for extractor?
  │     └─ YES → load_index (rec + localized groups merged) → return (fast path)
  │
  └─ NO → build from scratch (GPU inference on all reconstruction frames)
           → save_index() → writes to reconstruction/
           → return
```

After any successful `localize()` call, dashboard calls `add_localized_frame()` which appends to `localized/` in-memory and to zarr.

---

## Incremental update flow (new reconstruction cameras)

When new reconstruction frames are added (e.g. scene extended):

```python
# Update reconstruction/ in zarr with new frames
localizer.update_index(
    new_image_paths=[...],
    zarr_path=zarr_path,
    extractor_name="disk",
)
# update_index does NOT auto-clear localized frames.
# Caller must clear manually — localized frame assignments depend on old geometry.
CameraLocalizer.clear_localized_frames(zarr_path, extractor_name="disk")
# Reload full index (rec + empty localized)
localizer = CameraLocalizer.load_index(zarr_path, ...)
```

---

## Dashboard integration (`localize.py`)

```python
result = localizer.localize(query_image, query_intrinsics)
if result.pose is not None:
    localizer.add_localized_frame(
        image_path=query_image_path,
        pose=result.pose,
        intrinsics=query_intrinsics,
        features=result.query_features,
        zarr_path=zarr_path,
        extractor_name=extractor_name,
    )
```

Dashboard uses `localizer._frame_sources` to visually differentiate frame types (e.g. in frame picker, colour coding in 3D view).

---

## Edge Cases

| Case | Behaviour |
|---|---|
| Same `image_path` already in index | Log warning, skip. No duplicate frames. |
| Resolution mismatch (`features` from different HxW) | Log warning, skip. |
| BA/LC update invalidates reconstruction poses | Caller calls `clear_localized_frames()` + reloads. Document in `add_localized_frame` docstring. |
| `localized/` group absent in zarr | `load_index` loads reconstruction only; no error. |
| `load_index` on zarr with flat layout (pre-migration) | Raise `KeyError` on missing `reconstruction/` subgroup with clear message: "zarr uses old flat layout — rebuild index". |

---

## Files Changed

| File | Change |
|---|---|
| `collab_splats/pointcloud/localization.py` | `LocalizationResult.query_features`; `_frame_sources`; `add_localized_frame`; `clear_localized_frames`; extend `load_index` (merge rec+loc groups); `save_index` writes to `reconstruction/` subgroup; `localize()` populates `query_features` |
| `collab_splats/pointcloud/feedforward/base.py` | `FeedforwardResult._zarr_path` set in `load_zarr` (from feature-cache spec) |
| `collab_splats/dashboard/panes/localize.py` | Call `add_localized_frame` after successful localize; use `_frame_sources` for UI provenance |
| `tests/pointcloud/test_localization.py` | Round-trip (add → reload → localize finds localized frame); duplicate guard; `clear_localized_frames`; cross-session persistence; provenance list correct; flat-layout migration error |

---

## Testing Checklist

- [ ] `add_localized_frame` + reload: second session's index includes localized frame; `localize()` returns correspondence from it
- [ ] Duplicate `image_path`: warning logged, index length unchanged
- [ ] `clear_localized_frames`: `localized/` group absent after call; `load_index` returns reconstruction-only index
- [ ] `_frame_sources` list: correct length and values after mix of add calls
- [ ] Flat-layout migration: `load_index` raises with clear message on old zarr
- [ ] `localize()` always sets `query_features` (even on failure path)

---

## Open Questions for Implementer

1. **Migration path for existing zarrs**: bump a `layout_version` attr on the zarr group (`1` = flat, `2` = subgroup), or just check for `reconstruction/` key existence?
2. **`localized/` group initialization**: create empty arrays on first `save_index()` call, or lazily on first `add_localized_frame()`? Recommend lazy (avoids empty group when no queries ever localized).
3. **`_frame_sources` exposure**: expose as public `frame_sources` property or keep private? Dashboard needs it. Recommend public property returning a copy.
