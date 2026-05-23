# DataManager Feature Cache Path Invalidation Fix

**Date:** 2026-05-07  
**Status:** Ready for implementation

## Problem

Loading a `rade-features` model from a notebook (or any directory other than where training ran) causes `FeatureSplattingDataManager` to re-extract features for all images on every load. This is slow (minutes per load) and wasteful.

## Root Causes

### Primary: `test_mode="inference"` changes `image_filenames` paths

`FullImageDatamanager.__init__` (nerfstudio) does:

```python
if test_mode == "inference":
    self.dataparser.downscale_factor = 1  # Avoid opening images
```

This forces `downscale_factor=1`, which changes the paths nerfstudio returns in `image_filenames`. Nerfstudio's `_get_fname` returns `data_dir / "images_2" / filepath.name` when downscale=2, but `data_dir / filepath` when downscale=1. Result:

- Training (`test_mode="val"`): `image_filenames[0]` = `/data/preproc/images_2/frame_001.jpg`
- Notebook load (`test_mode="inference"`): `image_filenames[0]` = `/data/preproc/images/frame_001.jpg`

The cache comparison at `features.py:121` does a list equality check on full `Path` objects — this fails on every load where images were auto-downscaled, printing "Image filenames have changed, cache invalidated..." and re-extracting all features.

### Secondary: Relative `data` path in `config.yml`

If `Splatter` was invoked with a relative `file_path`, `preproc_data_path` is relative. `ns-train` stores it as-is in `config.yml`. When loaded from a different CWD:

```python
cache_dir = self.config.dataparser.data  # e.g. Path("preproc") — relative
cache_path = cache_dir / "feature-splatting_samclip-features.pt"
cache_path.exists()  # False from new CWD → "Cache does not exist, extracting features..."
```

## Fix

Two targeted changes to `FeatureSplattingDataManager.setup()` in `collab_splats/nerfstudio/datamanagers/features.py`:

### 1. Resolve `cache_dir` to absolute

```python
# Before:
cache_dir = self.config.dataparser.data

# After:
cache_dir = Path(self.config.dataparser.data).resolve()
```

Makes the cache file location CWD-independent. Absolute paths are a no-op (`.resolve()` returns same path). Relative paths are resolved once at load time using the current CWD — which for notebook loads is typically fine as long as data is accessible.

### 2. Compare by canonical filename stems

```python
def _cache_stems(filenames: list) -> list[str]:
    return sorted(Path(f).name for f in filenames)
```

Replace the comparison:

```python
# Before:
if cache_dict.get("image_filenames") != image_filenames:

# After:
if _cache_stems(cache_dict.get("image_filenames", [])) != _cache_stems(image_filenames):
```

**Why stems are canonical:** Image identity in nerfstudio comes from `transforms.json` `file_path` entries (e.g., `images/frame_001.jpg`). The stem (`frame_001.jpg`) is stable — it doesn't change with downscale factor, absolute vs relative prefix, or `test_mode`. The downscale folder (`images_2/`) is a load-time artifact, not part of the image's identity.

**Why sorted:** Guarantees order-independence in the comparison even if dataparser split logic changes ordering in future nerfstudio versions.

**Assumption:** All image filenames within a scene are unique (standard nerfstudio layout — all images in flat `images/` directory). If two images had the same filename in different subdirectories, stems would collide — but nerfstudio doesn't produce this layout.

## What Does Not Change

- Cache storage format: `{"image_filenames": [...], "features_dict": {...}}` — unchanged. Old caches continue to work; their `image_filenames` are now compared via stems, which is more lenient and correct.
- `split_train_test_features`: positional split `[:train_size]` / `[train_size:]` remains valid. Dataset split logic (which images go to train vs eval, and their order) is deterministic and unchanged between training and inference — only path prefixes differ.
- `extract_features`: unchanged.

## Files Changed

- `collab_splats/nerfstudio/datamanagers/features.py`: `setup()` — 3 line changes + 1 helper function

## Out of Scope

- Making `Splatter` always resolve `preproc_data_path` to absolute before calling `ns-train` (would prevent the secondary issue from ever occurring, but is a separate concern and not needed given the `resolve()` fix in the datamanager)
- Filename-keyed feature storage (robust to reordering, but over-engineering for current needs)
