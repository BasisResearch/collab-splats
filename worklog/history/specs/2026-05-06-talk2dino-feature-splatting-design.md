# Talk2DINO Feature Splatting — Design Spec

**Date:** 2026-05-06  
**Branch:** `refactor/core-modules`  
**Status:** Approved

## Goal

Enable Talk2DINO as a feature space for Gaussian splatting via the existing `rade-features` pipeline, with text-query inference at render time. Remove segmentation from non-SAM pipelines to eliminate noise — segmentation only applies when `"samclip"` is explicitly selected.

## Feature Types

| `main_features` | Extractor | Segmentation | Text queries |
|---|---|---|---|
| `"samclip"` (default) | `maskclip` | Yes (MobileSAMv2) | maskclip text encoder |
| `"maskclip"` | `maskclip` | No | maskclip text encoder |
| `"talk2dino"` | `talk2dino` | No | talk2dino text encoder |

`"samclip"` = current behavior, renamed. Default changes from `"maskclip"` → `"samclip"`.

## Changes

### 1. `FeatureSplattingDataManagerConfig` (`collab_splats/nerfstudio/datamanagers/features.py`)

- `main_features: Literal["maskclip", "samclip", "talk2dino"] = "samclip"`
- Update docstring to describe all three options

### 2. `FeatureSplattingDataManager.extract_features()`

Add alias map at module level:

```python
_EXTRACTOR_NAME: dict[str, str] = {"samclip": "maskclip"}
```

In `extract_features()`:
- Resolve extractor name: `extractor_name = _EXTRACTOR_NAME.get(self.config.main_features, self.config.main_features)`
- Gate segmentation: `use_seg = self.config.main_features == "samclip"`
- Only instantiate `Segmentation` when `use_seg` is True
- Only call `aggregate_masked_features` when `use_seg` is True

### 3. `RadegsFeaturesModel.populate_text_encoder()` (`collab_splats/nerfstudio/models/rade_features.py`)

Add at module level:

```python
_QUERYABLE_FEATURE_TYPES: frozenset[str] = frozenset({"maskclip", "samclip", "talk2dino"})
_TEXT_ENCODER_NAME: dict[str, str] = {"samclip": "maskclip"}
```

Replace `if "clip" in feature_type.lower()` with:

```python
if feature_type in _QUERYABLE_FEATURE_TYPES:
    encoder_name = _TEXT_ENCODER_NAME.get(feature_type, feature_type)
    self.text_encoder = BaseFeatureExtractor.get(encoder_name)(device=self.device)
    ...
    self.similarity_fx = self.text_encoder.score_queries
```

### 4. `rade-features` method config (`collab_splats/nerfstudio/method_configs/rade_features.py`)

Update default `FeatureSplattingDataManagerConfig` instantiation if it hardcodes `main_features`.

## Testing

### Segmentation gate (parametrized unit test)

File: `tests/nerfstudio/test_features_datamanager.py`

Mock `Segmentation` and `BaseFeatureExtractor.get`. Assert:
- `"samclip"` → `Segmentation` instantiated once
- `"maskclip"` → `Segmentation` never instantiated
- `"talk2dino"` → `Segmentation` never instantiated

### `populate_text_encoder` with `"talk2dino"` (unit test)

File: `tests/nerfstudio/test_features_datamanager.py` or `tests/test_models.py`

Mock metadata with `feature_type="talk2dino"`. Assert `model.similarity_fx` is not None after init.

## Notes

- `regularization_features` default remains `"dinov2"`. For `"talk2dino"`, callers should set `regularization_features=None` — structural grounding is built into talk2dino's DINO backbone. Not enforced, documented in config docstring.
- Talk2DINO center-crops to square during extraction — feature spatial coverage != full frame. This is existing behavior, not changed here.
- The two alias dicts (`_EXTRACTOR_NAME`, `_TEXT_ENCODER_NAME`) are intentionally not shared — their usage contexts are different enough that a shared constants module adds indirection without benefit.
