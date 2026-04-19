# Feature Extraction Batching & CUDA Auto-Detection

**Date:** 2026-04-19
**Branches:** `refactor/semantics` (extraction), `dashboard` (UI)

## Problem

Feature extraction in `collab_splats/semantics/` processes images one at a time:
- `DINOFeatureExtractor.forward()` hardcodes `[0]` — batch dim silently discarded
- `Talk2DinoExtractor.forward()` wraps single image in list
- `features_datamanager.py` loops with `trange`, one preprocess → forward → reshape per image
- Dashboard device dropdowns are static `["cpu", "cuda"]` — no CUDA auto-detection

This causes GPU underutilization (~10-20% occupancy on ViT models with batch_size=1) and unnecessary per-image kernel launch overhead.

## Goals

1. Make all `BaseFeatureExtractor` subclasses batch-native — batching is a general contract of the semantics module, not tied to any consumer.
2. Auto-scale batch size from available VRAM.
3. Dashboard auto-detects CUDA and defaults device dropdowns accordingly.
4. Each change fits in a commit < 50 lines.

## Design

### Component 1: Batch-native `forward()` in `features.py`

**`DINOFeatureExtractor.forward(image: torch.Tensor) -> torch.Tensor`**
- Input: `(B, C, H, W)` tensor (already the case in preprocess)
- Current bug: `["x_norm_patchtokens"][0]` drops batch dim
- Fix: return `(B, N_patches, D)` — caller reshapes per image
- `reshape()` updated to accept batch output: loops over B or accepts index

**`Talk2DinoExtractor`** — `forward()` signature stays unchanged (PIL → tensor). Add `forward_batch(images: List[Image.Image]) -> torch.Tensor` that calls `encode_image(images)` directly and returns `(B, N_patches, D)`.

**`MaskCLIPExtractor`** — no change, already batch-aware.

**Protocol on `BaseFeatureExtractor`**: add abstract `forward_batch()` so all extractors expose a consistent batch interface. The datamanager calls `forward_batch()` exclusively; `forward()` is unchanged for backward compatibility.

Constraint: all images in a batch must have the same spatial resolution after preprocessing. This holds for video-frame datasets (same camera → same resolution). If resolutions differ, images are grouped by resolution before batching.

### Component 2: `infer_batch_size()` utility in `features.py`

```python
def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return max(1, int(vram * headroom / mem_per_image_gb))
    return 1
```

Per-extractor VRAM estimates (conservative, at typical resolutions):
- `DINOFeatureExtractor` (ViTS14, 800px): `MEM_PER_IMAGE_GB = 1.5`
- `MaskCLIPExtractor` (ViT-L/14@336px): `MEM_PER_IMAGE_GB = 3.0`
- `Talk2DinoExtractor` (ViTB): `MEM_PER_IMAGE_GB = 1.0`

`headroom=0.3` leaves 70% VRAM for model weights and activations.

### Component 3: Batch extraction loop in `features_datamanager.py`

Replace:
```python
for i in trange(len(image_filenames), ...):
    image, H, W = extractor.preprocess(image_filenames[i])
    features = extractor.forward(image)
    features = extractor.reshape(features, H, W)
    features_list.append(features)
```

With chunked batch loop:
```python
batch_size = infer_batch_size(extractor.MEM_PER_IMAGE_GB)
chunks = [image_filenames[i:i+batch_size] for i in range(0, len(image_filenames), batch_size)]
for batch_paths in trange(len(chunks), desc=...):
    preprocessed = [extractor.preprocess(p) for p in batch_paths]
    features_batch = extractor.forward_batch(preprocessed)  # (B, ...)
    for j, meta in enumerate(preprocessed):
        features_list.append(extractor.reshape_batch(features_batch, j, *meta[1:]))
```

`forward_batch()` handles the PIL-vs-tensor difference per extractor — DINO/CLIP stack tensors via `torch.cat`, Talk2DINO passes PIL list directly to `encode_image()`. `reshape_batch(batch, idx, H, W)` extracts item `idx` and reshapes to `(C, H, W)`. Progress bar wraps outer chunk loop.

### Component 4: Dashboard CUDA auto-detection in `dashboard/semantics.py`

Three dropdowns currently hardcode `options=["cpu", "cuda"]` with no default.

Fix: compute default at widget creation time:
```python
_default_device = "cuda" if torch.cuda.is_available() else "cpu"
```

Apply to `device_dd`, `seg_device_dd`, `query_device_dd` — same default, no logic change beyond initialization. Options list stays `["cpu", "cuda"]` so user can still override.

## Commit Plan

| # | File(s) | Change | Branch | Est. lines |
|---|---------|--------|--------|-----------|
| 1 | `semantics/features.py` | Add `forward_batch()` + `reshape_batch()` to all extractors; `BaseFeatureExtractor` abstract protocol | `refactor/semantics` | ~30 |
| 2 | `semantics/features.py` | Add `infer_batch_size()` + per-extractor `MEM_PER_IMAGE_GB` constants | `refactor/semantics` | ~15 |
| 3 | `datamanagers/features_datamanager.py` | Replace per-image loop with chunked `forward_batch()` loop | `refactor/semantics` | ~25 |
| 4 | `dashboard/semantics.py` | CUDA auto-detect default for device dropdowns | `dashboard` | ~10 |

## Expected Speedup

At 20GB VRAM with `headroom=0.3`: batch_size ≈ 4 (DINO), 2 (CLIP), 6 (Talk2DINO).

ViT inference scales near-linearly with batch size up to memory limits. Expected wall-time speedup: **3-5x** for DINO/Talk2DINO, **2-3x** for CLIP (larger model, smaller batch).

## Out of Scope

- `grouping.py` DataLoader — consumer-specific, not part of the semantics module contract
- Multi-GPU support
- Mixed-precision inference (orthogonal optimization)
