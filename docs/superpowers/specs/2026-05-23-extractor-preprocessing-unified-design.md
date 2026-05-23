# Unified Extractor Preprocessing Interface

**Date:** 2026-05-23  
**Status:** Approved  
**Scope:** `collab_splats/semantics/features/{dino,maskclip,talk2dino}.py`

---

## Problem

Three extractors (DINOv2, MaskCLIP, Talk2DINO) each preprocess images differently:

| Extractor | `preprocess` return | `forward` paths | Normalization | resize_mode |
|-----------|---------------------|-----------------|---------------|-------------|
| DINOv2 | `(tensor, H, W)` tuple | 1 | `mean=[0.5]` ❌ should be ImageNet | none (always max_size) |
| MaskCLIP | `(C, H, W)` tensor ✓ | 1 | ImageNet stats ❌ should be CLIP stats | none (always max_size) |
| Talk2DINO | `PIL.Image` (square) / internal tensor (max_size) | 2 | reads from model ✓ | yes (`max_size`/`square`) |

Additionally Talk2DINO has two `_forward_*` paths selected in `forward()` — preprocessing is entangled with the forward pass rather than isolated in `preprocess()`.

---

## Design

### Principle

Preprocessing is `preprocess()`. Forward is `forward()`. The two must not overlap.

```
preprocess(image) -> torch.Tensor   # (C, H, W), on CPU, ready for stack+send to device
forward(images: list) -> list[torch.Tensor]  # stacks, batches, reshapes
```

### `preprocess(image) -> torch.Tensor`

Uniform signature across all three extractors. Takes anything `open_image()` accepts. Returns `(C, H, W)` float32 tensor **on CPU**. Moving to device happens in `forward()`.

**`resize_mode="max_size"` (default):**
1. `open_image(image).convert("RGB")`
2. `resize_image(img, longest_edge=image_resolution)` — proportional, no crop
3. Round H and W to nearest `patch_size` multiple
4. `T.ToTensor()` → `normalize()`

**`resize_mode="square"`:**
1. `open_image(image).convert("RGB")`
2. Center-crop to square (min(W, H))
3. Resize to `(image_resolution, image_resolution)` with BILINEAR
4. `T.ToTensor()` → `normalize()`

`square` mode produces square output; `max_size` preserves aspect ratio. Both round to `patch_size` multiples (max_size explicitly; square implicitly because `image_resolution` should be set to a `patch_size` multiple).

### `forward(images: list) -> list[torch.Tensor]`

Single path for all modes:

```python
def forward(self, images: list) -> list[torch.Tensor]:
    preprocessed = [self.preprocess(img) for img in images]
    batch = torch.stack(preprocessed).to(self.device)   # (B, C, H, W)
    with torch.no_grad():
        tokens = self._run_backbone(batch)               # (B, N, D)
    results = []
    for i, t in enumerate(preprocessed):
        _, H, W = t.shape
        results.append(_tokens_to_feature_map(tokens[i], H, W, self.patch_size))
    return results
```

**No** `_forward_max_size` / `_forward_square` split. Mode is fully absorbed by `preprocess`.

### Module-level helper

```python
def _tokens_to_feature_map(
    tokens: torch.Tensor, input_h: int, input_w: int, patch_size: int
) -> torch.Tensor:
    """Reshape (N, D) patch tokens → (D, H_p, W_p), L2-normalized."""
    ph = input_h // patch_size
    pw = input_w // patch_size
    assert tokens.shape[0] == ph * pw
    feat = tokens.reshape(ph, pw, -1).permute(2, 0, 1)
    return F.normalize(feat, dim=0)
```

Lives at module level in each file (or shared in `collab_splats/semantics/utils.py` if needed). Each extractor controls its own token-skip before passing to this function.

---

## Per-Extractor Spec

### DINOv2

**Backbone call:** `self.model(batch).last_hidden_state[:, 1:]` — drops CLS token (index 0).

**Normalization fix:**  
Current: `T.Normalize(mean=[0.5], std=[0.5])` ❌  
Correct: `T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` — standard ImageNet stats used in DINOv2 training.

**Parameter rename:** `resolution` → `image_resolution` for consistency. Default stays `800`.

**`preprocess` change:** Returns `(C, H, W)` tensor (CPU). Remove the `(tensor, H, W)` tuple return. `forward` infers H/W from the preprocessed tensor shape.

### MaskCLIP

**Backbone call:** `self.model.get_patch_encodings(batch)` — returns `(B, N, D)` already excluding CLS; no token skip needed.

**Normalization fix:**  
Current: `T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` ❌ (ImageNet)  
Correct: `T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])` — CLIP stats from `maskclip_onnx/clip.py:86`.

**`preprocess` change:** Remove optional `resolution` parameter from signature — caller can't change it per-image. Add `resize_mode` and `image_resolution` constructor params. Default `image_resolution=1024` (unchanged). Preprocess returns `(C, H, W)` tensor (already does).

**`forward` change:** Remove `resolution` parameter. Use `torch.stack` (already does this). Inline reshape/normalize now go through `_tokens_to_feature_map`. Drop `list(features)` — build list in loop.

### Talk2DINO

**Backbone call (both modes):** `self._model.model.forward_features(batch)[:, 5:]` — drops CLS + 4 register tokens (DINOv3 convention), returns `(B, N_patches, D)`.

`encode_image` is NOT used because it applies `T.Resize((N,N))` internally. Since `preprocess()` already normalizes and resizes (square or max_size), we call `forward_features` directly for both modes. Square mode sends a square tensor; max_size sends a non-square tensor. Same backbone call either way.

**Normalization:** Read from `self._model.image_transforms.transforms[-1]` (correct, keep as-is).

**Remove:** `_forward_max_size()`, `_forward_square()`, `preprocess()` returning PIL (square crop only). Replace with single `preprocess() -> Tensor` and single `forward()`.

**`image_resolution` in square mode:** Defaults to `512`. User should set this to a `patch_size` multiple (e.g., 518 for patch_size=14 is `518/14=37`; 504 is `36×14`). Document this.

---

## Parameters — Unified Constructor Signature

All three extractors gain:

```python
resize_mode: str = "max_size"    # "max_size" | "square"
image_resolution: int = <per-extractor default>
```

DINOv2: `image_resolution=800`  
MaskCLIP: `image_resolution=1024`  
Talk2DINO: `image_resolution=512`

---

## What Does NOT Change

- `encode_text()` signatures — untouched
- `score_queries()`, `compute_similarity()` in base — untouched
- `BaseFeatureExtractor._build_positional_basis()` calls `self.forward([zero_pil])` — still works because `forward` still accepts PIL images
- `svd_components` passthrough to base — unchanged
- `RegistryMixin` registration decorators — unchanged

---

## Files Changed

| File | Changes |
|------|---------|
| `collab_splats/semantics/features/dino.py` | Fix normalization; rename `resolution`→`image_resolution`; add `resize_mode`; `preprocess` returns tensor; `forward` uses `_tokens_to_feature_map` |
| `collab_splats/semantics/features/maskclip.py` | Fix normalization (CLIP stats); add `resize_mode`; remove per-call `resolution` param; `forward` uses `_tokens_to_feature_map` |
| `collab_splats/semantics/features/talk2dino.py` | Collapse to single `preprocess`→`forward` path; remove `_forward_max_size`/`_forward_square`; use `_tokens_to_feature_map`; keep normalization from model |

No changes to `base.py`, `compression.py`, `utils.py`, or any notebook.

---

## Test Impact

Existing extractor tests should pass without changes if they call `forward(images)`. Tests that call `preprocess()` directly and expect a tuple return from DINOv2 will need updating (return is now a plain tensor).
