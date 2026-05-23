# Semantics Features Refactor — Design Spec

**Date:** 2026-04-21
**Branch:** `refactor/core-modules`
**Goal:** Improve modularity and clarity of `collab_splats/semantics/features.py` by extracting utilities, unifying interfaces, and removing dead code.

---

## Problem

`features.py` (~575 lines) mixes three concerns:
1. General image utilities (PIL ops, resize, patch interpolation)
2. Torch/model utilities (GC, batch sizing, model loading, contrastive scoring)
3. Feature extractor classes (CLIP, DINO, Talk2DINO)

Additionally:
- `compute_similarity` is public on CLIP but private on Talk2DINO despite identical contract
- `compute_semantic_heatmap` duplicates functionality available in `utils/visualization.py`
- `MEM_PER_IMAGE_GB` is hardcoded per-extractor instead of measured dynamically
- DINO uses `torch.hub` while Talk2DINO uses HuggingFace `transformers` — inconsistent loading
- `clip-vit` backward-compat alias is only used by one test
- `_DEFAULT_NEGATIVE` is a module-level constant used by only two methods
- `_reshape` is a private method with one caller — should be inlined

## Design

### 1. New file: `collab_splats/utils/image.py`

Pure PIL image operations, zero torch dependency.

| Function | Source | Notes |
|----------|--------|-------|
| `open_image(image)` | `_open_image` from `features.py` | Made public (dropped leading `_`) |
| `resize_image(image, longest_edge)` | `features.py` | Unchanged |

### 2. New file: `collab_splats/semantics/utils.py`

Torch/model utilities serving the semantics pipeline.

| Function | Source | Notes |
|----------|--------|-------|
| `compute_semantic_contrast(raw_similarities, num_positive, softmax_temp, method)` | `_apply_similarity_method` from `features.py` | Renamed. Better documentation. Standard vs pairwise contrastive scoring. |
| `interpolate_to_patch_size(img_bchw, patch_size)` | `features.py` | Patch-specific, uses `F.interpolate` |
| `pytorch_gc()` | `features.py` | Unchanged |
| `infer_batch_size(mem_per_image_gb, headroom)` | `features.py` | Unchanged |
| `load_hf_weights(repo_id, filename)` | `features.py` | HF hub download |
| `load_torchhub_model(repo_id, model_name)` | `features.py` | Kept for `segmentation.py` (MobileSAM). Removable once MobileSAM migrates. |
| `batch_iterator(batch_size, *args)` | `features.py` | Unchanged |

### 3. `features.py` changes — extractors only

After extraction, `features.py` contains only `BaseFeatureExtractor` and the three extractor classes.

#### BaseFeatureExtractor

Add `_get_memory_per_image(self, sample_image) -> float`:
- Runs one dummy `forward([sample_image])` under `torch.no_grad()`
- Measures peak VRAM delta via `torch.cuda.max_memory_allocated()`
- Returns GB consumed per image
- Falls back to a reasonable default (e.g. 2.0 GB) if CUDA unavailable
- Replaces hardcoded `MEM_PER_IMAGE_GB` class constants on all extractors

#### MaskCLIPExtractor (`"samclip"`)

- Imports `open_image`, `resize_image` from `collab_splats.utils.image`
- Imports `compute_semantic_contrast` from `collab_splats.semantics.utils`
- `TORCH_HOME` stays here (only consumer)
- `maskclip_onnx` try/except import guard stays (CPU-only machines)
- `compute_similarity` calls `compute_semantic_contrast` internally
- `_DEFAULT_NEGATIVE` removed from module level; default `["object"]` inlined into `compute_similarity` method body
- **Delete** `clip-vit` backward-compat alias (line 334)

#### DINOFeatureExtractor (`"dinov2"`)

- **Migrate from torch.hub to HuggingFace transformers:**
  - `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` → `AutoModel.from_pretrained("facebook/dinov2-small")`
  - `model.forward_features(tensors)["x_norm_patchtokens"]` → `model(tensors).last_hidden_state[:, 1:]`
  - Both are equivalent: final LayerNorm (eps=1e-6) applied, CLS token stripped, patch tokens returned
  - `self.model.patch_size` → `self.model.config.patch_size`
  - Lazy `from transformers import AutoModel` inside `__init__` (matches Talk2DINO pattern)
  - Default `model_name` param becomes `"facebook/dinov2-small"`
- **Inline `_reshape`** into `forward` — 3 lines, one caller
- **Note:** Register-token variants (e.g. `dinov2_vits14_reg`) would need `[:, 1 + num_register_tokens:]` slicing. Not relevant for `dinov2-small` (0 registers) but worth a code comment if we support them later.

#### Talk2DinoExtractor (`"talk2dino"`)

- **Rename** `_compute_similarity` → `compute_similarity` (public, matches CLIP interface)
- **Delete** `compute_semantic_heatmap` entirely — visualization module has building blocks (`compute_heatmap`, `query_heatmap`)
- `_DEFAULT_NEGATIVE` inlined into `compute_similarity` as `negative = negative or ["object"]`

### 4. Protocol update: `semantics/protocols.py`

`SupportsTextQuery` simplified:

```python
@runtime_checkable
class SupportsTextQuery(Protocol):
    def encode_text(self, texts: List[str]) -> torch.Tensor: ...
    def compute_similarity(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]],
        softmax_temp: float,
        method: str,
    ) -> torch.Tensor: ...
```

Removes `compute_semantic_heatmap` from protocol. Both CLIP and Talk2DINO satisfy this protocol.

### 5. Downstream updates

#### `semantics/__init__.py`
- Update re-exports: utilities now come from `semantics/utils.py` and `utils/image.py`
- Keep extractor classes exported from `features.py`

#### `utils/features.py` (backward-compat shim)
- Update re-export paths to new locations
- Existing callers (`nerfstudio/datamanagers/features.py` etc.) keep working

#### `nerfstudio/datamanagers/features.py`
- `infer_batch_size` → calls `extractor._get_memory_per_image(sample)` instead of reading `extractor.MEM_PER_IMAGE_GB`
- Other imports updated to new paths (or keep working via shim)

#### `dashboard/semantics.py`
- Remove/defer `compute_semantic_heatmap` calls
- `isinstance(extractor, SupportsTextQuery)` still works with updated protocol

#### `tests/test_models.py`
- `"clip-vit"` → `"samclip"` in test fixture

## Out of scope

- Moving `compute_semantic_heatmap` logic to `utils/visualization.py` as standalone multi-label function (deferred — not needed now)
- MobileSAM migration from torch.hub to HF (separate task)
- `utils/features.py` shim retirement (wait until all callers migrated)

## Risks

| Risk | Mitigation |
|------|------------|
| DINO HF output differs from torch.hub | Verified: `last_hidden_state[:, 1:]` = `x_norm_patchtokens` for non-register models. Same LayerNorm, same CLS strip. |
| `_get_memory_per_image` slow on large models | Single forward pass overhead is one-time cost at batch-size inference time. Acceptable. |
| Dashboard breaks without `compute_semantic_heatmap` | Dashboard caller identified (`semantics.py:660`). Must update/remove before merge. |
| Breaking change: `model_name="dinov2_vits14"` no longer valid | Internal API. No external consumers. |
