# Feature Extraction Batching & CUDA Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make feature extraction batch-native in the semantics module and auto-detect CUDA in the dashboard.

**Architecture:** Add `forward_batch()` + `reshape_batch()` protocol to `BaseFeatureExtractor` (all subclasses implement); add `infer_batch_size()` VRAM-aware utility; replace the regularization-features per-image loop in `FeaturesPlattingDataManager` with a chunked batch loop. Separately, auto-detect CUDA for the three device dropdowns in the Panel dashboard.

**Tech Stack:** PyTorch, Panel (dashboard)

---

## Scope Note

Only the **regularization features loop** (L158-L166) in `features_datamanager.py` is batched. The **main features loop** (L182-L234) interleaves per-image SAM segmentation and cannot be batched without a larger refactor — leave it as-is.

## Branch Note

- Tasks 1–3: work on `refactor/semantics`
- Task 4: switch to `dashboard` branch

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/semantics/features.py` | Add `forward_batch()`, `reshape_batch()` to base + all 3 extractors; add `infer_batch_size()`; add `MEM_PER_IMAGE_GB` constants |
| `collab_splats/datamanagers/features_datamanager.py` | Replace per-image regularization loop (L158-L166) with batched chunk loop |
| `collab_splats/dashboard/semantics.py` | Auto-detect CUDA default for `device_dd`, `seg_device_dd`, `query_device_dd` |

---

## Task 1: Add `forward_batch` / `reshape_batch` Protocol

**Branch:** `refactor/semantics`

**Files:**
- Modify: `collab_splats/semantics/features.py`

- [ ] **Step 1.1: Add protocol stubs to `BaseFeatureExtractor`**

In `collab_splats/semantics/features.py`, add to `BaseFeatureExtractor` class after the `get()` classmethod:

```python
    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement forward_batch()")

    def reshape_batch(self, batch: torch.Tensor, idx: int, *args) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement reshape_batch()")
```

- [ ] **Step 1.2: Add `MEM_PER_IMAGE_GB`, `forward_batch`, `reshape_batch` to `DINOFeatureExtractor`**

In `DINOFeatureExtractor`, after `def __init__`, add:

```python
    MEM_PER_IMAGE_GB: float = 1.5  # DINOv2 ViTS14 at 800px

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of (tensor_1CHW, H, W) from preprocess()."""
        images = torch.cat([img for img, _, _ in preprocessed])  # (B, C, H, W)
        with torch.no_grad():
            return self.model.forward_features(images)["x_norm_patchtokens"]  # (B, N, D)

    def reshape_batch(self, batch: torch.Tensor, idx: int, target_H: int, target_W: int) -> torch.Tensor:
        return self.reshape(batch[idx], target_H, target_W)  # (C, H_patches, W_patches)
```

- [ ] **Step 1.3: Add `MEM_PER_IMAGE_GB`, `forward_batch`, `reshape_batch` to `MaskCLIPExtractor`**

In `MaskCLIPExtractor`, after `def __init__`, add:

```python
    MEM_PER_IMAGE_GB: float = 3.0  # CLIP ViT-L/14@336px

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of (C,H,W) tensors from preprocess()."""
        images = torch.stack(preprocessed)  # (B, C, H, W)
        return self.forward(images)  # (B, C_feat, pH, pW)

    def reshape_batch(self, batch: torch.Tensor, idx: int, *_) -> torch.Tensor:
        return batch[idx]  # (C_feat, pH, pW) — already correctly shaped
```

- [ ] **Step 1.4: Add `MEM_PER_IMAGE_GB`, `forward_batch`, `reshape_batch` to `Talk2DinoExtractor`**

In `Talk2DinoExtractor`, after `def __init__`, add:

```python
    MEM_PER_IMAGE_GB: float = 1.0  # Talk2DINO ViTB

    def forward_batch(self, preprocessed: list) -> torch.Tensor:
        """Batch inference. preprocessed: list of PIL Images from preprocess()."""
        with torch.no_grad():
            result = self._model.encode_image(preprocessed)
        # encode_image may return tensor (B,N,D) or list of (N,D) tensors
        return result if isinstance(result, torch.Tensor) else torch.stack(result)

    def reshape_batch(self, batch: torch.Tensor, idx: int, *_) -> torch.Tensor:
        return batch[idx]  # (N_patches, D) patch tokens
```

- [ ] **Step 1.5: Smoke check**

```bash
cd /workspace/collab-splats
python -c "
from collab_splats.semantics.features import BaseFeatureExtractor
assert hasattr(BaseFeatureExtractor, 'forward_batch')
assert hasattr(BaseFeatureExtractor, 'reshape_batch')
print('OK')
"
```

- [ ] **Step 1.6: Commit**

```bash
git add collab_splats/semantics/features.py
git commit -m "feat(semantics): add forward_batch/reshape_batch protocol to BaseFeatureExtractor and all extractors"
```

---

## Task 2: Add `infer_batch_size()` Utility

**Branch:** `refactor/semantics`

**Files:**
- Modify: `collab_splats/semantics/features.py` (module level, after `pytorch_gc()`)

- [ ] **Step 2.1: Add `infer_batch_size()` to `features.py`**

In `collab_splats/semantics/features.py`, after the `pytorch_gc()` function (around line 47), add:

```python
def infer_batch_size(mem_per_image_gb: float, headroom: float = 0.3) -> int:
    """Compute a safe batch size from available VRAM.

    Args:
        mem_per_image_gb: Estimated GPU memory per image in GB (use extractor.MEM_PER_IMAGE_GB).
        headroom: Fraction of VRAM to use for batch data; remainder reserved for model weights.

    Returns:
        Batch size >= 1. Returns 1 if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return max(1, int(vram_gb * headroom / mem_per_image_gb))
    return 1
```

- [ ] **Step 2.2: Smoke check**

```bash
cd /workspace/collab-splats
python -c "from collab_splats.semantics.features import infer_batch_size; print(infer_batch_size(1.5))"
```

Expected: prints an integer >= 1.

- [ ] **Step 2.3: Commit**

```bash
git add collab_splats/semantics/features.py
git commit -m "feat(semantics): add infer_batch_size() VRAM-aware batch size utility"
```

---

## Task 3: Batch Regularization Loop in `features_datamanager.py`

**Branch:** `refactor/semantics`

**Files:**
- Modify: `collab_splats/datamanagers/features_datamanager.py` (L158-L166)

**Scope reminder:** Only the regularization loop (L158-L166) is batched. The main features loop (L182-L234) involves per-image SAM segmentation and stays sequential.

- [ ] **Step 3.1: Add `infer_batch_size` to imports**

Find the existing import from `collab_splats.semantics.features` in `features_datamanager.py` and add `infer_batch_size`:

```python
from collab_splats.semantics.features import BaseFeatureExtractor, infer_batch_size
```

- [ ] **Step 3.2: Replace per-image regularization loop**

In `collab_splats/datamanagers/features_datamanager.py`, replace L158-L166:

**Remove:**
```python
            for i in trange(
                len(image_filenames),
                desc=f"Extracting {self.config.regularization_features} features",
            ):
                image, target_H, target_W = extractor.preprocess(image_filenames[i])
                features = extractor.forward(image)
                features = extractor.reshape(features, target_H, target_W)
                features = features.detach().cpu()
                features_dict[self.config.regularization_features].append(features)
```

**Add:**
```python
            batch_size = infer_batch_size(extractor.MEM_PER_IMAGE_GB)
            name = self.config.regularization_features
            for start in trange(0, len(image_filenames), batch_size, desc=f"Extracting {name} features"):
                batch_paths = image_filenames[start : start + batch_size]
                preprocessed = [extractor.preprocess(p) for p in batch_paths]
                features_batch = extractor.forward_batch(preprocessed)
                for j, (_, target_H, target_W) in enumerate(preprocessed):
                    features = extractor.reshape_batch(features_batch, j, target_H, target_W)
                    features_dict[name].append(features.detach().cpu())
```

- [ ] **Step 3.3: Smoke check**

```bash
cd /workspace/collab-splats
python -c "from collab_splats.datamanagers.features_datamanager import FeaturesPlattingDataManager; print('OK')"
```

- [ ] **Step 3.4: Commit**

```bash
git add collab_splats/datamanagers/features_datamanager.py
git commit -m "feat(datamanager): batch regularization feature extraction using forward_batch"
```

---

## Task 4: Dashboard CUDA Auto-Detection

**Branch:** `dashboard`

**Files:**
- Modify: `collab_splats/dashboard/semantics.py` (near L198, L202, L211)

**Switch to `dashboard` branch before starting this task:**
```bash
git checkout dashboard
```

- [ ] **Step 4.1: Check `torch` is imported in `dashboard/semantics.py`**

```bash
grep -n "^import torch\|^from torch" collab_splats/dashboard/semantics.py
```

If `torch` is **not** imported, add `import torch` to the imports section.

- [ ] **Step 4.2: Add `_DEFAULT_DEVICE` constant**

Add at module level (before `class SemanticsDashboard`):

```python
_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 4.3: Update all three device dropdowns**

**Line ~198:**
```python
        self.device_dd = pn.widgets.Select(name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100)
```

**Line ~202:**
```python
        self.seg_device_dd = pn.widgets.Select(name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100)
```

**Line ~211:**
```python
        self.query_device_dd = pn.widgets.Select(name="Device", options=["cpu", "cuda"], value=_DEFAULT_DEVICE, width=100)
```

- [ ] **Step 4.4: Smoke check**

```bash
cd /workspace/collab-splats
python -c "import collab_splats.dashboard.semantics as m; print('_DEFAULT_DEVICE:', m._DEFAULT_DEVICE)"
```

Expected: prints `_DEFAULT_DEVICE: cuda` (or `cpu` if no GPU present).

- [ ] **Step 4.5: Commit**

```bash
git add collab_splats/dashboard/semantics.py
git commit -m "feat(dashboard): auto-detect CUDA and default device dropdowns accordingly"
```

---

## Summary

| Task | Branch | Commit message |
|------|--------|----------------|
| 1 | `refactor/semantics` | `feat(semantics): add forward_batch/reshape_batch protocol to BaseFeatureExtractor and all extractors` |
| 2 | `refactor/semantics` | `feat(semantics): add infer_batch_size() VRAM-aware batch size utility` |
| 3 | `refactor/semantics` | `feat(datamanager): batch regularization feature extraction using forward_batch` |
| 4 | `dashboard` | `feat(dashboard): auto-detect CUDA and default device dropdowns accordingly` |

Expected speedup at 20GB VRAM (batch_size ≈ 4 for DINO): **3–5x** wall-time reduction for regularization feature extraction.
