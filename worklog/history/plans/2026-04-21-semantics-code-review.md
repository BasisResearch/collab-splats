# Semantics Code Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove four code quality issues from `collab_splats/semantics/` — duplicate image-loading logic, duplicate similarity branching, hardcoded default, and a debug print — with no public API changes.

**Architecture:** All four changes are additive private helpers or constants extracted from existing code. Existing tests cover public API and must pass unchanged. Two new test functions cover the extracted helpers directly.

**Tech Stack:** Python, PyTorch, PIL, pytest

---

## File Map

| File | Change |
|------|--------|
| `collab_splats/semantics/features.py` | Add `_DEFAULT_NEGATIVE`, `_open_image()`, `_apply_similarity_method()`; update 3× `preprocess()`, 2× similarity methods |
| `collab_splats/semantics/segmentation.py` | Add `import logging`, `logger`; replace 1× `print()` |
| `tests/semantics/test_features_guards.py` | Add tests for `_open_image()` and `_apply_similarity_method()` |
| `tests/semantics/test_segmentation.py` | New file — test `create_composite_mask` produces no stdout |

---

## Task 1: `_DEFAULT_NEGATIVE` constant

**Files:**
- Modify: `collab_splats/semantics/features.py` (add constant ~L32, update L304, L541)

- [ ] **Step 1: Add constant and update call sites**

  In `features.py`, after line 31 (`TORCH_HOME = ...`), add:

  ```python
  _DEFAULT_NEGATIVE: list[str] = ["object"]
  ```

  Then in `MaskCLIPExtractor.compute_similarity` (~L303):
  ```python
  # Before:
  if negative is None:
      negative = ["object"]
  # After:
  if negative is None:
      negative = _DEFAULT_NEGATIVE
  ```

  And in `Talk2DinoExtractor._compute_similarity` (~L540):
  ```python
  # Before:
  if negative is None:
      negative = ["object"]
  # After:
  if negative is None:
      negative = _DEFAULT_NEGATIVE
  ```

- [ ] **Step 2: Run existing tests**

  ```bash
  pytest tests/semantics/ -v
  ```

  Expected: same results as before (2 tests pass).

- [ ] **Step 3: Commit**

  ```bash
  git add collab_splats/semantics/features.py
  git commit -m "refactor(semantics): extract _DEFAULT_NEGATIVE constant"
  ```

---

## Task 2: `_open_image()` helper

**Files:**
- Modify: `collab_splats/semantics/features.py` (add helper ~L37, update 3× `preprocess()`)
- Modify: `tests/semantics/test_features_guards.py` (add 4 tests)

- [ ] **Step 1: Write failing tests**

  Append to `tests/semantics/test_features_guards.py`:

  ```python
  import numpy as np
  from pathlib import Path
  from PIL import Image


  def test_open_image_from_str_path(tmp_path):
      img = Image.new("RGB", (10, 10), color=(255, 0, 0))
      p = tmp_path / "test.png"
      img.save(p)
      result = feat_mod._open_image(str(p))
      assert isinstance(result, Image.Image)


  def test_open_image_from_path_object(tmp_path):
      img = Image.new("RGB", (10, 10))
      p = tmp_path / "test.png"
      img.save(p)
      result = feat_mod._open_image(p)
      assert isinstance(result, Image.Image)


  def test_open_image_from_ndarray():
      arr = np.zeros((10, 10, 3), dtype=np.uint8)
      result = feat_mod._open_image(arr)
      assert isinstance(result, Image.Image)


  def test_open_image_from_pil_returns_same():
      img = Image.new("RGB", (10, 10))
      result = feat_mod._open_image(img)
      assert result is img


  def test_open_image_invalid_type():
      with pytest.raises(ValueError, match="Unsupported image type"):
          feat_mod._open_image(42)
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  pytest tests/semantics/test_features_guards.py::test_open_image_from_str_path -v
  ```

  Expected: `AttributeError: module 'collab_splats.semantics.features' has no attribute '_open_image'`

- [ ] **Step 3: Add `_open_image()` to `features.py`**

  After `TORCH_HOME` and `_DEFAULT_NEGATIVE` lines (before the `########` section header), add:

  ```python
  def _open_image(image) -> Image.Image:
      if isinstance(image, (str, Path)):
          return Image.open(image)
      if isinstance(image, np.ndarray):
          return Image.fromarray(image)
      if isinstance(image, Image.Image):
          return image
      raise ValueError(f"Unsupported image type: {type(image)}")
  ```

- [ ] **Step 4: Update `MaskCLIPExtractor.preprocess()` (~L230)**

  ```python
  def preprocess(self, image, resolution: int = 1024) -> torch.Tensor:
      image = _open_image(image).convert("RGB")
      image = resize_image(image, longest_edge=resolution)
      return self.transform(image).to(self.device)
  ```

- [ ] **Step 5: Update `DINOFeatureExtractor.preprocess()` (~L368)**

  ```python
  def preprocess(self, image) -> Tuple[torch.Tensor, int, int]:
      image = _open_image(image)
      image = resize_image(image, longest_edge=self.resolution)
      image = self.transform(image)[:3].unsqueeze(0)
      image, target_H, target_W = interpolate_to_patch_size(image, self.model.patch_size)
      image = image.to(self.device)
      return image, target_H, target_W
  ```

  Note: DINO deliberately omits `.convert("RGB")` — it uses `[:3]` slice to handle channel count.

- [ ] **Step 6: Update `Talk2DinoExtractor.preprocess()` (~L468)**

  ```python
  def preprocess(self, image) -> Image.Image:
      """
      Center-crop image to square. Required by Talk2DINO (from hf_demo.ipynb).

      Returns PIL Image ready for encode_image / forward.
      """
      image = _open_image(image).convert("RGB")
      w, h = image.size
      crop_size = min(w, h)
      image = image.crop(
          (
              (w - crop_size) // 2,
              (h - crop_size) // 2,
              (w + crop_size) // 2,
              (h + crop_size) // 2,
          )
      )
      return image
  ```

- [ ] **Step 7: Run all tests**

  ```bash
  pytest tests/semantics/ -v
  ```

  Expected: all 7 tests pass (2 original + 5 new).

- [ ] **Step 8: Commit**

  ```bash
  git add collab_splats/semantics/features.py tests/semantics/test_features_guards.py
  git commit -m "refactor(semantics): extract _open_image() helper, remove 3x duplicated preprocess branching"
  ```

---

## Task 3: `_apply_similarity_method()` helper

**Files:**
- Modify: `collab_splats/semantics/features.py` (add helper, update 2× similarity methods)
- Modify: `tests/semantics/test_features_guards.py` (add 3 tests)

- [ ] **Step 1: Write failing tests**

  Append to `tests/semantics/test_features_guards.py`:

  ```python
  import torch


  def test_apply_similarity_method_standard_shape():
      # (3 queries, 6 patches), 2 positive
      raw = torch.rand(3, 6)
      result = feat_mod._apply_similarity_method(raw, num_positive=2, softmax_temp=0.05, method="standard")
      assert result.shape == (6,)


  def test_apply_similarity_method_pairwise_shape():
      # (3 queries, 6 patches), 1 positive, 2 negative
      raw = torch.rand(3, 6)
      result = feat_mod._apply_similarity_method(raw, num_positive=1, softmax_temp=0.05, method="pairwise")
      assert result.shape == (6,)


  def test_apply_similarity_method_unknown_raises():
      raw = torch.rand(2, 4)
      with pytest.raises(ValueError, match="Unknown method"):
          feat_mod._apply_similarity_method(raw, num_positive=1, softmax_temp=0.05, method="bad")
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  pytest tests/semantics/test_features_guards.py::test_apply_similarity_method_standard_shape -v
  ```

  Expected: `AttributeError: module ... has no attribute '_apply_similarity_method'`

- [ ] **Step 3: Add `_apply_similarity_method()` to `features.py`**

  Place this after `_open_image()` and before the `BaseFeatureExtractor` section header:

  ```python
  def _apply_similarity_method(
      raw_similarities: torch.Tensor,
      num_positive: int,
      softmax_temp: float,
      method: str,
  ) -> torch.Tensor:
      """Shared standard/pairwise similarity branching. Input: (num_queries, N). Output: (N,)."""
      if method == "standard":
          probs = (raw_similarities / softmax_temp).softmax(dim=0)
          return probs[:num_positive].sum(dim=0)
      if method == "pairwise":
          pos_similarities = raw_similarities[:num_positive]
          neg_similarities = raw_similarities[num_positive:]
          avg_pos = pos_similarities.mean(dim=0, keepdim=True)
          paired = torch.cat([avg_pos.expand(neg_similarities.shape[0], -1), neg_similarities], dim=0)
          probs = (paired / softmax_temp).softmax(dim=0)
          return torch.nan_to_num(probs[: neg_similarities.shape[0]].min(dim=0)[0], nan=0.0)
      raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")
  ```

- [ ] **Step 4: Update `MaskCLIPExtractor.compute_similarity()` (~L281)**

  Replace the full method body with:

  ```python
  def compute_similarity(
      self,
      features: torch.Tensor,
      positive: List[str],
      negative: Optional[List[str]] = None,
      softmax_temp: float = 0.05,
      method: str = "standard",
  ) -> torch.Tensor:
      """
      Compute similarity probability map between image features and text queries.

      Args:
          features (torch.Tensor): Image features of shape (C, H, W)
          positive (List[str]): List of positive text queries
          negative (List[str], optional): List of negative text queries.
                                                 If None, uses default negatives.
          softmax_temp (float): Temperature parameter for softmax
          method (str): "standard" or "pairwise"

      Returns:
          torch.Tensor: Similarity probability map of shape (H, W, 1)
      """
      if negative is None:
          negative = _DEFAULT_NEGATIVE
      queries = positive + negative
      text_embeddings = self.encode_text(queries)
      raw_similarities = torch.einsum("chw,nc->nhw", features, text_embeddings)
      raw_similarities = raw_similarities.reshape(raw_similarities.shape[0], -1)
      similarity = _apply_similarity_method(raw_similarities, len(positive), softmax_temp, method)
      return similarity.reshape(features.shape[1:] + (1,))
  ```

- [ ] **Step 5: Update `Talk2DinoExtractor._compute_similarity()` (~L524)**

  Replace the full method body with:

  ```python
  def _compute_similarity(
      self,
      features: torch.Tensor,
      positive: List[str],
      negative: Optional[List[str]] = None,
      softmax_temp: float = 0.05,
      method: str = "standard",
  ) -> torch.Tensor:
      """
      Compute per-patch similarity between image features and text queries.

      Port of compute_similarity() from hf_demo.ipynb.

      Returns:
          torch.Tensor: similarity scores of shape (N_patches,)
      """
      if negative is None:
          negative = _DEFAULT_NEGATIVE
      queries = positive + negative
      with torch.no_grad():
          text_embeddings = self._model.encode_text(queries)
      text_embeddings = F.normalize(text_embeddings, dim=-1)
      features_norm = F.normalize(features, dim=-1)
      raw_similarities = text_embeddings @ features_norm.T
      return _apply_similarity_method(raw_similarities, len(positive), softmax_temp, method)
  ```

- [ ] **Step 6: Run all tests**

  ```bash
  pytest tests/semantics/ -v
  ```

  Expected: all 10 tests pass.

- [ ] **Step 7: Commit**

  ```bash
  git add collab_splats/semantics/features.py tests/semantics/test_features_guards.py
  git commit -m "refactor(semantics): extract _apply_similarity_method(), remove 2x duplicated similarity branching"
  ```

---

## Task 4: Replace `print()` with `logger.debug()` in segmentation.py

**Files:**
- Modify: `collab_splats/semantics/segmentation.py` (add logging, update L273)
- Create: `tests/semantics/test_segmentation.py`

- [ ] **Step 1: Write failing test**

  Create `tests/semantics/test_segmentation.py`:

  ```python
  import numpy as np
  import pytest
  from collab_splats.semantics.segmentation import create_composite_mask


  def _make_results(n: int, iou: float = 0.9):
      """Make n minimal segmentation result dicts."""
      mask = np.ones((8, 8), dtype=np.uint8)
      return [{"segmentation": mask, "predicted_iou": iou} for _ in range(n)]


  def test_create_composite_mask_no_stdout(capsys):
      results = _make_results(3)
      create_composite_mask(results)
      captured = capsys.readouterr()
      assert captured.out == "", f"Unexpected stdout: {captured.out!r}"
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  pytest tests/semantics/test_segmentation.py::test_create_composite_mask_no_stdout -v
  ```

  Expected: FAIL — stdout contains `"Mask 1 has ... pixels"`.

- [ ] **Step 3: Update `segmentation.py`**

  Add after existing imports (after `from collab_splats.semantics.features import ...`):

  ```python
  import logging

  logger = logging.getLogger(__name__)
  ```

  Replace line 273:
  ```python
  # Before:
  print(f"Mask {i} has {mask.sum()} pixels")
  # After:
  logger.debug("Mask %d has %d pixels", i, mask.sum())
  ```

- [ ] **Step 4: Run all semantics tests**

  ```bash
  pytest tests/semantics/ -v
  ```

  Expected: all 11 tests pass.

- [ ] **Step 5: Run broader suite to confirm no regressions**

  ```bash
  pytest tests/ -v --ignore=tests/nerfstudio --ignore=tests/pointcloud/test_mapanything_creator.py
  ```

  Expected: all non-GPU tests pass. Pre-existing nerfstudio env failures and GPU smoke test excluded.

- [ ] **Step 6: Commit**

  ```bash
  git add collab_splats/semantics/segmentation.py tests/semantics/test_segmentation.py
  git commit -m "refactor(semantics): replace debug print with logger.debug in segmentation"
  ```

---

## Self-Review

**Spec coverage:**
- ✅ `_open_image()` — Task 2
- ✅ `_apply_similarity_method()` — Task 3
- ✅ `_DEFAULT_NEGATIVE` — Task 1 (also consumed by Task 3 refactor)
- ✅ `logger.debug()` replacing `print()` — Task 4
- ✅ No public API changes — verified: all method signatures unchanged
- ✅ All existing tests must pass — verified at each task step

**Placeholder scan:** No TBDs, no "similar to Task N", all code blocks complete.

**Type consistency:**
- `_open_image` returns `Image.Image` — consistent across Task 2 definition and usage
- `_apply_similarity_method` takes `(Tensor, int, float, str)` returns `Tensor` — consistent across Task 3 definition, MaskCLIP call, Talk2DINO call
- `_DEFAULT_NEGATIVE: list[str]` — consistent across Task 1 definition, Task 3 usage
