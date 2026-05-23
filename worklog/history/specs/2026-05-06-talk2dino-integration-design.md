# Talk2DINO Integration Design

**Date:** 2026-05-06  
**Branch:** `refactor/core-modules`  
**Goal:** Fix two bugs blocking notebook testing, then wire Talk2DINO as an alternative `main_features` extractor for `rade-features` training, enabling comparison with MaskCLIP.

---

## Context

The `feature_extraction.ipynb` notebook demonstrates MaskCLIP and Talk2DINO extractors. Two bugs block Talk2DINO from working:

1. **`clip` package missing** — Talk2DINOv3's HuggingFace remote code imports OpenAI `clip`, which is not installed in the nerfstudio env. Raises `ImportError: No module named 'clip'` at model load time.

2. **`score_queries` black-image bug** — when `negative=None` (no negative queries), `compute_semantic_contrast` falls back to raw dot-product similarities (range ~0.1–0.3). Visualization functions (`compute_heatmap`, `compute_masked_image`) expect softmax-normalized contrastive scores in [0,1]. Everything clips to zero → all-black image.

Additionally, `main_features` in `FeatureSplattingDataManagerConfig` is `Literal["maskclip"]` — Talk2DINO is blocked at the type level from being used as the training embedding space.

**Why Talk2DINO instead of MaskCLIP for training?** Talk2DINO uses a DINOv3 backbone with CLIP text projection — spatial structure is already baked into the features. MaskCLIP requires DINOv2 regularization to achieve structural grounding; Talk2DINO does not. This makes Talk2DINO a cleaner standalone embedding space.

**Center-crop misalignment (deferred):** Talk2DINO's `preprocess()` center-crops images to square before feature extraction. Training images are full (non-square) frames. The extracted features therefore cover only the center crop, not the full image. For this exploratory comparison this misalignment is acceptable; if results are promising, a follow-up spec will address it (e.g., masked feature loss outside cropped region).

---

## Changes

### 1. Environment: install `clip`

```bash
/opt/conda/envs/nerfstudio/bin/pip install git+https://github.com/openai/CLIP.git
```

Add to `setup.sh` under the nerfstudio-env pip installs so it persists across rebuilds.

### 2. `score_queries` default negative

**File:** `collab_splats/semantics/features.py:131`

Change `negative` default from `None` to `("background",)`. This ensures `compute_semantic_contrast` always takes the softmax path and returns values in [0,1], which visualization functions expect.

```python
# Before
negative: Optional[List[str]] = None,

# After
negative: List[str] = ("background",),
```

Tuple as default avoids the mutable-default antipattern; `List[str]` type hint accepts tuples fine. Update docstring to document the default. No import changes needed.

### 3. Widen `main_features` Literal

**File:** `collab_splats/nerfstudio/datamanagers/features.py:41`

```python
# Before
main_features: Literal["maskclip"] = "maskclip"

# After
main_features: Literal["maskclip", "talk2dino"] = "maskclip"
```

Update docstring. Add inline comment noting that `talk2dino` center-crops images during extraction (spatial coverage ≠ full frame).

### 4. Notebook fixes and comparison

**File:** `docs/semantics/feature_extraction.ipynb`

- **Cell 9:** Remove the redundant explicit `extractor_t2d.preprocess(pil_frame)` call (result unused; `forward()` calls it internally). Keep `sq_frame` derivation for visualization alignment.
- **Cell 6 (MaskCLIP):** No change needed — `score_queries` will now use `negative=("background",)` by default.
- **Add comparison section:** New cells after Talk2DINO section showing MaskCLIP and Talk2DINO side-by-side on the same image — PCA→RGB and similarity heatmap for each.

---

## Training Usage (post-implementation)

```bash
# Existing: maskclip + dinov2 regularization
ns-train rade-features --pipeline.datamanager.main_features maskclip

# New: talk2dino, no regularization needed
ns-train rade-features \
  --pipeline.datamanager.main_features talk2dino \
  --pipeline.datamanager.regularization_features None
```

---

## Verification

1. **Bug 1:** `Talk2DinoExtractor(model_name="lorebianchi98/Talk2DINOv3-ViTB")` loads without error in nerfstudio env.
2. **Bug 2:** `extractor.score_queries(features, positive=["bird"])` (no explicit negative) returns a non-black heatmap.
3. **Literal:** `FeatureSplattingDataManagerConfig(main_features="talk2dino")` instantiates without type error.
4. **Notebook:** All cells in `feature_extraction.ipynb` run top-to-bottom without error; comparison section renders side-by-side plots.
5. **Training smoke test:** `ns-train rade-features --pipeline.datamanager.main_features talk2dino` starts feature extraction without crashing (full training not required for verification).
