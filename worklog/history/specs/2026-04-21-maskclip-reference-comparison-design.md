# MaskCLIP Reference Comparison Notebook — Design Spec (v2)

## Context

`MaskCLIPExtractor` in `collab_splats/semantics/features.py` produces wrong similarity maps.

**Reference notebook** (`RogerQi/maskclip_onnx/clip_playground.ipynb`) uses:
- `ViT-B/16` at 224px input, ImageNet normalization (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- No text similarity — only feature extraction

Our implementation uses `ViT-L/14@336px` at 1024px, same ImageNet stats. The reference does not
implement text similarity at all, so `compute_similarity` / `compute_semantic_contrast` cannot be
validated against it directly.

**Actual open questions causing wrong maps:**

1. **Resolution**: `ViT-L/14@336px` was trained for 336px input. We pass 1024px. This may degrade
   positional embedding quality and feature quality.

2. **Feature normalization**: `get_patch_encodings()` may return unnormalized patch features.
   Text embeddings are explicitly L2-normalized (`embed /= embed.norm(...)`). If patch features
   have large/varying magnitudes, similarity is biased toward high-norm patches, not semantics.

3. **`compute_semantic_contrast` behavior**: softmax over query dimension with `softmax_temp=0.05`
   — needs visual validation that output probability distributions are sensible.

The goal is a diagnostic notebook that exposes each of these three questions with concrete
measurements and visualizations.

## Deliverable

**`docs/semantics/maskclip_reference_comparison.ipynb`**

Single notebook, 7 sections. Test image: `docs/semantics/dog-cat.jpg`.

## Notebook Structure

### §1 Setup
- Load `ViT-L/14@336px` model (raw) + `MaskCLIPExtractor`
- Load `docs/semantics/dog-cat.jpg`
- Define shared constants: `PATCH_SIZE=14`, `INPUT_RES=336`, `PATCH_H=PATCH_W=24`

### §2 Resolution comparison
Two preprocessing paths, same image, same ImageNet stats, different resize:
- **Native (336px)**: `T.Resize(336) + T.CenterCrop(336) + T.ToTensor() + ImageNet norm` → (1, 3, 336, 336)
- **Ours (1024px)**: `extractor.preprocess(image, resolution=1024)` → (C, H, W) aspect-ratio-preserved

Output:
- Tensor shapes and per-channel stats for both
- Visual comparison of the two preprocessed images (unnormalized for display)
- Note: different shapes, so no pixel-level diff — this is qualitative

### §3 Patch feature quality
Run `model.get_patch_encodings()` on both tensors:
- L2 norm histogram per patch for both resolutions
- If 1024px features have systematically different norms than 336px, resolution is hurting quality

### §4 Text embedding sanity check
Same text queries through both paths (`ref_model.encode_text` and `extractor.encode_text`):
- Cosine similarity between them — expected ≈ 1.0
- Confirms text path is clean

### §5 Feature normalization effect
Using 1024px features (our current path):
- Show L2 norm distribution of raw patch features
- Compute similarity map WITHOUT F.normalize vs WITH F.normalize(features, dim=0)
- Side-by-side heatmaps for query "dog" and "cat"
- This directly answers: does feature normalization matter for this model?

### §6 Resolution × normalization — three-column maps
For queries "dog" and "cat", show:
1. **336px + no normalize** (closest to reference intent)
2. **1024px + no normalize** (current behavior)
3. **1024px + F.normalize** (candidate fix)

Each column: similarity heatmap overlaid on image thumbnail.
Pearson correlation: col2 vs col1, col3 vs col1.

### §7 Diagnosis summary
Table:
| Question | Finding | Magnitude | Recommendation |
|----------|---------|-----------|----------------|
| Resolution hurts features? | ✓/✗ | Norm diff: X | Use 336px |
| Text embeddings correct? | ✓/✗ | Cosine sim: X | — |
| Feature norm needed? | ✓/✗ | Map Pearson diff: X | Add F.normalize |
| Best config | — | — | 336px + F.normalize |

## Files Modified / Created

| File | Action |
|------|--------|
| `docs/semantics/maskclip_reference_comparison.ipynb` | Create / rewrite |
| `collab_splats/semantics/features.py` | Out of scope — fix separately after notebook confirms |

## Verification

1. Set `IMAGE_PATH = "docs/semantics/dog-cat.jpg"` (already set)
2. Run all cells — no errors
3. §2 shows different crops/sizes for 336px vs 1024px
4. §5 shows whether F.normalize changes similarity maps
5. §6 shows which config produces the most semantically correct heatmaps
6. §7 table populated with real values
