# INSID3 In-Context Segmentation — Design Spec

**Date:** 2026-05-28
**Status:** approved

## Overview

Add `INSID3Segmentation` — a training-free in-context segmentation backend using frozen DINO
features and SVD-based positional debiasing. Given a reference image and a binary context mask,
it segments the same semantic category in any target image without text prompts or retraining.

Primary use case: annotate one frame of a scene (e.g., draw a tree mask), segment the same
category across all remaining frames.

Reference: [INSID3](https://github.com/visinf/INSID3) (visinf/INSID3).

---

## Architecture

```
BaseSegmentation
  ├── MobileSAMSegmentation   → segment(image)
  ├── SAM3Segmentation        → segment(image) + segment_with_text(image, prompt)
  └── INSID3Segmentation      → set_context + segment(image) + segment_with_mask
```

`INSID3Segmentation` extends `BaseSegmentation`, registered as `"insid3"`. It is a conditioned
segmenter: `segment()` requires a context to be set via `set_context()` first.
`segment_with_text` is inherited as `NotImplementedError` (no change to base).

Clustering utilities (`_agglomerative_clustering`, `_cluster_prototypes`) are module-level
private functions in `insid3.py` — too small to warrant a separate file.

---

## API

```python
@BaseSegmentation.register("insid3")
class INSID3Segmentation(BaseSegmentation):

    def __init__(
        self,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        device: str = "cuda",
    ) -> None: ...

    def set_context(
        self,
        ref_image: Image.Image | torch.Tensor,  # PIL or (C,H,W) tensor
        ref_mask: np.ndarray | torch.Tensor,    # (H,W) binary
    ) -> None:
        """Extract and cache ref features + prototype. Must call before segment()."""

    def clear_context(self) -> None:
        """Reset cached context state."""

    def segment(
        self,
        image: Image.Image | torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Segment image using cached context.

        Raises RuntimeError if no context is set.
        Returns (pred_mask (H,W) bool, metadata dict).
        """

    def segment_with_mask(
        self,
        image: Image.Image | torch.Tensor,
        ref_image: Image.Image | torch.Tensor,
        ref_mask: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """One-shot: set_context → segment → clear_context."""
```

### Stateful batch pattern

```python
seg = INSID3Segmentation()
seg.set_context(ref_image, ref_mask)   # DINO runs once on ref

seg.segment(frame_1)   # reuses cached prototype + ref features
seg.segment(frame_2)
seg.segment(frame_3)

seg.clear_context()
```

`set_context` caches `_prototype` and `_ref_feat_deb`. Subsequent `segment()` calls
do not re-extract ref features — only target features are extracted per call.

### metadata dict

`segment()` returns a metadata dict with:
- `candidate_mask` — `(H_p, W_p) bool` raw forward+backward candidate mask (patch resolution)
- `cluster_labels` — `(H_p, W_p) int` agglomerative cluster assignments
- `n_clusters` — int, number of clusters found

Useful for debugging and visualization without re-running the pipeline.

---

## Internal Pipeline

### set_context (runs once per context)

1. Extract DINO features for `ref_image` via `DINOFeatureExtractor.forward()`
2. L2-normalize features
3. Apply `extractor.debias()` — reuses existing `BaseFeatureExtractor` method; positional basis
   cached inside extractor after first call
4. Downsample `ref_mask` to patch grid resolution `(H_p, W_p)` with smart fallback:
   bilinear → nearest → single center pixel (handles tiny masks that vanish at patch resolution)
5. Mean of debiased ref features inside mask → L2-normalize → cache as `_prototype (D,)`
6. Cache debiased ref features as `_ref_feat_deb (D, H_p, W_p)`

### segment (runs per target)

1. Extract DINO features for target image, L2-normalize, debias
2. **Candidate localization**
   - Forward: cosine similarity of each target patch to `_prototype` > 0 → forward mask
   - Backward: for each target patch, find nearest neighbor in debiased ref feature map
     (`_ref_feat_deb`); check if that ref patch falls inside downsampled ref mask → backward mask
     (with S=1 context, "majority vote" degenerates to a single check — generalizes to S>1 later)
   - Candidate mask = forward ∩ backward
   - Fallback: if forward mask empty, use top-10% patches by prototype similarity
3. Early exit: if candidate mask is all-zero, return empty mask
4. **Agglomerative clustering** on raw (non-debiased) target features
   - Cosine distance matrix on `(H_p*W_p, D)` feature matrix
   - `sklearn.cluster.AgglomerativeClustering(metric='precomputed', linkage='average',
     distance_threshold=1-tau)`
   - Produces K coherent region clusters; K adapts to image complexity via threshold
5. **Seed + aggregate**
   - Find clusters overlapping candidate mask
   - Seed = cluster with highest cross-image similarity to `_prototype`
   - Score remaining clusters: `cross_image_sim × intra_image_sim_to_seed × area_weight`
   - Include clusters above `merge_threshold`; seed cluster always included
6. Bilinear upsample final mask to original image resolution → `(H, W) bool`

---

## Implementation Notes

### What we reuse from existing code

- `DINOFeatureExtractor.forward()` — feature extraction, no changes
- `BaseFeatureExtractor.debias()` — positional debiasing, no changes
- `BaseSegmentation` registry + `RegistryMixin` — no changes

### What is new

- `_locate_candidates(ref_feat_deb, tgt_feat_deb, ref_mask_down, prototype)` — forward+backward
- `_agglomerative_clustering(X, tau)` — sklearn wrapper, returns `(N,) int` labels
- `_cluster_prototypes(X, labels, K)` — L2-normalized per-cluster mean features
- `_seed_and_aggregate(...)` — cluster scoring and final mask assembly
- `INSID3Segmentation` class with context caching

### DINO model

`DINOFeatureExtractor` holds the encoder. `INSID3Segmentation.__init__` instantiates it
internally with `DINOFeatureExtractor(svd_components=svd_components, device=device)`.
No `get_intermediate_layers` call needed — `forward()` already returns patch features.

---

## Testing

File: `tests/semantics/test_insid3_segmentation.py`

Flat functions, no class-based. All use small synthetic tensors; `DINOFeatureExtractor.forward`
is mocked — no real DINO inference in unit tests.

| Test | What it checks |
|---|---|
| `test_set_context_caches_prototype` | `_prototype` + `_ref_feat_deb` non-None after `set_context` |
| `test_clear_context_resets_state` | both None after `clear_context` |
| `test_segment_raises_without_context` | `RuntimeError` if `segment()` before `set_context` |
| `test_segment_with_mask_one_shot` | output is `(H,W) bool`, state cleared after |
| `test_candidate_localization_forward_backward` | `_locate_candidates` with known sim maps |
| `test_agglomerative_clustering_shape` | returns `(N,)` int labels |
| `test_cluster_prototypes_normalized` | output is L2-normalized |
| `test_segment_output_shape` | mask matches input image spatial dims |
| `test_context_reused_across_calls` | mock extractor called once for ref, N times for N targets |

---

## Visualization

Add `plot_context_segmentation` to `collab_splats/utils/visualization.py`:

```python
def plot_context_segmentation(
    ref_image: Image.Image | np.ndarray,
    ref_mask: torch.Tensor | np.ndarray,
    tgt_image: Image.Image | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
) -> plt.Figure:
    """Side-by-side colored overlay: reference + context mask (red), target + prediction (green)."""
```

Returns a `Figure` (caller decides show vs save). Useful for notebooks and debugging without
needing to write files. Mirrors INSID3's `visualize_prediction_segmentation` approach.

---

## Files Changed

```
collab_splats/semantics/segmentation/
  insid3.py                          ← new
  __init__.py                        ← add INSID3Segmentation export

collab_splats/utils/visualization.py ← add plot_context_segmentation

tests/semantics/
  test_insid3_segmentation.py        ← new
```

No changes to `BaseSegmentation`, `BaseFeatureExtractor`, or any existing extractor.

**No new dependencies.** `sklearn` and `einops` already present.

---

## Out of Scope

- CRF mask refinement (`pydensecrf`) — future, noted; bilinear upsample sufficient for now
- Multi-reference (S>1 context images) — INSID3 supports it; start with single ref only
- SemanticsPane dashboard integration — follow-on spec
- Pre-extracted zarr tensor path — follow-on once basic API is solid
