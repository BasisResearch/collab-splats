# Design: Semantic Lifting — Shape-Aware API + Tutorial Notebook

**Date:** 2026-05-21
**Status:** Approved

---

## Problem

`BaseQueryableExtractor.compute_similarity` and `score_queries` hardcode `einsum("chw,nc->nhw")`, requiring 3D `(C, H, W)` input. `lift_features` returns `(P, D)` point arrays, so callers must manually reshape. This makes the semantic-lifting pattern clunky and undiscoverable.

---

## Design Decision: Guard Only in `compute_similarity`

`score_queries` delegates to `self.compute_similarity` internally. Adding the shape guard only in `compute_similarity` is sufficient:

- `compute_similarity(features=(P,D))` → reshapes to `(D, P, 1)` → einsum → squeezes → `(N_queries, P)`
- `score_queries` calls `compute_similarity`, gets `(N_queries, P)`, passes to `compute_semantic_contrast` which gets clean 2D input → returns `(P,)`
- `score_queries` body unchanged; only docstring updated

The alternative (guard in both methods) was rejected: it routes `(N_queries, P, 1)` into `compute_semantic_contrast`, relying on undocumented shape-agnosticism in that function.

---

## Shape Contract

| Input shape | `compute_similarity` output | `score_queries` output |
|---|---|---|
| `(C, H, W)` | `(N_queries, H, W)` | `(H, W)` — unchanged |
| `(P, D)`    | `(N_queries, P)`    | `(P,)` — new |

---

## Implementation

### Part 1 — `collab_splats/semantics/features.py`

`BaseQueryableExtractor.compute_similarity`:

```python
is_points = features.ndim == 2
if is_points:
    features = features.T.unsqueeze(-1)  # (D, P, 1)
text_embs = self.encode_text(queries)
out = torch.einsum("chw,nc->nhw", features, text_embs)
return out.squeeze(-1) if is_points else out
```

`score_queries`: docstring updated to document `(P, D) → (P,)` contract. Body unchanged.

### Part 2 — `tests/semantics/test_features.py`

4 tests using `MaskCLIPExtractor(device="cpu")`:
- `test_compute_similarity_point_array` — shape `(2, P)`
- `test_score_queries_point_array` — shape `(P,)` + range `[0, 1]`
- `test_compute_similarity_image_map_unchanged` — regression `(C, H, W)` path
- `test_score_queries_image_map_unchanged` — regression `(C, H, W)` path

### Part 3 — `docs/source/tutorials/semantics/semantic_lifting.ipynb`

7-section notebook demonstrating the full pipeline: keyframe extraction → VGGT-X reconstruction → MaskCLIP feature lifting → text queries → PyVista 3D viewer.

---

## Utilities Referenced

| Utility | Path |
|---|---|
| `BaseQueryableExtractor.compute_similarity` | `collab_splats/semantics/features.py:278` |
| `BaseQueryableExtractor.score_queries` | `collab_splats/semantics/features.py:297` |
| `compute_semantic_contrast` | `collab_splats/semantics/utils.py:33` |
| `lift_features` | `collab_splats/pointcloud/utils.py:706` |
| `sample_frames_optical_flow` | `collab_splats/utils/frame_sampling.py` |
| `VGGTXCreator` | `collab_splats/pointcloud/feedforward/vggtx.py` |
| `pointcloud_to_polydata` | `collab_splats/utils/visualization.py` |
