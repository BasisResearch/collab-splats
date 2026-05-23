# Semantics Module Code Review

**Date:** 2026-04-21
**Branch:** `refactor/core-modules`
**Scope:** `collab_splats/semantics/features.py`, `collab_splats/semantics/segmentation.py`

## Context

PR1 (`refactor/core-modules`) is ready for review. Before opening, clean up four code quality issues in `semantics/`: duplicated image-loading logic, duplicated similarity branching logic, hardcoded default values, and a debug `print()` in production code. Public API is frozen — no signature changes, no class renames.

---

## Changes

### 1. Extract `_open_image()` — `features.py`

**Problem:** All three `preprocess()` methods contain identical `isinstance` branching to convert input to `PIL.Image`:

```python
# Duplicated in MaskCLIPExtractor.preprocess (L231), DINOFeatureExtractor.preprocess (L369),
# Talk2DinoExtractor.preprocess (L474)
if isinstance(image, (str, Path)):
    image = Image.open(image)
elif isinstance(image, np.ndarray):
    image = Image.fromarray(image)
elif isinstance(image, Image.Image):
    pass
else:
    raise ValueError(f"Unsupported image type: {type(image)}")
```

**Fix:** Module-level private function in `features.py`:

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

Each `preprocess()` replaces its branching block with `image = _open_image(image)`, then applies `.convert("RGB")` as needed (each extractor keeps its own convert logic — DINO omits it intentionally since it does `[:3]` slice afterward).

---

### 2. Extract `_apply_similarity_method()` — `features.py`

**Problem:** The `standard`/`pairwise` branching after `raw_similarities` is computed is ~15 identical lines in both `MaskCLIPExtractor.compute_similarity` (L314–328) and `Talk2DinoExtractor._compute_similarity` (L552–563). The two methods differ in how they compute `raw_similarities` (different feature shapes), but the branching is identical.

**Fix:** Module-level private function:

```python
def _apply_similarity_method(
    raw_similarities: torch.Tensor,  # (num_queries, N)
    num_positive: int,
    softmax_temp: float,
    method: str,
) -> torch.Tensor:  # (N,)
    if method == "standard":
        probs = (raw_similarities / softmax_temp).softmax(dim=0)
        return probs[:num_positive].sum(dim=0)
    elif method == "pairwise":
        pos_similarities = raw_similarities[:num_positive]
        neg_similarities = raw_similarities[num_positive:]
        avg_pos = pos_similarities.mean(dim=0, keepdim=True)
        paired = torch.cat([avg_pos.expand(neg_similarities.shape[0], -1), neg_similarities], dim=0)
        probs = (paired / softmax_temp).softmax(dim=0)
        return torch.nan_to_num(probs[: neg_similarities.shape[0]].min(dim=0)[0], nan=0.0)
    raise ValueError(f"Unknown method: {method}. Choose 'standard' or 'pairwise'")
```

Each class keeps its own encode + einsum/matmul to produce `raw_similarities`, then delegates to `_apply_similarity_method`. No shape coupling or inheritance changes.

Note: `MaskCLIPExtractor.compute_similarity` returns `(H, W, 1)` — the `.reshape(features.shape[1:] + (1,))` call stays in that method after the shared function returns `(H*W,)`.

---

### 3. `_DEFAULT_NEGATIVE` constant — `features.py`

**Problem:** `["object"]` hardcoded as default negative query in two places:
- `MaskCLIPExtractor.compute_similarity` L304
- `Talk2DinoExtractor._compute_similarity` L541

**Fix:** One module-level constant at the top of `features.py`:

```python
_DEFAULT_NEGATIVE: list[str] = ["object"]
```

Both methods replace `negative = ["object"]` with `negative = _DEFAULT_NEGATIVE`.

---

### 4. Replace `print()` with `logger.debug()` — `segmentation.py`

**Problem:** `segmentation.py:273` has a debug print that hits stdout unconditionally in production:

```python
print(f"Mask {i} has {mask.sum()} pixels")
```

**Fix:** Add logger at module level and convert to `debug`:

```python
import logging
logger = logging.getLogger(__name__)
# ...
logger.debug("Mask %d has %d pixels", i, mask.sum())
```

---

## Files Modified

| File | Lines touched |
|------|--------------|
| `collab_splats/semantics/features.py` | Add `_open_image`, `_apply_similarity_method`, `_DEFAULT_NEGATIVE`; update 3× `preprocess()`, 2× similarity methods |
| `collab_splats/semantics/segmentation.py` | Add `import logging`, `logger`, replace 1× `print()` |

## Files NOT modified

Everything else in `collab_splats/semantics/` — `protocols.py`, `frame_sampling.py`, `__init__.py` — untouched.

## Constraints

- No public API changes (class names, method signatures, `__init__` params unchanged)
- No new classes, no new files
- `BaseFeatureExtractor` inheritance chain unchanged
- All existing tests must pass without modification

## Verification

```bash
pytest tests/semantics/ -v
pytest tests/ -v --ignore=tests/nerfstudio --ignore=tests/pointcloud/test_mapanything_creator.py
```

Pre-existing failures (nerfstudio env + GPU smoke test) are excluded — not caused by this change.
