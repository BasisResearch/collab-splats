# Semantics Query API Redesign — Design Spec

**Date:** 2026-04-22
**Branch:** `refactor/core-modules`

---

## Problem

Four issues in the current `features.py` + `semantics/utils.py`:

1. **`compute_similarity` fuses similarity and contrast** — returns a contrastive score shaped like a similarity map. Callers cannot access raw cosine similarities for visualization or downstream use.
2. **`compute_semantic_contrast` requires negatives** — `softmax(dim=0)` over queries is undefined without at least one negative; default `["background"]` is silently injected.
3. **Saturation with multiple positives** — old `"standard"` method sums positive probs in a joint softmax. With N positives + 1 weak negative, sum saturates near 1 everywhere.
4. **Resolution is locked** — `MaskCLIPExtractor` hardcodes `resolution=1024` inside `preprocess`. `DINOFeatureExtractor` sets it at init with no override path. Neither supports per-call resolution.

Additionally:
- `compute_similarity` and `score_queries` logic is duplicated between `MaskCLIPExtractor` and `Talk2DinoExtractor`.
- `model_name` is inconsistent across extractors (`hf_model_id` used in Talk2DINO notebook).
- `SupportsTextQuery` Protocol is redundant once a proper ABC exists.
- `softmax_temp` is a non-standard name; ML literature uses `temperature`.
- Old `"pairwise"` method had a bug: `avg_pos.expand(num_neg, -1)` gave positives `num_neg` slots in the softmax denominator, inflating scores proportionally to negative count.

---

## Design

### 1. Class hierarchy

```
BaseFeatureExtractor              registry, forward() abstract
├── DINOFeatureExtractor          patch features only — no text
└── BaseQueryableExtractor        text query ABC — owns compute_similarity + score_queries
    ├── MaskCLIPExtractor
    └── Talk2DinoExtractor
```

- `BaseQueryableExtractor` is abstract; cannot be instantiated directly.
- Registry (`@BaseFeatureExtractor.register`) stays on root class; subclasses inherit registration.
- `SupportsTextQuery` Protocol in `semantics/protocols.py` is **deleted**. Callers use `isinstance(x, BaseQueryableExtractor)`.

---

### 2. `BaseQueryableExtractor` interface

Subclasses implement only `encode_text` + `forward` + `preprocess`. All query logic lives in the base.

```python
class BaseQueryableExtractor(BaseFeatureExtractor):
    """Abstract base for feature extractors that support text-conditioned similarity queries.

    Subclasses must implement encode_text() and forward(). compute_similarity()
    and score_queries() are shared here — no duplication across subclasses.

    All subclasses accept model_name as the first __init__ parameter.
    """

    @abstractmethod
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Return normalized text embeddings of shape (N, D)."""
        ...

    def compute_similarity(
        self,
        features: torch.Tensor,
        queries: List[str],
    ) -> torch.Tensor:
        """Raw cosine similarities between patch features and text queries.

        Args:
            features: (C, H, W) patch feature map from forward().
            queries: text strings to compare against features.

        Returns:
            (N_queries, H, W) — one map per query. No reduction, no softmax.
            Use score_queries() for contrastive scoring.
        """
        # Requires features in (C, H, W) layout — all BaseQueryableExtractor subclasses
        # must return this shape from forward(). See Talk2DinoExtractor note below.
        text_embs = self.encode_text(queries)
        return torch.einsum("chw,nc->nhw", features, text_embs)

    def score_queries(
        self,
        features: torch.Tensor,
        positive: List[str],
        negative: Optional[List[str]] = None,
        temperature: float = 0.05,
        reduction: str = "max",
    ) -> torch.Tensor:
        """Contrastive score map: how well each patch matches positive vs negative queries.

        Computes raw similarities internally via compute_similarity, then applies
        contrastive scoring. Use compute_similarity() directly if you need raw maps.

        Args:
            features: (C, H, W) patch feature map from forward().
            positive: target concept queries.
            negative: background/contrast queries. None → raw reduction over positives only.
            temperature: softmax temperature τ. Lower = sharper spatial contrast.
            reduction: "max" or "pool" (see compute_semantic_contrast docs).

        Returns:
            (H, W) scores in [0, 1].
        """
        queries = positive + (negative or [])
        similarity = self.compute_similarity(features, queries)
        return compute_semantic_contrast(similarity, len(positive), temperature, reduction)
```

---

### 3. `compute_semantic_contrast` — fixed scoring

Replaces old version in `semantics/utils.py`. Drops `method` param, adds `reduction`, renames `softmax_temp` → `temperature`, fixes pairwise bug, handles no-negative case.

```python
def compute_semantic_contrast(
    raw_similarities: torch.Tensor,
    num_positive: int,
    temperature: float = 0.05,
    reduction: str = "max",
) -> torch.Tensor:
    """Contrastive scoring: how strongly positive queries match relative to negatives.

    When no negatives are present (num_positive == raw_similarities.shape[0]),
    falls back to raw reduction over positives — contrastive scoring is undefined
    without a negative to push against.

    Args:
        raw_similarities: (N_queries, N) dot-product similarities per patch.
        num_positive: rows [0:num_positive] are positive queries; rest are negative.
        temperature: scaling parameter τ. Lower = sharper. Ignored when no negatives.
        reduction: aggregation over positive queries:
            "max"  — each positive independently scored against all negatives via
                     binary softmax; max over per-positive scores. Semantics: does
                     the best-matching positive beat all negatives?
            "pool" — positives averaged in similarity space before softmax; one
                     representative competes against all negatives. Semantics: does
                     the combined concept beat all negatives? Preferred for synonyms.

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    # No negatives: skip contrastive step, reduce directly over raw similarities
    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        # Each positive independently: binary softmax against all negatives, then max
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    if reduction == "pool":
        # Average positives first, one representative vs all negatives
        avg_pos = pos.mean(dim=0, keepdim=True)
        stacked = torch.cat([avg_pos, neg], dim=0)
        return stacked.div(temperature).softmax(dim=0)[0]

    raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")
```

**Changes from old version:**
- `method="standard"|"pairwise"` → `reduction="max"|"pool"`
- `softmax_temp` → `temperature`
- `"max"` uses independent per-positive binary softmax (fixes inter-positive dilution in joint softmax)
- `"pool"` fixes old pairwise bug (`avg_pos.expand(num_neg)` gave positive `num_neg` slots — inflated score)
- No-negative fallback added

---

### 4. Resolution decoupling

All extractors accept `model_name` as first param and `resolution` override at call time.

#### `MaskCLIPExtractor`

```python
def __init__(self, model_name: str = "ViT-L/14@336px", resolution: int = 1024, ...):
    self.default_resolution = resolution
    ...

def preprocess(self, image, resolution: int | None = None) -> torch.Tensor:
    """Resize to longest edge, normalize. resolution overrides instance default."""
    resolution = resolution or self.default_resolution
    ...

def forward(self, images: list, resolution: int | None = None) -> list[torch.Tensor]:
    """Extract patch features. resolution passed through to preprocess."""
    preprocessed = [self.preprocess(img, resolution) for img in images]
    ...
```

#### `DINOFeatureExtractor`

Same pattern: `default_resolution` at init, `resolution` kwarg on `preprocess` + `forward`. Removes the current `self.resolution` used only in `preprocess`.

#### `Talk2DinoExtractor`

No `resolution` param. Center-crop to square is a model constraint, not a resolution choice. Document this clearly in the class docstring.

**`forward()` shape fix required:** Current `Talk2DinoExtractor.forward()` returns `(N_patches, D)` flat tensors. `BaseQueryableExtractor.compute_similarity` expects `(C, H, W)`. `Talk2DinoExtractor.forward()` must reshape to `(D, pH, pW)` — patch grid dimensions are derivable from the square crop size and `self.patch_size`.

---

### 5. Consistent `model_name` parameter

All three extractors use `model_name` as the first `__init__` parameter. No `hf_model_id` or other aliases anywhere (class bodies, notebooks, tests).

---

### 6. Deletions

| Item | Reason |
|---|---|
| `SupportsTextQuery` Protocol | Replaced by `isinstance(x, BaseQueryableExtractor)` |
| `method="standard"|"pairwise"` param | Replaced by `reduction="max"|"pool"` |
| `softmax_temp` param name | Renamed to `temperature` |
| Duplicated `compute_similarity` bodies in `MaskCLIPExtractor`, `Talk2DinoExtractor` | Moved to `BaseQueryableExtractor` |

---

### 7. Downstream migration — all callers

Every file that must change as part of this spec:

#### `collab_splats/semantics/protocols.py`
- **Delete entire file.** `SupportsTextQuery` is replaced by `isinstance(x, BaseQueryableExtractor)`.

#### `collab_splats/semantics/__init__.py`
- Remove `from .protocols import SupportsTextQuery`
- Remove `"SupportsTextQuery"` from `__all__`

#### `collab_splats/semantics/features.py`
- Add `BaseQueryableExtractor` class (see Section 2)
- `MaskCLIPExtractor` inherits from `BaseQueryableExtractor`; delete its `compute_similarity` body
- `Talk2DinoExtractor` inherits from `BaseQueryableExtractor`; delete its `compute_similarity` body
- `Talk2DinoExtractor.__init__`: rename `hf_model_id` → `model_name`
- `Talk2DinoExtractor.forward()`: reshape output from `(N_patches, D)` → `(D, pH, pW)`
- All extractors: add `default_resolution` + `resolution` kwarg on `preprocess` + `forward`

#### `collab_splats/semantics/utils.py`
- Replace `compute_semantic_contrast` with fixed version (Section 3)
- Rename `softmax_temp` → `temperature`, `method` → `reduction`

#### `collab_splats/dashboard/semantics.py`
- Line 661: `Talk2DinoExtractor(hf_model_id=...)` → `Talk2DinoExtractor(model_name=...)`
- Line 685: `extractor.compute_similarity(img_embed, positive, negative, self.temp_slider.value, self.method_dd.value)`
  → `extractor.score_queries(img_embed, positive, negative, temperature=self.temp_slider.value, reduction=self.reduction_dd.value)`
- Any `isinstance(extractor, SupportsTextQuery)` → `isinstance(extractor, BaseQueryableExtractor)`
- Import `BaseQueryableExtractor` instead of `SupportsTextQuery`

#### `collab_splats/nerfstudio/models/rade_features.py`
- Line 135: `self.similarity_fx = self.text_encoder.compute_similarity` — this stores a reference to the old fused `compute_similarity`. Callers of `similarity_fx` must be updated to call `score_queries` instead, or `similarity_fx` reassigned to `score_queries`. **Audit all `similarity_fx` call sites before implementing.**

#### `docs/semantics/feature_extraction.ipynb`
- `Talk2DinoExtractor(hf_model_id=...)` → `Talk2DinoExtractor(model_name=...)`
- All `extractor.compute_similarity(features, positive=..., negative=..., softmax_temp=..., method=...)` calls → `extractor.score_queries(features, positive, negative, temperature=..., reduction=...)`

#### `docs/semantics/maskclip_reference_comparison.ipynb`
- All `compute_similarity(...)` calls that expected `(H, W, 1)` output → update to `score_queries(...)` or handle `(N_queries, H, W)` raw output explicitly

#### Out of scope (do not touch)
- `stage/feedforward.py` and `stage/FeedforwardMeshing.ipynb` — use their own `softmax_temp` variable locally, not imported from this module

---

## Out of scope

- Resolution non-specificity in tree similarity maps — CLIP architecture limitation, not a preprocessing issue. Talk2DINO recommended for patch-level localization tasks.
- Vectorizing the `"max"` reduction loop in `compute_semantic_contrast` — correctness first.
- Batching `score_queries` across multiple images.

---

## Risks

- **Breaking change on `compute_similarity`**: old signature returned a `(H, W, 1)` contrastive score; new returns `(N_queries, H, W)` raw sims. All callers must switch to `score_queries()` for scored output. Notebooks need updating.
- **`method` param removal**: any external code passing `method="standard"` or `method="pairwise"` to `compute_semantic_contrast` will break. No backward-compat shim — this is a deliberate API clean-up.
- **`temperature` rename**: callers passing `softmax_temp=` as keyword arg will break.
- **`Talk2DinoExtractor.forward()` shape change**: currently returns `(N_patches, D)` flat; after this change returns `(D, pH, pW)`. Any caller that reshapes the output manually will break.
