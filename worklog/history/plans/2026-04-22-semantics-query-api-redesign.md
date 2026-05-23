# Semantics Query API Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the semantics feature extraction API to separate raw similarity from contrastive scoring, fix scoring math bugs, add resolution flexibility, and eliminate code duplication via a `BaseQueryableExtractor` ABC.

**Architecture:** Introduce `BaseQueryableExtractor(BaseFeatureExtractor)` that owns `compute_similarity` (raw cosine sims, `(N_queries, H, W)`) and `score_queries` (contrastive scoring, `(H, W)`). Subclasses implement only `encode_text` + `forward`. Fix `compute_semantic_contrast` with `reduction="max"|"pool"`, renamed `temperature` param, no-negative fallback, and correct independent-softmax scoring. Delete `SupportsTextQuery` Protocol; migrate all call sites.

**Tech Stack:** PyTorch, Python ABCs (`abc.abstractmethod`), HuggingFace `transformers`, `maskclip_onnx`, `panel` (dashboard), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/semantics/utils.py` | Modify | Replace `compute_semantic_contrast` with fixed version |
| `collab_splats/semantics/features.py` | Modify | Add `BaseQueryableExtractor`; migrate `MaskCLIPExtractor` + `Talk2DinoExtractor` |
| `collab_splats/semantics/protocols.py` | **Delete** | Replaced by `isinstance(x, BaseQueryableExtractor)` |
| `collab_splats/semantics/__init__.py` | Modify | Swap `SupportsTextQuery` for `BaseQueryableExtractor` |
| `collab_splats/dashboard/semantics.py` | Modify | Update 3 call sites + widget options |
| `collab_splats/nerfstudio/models/rade_features.py` | Modify | Replace `similarity_fx` reference + fix shape assumptions |
| `docs/semantics/feature_extraction.ipynb` | Modify | Update `compute_similarity` → `score_queries`, `hf_model_id` → `model_name` |
| `docs/semantics/maskclip_reference_comparison.ipynb` | Modify | Update `compute_similarity` call sites |
| `tests/semantics/test_semantics_utils.py` | Modify | Update existing tests to new param names |
| `tests/semantics/test_query_api.py` | **Create** | New tests for scoring math, ABC contract, resolution, shapes |

---

### Task 1: Fix `compute_semantic_contrast` in `utils.py`

**Files:**
- Modify: `collab_splats/semantics/utils.py`
- Modify: `tests/semantics/test_semantics_utils.py`
- Create: `tests/semantics/test_query_api.py`

- [ ] **Step 1: Update existing tests in `test_semantics_utils.py` to new API**

The existing tests use `softmax_temp=` and `method=` — update them to `temperature=` and `reduction=`. Replace the full test file content:

```python
"""Tests for collab_splats.semantics.utils — torch/model utilities."""

import pytest
import torch

from collab_splats.semantics.utils import (
    compute_semantic_contrast,
    interpolate_to_patch_size,
    pytorch_gc,
    infer_batch_size,
    batch_iterator,
)


def test_max_contrast_shape():
    raw = torch.randn(5, 100)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="max")
    assert result.shape == (100,)


def test_max_contrast_bounds():
    raw = torch.randn(3, 50)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="max")
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_pool_contrast_shape():
    raw = torch.randn(4, 100)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="pool")
    assert result.shape == (100,)


def test_pool_contrast_bounds():
    raw = torch.randn(4, 50)
    result = compute_semantic_contrast(raw, num_positive=2, temperature=0.05, reduction="pool")
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_unknown_reduction_raises():
    raw = torch.randn(3, 50)
    with pytest.raises(ValueError, match="Unknown reduction"):
        compute_semantic_contrast(raw, num_positive=1, temperature=0.05, reduction="invalid")


def test_no_negatives_max_returns_raw_max():
    """No negatives: falls back to max over positives, no softmax."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])  # 2 positives, 2 patches
    result = compute_semantic_contrast(sims, num_positive=2, temperature=0.05, reduction="max")
    assert torch.allclose(result, sims.max(dim=0).values)


def test_no_negatives_pool_returns_raw_mean():
    """No negatives: falls back to mean over positives."""
    sims = torch.tensor([[0.8, 0.2], [0.6, 0.9]])
    result = compute_semantic_contrast(sims, num_positive=2, temperature=0.05, reduction="pool")
    assert torch.allclose(result, sims.mean(dim=0))


def test_max_single_positive_wins_on_high_sim_patch():
    """Patch with high positive similarity and low negative similarity scores > 0.5."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])  # (pos, neg) x 2 patches
    result = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="max")
    assert result[0] > 0.5  # patch 0: positive dominates
    assert result[1] < 0.5  # patch 1: negative dominates


def test_max_not_inflated_by_multiple_positives():
    """Adding synonym positives should not drastically inflate or deflate score."""
    # 1 positive, 1 negative
    sims_1pos = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    # 3 synonym positives, same negative — patch 0 should still score high
    sims_3pos = torch.tensor([[0.9, 0.1], [0.85, 0.05], [0.88, 0.08], [-0.1, 0.9]])
    r1 = compute_semantic_contrast(sims_1pos, num_positive=1, temperature=0.05, reduction="max")
    r3 = compute_semantic_contrast(sims_3pos, num_positive=3, temperature=0.05, reduction="max")
    assert abs(r1[0].item() - r3[0].item()) < 0.15


def test_pool_single_pos_matches_max():
    """With one positive, pool and max should return identical results."""
    sims = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
    r_max = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="max")
    r_pool = compute_semantic_contrast(sims, num_positive=1, temperature=0.05, reduction="pool")
    assert torch.allclose(r_max, r_pool)


def test_interpolate_divisible():
    img = torch.randn(1, 3, 224, 224)
    result, h, w = interpolate_to_patch_size(img, patch_size=14)
    assert h % 14 == 0
    assert w % 14 == 0
    assert result.shape == (1, 3, h, w)


def test_interpolate_non_divisible():
    img = torch.randn(1, 3, 230, 230)
    result, h, w = interpolate_to_patch_size(img, patch_size=14)
    assert h % 14 == 0
    assert h == 224


def test_pytorch_gc_no_error():
    pytorch_gc()


def test_infer_batch_size_cpu():
    if not torch.cuda.is_available():
        assert infer_batch_size(3.0) == 1


def test_infer_batch_size_negative_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(-1.0)


def test_infer_batch_size_zero_raises():
    with pytest.raises(ValueError, match="must be positive"):
        infer_batch_size(0.0)


def test_batch_iterator_basic():
    items = list(range(10))
    batches = list(batch_iterator(3, items))
    assert len(batches) == 4
    assert batches[0] == [[0, 1, 2]]
    assert batches[-1] == [[9]]


def test_batch_iterator_multiple_args():
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    batches = list(batch_iterator(2, a, b))
    assert len(batches) == 2
    assert batches[0] == [[1, 2], [5, 6]]


def test_batch_iterator_mismatched_raises():
    with pytest.raises(AssertionError):
        list(batch_iterator(2, [1, 2], [3]))
```

- [ ] **Step 2: Run tests to verify they fail with current code**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_semantics_utils.py -v 2>&1 | tail -20
```

Expected: Several FAIL — `temperature` and `reduction` kwargs not recognized.

- [ ] **Step 3: Replace `compute_semantic_contrast` in `collab_splats/semantics/utils.py`**

Find and replace the entire `compute_semantic_contrast` function (keep all other functions intact):

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
                     binary softmax; max over per-positive scores. Use for distinct
                     concepts where any match counts.
            "pool" — positives averaged in similarity space before softmax; one
                     representative competes against all negatives. Use for synonymous
                     concepts that should be treated as one combined query.

    Returns:
        (N,) contrastive scores in [0, 1].
    """
    pos = raw_similarities[:num_positive]
    neg = raw_similarities[num_positive:]

    # No negatives: skip contrastive step, reduce directly over raw similarities
    if neg.shape[0] == 0:
        return pos.max(dim=0).values if reduction == "max" else pos.mean(dim=0)

    if reduction == "max":
        # Each positive independently: binary softmax against all negatives, then max.
        # Avoids inter-positive dilution that occurs in joint softmax.
        scores = []
        for p_i in pos:
            stacked = torch.cat([p_i.unsqueeze(0), neg], dim=0)
            scores.append(stacked.div(temperature).softmax(dim=0)[0])
        return torch.stack(scores).max(dim=0).values

    if reduction == "pool":
        # Average positives first — one representative vs all negatives.
        # One slot in denominator regardless of how many positives were passed.
        avg_pos = pos.mean(dim=0, keepdim=True)
        stacked = torch.cat([avg_pos, neg], dim=0)
        return stacked.div(temperature).softmax(dim=0)[0]

    raise ValueError(f"Unknown reduction '{reduction}'. Choose 'max' or 'pool'.")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_semantics_utils.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/utils.py tests/semantics/test_semantics_utils.py
git commit -m "fix(semantics): replace compute_semantic_contrast — temperature param, max/pool reduction, no-negative fallback, fix pairwise bug"
```

---

### Task 2: Add `BaseQueryableExtractor` to `features.py`

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Create: `tests/semantics/test_query_api.py`

- [ ] **Step 1: Create `tests/semantics/test_query_api.py` with contract tests**

```python
"""Tests for BaseQueryableExtractor contract and query API."""

import pytest
import torch

from collab_splats.semantics.features import BaseQueryableExtractor


class _ConcreteExtractor(BaseQueryableExtractor):
    """Minimal concrete subclass for testing the base class contract."""

    def encode_text(self, texts):
        # Returns unit vectors of shape (N, 8)
        n = len(texts)
        emb = torch.zeros(n, 8)
        emb[:, 0] = 1.0
        return emb

    def forward(self, images):
        # Returns (8, 4, 4) feature map per image
        return [torch.zeros(8, 4, 4) for _ in images]


def test_base_not_instantiable_without_encode_text():
    """BaseQueryableExtractor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseQueryableExtractor()


def test_isinstance_check():
    """Concrete subclass satisfies isinstance check."""
    extractor = _ConcreteExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)


def test_compute_similarity_shape():
    """compute_similarity returns (N_queries, H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.compute_similarity(features, ["cat", "dog", "background"])
    assert result.shape == (3, 4, 4)


def test_score_queries_shape_with_negative():
    """score_queries returns (H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"], negative=["background"])
    assert result.shape == (4, 4)


def test_score_queries_shape_no_negative():
    """score_queries with no negative does not raise and returns (H, W)."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"])
    assert result.shape == (4, 4)


def test_score_queries_bounds():
    """score_queries output is in [0, 1]."""
    extractor = _ConcreteExtractor()
    features = torch.randn(8, 4, 4)
    result = extractor.score_queries(features, positive=["cat"], negative=["background"])
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_query_api.py -v 2>&1 | tail -15
```

Expected: FAIL — `BaseQueryableExtractor` not yet defined.

- [ ] **Step 3: Add `BaseQueryableExtractor` to `collab_splats/semantics/features.py`**

Add `from abc import abstractmethod` to the top-level imports.

After the `BaseFeatureExtractor` class definition and before the `@BaseFeatureExtractor.register("maskclip")` decorator, insert:

```python
class BaseQueryableExtractor(BaseFeatureExtractor):
    """Abstract base for feature extractors that support text-conditioned similarity queries.

    Subclasses must implement encode_text() and forward(). compute_similarity()
    and score_queries() are provided here and shared across all queryable extractors.
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
            features: (C, H, W) patch feature map from forward(). All
                BaseQueryableExtractor subclasses must return (C, H, W) from forward().
            queries: text strings to compare against features.

        Returns:
            (N_queries, H, W) — one similarity map per query. No reduction, no softmax.
            Use score_queries() for contrastive scoring.
        """
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

        Computes raw similarities internally then applies contrastive scoring.
        Use compute_similarity() directly to access raw per-query maps for visualization.

        Args:
            features: (C, H, W) patch feature map from forward().
            positive: target concept queries.
            negative: background/contrast queries. None → raw reduction over positives only.
            temperature: softmax temperature τ. Lower = sharper spatial contrast.
            reduction: "max" (each positive independently vs all negatives, take best) or
                "pool" (average positives to one representative, then contrast).

        Returns:
            (H, W) scores in [0, 1].
        """
        queries = positive + (negative or [])
        similarity = self.compute_similarity(features, queries)
        return compute_semantic_contrast(similarity, len(positive), temperature, reduction)
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_query_api.py tests/semantics/test_semantics_utils.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_query_api.py
git commit -m "feat(semantics): add BaseQueryableExtractor ABC — shared compute_similarity and score_queries"
```

---

### Task 3: Migrate `MaskCLIPExtractor` to `BaseQueryableExtractor`

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `tests/semantics/test_query_api.py`

- [ ] **Step 1: Add resolution tests**

Append to `tests/semantics/test_query_api.py`:

```python
def test_maskclip_default_resolution():
    """MaskCLIPExtractor stores default_resolution at init."""
    pytest.importorskip("maskclip_onnx")
    from collab_splats.semantics.features import MaskCLIPExtractor
    extractor = MaskCLIPExtractor(resolution=512)
    assert extractor.default_resolution == 512


def test_maskclip_is_queryable():
    """MaskCLIPExtractor is a BaseQueryableExtractor."""
    pytest.importorskip("maskclip_onnx")
    from collab_splats.semantics.features import MaskCLIPExtractor
    extractor = MaskCLIPExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_query_api.py::test_maskclip_default_resolution -v 2>&1 | tail -10
```

Expected: FAIL — `default_resolution` not present.

- [ ] **Step 3: Update `MaskCLIPExtractor` in `features.py`**

Make these changes:

**a) Change class declaration:**
```python
# Before:
class MaskCLIPExtractor(BaseFeatureExtractor):
# After:
class MaskCLIPExtractor(BaseQueryableExtractor):
```

**b) Update `__init__` — add `resolution` param, store `default_resolution`:**
```python
def __init__(
    self,
    model_name: str = "ViT-L/14@336px",
    resolution: int = 1024,
    cache_dir: str = TORCH_HOME,
    device: Optional[str] = None,
):
    if device is None:
        device = get_device()
    if not _MASKCLIP_AVAILABLE:
        raise ImportError(
            "maskclip_onnx is not installed. Install with: pip install maskclip_onnx"
        )
    super().__init__()
    self.default_resolution = resolution

    self.model, _ = maskclip_onnx.clip.load(model_name, download_root=cache_dir)
    self.model = self.model.to(device)
    self.model.eval()

    self.patch_size = self.model.visual.patch_size
    self.transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
```

**c) Update `preprocess` — add `resolution` override:**
```python
def preprocess(self, image, resolution: Optional[int] = None) -> torch.Tensor:
    """Open, resize, normalize an image and return a (C, H, W) tensor on device.

    Args:
        image: PIL Image or path accepted by open_image().
        resolution: longest-edge resize target. Defaults to self.default_resolution.
    """
    resolution = resolution or self.default_resolution
    image = open_image(image).convert("RGB")
    image = resize_image(image, longest_edge=resolution)
    return self.transform(image).to(self.device)
```

**d) Update `forward` — add `resolution` kwarg:**
```python
def forward(self, images: list, resolution: Optional[int] = None) -> list[torch.Tensor]:
    """Preprocess images, extract patch features, return one (C, pH, pW) tensor per image.

    Args:
        images: list of PIL Images or paths.
        resolution: longest-edge resize target. Defaults to self.default_resolution.
    """
    preprocessed = [self.preprocess(img, resolution) for img in images]
    stacked = torch.stack(preprocessed)  # (B, C, H, W)
    b, _, H, W = stacked.shape
    patch_h = H // self.patch_size
    patch_w = W // self.patch_size
    with torch.no_grad():
        features = self.model.get_patch_encodings(stacked).to(torch.float32)
        features = F.normalize(features, dim=-1)
        features = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
    return list(features)
```

**e) Delete the entire `compute_similarity` method** — it is now inherited from `BaseQueryableExtractor`. The `encode_text` method stays as-is.

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/ -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_query_api.py
git commit -m "refactor(semantics): migrate MaskCLIPExtractor to BaseQueryableExtractor — resolution param, delete duplicated compute_similarity"
```

---

### Task 4: Migrate `Talk2DinoExtractor` to `BaseQueryableExtractor`

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Modify: `tests/semantics/test_query_api.py`

- [ ] **Step 1: Add tests**

Append to `tests/semantics/test_query_api.py`:

```python
def test_talk2dino_accepts_model_name():
    """Talk2DinoExtractor uses model_name, not hf_model_id."""
    pytest.importorskip("transformers")
    import inspect
    from collab_splats.semantics.features import Talk2DinoExtractor
    sig = inspect.signature(Talk2DinoExtractor.__init__)
    assert "model_name" in sig.parameters
    assert "hf_model_id" not in sig.parameters


def test_talk2dino_forward_returns_spatial():
    """Talk2DinoExtractor.forward() returns (D, pH, pW), not (N_patches, D)."""
    pytest.importorskip("transformers")
    from collab_splats.semantics.features import Talk2DinoExtractor
    from PIL import Image
    import numpy as np
    extractor = Talk2DinoExtractor()
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    features = extractor.forward([img])
    assert len(features) == 1
    assert features[0].ndim == 3  # (D, pH, pW), not flat


def test_talk2dino_is_queryable():
    """Talk2DinoExtractor is a BaseQueryableExtractor."""
    pytest.importorskip("transformers")
    from collab_splats.semantics.features import Talk2DinoExtractor
    extractor = Talk2DinoExtractor()
    assert isinstance(extractor, BaseQueryableExtractor)
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/test_query_api.py::test_talk2dino_accepts_model_name tests/semantics/test_query_api.py::test_talk2dino_forward_returns_spatial -v 2>&1 | tail -15
```

Expected: FAIL — `hf_model_id` still present; forward returns flat tensor.

- [ ] **Step 3: Update `Talk2DinoExtractor` in `features.py`**

**a) Change class declaration:**
```python
# Before:
class Talk2DinoExtractor(BaseFeatureExtractor):
# After:
class Talk2DinoExtractor(BaseQueryableExtractor):
```

**b) Update `__init__` — rename `hf_model_id` → `model_name`:**
```python
def __init__(
    self,
    model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
    device: Optional[str] = None,
):
    """
    Args:
        model_name: HuggingFace Hub model ID.
            DINOv3 (default): "lorebianchi98/Talk2DINOv3-ViTB"
            DINOv2: "lorebianchi98/Talk2DINO-ViTB"
        device: Torch device string ("cpu" or "cuda").

    Note: No resolution param. Talk2DINO requires center-crop to square — this
    is a model architecture constraint, not a configurable resolution.
    """
    if device is None:
        device = get_device()
    super().__init__()
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise ImportError(
            "transformers is required for Talk2DinoExtractor. "
            "Install via: pip install transformers"
        ) from e

    self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    self.patch_size: int = getattr(self._model.config, "patch_size", 14)
    self._device = torch.device(device)
```

**c) Update `forward()` — reshape output to `(D, pH, pW)`:**
```python
def forward(self, images: list) -> list[torch.Tensor]:
    """Preprocess images, extract patch tokens, return one (D, pH, pW) tensor per image.

    Talk2DINO center-crops to square, so the patch grid is always square (pH == pW).
    Output shape matches MaskCLIPExtractor and DINOFeatureExtractor — required by
    BaseQueryableExtractor.compute_similarity which expects (C, H, W) spatial layout.
    """
    preprocessed = [self.preprocess(img) for img in images]
    with torch.no_grad():
        result = self._model.encode_image(preprocessed)
    patch_tokens = list(result) if isinstance(result, torch.Tensor) else result

    outputs = []
    for tokens in patch_tokens:
        # tokens: (N_patches, D) — reshape to (D, pH, pW)
        # Center-crop guarantees square patch grid: pH == pW == sqrt(N_patches)
        n_patches = tokens.shape[0]
        ph = pw = int(n_patches ** 0.5)
        outputs.append(tokens.reshape(ph, pw, -1).permute(2, 0, 1))  # (D, pH, pW)
    return outputs
```

**d) Delete the entire `compute_similarity` method** — inherited from `BaseQueryableExtractor`. Keep `encode_text` and `preprocess`.

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/semantics/ -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/features.py tests/semantics/test_query_api.py
git commit -m "refactor(semantics): migrate Talk2DinoExtractor — BaseQueryableExtractor, model_name, (D,pH,pW) forward shape"
```

---

### Task 5: Delete `protocols.py` and update `__init__.py`

**Files:**
- Delete: `collab_splats/semantics/protocols.py`
- Modify: `collab_splats/semantics/__init__.py`

- [ ] **Step 1: Delete `protocols.py`**

```bash
rm /workspace/collab-splats/collab_splats/semantics/protocols.py
```

- [ ] **Step 2: Update `collab_splats/semantics/__init__.py`**

Remove:
```python
from .protocols import SupportsTextQuery
```
and from `__all__`:
```python
    "SupportsTextQuery",
```

Add `BaseQueryableExtractor` to the features import block:
```python
from .features import (
    BaseFeatureExtractor,
    BaseQueryableExtractor,
    MaskCLIPExtractor,
    DINOFeatureExtractor,
    Talk2DinoExtractor,
)
```

Add to `__all__` alongside `BaseFeatureExtractor`:
```python
    "BaseFeatureExtractor",
    "BaseQueryableExtractor",
```

- [ ] **Step 3: Verify no remaining `SupportsTextQuery` or `protocols` imports in main package**

```bash
grep -rn 'SupportsTextQuery\|from.*protocols' /workspace/collab-splats/collab_splats --include='*.py'
```

Expected: no output.

- [ ] **Step 4: Run full test suite**

```bash
cd /workspace/collab-splats && python -m pytest tests/ -v 2>&1 | tail -25
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/__init__.py
git rm collab_splats/semantics/protocols.py
git commit -m "refactor(semantics): delete SupportsTextQuery protocol — replaced by isinstance(x, BaseQueryableExtractor)"
```

---

### Task 6: Migrate `dashboard/semantics.py`

**Files:**
- Modify: `collab_splats/dashboard/semantics.py`

- [ ] **Step 1: Update `SupportsTextQuery` isinstance check**

Find the import and isinstance check (search: `grep -n 'SupportsTextQuery\|protocols' collab_splats/dashboard/semantics.py`).

Replace:
```python
from collab_splats.semantics.protocols import SupportsTextQuery
if isinstance(extractor, SupportsTextQuery):
```
With:
```python
from collab_splats.semantics import BaseQueryableExtractor
if isinstance(extractor, BaseQueryableExtractor):
```

- [ ] **Step 2: Update `Talk2DinoExtractor` instantiation (line ~661)**

```python
# Before:
extractor = Talk2DinoExtractor(hf_model_id=self.hf_model_dd.value, device=self.query_device_dd.value)
# After:
extractor = Talk2DinoExtractor(model_name=self.hf_model_dd.value, device=self.query_device_dd.value)
```

- [ ] **Step 3: Update `method_dd` widget and call site**

Widget definition (line ~237) — rename widget and update options from `"standard"/"pairwise"` to `"max"/"pool"`:
```python
# Before:
self.method_dd = pn.widgets.Select(name="Method", options=["standard", "pairwise"], width=150)
# After:
self.reduction_dd = pn.widgets.Select(name="Reduction", options=["max", "pool"], width=150)
```

Update call site (line ~685):
```python
# Before:
sim = extractor.compute_similarity(img_embed, positive, negative, self.temp_slider.value, self.method_dd.value)
# After:
sim = extractor.score_queries(img_embed, positive, negative, temperature=self.temp_slider.value, reduction=self.reduction_dd.value)
```

Update widget layout (lines ~798-799) — replace `self.method_dd` with `self.reduction_dd` wherever it appears in panel layouts.

- [ ] **Step 4: Verify no remaining old API references**

```bash
grep -n 'SupportsTextQuery\|hf_model_id\|method_dd\|softmax_temp\|compute_similarity' /workspace/collab-splats/collab_splats/dashboard/semantics.py
```

Expected: no output.

- [ ] **Step 5: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/dashboard/ tests/semantics/ -v 2>&1 | tail -20
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/dashboard/semantics.py
git commit -m "fix(dashboard): update semantics widget — score_queries API, model_name, reduction_dd replaces method_dd"
```

---

### Task 7: Migrate `rade_features.py`

**Files:**
- Modify: `collab_splats/nerfstudio/models/rade_features.py`

The call site (lines ~135, 507-512) does three things that all need updating:

```python
# Line 135 — stores reference to old fused compute_similarity
self.similarity_fx = self.text_encoder.compute_similarity

# Lines 507-512 — calls with old API and expects (H, W, 1) output
similarity_map = self.similarity_fx(
    features=decoded_features_dict[self.main_features_name],
    positive=self.positive_queries,
    negative=self.negative_queries,
    method=self.config.similarity_method,
)
assert similarity_map.shape[2] == 1                          # expects (H, W, 1)
similarity_map.permute(2, 0, 1)[None]                        # treats it as (H, W, 1)
```

- [ ] **Step 1: Update `similarity_fx` assignment (line ~135)**

```python
# Before:
self.similarity_fx = self.text_encoder.compute_similarity
# After:
self.similarity_fx = self.text_encoder.score_queries
```

- [ ] **Step 2: Update call site (lines ~507-512)**

```python
# Before:
similarity_map = self.similarity_fx(
    features=decoded_features_dict[self.main_features_name],
    positive=self.positive_queries,
    negative=self.negative_queries,
    method=self.config.similarity_method,
)
assert similarity_map.shape[2] == 1
# ... similarity_map.permute(2, 0, 1)[None] ...

# After:
similarity_map = self.similarity_fx(
    features=decoded_features_dict[self.main_features_name],
    positive=self.positive_queries,
    negative=self.negative_queries,
    reduction=self.config.similarity_method,  # config values must be "max" or "pool"
)
# score_queries returns (H, W) — add channel dim for interpolation
assert similarity_map.ndim == 2
# ... similarity_map[None, None] instead of similarity_map.permute(2,0,1)[None] ...
```

Find the full interpolation block referencing `similarity_map.permute(2, 0, 1)[None]` and update to `similarity_map[None, None]` (adds batch + channel dims: `(1, 1, H, W)`).

- [ ] **Step 3: Update `similarity_method` config values**

Find the model config class (likely a dataclass near the top of the file). If `similarity_method` has a default of `"standard"` or `"pairwise"`, update to `"max"`:

```python
# Before:
similarity_method: str = "standard"
# After:
similarity_method: str = "max"
```

- [ ] **Step 4: Run tests**

```bash
cd /workspace/collab-splats && python -m pytest tests/nerfstudio/ tests/semantics/ -v 2>&1 | tail -20
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/nerfstudio/models/rade_features.py
git commit -m "fix(nerfstudio): update rade_features — score_queries API, (H,W) similarity map, reduction param"
```

---

### Task 8: Update notebooks

**Files:**
- Modify: `docs/semantics/feature_extraction.ipynb`
- Modify: `docs/semantics/maskclip_reference_comparison.ipynb`

- [ ] **Step 1: Update `feature_extraction.ipynb` — `hf_model_id` → `model_name`**

Find the cell containing:
```python
extractor_t2d = Talk2DinoExtractor(hf_model_id="lorebianchi98/Talk2DINOv3-ViTB")
```
Change to:
```python
extractor_t2d = Talk2DinoExtractor(model_name="lorebianchi98/Talk2DINOv3-ViTB")
```

- [ ] **Step 2: Update `feature_extraction.ipynb` — `compute_similarity` → `score_queries`**

Find all cells with `extractor.compute_similarity(...)` and replace with `score_queries`:
```python
# Before:
sim_map_clip = extractor_clip.compute_similarity(
    features_clip, positive=["bird"], negative=["background", "sky"], softmax_temp=0.05
)
# After:
sim_map_clip = extractor_clip.score_queries(
    features_clip, positive=["bird"], negative=["background", "sky"], temperature=0.05
)
```

Same for Talk2DINO section:
```python
# Before:
sim_t2d = extractor_t2d.compute_similarity(
    features_t2d[0], positive=positive, negative=negative
)
# After:
sim_t2d = extractor_t2d.score_queries(
    features_t2d[0], positive=positive, negative=negative
)
```

- [ ] **Step 3: Update `feature_extraction.ipynb` — remove Talk2DINO manual reshape**

After the forward shape fix in Task 4, `Talk2DinoExtractor.forward()` returns `(D, pH, pW)`. Any cell that manually reshapes `(N_patches, D)` → `(D, pH, pW)` should be removed or updated with a comment:

```python
# No reshape needed — forward() now returns (D, pH, pW) directly
features_t2d_spatial = features_t2d[0]  # (D, pH, pW)
```

- [ ] **Step 4: Update `maskclip_reference_comparison.ipynb`**

Find all `extractor.compute_similarity(...)` calls. Each one expected `(H, W, 1)` output via `[..., 0]` squeeze. Replace with `score_queries`:

```python
# Before:
sim_raw = extractor.compute_similarity(feat_1024_chw, positive=[query], negative=neg)[..., 0]
# After — score_queries returns (H, W) directly, no squeeze:
sim_raw = extractor.score_queries(feat_1024_chw, positive=[query], negative=neg)
```

If any cell needs raw per-query maps for diagnostic purposes, use the new `compute_similarity` signature:
```python
# Raw similarities (N_queries, H, W) — for visualization/debugging:
raw_sims = extractor.compute_similarity(features, queries=positive + negative)
# raw_sims[0] is the first positive query map, raw_sims[-1] is the negative map
```

- [ ] **Step 5: Commit**

```bash
git add docs/semantics/feature_extraction.ipynb docs/semantics/maskclip_reference_comparison.ipynb
git commit -m "docs(semantics): update notebooks — score_queries API, model_name, Talk2DINO spatial forward"
```

---

## Self-Review

**Spec coverage:**
- ✅ Section 1 (hierarchy): Tasks 2–4
- ✅ Section 2 (BaseQueryableExtractor interface): Task 2
- ✅ Section 3 (compute_semantic_contrast fixed): Task 1
- ✅ Section 4 (resolution decoupling): Task 3
- ✅ Section 5 (model_name consistency): Tasks 4, 6, 8
- ✅ Section 6 (deletions): Tasks 3–5 (`compute_similarity` bodies, protocol)
- ✅ Section 7 (downstream migration): Tasks 5–8

**Placeholder scan:** None. All code steps contain complete implementations.

**Type consistency:**
- `compute_semantic_contrast(raw_similarities, num_positive, temperature, reduction)` — Task 1 defines; Task 2 calls via `score_queries`; consistent throughout.
- `score_queries(features, positive, negative, temperature, reduction) → (H, W)` — Task 2 defines; Tasks 6, 7, 8 call sites use matching kwargs.
- `compute_similarity(features, queries) → (N_queries, H, W)` — Task 2 defines; Task 8 notebook callers remove old `[..., 0]` squeeze.
- `Talk2DinoExtractor(model_name=...)` — Task 4 defines; Tasks 6, 8 use `model_name=`.
- `similarity_map` in `rade_features.py` changes from `(H, W, 1)` → `(H, W)` — Task 7 updates both the assert and the permute/unsqueeze downstream.
