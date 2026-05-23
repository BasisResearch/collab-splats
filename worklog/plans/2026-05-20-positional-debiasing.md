# Positional Debiasing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SVD-based positional debiasing (INSID3 algorithm) to `BaseFeatureExtractor` as a standalone `debias()` method. Subclass `forward()` methods are completely untouched.

**Architecture:** `debias(features)` is a new public method on `BaseFeatureExtractor` that operates on already-extracted patch features. It builds a positional basis lazily from a zero-pixel image (using the subclass's own `forward()`) on first call per resolution, then caches. `get_bias_visualization()` provides a PCA-based RGB heatmap for inspection.

**Tech Stack:** PyTorch (`torch.linalg.svd`, `F.normalize`, `F.interpolate`), PIL, NumPy.

**Implementation notes:** All imports at the top of the file. Every non-obvious line gets an inline comment explaining WHY it is there (not just what it does).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/semantics/features.py` | Modify | Add `import numpy as np`, `_DEBIAS_VALIDATED`, `BaseFeatureExtractor.__init__`, `debias()`, `_build_positional_basis()`, `_apply_debias()`, `get_bias_visualization()`, `**kwargs` in concrete `__init__` methods |
| `tests/semantics/test_positional_debiasing.py` | Create | All tests — uses lightweight fake extractor, no real model weights |

---

### Task 1: Add `BaseFeatureExtractor.__init__` with debiasing state + `**kwargs` in concrete classes

No behavioral change yet — just wires up the new state so later tasks can use it.

**Files:**
- Modify: `collab_splats/semantics/features.py`
- Create: `tests/semantics/test_positional_debiasing.py`

- [ ] **Step 1: Create test file with fake extractor and smoke test**

Create `tests/semantics/test_positional_debiasing.py`:

```python
"""Tests for positional debiasing in BaseFeatureExtractor.

Uses _FakeExtractor — a lightweight subclass with no model weights — to test
the infrastructure in BaseFeatureExtractor without requiring GPU or HuggingFace downloads.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from collab_splats.semantics.features import BaseFeatureExtractor, _DEBIAS_VALIDATED


# ---------------------------------------------------------------------------
# Fake extractor: minimal concrete implementation for testing base-class logic
# ---------------------------------------------------------------------------

class _FakeExtractor(BaseFeatureExtractor):
    """Returns deterministic features based on pixel mean so image content affects output.

    patch_size=14 matches standard ViT-B stride — required by _build_positional_basis
    so the zero-image dimensions are computed correctly.
    """

    patch_size = 14

    def __init__(self, feature_dim: int = 16, h_p: int = 4, w_p: int = 4, **kwargs):
        super().__init__(**kwargs)  # passes svd_components and future base params through
        self._feature_dim = feature_dim  # D: output patch feature dimensionality
        self._h_p = h_p                  # H_p: fixed patch grid height (regardless of input size)
        self._w_p = w_p                  # W_p: fixed patch grid width

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")  # CPU-only — no GPU needed in tests

    def forward(self, images: list) -> list[torch.Tensor]:
        results = []
        for img in images:
            arr = np.array(img)
            # Seed from pixel mean: zero-pixel image (mean=0) gives different features than
            # content images (mean>0), which is required for debiasing tests to be meaningful.
            seed = int(arr.mean() * 1000) % (2 ** 31)
            gen = torch.Generator()
            gen.manual_seed(seed)
            feat = torch.randn(self._feature_dim, self._h_p, self._w_p, generator=gen)
            results.append(F.normalize(feat, p=2, dim=0))  # unit-norm patches, matches real extractor contract
        return results


class _UnvalidatedExtractor(_FakeExtractor):
    """Subclass of _FakeExtractor NOT listed in _DEBIAS_VALIDATED — used to test warning path."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(h: int = 64, w: int = 64, value: int = 100) -> Image.Image:
    """Create a solid-color PIL image for use as test input."""
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Task 1: smoke test — base class state is initialised
# ---------------------------------------------------------------------------

def test_base_state_initialised():
    """BaseFeatureExtractor.__init__ must set svd_components, _pos_basis_cache, _zero_feats_cache."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=8)
    assert extractor.svd_components == 8
    assert extractor._pos_basis_cache == {}
    assert extractor._zero_feats_cache == {}


def test_forward_unchanged():
    """Subclass forward() must work exactly as before — debias state must not affect it."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4)
    img = _make_rgb_image()
    result = extractor.forward([img])
    assert isinstance(result, list) and len(result) == 1
    assert result[0].shape == (16, 4, 4)
```

- [ ] **Step 2: Run tests — expect ImportError on `_DEBIAS_VALIDATED`**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py -v 2>&1 | tail -20
```

Expected: FAIL — `_DEBIAS_VALIDATED` not importable, `BaseFeatureExtractor.__init__` not defined.

- [ ] **Step 3: Add `import numpy as np` and `_DEBIAS_VALIDATED` to `features.py`**

In `features.py`, insert after `import logging` (line 10):

```python
import logging
import numpy as np  # needed by _build_positional_basis to create the zero-pixel PIL image
import os
```

After `logger = logging.getLogger(__name__)` (currently line 34), add:

```python
# Extractors where positional debiasing has been empirically validated against DINO-family models.
# Add a class name here after verifying that debiasing improves quality for that model family.
# Extractors NOT in this set still work with debias() but receive a warning at call time.
_DEBIAS_VALIDATED: frozenset = frozenset({"DINOFeatureExtractor", "Talk2DinoExtractor"})
```

- [ ] **Step 4: Add `__init__` to `BaseFeatureExtractor`**

In `BaseFeatureExtractor`, insert `__init__` before the `@abstractmethod forward`:

```python
    def __init__(self, svd_components: int = 500, **kwargs) -> None:
        # svd_components: number of top singular vectors kept for the positional subspace.
        # 500 matches the INSID3 default (see reference implementation).
        super().__init__(**kwargs)  # passes remaining kwargs up to nn.Module
        self.svd_components = svd_components  # stored so _build_positional_basis can read it later
        # Caches keyed by (H_p, W_p) so different input resolutions each get their own basis.
        self._pos_basis_cache: dict = {}   # (H_p, W_p) → Tensor(D, K) positional subspace basis
        self._zero_feats_cache: dict = {}  # (H_p, W_p) → Tensor(D, H_p, W_p) zero-image features for viz
```

- [ ] **Step 5: Add `**kwargs` to `MaskCLIPExtractor.__init__`**

Change the `__init__` signature and `super().__init__()` call:

```python
    def __init__(
        self,
        model_name: str = "ViT-L/14@336px",
        resolution: int = 1024,
        cache_dir: str = TORCH_HOME,
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.default_resolution = resolution
        # ... rest of body unchanged
```

- [ ] **Step 6: Add `**kwargs` to `DINOFeatureExtractor.__init__`**

```python
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        resolution: int = 800,
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size: int = self.model.config.patch_size
        self.resolution = resolution
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
```

- [ ] **Step 7: Add `**kwargs` to `Talk2DinoExtractor.__init__`**

```python
    def __init__(
        self,
        model_name: str = "lorebianchi98/Talk2DINOv3-ViTB",
        device: Optional[str] = None,
        **kwargs,  # passes svd_components and any future BaseFeatureExtractor params through
    ):
        if device is None:
            device = get_device()
        super().__init__(**kwargs)  # forwards svd_components to BaseFeatureExtractor.__init__
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        # config.patch_size reflects the ViT token grid, not the pixel stride seen
        # by the caller — encode_image upscales internally (e.g. 224→448) so the
        # effective stride in original-input pixels is smaller. Derive from conv layer.
        try:
            conv = self._model.model.patch_embed.proj
            self.patch_size: int = conv.stride[0]
        except AttributeError:
            self.patch_size = getattr(self._model.config, "patch_size", 14)
        self._device = torch.device(device)
```

- [ ] **Step 8: Run Task 1 tests — expect PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py::test_base_state_initialised tests/semantics/test_positional_debiasing.py::test_forward_unchanged -v 2>&1 | tail -20
```

Expected: both PASS.

- [ ] **Step 9: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/semantics/features.py tests/semantics/test_positional_debiasing.py
git commit -m "feat(semantics): add BaseFeatureExtractor debiasing state + _DEBIAS_VALIDATED"
```

---

### Task 2: SVD debiasing — `_build_positional_basis`, `_apply_debias`, `debias()`, and validation warning

**Files:**
- Modify: `collab_splats/semantics/features.py` (add methods to `BaseFeatureExtractor`)
- Modify: `tests/semantics/test_positional_debiasing.py` (add tests)

- [ ] **Step 1: Add debiasing tests**

Append to `tests/semantics/test_positional_debiasing.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: debias() — SVD projection, shape, caching, warning
# ---------------------------------------------------------------------------

def test_debias_returns_list_of_tensors_same_shape():
    """debias() must return list[Tensor] with same shape as input features."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    features = extractor.forward([img])
    debiased = extractor.debias(features)
    assert isinstance(debiased, list) and len(debiased) == 1
    assert debiased[0].shape == features[0].shape


def test_debias_output_differs_from_input():
    """debias() must produce different features than the raw forward() output."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)  # non-zero pixel value → differs from zero-image basis
    features = extractor.forward([img])
    debiased = extractor.debias(features)
    # Debiasing subtracts a structured positional component — outputs must differ.
    assert not torch.allclose(features[0], debiased[0])


def test_debias_output_is_unit_norm():
    """_apply_debias re-normalizes L2 — each patch vector in output must be unit norm."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    debiased = extractor.debias(extractor.forward([img]))
    norms = debiased[0].norm(dim=0)  # (H_p, W_p) — norm of each patch vector along feature dim
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_positional_basis_cached_after_first_debias_call():
    """_pos_basis_cache must be populated after first debias() call and NOT rebuilt on second."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    assert len(extractor._pos_basis_cache) == 0  # empty before first call

    features = extractor.forward([img])
    extractor.debias(features)
    assert (4, 4) in extractor._pos_basis_cache  # cached after first call

    # Monkey-patch to detect if _build_positional_basis is called a second time.
    calls = []
    original = extractor._build_positional_basis
    extractor._build_positional_basis = lambda *a, **kw: calls.append(1) or original(*a, **kw)

    extractor.debias(features)  # second call at same resolution
    assert len(calls) == 0  # basis must NOT be rebuilt — must reuse from cache


def test_unvalidated_extractor_warns_on_debias(caplog):
    """debias() on an extractor not in _DEBIAS_VALIDATED must log WARNING (not raise)."""
    assert _UnvalidatedExtractor.__name__ not in _DEBIAS_VALIDATED  # confirm test precondition
    extractor = _UnvalidatedExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    features = extractor.forward([img])
    import logging
    with caplog.at_level(logging.WARNING, logger="collab_splats.semantics.features"):
        debiased = extractor.debias(features)  # must not raise
    assert any("not yet validated" in r.message for r in caplog.records)
    assert debiased[0].shape == features[0].shape  # still returns correct output despite warning
```

- [ ] **Step 2: Run tests — expect FAIL (debias not defined)**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py -k "debias" -v 2>&1 | tail -25
```

Expected: FAIL — `AttributeError: 'debias' not found`.

- [ ] **Step 3: Add `_build_positional_basis`, `_apply_debias`, and `debias` to `BaseFeatureExtractor`**

Insert these three methods after `_get_memory_per_image` in `BaseFeatureExtractor`:

```python
    def _build_positional_basis(self, H_p: int, W_p: int) -> None:
        """Estimate the positional subspace from a zero-pixel image using SVD (INSID3 algorithm).

        A black (zero-pixel) image, after each extractor's own normalization, produces features
        driven entirely by the model's positional embeddings — no semantic content to interfere.
        SVD then extracts the top-K directions of variance, representing the positional subspace.

        Calls self.forward() so each subclass handles its own preprocessing identically to
        real inference — no separate code path, no reimplementation of preprocessing.

        Stores results in _pos_basis_cache[(H_p, W_p)] and _zero_feats_cache[(H_p, W_p)].
        Only called once per resolution; all subsequent debias() calls at (H_p, W_p) reuse the cache.

        Args:
            H_p: Number of patch rows in the target feature map.
            W_p: Number of patch columns in the target feature map.
        """
        # Require patch_size to be set explicitly — no silent fallback.
        # If missing, the AttributeError here is far better than producing a silently wrong zero image.
        if not hasattr(self, "patch_size"):
            raise AttributeError(
                f"{type(self).__name__} must define `self.patch_size` "
                "(the pixel stride of each patch token) before calling debias()."
            )
        patch_size = self.patch_size  # pixel stride per patch; set in each concrete subclass __init__
        H_img = H_p * patch_size  # image height that produces H_p patch rows under stride-exact preprocessing
        W_img = W_p * patch_size  # image width  that produces W_p patch cols under stride-exact preprocessing

        # Zero-pixel (black) image — matches INSID3's torch.zeros approach, expressed as a PIL Image
        # so it passes naturally through each subclass's own forward() without special-casing.
        # After normalization inside forward(), zero pixels become a fixed non-semantic input
        # that elicits the model's positional response with no image-content signal.
        zero_arr = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        zero_pil = Image.fromarray(zero_arr)  # PIL Image so forward() preprocessing runs unchanged

        with torch.no_grad():
            [zero_feat] = self.forward([zero_pil])  # (D, H_p', W_p') — subclass forward handles preprocessing

        if zero_feat.shape[1:] != (H_p, W_p):
            # The subclass's internal preprocessing (e.g. Talk2DINO upscaling) changed the grid size.
            # Interpolate to the target (H_p, W_p) — positional bias is spatially smooth so this is valid.
            zero_feat = F.interpolate(
                zero_feat.unsqueeze(0),  # (1, D, H_p', W_p') — F.interpolate requires a leading batch dim
                size=(H_p, W_p),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # remove batch dim → (D, H_p, W_p)

        self._zero_feats_cache[(H_p, W_p)] = zero_feat  # saved for get_bias_visualization

        D = zero_feat.shape[0]  # feature dimensionality — needed to reshape E for SVD
        E = zero_feat.reshape(D, -1)               # (D, H_p*W_p) — one column per patch, one row per feature
        E = E - E.mean(dim=1, keepdim=True)        # center: remove mean activation per channel before SVD

        # SVD: columns of U are principal directions of feature variance across spatial patch positions.
        # Because the image has no content, these directions capture positional variation only.
        U, _, _ = torch.linalg.svd(E, full_matrices=False)  # U: (D, min(D, H_p*W_p))

        # Keep top-K directions — they represent the positional subspace; the tail captures noise.
        self._pos_basis_cache[(H_p, W_p)] = U[:, : self.svd_components].contiguous()  # (D, K)

    def _apply_debias(self, fmap: torch.Tensor) -> torch.Tensor:
        """Project fmap onto the orthogonal complement of the positional subspace.

        Removes the positional component from each patch feature vector, then re-normalizes
        L2 so downstream cosine-similarity comparisons remain well-defined.

        Args:
            fmap: (D, H_p, W_p) patch feature map — same spatial resolution as a cached basis.

        Returns:
            (D, H_p, W_p) debiased feature map with unit-norm patch vectors.
        """
        D, H_p, W_p = fmap.shape
        basis = self._pos_basis_cache[(H_p, W_p)].to(fmap.device)  # (D, K) — move to same device as features

        # P_perp = I - U @ U.T projects out the component in span(U) (the positional subspace).
        # Applying P_perp to a feature vector zeroes its positional component, keeping only semantics.
        P_perp = torch.eye(D, device=fmap.device, dtype=fmap.dtype) - basis @ basis.T  # (D, D)

        X = fmap.reshape(D, -1)    # (D, H_p*W_p) — flatten spatial dims for matrix multiply
        X_deb = P_perp @ X         # (D, H_p*W_p) — positional component removed from each patch vector

        # Re-normalize: after projection the vectors are no longer unit-norm.
        # Cosine-similarity comparisons in downstream code require unit-norm patch vectors.
        X_deb = F.normalize(X_deb, p=2, dim=0)  # normalize across feature dim (dim=0 for (D, N) layout)

        return X_deb.reshape(D, H_p, W_p)  # restore spatial layout

    def debias(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Remove positional bias from extracted patch features using SVD projection (INSID3 algorithm).

        Operates on features already returned by forward(). Builds and caches the positional
        basis from a zero-pixel image (using this extractor's own forward()) on first call at
        each patch-grid resolution, then reuses the cached basis for all subsequent calls.

        Args:
            features: list of (D, H_p, W_p) tensors — output of forward(). All tensors must
                      have the same spatial resolution (H_p, W_p).

        Returns:
            list of (D, H_p, W_p) tensors with positional bias removed and L2 re-normalized.
        """
        if type(self).__name__ not in _DEBIAS_VALIDATED:
            # Algorithm is general but has only been verified for DINO-family models.
            # Warn rather than raise so researchers can experiment with other extractors.
            logger.warning(
                "[%s] Positional debiasing not yet validated for this extractor. "
                "Proceeding — results may be suboptimal.",
                type(self).__name__,
            )
        _, H_p, W_p = features[0].shape  # read patch grid dimensions from the first feature map
        if (H_p, W_p) not in self._pos_basis_cache:
            # First debias() call at this resolution — build and cache the positional basis.
            self._build_positional_basis(H_p, W_p)
        return [self._apply_debias(f) for f in features]  # project each image's feature map
```

- [ ] **Step 4: Run all Task 2 tests — expect PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/semantics/features.py tests/semantics/test_positional_debiasing.py
git commit -m "feat(semantics): add debias() with SVD positional basis to BaseFeatureExtractor"
```

---

### Task 3: Bias visualization — `get_bias_visualization`

**Files:**
- Modify: `collab_splats/semantics/features.py` (add method to `BaseFeatureExtractor`)
- Modify: `tests/semantics/test_positional_debiasing.py` (add tests)

- [ ] **Step 1: Add visualization tests**

Append to `tests/semantics/test_positional_debiasing.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: get_bias_visualization
# ---------------------------------------------------------------------------

def test_get_bias_visualization_correct_shape_and_dtype():
    """get_bias_visualization must return (H_p, W_p, 3) uint8 after a debias() call."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    extractor.debias(extractor.forward([img]))  # populates _zero_feats_cache
    viz = extractor.get_bias_visualization(4, 4)
    assert isinstance(viz, np.ndarray)
    assert viz.shape == (4, 4, 3)
    assert viz.dtype == np.uint8


def test_get_bias_visualization_raises_without_prior_debias_call():
    """get_bias_visualization must raise KeyError if debias() has not been called first."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    with pytest.raises(KeyError, match="No positional bias cached"):
        extractor.get_bias_visualization(4, 4)


def test_get_bias_visualization_values_in_range():
    """get_bias_visualization output values must lie in [0, 255]."""
    extractor = _FakeExtractor(feature_dim=16, h_p=4, w_p=4, svd_components=4)
    img = _make_rgb_image(value=128)
    extractor.debias(extractor.forward([img]))
    viz = extractor.get_bias_visualization(4, 4)
    assert int(viz.min()) >= 0
    assert int(viz.max()) <= 255
```

- [ ] **Step 2: Run visualization tests — expect FAIL**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py -k "visualization" -v 2>&1 | tail -20
```

Expected: FAIL — `AttributeError: 'get_bias_visualization' not found`.

- [ ] **Step 3: Add `get_bias_visualization` to `BaseFeatureExtractor`**

Insert after `debias()`:

```python
    def get_bias_visualization(self, H_p: int, W_p: int) -> "np.ndarray":
        """Return an RGB heatmap of the positional bias at a given patch grid resolution.

        Uses PCA (via SVD, no sklearn required) to project the zero-image features onto the
        top 3 principal components, producing an image where color encodes the dominant axes
        of positional variation. Useful for verifying that debiasing captures spatial structure.

        Requires a prior debias() call at this (H_p, W_p) resolution to populate the cache.

        Args:
            H_p: Patch grid height — must match a resolution used in a prior debias() call.
            W_p: Patch grid width — must match a resolution used in a prior debias() call.

        Returns:
            np.ndarray of shape (H_p, W_p, 3) dtype uint8 — PCA-derived RGB visualization.

        Raises:
            KeyError: if (H_p, W_p) has not been cached — call debias() at this resolution first.
        """
        if (H_p, W_p) not in self._zero_feats_cache:
            raise KeyError(
                f"No positional bias cached for patch grid ({H_p}, {W_p}). "
                "Call debias() at this resolution first."
            )
        zero_feats = self._zero_feats_cache[(H_p, W_p)]  # (D, H_p, W_p) — zero-image patch features
        D = zero_feats.shape[0]  # feature dimensionality — needed to reshape before SVD

        # PCA via SVD on the (N, D) matrix where N = H_p*W_p patch locations.
        # Each row is one patch's feature vector; SVD gives the principal axes of spatial variation.
        E = zero_feats.reshape(D, -1).T  # (N, D) — transpose so rows are per-patch observations
        E = E - E.mean(dim=0, keepdim=True)  # center: subtract mean patch feature across all positions

        # SVD: right singular vectors Vt span the principal directions in feature space.
        # We only need Vt[:3] (top-3 right singular vectors) to project onto the 3 dominant PCs.
        _, _, Vt = torch.linalg.svd(E, full_matrices=False)  # Vt: (min(N,D), D)

        rgb = (E @ Vt[:3].T).cpu().numpy()  # (N, 3) — project each patch onto the top-3 PCs

        # Normalize to [0, 1] for display: shift minimum to zero, then scale by range.
        # +1e-8 prevents divide-by-zero when all features are identical (e.g. all-black image edge case).
        rgb -= rgb.min()
        rgb /= rgb.max() + 1e-8

        return (rgb.reshape(H_p, W_p, 3) * 255).astype(np.uint8)  # scale to uint8 RGB for display
```

- [ ] **Step 4: Run all tests — expect all PASS**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/semantics/test_positional_debiasing.py -v 2>&1 | tail -30
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/semantics/features.py tests/semantics/test_positional_debiasing.py
git commit -m "feat(semantics): add get_bias_visualization — PCA heatmap of positional basis"
```

---

## Self-Review

**Spec coverage:**
- ✅ `debias(features)` public method — Task 2
- ✅ `get_bias_visualization(H_p, W_p)` — Task 3
- ✅ `_build_positional_basis` (zero PIL → subclass forward → SVD → cache) — Task 2
- ✅ `_apply_debias` (P_perp projection, L2 renorm) — Task 2
- ✅ `svd_components`, `_pos_basis_cache`, `_zero_feats_cache` in `BaseFeatureExtractor.__init__` — Task 1
- ✅ `_DEBIAS_VALIDATED` + warning in `debias()` — Task 2
- ✅ `**kwargs` in all three concrete `__init__` methods — Task 1
- ✅ `import numpy as np` at file top — Task 1
- ✅ No `forward` renamed or modified anywhere — confirmed throughout
- ✅ `_build_positional_basis` calls `self.forward([zero_pil])` — subclass preprocessing unchanged

**Placeholder scan:** None. All code blocks are complete.

**Type consistency:**
- `debias(features: list[Tensor]) -> list[Tensor]` — defined Task 2, consistent in all tests
- `_build_positional_basis(H_p: int, W_p: int) -> None` — called from `debias()`, defined Task 2
- `_apply_debias(fmap: Tensor) -> Tensor` — called from `debias()`, defined Task 2
- `get_bias_visualization(H_p: int, W_p: int) -> np.ndarray` — defined Task 3
- Cache key `(H_p, W_p)` — consistent across `_pos_basis_cache` and `_zero_feats_cache` ✅
- `_FakeExtractor` uses `patch_size = 14` (class attr), `h_p=4`, `w_p=4`, `svd_components=4` — `min(D=16, H_p*W_p=16) = 16 ≥ 4` so SVD truncation is valid ✅
- `_UnvalidatedExtractor.__name__` = `"_UnvalidatedExtractor"` — confirmed absent from `_DEBIAS_VALIDATED` ✅
