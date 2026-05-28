# INSID3 In-Context Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `INSID3Segmentation` — a training-free in-context segmentation backend that segments a target image conditioned on a reference image + binary mask, using frozen DINOv2 features and SVD positional debiasing.

**Architecture:** `INSID3Segmentation(BaseSegmentation)` registered as `"insid3"`. Stateful: `set_context(ref_image, ref_mask)` extracts and caches reference features once; `segment(image)` reuses the cache across N target frames. Module-level private functions handle clustering and candidate localization.

**Tech Stack:** PyTorch, `sklearn.cluster.AgglomerativeClustering`, `DINOFeatureExtractor` (existing), `BaseFeatureExtractor.debias()` (existing), PIL, numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-05-28-insid3-design.md`

**Python env:** always use `/opt/conda/envs/reconstruction/bin/python` (py3.11). Tests: `/opt/conda/envs/reconstruction/bin/python -m pytest`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `collab_splats/semantics/segmentation/insid3.py` | Create | All INSID3 logic: clustering utils, candidate localization, seed+aggregate, `INSID3Segmentation` class |
| `collab_splats/semantics/segmentation/__init__.py` | Modify | Add `INSID3Segmentation` export |
| `collab_splats/utils/visualization.py` | Modify | Add `plot_context_segmentation` |
| `tests/semantics/test_insid3_segmentation.py` | Create | All tests (flat functions, mocked extractor) |

---

## Task 1: Scaffold `insid3.py` + clustering utilities

**Files:**
- Create: `collab_splats/semantics/segmentation/insid3.py`
- Create: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Write failing tests for clustering utilities**

```python
# tests/semantics/test_insid3_segmentation.py
"""Tests for INSID3 in-context segmentation backend."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def test_agglomerative_clustering_returns_integer_labels():
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(16, 32), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    assert labels.shape == (16,)
    assert labels.dtype == torch.long
    assert labels.min() >= 0


def test_agglomerative_clustering_device_preserved():
    from collab_splats.semantics.segmentation.insid3 import _agglomerative_clustering
    X = F.normalize(torch.randn(8, 16), p=2, dim=1)
    labels = _agglomerative_clustering(X, tau=0.6)
    assert labels.device == X.device


def test_cluster_prototypes_shape_and_normalized():
    from collab_splats.semantics.segmentation.insid3 import _cluster_prototypes
    X = F.normalize(torch.randn(20, 32), p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2, 0, 1])
    K = 3
    protos = _cluster_prototypes(X, labels, K)
    assert protos.shape == (K, 32)
    norms = protos.norm(dim=1)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5)
```

- [ ] **Step 2: Run tests — verify they fail with ImportError**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_agglomerative_clustering'`

- [ ] **Step 3: Create `insid3.py` with scaffold and clustering utilities**

```python
# collab_splats/semantics/segmentation/insid3.py
"""INSID3 in-context segmentation backend.

Provides:
  INSID3Segmentation — training-free in-context segmentation via frozen DINOv2 features
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseSegmentation

logger = logging.getLogger(__name__)


########################################################
########## Clustering utilities ########################
########################################################


def _agglomerative_clustering(X: torch.Tensor, tau: float) -> torch.Tensor:
    """Partition N patches into clusters via cosine-distance agglomerative clustering.

    Args:
        X: (N, D) L2-normalized patch features.
        tau: similarity threshold — clusters are merged above this cosine similarity.

    Returns:
        (N,) long tensor of integer cluster labels on the same device as X.
    """
    from sklearn.cluster import AgglomerativeClustering
    S = (X @ X.T).clamp(-1, 1)
    D = (1.0 - S).cpu().numpy()
    ac = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(1.0 - tau),
    )
    labels = ac.fit_predict(D)
    return torch.from_numpy(labels).long().to(X.device)


def _cluster_prototypes(X: torch.Tensor, labels: torch.Tensor, K: int) -> torch.Tensor:
    """Compute an L2-normalized prototype (mean) for each cluster.

    Args:
        X: (N, D) patch features.
        labels: (N,) integer cluster assignments.
        K: number of clusters.

    Returns:
        (K, D) L2-normalized prototypes.
    """
    protos = []
    for k in range(K):
        idx = labels == k
        mu = X[idx].mean(dim=0) if idx.any() else torch.zeros(X.shape[1], device=X.device)
        protos.append(F.normalize(mu, p=2, dim=0).unsqueeze(0))
    return torch.cat(protos, dim=0)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py::test_agglomerative_clustering_returns_integer_labels tests/semantics/test_insid3_segmentation.py::test_agglomerative_clustering_device_preserved tests/semantics/test_insid3_segmentation.py::test_cluster_prototypes_shape_and_normalized -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/segmentation/insid3.py tests/semantics/test_insid3_segmentation.py
git commit -m "feat(segmentation): scaffold insid3.py + clustering utilities"
```

---

## Task 2: Mask downsampling + candidate localization

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`
- Modify: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/semantics/test_insid3_segmentation.py`:

```python
def test_downsample_mask_reduces_to_patch_resolution():
    from collab_splats.semantics.segmentation.insid3 import _downsample_mask
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[20:44, 20:44] = True
    down = _downsample_mask(mask, h=8, w=8)
    assert down.shape == (8, 8)
    assert down.any(), "mask should be non-empty after downsampling"


def test_downsample_mask_tiny_mask_fallback():
    from collab_splats.semantics.segmentation.insid3 import _downsample_mask
    # Single pixel mask — bilinear would vanish at patch resolution
    mask = torch.zeros(64, 64, dtype=torch.bool)
    mask[32, 32] = True
    down = _downsample_mask(mask, h=8, w=8)
    assert down.shape == (8, 8)
    assert down.sum() == 1, "fallback should produce exactly one True patch"


def test_locate_candidates_returns_bool_mask():
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    D, H, W = 16, 6, 6
    tgt = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref_mask_down = torch.zeros(H, W, dtype=torch.bool)
    ref_mask_down[1:4, 1:4] = True
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    out = _locate_candidates(tgt, ref, ref_mask_down, proto)
    assert out.shape == (H, W)
    assert out.dtype == torch.bool


def test_locate_candidates_perfect_match():
    from collab_splats.semantics.segmentation.insid3 import _locate_candidates
    # When target IS the reference, backward mask = forward mask = ref_mask_down (approx)
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    ref_mask_down = torch.zeros(H, W, dtype=torch.bool)
    ref_mask_down[0:3, 0:3] = True
    # prototype = mean of masked ref features
    proto = F.normalize(feat[:, ref_mask_down].mean(dim=1), p=2, dim=0)
    out = _locate_candidates(feat, feat, ref_mask_down, proto)
    assert out.shape == (H, W)
    # At least some candidates should be inside the masked region
    assert (out & ref_mask_down).any()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "downsample or locate" -v 2>&1 | tail -10
```

Expected: 4 FAILED with ImportError

- [ ] **Step 3: Add `_downsample_mask`, `_upsample_mask`, `_tensor_to_pil`, `_locate_candidates` to `insid3.py`**

Insert after the clustering utilities section (before the class):

```python
########################################################
########## Mask and image helpers ######################
########################################################


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample (H, W) bool mask to (h, w) with fallback for tiny masks.

    Tries bilinear → nearest → single center pixel to ensure non-empty output.
    """
    m = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    down = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
    if down.sum() == 0:
        down = F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5
    if down.sum() == 0:
        center = torch.argwhere(mask).float().mean(dim=0)
        scale = torch.tensor([h / mask.shape[0], w / mask.shape[1]], device=mask.device)
        cy, cx = (center * scale).long()
        cy = cy.clamp(0, h - 1)
        cx = cx.clamp(0, w - 1)
        down = torch.zeros(h, w, dtype=torch.bool, device=mask.device)
        down[cy, cx] = True
    return down


def _upsample_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Bilinear upsample (h, w) bool mask to (H, W)."""
    return F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0] > 0.5


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert (C, H, W) float tensor in [0, 1] to PIL Image."""
    arr = (t.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


########################################################
########## Candidate localization ######################
########################################################


def _locate_candidates(
    tgt_feat_deb: torch.Tensor,   # (D, Ht, Wt) debiased target features
    ref_feat_deb: torch.Tensor,   # (D, Hr, Wr) debiased ref features
    ref_mask_down: torch.Tensor,  # (Hr, Wr) bool — ref mask at patch resolution
    prototype: torch.Tensor,      # (D,) L2-normalized ref prototype
) -> torch.Tensor:                # (Ht, Wt) bool candidate mask
    """Forward+backward candidate localization.

    Forward: target patches with positive cosine sim to prototype.
    Backward: for each target patch, its nearest ref patch must be inside ref_mask_down.
    Returns intersection of both masks.
    """
    D, Ht, Wt = tgt_feat_deb.shape
    _, Hr, Wr = ref_feat_deb.shape

    # Forward: cosine similarity to prototype
    sim_fwd = torch.einsum("dhw,d->hw", tgt_feat_deb, prototype)  # (Ht, Wt)
    forward_mask = sim_fwd > 0
    if forward_mask.sum() == 0:
        thresh = float(torch.quantile(sim_fwd.float(), 0.9))
        forward_mask = sim_fwd > thresh

    # Backward: NN in ref; check if inside ref mask
    tgt_flat = tgt_feat_deb.reshape(D, -1).T          # (Ht*Wt, D)
    ref_flat = ref_feat_deb.reshape(D, -1).T           # (Hr*Wr, D)
    sim_t_to_r = tgt_flat @ ref_flat.T                 # (Ht*Wt, Hr*Wr)
    best_idx = sim_t_to_r.argmax(dim=1)                # (Ht*Wt,)
    rows = (best_idx // Wr).clamp(0, Hr - 1)
    cols = (best_idx % Wr).clamp(0, Wr - 1)
    backward_mask = ref_mask_down[rows, cols].reshape(Ht, Wt)

    return forward_mask & backward_mask
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "downsample or locate" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/segmentation/insid3.py tests/semantics/test_insid3_segmentation.py
git commit -m "feat(segmentation): add mask downsampling and candidate localization"
```

---

## Task 3: Seed and aggregate

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`
- Modify: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/semantics/test_insid3_segmentation.py`:

```python
def test_seed_and_aggregate_returns_bool_mask():
    from collab_splats.semantics.segmentation.insid3 import _seed_and_aggregate, _cluster_prototypes
    D, H, W = 16, 6, 6
    N = H * W
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    feat_deb = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    candidate_mask = torch.zeros(H, W, dtype=torch.bool)
    candidate_mask[1:4, 1:4] = True
    # Create simple cluster labels: checkerboard of 2 clusters
    labels = torch.zeros(H, W, dtype=torch.long)
    labels[::2, ::2] = 1
    K = 2
    out = _seed_and_aggregate(candidate_mask, feat, feat_deb, proto, labels, K, merge_threshold=0.2)
    assert out.shape == (H, W)
    assert out.dtype == torch.bool


def test_seed_and_aggregate_empty_candidate_returns_candidate():
    from collab_splats.semantics.segmentation.insid3 import _seed_and_aggregate
    D, H, W = 16, 6, 6
    feat = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    feat_deb = F.normalize(torch.randn(D, H, W), p=2, dim=0)
    proto = F.normalize(torch.randn(D), p=2, dim=0)
    candidate_mask = torch.zeros(H, W, dtype=torch.bool)  # all-zero
    labels = torch.zeros(H, W, dtype=torch.long)
    out = _seed_and_aggregate(candidate_mask, feat, feat_deb, proto, labels, K=1, merge_threshold=0.2)
    assert out.shape == (H, W)
    assert not out.any()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "seed_and_aggregate" -v 2>&1 | tail -10
```

Expected: 2 FAILED with ImportError

- [ ] **Step 3: Add `_seed_and_aggregate` to `insid3.py`**

Insert after `_locate_candidates` (before the class placeholder):

```python
########################################################
########## Seed selection and cluster aggregation ######
########################################################


def _seed_and_aggregate(
    candidate_mask: torch.Tensor,   # (H_p, W_p) bool
    tgt_feat: torch.Tensor,         # (D, H_p, W_p) raw (non-debiased) target features
    tgt_feat_deb: torch.Tensor,     # (D, H_p, W_p) debiased target features
    prototype: torch.Tensor,        # (D,) debiased ref prototype
    cluster_labels: torch.Tensor,   # (H_p, W_p) long
    K: int,
    merge_threshold: float,
) -> torch.Tensor:                  # (H_p, W_p) bool
    """Select seed cluster and aggregate remaining clusters by combined similarity score.

    Returns empty mask if no clusters overlap the candidate region.
    """
    D, H, W = tgt_feat.shape
    matched_mask = candidate_mask & (cluster_labels >= 0)
    if matched_mask.sum() == 0:
        return candidate_mask

    matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

    # Area weight: fraction of candidate pixels each cluster contributes
    all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
    area_weights = torch.zeros(K, device=cluster_labels.device)
    area_weights[matched_ids] = n_pixels.float() / all_areas.float()[matched_ids]

    # Cluster prototypes in debiased space for cross-image similarity
    tgt_deb_flat = tgt_feat_deb.reshape(D, -1).T   # (N, D)
    deb_protos = _cluster_prototypes(tgt_deb_flat, cluster_labels.view(-1), K)  # (K, D)

    # Seed: matched cluster with highest cross-image similarity to ref prototype
    cross_sim_matched = deb_protos[matched_ids] @ prototype  # (n_matched,)
    seed_idx = int(torch.argmax(cross_sim_matched).item())
    seed_id = int(matched_ids[seed_idx].item())

    # Raw cluster prototypes for intra-image similarity to seed
    tgt_flat = tgt_feat.reshape(D, -1).T           # (N, D)
    raw_protos = _cluster_prototypes(tgt_flat, cluster_labels.view(-1), K)  # (K, D)
    intra_sim = raw_protos @ raw_protos[seed_id]   # (K,)

    # Per-patch cross-image similarity map
    fg_sim = torch.einsum("dhw,d->hw", tgt_feat_deb, prototype)  # (H_p, W_p)
    cross_sim = torch.zeros(K, device=fg_sim.device)
    for k in range(K):
        idx = cluster_labels == k
        cross_sim[k] = fg_sim[idx].mean() if idx.any() else 0.0

    # Combined score; seed cluster always included (area_weight = 1.0)
    combined = cross_sim * intra_sim
    area_weights[seed_id] = 1.0
    combined = combined * area_weights

    final_mask = torch.zeros(H, W, dtype=torch.bool, device=cluster_labels.device)
    valid = cluster_labels >= 0
    final_mask[valid] = combined[cluster_labels[valid]] > merge_threshold
    return final_mask
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "seed_and_aggregate" -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/segmentation/insid3.py tests/semantics/test_insid3_segmentation.py
git commit -m "feat(segmentation): add seed selection and cluster aggregation"
```

---

## Task 4: `INSID3Segmentation` — `__init__`, `set_context`, `clear_context`

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`
- Modify: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/semantics/test_insid3_segmentation.py`:

```python
from unittest.mock import MagicMock, patch


def _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6):
    """Helper: INSID3Segmentation with mocked DINOFeatureExtractor."""
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    feat = F.normalize(torch.randn(D, H_p, W_p), p=2, dim=0)
    extractor = MagicMock()
    extractor.forward.return_value = [feat.clone()]
    extractor.debias.return_value = [feat.clone()]
    seg = INSID3Segmentation.__new__(INSID3Segmentation)
    seg._extractor = extractor
    seg._tau = 0.6
    seg._merge_threshold = 0.2
    seg._prototype = None
    seg._ref_feat_deb = None
    seg._ref_mask_down = None
    return seg, extractor, feat


def test_set_context_caches_prototype_and_features():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    assert seg._prototype is not None
    assert seg._ref_feat_deb is not None
    assert seg._ref_mask_down is not None
    assert seg._prototype.shape == (16,)  # D=16
    assert seg._prototype.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_clear_context_resets_state():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = torch.zeros(48, 48, dtype=torch.bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    seg.clear_context()
    assert seg._prototype is None
    assert seg._ref_feat_deb is None
    assert seg._ref_mask_down is None


def test_set_context_accepts_numpy_mask():
    seg, extractor, feat = _make_seg_with_mock_extractor()
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = np.zeros((48, 48), dtype=bool)
    ref_mask[16:32, 16:32] = True
    seg.set_context(ref_image, ref_mask)
    assert seg._prototype is not None


def test_registered_as_insid3():
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    from collab_splats.semantics.segmentation.base import BaseSegmentation
    assert BaseSegmentation.get("insid3") is INSID3Segmentation
```

Add `from PIL import Image` to the top of the test file (alongside existing imports).

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "context or registered" -v 2>&1 | tail -15
```

Expected: all FAILED with ImportError/AttributeError

- [ ] **Step 3: Add `INSID3Segmentation` class with `__init__`, `set_context`, `clear_context`**

Append to `collab_splats/semantics/segmentation/insid3.py`:

```python
########################################################
########## INSID3Segmentation ##########################
########################################################


@BaseSegmentation.register("insid3")
class INSID3Segmentation(BaseSegmentation):
    """Training-free in-context segmentation using frozen DINOv2 features.

    Call set_context(ref_image, ref_mask) once per semantic category, then segment()
    for each target frame. Context is cached — ref features are extracted only once.

    Args:
        svd_components: SVD rank for positional debiasing (default 500, matches INSID3).
        tau: Agglomerative clustering similarity threshold (default 0.6).
        merge_threshold: Minimum combined score to include a cluster (default 0.2).
        device: Torch device string.
    """

    def __init__(
        self,
        svd_components: int = 500,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        device: str = "cuda",
    ) -> None:
        from collab_splats.semantics.features.dino import DINOFeatureExtractor
        self._extractor = DINOFeatureExtractor(svd_components=svd_components, device=device)
        self._tau = tau
        self._merge_threshold = merge_threshold
        self._prototype: torch.Tensor | None = None
        self._ref_feat_deb: torch.Tensor | None = None
        self._ref_mask_down: torch.Tensor | None = None

    def set_context(
        self,
        ref_image: Image.Image | torch.Tensor,
        ref_mask: np.ndarray | torch.Tensor,
    ) -> None:
        """Extract and cache ref features + prototype. Must call before segment().

        Args:
            ref_image: Reference image as PIL Image or (C, H, W) float tensor in [0, 1].
            ref_mask: Binary context mask as numpy bool array or torch bool tensor, (H, W).
        """
        if isinstance(ref_image, torch.Tensor):
            ref_image = _tensor_to_pil(ref_image)

        [feat] = self._extractor.forward([ref_image])  # (D, H_p, W_p)
        feat_norm = F.normalize(feat, p=2, dim=0)
        [feat_deb] = self._extractor.debias([feat_norm])  # (D, H_p, W_p)

        H_p, W_p = feat_deb.shape[1:]
        device = feat_deb.device

        if isinstance(ref_mask, np.ndarray):
            ref_mask = torch.from_numpy(ref_mask)
        ref_mask = ref_mask.bool().to(device)

        mask_down = _downsample_mask(ref_mask, H_p, W_p)

        # Prototype: L2-normalized mean of debiased ref features inside the mask
        fg = feat_deb[:, mask_down]              # (D, N_fg)
        prototype = F.normalize(fg.mean(dim=1), p=2, dim=0)  # (D,)

        self._prototype = prototype
        self._ref_feat_deb = feat_deb
        self._ref_mask_down = mask_down

    def clear_context(self) -> None:
        """Reset cached context state."""
        self._prototype = None
        self._ref_feat_deb = None
        self._ref_mask_down = None
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "context or registered" -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/segmentation/insid3.py tests/semantics/test_insid3_segmentation.py
git commit -m "feat(segmentation): INSID3Segmentation class with set_context and clear_context"
```

---

## Task 5: `segment()` + `segment_with_mask()`

**Files:**
- Modify: `collab_splats/semantics/segmentation/insid3.py`
- Modify: `tests/semantics/test_insid3_segmentation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/semantics/test_insid3_segmentation.py`:

```python
def test_segment_raises_without_context():
    from collab_splats.semantics.segmentation.insid3 import INSID3Segmentation
    seg = INSID3Segmentation.__new__(INSID3Segmentation)
    seg._prototype = None
    seg._ref_feat_deb = None
    seg._ref_mask_down = None
    tgt = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="set_context"):
        seg.segment(tgt)


def test_segment_output_shape_matches_input():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    H, W = 48, 48
    # Manually set context state so extractor mock controls feature shape
    seg._prototype = F.normalize(torch.randn(16), p=2, dim=0)
    seg._ref_feat_deb = feat.clone()
    seg._ref_mask_down = torch.zeros(6, 6, dtype=torch.bool)
    seg._ref_mask_down[1:4, 1:4] = True
    tgt = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    pred_mask, meta = seg.segment(tgt)
    assert pred_mask.shape == (H, W)
    assert pred_mask.dtype == torch.bool
    assert "candidate_mask" in meta
    assert "cluster_labels" in meta
    assert "n_clusters" in meta


def test_segment_context_reused_across_calls():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    seg._prototype = F.normalize(torch.randn(16), p=2, dim=0)
    seg._ref_feat_deb = feat.clone()
    seg._ref_mask_down = torch.zeros(6, 6, dtype=torch.bool)
    seg._ref_mask_down[1:4, 1:4] = True
    tgt = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    seg.segment(tgt)
    seg.segment(tgt)
    seg.segment(tgt)
    # forward() called 3 times (one per target), NOT for ref (already cached)
    assert extractor.forward.call_count == 3


def test_segment_with_mask_one_shot_clears_context():
    seg, extractor, feat = _make_seg_with_mock_extractor(D=16, H_p=6, W_p=6)
    ref_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    tgt_image = Image.fromarray(np.zeros((48, 48, 3), dtype=np.uint8))
    ref_mask = np.zeros((48, 48), dtype=bool)
    ref_mask[16:32, 16:32] = True
    pred_mask, meta = seg.segment_with_mask(tgt_image, ref_image, ref_mask)
    assert pred_mask.shape == (48, 48)
    assert pred_mask.dtype == torch.bool
    # Context cleared after one-shot
    assert seg._prototype is None
    assert seg._ref_feat_deb is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "segment" -v 2>&1 | tail -15
```

Expected: 4 FAILED

- [ ] **Step 3: Add `segment()` and `segment_with_mask()` to `INSID3Segmentation`**

Add these methods inside the `INSID3Segmentation` class, after `clear_context`:

```python
    def segment(self, image: Image.Image | torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Segment image using cached context.

        Raises RuntimeError if no context is set — call set_context() first.
        Returns (pred_mask (H, W) bool, metadata dict).
        Metadata keys: candidate_mask (H_p, W_p), cluster_labels (H_p, W_p), n_clusters (int).
        """
        if self._prototype is None or self._ref_feat_deb is None or self._ref_mask_down is None:
            raise RuntimeError(
                "No context set. Call set_context(ref_image, ref_mask) before segment()."
            )

        if isinstance(image, torch.Tensor):
            orig_H, orig_W = int(image.shape[-2]), int(image.shape[-1])
            image = _tensor_to_pil(image)
        else:
            orig_H, orig_W = image.height, image.width

        [feat] = self._extractor.forward([image])   # (D, H_p, W_p)
        D, H_p, W_p = feat.shape
        feat_norm = F.normalize(feat, p=2, dim=0)
        [feat_deb] = self._extractor.debias([feat_norm])   # (D, H_p, W_p)

        # Candidate localization
        candidate_mask = _locate_candidates(
            feat_deb, self._ref_feat_deb, self._ref_mask_down, self._prototype
        )  # (H_p, W_p) bool

        # Early exit: no candidates found
        if candidate_mask.sum() == 0:
            empty = torch.zeros(H_p, W_p, dtype=torch.bool, device=feat.device)
            return _upsample_mask(empty, orig_H, orig_W), {
                "candidate_mask": candidate_mask,
                "cluster_labels": torch.full((H_p, W_p), -1, dtype=torch.long),
                "n_clusters": 0,
            }

        # Agglomerative clustering on raw (non-debiased) L2-normalized features
        feat_flat = feat_norm.reshape(D, -1).T   # (H_p*W_p, D)
        cluster_labels = _agglomerative_clustering(feat_flat, self._tau)  # (H_p*W_p,)
        K = int(cluster_labels.max().item()) + 1
        cluster_labels_2d = cluster_labels.reshape(H_p, W_p)

        # Seed cluster selection + cluster aggregation
        pred_mask = _seed_and_aggregate(
            candidate_mask, feat_norm, feat_deb, self._prototype,
            cluster_labels_2d, K, self._merge_threshold,
        )  # (H_p, W_p) bool

        pred_mask_up = _upsample_mask(pred_mask, orig_H, orig_W)
        return pred_mask_up, {
            "candidate_mask": candidate_mask,
            "cluster_labels": cluster_labels_2d,
            "n_clusters": K,
        }

    def segment_with_mask(
        self,
        image: Image.Image | torch.Tensor,
        ref_image: Image.Image | torch.Tensor,
        ref_mask: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """One-shot in-context segmentation: set_context → segment → clear_context.

        Args:
            image: Target image to segment.
            ref_image: Reference image containing the context category.
            ref_mask: Binary mask on ref_image indicating the context region.

        Returns:
            (pred_mask (H, W) bool, metadata dict) — same as segment().
        """
        self.set_context(ref_image, ref_mask)
        result = self.segment(image)
        self.clear_context()
        return result
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -k "segment" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run the full test file**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/semantics/test_insid3_segmentation.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add collab_splats/semantics/segmentation/insid3.py tests/semantics/test_insid3_segmentation.py
git commit -m "feat(segmentation): segment() and segment_with_mask() on INSID3Segmentation"
```

---

## Task 6: Export in `segmentation/__init__.py`

**Files:**
- Modify: `collab_splats/semantics/segmentation/__init__.py`

- [ ] **Step 1: Add the import and `__all__` entry**

In `collab_splats/semantics/segmentation/__init__.py`, add after the `SAM3Segmentation` import line:

```python
from .insid3 import INSID3Segmentation
```

And add `"INSID3Segmentation"` to `__all__`.

The file should look like:

```python
"""Segmentation backends and mask utilities.

Import from here — submodule structure is an implementation detail.
"""
from __future__ import annotations

# ── Abstract base and mask utilities ──────────────────────────────────────────
from .base import (
    BaseSegmentation,
    create_patch_mask,
    create_composite_mask,
    mask_id_to_binary_mask,
    convert_matched_mask,
    aggregate_masked_features,
)

# ── Concrete backends ─────────────────────────────────────────────────────────
from .mobile_sam import MobileSAMSegmentation, load_mobile_sam
from .sam3 import SAM3Segmentation
from .insid3 import INSID3Segmentation

__all__ = [
    # abstract base
    "BaseSegmentation",
    # mask utilities
    "create_patch_mask",
    "create_composite_mask",
    "mask_id_to_binary_mask",
    "convert_matched_mask",
    "aggregate_masked_features",
    # backends
    "MobileSAMSegmentation",
    "load_mobile_sam",
    "SAM3Segmentation",
    "INSID3Segmentation",
]
```

- [ ] **Step 2: Verify import works**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.semantics.segmentation import INSID3Segmentation; print('OK', INSID3Segmentation)"
```

Expected: `OK <class '...INSID3Segmentation'>`

- [ ] **Step 3: Verify registry**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.semantics.segmentation import BaseSegmentation, INSID3Segmentation; assert BaseSegmentation.get('insid3') is INSID3Segmentation; print('registry OK')"
```

Expected: `registry OK`

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass; INSID3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/semantics/segmentation/__init__.py
git commit -m "feat(segmentation): export INSID3Segmentation from package"
```

---

## Task 7: `plot_context_segmentation` visualization

**Files:**
- Modify: `collab_splats/utils/visualization.py`

- [ ] **Step 1: Add `plot_context_segmentation` to `visualization.py`**

Read the existing file first to find the `# ── Semantic Visualization ──` section. Add the new function after `plot_heatmap` (which ends around line 90-ish based on the file structure). The function goes in the Semantic Visualization section.

```python
def plot_context_segmentation(
    ref_image: "np.ndarray | Image.Image",
    ref_mask: "np.ndarray | torch.Tensor",
    tgt_image: "np.ndarray | Image.Image",
    pred_mask: "np.ndarray | torch.Tensor",
    alpha: float = 0.45,
) -> "plt.Figure":
    """Side-by-side colored overlay for in-context segmentation results.

    Reference + context mask shown in red; target + predicted mask shown in green.
    Returns the Figure — caller decides whether to show or save.

    Args:
        ref_image: Reference image as numpy (H, W, 3) uint8 or PIL Image.
        ref_mask: Binary context mask as numpy bool or torch bool tensor (H, W).
        tgt_image: Target image as numpy (H, W, 3) uint8 or PIL Image.
        pred_mask: Predicted binary mask as numpy bool or torch bool tensor (H, W).
        alpha: Overlay opacity (default 0.45).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image as _Image

    def _to_np_image(img):
        if isinstance(img, _Image.Image):
            return np.array(img.convert("RGB"))
        return np.asarray(img)

    def _to_np_mask(mask, ref_shape):
        import torch as _torch
        if isinstance(mask, _torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask).squeeze().astype(bool)
        if mask.shape != ref_shape[:2]:
            from PIL import Image as _PIL
            mask = np.array(
                _PIL.fromarray(mask.astype(np.uint8) * 255).resize(
                    (ref_shape[1], ref_shape[0]), resample=_PIL.NEAREST
                )
            ) > 0
        return mask

    def _overlay(image_np, mask_np, color, alpha):
        out = image_np.astype(np.float32).copy()
        color_arr = np.array(color, dtype=np.float32) * 255.0
        out[mask_np] = (1.0 - alpha) * out[mask_np] + alpha * color_arr
        return np.clip(out, 0, 255).astype(np.uint8)

    ref_np = _to_np_image(ref_image)
    tgt_np = _to_np_image(tgt_image)
    ref_mask_np = _to_np_mask(ref_mask, ref_np.shape)
    pred_mask_np = _to_np_mask(pred_mask, tgt_np.shape)

    ref_overlay = _overlay(ref_np, ref_mask_np, color=(0.95, 0.25, 0.2), alpha=alpha)
    tgt_overlay = _overlay(tgt_np, pred_mask_np, color=(0.15, 0.8, 0.35), alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(ref_overlay)
    axes[0].set_title("Reference + context mask")
    axes[0].axis("off")
    axes[1].imshow(tgt_overlay)
    axes[1].set_title("Target + prediction")
    axes[1].axis("off")
    return fig
```

- [ ] **Step 2: Verify import**

```bash
/opt/conda/envs/reconstruction/bin/python -c "from collab_splats.utils.visualization import plot_context_segmentation; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Smoke-test the function**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import numpy as np
import torch
from collab_splats.utils.visualization import plot_context_segmentation
ref = np.zeros((64, 64, 3), dtype='uint8')
tgt = np.zeros((64, 64, 3), dtype='uint8')
ref_mask = np.zeros((64, 64), dtype=bool); ref_mask[20:44, 20:44] = True
pred_mask = torch.zeros(64, 64, dtype=torch.bool); pred_mask[10:30, 10:30] = True
fig = plot_context_segmentation(ref, ref_mask, tgt, pred_mask)
print('shape OK', fig.get_size_inches())
import matplotlib.pyplot as plt; plt.close(fig)
"
```

Expected: `shape OK [12.  6.]`

- [ ] **Step 4: Commit**

```bash
git add collab_splats/utils/visualization.py
git commit -m "feat(viz): add plot_context_segmentation for in-context segmentation results"
```

---

## Final check

- [ ] **Run full test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass, no regressions.
