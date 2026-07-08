# LoMa Matcher Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LoMa (LoMa-B and LoMa-G) as registered local feature extractors for camera localization, alongside DISK and XFeat.

**Architecture:** One adapter class per variant (`LomaExtractor` for LoMa-B, `LomaGExtractor(LomaExtractor)` for LoMa-G) implementing the existing `BaseLocalExtractor` contract in `collab_splats/pointcloud/localization.py`. Distinct classes per registry name keep `CameraLocalizer.from_feedforward`'s class→name reverse lookup for zarr cache keying unambiguous (this supersedes the spec's instance-attribute suggestion — same requirement, cleaner mechanism). `lomatch` becomes a required base dependency; the existing `[tool.uv] override-dependencies` pin `torchvision==0.20.1+cu121` already neutralizes lomatch's `torchvision>=0.23` floor.

**Tech Stack:** `lomatch` (pip), torch 2.5.1+cu121, existing pycolmap PnP stage (unchanged).

**Spec:** `docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md`

**Environment notes for the engineer:**
- Python: `/opt/venv/reconstruction/bin/python` (py3.11). NOT the base-shell `python`.
- `docs/superpowers/` is gitignored — commit plan/spec/decision docs with `git add -f`.
- Tests that download model weights get `@pytest.mark.slow` (see `tests/pointcloud/test_localization.py:39`).
- LoMa weights auto-download on first construction: LoMa-B 723 MB, LoMa-G 1.4 GB, plus DaD detector `dad.pth` and DINOv2 `dinov2_vitl14_pretrain.pth`, all into the torch hub cache. First slow-test run takes minutes.

## lomatch API cheat sheet (verified against LoMa source, main branch)

- `from loma import LoMa, LoMaB, LoMaG` — presets are frozen dataclasses subclassing `LoMa.Cfg`. `LoMa(LoMaB())` builds matcher + frozen DaD detector + DeDoDe-G (DINOv2 ViT-L) descriptor; weights load via `torch.hub.load_state_dict_from_url`.
- Cfg defaults: `filter_threshold=0.1`, `mp=True` (internal autocast), `compile=False`, `num_keypoints=2048`. LoMa-B: `input_dim=256` → descriptors D=256.
- Detect: `model.detect(batch, num_keypoints=N)` → `{"keypoints": (B,N,2), "keypoint_probs": (B,N)}` — keypoints are **(x, y) normalized [-1, 1]** (grid_sample convention).
- Describe: `model._descriptor.describe_keypoints(images, keypoints)` → `{"descriptions": (B,N,D)}` — keypoints must be normalized [-1,1]. (`model.describe` requires a 2-image `Batch`; do not use it for single images.)
- Match: `scores = model(kpts0, kpts1, desc0, desc1)["scores"]`, then `from loma.loma import filter_matches`; `m0, m1, mscores0, mscores1 = filter_matches(scores, threshold)`; `m0 == -1` means unmatched. No image sizes needed (coords already normalized).
- Pixel conversion: `to_pixel_coords` does `x_px = w * (x + 1) / 2`; the inverse is `x_norm = 2 * x_px / w - 1`.
- Inputs: `(B,3,H,W)` float RGB in **[0,1]**, NOT ImageNet-normalized (models normalize internally). Reference resolution 784×784 (used by LoMa's own tests and DeDoDe path loading; 784 = 14·56, divisible by DINOv2 patch size).
- Device: module-level global `loma.device.device` (cuda > mps > cpu, chosen at import); `LoMa.__init__` moves everything there. All inference methods are `@torch.inference_mode()`; matcher forward autocasts internally when `cfg.mp`.
- Verified torch 2.5.1 / torchvision 0.20.1 compatible: no `tv_tensors`/`transforms.v2`; SDPA called positionally; `torch.compiler.disable` needs only torch≥2.1.

---

### Task 1: Dependency — lomatch in pyproject, sync, smoke-test gate

**Files:**
- Modify: `pyproject.toml` (base dependencies ~line 61; `[tool.uv] override-dependencies` at lines 228-232)

- [ ] **Step 1: Add lomatch to base dependencies**

In `pyproject.toml`, after the `"kornia",` line (~line 61), add:

```toml
    # LoMa sparse matcher (ECCV 2026) — localization extractor alternative to DISK/XFeat.
    # Declares torchvision>=0.23; neutralized by the torchvision override below.
    "lomatch>=1.0.0",
```

- [ ] **Step 2: Neutralize lomatch's bare `dataclasses` requirement**

lomatch declares `dataclasses>=0.8` unconditionally — a py3.6 backport that must not install on py3.11. In `[tool.uv] override-dependencies` (line 228), add one entry:

```toml
    # lomatch declares the py3.6 `dataclasses` backport unconditionally; never install it.
    "dataclasses>=0.8; python_version < '3.7'",
```

- [ ] **Step 3: Sync**

Run: `cd /workspace/collab-splats && /root/.local/bin/uv sync --all-extras`
Expected: resolves and installs `lomatch` 1.x; no torchvision change (stays 0.20.1+cu121); no `dataclasses` package installed.
Verify: `/opt/venv/reconstruction/bin/python -c "import loma; print(loma.__name__)"` → `loma`

- [ ] **Step 4: Smoke test — VERIFICATION GATE**

Run:

```bash
/opt/venv/reconstruction/bin/python - <<'EOF'
import torch
from loma import LoMa, LoMaB
model = LoMa(LoMaB())                       # downloads ~723MB on first run
a = torch.rand(1, 3, 784, 784)
b = torch.rand(1, 3, 784, 784)
det = model.detect(a, num_keypoints=512)
print("keypoints:", det["keypoints"].shape, "probs:", det["keypoint_probs"].shape)
desc = model._descriptor.describe_keypoints(a, det["keypoints"])["descriptions"]
print("descriptors:", desc.shape)
kA, kB = model.match(a, b)
print("e2e match:", kA.shape, kB.shape)
EOF
```

Expected: `keypoints: torch.Size([1, 512, 2]) probs: torch.Size([1, 512])`, `descriptors: torch.Size([1, 512, 256])`, e2e match prints two `(M, 2)` shapes (M may be small — random noise). No exceptions.

**If this fails on torch 2.5.1/torchvision 0.20.1: STOP. Do not work around it. Report the exact error — fallback (vendoring) needs user sign-off per the spec.**

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(localization): add lomatch dependency for LoMa matcher"
```

---

### Task 2: LomaExtractor + LomaGExtractor adapter (TDD)

**Files:**
- Modify: `collab_splats/pointcloud/localization.py` (imports ~line 33; new classes after `XFeatExtractor`, ~line 445)
- Test: `tests/pointcloud/test_loma_extractor.py` (new)

- [ ] **Step 1: Write failing registry tests**

Create `tests/pointcloud/test_loma_extractor.py`:

```python
import numpy as np
import pytest
import torch

from collab_splats.pointcloud.localization import (
    BaseLocalExtractor,
    LocalFeatures,
    LomaExtractor,
    LomaGExtractor,
)


def test_loma_registry():
    assert BaseLocalExtractor.get("loma") is LomaExtractor
    assert BaseLocalExtractor.get("loma-g") is LomaGExtractor


def test_loma_g_is_distinct_class():
    # from_feedforward reverse-looks-up class -> registry name for zarr cache
    # keying; each variant must be its own class so the lookup is unambiguous.
    assert LomaGExtractor is not LomaExtractor
    assert issubclass(LomaGExtractor, LomaExtractor)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loma_extractor.py -v`
Expected: FAIL — `ImportError: cannot import name 'LomaExtractor'`

- [ ] **Step 3: Implement the adapter**

In `collab_splats/pointcloud/localization.py`, extend the kornia import block (~line 33) with:

```python
import torch.nn.functional as F
from loma import LoMa, LoMaB, LoMaG
from loma.loma import filter_matches
```

(`F` may already be imported — check; keep a single import.)

After `XFeatExtractor` (~line 445), add:

```python
@BaseLocalExtractor.register("loma")
class LomaExtractor(BaseLocalExtractor):
    """LoMa-B local feature extractor and matcher (ECCV 2026).

    DaD keypoint detector + DeDoDe-G (DINOv2 ViT-L) descriptors + LoMa
    transformer matcher, from the `lomatch` package. Detection and description
    run once per image at a fixed 784x784 inference resolution; keypoints are
    stored in original-image pixel coordinates so the zarr feature cache and
    the 2D->3D assignment stage work unchanged.

    Weights: auto-downloaded to torch hub cache on first use (~723 MB for
    LoMa-B, plus DaD and DINOv2 backbones).
    """

    _cfg_factory = LoMaB
    _inference_hw = 784  # 14 * 56 — divisible by the DINOv2 patch size

    def __init__(self, top_k: int = 2048, filter_threshold: float = 0.1):
        # LoMa pins all modules and inputs to the module-level loma.device
        # global (cuda > cpu, chosen at import) — adopt it rather than fight it.
        from loma.device import device as loma_device
        self._device = loma_device
        self._top_k = top_k
        self._filter_threshold = filter_threshold

        # Load matcher + frozen detector/descriptor; downloads weights on first use
        self._loma = LoMa(self._cfg_factory()).eval()

        logger.debug("%s: loaded on %s", type(self).__name__, self._device)

    def extract(self, image: np.ndarray) -> LocalFeatures:
        """Extract DaD keypoints and DeDoDe descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            LocalFeatures with keypoints (N,2) in original pixel coords,
            descriptors (N,256), scores (N,).
        """
        H, W = image.shape[:2]

        # HxWx3 uint8 -> (1,3,784,784) float [0,1]; LoMa normalizes internally.
        # Square resize is safe: keypoints come back normalized [-1,1], which is
        # resolution-invariant, and are denormalized against the ORIGINAL (W,H).
        img_t = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        img_t = F.interpolate(
            img_t, size=(self._inference_hw, self._inference_hw),
            mode="bilinear", align_corners=False,
        ).to(self._device)

        # Detect (normalized [-1,1] xy) then describe at those keypoints
        with torch.inference_mode():
            det = self._loma.detect(img_t, num_keypoints=self._top_k)
            kpts_n = det["keypoints"]            # (1, N, 2) normalized xy
            probs = det["keypoint_probs"]        # (1, N)
            descs = self._loma._descriptor.describe_keypoints(img_t, kpts_n)[
                "descriptions"
            ]                                    # (1, N, 256)

        # Denormalize to original pixel coords: x_px = W * (x + 1) / 2
        wh = torch.tensor([W, H], dtype=torch.float32)
        kpts_px = (kpts_n[0].cpu().float() + 1.0) * wh / 2.0

        return LocalFeatures(
            keypoints=kpts_px,
            descriptors=descs[0].cpu().float(),
            scores=probs[0].cpu().float(),
        )

    def match(
        self,
        query: LocalFeatures,
        db: LocalFeatures,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Match query features against database features with the LoMa matcher.

        Args:
            query:    LocalFeatures from the query image.
            db:       LocalFeatures from the database image.
            image_hw: (H, W) — used to re-normalize pixel coords to [-1, 1].

        Returns:
            matches: (K, 2) int64 — [query_idx, db_idx] pairs.
        """
        # Pixel -> normalized [-1,1] (inverse of loma.loma.to_pixel_coords)
        wh = torch.tensor([image_hw[1], image_hw[0]], dtype=torch.float32)
        k0 = (2.0 * query.keypoints / wh - 1.0).unsqueeze(0).to(self._device)
        k1 = (2.0 * db.keypoints / wh - 1.0).unsqueeze(0).to(self._device)
        d0 = query.descriptors.unsqueeze(0).to(self._device)
        d1 = db.descriptors.unsqueeze(0).to(self._device)

        # Matcher forward autocasts internally (cfg.mp); returns log-assignment scores
        with torch.inference_mode():
            scores = self._loma(k0, k1, d0, d1)["scores"]
        m0, _, _, _ = filter_matches(scores, self._filter_threshold)

        # m0[i] = index in db for query kpt i, or -1 if unmatched
        m0 = m0[0].cpu()
        valid = m0 > -1
        idx_q = torch.where(valid)[0]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, m0[valid]], dim=1).long()


@BaseLocalExtractor.register("loma-g")
class LomaGExtractor(LomaExtractor):
    """LoMa-G variant: larger matcher (embed_dim=1024), best accuracy, ~1.4 GB weights."""

    _cfg_factory = LoMaG
```

Implementation checkpoints (verify against the installed package, adjust if needed):
- `filter_matches(scores, th)` return arity/batching — check `demo.py` in the lomatch source (`/opt/venv/reconstruction/lib/python3.11/site-packages/loma/` or the GitHub repo) for the exact composed flow.
- `describe_keypoints` output key is `"descriptions"`.
- If `torch.nn.functional` is already imported under a different alias, reuse it.

- [ ] **Step 4: Run registry tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loma_extractor.py -v`
Expected: 2 PASS (no weight download — registry tests never instantiate).

- [ ] **Step 5: Write slow extract/match tests**

Append to `tests/pointcloud/test_loma_extractor.py`:

```python
@pytest.fixture(scope="module")
def loma_extractor():
    """Shared instance — construction downloads ~723 MB of weights."""
    return LomaExtractor(top_k=512)


@pytest.mark.slow
def test_loma_extract_shapes(loma_extractor):
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = loma_extractor.extract(image)
    assert isinstance(feats, LocalFeatures)
    assert feats.keypoints.ndim == 2 and feats.keypoints.shape[1] == 2
    assert feats.descriptors.ndim == 2 and feats.descriptors.shape[1] == 256
    assert len(feats.keypoints) == len(feats.descriptors) == len(feats.scores)
    assert len(feats.keypoints) > 0
    # Keypoints are in ORIGINAL pixel coords (not 784x784 inference coords)
    assert feats.keypoints[:, 0].max() <= 640
    assert feats.keypoints[:, 1].max() <= 480


@pytest.mark.slow
def test_loma_match_returns_index_pairs(loma_extractor):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    feats = loma_extractor.extract(img)
    matches = loma_extractor.match(feats, feats, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
    assert matches.dtype == torch.long
    # Self-match: most matches should be the identity pair
    if len(matches) > 0:
        diag = (matches[:, 0] == matches[:, 1]).sum()
        assert diag > 0
```

- [ ] **Step 6: Run slow tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loma_extractor.py -v -m slow`
Expected: 2 PASS (first run downloads weights — allow several minutes).

- [ ] **Step 7: Run the full localization test files for regressions**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_localization.py tests/pointcloud/test_localization_cache.py tests/pointcloud/test_loma_extractor.py -v`
Expected: all PASS (check `docs/known-test-failures.md` if anything unrelated fails).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_loma_extractor.py
git commit -m "feat(localization): LoMa-B/LoMa-G local extractors via lomatch"
```

---

### Task 3: Package exports

**Files:**
- Modify: `collab_splats/pointcloud/__init__.py:20-27` (localization import block) and `__all__` (lines 75-97)
- Test: `tests/pointcloud/test_loma_extractor.py`

- [ ] **Step 1: Write failing export test**

Append to `tests/pointcloud/test_loma_extractor.py`:

```python
def test_loma_exported_from_pointcloud_package():
    from collab_splats.pointcloud import LomaExtractor as le
    from collab_splats.pointcloud import LomaGExtractor as lge
    assert le is LomaExtractor and lge is LomaGExtractor
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loma_extractor.py::test_loma_exported_from_pointcloud_package -v`
Expected: FAIL — `ImportError`

- [ ] **Step 2: Add exports**

In `collab_splats/pointcloud/__init__.py`, extend the `from .localization import (...)` block (lines 20-27) with `LomaExtractor,` and `LomaGExtractor,` (alphabetical: after `LocalFeatures`), and add both names to `__all__` (alphabetical order).

- [ ] **Step 3: Run test to verify it passes**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_loma_extractor.py -v`
Expected: PASS (non-slow tests).

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/__init__.py tests/pointcloud/test_loma_extractor.py
git commit -m "feat(pointcloud): export LomaExtractor and LomaGExtractor"
```

---

### Task 4: RoMaV2 deferral decision doc

**Files:**
- Create: `docs/superpowers/decisions/014-defer-romav2.md`

- [ ] **Step 1: Write the decision doc**

```markdown
# 014 — Defer RoMaV2 dense matcher integration

**Date:** 2026-07-08
**Status:** Accepted

## Context

While integrating LoMa as a localization matcher
(spec: ../specs/2026-07-08-loma-matcher-integration-design.md), we evaluated
RoMaV2 (github.com/Parskatt/RoMaV2) — the current top dense matcher
(Mega-1500 pose AUC@5 62.8, ScanNet-1500 34.0).

## Decision

Do not integrate RoMaV2 now. Reasons:

1. Its required `fused-local-corr` CUDA kernel pins `torch==2.11.0`, and
   `torchvision>=0.23` implies torch>=2.8 — our stack is torch 2.5.1+cu121.
2. DINOv3 backbone weights carry Meta's custom DINOv3 license
   (the 1045 MB release checkpoint embeds DINOv3-derived weights).
3. Dense warp output does not fit the `BaseLocalExtractor` keypoint
   extract/match contract — it would need a separate dense-matcher path.
4. Open stability issues upstream: VRAM leak in the fused kernel (#40),
   OOM (#15, #17), high outlier ratios (#21).
5. LoMa-G already matches or beats RoMa v2 on WxBS and IMC22, with clean
   MIT/Apache licensing and no custom CUDA ops.

## Revisit trigger

Torch stack upgrade to >=2.8 (kernel pin may also relax upstream). Re-check
the DINOv3 license question at that point.
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/decisions/014-defer-romav2.md
git commit -m "docs(decisions): 014 — defer RoMaV2 dense matcher"
```

---

### Task 5: Notebook demo — LoMa vs DISK in the localization tutorial

**Files:**
- Modify: `docs/source/tutorials/07_localization/localization.ipynb`

- [ ] **Step 1: Read the notebook structure**

Read `docs/source/tutorials/07_localization/localization.ipynb` (use the Read tool / NotebookEdit — it renders cells). Note:
- the variable holding the feedforward/reconstruction result passed to `CameraLocalizer.from_feedforward`,
- the query image + intrinsics variables,
- how the DISK localization result and match visualization are produced (`LocalizationResult` fields: `n_correspondences`, `n_inliers`, `pts2d`, `pts2d_ref`, `inlier_mask`).

- [ ] **Step 2: Append a "LoMa matcher" section**

Add a markdown cell + code cells after the existing DISK localization results (adapt variable names to what Step 1 found — the logic below is fixed):

Markdown cell:

```markdown
## LoMa matcher

[LoMa](https://github.com/davnords/LoMa) (ECCV 2026) is a sparse keypoint
matcher — DaD detector + DINOv2 descriptors + a LightGlue-style transformer
matcher — that substantially outperforms LightGlue pipelines on WxBS/IMC22.
It is registered as `"loma"` (LoMa-B) and `"loma-g"` (LoMa-G, best accuracy).
Weights auto-download on first use (~723 MB / ~1.4 GB).
```

Code cell (localize with LoMa; reuse the notebook's existing `result`, `query_image`, `query_K` names):

```python
from collab_splats.pointcloud import LomaExtractor

localizer_loma = CameraLocalizer.from_feedforward(result, extractor=LomaExtractor())
res_loma = localizer_loma.localize(query_image, query_K)
print(f"LoMa-B: {res_loma.n_inliers}/{res_loma.n_correspondences} inliers")
```

Code cell (side-by-side comparison against the DISK result computed earlier in the notebook, e.g. `res_disk`):

```python
import pandas as pd

pd.DataFrame(
    [
        ("DISK+LightGlue", res_disk.n_correspondences, res_disk.n_inliers),
        ("LoMa-B", res_loma.n_correspondences, res_loma.n_inliers),
    ],
    columns=["matcher", "correspondences", "RANSAC inliers"],
)
```

Code cell: reuse the notebook's existing match-visualization code on `res_loma` (`pts2d`, `pts2d_ref`, `inlier_mask`) to show LoMa matches next to the DISK ones. If the notebook computes a pose error against a held-out ground-truth pose, add the same computation for `res_loma`.

- [ ] **Step 3: Execute the notebook — VERIFICATION GATE (spec gate 3)**

Run: `/opt/venv/reconstruction/bin/python -m jupyter nbconvert --to notebook --execute --inplace docs/source/tutorials/07_localization/localization.ipynb --ExecutePreprocessor.timeout=1800`

Expected: executes end-to-end; the LoMa cell reports a pose (non-None) and an inlier count in the same ballpark as (or better than) DISK. Record both numbers in the final report.

If the notebook needs scene data that is not present locally (collab-data rclone mount), report the blocker with the exact missing path instead of fabricating outputs — do not commit a half-executed notebook.

- [ ] **Step 4: Commit**

```bash
git add docs/source/tutorials/07_localization/localization.ipynb
git commit -m "docs(tutorials): LoMa matcher demo in localization notebook"
```

---

### Task 6: Update CLAUDE.md in-flight list

**Files:**
- Modify: `CLAUDE.md` (In-Flight Work section)

- [ ] **Step 1: Add the entry**

Add under "In-Flight Work":

```markdown
- **loma-matcher** — LoMa local matcher for localization ([spec](docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md) · [plan](docs/superpowers/plans/2026-07-08-loma-matcher-integration.md))
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: track loma-matcher in-flight work"
```
