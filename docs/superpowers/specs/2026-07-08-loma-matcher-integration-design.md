# LoMa Matcher Integration — Design

**Date:** 2026-07-08
**Status:** Approved
**Scope decision:** Integrate LoMa only. RoMaV2 deferred (see "RoMaV2 deferral" below).

## Goal

Add LoMa as an alternative local feature matcher for camera localization, alongside the
existing DISK+LightGlue and XFeat extractors, with minimal new code and dependencies.

## Background: current matcher architecture

All local-feature matching lives in `collab_splats/pointcloud/localization.py`:

- `BaseLocalExtractor(RegistryMixin, ABC)` — combines detection + matching. Contract:
  - `extract(image: HxWx3 uint8) -> LocalFeatures` (`keypoints (N,2) f32`,
    `descriptors (N,D) f32`, `scores (N,)|None`) — cached per reference frame in zarr,
    keyed by extractor name.
  - `match(query: LocalFeatures, db: LocalFeatures, image_hw) -> (K,2) int64`
    index pairs `[query_idx, db_idx]`.
- Registered: `"disk"` (kornia DISK + LightGlue, D=128), `"xfeat"` (vendored, D=64).
- Pose: `pycolmap.estimate_and_refine_absolute_pose` (LO-RANSAC + Ceres) in
  `CameraLocalizer.localize`.
- Webapp selects extractor by string in `webapp/routers/localize.py`;
  default in `webapp/state.py`.

## Candidate analysis

### LoMa (github.com/davnords/LoMa) — integrate

- Sparse keypoint matcher, ECCV 2026 (Nordström, Edstedt, et al. — same group as RoMa).
  DaD detector + DINOv2/DeDoDe descriptors + LightGlue-style matcher.
- Accuracy: claims significant win over LightGlue; LoMa-G surpasses RoMa v1 **and** v2
  on WxBS and IMC22 (LoMa-B WxBS mAA_10px 0.6876).
- Packaging: `pip install lomatch` (PyPI 1.0.0), pure PyTorch, no custom CUDA ops.
- License: MIT; matcher module Apache-2.0 (inherited from LightGlue). Clean.
- Source layout separates `detector/dad.py`, `descriptor/`, and matcher (`loma.py`),
  with dedicated detect-only / describe-only / match-only throughput tests — so it can
  be split to fit our `extract`/`match` contract and reuse the zarr feature cache.
- Weights auto-download on first instantiation (torch.hub cache):
  LoMa-B 723 MB, LoMa-G 1.4 GB, from `github.com/davnords/storage` releases.
  Pinned URLs (link-rot hedge):
  - LoMa-B: https://github.com/davnords/storage/releases/download/loma/loma_B.pt
  - LoMa-G: https://github.com/davnords/storage/releases/download/loma/loma_G.pth
- Compatibility risk: declares `torchvision>=0.23.0` (implies torch>=2.8); our env is
  torch 2.5.1+cu121 / torchvision 0.20.1. torch itself is unpinned. Mitigation: uv
  `override-dependencies` + smoke-test gate (below).

### RoMaV2 (github.com/Parskatt/RoMaV2) — deferral

Deferred despite top dense-matching accuracy (Mega-1500 AUC@5 62.8, ScanNet-1500 34.0):

1. Required `fused-local-corr` CUDA kernel pins `torch==2.11.0`; `torchvision>=0.23`
   implies torch>=2.8. We run torch 2.5.1 — hard incompatibility.
2. DINOv3 backbone weights carry Meta's custom DINOv3 license (checkpoint 1045 MB) —
   needs legal review before shipping.
3. Dense warp output does not fit the keypoint `extract`/`match` contract — would need
   a separate dense-matcher path (larger design surface).
4. Open stability issues: VRAM leak in fused kernel (#40), OOM (#15/#17),
   outlier ratios (#21).
5. LoMa-G already matches or beats RoMa v2 on WxBS/IMC22.

Revisit when the stack reaches torch>=2.8. Record as a decision doc entry during
implementation.

## Design

### Dependency

- Add `lomatch>=1.0.0` to the **required base dependencies** in `pyproject.toml` —
  localization is a core pipeline stage, not an optional feature.
- Add `[tool.uv] override-dependencies` entry relaxing lomatch's `torchvision>=0.23.0`
  floor to our installed 0.20.1.
- Hard import at the top of `localization.py`, matching the module's existing imports
  (per code style: hard imports, no stub backends).

### Adapter

One class in `collab_splats/pointcloud/localization.py`:

```python
@BaseLocalExtractor.register("loma")     # LoMa-B (723 MB, faster default)
@BaseLocalExtractor.register("loma-g")   # LoMa-G (1.4 GB, best accuracy)
class LomaExtractor(BaseLocalExtractor):
    def extract(self, image) -> LocalFeatures: ...   # DaD detect + describe; D=256
    def match(self, query, db, image_hw) -> Tensor: ...  # (K,2) int64 index pairs
```

- Registered names select the variant via a constructor arg; same class, config-only
  difference.
- Note: `CameraLocalizer.from_feedforward` reverse-looks-up class → registry name for
  zarr cache keying; with two names mapping to one class this lookup is ambiguous.
  The implementation must key the cache by variant (e.g. an instance-level name
  attribute), not by class identity.
- `extract` runs DaD detection + descriptor sampling; returns `LocalFeatures` so the
  existing zarr cache path works unchanged (cache already keyed by extractor name).
- `match` runs the LoMa matcher on two cached descriptor sets and converts its output
  to `(K,2)` index pairs.
- Exports added to `collab_splats/pointcloud/__init__.py`.

### Documentation demo

- Extend the existing localization tutorial notebook
  (`docs/source/tutorials/.../localization.ipynb`) with a LoMa section: swap the
  extractor to `"loma"` / `"loma-g"`, localize the same query frame, and show a
  side-by-side comparison against the DISK+LightGlue baseline (match visualization,
  inlier count, pose error).

### Verification gates (in order)

1. **Smoke test:** `lomatch` imports and matches an image pair on
   torch 2.5.1 / torchvision 0.20.1. **If this fails, stop and report** — fallback
   decision point is vendoring (rejected Approach B), which needs user sign-off.
2. **Unit tests** (`tests/pointcloud/`, flat functions): registry resolution for
   `"loma"`/`"loma-g"`, `extract` and `match` output shapes/dtypes.
3. **Integration:** localize a query frame on an existing 7-Scenes chess sequence;
   compare inlier count and pose error against the DISK+LightGlue baseline.

### Alternatives considered

- **Vendor into `third_party/` (like xfeat):** full control, no resolver fight, but
  ~40 source files, manual sync, carried licenses. XFeat was vendored only because no
  pip package existed; lomatch has one. Rejected — kept as fallback if the smoke test
  fails.
- **Separate dense-matcher base class:** only needed for RoMaV2's dense output; LoMa
  is sparse and fits the existing contract. Dropped with the RoMaV2 deferral.

## Out of scope

- RoMaV2 integration (deferred, see above).
- Webapp wiring (`webapp/routers/localize.py`, `webapp/state.py`) — webapp is
  currently minimal for reconstruction; add loma picker options later if needed.
- Loop-closure wiring — loop closure gates on DINO-SALAD global retrieval + VGGT
  confidence; it does not use local matchers.
- Any retraining or fine-tuning.
- Offline weight pre-seeding automation (direct URLs documented above if needed).
