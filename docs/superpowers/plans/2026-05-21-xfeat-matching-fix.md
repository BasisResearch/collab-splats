# XFeat Matching Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `XFeatExtractor` to use canonical `match_lighterglue` instead of `kornia.match_mnn`, and unify the `extract()` / `match()` interface across both extractors so `scores` flow through the pipeline.

**Architecture:** Both `DiskExtractor` and `XFeatExtractor` gain a uniform 3-tuple `extract()` return `(kpts, scores, descs)` and a unified `match()` signature accepting `scores_q, scores_db`. `XFeatExtractor.match()` uses `xfeat.match_lighterglue(d0, d1)` with proper dict format. `CameraLocalizer` stores and unpacks 3-tuples; inner PnP loop is unchanged.

**Tech Stack:** kornia 0.8.1, vendor/xfeat (accelerated_features.XFeat), PyTorch, NumPy, OpenCV, pytest.

---

## Files

- Modify: `collab_splats/pointcloud/localization.py` — extractor classes + CameraLocalizer
- Modify: `tests/pointcloud/test_localization.py` — update extractor tests + `_MockExtractor`

---

## Task 1: Update DiskExtractor

**Files:**
- Modify: `collab_splats/pointcloud/localization.py:162-230`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Update failing tests for DiskExtractor**

Replace the two existing slow DISK tests in `tests/pointcloud/test_localization.py`:

```python
@pytest.mark.slow
def test_disk_extractor_returns_keypoints_and_descriptors():
    """Requires network access to download DISK weights (~4 MB)."""
    from collab_splats.pointcloud.localization import DiskExtractor
    extractor = DiskExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, scores, descs = extractor.extract(image)
    assert kpts.ndim == 2 and kpts.shape[1] == 2
    assert scores.ndim == 1
    assert descs.ndim == 2 and descs.shape[1] == 128
    assert len(kpts) == len(scores) == len(descs)
    assert len(kpts) > 0

@pytest.mark.slow
def test_disk_extractor_match_returns_index_pairs():
    from collab_splats.pointcloud.localization import DiskExtractor
    extractor = DiskExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, scores, descs = extractor.extract(img)
    matches = extractor.match(kpts, scores, descs, kpts, scores, descs, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
    diag = (matches[:, 0] == matches[:, 1]).sum()
    assert diag > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "disk_extractor" -v -m slow
```

Expected: FAIL — `not enough values to unpack` (extract returns 2-tuple) or similar.

- [ ] **Step 3: Update DiskExtractor.extract() to return 3-tuple**

Replace the `extract` method body in `collab_splats/pointcloud/localization.py` (lines ~162–183):

```python
    def extract(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract DISK keypoints, scores, and descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            keypoints:   (N, 2) float32 pixel coordinates [x, y].
            scores:      (N,) float32 detection scores.
            descriptors: (N, 128) float32.
        """
        # Convert HxWx3 uint8 to 1xCxHxW float tensor in [0, 1]
        img_t = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(self._device)

        # Detect keypoints and compute descriptors
        with torch.no_grad():
            features = self._disk(img_t, self._top_k, pad_if_not_divisible=True)
        kpts = features[0].keypoints.cpu()          # (N, 2)
        scores = features[0].detection_scores.cpu() # (N,)
        descs = features[0].descriptors.cpu()       # (N, 128)

        return kpts, scores, descs
```

- [ ] **Step 4: Update DiskExtractor.match() signature**

Replace the `match` method signature and docstring in `collab_splats/pointcloud/localization.py` (lines ~185–230). The implementation body is unchanged — only the signature and docstring change:

```python
    def match(
        self,
        kpts_q: torch.Tensor,
        scores_q: torch.Tensor,
        descs_q: torch.Tensor,
        kpts_db: torch.Tensor,
        scores_db: torch.Tensor,
        descs_db: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Match query features against database features using LightGlue.

        Args:
            kpts_q:    (N, 2) query keypoints.
            scores_q:  (N,) query scores (unused — LightGlue does not require them).
            descs_q:   (N, 128) query descriptors.
            kpts_db:   (M, 2) database keypoints.
            scores_db: (M,) database scores (unused).
            descs_db:  (M, 128) database descriptors.
            image_hw:  (H, W) — required for LightGlue coordinate normalisation.

        Returns:
            matches: (K, 2) int64 — [query_idx, db_idx] pairs.
        """
        # LightGlue normalize_keypoints expects image_size as [W, H] (not [H, W])
        wh = torch.tensor([[image_hw[1], image_hw[0]]], dtype=torch.float32)
        data = {
            "image0": {
                "keypoints": kpts_q.unsqueeze(0).to(self._device),
                "descriptors": descs_q.unsqueeze(0).to(self._device),
                "image_size": wh.to(self._device),
            },
            "image1": {
                "keypoints": kpts_db.unsqueeze(0).to(self._device),
                "descriptors": descs_db.unsqueeze(0).to(self._device),
                "image_size": wh.to(self._device),
            },
        }

        # matches0[i] = index in image1 for kpt i in image0, or -1 if unmatched
        with torch.no_grad():
            result = self._lightglue(data)

        matches0 = result["matches0"][0].cpu()   # (N,)
        valid = matches0 > -1
        idx_q = torch.where(valid)[0]
        idx_db = matches0[valid]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, idx_db], dim=1)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "disk_extractor" -v -m slow
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "refactor(localization): DiskExtractor.extract returns 3-tuple (kpts, scores, descs)"
```

---

## Task 2: Fix XFeatExtractor

**Files:**
- Modify: `collab_splats/pointcloud/localization.py:233-294`
- Modify: `tests/pointcloud/test_localization.py`

- [ ] **Step 1: Update failing tests for XFeatExtractor**

Replace the two existing slow XFeat tests in `tests/pointcloud/test_localization.py`:

```python
@pytest.mark.slow
def test_xfeat_extractor_returns_keypoints_and_descriptors():
    from collab_splats.pointcloud.localization import XFeatExtractor
    extractor = XFeatExtractor()
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, scores, descs = extractor.extract(image)
    assert kpts.ndim == 2 and kpts.shape[1] == 2
    assert scores.ndim == 1
    assert descs.ndim == 2 and descs.shape[1] == 64
    assert len(kpts) == len(scores) == len(descs)
    assert len(kpts) > 0

@pytest.mark.slow
def test_xfeat_extractor_match_returns_index_pairs():
    from collab_splats.pointcloud.localization import XFeatExtractor
    extractor = XFeatExtractor()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    kpts, scores, descs = extractor.extract(img)
    matches = extractor.match(kpts, scores, descs, kpts, scores, descs, image_hw=(480, 640))
    assert matches.ndim == 2 and matches.shape[1] == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "xfeat_extractor" -v -m slow
```

Expected: FAIL — `not enough values to unpack` or `TypeError` on match signature.

- [ ] **Step 3: Update XFeatExtractor.extract() to return 3-tuple**

Replace the `extract` method body in `collab_splats/pointcloud/localization.py` (lines ~250–270):

```python
    def extract(self, image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract XFeat keypoints, scores, and descriptors.

        Args:
            image: HxWx3 uint8 RGB image.

        Returns:
            keypoints:   (N, 2) float32 pixel coordinates [x, y].
            scores:      (N,) float32 keypoint saliency scores.
            descriptors: (N, 64) float32 L2-normalised.
        """
        # Convert HxWx3 uint8 to 1xCxHxW float tensor in [0, 1]
        img_t = (
            torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )

        # Detect keypoints and compute descriptors
        out = self._xfeat.detectAndCompute(img_t, top_k=self._top_k)
        kpts = out[0]["keypoints"].cpu()      # (N, 2)
        scores = out[0]["scores"].cpu()       # (N,)
        descs = out[0]["descriptors"].cpu()   # (N, 64)

        return kpts, scores, descs
```

- [ ] **Step 4: Replace XFeatExtractor.match() to use match_lighterglue**

Replace the entire `match` method in `collab_splats/pointcloud/localization.py` (lines ~272–294):

```python
    def match(
        self,
        kpts_q: torch.Tensor,
        scores_q: torch.Tensor,
        descs_q: torch.Tensor,
        kpts_db: torch.Tensor,
        scores_db: torch.Tensor,
        descs_db: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Match query features against database features using XFeat LighterGlue.

        Uses xfeat.match_lighterglue — XFeat's built-in lighter LightGlue variant.
        Canonical usage: detectAndCompute → dict with image_size → match_lighterglue.

        Args:
            kpts_q:    (N, 2) query keypoints.
            scores_q:  (N,) query saliency scores — required by LighterGlue.
            descs_q:   (N, 64) query descriptors.
            kpts_db:   (M, 2) database keypoints.
            scores_db: (M,) database saliency scores.
            descs_db:  (M, 64) database descriptors.
            image_hw:  (H, W) — required for LighterGlue coordinate normalisation.

        Returns:
            matches: (K, 2) int64 — [query_idx, db_idx] pairs.
        """
        # match_lighterglue expects image_size as (W, H)
        W, H = image_hw[1], image_hw[0]
        d0 = {
            "keypoints": kpts_q.to(self._device),
            "scores": scores_q.to(self._device),
            "descriptors": descs_q.to(self._device),
            "image_size": (W, H),
        }
        d1 = {
            "keypoints": kpts_db.to(self._device),
            "scores": scores_db.to(self._device),
            "descriptors": descs_db.to(self._device),
            "image_size": (W, H),
        }

        # Returns mkpts_0, mkpts_1, idx — idx is (K, 2) index pairs
        _, _, idx = self._xfeat.match_lighterglue(d0, d1)
        if len(idx) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.from_numpy(idx).long()
```

Also remove `match_mnn` from the top-level import since it's no longer used. Change line 27:

```python
from kornia.feature import DISK, LightGlue
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "xfeat_extractor" -v -m slow
```

Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "fix(localization): XFeatExtractor uses match_lighterglue, returns scores from extract"
```

---

## Task 3: Update CameraLocalizer and _MockExtractor

**Files:**
- Modify: `collab_splats/pointcloud/localization.py:384-570`
- Modify: `tests/pointcloud/test_localization.py` — `_MockExtractor`

- [ ] **Step 1: Update _MockExtractor in tests**

Replace the `_MockExtractor` class in `tests/pointcloud/test_localization.py`:

```python
class _MockExtractor:
    """Extractor that returns exact projected 3D positions as keypoints.

    descriptor[j] = one-hot-ish encoding of pt3d index j — matching is exact.
    Scores are uniform 1.0 (synthetic data has no saliency notion).
    """

    def __init__(self, pts3d, extrinsics, intrinsics):
        self._pts3d = pts3d
        self._extrinsics = extrinsics
        self._intrinsics = intrinsics
        self._call = 0
        n = len(pts3d)
        # Descriptors: identity matrix padded/truncated to 128 dims
        self._descs = torch.zeros(n, 128)
        for j in range(min(n, 128)):
            self._descs[j, j] = 1.0

    def extract(self, image):
        i = self._call % len(self._extrinsics)
        self._call += 1
        R = self._extrinsics[i, :3, :3]
        t = self._extrinsics[i, :3, 3]
        K = self._intrinsics[i]
        pts_cam = self._pts3d @ R.T + t
        pts_proj = pts_cam @ K.T
        kpts = torch.from_numpy(pts_proj[:, :2] / pts_proj[:, 2:3]).float()
        scores = torch.ones(len(kpts))
        return kpts, scores, self._descs

    def match(self, kpts_q, scores_q, descs_q, kpts_db, scores_db, descs_db, image_hw):
        sim = descs_q @ descs_db.T   # (N, M)
        best = sim.argmax(dim=1)
        valid = sim.max(dim=1).values > 0.5
        idx_q = torch.where(valid)[0]
        idx_db = best[valid]
        if len(idx_q) == 0:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.stack([idx_q, idx_db], dim=1)
```

- [ ] **Step 2: Run existing CameraLocalizer tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -k "camera_localizer" -v
```

Expected: FAIL — `CameraLocalizer.__init__` still calls `extract()` expecting 2-tuple.

- [ ] **Step 3: Update CameraLocalizer.__init__ to store 3-tuples**

In `collab_splats/pointcloud/localization.py`, update the feature extraction loop inside `CameraLocalizer.__init__` (lines ~426–443):

```python
        # Extract local features for all reference frames
        self._frame_features: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        first_hw: tuple[int, int] | None = None
        for path in image_paths:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"CameraLocalizer: cannot read {path}")
            rgb = bgr[..., ::-1].copy()
            if first_hw is None:
                first_hw = (rgb.shape[0], rgb.shape[1])
            kpts, scores, descs = self._extractor.extract(rgb)
            self._frame_features.append((kpts, scores, descs))
            logger.debug(
                "  frame %s: %d keypoints",
                path.name if hasattr(path, "name") else path,
                len(kpts),
            )
```

- [ ] **Step 4: Update _build_frame_assignments call (kpts still index 0)**

Still inside `CameraLocalizer.__init__`, the `_build_frame_assignments` call uses `f[0]` for keypoints — no change needed. Verify it reads:

```python
        self._assignments = _build_frame_assignments(
            pts3d=pts3d,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            frame_keypoints=[f[0] for f in self._frame_features],
            image_hw=self._image_hw,
            radius=radius,
        )
```

- [ ] **Step 5: Update CameraLocalizer.localize() to unpack 3-tuples and pass scores**

In `collab_splats/pointcloud/localization.py`, update `localize()` (lines ~496–526):

```python
        # Extract local features from query image
        kpts_q, scores_q, descs_q = self._extractor.extract(query_image)

        logger.debug("CameraLocalizer.localize: query has %d keypoints", len(kpts_q))

        # Match query against all reference frames; deduplicate via best score per 3D point
        best_score: dict[int, float] = {}    # pt3d_idx → best confidence so far
        best_2d: dict[int, np.ndarray] = {}  # pt3d_idx → corresponding query 2D point

        for i, (kpts_db, scores_db, descs_db) in enumerate(self._frame_features):
            if len(kpts_db) == 0:
                continue
            matches = self._extractor.match(
                kpts_q, scores_q, descs_q,
                kpts_db, scores_db, descs_db,
                self._image_hw,
            )
            if len(matches) == 0:
                continue

            # Resolve matched db keypoints to world 3D points via assignment map
            assignment = self._assignments[i]
            for q_idx, db_idx in matches.numpy():
                db_idx = int(db_idx)
                if db_idx not in assignment:
                    continue
                pt3d_idx = assignment[db_idx]
                pt2d = kpts_q[int(q_idx)].numpy().astype(np.float32)
                score = 1.0  # uniform — match_lighterglue idx does not carry per-match confidence

                if pt3d_idx not in best_score or score > best_score[pt3d_idx]:
                    best_score[pt3d_idx] = score
                    best_2d[pt3d_idx] = pt2d
```

- [ ] **Step 6: Run all localization tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_localization.py -v
```

Expected: all non-slow tests PASS. Slow tests skipped unless `-m slow` flag added.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/localization.py tests/pointcloud/test_localization.py
git commit -m "fix(localization): CameraLocalizer unpacks 3-tuple from extract, passes scores to match"
```

---

## Self-Review

**Spec coverage:**
- ✅ `DiskExtractor.extract()` returns 3-tuple with `detection_scores` — Task 1
- ✅ `DiskExtractor.match()` gains `scores_q/scores_db` params (ignored) — Task 1
- ✅ `XFeatExtractor.extract()` returns 3-tuple, stops dropping scores — Task 2
- ✅ `XFeatExtractor.match()` uses `match_lighterglue` with proper dict — Task 2
- ✅ `image_hw` now required (not optional) for `XFeatExtractor.match()` — Task 2
- ✅ `CameraLocalizer` stores 3-tuples in `_frame_features` — Task 3
- ✅ `CameraLocalizer.localize()` unpacks and passes scores — Task 3
- ✅ Tests updated for 3-tuple extract, new match signature — Tasks 1–3
- ✅ `match_mnn` import removed (no longer used) — Task 2 Step 4

**Type consistency:**
- `extract()` → `(Tensor(N,2), Tensor(N,), Tensor(N,D))` used as `kpts, scores, descs` throughout ✅
- `match(kpts_q, scores_q, descs_q, kpts_db, scores_db, descs_db, image_hw)` → `Tensor(K,2)` consistent across both extractors and all callers ✅
- `_frame_features: list[tuple[Tensor, Tensor, Tensor]]` — `f[0]` for kpts in `_build_frame_assignments` unchanged ✅
- `match_lighterglue` returns `(mkpts_0, mkpts_1, idx)` — we unpack as `_, _, idx` ✅
