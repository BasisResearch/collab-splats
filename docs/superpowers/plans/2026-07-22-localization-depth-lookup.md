# Localization Depth-Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lossy reverse-projection 2D→3D stage with pose_from_cluster-style per-frame `world_points` depth sampling; add an XFeat\* semi-dense matcher; decouple `LocalizationResult` from viz helpers.

**Architecture:** `match()` returns matched **pixel pairs** (`MatchResult`); `localize()` samples the reference frame's dense `world_points (H,W,3)` at each matched ref pixel (bilinear, invalid→drop) to get 3D, then solves PnP exactly as today. `_build_frame_assignments`, `radius`, and `self._assignments` are deleted. Spec: `docs/superpowers/specs/2026-07-22-localization-depth-lookup-design.md`.

**Tech Stack:** torch `grid_sample`, pycolmap (unchanged), vendored XFeat (`third_party/xfeat/modules/xfeat.py`), zarr CSR feature cache.

**Env:** `/opt/venv/reconstruction/bin/python`. Run tests as `/opt/venv/reconstruction/bin/python -m pytest <path> -x -q`.

---

### Task 1: `MatchResult` + sparse extractors return pixel pairs

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Test: `tests/localization/test_extractors.py` (existing — adapt)

- [ ] **Step 1: Write failing test** (append to `tests/localization/test_extractors.py`; reuse its existing fixtures/synthetic images):

```python
def test_match_returns_pixel_pairs_disk(synthetic_pair):
    """match() returns MatchResult whose pixels correspond to detected keypoints."""
    from collab_splats.localization.extractors import DiskExtractor, MatchResult

    img0, img1 = synthetic_pair  # reuse existing fixture; else two np.uint8 HxWx3 arrays
    ex = DiskExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult)
    assert m.query_px.shape == m.ref_px.shape and m.query_px.shape[1] == 2
    # every returned pixel must be one of the detected keypoints
    if len(m.query_px):
        kq = f0.keypoints.numpy()
        assert all(((kq == q).all(axis=1)).any() for q in m.query_px[:5])
```

If no synthetic-pair fixture exists, create two 128×128 uint8 images with random-dot texture (`rng.integers(0,255,(128,128,3))`, second = `np.roll(first, 4, axis=1)`).

- [ ] **Step 2: Run — expect FAIL** (`ImportError: MatchResult`).

- [ ] **Step 3: Implement.** In `extractors.py`:

```python
@dataclass
class MatchResult:
    """Matched pixel coordinates between a query and one reference image."""

    query_px: np.ndarray  # (K, 2) float32 xy in query image
    ref_px: np.ndarray    # (K, 2) float32 xy in reference image

    def __len__(self) -> int:
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """Zero-length MatchResult."""
    z = np.zeros((0, 2), dtype=np.float32)
    return MatchResult(query_px=z, ref_px=z)
```

Change `BaseLocalExtractor.match` abstract docstring/return to `MatchResult`. In **each** of `DiskExtractor.match`, `XFeatExtractor.match`, `LomaExtractor.match`: keep the index logic, replace the final `return torch.stack(...)` with:

```python
        if len(idx_q) == 0:
            return _empty_match()
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
        )
```

(XFeat path: `idx` is a `(K,2)` numpy array — `idx_q, idx_db = idx[:, 0], idx[:, 1]`. Loma path: `idx_db = m0[valid]`.) Drop the now-unused `scores` idea — YAGNI (spec open-question 1 resolved: rely on RANSAC).

- [ ] **Step 4: Run test — PASS.** Also run the full extractor test file; fix any test that asserted the old `(K,2)` index tensor to assert `MatchResult` instead.

- [ ] **Step 5: Commit** `refactor(localization): match() returns MatchResult pixel pairs`

---

### Task 2: `sample_world_points` helper

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/test_localizer.py`

- [ ] **Step 1: Write failing test:**

```python
def test_sample_world_points_bilinear_and_invalid():
    from collab_splats.localization.localizer import sample_world_points

    # 4x4 grid whose world point at (row r, col c) is (c, r, 1)
    H = W = 4
    wp = np.stack(np.meshgrid(np.arange(W), np.arange(H)) + [np.ones((H, W))], axis=-1).astype(np.float32)
    wp[0, 0] = np.nan  # unmapped pixel

    px = np.array([[2.0, 1.0], [1.5, 2.5], [0.0, 0.0]], dtype=np.float32)  # xy
    pts, valid = sample_world_points(wp, px)
    assert pts.shape == (3, 3) and valid.dtype == bool
    np.testing.assert_allclose(pts[0], [2.0, 1.0, 1.0], atol=1e-5)   # exact pixel
    np.testing.assert_allclose(pts[1], [1.5, 2.5, 1.0], atol=1e-5)   # bilinear midpoint
    assert not valid[2] and valid[0] and valid[1]                     # NaN cell dropped
```

- [ ] **Step 2: Run — FAIL** (`ImportError`).

- [ ] **Step 3: Implement** in `localizer.py` (below `seed_intrinsics`):

```python
def sample_world_points(
    world_points: np.ndarray,  # (H, W, 3) world-space per-pixel points
    px: np.ndarray,            # (K, 2) float32 xy pixel coords
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-sample per-pixel world points at px (hloc interpolate_scan analog).

    Returns (pts3d (K,3) float32, valid (K,) bool) — invalid where the sample
    touches NaN (unmapped pixels) or falls outside the image.
    """
    H, W, _ = world_points.shape
    # Normalize to [-1, 1] for grid_sample (align_corners=True convention)
    grid = torch.from_numpy(px / np.array([[W - 1, H - 1]], dtype=np.float32) * 2 - 1)
    wp = torch.from_numpy(world_points).permute(2, 0, 1)[None].float()  # (1,3,H,W)
    interp = F.grid_sample(wp, grid[None, None].float(), align_corners=True, mode="bilinear")[0, :, 0]  # (3,K)
    valid = ~torch.any(torch.isnan(interp), dim=0)
    # Out-of-bounds px → invalid (grid_sample pads with border values otherwise)
    in_bounds = torch.from_numpy(
        (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
    )
    valid = (valid & in_bounds).numpy()
    return interp.T.numpy().astype(np.float32), valid
```

Add `import torch.nn.functional as F` at top.

- [ ] **Step 4: Run test — PASS.**

- [ ] **Step 5: Commit** `feat(localization): sample_world_points depth lookup`

---

### Task 3: Rewire `CameraLocalizer` — world_points in, assignments out

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/` (multiple files — adapt constructor calls)

- [ ] **Step 1: Write failing test** (synthetic end-to-end, no GPU extractor — stub extractor):

```python
def test_localize_via_depth_lookup():
    """Query identical to ref frame 0 localizes at ref 0's pose via world_points sampling."""
    from collab_splats.localization.localizer import CameraLocalizer
    from collab_splats.localization.extractors import BaseLocalExtractor, LocalFeatures, MatchResult

    H = W = 64
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)

    # Planar scene at z=2: world point for pixel (x,y) with identity K-ish pinhole
    f = 50.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    z = 2.0
    wp = np.stack([(xs - W / 2) / f * z, (ys - H / 2) / f * z, np.full_like(xs, z, dtype=np.float64)], -1).astype(np.float32)

    class StubExtractor(BaseLocalExtractor):
        def extract(self, image):
            k = torch.tensor([[8.0 + 3 * i, 8.0 + 2 * i] for i in range(12)])
            return LocalFeatures(keypoints=k, descriptors=torch.zeros(12, 4))
        def match(self, query, db, image_hw):
            px = query.keypoints.numpy().astype(np.float32)
            return MatchResult(query_px=px, ref_px=px.copy())  # identity matches

    loc = CameraLocalizer(
        world_points=wp[None],                      # (1, H, W, 3)
        extrinsics=np.eye(4, dtype=np.float32)[None],
        images=[img], ids=["frame_0"], extractor=StubExtractor(),
    )
    res = loc.localize(img, query_intrinsics=K)
    assert res.pose is not None
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)  # camera at identity
    assert res.n_correspondences >= 10
```

- [ ] **Step 2: Run — FAIL** (ctor has no `world_points` param).

- [ ] **Step 3: Implement.** In `localizer.py`:

1. **Ctor signature:** `__init__(self, world_points, extrinsics, images, ids, extractor=None, config=None, progress_callback=None)`.
   - Delete params `pts3d`, `intrinsics`, `radius`. Store `self._world_points = world_points`. Keep `self._extrinsics` (used by the `extrinsics` property). First `grep -n "_pts3d\|_intrinsics\b" collab_splats/` — if the dashboard/webapp reads them off the localizer, keep a property; expected: unused → delete.
   - Delete the `_build_frame_assignments` call and the whole function (lines ~80-158).
2. **`localize()` correspondence loop** — replace the dedup-dict block (lines ~869-920) with:

```python
        # Match against each reference frame; 3D via depth lookup at the matched ref pixel
        all_q, all_3d, all_ref, all_frame = [], [], [], []
        for i, db_feats in enumerate(self._frame_features):
            if self._frame_sources[i] != "reconstruction":
                continue  # localized frames carry no world_points
            if len(db_feats.keypoints) == 0:
                continue
            m = self._extractor.match(query_feats, db_feats, self._image_hw)
            if len(m) == 0:
                continue
            pts3d, valid = sample_world_points(self._world_points[i], m.ref_px)
            if not valid.any():
                continue
            all_q.append(m.query_px[valid])
            all_3d.append(pts3d[valid])
            all_ref.append(m.ref_px[valid])
            all_frame.append(np.full(int(valid.sum()), i, dtype=np.int32))

        n_corr = sum(len(a) for a in all_q)
        if n_corr < 4:
            ...  # existing <4 early-return, with n_correspondences=n_corr
        pts2d = np.concatenate(all_q).astype(np.float32)
        pts3d_matched = np.concatenate(all_3d).astype(np.float32)
        pts2d_ref = np.concatenate(all_ref).astype(np.float32)
        ref_frame_indices = np.concatenate(all_frame)
```

   PnP block onward unchanged.
3. **`from_feedforward`:** pass `world_points=result.world_points` (raise `ValueError("FeedforwardResult has no world_points — re-save zarr or load with load_world_points=True")` when `None`); drop `pts3d=`/`intrinsics=` forwarding; same for the `load_index` cache-hit call.
4. **`save_index`/`load_index`:** remove the assignment rebuild in `load_index`; its geometry params become `world_points`, `extrinsics`. Cached arrays (keypoints/descriptors/scores CSR) unchanged.
5. Remove `radius` from every internal mention (docstrings, kwargs filter `("config", "radius")` → `("config",)`).

- [ ] **Step 4: Run new test — PASS.** Then run `tests/localization/ -x -q`; mechanically update every `CameraLocalizer(pts3d=..., intrinsics=...)` call in tests to the new signature (tests own synthetic geometry — build `world_points` grids like the new test does). `test_decoupling_parity.py` compares against a stored baseline built under the old model — if it asserts old correspondence counts, update the baseline expectations, noting why in the test docstring.

- [ ] **Step 5: Commit** `feat(localization): 2D→3D via per-frame world_points depth lookup (pose_from_cluster parity)`

---

### Task 4: Wire load sites

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (~line 376), `collab_splats/webapp/routers/localize.py` (~line 115), `collab_splats/dashboard/pipeline.py` (~line 412)

- [ ] **Step 1:** At each `FeedforwardResult.load_zarr(...)` feeding a `CameraLocalizer`, add `load_world_points=True`. Confirm each `from_feedforward(...)` call needs no other change (it reads `result.world_points` itself).
- [ ] **Step 2:** `grep -rn "radius" collab_splats/wrapper collab_splats/dashboard collab_splats/webapp configs/` — remove any localizer-`radius` plumbing/config keys found.
- [ ] **Step 3:** Run `tests/wrapper tests/dashboard tests/webapp -q` (whichever exist); fix fallout.
- [ ] **Step 4:** Dashboard touched → run smoke gate: `/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke` → must print `SMOKE PASS`.
- [ ] **Step 5: Commit** `fix(localization): load world_points at localizer build sites`

---

### Task 5: `xfeat-star` extractor

**Files:**
- Modify: `collab_splats/localization/extractors.py`, `collab_splats/localization/localizer.py` (cache: optional `scales`)
- Read first: `third_party/xfeat/modules/xfeat.py` — verify exact `detectAndComputeDense` / `batch_match` / `refine_matches` signatures before coding (vendored, so read locally; do NOT trust this plan's recollection).
- Test: `tests/localization/test_extractors.py`

- [ ] **Step 1: Write failing test:**

```python
def test_xfeat_star_matches_shifted_image():
    from collab_splats.localization.extractors import BaseLocalExtractor, MatchResult

    rng = np.random.default_rng(1)
    img0 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    img1 = np.roll(img0, 6, axis=1)
    ex = BaseLocalExtractor.create("xfeat-star", top_k=2048)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    assert f0.scales is not None                      # dense path caches scales
    m = ex.match(f0, f1, img0.shape[:2])
    assert isinstance(m, MatchResult) and len(m) > 50
    dx = m.ref_px[:, 0] - m.query_px[:, 0]
    assert abs(np.median(dx) - 6) < 1.5               # recovers the shift
```

- [ ] **Step 2: Run — FAIL** (`'xfeat-star' not registered`).

- [ ] **Step 3: Implement.**
   1. `LocalFeatures` gains `scales: torch.Tensor | None = None  # (N,) — dense XFeat* only`.
   2. New extractor, registered `"xfeat-star"`, mirroring how `match_xfeat_star` composes the vendored pieces (verify against the vendored source read in this task):

```python
@BaseLocalExtractor.register("xfeat-star")
class XFeatStarExtractor(BaseLocalExtractor):
    """XFeat* semi-dense matcher: cached dense features + pairwise match refinement.

    extract() caches detectAndComputeDense output per frame; match() runs XFeat's
    batch_match + refine_matches on the cached features and returns refined
    subpixel pixel pairs (accelerated_features xfeat_matching notebook flow).
    """

    def __init__(self, top_k: int = 4096, device: str | None = None):
        self._top_k = top_k
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._xfeat = XFeat()

    def extract(self, image: np.ndarray) -> LocalFeatures:
        img_t = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        out = self._xfeat.detectAndComputeDense(img_t, top_k=self._top_k)
        return LocalFeatures(
            keypoints=out["keypoints"][0].cpu(),
            descriptors=out["descriptors"][0].cpu(),
            scales=out["scales"][0].cpu(),
        )

    def match(self, query, db, image_hw) -> MatchResult:
        d0 = {"keypoints": query.keypoints.to(self._device),
              "descriptors": query.descriptors.to(self._device),
              "scales": query.scales.to(self._device)}
        d1 = {"keypoints": db.keypoints.to(self._device),
              "descriptors": db.descriptors.to(self._device),
              "scales": db.scales.to(self._device)}
        idxs = self._xfeat.batch_match(d0["descriptors"][None], d1["descriptors"][None], min_cossim=-1)
        matches = self._xfeat.refine_matches(d0, d1, matches=idxs, batch_idx=0)  # (K,4) x1,y1,x2,y2
        m = matches.cpu().numpy().astype(np.float32)
        if len(m) == 0:
            return _empty_match()
        return MatchResult(query_px=m[:, :2], ref_px=m[:, 2:])
```

   Adjust arg shapes to whatever the vendored source actually expects (batch dims differ between versions) — the test is the referee.
   3. Zarr cache: in `save_index`/`load_index`, persist/restore `scales` exactly like `scores` (optional array, present when the extractor produced it).

- [ ] **Step 4: Run test — PASS** (GPU box). Run full `tests/localization/ -q`.

- [ ] **Step 5: Commit** `feat(localization): xfeat-star semi-dense extractor`

---

### Task 6: Decouple viz from `LocalizationResult`

**Files:**
- Modify: `collab_splats/localization/viz.py`, `collab_splats/dashboard/localize.py` (~line 750)
- Notebook: `docs/source/tutorials/07_localization/localization.ipynb` (~cell with `plot_correspondences`)
- Test: `tests/localization/test_viz_correspondences.py`, `test_viz_distribution.py`, `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Write failing test:**

```python
def test_plot_correspondences_plain_arrays():
    """Array-based signature — no LocalizationResult required."""
    from collab_splats.localization.viz import plot_correspondences

    rng = np.random.default_rng(2)
    q = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    r = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    q_px = rng.uniform(0, 79, (10, 2)).astype(np.float32)
    r_px = rng.uniform(0, 79, (10, 2)).astype(np.float32)
    fig = plot_correspondences(q, r, q_px, r_px, inlier_mask=np.arange(10) % 2 == 0, show=False)
    assert fig is not None


def test_correspondences_for_ref_duck_typed():
    from types import SimpleNamespace
    from collab_splats.localization.viz import correspondences_for_ref

    loc = SimpleNamespace(
        pts2d=np.zeros((4, 2), np.float32), pts2d_ref=np.ones((4, 2), np.float32),
        ref_frame_indices=np.array([0, 1, 1, 0]), inlier_mask=np.array([True, False, True, True]),
    )
    q_px, r_px, mask = correspondences_for_ref(loc, 1)
    assert len(q_px) == 2 and mask.tolist() == [False, True]
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement.**
   1. Rewrite `plot_correspondences(query_image, ref_image, query_px, ref_px, inlier_mask=None, max_pairs=200, warp_corners=False, show=True)` — same drawing body, `loc.*` reads replaced by the array params; `inlier_mask=None` → all drawn as inliers; `warp_corners` homography uses the passed pixel arrays. Remove `ref_idx` (slicing moves to the adapter) and any `loc.pose` use.
   2. Add the adapter:

```python
def correspondences_for_ref(loc, ref_idx: int):
    """(query_px, ref_px, inlier_mask) for one reference frame, from any object
    exposing pts2d / pts2d_ref / ref_frame_indices / inlier_mask arrays."""
    sel = loc.ref_frame_indices == ref_idx
    mask = loc.inlier_mask[sel] if loc.inlier_mask is not None else None
    return loc.pts2d[sel], loc.pts2d_ref[sel], mask
```

   3. Audit `viz.py` for other `LocalizationResult`-typed helpers (e.g. the inlier-distribution plot exercised by `test_viz_distribution.py`) — convert each to plain-array params the same way; remove the `LocalizationResult` import from `viz.py` entirely.
   4. Update callers: `dashboard/localize.py:750` → `plot_correspondences(out.query_frame, ref_image, *correspondences_for_ref(loc, int(ref)), max_pairs=config.max_pairs, show=False)`; nb07 cell likewise (NotebookEdit, don't execute).
   5. Update the old-signature tests + the `test_localize_page.py` monkeypatch.

- [ ] **Step 4: Run the three viz/dashboard test files — PASS.** Smoke gate again (dashboard touched): must print `SMOKE PASS`.

- [ ] **Step 5: Commit** `refactor(localization): array-based viz, LocalizationResult only at localizer boundary`

---

### Task 7: Full-suite + real-data verification

- [ ] **Step 1:** `/opt/venv/reconstruction/bin/python -m pytest tests/ -q` — green modulo `docs/known-test-failures.md`.
- [ ] **Step 2:** Real regression (the original bug), in tmux (memory cap). Script against the tutorial scene: load tutorial `feedforward.zarr` (`data/tutorial/`, RECON from nb02) with `load_world_points=True`, build `CameraLocalizer.from_feedforward` per extractor ∈ {disk, xfeat, xfeat-star}, localize `docs/source/tutorials/07_localization/ref_image.jpg`. Record `n_correspondences` / `n_inliers` / `pose is not None` per extractor. Success: pose recovered where all previously failed; report the numbers to the user (no hedging).
- [ ] **Step 3:** `black . && isort .`; `git status` clean-up; final commit if formatting changed.
- [ ] **Step 4:** Update `CLAUDE.md` In-Flight list + memory file per outcome.
