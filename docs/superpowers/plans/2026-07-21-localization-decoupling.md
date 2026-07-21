# Localization Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `frames_zarr` artifact coupling from the general localization + viz API; feed pixels as core `np.ndarray` objects built at the pipeline boundary.

**Architecture:** The localizer and viz operate only over core objects — an iterable of RGB arrays (`images`) plus stable string labels (`ids`). Pipeline artifacts (`frames.zarr` / `FrameStore`) are adapted to those core objects at the *caller* (boundary glue in wrapper/dashboard), never threaded into the general functions. Best-reference selection moves onto `LocalizationResult` as a pure property. Semantics 2D-feature extraction routes through the existing `extract_and_cache_from_zarr` instead of a temp-JPG bridge.

**Tech Stack:** Python 3.11, numpy, zarr 3.x, pycolmap, PIL (`utils/image.open_image`), matplotlib, pytest.

---

## Environment

- **Worktree:** `/workspace/collab-splats/.worktrees/localization-decoupling` (branch `refactor/localization-decoupling`). All work happens here.
- **Python:** `/opt/venv/reconstruction/bin/python` (py3.11).
- **CRITICAL — editable install points at the MAIN repo.** To make the worktree's `collab_splats` win, prefix every python/pytest command with `PYTHONPATH=<worktree-root>`. Set once per shell:
  ```bash
  cd /workspace/collab-splats/.worktrees/localization-decoupling
  export WT=$(pwd)
  ```
  Then run tests as `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest ...`.
- **Verify resolution before starting:**
  ```bash
  PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
  # MUST print a path under .worktrees/localization-decoupling
  ```

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `collab_splats/localization/localizer.py` | `LocalizationResult` dataclass + `CameraLocalizer` | Add `ranked_ref_frames` property; `__init__`/`update_index`/`from_feedforward` take `(images, ids)`, drop `frames_zarr` + IO |
| `collab_splats/localization/viz.py` | `plot_correspondences` | Take `ref_image: np.ndarray` + required `ref_idx`; drop `image_paths`/`frames_zarr`/selection/IO |
| `collab_splats/wrapper/reconstructor.py` | boundary glue | `_build_localization_db` builds `(images, ids)`; `_extract_2d_features` routes through `_from_zarr` |
| `collab_splats/dashboard/pipeline.py` | boundary glue | `_build_localizer` builds `(images, ids)` from store |
| `collab_splats/dashboard/localize.py` | boundary glue | `_build_result_figures` resolves ref arrays, uses `ranked_ref_frames` |
| `docs/source/tutorials/07_localization/localization.ipynb` | tutorial | Update `plot_correspondences` call to new API |
| `tests/localization/*`, `tests/dashboard/*`, `tests/wrapper/*`, `tests/semantics/*` | tests | Update to decoupled API + add property/parity tests |

## Conventions for this plan

- **RGB from a path:** `np.asarray(open_image(p).convert("RGB"))` — import `from collab_splats.utils.image import open_image`. Never `cv2.imread` (BGR trap).
- **RGB from the store:** `store.image_by_frame_idx(fi)` (already RGB) or `store.image(row)`.
- **ids convention:** `f"frame_{fi:06d}.jpg"` for store-sourced frames (matches existing reconstruction naming); `p.name` for dir-sourced.
- The localizer keeps its internal attribute name `self._image_paths` (now holding **string ids**) to minimise churn — only constructor params and the pixel-read source change.

---

## Task 0: Baseline

**Files:** none (verification only).

- [ ] **Step 1: Confirm worktree import resolution**

Run:
```bash
cd /workspace/collab-splats/.worktrees/localization-decoupling && export WT=$(pwd)
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
```
Expected: path under `.worktrees/localization-decoupling`.

- [ ] **Step 2: Record baseline test state**

Run:
```bash
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization tests/preproc/test_viz.py tests/dashboard tests/wrapper tests/semantics -q 2>&1 | tail -25
```
Expected: note the pass/fail/xfail counts. Cross-reference `docs/known-test-failures.md` for pre-existing failures (e.g. 2 pre-existing `run_pipeline` fails). Any NEW failure introduced later must be judged against this baseline.

- [ ] **Step 3: Confirm the coupling exists (pre-condition for the sanity gate)**

Run: `grep -rn "frames_zarr" collab_splats/localization`
Expected: non-empty (viz.py + localizer.py hits). This must become EMPTY by Task 6.

---

## Task 1: `LocalizationResult.ranked_ref_frames` property

**Files:**
- Modify: `collab_splats/localization/localizer.py` (dataclass at `:26-47`)
- Test: `tests/localization/test_localization_result.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/localization/test_localization_result.py`:
```python
"""LocalizationResult.ranked_ref_frames: pure inlier-count ranking over result fields."""

import numpy as np

from collab_splats.localization.localizer import LocalizationResult


def _make(ref_frame_indices, inlier_mask):
    return LocalizationResult(
        pose=None,
        n_inliers=int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        n_correspondences=len(ref_frame_indices) if ref_frame_indices is not None else 0,
        pts2d=None,
        pts2d_ref=None,
        inlier_mask=inlier_mask,
        ref_frame_indices=ref_frame_indices,
    )


def test_ranked_orders_by_inlier_count_desc():
    # frame 2 has 3 inliers, frame 0 has 2, frame 1 has 1
    ref = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    mask = np.array([True, True, True, True, True, True])
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [2, 0, 1]


def test_ranked_excludes_zero_inlier_frames():
    ref = np.array([0, 1, 1], dtype=np.int32)
    mask = np.array([False, True, True])  # frame 0 has 0 inliers
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [1]


def test_ranked_empty_when_no_result():
    assert _make(None, None).ranked_ref_frames == []
    ref = np.array([0, 1], dtype=np.int32)
    assert _make(ref, None).ranked_ref_frames == []
```

Check the exact `LocalizationResult` field list first (localizer.py `:42-47`) and match the constructor kwargs — the dataclass has `pose, n_inliers, n_correspondences, pts2d, pts2d_ref, inlier_mask, ref_frame_indices`. Adjust `_make` if a field name differs.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localization_result.py -q`
Expected: FAIL — `AttributeError: 'LocalizationResult' object has no attribute 'ranked_ref_frames'`.

- [ ] **Step 3: Add the property**

In `collab_splats/localization/localizer.py`, inside the `LocalizationResult` dataclass (after the fields, before the closing of the class ~`:47`), add:
```python
    @property
    def ranked_ref_frames(self) -> list[int]:
        """Reference-frame indices ordered by inlier-match count, most first.

        Excludes frames with zero inliers. Empty when pose failed / no inliers.
        Slice for the n best (``[:n]``) or take ``[0]`` for the single best.
        """
        if self.inlier_mask is None or self.ref_frame_indices is None:
            return []
        counts = np.bincount(self.ref_frame_indices[self.inlier_mask].astype(np.intp))
        order = np.argsort(counts)[::-1]
        return [int(i) for i in order if counts[i] > 0]
```
(`np` is already imported at module top.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localization_result.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization/test_localization_result.py
git commit -m "feat(localization): add LocalizationResult.ranked_ref_frames property"
```

---

## Task 2: `plot_correspondences` — array-based ref, no IO

**Files:**
- Modify: `collab_splats/localization/viz.py`
- Test: `tests/localization/test_viz_correspondences.py`, `tests/localization/test_viz_distribution.py`

- [ ] **Step 1: Update the tests to the new API (they will fail)**

In `tests/localization/test_viz_correspondences.py`, the two tests currently build image *paths* on disk and pass `image_paths`. Change them to pass a single `ref_image` array + required `ref_idx`. Read the file first; replace each `plot_correspondences(...)` call so it matches the new signature:
```python
plot_correspondences(
    loc,
    query_image,          # np.ndarray HxWx3 RGB
    ref_image,            # np.ndarray HxWx3 RGB  (was image_paths list)
    ref_idx=<the ref index these fixtures exercise>,
    show=False,
)
```
Build `ref_image` as a synthetic array of the intended reference resolution (the test's point is resolution-mismatch handling), e.g. `ref_image = np.zeros((H, W, 3), np.uint8)`. Remove any now-unused tmp-file/path setup.

In `tests/localization/test_viz_distribution.py`:
- The `ref_idx=2` call (`:99`): pass `ref_image=<array>`, `ref_idx=2`.
- The no-`ref_idx` call (`:109`): the function no longer auto-selects. Replace with:
  ```python
  ranked = loc.ranked_ref_frames
  ri = ranked[0]
  fig = plot_correspondences(loc, query, ref_image, ref_idx=ri, show=False)
  ```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_viz_correspondences.py tests/localization/test_viz_distribution.py -q`
Expected: FAIL — `plot_correspondences() got an unexpected keyword argument 'image_paths'` or missing `ref_image` / unexpected `frames_zarr`.

- [ ] **Step 3: Rewrite `plot_correspondences`**

In `collab_splats/localization/viz.py`:
1. Delete imports `import cv2` **only if** no other function in the file uses it — `plot_correspondences` still uses `cv2.findHomography`/`cv2.perspectiveTransform`/`cv2.line` for `warp_corners`, so **keep `cv2`**. Remove `from collab_splats.preproc.frame_store import FrameStore` and the `from pathlib import Path` import **only if** unused elsewhere in the file (grep first).
2. New signature (drop `image_paths`, `frames_zarr`; add `ref_image`; make `ref_idx` required):
```python
def plot_correspondences(
    loc: LocalizationResult,
    query_image: np.ndarray,
    ref_image: np.ndarray,
    ref_idx: int,
    max_pairs: int = 200,
    warp_corners: bool = False,
    show: bool = True,
) -> "plt.Figure | None":
```
3. Update the docstring: `ref_image` = HxWx3 uint8 RGB reference frame (caller-resolved); `ref_idx` = its index in the reference set (caller picks it, e.g. `loc.ranked_ref_frames[0]`). Remove the `image_paths`/`frames_zarr` lines.
4. Replace the selection block (`:59-70`, the `if ref_idx is not None ... else ... bincount ...`) with a direct use of the passed `ref_idx`:
```python
    if loc.pose is None or loc.inlier_mask is None or loc.pts2d_ref is None or loc.ref_frame_indices is None:
        logger.warning("plot_correspondences: no valid localization result to plot")
        return None

    best_ref_idx = int(ref_idx)
    frame_mask = loc.ref_frame_indices == best_ref_idx
    if not frame_mask.any():
        logger.warning("plot_correspondences: no correspondences for frame %d", best_ref_idx)
        return None
```
5. Delete the whole pixel-read block (`:104-111`, the `if frames_zarr is not None: ... else: cv2.imread ...`). `ref_image` is already the array. Everything downstream that used the local `ref_image` variable is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_viz_correspondences.py tests/localization/test_viz_distribution.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/viz.py tests/localization/test_viz_correspondences.py tests/localization/test_viz_distribution.py
git commit -m "refactor(localization): plot_correspondences takes ref_image array, drops frames_zarr"
```

---

## Task 3: `CameraLocalizer.__init__` — (images, ids), no IO

**Files:**
- Modify: `collab_splats/localization/localizer.py` (`__init__` `:144-252`)
- Test: `tests/localization/` (find the test that constructs `CameraLocalizer(...)` directly)

- [ ] **Step 1: Find and update the constructor test**

Run: `grep -rn "CameraLocalizer(" tests/localization` to find direct-construction tests. For each, replace `image_paths=<paths>` (and any `frames_zarr=`) with:
```python
images=[np.asarray(open_image(p).convert("RGB")) for p in paths],  # or synthetic arrays
ids=[p.name for p in paths],
```
Add `from collab_splats.utils.image import open_image` if reading from paths, or build synthetic `images=[np.zeros((H, W, 3), np.uint8), ...]` with matching `ids=["frame_000000.jpg", ...]`. If a test asserts `localizer.image_paths == [...]`, update the expectation to the `ids` strings.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k "index or build or localizer"`
Expected: FAIL — unexpected keyword `image_paths`/`frames_zarr`, or missing `images`/`ids`.

- [ ] **Step 3: Rewrite `__init__`**

In `collab_splats/localization/localizer.py`:
1. Signature — replace `image_paths: list` and `frames_zarr: str | Path | None = None` params with:
```python
        images,                    # Iterable[np.ndarray] — RGB arrays, one per reference frame
        ids: list[str],            # stable per-frame labels, index-aligned with images
```
Keep `pts3d, extrinsics, intrinsics` first, then `images, ids`, then `extractor, radius, config, progress_callback` (drop `frames_zarr`).
2. Update the docstring block for `images`/`ids`; delete the `frames_zarr` doc paragraph.
3. Replace `self._image_paths: list[Path] = [Path(p) for p in image_paths]` with:
```python
        self._image_paths: list[str] = list(ids)
```
4. Delete the `store = FrameStore.open(frames_zarr) ...` line (`:205`).
5. Replace the build loop (`:208-236`) with iteration over `zip(images, ids)`:
```python
        self._frame_features: list[LocalFeatures] = []
        first_hw: tuple[int, int] | None = None
        total = len(ids)
        # tqdm only when no external progress_callback is wired
        if progress_callback is None:
            try:
                from tqdm.auto import tqdm as _tqdm
                _iter = _tqdm(zip(images, ids), total=total, desc="Indexing frames", unit="frame", leave=False)
            except ImportError:
                _iter = zip(images, ids)
        else:
            _iter = zip(images, ids)
        for rgb, fid in _iter:
            if first_hw is None:
                first_hw = (rgb.shape[0], rgb.shape[1])
            feats = self._extractor.extract(rgb)
            self._frame_features.append(feats)
            if progress_callback is not None:
                progress_callback(len(self._frame_features) - 1, total)
            logger.debug("  frame %s: %d keypoints", fid, len(feats.keypoints))
```
6. Update the `len(image_paths)` reference in the info log (`:196`) to `len(ids)`.
7. At module top: remove `import cv2` and `from collab_splats.preproc.frame_store import FrameStore` **iff** no longer referenced anywhere else in `localizer.py` (grep `cv2\.` and `FrameStore` in the file after editing; `FrameStore.frame_idx_from_path` usages must be gone).

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k "index or build or localizer"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization
git commit -m "refactor(localization): CameraLocalizer.__init__ takes (images, ids), no frames_zarr"
```

---

## Task 4: `CameraLocalizer.update_index` — (new_images, new_ids)

**Files:**
- Modify: `collab_splats/localization/localizer.py` (`update_index` `:487-524`)
- Test: `tests/localization/` (find any `update_index(` test)

- [ ] **Step 1: Update the test (if one exists)**

Run: `grep -rn "update_index(" tests/localization`. If found, change `new_image_paths=<...>` + `frames_zarr=<...>` to `new_images=[<arrays>]`, `new_ids=[<strings>]`. If no direct test exists, add a minimal one that appends one synthetic frame and asserts `localizer.image_paths[-1] == new_ids[-1]` and `len(localizer._frame_features)` grew by 1.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k update`
Expected: FAIL on the new/updated test.

- [ ] **Step 3: Rewrite `update_index`**

In `collab_splats/localization/localizer.py`:
1. Signature: replace `new_image_paths: list` + `frames_zarr` with `new_images` (Iterable[np.ndarray]) + `new_ids: list[str]`. Update docstring; drop the `frames_zarr` line.
2. Replace the extraction loop (`:503-518`) — delete the `store = FrameStore.open(...)` and the store-vs-imread branch:
```python
        new_features: list[LocalFeatures] = []
        for i, (rgb, fid) in enumerate(zip(new_images, new_ids)):
            feats = self._extractor.extract(rgb)
            new_features.append(feats)
            if progress_callback is not None:
                progress_callback(i, len(new_ids))
            logger.debug("update_index: frame %s: %d kpts", fid, len(feats.keypoints))
```
3. In the in-memory update loop (`:521-524`) replace `new_image_paths` with `new_ids` and append the id string directly:
```python
        for fid, feats in zip(new_ids, new_features):
            self._frame_features.append(feats)
            self._frame_sources.append("reconstruction")
            self._image_paths.append(fid)
```
4. In the zarr-attr update (`:538-540`) replace `new_image_paths` with `new_ids`:
```python
        existing_paths.extend([str(f) for f in new_ids])
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k update`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization
git commit -m "refactor(localization): update_index takes (new_images, new_ids), no frames_zarr"
```

---

## Task 5: `CameraLocalizer.from_feedforward` — boundary adapter, drop frames_zarr

**Files:**
- Modify: `collab_splats/localization/localizer.py` (`from_feedforward` `:738-824`)
- Test: `tests/localization/` (find `from_feedforward(` tests)

- [ ] **Step 1: Update tests to new signature**

Run: `grep -rn "from_feedforward(" tests/localization tests/dashboard tests/wrapper`. For localization unit tests that hit the cache-**miss** path, add `images=<iterable of arrays>` and `ids=<list[str]>` args and drop any `frames_zarr=`. Cache-**hit** tests can pass `images=iter(())`, `ids=[]` (never consumed on a hit) — but simplest: pass the real aligned pair. Adjust assertions that referenced `frames_zarr`.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k feedforward`
Expected: FAIL — unexpected `frames_zarr` / missing `images`,`ids`.

- [ ] **Step 3: Rewrite `from_feedforward`**

In `collab_splats/localization/localizer.py`:
1. Signature: drop `frames_zarr=None`; add `images` and `ids` params (both `= None` defaults so cache-hit callers may omit):
```python
    @classmethod
    def from_feedforward(
        cls,
        result,
        images=None,          # Iterable[np.ndarray]; required only on a cache MISS
        ids=None,             # list[str]; required only on a cache MISS
        extractor=None,
        progress_callback=None,
        zarr_path=None,
        extractor_name=None,
        **kwargs,
    ) -> "CameraLocalizer":
```
2. Update the docstring: `images`/`ids` are the caller-built pixel source + labels used only when the zarr cache misses; the boundary caller aligns them. Remove the `frames_zarr` paragraph.
3. The cache-hit path (`:781-799`) is unchanged (it uses `result` + zarr only).
4. In the staleness check (`:787-789`), compare cached labels against `ids` when provided, else fall back to `result.image_paths`:
```python
                    cached_paths = [str(p) for p in store[rec_key].attrs["image_paths"]]
                    expected = [str(x) for x in (ids if ids is not None else result.image_paths)]
                    if cached_paths != expected:
                        logger.warning("CameraLocalizer: cached image_paths differ from expected — cache may be stale")
```
5. The cache-miss construction (`:806-815`) — replace `image_paths=result.image_paths` + `frames_zarr=frames_zarr` with `images`/`ids`, guarding for the miss path:
```python
        if images is None or ids is None:
            raise ValueError("from_feedforward: cache miss requires images and ids (caller must build them)")
        localizer = cls(
            pts3d=result.points,
            extrinsics=result.extrinsics,
            intrinsics=result.intrinsics,
            images=images,
            ids=ids,
            extractor=extractor_inst,
            progress_callback=progress_callback,
            **kwargs,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -k feedforward`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization
git commit -m "refactor(localization): from_feedforward takes caller-built (images, ids), drops frames_zarr"
```

---

## Task 6: Boundary callers build (images, ids) — and the sanity gate

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_build_localization_db` `:309-336`)
- Modify: `collab_splats/dashboard/pipeline.py` (`_build_localizer` `:370-410`)
- Modify: `collab_splats/dashboard/localize.py` (`_build_result_figures` `:703-753`)
- Modify: `docs/source/tutorials/07_localization/localization.ipynb` (the `plot_correspondences` cell)
- Test: `tests/dashboard/test_run_localization.py`, `tests/dashboard/test_pipeline.py`, `tests/dashboard/test_localize_page.py`, `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: `_build_localization_db` (wrapper)**

In `collab_splats/wrapper/reconstructor.py::_build_localization_db`, build the aligned pair from the store and pass to `from_feedforward` (keep the `frames_zarr: Path` param — it is glue here). Replace the `from_feedforward(...)` call:
```python
    from collab_splats.preproc.frame_store import FrameStore

    ff = FeedforwardResult.load_zarr(feedforward_zarr, load_images=True)
    extractor = BaseLocalExtractor.get(extractor_name)()

    # Boundary adapter: canonical store → (images, ids) core objects. Lazy genexpr means
    # zero reads on a cache hit; one partial-read per frame on a miss.
    store = FrameStore.open(frames_zarr)
    frame_indices = store.frame_indices()
    images = (store.image_by_frame_idx(fi) for fi in frame_indices)
    ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]

    CameraLocalizer.from_feedforward(
        ff,
        images=images,
        ids=ids,
        extractor=extractor,
        extractor_name=extractor_name,
        zarr_path=feedforward_zarr,
        radius=radius,
    )
```
Ensure `FrameStore` is imported (it is already used elsewhere in the module — check the top-of-file imports; if the inline heavy-dep block is the pattern, follow it).

- [ ] **Step 2: `_build_localizer` (dashboard pipeline)**

In `collab_splats/dashboard/pipeline.py::_build_localizer`, keep the `frames_zarr` param (glue). Before the `from_feedforward` call, build `(images, ids)` from the store when `frames_zarr` is present; else from `result.image_paths`:
```python
    from collab_splats.preproc.frame_store import FrameStore
    from collab_splats.utils.image import open_image

    if frames_zarr is not None:
        store = FrameStore.open(frames_zarr)
        frame_indices = store.frame_indices()
        images = (store.image_by_frame_idx(fi) for fi in frame_indices)
        ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
    else:
        paths = [Path(p) for p in result.image_paths]
        images = (np.asarray(open_image(p).convert("RGB")) for p in paths)
        ids = [p.name for p in paths]

    localizer = CameraLocalizer.from_feedforward(
        result,
        images=images,
        ids=ids,
        extractor=extractor,
        extractor_name=config.extractor,
        zarr_path=zarr_path,
        progress_callback=on_progress,
    )
```
Confirm `np` and `Path` are imported at the top of `pipeline.py` (add if missing).

- [ ] **Step 3: `_build_result_figures` (dashboard localize)**

In `collab_splats/dashboard/localize.py::_build_result_figures`, replace the top-k loop (`:733-751`) to (a) use `loc.ranked_ref_frames`, (b) resolve each ref frame's pixels to an array, (c) pass `ref_image`/`ref_idx` to `plot_correspondences`:
```python
        from collab_splats.preproc.frame_store import FrameStore
        from collab_splats.utils.image import open_image

        store = FrameStore.open(frames_zarr) if frames_zarr is not None else None

        match_figs = []
        for ref in loc.ranked_ref_frames[: config.top_k_viz]:
            if not Path(out.ref_image_paths[ref]).exists() and not (
                store is not None and ref < len(out.frame_sources) and out.frame_sources[ref] == "reconstruction"
            ):
                continue
            is_reconstruction = ref < len(out.frame_sources) and out.frame_sources[ref] == "reconstruction"
            if is_reconstruction and store is not None:
                # reconstruction ref → canonical store (lossless RGB)
                fi = FrameStore.frame_idx_from_path(out.ref_image_paths[ref])
                ref_image = store.image_by_frame_idx(fi)
            else:
                # localized/ frame → disk (never written to frames.zarr)
                ref_image = np.asarray(open_image(out.ref_image_paths[ref]).convert("RGB"))
            mfig = plot_correspondences(
                loc,
                out.query_frame,
                ref_image,
                ref_idx=int(ref),
                max_pairs=config.max_pairs,
                show=False,
            )
            if mfig is not None:
                match_figs.append(mfig)
```
Delete the now-unused `counts = np.bincount(...)` / `top = np.argsort(...)` lines above the loop. Keep `n_frames`, `dist_fig`, `stats_html`, and the final `return` dict unchanged. Confirm `np` and `Path` are imported in `localize.py`.

- [ ] **Step 4: Tutorial notebook cell**

In `docs/source/tutorials/07_localization/localization.ipynb`, the cell `plot_correspondences(loc, query_image, result_trimmed.image_paths, frames_zarr=FRAMES_ZARR)` becomes (glue resolves the ref array):
```python
from collab_splats.utils.image import open_image
ranked = loc.ranked_ref_frames
if ranked:
    ri = ranked[0]
    fi = FrameStore.frame_idx_from_path(result_trimmed.image_paths[ri])
    ref_image = FrameStore.open(FRAMES_ZARR).image_by_frame_idx(fi)
    plot_correspondences(loc, query_image, ref_image, ref_idx=ri)
```
Edit the notebook JSON `source` array for that cell (keep it a single logical cell). Do not execute the notebook here (GPU/heavy) — the docs-site build owns execution.

- [ ] **Step 5: Update dashboard/wrapper tests**

Run: `grep -rn "frames_zarr\|plot_correspondences\|from_feedforward\|update_index\|image_paths=" tests/dashboard tests/wrapper`. Update mocks/assertions:
- `tests/dashboard/test_localize_page.py:346` patches `plot_correspondences` — ensure the patched call assertion (if any) matches the new positional/kw args (`ref_image`, `ref_idx`), not `image_paths`/`frames_zarr`.
- `tests/dashboard/test_run_localization.py`, `test_pipeline.py`: any `frames_zarr=` passed to `_build_localizer`/`from_feedforward` still valid as glue param on `_build_localizer`; assertions on `from_feedforward` kwargs must drop `frames_zarr` and expect `images`/`ids`.
- `tests/wrapper/test_reconstructor.py`: `_build_localization_db` still takes `frames_zarr` — keep; assert `from_feedforward` is called with `images`/`ids` if the test inspects the call.

- [ ] **Step 6: Run the boundary + gate**

Run:
```bash
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/dashboard tests/wrapper -q
grep -rn "frames_zarr" collab_splats/localization
```
Expected: tests PASS; grep prints **nothing** (localization module fully decoupled). If grep is non-empty, a leaked reference remains — fix before committing.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/localize.py docs/source/tutorials/07_localization/localization.ipynb tests/dashboard tests/wrapper
git commit -m "refactor(dashboard,wrapper): build (images, ids) at boundary; localization decoupled from frames.zarr"
```

---

## Task 7: Track B — semantics routes through extract_and_cache_from_zarr

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_extract_2d_features` `:177-195`)
- Test: `tests/wrapper/` (find `_extract_2d_features` / feature-extraction test), `tests/semantics/features/test_extract_from_zarr.py`

- [ ] **Step 1: Update/confirm the test**

Run: `grep -rn "_extract_2d_features\|extract_and_cache" tests/wrapper tests/semantics`. Update any test that asserts the temp-export bridge behaviour of `_extract_2d_features` to expect a direct `extract_and_cache_from_zarr(frames_zarr, cache_dir)` call (patch `extract_and_cache_from_zarr` and assert it is invoked with the frames zarr path; assert no `TemporaryDirectory`/`export` is used). `tests/semantics/features/test_extract_from_zarr.py` already covers the `_from_zarr` method — no change expected there.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/wrapper -q -k "feature or extract"`
Expected: FAIL on the updated assertion.

- [ ] **Step 3: Rewrite `_extract_2d_features`**

Replace the body of `collab_splats/wrapper/reconstructor.py::_extract_2d_features`:
```python
def _extract_2d_features(
    extractor_name: str,
    frames_zarr: Path,
    features_dir: Path,
) -> Path:
    """Extract 2D features for all frames straight from the canonical store.

    Delegates to BaseFeatureExtractor.extract_and_cache_from_zarr, which iterates the
    frames zarr lazily (one chunk at a time) and writes cache_dir/{name}.zarr — no temp
    JPG export, no full-RAM load.
    """
    extractor = _get_extractor(extractor_name)
    cache_dir = features_dir / extractor_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return extractor.extract_and_cache_from_zarr(frames_zarr, cache_dir)
```
Remove the now-unused `tempfile` import from the module **iff** nothing else in `reconstructor.py` uses it (grep `tempfile` in the file — note `_reconstruct`/preprocess bridges may still use it; keep if so).

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/wrapper tests/semantics -q -k "feature or extract or zarr"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py tests/wrapper
git commit -m "refactor(wrapper): _extract_2d_features reads frames.zarr directly, drops temp-JPG bridge"
```

---

## Task 8: Parity equivalence check (localization is parity-sensitive)

**Files:**
- Test: `tests/localization/test_decoupling_parity.py` (create)

- [ ] **Step 1: Write the parity test**

Localization output must be pixel-for-pixel stable across the refactor. This test asserts that building the local-features index via the new `(images, ids)` path produces the **same keypoints/descriptors** as extracting each frame directly — i.e. the decoupling introduced no accidental colour-swap or ordering change.

Create `tests/localization/test_decoupling_parity.py`:
```python
"""Decoupling parity: (images, ids) index build == direct per-frame extraction."""

import numpy as np
import torch

from collab_splats.localization.extractors import DiskExtractor
from collab_splats.localization.localizer import CameraLocalizer


def test_index_features_match_direct_extraction():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(3)]
    ids = [f"frame_{i:06d}.jpg" for i in range(3)]
    extractor = DiskExtractor()

    # Reference: extract each frame directly (the pure operation).
    direct = [extractor.extract(f) for f in frames]

    # Under test: build the index via the decoupled (images, ids) path.
    loc = CameraLocalizer(
        pts3d=np.zeros((1, 3), np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
        images=iter(frames),
        ids=ids,
        extractor=extractor,
    )

    assert len(loc._frame_features) == 3
    for got, want in zip(loc._frame_features, direct):
        assert torch.equal(got.keypoints, want.keypoints)
        assert torch.equal(got.descriptors, want.descriptors)
    assert loc.image_paths == ids
```
If `DiskExtractor` requires a GPU and the CI box lacks one, mark the module `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="extractor needs CUDA")` and record it in `docs/known-test-failures.md`.

- [ ] **Step 2: Run the parity test**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_decoupling_parity.py -q`
Expected: PASS (or SKIP on a CUDA-less box — then run once on a GPU box before merge).

- [ ] **Step 3: Commit**

```bash
git add tests/localization/test_decoupling_parity.py
git commit -m "test(localization): parity — decoupled index build matches direct extraction"
```

---

## Task 9: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Full targeted suite**

Run:
```bash
PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/localization tests/preproc/test_viz.py tests/dashboard tests/wrapper tests/semantics -q 2>&1 | tail -30
```
Expected: no NEW failures vs the Task 0 baseline. Investigate any regression.

- [ ] **Step 2: Dashboard smoke gate (mandatory pre-commit for dashboard changes)**

Run: `PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke`
Expected: prints `SMOKE PASS`.

- [ ] **Step 3: Final sanity gate**

Run: `grep -rn "frames_zarr" collab_splats/localization`
Expected: EMPTY.

- [ ] **Step 4: Format**

Run: `cd $WT && black collab_splats/localization collab_splats/wrapper collab_splats/dashboard tests && isort collab_splats/localization collab_splats/wrapper collab_splats/dashboard tests`
Then re-run Step 1 to confirm formatting didn't break anything. Commit any formatting churn:
```bash
git add -A && git commit -m "style: black + isort"
```

---

## Task 10: Finish the branch

- [ ] **Step 1: Review the full diff**

Run: `git log --oneline refactor/cu121-uv-migration..HEAD` and `git diff refactor/cu121-uv-migration...HEAD --stat`
Confirm: no changes to `collab_splats/geometry/loop_closure/merge.py` (concurrent-session file); no reintroduced `images/` dir; `frames_zarr` present only in wrapper/dashboard glue.

- [ ] **Step 2: Use finishing-a-development-branch skill**

Invoke superpowers:finishing-a-development-branch to merge `refactor/localization-decoupling` back into `refactor/cu121-uv-migration` and clean up the worktree.

---

## Self-Review Notes (author)

- **Spec coverage:** §A1 __init__ → Task 3; §A2 update_index → Task 4; §A3 from_feedforward → Task 5; §A4 viz → Task 2 (+ ranked_ref_frames Task 1); boundary callers → Task 6; sanity gate → Tasks 6/9; Track B → Task 7; parity → Task 8; tests → each task; smoke → Task 9. All spec sections mapped.
- **Deviations from spec (intentional, discussed with user):** (1) internal attr kept as `self._image_paths` holding string ids (spec said rename to `_ids`) — lower churn, property/save/load untouched. (2) Best-ref selection became `LocalizationResult.ranked_ref_frames` property (not mentioned in spec) — forced by decoupling: caller must know the ref index before fetching its pixels; property also collapses the dashboard's duplicate `bincount`. Both keep the "no new classes / no IO helper" constraints (pure property, no new module).
- **Types consistent:** `images` = Iterable[np.ndarray]; `ids` = list[str]; `ranked_ref_frames` = list[int]; `plot_correspondences(..., ref_image: np.ndarray, ref_idx: int)` used identically in Task 2 definition and Task 6 callers.
