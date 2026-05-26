# Incremental Bundle Adjustment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `BundleAdjustment` with a growing-window incremental mode and a zarr track cache so repeated runs skip expensive VGGSfM extraction.

**Architecture:** Add `add_size` and `tracks_cache_dir` to `BundleAdjustmentConfig`; `refine()` dispatches to `_refine_allonce()` (current behaviour) or `_refine_incremental()` (growing-window loop) based on `add_size`; tracks cached to `{tracks_cache_dir}/tracks.zarr` with a SHA-256 hash key for invalidation.

**Tech Stack:** Python 3.11, numpy, zarr v3 + BloscCodec, `unittest.mock.patch`, `/opt/conda/envs/nerfstudio/bin/python`

---

## File Map

| File | Action | What changes |
|---|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | Modify | Config fields; `_load_or_extract_tracks()`; `_refine_allonce()`; `_refine_incremental()`; `refine()` restructure; `_last_loss_history` type |
| `evals/eval_gt.py` | Modify | `incremental_ba-N` condition in `_validate_condition` + `_make_creator` |
| `evals/eval_suite.sh` | Modify | Add `incremental_ba-5` to condition sweep |
| `tests/pointcloud/test_bundle_adjustment.py` | Modify | 5 new tests + update 2 stale loss-history tests |

---

## Task 0: Pre-implementation eval gate

**Files:** (none modified — eval only)

- [ ] **Step 1: Run current BA eval on chess seq-01**

```bash
cd /workspace/collab-splats
tmux new-session -d -s ba_eval "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir data/7scenes/chess/chess/seq-01 \
    --output_dir evals/results/chess_seq01_baseline_check \
    --max_frames 50 \
    --conditions baseline ba 2>&1 | tee /tmp/ba_eval_precheck.log"
tmux attach -t ba_eval
```

- [ ] **Step 2: Record results**

After completion, note the ATE RMSE for `baseline` and `ba` from the log. This is the target to beat. Record in a comment at the top of the plan or in a scratch note.

---

## Task 1: Config fields + imports

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/pointcloud/test_bundle_adjustment.py` inside the "Tests for BundleAdjustment class" section:

```python
def test_ba_config_new_fields_default():
    """BundleAdjustmentConfig has add_size=0 and tracks_cache_dir=None by default."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustmentConfig
    cfg = BundleAdjustmentConfig()
    assert cfg.add_size == 0
    assert cfg.tracks_cache_dir is None
```

- [ ] **Step 2: Run to confirm it fails**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_ba_config_new_fields_default -v
```

Expected: `FAILED` — `BundleAdjustmentConfig has no attribute 'add_size'`

- [ ] **Step 3: Add fields to config and imports**

In `collab_splats/pointcloud/bundle_adjustment.py`, add imports at the top (after existing imports):

```python
import hashlib
import json
import logging
import shutil
from pathlib import Path

import zarr
from zarr.codecs import BloscCodec
```

Add `logger` at module level (after `__all__`):

```python
logger = logging.getLogger(__name__)
```

In `BundleAdjustmentConfig`, add after `capture_loss_history`:

```python
add_size: int = 0                      # 0 or >= N → all-at-once; 1..N-1 → incremental growing-window BA
tracks_cache_dir: Path | None = None   # zarr cache dir for tracks; None = always extract
```

Also update `__init__` type annotation:

```python
# Before:
self._last_loss_history: list[float] = []
# After:
self._last_loss_history: list[list[float]] = []
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_ba_config_new_fields_default tests/pointcloud/test_bundle_adjustment.py::test_bundle_adjustment_default_config -v
```

Expected: both PASS (the existing `test_bundle_adjustment_default_config` should still pass since we only added fields with defaults)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "feat(ba): add add_size + tracks_cache_dir config fields, update loss_history type"
```

---

## Task 2: Fix `_last_loss_history` from assign to append

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

The type changed from `list[float]` to `list[list[float]]`. `_optimize` currently does `self._last_loss_history = loss_hist` (assign). Change to `self._last_loss_history.append(loss_hist)` so multiple incremental calls each append one inner list. The reset happens at the start of `refine()`.

- [ ] **Step 1: Update two stale tests**

Find `test_optimize_captures_loss_history_when_flag_set` and `test_optimize_no_loss_history_by_default`. Update them:

```python
@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_captures_loss_history_when_flag_set():
    """_optimize appends one inner list to _last_loss_history when capture_loss_history=True."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(capture_loss_history=True, lm_steps=n_steps, min_inliers_per_frame=10)
    ba = BundleAdjustment(config=cfg)
    assert ba._last_loss_history == []

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), max_reproj_error=None)

    hist = ba._last_loss_history
    assert isinstance(hist, list)
    assert len(hist) == 1, f"one _optimize call → one inner list; got {len(hist)}"
    assert len(hist[0]) == n_steps
    assert all(isinstance(v, float) for v in hist[0])
    assert all(v >= 0 for v in hist[0])
```

```python
@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_no_loss_history_by_default():
    """Without capture_loss_history, _last_loss_history stays empty after _optimize."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment(config=BundleAdjustmentConfig(min_inliers_per_frame=10))
    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), max_reproj_error=None)
    assert ba._last_loss_history == []
```

- [ ] **Step 2: Find and update the assign in `_optimize`**

In `bundle_adjustment.py`, find the line:
```python
self._last_loss_history = loss_hist
```

Change it to:
```python
self._last_loss_history.append(loss_hist)
```

Also find the else-branch that does:
```python
self._last_loss_history = []
```

**Remove that line** (do not reset inside `_optimize`; reset happens in `refine()`).

At the top of `refine()`, add the reset:
```python
def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
    """Refine camera poses; return updated FeedforwardResult with new extrinsics/intrinsics."""
    self._last_loss_history = []   # ← add this line
    cfg = self.config
    ...
```

- [ ] **Step 3: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v -k "loss_history"
```

Expected: all loss_history tests PASS (CUDA tests skip if no GPU)

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "refactor(ba): _last_loss_history append per-call, reset in refine()"
```

---

## Task 3: Track zarr cache

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Write failing cache tests**

Add to `tests/pointcloud/test_bundle_adjustment.py`:

```python
def test_tracks_cache_save_load(tmp_path):
    """_load_or_extract_tracks saves to zarr; second call returns cached arrays without extracting."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_tracks = np.ones((N, 5, 2), dtype=np.float32)
    fake_vis = np.ones((N, 5), dtype=np.float32) * 0.9
    fake_pts3d = np.ones((5, 3), dtype=np.float32) * 2.0

    cfg = BundleAdjustmentConfig(tracks_cache_dir=tmp_path)
    ba = BundleAdjustment(config=cfg)

    extract_calls = []

    def fake_extract(images, confidence, world_points, max_query_pts, query_frame_num, device=None):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        t1, v1, p1 = ba._load_or_extract_tracks(result)
        t2, v2, p2 = ba._load_or_extract_tracks(result)

    assert len(extract_calls) == 1, "second call should use cache, not re-extract"
    np.testing.assert_array_equal(t1, fake_tracks)
    np.testing.assert_array_equal(t2, fake_tracks)
    np.testing.assert_array_equal(p1, fake_pts3d)
    np.testing.assert_array_equal(p2, fake_pts3d)


def test_tracks_cache_invalidates_on_config_change(tmp_path):
    """Cache is invalidated when query_frame_num changes; extraction runs again."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_a = (np.ones((N, 5, 2), dtype=np.float32),
              np.ones((N, 5), dtype=np.float32),
              np.ones((5, 3), dtype=np.float32))
    fake_b = (np.zeros((N, 5, 2), dtype=np.float32),
              np.zeros((N, 5), dtype=np.float32),
              np.zeros((5, 3), dtype=np.float32))
    extractions = [fake_a, fake_b]

    def fake_extract(*args, **kwargs):
        return extractions.pop(0)

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=5, tracks_cache_dir=tmp_path))
        ba1._load_or_extract_tracks(result)

        # Change query_frame_num — different key → cache miss
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(query_frame_num=10, tracks_cache_dir=tmp_path))
        t2, _, _ = ba2._load_or_extract_tracks(result)

    np.testing.assert_array_equal(t2, fake_b[0])
```

- [ ] **Step 2: Run to confirm they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_tracks_cache_save_load tests/pointcloud/test_bundle_adjustment.py::test_tracks_cache_invalidates_on_config_change -v
```

Expected: both FAIL — `BundleAdjustment has no attribute '_load_or_extract_tracks'`

- [ ] **Step 3: Implement `_compute_tracks_cache_key` and `_load_or_extract_tracks`**

Add module-level function after `_UNSET` sentinel at top of `bundle_adjustment.py`:

```python
def _compute_tracks_cache_key(result: "FeedforwardResult", cfg: "BundleAdjustmentConfig") -> str:
    """SHA-256 key for track cache invalidation — covers image paths + extraction config."""
    meta = {
        "image_paths": sorted(str(p) for p in (result.image_paths or [])),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
```

Add method to `BundleAdjustment` class after `__init__`:

```python
def _load_or_extract_tracks(
    self, result: "FeedforwardResult"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tracks, vis_scores, pts3d_tracks) from zarr cache or VGGSfM extraction."""
    cfg = self.config

    def _extract() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _extract_tracks_vggsfm(
            result.images, result.confidence, result.world_points,
            max_query_pts=cfg.max_query_pts,
            query_frame_num=cfg.query_frame_num,
            device=cfg.device,
        )

    if cfg.tracks_cache_dir is None or not result.image_paths:
        return _extract()

    cache_path = Path(cfg.tracks_cache_dir) / "tracks.zarr"
    expected_key = _compute_tracks_cache_key(result, cfg)

    if cache_path.exists():
        try:
            store = zarr.open(str(cache_path), mode="r")
            if store.attrs.get("cache_key") == expected_key:
                logger.debug("Track cache hit: %s", cache_path)
                return (
                    store["tracks"][:],
                    store["vis_scores"][:],
                    store["pts3d_tracks"][:],
                )
            logger.warning("Track cache key mismatch, re-extracting: %s", cache_path)
        except Exception as exc:
            logger.warning("Track cache unreadable (%s), re-extracting: %s", exc, cache_path)
        shutil.rmtree(cache_path)

    tracks, vis_scores, pts3d_tracks = _extract()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lz4 = BloscCodec(cname="lz4")
    store = zarr.open(str(cache_path), mode="w")
    for key, arr in [("tracks", tracks), ("vis_scores", vis_scores), ("pts3d_tracks", pts3d_tracks)]:
        store.create_array(key, data=arr, chunks=arr.shape, compressors=lz4)
    store.attrs["cache_key"] = expected_key
    logger.debug("Track cache saved: %s", cache_path)

    return tracks, vis_scores, pts3d_tracks
```

- [ ] **Step 4: Run cache tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py::test_tracks_cache_save_load tests/pointcloud/test_bundle_adjustment.py::test_tracks_cache_invalidates_on_config_change -v
```

Expected: both PASS

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v
```

Expected: all previously passing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "feat(ba): zarr track cache with SHA-256 invalidation"
```

---

## Task 4: Extract `_refine_allonce()` + restructure `refine()`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`

Pure refactor — no behaviour change.

- [ ] **Step 1: Add `_refine_allonce()` method**

Add after `_load_or_extract_tracks()` in the `BundleAdjustment` class:

```python
def _refine_allonce(
    self,
    result: "FeedforwardResult",
    tracks: np.ndarray,
    vis_scores: np.ndarray,
    pts3d_tracks: np.ndarray,
    intrinsics_model: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run BA on all N frames simultaneously (current behaviour)."""
    extrinsics_3x4 = result.extrinsics[:, :3, :]
    _, refined_extrinsics, refined_intrinsics_model = self._optimize(
        pts3d_tracks, extrinsics_3x4, intrinsics_model, tracks, vis_scores,
    )
    return refined_extrinsics, refined_intrinsics_model
```

- [ ] **Step 2: Restructure `refine()`**

Replace the current body of `refine()` with:

```python
def refine(self, result: "FeedforwardResult") -> "FeedforwardResult":
    """Refine camera poses; return updated FeedforwardResult with new extrinsics/intrinsics.

    points, colors, and pixel_indices are unchanged — call creator.reproject(result)
    after to re-extract points from refined poses.
    """
    self._last_loss_history = []

    # Load cached tracks or extract via VGGSfM (one extraction shared across all k-steps)
    tracks, vis_scores, pts3d_tracks = self._load_or_extract_tracks(result)

    # Scale intrinsics once — VGGSfM tracks in model-res, intrinsics stored at original-res
    intrinsics_model, sx, sy, tl_x, tl_y = _scale_intrinsics_to_model(
        result.intrinsics, result.images, result.original_coords,
    )

    N = len(result.images)
    add_size = self.config.add_size
    if add_size == 0 or add_size >= N:
        refined_extrinsics, refined_intrinsics_model = self._refine_allonce(
            result, tracks, vis_scores, pts3d_tracks, intrinsics_model,
        )
    else:
        refined_extrinsics, refined_intrinsics_model = self._refine_incremental(
            result, tracks, vis_scores, pts3d_tracks, intrinsics_model, add_size,
        )

    # Rescale refined intrinsics back to original-image space
    refined_intrinsics = refined_intrinsics_model.copy()
    refined_intrinsics[:, 0, 0] /= sx
    refined_intrinsics[:, 1, 1] /= sy
    refined_intrinsics[:, 0, 2] = refined_intrinsics_model[:, 0, 2] / sx + tl_x
    refined_intrinsics[:, 1, 2] = refined_intrinsics_model[:, 1, 2] / sy + tl_y

    refined_extrinsics_4x4 = extrinsics_to_homogeneous(refined_extrinsics)
    return replace(result, extrinsics=refined_extrinsics_4x4, intrinsics=refined_intrinsics)
```

- [ ] **Step 3: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v
```

Expected: all tests PASS (pure refactor, no behaviour change for `add_size=0`)

- [ ] **Step 4: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py
git commit -m "refactor(ba): extract _refine_allonce(), restructure refine() for dispatch"
```

---

## Task 5: Implement `_refine_incremental()`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py`
- Modify: `tests/pointcloud/test_bundle_adjustment.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/pointcloud/test_bundle_adjustment.py`:

```python
def test_incremental_ba_add_size_n_matches_allonce():
    """add_size >= N dispatches to all-at-once path: _optimize called exactly once with all N frames."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 4, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)
    refined_ext = np.tile(np.eye(3, 4), (N, 1, 1)).astype(np.float32)
    refined_intr = np.tile(np.eye(3), (N, 1, 1)).astype(np.float32)

    track_calls = []
    optimize_frame_counts = []

    def mock_optimize(pts3d, extrinsics, intrinsics, tracks, vis_scores):
        optimize_frame_counts.append(len(tracks))
        return (fake_pts3d, refined_ext[:len(tracks)], refined_intr[:len(tracks)])

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize):

        BundleAdjustment(BundleAdjustmentConfig(add_size=0)).refine(result)
        BundleAdjustment(BundleAdjustmentConfig(add_size=N)).refine(result)
        BundleAdjustment(BundleAdjustmentConfig(add_size=N + 10)).refine(result)

    assert optimize_frame_counts == [N, N, N], (
        f"add_size=0/N/N+10 should all call _optimize once with N frames; got {optimize_frame_counts}"
    )


def test_incremental_ba_warm_start_updates_registered_frames():
    """_refine_incremental updates extrinsics[:k] after each step (warm start propagates)."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 6, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)

    # _optimize returns extrinsics perturbed by step index so we can detect updates
    step_counter = [0]
    received_extrinsics = []

    def mock_optimize(pts3d, extrinsics, intrinsics, tracks, vis_scores):
        k = len(tracks)
        received_extrinsics.append(extrinsics.copy())
        # Return slightly modified extrinsics (add step_counter to diagonal) to simulate refinement
        refined = extrinsics.copy()
        refined[:, 0, 0] += float(step_counter[0] + 1)
        step_counter[0] += 1
        return (fake_pts3d, refined, intrinsics.copy())

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize", side_effect=mock_optimize):

        ba = BundleAdjustment(BundleAdjustmentConfig(add_size=2))
        ba.refine(result)

    # With N=6 and add_size=2: steps are k=2, k=4, k=6 → 3 _optimize calls
    assert len(received_extrinsics) == 3, f"expected 3 steps for N=6 add_size=2, got {len(received_extrinsics)}"

    # Step 2 (k=4): the first 2 extrinsics should be the refined output from step 1 (diagonal+1)
    # (They are NOT the original identity extrinsics from the feedforward result)
    # The warm start means step 1's refined[0:2] feeds into step 2's extrinsics[0:2]
    assert received_extrinsics[1][:2, 0, 0].mean() > 1.0, (
        "warm start failed: step-2 extrinsics[:2] should be step-1 refined output, not original feedforward"
    )


def test_incremental_ba_loss_history_has_one_entry_per_step():
    """_last_loss_history contains one inner list per k-step when capture_loss_history=True."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 6, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    P = 5
    fake_tracks = np.zeros((N, P, 2), dtype=np.float32)
    fake_vis = np.ones((N, P), dtype=np.float32)
    fake_pts3d = np.zeros((P, 3), dtype=np.float32)

    # Simulate _optimize appending to loss_history (as actual _optimize does when capture=True)
    def mock_optimize_with_hist(self_ba, pts3d, extrinsics, intrinsics, tracks, vis_scores):
        k = len(tracks)
        if self_ba.config.capture_loss_history:
            self_ba._last_loss_history.append([float(k) * 0.1])
        return (fake_pts3d, extrinsics.copy(), intrinsics.copy())

    with patch("collab_splats.pointcloud.bundle_adjustment._extract_tracks_vggsfm",
               return_value=(fake_tracks, fake_vis, fake_pts3d)), \
         patch.object(BundleAdjustment, "_optimize",
                      lambda self_ba, *a, **kw: mock_optimize_with_hist(self_ba, *a, **kw)):

        ba = BundleAdjustment(BundleAdjustmentConfig(add_size=2, capture_loss_history=True))
        ba.refine(result)

    # N=6, add_size=2 → steps k=2,4,6 → 3 _optimize calls → 3 inner lists
    assert len(ba._last_loss_history) == 3, (
        f"expected 3 inner lists for 3 steps; got {len(ba._last_loss_history)}"
    )
    assert all(isinstance(entry, list) for entry in ba._last_loss_history)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_add_size_n_matches_allonce \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_warm_start_updates_registered_frames \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_loss_history_has_one_entry_per_step \
  -v
```

Expected: all FAIL — `_refine_incremental` not implemented / dispatch not wired

- [ ] **Step 3: Implement `_refine_incremental()`**

Add after `_refine_allonce()` in the `BundleAdjustment` class:

```python
def _refine_incremental(
    self,
    result: "FeedforwardResult",
    tracks: np.ndarray,
    vis_scores: np.ndarray,
    pts3d_tracks: np.ndarray,
    intrinsics_model: np.ndarray,
    add_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow the registered frame set by add_size per step; warm-start each BA from prior step."""
    N = len(result.images)
    # Initialise warm state from feedforward poses
    refined_extrinsics = result.extrinsics[:, :3, :].copy()
    refined_intrinsics = intrinsics_model.copy()

    # Build step sequence: add_size, 2*add_size, ..., N (always ends at exactly N)
    steps = sorted({min(k, N) for k in range(add_size, N + add_size, add_size)})

    for k in steps:
        logger.info("Incremental BA: %d/%d frames registered", k, N)
        _, refined_ext_k, refined_intr_k = self._optimize(
            pts3d_tracks,
            refined_extrinsics[:k].copy(),   # warm start from previous step
            refined_intrinsics[:k].copy(),
            tracks[:k],
            vis_scores[:k],
        )
        refined_extrinsics[:k] = refined_ext_k
        refined_intrinsics[:k] = refined_intr_k

    return refined_extrinsics, refined_intrinsics
```

- [ ] **Step 4: Run new tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_add_size_n_matches_allonce \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_warm_start_updates_registered_frames \
  tests/pointcloud/test_bundle_adjustment.py::test_incremental_ba_loss_history_has_one_entry_per_step \
  -v
```

Expected: all PASS

- [ ] **Step 5: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py tests/pointcloud/test_bundle_adjustment.py
git commit -m "feat(ba): incremental growing-window BA via add_size config field"
```

---

## Task 6: Add `incremental_ba-N` eval condition

**Files:**
- Modify: `evals/eval_gt.py`
- Modify: `evals/eval_suite.sh`

- [ ] **Step 1: Update `_validate_condition` in `eval_gt.py`**

Find the `_validate_condition` function. After the `ba_track-density` regex block, add:

```python
m2 = re.fullmatch(r"incremental_ba-(\d+)", cond)
if m2:
    n = int(m2.group(1))
    if n <= 0:
        raise ValueError(
            f"incremental_ba-{{N}} requires N > 0, got {cond!r}"
        )
    return
```

Also update the error message at the bottom to include the new pattern:
```python
raise ValueError(
    f"Unknown condition {cond!r}. "
    f"Valid: {sorted(_FIXED_CONDITIONS)} or ba_track-density-{{N}} "
    f"or incremental_ba-{{N}} (e.g. incremental_ba-5)"
)
```

- [ ] **Step 2: Update `_make_creator` in `eval_gt.py`**

After the `ba_track-density` block inside `_make_creator`, add:

```python
m2 = re.fullmatch(r"incremental_ba-(\d+)", condition)
if m2:
    add_size = int(m2.group(1))
    cfg = BundleAdjustmentConfig(add_size=add_size)
    if submap_size is not None:
        _no_lc_cfg = LoopClosureConfig(submap_size=submap_size, lc_cosine_threshold=1.0)
        windowed = LoopClosure(base, config=_no_lc_cfg)
        return windowed, cfg
    return base, cfg
```

- [ ] **Step 3: Smoke-test the new condition parses correctly**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from evals.eval_gt import _validate_condition, _make_creator
_validate_condition('incremental_ba-5')
_validate_condition('incremental_ba-1')
print('validation OK')
"
```

Expected: prints `validation OK` with no exception.

- [ ] **Step 4: Update `eval_suite.sh`**

Open `evals/eval_suite.sh` and find the conditions list (likely a variable like `CONDITIONS="baseline ba lc ..."`). Add `incremental_ba-5` to it. The exact line will look like:

```bash
CONDITIONS="${CONDITIONS:-baseline ba lc ba_track-density-4096 incremental_ba-5}"
```

Adjust to match whatever format the existing file uses.

- [ ] **Step 5: Commit**

```bash
git add evals/eval_gt.py evals/eval_suite.sh
git commit -m "feat(evals): add incremental_ba-N condition to eval_gt + eval_suite"
```

---

## Task 7: Eval validation

**Files:** (none modified — eval only)

- [ ] **Step 1: Run incremental BA eval**

```bash
tmux new-session -d -s incremental_eval "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir data/7scenes/chess/chess/seq-01 \
    --output_dir evals/results/chess_seq01_incremental_ba \
    --max_frames 50 \
    --conditions baseline ba incremental_ba-5 2>&1 | tee /tmp/incremental_ba_eval.log"
tmux attach -t incremental_eval
```

- [ ] **Step 2: Interpret results**

Compare ATE RMSE for `incremental_ba-5` vs `baseline` vs `ba`:

| Outcome | Interpretation |
|---|---|
| `incremental_ba-5` ATE < `baseline` | Warm-initialization hypothesis confirmed |
| `incremental_ba-5` ATE ≈ `ba` (both > baseline) | Problem is in track quality or LM config, not architecture — file a separate investigation |
| `incremental_ba-5` ATE > `ba` | Growing-window overhead hurts; try larger `add_size` (e.g. `incremental_ba-10`) |

- [ ] **Step 3: Commit results note**

Add a one-line result comment to the worklog:

```bash
# Edit worklog/WORKLOG.md and append the result table under today's date
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `add_size` config field → Task 1
- [x] `tracks_cache_dir` config field → Task 1
- [x] `_load_or_extract_tracks()` zarr cache → Task 3
- [x] `_last_loss_history: list[list[float]]` → Task 2
- [x] `_refine_allonce()` extraction → Task 4
- [x] `_refine_incremental()` growing-window loop → Task 5
- [x] Warm extrinsics/intrinsics threading → Task 5
- [x] pts3d NOT threaded (L_k shape mismatch) → Task 5 (uses `pts3d_tracks` each step)
- [x] Reprojection agnostic (no change needed) → documented in spec, no task needed
- [x] Pre-implementation eval gate → Task 0
- [x] 5 new tests → Tasks 3 + 5
- [x] 2 stale tests updated → Task 2
- [x] `incremental_ba-N` condition in `eval_gt.py` → Task 6
- [x] `eval_suite.sh` update → Task 6
- [x] Eval validation → Task 7

**Type consistency:**
- `_load_or_extract_tracks(result)` → used in `refine()` ✓
- `_refine_allonce(result, tracks, vis_scores, pts3d_tracks, intrinsics_model)` → called in `refine()` ✓
- `_refine_incremental(result, tracks, vis_scores, pts3d_tracks, intrinsics_model, add_size)` → called in `refine()` ✓
- `_optimize(pts3d, extrinsics_3x4, intrinsics_model, tracks, vis_scores)` → unchanged signature ✓
- `_last_loss_history: list[list[float]]` → consistent across `__init__`, `refine()`, `_optimize` ✓
