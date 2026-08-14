# Geometric Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A geometrically verified sparse cloud + pose verification statistics for any feedforward backbone, produced by pycolmap known-pose triangulation over the localization matchers' tracks.

**Architecture:** `MatchResult` gains keypoint indices (COLMAP's match format); a new `geometry/verification.py` writes features/matches into a `pycolmap.Database`, runs `verify_matches` (Tier 1: per-pair epipolar + relative-pose stats) and `triangulate_points` (Tier 2: verified points with real tracks + per-frame survival/reprojection stats); `Reconstructor` wires it as leaf stage `verify` behind one boolean `pointcloud.geometric_verification`, sharing the localization extractor's zarr feature cache. Raw model outputs are immutable — everything lands under `<backend>/colmap/`.

**Tech Stack:** pycolmap 4.0.4 (CPU; `has_cuda=False`), existing localization extractors (disk/xfeat/loma/loma-g), zarr feature cache, numpy.

**Spec:** `docs/superpowers/specs/2026-08-14-geometric-verification-design.md`
**Decision record:** `docs/superpowers/specs/2026-08-14-geometric-consistency-proposal.md`

---

## Ground truth about pycolmap 4.0.4 (probed on this machine — do not re-derive)

These were verified with a synthetic 3-view probe (60 points, 3 cameras, identity matches → 60/60 triangulated, track length 3, max 3D error 2e-6):

1. `pycolmap.Database.open(str(path))` — static method. A bare `pycolmap.Database()` is an abstract trampoline and SIGABRTs with "pure virtual function" on first write.
2. **DB ids must mirror the reconstruction exactly** — one camera per image, `camera_id == image_id`. A single shared DB camera against `build_pycolmap_reconstruction`'s per-image trivial rigs fails inside `triangulate_points` with `Check failed: existing_frame.RigId() == frame.RigId()`.
3. `db.write_image(pycolmap.Image(name=..., camera_id=..., image_id=...), use_image_id=True)` — signature confirmed; `write_camera(cam, use_camera_id=True)`; `write_keypoints(image_id, float64 (N,2))`; `write_matches(id1, id2, uint32 (K,2))`.
4. `pycolmap.verify_matches(db_path, pairs_path, options=opts)` needs `opts.compute_relative_pose = True` or every `TwoViewGeometry.cam2_from_cam1` is `None`. pairs file = lines of `"name1 name2"`.
5. `db.read_two_view_geometries()` returns `(pair_ids: list[int], geoms: list[TwoViewGeometry])`. Decode: `id1, id2 = divmod(pair_id, 2147483647)`. There is **no** `pair_id_to_image_pair` helper in this build.
6. `pycolmap.triangulate_points(recon, db_path, image_dir, out_dir)` clears model points by default (`clear_points=True`), chains tracks internally, and applies mapper filtering at COLMAP defaults (`filter_max_reproj_error=4.0`, `filter_min_tri_angle=1.5` — confirmed on `IncrementalPipelineOptions().mapper`). **No `ObservationManager` / `filter_all_points3D` call is needed** — the pipeline already filters. `image_dir` may be an existing directory with no images (keypoints come from the DB).
7. `Image.cam_from_world()` is a **method** (returns `Rigid3d`) in this build. `Rigid3d` has `inverse()`, `__mul__`, `.rotation.matrix()`, `.translation`. `Point2D.has_point3D()` and `.point3D_id` exist.

Run all tests with `/opt/venv/reconstruction/bin/python -m pytest ... -p no:randomly`.
Commit docs with `git add -f` (docs/superpowers is gitignored). End commit messages with the
`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` trailer. Always commit with explicit pathspecs.

## Validation: datasets and metrics (answers "how do we validate this?")

- **Primary dataset: 7-Scenes chess/seq-01** — already used for the mv-confidence report and LC evals; loader exists (`evals/datasets.py::_load_7scenes`). It has **both GT poses** (`frame-XXXXXX.pose.txt`, c2w) — which validate Tier 1 relative-pose errors directly — **and GT depth** (`frame-XXXXXX.depth.png`, uint16 mm, 65535 = invalid) for Tier 2 depth accuracy. Caveat from the LoGeR benchmark: a GT column can be a noise floor — always report the reference-free triangulated-vs-model agreement alongside it.
- **Literature benchmark: ETH3D** is the standard for triangulation completeness/accuracy (used by COLMAP's own evaluations and VGGSfM). It requires a download and is **deferred** — noted so the follow-on knows where to go for cross-paper numbers.
- **Metrics** (all distributions as median/p90/p99, never median alone):
  - *Tier 1 (pose):* per-pair rotation error and translation-direction error, estimated-vs-model — and on 7-Scenes, estimated-vs-GT and model-vs-GT for the same pairs; epipolar inlier ratio per pair.
  - *Tier 2 (points):* triangulated-vs-model relative depth difference at track pixels (same world frame, scale-free); triangulated-vs-GT and model-vs-GT relative depth error after one **global** median scale (backbones are non-metric), plus out10 (fraction of pixels >10% error — the tail metric the mv report standardized on).
  - *Yield:* point count, track-length distribution, per-frame track survival (tracks / keypoints) — the input to the sparse_pc replace-vs-ship-both decision.
  - *Negative control (mandatory):* +2° rotation on a subset of poses must be flagged by Tier 1 pair errors and depressed Tier 2 survival — "a threshold nobody mutated is presumed inert".

## File structure

- Modify: `collab_splats/localization/extractors.py` — `MatchResult` indices (Task 1)
- Modify: `collab_splats/localization/localizer.py` — extract cache reader `load_reconstruction_features` (Task 2)
- Create: `collab_splats/geometry/verification.py` — pair selection, DB export, Tier 1/2, `verify_reconstruction` (Tasks 3–5)
- Create: `tests/geometry/test_verification.py` — synthetic-scene tests + negative control (Tasks 3–6)
- Modify: `collab_splats/wrapper/reconstructor.py`, `configs/base.yaml`, `collab_splats/remote/sources.py`, `configs/README.md` — wiring (Task 7)
- Create: `tests/wrapper/test_verify_stage.py` (Task 7)
- Create: `evals/scripts/eval_verification.py` — measurement script (Task 8)

---

### Task 1: `MatchResult` keypoint indices

**Files:**
- Modify: `collab_splats/localization/extractors.py`
- Test: `tests/localization/test_extractors.py`, `tests/localization/test_loma_extractor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/localization/test_extractors.py` (it already has the `image_pair` fixture — two 128×128 random-dot images, second shifted 4 px):

```python
def test_match_returns_indices_disk(image_pair):
    """idx_q/idx_db index the keypoint tables the pixel pairs came from."""
    img0, img1 = image_pair
    ex = DiskExtractor(top_k=256)
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert m.idx_q is not None and m.idx_q.dtype == np.int64
    assert m.idx_db is not None and m.idx_db.dtype == np.int64
    np.testing.assert_allclose(m.query_px, f0.keypoints.numpy()[m.idx_q])
    np.testing.assert_allclose(m.ref_px, f1.keypoints.numpy()[m.idx_db])


def test_match_returns_indices_xfeat(image_pair):
    """XFeat's LighterGlue index pairs survive into MatchResult."""
    img0, img1 = image_pair
    ex = XFeatExtractor()
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert m.idx_q is not None and m.idx_db is not None
    np.testing.assert_allclose(m.query_px, f0.keypoints.numpy()[m.idx_q])
    np.testing.assert_allclose(m.ref_px, f1.keypoints.numpy()[m.idx_db])


def test_xfeat_star_has_no_indices(image_pair):
    """XFeatStar refines pixels per pair — no stable keypoint-table indices exist."""
    from collab_splats.localization.extractors import XFeatStarExtractor

    img0, img1 = image_pair
    ex = XFeatStarExtractor()
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert m.idx_q is None and m.idx_db is None


def test_empty_match_has_empty_indices():
    """_empty_match carries zero-length int64 index arrays, not None."""
    m = _empty_match()
    assert m.idx_q.shape == (0,) and m.idx_q.dtype == np.int64
    assert m.idx_db.shape == (0,) and m.idx_db.dtype == np.int64
```

Add `_empty_match` (and `XFeatStarExtractor` if absent) to the test file's imports from `collab_splats.localization.extractors`.

Append to `tests/localization/test_loma_extractor.py` a mirror of the disk test using the file's existing extractor fixture/style (LomaG inherits `match` from `LomaExtractor`, so one loma test covers both):

```python
def test_loma_match_returns_indices(image_pair):
    """LoMa filter_matches indices survive into MatchResult."""
    img0, img1 = image_pair
    ex = LomaExtractor()
    f0, f1 = ex.extract(img0), ex.extract(img1)
    m = ex.match(f0, f1, img0.shape[:2])
    assert m.idx_q is not None and m.idx_db is not None
    np.testing.assert_allclose(m.query_px, f0.keypoints.numpy()[m.idx_q])
    np.testing.assert_allclose(m.ref_px, f1.keypoints.numpy()[m.idx_db])
```

(If `test_loma_extractor.py` has no `image_pair` fixture, copy the one from `test_extractors.py` verbatim into it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_extractors.py -k indices -p no:randomly -v`
Expected: FAIL — `MatchResult` has no attribute `idx_q` / unexpected keyword.

- [ ] **Step 3: Implement**

In `collab_splats/localization/extractors.py`:

Replace the `MatchResult` dataclass (currently lines 49–57) with:

```python
@dataclass
class MatchResult:
    """Matched pixel coordinates between a query and one reference image.

    idx_q/idx_db are the keypoint-table indices behind the pixel pairs — COLMAP's match
    format, consumed by geometry/verification.py. None when the matcher cannot provide
    stable indices (XFeatStar's per-pair subpixel refinement moves the same keypoint to
    different coordinates in different pairs, so no single table row describes it).
    """

    query_px: np.ndarray  # (K, 2) float32 xy in query image
    ref_px: np.ndarray  # (K, 2) float32 xy in reference image
    idx_q: np.ndarray | None = None  # (K,) int64 into the query keypoint table
    idx_db: np.ndarray | None = None  # (K,) int64 into the reference keypoint table

    def __len__(self) -> int:
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """Zero-length MatchResult (with empty index arrays — a zero match is indexable)."""
    z = np.zeros((0, 2), dtype=np.float32)
    zi = np.zeros(0, dtype=np.int64)
    return MatchResult(query_px=z, ref_px=z, idx_q=zi, idx_db=zi)
```

Disk matcher — the `return MatchResult(...)` at the end of `DiskExtractor.match` becomes:

```python
        return MatchResult(
            query_px=query.keypoints[idx_q].numpy().astype(np.float32),
            ref_px=db.keypoints[idx_db].numpy().astype(np.float32),
            idx_q=idx_q.numpy().astype(np.int64),
            idx_db=idx_db.numpy().astype(np.int64),
        )
```

`XFeatExtractor.match` — identical two extra kwargs on its `return MatchResult(...)` (its `idx_q`/`idx_db` are already torch tensors two lines above).

`LomaExtractor.match` — identical two extra kwargs on its `return MatchResult(...)`.

`XFeatStarExtractor.match` — **unchanged** (its `MatchResult(query_px=pairs[:, :2], ref_px=pairs[:, 2:])` leaves the new fields at their `None` defaults, which is the contract).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_extractors.py tests/localization/test_loma_extractor.py -p no:randomly -v`
Expected: all PASS (pre-existing tests included — pixel fields are untouched).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/extractors.py tests/localization/test_extractors.py tests/localization/test_loma_extractor.py
git commit -m "feat(localization): expose keypoint indices on MatchResult"
```

---

### Task 2: shared feature-cache reader

**Files:**
- Modify: `collab_splats/localization/localizer.py`
- Test: `tests/localization/test_localization_cache.py`

The read half of `save_index` currently lives inline in `CameraLocalizer.load_index` (the block reading `local_features/{name}/reconstruction` CSR arrays, lines ~317–358). Verification needs exactly that read without a localizer (no world_points/extrinsics). Extract it as a module function; `load_index` calls it — DRY, zero behavior change.

- [ ] **Step 1: Write the failing test**

Append to `tests/localization/test_localization_cache.py` (reuse its `_make_features` helper):

```python
def test_load_reconstruction_features_roundtrip(tmp_path):
    """Module-level cache reader returns exactly what save_index wrote."""
    from collab_splats.localization.localizer import load_reconstruction_features

    feats = [_make_features(n_kpts=5), _make_features(n_kpts=8)]
    ids = ["frame_000000.jpg", "frame_000001.jpg"]
    loc = CameraLocalizer(
        world_points=np.zeros((2, 4, 4, 3), dtype=np.float32),
        extrinsics=np.stack([np.eye(4, dtype=np.float32)] * 2),
        frame_features=feats,
        image_paths=ids,
        image_hw=(64, 64),
    )
    zarr_path = tmp_path / "feedforward.zarr"
    loc.save_index(zarr_path, "xfeat")

    out_feats, out_ids, hw = load_reconstruction_features(zarr_path, "xfeat")
    assert out_ids == ids and hw == (64, 64)
    assert [len(f.keypoints) for f in out_feats] == [5, 8]
    np.testing.assert_allclose(out_feats[1].keypoints.numpy(), feats[1].keypoints.numpy())
    np.testing.assert_allclose(out_feats[0].descriptors.numpy(), feats[0].descriptors.numpy())


def test_load_reconstruction_features_missing_raises(tmp_path):
    """Missing cache raises KeyError naming the extractor."""
    from collab_splats.localization.localizer import load_reconstruction_features

    with pytest.raises(KeyError, match="disk"):
        load_reconstruction_features(tmp_path / "feedforward.zarr", "disk")
```

**Check the `CameraLocalizer` constructor kwargs against the file before writing the test** — mirror whichever construction pattern the existing round-trip tests in `test_localization_cache.py` use (they already build a localizer and call `save_index`); the kwargs above are indicative, the existing tests are authoritative.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localization_cache.py -k load_reconstruction_features -p no:randomly -v`
Expected: FAIL — `ImportError: cannot import name 'load_reconstruction_features'`.

- [ ] **Step 3: Implement**

In `localizer.py`, add a module-level function above `CameraLocalizer` (move, don't copy, the body from `load_index`):

```python
def load_reconstruction_features(
    zarr_path: "str | Path", extractor_name: str
) -> tuple[list[LocalFeatures], list[str], tuple[int, int]]:
    """Read the per-frame reconstruction feature cache for one extractor.

    Returns (per-frame LocalFeatures, image ids, (H, W)). Raises KeyError when the
    cache is missing. This is the read half of CameraLocalizer.save_index; load_index
    builds a localizer on top of it, geometry/verification.py consumes it directly.
    """
    zarr_path = pathlib.Path(zarr_path)
    store = zarr.open(str(zarr_path), mode="r")

    rec_key = f"local_features/{extractor_name}/reconstruction"
    if rec_key not in store:
        raise KeyError(
            f"No feature cache for extractor '{extractor_name}' in {zarr_path}. "
            "Rebuild via CameraLocalizer.from_feedforward()."
        )

    # ── moved verbatim from load_index: bulk CSR decode of the reconstruction group ──
    rec_group = store[rec_key]
    rec_image_paths = [str(p) for p in rec_group.attrs["image_paths"]]
    hw = tuple(int(x) for x in rec_group.attrs["hw"])
    offsets = rec_group["frame_offsets"][:]
    logger.info("CameraLocalizer: reading feature DB (%d frames) from zarr", len(offsets) - 1)
    t0 = time.perf_counter()
    all_kpts = (
        rec_group["keypoints"][:] if rec_group["keypoints"].shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
    )
    all_descs = (
        rec_group["descriptors"][:] if rec_group["descriptors"].shape[0] > 0 else np.zeros((0, 1), dtype=np.float32)
    )
    all_scores = rec_group["scores"][:] if "scores" in rec_group else None
    all_scales = rec_group["scales"][:] if "scales" in rec_group else None
    logger.info(
        "CameraLocalizer: read %s keypoints / %.0f MB descriptors in %.1fs",
        f"{len(all_kpts):,}",
        all_descs.nbytes / 1e6,
        time.perf_counter() - t0,
    )

    rec_features: list[LocalFeatures] = []
    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        rec_features.append(
            LocalFeatures(
                keypoints=torch.from_numpy(all_kpts[s:e]),
                descriptors=torch.from_numpy(all_descs[s:e]),
                scores=torch.from_numpy(all_scores[s:e]) if all_scores is not None else None,
                scales=torch.from_numpy(all_scales[s:e]) if all_scales is not None else None,
            )
        )
    return rec_features, rec_image_paths, hw
```

Then in `load_index`, replace the moved block with:

```python
        rec_features, rec_image_paths, hw = load_reconstruction_features(zarr_path, extractor_name)
```

keeping everything after it (the `localized/` group handling and localizer construction) unchanged. Adjust for how the surrounding code opens `store` — `load_index` still needs its own `store` handle for the `localized/` group; keep that open where it is.

- [ ] **Step 4: Run the full localization cache tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localization_cache.py tests/localization/test_localizer.py -p no:randomly -v`
Expected: all PASS (round-trip parity is the regression gate for the refactor).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/localization/localizer.py tests/localization/test_localization_cache.py
git commit -m "refactor(localization): extract load_reconstruction_features cache reader"
```

---

### Task 3: `geometry/verification.py` — pair selection

**Files:**
- Create: `collab_splats/geometry/verification.py`
- Create: `tests/geometry/test_verification.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_verification.py`:

```python
"""Tests for geometric verification: pair selection, DB export, triangulation, negative control."""

import numpy as np
import pytest

from collab_splats.geometry.verification import (
    select_loop_pairs,
    select_sequential_pairs,
)


def test_sequential_pairs_window():
    """All (i, j) with 0 < j - i <= window, no duplicates, no self-pairs."""
    pairs = select_sequential_pairs(5, window=2)
    assert pairs == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def test_sequential_pairs_window_covers_all_when_large():
    """window >= n-1 yields the complete pair set."""
    pairs = select_sequential_pairs(4, window=10)
    assert len(pairs) == 6  # C(4,2)


def test_loop_pairs_finds_far_similar_frames():
    """Loop pairs link similar frames outside the sequential window; near frames excluded."""
    rng = np.random.default_rng(0)
    descs = rng.standard_normal((30, 8)).astype(np.float32)
    descs /= np.linalg.norm(descs, axis=1, keepdims=True)
    descs[25] = descs[0]  # frame 25 revisits frame 0
    pairs = select_loop_pairs(descs, window=5, top_k=1)
    assert (0, 25) in pairs
    assert all(j - i >= 10 for i, j in pairs)  # 2*window gap enforced
    assert all(i < j for i, j in pairs)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -p no:randomly -v`
Expected: FAIL — `ModuleNotFoundError: collab_splats.geometry.verification`.

- [ ] **Step 3: Implement**

Create `collab_splats/geometry/verification.py`:

```python
"""Geometric verification of feedforward reconstructions via pycolmap.

Layer on top of any backbone: takes the final COLMAP-frame reconstruction (pose/camera
authority) plus the localization extractor's per-frame features, and produces
  - Tier 1: per-pair epipolar inlier counts and estimated-vs-model relative-pose errors,
  - Tier 2: a triangulated sparse cloud with real feature tracks, filtered at COLMAP's
    defaults (4.0 px reprojection, 1.5 deg triangulation angle), with per-frame stats.
Spec: docs/superpowers/specs/2026-08-14-geometric-verification-design.md.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from collab_splats.localization.extractors import BaseLocalExtractor, LocalFeatures, MatchResult

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# COLMAP pair-id convention: pair_id = image_id1 * kMaxNumImages + image_id2.
# This build of pycolmap (4.0.4) exposes no decoding helper, so we mirror the constant.
_COLMAP_MAX_NUM_IMAGES = 2147483647

# Sequential adjacency window (frames) and loop-retrieval candidates per frame.
# Module defaults on purpose — repo precedent (mv confidence) keeps tuning knobs out of config.
DEFAULT_WINDOW = 10
DEFAULT_LOOP_TOP_K = 2


########################################
# Pair selection
########################################


def select_sequential_pairs(n_frames: int, window: int = DEFAULT_WINDOW) -> list[tuple[int, int]]:
    """All frame-index pairs (i, j) with 0 < j - i <= window. O(N * window), never O(N^2)."""
    return [
        (i, j)
        for i in range(n_frames)
        for j in range(i + 1, min(i + window + 1, n_frames))
    ]


def select_loop_pairs(
    descriptors: np.ndarray,
    window: int = DEFAULT_WINDOW,
    top_k: int = DEFAULT_LOOP_TOP_K,
) -> list[tuple[int, int]]:
    """Loop-candidate pairs from (N, D) L2-normalized global descriptors.

    For each frame, its top_k most-similar frames at least 2*window away — far enough
    that the sequential window cannot already cover the pair.
    """
    sim = descriptors @ descriptors.T
    n = len(sim)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        far = np.abs(np.arange(n) - i) >= 2 * window
        if not far.any():
            continue
        # Rank only the far frames; -inf keeps near frames out of the top_k
        ranked = np.argsort(np.where(far, sim[i], -np.inf))[::-1][:top_k]
        for j in ranked:
            if far[j]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -p no:randomly -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/verification.py tests/geometry/test_verification.py
git commit -m "feat(geometry): verification pair selection (sequential window + loop retrieval)"
```

---

### Task 4: COLMAP database export + Tier 1 epipolar verification

**Files:**
- Modify: `collab_splats/geometry/verification.py`
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_verification.py`. The synthetic scene and stub matcher are shared by Tasks 4–6 — define them once here:

```python
import torch

from collab_splats.localization.extractors import LocalFeatures, MatchResult
from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction

W, H = 640, 480
_K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float32)


def _synthetic_scene(n_cams: int = 3, n_pts: int = 60, seed: int = 0):
    """World points + lateral camera array + exact projections.

    Returns (pts_w (P,3), extrinsics (N,3,4) w2c, per-frame keypoints list of (P,2))."""
    rng = np.random.default_rng(seed)
    pts_w = np.stack(
        [rng.uniform(-1, 1, n_pts), rng.uniform(-0.8, 0.8, n_pts), rng.uniform(3.0, 5.0, n_pts)],
        axis=1,
    ).astype(np.float32)
    extrinsics = []
    for i in range(n_cams):
        E = np.eye(4, dtype=np.float32)
        E[0, 3] = -0.3 * i  # camera at x = +0.3*i; w2c translation is the negative
        extrinsics.append(E[:3])
    extrinsics = np.stack(extrinsics)
    kps = []
    for E in extrinsics:
        pc = (E[:3, :3] @ pts_w.T + E[:3, 3:4]).T
        uv = (_K @ (pc / pc[:, 2:3]).T).T[:, :2]
        kps.append(uv.astype(np.float32))
    return pts_w, extrinsics, kps


def _make_recon(extrinsics):
    """Poses-only pycolmap reconstruction over the synthetic cameras."""
    n = len(extrinsics)
    return build_pycolmap_reconstruction(
        pts3d=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.stack([_K] * n),
        image_width=W,
        image_height=H,
        image_names=[f"frame_{i:05d}" for i in range(n)],
    )


def _features_from_keypoints(kps):
    """Wrap projected keypoints as LocalFeatures (descriptors unused by the stub matcher)."""
    return [
        LocalFeatures(keypoints=torch.from_numpy(k), descriptors=torch.zeros(len(k), 4))
        for k in kps
    ]


class _IdentityMatcher(BaseLocalExtractor):
    """Stub matcher: keypoint i in every frame observes world point i (ground-truth tracks)."""

    def extract(self, image):  # pragma: no cover - never called in these tests
        raise NotImplementedError

    def match(self, query, db, image_hw):
        n = min(len(query.keypoints), len(db.keypoints))
        idx = np.arange(n, dtype=np.int64)
        return MatchResult(
            query_px=query.keypoints.numpy()[:n],
            ref_px=db.keypoints.numpy()[:n],
            idx_q=idx,
            idx_db=idx.copy(),
        )


class _NoIndexMatcher(_IdentityMatcher):
    """Stub matcher mimicking XFeatStar: pixels only, no table indices."""

    def match(self, query, db, image_hw):
        m = super().match(query, db, image_hw)
        return MatchResult(query_px=m.query_px, ref_px=m.ref_px)


def test_tier1_pair_stats_on_clean_scene(tmp_path):
    """verify_matches recovers each pair's relative pose to within a fraction of a degree."""
    from collab_splats.geometry.verification import verify_reconstruction

    _, extrinsics, kps = _synthetic_scene()
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    assert len(result.pair_stats) == 3  # window covers all pairs of 3 frames
    for p in result.pair_stats:
        assert p.num_inliers >= 55
        assert p.rot_error_deg < 0.1
        assert p.t_direction_error_deg < 1.0


def test_no_index_matcher_rejected(tmp_path):
    """A matcher without keypoint indices (XFeatStar) is rejected with a clear error."""
    from collab_splats.geometry.verification import verify_reconstruction

    _, extrinsics, kps = _synthetic_scene()
    with pytest.raises(ValueError, match="indices"):
        verify_reconstruction(
            recon=_make_recon(extrinsics),
            features=_features_from_keypoints(kps),
            matcher=_NoIndexMatcher(),
            output_dir=tmp_path,
        )


def test_keypoint_bounds_guard(tmp_path):
    """Keypoints outside the camera grid abort the export (resolution-mismatch class)."""
    from collab_splats.geometry.verification import verify_reconstruction

    _, extrinsics, kps = _synthetic_scene()
    kps[1][0] = [W * 2.0, H * 2.0]  # simulate a cache built at a different resolution
    with pytest.raises(ValueError, match="bounds"):
        verify_reconstruction(
            recon=_make_recon(extrinsics),
            features=_features_from_keypoints(kps),
            matcher=_IdentityMatcher(),
            output_dir=tmp_path,
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -k "tier1 or rejected or bounds" -p no:randomly -v`
Expected: FAIL — `ImportError: cannot import name 'verify_reconstruction'`.

- [ ] **Step 3: Implement DB export + Tier 1 (and the `verify_reconstruction` shell)**

Append to `collab_splats/geometry/verification.py`. Task 5 extends `verify_reconstruction` with triangulation — write it here already structured for that (the triangulation lines land in the marked spot):

```python
########################################
# Results
########################################


@dataclass
class PairStats:
    """Epipolar verification of one image pair against the model's relative pose."""

    name1: str
    name2: str
    num_matches: int
    num_inliers: int
    rot_error_deg: float  # estimated-vs-model relative rotation, degrees
    t_direction_error_deg: float  # translation-direction angle, degrees (nan if degenerate)


@dataclass
class VerificationResult:
    """Verified reconstruction + Tier 1/2 statistics."""

    reconstruction: pycolmap.Reconstruction  # model poses + triangulated tracked points
    pair_stats: list[PairStats]
    frame_stats: dict[str, dict]  # per image name: n_keypoints, n_tracks, mean_reproj_error_px
    summary: dict = field(default_factory=dict)


########################################
# Pose-error helpers
########################################


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _pair_pose_errors(estimated: pycolmap.Rigid3d, model_rel: pycolmap.Rigid3d) -> tuple[float, float]:
    """(rotation error deg, translation-direction error deg) of estimated vs model relative pose.

    The epipolar estimate fixes translation only up to scale, so the direction angle is
    the honest comparison; a near-zero baseline on either side makes it undefined (nan).
    """
    rot_err = _rotation_angle_deg(estimated.rotation.matrix() @ model_rel.rotation.matrix().T)
    t_est, t_mod = estimated.translation, model_rel.translation
    n_est, n_mod = np.linalg.norm(t_est), np.linalg.norm(t_mod)
    if n_est < 1e-9 or n_mod < 1e-9:
        return rot_err, float("nan")
    cos = np.clip(np.dot(t_est / n_est, t_mod / n_mod), -1.0, 1.0)
    return rot_err, float(np.degrees(np.arccos(cos)))


########################################
# COLMAP database export
########################################


def _write_database(
    recon: pycolmap.Reconstruction,
    features: list[LocalFeatures],
    matches: dict[tuple[int, int], np.ndarray],
    db_path: Path,
) -> None:
    """Write cameras/images/keypoints/matches into a fresh COLMAP database.

    Ids mirror `recon` exactly (one camera per image, camera_id == image_id) —
    triangulate_points joins DB rows to reconstruction frames by id, and a shared DB
    camera against per-image trivial rigs fails COLMAP's RigId check.
    """
    if db_path.exists():
        db_path.unlink()  # stale DBs accumulate duplicate rows; always start fresh
    db = pycolmap.Database.open(str(db_path))
    image_ids = sorted(recon.images)
    for image_id, feats in zip(image_ids, features):
        image = recon.images[image_id]
        camera = recon.cameras[image.camera_id]
        kpts = feats.keypoints.numpy().astype(np.float64)
        # Bounds guard: catches a feature cache built at a different resolution than the
        # reconstruction's cameras (the 2026-08-11 model-res-vs-original-res class)
        if len(kpts) and (
            kpts.min() < 0.0
            or kpts[:, 0].max() >= camera.width
            or kpts[:, 1].max() >= camera.height
        ):
            raise ValueError(
                f"Keypoints for {image.name} exceed camera bounds ({camera.width}x{camera.height}) "
                "— the feature cache and the reconstruction disagree on image resolution."
            )
        db.write_camera(camera, use_camera_id=True)
        # Set image_id via the property, not the ctor (the ctor kwarg is unverified in
        # this pycolmap build; the property + use_image_id=True path is the documented one)
        image_row = pycolmap.Image(name=image.name, camera_id=image.camera_id)
        image_row.image_id = image_id
        db.write_image(image_row, use_image_id=True)
        db.write_keypoints(image_id, kpts)
    for (id1, id2), m in matches.items():
        db.write_matches(id1, id2, m)
    db.close()


########################################
# Entry point
########################################


def verify_reconstruction(
    recon: pycolmap.Reconstruction,
    features: list[LocalFeatures],
    matcher: BaseLocalExtractor,
    output_dir: str | Path,
    window: int = DEFAULT_WINDOW,
    loop_descriptors: np.ndarray | None = None,
    loop_top_k: int = DEFAULT_LOOP_TOP_K,
) -> VerificationResult:
    """Triangulate and epipolar-verify a reconstruction's poses with independent features.

    Args:
        recon: pose/camera authority (original-resolution K); its points are ignored.
        features: per-image LocalFeatures, aligned with sorted(recon.images) order.
        matcher: a BaseLocalExtractor whose match() exposes keypoint indices.
        output_dir: writes database.db, verified/ (COLMAP model), verification.json.
        window: sequential pair adjacency window (frames).
        loop_descriptors: optional (N, D) global descriptors adding loop pairs.
        loop_top_k: loop candidates per frame when loop_descriptors given.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(recon.images)
    if len(features) != len(image_ids):
        raise ValueError(
            f"{len(features)} feature frames vs {len(image_ids)} reconstruction images — "
            "the cache and the reconstruction describe different runs."
        )

    # ── Pair selection: adjacency window + optional retrieval loop pairs ──
    pairs = select_sequential_pairs(len(image_ids), window)
    if loop_descriptors is not None:
        pairs = sorted(set(pairs) | set(select_loop_pairs(loop_descriptors, window, loop_top_k)))

    # ── Match every pair; index pairs are COLMAP's match format ──
    cam0 = recon.cameras[recon.images[image_ids[0]].camera_id]
    hw = (cam0.height, cam0.width)
    matches: dict[tuple[int, int], np.ndarray] = {}
    for i, j in pairs:
        m = matcher.match(features[i], features[j], hw)
        if m.idx_q is None or m.idx_db is None:
            raise ValueError(
                f"{type(matcher).__name__} does not expose keypoint indices "
                "(per-pair refined matchers cannot feed COLMAP tracks) — use disk/xfeat/loma."
            )
        if len(m) == 0:
            continue
        matches[(image_ids[i], image_ids[j])] = np.stack([m.idx_q, m.idx_db], axis=1).astype(np.uint32)
    logger.info("Verification: %d/%d pairs matched", len(matches), len(pairs))

    # ── COLMAP database + epipolar verification (Tier 1) ──
    db_path = output_dir / "database.db"
    _write_database(recon, features, matches, db_path)
    pairs_path = output_dir / "pairs.txt"
    pairs_path.write_text(
        "\n".join(f"{recon.images[a].name} {recon.images[b].name}" for a, b in matches)
    )
    tvg_options = pycolmap.TwoViewGeometryOptions()
    tvg_options.compute_relative_pose = True  # cam2_from_cam1 stays None without this
    pycolmap.verify_matches(str(db_path), str(pairs_path), options=tvg_options)

    db = pycolmap.Database.open(str(db_path))
    pair_ids, geoms = db.read_two_view_geometries()
    db.close()
    pair_stats = []
    for pid, g in zip(pair_ids, geoms):
        id1, id2 = divmod(int(pid), _COLMAP_MAX_NUM_IMAGES)
        im1, im2 = recon.images[id1], recon.images[id2]
        model_rel = im2.cam_from_world() * im1.cam_from_world().inverse()
        if g.cam2_from_cam1 is not None:
            rot_err, tdir_err = _pair_pose_errors(g.cam2_from_cam1, model_rel)
        else:
            rot_err = tdir_err = float("nan")  # too few inliers for a pose estimate
        pair_stats.append(
            PairStats(
                name1=im1.name,
                name2=im2.name,
                num_matches=int(matches[(id1, id2)].shape[0]),
                num_inliers=len(g.inlier_matches),
                rot_error_deg=rot_err,
                t_direction_error_deg=tdir_err,
            )
        )

    # ── Tier 2: known-pose triangulation (Task 5 fills in from here) ──
    verified, frame_stats, summary = _triangulate_and_summarize(
        recon, db_path, output_dir, pair_stats, features
    )
    result = VerificationResult(
        reconstruction=verified, pair_stats=pair_stats, frame_stats=frame_stats, summary=summary
    )
    _write_report(result, output_dir / "verification.json")
    return result
```

Also add a **temporary** minimal `_triangulate_and_summarize` and `_write_report` so Tier 1 tests run before Task 5 (Task 5 replaces them):

```python
def _triangulate_and_summarize(recon, db_path, output_dir, pair_stats, features):
    """Placeholder until Task 5: no triangulation, empty stats."""
    return recon, {}, {}


def _write_report(result, path):
    """Placeholder until Task 5."""
    path.write_text("{}")
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -p no:randomly -v`
Expected: all PASS (Tier 1, rejection, bounds guard, and Task 3's pair tests).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/verification.py tests/geometry/test_verification.py
git commit -m "feat(geometry): COLMAP DB export + Tier 1 epipolar pose verification"
```

---

### Task 5: known-pose triangulation + Tier 2 stats + report

**Files:**
- Modify: `collab_splats/geometry/verification.py`
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_verification.py`:

```python
def test_triangulation_recovers_scene(tmp_path):
    """Tier 2 triangulates the synthetic scene: full yield, full tracks, ~zero error."""
    from collab_splats.geometry.verification import verify_reconstruction

    pts_w, extrinsics, kps = _synthetic_scene()
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    verified = result.reconstruction
    assert verified.num_points3D() >= 55  # of 60; COLMAP may drop boundary cases
    # Every surviving point carries a real (non-empty) track and lies on a GT point
    for p in verified.points3D.values():
        assert p.track.length() == 3
        assert np.linalg.norm(pts_w - p.xyz, axis=1).min() < 1e-3
    # Per-frame stats populated for all frames; reprojection error is sub-pixel
    assert set(result.frame_stats) == {f"frame_{i:05d}" for i in range(3)}
    for s in result.frame_stats.values():
        assert s["n_tracks"] >= 55
        assert s["mean_reproj_error_px"] < 0.5
    # Summary distributions present (median/p90/p99 — never median alone)
    assert result.summary["n_points"] >= 55
    assert set(result.summary["track_length"]) == {"median", "p90", "p99"}
    # Report written and loadable
    report = json.loads((tmp_path / "verification.json").read_text())
    assert report["summary"]["n_points"] == result.summary["n_points"]
    assert len(report["pair_stats"]) == 3
    # COLMAP model on disk for downstream tooling
    assert (tmp_path / "verified" / "points3D.bin").exists()
```

Add `import json` to the test file's imports.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -k triangulation_recovers -p no:randomly -v`
Expected: FAIL — placeholder returns `recon` with 0 points / empty stats.

- [ ] **Step 3: Implement**

Replace the two placeholders in `verification.py` with:

```python
########################################
# Tier 2: triangulation + summaries
########################################


def _distribution(values) -> dict | None:
    """median/p90/p99 of a value list — never median alone. None when empty/all-nan."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return {
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
    }


def _triangulate_and_summarize(
    recon: pycolmap.Reconstruction,
    db_path: Path,
    output_dir: Path,
    pair_stats: list[PairStats],
    features: list[LocalFeatures],
) -> tuple[pycolmap.Reconstruction, dict, dict]:
    """Run known-pose triangulation and derive per-frame and scene-level statistics.

    triangulate_points clears the model's points, chains tracks from the DB matches, and
    filters at COLMAP defaults (4.0 px reprojection, 1.5 deg angle) inside its own
    pipeline — no extra filter call belongs here.
    """
    verified_dir = output_dir / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    # image dir is unused (keypoints live in the DB) but must exist
    verified = pycolmap.triangulate_points(recon, str(db_path), str(output_dir), str(verified_dir))
    logger.info("Verification: triangulated %d points", verified.num_points3D())

    # Per-frame survival + reprojection error via each frame's track observations
    frame_stats: dict[str, dict] = {}
    id_to_pos = {iid: k for k, iid in enumerate(sorted(verified.images))}
    for image_id in sorted(verified.images):
        image = verified.images[image_id]
        errors = [
            verified.points3D[p2d.point3D_id].error
            for p2d in image.points2D
            if p2d.has_point3D()
        ]
        n_kpts = len(features[id_to_pos[image_id]].keypoints)
        frame_stats[image.name] = {
            "n_keypoints": int(n_kpts),
            "n_tracks": len(errors),
            "track_survival": (len(errors) / n_kpts) if n_kpts else 0.0,
            "mean_reproj_error_px": float(np.mean(errors)) if errors else None,
        }

    track_lengths = [p.track.length() for p in verified.points3D.values()]
    reproj_errors = [p.error for p in verified.points3D.values()]
    inlier_ratios = [p.num_inliers / p.num_matches for p in pair_stats if p.num_matches]
    summary = {
        "n_points": int(verified.num_points3D()),
        "n_pairs": len(pair_stats),
        "track_length": _distribution(track_lengths),
        "reproj_error_px": _distribution(reproj_errors),
        "pair_inlier_ratio": _distribution(inlier_ratios),
        "pair_rot_error_deg": _distribution(p.rot_error_deg for p in pair_stats),
        "pair_t_direction_error_deg": _distribution(p.t_direction_error_deg for p in pair_stats),
    }
    return verified, frame_stats, summary


def _write_report(result: VerificationResult, path: Path) -> None:
    """Serialize pair/frame/summary stats to verification.json (nan -> null)."""

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    payload = _clean(
        {
            "pair_stats": [asdict(p) for p in result.pair_stats],
            "frame_stats": result.frame_stats,
            "summary": result.summary,
        }
    )
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Verification report written to %s", path)
```

- [ ] **Step 4: Run all verification tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -p no:randomly -v`
Expected: all PASS. The Tier 1 test still passes — `verify_reconstruction`'s flow is unchanged, only the two helpers grew real bodies.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/verification.py tests/geometry/test_verification.py
git commit -m "feat(geometry): known-pose triangulation + Tier 2 stats + verification.json"
```

---

### Task 6: negative control — perturbed poses must be flagged

This is the mandatory mutation test: a verifier validated only on good poses is unvalidated.

**Files:**
- Test: `tests/geometry/test_verification.py`

- [ ] **Step 1: Write the test (it should pass immediately if Tasks 4–5 are correct — if it fails, the verifier is broken, not the test)**

```python
def _rot_x(deg: float) -> np.ndarray:
    """Rotation about x, degrees."""
    a = np.radians(deg)
    return np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32
    )


def test_negative_control_perturbed_pose_flagged(tmp_path):
    """+2 deg rotation on one camera shows up in exactly that camera's pair errors and survival."""
    from collab_splats.geometry.verification import verify_reconstruction

    _, extrinsics, kps = _synthetic_scene(n_cams=5)
    bad = 2
    perturbed = extrinsics.copy()
    perturbed[bad, :3, :3] = _rot_x(2.0) @ perturbed[bad, :3, :3]

    # Features come from the TRUE geometry; only the model's pose for frame `bad` lies.
    result = verify_reconstruction(
        recon=_make_recon(perturbed),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    bad_name = f"frame_{bad:05d}"
    for p in result.pair_stats:
        involved = bad_name in (p.name1, p.name2)
        if involved:
            # The epipolar estimate follows the matches (truth), so it disagrees with the
            # model's perturbed relative pose by ~the injected 2 degrees.
            assert p.rot_error_deg > 1.0, f"{p.name1}-{p.name2} not flagged: {p.rot_error_deg}"
        else:
            assert p.rot_error_deg < 0.2, f"clean pair {p.name1}-{p.name2}: {p.rot_error_deg}"
    # Tier 2: reprojection through the wrong pose kills that frame's observations
    clean_survival = [
        s["track_survival"] for n, s in result.frame_stats.items() if n != bad_name
    ]
    assert result.frame_stats[bad_name]["track_survival"] < min(clean_survival)
```

- [ ] **Step 2: Run it**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -k negative_control -p no:randomly -v`
Expected: PASS. If the survival assertion is flaky at exactly 2° (COLMAP may still triangulate through 4 px gates at this focal length — 2° ≈ 17 px at f=500, so it should not be), record the measured margin in the test as a comment rather than loosening blindly; if rot-error assertions fail, debug the verifier — this test is the point of the feature.

- [ ] **Step 3: Run the full geometry suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -p no:randomly -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/geometry/test_verification.py
git commit -m "test(geometry): negative control — perturbed pose flagged by both tiers"
```

---

### Task 7: pipeline wiring — `verify` leaf stage, config boolean, storage contract

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py`
- Modify: `configs/base.yaml`
- Modify: `collab_splats/remote/sources.py`
- Modify: `configs/README.md`
- Test: `tests/wrapper/test_verify_stage.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/wrapper/test_verify_stage.py`:

```python
"""Verify-stage wiring: leaf-stage registration + config default."""

from pathlib import Path

import yaml

from collab_splats.wrapper.reconstructor import _STAGE_DEPS, _STAGE_ORDER, LEAF_STAGES

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_verify_is_a_leaf_stage():
    """verify is registered, depends only on pointcloud, and is re-runnable on its own."""
    assert "verify" in _STAGE_ORDER
    assert _STAGE_DEPS["verify"] == ["pointcloud"]
    assert "verify" in LEAF_STAGES


def test_geometric_verification_defaults_off():
    """Ships off until the first measured report (spec: Validation gates the default)."""
    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text())
    assert cfg["pointcloud"]["geometric_verification"] is False


def test_database_db_not_pushed():
    """database.db is a rebuildable local artifact — excluded from GCS pushes."""
    from collab_splats.remote.sources import PUSH_EXCLUDES

    assert "/*/colmap/database.db" in PUSH_EXCLUDES
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_verify_stage.py -p no:randomly -v`
Expected: 3 FAIL.

- [ ] **Step 3: Implement**

`configs/base.yaml` — under `pointcloud:`, directly after the `use_multiview_confidence: false` line, add:

```yaml
  # Geometric verification: triangulate independent feature tracks (the localization
  # extractor's — shared zarr cache, one extraction serves both stages) against the final
  # poses via pycolmap. Writes <backend>/colmap/{verified/, verification.json, database.db}.
  # Reports only — never gates the pipeline or mutates model outputs (feedforward.zarr,
  # sparse_pc.ply, mesh untouched). Off until the first measured report justifies a default.
  geometric_verification: false
```

`collab_splats/remote/sources.py` — extend `PUSH_EXCLUDES` (keep the anchoring comment discipline; unanchored patterns match at ANY depth):

```python
PUSH_EXCLUDES = (
    "/semantics/**",
    # COLMAP match database is a local build artifact, rebuildable from the zarr feature
    # cache + poses (geometry/verification.py). Anchored at <backend>/colmap depth.
    "/*/colmap/database.db",
    "*.[Mm][Pp]4",
    "*.[Mm][Oo][Vv]",
    "*.[Aa][Vv][Ii]",
)
```

`collab_splats/wrapper/reconstructor.py`:

1. Stage registration (lines 46–53):

```python
_STAGE_ORDER = ["preproc", "pointcloud", "semantics", "mesh", "localize", "verify"]
_STAGE_DEPS: dict[str, list[str]] = {
    "preproc": [],
    "pointcloud": ["preproc"],
    "semantics": ["pointcloud"],
    "mesh": ["pointcloud"],
    "localize": ["pointcloud"],
    # verify reuses the localize feature cache but builds it itself when absent, so its
    # only hard dependency is the reconstruction
    "verify": ["pointcloud"],
}
```

(`LEAF_STAGES` is derived — `verify` joins automatically.)

2. New method after `build_localization_db` (mirror `mesh()`'s structure):

```python
    def verify(self, overwrite: bool = False) -> Path:
        """Geometrically verify poses/points: pycolmap triangulation over feature tracks.

        Reuses the localization extractor's zarr feature cache (building it if absent) and
        writes colmap/{verified/, verification.json, database.db}. Reports only — nothing
        upstream is mutated.
        """
        out_json = self.backend_dir / "colmap" / "verification.json"
        if not overwrite and self._stage_output_exists("verify"):
            logger.info("Verification exists at %s, skipping", out_json)
            return out_json

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # One extractor serves localization and verification by design — the cache is
        # keyed by extractor, so sharing it means one extraction pass, zero drift.
        self.build_localization_db()
        extractor_name = self.config["localization"]["extractor"]

        # Heavy deps kept inline so the module imports without GPU/model libs
        from collab_splats.geometry.verification import verify_reconstruction
        from collab_splats.localization.extractors import BaseLocalExtractor
        from collab_splats.localization.localizer import load_reconstruction_features

        features, ids, _ = load_reconstruction_features(
            self.backend_dir / "feedforward.zarr", extractor_name
        )
        # The cache ids are frame_XXXXXX.jpg, the reconstruction registers frame_XXXXXX
        # (no extension) — compare stems so a reordered/rebuilt cache cannot slip through.
        recon = result.reconstruction
        recon_names = [recon.images[i].name for i in sorted(recon.images)]
        if [Path(n).stem for n in ids] != [Path(n).stem for n in recon_names]:
            raise ValueError(
                "Feature cache and reconstruction disagree on frame order/naming — "
                "rebuild the localization DB (overwrite=True)."
            )
        matcher = BaseLocalExtractor.get(extractor_name)()
        # v1 wiring uses sequential pairs only (loop_descriptors=None): the verify stage
        # must run against a processed scene where only colmap/ + feedforward.zarr are
        # guaranteed, and global retrieval descriptors are not cached anywhere. The loop
        # path exists and is tested (select_loop_pairs); callers with descriptors (eval,
        # future LC-aware wiring) pass them explicitly.
        verify_reconstruction(
            recon=recon,
            features=features,
            matcher=matcher,
            output_dir=self.backend_dir / "colmap",
        )
        logger.info("Verification written to %s", out_json)
        return out_json
```

3. `_stage_output_exists` — add before the final `return False`:

```python
        if stage == "verify":
            return (self.backend_dir / "colmap" / "verification.json").exists()
```

4. `run_pipeline` — default-stage assembly gains:

```python
            if self.config["pointcloud"]["geometric_verification"]:
                stages.append("verify")
```

and the dispatch loop gains:

```python
            elif stage == "verify":
                self.verify(overwrite=overwrite)
```

and the docstring's stage list becomes `["preproc", "pointcloud", "semantics", "mesh", "localize", "verify"]`.

`configs/README.md` — append to the "Re-running one stage against a processed scene" section:

```markdown
`verify` is a leaf stage: `--stages verify` re-runs geometric verification against a
processed scene (needs `colmap/` + `feedforward.zarr` locally). Outputs under
`<backend>/colmap/`: `verified/` (COLMAP model whose points carry real feature tracks;
poses/cameras identical to `sparse/0`), `verification.json` (per-pair epipolar +
relative-pose stats, per-frame track survival and reprojection error), and `database.db`
(local build artifact, excluded from pushes). `sparse/0` is never modified.
```

- [ ] **Step 4: Run tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_verify_stage.py tests/wrapper/ tests/remote/ -p no:randomly -v`
Expected: new tests PASS; pre-existing wrapper/remote tests unchanged. If any pre-existing test enumerates stage names, update it to include `verify` (that is the test doing its job).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py configs/base.yaml collab_splats/remote/sources.py configs/README.md tests/wrapper/test_verify_stage.py
git commit -m "feat(wrapper): geometric_verification boolean + verify leaf stage"
```

---

### Task 8: measurement script + first experiment (chess/seq-01)

**Files:**
- Create: `evals/scripts/eval_verification.py`

Compute script — CLI/tmux only, never a notebook. It measures an **already reconstructed** scene (prerequisite: run the pipeline on 7-Scenes chess/seq-01 with `localization.enabled: true` so the cache exists, or let `verify()` build it).

- [ ] **Step 1: Write the script**

Create `evals/scripts/eval_verification.py`:

```python
"""Measure geometric verification on a reconstructed scene against 7-Scenes ground truth.

Reports (all distributions median/p90/p99 + out10 where applicable):
  Tier 1: per-pair rotation / translation-direction errors — estimated-vs-model,
          and (with --gt_dir) estimated-vs-GT and model-vs-GT for the same pairs.
  Tier 2: triangulated-vs-model relative depth agreement at track pixels (scale-free,
          reference-free control), triangulated-vs-GT and model-vs-GT after one global
          median scale, and out10 for each.
  Yield:  point count, track lengths, per-frame survival (from verification.json).
  Negative control: --perturb_deg rotates every 5th pose; those frames must be flagged.

Usage (tmux, never a notebook):
  /opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
      --backend_dir data/outputs/<scene>/vggt_omega \
      --gt_dir /data/7scenes/chess/seq-01 \
      --extractor xfeat --out evals/results/verification/chess_xfeat.json
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from collab_splats.geometry.verification import (
    _pair_pose_errors,
    verify_reconstruction,
)
from collab_splats.localization.extractors import BaseLocalExtractor
from collab_splats.localization.localizer import load_reconstruction_features
from collab_splats.pointcloud.feedforward.base import (
    FeedforwardResult,
    build_pycolmap_reconstruction,
)
from collab_splats.preproc.frame_store import FrameStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_7SCENES_INVALID_DEPTH = 65535  # sentinel in 7-Scenes depth PNGs (uint16 millimetres)


def _rot_x(deg: float) -> np.ndarray:
    """Rotation about x, degrees."""
    a = np.radians(deg)
    return np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float64
    )


def _recon_to_arrays(recon: pycolmap.Reconstruction):
    """Extract (extrinsics (N,3,4), intrinsics (N,3,3), names, W, H) from a reconstruction."""
    ids = sorted(recon.images)
    extr, intr, names = [], [], []
    for iid in ids:
        im = recon.images[iid]
        cam = recon.cameras[im.camera_id]
        extr.append(im.cam_from_world().matrix())  # (3, 4)
        fx, fy, cx, cy = cam.params  # PINHOLE
        intr.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
        names.append(im.name)
    cam0 = recon.cameras[recon.images[ids[0]].camera_id]
    return np.stack(extr).astype(np.float32), np.stack(intr), names, cam0.width, cam0.height


def _perturb(recon: pycolmap.Reconstruction, deg: float) -> tuple[pycolmap.Reconstruction, list[str]]:
    """Rebuild the reconstruction with every 5th pose rotated by deg (negative control)."""
    extr, intr, names, w, h = _recon_to_arrays(recon)
    bad = list(range(0, len(names), 5))
    for i in bad:
        extr[i, :3, :3] = (_rot_x(deg) @ extr[i, :3, :3]).astype(np.float32)
    out = build_pycolmap_reconstruction(
        pts3d=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=extr,
        intrinsics=intr,
        image_width=w,
        image_height=h,
        image_names=names,
    )
    return out, [names[i] for i in bad]


def _load_gt_poses(gt_dir: Path, frame_indices) -> dict[str, np.ndarray]:
    """7-Scenes frame-XXXXXX.pose.txt (c2w) -> w2c 4x4, keyed by reconstruction image name."""
    poses = {}
    for fi in frame_indices:
        p = gt_dir / f"frame-{int(fi):06d}.pose.txt"
        c2w = np.loadtxt(p).reshape(4, 4)
        poses[f"frame_{int(fi):06d}"] = np.linalg.inv(c2w)
    return poses


def _gt_depth(gt_dir: Path, frame_idx: int) -> np.ndarray:
    """7-Scenes depth PNG in metres, invalid pixels = nan."""
    d = cv2.imread(str(gt_dir / f"frame-{int(frame_idx):06d}.depth.png"), cv2.IMREAD_UNCHANGED)
    d = d.astype(np.float64)
    d[d == _7SCENES_INVALID_DEPTH] = np.nan
    d[d == 0] = np.nan
    return d / 1000.0


def _dist(v) -> dict | None:
    """median/p90/p99 (+out10 as the fraction > 0.10 for relative errors)."""
    v = np.asarray(list(v), dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
        "out10": float(np.mean(v > 0.10)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend_dir", type=Path, required=True, help="e.g. .../<scene>/vggt_omega")
    ap.add_argument("--gt_dir", type=Path, default=None, help="7-Scenes seq dir (poses + depth)")
    ap.add_argument("--extractor", default="xfeat", help="localization extractor registry key")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--perturb_deg", type=float, default=0.0, help="negative control rotation")
    ap.add_argument("--out", type=Path, required=True, help="results JSON path")
    args = ap.parse_args()

    # ── Load reconstruction (pose authority) + feature cache + model depth ──
    recon = pycolmap.Reconstruction()
    recon.read(str(args.backend_dir / "colmap" / "sparse" / "0"))
    features, ids, _ = load_reconstruction_features(
        args.backend_dir / "feedforward.zarr", args.extractor
    )
    ff = FeedforwardResult.load_zarr(args.backend_dir / "feedforward.zarr")
    frame_indices = FrameStore.open(args.backend_dir.parent / "frames.zarr").frame_indices()

    perturbed_names: list[str] = []
    if args.perturb_deg:
        recon, perturbed_names = _perturb(recon, args.perturb_deg)

    # ── Run verification into a results-local dir (never into the scene's colmap/) ──
    work_dir = args.out.parent / (args.out.stem + "_work")
    matcher = BaseLocalExtractor.get(args.extractor)()
    result = verify_reconstruction(
        recon=recon, features=features, matcher=matcher, output_dir=work_dir, window=args.window
    )
    report: dict = {
        "backend_dir": str(args.backend_dir),
        "extractor": args.extractor,
        "window": args.window,
        "perturb_deg": args.perturb_deg,
        "perturbed_frames": perturbed_names,
        "summary": result.summary,
        "frame_stats": result.frame_stats,
    }

    # ── Tier 1 vs GT: same pairs, three comparisons ──
    if args.gt_dir is not None:
        gt_w2c = _load_gt_poses(args.gt_dir, frame_indices)
        name_to_image = {recon.images[i].name: recon.images[i] for i in recon.images}
        est_vs_model, model_vs_gt = [], []
        for p in result.pair_stats:
            if np.isnan(p.rot_error_deg):
                continue
            # GT relative pose for the same pair, in w2c convention
            T_rel = gt_w2c[p.name2] @ np.linalg.inv(gt_w2c[p.name1])
            gt_rel = pycolmap.Rigid3d(
                pycolmap.Rotation3d(T_rel[:3, :3]), T_rel[:3, 3]
            )
            im1, im2 = name_to_image[p.name1], name_to_image[p.name2]
            model_rel = im2.cam_from_world() * im1.cam_from_world().inverse()
            model_vs_gt.append(_pair_pose_errors(model_rel, gt_rel)[0])
            est_vs_model.append(p.rot_error_deg)
        # Two columns tell the story together: if estimated-vs-model tracks model-vs-GT
        # pair-by-pair, the epipolar estimate is seeing the same pose errors GT sees.
        report["tier1"] = {
            "model_vs_gt_rot_deg": _dist(model_vs_gt),
            "estimated_vs_model_rot_deg": _dist(est_vs_model),
        }

    # ── Tier 2 depth accuracy at track pixels ──
    verified = result.reconstruction
    name_to_pos = {f"frame_{int(fi):06d}": k for k, fi in enumerate(frame_indices)}
    model_h, model_w = ff.depth.shape[1:3]
    cam0 = recon.cameras[recon.images[sorted(recon.images)[0]].camera_id]
    sx, sy = model_w / cam0.width, model_h / cam0.height  # original-res px -> model grid

    tri_vs_model, tri_vs_gt_raw, model_vs_gt_raw = [], [], []
    for point in verified.points3D.values():
        for el in point.track.elements:
            image = verified.images[el.image_id]
            pos = name_to_pos[image.name]
            px = image.points2D[el.point2D_idx].xy  # original-res [x, y]
            # Triangulated depth: point through this frame's (model) pose
            z_tri = (image.cam_from_world() * point.xyz)[2]
            # Model depth: nearest sample on the model-resolution grid
            mx, my = int(round(px[0] * sx)), int(round(px[1] * sy))
            if not (0 <= mx < model_w and 0 <= my < model_h):
                continue
            z_model = float(ff.depth[pos, my, mx])
            if z_model <= 0 or z_tri <= 0:
                continue
            tri_vs_model.append(abs(z_tri - z_model) / z_model)
            if args.gt_dir is not None:
                gt = _gt_depth(args.gt_dir, frame_indices[pos])
                gy, gx = int(round(px[1])), int(round(px[0]))
                if 0 <= gy < gt.shape[0] and 0 <= gx < gt.shape[1] and np.isfinite(gt[gy, gx]):
                    tri_vs_gt_raw.append((z_tri, gt[gy, gx]))
                    model_vs_gt_raw.append((z_model, gt[gy, gx]))

    report["tier2_depth"] = {"triangulated_vs_model_rel": _dist(tri_vs_model)}
    if tri_vs_gt_raw:
        # One GLOBAL median scale (backbones are non-metric); per-frame scaling would
        # hide exactly the per-frame pose errors we are trying to see.
        s = float(np.median([g / z for z, g in model_vs_gt_raw]))
        report["tier2_depth"]["global_scale"] = s
        report["tier2_depth"]["triangulated_vs_gt_rel"] = _dist(
            abs(z * s - g) / g for z, g in tri_vs_gt_raw
        )
        report["tier2_depth"]["model_vs_gt_rel"] = _dist(
            abs(z * s - g) / g for z, g in model_vs_gt_raw
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Report: %s", args.out)
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report.get("tier2_depth", {}), indent=2))


if __name__ == "__main__":
    main()
```

Before committing, sanity-fix against reality (these are the known soft spots, resolve them by reading the actual code, not by guessing): (a) `FeedforwardResult.load_zarr` flag names for loading depth (`load_images`/`load_world_points` exist; depth loads by default — confirm), (b) `GT depth caveat`: 7-Scenes depth is Kinect-registered to color only approximately — it is a noise floor, not truth; the reference-free `triangulated_vs_model_rel` column is the primary signal, (c) `pycolmap.Rigid3d(Rotation3d(...), t)` ctor matches `build_pycolmap_reconstruction`'s usage — copy its exact spelling.

- [ ] **Step 2: Smoke the script on the synthetic path**

Run: `/opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py --help`
Expected: usage text, no import errors.

- [ ] **Step 3: Commit**

```bash
git add evals/scripts/eval_verification.py
git commit -m "feat(evals): geometric verification measurement script"
```

- [ ] **Step 4: First experiment (tmux — compute; do NOT run in a side shell during other heavy jobs)**

Prerequisite once: a chess/seq-01 reconstruction with the localization cache. If none exists under `data/outputs/` or `environments-processed/`, build one (check `docs/examples/run_pipeline.py --help` for the exact flags; ~60 frames matches the mv-confidence report's setup).

```bash
tmux new -s verif_eval
# XFeat (fast) — accuracy + yield
/opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
  --backend_dir <chess_scene>/vggt_omega --gt_dir <7scenes>/chess/seq-01 \
  --extractor xfeat --out evals/results/verification/chess_xfeat.json
# LoMa (heavy) — extractor comparison
/opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
  --backend_dir <chess_scene>/vggt_omega --gt_dir <7scenes>/chess/seq-01 \
  --extractor loma --out evals/results/verification/chess_loma.json
# Negative control on the real scene — perturbed frames must be flagged
/opt/venv/reconstruction/bin/python evals/scripts/eval_verification.py \
  --backend_dir <chess_scene>/vggt_omega --gt_dir <7scenes>/chess/seq-01 \
  --extractor xfeat --perturb_deg 2.0 --out evals/results/verification/chess_negctl.json
```

Record the numbers in `docs/superpowers/specs/2026-08-14-geometric-verification-measured-report.md` (append, never replace concurrent work). Decision inputs, per the spec's falsification clause: if triangulated points are not measurably more accurate than model points where they disagree, the verified cloud is an audit artifact only and the BA follow-on loses its premise.

---

## Execution notes

- Tasks 1–2 are independently landable; 3→4→5→6 are sequential; 7 needs 2+5; 8 needs 7.
- Tests in Tasks 4–6 exercise real pycolmap (CPU, sub-second on the synthetic scene) — no GPU, no model downloads. Task 1's tests download DISK/XFeat/LoMa weights on first run (cached thereafter).
- Never run repo-wide `black .` (venv black is newer than repo formatting); format only touched files if needed.
- The dashboard smoke gate applies only if a dashboard file is touched — none is in this plan.
