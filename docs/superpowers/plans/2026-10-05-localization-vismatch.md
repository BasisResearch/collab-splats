# Localization on vismatch: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every localization path on vismatch's cached-feature fast path, add matcher tracks for BA (`geometry/tracks.py`), and bring `localization` under the package contract.

**Architecture:**
- `LocalMatcher` keeps only xfeat and loma, the models that match cached features.
- `CameraLocalizer` has one match path: cached features + `match()`, refs from DINO-SALAD or from the caller, ref px mapped through the preprocess crop.
- `geometry/tracks.py` promotes the measured star-chain prototype: full-res extract → model grid → sequential + retrieval pairs → depth filter → pycolmap verify → star queries. BA reaches it through `track_source`.

**Tech Stack:** vismatch (pinned, `known_models`), pycolmap (`Database`, `verify_matches`, `DatabaseCache`), torch, zarr v3, pytest.

**Spec:** [2026-10-05-localization-vismatch-design.md](../specs/2026-10-05-localization-vismatch-design.md)

---

## Ground rules (every task)

- **Worktree:** `/workspace/collab-splats/.worktrees/localization`, branch `clean/localization`.
  - Run everything from there: `cd /workspace/collab-splats/.worktrees/localization && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python ...`.
  - Before the first test run of a session, print `collab_splats.__file__` once and confirm it points into the worktree.
- **Python:** `/opt/venv/reconstruction/bin/python` only. Check `python -c "import gsplat; print(gsplat.__version__)"` at session start (the shared venv flips it).
- **Never edit** site-packages, `third_party/`, or `/workspace/vismatch` (it holds uncommitted `feat/batch-forward` work; no reset, no stash).
- **Never delete** `scratch/match_tracks_prototype/`. Task 0 copies from it.
- **Commits:** `git add <paths>` then `git commit --only <paths> -m ...`. Files under `docs/superpowers/` need `git add -f`. Trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- **Pytest gates:** run in the foreground with a timeout and keep the exit code. No `| tail`.
  - Pattern: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest <paths> -q -p no:cacheprovider; echo EXIT=$?`
- **Contract style** on every touched file:
  - Docstrings: `"""` on its own line; one summary line ≤ 100 chars that does not restate the name; `- ` bullets; `Args:`/`Returns:`/`Raises:` for public defs. Private defs get the summary plus optional bullets only.
  - Every parameter and return is annotated.
  - Block comments are ONE plain line (memory override of the CLAUDE.md header+bullet rule).
  - A blank line goes above every block comment and around every `for`/`if`/`with`/`try`.
  - One call per line, never `f(g(x))`.
  - Tunables are keyword defaults, never module constants. `logger`, not `print`. No inline imports.
- **Formatting:** `black` and `isort` on touched files only, never repo-wide.
- **Kill by pid**, never `pkill -f`. No parallel heavy GPU jobs. Heavy gates run in tmux.
- **Hands off** the main checkout's uncommitted files: `CLAUDE.md`, `README.md`, the dead-code spec, `ref_image.jpg`.

## Deviations from the spec, found while planning

1. **`image_paths` are stems, not paths.** BA's `image_paths` are `Path("frame_000042")` stems (`feedforward/base.py:115`), so they cannot locate full-res frames. `refine` therefore gains a separate `frame_paths` keyword, which the reconstructor resolves via `store_rows`. `image_paths` keeps keying the cache, so existing VGGSfM keys stay stable apart from the added `track_source` field.
2. **Lazy chunking uses `itertools.islice`.** `batch_iterator` needs sequences, and the localizer's `images` is a lazy iterable (zero reads on a cache hit), so it chunks with `itertools.islice`. Callers still use `batch_iterator` over index lists.
3. **`from_pointcloud` keeps a `config` keyword**, which carries the solver options `__init__` already takes. Its `extractor_name` comes from `extractor.model_name`.
4. **`_pairs` skips suppressed candidates.** The prototype takes `argsort[:k]` blindly. With fewer than k frames outside the nms band, that picks `-inf` entries, including `(a, a)` self pairs. `tracks._pairs` drops non-finite candidates. Parity (Task 3) is unaffected: every frame of the 12-frame scene has ≥ 2 candidates, and so does every frame at gh1k scale.
5. **`LocalMatcher.to_device(features)`** is the one device move (spec §2 "move once"). The localizer and `tracks` call it; `to_numpy` replaces every `.numpy()` on features.
6. **`from_pointcloud` cache stamps.** The DB stamps `hw` and `max_num_keypoints` beside `image_paths`; any mismatch rebuilds.
7. **Reference features stay on CPU.** Each chosen ref moves to the matcher device per `localize`, which bounds memory at large N.
8. **`update_index` is persist-only.** It writes the DB; the in-memory localizer is unchanged.
9. **Gates ran on the 100-frame tutorial scene.** The gh1k data is gone.
   - gh1k absolute budgets unverified: extract ≤15 s, match ≤28 s, 108,518 tracks
10. **BA vggsfm gate relaxed** to within base-vs-base noise on shared cached tracks.
    - base-vs-branch 5.5e-6 vs base-vs-base 7.2e-6
    - VGGSfM extraction is unseeded
11. **Full-res → model pixel maps differ.** `tracks` uses pixel-center; the localizer uses pixel-corner, as the spec mandates.
    - 0.36 model px apart; harmless at PnP `max_error` 50
12. **Matcher `track_source` is refused with LC.** Window BA (`loop_closure/wrapper.py`) calls `ba.refine` without `frame_paths`, since per-window frame paths are not threaded.
    - xfeat / loma + LC raises ValueError at config validation
    - otherwise each window's refine raises, is caught, and BA silently no-ops
13. **Query camera is SIMPLE_PINHOLE; the refined K is returned.** `_solve_pnp` used PINHOLE and returned the seed K while pycolmap refined the focal in place.
    - bug pre-existing at base, not introduced by this branch
    - one focal (fx == fy); principal point held at the input K's, as COLMAP always fixes it
    - `refine_extra_params` defaults False: a no-op for SIMPLE_PINHOLE
    - failed path keeps the input K; dashboard now stores the returned K, calibrated or seeded
14. **Localized frames persist pose and id only.** `add_localized_frame` no longer stores query features or K, and `LocalizationResult.query_features` is gone. Nothing read them; only the pose and the path are consumed.
    - `localized/` group: `extrinsics` + `image_paths` and `provenance` attrs
    - `update_index`, `clear_localized_frames`, `_append_csr` kept
15. **Query shrunk to the reference long side.** `localize` matches on a query no larger than the references; K and px return on the original grid.
    - the pure-scale px back-map carries the same half-pixel convention gap deviation 11 accepts (~0.5·(s-1) px, negligible at PnP `max_error` 50)
16. **`sample_world_points` lives in `geometry.projection`.** `tracks` and the localizer import it there.
    - breaks the cycle localizer → geometry → bundle_adjustment → tracks → localizer
    - no `localization` re-export
17. **`localization.top_k` config key removed.** `CameraLocalizer(top_k=8)` is the only knob.
    - `batch_size` left `from_pointcloud` / `update_index`
18. **`preproc.frames.read_frames_chunked` streams frames.** The reconstructor and the dashboard draw frames through it; `geometry/tracks` keeps its `batch_iterator` + `read_frames` loop, since `extract` takes whole batches.
    - in place of `store_rows` + `batch_iterator`

## File map

| File | Change |
|---|---|
| `tests/geometry/data/match_tracks_prototype.py.txt` | create: verbatim copy of `scratch/match_tracks_prototype/match_tracks.py` (`.txt` keeps it out of the import-style scan) |
| `collab_splats/localization/extractors.py` | rewrite: xfeat/loma only, `max_num_keypoints`, `device`, `to_device`, contract docstrings |
| `collab_splats/geometry/tracks.py` | create: `build_tracks` + private helpers |
| `collab_splats/geometry/bundle_adjustment.py` | modify: `track_source`, `extract_tracks` dispatch, `refine(frame_paths=)`, cache key |
| `collab_splats/localization/localizer.py` | rewrite: one match path, `refs=`, crop map, `global_desc`, `from_pointcloud`, `localization_db_exists` |
| `collab_splats/localization/__init__.py`, `retrieval.py`, `viz.py` | contract docstrings |
| `collab_splats/reconstructor.py` | `_build_localization_db` → `from_pointcloud`, chunked `read_frames`; `_localization_db_exists` → `localization_db_exists`; `refine` passes `frame_paths` |
| `collab_splats/dashboard/pipeline.py` | `_build_localizer` → `from_pointcloud`, chunked `read_frames`, top-level imports; `_load_feedforward_result` drops `load_images` |
| `collab_splats/dashboard/localize.py` | `_METHODS = ["xfeat", "loma"]` |
| `configs/base.yaml` | matcher comment; `bundle_adjustment.track_source: vggsfm` |
| `tests/localization/conftest.py` | create: `FakeSalad` autouse + `StubMatcher` fixture |
| `tests/localization/test_*.py`, `tests/geometry/test_tracks.py`, `tests/geometry/test_bundle_adjustment.py`, `tests/reconstructor/*`, `tests/dashboard/*` | update / create |
| `tests/test_docstring_contract.py` | `PACKAGES += ("localization",)` |
| `tests/test_import_cycles.py` | create: fresh-process imports |

---

### Task 0: Copy the prototype into the worktree

**Files:**
- Create: `tests/geometry/data/match_tracks_prototype.py.txt`

- [ ] **Step 1: Copy the files. The scratch originals stay where they are.**

```bash
cd /workspace/collab-splats/.worktrees/localization
mkdir -p tests/geometry/data
cp scratch/match_tracks_prototype/match_tracks.py tests/geometry/data/match_tracks_prototype.py.txt
sha256sum scratch/match_tracks_prototype/match_tracks.py tests/geometry/data/match_tracks_prototype.py.txt
ls scratch/match_tracks_prototype/
```

Expected: both hashes are identical, and `ls` still lists all five scratch files.

- [ ] **Step 2: Commit**

```bash
git add tests/geometry/data/match_tracks_prototype.py.txt
git commit --only tests/geometry/data/match_tracks_prototype.py.txt -m "test(geometry): vendor the star-chain match_tracks prototype as the tracks parity reference

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: `extractors.py`: xfeat and loma only

**Files:**
- Modify: `collab_splats/localization/extractors.py` (whole file)
- Test: `tests/localization/test_local_matcher.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/localization/test_local_matcher.py`)

```python
def _batch_vismatch_matcher(n_kpts: int = 8, d: int = 64) -> MagicMock:
    """
    Mock vismatch matcher that supports cached-feature matching.
    """
    m = _fake_vismatch_matcher(n_kpts=n_kpts, d=d)
    m.supports_batches = True
    return m


@patch("vismatch.get_matcher")
def test_non_batch_model_refused(mock_get):
    mock_get.return_value = _fake_vismatch_matcher()

    with pytest.raises(ValueError, match="supports_batches"):
        LocalMatcher("disk-lightglue", device="cpu")


@patch("vismatch.get_matcher")
def test_max_num_keypoints_forwarded(mock_get):
    mock_get.return_value = _batch_vismatch_matcher()

    LocalMatcher("xfeat", device="cpu", max_num_keypoints=512)

    mock_get.assert_called_once_with("xfeat", device="cpu", max_num_keypoints=512)


@patch("vismatch.get_matcher")
def test_to_device_moves_every_tensor(mock_get):
    mock_get.return_value = _batch_vismatch_matcher()
    lm = LocalMatcher("xfeat", device="cpu")
    feats = LocalFeatures(
        keypoints=torch.zeros(3, 2, dtype=torch.float64),
        descriptors=torch.zeros(3, 4),
        keypoints_normalized=torch.zeros(3, 2),
        image_size=(10, 10),
    )

    moved = lm.to_device(feats)

    assert moved.keypoints.device.type == "cpu" and moved.keypoints_normalized is not None
    assert moved.image_size == (10, 10)


@patch("vismatch.get_matcher")
def test_match_returns_native_indices(mock_get):
    m = _batch_vismatch_matcher()
    m.match.return_value = {
        "matched_kpts0": np.ones((2, 2), np.float32),
        "matched_kpts1": np.ones((2, 2), np.float32),
        "matched_idxs0": np.array([0, 3]),
        "matched_idxs1": np.array([1, 2]),
    }
    mock_get.return_value = m
    lm = LocalMatcher("xfeat", device="cpu")
    f = LocalFeatures(keypoints=torch.zeros(4, 2), descriptors=torch.zeros(4, 8), image_size=(10, 10))

    res = lm.match(f, f)

    np.testing.assert_array_equal(res.idx_q, [0, 3])
    np.testing.assert_array_equal(res.idx_db, [1, 2])
    assert res.idx_q.dtype == np.int64
```

- [ ] **Step 2: Run them and watch them fail**

Run: `timeout 300 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -k "refused or forwarded or to_device or native_indices" -p no:cacheprovider; echo EXIT=$?`

Expected: FAIL. Today a non-batch model is accepted, `get_matcher` is called without `max_num_keypoints`, and `to_device` does not exist.

- [ ] **Step 3: Rewrite `collab_splats/localization/extractors.py`**

```python
"""
Stage 2 local matching: vismatch features, cached once, matched pair by pair.

- xfeat and loma only: vismatch supports_batches, so match() reads cached features
- LocalFeatures is one cache row; MatchResult carries native keypoint-table indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import torch

import vismatch

from collab_splats.utils.torch_utils import get_device, to_numpy

logger = logging.getLogger(__name__)


########################################
# Containers
########################################


@dataclass
class LocalFeatures:
    """
    Keypoints and descriptors of one image, as the feature cache stores them.

    - scores / scales: optional per-keypoint extras; None when the model has none
    - keypoints_normalized: loma's model-grid coords, which its matcher consumes
    - image_size: (W, H) of the image the keypoints were detected in
    """

    keypoints: torch.Tensor  # (N, 2) float32 pixel xy
    descriptors: torch.Tensor  # (N, D) float32
    scores: torch.Tensor | None = None  # (N,) float32
    scales: torch.Tensor | None = None  # (N,)
    keypoints_normalized: torch.Tensor | None = None  # (N, 2)
    image_size: tuple[int, int] | None = None  # (W, H)


@dataclass
class MatchResult:
    """
    Matched pixels between a query and one reference image, with their keypoint rows.

    - idx_q / idx_db index the two keypoint tables, COLMAP's match format
    """

    query_px: np.ndarray  # (K, 2) float32 xy in the query image
    ref_px: np.ndarray  # (K, 2) float32 xy in the reference image
    idx_q: np.ndarray  # (K,) int64 into the query keypoint table
    idx_db: np.ndarray  # (K,) int64 into the reference keypoint table

    def __len__(self) -> int:
        """
        Number of matches.
        """
        return len(self.query_px)


def _empty_match() -> MatchResult:
    """
    Zero-length MatchResult with empty index arrays.
    """
    px = np.zeros((0, 2), dtype=np.float32)
    idx = np.zeros(0, dtype=np.int64)
    return MatchResult(query_px=px, ref_px=px, idx_q=idx, idx_db=idx)


########################################
# Matcher
########################################


class LocalMatcher:
    """
    One vismatch model: extract() fills the feature cache, match() pairs two cached frames.

    - the model name is data, not a subclass
    - models without vismatch supports_batches are refused: they cannot match cached features
    """

    def __init__(self, model_name: str, device: str | None = None, *, max_num_keypoints: int = 2048) -> None:
        """
        Load the vismatch model; refuse one that cannot match cached features.

        Args:
            model_name: vismatch model name, "xfeat" or "loma".
            device: torch device; None picks CUDA when available.
            max_num_keypoints: per-image keypoint cap, forwarded to vismatch unchanged.

        Raises:
            ValueError: the model lacks vismatch supports_batches.
        """
        self._model_name = model_name
        self._device = device or get_device()
        self._matcher = vismatch.get_matcher(model_name, device=self._device, max_num_keypoints=max_num_keypoints)

        # Cached-feature matching runs on vismatch's batch path only
        if not self._matcher.supports_batches:
            raise ValueError(
                f"LocalMatcher: vismatch '{model_name}' cannot match cached features "
                "(supports_batches is False); use xfeat or loma"
            )

        # Callers verify geometry with pycolmap; vismatch's homography RANSAC is wasted work
        self._matcher.skip_ransac = True

    @property
    def model_name(self) -> str:
        """
        vismatch model name; also the feature-cache key.
        """
        return self._model_name

    @property
    def device(self) -> str:
        """
        Torch device the model runs on.
        """
        return self._device

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        HxWx3 RGB to (3, H, W) on device, dtype kept; vismatch scales uint8 itself.
        """
        array = np.ascontiguousarray(image)
        tensor = torch.from_numpy(array).to(self._device)
        return tensor.permute(2, 0, 1)

    @staticmethod
    def _check_pixel_frame(kpts: np.ndarray, hw: tuple[int, int], what: str) -> None:
        """
        Refuse keypoints outside the input image's pixel frame (the 92f2e4a bug class).
        """
        if len(kpts) and (kpts.min() < -0.5 or kpts[:, 0].max() > hw[1] - 0.5 or kpts[:, 1].max() > hw[0] - 0.5):
            raise ValueError(
                f"vismatch '{what}' keypoints outside input pixel frame {hw}: "
                f"x range [{kpts[:, 0].min():.1f}, {kpts[:, 0].max():.1f}], "
                f"y range [{kpts[:, 1].min():.1f}, {kpts[:, 1].max():.1f}] — "
                "model likely returns coords at its internal resolution"
            )

    def extract(self, images: np.ndarray | list[np.ndarray]) -> LocalFeatures | list[LocalFeatures]:
        """
        Keypoints and descriptors for one image, or for every image of a list in one vismatch call.

        Args:
            images: HxWx3 RGB uint8 (or float in [0, 1]), or a list of them.

        Returns:
            One LocalFeatures for one image; a list, in order, for a list. Tensors on CPU.

        Raises:
            ValueError: the model returned keypoints outside an input image.
        """
        batch = isinstance(images, list)
        images = images if batch else [images]
        tensors = [self._to_tensor(image) for image in images]

        with torch.inference_mode():
            outs = self._matcher.extract(tensors)

        # vismatch hands back numpy or on-device tensors depending on the model
        feats = []

        for image, out in zip(images, outs):
            kpts = to_numpy(out["all_kpts0"])
            kpts = kpts.astype(np.float32, copy=False)
            self._check_pixel_frame(kpts, image.shape[:2], self._model_name)
            desc = to_numpy(out["all_desc0"])
            desc = desc.astype(np.float32, copy=False)
            norm = out.get("kpts_normalized")

            if norm is not None:
                norm = to_numpy(norm)
                norm = norm.astype(np.float32, copy=False)
                norm = torch.from_numpy(norm)

            feats.append(
                LocalFeatures(
                    keypoints=torch.from_numpy(kpts),
                    descriptors=torch.from_numpy(desc),
                    keypoints_normalized=norm,
                    image_size=(image.shape[1], image.shape[0]),
                )
            )

        return feats if batch else feats[0]

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        """
        The same features on the matcher's device, so vismatch's per-match upload is a no-op.

        Args:
            features: features from extract() or the cache.

        Returns:
            A copy whose tensors sit on the matcher's device.
        """
        moved = {}

        for name in ("keypoints", "descriptors", "scores", "scales", "keypoints_normalized"):
            value = getattr(features, name)
            moved[name] = None if value is None else value.to(self._device)

        return replace(features, **moved)

    def match(self, query: LocalFeatures, db: LocalFeatures) -> MatchResult:
        """
        Match two cached frames; index rows are native keypoint-table indices.

        Args:
            query: query image features.
            db: reference image features.

        Returns:
            Pre-RANSAC matches; empty when either side has no keypoints.
        """
        if len(query.descriptors) == 0 or len(db.descriptors) == 0:
            return _empty_match()

        query_in = self._vismatch_features(query)
        db_in = self._vismatch_features(db)

        with torch.inference_mode():
            out = self._matcher.match(query_in, db_in)

        if len(out["matched_idxs0"]) == 0:
            return _empty_match()

        query_px = to_numpy(out["matched_kpts0"])
        ref_px = to_numpy(out["matched_kpts1"])
        idx_q = to_numpy(out["matched_idxs0"])
        idx_db = to_numpy(out["matched_idxs1"])

        # MatchResult dtypes: float32 pixels, int64 keypoint indices
        query_px = query_px.astype(np.float32, copy=False)
        ref_px = ref_px.astype(np.float32, copy=False)
        idx_q = idx_q.astype(np.int64)
        idx_db = idx_db.astype(np.int64)
        return MatchResult(query_px=query_px, ref_px=ref_px, idx_q=idx_q, idx_db=idx_db)

    @staticmethod
    def _vismatch_features(feats: LocalFeatures) -> dict:
        """
        LocalFeatures as a vismatch extract() dict; keypoints_normalized feeds loma.
        """
        out = {"all_kpts0": feats.keypoints, "all_desc0": feats.descriptors, "image_size": feats.image_size}

        if feats.keypoints_normalized is not None:
            out["kpts_normalized"] = feats.keypoints_normalized

        return out
```

Notes for the implementer:
- `torch.inference_mode()` around `match` is new. vismatch already runs `match` without grad, and inference mode only skips autograd bookkeeping.
- Remove `MatchResult`'s old index-less constructor uses. Any `MatchResult(query_px=..., ref_px=...)` in tests now needs `idx_q` / `idx_db`; pass `np.arange(K)` for both.

- [ ] **Step 4: Bring the rest of `tests/localization/test_local_matcher.py` in line**
  - **Delete** these tests outright. Each one tests code §1 removed:
    - every test whose name contains `blocked`, `probe`, `recover_indices`, `match_images`, `stable_indices` or `pairwise`
    - the `NotImplementedError` "unsupported" test (`test_non_batch_model_refused` replaces it)
  - **Edit:**
    - Drop every `probe=False` argument.
    - Every test that constructs `LocalMatcher` gets `_batch_vismatch_matcher()`, not `_fake_vismatch_matcher()`. The model name in those tests becomes `"xfeat"`.
    - `assert_called_once_with("disk-lightglue", device="cpu")` becomes `assert_called_once_with("xfeat", device="cpu", max_num_keypoints=2048)`.
    - In `_fake_vismatch_matcher`, drop the `stable_indices` parameter and its refined-coordinates branch.
  - **Real-model parity tests** (they call `match_images` as the reference): the reference becomes vismatch's own pairwise call on the same matcher instance:
    ```python
    with torch.inference_mode():
        ref = lm._matcher(lm._to_tensor(img0), lm._to_tensor(img1))
    ```
    Compare `ref["matched_kpts0"]` to `lm.match(*lm.extract([img0, img1])).query_px` with the existing tolerance. Keep their skip markers.

- [ ] **Step 5: Run the file**

Run: `timeout 600 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_local_matcher.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS, `EXIT=0`. The real-model tests either pass or skip, matching their state before the change.

- [ ] **Step 6: Find the remaining callers of the deleted names**

Run: `grep -rn "probe=\|match_images\|has_stable_indices\|_recover_indices\|_VISMATCH_\|_probe_index" collab_splats tests evals configs`

Expected: hits only in `localizer.py` (Task 4 rewrites it), in tests that Tasks 4 and 5 rewrite, and in the `configs/base.yaml` comment (Task 5). Fix any other hit here.

- [ ] **Step 7: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m isort collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
/opt/venv/reconstruction/bin/python -m black collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git add collab_splats/localization/extractors.py tests/localization/test_local_matcher.py
git commit --only collab_splats/localization/extractors.py tests/localization/test_local_matcher.py -m "refactor(localization): LocalMatcher keeps xfeat/loma only, native indices, to_device

- non-batch vismatch models refused at init (ValueError)
- probe, match_images, _recover_indices, both blocklists deleted
- max_num_keypoints forwarded to vismatch; import vismatch at module top

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `geometry/tracks.py`: helpers, by TDD

**Files:**
- Create: `collab_splats/geometry/tracks.py`
- Test: `tests/geometry/test_tracks.py`

The synthetic scene below is shared by Tasks 2 and 3. It is a smooth non-planar surface seen by translating, slightly yawing cameras. Frame `i` is identified by pixel `(0, 0)` holding the value `i`, so the fake matcher and fake retrieval need no image analysis.

- [ ] **Step 1: Write the scene, the fakes and the helper tests** in `tests/geometry/test_tracks.py`

```python
"""
Matcher tracks: synthetic posed scene, helper units, star-chain parity with the prototype.
"""

import importlib.machinery
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry import tracks as tracks_mod
from collab_splats.localization.extractors import LocalFeatures, MatchResult
from collab_splats.preproc.frames import write_frames

PROTOTYPE = Path(__file__).parent / "data" / "match_tracks_prototype.py.txt"


########################################
# Synthetic scene
########################################


def _surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smooth non-planar height field z = f(x, y), world units.
    """
    return 5.0 + 0.3 * np.sin(2.0 * x) + 0.3 * np.cos(2.0 * y)


def _scene(n_frames: int = 12, height: int = 60, width: int = 80, upscale: int = 2, n_points: int = 600) -> dict:
    """
    Posed cameras over the surface: dense world points, depth, and per-frame keypoint tables.

    - keypoints are exact projections of shared surface points, restricted to [0, W-1] x [0, H-1]
    - full-res keypoints invert the size map (kp + 0.5) * W / w - 0.5
    """
    rng = np.random.default_rng(0)
    K = np.array([[70.0, 0, width / 2], [0, 70.0, height / 2], [0, 0, 1]])
    intrinsics = np.repeat(K[None], n_frames, 0).astype(np.float32)

    # Cameras translate along x and yaw slightly
    extrinsics = np.tile(np.eye(4), (n_frames, 1, 1))

    for i in range(n_frames):
        yaw = 0.01 * i
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        centre = np.array([0.1 * i, 0.0, 0.0])
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = -R @ centre

    # Dense world points: per-pixel ray / surface intersection by fixed-point iteration
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    rays_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    world_points = np.zeros((n_frames, height, width, 3))
    depth = np.zeros((n_frames, height, width))

    for i in range(n_frames):
        R = extrinsics[i, :3, :3]
        centre = -R.T @ extrinsics[i, :3, 3]
        rays = rays_cam @ R
        s = np.full((height, width), 5.0)

        for _ in range(50):
            pts = centre + rays * s[..., None]
            s = (_surface(pts[..., 0], pts[..., 1]) - centre[2]) / rays[..., 2]

        world_points[i] = centre + rays * s[..., None]
        depth[i] = s

    # Shared surface points, then their projections per frame
    xy = rng.uniform([-1.5, -1.5], [2.5, 1.5], (n_points, 2))
    points = np.column_stack([xy, _surface(xy[:, 0], xy[:, 1])])
    kp_model, kp_ids = [], []

    for i in range(n_frames):
        cam = points @ extrinsics[i, :3, :3].T + extrinsics[i, :3, 3]
        px = cam @ K.T
        px = px[:, :2] / px[:, 2:]
        inside = (px[:, 0] >= 0) & (px[:, 0] <= width - 1) & (px[:, 1] >= 0) & (px[:, 1] <= height - 1)
        kp_model.append(px[inside].astype(np.float32))
        kp_ids.append(np.flatnonzero(inside))

    kp_full = [(kp + 0.5) * upscale - 0.5 for kp in kp_model]
    return {
        "K": intrinsics,
        "extrinsics": extrinsics.astype(np.float32),
        "world_points": world_points.astype(np.float32),
        "depth": depth.astype(np.float32),
        "kp_full": kp_full,
        "kp_ids": kp_ids,
        "hw": (height, width),
        "full_hw": (height * upscale, width * upscale),
    }


def _write_store(tmp_path: Path, scene: dict) -> list[Path]:
    """
    Full-res frame store: frame i is a flat image whose pixel (0, 0) holds i.
    """
    H, W = scene["full_hw"]
    n = len(scene["extrinsics"])
    frames = np.full((n, H, W, 3), 200, np.uint8)
    frames[:, 0, 0, 0] = np.arange(n)
    return write_frames(tmp_path / "images", frames, list(range(n)))


def _model_images(scene: dict) -> np.ndarray:
    """
    (N, 3, H, W) model-grid images in [0, 1]; pixel (0, 0) of channel 0 encodes the frame.
    """
    H, W = scene["hw"]
    n = len(scene["extrinsics"])
    images = np.full((n, 3, H, W), 200 / 255, np.float32)
    images[:, 0, 0, 0] = np.arange(n) / 255
    return images


class _FakeMatcher:
    """
    Matcher over the scene's keypoint tables; descriptors carry the surface point id.
    """

    def __init__(self, scene: dict) -> None:
        self._scene = scene
        self.device = "cpu"
        self._matcher = types.SimpleNamespace(max_num_keypoints=2048)

    def _one(self, image: np.ndarray) -> LocalFeatures:
        i = int(image[0, 0, 0])
        kp = torch.from_numpy(self._scene["kp_full"][i].copy())
        ids = torch.from_numpy(self._scene["kp_ids"][i].astype(np.float32))
        return LocalFeatures(keypoints=kp, descriptors=ids[:, None], image_size=(image.shape[1], image.shape[0]))

    def extract(self, images):
        if isinstance(images, list):
            return [self._one(im) for im in images]
        return self._one(images)

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        return features

    def match(self, q: LocalFeatures, db: LocalFeatures) -> MatchResult:
        q_ids = q.descriptors[:, 0].numpy()
        db_ids = db.descriptors[:, 0].numpy()
        _, idx_q, idx_db = np.intersect1d(q_ids, db_ids, return_indices=True)
        return MatchResult(
            query_px=q.keypoints.numpy()[idx_q],
            ref_px=db.keypoints.numpy()[idx_db],
            idx_q=idx_q.astype(np.int64),
            idx_db=idx_db.astype(np.int64),
        )


class _FakeSalad(torch.nn.Module):
    """
    Global descriptor seeded by the frame index in pixel (0, 0); deterministic, CPU.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        idx = (images[:, 0, 0, 0] * 255).round().long()
        desc = torch.stack([torch.from_numpy(np.random.default_rng(int(i)).normal(size=16)) for i in idx])
        return torch.nn.functional.normalize(desc.float(), dim=-1)


@pytest.fixture
def scene() -> dict:
    return _scene()


@pytest.fixture(autouse=True)
def fake_salad(monkeypatch):
    monkeypatch.setattr(tracks_mod, "DinoSaladExtractor", _FakeSalad)


########################################
# Helpers
########################################


def test_to_model_grid_size_map():
    feats = LocalFeatures(keypoints=torch.tensor([[0.0, 0.0], [159.0, 119.0]]), descriptors=torch.zeros(2, 1), image_size=(160, 120))

    out = tracks_mod._to_model_grid(feats, (60, 80))

    np.testing.assert_allclose(out.keypoints.numpy(), [[-0.25, -0.25], [79.25, 59.25]])


def test_to_model_grid_refuses_cropped_aspect():
    feats = LocalFeatures(keypoints=torch.zeros(1, 2), descriptors=torch.zeros(1, 1), image_size=(1920, 1080))

    with pytest.raises(ValueError, match="aspect"):
        tracks_mod._to_model_grid(feats, (518, 518))


def test_extract_full_res_refuses_mixed_dirs(tmp_path, scene):
    paths = _write_store(tmp_path, scene)
    stray = tmp_path / "other" / paths[0].name

    with pytest.raises(ValueError, match="one directory"):
        tracks_mod._extract_full_res(_FakeMatcher(scene), [stray, *paths[1:]], batch_size=4)


def test_extract_full_res_chunks_preserve_order(tmp_path, scene):
    paths = _write_store(tmp_path, scene)

    feats = tracks_mod._extract_full_res(_FakeMatcher(scene), paths, batch_size=5)

    assert len(feats) == len(paths)

    for i, f in enumerate(feats):
        np.testing.assert_array_equal(f.keypoints.numpy(), scene["kp_full"][i])


def test_pairs_sequential_plus_retrieval(scene):
    pairs = tracks_mod._pairs(_model_images(scene), window=3, retrieval_k=2, retrieval_nms=4, batch_size=5)

    seq = [(a, b) for a in range(12) for b in range(a + 1, min(12, a + 4))]
    assert pairs[: len(seq)] == seq
    extra = pairs[len(seq) :]
    assert extra == sorted(extra) and all(b - a > 4 for a, b in extra)
    assert len(set(pairs)) == len(pairs)


def test_pairs_skip_suppressed_candidates_when_few_frames(scene):
    images = _model_images(scene)[:6]

    pairs = tracks_mod._pairs(images, window=1, retrieval_k=2, retrieval_nms=4, batch_size=5)

    # Only frames 0 and 5 lie outside each other's nms band; the prototype would add (a, a) self pairs
    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]


def test_depth_filter_drops_planted_outlier(scene):
    kps = [(kp + 0.5) / 2 - 0.5 for kp in scene["kp_full"]]
    lifted = tracks_mod._lift_keypoints(scene["world_points"], kps)
    _, ia, ib = np.intersect1d(scene["kp_ids"][0], scene["kp_ids"][1], return_indices=True)

    # Swap one match's reference keypoint for a far one
    ib_bad = ib.copy()
    ib_bad[0] = ib[-1]

    ok = tracks_mod._depth_filter(lifted[0][ia], lifted[1][ib_bad], kps[0][ia], kps[1][ib_bad], scene["extrinsics"][[0, 1]], scene["K"][[0, 1]], depth_tol=2.0)

    assert not ok[0] and ok[1:].all()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `timeout 300 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_tracks.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: collection ERROR, `cannot import name 'tracks' from 'collab_splats.geometry'`.

- [ ] **Step 3: Create `collab_splats/geometry/tracks.py`**

```python
"""
Matcher tracks for bundle adjustment: star queries over verified cached-feature matches.

- full-res keypoints mapped to the model grid; pairs = sequential window + DINO-SALAD retrieval
- pycolmap verifies and builds the correspondence graph; each seed keypoint's direct matches form a track
- same (tracks, vis, pts3d) triple as extract_tracks_vggsfm
- promotes the star-chain prototype that passed the gh1k runs (gh1kq60, gstar294r, 2026-10)
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
import torch

from collab_splats.geometry.projection import project
from collab_splats.localization.extractors import LocalFeatures, LocalMatcher
from collab_splats.localization.localizer import sample_world_points
from collab_splats.localization.retrieval import DinoSaladExtractor
from collab_splats.preproc.frames import frame_idx_from_path, read_frames
from collab_splats.utils.torch_utils import batch_iterator, pytorch_gc, to_numpy

logger = logging.getLogger(__name__)


########################################
# Public API
########################################


def build_tracks(
    matcher: LocalMatcher,
    images: np.ndarray | torch.Tensor,
    frame_paths: list[Path],
    world_points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    window: int = 10,
    retrieval_k: int = 20,
    retrieval_nms: int = 25,
    seed_frames: int = 30,
    min_matches: int = 50,
    depth_tol: float = 8.0,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Star tracks from verified matcher correspondences, on the model grid.

    - full-res keypoints mapped to the model grid by size; a cropped preprocess is refused
    - pairs: sequential window plus DINO-SALAD top-k outside the nms band
    - a match survives only if both directions reproject within depth_tol under the prior poses
    - pycolmap verifies; each seed keypoint plus its direct verified matches is one track
    - a frame holding two keypoints of one track is dropped from it; tracks keep >= 2 observations

    Args:
        matcher: xfeat or loma LocalMatcher.
        images: (N, 3, H, W) RGB in [0, 1] on the model grid; retrieval only.
        frame_paths: full-res store frame per model frame, all in one directory.
        world_points: (N, H, W, 3) model-grid world points.
        extrinsics: (N, 4, 4) world-to-camera.
        intrinsics: (N, 3, 3) model-grid K.
        window: sequential pair span, frames.
        retrieval_k: retrieval pairs per frame.
        retrieval_nms: frame gap at or under which retrieval pairs are suppressed.
        seed_frames: evenly spaced frames whose keypoints seed tracks.
        min_matches: verified inliers a pair needs to join the correspondence graph.
        depth_tol: symmetric transfer-error gate, model px.
        batch_size: frames per extract and retrieval chunk.

    Returns:
        tracks (N, P, 2) model px, vis (N, P) 1.0 where observed, pts3d (P, 3) world; float32.

    Raises:
        ValueError: frame_paths span directories, or a frame's aspect differs from the model grid's.
    """
    world_points = np.asarray(world_points)
    N, H, W = world_points.shape[:3]
    timings = {}

    # Full-res features mapped onto the model grid, then moved to the matcher once
    t = time.perf_counter()
    feats = _extract_full_res(matcher, frame_paths, batch_size)
    feats = [_to_model_grid(f, (H, W)) for f in feats]
    kps = [to_numpy(f.keypoints) for f in feats]
    feats = [matcher.to_device(f) for f in feats]
    lifted = _lift_keypoints(world_points, kps)
    timings["extract"] = time.perf_counter() - t

    # Candidate pairs
    t = time.perf_counter()
    pairs = _pairs(images, window, retrieval_k, retrieval_nms, batch_size)
    timings["pairs"] = time.perf_counter() - t

    with tempfile.TemporaryDirectory(prefix="tracks_") as tmp:
        db_path = Path(tmp) / "database.db"

        # Match every pair, depth-filter, and write the database
        t = time.perf_counter()
        n_raw, n_kept = _write_database(db_path, matcher, feats, kps, lifted, pairs, extrinsics, intrinsics, (H, W), depth_tol)
        timings["match"] = time.perf_counter() - t

        # Epipolar verification over exactly the matched pairs
        t = time.perf_counter()
        pairs_path = Path(tmp) / "pairs.txt"
        lines = [f"{a:05d}.png {b:05d}.png" for a, b in pairs]
        pairs_path.write_text("\n".join(lines))
        pycolmap.verify_matches(str(db_path), str(pairs_path))
        timings["verify"] = time.perf_counter() - t

        # Correspondence graph of pairs with at least min_matches inliers
        t = time.perf_counter()
        options = pycolmap.DatabaseCacheOptions()
        options.min_num_matches = min_matches
        db = pycolmap.Database.open(str(db_path))
        cache = pycolmap.DatabaseCache.create(db, options)
        graph = cache.correspondence_graph
        db.close()
        observations = _star_tracks(graph, kps, seed_frames)
        timings["chain"] = time.perf_counter() - t

    tracks, vis, pts3d = _assemble(observations, kps, lifted, N)
    logger.info(
        "tracks: %d frames, %d pairs, %d/%d matches kept by depth, %d tracks, %d observations; timings %s",
        N,
        len(pairs),
        n_kept,
        n_raw,
        tracks.shape[1],
        int(vis.sum()),
        {k: round(v, 1) for k, v in timings.items()},
    )
    return tracks, vis, pts3d


########################################
# Steps
########################################


def _extract_full_res(matcher: LocalMatcher, frame_paths: list[Path], batch_size: int) -> list[LocalFeatures]:
    """
    Features of every full-res store frame, read and extracted one chunk at a time.

    - chunked so the full-res stack (~6 GB on gh1k) is never resident
    """
    frames_dir = Path(frame_paths[0]).parent

    # read_frames serves one directory by source index
    if any(Path(p).parent != frames_dir for p in frame_paths):
        raise ValueError(f"build_tracks: frame_paths must sit in one directory, got more than {frames_dir}")

    idxs = [frame_idx_from_path(p) for p in frame_paths]
    feats = []

    for (chunk,) in batch_iterator(batch_size, idxs):
        frames = read_frames(frames_dir, chunk)
        feats += matcher.extract(list(frames))

    return feats


def _to_model_grid(features: LocalFeatures, model_hw: tuple[int, int]) -> LocalFeatures:
    """
    Full-res keypoints on the model grid, pixel-center size map (prototype map verbatim).

    - a frame whose aspect differs from the grid's by more than one model px is refused
    """
    H, W = model_hw
    w, h = features.image_size

    # Size-only mapping would misplace every keypoint of a cropped preprocess
    if abs(w * H / h - W) > 1:
        raise ValueError(f"build_tracks: frame aspect {w}x{h} differs from model grid {W}x{H} (cropped preprocess)")

    scale = torch.tensor([W / w, H / h])
    keypoints = (features.keypoints + 0.5) * scale - 0.5
    return replace(features, keypoints=keypoints)


def _pairs(
    images: np.ndarray | torch.Tensor, window: int, retrieval_k: int, retrieval_nms: int, batch_size: int
) -> list[tuple[int, int]]:
    """
    Sequential pairs within window, then sorted DINO-SALAD pairs outside the nms band.

    - retrieval: cosine top-k per frame, |a - b| <= nms suppressed, deduped against the sequential set
    """
    N = len(images)
    pairs = [(a, b) for a in range(N) for b in range(a + 1, min(N, a + window + 1))]

    # Global descriptors per chunk of model-grid images
    salad = DinoSaladExtractor()
    images = torch.as_tensor(images)
    descs = []

    for start in range(0, N, batch_size):
        chunk = images[start : start + batch_size]
        chunk = chunk.float()
        descs.append(salad(chunk))

    desc = torch.cat(descs)
    del salad
    pytorch_gc()

    # Cosine similarity with the near-diagonal band suppressed
    sim = desc @ desc.T
    sim = sim.numpy()
    frame = np.arange(N)
    sim[np.abs(frame[:, None] - frame[None, :]) <= retrieval_nms] = -np.inf

    # Top-k per frame, as (min, max) pairs not already sequential
    extra = set()

    for a in range(N):
        order = np.argsort(-sim[a])
        order = order[:retrieval_k]

        # Fewer than k candidates outside the band: argsort reaches suppressed -inf entries, self included
        for b in order:
            if not np.isfinite(sim[a, b]):
                continue

            extra.add((min(a, int(b)), max(a, int(b))))

    extra -= set(pairs)
    return pairs + sorted(extra)


def _lift_keypoints(world_points: np.ndarray, kps: list[np.ndarray]) -> list[np.ndarray]:
    """
    World point under every keypoint of every frame; NaN where the lookup is invalid.
    """
    lifted = []

    for i, kp in enumerate(kps):
        pts, valid = sample_world_points(world_points[i], kp)
        pts[~valid] = np.nan
        lifted.append(pts)

    return lifted


def _transfer_err(points: np.ndarray, px: np.ndarray, world_to_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Pixel error of world points projected into a camera vs their matched keypoints; inf if invalid.
    """
    points = points.astype(np.float64)
    world_to_cam = world_to_cam.astype(np.float64)
    K = K.astype(np.float64)
    px = px.astype(np.float64)
    points_t = torch.from_numpy(points)
    w2c_t = torch.from_numpy(world_to_cam)
    K_t = torch.from_numpy(K)
    px_t = torch.from_numpy(px)
    pixels, points_cam = project(points_t, w2c_t, K_t)
    err = torch.linalg.norm(pixels - px_t, dim=1)

    # NaN lifts and points behind the camera never pass
    err[~(points_cam[:, 2] > 0)] = torch.inf
    err[torch.isnan(err)] = torch.inf
    return err.numpy()


def _depth_filter(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    px_a: np.ndarray,
    px_b: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    depth_tol: float,
) -> np.ndarray:
    """
    Matches whose symmetric transfer error stays under depth_tol model px.

    - world_to_cam / intrinsics stack frame a then frame b
    """
    err_ab = _transfer_err(pts_a, px_b, world_to_cam[1], intrinsics[1])
    err_ba = _transfer_err(pts_b, px_a, world_to_cam[0], intrinsics[0])
    return np.maximum(err_ab, err_ba) < depth_tol


def _write_database(
    db_path: Path,
    matcher: LocalMatcher,
    feats: list[LocalFeatures],
    kps: list[np.ndarray],
    lifted: list[np.ndarray],
    pairs: list[tuple[int, int]],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    model_hw: tuple[int, int],
    depth_tol: float,
) -> tuple[int, int]:
    """
    pycolmap database of cameras, images, keypoints and depth-filtered matches; (raw, kept) counts.

    - image i is image_id i + 1, named f"{i:05d}.png", one PINHOLE camera per frame
    """
    H, W = model_hw
    db = pycolmap.Database.open(str(db_path))

    # One camera and image row per frame, with its model-grid keypoints
    for i in range(len(kps)):
        K = intrinsics[i]
        params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        camera = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=params, camera_id=i + 1)
        db.write_camera(camera, use_camera_id=True)
        row = pycolmap.Image(name=f"{i:05d}.png", camera_id=i + 1)
        row.image_id = i + 1
        db.write_image(row, use_image_id=True)
        kp = kps[i].astype(np.float64)
        db.write_keypoints(i + 1, kp)

    n_raw = n_kept = 0

    # Match each pair; keep depth-consistent matches only
    for a, b in pairs:
        m = matcher.match(feats[a], feats[b])
        n_raw += len(m)

        if len(m) == 0:
            continue

        ok = _depth_filter(
            lifted[a][m.idx_q],
            lifted[b][m.idx_db],
            kps[a][m.idx_q],
            kps[b][m.idx_db],
            extrinsics[[a, b]],
            intrinsics[[a, b]],
            depth_tol,
        )

        if not ok.any():
            continue

        idx = np.stack([m.idx_q[ok], m.idx_db[ok]], axis=1)
        idx = idx.astype(np.uint32)
        db.write_matches(a + 1, b + 1, idx)
        n_kept += int(ok.sum())

    db.close()
    return n_raw, n_kept


def _star_tracks(graph: pycolmap.CorrespondenceGraph, kps: list[np.ndarray], seed_frames: int) -> list[list[tuple[int, int]]]:
    """
    One track per seed keypoint: itself plus its direct verified matches, as (frame, keypoint) lists.

    - seeds: seed_frames evenly spaced frames; frames absent from the graph are skipped
    - a frame contributing two keypoints to one track is dropped from it; >= 2 observations kept
    """
    N = len(kps)
    seeds = np.linspace(0, N - 1, seed_frames).round().astype(int)
    tracks = []
    n_conflict = 0

    for seed in seeds:
        image_id = int(seed) + 1

        # A seed with no verified matches is absent from the graph
        if not graph.exists_image(image_id):
            logger.info("tracks: seed frame %d has no verified matches, skipped", seed)
            continue

        for idx in range(len(kps[seed])):
            corrs = graph.extract_transitive_correspondences(image_id, idx, 1)
            members = {(c.image_id, c.point2D_idx) for c in corrs}
            members.add((image_id, idx))

            # Group by frame; an ambiguous frame is dropped from the track
            per_image = defaultdict(list)

            for img_id, kp in members:
                per_image[img_id].append(kp)

            obs = []

            for img_id, kp_list in sorted(per_image.items()):
                if len(kp_list) > 1:
                    n_conflict += 1
                    continue

                obs.append((img_id - 1, kp_list[0]))

            if len(obs) >= 2:
                tracks.append(obs)

    logger.info("tracks: %d star tracks, %d ambiguous frame drops", len(tracks), n_conflict)
    return tracks


def _assemble(
    observations: list[list[tuple[int, int]]], kps: list[np.ndarray], lifted: list[np.ndarray], n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense (N, P) layout; pts3d is the world point at the first observation, invalid tracks dropped.
    """
    P = len(observations)
    tracks = np.zeros((n_frames, P, 2), np.float32)
    vis = np.zeros((n_frames, P), np.float32)
    pts3d = np.zeros((P, 3), np.float32)

    for j, obs in enumerate(observations):
        for i, k in obs:
            tracks[i, j] = kps[i][k]
            vis[i, j] = 1.0

        i0, k0 = obs[0]
        pts3d[j] = lifted[i0][k0]

    # A first observation with no world point has nothing to lift the track from
    valid = ~np.isnan(pts3d).any(axis=1)
    return tracks[:, valid], vis[:, valid], pts3d[valid]
```

Notes:
- `_write_database` carries ten parameters because it is the prototype's inner loop. Do not fold it into `build_tracks`: that would make one 150-line function.
- If `test_import_style` or isort reorders anything, accept isort's order.

- [ ] **Step 4: Run the helper tests**

Run: `timeout 300 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_tracks.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS for all six. If the depth-filter test fails on `ok[1:]`, print the errors. The bilinear lookup on the synthetic surface should stay well under 2 px; a large error means the scene's world points and keypoints disagree, which is a test bug, not a reason to loosen `depth_tol`.

- [ ] **Step 5: Check the import cycle in a fresh process**

Run: `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud, collab_splats.geometry.tracks; print('ok')"`

Expected: `ok`. A circular-import error at this point means `localizer.py` still imports from pointcloud at module level. It does not yet; Task 4 adds a `TYPE_CHECKING`-only import.

- [ ] **Step 6: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m isort collab_splats/geometry/tracks.py tests/geometry/test_tracks.py
/opt/venv/reconstruction/bin/python -m black collab_splats/geometry/tracks.py tests/geometry/test_tracks.py
git add collab_splats/geometry/tracks.py tests/geometry/test_tracks.py
git commit --only collab_splats/geometry/tracks.py tests/geometry/test_tracks.py -m "feat(geometry): matcher star tracks for BA (geometry/tracks.py)

- promotes the xfeat star-chain prototype; env vars become keyword defaults
- full-res chunked extract, sequential + DINO-SALAD pairs, world-point depth filter
- pycolmap verify_matches + DatabaseCache correspondence graph, star queries per seed keypoint

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `build_tracks` end to end, and parity with the prototype

**Files:**
- Test: `tests/geometry/test_tracks.py` (append)

- [ ] **Step 1: Write the end-to-end and parity tests**

```python
########################################
# End to end
########################################


def _build(tmp_path: Path, scene: dict, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build_tracks over the synthetic scene with small-scene keyword values.
    """
    paths = _write_store(tmp_path, scene)
    options = {"window": 3, "retrieval_k": 2, "retrieval_nms": 4, "seed_frames": 4, "min_matches": 15, "depth_tol": 2.0, "batch_size": 5}
    options.update(kwargs)
    return tracks_mod.build_tracks(_FakeMatcher(scene), _model_images(scene), paths, scene["world_points"], scene["extrinsics"], scene["K"], **options)


def test_build_tracks_recovers_star_tracks(tmp_path, scene):
    tracks, vis, pts3d = _build(tmp_path, scene)

    assert tracks.dtype == vis.dtype == pts3d.dtype == np.float32
    assert tracks.shape[1] == vis.shape[1] == len(pts3d) > 50
    assert (vis.sum(axis=0) >= 2).all()

    # Every observation reprojects onto its keypoint under the true poses
    for i in range(len(vis)):
        seen = vis[i] > 0
        cam = pts3d[seen] @ scene["extrinsics"][i, :3, :3].T + scene["extrinsics"][i, :3, 3]
        px = cam @ scene["K"][i].T
        px = px[:, :2] / px[:, 2:]
        np.testing.assert_allclose(px, tracks[i, seen], atol=0.5)


def test_build_tracks_pts3d_from_world_points(tmp_path, scene):
    tracks, vis, pts3d = _build(tmp_path, scene)
    first = vis.argmax(axis=0)

    for j in range(0, len(pts3d), 25):
        i = first[j]
        expect, ok = tracks_mod.sample_world_points(scene["world_points"][i], tracks[i, j][None])
        assert ok[0]
        np.testing.assert_allclose(pts3d[j], expect[0], atol=1e-5)


########################################
# Prototype parity
########################################


def _load_prototype():
    """
    The vendored star-chain prototype as a module.
    """
    loader = importlib.machinery.SourceFileLoader("match_tracks_prototype", str(PROTOTYPE))
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def test_build_tracks_matches_prototype(tmp_path, scene, monkeypatch):
    proto = _load_prototype()
    fake = _FakeMatcher(scene)
    paths = _write_store(tmp_path, scene)

    # Prototype arm of the gh1k runs: MT_CHAIN=star, MT_FULL_DIR set, depth filter on
    monkeypatch.setattr(proto, "LocalMatcher", lambda name, probe=False: fake)
    monkeypatch.setattr(proto, "DinoSaladExtractor", _FakeSalad)
    monkeypatch.setattr(proto, "CHAIN", "star")
    monkeypatch.setattr(proto, "QUERY_FRAMES", 4)
    monkeypatch.setattr(proto, "RETR_K", 2)
    monkeypatch.setattr(proto, "RETR_NMS", 4)
    monkeypatch.setattr(proto, "MIN_MATCHES", 15)
    monkeypatch.setattr(proto, "DEPTH_TOL", 2.0)
    monkeypatch.setattr(proto, "FULL_DIR", str(paths[0].parent))
    proto.set_poses(scene["extrinsics"], scene["K"])
    proto.set_depth(scene["depth"])
    extract = proto.make_extract("xfeat", 3, "wp")
    p_tracks, p_vis, p_pts3d = extract(torch.from_numpy(_model_images(scene)), None, scene["world_points"], None)

    tracks, vis, pts3d = _build(tmp_path / "ours", scene)

    # Same tracks, same observations; pts3d differ only by bilinear vs nearest lookup
    assert tracks.shape == p_tracks.shape
    np.testing.assert_array_equal(vis, p_vis)
    np.testing.assert_allclose(tracks, p_tracks, atol=1e-5)
    np.testing.assert_allclose(pts3d, p_pts3d, atol=0.05)
    assert proto._state["stats"][-1]["pairs"] == len(tracks_mod._pairs(_model_images(scene), 3, 2, 4, 5))
```

- [ ] **Step 2: Run them**

Run: `timeout 600 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_tracks.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS. If parity fails, debug instead of loosening:
- **Different pairs:** compare `_pairs` with the prototype's pair list line by line.
- **Different track count:** dump both obs lists for the first seed. The prototype and `_star_tracks` must iterate seeds and keypoints in the same order.
- **pycolmap verification drops pairs on the synthetic scene:** print `pycolmap.Database.open(db).read_two_view_geometry(...)` configs. Raising `n_points` is a legal fix; changing `min_matches` in only one arm is not.

- [ ] **Step 3: Commit**

```bash
/opt/venv/reconstruction/bin/python -m black tests/geometry/test_tracks.py
git add tests/geometry/test_tracks.py
git commit --only tests/geometry/test_tracks.py -m "test(geometry): build_tracks end to end and parity with the star-chain prototype

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: BA hook (`track_source`)

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py` (config ~61-100; `extract_tracks` ~159-226; `refine` ~296-345; `_compute_tracks_cache_key` ~993-1008)
- Modify: `collab_splats/reconstructor.py` (`refine` stage ~776-795)
- Modify: `configs/base.yaml` (`pointcloud.bundle_adjustment` block)
- Test: `tests/geometry/test_bundle_adjustment.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_tracks_vggsfm_source_unchanged(monkeypatch):
    calls = []

    def fake_vggsfm(images, conf, world_points, cfg):
        calls.append((images, conf, world_points, cfg))
        return np.zeros((2, 1, 2), np.float32), np.ones((2, 1), np.float32), np.zeros((1, 3), np.float32)

    monkeypatch.setattr(ba_mod, "extract_tracks_vggsfm", fake_vggsfm)
    ba = ba_mod.BundleAdjustment(ba_mod.BundleAdjustmentConfig())

    ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=None)

    assert calls == [("imgs", "conf", "wp", ba.config)]


def test_extract_tracks_matcher_source_dispatches(monkeypatch):
    seen = {}

    def fake_build(matcher, images, frame_paths, world_points, extrinsics, intrinsics):
        seen.update(matcher=matcher, frame_paths=frame_paths, extrinsics=extrinsics, intrinsics=intrinsics)
        return np.zeros((2, 1, 2), np.float32), np.ones((2, 1), np.float32), np.zeros((1, 3), np.float32)

    monkeypatch.setattr(ba_mod, "build_tracks", fake_build)
    monkeypatch.setattr(ba_mod, "LocalMatcher", lambda name: f"matcher:{name}")
    ba = ba_mod.BundleAdjustment(ba_mod.BundleAdjustmentConfig(track_source="xfeat"))

    ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=["a.png"])

    assert seen == {"matcher": "matcher:xfeat", "frame_paths": ["a.png"], "extrinsics": "ext", "intrinsics": "K"}


def test_extract_tracks_matcher_source_needs_frame_paths():
    ba = ba_mod.BundleAdjustment(ba_mod.BundleAdjustmentConfig(track_source="xfeat"))

    with pytest.raises(ValueError, match="frame_paths"):
        ba.extract_tracks("imgs", "conf", "wp", None, extrinsics="ext", intrinsics="K", frame_paths=None)


def test_tracks_cache_key_includes_source():
    wp = np.zeros((2, 2, 2, 3), np.float32)
    k_v = ba_mod._compute_tracks_cache_key(["a"], wp, ba_mod.BundleAdjustmentConfig())
    k_x = ba_mod._compute_tracks_cache_key(["a"], wp, ba_mod.BundleAdjustmentConfig(track_source="xfeat"))

    assert k_v != k_x
```

At the top of the test file, use `from collab_splats.geometry import bundle_adjustment as ba_mod` if the file does not already import the module under some name; if it does, reuse that name.

- [ ] **Step 2: Run them and watch them fail**

Run: `timeout 300 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -q -k "track_source or tracks_cache_key_includes or vggsfm_source or matcher_source" -p no:cacheprovider; echo EXIT=$?`

Expected: FAIL (`unexpected keyword argument 'track_source'` / `'extrinsics'`).

- [ ] **Step 3: Implement**
  - **Imports**, added to the `collab_splats` group:
    ```python
    from collab_splats.geometry.tracks import build_tracks
    from collab_splats.localization.extractors import LocalMatcher
    ```
  - **Config:**
    - Tracks group, after `tracks_cache_dir`:
      ```python
      track_source: Literal["vggsfm", "xfeat", "loma"] = "vggsfm"
      ```
    - Docstring `Args:`, after `tracks_cache_dir`:
      ```
              track_source: "vggsfm" predicts tracks; "xfeat" / "loma" build matcher star tracks (geometry/tracks.py).
      ```
    - Append to `__post_init__`:
      ```python
      if self.track_source not in ("vggsfm", "xfeat", "loma"):
          raise ValueError(f"BundleAdjustmentConfig.track_source must be vggsfm, xfeat or loma, got {self.track_source!r}")
      ```
  - **`extract_tracks`:**
    - New signature:
      ```python
      def extract_tracks(
          self,
          images: np.ndarray,
          confidence: np.ndarray,
          world_points: np.ndarray,
          image_paths: list | None,
          *,
          extrinsics: np.ndarray | None = None,
          intrinsics: np.ndarray | None = None,
          frame_paths: list[Path] | None = None,
      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
      ```
    - Docstring: summary becomes `Tracks for the frames from the configured source; cached under tracks_cache_dir when set.`
    - Add these `Args:`:
      ```
              extrinsics: (N, 4, 4) world-to-cam; matcher sources only.
              intrinsics: (N, 3, 3) model-grid K; matcher sources only.
              frame_paths: full-res store frame per model frame; matcher sources only.
      ```
    - Add `Raises:`:
      ```
          Raises:
              ValueError: a matcher source without extrinsics, intrinsics or frame_paths.
      ```
    - Replace both `extract_tracks_vggsfm(images, confidence, world_points, cfg)` calls with:
      ```python
      self._extract_uncached(images, confidence, world_points, extrinsics, intrinsics, frame_paths)
      ```
    - Change the cache-hit log to `"Track cache hit (skipping extraction): %s"`.
  - **New private method**, below `extract_tracks`:
    ```python
    def _extract_uncached(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
        frame_paths: list[Path] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks from the configured source, uncached.

        - matcher sources need poses, model-grid K and full-res frame paths; confidence is unused there
        """
        cfg = self.config

        if cfg.track_source == "vggsfm":
            return extract_tracks_vggsfm(images, confidence, world_points, cfg)

        # Matcher tracks read prior poses and full-res frames
        if extrinsics is None or intrinsics is None or not frame_paths:
            raise ValueError(f"BA track_source {cfg.track_source!r} needs extrinsics, intrinsics and frame_paths")

        matcher = LocalMatcher(cfg.track_source)
        return build_tracks(matcher, images, frame_paths, world_points, extrinsics, intrinsics)
    ```
  - **`refine`:**
    - Add a keyword after `depth`: `frame_paths: list[Path] | None = None`.
    - `Args:` line: `frame_paths: full-res store frame per model frame; read by matcher track sources.`
    - The call becomes:
      ```python
      tracks, vis_scores, pts3d_tracks = self.extract_tracks(
          images,
          confidence,
          world_points,
          image_paths,
          extrinsics=extrinsics,
          intrinsics=intrinsics,
          frame_paths=frame_paths,
      )
      ```
    - Update its block comment to `# Load cached tracks or extract from the configured source (one extraction shared across k-steps)`.
  - **`_compute_tracks_cache_key`:** add `"track_source": cfg.track_source,` to `meta`. Existing VGGSfM caches miss once. That is expected: the gate is bit-identical refine output, not cache reuse.
  - **`reconstructor.refine`:** after `ff = PointcloudResult.load_zarr(...)`, resolve the frames:
    ```python
    # Full-res store frames in zarr order, for matcher track sources
    all_paths = frames.frame_paths(self.images_dir)
    rows = store_rows(self.images_dir, ff.image_paths)
    frame_paths = [all_paths[row] for row in rows]
    ```
    Then pass `frame_paths=frame_paths` to `ba.refine(...)`.
  - **`configs/base.yaml`:** in `pointcloud.bundle_adjustment`, add
    ```yaml
          track_source: vggsfm    # vggsfm | xfeat | loma (matcher star tracks, geometry/tracks.py)
    ```
    Match the block's existing indentation and comment style. Read the block first.

- [ ] **Step 4: Run the BA tests and the reconstructor refine tests**

Run: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py tests/reconstructor -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS. A reconstructor test that stubs `ba.refine` with a fixed positional signature needs `**kwargs` added to the stub. Fix the stub, not the call.

- [ ] **Step 5: Fresh-process import check**

Run: `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud; import collab_splats.geometry; import collab_splats.reconstructor; print('ok')"`

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
/opt/venv/reconstruction/bin/python -m isort collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
/opt/venv/reconstruction/bin/python -m black collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git add collab_splats/geometry/bundle_adjustment.py collab_splats/reconstructor.py configs/base.yaml tests/geometry/test_bundle_adjustment.py
git commit --only collab_splats/geometry/bundle_adjustment.py collab_splats/reconstructor.py configs/base.yaml tests/geometry/test_bundle_adjustment.py -m "feat(ba): track_source selects VGGSfM or matcher star tracks

- extract_tracks/refine take extrinsics, intrinsics and full-res frame_paths
- frame_paths separate from image_paths: result image_paths are stems
- cache key includes track_source; vggsfm path calls extract_tracks_vggsfm unchanged

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Do NOT run black on `reconstructor.py`; it has its own formatting history. Run `isort --check` on it only.

---

### Task 5: Localizer test fixtures

**Files:**
- Create: `tests/localization/conftest.py`

- [ ] **Step 1: Write the fixtures**

```python
"""
Localization test fixtures: a CPU DINO-SALAD stand-in and a duck-typed matcher.
"""

import numpy as np
import pytest
import torch

from collab_splats.localization.extractors import LocalFeatures, MatchResult


class FakeSalad(torch.nn.Module):
    """
    Deterministic global descriptor: per-channel mean and std, L2-normalized.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        flat = images.float().flatten(2)
        desc = torch.cat([flat.mean(-1), flat.std(-1)], dim=1)
        return torch.nn.functional.normalize(desc, dim=-1).cpu()


class StubMatcher:
    """
    LocalMatcher stand-in: extract replays a callback, match pairs equal descriptor rows.

    - keypoints(image) -> (K, 2) px; descriptors are row ids, so equal ids match
    """

    model_name = "stub"
    device = "cpu"

    def __init__(self, keypoints) -> None:
        self._keypoints = keypoints
        self.n_extract = 0

    def _one(self, image: np.ndarray) -> LocalFeatures:
        self.n_extract += 1
        kp = torch.as_tensor(self._keypoints(image), dtype=torch.float32)
        ids = torch.arange(len(kp), dtype=torch.float32)[:, None]
        return LocalFeatures(keypoints=kp, descriptors=ids, image_size=(image.shape[1], image.shape[0]))

    def extract(self, images):
        if isinstance(images, list):
            return [self._one(im) for im in images]
        return self._one(images)

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        return features

    def match(self, q: LocalFeatures, db: LocalFeatures) -> MatchResult:
        _, iq, idb = np.intersect1d(q.descriptors[:, 0].numpy(), db.descriptors[:, 0].numpy(), return_indices=True)
        return MatchResult(
            query_px=q.keypoints.numpy()[iq],
            ref_px=db.keypoints.numpy()[idb],
            idx_q=iq.astype(np.int64),
            idx_db=idb.astype(np.int64),
        )


@pytest.fixture(autouse=True)
def fake_salad(monkeypatch):
    """
    Every localizer in these tests embeds with FakeSalad, never the real checkpoint.
    """
    monkeypatch.setattr("collab_splats.localization.localizer.DinoSaladExtractor", FakeSalad)
    return FakeSalad


@pytest.fixture
def stub_matcher():
    """
    The StubMatcher class, for tests that need a duck-typed matcher.
    """
    return StubMatcher
```

The `monkeypatch.setattr` target only resolves after Task 6 adds `DinoSaladExtractor` to `localizer.py`'s namespace. Task 6 Step 1 commits this file together with its tests, so there is no commit step here.

---

### Task 6: `localizer.py`: one match path

**Files:**
- Modify: `collab_splats/localization/localizer.py` (whole file)
- Test: `tests/localization/test_localizer.py`, `tests/localization/test_localization_cache.py`, `tests/localization/test_provenance.py`, `tests/localization/test_decoupling_parity.py`, `tests/localization/test_reference_alignment.py`, `tests/localization/test_local_matcher.py` (localizer parts)

- [ ] **Step 1: Write the failing tests** (append to `tests/localization/test_localizer.py`)

```python
def _plane_scene(H: int = 64, W: int = 64, f: float = 50.0, z: float = 2.0):
    """
    Fronto-parallel plane at depth z under a pinhole K, as one (1, H, W, 3) world map.
    """
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    wp = np.stack([(xs - W / 2) / f * z, (ys - H / 2) / f * z, np.full(xs.shape, z)], -1).astype(np.float32)
    return wp[None], K


def _grid_keypoints(image: np.ndarray) -> np.ndarray:
    """
    4x3 non-collinear keypoint grid scaled to the image.
    """
    h, w = image.shape[:2]
    return np.array([[w * (0.125 + 0.22 * (i % 4)), h * (0.125 + 0.28 * (i // 4))] for i in range(12)])


def test_localize_refs_uses_only_chosen_frames(stub_matcher):
    wp, K = _plane_scene()
    wp3 = np.repeat(wp, 3, 0)
    img = np.zeros((64, 64, 3), np.uint8)
    loc = CameraLocalizer(wp3, np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)), [img] * 3, ["a", "b", "c"], extractor=stub_matcher(_grid_keypoints))

    res = loc.localize(img, query_intrinsics=K, refs=[2])

    assert res.pose is not None
    assert set(res.ref_frame_indices.tolist()) == {2}
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)


def test_localize_refs_refuses_non_reconstruction_frame(stub_matcher):
    wp, K = _plane_scene()
    img = np.zeros((64, 64, 3), np.uint8)
    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [img], ["a"], extractor=stub_matcher(_grid_keypoints))

    with pytest.raises(ValueError, match="reconstruction"):
        loc.localize(img, query_intrinsics=K, refs=[1])


def test_crop_map_samples_matching_model_pixel():
    box = np.array([100, 0, 1100, 1000, 1200, 1000], np.float32)
    px_full = np.array([[100 + 20 * 20, 10 * 20]], np.float32)

    px_model = localizer_mod._crop_to_model_grid(px_full, box, (50, 50))

    np.testing.assert_allclose(px_model, [[20.0, 10.0]])


def test_localize_fullres_cropped_refs(stub_matcher):
    # Model grid 64x64; refs are 128x96 full-res frames whose center 96x96 crop became the grid
    wp, K_model = _plane_scene()
    box = np.array([[16, 0, 112, 96, 128, 96]], np.float32)
    scale = 96 / 64
    K_full = K_model.copy()
    K_full[:2] *= scale
    K_full[0, 2] += 16
    ref = np.zeros((96, 128, 3), np.uint8)

    # Keypoints placed where the plane's model-grid pixels land in the full-res crop
    def keypoints(image):
        grid = _grid_keypoints(np.zeros((64, 64, 3)))
        return grid * scale + np.array([16, 0])

    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [ref], ["a"], extractor=stub_matcher(keypoints), original_coords=box)

    res = loc.localize(ref, query_intrinsics=K_full, refs=[0])

    assert res.pose is not None
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)
    assert res.ref_hw == (96, 128)
```

At the top of the file, add `from collab_splats.localization import localizer as localizer_mod`.

Append to `tests/localization/test_localization_cache.py`:

```python
def test_global_desc_round_trip(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    imgs = [np.full((8, 8, 3), v, np.uint8) for v in (10, 200)]
    loc = CameraLocalizer(wp, np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)), imgs, ["a", "b"], extractor=stub_matcher(lambda im: np.zeros((0, 2))))
    loc.save_index(tmp_path / "pc.zarr", "stub")

    loaded = CameraLocalizer.load_index(tmp_path / "pc.zarr", "stub", wp, loc.extrinsics, extractor=stub_matcher(lambda im: np.zeros((0, 2))))

    np.testing.assert_allclose(loaded._global_desc, loc._global_desc)


def test_load_index_without_global_desc_raises_keyerror(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [np.zeros((8, 8, 3), np.uint8)], ["a"], extractor=stub_matcher(lambda im: np.zeros((0, 2))))
    loc.save_index(tmp_path / "pc.zarr", "stub")
    store = zarr.open(str(tmp_path / "pc.zarr"), mode="a")
    del store["local_features/stub/reconstruction/global_desc"]

    with pytest.raises(KeyError, match="global_desc"):
        CameraLocalizer.load_index(tmp_path / "pc.zarr", "stub", wp, loc.extrinsics, extractor=stub_matcher(lambda im: np.zeros((0, 2))))


def test_from_pointcloud_cache_hit_reads_zero_frames(tmp_path, stub_matcher):
    wp = np.zeros((2, 8, 8, 3), np.float32)
    result = _mock_result(wp)
    ids = ["frame_000000.png", "frame_000001.png"]
    built = CameraLocalizer.from_pointcloud(result, zarr_path=tmp_path / "pc.zarr", images=[np.zeros((8, 8, 3), np.uint8)] * 2, ids=ids, extractor=stub_matcher(lambda im: np.zeros((3, 2))))
    reads = []

    def lazy():
        for _ in range(2):
            reads.append(1)
            yield np.zeros((8, 8, 3), np.uint8)

    hit = CameraLocalizer.from_pointcloud(result, zarr_path=tmp_path / "pc.zarr", images=lazy(), ids=ids, extractor=stub_matcher(lambda im: np.zeros((3, 2))))

    assert reads == [] and hit.image_paths == built.image_paths


def test_from_pointcloud_stale_ids_rebuild(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    result = _mock_result(wp)
    CameraLocalizer.from_pointcloud(result, zarr_path=tmp_path / "pc.zarr", images=[np.zeros((8, 8, 3), np.uint8)], ids=["frame_000000.jpg"], extractor=stub_matcher(lambda im: np.zeros((3, 2))))
    matcher = stub_matcher(lambda im: np.zeros((3, 2)))

    CameraLocalizer.from_pointcloud(result, zarr_path=tmp_path / "pc.zarr", images=[np.zeros((8, 8, 3), np.uint8)], ids=["frame_000000.png"], extractor=matcher)

    assert matcher.n_extract == 1


def test_localized_frame_keeps_normalized_and_size(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [np.zeros((8, 8, 3), np.uint8)], ["a"], extractor=stub_matcher(lambda im: np.zeros((3, 2))))
    loc.save_index(tmp_path / "pc.zarr", "stub")
    feats = LocalFeatures(keypoints=torch.zeros(3, 2), descriptors=torch.zeros(3, 1), keypoints_normalized=torch.ones(3, 2), image_size=(40, 30))

    loc.add_localized_frame("q.png", np.eye(4), np.eye(3), feats, zarr_path=tmp_path / "pc.zarr", extractor_name="stub")
    loaded = CameraLocalizer.load_index(tmp_path / "pc.zarr", "stub", wp, np.eye(4, dtype=np.float32)[None], extractor=stub_matcher(lambda im: np.zeros((3, 2))))

    got = loaded._frame_features[-1]
    assert got.image_size == (40, 30)
    np.testing.assert_array_equal(got.keypoints_normalized.numpy(), np.ones((3, 2)))


def test_update_index_one_write_per_array(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [np.zeros((8, 8, 3), np.uint8)], ["a"], extractor=stub_matcher(lambda im: np.zeros((3, 2))))
    loc.save_index(tmp_path / "pc.zarr", "stub")

    loc.update_index([np.zeros((8, 8, 3), np.uint8)] * 5, [f"n{i}" for i in range(5)], tmp_path / "pc.zarr", "stub")

    feats, paths, _ = read_localization_db(tmp_path / "pc.zarr", "stub")
    assert paths == ["a", "n0", "n1", "n2", "n3", "n4"] and len(feats) == 6
    store = zarr.open(str(tmp_path / "pc.zarr"), mode="r")
    assert store["local_features/stub/reconstruction/global_desc"].shape[0] == 6
```

Use the file's existing `_mock_result` helper. Make it return an object with `world_points`, `extrinsics`, `image_paths` (stems) and `original_coords` (`np.tile([0, 0, W, H, W, H], (N, 1))` with W, H from the world map), and nothing private.

Add `test_add_localized_frame_exact_id_guard` in place of the old duplicate-across-extensions test:

```python
def test_add_localized_frame_exact_id_guard(tmp_path, stub_matcher):
    wp = np.zeros((1, 8, 8, 3), np.float32)
    loc = CameraLocalizer(wp, np.eye(4, dtype=np.float32)[None], [np.zeros((8, 8, 3), np.uint8)], ["q.png"], extractor=stub_matcher(lambda im: np.zeros((3, 2))))
    feats = LocalFeatures(keypoints=torch.zeros(3, 2), descriptors=torch.zeros(3, 1), image_size=(8, 8))

    loc.add_localized_frame("q.png", np.eye(4), np.eye(3), feats)
    loc.add_localized_frame("q.jpg", np.eye(4), np.eye(3), feats)

    assert loc.image_paths == ["q.png", "q.jpg"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `timeout 600 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localizer.py tests/localization/test_localization_cache.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: many FAILs and ERRORs (`refs`, `original_coords`, `from_pointcloud`, `_crop_to_model_grid` do not exist yet; `fake_salad` cannot resolve `DinoSaladExtractor`).

- [ ] **Step 3: Rewrite `collab_splats/localization/localizer.py`**

```python
"""
Stage 3 pose estimation: cached-feature matches, world-point lookup, pycolmap absolute pose.

- feature DB: pointcloud.zarr local_features/<matcher>/{reconstruction,localized}
- refs: DINO-SALAD top-k reconstruction frames, or frames the caller chooses
- ref px map through the preprocess crop onto the world_points grid before lookup
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import zarr
from tqdm.auto import tqdm

from collab_splats.localization.extractors import LocalFeatures, LocalMatcher
from collab_splats.localization.retrieval import DinoSaladExtractor
from collab_splats.utils.io import LZ4
from collab_splats.utils.torch_utils import to_numpy

if TYPE_CHECKING:
    from collab_splats.pointcloud.base import PointcloudResult

logger = logging.getLogger(__name__)


########################################
# Helpers
########################################


def seed_intrinsics(height: int, width: int) -> np.ndarray:
    """
    Pinhole K seed from image proportions, COLMAP's f = 1.2 * max(W, H) rule.

    - centered principal point, square pixels; pycolmap focal refinement solves the true focal

    Args:
        height: image height, px.
        width: image width, px.

    Returns:
        (3, 3) float32 K.
    """
    f = 1.2 * max(width, height)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def sample_world_points(world_points: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bilinear world points at pixel coordinates (hloc interpolate_scan analog).

    - align_corners=True: pixel index i sits at coordinate i
    - invalid where the sample touches NaN or px falls outside the map; never a zero check

    Args:
        world_points: (H, W, 3) per-pixel world points.
        px: (K, 2) float32 xy on that grid.

    Returns:
        pts3d (K, 3) float32 and valid (K,) bool.
    """
    H, W, _ = world_points.shape

    # Normalize to [-1, 1] for grid_sample
    norm = px / np.array([[W - 1, H - 1]], dtype=np.float32) * 2 - 1
    norm = norm.astype(np.float32)
    grid = torch.from_numpy(norm)
    wp = torch.from_numpy(world_points)
    wp = wp.permute(2, 0, 1)[None].float()
    interp = F.grid_sample(wp, grid[None, None], align_corners=True, mode="bilinear")[0, :, 0]

    # NaN marks unmapped pixels; grid_sample pads out-of-bounds samples, so bounds are checked too
    valid = ~torch.any(torch.isnan(interp), dim=0)
    in_bounds = (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
    valid = valid.numpy() & in_bounds
    return interp.T.numpy().astype(np.float32), valid


def _crop_to_model_grid(px: np.ndarray, box: np.ndarray, model_hw: tuple[int, int]) -> np.ndarray:
    """
    Full-res pixels on the model grid: inverse of PointcloudResult.__post_init__'s crop map.

    - box is one original_coords row [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], pixel-corner
    - pixels outside the grid come back outside it; sample_world_points marks them invalid
    """
    H, W = model_hw
    crop_wh = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32)
    scale = np.array([W, H], dtype=np.float32) / crop_wh
    shifted = px - box[:2]
    return (shifted * scale).astype(np.float32)


def _chunked(items: Iterable[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    """
    Consecutive lists of up to size items, drawn lazily.
    """
    it = iter(items)

    while True:
        window = itertools.islice(it, size)
        chunk = list(window)

        if not chunk:
            return

        yield chunk


def _full_frame_coords(n: int, height: int, width: int) -> np.ndarray:
    """
    original_coords rows for uncropped frames whose model grid is the image itself.
    """
    row = np.array([0, 0, width, height, width, height], dtype=np.float32)
    return np.tile(row, (n, 1))


def localization_db_exists(zarr_path: Path | str, extractor_name: str) -> bool:
    """
    True when pointcloud.zarr holds a reconstruction feature DB for the extractor.

    - a missing store, or one that is not a group, reads as absent

    Args:
        zarr_path: pointcloud.zarr path.
        extractor_name: feature-cache key.

    Returns:
        Whether local_features/<extractor_name>/reconstruction exists.
    """
    # Open read-only; a missing or non-group store has no DB
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except (FileNotFoundError, zarr.errors.NodeNotFoundError, zarr.errors.ContainsArrayError):
        return False

    return f"local_features/{extractor_name}/reconstruction" in store


def _append_rows(group: zarr.Group, name: str, rows: np.ndarray) -> None:
    """
    Append rows to an existing array in one resize and one write.
    """
    arr = group[name]
    n = arr.shape[0]
    arr.resize((n + len(rows), *arr.shape[1:]))
    arr[n:] = rows


def _features_from_csr(group: zarr.Group, image_sizes: list[tuple[int, int]]) -> list[LocalFeatures]:
    """
    Per-frame LocalFeatures from one CSR feature group (reconstruction or localized).
    """
    offsets = group["frame_offsets"][:]
    kpts = group["keypoints"][:]
    descs = group["descriptors"][:]
    optional = {}

    for name in ("scores", "scales", "keypoints_normalized"):
        optional[name] = group[name][:] if name in group else None

    feats = []

    for i in range(len(offsets) - 1):
        s, e = int(offsets[i]), int(offsets[i + 1])
        extras = {k: None if v is None else torch.from_numpy(v[s:e]) for k, v in optional.items()}
        feats.append(
            LocalFeatures(
                keypoints=torch.from_numpy(kpts[s:e]),
                descriptors=torch.from_numpy(descs[s:e]),
                image_size=image_sizes[i],
                **extras,
            )
        )

    return feats


def _write_csr(group: zarr.Group, feats: list[LocalFeatures]) -> None:
    """
    Create the CSR arrays of a feature group: offsets, keypoints, descriptors, optional extras.

    - scores / scales: zero-filled for frames without them, absent when no frame has them
    - keypoints_normalized: written only when every frame has it; zeros would be wrong data
    """
    counts = [len(f.keypoints) for f in feats]
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    d = feats[0].descriptors.shape[1] if feats else 1
    kpts = [to_numpy(f.keypoints) for f in feats]
    descs = [to_numpy(f.descriptors) for f in feats]
    all_kpts = np.concatenate(kpts).astype(np.float32) if offsets[-1] else np.zeros((0, 2), np.float32)
    all_descs = np.concatenate(descs).astype(np.float32) if offsets[-1] else np.zeros((0, d), np.float32)
    group.create_array("frame_offsets", data=offsets, chunks=offsets.shape, compressors=LZ4)
    group.create_array("keypoints", data=all_kpts, chunks=(max(len(all_kpts), 1), 2), compressors=LZ4)
    group.create_array("descriptors", data=all_descs, chunks=(max(len(all_descs), 1), max(d, 1)), compressors=LZ4)

    # Per-keypoint extras
    for name in ("scores", "scales"):
        if not any(getattr(f, name) is not None for f in feats):
            continue

        parts = []

        for f in feats:
            value = getattr(f, name)
            parts.append(np.zeros(len(f.keypoints), np.float32) if value is None else to_numpy(value))

        data = np.concatenate(parts).astype(np.float32)
        group.create_array(name, data=data, chunks=(max(len(data), 1),), compressors=LZ4)

    # Normalized keypoints only when every frame carries them
    if feats and offsets[-1] and all(f.keypoints_normalized is not None for f in feats):
        parts = [to_numpy(f.keypoints_normalized) for f in feats]
        data = np.concatenate(parts).astype(np.float32)
        group.create_array("keypoints_normalized", data=data, chunks=(max(len(data), 1), 2), compressors=LZ4)


def _append_csr(group: zarr.Group, feats: list[LocalFeatures]) -> None:
    """
    Append frames to an existing CSR feature group, one resize and write per array.
    """
    counts = [len(f.keypoints) for f in feats]
    last = int(group["frame_offsets"][-1])
    offsets = last + np.cumsum(counts, dtype=np.int64)
    _append_rows(group, "frame_offsets", offsets)
    kpts = [to_numpy(f.keypoints) for f in feats]
    descs = [to_numpy(f.descriptors) for f in feats]
    kpts = np.concatenate(kpts, dtype=np.float32)
    descs = np.concatenate(descs, dtype=np.float32)
    _append_rows(group, "keypoints", kpts)
    _append_rows(group, "descriptors", descs)

    # Extras the group already stores; frames without one append zeros (scores, scales)
    for name in ("scores", "scales"):
        if name not in group:
            continue

        parts = []

        for f in feats:
            value = getattr(f, name)
            parts.append(np.zeros(len(f.keypoints), np.float32) if value is None else to_numpy(value))

        data = np.concatenate(parts, dtype=np.float32)
        _append_rows(group, name, data)

    # Normalized keypoints must exist for every appended frame, or the array would misalign
    if "keypoints_normalized" in group:
        if any(f.keypoints_normalized is None for f in feats):
            raise ValueError("feature DB stores keypoints_normalized; every appended frame must carry it")

        parts = [to_numpy(f.keypoints_normalized) for f in feats]
        data = np.concatenate(parts, dtype=np.float32)
        _append_rows(group, "keypoints_normalized", data)


########################################
# Result
########################################


@dataclass
class LocalizationResult:
    """
    Output of CameraLocalizer.localize: pose plus the correspondences PnP saw.

    - pose None when PnP fails or fewer than 4 correspondences exist
    - pts2d_ref lives in reference-image pixels of size ref_hw; rescale before drawing elsewhere
    - ref_frame_indices: source reference frame per correspondence
    """

    pose: np.ndarray | None  # (4, 4) world-to-camera
    n_correspondences: int  # M, 2D-3D pairs before RANSAC
    n_inliers: int
    pts2d: np.ndarray | None  # (M, 2) query px
    pts3d_matched: np.ndarray | None  # (M, 3) world
    inlier_mask: np.ndarray | None  # (M,) bool
    pts2d_ref: np.ndarray | None = None  # (M, 2) reference px
    ref_frame_indices: np.ndarray | None = None  # (M,) int32
    query_features: LocalFeatures | None = None  # pass to add_localized_frame
    query_intrinsics: np.ndarray | None = None  # (3, 3) K used for PnP
    ref_hw: tuple[int, int] | None = None  # (H, W) of the reference images

    @property
    def ranked_ref_frames(self) -> list[int]:
        """
        Reference frames ordered by inlier count, most first; zero-inlier frames dropped.

        - ties broken by lowest frame index; empty when pose failed

        Returns:
            Reference-frame indices.
        """
        if self.inlier_mask is None or self.ref_frame_indices is None:
            return []

        # Inliers per reference frame, descending, stable tie-break
        frames = self.ref_frame_indices[self.inlier_mask]
        frames = frames.astype(np.intp)
        counts = np.bincount(frames)
        order = np.argsort(-counts, kind="stable")
        return [int(i) for i in order if counts[i] > 0]


def read_localization_db(zarr_path: Path | str, extractor_name: str) -> tuple[list[LocalFeatures], list[str], tuple[int, int]]:
    """
    Read the reconstruction feature DB of one extractor.

    - the read half of CameraLocalizer.save_index; load_index builds a localizer on it

    Args:
        zarr_path: pointcloud.zarr path.
        extractor_name: feature-cache key.

    Returns:
        Per-frame LocalFeatures, frame ids, and the reference images' (H, W).

    Raises:
        KeyError: the store holds no DB for the extractor.
    """
    store = zarr.open(str(zarr_path), mode="r")
    rec_key = f"local_features/{extractor_name}/reconstruction"

    if rec_key not in store:
        raise KeyError(f"No feature DB for '{extractor_name}' in {zarr_path}; build via CameraLocalizer.from_pointcloud()")

    # Bulk decode is the slow phase of a cache-hit load; time it
    group = store[rec_key]
    ids = [str(p) for p in group.attrs["image_paths"]]
    hw = tuple(int(x) for x in group.attrs["hw"])
    t0 = time.perf_counter()
    feats = _features_from_csr(group, [(hw[1], hw[0])] * len(ids))
    logger.info("CameraLocalizer: read feature DB (%d frames) in %.1fs", len(feats), time.perf_counter() - t0)
    return feats, ids, hw


########################################
# Localizer
########################################


class CameraLocalizer:
    """
    Locates a query camera in a known scene from cached reference-frame features.

    - refs: DINO-SALAD top-k reconstruction frames, or the caller's chosen frames
    - 3D from the ref's dense world_points at each matched ref px; pose by pycolmap LO-RANSAC + refinement
    - poses are (4, 4) world-to-camera, as PointcloudResult.extrinsics
    """

    def __init__(
        self,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        images: Iterable[np.ndarray],
        ids: list[str],
        extractor: LocalMatcher | None = None,
        config: dict | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        *,
        original_coords: np.ndarray | None = None,
        top_k: int = 8,
        batch_size: int = 32,
    ) -> None:
        """
        Extract and embed every reference frame, one chunk at a time.

        Args:
            world_points: (N, H, W, 3) model-grid world points, one map per reference frame.
            extrinsics: (N, 4, 4) world-to-camera.
            images: (h, w, 3) uint8 RGB reference frames, drawn lazily; no image IO here.
            ids: stable per-frame labels, index-aligned with images.
            extractor: local matcher; None builds LocalMatcher("loma").
            config: pycolmap options: "estimation" {"ransac": {"max_error"}} (50),
                "refinement" {"refine_focal_length", "refine_extra_params"} (True, True).
            progress_callback: called (frame_index, total) as frames are indexed.
            original_coords: (N, 6) preprocess crop per frame; None means images are the uncropped grid source.
            top_k: retrieved reference frames per localize(refs=None).
            batch_size: frames per extract and embed call.

        Raises:
            ValueError: images is empty.
        """
        self.config = config or {}
        self._world_points = world_points
        self._extrinsics = extrinsics
        self._extractor = extractor if extractor is not None else LocalMatcher("loma")
        self._retrieval: DinoSaladExtractor | None = None
        self._top_k = top_k
        self._image_paths = [str(i) for i in ids]
        self._localized_extrinsics: list[np.ndarray] = []
        self._frame_features: list[LocalFeatures] = []
        descs = []

        # Extract and embed per chunk; tqdm only without an external progress sink
        bar = tqdm(total=len(ids), desc="Indexing frames", unit="frame", leave=False, disable=progress_callback is not None)

        for chunk in _chunked(images, batch_size):
            start = len(self._frame_features)
            feats = self._extractor.extract(chunk)
            self._frame_features += [self._extractor.to_device(f) for f in feats]
            descs.append(self._embed(chunk))
            bar.update(len(chunk))

            if progress_callback is not None:
                for k in range(start, len(self._frame_features)):
                    progress_callback(k, len(ids))

        bar.close()

        if not self._frame_features:
            raise ValueError("CameraLocalizer: no reference frames")

        # Reference image size, and the crop each frame went through
        w, h = self._frame_features[0].image_size
        self._image_hw = (h, w)
        n = len(self._frame_features)
        self._coords = original_coords if original_coords is not None else _full_frame_coords(n, h, w)
        self._global_desc = np.concatenate(descs)
        self._frame_sources = ["reconstruction"] * n
        logger.info("CameraLocalizer: indexed %d frames", n)

    ########################################
    # Views
    ########################################

    @property
    def frame_sources(self) -> list[str]:
        """
        Provenance per frame: 'reconstruction' or 'localized'.
        """
        return list(self._frame_sources)

    @property
    def image_paths(self) -> list[str]:
        """
        Frame ids, index-aligned with frame_sources and extrinsics.
        """
        return list(self._image_paths)

    @property
    def extrinsics(self) -> np.ndarray:
        """
        (N, 4, 4) world-to-camera for every frame: reconstruction then localized.
        """
        if not self._localized_extrinsics:
            return self._extrinsics

        localized = np.stack(self._localized_extrinsics)
        return np.concatenate([self._extrinsics, localized], axis=0)

    ########################################
    # Retrieval
    ########################################

    def _embed(self, images: list[np.ndarray]) -> np.ndarray:
        """
        DINO-SALAD descriptors of uint8 RGB frames; the model is built on first use.
        """
        if self._retrieval is None:
            self._retrieval = DinoSaladExtractor()

        # Upload uint8, convert on device
        device = next(self._retrieval.parameters()).device
        stack = np.stack(images)
        tensor = torch.from_numpy(stack).to(device)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255
        desc = self._retrieval(tensor)
        return to_numpy(desc).astype(np.float32)

    def _rank_refs(self, query_image: np.ndarray) -> list[int]:
        """
        Top-k reconstruction frames by cosine similarity to the query.
        """
        query_desc = self._embed([query_image])[0]
        sims = self._global_desc @ query_desc
        order = np.argsort(-sims, kind="stable")
        return [int(i) for i in order[: self._top_k]]

    ########################################
    # Persistence
    ########################################

    def save_index(self, zarr_path: Path | str, extractor_name: str, attrs: dict | None = None) -> None:
        """
        Persist the reconstruction features and global descriptors to pointcloud.zarr.

        - overwrites the extractor's reconstruction group; single writer
        - attrs replace the extractor group's attrs wholesale; extractor is always stamped

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            attrs: build provenance (backbone, ba, lc, built_at, ...).
        """
        store = zarr.open(str(zarr_path), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"

        # Clean overwrite
        if rec_key in store:
            del store[rec_key]

        group = store.require_group(rec_key)
        ext_group = store.require_group(f"local_features/{extractor_name}")
        ext_group.attrs.put({"extractor": extractor_name, **(attrs or {})})

        # Reconstruction frames only; localized frames live in their own group
        n_rec = self._frame_sources.count("reconstruction")
        group.attrs["image_paths"] = self._image_paths[:n_rec]
        group.attrs["hw"] = list(self._image_hw)
        _write_csr(group, self._frame_features[:n_rec])
        group.create_array("global_desc", data=self._global_desc, chunks=self._global_desc.shape, compressors=LZ4)
        logger.info("CameraLocalizer.save_index: %d frames to %s [%s]", n_rec, zarr_path, extractor_name)

    @classmethod
    def load_index(
        cls,
        zarr_path: Path | str,
        extractor_name: str,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        config: dict | None = None,
        extractor: LocalMatcher | None = None,
        *,
        original_coords: np.ndarray | None = None,
        top_k: int = 8,
    ) -> CameraLocalizer:
        """
        Localizer from the zarr feature DB, attached to the current scene geometry.

        - merges reconstruction and localized groups; the retrieval model loads on first localize

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            world_points: (N, H, W, 3) model-grid world points.
            extrinsics: (N, 4, 4) world-to-camera.
            config: pycolmap options, as __init__.
            extractor: local matcher; None builds LocalMatcher("loma").
            original_coords: (N, 6) preprocess crop per frame; None means uncropped.
            top_k: retrieved reference frames per localize(refs=None).

        Returns:
            The localizer.

        Raises:
            KeyError: no DB for the extractor, or it lacks global_desc (rebuild it).
        """
        rec_features, rec_ids, hw = read_localization_db(zarr_path, extractor_name)
        store = zarr.open(str(zarr_path), mode="r")
        rec_group = store[f"local_features/{extractor_name}/reconstruction"]

        # A DB without global descriptors predates this layout; the caller rebuilds
        if "global_desc" not in rec_group:
            raise KeyError(f"feature DB for '{extractor_name}' has no global_desc; rebuild it")

        # Localized frames, if any, with their own image sizes
        loc_key = f"local_features/{extractor_name}/localized"
        loc_features: list[LocalFeatures] = []
        loc_ids: list[str] = []
        loc_extrinsics: list[np.ndarray] = []

        if loc_key in store:
            loc_group = store[loc_key]
            loc_ids = [str(p) for p in loc_group.attrs.get("image_paths", [])]
            sizes = [(int(w), int(h)) for w, h in loc_group["image_sizes"][:]]
            loc_features = _features_from_csr(loc_group, sizes)
            loc_extrinsics = list(loc_group["extrinsics"][:])

        # Assemble without the extraction loop
        obj = object.__new__(cls)
        obj.config = config or {}
        obj._world_points = world_points
        obj._extrinsics = extrinsics
        obj._extractor = extractor if extractor is not None else LocalMatcher("loma")
        obj._retrieval = None
        obj._top_k = top_k
        obj._image_hw = hw
        obj._coords = original_coords if original_coords is not None else _full_frame_coords(len(rec_ids), *hw)
        obj._global_desc = rec_group["global_desc"][:]
        features = rec_features + loc_features
        obj._frame_features = [obj._extractor.to_device(f) for f in features]
        obj._frame_sources = ["reconstruction"] * len(rec_features) + ["localized"] * len(loc_features)
        obj._image_paths = rec_ids + loc_ids
        obj._localized_extrinsics = loc_extrinsics
        logger.info("CameraLocalizer.load_index: %d rec + %d loc frames [%s]", len(rec_features), len(loc_features), extractor_name)
        return obj

    def update_index(
        self,
        new_images: Iterable[np.ndarray],
        new_ids: list[str],
        zarr_path: Path | str,
        extractor_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
        *,
        batch_size: int = 32,
    ) -> None:
        """
        Extract and embed new reconstruction frames; append them to the zarr DB in one write per array.

        - world_points / extrinsics are not updated: reload via load_index with the new geometry
        - until then the new frames sit after any localized frames in memory

        Args:
            new_images: (h, w, 3) uint8 RGB frames, drawn lazily.
            new_ids: labels, index-aligned with new_images.
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
            progress_callback: called (frame_index, total) as frames are extracted.
            batch_size: frames per extract and embed call.
        """
        new_features: list[LocalFeatures] = []
        descs = []

        # Extract and embed per chunk
        for chunk in _chunked(new_images, batch_size):
            start = len(new_features)
            feats = self._extractor.extract(chunk)
            new_features += [self._extractor.to_device(f) for f in feats]
            descs.append(self._embed(chunk))

            if progress_callback is not None:
                for k in range(start, len(new_features)):
                    progress_callback(k, len(new_ids))

        new_desc = np.concatenate(descs)

        # In-memory state
        self._frame_features += new_features
        self._frame_sources += ["reconstruction"] * len(new_features)
        self._image_paths += [str(i) for i in new_ids]
        self._global_desc = np.concatenate([self._global_desc, new_desc])
        store = zarr.open(str(zarr_path), mode="a")
        rec_key = f"local_features/{extractor_name}/reconstruction"

        # No DB yet: write it whole
        if rec_key not in store:
            logger.warning("update_index: no reconstruction DB, writing it whole")
            self.save_index(zarr_path, extractor_name)
            return

        group = store[rec_key]
        group.attrs["image_paths"] = list(group.attrs["image_paths"]) + [str(i) for i in new_ids]
        _append_csr(group, new_features)
        _append_rows(group, "global_desc", new_desc)
        logger.info("CameraLocalizer.update_index: appended %d frames [%s]", len(new_ids), extractor_name)

    def add_localized_frame(
        self,
        image_path: Path | str,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: LocalFeatures,
        zarr_path: Path | str | None = None,
        extractor_name: str | None = None,
        provenance: dict | None = None,
    ) -> None:
        """
        Record a localized frame for provenance; it is never a match source.

        - duplicate guard on the exact id; a repeat is logged and skipped
        - persisted to the localized group when zarr_path and extractor_name are given; single writer
        - call clear_localized_frames after BA / LC updates that invalidate poses

        Args:
            image_path: frame id.
            pose: (4, 4) world-to-camera.
            intrinsics: (3, 3) K used for PnP.
            features: the query's LocalFeatures (LocalizationResult.query_features).
            zarr_path: pointcloud.zarr path; None keeps the frame in memory only.
            extractor_name: feature-cache key; None keeps the frame in memory only.
            provenance: opaque per-frame metadata stored beside the frame.
        """
        frame_id = str(image_path)

        if frame_id in self._image_paths:
            logger.warning("CameraLocalizer.add_localized_frame: %s already in index, skipping", frame_id)
            return

        self._frame_features.append(features)
        self._frame_sources.append("localized")
        self._image_paths.append(frame_id)
        self._localized_extrinsics.append(np.asarray(pose))

        if zarr_path is not None and extractor_name is not None:
            self._append_localized_to_zarr(frame_id, pose, intrinsics, features, zarr_path, extractor_name, provenance)

    @staticmethod
    def _append_localized_to_zarr(
        frame_id: str,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        features: LocalFeatures,
        zarr_path: Path | str,
        extractor_name: str,
        provenance: dict | None,
    ) -> None:
        """
        Append one localized frame, its pose, K and image size to the localized group.
        """
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"
        size = np.array([features.image_size], dtype=np.int64)

        # First frame creates the group; later frames append
        if loc_key not in store:
            group = store.require_group(loc_key)
            group.attrs["image_paths"] = [frame_id]
            group.attrs["provenance"] = [provenance or {}]
            _write_csr(group, [features])
            group.create_array("extrinsics", data=np.asarray(pose)[None], chunks=(1, 4, 4), compressors=LZ4)
            group.create_array("intrinsics", data=np.asarray(intrinsics)[None], chunks=(1, 3, 3), compressors=LZ4)
            group.create_array("image_sizes", data=size, chunks=(1, 2), compressors=LZ4)
            return

        group = store[loc_key]
        group.attrs["image_paths"] = list(group.attrs["image_paths"]) + [frame_id]
        group.attrs["provenance"] = list(group.attrs["provenance"]) + [provenance or {}]
        _append_csr(group, [features])
        _append_rows(group, "extrinsics", np.asarray(pose)[None])
        _append_rows(group, "intrinsics", np.asarray(intrinsics)[None])
        _append_rows(group, "image_sizes", size)

    @staticmethod
    def clear_localized_frames(zarr_path: Path | str, extractor_name: str) -> None:
        """
        Delete the extractor's localized group; reconstruction data is untouched.

        - call after BA / LC updates that invalidate localized poses, then reload via load_index

        Args:
            zarr_path: pointcloud.zarr path.
            extractor_name: feature-cache key.
        """
        store = zarr.open(str(zarr_path), mode="a")
        loc_key = f"local_features/{extractor_name}/localized"

        if loc_key in store:
            del store[loc_key]
            logger.info("CameraLocalizer.clear_localized_frames: cleared '%s' from %s", extractor_name, zarr_path)

    @classmethod
    def from_pointcloud(
        cls,
        result: PointcloudResult,
        *,
        zarr_path: Path,
        images: Iterable[np.ndarray] | None = None,
        ids: list[str] | None = None,
        extractor: LocalMatcher | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        top_k: int = 8,
        batch_size: int = 32,
        config: dict | None = None,
    ) -> CameraLocalizer:
        """
        Localizer for a reconstruction: loads the zarr feature DB, or builds and saves it.

        - a cache hit draws nothing from images; ids must equal the DB's ids exactly, or it rebuilds
        - a DB missing global_desc rebuilds

        Args:
            result: the reconstruction; reads world_points, extrinsics, image_paths, original_coords.
            zarr_path: pointcloud.zarr path holding the DB.
            images: (h, w, 3) uint8 RGB reference frames, aligned to result; drawn only on a rebuild.
            ids: frame labels aligned to images; None compares against result.image_paths.
            extractor: local matcher; None builds LocalMatcher("loma"). Its model_name keys the DB.
            progress_callback: called (frame_index, total) during a rebuild.
            top_k: retrieved reference frames per localize(refs=None).
            batch_size: frames per extract and embed call.
            config: pycolmap options, as __init__.

        Returns:
            The localizer.

        Raises:
            ValueError: a rebuild is needed but images or ids is None.
        """
        extractor = extractor if extractor is not None else LocalMatcher("loma")
        name = extractor.model_name
        expected = ids if ids is not None else [str(p) for p in result.image_paths]

        # Cache hit: ids match exactly and the DB has every array
        if localization_db_exists(zarr_path, name):
            store = zarr.open(str(zarr_path), mode="r")
            cached = [str(p) for p in store[f"local_features/{name}/reconstruction"].attrs["image_paths"]]

            if cached == expected:
                try:
                    return cls.load_index(
                        zarr_path,
                        name,
                        result.world_points,
                        result.extrinsics,
                        config=config,
                        extractor=extractor,
                        original_coords=result.original_coords,
                        top_k=top_k,
                    )
                except KeyError as exc:
                    logger.info("CameraLocalizer: %s, rebuilding", exc)
            else:
                logger.info("CameraLocalizer: feature DB ids differ from the reconstruction's, rebuilding")

        # Rebuild from the caller's frames
        if images is None or ids is None:
            raise ValueError("from_pointcloud: building the feature DB needs images and ids")

        localizer = cls(
            result.world_points,
            result.extrinsics,
            images,
            ids,
            extractor=extractor,
            config=config,
            progress_callback=progress_callback,
            original_coords=result.original_coords,
            top_k=top_k,
            batch_size=batch_size,
        )
        localizer.save_index(zarr_path, name)
        return localizer

    ########################################
    # Localization
    ########################################

    def localize(
        self,
        query_image: np.ndarray,
        query_intrinsics: np.ndarray | None = None,
        refs: Sequence[int] | None = None,
    ) -> LocalizationResult:
        """
        World-to-camera pose of a query image.

        - refs None: DINO-SALAD ranks reconstruction frames, top_k are matched
        - refs given: exactly those reconstruction frames (align to a chosen scene image)
        - per ref: cached features + match(); ref px through the crop map; world_points lookup

        Args:
            query_image: HxWx3 uint8 RGB.
            query_intrinsics: (3, 3) K; None seeds from proportions and pycolmap refines focal.
            refs: reconstruction frame indices to match against.

        Returns:
            LocalizationResult; pose None when PnP fails.

        Raises:
            ValueError: refs names a frame with no world_points (not a reconstruction frame).
        """
        query_feats = self._extractor.extract(query_image)
        query_feats = self._extractor.to_device(query_feats)

        if query_intrinsics is None:
            query_intrinsics = seed_intrinsics(*query_image.shape[:2])

        # Chosen frames must carry world points
        n_rec = len(self._world_points)

        if refs is not None and any(not 0 <= i < n_rec for i in refs):
            raise ValueError(f"localize: refs {list(refs)} must index reconstruction frames [0, {n_rec})")

        refs = self._rank_refs(query_image) if refs is None else list(refs)
        model_hw = self._world_points.shape[1:3]
        all_q, all_3d, all_ref, all_frame = [], [], [], []

        # Match each ref and lift its matched px through the world map
        for i in refs:
            m = self._extractor.match(query_feats, self._frame_features[i])

            if len(m) == 0:
                continue

            px_model = _crop_to_model_grid(m.ref_px, self._coords[i], model_hw)
            pts3d, valid = sample_world_points(self._world_points[i], px_model)

            if not valid.any():
                continue

            all_q.append(m.query_px[valid])
            all_3d.append(pts3d[valid])
            all_ref.append(m.ref_px[valid])
            all_frame.append(np.full(int(valid.sum()), i, dtype=np.int32))

        return self._solve_pnp(all_q, all_3d, all_ref, all_frame, query_image, query_feats, query_intrinsics)
```

Then keep `_solve_pnp` with these edits:
- Signature: `(self, all_q: list[np.ndarray], all_3d: list[np.ndarray], all_ref: list[np.ndarray], all_frame: list[np.ndarray], query_image: np.ndarray, query_feats: LocalFeatures, query_intrinsics: np.ndarray) -> LocalizationResult`.
- Drop the `ref_hw` parameter; set `ref_hw = self._image_hw` at the top.
- Docstring: summary line `LO-RANSAC + refinement PnP over the accumulated correspondences.` No `Args:` (private).
- Body: keep it as is. Split its nested calls one per line where they appear (`pts2d.astype(np.float64)` arguments go on their own lines). Use single-line block comments.

Deleted on purpose (never re-add):
- `_localize_pairwise`, `_build_pairwise_refs`, the `pairwise_refs` keyword
- `_ref_images`, `_ref_global_desc`
- the descriptor-path loop and its size-only rescale
- the PIL import and the `to_uint8_hwc` import
- `from_feedforward`

- [ ] **Step 4: Bring the existing localization tests in line**

Apply these rules to every file in `tests/localization/`:
- **Extractor stubs:** a stub `extract(image)` that takes one image must also accept a list. Each stub needs `model_name`, `device`, `to_device(f) -> f` and `image_size` on every `LocalFeatures`. Prefer replacing local stubs with the `stub_matcher` fixture where the stub was only "identity matches on a keypoint grid".
- **MatchResult constructions** need `idx_q` / `idx_db`.
- **`test_localizer.py`:**
  - Replace `test_camera_localizer_from_feedforward_classmethod` with a `from_pointcloud` equivalent. It passes `zarr_path=tmp_path / "pc.zarr"` and a result object carrying `original_coords`.
  - Delete `test_localize_fullres_images_modelres_world_points`; `test_localize_fullres_cropped_refs` replaces it.
  - Keep `test_localize_via_depth_lookup`, rewritten on `stub_matcher(_grid_keypoints)` with `refs=None` (the fake retrieval ranks the single frame first).
  - Delete the `_build_pairwise_refs` tests.
  - In `_localizer_replaying`, replace `MagicMock(spec=LocalMatcher)` with a stub whose `extract` returns the given features list per call (list in, list out) and whose features carry `image_size`.
- **`test_localization_cache.py`:**
  - `from_feedforward` tests become `from_pointcloud` tests with an explicit `zarr_path`. The "stale warns but loads" test becomes "stale rebuilds" (`test_from_pointcloud_stale_ids_rebuild` above).
  - Delete the duplicate-across-extensions (stem) test; `test_add_localized_frame_exact_id_guard` replaces it.
  - Any test reading `pairwise_refs` is deleted.
- **`test_local_matcher.py`:** delete the pairwise localizer tests (they built `CameraLocalizer` with `pairwise_refs`). A `read_localization_db` test there keeps working; give its features `image_size`.
- **`test_decoupling_parity.py`, `test_provenance.py`, `test_reference_alignment.py`:** apply the stub rules. `test_reference_alignment` is the natural home for one `refs=[i]` alignment test if it is not already covered above.
- After the edits, `grep -n "from_feedforward\|pairwise\|match_images\|probe" tests/localization` must print nothing.

- [ ] **Step 5: Run the localization suite**

Run: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/localization -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS, `EXIT=0`. `test_retrieval.py` is unaffected: `fake_salad` only patches the localizer's namespace.

- [ ] **Step 6: Fresh-process import check**

Run: `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats.localization; import collab_splats.pointcloud; import collab_splats.geometry.tracks; print('ok')"` and then, in a separate process, `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.tracks; print('ok')"`.

Expected: `ok` twice.

- [ ] **Step 7: Format and commit**

```bash
/opt/venv/reconstruction/bin/python -m isort collab_splats/localization/localizer.py tests/localization
/opt/venv/reconstruction/bin/python -m black collab_splats/localization/localizer.py tests/localization
git add collab_splats/localization/localizer.py tests/localization
git commit --only collab_splats/localization/localizer.py tests/localization -m "refactor(localization): one cached-feature match path, refs=, crop map, from_pointcloud

- localize matches cached features via match(); refs=None retrieves top-k, refs=[i] aligns to chosen frames
- ref px map through original_coords onto the world_points grid
- global_desc persisted; retrieval model lazy; update_index one write per array
- localized frames keep keypoints_normalized + image_size; exact-id duplicate guard
- from_pointcloud replaces from_feedforward: typed, explicit zarr_path, exact-id staleness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Callers: reconstructor and dashboard

**Files:**
- Modify: `collab_splats/reconstructor.py` (`_localization_db_exists` ~161-177, `_build_localization_db` ~180-221, and its call at ~481)
- Modify: `collab_splats/dashboard/pipeline.py` (`_load_feedforward_result` ~416-430, `_build_localizer` ~482-523)
- Modify: `collab_splats/dashboard/localize.py:31`
- Modify: `configs/base.yaml:198-203`
- Test: `tests/reconstructor/test_sfm_stage.py`, `tests/reconstructor/test_localization_db_overwrite.py`, `tests/dashboard/test_run_localization.py`, `tests/dashboard/test_localize_page.py`

- [ ] **Step 1: Update the caller tests first**
  - **Patch targets:** `"collab_splats.localization.localizer.CameraLocalizer.from_feedforward"` becomes `"...from_pointcloud"`. Spy signatures become `(result, *, zarr_path, images, ids, **kwargs)`.
  - **`test_sfm_stage.py:233`:** keep its assertion (M ids, M frames, zarr order). The frames it collects now come from the chunked `read_frames` generator, so consume `images` with `list(images)`.
  - **`test_run_localization.py`:** the genexpr capture at ~292-297 asserts zero reads before consumption. Keep that assertion.
  - **`"disk-lightglue"`:** in `test_run_localization.py:307` and `test_localize_page.py:485,540,598`, replace it with `"xfeat"`. Where a test asserts the method list, the list becomes `["xfeat", "loma"]`.

Run: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor tests/dashboard -q -p no:cacheprovider; echo EXIT=$?`

Expected: FAIL. Production code still calls `from_feedforward` (`AttributeError`).

- [ ] **Step 2: `reconstructor.py`**
  - Delete `_localization_db_exists`. Import `localization_db_exists` from `collab_splats.localization.localizer`, merging it into the existing import of that module, and use it at the old call site (~481).
  - Replace the body of `_build_localization_db` after the group delete:
    ```python
        # Load the reconstruction and the matcher; only the pairwise path ever needed images
        ff = PointcloudResult.load_zarr(pointcloud_zarr, load_world_points=True)
        extractor = LocalMatcher(extractor_name)

        # Store frames in the zarr's image_paths order, read lazily per chunk
        all_paths = frames.frame_paths(images_dir)
        rows = store_rows(images_dir, ff.image_paths)
        paths = [all_paths[row] for row in rows]
        idxs = [frames.frame_idx_from_path(p) for p in paths]
        images = (image for (chunk,) in batch_iterator(batch_size, idxs) for image in frames.read_frames(images_dir, chunk))

        # Localization ids name the store's own files
        ids = [p.name for p in paths]
        CameraLocalizer.from_pointcloud(ff, zarr_path=pointcloud_zarr, images=images, ids=ids, extractor=extractor, top_k=top_k, batch_size=batch_size)
    ```
  - Add a `batch_size: int = 32` keyword to `_build_localization_db`. Update its docstring:
    - bullet `top_k is the pairwise (vismatch) fan-out; the descriptor path ignores it` becomes `top_k: retrieved reference frames per localize`
    - add bullet `frames are read per batch_size chunk; a cache hit reads none` (it never hits here, since the group was dropped)
    - fix the stale block comment `# Drop the stale group; from_feedforward has no overwrite...` to name `from_pointcloud`
  - Imports: `from collab_splats.utils.torch_utils import batch_iterator` (merge into the existing import line if one exists). Drop `read_image` if nothing else in the file uses it (`grep -n "read_image" collab_splats/reconstructor.py`).
- [ ] **Step 3: `dashboard/pipeline.py`**
  - Move the two inline imports in `_build_localizer` to the top of the file, in the `collab_splats` group:
    ```python
    from collab_splats.localization import CameraLocalizer
    from collab_splats.localization.extractors import LocalMatcher
    ```
    Then run the fresh-process import check from Task 6 Step 6 plus `python -c "import collab_splats.dashboard.pipeline"`. If the move creates a cycle, keep them inline. The CLAUDE.md exception covers heavy optional deps, and vismatch is one; add a comment saying so.
  - Replace the boundary adapter and the call:
    ```python
        # Store frames read lazily per chunk; a cache hit reads none
        paths = fr.frame_paths(images_dir) if images_dir is not None else [Path(p) for p in result.image_paths]
        ids = [p.name for p in paths]

        if images_dir is not None:
            idxs = [fr.frame_idx_from_path(p) for p in paths]
            images = (image for (chunk,) in batch_iterator(32, idxs) for image in fr.read_frames(images_dir, chunk))
        else:
            images = (read_image(p) for p in paths)

        localizer = CameraLocalizer.from_pointcloud(
            result,
            zarr_path=zarr_path,
            images=images,
            ids=ids,
            extractor=extractor,
            progress_callback=on_progress,
        )
    ```
    The `images_dir is None` arm keeps its per-path `read_image`: export paths may span directories, which `read_frames` cannot serve.
  - In `_load_feedforward_result`, drop the localizer's `load_images=True` reason. The comment at ~422 says the localizer needs images; it no longer does. Change the call at ~628 to `load_images=False` only if nothing else on that path reads `result.images`. Check with `grep -n "\.images" collab_splats/dashboard/pipeline.py` and leave it alone if anything does.
  - Add `batch_iterator` to the existing `collab_splats.utils.torch_utils` import.
- [ ] **Step 4: `dashboard/localize.py:31`:** `_METHODS = ["xfeat", "loma"]`.
- [ ] **Step 5: `configs/base.yaml:198-203`:** rewrite the matcher comment as one or two lines:
    ```yaml
      matcher: loma    # xfeat | loma: vismatch models that match cached features (others are refused)
    ```
    Read the block first and keep its keys and indentation.
- [ ] **Step 6: Run the caller suites**

Run: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/reconstructor tests/dashboard tests/localization -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS. Dashboard tests that build a real `CameraLocalizer` (no `from_pointcloud` stub) would load the real DINO-SALAD checkpoint. Add the same `monkeypatch.setattr("collab_splats.localization.localizer.DinoSaladExtractor", ...)` fixture to `tests/dashboard/conftest.py` with an inline `FakeSalad` copy. Tests cannot import from `tests/localization/conftest.py`.

- [ ] **Step 7: Run the dashboard smoke gate** (mandatory before dashboard commits)

Run: `timeout 300 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m collab_splats.dashboard.serve --smoke; echo EXIT=$?`

Expected: `EXIT=0`. If `serve.py` lives elsewhere, find it with `grep -rn "\-\-smoke" collab_splats/dashboard`.

- [ ] **Step 8: Commit**

```bash
/opt/venv/reconstruction/bin/python -m isort --check collab_splats/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/localize.py
git add collab_splats/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/localize.py configs/base.yaml tests/reconstructor tests/dashboard
git commit --only collab_splats/reconstructor.py collab_splats/dashboard/pipeline.py collab_splats/dashboard/localize.py configs/base.yaml tests/reconstructor tests/dashboard -m "refactor(localization): callers build via from_pointcloud with chunked read_frames

- reconstructor + dashboard stop loading model-res images for the localizer
- one localization_db_exists helper; dashboard methods xfeat | loma

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Contract pass on `localization`

**Files:**
- Modify: `tests/test_docstring_contract.py:22`
- Modify: `collab_splats/localization/__init__.py`, `retrieval.py`, `viz.py`
- Create: `tests/test_import_cycles.py`

- [ ] **Step 1: Add the package to the contract and the cycle test**

In `tests/test_docstring_contract.py`, the package joins all three tiers (docstring shape, release checks, round-3 checks):
```python
PACKAGES = ("preproc", "semantics", "pointcloud", "geometry", "splats", "mesh", "localization")

RELEASED: frozenset[str] = frozenset({"preproc", "semantics", "geometry", "splats", "mesh", "evals", "localization"})
```
and in the `test_round3_rules` parametrize:
```python
        if _package_path(p).parts[0] in ("geometry", "mesh", "localization")
```
Update the round-3 comment above `ROUND3_CHECKS` to "enforced for geometry, mesh and localization".

Create `tests/test_import_cycles.py`:
```python
"""
Fresh-process imports of the packages whose import graph crosses geometry and localization.

- geometry.tracks imports localization; localizer imports pointcloud only under TYPE_CHECKING
"""

import subprocess
import sys

import pytest

MODULES = (
    "collab_splats.pointcloud",
    "collab_splats.geometry",
    "collab_splats.geometry.tracks",
    "collab_splats.localization",
    "collab_splats.reconstructor",
)


@pytest.mark.parametrize("module", MODULES)
def test_fresh_import(module):
    proc = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stderr
```

- [ ] **Step 2: Run the contract and watch it fail**

Run: `timeout 600 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py -q -k localization -p no:cacheprovider; echo EXIT=$?`

Release checks on the kept files pass today (no banned words, scene ids, UPPER numeric constants or silent fallbacks). Round-3 currently reports 7 quote-line docstrings in `retrieval.py`, 4 in `viz.py` plus one prose comment pair at `viz.py:199`, and 1 in `__init__.py`.

Expected: FAIL on `__init__.py` (prose docstring), `retrieval.py` (`BaseRetrievalExtractor`, `DinoSaladExtractor.__init__`, `PECLIPExtractor.*` one-line or prose docstrings, missing `Args:`) and `viz.py`. Any failure in `extractors.py` / `localizer.py` is a Task 1 / 6 miss; fix it there.

- [ ] **Step 3: Fix each reported def**
  - **`__init__.py`:**
    ```python
    """
    Camera localization: pose of a query image in a known reconstruction.

    - retrieval: DINO-SALAD global descriptors rank reference frames
    - extractors: vismatch xfeat / loma features, cached and matched pair by pair
    - localizer: matched ref px -> world_points lookup -> pycolmap absolute pose
    """
    ```
  - **`retrieval.py` and `viz.py`:** convert every public class, method and function docstring to the contract shape, using the failure messages as the checklist.
    - Keep the content; drop restated names; move types out of `Args:`.
    - Annotate missing parameters and returns (`viz.py:19`'s signature has an unannotated `ref_frame` int).
    - Do not change behavior. `viz.py`'s display rescale stays (spec §6).
    - `viz.py:199`'s 2-line comment becomes one line, or a header plus a `- ` bullet line.
- [ ] **Step 4: Run the contract, the import style and the cycle tests**

Run: `timeout 900 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/test_docstring_contract.py tests/test_import_style.py tests/test_import_cycles.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: PASS, `EXIT=0`.

- [ ] **Step 5: Measure the import cost geometry now pays for tracks**

Run: `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import time; t=time.perf_counter(); import collab_splats.geometry; print(round(time.perf_counter()-t, 2))"`, three times, on an idle machine.

Baseline before the branch was 8.5 s, with localization alone at 6.8 s and both together at 9.8 s. Expected: ≤ 10 s. Report the numbers in the final summary. A much larger number means something heavy is imported twice; investigate before moving on.

- [ ] **Step 6: Commit**

```bash
/opt/venv/reconstruction/bin/python -m isort collab_splats/localization tests/test_import_cycles.py
/opt/venv/reconstruction/bin/python -m black collab_splats/localization/__init__.py collab_splats/localization/retrieval.py collab_splats/localization/viz.py tests/test_import_cycles.py
git add collab_splats/localization/__init__.py collab_splats/localization/retrieval.py collab_splats/localization/viz.py tests/test_docstring_contract.py tests/test_import_cycles.py
git commit --only collab_splats/localization/__init__.py collab_splats/localization/retrieval.py collab_splats/localization/viz.py tests/test_docstring_contract.py tests/test_import_cycles.py -m "docs(localization): package joins the docstring contract; fresh-process import-cycle test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Profile-gated levers (§5 d, f)

Levers a, b, c and e are already in Tasks 6 and 7. The remaining two are applied only if profiling shows they matter.

**Files:**
- Create: `scratch/localization_profile/profile_localize.py` (gitignored scratch)

- [ ] **Step 1: Write the profile script**

```python
"""
Profile localization on gh1k: cache-hit load and one top-k localize.
"""

import cProfile
import pstats
import sys
from pathlib import Path

from collab_splats.localization import CameraLocalizer, LocalMatcher
from collab_splats.localization.localizer import read_localization_db
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.utils.io import read_image

zarr_path = Path(sys.argv[1])
query = read_image(Path(sys.argv[2]))
result = PointcloudResult.load_zarr(zarr_path)
matcher = LocalMatcher("loma")

# DB ids are store file names, not result.image_paths stems; pass them so the load is a hit
_, ids, _ = read_localization_db(zarr_path, "loma")

with cProfile.Profile() as prof:
    loc = CameraLocalizer.from_pointcloud(result, zarr_path=zarr_path, ids=ids, extractor=matcher)

pstats.Stats(prof).sort_stats("cumulative").print_stats(15)

with cProfile.Profile() as prof:
    res = loc.localize(query)

pstats.Stats(prof).sort_stats("cumulative").print_stats(15)
print("inliers", res.n_inliers)
```

Before running it, ask the user for the gh1k pointcloud.zarr path and a query image, unless the session already has them. Run it in tmux on an idle GPU, with an existing DB (built once via `_build_localization_db`).

- [ ] **Step 2: Decide**
  - **(d) row-chunked feature arrays:** apply only if `read_localization_db` takes ≥ 20% of cache-hit wall time. The change is `chunks=(65536, ...)` in `_write_csr`, plus a test that the round trip is unchanged.
  - **(f) one `sample_world_points` over the top-k maps:** apply only if `sample_world_points` takes ≥ 10% of `localize` wall time.
  - Record both numbers and both decisions in the final summary. Skipped levers stay out of the code.

---

### Task 10: Gates

Every heavy gate runs in tmux on an idle GPU, one at a time. Record every number in the final summary.

- [ ] **Gate 1: gh1k matcher tracks**

Script `scratch/localization_profile/gate_tracks.py` (scratch):

```python
"""
gh1k matcher tracks at the prototype's gh1kq60 settings; compare with 108,518 tracks / 929,663 observations.
"""

import sys
import time
from pathlib import Path

from collab_splats.geometry.tracks import build_tracks
from collab_splats.localization import LocalMatcher
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.preproc import frames
from collab_splats.reconstructor import store_rows

zarr_path, images_dir = Path(sys.argv[1]), Path(sys.argv[2])
ff = PointcloudResult.load_zarr(zarr_path, load_images=True)
all_paths = frames.frame_paths(images_dir)
rows = store_rows(images_dir, ff.image_paths)
frame_paths = [all_paths[r] for r in rows]
t = time.perf_counter()
tracks, vis, pts3d = build_tracks(LocalMatcher("xfeat"), ff.images, frame_paths, ff.world_points, ff.extrinsics, ff.model_intrinsics, seed_frames=60)
print("tracks", tracks.shape[1], "obs", int(vis.sum()), "wall", round(time.perf_counter() - t, 1))
```

Find the gh1k zarr and store paths in `scratch/match_tracks_prototype/HANDOFF_vismatch_tracks.md` (the gh1kq60 run); ask the user if they are not there.

Pass criteria:
- the `tracks:` log line reports extract ≤ 15 s and match ≤ 28 s
- verify and chain times are reported
- tracks within 5% of 108,518 and observations within 5% of 929,663

If extract misses its budget, check `batch_size`: a GPU OOM retry means it is too large, so halve it. Report the outcome; do not tune silently.

- [ ] **Gate 2: localize with LoMa, top-K 8, tutorial query.** Measure wall time and inliers before the branch and after it.
  - Before: run at `32e73933` in a temporary worktree, using `from_feedforward` with `load_images=True`.
  - After: run on this branch.
  - Pass: inliers ≥ before. Report the wall time.
- [ ] **Gate 3: tutorial `ref_image` with `refs=[i]`.** Load `docs/source/tutorials/07_localization/ref_image.jpg` read-only from the main checkout, without copying or committing it. Choose `i` as the scene frame the tutorial aligns it to (the notebook names it).
  - Pass: pose is not None and ≥ 4 inliers.
- [ ] **Gate 4: cache-hit `from_pointcloud` on gh1k.** Report wall time before and after. Pass: profile shows no DINO forward over N frames (`_embed` absent from the hit profile).
- [ ] **Gate 5: `_build_localization_db` on gh1k.** Report wall time before and after.
- [ ] **Gate 6: `update_index`, 100 frames onto 1k.** Report wall time before and after.
- [ ] **Gate 7: BA `track_source: vggsfm` bit-identical.** Run seeded `BundleAdjustment.refine` on the tutorial zarr at `32e73933` and on this branch. Save `extrinsics, intrinsics` to npz. Compare with `sha256sum` or `rtk proxy cmp`; never trust the rewritten `diff`.
  - Pass: the hashes are equal.
  - Delete `tracks.zarr` in both runs' cache dirs first, so both extract.
- [ ] **Gate 8: suite.**

Run: `timeout 3600 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/localization tests/dashboard tests/reconstructor tests/geometry tests/test_docstring_contract.py tests/test_import_style.py tests/test_import_cycles.py -q -p no:cacheprovider; echo EXIT=$?`

Expected: `EXIT=0`. Compare any failure against `docs/known-test-failures.md` before calling it pre-existing, and verify it on the base commit.

- [ ] **Step 9: Bookkeeping commit**
  - Run `graphify update .` in the worktree.
  - Add a CLAUDE.md In-Flight entry for **localization-vismatch** in the worktree's CLAUDE.md, linking the spec and plan. The main checkout's uncommitted CLAUDE.md edits are not touched.
  - Commit with `git commit --only CLAUDE.md`.

---

## Self-review notes

- **Spec coverage:**
  - §1 → Task 1; §2 → Task 6; §3 → Tasks 2–4; §4 → Task 6 (`_crop_to_model_grid` and its tests).
  - §5 a/c/e → Task 6, b → Task 7, d/f → Task 9.
  - §6 → Tasks 6–8; §7 → tests in Tasks 1–8; Gates → Task 10.
  - The rebase onto `clean/final` after `feat/rgbd-ba-cf` lands is outside this plan; it happens before merge.
- **Names used across tasks:**
  - `LocalMatcher.to_device` and `LocalMatcher.device` (Task 1) are used in Tasks 2 and 6.
  - `localization_db_exists` (Task 6) is used in Task 7.
  - `build_tracks` keyword names match the spec signature.
  - `refine(frame_paths=)` (Task 4) is used by `reconstructor.refine` (Task 4) and Gate 1 (direct `build_tracks`).
