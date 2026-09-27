# Geometry Release Round 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Land the round-3 review of `collab_splats/geometry` as set out in
[the spec](../specs/2026-09-26-geometry-round3-design.md): contract prose, no nested
functions, array-in BA, QA-contract JSON reports, and duplicated or dead code removed.

**Architecture:** Round 1 is one commit that changes prose only, proved by AST equality.
Round 2 is one commit per spec row, with a gate after each commit. Three scratch
equivalence harnesses are captured on the untouched tree first:

- LC parity
- BA poses
- report values

Every later commit is compared against those captures.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), pytest, numpy,
zarr 3, pypose/bae (CUDA A40), pycolmap.

---

## Standing rules (every task)

- Worktree: `WT=/workspace/collab-splats/.worktrees/geometry-round3`, branch `clean/geometry-round3`.
- Every python command runs as `cd $WT && PYTHONUTF8=1 PYTHONPATH=$WT /opt/venv/reconstruction/bin/python ...`.
  - `cd` does not persist between Bash calls, so it is always in the same command.
- Every gate prints a proof line first:
  - `python -c "import collab_splats; print(collab_splats.__file__)"`
  - the output must start with `$WT/`
- Git hygiene:
  - commit with `git commit --only <paths>`
  - use `git add -f` for `docs/superpowers/**`
  - never amend, rebase, reset or run a bare `git stash`
- Do not install or sync anything, merge, push, or edit notebooks.
- Never touch `.worktrees/tutorial-rework` or the main checkout's dirty files.
- pytest hygiene:
  - never pipe pytest through `tail`
  - never pass `--tb=no`
  - never run the full suite
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Contract style for all new or edited code (CLAUDE.md "Code Style"):
  - `"""` sits on its own line
  - summary is at most 100 chars
  - `- ` bullets
  - `Args:` and `Returns:` carry no types
  - a comment run of 3+ lines is a header plus `- ` bullets
  - blank line above every block comment
  - US spelling
- `SP=/tmp/claude-0/-workspace-collab-splats/351261bd-d06a-403c-8d8b-cd81503fd03c/scratchpad` holds the harnesses.

**G′ gate** (after every commit; the counts must match the Task 0 baseline, plus any tests the task adds or deletes):

```bash
cd $WT && PYTHONUTF8=1 PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && \
PYTHONUTF8=1 PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/geometry tests/evals tests/pointcloud/feedforward tests/pointcloud/test_pose_extraction.py tests/wrapper tests/test_docstring_contract.py -q -p no:cacheprovider > $SP/r3_gate_<task>.log 2>&1; echo exit=$?; grep -E "passed|failed" $SP/r3_gate_<task>.log | tail -3
```

## Spec deltas (decided while planning)

| Spec row | Delta | Why |
|---|---|---|
| C1 | lands after every hoist (Task 19), not first | the new nested-def check fails until C2, C14 and E4 remove the nested defs |
| C13 | merged into E2 (Task 22) | `check_scale_method` still guards `none` until E2 deletes it |
| C12, E1 | before C11 | `clean_for_json` moves first; the report reads the new verification.json |
| C10 | reorder only, and after C11 | no existing helper covers `_scale_intrinsics_to_original` (the ratio-with-crop-origin form exists nowhere else) |
| C11 | frame column is `median_abs_rel_depth_error` | it is the median of `abs()` per frame; the spec's `median_rel_depth_error` would read as signed |
| E1 | `_distribution` deleted, not moved | `eval_verification.py` already has `_dist`, which covers it |
| C14 (K 4×4) | the 4× K_4x4 builds go into Task 8 via `intrinsics_4x4` | the helper lives in transforms with `decompose_camera` |
| Gates (C11/E1) | equality is checked on the synthetic `_write_tiny_scene` scene, not `data/outputs` | tutorial-rework retires the shared outputs cache; a fixture is reproducible |
| E6 | dropped | the reconstructor has only one hand-rolled temp+replace (the new report write), below the ≥2 bar |

## File map

| File | Responsibility after this plan |
|---|---|
| `collab_splats/geometry/transforms.py` | + `project_to_so3`, `intrinsics_4x4`, `decompose_camera`, `clean_for_json`, `_umeyama_weights` |
| `collab_splats/geometry/bundle_adjustment.py` | array-in `refine`, public `check_model_resolution`, grouped config, digest cache key |
| `collab_splats/geometry/verification.py` | columnar `verification.json`, one id-map helper |
| `collab_splats/geometry/metrics.py` | `compute_reconstruction_quality(arrays)`; no IO beyond nothing |
| `collab_splats/geometry/loop_closure/graph.py` | one inner-chain loop, hoisted helpers, no `scale_method` |
| `collab_splats/geometry/loop_closure/wrapper.py` | no `scale_method`, no `_last_*` hooks |
| `collab_splats/wrapper/reconstructor.py` | report stage owns zarr load, dense pass, stale check, atomic write |
| `tests/geometry/loop_closure/_helpers.py` | `record_driven_submaps()` spy (replaces `_last_*`) |
| `tests/test_docstring_contract.py` | nested-def, quote-line docstring, 2-line prose-comment checks (geometry only) |

---

### Task 0: Baseline gate + equivalence harnesses

**Files:**
- Create: `$SP/lc_parity_r3.py`, `$SP/ba_eq_r3.py`, `$SP/report_eq_r3.py`
- Outputs: `$SP/r3_gate_base.log`, `$SP/lc_r3_base.npz`, `$SP/ba_r3_base.npz`, `$SP/report_r3_base/`

- [x] **Step 1: Run the G′ gate on the untouched tree.** Record the passed/failed/skipped counts and the failing ids (`grep -E "^FAILED|^ERROR"`) in `$SP/r3_baseline.txt`. Every later gate is compared with this file.

- [x] **Step 2: Create `lc_parity_r3.py`** as a copy of `$SP/lc_parity_t29b.py` with three changes:
  - `n_lc = int(lc.base.n_loops_applied)` replaces `len(lc._last_lc_submaps)`, so C16 does not break the harness
  - `LoopClosureConfig(...)` receives `scale_method=` only when it is not `"rotation_only"`: `**({} if scale_method == "rotation_only" else {"scale_method": scale_method})`, so the harness survives E2
  - usage becomes `capture|compare <npz> [methods]`, where `methods` is comma-separated (default `rotation_only,none`); `compare` checks only keys whose prefix is in `methods + ("find",)`

- [x] **Step 3: Capture LC.** Run `lc_parity_r3.py capture $SP/lc_r3_base.npz`, then `compare $SP/lc_r3_base.npz`. Expected: `PARITY OK`, and all keys bit-exact against itself.

- [x] **Step 4: Create `ba_eq_r3.py capture|compare <npz>`**:

```python
"""
BA equivalence: refine on a synthetic scene with fixed tracks, CUDA, bit-compare.

Usage: ba_eq_r3.py capture|compare <npz>
"""
import inspect
import sys
from dataclasses import replace
from unittest.mock import patch

import numpy as np

import collab_splats
from collab_splats.geometry import bundle_adjustment as bam
from tests.geometry.test_bundle_adjustment import _build_synthetic_scene

print(collab_splats.__file__)
N, P, H, W = 6, 300, 128, 128
pts3d, ext, K, tracks, vis = _build_synthetic_scene(N, P, H, W, seed=3)
rng = np.random.default_rng(0)
noisy = ext.copy()
noisy[:, :3, 3] += rng.normal(0, 0.05, (N, 3)).astype(np.float32)
fixed = (tracks.astype(np.float32), vis.astype(np.float32), pts3d.astype(np.float32))
images = np.zeros((N, 3, H, W), np.float32)
conf = np.ones((N, H, W), np.float32)
wp = np.zeros((N, H, W, 3), np.float32)
coords = np.tile(np.array([0, 0, W, H, W, H], np.float32), (N, 1))

out = {}
with patch.object(bam, "_extract_tracks_vggsfm", return_value=fixed):
    for inc in (0, 3):
        ba = bam.BundleAdjustment(bam.BundleAdjustmentConfig(increment_size=inc, lm_steps=10))
        params = inspect.signature(ba.refine).parameters
        if "result" in params:  # pre-C6 API
            from types import SimpleNamespace
            r = SimpleNamespace(images=images, confidence=conf, world_points=wp,
                                extrinsics=bam.extrinsics_to_homogeneous(noisy), intrinsics=K,
                                original_coords=coords, image_paths=None)
            with patch.object(bam, "replace", lambda res, **kw: SimpleNamespace(**kw)):
                got = ba.refine(r)
            e, k = got.extrinsics, got.intrinsics
        else:  # post-C6 API
            bam.check_model_resolution(K, images, coords)
            e, k = ba.refine(images, conf, wp, bam.extrinsics_to_homogeneous(noisy), K)
        out[f"ext_{inc}"], out[f"K_{inc}"] = np.asarray(e), np.asarray(k)

mode, path = sys.argv[1], sys.argv[2]
if mode == "capture":
    np.savez(path, **out)
    print("captured", sorted(out))
else:
    ref = np.load(path)
    bad = [k for k in ref.files if not np.array_equal(ref[k], out[k])]
    print("BA EQ OK" if not bad else f"BA EQ FAIL: {bad}")
    sys.exit(1 if bad else 0)
```

  Run `capture`, then `compare` twice. Expected: `BA EQ OK` both times, which shows the solve is deterministic. If it is not deterministic:
  - compare with `np.allclose(atol=1e-6)`
  - record the tolerance in the task report

- [x] **Step 5: Create `report_eq_r3.py capture|compare <dir>`.** It builds one synthetic scene and captures both reports:
  - Scene: `tests/geometry/test_metrics.py::_write_tiny_scene` into `<dir>/scene`, plus RGB frames written with `cv2.imwrite(images/frame_{i:06d}.png)` at the original resolution the crop coords declare.
  - Verification input: a `colmap/verification.json` in the old format (`pair_stats` rows, `frame_stats` by name).
  - `capture` runs the current `build_reconstruction_quality_report` and saves `old_report.json`.
  - `compare` loads `old_report.json`, runs the current code path, and asserts every surviving value is equal.
    - Old code path: the function.
    - New code path: `Reconstructor`-free.
      - Call `compute_reconstruction_quality` with the same arrays.
      - Stack the old/new pairs per the mapping table in Task 12, Step 4.
  - Write `capture` now. Write `compare` in Task 12, once the new API exists.

- [x] **Step 6: Nothing to commit** (the harnesses live in the scratchpad). Report the baseline counts.

---

### Task 1: Round 1 — prose only (R1–R5)

**Files:** every file under `collab_splats/geometry/` (the only edits are docstrings, comments, blank lines and the `log` → `logger` rename).

The `log` → `logger` rename in `graph.py` changes a name, so it is **not** prose. It goes into Task 14 (C14) instead. Round 1 is AST-proved, so it touches only docstrings and comments.

- [x] **Step 1: R1 docstrings.** Every def and class in geometry, private ones included, gets a contract docstring:
  - `"""` on its own line; a one-line summary of at most 100 chars
  - `- ` bullets, then `Args:` for every parameter (ctor params on `__init__`, or on the class for dataclasses) and `Returns:`
  - no quote-line one-liners (`"""Foo."""`) anywhere
  - A private helper may omit `Args:` only when it takes no parameters.

- [x] **Step 2: R2 comments.**
  - Every 2+ line prose comment run becomes a header line plus `- ` bullet fragments.
  - Every block comment gets a blank line above it.
  - Every uncommented wall of 6+ statements gets a block comment.

- [x] **Step 3: R3 explain once.** Grep each explanation and keep one canonical copy, turning the others into a short citation (`# see <function>`):
  - `R^T` / "transpose is inverse" (×5): canonical in `transforms.invert_poses`
  - intrinsics ratio (×7): canonical in `bundle_adjustment._check_model_resolution` and `graph.decompose_camera`
  - overlap (×4): canonical in `LoopClosureConfig`
  - 2-frame carrier (×3): canonical in `wrapper._run_lc_loop`
  - `_lc_assembled` (×4): canonical at its assignment

- [x] **Step 4: R4 dividers.** Every divider becomes `#` × 72, then `# Title`, then `#` × 72, matching `preproc/qa.py`. The `########...##########` title-in-hashes style in `bundle_adjustment.py` goes.

- [x] **Step 5: R5.** In the `verification.py` `Args:` entries, prose sentences become fragments.

- [x] **Step 6: Prove prose-only.**

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python $SP/prose_proof.py ac93e96b $(git diff --name-only -- collab_splats/geometry)
```

  Expected: `PROSE ONLY`.
  - Sanity mutation: temporarily change one `0` to `1` in `transforms.py`; the proof must print `CODE CHANGED: collab_splats/geometry/transforms.py`.
  - Revert the mutation with Edit, not with git.

- [x] **Step 7: Run the G′ gate.** Expected: same counts as the baseline. The contract test must stay green for geometry.

- [x] **Step 8: Commit.**

```bash
git commit --only collab_splats/geometry -m "docs(geometry): round-3 prose — contract docstrings, bulleted comments, explain once"
```

---

### Task 2: C2 — hoist `_extract`

**Files:** Modify `collab_splats/geometry/bundle_adjustment.py` (`_load_or_extract_tracks`).

- [x] **Step 1:** Replace the nested `_extract` with a module-level function:

```python
def _extract_tracks(
    images: np.ndarray, confidence: np.ndarray, world_points: np.ndarray, cfg: BundleAdjustmentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One VGGSfM extraction with the config's knobs, logged because it is the slow step.

    Args:
        images: (N, 3, H, W) model-grid frames.
        confidence: (N, H, W) per-pixel confidence.
        world_points: (N, H, W, 3) per-pixel world points.
        cfg: supplies max_query_pts, query_frame_num, fine_tracking and device.

    Returns:
        (tracks, vis_scores, pts3d_tracks).
    """
    logger.info(
        "Extracting VGGSfM tracks: %d frames, max_query_pts=%d, query_frame_num=%d, fine_tracking=%s (slow step)",
        len(images), cfg.max_query_pts, cfg.query_frame_num, cfg.fine_tracking,
    )
    return _extract_tracks_vggsfm(
        images, confidence, world_points,
        max_query_pts=cfg.max_query_pts, query_frame_num=cfg.query_frame_num,
        fine_tracking=cfg.fine_tracking, device=cfg.device,
    )
```

  - Place it just below `BundleAdjustmentConfig`, which is already defined above it.
  - Both call sites become `_extract_tracks(result.images, result.confidence, result.world_points, cfg)`.
  - `black`-format only the touched lines; do not run a repo-wide black.

- [x] **Step 2:** Run `pytest tests/geometry/test_bundle_adjustment.py -q -p no:cacheprovider`. Expected: same count as the baseline for this file.
- [x] **Step 3:** Run `ba_eq_r3.py compare $SP/ba_r3_base.npz`. Expected: `BA EQ OK`.
- [x] **Step 4:** Run the G′ gate, then commit with `refactor(ba): hoist the nested track-extraction closure`.

---

### Task 3: C3 — `_refine_allonce` → `_refine_global`

- [x] **Step 1:** Rename the def and every caller in one pass:

```bash
cd $WT && rtk proxy grep -rn "_refine_allonce" collab_splats tests evals docs --include=*.py --include=*.md --include=*.rst
```

  - Rename in `collab_splats/`, `tests/` and `evals/` only.
  - Leave `docs/superpowers/**` history untouched.
  - Update the log line to `"Global BA over %d frames"`.

- [x] **Step 2:** Run the BA test file; it must pass. Then run `ba_eq_r3.py compare`.
- [x] **Step 3:** Run the G′ gate, then commit with `refactor(ba): _refine_allonce -> _refine_global`.

---

### Task 4: C4 — group `BundleAdjustmentConfig`

- [x] **Step 1:** Keep all 11 fields and their defaults. The only changes are order, one block comment per group, and moving the trailing comments into the docstring:

```python
@dataclass
class BundleAdjustmentConfig:
    """
    LM bundle adjustment settings, grouped by the step that reads them.

    - tracks: VGGSfM extraction and its cache
    - filter: which observations enter the solve
    - solve: the LM problem itself
    - runtime: where it runs

    Attributes:
        max_query_pts: extraction query points (upstream demo default).
        query_frame_num: extraction query frames (upstream demo default).
        fine_tracking: VGGSfM fine refinement; coarse-only tracks are ~1-2 px off.
        tracks_cache_dir: zarr track-cache dir; None always extracts.
        vis_thresh: min VGGSfM visibility score for an observation.
        max_reproj_error: pre-solve pixel reprojection gate; None skips the filter.
        min_inliers_per_frame: frames below this inlier count sit out the solve.
        lm_steps: LM iterations per solve, all run, no early stop.
        shared_camera: one focal per scene (per-frame K spread is model noise); False fits one per frame.
        increment_size: 0 = one global solve; 1..N-1 = frames added per incremental solve.
        device: CUDA device, None = auto; CPU unsupported (bae LM is CUDA-only).
    """

    # Tracks
    max_query_pts: int = 4096
    query_frame_num: int = 8
    fine_tracking: bool = True
    tracks_cache_dir: Path | None = None

    # Filter
    vis_thresh: float = 0.2
    max_reproj_error: float | None = 4.0
    min_inliers_per_frame: int = 64

    # Solve
    lm_steps: int = 40
    shared_camera: bool = True
    increment_size: int = 0

    # Runtime
    device: str | None = None
```

  - Check that no caller constructs the config positionally: `rtk proxy grep -rn "BundleAdjustmentConfig(" collab_splats evals tests docs/source` must show keyword-only calls.
  - `max_reproj_error` is annotated `float | None`, because `None` is already a documented value.

- [x] **Step 2:** Run the BA tests and `ba_eq_r3.py compare`, then the G′ gate. Commit with `refactor(ba): group BundleAdjustmentConfig fields by step`.

---

### Task 5: C5 — `world_points` digest in the track-cache key

**Files:**
- Modify: `bundle_adjustment.py:_compute_tracks_cache_key`
- Modify: `evals/scripts/eval.py` (`_ba_cache`)
- Modify: `evals/scripts/ba_start_at_gt.py:134`
- Test: `tests/geometry/test_bundle_adjustment.py`

- [x] **Step 1: Failing test.** Add it next to the existing cache tests:

```python
def test_tracks_cache_key_changes_with_world_points():
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig, _compute_tracks_cache_key

    cfg = BundleAdjustmentConfig()
    paths = [Path("a/frame_000000.png"), Path("a/frame_000001.png")]
    wp = np.zeros((2, 4, 4, 3), np.float32)
    k0 = _compute_tracks_cache_key(paths, wp, cfg)
    wp2 = wp.copy()
    wp2[0, 0, 0, 0] = 1.0
    assert _compute_tracks_cache_key(paths, wp2, cfg) != k0
    assert _compute_tracks_cache_key(paths, wp.copy(), cfg) == k0
```

  Run it. Expected: FAIL with a `TypeError` (the old signature takes `(result, cfg)`).

- [x] **Step 2: Implement.**

```python
def _compute_tracks_cache_key(image_paths: list, world_points: np.ndarray, cfg: BundleAdjustmentConfig) -> str:
    """
    SHA-256 track-cache key over image paths, world points and extraction config.

    - world_points digest: two backbones over the same images must not share tracks

    Args:
        image_paths: frame paths, order-insensitive.
        world_points: (N, H, W, 3) the tracks are lifted from.
        cfg: supplies the extraction knobs.

    Returns:
        Hex digest.
    """
    meta = {
        "image_paths": sorted(str(p) for p in image_paths),
        "world_points": hashlib.sha256(np.ascontiguousarray(world_points).tobytes()).hexdigest(),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "fine_tracking": cfg.fine_tracking,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
```

  - The caller in `_load_or_extract_tracks` becomes `_compute_tracks_cache_key(result.image_paths, result.world_points, cfg)`.
  - `evals/scripts/eval.py`:
    - drop the `/ backbone` suffix on the tracks cache dir, so the plain `tracks_cache_dir` is used
    - drop the workaround comment
  - `ba_start_at_gt.py:134`: the call becomes `_compute_tracks_cache_key(result.image_paths, result.world_points, cfg)`.
  - Existing cache tests that hand-build a key are updated to the new signature.

- [x] **Step 3:** Run the BA tests, `tests/evals` and `ba_eq_r3.py compare`, then the G′ gate. Commit `evals/scripts/eval.py`, `evals/scripts/ba_start_at_gt.py`, `bundle_adjustment.py` and the test with `fix(ba): key the track cache on world_points too`.

---

### Task 6: C6 — `refine` takes arrays

**Files:**
- Modify: `bundle_adjustment.py`
- Modify: `collab_splats/wrapper/reconstructor.py:refine_poses`
- Modify: `evals/scripts/eval.py` (the `ba.refine(creator.outputs)` site)
- Modify: `evals/scripts/ba_start_at_gt.py`
- Tests: `tests/geometry/test_bundle_adjustment.py`, `tests/wrapper/*` (refine-stage tests)

- [x] **Step 1: New API in `bundle_adjustment.py`.**

  (a) `_check_model_resolution` → `check_model_resolution` (public, in `__all__`). Its body is unchanged.

  (b) `_load_or_extract_tracks(self, images, confidence, world_points, image_paths)`:
  - extracts directly when `cfg.tracks_cache_dir is None or not image_paths`
  - otherwise builds the key from `(image_paths, world_points, cfg)`

  (c) `_refine_global(self, extrinsics, tracks, vis_scores, pts3d_tracks, intrinsics)` and `_refine_incremental(self, extrinsics, tracks, vis_scores, pts3d_tracks, intrinsics, increment_size)` take `extrinsics` (N, 4, 4) in place of `result`, and slice `[:, :3, :]` themselves.

  (d) The new `refine`:

```python
    def refine(
        self,
        images: np.ndarray,
        confidence: np.ndarray,
        world_points: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        image_paths: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine camera poses and focal against VGGSfM tracks.

        - K must be at model resolution; callers run check_model_resolution first
        - points are not touched; the caller re-derives them under the new poses

        Args:
            images: (N, 3, H, W) model-grid frames, the track source.
            confidence: (N, H, W) per-pixel confidence for query-point sampling.
            world_points: (N, H, W, 3) the tracks are lifted from.
            extrinsics: (N, 4, 4) world-to-cam start poses.
            intrinsics: (N, 3, 3) model-resolution K.
            image_paths: frame paths for the track-cache key; None skips the cache.

        Returns:
            (N, 4, 4) refined extrinsics and (N, 3, 3) refined intrinsics.

        Raises:
            ValueError: fewer than 2 frames or 2 points stay active after filtering.
            RuntimeError: the resolved device is not CUDA (bae LM is CUDA-only).
        """
```

  - The body keeps the current logic.
  - It drops `replace`, returning `extrinsics_to_homogeneous(refined_extrinsics), refined_intrinsics_model`.
  - `from dataclasses import replace` and the `TYPE_CHECKING` `FeedforwardResult` import are deleted from `bundle_adjustment.py`.

- [x] **Step 2: Test helpers.** Add them to `tests/geometry/test_bundle_adjustment.py` (after the imports) and convert every `ba.refine(result)` site (:596, :618, :639, :887-889, :924, :958, :1230) and every `_load_or_extract_tracks(result)` site (:748-858, :1193, :1208):

```python
def _refine(ba, result):
    """Old-style call: check K, refine arrays, return the result with refined cameras."""
    check_model_resolution(result.intrinsics, result.images, result.original_coords)
    ext, K = ba.refine(
        result.images, result.confidence, result.world_points, result.extrinsics, result.intrinsics, result.image_paths
    )
    return replace(result, extrinsics=ext, intrinsics=K)


def _tracks(ba, result):
    """Track load with the result's arrays."""
    return ba._load_or_extract_tracks(result.images, result.confidence, result.world_points, result.image_paths)
```

  - The `_check_model_resolution` import at :24 becomes `check_model_resolution`.
  - Tests that asserted `refine` raises on an original-res K now call `_refine` and keep their assertion.
  - Tests may use `replace` on a `FeedforwardResult`; the pointcloud type in tests is fine.

- [x] **Step 3: Update the callers.**

  `reconstructor.refine_poses`:

```python
        # Refine poses with LM BA, then re-derive the point set under the new cameras
        # - VGGSfM tracks live on the model grid, so K must too; checked before the slow extraction
        check_model_resolution(ff.intrinsics, ff.images, ff.original_coords)
        ba = BundleAdjustment(BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir))
        extrinsics, intrinsics = ba.refine(
            ff.images, ff.confidence, ff.world_points, ff.extrinsics, ff.intrinsics, ff.image_paths
        )
        ff = replace(ff, extrinsics=extrinsics, intrinsics=intrinsics).reproject()
```

  - Add `check_model_resolution` to the inline import, which C18 hoists later.
  - Add `from dataclasses import replace` at the top of `reconstructor.py` if it is absent.

  In `evals/scripts/eval.py`, the `ba.refine(creator.outputs).reproject()` site becomes the same three-line pattern on `creator.outputs`.

  `ba_start_at_gt.py`:
  - :140 becomes `check_model_resolution(...)`
  - :143 becomes `ba_probe._load_or_extract_tracks(result.images, result.confidence, result.world_points, result.image_paths)`
  - :160-162 become `ext, _ = ba.refine(start.images, start.confidence, start.world_points, start.extrinsics, start.intrinsics, start.image_paths)`; `refined.extrinsics` becomes `ext`

- [x] **Step 4:** Run the BA tests, `tests/wrapper -k refine` and `tests/evals`, then `ba_eq_r3.py compare`. Expected: `BA EQ OK`, via the post-C6 branch of the harness.
  - Name every CUDA-skipped test in the report: `grep SKIPPED` on the BA file output, run with `-rs`.

- [x] **Step 5:** Run the G′ gate. Commit with `refactor(ba): refine takes arrays; callers check K resolution`.

---

### Task 7: C7 — `project_to_so3`

**Files:**
- Modify: `transforms.py` (Geometry helpers section), `bundle_adjustment.py` (SVD snap), `graph.py:decompose_camera`
- Test: `tests/geometry/test_transforms.py`

- [x] **Step 1: Failing tests.**

```python
def test_project_to_so3_returns_nearest_rotation_with_det_one():
    from collab_splats.geometry.transforms import project_to_so3

    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 3, 3))
    R = project_to_so3(M)
    assert np.allclose(R @ np.swapaxes(R, -1, -2), np.eye(3), atol=1e-10)
    assert np.allclose(np.linalg.det(R), 1.0)


def test_project_to_so3_fixes_a_reflection_and_keeps_a_rotation_bit_exact():
    from collab_splats.geometry.transforms import project_to_so3

    refl = np.diag([1.0, 1.0, -1.0])
    assert np.isclose(np.linalg.det(project_to_so3(refl)), 1.0)
    c, s = np.cos(0.3), np.sin(0.3)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    U, _, Vt = np.linalg.svd(Rz)
    assert np.array_equal(project_to_so3(Rz), U @ Vt)
```

- [x] **Step 2: Implement** in `transforms.py`:

```python
def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    Nearest rotation to each 3x3 matrix, by SVD, with the reflection case flipped.

    - U @ Vt is orthogonal but may have det -1; negating U's last column fixes it
    - bit-identical to U @ Vt when det is already +1

    Args:
        R: (..., 3, 3) near-rotation matrices.

    Returns:
        (..., 3, 3) rotations, det +1.
    """
    U, _, Vt = np.linalg.svd(R)
    U[..., :, -1] *= np.where(np.linalg.det(U @ Vt) < 0, -1.0, 1.0)[..., None]
    return U @ Vt
```

  - `np.where` form works for both (3, 3) and (N, 3, 3); multiplying by 1.0 keeps the det>0 case bit-exact.

- [x] **Step 3: Replace the two snaps.**
  - In BA, `ext_4x4[:, :3, :3] = project_to_so3(ext_4x4[:, :3, :3])`.
  - In `decompose_camera`, `R = project_to_so3(R)`.
- [x] **Step 4:** Run `lc_parity_r3.py compare $SP/lc_r3_base.npz` and `ba_eq_r3.py compare`. Both must be OK.
  - If LC parity fails, `decompose_camera` was seeing det<0 inputs.
  - In that case, revert the graph hunk only, and keep a plain `U @ Vt` there with a comment `# no det fix: <reason from the failing case>`.
  - Record this in the report.
- [x] **Step 5:** Run the G′ gate. Commit with `refactor(geometry): one project_to_so3 for both SVD snaps`.

---

### Task 8: C8 — `decompose_camera` → `transforms.py`, plus `intrinsics_4x4`

**Files:**
- Modify: `transforms.py`, `graph.py`, `submap.py`, `loop_closure/__init__.py`
- Modify: `tests/pointcloud/test_pose_extraction.py:14`, `tests/geometry/loop_closure/test_graph.py:14`

- [x] **Step 1:** Move `decompose_camera` verbatim (body plus docstring) from `graph.py` into `transforms.py` (Geometry helpers section). `graph.py` imports it from `collab_splats.geometry.transforms`.
- [x] **Step 2:** In `submap.get_all_poses_world`, delete the inline `from .graph import decompose_camera` and import it at the top of `submap.py` from transforms. This creates no cycle, because transforms imports nothing from loop_closure.
- [x] **Step 3:** Add `intrinsics_4x4` to `transforms.py`:

```python
def intrinsics_4x4(K: np.ndarray) -> np.ndarray:
    """
    Embed (..., 3, 3) K in the top-left of (..., 4, 4) identities.

    Args:
        K: (..., 3, 3) intrinsics.

    Returns:
        (..., 4, 4) with K top-left and 1 at [3, 3].
    """
    out = np.tile(np.eye(4, dtype=K.dtype), K.shape[:-2] + (1, 1))
    out[..., :3, :3] = K
    return out
```

  - Test: `intrinsics_4x4(K)[..., :3, :3] == K` and `[..., 3, 3] == 1` for shapes (3,3) and (5,3,3).
  - Replace every hand-built K 4×4 in `graph.py` (`add_submap`, `add_loop_edge` ×4) and in `submap.py` (`get_all_poses_world`) with it.
  - Match dtype: if a site built `np.eye(4)` float64 with a float32 K, pass `K.astype(np.float64)` so the output stays bit-exact. LC parity is the judge.
- [x] **Step 4:** In `loop_closure/__init__.py`, keep `decompose_camera` exported until E3; re-export it from transforms. Tests import from `collab_splats.geometry.transforms`.
- [x] **Step 5:** Run LC parity, then the G′ gate. Commit with `refactor(geometry): decompose_camera and intrinsics_4x4 in transforms`.

---

### Task 9: C9 — shared Umeyama input guard

- [x] **Step 1:** Read `umeyama_se3` (:250) and `umeyama_sim3` (:297). Extract their identical guard, meaning the weight validation, normalization and the shape/count checks, into:

```python
def _umeyama_weights(source: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Validated, normalized per-point weights shared by both Umeyama solvers.

    Args:
        source: (N, 3) points being aligned.
        weights: (N,) non-negative weights, or None for uniform.

    Returns:
        (N,) float64 weights summing to 1.

    Raises:
        ValueError: shapes disagree, fewer than 3 points, or weights sum to 0.
    """
```

  - The body is exactly the lines both functions share today, including the same error messages.
  - Both solvers call it.
- [x] **Step 2:** Run `pytest tests/geometry/test_transforms.py`, including the existing error-message tests, then LC parity, then the G′ gate. Commit with `refactor(geometry): one Umeyama weight guard`.

---

### Task 10: C12 — `clean_for_json` → transforms; one id map

**Files:** Modify `verification.py`, `transforms.py`, `metrics.py` (import), and any test that imports `clean_for_json` from verification.

- [x] **Step 1:** Move `clean_for_json` verbatim to `transforms.py` (new section `# JSON`). Update every importer:

```bash
rtk proxy grep -rn "clean_for_json" collab_splats tests evals
```

- [x] **Step 2:** In `verification.py`, the three `{image_id: position}` builds (in `verify_reconstruction` at ~:249 and :328, and in `_triangulate_and_summarize`) become one helper:

```python
def _image_positions(recon: "pycolmap.Reconstruction") -> dict[int, int]:
    """
    COLMAP image_id -> position in name-sorted order, the order every table uses.

    Args:
        recon: the reconstruction being verified.

    Returns:
        {image_id: 0-based position}.
    """
```

  - First read all three sites, and confirm they compute the same mapping (the same sort key).
  - If one differs, leave that site and record why.
- [x] **Step 3:** Run `tests/geometry/test_verification.py` and `tests/geometry/test_metrics.py`, then the G′ gate. Commit with `refactor(geometry): clean_for_json to transforms; one image-position map`.

---

### Task 11: E1 — `verification.json` contract

**Files:**
- Modify: `verification.py` (`_write_report`, `VerificationResult`, delete `_distribution` and `summary`)
- Modify: `evals/scripts/eval_verification.py`
- Modify: `collab_splats/wrapper/reconstructor.py:verify` (stale check)
- Tests: `tests/geometry/test_verification.py`, `tests/wrapper/*` verify tests

- [x] **Step 1: Failing tests** in `test_verification.py`. They replace the `summary`/`phase_seconds` asserts at :197-201, :246, :333 and :383-388:

```python
def test_verification_json_is_columnar(tmp_path):
    # Build inputs exactly as test_tier1_pair_stats_on_clean_scene does (copy its setup lines)
    # - _synthetic_scene -> _make_recon -> _features_from_keypoints -> verify_reconstruction
    overlap = 10
    out_dir = tmp_path / "verify"
    result = verify_reconstruction(recon, features, matcher, output_dir=out_dir, overlap=overlap)
    data = json.loads((out_dir / "verification.json").read_text())
    assert set(data) == {"params", "pairs", "frames"}
    assert data["params"] == {"overlap": overlap}
    n_pairs = len(result.pair_stats)
    assert all(len(v) == n_pairs for v in data["pairs"].values())
    assert set(data["pairs"]) == {f.name for f in dataclasses.fields(PairStats)}
    assert "name" in data["frames"] and "idx" in data["frames"]
```

  Also add a test that no key named `summary` exists in the JSON or on `VerificationResult`.

- [x] **Step 2: Implement.**
  - `_write_report(path, overlap, pair_stats, frame_stats)` writes `{"params": {"overlap": overlap}, "pairs": {field: [getattr(p, field) for p in pair_stats]}, "frames": {col: [...]}}`.
    - Frame rows are ordered by position.
    - `name` is a column, and so is `idx` (the position).
    - It writes through `clean_for_json`, then `path.with_suffix(".json.tmp")`, then `os.replace`.
  - `VerificationResult` drops `summary`.
  - `phase_seconds` is logged only, with `logger.info("verify phases: %s", ...)`.
  - `_distribution` is deleted.
  - `reconstructor.verify`, on the reuse path:

```python
        if not overwrite and self._stage_output_exists("verify"):
            if "pairs" not in json.loads(out_json.read_text()):
                raise ValueError(f"{out_json} is a stale verification report (no 'pairs'); delete it and re-run")
            logger.info("Verification exists at %s, skipping", out_json)
            return out_json
```

  - Wrapper test: write an old-format `{"pair_stats": [], "frame_stats": {}, "summary": {}}` file; `verify()` raises `ValueError` matching "stale".
  - `eval_verification.py`:
    - `report["summary"]` is replaced by `"yield": {"n_points": len(result.reconstruction.points3D), "track_length": _dist(len(p.track.elements) for p in result.reconstruction.points3D.values()), "rot_error_deg": _dist(p.rot_error_deg for p in result.pair_stats)}`
    - `"frame_stats": result.frame_stats` stays as is (it is the dataclass-free dict the result still carries)
    - the final print uses `report["yield"]`
- [x] **Step 3:** Run the tests, then the G′ gate. Commit `verification.py`, `eval_verification.py`, `reconstructor.py` and the tests with `refactor(geometry): verification.json follows the QA report contract`.

---

### Task 12: C11 + T1 — reconstruction quality report contract

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Modify: `collab_splats/wrapper/reconstructor.py:reconstruction_quality_report`
- Modify: `configs/README.md` (the report section)
- Tests: `tests/geometry/test_metrics.py`, `tests/wrapper/*` report tests

- [x] **Step 1: The new metrics API.** Signatures and return shapes:

```python
def compute_depth_error(collected: dict, focal_px: float) -> tuple[dict, dict]:
    """
    Per-direction depth disagreement and the pixel residual histogram, both columnar.

    Args:
        collected: the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px: mean focal, pixels; states the residual in pixel units.

    Returns:
        (depth_pairs, depth_residual_histogram)
        - depth_pairs: {idx1, idx2, n_pixels, median_rel_depth_error, iqr_rel_depth_error,
          median_parallax_deg, median_depth, depth_error_px}, one entry per pair direction
        - depth_residual_histogram: {counts, bin_edges}
    """

def compute_photometric_ncc(images, depth, intrinsics, extrinsics, original_coords=None,
                            max_separation=2, min_samples=32) -> dict:
    """... Returns: {idx1, idx2, photometric_ncc, n_pixels}; empty lists when no pair correlates."""

def compute_reconstruction_quality(
    collected: dict,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray,
    image_names: list[str],
    confidence: np.ndarray | None,
    images: np.ndarray | None,
    verification: dict | None,
) -> dict:
    """
    Every table the report holds, from arrays; the Reconstructor stage owns all IO.

    Returns:
        {"frames", "depth_pairs", "depth_residual_histogram", "photometric_pairs", "epipolar_pairs"}
        - photometric_pairs None without images; epipolar_pairs None without verification
    """
```

  `frames` columns, one entry per reconstruction frame `k`:
  - `frame_idx`: `frames.frame_idx_from_path(name)` if `_FRAME_STEM_RE.fullmatch(Path(name).stem)`, else None
  - `covered_fraction`: the old `crop_coverage` expression
  - `median_abs_rel_depth_error`: median of `abs(p.median_rel_depth_error)` over the pairs touching `k`, or None
  - `confidence_median`: `float(np.median(confidence[k]))`, or None without confidence
  - `mean_reproj_error_px` and `mean_reproj_error_frac_width`:
    - joined from `verification["frames"]` by `name == Path(image_names[k]).name`
    - `frac = px / image_width`, where `image_width = int(original_coords[0][4])`
    - None without verify or without a match

  `epipolar_pairs` is `verification["pairs"]` (already columnar) plus the `inlier_ratio` column, computed the old way.

  Deleted:
  - `read_quantiles`, `_running_error` and every correlation
  - ranks, `notes`, `available`/`reason`/`grid`/`units`
  - `extract_photometric` and `build_reconstruction_quality_report`
  - the `scipy.stats`, `time` and `json` imports, when unused
  - the old `notes` text, which moves into the module docstring as bullets together with every column, its units and its grid:
    - model-grid: depth tables
    - original-grid: photometric and epipolar tables

- [x] **Step 2: The stage.** `Reconstructor.reconstruction_quality_report`:

```python
        out_json = self.backend_dir / "reconstruction_quality_report.json"
        if not overwrite and self._stage_output_exists("reconstruction_quality_report"):
            if "frames" not in json.loads(out_json.read_text()):
                raise ValueError(
                    f"{out_json} is a stale reconstruction quality report (no 'frames'); delete it and re-run"
                )
            logger.info("Reconstruction quality report exists at %s, skipping", out_json)
            return out_json
        ...  # _resolve_result check and optional verify() unchanged

        # Load the result and run the dense multiview pass once
        # - abs_thresh 0.0 keeps the pass scale-invariant across backbones
        rel_thresh = 0.05
        ff = FeedforwardResult.load_zarr(self.pointcloud_zarr)
        collected: dict = {}
        compute_multiview_depth_confidence(
            ff.depth, ff.intrinsics, ff.extrinsics, abs_thresh=0.0, rel_thresh=rel_thresh, collect=collected
        )

        # Optional inputs: keyframes for photometric, verify's tables for epipolar
        # - read_frames is in filename order, the reconstruction's order; capped at N
        images = frames.read_frames(self.images_dir)[: len(ff.depth)].astype(np.float32) if frames.frame_paths(self.images_dir) else None
        verification = json.loads(verification_json.read_text()) if verification_json.exists() else None
        if verification is not None and "pairs" not in verification:
            raise ValueError(f"{verification_json} is a stale verification report (no 'pairs'); delete it and re-run")

        tables = compute_reconstruction_quality(
            collected, ff.depth, ff.intrinsics, ff.extrinsics, ff.original_coords,
            [Path(str(p)).name for p in ff.image_paths], ff.confidence, images, verification,
        )
        # scene: backend and model_resolution use the exact expressions the old stage/report used
        # - backend: the value the stage passed as build_reconstruction_quality_report(backend=)
        # - model_resolution: copy metrics.py's old `model_res` expression verbatim
        report = {
            "scene": {
                "backend": backend,
                "n_frames": len(ff.depth),
                "model_resolution": model_res,
                "image_width": int(ff.original_coords[0][4]),
                "zarr": str(self.pointcloud_zarr),
            },
            "params": {"rel_thresh": rel_thresh},
            **tables,
        }

        # Atomic write: a crash mid-dump must not leave a half file that reuse-by-existence trusts
        tmp = out_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(clean_for_json(report), indent=2, default=lambda o: o.item()))
        os.replace(tmp, out_json)
```

  - Imports: `FeedforwardResult` and `compute_multiview_depth_confidence` from `pointcloud.feedforward.base`, `compute_reconstruction_quality` from metrics, `clean_for_json` from transforms, and `frames` from preproc. Keep them inline for now; C18 decides whether to hoist.
  - Update the docstring bullets to match. The "{available: false}" bullet goes, and "null table without verify / images" takes its place.

- [x] **Step 3: T1 tests.**
  - Delete the metrics tests that assert deleted keys: correlations, quantiles, running error, ranks, notes, `available`, and `extract_photometric`/`build_*` IO.
  - Add column tests on `_write_tiny_scene`:
    - every table's columns have equal length
    - `frames` has N rows
    - `depth_pairs` idx columns equal the collected pair order
    - `photometric_pairs is None` without images
    - `epipolar_pairs is None` without verification
    - `frame_idx` is None for an off-contract name
    - `mean_reproj_error_frac_width == px / image_width`
    - nan is written as null (through the stage)
  - Stage tests in `tests/wrapper`:
    - the stage writes `{scene, params, frames, depth_pairs, depth_residual_histogram, photometric_pairs, epipolar_pairs}`
    - a stale old-format report raises "stale"
    - no `.json.tmp` is left behind
  - Keep every test of `residual_bin_edges`, `bounded_residual`, `depth_error_in_pixels` and the NCC math.

- [x] **Step 4: Write `report_eq_r3.py compare`.** It asserts, on the Task 0 scene, that the old and new values are equal:

| old | new |
|---|---|
| `measurements.depth.pair_directions[i][c]` | `depth_pairs[c][i]` for the 8 kept columns |
| `measurements.depth.residual_histogram.counts / bin_edges` | `depth_residual_histogram.counts / bin_edges` |
| `measurements.photometric.pairs[i][c]` | `photometric_pairs[c][i]` for idx1, idx2, photometric_ncc, n_pixels |
| `measurements.epipolar.pairs[i][c]` | `epipolar_pairs[c][i]` |
| `measurements.epipolar.frames` by name, px / frac | `frames.mean_reproj_error_px / _frac_width` |
| `source_frame_indices[k]` | `frames.frame_idx[k]` |
| `crop_coverage[k].covered_fraction` | `frames.covered_fraction[k]` |
| `scene.backend / n_frames / model_resolution` | same |

  The old epipolar values come from the old-format fixture. The new ones come from the same fixture converted to the columnar format with Task 11's writer.

  Run it. Expected: `REPORT EQ OK`.

- [x] **Step 5: `configs/README.md`.** Replace the report section's key list with the new layout. Copy the table from the metrics module docstring.
- [x] **Step 6:** Run the G′ gate. The test count drops by the deleted tests and rises by the new ones; list both in the report. Commit the metrics, reconstructor, README and tests with `refactor(geometry): reconstruction quality report follows the QA report contract`.

---

### Task 13: C10 — `metrics.py` order like `qa.py`

- [x] **Step 1:** Reorder `metrics.py` sections to match `qa.py`, with `#`×72 dividers:
  1. Constants (`_FRAME_STEM_RE`)
  2. Shared: residual axis and pixel equivalent (`residual_bin_edges`, `bounded_residual`, `depth_error_in_pixels`)
  3. Per pair: depth (`compute_depth_error`)
  4. Per pair: photometric (`_scale_intrinsics_to_original`, `compute_photometric_ncc`)
  5. Per frame + assembly (`compute_reconstruction_quality`)

  This is a pure move. Prove it by comparing the sorted top-level statements of HEAD and the working tree:

```bash
cd $WT && /opt/venv/reconstruction/bin/python - <<'PY'
import ast, subprocess
f = "collab_splats/geometry/metrics.py"
old = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True, check=True).stdout
new = open(f).read()
dump = lambda s: sorted(ast.dump(n) for n in ast.parse(s).body)
print("MOVE ONLY" if dump(old) == dump(new) else "CODE CHANGED")
PY
```

  Expected: `MOVE ONLY`. Divider comments are not AST nodes, so they may change freely.
- [x] **Step 2:** Run the G′ gate. Commit with `refactor(geometry): metrics sections ordered like preproc/qa.py`.

---

### Task 14: C14 — `graph.py` hoists and dedupe

**Files:** Modify `collab_splats/geometry/loop_closure/graph.py`

- [x] **Step 1:** `log = logging.getLogger(__name__)` at :192 moves to the top as `logger`; replace every `log.` with `logger.` in the file.
- [x] **Step 2:** Hoist `_lc_anchor_scale`'s nested `_frame_data` to a module-level `_frame_points(submap, idx, flat_idx)`, using its current body verbatim. Add a contract docstring.
- [x] **Step 3:** In `add_submap`, replace the duplicated inner-chain loop (once in the first-submap branch, once in the else-branch) with a method call placed after the if/else:

```python
    def _add_inner_chain(self, submap: Submap, node_ids: list[int]) -> None:
        """
        Chain edges between consecutive frames inside one submap.

        Args:
            submap: the submap whose frames are chained.
            node_ids: graph node id per frame, in frame order.
        """
        # body = the current loop, verbatim
```

  - Before merging, read both copies and confirm they are identical apart from indentation.
  - If they differ, keep both and record the difference.
- [x] **Step 4:** Replace the inline cam-local points loop in `add_submap` with the existing `_cam_local_points` helper, but only if it computes the same thing (same order, same dtype).
- [x] **Step 5:** Run `lc_parity_r3.py compare $SP/lc_r3_base.npz` (all methods). Expected: `PARITY OK`. Then run the G′ gate. Commit with `refactor(lc): one inner-chain loop, hoisted frame-point helper, logger`.

---

### Task 15: C14b — merge the confidence fallback

- [x] **Step 1:** Add one helper that `add_submap` and `_lc_anchor_scale` both call:

```python
def _conf_fallback_mask(
    curr_conf: np.ndarray, prior_conf: np.ndarray, conf_threshold: float, min_conf_points: int
) -> np.ndarray:
    """
    Points both frames trust, relaxed in tiers until enough survive.

    - tiers: joint > thr, then prior > thr, then prior > 0
    - the last tier still excludes zero-confidence (no-observation) pixels

    Args:
        curr_conf: (P,) confidence in the current frame.
        prior_conf: (P,) confidence in the prior frame.
        conf_threshold: confidence floor for the first two tiers.
        min_conf_points: survivors needed to stop relaxing.

    Returns:
        (P,) bool mask of the first tier with at least min_conf_points survivors, else the last tier.
    """
    for mask in (
        (curr_conf > conf_threshold) & (prior_conf > conf_threshold),
        prior_conf > conf_threshold,
        prior_conf > 0,
    ):
        if mask.sum() >= min_conf_points:
            return mask
    return mask
```

  Before writing it, read both current copies:
  - Match the comparison operators (`>` vs `>=`) and the count rule exactly as `_lc_anchor_scale` has them.
  - `add_submap` gains the `prior > 0` tier; this is the behavior change.

- [x] **Step 2:** Run LC parity. Expected: `PARITY OK`.
  - If it FAILS, revert this task's edit (Edit back; no git reset).
  - Add a comment at `add_submap`'s copy: `# no prior>0 tier here, unlike _lc_anchor_scale: parity case <key> changes`.
  - Commit that as `docs(lc): record why the two confidence fallbacks differ`, then move on.
- [x] **Step 3:** Run the G′ gate. Commit with `refactor(lc): one confidence fallback for window and loop scale`.

---

### Task 16: C15 — `wrapper.py` tidy

**Files:** Modify `loop_closure/wrapper.py` and `loop_closure/__init__.py`; tests that call `add_points`.

- [x] **Step 1:** Delete `_camera_centers_from_poses`. Its callers use `invert_poses(poses)[:, :3, 3]` (imported from transforms).
  - First confirm numerically, with a one-off script on random poses, that the two agree to 1e-6.
- [x] **Step 2:** Replace the view count repeated at `_enough_frames`, `run_predictions` and `_run_lc_loop` with a method `_n_views(self) -> int`.
  - First read all three sites and confirm they compute the same expression.
- [x] **Step 3:** Rename `add_points` to `_add_window_submap`, updating the callers in the package and the tests.
  - `rtk proxy grep -rn "add_points" collab_splats tests evals docs/source` first.
  - If a notebook or doc calls it, stop and report; do not edit notebooks.
- [x] **Step 4:** Add `LoopClosureConfig` to `wrapper.__all__`, if the module defines `__all__`.
- [x] **Step 5:** Run LC parity, then the G′ gate. Commit with `refactor(lc): tidy wrapper — invert_poses centers, one view count, private window add`.

---

### Task 17: C16 — dead code out, test hooks into tests

**Files:**
- Modify: `submap.py` (`conf_masks`), `map.py`, `graph.py`, `wrapper.py` (`_last_submaps`, `_last_lc_submaps`)
- Modify: `tests/geometry/loop_closure/_helpers.py`, `test_wrapper.py`

- [x] **Step 1:** Delete `Submap.conf_masks` (:48), after `rtk proxy grep -rn "conf_masks" collab_splats tests evals` shows no reader. Also delete any other `_last_*`/debug-only attribute in `map.py`/`graph.py` that the same grep proves has no production reader.
- [x] **Step 2:** Add the spy to `tests/geometry/loop_closure/_helpers.py`:

```python
@contextmanager
def record_driven_submaps():
    """
    Record every submap LoopClosure hands the pose graph, in call order.

    Yields:
        dict filled on exit: "submaps" (add_submap) and "lc_submaps" (add_loop_edge).
    """
    with (
        patch.object(PoseGraph, "add_submap", autospec=True, side_effect=PoseGraph.add_submap) as add,
        patch.object(PoseGraph, "add_loop_edge", autospec=True, side_effect=PoseGraph.add_loop_edge) as loop,
    ):
        driven: dict = {}
        yield driven
    driven["submaps"] = [c.args[1] for c in add.call_args_list]
    driven["lc_submaps"] = [c.args[1] for c in loop.call_args_list]
```

  - Check the positional index: `add_loop_edge`'s `args[1]` must be the submap the old `_last_lc_submaps` recorded. Read the wrapper's `_add_loop_edge` to confirm what was appended, and pick the matching arg.
  - If `_last_lc_submaps` held something not passed to `add_loop_edge`, spy on the wrapper method that receives it instead.
- [x] **Step 3:** Rewrite the `test_wrapper.py` uses (:546-547, :642, :726, :773, :849, :1018, :1211, :1280) as `with record_driven_submaps() as driven: lc.run(...)`, then assert on `driven[...]`. Then delete `_last_submaps`/`_last_lc_submaps` from `wrapper.py` (:136-137, :626-627).
- [x] **Step 4:** Run `pytest tests/geometry/loop_closure`. Expected: same count. Run LC parity (the harness already uses `n_loops_applied`), then the G′ gate. Commit with `refactor(lc): drop dead conf_masks and test-only hooks; tests spy instead`.

---

### Task 18: E3 — trim `loop_closure/__init__.py`

- [x] **Step 1:** For each name in `__all__`, run `rtk proxy grep -rn "from collab_splats.geometry.loop_closure import" collab_splats evals docs/source .worktrees/tutorial-rework/docs 2>/dev/null` (the tutorial-rework grep is read-only).
  - Keep every name any non-test module imports through the package, and always keep `PoseGraph`, `LoopClosure` and `LoopClosureConfig`.
  - Drop the rest (`dedup_overlap`, `estimate_scale_pairwise`, `assert_world_to_cam`, `decompose_camera`, …) from the package exports.
  - Tests import those from their submodules.
- [x] **Step 2:** Run `pytest tests/geometry`, then the G′ gate. Commit with `refactor(lc): package exports only what callers import`.

---

### Task 19: C1 — new contract checks, geometry only

**Files:** Modify `tests/test_docstring_contract.py`

- [x] **Step 1: Checks plus fixture tests.** Add these after `_silent_fallbacks`:

```python
def _nested_defs(src: str) -> list[str]:
    """
    Functions defined inside functions.

    Args:
        src: module source.

    Returns:
        "<line>: nested def <inner> in <outer>" per offender.
    """
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    out, seen = [], set()

    # ast.walk is breadth-first, so each inner def is reported against its outermost function
    for outer in ast.walk(ast.parse(src)):
        if not isinstance(outer, funcs):
            continue
        for inner in ast.walk(outer):
            if inner is not outer and isinstance(inner, funcs) and inner.lineno not in seen:
                seen.add(inner.lineno)
                out.append(f"{inner.lineno}: nested def {inner.name} in {outer.name}")
    return out


def _quote_line_docstrings(src: str) -> list[str]:
    """
    Docstrings whose summary starts on the opening-quote line.

    Args:
        src: module source.

    Returns:
        "<label>: summary on the quote line" per offender.
    """
    tree = ast.parse(src)
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    out = []
    for n in nodes:
        doc = ast.get_docstring(n, clean=False)
        if doc is not None and not doc.startswith("\n"):
            out.append(f"{getattr(n, 'name', 'module')}: summary on the quote line")
    return out


def _prose_comment_pairs(src: str) -> list[str]:
    """
    Two-line comment runs whose second line is neither a bullet nor an indented continuation.

    - MIN_RUN already covers runs of 3+; this closes the 2-line gap
    - dividers and runs led by RUN_SKIP_PREFIXES are skipped

    Args:
        src: module source.

    Returns:
        "<line>: 2-line comment is prose" per offender.
    """
    lines = src.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("#"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        run = [lines[k].strip()[1:] for k in range(i, j)]
        text = [r for r in run if r.strip()]
        if j - i == 2 and len(text) == 2 and not text[0].strip().startswith(RUN_SKIP_PREFIXES):
            second = text[1]
            if not second.strip().startswith("- ") and not second.startswith("   "):
                out.append(f"{i + 1}: 2-line comment is prose — {text[0].strip()[:60]!r}")
        i = j
    return out
```

  - Lambdas are not defs; methods of a class are not nested.
  - Fixture tests:

```python
def test_nested_def_check_flags_an_inner_function():
    src = "def f() -> None:\n    def g() -> None:\n        pass\n    g()\n\nclass C:\n    def m(self) -> None:\n        pass\n"
    assert _nested_defs(src) == ["2: nested def g in f"]


def test_quote_line_docstring_check_flags_a_one_liner():
    ok = 'def f() -> None:\n    """\n    Summary.\n    """\n'
    bad = 'def _g() -> None:\n    """Summary."""\n'
    assert _quote_line_docstrings(ok) == []
    assert _quote_line_docstrings(bad) == ["_g: summary on the quote line"]


def test_prose_comment_pair_check_flags_prose_and_spares_bullets():
    assert _prose_comment_pairs("# a header\n# - a bullet\nx = 1\n") == []
    assert _prose_comment_pairs("# first sentence of prose\n# second sentence\nx = 1\n") == [
        "1: 2-line comment is prose — 'first sentence of prose'"
    ]
    assert _prose_comment_pairs("# a header\n#    continued\nx = 1\n") == []
```

- [x] **Step 2: Wire up, geometry only.**

```python
# Round-3 checks: enforced for geometry only; other packages are a changelog follow-up
ROUND3_CHECKS = {
    "nested-def": _nested_defs,
    "quote-line-docstring": _quote_line_docstrings,
    "prose-comment-pair": _prose_comment_pairs,
}


@pytest.mark.parametrize(
    ("path", "check"),
    [
        pytest.param(p, name, id=f"{pid}::{name}")
        for p, pid in zip(SOURCES, SOURCE_IDS)
        if p.relative_to(ROOT / "collab_splats").parts[0] == "geometry"
        for name in ROUND3_CHECKS
    ],
)
def test_round3_rules(path, check):
    bad = ROUND3_CHECKS[check](path.read_text())
    assert not bad, "\n".join(bad)
```

- [x] **Step 3:** Run `pytest tests/test_docstring_contract.py -q`. Expected: all geometry files pass.
  - A failure is a Round 1 miss; fix the prose in geometry (prose only, then `prose_proof.py HEAD`) in the same commit.
  - Mutation check: temporarily nest a def in `transforms.py`; `test_round3_rules[...transforms.py::nested-def]` must fail. Revert with Edit.
- [x] **Step 4:** Run the G′ gate. Commit with `test(contract): nested defs, quote-line docstrings, 2-line prose comments (geometry)`.

---

### Task 20: C18 — hoist the function-level imports

**Files:** Modify `collab_splats/wrapper/reconstructor.py`, `evals/scripts/eval.py`

- [x] **Step 1: Measure each candidate.** Candidates: `geometry.bundle_adjustment`, `geometry.metrics`, `geometry.transforms`, `pointcloud.feedforward.base`, `preproc.frames`. For each, run:

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -c "
import sys, importlib; importlib.import_module('collab_splats.wrapper.reconstructor'); before=set(sys.modules)
importlib.import_module('<candidate>')
new={m.split('.')[0] for m in set(sys.modules)-before}
print('<candidate>', sorted(new & {'vismatch','vggt','vggsfm','bae','pypose','torch','mapanything'}))"
```

  - Hoist a module only if its new-heavy set is empty, or already present after importing `reconstructor`.
  - Check that `python -c "import collab_splats.wrapper.reconstructor"` still works after the hoist, meaning there is no cycle.
  - `bundle_adjustment` pulls in bae/pypose/vggt; if `reconstructor` does not import them already, BA stays inline.
  - Verification stays inline (vismatch).
  - Record the measurements in the commit body.
- [x] **Step 2:** Apply the same rule to `eval.py`'s function-level imports.
- [x] **Step 3:** Run the G′ gate, including `tests/wrapper` (import-time failures show there). Commit with `refactor(wrapper): hoist geometry imports that pull no heavy deps`.

---

### Task 21: E4 — `eval.py` nested `_ba` and the ×3 no-LC config

- [x] **Step 1:** Read `_make_creator` (:230-310). Hoist the nested `_ba` to a module-level `_run_ba(creator, cfg)`, using the same body with its closure variables passed in.
- [x] **Step 2:** Replace the three identical no-LC `LoopClosureConfig(...)` literals with one:

```python
def _no_lc_config(submap_size: int, submap_overlap: int, **extra) -> LoopClosureConfig:
    """
    Loop-closure config that windows the sequence but never closes a loop.

    Args:
        submap_size: frames per submap.
        submap_overlap: frames shared by consecutive submaps.
        extra: remaining LoopClosureConfig fields, passed through.

    Returns:
        The config.
    """
```

  - Its body is the literal's fields, taken from the current code verbatim.
  - First diff the three literals; if they are not identical, the helper takes the differing fields as args.
- [x] **Step 3:** Run `pytest tests/evals -q`, then the G′ gate. Commit with `refactor(evals): hoist _ba; one no-LC config builder`.

---

### Task 22: E2 + C13 — delete `scale_method`

**Files:**
- Modify: `graph.py` (`SCALE_METHODS`, `check_scale_method`, `none` branches), `wrapper.py` (field, Literal, `__post_init__` check)
- Modify: `evals/scripts/eval.py` (:102, :120, :135, :157, :234, :264, :383, :403), `configs/loop_closure.yaml`, `evals/configs/*.yaml`, `evals/README.md`
- Tests: `tests/geometry/loop_closure/*` (drop the `none` cases and the `scale_method` params in `_helpers.drive_pose_graph`), `tests/evals/*`

- [x] **Step 1: Last full-method parity.** Run `lc_parity_r3.py compare $SP/lc_r3_base.npz rotation_only,none`. Expected: `PARITY OK`.
- [x] **Step 2: Delete.**
  - Delete `SCALE_METHODS`, `check_scale_method`, every `if scale_method == "none"` branch (keeping the `rotation_only` path inline), the `scale_method` parameter on graph methods, `LoopClosureConfig.scale_method` and its post-init check, and `--lc_scale_method` together with every eval pass-through.
  - In YAML, delete the `scale_method:` keys.
  - First grep: `rtk proxy grep -rn "scale_method" collab_splats evals configs tests docs/source`.
    - After the edit it must return nothing outside `docs/superpowers`.
    - One exception is allowed: a yaml-loader rejection test, if the config loader rejects unknown keys. Check how `LoopClosureConfig(**yaml)` behaves.
    - If an old yaml with `scale_method` would now raise `TypeError`, that is acceptable (a loud error). Note it in the hand-off.
- [x] **Step 3:** Run `lc_parity_r3.py compare $SP/lc_r3_base.npz rotation_only`. Expected: `PARITY OK`, meaning `rotation_only` is unchanged by the deletion. Then re-baseline with `lc_parity_r3.py capture $SP/lc_r3_final.npz rotation_only`.
- [x] **Step 4:** Run the G′ gate; the test count drops by the removed `none` cases, which must be listed. Commit with `refactor(lc): delete scale_method (none was the only alternative)`.

---

### Task 23: C19 — docs, changelog, CLAUDE.md, hand-off

**Files:**
- Modify: `docs/source/api/geometry.rst`, `docs/parity.md`, `docs/superpowers/CHANGELOG.md`, `CLAUDE.md` (the tree only), `docs/known-test-failures.md` (only if the baseline entries moved)
- Create: `docs/superpowers/handoffs/2026-09-26-geometry-round3-tutorial.md` (if the dir exists; else append to the changelog entry)

- [x] **Step 1: `geometry.rst`.**
  - Add `project_to_so3`, `intrinsics_4x4`, `decompose_camera`, `clean_for_json`, `check_model_resolution` and `compute_reconstruction_quality`.
  - Remove `build_reconstruction_quality_report` and `extract_photometric`.
  - Check with `rtk proxy grep -n "autofunction\|automodule\|autoclass" docs/source/api/geometry.rst`, and confirm each target exists by importing it.
- [x] **Step 2: `docs/parity.md`.**
  - Drop the `scale_method: none` parity rows.
  - Note that the parity harness is `rotation_only`-only after E2.
- [x] **Step 3: CHANGELOG entry.** Add **geometry-round3 (2026-09-26)**, with bullets for:
  - the rows landed, the spec deltas, and the LOC delta measured with `git diff --stat ac93e96b -- collab_splats/geometry tests evals`
  - the gate counts, before and after
  - follow-ups:
    - C1 checks for pointcloud (8 nested defs) and preproc (1)
    - C17 not done
    - the track caches re-extract once (C5)
    - old reports raise "stale"
- [x] **Step 4: CLAUDE.md tree.**
  - The `geometry/` lines mention `transforms.py`'s new helpers and `metrics.py`/`verification.py`.
  - Keep CLAUDE.md in-flight only; do not add a completed entry, since the hook enforces this.
- [x] **Step 5: Hand-off for tutorial-rework.** Name every break:
  - `02_pointcloud/bundle_adjustment.ipynb`: the `ba.refine(result)` call becomes the array form plus `check_model_resolution`
  - any notebook reading `reconstruction_quality_report.json` or `verification.json` keys
  - `LoopClosureConfig(scale_method=...)`
  - `from collab_splats.geometry.loop_closure import decompose_camera` and the other dropped exports
  - Find them with a read-only grep over `.worktrees/tutorial-rework/docs` and `docs/source/tutorials`.
- [x] **Step 6: Final gates.**
  - G′
  - `lc_parity_r3.py compare $SP/lc_r3_final.npz rotation_only`
  - `ba_eq_r3.py compare $SP/ba_r3_base.npz`
  - `report_eq_r3.py compare`
  - `git diff --stat ac93e96b -- collab_splats/geometry`, for the LOC delta
- [x] **Step 7: Commit** with `git add -f` for docs/superpowers, then `docs(geometry): round-3 docs, changelog, tutorial hand-off`. Do not merge or push.
