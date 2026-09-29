# Pointcloud Release Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the duplicates and overengineering left in `pointcloud/` after unify, share the create template across feedforward and sfm, and bring docstrings/comments to geometry style — every commit net-negative in lines.

**Architecture:** Spec [2026-09-28-pointcloud-release-dedup-design.md](../specs/2026-09-28-pointcloud-release-dedup-design.md). One per-pair helper (`depth_agreement`) in `geometry/projection.py` feeds both the feedforward multiview filter and the report's pair stats. `BasePointcloudCreator` becomes a dataclass owning `create_pointcloud`; subclasses implement `_reconstruct(paths, out_dir)`. MapAnything stacks its per-view list into the base raw dict so one base `_postprocess` serves all four feedforward backends.

**Tech Stack:** Python 3.11, numpy, torch, pycolmap, pytest; venv `/opt/venv/reconstruction`.

---

## Ground rules (every task)

- Worktree: `/workspace/collab-splats/.worktrees/pointcloud-release`. `cd` there at the start of every Bash call; the shell cwd resets.
- Python: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python ...`. First run of a session prints `python -c "import collab_splats; print(collab_splats.__file__)"` and it must point into the worktree.
- Format per touched file only: `black --target-version py311 -l 120 <file> && isort <file>`. Never repo-wide.
- Commit: `git commit --only <paths> -m "..."` with trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `git add -f` for anything under `docs/superpowers/`. No amend, rebase, reset, revert, bare stash, merge, push. Restore a file via `cp`, never `git checkout --`.
- Net-negative: before each commit, `git diff --cached --shortstat` (or `git diff --shortstat <paths>`) shows deletions > insertions. A task that can't meet this stops and reports.
- No K-undo wrapper, no layout constants, no nested `f(g(x))` (one call per line), US spelling.
- Gate: `WT=$WT bash /workspace/scratch/pc-release/unify_gate.sh <tag> 2>&1 | grep -av Warp` (Bash timeout 600000, foreground). rc is 0 even on failure — **read** the G′ line, `side rc=`, and the parity `SUMMARY`/`FAIL` lines.
  - baseline chain: `/workspace/scratch/pc-release/parity_baseline_unify` only; never touch `parity_baseline/`
  - a parity FAIL on an expected-move case: STOP, report the diff to the user, and wait. Re-record only after the user accepts.
- `/ffbase.bak` is the user's to delete; leave it.
- `docs/source/**` notebooks are out of scope (no notebook edits).

Expected baseline moves, each reported before it is accepted:

| Task | Case | Why |
|---|---|---|
| 3 | mapanything | abs_thresh 0.02 dropped; `seen == 0` judged pixels now kept |
| 6 | mapanything, only if moved | world points from `unproject` instead of upstream pts3d |
| 8 | sfm tests (not in harness) | sfm SOR deleted; postprocess default off |
| 12 | depth_align, only if moved | pixel mapping floor then clip |
| 16 | vggtx, vggt_omega, loger | `>=` percentile → `confidence_mask` strict `>` |

---

## File map

| File | Change |
|---|---|
| `collab_splats/geometry/projection.py` | `depth_residual` also returns `pixels`; new `depth_agreement`, `multiview_depth_confidence` |
| `collab_splats/geometry/metrics.py` | own pair loop via `depth_agreement`; `multiview_agreement` per frame; photometric takes images-grid K |
| `collab_splats/pointcloud/base.py` | `PointcloudResult.mv_*` removed; `BasePointcloudCreator` dataclass + `create_pointcloud` |
| `collab_splats/pointcloud/utils.py` | `clean_pointcloud` → `outlier_mask`; new `clean_pointcloud(result, ...)`; `mean_top_quarter` deleted |
| `collab_splats/pointcloud/feedforward/base.py` | mv block, frame decoding, `run`/`create`/`postprocess`/`clean_outputs`/`write_colmap_model`, `_verify_geometry`, `_raw_to_world_points`, `_mask_to_points`, `unproject_and_filter_points`, `full_frame_coords` deleted; `_reconstruct` added |
| `collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,loger,mapanything}.py` | path-based `_preprocess`; MapAnything `_stack_predictions` |
| `collab_splats/pointcloud/feedforward/__init__.py` | exports trimmed |
| `collab_splats/pointcloud/sfm/base.py` | SOR deleted; `_reconstruct`; export override |
| `collab_splats/pointcloud/depth.py` | floor pixel rule; fit folded into `align_depth`; VDA note |
| `collab_splats/geometry/loop_closure/wrapper.py` | `create_pointcloud` + `_reconstruct`; `reconstruct`/`run`/`postprocess` deleted; collate branch deleted |
| `collab_splats/semantics/lifting.py` | drops its `project` call |
| `collab_splats/wrapper/reconstructor.py` | callers, config keys, report, refine_poses, localize ids |
| `collab_splats/dashboard/pipeline.py`, `evals/scripts/{eval,eval_run_backend,eval_multiview_conf,eval_similarity_calibration,ba_start_at_gt}.py` | callers |
| `configs/base.yaml` | `use_multiview_confidence` → `min_views`, `mv_rel_thresh` |
| `docs/parity.md`, `docs/pointcloud.md` (or the existing pointcloud doc) | `mean_top_quarter` inline; VDA license |
| `/workspace/scratch/pc-release/parity.py` | `_run` on the new API |

---

## Section 1 — Multiview depth confidence

### Task 1: `depth_agreement` + `multiview_depth_confidence` in projection.py

**Files:**
- Modify: `collab_splats/geometry/projection.py` (after `depth_residual`, line ~137)
- Test: `tests/geometry/test_projection.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/geometry/test_projection.py`:

```python
def _two_view_scene(depth_b: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two identical cameras at the origin looking down +Z at a fronto-parallel plane.

    - view 0 at depth 2 everywhere; view 1 at depth_b everywhere
    """
    h, w = 8, 10
    depth = torch.stack([torch.full((h, w), 2.0), torch.full((h, w), depth_b)])
    intrinsics = torch.tensor([[10.0, 0, (w - 1) / 2], [0, 10.0, (h - 1) / 2], [0, 0, 1]]).expand(2, 3, 3)
    extrinsics = torch.eye(4).expand(2, 4, 4)
    return depth, intrinsics, extrinsics


def test_multiview_depth_confidence_agree():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert agree.shape == seen.shape == (2, 8, 10)
    assert (agree == 1).all() and (seen == 1).all()


def test_multiview_depth_confidence_occluded_is_unseen():
    # View 1 sees a surface in front of view 0's points: occluded, left out of seen
    depth, intrinsics, extrinsics = _two_view_scene(1.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 0).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_free_space_violation_is_seen_not_agree():
    # View 1 sees behind view 0's points: counted as seen, not as agreeing
    depth, intrinsics, extrinsics = _two_view_scene(3.0)

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 1).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_out_of_bounds_is_unseen():
    # View 1 shifted far sideways: nothing projects in bounds
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    extrinsics = extrinsics.clone()
    extrinsics[1, 0, 3] = 100.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 0).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_zero_source_depth_counts_nothing():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    depth[0, :2] = 0.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0, :2] == 0).all() and (agree[0, :2] == 0).all()
    assert (seen[0, 2:] == 1).all()


def test_multiview_depth_confidence_zero_target_depth_is_seen_not_agree():
    # A hole in view 1: the pixel is seen (not occluded) but has nothing to agree with
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    depth[1] = 0.0

    agree, seen = multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)

    assert (seen[0] == 1).all() and (agree[0] == 0).all()


def test_multiview_depth_confidence_rejects_original_res_intrinsics():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 2] = 50.0

    with pytest.raises(ValueError, match="principal point"):
        multiview_depth_confidence(depth.numpy(), intrinsics.numpy(), extrinsics.numpy(), rel_thresh=0.01)


def test_multiview_depth_confidence_rejects_length_mismatch():
    depth, intrinsics, extrinsics = _two_view_scene(2.0)

    with pytest.raises(ValueError, match="length mismatch"):
        multiview_depth_confidence(depth.numpy(), intrinsics[:1].numpy(), extrinsics.numpy(), rel_thresh=0.01)
```

Add `multiview_depth_confidence` to the file's existing `from collab_splats.geometry.projection import ...` line, and `import pytest` if absent.

- [ ] **Step 2: Run to verify they fail**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_projection.py -q`
Expected: collection error `ImportError: cannot import name 'multiview_depth_confidence'`.

- [ ] **Step 3: Implement** — append after `depth_residual` in `collab_splats/geometry/projection.py` (add `import numpy as np` and `from collab_splats.utils.torch_utils import get_device` at the top if absent; check for an import cycle with `python -c "import collab_splats.geometry.projection"`):

```python
def depth_agreement(
    points_world: Tensor, world_to_cam: Tensor, intrinsics: Tensor, depth: Tensor, rel_thresh: float
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Whether world points agree with another view's depth map, within a relative tolerance.

    - occluded (the view sees a nearer surface) is no evidence: not seen
    - a hole in the view's depth map is seen but cannot agree
    - scale-free: tolerance is rel_thresh times the point's own depth

    Args:
        points_world: (P, 3) world points.
        world_to_cam: (4, 4) w2c of the view whose depth is read.
        intrinsics: (3, 3) camera matrix on the depth map's pixel grid.
        depth: (H, W) z-depth of that view; 0 marks no depth.
        rel_thresh: tolerance as a fraction of the point's depth.

    Returns:
        (agree, seen, rel_residual), each (P,)
        - rel_residual: (sampled - expected) / expected, meaningful where seen and sampled > 0
    """
    residual, expected, sampled, valid = depth_residual(points_world, world_to_cam, intrinsics, depth)

    # Nearer sampled surface = occluded; farther = free-space violation, still seen
    tol = rel_thresh * expected.abs()
    has_depth = sampled > 0
    occluded = (sampled < expected - tol) & has_depth
    seen = valid & ~occluded
    agree = seen & has_depth & (residual.abs() < tol)

    return agree, seen, (sampled - expected) / expected


def multiview_depth_confidence(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, rel_thresh: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel count of other views that see the pixel and that agree with its depth.

    - port of mapanything/utils/multiview_confidence.py:125 (facebookresearch/map-anything)
    - every ordered pair, O(N^2); pixels with no source depth count nothing
    - depth, intrinsics and extrinsics on one pixel grid, OpenCV, +Z z-depth

    Args:
        depth: (N, H, W) z-depth per frame; 0 marks no depth.
        intrinsics: (N, 3, 3) K on the depth grid.
        extrinsics: (N, 4, 4) w2c.
        rel_thresh: agreement tolerance as a fraction of depth.

    Returns:
        (agree, seen), both (N, H, W) int32.

    Raises:
        ValueError: N disagrees across the arrays, or the principal point lies outside the depth
            grid (model-res depth paired with original-res intrinsics).
    """
    n, h, w = depth.shape

    # Alignment contract: same N, K on the depth grid
    if len(intrinsics) != n or len(extrinsics) != n:
        raise ValueError(
            f"length mismatch: depth has {n} frames, intrinsics {len(intrinsics)}, extrinsics {len(extrinsics)}"
        )
    cx, cy = float(intrinsics[0][0, 2]), float(intrinsics[0][1, 2])
    if not (0 < cx < w and 0 < cy < h):
        raise ValueError(
            f"intrinsics/depth resolution mismatch: principal point ({cx:.1f}, {cy:.1f}) lies outside a "
            f"{w}x{h} depth grid; model-resolution depth was probably paired with original-resolution intrinsics"
        )

    device = torch.device(get_device())
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    agree = torch.zeros(n, h * w, dtype=torch.int32, device=device)
    seen = torch.zeros(n, h * w, dtype=torch.int32, device=device)

    # Every source frame against every other frame
    for i in range(n):
        points = unproject(depth_t[i], extrinsics_t[i], intrinsics_t[i]).reshape(-1, 3)
        has_source = depth_t[i].reshape(-1) > 0

        for j in range(n):
            if i == j:
                continue
            agree_ij, seen_ij, _ = depth_agreement(points, extrinsics_t[j], intrinsics_t[j], depth_t[j], rel_thresh)
            agree[i] += (agree_ij & has_source).int()
            seen[i] += (seen_ij & has_source).int()

    return agree.reshape(n, h, w).cpu().numpy(), seen.reshape(n, h, w).cpu().numpy()
```

- [ ] **Step 4: Run to verify they pass**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_projection.py -q`
Expected: all pass.

- [ ] **Step 5: Format + commit** (this task alone is net-positive; it is folded into Task 2's commit — do NOT commit yet, carry the diff into Task 2).

### Task 2: Feedforward filter, config swap, deletions

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py` (delete lines ~210-514: `_frustum_world_aabbs`, `_aabbs_overlap`, `MultiviewConfidence`, `multiview_mask`, `_mv_result_fields`, `compute_multiview_depth_confidence`; `_multiview` ~895-920; fields ~685-697; module docstring line 4)
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py` (fields `use_multiview_confidence`, `mv_conf_abs_thresh`, `mv_conf_rel_thresh`, `min_views`; `_multiview` call ~318-324; `**_mv_result_fields`)
- Modify: `collab_splats/pointcloud/feedforward/__init__.py` (`__all__` drops `MultiviewConfidence`, `compute_multiview_depth_confidence`, `multiview_mask`)
- Modify: `collab_splats/pointcloud/base.py` (`mv_ratio`, `mv_inlier_count`, `mv_valid_count` fields ~67-69, docs ~95-97, zarr ~180-182 and the load path)
- Modify: `collab_splats/wrapper/reconstructor.py:413,436,440,489,1096` and `configs/base.yaml:51-58`
- Modify: `evals/scripts/eval_multiview_conf.py:27-28,137-139`, `evals/scripts/eval_run_backend.py` (drop `use_multiview_confidence`)
- Tests: `tests/pointcloud/test_mv_conf.py`, `test_mv_creator_wiring.py`, `test_mv_zarr_roundtrip.py`, `tests/wrapper/test_reconstructor_mv_config.py`, `tests/pointcloud/feedforward/test_mapanything_creator.py`

- [ ] **Step 1: Rewrite the wiring test** — replace the body of `tests/pointcloud/test_mv_creator_wiring.py` with:

```python
"""
Multiview filter wiring in the feedforward _postprocess.
"""

import numpy as np

from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from tests.pointcloud._stubs import vggt_raw_outputs


def _run(creator: VGGTXCreator) -> int:
    raw = vggt_raw_outputs(n=3, h=8, w=10)
    creator.image_paths = [f"frame_{i:06d}" for i in range(3)]
    creator.original_coords = np.tile(np.array([0, 0, 10, 8, 10, 8], np.float32), (3, 1))
    return len(creator._postprocess(raw).points)


def test_min_views_zero_is_off():
    assert _run(VGGTXCreator(min_views=0)) == _run(VGGTXCreator(min_views=0, mv_rel_thresh=1e-9))


def test_min_views_filters_on_disagreement():
    assert _run(VGGTXCreator(min_views=1, mv_rel_thresh=1e-9)) < _run(VGGTXCreator(min_views=0))
```

Read `tests/pointcloud/_stubs.py` first. If it has no raw-dict builder, add `vggt_raw_outputs(n, h, w)` there: random depth in `[1, 2]` shaped `(n, h, w, 1)`, identity-plus-small-translation `extrinsic` `(n, 3, 4)`, a K with `cx=(w-1)/2`, `cy=(h-1)/2`, `depth_conf` ones `(n, h, w)`, `images` rand `(n, 3, h, w)`, seeded `np.random.default_rng(0)`.

- [ ] **Step 2: Run to verify it fails**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_creator_wiring.py -q`
Expected: FAIL, `TypeError: ... unexpected keyword argument 'mv_rel_thresh'`.

- [ ] **Step 3: Implement the ff filter** — in `feedforward/base.py`:

Fields (replace the four mv fields and their comments at ~685-697):

```python
    # Multiview filter: keep a pixel when min(min_views, seen) other views agree; 0 = off
    min_views: int = 0
    mv_rel_thresh: float = 0.01
```

Replace `_multiview` (~895-920) with:

```python
    def _multiview_mask(self, depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """
        Keep-mask of pixels that min(min_views, seen) other views agree with; all-True when off.

        - seen == 0 keeps the pixel: nothing to disagree with
        """
        if self.min_views == 0:
            return np.ones(depth.shape, dtype=bool)
        agree, seen = multiview_depth_confidence(depth, intrinsics, extrinsics, rel_thresh=self.mv_rel_thresh)
        return agree >= np.minimum(self.min_views, seen)
```

In `_postprocess` replace the two mv lines with:

```python
        # Optional cross-view depth consistency, ANDed onto the confidence mask
        mv_mask = self._multiview_mask(depth, intrinsic, extrinsic_4x4)
```

and drop `**_mv_result_fields(mv_conf),`. `extra_mask=mv_mask` stays until Task 16.

Delete the six mv definitions listed under **Files**, the module-docstring line naming them, and every import only they used (`tqdm`, `time` if unused elsewhere, `residual_bin_edges`, `bounded_residual`, `PairStats`, `depth_residual`). Add `from collab_splats.geometry.projection import multiview_depth_confidence` (merge into the existing projection import).

- [ ] **Step 4: MapAnything** — in `mapanything.py` replace the four mv fields with:

```python
    # Upstream's multiview calibration
    min_views: int = 1
    mv_rel_thresh: float = 0.02
```

Replace the `mv_conf, combined_mask = self._multiview(...)` block with:

```python
        # Shared cross-view depth filter on top of the upstream mask
        extrinsics_4x4 = extrinsics_to_homogeneous(np.stack(extrinsics_list))
        combined_mask &= self._multiview_mask(np.stack(depth_list), np.stack(intrinsics_list), extrinsics_4x4)
```

and drop `**_mv_result_fields(mv_conf)` and the `_mv_result_fields` import.

- [ ] **Step 5: PointcloudResult** — in `pointcloud/base.py` delete the three `mv_*` fields, their docstring bullets, their `save_zarr` writes and the `load_zarr` reads. Delete `tests/pointcloud/test_mv_zarr_roundtrip.py` (it only tests those fields; confirm with a read first).

- [ ] **Step 6: Config + reconstructor** — `configs/base.yaml` replace lines 51-58 with:

```yaml
  # Cross-view depth consistency filter (feedforward only); 0 = off
  # - keeps a pixel when min(min_views, seen) other views agree within mv_rel_thresh
  # - off: on chess/seq-01 it cuts the >10%-error fraction ~5% for ~2% of pixels at O(N^2)
  min_views: 0
  mv_rel_thresh: 0.01
```

`reconstructor.py`: in `_run_feedforward` rename the `use_multiview_confidence: bool` param to `min_views: int, mv_rel_thresh: float`, update its `Args:` entries, the reserved-kwargs doc line, and:

```python
    explicit = {"max_points": max_points, "min_views": min_views, "mv_rel_thresh": mv_rel_thresh, "clean": clean}
```

At the call site (~1096) pass `min_views=pc_cfg["min_views"], mv_rel_thresh=pc_cfg["mv_rel_thresh"]`. Grep `configs/` for other `use_multiview_confidence` keys and swap them the same way.

- [ ] **Step 7: Evals scripts** — `evals/scripts/eval_multiview_conf.py`: import `multiview_depth_confidence` from `collab_splats.geometry.projection`; at 137-139 replace the old call + `multiview_mask` with:

```python
        agree, seen = multiview_depth_confidence(depth, intrinsics, extrinsics, rel_thresh=rel)
        keep = agree >= np.minimum(k, seen)
```

(keep the script's own variable names for `k`/`rel`; read the loop first). `eval_run_backend.py`: delete the `use_multiview_confidence=` kwarg.

- [ ] **Step 8: Old tests** — `tests/pointcloud/test_mv_conf.py` tests the deleted API. Keep only tests whose property still holds and port them to `multiview_depth_confidence` (scale invariance at any `s`, principal-point guard is already in Task 1 → delete duplicates). Delete the rest (`MultiviewConfidence`, `judged`, `pair_gate`, `abs_thresh`, `collect=`). `tests/wrapper/test_reconstructor_mv_config.py`: swap `use_multiview_confidence` for `min_views`/`mv_rel_thresh`. `test_mapanything_creator.py`: drop asserts on removed fields.

- [ ] **Step 9: Grep clean**

Run: `cd $WT && rtk proxy grep -rn "use_multiview_confidence\|mv_conf_\|MultiviewConfidence\|multiview_mask\|_mv_result_fields\|compute_multiview_depth_confidence\|mv_ratio\|mv_inlier\|mv_valid\|_aabbs_overlap\|_frustum_world" --include=*.py --include=*.yaml collab_splats evals tests configs`
Expected: only `reconstructor.py`'s report call (Task 3) and `metrics.py` remain.

- [ ] **Step 10: Run tests**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/geometry/test_projection.py tests/wrapper -q -x` — the report test may still fail until Task 3; note it and carry on.

Task 2 does not commit alone; Task 3 finishes Section 1.

### Task 3: Report pair loop in metrics + `multiview_agreement`

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (`compute_reconstruction_quality` ~300-400)
- Modify: `collab_splats/wrapper/reconstructor.py:44,100,1519-1556`
- Test: `tests/geometry/test_metrics.py`, `tests/geometry/test_metrics_controls.py`

- [ ] **Step 1: Failing test** — append to `tests/geometry/test_metrics.py` (reuse that file's existing synthetic-scene fixture/helper that builds `depth`, `model_intrinsics`, `extrinsics`, `original_coords`; read it first and use its name):

```python
def test_report_multiview_agreement_is_one_on_consistent_scene(consistent_scene):
    tables = compute_reconstruction_quality(
        consistent_scene.depth,
        consistent_scene.model_intrinsics,
        consistent_scene.extrinsics,
        consistent_scene.original_coords,
        consistent_scene.image_names,
        confidence=None,
        images=None,
        intrinsics=None,
    )

    agreement = tables["frames"]["multiview_agreement"]
    assert len(agreement) == len(consistent_scene.depth)
    assert all(a == pytest.approx(1.0) for a in agreement if a is not None)
```

- [ ] **Step 2: Run** — `pytest tests/geometry/test_metrics.py -q -k multiview_agreement` → FAIL (signature).

- [ ] **Step 3: Implement** — `compute_reconstruction_quality` new signature:

```python
def compute_reconstruction_quality(
    depth: np.ndarray,
    model_intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray,
    image_names: list[str],
    confidence: np.ndarray | None,
    images: np.ndarray | None,
    intrinsics: np.ndarray | None,
    rel_thresh: float = 0.05,
) -> dict:
```

`intrinsics` is the full-res K (used from Task 11 on; until then photometric keeps `model_intrinsics` + `original_coords` exactly as today). Add `rel_thresh` and `multiview_agreement` to `Args`/`Returns`.

Add a private pair loop in `metrics.py`, lifted from the deleted `compute_multiview_depth_confidence` collect branch, built on `depth_agreement`:

```python
def _collect_pairs(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray, rel_thresh: float
) -> tuple[dict, list[float | None]]:
    """
    Per-pair depth stats, the residual histogram and per-frame agreement, in one O(N^2) pass.

    - collected: "pairs" (list[PairStats]), "rel_depth_error_counts", "rel_depth_error_edges"
    - agreement: share of seen pixels with at least one agreeing view; None when nothing is seen
    """
    n, h, w = depth.shape
    device = torch.device(get_device())
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    centers = torch.as_tensor(invert_poses(extrinsics)[:, :3, 3], dtype=torch.float32, device=device)

    # One histogram over every ordered pair's residuals: n*(n-1)*h*w at most
    edges = residual_bin_edges(n * (n - 1) * h * w)
    collected = {"pairs": [], "rel_depth_error_edges": edges, "rel_depth_error_counts": np.zeros(len(edges) - 1, np.int64)}
    agreement = []

    for i in tqdm(range(n), desc=f"Cross-view depth check ({n * (n - 1)} pair directions)", unit="frame", leave=False):
        points = unproject(depth_t[i], extrinsics_t[i], intrinsics_t[i]).reshape(-1, 3)
        has_source = depth_t[i].reshape(-1) > 0
        any_agree = torch.zeros_like(has_source)
        any_seen = torch.zeros_like(has_source)

        for j in range(n):
            if i == j:
                continue
            agree, seen, rel = depth_agreement(points, extrinsics_t[j], intrinsics_t[j], depth_t[j], rel_thresh)
            seen &= has_source
            any_agree |= agree & has_source
            any_seen |= seen

            # Residuals over seen pixels with a sampled depth; near-zero depth degenerates
            # - sampled == 0 gives rel == -1 exactly, the has_depth test of the old loop
            z = (points @ extrinsics_t[j, :3, :3].T + extrinsics_t[j, :3, 3])[:, 2]
            sel = seen & (rel != -1) & (z > 1e-6)
            if not bool(sel.any()):
                continue
            rel_sel = rel[sel]

            # Parallax from ray directions: scale-free, needs no focal length
            ray_i = points[sel] - centers[i]
            ray_j = points[sel] - centers[j]
            cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
            parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            residuals = rel_sel.cpu().numpy()
            collected["rel_depth_error_counts"] += np.histogram(bounded_residual(residuals), bins=edges)[0]
            q = torch.quantile(rel_sel, torch.tensor([0.25, 0.5, 0.75], device=device))
            collected["pairs"].append(
                PairStats(
                    idx1=i,
                    idx2=j,
                    n_pixels=int(sel.sum()),
                    median_rel_depth_error=float(q[1]),
                    iqr_rel_depth_error=float(q[2] - q[0]),
                    median_parallax_deg=float(parallax.median()),
                    median_depth=float(z[sel].median()),
                )
            )

        n_seen = int(any_seen.sum())
        agreement.append(int(any_agree.sum()) / n_seen if n_seen else None)

    return collected, agreement
```

`rel == -1` exactly iff `sampled == 0` (`(0 - e)/e`), which reproduces the old `has_depth` selector. If the parity/metrics tests show a float `-1` mismatch, return `sampled` from `depth_agreement` as a 4th element instead — do not add a second projection.

In `compute_reconstruction_quality`: first line `collected, agreement = _collect_pairs(depth, model_intrinsics, extrinsics, rel_thresh)`; add `"multiview_agreement": agreement` to `frames_table`. Imports: `depth_agreement` from projection, `invert_poses` from transforms, `get_device`, `torch`, `tqdm`.

- [ ] **Step 4: Reconstructor report** — replace lines ~1519-1545 with:

```python
        # Load the result; the report runs its own cross-view pass
        ff = PointcloudResult.load_zarr(self.pointcloud_zarr)

        # Optional keyframes for photometric, picked by the zarr's own rows
        # - images/ may hold frames an incremental sfm model dropped; filename order would mispair
        images = None
        if frames.frame_paths(self.images_dir):
            images = _scene_frames(self.images_dir)[_store_rows(self.images_dir, ff.image_paths)].astype(np.float32)

        tables = compute_reconstruction_quality(
            ff.depth,
            ff.model_intrinsics,
            ff.extrinsics,
            ff.original_coords,
            [Path(str(p)).name for p in ff.image_paths],
            ff.confidence,
            images,
            ff.intrinsics,
        )
```

`"params": {"rel_thresh": 0.05}` — or drop the key if nothing reads it (grep `rel_thresh` under `tests/wrapper`, `dashboard`). Delete `_REPORT_REL_THRESH` and the `compute_multiview_depth_confidence` import.

- [ ] **Step 5: Update metrics tests** — every `compute_reconstruction_quality(collected, ...)` call in `tests/geometry/test_metrics*.py` drops `collected` and appends `intrinsics=None`. Tests that built `collected` by hand to test `compute_depth_error` keep calling `compute_depth_error` directly.

- [ ] **Step 6: Run tests**

Run: `cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python -m pytest tests/geometry tests/pointcloud tests/wrapper -q`
Expected: pass (compare failures against `docs/known-test-failures.md`).

- [ ] **Step 7: Parity harness** — the mapanything case sets no mv field; the class default now means `min_views=1, mv_rel_thresh=0.02` (was abs 0.02 + rel 0.02, min_views 1). Run the gate:

`cd $WT && WT=$WT bash /workspace/scratch/pc-release/unify_gate.sh s1 2>&1 | grep -av Warp`

Expected: G′ matches baseline; side suite pass; parity vggtx/omega/loger/depth_align PASS; mapanything FAIL (expected move). **STOP: report the mapanything diff (point count before/after) to the user.** After acceptance, re-record only that case into `parity_baseline_unify` (read `parity.py --help` for the record flag).

- [ ] **Step 8: Net-negative + commit**

```bash
cd $WT && for f in $(git diff --name-only); do black --target-version py311 -l 120 $f; isort $f; done
git diff --shortstat   # deletions > insertions
git commit --only collab_splats/geometry/projection.py collab_splats/geometry/metrics.py \
  collab_splats/pointcloud/base.py collab_splats/pointcloud/feedforward configs/base.yaml \
  collab_splats/wrapper/reconstructor.py evals/scripts/eval_multiview_conf.py evals/scripts/eval_run_backend.py \
  tests/geometry tests/pointcloud tests/wrapper \
  -m "refactor(pointcloud): one multiview_depth_confidence on depth_agreement

Counts (agree, seen) per pixel replace MultiviewConfidence, multiview_mask, the
frustum gate and the PointcloudResult mv fields; the report runs its own pair
loop on the same helper and adds per-frame multiview_agreement.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(`git rm` deleted test files first so `--only` sees them.)

---

## Section 2 — Creator structure

### Task 4: `outlier_mask` rename + `clean_pointcloud(result)`

**Files:**
- Modify: `collab_splats/pointcloud/utils.py`
- Test: `tests/pointcloud/test_pointcloud_utils.py`

- [ ] **Step 1: Failing tests** — in `test_pointcloud_utils.py` rename every `clean_pointcloud(points)` use to `outlier_mask`, then append:

```python
def _result(points: np.ndarray) -> PointcloudResult:
    n = len(points)
    return PointcloudResult(
        points=points.astype(np.float32),
        colors=np.zeros((n, 3), np.uint8),
        pixel_indices=np.zeros((n, 3), np.int32),
        extrinsics=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.array([[[10.0, 0, 5], [0, 10.0, 4], [0, 0, 1]]], np.float32),
        model_intrinsics=None,
        image_paths=["frame_000000"],
        original_coords=np.array([[0, 0, 10, 8, 10, 8]], np.float32),
        model_width=10,
        model_height=8,
    )


def test_clean_pointcloud_caps_without_outlier_removal():
    points = np.random.default_rng(0).normal(size=(100, 3))

    out = clean_pointcloud(_result(points), remove_outliers=False, max_points=10)

    assert len(out.points) == 10 and len(out.colors) == 10 and len(out.pixel_indices) == 10


def test_clean_pointcloud_drops_outliers():
    points = np.random.default_rng(0).normal(size=(200, 3))
    points[0] = 1e3

    out = clean_pointcloud(_result(points), remove_outliers=True, max_points=1000)

    assert len(out.points) < 200 and not (np.abs(out.points) > 100).any()
```

Adjust the `PointcloudResult(...)` kwargs to its real required fields (read `pointcloud/base.py` dataclass first; `tests/pointcloud/test_base.py` has a builder — import that instead of writing `_result` if one exists).

- [ ] **Step 2: Run** → FAIL (`clean_pointcloud` takes points).

- [ ] **Step 3: Implement** in `pointcloud/utils.py`: rename `def clean_pointcloud(points, ...)` → `def outlier_mask(points, ...)` (docstring summary "Statistical outlier keep-mask; all-True when too few points."), then add:

```python
def clean_pointcloud(result: "PointcloudResult", *, remove_outliers: bool, max_points: int) -> "PointcloudResult":
    """
    Optional statistical outlier removal, then a seeded cap, on one result.

    - one keep mask selects every per-point array, so they stay row-aligned

    Args:
        result: the cloud to clean.
        remove_outliers: run outlier_mask first.
        max_points: cap drawn after the outlier removal.

    Returns:
        The selected result.
    """
    keep = outlier_mask(result.points) if remove_outliers else np.ones(len(result.points), dtype=bool)
    keep = subsample_points(keep, max_points)
    return result.select_points(keep)
```

Add `from typing import TYPE_CHECKING` and `if TYPE_CHECKING: from collab_splats.pointcloud.base import PointcloudResult` (base imports utils).

Check `subsample_points(mask, max_points)` is a no-op when `mask.sum() <= max_points` (read it); if it isn't, the ff path's old `if len > max_points` guard must be kept inside `clean_pointcloud`.

- [ ] **Step 4: Update the other callers of the old name** — `rtk proxy grep -rn "clean_pointcloud" --include=*.py collab_splats evals tests`: ff `clean_outputs`, sfm `base.py`, `reconstructor.refine_poses`, LC wrapper. Temporarily rename those uses to `outlier_mask` (they are deleted in Tasks 5, 7, 11).

- [ ] **Step 5: Run** `pytest tests/pointcloud -q` → pass. No commit yet (net-positive alone); commit with Task 5.

### Task 5: Base dataclass `create_pointcloud` + feedforward `_reconstruct` + path `_preprocess`

**Files:**
- Modify: `collab_splats/pointcloud/base.py:404-` (`BasePointcloudCreator`)
- Modify: `collab_splats/pointcloud/feedforward/base.py` (fields ~680-712; `create`, `run`, `setup_inference`, `_decode_source`, `postprocess`, `clean_outputs`, `write_colmap_model`, `_preprocess` abstract; frame-decoding section ~516-640 minus `center_crop_coords`)
- Modify: `feedforward/vggtx.py:80-110`, `vggt_omega.py:~100-135`, `loger.py:221-270`, `mapanything.py` `_preprocess`
- Tests: `tests/pointcloud/test_base.py`, `test_feedforward_preprocess_store.py`, `test_vggtx_preproc.py`, creator tests

- [ ] **Step 1: Failing test** — append to `tests/pointcloud/test_base.py`:

```python
@dataclass
class _FakeCreator(BasePointcloudCreator):
    seen: list = field(default_factory=list)

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        self.seen = paths
        return _result(np.random.default_rng(0).normal(size=(50, 3)))


def test_create_pointcloud_reads_frames_cleans_and_writes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (10, 8)).save(images / f"frame_{i:06d}.png")

    creator = _FakeCreator(max_points=20, clean=False)
    result = creator.create_pointcloud(images, tmp_path / "out", tmp_path / "model")

    assert [p.name for p in creator.seen] == ["frame_000000.png", "frame_000001.png"]
    assert len(result.points) == 20
    assert (tmp_path / "model" / "cameras.bin").exists()


def test_create_pointcloud_postprocess_off_keeps_every_point(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 8)).save(images / "frame_000000.png")

    result = _FakeCreator(max_points=20, postprocess=False).create_pointcloud(images, tmp_path / "out")

    assert len(result.points) == 50


def test_create_pointcloud_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        _FakeCreator().create_pointcloud(tmp_path / "nope", tmp_path / "out")
```

(`_result` is Task 4's builder; import it or the shared one.)

- [ ] **Step 2: Run** → FAIL (`BasePointcloudCreator` has abstract `create`).

- [ ] **Step 3: Base** — replace `BasePointcloudCreator(ABC)` in `pointcloud/base.py` with:

```python
@dataclass
class BasePointcloudCreator(ABC):
    """
    Images directory to a cleaned PointcloudResult and an optional COLMAP model.

    - subclasses implement _reconstruct only
    - postprocess: outlier removal (when clean) then the max_points cap

    Attributes:
        max_points: cap drawn after the outlier removal.
        clean: run statistical outlier removal in the postprocess.
        postprocess: run the postprocess at all; sfm defaults it off.
    """

    max_points: int = 500_000
    clean: bool = True
    postprocess: bool = True

    def create_pointcloud(self, images_dir: Path, out_dir: Path, model_dir: Path | None = None) -> PointcloudResult:
        """
        Reconstruct the frames in images_dir, postprocess, and optionally export COLMAP.

        Args:
            images_dir: directory of keyframe images.
            out_dir: run directory, created if absent.
            model_dir: COLMAP model directory; None skips the export.

        Returns:
            The PointcloudResult.

        Raises:
            FileNotFoundError: images_dir is missing or holds no images.
        """
        paths = frames.frame_paths(images_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        result = self._reconstruct(paths, Path(out_dir))

        # Outlier removal, then the cap
        if self.postprocess:
            result = clean_pointcloud(result, remove_outliers=self.clean, max_points=self.max_points)

        if model_dir is not None:
            write_colmap_reconstruction(result.to_colmap(), Path(model_dir))
        return result

    @abstractmethod
    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        Frames to an unpostprocessed PointcloudResult.
        """
        ...
```

Read `frames.frame_paths` first: if it returns `[]` for a missing/empty dir instead of raising, add `if not paths: raise FileNotFoundError(f"no images in {images_dir}")` after the call. Imports: `frames` from `collab_splats.preproc`, `clean_pointcloud` from `.utils`, `write_colmap_reconstruction` from wherever sfm/ff import it today. Check the circular import: `python -c "import collab_splats.pointcloud"`.

- [ ] **Step 4: Feedforward base** — in `BaseFeedforwardCreator`:
  - delete fields `max_points`, `clean` (inherited; ff default `postprocess=True` inherited)
  - delete `create`, `run`, `_decode_source`, `postprocess`, `clean_outputs`, `write_colmap_model`
  - `load_model`: first line `if self.model is not None: return`
  - `setup_inference(self, paths: list[Path]) -> None`:

```python
    def setup_inference(self, paths: list[Path]) -> None:
        """
        Step 2: model-specific preprocess of the frame files.

        Args:
            paths: frame image files, reconstruction order.
        """
        t0 = time.perf_counter()
        self.image_paths = [Path(p.stem) for p in paths]
        self.views, self.original_coords = self._preprocess(paths)
        logger.info("Preprocessed %d images in %.1fs", len(paths), time.perf_counter() - t0)
```

  - add

```python
    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        Load, preprocess, forward, and unproject; the base postprocess runs after.
        """
        self.load_model()
        self.setup_inference(paths)
        self.run_inference()
        self.outputs = self._postprocess(self.raw_outputs)
        return self.outputs
```

  - abstract `_preprocess(self, paths: list[Path]) -> tuple[Any, np.ndarray]`, Args `paths: frame image files.`
  - class docstring: steps become `load_model -> setup_inference -> run_inference -> _postprocess`; drop the `conf_threshold` raw-semantics note only in Task 16
  - delete the in-memory frame decoding section except `center_crop_coords` (move it to the bottom helper section): `frames_as_pil_source`, `_STORE_FRAME_NAME`, `_source_frame_idxs`, `_decode_dir_to_frames`; drop now-unused imports (`re`, `PIL`, `contextmanager` only if `capture_qk` doesn't use it — it does, so keep `contextmanager`; drop `subsample_points`, `clean_pointcloud`/`outlier_mask`, `write_colmap_reconstruction`, `fr` if unused)

- [ ] **Step 5: Backend `_preprocess(paths)`** — each backend reads files directly:
  - vggtx / vggt_omega: `sizes = [Image.open(p).size for p in paths]` (width, height) feeds the existing crop-box math; the loader call becomes `load_and_preprocess_images([str(p) for p in paths], ...)` with the same mode/size args it uses today. Delete the `frames_as_pil_source` context.
  - mapanything: `load_images([str(p) for p in paths], ...)` with today's args; crop boxes from `Image.open(p).size`.
  - loger: `frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])` after the uniform-size check `if len({Image.open(p).size for p in paths}) > 1: raise ValueError(...)` (keep today's message); delete the ascending `frame_idxs` check and its comment.

  Read each current `_preprocess` fully before editing; the crop/resize math must stay byte-identical.

- [ ] **Step 6: Tests** — `test_feedforward_preprocess_store.py` and `test_vggtx_preproc.py` call `_preprocess(frames, frame_idxs)`: rewrite to write PNGs into `tmp_path` and pass paths. Creator tests that call `creator.run(...)`/`creator.create(...)`/`creator.postprocess()` switch to `create_pointcloud(...)` or `_postprocess(raw)`. Delete tests of deleted helpers (`frames_as_pil_source`, `_source_frame_idxs`, `_decode_dir_to_frames`).

- [ ] **Step 7: Run** `pytest tests/pointcloud -q` → pass except LC/MapAnything/sfm (Tasks 6-7).

### Task 6: MapAnything `_stack_predictions` + LC wrapper

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py` (`_forward`, `_lc_collate_outputs`, `_postprocess`, `extract_intermediate_features` untouched)
- Modify: `collab_splats/pointcloud/feedforward/base.py` `_postprocess` (optional `mask` key, `depth.ndim` squeeze)
- Modify: `collab_splats/geometry/loop_closure/wrapper.py:98-107,188-235,306-360`
- Tests: `tests/pointcloud/feedforward/test_mapanything_creator.py`, `tests/pointcloud/test_lc_collate_window.py`, `tests/geometry/loop_closure/test_wrapper.py`

- [ ] **Step 1: Failing test** — in `test_mapanything_creator.py` add (reuse the file's per-view prediction stub):

```python
def test_stack_predictions_gives_base_raw_dict(ma_preds_and_views):
    preds, views = ma_preds_and_views
    creator = MapAnythingCreator()

    raw = creator._stack_predictions(preds, views, masked=True)

    n = len(preds)
    assert raw["extrinsic"].shape == (n, 3, 4)
    assert raw["intrinsics"].shape == (n, 3, 3)
    assert raw["depth"].ndim == 3 and raw["depth_conf"].shape == raw["depth"].shape
    assert raw["images"].shape[:2] == (n, 3)
    assert raw["mask"].dtype == bool


def test_stack_predictions_unmasked_has_no_mask(ma_preds_and_views):
    preds, views = ma_preds_and_views

    raw = MapAnythingCreator()._stack_predictions(preds, views, masked=False)

    assert "mask" not in raw
```

If the file has no fixture, build one from the parity harness's mapanything case (`/workspace/scratch/pc-release/parity.py:263-320`), copied into `tests/pointcloud/feedforward/conftest.py` as `ma_preds_and_views`.

- [ ] **Step 2: Run** → FAIL (`_stack_predictions` missing).

- [ ] **Step 3: Implement `_stack_predictions`** in `mapanything.py`, replacing `_lc_collate_outputs` and the body of `_postprocess`:

```python
    def _stack_predictions(self, preds: list[dict], views: list[dict], *, masked: bool) -> dict[str, np.ndarray]:
        """
        Upstream postprocess, then per-view predictions stacked into the base raw dict.

        - masked: upstream's edge + confidence-percentile mask lands under "mask"
        - bf16 pts3d/pts3d_cam cast to float32: upstream grid_sample needs matching dtypes
        """
        for pred in preds:
            pred["pts3d_cam"] = pred["pts3d_cam"].float()
            pred["pts3d"] = pred["pts3d"].float()

        if masked:
            processed = postprocess_model_outputs_for_inference(
                preds,
                views,
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=True,
                confidence_percentile=self.confidence_percentile,
            )
        else:
            processed = postprocess_model_outputs_for_inference(preds, views, apply_mask=False)

        # Upstream ranks: depth_z (1, H, W, 1), img_no_norm (1, H, W, 3), conf (1, H, W)
        raw = {
            "extrinsic": np.stack([invert_poses(p["camera_poses"][0].cpu().float().numpy())[:3] for p in processed]),
            "intrinsics": np.stack([p["intrinsics"][0].cpu().float().numpy() for p in processed]),
            "depth": np.stack([p["depth_z"][0, ..., 0].cpu().float().numpy() for p in processed]),
            "depth_conf": np.stack([p["conf"][0].cpu().float().numpy() for p in processed]),
            "images": np.stack([p["img_no_norm"][0].cpu().float().numpy().transpose(2, 0, 1) for p in processed]),
        }
        if masked:
            raw["mask"] = np.stack([p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in processed])
        return raw
```

`_forward` ends with `return self._stack_predictions(preds, forward_views, masked=views is None)` — read `_forward`'s full-vs-window branch first; the full sequence is masked, the LC window is not. Delete `_lc_window_views` and `_lc_collate_outputs`. Delete MapAnything's `_postprocess` override entirely.

- [ ] **Step 4: Base `_postprocess`** — depth line becomes:

```python
        depth = raw_outputs["depth"]
        if depth.ndim == 4:
            depth = depth.squeeze(-1)
```

and the confidence mask becomes (still `>=` percentile until Task 16):

```python
        # Backend mask (MapAnything upstream) or the depth-confidence percentile
        valid = depth > 0
        if "mask" in raw_outputs:
            valid &= raw_outputs["mask"]
        else:
            valid &= raw_outputs["depth_conf"] >= np.percentile(raw_outputs["depth_conf"], self.conf_threshold)
        valid &= self._multiview_mask(depth, intrinsic, extrinsic_4x4)
```

then pass `valid` where `unproject_and_filter_points` used its own mask — inline the gather here now (Task 16 needs it anyway):

```python
        # Kept pixels as rows of the dense grid, in row-major order
        colors_grid = (np.asarray(raw_outputs["images"], dtype=np.float32).transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        pixel_indices = np.stack(np.where(valid), axis=1).astype(np.int32)
```

and `points=world_points[valid].astype(np.float32), colors=colors_grid[valid]`. Keep `conf_threshold <= 1.0` raw semantics out of scope here: `if self.conf_threshold <= 1.0` → raw compare, else percentile, exactly as `unproject_and_filter_points` does today, so vggtx/omega/loger parity stays bit-exact. Delete `unproject_and_filter_points` and `_mask_to_points`. Check `images` is a tensor for vggt (`.cpu()` path) — keep a `torch.as_tensor(...).float().numpy()` if so.

- [ ] **Step 5: MapAnything world points** — the base `_postprocess` unprojects depth through the stacked K and poses; MapAnything used upstream `pts3d`. Run the parity gate now (`unify_gate.sh s2ma`). If mapanything moves: **report the diff to the user**; if they want no move, add `"world_points": np.stack([p["pts3d"][0].cpu().float().numpy() for p in processed])` to the stacked dict and use `raw_outputs.get("world_points")` over the unprojection in the base. Otherwise leave it.

- [ ] **Step 6: LC wrapper** — in `loop_closure/wrapper.py`:
  - delete `reconstruct` (188-205), `run` (207-223), `postprocess` (225-235)
  - add, near the top of the class body:

```python
    create_pointcloud = BasePointcloudCreator.create_pointcloud

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        The base creator's steps with the windowed LC forward; assembled result when LC closed.
        """
        self.base.load_model()
        self.base.setup_inference(paths)
        self.run_inference()
        if self._lc_assembled:
            return self.base.outputs
        return self.base._postprocess(self.base.raw_outputs)
```

  Read the current `run`/`reconstruct` to confirm the attribute names (`_lc_assembled`, `run_inference`) and that `create_pointcloud` reads `self.postprocess`/`self.clean`/`self.max_points` through `__getattr__` onto `base` — it does, since the wrapper does not define them. `self.postprocess` must resolve to the dataclass bool: grep the wrapper for any `postprocess` attribute after the deletion.
  - line 307: `raw_lc = raw` (drop the collate branch; delete the comment above it)
  - colors (344-360): replace the `"colors" in raw_lc` branch with

```python
            # Colors from the backend's own [0, 1] images when it returns them (MapAnything is normalized otherwise)
            if "images" in raw_lc:
                dense_colors = (np.asarray(raw_lc["images"]).transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            else:
```

  keeping the frames-heuristic `else` body. Check VGGT's raw also has `"images"` (it does for `_postprocess`); if so the heuristic `else` is dead — delete it and use the `images` path for all backends only if the LC tests pass unchanged.
  - quickstart docstring (98-107): `LoopClosure(base, config).create_pointcloud(images_dir, out_dir)`.
  - `_assemble_result` cap stays.

- [ ] **Step 7: Tests** — `test_lc_collate_window.py` tests `_lc_collate_outputs`: port its window-view assertion to `_stack_predictions(..., masked=False)` or delete it if Task 6 Step 1 covers it. `tests/geometry/loop_closure/test_wrapper.py`: `.run(` / `.reconstruct(` → `.create_pointcloud(`.

- [ ] **Step 8: Run** `pytest tests/pointcloud tests/geometry -q` → pass.

### Task 7: sfm opt-in postprocess + export override

**Files:**
- Modify: `collab_splats/pointcloud/sfm/base.py` (whole create flow; SOR 121-131)
- Test: `tests/pointcloud/sfm/test_base.py`

- [ ] **Step 1: Failing test** — append to `tests/pointcloud/sfm/test_base.py` (reuse its stub creator that returns a small `pycolmap.Reconstruction`; read the file first):

```python
def test_postprocess_off_writes_tracks(stub_sfm_creator, images_dir, tmp_path):
    stub_sfm_creator.create_pointcloud(images_dir, tmp_path / "out", tmp_path / "model")

    recon = pycolmap.Reconstruction(tmp_path / "model")
    assert any(len(p.track.elements) > 0 for p in recon.points3D.values())


def test_postprocess_on_writes_to_colmap(stub_sfm_creator, images_dir, tmp_path):
    stub_sfm_creator.postprocess = True
    stub_sfm_creator.create_pointcloud(images_dir, tmp_path / "out", tmp_path / "model")

    recon = pycolmap.Reconstruction(tmp_path / "model")
    assert all(len(p.track.elements) == 0 for p in recon.points3D.values())
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — `BaseSfmCreator` (read its current name/fields first):
  - `postprocess: bool = False` field override; delete its own `clean`/`max_points` if duplicated
  - rename `create` → `_reconstruct(self, paths, out_dir)`; it keeps `self._recon = recon` and returns `align_depth(...)` as today, minus the SOR block (121-131) and minus the `write_colmap_reconstruction(recon, model_dir)` line
  - override:

```python
    def create_pointcloud(self, images_dir: Path, out_dir: Path, model_dir: Path | None = None) -> PointcloudResult:
        """
        The base flow; with the postprocess off, the model is written from the sfm recon, tracks kept.

        Args:
            images_dir: directory of keyframe images.
            out_dir: run directory, created if absent.
            model_dir: COLMAP model directory; None skips the export.

        Returns:
            The PointcloudResult.
        """
        if self.postprocess:
            return super().create_pointcloud(images_dir, out_dir, model_dir)

        result = super().create_pointcloud(images_dir, out_dir)
        if model_dir is not None:
            write_colmap_reconstruction(self._recon, Path(model_dir))
        return result
```

  Read how `create` holds `recon` today; if `align_depth` needs `recon` before the write, keep that order.
  - the frame-listing inside today's `create` (`frame_paths`) is gone: `_reconstruct` gets `paths`. Stems: sfm already uses stems for image names — confirm.

- [ ] **Step 4: Run** `pytest tests/pointcloud/sfm -q` → pass.

### Task 8: Callers + parity harness

**Files:**
- Modify: `collab_splats/wrapper/reconstructor.py` (`_run_feedforward` ~489-516, `_run_sfm` ~1122-1147)
- Modify: `evals/scripts/eval.py:318,355-372`, `evals/scripts/ba_start_at_gt.py:98-101`, `evals/scripts/eval_run_backend.py:73-74`, `collab_splats/dashboard/pipeline.py:376-395`
- Modify: `/workspace/scratch/pc-release/parity.py:379-384` and its mapanything case
- Tests: `tests/wrapper/test_reconstructor*.py`, `tests/test_feedforward_logging.py`, `tests/integration/test_pipeline_cu121.py`

- [ ] **Step 1: Reconstructor** — `_run_feedforward`: one call for both LC and non-LC:

```python
    result = creator.create_pointcloud(images_dir, output_dir, model_dir)
```

(where `creator` is the LoopClosure wrapper when LC is on). `_run_sfm`: pass `clean=pc_cfg["clean"]["enabled"], max_points=pc_cfg["max_points"]` into the sfm creator and call `create_pointcloud(images_dir, output_dir, model_dir)`.

- [ ] **Step 2: evals** — `eval.py` BA path:

```python
        result = creator.create_pointcloud(image_dir, output_dir)
        result = <existing BA call on result>
        write_colmap_reconstruction(result.to_colmap(), model_dir)
```

non-BA path: `creator.create_pointcloud(image_dir, output_dir, model_dir)`. Read 318 and 355-372 first. `ba_start_at_gt.py:98-101` and `eval_run_backend.py:73-74`: `create_pointcloud(staged, args.out)`.

- [ ] **Step 3: dashboard** — `pipeline.py:376-395`: keep the explicit `creator.load_model()` for the progress step, then `result = creator.create_pointcloud(images_dir, out_dir)`; `load_model` is idempotent (Task 5).

- [ ] **Step 4: Name-keyed GT check** — `git -C $WT show clean/evals-release:evals/datasets.py | rtk proxy grep -n "image_paths\|frame_\|stem\|name"` and the same for `evals/scripts/eval.py` / `gt_metrics.py`. If any GT join keys on `frame_{idx:06d}` synthetic names, STOP and report (7-Scenes image_paths become real stems like `frame-000000.color`).

- [ ] **Step 5: Parity harness** — in `parity.py` `_run`:

```python
def _run(creator, seed: int) -> dict:
    """Seed both RNGs, run _postprocess then the base clean, encode the result."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    result = creator._postprocess(creator.raw_outputs)
    result = clean_pointcloud(result, remove_outliers=creator.clean, max_points=creator.max_points)
    return _fields(result)
```

(match its existing seeding lines.) The mapanything case: `creator.raw_outputs = creator._stack_predictions(raw, views, masked=True)`; drop `_processed_views`. Update the module docstring line 8. This file is outside the repo: no commit, note the change in the Task 8 commit body.

- [ ] **Step 6: Tests** — update remaining `create(`/`run(`/`postprocess()` callers found by:

`cd $WT && rtk proxy grep -rn "\.create(\|\.run(\|\.postprocess()\|write_colmap_model\|clean_outputs\|_lc_collate_outputs\|frames_as_pil_source" --include=*.py collab_splats evals tests`
Expected after edits: no hits except unrelated `.run(` (subprocess, tmux).

- [ ] **Step 7: Gate**

`cd $WT && WT=$WT bash /workspace/scratch/pc-release/unify_gate.sh s2 2>&1 | grep -av Warp`
Expected: vggtx/omega/loger/depth_align PASS; mapanything PASS (or the Task 6 Step 5 move, already reported). G′: any sfm test that asserted SOR-cleaned counts moves → **report to the user** with before/after counts; update the test only after acceptance.

- [ ] **Step 8: Commit Section 2** — format touched files, check `--shortstat` net-negative, then

```bash
git commit --only collab_splats evals tests \
  -m "refactor(pointcloud): shared create_pointcloud template; sfm postprocess opt-in

BasePointcloudCreator owns frame listing, clean + cap, and the COLMAP export;
feedforward, sfm and LoopClosure implement _reconstruct. _preprocess reads frame
files directly; MapAnything stacks into the base raw dict. sfm SOR deleted,
postprocess default off. clean_pointcloud(points) renamed outlier_mask.
Parity harness _run moved to _postprocess + clean_pointcloud.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(list exact paths from `git status --short`, not whole directories, if unrelated dirty files exist there.)

---

## Section 3 — Unify duplicates

### Task 9: `depth_residual` returns pixels; lifting drops `project`

**Files:**
- Modify: `collab_splats/geometry/projection.py` (`depth_residual`, `depth_agreement`)
- Modify: `collab_splats/semantics/lifting.py:167-174`
- Test: `tests/geometry/test_projection.py`, `tests/semantics/test_lifting*.py`

- [ ] **Step 1: Failing test** — in `test_projection.py`:

```python
def test_depth_residual_returns_project_pixels():
    points = torch.tensor([[0.1, -0.2, 2.0], [0.0, 0.0, 3.0]])
    world_to_cam = torch.eye(4)
    intrinsics = torch.tensor([[10.0, 0, 4.5], [0, 10.0, 3.5], [0, 0, 1]])

    *_, pixels = depth_residual(points, world_to_cam, intrinsics, torch.ones(8, 10))
    expected, _ = project(points, world_to_cam, intrinsics)

    assert torch.equal(pixels, expected)
```

- [ ] **Step 2: Run** → FAIL (4-tuple).
- [ ] **Step 3: Implement** — `depth_residual` returns `residual, expected, sampled, in_front & in_bounds, pixels`; docstring `Returns:` gains `- pixels: (P, 2) project's pixels`; the bullet "pixels are project's, so a caller that projects the same points reads the same pixel" becomes "- pixels returned, so callers need no second project". `depth_agreement` unpacks `residual, expected, sampled, valid, _`. Update every other caller: `rtk proxy grep -rn "depth_residual(" --include=*.py collab_splats tests evals`.
- [ ] **Step 4: Lifting** — read `lifting.py:160-180`; delete the `project(...)` call and use the 5th return of `depth_residual`. Drop `project` from its imports if unused.
- [ ] **Step 5: Run** `pytest tests/geometry tests/semantics -q` → pass.

### Task 10: `refine_poses` uses `clean_pointcloud`

**Files:** Modify `collab_splats/wrapper/reconstructor.py:1149-1215`. Test: `tests/wrapper/test_reconstructor*.py` (refine tests).

- [ ] **Step 1:** Read 1149-1215. Replace the inline SOR + cap (~1190-1201) with:

```python
        # Outlier removal, then the cap, on the reprojected cloud
        refined = clean_pointcloud(refined, remove_outliers=pc_cfg["clean"]["enabled"], max_points=pc_cfg["max_points"])
```

and the export with `write_colmap_reconstruction(refined.to_colmap(), model_dir)` if it isn't already. Import `clean_pointcloud` from `collab_splats.pointcloud.utils`; drop `outlier_mask`/`subsample_points` imports if now unused.

- [ ] **Step 2:** `pytest tests/wrapper -q -k refine` → pass (identical counts: same mask then same seeded draw). If a count moves, the old code drew the cap before the SOR — **report**, don't adjust.

### Task 11: metrics photometric on images-grid K

**Files:** Modify `collab_splats/geometry/metrics.py:224-244` (`compute_photometric_ncc`) and `compute_reconstruction_quality`. Test: `tests/geometry/test_metrics*.py`.

- [ ] **Step 1: Failing test** — change the existing photometric test to pass full-res `intrinsics` (a K on the images grid) and no `original_coords`:

```python
    pairs = compute_photometric_ncc(images, depth, intrinsics_fullres, extrinsics)
```

where `intrinsics_fullres` is what `PointcloudResult.__post_init__` derives (build the test `PointcloudResult` and read `.intrinsics`, or compute it with `rescale_intrinsics` then `shift_intrinsics`, one call per line). Run → FAIL (unexpected/missing arg).

- [ ] **Step 2: Implement** — `compute_photometric_ncc(images, depth, intrinsics, extrinsics)`: `intrinsics` Args entry "K on the images' pixel grid"; delete the lift block (236-243) and the `rescale_intrinsics, shift_intrinsics` import. Depth is still model-res: read the function to confirm how it upsamples depth (`upsample_depths` uses `original_coords`?). If `upsample_depths` needs `original_coords`, the parameter stays and only the K lift goes.
- [ ] **Step 3:** In `compute_reconstruction_quality` pass `intrinsics[:m]` (the new full-res arg). `original_coords` stays (covered_fraction). The reconstructor already passes `ff.intrinsics` (Task 3).
- [ ] **Step 4:** `pytest tests/geometry -q` → pass; photometric values identical to before (the existing test's expected numbers must not change).

### Task 12: depth.py pixel rule + fold the scale fit

**Files:** Modify `collab_splats/pointcloud/depth.py` (`_pixel_indices_from_reconstruction`, `_depth_correspondences`, `_fit_depth_scales`, `align_depth`). Test: `tests/pointcloud/test_depth.py`.

- [ ] **Step 1: Failing test** — append:

```python
def test_pixel_indices_floor_negative_subpixel():
    # int() truncates -0.5 to 0; floor gives -1, which the clip then puts at 0
    xy = np.array([[-0.5, 2.7], [3.9, -0.1]])
    rows, cols = _pixel_indices(xy, height=4, width=5)
    assert cols.tolist() == [0, 3] and rows.tolist() == [2, 0]
```

Read `_pixel_indices_from_reconstruction` first; if it can't be called on raw xy, test through its public caller with a pycolmap stub from `test_depth.py`'s existing fixtures instead, asserting the same floor+clip result.

- [ ] **Step 2: Implement** — both paths use `np.floor(xy).astype(int)` then `np.clip(..., 0, W-1 / H-1)`. Then fold `_depth_correspondences` + `_fit_depth_scales` into `align_depth` as inline blocks (~20 lines): per-frame `median(sparse / dense)` over frames with `>= min_obs` correspondences; global median fallback for the rest; the `global_scale`, `fallback_frames` stats and the before/after ratio p10/50/90 log kept; `ValueError` when no frame fits. Replace `full_frame_coords(orig_w, orig_h, n)` with:

```python
    original_coords = np.tile(np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32), (n, 1))
```

Delete the two helpers and their dedicated tests; keep behavior tests on `align_depth`.

- [ ] **Step 3: Run** `pytest tests/pointcloud/test_depth.py -q` → pass.
- [ ] **Step 4: Gate + commit Section 3**

`cd $WT && WT=$WT bash /workspace/scratch/pc-release/unify_gate.sh s3 2>&1 | grep -av Warp`
Expected: all parity PASS; depth_align may move from the floor rule → **report** if so. Then format, `--shortstat`, and

```bash
git commit --only <paths> -m "refactor(pointcloud): fold duplicate projections, refine clean and depth fit

depth_residual returns project's pixels, so lifting projects once; refine_poses
uses clean_pointcloud; photometric takes the images-grid K instead of re-lifting;
depth.py pixel mapping is floor then clip and the scale fit lives in align_depth.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Section 4 — VDA loader

### Task 13: VDA packaging try, license note

**Files:** Modify `collab_splats/pointcloud/depth.py` (`VDA_ROOT`, `_load_vda_model`), `pyproject.toml` (only if the dep installs), `docs/pointcloud.md` (or the existing user doc for `pointcloud/` — `ls docs/*.md`).

- [ ] **Step 1: Try the git dep in a scratch venv** — never the shared venv:

```bash
cd /tmp/claude-0/-workspace-collab-splats/a8f9c053-6a5b-4f92-be46-423d373f50fa/scratchpad && \
uv venv vda-try --python 3.11 && \
VIRTUAL_ENV=vda-try uv pip install --no-deps "git+https://github.com/DepthAnything/Video-Depth-Anything@4f5ae23" 2>&1 | tail -5
```

Expected (likely): fails — no `setup.py`/`pyproject.toml`, and `video_depth.py:27` imports top-level `utils`. If it **installs and** `python -c "from video_depth_anything.video_depth import VideoDepthAnything"` works: report to the user before touching `pyproject.toml` (the user owns the uv lock). If it fails: record the error line in the commit body and keep the pinned clone.

- [ ] **Step 2: Trim** `_load_vda_model` comments to the header+bullets style; confirm the checkpoint path is `hf_hub_download` (already). Add to the `depth.py` module docstring:

```
- estimate_depth uses Metric-Video-Depth-Anything-Large, licensed cc-by-nc-4.0 (non-commercial)
```

and one line in the user doc's depth section saying the same, naming Metric-Small (Apache-2.0) as the commercial-safe alternative not taken.

- [ ] **Step 3: Gate** `unify_gate.sh s4` → unchanged. Commit `docs(pointcloud): VDA license note; loader comments` (net-negative from the comment trim; if not, fold into the Section 5 commit).

---

## Section 5 — Small items and style

### Task 14: Inlines

**Files:** `collab_splats/pointcloud/utils.py` (`mean_top_quarter`), `feedforward/base.py` (`_verify_loop_candidate` ~1022, `_raw_to_world_points`, `full_frame_coords`), `feedforward/loger.py:30,267`, `evals/scripts/eval_similarity_calibration.py:49,228`, `docs/parity.md:40,205`. Tests: `tests/pointcloud/feedforward/test_full_frame_coords.py` (delete), `tests/pointcloud/test_pointcloud_utils.py`.

- [ ] **Step 1:** `_verify_loop_candidate`:

```python
        # Gate on the mean of the top quarter of the per-token cross-frame attention ratio
        ratio = cross_frame_attention_ratio(features["k"], features["q"], token_offset=self._lc_token_offset)
        score = float(ratio[ratio >= np.percentile(ratio, 75)].mean())
```

(read `mean_top_quarter` first: if it converts a tensor to numpy, add that conversion as its own line.) `eval_similarity_calibration.py:228` the same two lines. `docs/parity.md:40,205`: name the expression instead of the function. Delete `mean_top_quarter` and its tests; fix `utils.py:132` docstring mention.
- [ ] **Step 2:** `loger.py:267`: `original_coords = np.tile(np.array([0, 0, orig_w, orig_h, orig_w, orig_h], dtype=np.float32), (len(frames), 1))`; delete `full_frame_coords` and `test_full_frame_coords.py`.
- [ ] **Step 3:** `_raw_to_world_points`: the only live caller is `_postprocess` with `subsample=1` (the LC `subsample=8` path is dead — grep to confirm). Inline in `_postprocess`:

```python
        # Dense world-point grid at model resolution, unprojected once
        depth_t = torch.as_tensor(depth, dtype=torch.float32)
        world_to_cam = torch.as_tensor(extrinsic, dtype=torch.float32)
        intrinsics_t = torch.as_tensor(intrinsic, dtype=torch.float32)
        world_points = unproject(depth_t, world_to_cam, intrinsics_t).numpy()
```

Delete `_raw_to_world_points`.
- [ ] **Step 4:** `pytest tests/pointcloud tests/geometry -q` → pass.

### Task 15: `_verify_geometry` deleted; real file names; exports

**Files:** `feedforward/base.py:114-135`, `vggtx.py:21,167`, `vggt_omega.py:25,185`, `wrapper/reconstructor.py:799`, `feedforward/__init__.py`. Tests: `tests/pointcloud/feedforward/test_verify_lc_data.py`.

- [ ] **Step 1:** In both `extract_intermediate_features`, replace `captured.update(_verify_geometry(raw))` with:

```python
        # Pair poses and geometry from the SAME forward
        raw = _decode_depth_head(predictions, frames.shape[-2:], pose_encoding_to_extri_intri)
        depth = torch.from_numpy(raw["depth"][..., 0])
        world_to_cam = torch.from_numpy(raw["extrinsic"])
        intrinsics = torch.from_numpy(raw["intrinsics"])
        captured["world_points"] = unproject(depth, world_to_cam, intrinsics).numpy()
        captured["poses"] = extrinsics_to_homogeneous(raw["extrinsic"])
        captured["conf"] = raw["depth_conf"]
```

Delete `_verify_geometry`. `test_verify_lc_data.py` keeps its output-shape assertions against `extract_intermediate_features` (stub model) or is deleted if it only tested the helper.

Net-negative check: two call sites × 6 lines vs a 22-line helper. If the diff is net-positive, keep `_verify_geometry` and report.
- [ ] **Step 2:** `reconstructor.py:799`: `ids = [p.name for p in paths]` (read 790-805; `paths` from `frames.frame_paths(self.images_dir)`).
- [ ] **Step 3:** `feedforward/__init__.py`: `__all__` lists only names that still exist (`python -c "import collab_splats.pointcloud.feedforward as f; [getattr(f, n) for n in f.__all__]"`).
- [ ] **Step 4:** Grep clean:

`cd $WT && rtk proxy grep -rn "mean_top_quarter\|full_frame_coords\|_mask_to_points\|_raw_to_world_points\|unproject_and_filter_points\|_verify_geometry" --include=*.py --include=*.md collab_splats evals tests docs/parity.md`
Expected: no hits (plans/specs under `docs/superpowers/` may keep history).

### Task 16: `confidence_mask` replaces the percentile compare

**Files:** `feedforward/base.py` `_postprocess` mask block and class docstring; `configs/base.yaml:43-46` (loger `conf_threshold` comment).

- [ ] **Step 1:** Replace the Task 6 percentile/raw branch with:

```python
        # Backend mask (MapAnything upstream) or the depth-confidence percentile
        valid = depth > 0
        if "mask" in raw_outputs:
            valid &= raw_outputs["mask"]
        else:
            valid &= confidence_mask(raw_outputs["depth_conf"], self.conf_threshold)
        valid &= self._multiview_mask(depth, intrinsic, extrinsic_4x4)
```

Class docstring: `conf_threshold: depth-confidence percentile (0-100); pixels strictly above it are kept.` `base.yaml` loger comment: drop the raw-value paragraph, keep `conf_threshold: 50.0  # depth-confidence percentile`.
- [ ] **Step 2:** `pytest tests/pointcloud -q`; tests asserting raw `<= 1.0` semantics are deleted.
- [ ] **Step 3: Gate** — `unify_gate.sh s5` — vggtx/omega/loger expected to move (ties at the percentile, `depth > 0` now required for the VGGT family). **STOP and report** point-count diffs per case. Re-record after acceptance.
- [ ] **Step 4: Commit Section 5.1-5.6** — format, `--shortstat`, commit `refactor(pointcloud): inline one-caller helpers; confidence_mask in _postprocess` with trailer.

### Task 17: Style pass (5.7) — behavior-free

**Files:** every `.py` under `collab_splats/pointcloud/`.

- [ ] **Step 1: Snapshot AST** before editing:

```bash
cd $WT && PYTHONPATH=$WT /opt/venv/reconstruction/bin/python - <<'EOF' > /tmp/claude-0/-workspace-collab-splats/a8f9c053-6a5b-4f92-be46-423d373f50fa/scratchpad/ast_before.txt
import ast, pathlib
for p in sorted(pathlib.Path("collab_splats/pointcloud").rglob("*.py")):
    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    print(p, ast.dump(tree))
EOF
```

- [ ] **Step 2: Edit** per the spec 5.7 list: one-line summaries (≤100 chars), fragment bullets, no provenance essays, one upstream cite per file at top, classes top / helpers bottom (moving a def reorders the module body — the AST dump compares a sorted list of top-level defs, so change the script's print to `sorted(ast.dump(n) for n in tree.body)` if any def moves), blank lines around blocks, US spelling.
- [ ] **Step 3: Prove** — rerun Step 1's script to `ast_after.txt`; `diff ast_before.txt ast_after.txt` → empty. Sanity mutation: change one `+` to `-` in a scratch copy, rerun, see a diff, revert via `cp`.
- [ ] **Step 4:** `pytest tests/test_docstring_contract.py -q` → pass. Gate `unify_gate.sh s6` → unchanged.
- [ ] **Step 5: Commit** `style(pointcloud): geometry-style docstrings and comments` with trailer.

---

## After the plan

Handoff items (not in this plan): instantsfm smoke re-run, CHANGELOG + CLAUDE.md row, final whole-branch review.
