# BA Track-Quality Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring BA track quality and observation filtering to upstream `demo_colmap.py` parity so BA stops hurting sub-cm baselines: visibility gate, fine tracking, upstream track density, shared camera, upstream filter order.

**Architecture:** All code changes land in `collab_splats/geometry/bundle_adjustment.py` — new `BundleAdjustmentConfig` fields (defaults are the only config surface; pipeline and eval both construct the dataclass bare), a new pure helper `_filter_observations` extracted from `_optimize` for testability, and one new eval condition `ba_percam` for the shared-camera ablation. No YAML changes.

**Tech Stack:** numpy, pypose/bae (mocked in tests), VGGSfM `predict_tracks`, pytest. Python: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-19-ba-track-quality-parity-design.md`

---

### Task 1: Config fields + new defaults

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:61-76` (`BundleAdjustmentConfig`)
- Test: `tests/geometry/test_bundle_adjustment.py` (extend, near `test_ba_config_new_fields_default` at line 644)

- [ ] **Step 1: Write the failing test**

Add after `test_ba_config_new_fields_default` (follows its direct-import style):

```python
def test_ba_config_track_quality_defaults():
    """Track-quality parity defaults: vis gate, fine tracking, shared camera, upstream density."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig
    cfg = BundleAdjustmentConfig()
    assert cfg.vis_thresh == 0.2
    assert cfg.fine_tracking is True
    assert cfg.shared_camera is True
    assert cfg.max_query_pts == 4096
    assert cfg.query_frame_num == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py::test_ba_config_track_quality_defaults -v`
Expected: FAIL — `TypeError` on unknown attr / `AttributeError: vis_thresh`

- [ ] **Step 3: Update the dataclass**

In `BundleAdjustmentConfig`, change:

```python
    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = True  # one physical camera per scene; per-frame K spread is model noise
    vis_thresh: float = 0.2  # min VGGSfM visibility score for an observation to enter BA
    fine_tracking: bool = True  # VGGSfM fine refinement stage (upstream always on; coarse-only ~1-2px error)
    min_inliers_per_frame: int = 64
    max_query_pts: int = 4096  # track extraction: max query points (upstream demo default)
    query_frame_num: int = 8  # track extraction: number of query frames (upstream demo default)
```

(Only `shared_camera` moves position/value; `vis_thresh`/`fine_tracking` are new; other fields keep their lines.)

- [ ] **Step 4: Run test + neighbours to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "config" -v`
Expected: PASS (including existing `test_ba_config_new_fields_default`, `test_bundle_adjustment_default_config` — neither asserts the changed fields)

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "feat(ba): track-quality parity defaults — vis_thresh, fine_tracking, shared camera, 4096/8"
```

---

### Task 2: Thread `fine_tracking` into extraction + cache key

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:46-53` (`_compute_tracks_cache_key`), `:100-114` (`_load_or_extract_tracks` `_extract` closure)
- Test: `tests/geometry/test_bundle_adjustment.py` (extend cache tests near line 700; update `fake_extract` in `test_tracks_cache_save_load`)

- [ ] **Step 1: Write the failing tests**

Add after `test_tracks_cache_invalidates_on_config_change` (mirrors its body):

```python
def test_tracks_cache_invalidates_on_fine_tracking_change(tmp_path):
    """fine_tracking is part of the cache key — flipping it must re-extract."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

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

    with patch("collab_splats.geometry.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=True, tracks_cache_dir=tmp_path))
        ba1._load_or_extract_tracks(result)

        # Flip fine_tracking — different key → cache miss, re-extract
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=False, tracks_cache_dir=tmp_path))
        t2, _, _ = ba2._load_or_extract_tracks(result)

    np.testing.assert_array_equal(t2, fake_b[0])


def test_tracks_cache_hit_on_vis_thresh_change(tmp_path):
    """vis_thresh is applied post-extraction — changing it must reuse the cache."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)

    fake_tracks = np.ones((N, 5, 2), dtype=np.float32)
    fake_vis = np.ones((N, 5), dtype=np.float32) * 0.9
    fake_pts3d = np.ones((5, 3), dtype=np.float32) * 2.0

    extract_calls = []

    def fake_extract(*args, **kwargs):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d

    with patch("collab_splats.geometry.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba1 = BundleAdjustment(config=BundleAdjustmentConfig(vis_thresh=0.2, tracks_cache_dir=tmp_path))
        ba1._load_or_extract_tracks(result)
        ba2 = BundleAdjustment(config=BundleAdjustmentConfig(vis_thresh=0.5, tracks_cache_dir=tmp_path))
        ba2._load_or_extract_tracks(result)

    assert len(extract_calls) == 1, "vis_thresh change must NOT invalidate the track cache"


def test_extract_receives_fine_tracking_kwarg(tmp_path):
    """_load_or_extract_tracks passes cfg.fine_tracking through to _extract_tracks_vggsfm."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, H, W = 3, 8, 8
    result = _make_ff_result_for_ba(N, H, W)
    seen_kwargs = {}

    def fake_extract(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return (np.ones((N, 5, 2), dtype=np.float32),
                np.ones((N, 5), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32))

    with patch("collab_splats.geometry.bundle_adjustment._extract_tracks_vggsfm", side_effect=fake_extract):
        ba = BundleAdjustment(config=BundleAdjustmentConfig(fine_tracking=False))
        ba._load_or_extract_tracks(result)

    assert seen_kwargs.get("fine_tracking") is False
```

Also update the `fake_extract` signature inside the existing `test_tracks_cache_save_load` (line ~716) so the new kwarg is accepted:

```python
    def fake_extract(images, confidence, world_points, max_query_pts, query_frame_num, fine_tracking=True, device=None):
        extract_calls.append(1)
        return fake_tracks, fake_vis, fake_pts3d
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "cache or fine_tracking" -v`
Expected: `test_tracks_cache_invalidates_on_fine_tracking_change` FAIL (cache wrongly hit → t2 == fake_a), `test_extract_receives_fine_tracking_kwarg` FAIL (kwarg absent); the two pre-existing cache tests PASS.

- [ ] **Step 3: Implement**

`_compute_tracks_cache_key` meta dict gains one entry:

```python
    meta = {
        "image_paths": sorted(str(p) for p in (result.image_paths or [])),
        "max_query_pts": cfg.max_query_pts,
        "query_frame_num": cfg.query_frame_num,
        "fine_tracking": cfg.fine_tracking,
    }
```

`_load_or_extract_tracks`'s `_extract` closure passes it through (and logs it):

```python
        def _extract() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            logger.info(
                "Extracting VGGSfM tracks: %d frames, max_query_pts=%d, query_frame_num=%d, fine_tracking=%s (slow step)",
                len(result.images),
                cfg.max_query_pts,
                cfg.query_frame_num,
                cfg.fine_tracking,
            )
            return _extract_tracks_vggsfm(
                result.images,
                result.confidence,
                result.world_points,
                max_query_pts=cfg.max_query_pts,
                query_frame_num=cfg.query_frame_num,
                fine_tracking=cfg.fine_tracking,
                device=cfg.device,
            )
```

`_extract_tracks_vggsfm` keeps its signature (`fine_tracking` param already exists) but its default flips to match the config era:

```python
    fine_tracking: bool = True,
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "cache or fine_tracking or extract" -v`
Expected: PASS all.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "fix(ba): thread fine_tracking into VGGSfM extraction + track cache key"
```

---

### Task 3: `_filter_observations` — vis gate + upstream filter order

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:288-310` (`_optimize` filtering block → helper call), new module-level helper in the Helpers section (after `_scale_intrinsics_to_model`)
- Test: `tests/geometry/test_bundle_adjustment.py`

**Scope guard:** `_filter_observations` is a RELOCATION of the existing filtering block out of `_optimize`, not new abstraction — net code size unchanged, and it makes the filter logic testable without CUDA or mocked-LM machinery. Do not generalize it further (no config object param, no class, single call site is fine).

- [ ] **Step 1: Write the failing tests**

Add near the other CPU-runnable `_optimize` tests (direct-import style):

```python
def test_filter_observations_vis_threshold():
    """Observations under vis_thresh are dropped; landmarks left with <2 obs die with them."""
    from collab_splats.geometry.bundle_adjustment import _filter_observations

    vis_scores = np.array([[0.9, 0.1], [0.9, 0.9]], dtype=np.float32)
    tracks = np.zeros((2, 2, 2), dtype=np.float32)
    pts3d = np.zeros((2, 3), dtype=np.float64)
    ext = np.zeros((2, 3, 4), dtype=np.float32)
    intr = np.zeros((2, 3, 3), dtype=np.float32)

    vis = _filter_observations(
        vis_scores, tracks, pts3d, ext, intr,
        vis_thresh=0.2, max_reproj=None, min_inliers_per_frame=1,
    )
    # (0,1) fails the 0.2 gate; landmark 1 then has a single obs -> dropped everywhere
    assert not vis[0, 1] and not vis[1, 1]
    assert vis[0, 0] and vis[1, 0]


def test_filter_observations_no_single_obs_landmark_after_frame_drop():
    """Upstream order: frames drop BEFORE the >=2-obs landmark check, so no landmark
    can survive on observations from dropped frames (old code kept single-obs landmarks)."""
    from collab_splats.geometry.bundle_adjustment import _filter_observations

    vis_scores = np.array([
        [0.9, 0.9, 0.9],   # frame 0: 3 obs
        [0.0, 0.0, 0.9],   # frame 1: 1 obs -> under min_inliers=2, whole frame drops
        [0.9, 0.9, 0.0],   # frame 2: 2 obs
    ], dtype=np.float32)
    tracks = np.zeros((3, 3, 2), dtype=np.float32)
    pts3d = np.zeros((3, 3), dtype=np.float64)
    ext = np.zeros((3, 3, 4), dtype=np.float32)
    intr = np.zeros((3, 3, 3), dtype=np.float32)

    vis = _filter_observations(
        vis_scores, tracks, pts3d, ext, intr,
        vis_thresh=0.2, max_reproj=None, min_inliers_per_frame=2,
    )
    # Landmark 2 was seen only by frames 0 and (dropped) 1 -> single obs -> fully dropped
    assert not vis[:, 2].any()
    # Invariant: every surviving landmark has >=2 observations
    assert (vis.sum(0)[vis.any(0)] >= 2).all()
```

(No separate bool-mask test — existing `_optimize` tests pass bool `vis_mask` arrays through the full path and would catch a bool/float regression; `bool > float` comparison is valid numpy.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "filter_observations" -v`
Expected: FAIL — `ImportError: cannot import name '_filter_observations'`

- [ ] **Step 3: Implement the helper**

Add to the Helpers section of `bundle_adjustment.py` (after `_scale_intrinsics_to_model`):

```python
def _filter_observations(
    vis_scores: np.ndarray,
    tracks: np.ndarray,
    pts3d: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    vis_thresh: float,
    max_reproj: float | None,
    min_inliers_per_frame: int,
) -> np.ndarray:
    """Return bool (N, P) observation mask: visibility gate, reprojection filter,
    frame min-inlier drop, then landmark >=2-obs/in-range drop (upstream demo_colmap order)."""
    # Visibility gate: keep observations the tracker is confident about
    vis = vis_scores > vis_thresh

    # Remove observations with high reprojection error under the current poses
    if max_reproj is not None:
        proj2d, proj_cam = project_3D_points_np(pts3d, extrinsics, intrinsics)
        # Behind-camera points get large sentinel projection so they fail the threshold
        behind = proj_cam[:, 2, :] <= 0
        proj2d = proj2d.copy()
        proj2d[behind] = 1e6
        reproj_err = np.linalg.norm(proj2d - tracks, axis=-1)
        vis[reproj_err > max_reproj] = False

    # Drop under-constrained frames FIRST so the landmark count below reflects only
    # surviving frames — otherwise single-observation landmarks slip through
    vis[vis.sum(1) < min_inliers_per_frame] = False

    # Drop landmarks seen from fewer than 2 surviving frames or outside valid world range
    seen_enough = vis.sum(0) >= 2
    in_range = (np.abs(pts3d) < 3000).all(axis=-1)
    vis[:, ~(seen_enough & in_range)] = False
    return vis
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "filter_observations" -v`
Expected: PASS both.

- [ ] **Step 5: Integrate into `_optimize`**

Replace `bundle_adjustment.py` lines 288-310 (from `# Work on copies...` through the frame-drop line) with:

```python
        # Work on copies
        refined_extrinsics = extrinsics.astype(np.float32).copy()
        refined_intrinsics = intrinsics.astype(np.float32).copy()
        refined_pts3d = pts3d.astype(np.float64).copy()

        # Visibility gate + reprojection filter + frame/landmark drops (upstream order)
        vis = _filter_observations(
            vis_scores,
            tracks,
            pts3d,
            extrinsics,
            intrinsics,
            vis_thresh=cfg.vis_thresh,
            max_reproj=max_reproj,
            min_inliers_per_frame=cfg.min_inliers_per_frame,
        )
```

(The old `vis = vis_scores.astype(bool).copy()` line and the three filtering blocks are deleted — the helper is their only home now.)

- [ ] **Step 6: Run the whole BA test module**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -v`
Expected: all PASS (existing `_optimize` tests pass bool masks — covered by `test_filter_observations_bool_mask_input` semantics). CUDA-marked tests skip on non-GPU shells; on this machine they run — fine either way.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "fix(ba): visibility gate + upstream filter order via _filter_observations"
```

---

### Task 4 (CONDITIONAL — skip unless Task 6's gate demands it): `ba_percam` eval condition

**Execute ONLY if the Task 6 measurement is ambiguous** — mapanything regresses, or vggtx improves but stays above baseline — and knob attribution is needed. If the parity run wins cleanly, this task is never built (YAGNI).

**Files:**
- Modify: `evals/scripts/eval.py` (`_FIXED_CONDITIONS` line 59, `_COLORS` line 60, `_make_creator` single-pass + windowed `ba` branches ~lines 256-266)
- Test: `tests/evals/test_eval_gt_helpers.py` (mirror `test_make_creator_ba_track_density_4096` at line 103)

- [ ] **Step 1: Write the failing test**

```python
def test_make_creator_ba_percam(monkeypatch):
    """ba_percam → BA with shared_camera=False (per-frame focal ablation)."""
    import eval as eval_gt
    from collab_splats.geometry import BundleAdjustmentConfig
    from unittest.mock import MagicMock

    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    creator, ba_cfg = eval_gt._make_creator("ba_percam")
    assert isinstance(ba_cfg, BundleAdjustmentConfig)
    assert ba_cfg.shared_camera is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py::test_make_creator_ba_percam -v`
Expected: FAIL — `ValueError` (unknown condition) from `_make_creator`'s validation.

- [ ] **Step 3: Implement**

In `evals/scripts/eval.py`:

```python
_FIXED_CONDITIONS = {"baseline", "ba", "ba_percam", "lc"}
```

`_COLORS` gains `"ba_percam": "tab:purple"`.

In `_make_creator`, both `ba` branches gain the sibling (windowed branch ~line 260 and single-pass ~line 264):

```python
        if condition == "ba":
            return windowed, BundleAdjustmentConfig()
        if condition == "ba_percam":
            return windowed, BundleAdjustmentConfig(shared_camera=False)
        return windowed, None  # baseline
    # Default: single-pass (short sequences that fit in GPU memory)
    if condition == "ba":
        return base, BundleAdjustmentConfig()
    if condition == "ba_percam":
        return base, BundleAdjustmentConfig(shared_camera=False)
    return base, None  # baseline
```

- [ ] **Step 4: Run eval tests**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v`
Expected: all PASS (density tests unaffected — `ba_track-density-{N}` keeps its own branch, and its `query_frame_num=max(5, n//512)` formula is untouched).

- [ ] **Step 5: Commit**

```bash
git add evals/scripts/eval.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(evals): ba_percam condition — shared-camera ablation for BA parity sweep"
```

---

### Task 5: Full-suite regression gate

**Files:** none (verification only)

- [ ] **Step 1: Run geometry + evals + wrapper suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ tests/evals/ tests/wrapper/ -q`
Expected: green, except the 5 known wrapper "failures" caused by another session's uncommitted `configs/base.yaml` (see handoff §Environment) and entries in `docs/known-test-failures.md`. Anything new → fix before proceeding.

- [ ] **Step 2: Check formatting**

Run: `/opt/venv/reconstruction/bin/python -m black --check collab_splats/geometry/bundle_adjustment.py evals/scripts/eval.py tests/geometry/test_bundle_adjustment.py tests/evals/test_eval_gt_helpers.py`
Expected: clean. If not: run black on ONLY these files (never repo-wide — venv black 26.5.1 is newer than repo formatting), re-run tests, amend the last commit.

---

### Task 6: Validation sweep (HUMAN-GATED — heavy compute, tmux only, serial)

**Files:**
- Results: `evals/results/ba_parity_chess/<backbone>/` (gitignored)
- Modify after measurement: `docs/superpowers/specs/2026-08-19-ba-track-quality-parity-design.md` (§Validation), `CLAUDE.md` (BA verdict line), `docs/superpowers/handoffs/` if handing off

- [ ] **Step 1: Confirm with the user before launching** — two serial GPU runs (third only if the gate demands attribution), ~cgroup-capped machine. Do not run alongside other heavy jobs.

- [ ] **Step 2: Run vggtx (worst regression) — new defaults**

```bash
tmux new -s ba_parity -d
tmux send-keys -t ba_parity "/opt/venv/reconstruction/bin/python evals/scripts/eval.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --output_dir evals/results/ba_parity_chess/vggtx \
  --max_frames 100 --backbone vggtx \
  --conditions baseline ba \
  --output_ate evals/results/ba_parity_chess/vggtx/ate.json 2>&1 | tee evals/results/ba_parity_chess/vggtx.log" Enter
```

Reference numbers to beat (2026-08-19 sweep, old code): vggtx baseline ATE 0.0077 / ba 0.0099-0.0133 band — exact per-metric table in `docs/superpowers/specs/2026-08-18-ba-pipeline-wiring-design.md` §Validation.

- [ ] **Step 3: Run mapanything — regression guard (BA helped it; must not lose that)**

Same command with `--backbone mapanything`, output dir `.../mapanything`. Guard: ba ATE must stay ≤ its baseline (0.0133) and ideally ≤ the old measured ba (0.0120).

- [ ] **Step 4: Decision gate (from spec)**

- vggtx `ba` ATE ≤ baseline AND mapanything holds its gain → parity restored; proceed to Step 5. Task 4 stays unbuilt.
- Improved but still > baseline → build Task 4, run `--conditions ba_percam` (output dir `.../vggtx_percam`; cheap — track cache reused, `shared_camera` not in the cache key) to attribute, then next effort is option B (1024/original-res track extraction). Record numbers first.
- No improvement → same attribution run, then option C (focal freeze / pose-only BA). Record numbers first.
- mapanything regressed → build Task 4 and ablate which new knob costs it before landing anything.

- [ ] **Step 5: Document measured results**

Append the measured table (baseline / ba / ba_percam × ATE, RPE-t, RPE-rot, AUC@5) to the spec's §Validation, update the CLAUDE.md "Recently completed" BA verdict if it changes, and commit:

```bash
git add -f docs/superpowers/specs/2026-08-19-ba-track-quality-parity-design.md CLAUDE.md
git commit -m "docs(specs): BA track-quality parity — measured chess/seq-01 results"
```

---

## Execution notes

- Tests must not require CUDA except those already marked `@pytest.mark.skipif(not _cuda_and_bae_available())`.
- `_build_synthetic_scene` returns a bool `vis_mask`; the new gate (`vis_scores > vis_thresh`) accepts bool arrays unchanged — do NOT convert existing tests to float scores.
- Track caches in existing backend dirs invalidate automatically (cache key gains `fine_tracking`) — no manual cleanup needed.
- Concurrent-session trap: `configs/base.yaml` may be dirty from another session — never `git add -A`; stage named files only.
