# BA LM Convergence + Dropped-Frame Gauge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make bundle adjustment run its configured 40 LM steps in every code path, keep frames dropped by the inlier gate consistent with the refined reconstruction, and let eval reuse extracted tracks.

**Architecture:** Four independent changes in `collab_splats/geometry/bundle_adjustment.py` plus one helper relocation and two eval-script additions. The LM change *deletes* a branch (the `capture_loss_history=False` path that selects `StopOnPlateau`) rather than adding one. The gauge fix is a new pure-numpy module-level helper alongside the existing `_filter_observations`, so it is testable without CUDA.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, pypose 0.7.5, bae 0.2.4, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-ba-lm-convergence-design.md`

**Branch:** work directly on `refactor/cu121-uv-migration`. Do **not** create a worktree — repo convention.

**Staging discipline:** a concurrent session keeps `configs/base.yaml`, `collab_splats/remote/rerun.py`, `docs/examples/run_pipeline_remote.py`, `pyproject.toml`, `data/tutorial/README.md`, `tests/examples/test_run_pipeline_remote.py` dirty. **Never `git add -A`.** Stage only the files each task names. **Never run repo-wide `black .`** — the venv's black is newer than the repo's formatting and fails at the base commit too.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `collab_splats/geometry/transforms.py` | pure-numpy pose/geometry helpers, no heavy imports | **gains** `umeyama_se3`, `umeyama_sim3` |
| `collab_splats/geometry/loop_closure/graph.py` | SL(4) pose graph; imports `gtsam` at module level | **loses** the two `umeyama_*` bodies, re-imports them |
| `collab_splats/geometry/loop_closure/eval.py` | LC trajectory metrics | import line updated |
| `collab_splats/geometry/bundle_adjustment.py` | LM BA | LM loop, `_carry_dropped_frames`, focal writeback, config field removal |
| `collab_splats/wrapper/reconstructor.py` | pipeline stages | drops one kwarg |
| `evals/scripts/eval.py` | eval compute driver | `--tracks_cache_dir`, `ba_coarse` |
| `tests/geometry/test_transforms.py` | transforms unit tests | new umeyama tests |
| `tests/geometry/test_bundle_adjustment.py` | BA unit tests | loss-history + gauge tests |
| `tests/evals/test_eval_gt_helpers.py` | eval helper unit tests | cache-dir + `ba_coarse` tests |

---

### Task 1: Move the umeyama helpers into `transforms.py`

`umeyama_sim3` currently lives in `collab_splats/geometry/loop_closure/graph.py`, which does
`import gtsam` at module level. Bundle adjustment must not pull gtsam in, so the helper moves to
`transforms.py` (pure numpy, already imported by `bundle_adjustment.py`). Both helpers move
together — splitting the pair across modules is worse than moving the one BA does not use yet.
This is a pure relocation: no behaviour change.

**Files:**
- Modify: `collab_splats/geometry/transforms.py` (append at end of file)
- Modify: `collab_splats/geometry/loop_closure/graph.py:460-545` (delete both function bodies)
- Modify: `collab_splats/geometry/loop_closure/eval.py:14` (import line)
- Test: `tests/geometry/test_transforms.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_transforms.py`:

```python
def test_umeyama_sim3_recovers_known_similarity():
    """umeyama_sim3 recovers the (s, R, t) that generated the target points."""
    from collab_splats.geometry.transforms import umeyama_sim3

    rng = np.random.default_rng(0)
    src = rng.normal(size=(12, 3))
    # Known 90 deg rotation about z, scale 2.5, translation (1, -2, 3)
    R_true = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s_true, t_true = 2.5, np.array([1.0, -2.0, 3.0])
    dst = s_true * (R_true @ src.T).T + t_true

    s, R, t = umeyama_sim3(src, dst)
    assert np.isclose(s, s_true, atol=1e-5)
    assert np.allclose(R, R_true, atol=1e-5)
    assert np.allclose(t, t_true, atol=1e-4)


def test_umeyama_sim3_too_few_points_returns_identity():
    """Fewer than 3 correspondences cannot fix a Sim(3): identity is returned."""
    from collab_splats.geometry.transforms import umeyama_sim3

    s, R, t = umeyama_sim3(np.zeros((2, 3)), np.ones((2, 3)))
    assert s == 1.0
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, np.zeros(3))


def test_umeyama_se3_recovers_known_rigid_transform():
    """umeyama_se3 recovers a rigid transform as a (4,4) homogeneous matrix."""
    from collab_splats.geometry.transforms import umeyama_se3

    rng = np.random.default_rng(1)
    src = rng.normal(size=(10, 3))
    R_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    t_true = np.array([0.5, 0.25, -1.0])
    dst = (R_true @ src.T).T + t_true

    T = umeyama_se3(src, dst)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], R_true, atol=1e-5)
    assert np.allclose(T[:3, 3], t_true, atol=1e-5)


def test_transforms_does_not_import_gtsam():
    """transforms.py must stay light: importing it must not pull gtsam in."""
    import subprocess
    import sys

    code = (
        "import collab_splats.geometry.transforms; "
        "import sys; "
        "sys.exit(1 if 'gtsam' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
```

If `tests/geometry/test_transforms.py` does not exist, create it with this header first:

```python
"""Unit tests for collab_splats.geometry.transforms."""

import numpy as np
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -k umeyama -v -p no:randomly`
Expected: FAIL with `ImportError: cannot import name 'umeyama_sim3' from 'collab_splats.geometry.transforms'`

- [ ] **Step 3: Move both functions**

Cut `umeyama_se3` (currently `graph.py:460-497`) and `umeyama_sim3` (`graph.py:499-545`) verbatim
— docstrings and bodies unchanged — and append them to the end of
`collab_splats/geometry/transforms.py`, under a section divider matching the file's style:

```python
########################################
########## Point-set alignment #########
########################################
```

In `graph.py`, delete both definitions and add to its imports (after `from scipy.linalg import rq`):

```python
from collab_splats.geometry.transforms import umeyama_se3, umeyama_sim3
```

`graph.py` re-exports them by virtue of the import, so
`collab_splats/geometry/loop_closure/eval.py:14` (`from .graph import PoseGraph, umeyama_se3, umeyama_sim3`)
keeps working unchanged. Leave it as-is.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -v -p no:randomly`
Expected: PASS (including `test_transforms_does_not_import_gtsam`)

- [ ] **Step 5: Verify the loop-closure suite is unaffected**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -q -p no:randomly`
Expected: same pass/fail counts as before the change (BA tests may already be failing from other
tasks — compare against `git stash` baseline if unsure).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/transforms.py \
        collab_splats/geometry/loop_closure/graph.py \
        tests/geometry/test_transforms.py
git commit -m "refactor(geometry): move umeyama_se3/sim3 to transforms.py

graph.py imports gtsam at module level; bundle_adjustment needs
umeyama_sim3 and must stay free of that dependency. Pure relocation."
```

---

### Task 2: One LM loop — delete the `StopOnPlateau` branch

`_optimize` picks its stopping rule from `cfg.capture_loss_history`: `True` runs a plain
`for i in range(n_steps)` loop (all 40 steps), `False` runs `StopOnPlateau`, which aborts after
1–2 steps because `pypose/optim/scheduler.py:153-155` stops whenever
`optimizer.reject_count > 0` and bae's LM increments that on any trust-region damping retry.
Production (`reconstructor.py:927`) passes `True`; eval passes nothing. Delete the branch, always
take the loop that already exists, and delete the now-dead flag.

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:75` (delete config field), `:96-97` (comment), `:410-436` (the branch)
- Modify: `collab_splats/wrapper/reconstructor.py:927`
- Test: `tests/geometry/test_bundle_adjustment.py:668-709`

- [ ] **Step 1: Write the failing tests**

Replace `test_optimize_captures_loss_history_when_flag_set` and
`test_optimize_no_loss_history_by_default` (`tests/geometry/test_bundle_adjustment.py:676-709`)
with:

```python
@pytest.mark.skipif(not _cuda_and_bae_available(), reason="requires CUDA, pypose, and bae")
def test_optimize_captures_loss_history_unconditionally():
    """_optimize always records one inner list of per-step losses (no flag gates it)."""
    from collab_splats.geometry.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(lm_steps=n_steps, min_inliers_per_frame=10)
    ba = BundleAdjustment(config=cfg)
    assert ba._last_loss_history == []

    ba._optimize(pts3d, extrinsics, intrinsics, tracks, vis_mask.astype(np.float32), max_reproj_error=None)

    hist = ba._last_loss_history
    assert len(hist) == 1, f"one _optimize call -> one inner list; got {len(hist)}"
    assert len(hist[0]) == n_steps, (
        f"expected all {n_steps} LM steps; got {len(hist[0])}. A short history means the "
        "StopOnPlateau reject_count abort is back."
    )
    assert all(isinstance(v, float) for v in hist[0])


def test_ba_config_has_no_capture_loss_history_field():
    """capture_loss_history is deleted: history is always captured, so the flag is dead."""
    import dataclasses

    from collab_splats.geometry.bundle_adjustment import BundleAdjustmentConfig

    names = {f.name for f in dataclasses.fields(BundleAdjustmentConfig)}
    assert "capture_loss_history" not in names
```

In `test_bundle_adjustment_default_config` (`:668-672`), delete the line
`assert ba.config.capture_loss_history is False` and its preceding comment; keep
`assert ba._last_loss_history == []`.

In `test_incremental_ba_loss_history_per_step` (`:920-949`), the fake at `:933-934` reads the
flag. Replace those two lines with the unconditional form:

```python
        self_ba._last_loss_history.append([float(k) * 0.1])
```

and at `:942` drop the kwarg:

```python
        ba = BundleAdjustment(BundleAdjustmentConfig(increment_size=2))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "capture_loss_history or loss_history" -v -p no:randomly`
Expected: `test_ba_config_has_no_capture_loss_history_field` FAILS (`assert 'capture_loss_history' not in names`). The CUDA-marked test is skipped on a CPU box; that is fine — Step 5 runs it on GPU.

- [ ] **Step 3: Delete the branch and the flag**

In `collab_splats/geometry/bundle_adjustment.py`, delete line 75:

```python
    capture_loss_history: bool = False  # record per-step LM loss; read via BundleAdjustment._last_loss_history
```

Update the comment at `:96` to:

```python
        # Populated per _optimize() call with that call's per-step LM losses
```

Replace the whole `if cfg.capture_loss_history: ... else: ...` block (`:410-436`) with:

```python
            # Manual LM loop. pypose's StopOnPlateau is unusable here: it aborts the whole
            # optimization as soon as bae's LM needs a single trust-region damping retry
            # (reject_count > 0), which routinely fires on step 1, and its plateau test is an
            # absolute difference against 1e-3 so it never fires at our loss scale (~1e5-1e6).
            loss_hist: list[float] = []
            for i in range(n_steps):
                step_loss = optimizer.step(input=input_dict)
                loss_hist.append(float(step_loss))
                logger.info("LM step %d/%d: loss=%.6e", i + 1, n_steps, float(step_loss))
            self._last_loss_history.append(loss_hist)
```

In `collab_splats/wrapper/reconstructor.py:927`:

```python
        cfg = BundleAdjustmentConfig(tracks_cache_dir=self.backend_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -v -p no:randomly`
Expected: PASS (CUDA-marked tests skip on a CPU box).

Also confirm nothing else references the flag:

Run: `grep -rn "capture_loss_history" --include=*.py . | grep -v '\.worktrees'`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py \
        collab_splats/wrapper/reconstructor.py \
        tests/geometry/test_bundle_adjustment.py
git commit -m "fix(ba): run all configured LM steps; drop capture_loss_history

pypose StopOnPlateau stops as soon as bae's LM needs one damping retry
(scheduler.py:153-155 vs optimizer.py:32), so the non-capture path ran
1-2 of 40 steps. Production already took the plain loop via
capture_loss_history=True; eval did not, so the two paths ran different
solvers. Keep the plain loop, delete the flag."
```

---

### Task 3: `_carry_dropped_frames` helper

Frames dropped by `min_inliers_per_frame` keep their pre-BA pose while every other frame moves to
the solved gauge — and `_BAModel` fixes neither a frame nor scale, so that gauge can drift as a
whole. This helper transforms dropped frames by the Sim(3) the active set underwent. Pure numpy
and module-level (like the existing `_filter_observations`) so it is testable without CUDA.

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py` (add helper next to `_filter_observations`)
- Test: `tests/geometry/test_bundle_adjustment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_bundle_adjustment.py`:

```python
def _w2c_from_rt(R, t):
    """Stack (3,3) rotations and (3,) translations into (N,3,4) world-to-cam extrinsics."""
    return np.concatenate([R, t[..., None]], axis=-1).astype(np.float32)


def test_carry_dropped_frames_applies_active_set_sim3():
    """A dropped frame's camera centre lands at s*R_g@C + t_g, the same map the points took."""
    from collab_splats.geometry.bundle_adjustment import _carry_dropped_frames
    from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses

    rng = np.random.default_rng(7)
    N = 5
    # Random-but-valid original poses (orthogonalize a random matrix per frame)
    R0 = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(N)])
    R0[np.linalg.det(R0) < 0] *= -1.0
    t0 = rng.normal(size=(N, 3))
    original = _w2c_from_rt(R0, t0)

    # Known world-gauge Sim(3): 90 deg about z, scale 1.5, translation (0.3, -0.7, 2.0)
    R_g = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    s_g, t_g = 1.5, np.array([0.3, -0.7, 2.0])
    # Apply it to every frame to build the "refined" set, then revert frame 3 to its original
    R_ref = R0 @ R_g.T
    t_ref = s_g * t0 - np.einsum("nij,j->ni", R_ref, t_g)
    refined = _w2c_from_rt(R_ref, t_ref)
    refined[3] = original[3]
    active = np.array([0, 1, 2, 4])

    out, scale = _carry_dropped_frames(refined, original, active)

    centers_orig = invert_poses(extrinsics_to_homogeneous(original))[:, :3, 3]
    centers_out = invert_poses(extrinsics_to_homogeneous(out))[:, :3, 3]
    expected = s_g * (R_g @ centers_orig[3]) + t_g
    assert np.allclose(centers_out[3], expected, atol=1e-4), (
        f"dropped frame centre {centers_out[3]} != gauge-mapped {expected}"
    )
    assert np.isclose(scale, s_g, atol=1e-4)
    # Active frames are untouched
    assert np.allclose(out[active], refined[active], atol=1e-6)


def test_carry_dropped_frames_noop_when_all_active():
    """No dropped frames -> array returned unchanged and scale is None."""
    from collab_splats.geometry.bundle_adjustment import _carry_dropped_frames

    refined = np.tile(np.eye(4, dtype=np.float32)[:3], (4, 1, 1))
    out, scale = _carry_dropped_frames(refined, refined.copy(), np.arange(4))
    assert scale is None
    assert np.allclose(out, refined)


def test_carry_dropped_frames_needs_three_active_frames():
    """Fewer than 3 active frames cannot fix a Sim(3): dropped frames are left alone."""
    from collab_splats.geometry.bundle_adjustment import _carry_dropped_frames

    refined = np.tile(np.eye(4, dtype=np.float32)[:3], (4, 1, 1))
    original = refined.copy()
    refined[:2, :, 3] += 5.0  # move only the active pair
    out, scale = _carry_dropped_frames(refined, original, np.array([0, 1]))
    assert scale is None
    assert np.allclose(out, refined)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k carry_dropped -v -p no:randomly`
Expected: FAIL with `ImportError: cannot import name '_carry_dropped_frames'`

- [ ] **Step 3: Write the helper**

In `collab_splats/geometry/bundle_adjustment.py`, add `umeyama_sim3` and `invert_poses` to the
existing transforms import:

```python
from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses, umeyama_sim3
```

Add immediately after `_filter_observations`:

```python
def _carry_dropped_frames(
    refined_extrinsics: np.ndarray,
    extrinsics: np.ndarray,
    active_frames: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Transform frames outside active_frames by the Sim(3) the active set underwent.

    BA fixes no frame and no scale, so the refined set can drift as a whole. A frame the
    inlier gate dropped keeps its pre-BA pose and would otherwise sit in the pre-BA gauge,
    inconsistent with every refined frame around it.

    Args:
        refined_extrinsics: (N, 3, 4) poses with active rows already refined.
        extrinsics: (N, 3, 4) original pre-BA poses.
        active_frames: (K,) indices refined by the solve.

    Returns:
        (extrinsics, scale): a copy with dropped frames carried, and the Sim(3) scale.
        scale is None when nothing was dropped or the gauge could not be estimated.
    """
    inactive = np.setdiff1d(np.arange(refined_extrinsics.shape[0]), active_frames)
    if len(inactive) == 0 or len(active_frames) < 3:
        return refined_extrinsics, None

    # Estimate the world-gauge Sim(3) from how the active cameras' centres moved
    src_c = invert_poses(extrinsics_to_homogeneous(extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    dst_c = invert_poses(extrinsics_to_homogeneous(refined_extrinsics[active_frames].astype(np.float64)))[:, :3, 3]
    s, R_g, t_g = umeyama_sim3(src_c, dst_c)

    # World gauge X' = s R_g X + t_g maps a world-to-cam [R|t] to [R R_g^T | s t - R R_g^T t_g],
    # which puts the dropped camera's centre at s R_g C + t_g — the same map the points took.
    out = refined_extrinsics.copy()
    R_in = refined_extrinsics[inactive, :, :3].astype(np.float64)
    t_in = refined_extrinsics[inactive, :, 3].astype(np.float64)
    R_new = R_in @ R_g.T.astype(np.float64)
    out[inactive, :, :3] = R_new.astype(np.float32)
    out[inactive, :, 3] = (s * t_in - np.einsum("nij,j->ni", R_new, t_g.astype(np.float64))).astype(np.float32)
    return out, float(s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k carry_dropped -v -p no:randomly`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "feat(ba): _carry_dropped_frames maps dropped poses into the solved gauge"
```

---

### Task 4: Wire the gauge carry and the shared focal into `_optimize`

**Files:**
- Modify: `collab_splats/geometry/bundle_adjustment.py:438-457` (writeback block)
- Test: `tests/geometry/test_bundle_adjustment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_bundle_adjustment.py`:

```python
def test_shared_focal_written_to_every_frame(monkeypatch):
    """With shared_camera=True the solved focal lands on dropped frames too.

    A dropped frame keeping its own focal in an otherwise single-camera reconstruction is
    exactly the mixed-K state shared_camera exists to remove.
    """
    import inspect

    from collab_splats.geometry import bundle_adjustment as ba_mod

    src = inspect.getsource(ba_mod._optimize) if hasattr(ba_mod, "_optimize") else inspect.getsource(
        ba_mod.BundleAdjustment._optimize
    )
    focal_block = src.split("Write optimised focal lengths back")[1]
    assert "refined_intrinsics[active_frames, 0, 0] = focal_val" not in focal_block, (
        "shared focal must be written to all frames, not only active_frames"
    )
    assert "refined_intrinsics[:, 0, 0] = focal_val" in focal_block


def test_optimize_calls_carry_dropped_frames(monkeypatch):
    """_optimize routes its writeback through _carry_dropped_frames."""
    import inspect

    from collab_splats.geometry import bundle_adjustment as ba_mod

    src = inspect.getsource(ba_mod.BundleAdjustment._optimize)
    assert "_carry_dropped_frames(" in src
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -k "shared_focal_written or calls_carry" -v -p no:randomly`
Expected: both FAIL (`assert '_carry_dropped_frames(' in src`, and the focal assertion).

- [ ] **Step 3: Wire it in**

In `_optimize`, replace the writeback block (currently `:444-455`) with:

```python
        refined_extrinsics[active_frames] = opt_extrinsics_3x4.astype(np.float32)
        refined_pts3d[active_pts] = opt_pts.astype(np.float64)

        # Frames the inlier gate dropped keep their pre-BA pose, which sits in the pre-BA
        # gauge — BA fixes no frame and no scale, so the refined set can drift as a whole.
        # Carry them by the Sim(3) the active set underwent so they stay consistent.
        n_dropped = vis.shape[0] - len(active_frames)
        if n_dropped:
            inactive = np.setdiff1d(np.arange(vis.shape[0]), active_frames)
            refined_extrinsics, gauge_scale = _carry_dropped_frames(
                refined_extrinsics, extrinsics, active_frames
            )
            if gauge_scale is None:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s) and left "
                    "in the pre-BA gauge — fewer than 3 active frames, cannot estimate it",
                    n_dropped, vis.shape[0], cfg.min_inliers_per_frame, inactive.tolist(),
                )
            else:
                logger.warning(
                    "BA: %d/%d frames dropped (min_inliers_per_frame=%d, indices %s); carried "
                    "by the active-set Sim(3) (scale %.6f) but not refined",
                    n_dropped, vis.shape[0], cfg.min_inliers_per_frame, inactive.tolist(),
                    gauge_scale,
                )

        # Write optimised focal lengths back to intrinsics matrix. The shared focal goes to
        # every frame, including dropped ones: a per-frame focal surviving in an otherwise
        # single-camera reconstruction is the mixed-K state shared_camera exists to remove.
        if cfg.shared_camera and model.shared_intr is not None:
            focal_val = float(model.shared_intr.data.detach().cpu().numpy().mean())
            refined_intrinsics[:, 0, 0] = focal_val
            refined_intrinsics[:, 1, 1] = focal_val
        elif not cfg.shared_camera:
            opt_focal = opt_cam[:, 7]
            refined_intrinsics[active_frames, 0, 0] = opt_focal
            refined_intrinsics[active_frames, 1, 1] = opt_focal
```

Note the `elif not cfg.shared_camera` branch keeps `active_frames`: with per-frame focals there is
no scene-wide value to give a dropped frame, so it keeps its own.

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_bundle_adjustment.py -v -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/bundle_adjustment.py tests/geometry/test_bundle_adjustment.py
git commit -m "fix(ba): keep dropped frames in the solved gauge, share focal scene-wide

_optimize wrote back only active_frames, so a frame dropped by the
min-inlier gate kept its pre-BA pose and per-frame focal while every
other frame moved to the solved gauge — silently mixing refined and
unrefined poses in one reconstruction."
```

---

### Task 5: `--tracks_cache_dir` in eval

`tracks_cache_dir` defaults to `None` (always extract) and `eval.py` builds
`BundleAdjustmentConfig()` bare, so every eval BA condition re-extracts tracks (~11 min of a
~14 min run).

The spec says "`_make_creator` passes it into every `BundleAdjustmentConfig` it constructs".
Implement that through a local `_ba()` factory rather than repeating the kwarg at all eight
construction sites — same observable behaviour, one place to change.

**Files:**
- Modify: `evals/scripts/eval.py:217-278` (`_make_creator`), `:281-299` (`_run_condition`), `:415-440` (argparse), `:505-513` (call site)
- Test: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/evals/test_eval_gt_helpers.py`:

```python
def test_make_creator_threads_tracks_cache_dir(monkeypatch, tmp_path):
    """tracks_cache_dir reaches the BA config for every BA condition."""
    import eval as eval_gt
    from unittest.mock import MagicMock

    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    for cond in ("ba", "ba_percam", "ba_track-density-4096", "incremental_ba-5"):
        _, ba_cfg = eval_gt._make_creator(cond, tracks_cache_dir=tmp_path)
        assert ba_cfg.tracks_cache_dir == tmp_path, f"{cond} dropped tracks_cache_dir"


def test_make_creator_tracks_cache_dir_defaults_to_none(monkeypatch):
    """Unset --tracks_cache_dir leaves the config default (always extract)."""
    import eval as eval_gt
    from unittest.mock import MagicMock

    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    _, ba_cfg = eval_gt._make_creator("ba")
    assert ba_cfg.tracks_cache_dir is None


def test_tracks_cache_dir_arg_parses_path():
    """--tracks_cache_dir parses to a Path and defaults to None."""
    from pathlib import Path

    import eval as eval_gt

    parser = eval_gt._build_parser()
    assert parser.parse_args(["--dataset", "7scenes"]).tracks_cache_dir is None
    parsed = parser.parse_args(["--dataset", "7scenes", "--tracks_cache_dir", "/tmp/tc"])
    assert parsed.tracks_cache_dir == Path("/tmp/tc")
```

If `eval.py` builds its parser inline in `main()` rather than in a `_build_parser()` function,
replace the third test with one that reads the parser through whatever accessor exists; check
`test_slam_tum_arg_parses_path` (`tests/evals/test_eval_gt_helpers.py:257`) and copy its pattern
exactly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -k tracks_cache -v -p no:randomly`
Expected: FAIL with `TypeError: _make_creator() got an unexpected keyword argument 'tracks_cache_dir'`

- [ ] **Step 3: Thread the parameter**

In `evals/scripts/eval.py`, add the parameter to `_make_creator` (`:217-224`):

```python
def _make_creator(
    condition: str,
    submap_size: int | None = None,
    backbone: str = "vggt_omega",
    lc_scale_method: str = "rotation_only",
    max_loops_per_submap: int | None = None,
    loop_edge_timing: str = "deferred",
    tracks_cache_dir: Path | None = None,
):
```

Immediately after `_lc_extra` is built and before `base = get_creator(backbone)()`, add:

```python
    # Every BA config in this function shares the same track cache dir; extraction dominates
    # BA runtime, so re-running conditions against a warm cache is the main speed lever.
    def _ba(**kw) -> BundleAdjustmentConfig:
        return BundleAdjustmentConfig(tracks_cache_dir=tracks_cache_dir, **kw)
```

Replace every `BundleAdjustmentConfig(...)` construction in the function body with `_ba(...)`:

| line | before | after |
|---|---|---|
| ~245 | `BundleAdjustmentConfig(max_query_pts=n, query_frame_num=max(5, n // 512))` | `_ba(max_query_pts=n, query_frame_num=max(5, n // 512))` |
| ~257 | `BundleAdjustmentConfig(increment_size=increment_size)` | `_ba(increment_size=increment_size)` |
| ~268 | `return windowed, BundleAdjustmentConfig()` | `return windowed, _ba()` |
| ~271 | `return windowed, BundleAdjustmentConfig(shared_camera=False)` | `return windowed, _ba(shared_camera=False)` |
| ~275 | `return base, BundleAdjustmentConfig()` | `return base, _ba()` |
| ~277 | `return base, BundleAdjustmentConfig(shared_camera=False)` | `return base, _ba(shared_camera=False)` |

Add the same parameter to `_run_condition` (`:281-289`), defaulting to `None`, and pass it through
in the `_make_creator` call at `:292-298`:

```python
    tracks_cache_dir: Path | None = None,
) -> tuple[np.ndarray, Any]:
    """Run condition, return (extrinsics (N,4,4), creator)."""
    creator, ba_cfg = _make_creator(
        name,
        submap_size=submap_size,
        backbone=backbone,
        lc_scale_method=lc_scale_method,
        max_loops_per_submap=max_loops_per_submap,
        loop_edge_timing=loop_edge_timing,
        tracks_cache_dir=tracks_cache_dir,
    )
```

Add the argparse flag next to `--submap_size` (`:420`):

```python
    parser.add_argument(
        "--tracks_cache_dir",
        type=Path,
        default=None,
        help="Reuse extracted VGGSfM tracks across BA conditions. Single-slot cache: "
        "conditions with different extraction settings (e.g. ba vs ba_coarse, which differ "
        "in fine_tracking) evict each other — give those separate dirs.",
    )
```

And pass it at the `_run_condition` call site (`:505-513`):

```python
        loop_edge_timing=getattr(args, "loop_edge_timing", "deferred"),
        tracks_cache_dir=getattr(args, "tracks_cache_dir", None),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/scripts/eval.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(evals): --tracks_cache_dir so BA conditions reuse extracted tracks

Track extraction is ~11 min of a ~14 min BA eval run and eval built the
config bare, so every condition re-extracted. Default None keeps current
behaviour."
```

---

### Task 6: `ba_coarse` eval condition

`fine_tracking=True` was flipped as one of five simultaneous changes, was never attributed on its
own, and dominates BA runtime. Add the ablation, mirroring the existing `ba_percam` sibling.

**Files:**
- Modify: `evals/scripts/eval.py:59` (`_FIXED_CONDITIONS`), `:60-65` (`_COLORS`), `:266-277` (both `_make_creator` branches)
- Test: `tests/evals/test_eval_gt_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/evals/test_eval_gt_helpers.py`:

```python
def test_make_creator_ba_coarse(monkeypatch):
    """ba_coarse -> BA with fine_tracking=False (coarse-track ablation)."""
    import eval as eval_gt
    from collab_splats.geometry import BundleAdjustmentConfig
    from unittest.mock import MagicMock

    monkeypatch.setattr(eval_gt, "get_creator", lambda name: lambda: MagicMock())
    creator, ba_cfg = eval_gt._make_creator("ba_coarse")
    assert isinstance(ba_cfg, BundleAdjustmentConfig)
    assert ba_cfg.fine_tracking is False
    assert ba_cfg.shared_camera is True  # only fine_tracking differs from `ba`


def test_ba_coarse_is_a_validated_condition():
    """ba_coarse validates and has its own plot colour."""
    import eval as eval_gt

    eval_gt._validate_condition("ba_coarse")  # must not raise
    assert "ba_coarse" in eval_gt._COLORS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -k ba_coarse -v -p no:randomly`
Expected: FAIL — `_make_creator("ba_coarse")` silently returns `(base, None)` so `ba_cfg` is
`None`, and `_validate_condition` raises `ValueError: Unknown condition 'ba_coarse'`.

- [ ] **Step 3: Add the condition**

In `evals/scripts/eval.py:59`:

```python
_FIXED_CONDITIONS = {"baseline", "ba", "ba_coarse", "ba_percam", "lc"}
```

In `_COLORS` (after the `"ba_percam"` entry at `:64`):

```python
    "ba_coarse": "tab:olive",
```

In the windowed branch of `_make_creator`, after the `ba_percam` return (~`:271`):

```python
        # ba_coarse: track-quality ablation — skip the VGGSfM fine refinement stage
        if condition == "ba_coarse":
            return windowed, _ba(fine_tracking=False)
```

And in the single-pass branch, after the `ba_percam` return (~`:277`):

```python
    if condition == "ba_coarse":
        return base, _ba(fine_tracking=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_gt_helpers.py -v -p no:randomly`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add evals/scripts/eval.py tests/evals/test_eval_gt_helpers.py
git commit -m "feat(evals): ba_coarse condition for fine_tracking attribution"
```

---

### Task 7: Full suite

**Files:** none modified — verification only.

- [ ] **Step 1: Run the geometry, evals, and wrapper suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ tests/evals/ tests/wrapper/ -q -p no:randomly`
Expected: no new failures. Five `tests/wrapper/` failures are **pre-existing**, caused by a
concurrent session's dirty `configs/base.yaml` — confirm they are the same five by name, and do
not attempt to fix them.

- [ ] **Step 2: Run the full suite**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly`
Expected: matches the pre-change baseline apart from the tests this plan added. Known failures are
listed in `docs/known-test-failures.md`.

- [ ] **Step 3: Check formatting of only the files touched**

Run: `/opt/venv/reconstruction/bin/python -m black --check collab_splats/geometry/bundle_adjustment.py collab_splats/geometry/transforms.py evals/scripts/eval.py`
Expected: the venv's black is newer than the repo's formatting and **fails at the base commit
too**. Compare against `git stash && black --check <same files>` before changing anything. Do
**not** reformat files this plan did not otherwise touch, and never run repo-wide `black .`.

- [ ] **Step 4: Commit any fixups**

Only if Step 1-3 required changes:

```bash
git add <the specific files you changed>
git commit -m "fix(ba): <what the suite caught>"
```

---

### Task 8: Validation sweep (human-gated compute)

**Do not start these runs without asking the user first.** They are heavy GPU eval runs on
7-Scenes chess/seq-01, must run serially in tmux (container cgroup cap 46.6 GB), and take
substantially longer than the previous sweep because BA now runs 40 LM steps instead of 1–2.

**Files:**
- Create: `<scratchpad>/ba_convergence_runs.sh`
- Modify: `docs/superpowers/specs/2026-08-19-ba-lm-convergence-design.md` (§Measured results)
- Modify: `CLAUDE.md` (verdict entry)

- [ ] **Step 1: Ask the user to authorize the runs**

State the expected wall-clock cost and confirm no other GPU work is running.

- [ ] **Step 2: Write the driver script**

Create `<scratchpad>/ba_convergence_runs.sh`. Note each backbone gets its **own** cache dir, and
`ba_coarse` gets a separate one again because `fine_tracking` is part of the extraction key:

```bash
#!/bin/bash
# BA LM-convergence validation. Serial — one GPU job at a time.
cd /workspace/collab-splats
PY=/opt/venv/reconstruction/bin/python
OUT=evals/results/ba_convergence_chess
SEQ="--dataset 7scenes --seq_dir data/7scenes/chess/seq-01 --max_frames 100"

for BB in vggtx mapanything vggt_omega; do
  echo "=== START $BB $(date -Is) ==="
  $PY evals/scripts/eval.py $SEQ \
    --output_dir $OUT/$BB --backbone $BB \
    --conditions baseline ba \
    --tracks_cache_dir $OUT/cache/$BB \
    --output_ate $OUT/$BB/ate.json 2>&1 | tee $OUT/$BB.log
  echo "=== DONE $BB exit=${PIPESTATUS[0]} $(date -Is) ==="
done

echo "=== START vggtx ba_coarse $(date -Is) ==="
$PY evals/scripts/eval.py $SEQ \
  --output_dir $OUT/vggtx_coarse --backbone vggtx \
  --conditions ba_coarse \
  --tracks_cache_dir $OUT/cache/vggtx_coarse \
  --output_ate $OUT/vggtx_coarse/ate.json 2>&1 | tee $OUT/vggtx_coarse.log
echo "=== DONE vggtx ba_coarse exit=${PIPESTATUS[0]} $(date -Is) ==="
echo "=== ALL RUNS COMPLETE $(date -Is) ==="
```

- [ ] **Step 3: Launch in tmux and monitor**

```bash
tmux new-session -d -s ba_conv 'bash <scratchpad>/ba_convergence_runs.sh 2>&1 | tee evals/results/ba_convergence_chess/driver.log'
```

- [ ] **Step 4: Confirm BA actually converged**

Run: `for f in evals/results/ba_convergence_chess/*.log; do echo "$f: $(grep -c 'LM step' $f)"; done`
Expected: **40 per BA condition**, not 1–2. A count of 1–2 means the `StopOnPlateau` abort is back
and every downstream number is invalid.

Also check whether any frame was dropped:

Run: `grep -h "frames dropped\|LM optimize" evals/results/ba_convergence_chess/*.log`
Expected: `LM optimize: 100/100 frames` and no "frames dropped" warnings, matching the previous
sweep. Any drop warning means the Task 3/4 path went live and its indices should be recorded.

- [ ] **Step 5: Record measured results**

Add a §Measured results section to
`docs/superpowers/specs/2026-08-19-ba-lm-convergence-design.md` with a per-backbone table of
ATE rmse/median/max, RPE-t, RPE-rot, AUC@5/15/30, and wall-clock, each against both the baseline
and the 1–2 step numbers in `2026-08-19-ba-track-quality-parity-design.md` §Measured results.
Note explicitly that these are the first measurements of the solver the **production** pipeline
runs. Prefer ATE/RPE over AUC@5 — chess/seq-01's 10–20 mm inter-frame baselines make the bearing
term ill-conditioned.

Then update the `ba-track-quality-parity` entry in `CLAUDE.md` with the new verdict and whether
`pointcloud.bundle_adjustment` should stay `false`.

- [ ] **Step 6: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-19-ba-lm-convergence-design.md
git add CLAUDE.md
git commit -m "docs(specs): BA LM convergence measured results"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §Design 1 — one LM loop, delete `capture_loss_history` | Task 2 |
| §Design 2 — dropped-frame Sim(3) carry | Tasks 3, 4 |
| §Design 2 — shared focal to all frames | Task 4 |
| §Design 2 — warn on drop | Task 4 |
| §Design 2 — move `umeyama_*` out of the gtsam-importing module | Task 1 |
| §Design 3 — `--tracks_cache_dir` | Task 5 |
| §Design 4 — `ba_coarse` | Task 6 |
| §Testing | Tasks 1-6 (per task) + Task 7 |
| §Validation | Task 8 |
| §Considered and deferred | no task, by design |

**Deviations from the spec, deliberate:**
- The spec says `_make_creator` "passes it into every `BundleAdjustmentConfig` it constructs";
  Task 5 does that through a local `_ba()` factory instead of repeating the kwarg eight times.
  Same behaviour, one place to change.
- The spec sketches `_camera_centers` as a possible new two-line helper. Task 3 uses the existing
  `invert_poses(extrinsics_to_homogeneous(...))[:, :3, 3]` instead — no new helper needed.

**Type consistency:** `_carry_dropped_frames(refined_extrinsics, extrinsics, active_frames) ->
(np.ndarray, float | None)` is defined in Task 3 and called with exactly that signature in Task 4.
`_ba(**kw) -> BundleAdjustmentConfig` is defined in Task 5 and used in Task 6.
`tracks_cache_dir: Path | None` is spelled identically in `_make_creator`, `_run_condition`, and
argparse.

**Known risk:** Task 4's two tests assert on `inspect.getsource` text rather than behaviour,
because the surrounding code path needs CUDA. They are guardrails against silent reversion, not
correctness proofs — the behavioural coverage for the gauge maths is in Task 3, and the live
check is Task 8 Step 4.
