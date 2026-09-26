# Consistency Phase 1 — Convention Bugs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every phase-1 convention bug in [the consistency spec](../specs/2026-09-26-consistency-design.md), each behind a test that fails on the fork point.

**Architecture:** One long-lived worktree `.worktrees/consistency` on branch `clean/consistency`. Each fix is the smallest hunk at the defect, so rebases onto `clean/r4-lc` and `clean/pointcloud-release` conflict locally. The one new shared helper is `geometry/transforms.rescale_intrinsics`; everything else is in place.

**Tech Stack:** numpy, pycolmap 4.0.4, zarr 3, pytest; MapAnything `c845b8f` (installed) for the crop-box reference.

---

## Conventions for every task

- Worktree: `/workspace/collab-splats/.worktrees/consistency`. Every command below starts with
  `cd /workspace/collab-splats/.worktrees/consistency &&` — the cwd resets between Bash calls.
- Run Python as `PYTHONPATH=. /opt/venv/reconstruction/bin/python`. The venv's editable finder
  points at the MAIN tree, so without `PYTHONPATH=.` pytest silently tests the wrong code.
- Every test gate begins with the proof line:
  `PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"`
  and must print a path under `.worktrees/consistency/`.
- Never pipe pytest into `tail`/`head` (eats the exit code). Use `-q` and read the summary line.
- No full-suite runs; the GPU is shared. Per-package gates only.
- Commit in the worktree with `git add <files> && git commit`. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Code style (CLAUDE.md): block comments per logical block; docstrings open on their own line,
  1-line summary, `- ` bullets, `Args:` / `Returns:` / `Raises:`; comment runs of 3+ lines are a
  header line then `- ` bullets. `tests/test_docstring_contract.py` enforces it for `geometry`
  and `pointcloud`.
- A test that passes on the unfixed code is rejected. Every "verify it fails" step is mandatory.

## Dropped from the spec during planning

- **Localizer ref-pixel lookup (`localizer.py:931-942`)**: that branch is the descriptor path,
  taken only by non-`LocalMatcher` extractors (test stubs). The only production extractor,
  `LocalMatcher`, goes through `_localize_pairwise` (`localizer.py:916-917`) against model-res
  refs on the `world_points` grid, so no rescale runs. Fixing it means threading
  `original_coords` into `CameraLocalizer` for a path no production extractor reaches. Not fixed.
- **Viewer FOV**: already dropped in the spec (correct as written).
- **`mesh/io.py:317` JPEG comment**: the call site already says why (`io.py:316`,
  "Textures are jpg: an 8192 atlas is ~180 MB as png and ~10 MB here"). No change.

## File map

| File | Change |
|---|---|
| `evals/trajectory_metrics.py` | `rpe` takes w2c, inverts, Sim3-scales |
| `evals/scripts/eval_verification.py` | K from `cam.calibration_matrix()` |
| `collab_splats/geometry/verification.py` | `clean_for_json` handles numpy, tuples, inf |
| `collab_splats/geometry/metrics.py` | strict JSON write; K via `rescale_intrinsics` |
| `collab_splats/geometry/transforms.py` | new `rescale_intrinsics` |
| `collab_splats/pointcloud/feedforward/base.py` | crop-aware `_rescale_reconstruction_to_original_dimensions` |
| `evals/scripts/eval_splats.py` | K via `rescale_intrinsics` |
| `collab_splats/pointcloud/feedforward/mapanything.py` | real crop box in original pixels |
| `collab_splats/dashboard/pipeline.py` | provenance from either schema; ref paths by stem; PNG query frames |
| `collab_splats/localization/localizer.py` | staleness compares stems |
| `collab_splats/wrapper/reconstructor.py` | localization ids `.png`; delete stale comment |
| `collab_splats/pointcloud/sfm/instantsfm.py` | fallback name `.png` |

---

### Task 0: Worktree and control gate

**Files:** none (setup only)

- [ ] **Step 1: Check for in-progress git operations and create the worktree**

```bash
cd /workspace/collab-splats && ls .git/sequencer 2>/dev/null; git status --short | head
git worktree add .worktrees/consistency -b clean/consistency clean/final
git -C .worktrees/consistency log --oneline -1
```

Expected: no `sequencer` listing; the worktree log line shows the current `clean/final` tip. Write that SHA
into the "Fork point" line at the bottom of this plan (commit it on `clean/consistency` in Task 12).

- [ ] **Step 2: Symlink gitignored third_party so guarded tests do not skip**

```bash
cd /workspace/collab-splats/.worktrees/consistency && mkdir -p third_party && for d in /workspace/collab-splats/third_party/*; do ln -sfn "$d" third_party/; done && ls third_party
```

- [ ] **Step 3: Record the control gate on the fork point**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && for p in evals geometry pointcloud localization dashboard wrapper; do echo "== $p"; PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/$p -q -p no:cacheprovider 2>&1 | grep -E "passed|failed|error" | tail -1; done | tee /tmp/claude-0/consistency-control.txt
```

Expected: one summary line per package. Save the failing node ids too:

```bash
cd /workspace/collab-splats/.worktrees/consistency && for p in evals geometry pointcloud localization dashboard wrapper; do PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/$p -q -rfE -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)"; done | sort > /tmp/claude-0/consistency-control-failures.txt; wc -l /tmp/claude-0/consistency-control-failures.txt
```

These are the baseline. Task 12 diffs against them; only a new failure blocks.

---

### Task 1: RPE frame convention

**Files:**
- Modify: `evals/trajectory_metrics.py:1-6` (module docstring), `:76-104` (`rpe`)
- Test: `tests/evals/test_trajectory_metrics.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/evals/test_trajectory_metrics.py`:

```python
def _random_c2w(n: int, seed: int) -> np.ndarray:
    """Non-identity camera-to-world poses: random rotations, spread-out centres."""
    rng = np.random.default_rng(seed)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, :3, :3] = R.random(n, random_state=seed).as_matrix()
    poses[:, :3, 3] = rng.normal(scale=3.0, size=(n, 3))
    return poses


def test_rpe_is_zero_for_a_sim3_copy_of_gt():
    """w2c input: a pred that is gt under a world Sim3 (scale 2.5, rotated, shifted) has no RPE."""
    gt_c2w = _random_c2w(8, seed=0)
    R_w = R.from_euler("xyz", [30, -50, 70], degrees=True).as_matrix()
    s, t_w = 2.5, np.array([4.0, -1.0, 2.0])
    pred_c2w = gt_c2w.copy()
    pred_c2w[:, :3, :3] = R_w @ gt_c2w[:, :3, :3]
    pred_c2w[:, :3, 3] = s * gt_c2w[:, :3, 3] @ R_w.T + t_w

    result = rpe(np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w), delta=1)

    assert result["trans_rmse"] < 1e-6
    assert result["rot_rmse_deg"] < 1e-4


def test_rpe_rotation_error_isolated_to_the_perturbed_frame():
    """One frame rotated by theta about its own x axis: pairs (k-1,k) and (k,k+1) each err theta."""
    n, k, theta = 8, 4, 5.0
    gt_c2w = _random_c2w(n, seed=1)
    pred_c2w = gt_c2w.copy()
    pred_c2w[k, :3, :3] = gt_c2w[k, :3, :3] @ R.from_euler("x", theta, degrees=True).as_matrix()

    result = rpe(np.linalg.inv(pred_c2w), np.linalg.inv(gt_c2w), delta=1)

    # Two of the n-1 pairs carry theta, the rest zero
    assert result["rot_rmse_deg"] == pytest.approx(theta * np.sqrt(2 / (n - 1)), rel=1e-6)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_trajectory_metrics.py -q -k "sim3_copy or perturbed_frame"
```

Expected: `test_rpe_is_zero_for_a_sim3_copy_of_gt` FAILS (trans_rmse far above 1e-6 — w2c translations
under a 2.5x scale). If both pass, stop: the fixture cannot see the bug; fix the fixture first.

- [ ] **Step 3: Implement**

In `evals/trajectory_metrics.py`, change the module docstring's last bullet to:

```python
- ATE and AUC align to ground truth first (Umeyama); RPE removes only the Sim3 scale, which is
  the one part of the alignment that survives in relative poses
```

Replace `rpe` (lines 76-104) with:

```python
def rpe(pred: np.ndarray, gt: np.ndarray, delta: int = 1) -> dict:
    """
    Relative Pose Error at frame stride delta, on world-to-camera input.

    - both trajectories are inverted to camera-to-world before relative poses are formed
    - pred centres are scaled by the Sim3 scale of `umeyama_sim3`; the alignment's rotation and
      translation cancel exactly in inv(A_i) @ A_j, so the scale is all that is applied
    - coincident centres give scale 1 (umeyama_sim3's own rule)

    Args:
        pred: (N, 4, 4) predicted world-to-camera poses.
        gt: (N, 4, 4) ground-truth world-to-camera poses.
        delta: frame stride between compared pose pairs.

    Returns:
        {'trans_rmse', 'rot_rmse_deg'}; trans_rmse is in gt units.

    Raises:
        ValueError: delta >= N, so no pose pairs exist.
    """
    if delta >= len(pred):
        raise ValueError(f"delta={delta} >= N={len(pred)}, no pose pairs available")

    # World-to-camera in, camera-to-world for the relative-pose formula
    pred_c2w = np.linalg.inv(pred.astype(np.float64))
    gt_c2w = np.linalg.inv(gt.astype(np.float64))

    # Sim3 scale from the camera centres; needs 3 centres, so 2-frame input keeps scale 1
    scale = 1.0
    if len(pred) >= 3:
        scale, _, _ = umeyama_sim3(source=pred_c2w[:, :3, 3], target=gt_c2w[:, :3, 3])
    pred_c2w[:, :3, 3] *= scale

    # Relative pose between frame i and i+delta for each trajectory, then the error transform
    rel_pred = np.linalg.inv(pred_c2w[:-delta]) @ pred_c2w[delta:]  # (N-δ, 4, 4)
    rel_gt = np.linalg.inv(gt_c2w[:-delta]) @ gt_c2w[delta:]  # (N-δ, 4, 4)
    err = np.linalg.inv(rel_gt) @ rel_pred  # (N-δ, 4, 4)

    # Translation error is the error transform's norm; rotation error its geodesic angle
    t_err = np.linalg.norm(err[:, :3, 3], axis=1)
    cos_angle = np.clip((np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    r_err_deg = np.degrees(np.arccos(cos_angle))

    return {
        "trans_rmse": float(np.sqrt((t_err**2).mean())),
        "rot_rmse_deg": float(np.sqrt((r_err_deg**2).mean())),
    }
```

The `len(pred) >= 3` guard exists because `umeyama_sim3` raises below 3 points, and
`tests/evals/test_pose_graph_diagnostics.py:186` calls `rpe` with 2 poses.

- [ ] **Step 4: Run the RPE tests and the existing callers' tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_trajectory_metrics.py tests/evals/test_pose_graph_diagnostics.py -q
```

Expected: all pass, including the pre-existing `test_rpe_perfect`, `test_rpe_returns_correct_keys`,
`test_rpe_returns_error_dict`.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add evals/trajectory_metrics.py tests/evals/test_trajectory_metrics.py && git commit -m "fix(evals): rpe inverts w2c input and removes the Sim3 scale

eval.py passes world-to-camera poses; the c2w relative-pose formula on
w2c input reported wrong-frame errors in model scale.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `cam.params` unpacking in eval_verification

**Files:**
- Modify: `evals/scripts/eval_verification.py:62-63`
- Create: `tests/evals/test_eval_verification.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evals/test_eval_verification.py`:

```python
import numpy as np

from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction
from evals.scripts.eval_verification import _recon_to_arrays


def _recon(camera_model: str):
    """Two-frame poses-only reconstruction; K has fx == fy so both models can hold it."""
    K = np.array([[500.0, 0, 330.0], [0, 500.0, 250.0], [0, 0, 1]], dtype=np.float32)
    extrinsics = np.tile(np.eye(3, 4, dtype=np.float32), (2, 1, 1))
    extrinsics[1, 0, 3] = 0.5
    return build_pycolmap_reconstruction(
        pts3d=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.stack([K, K]),
        image_width=640,
        image_height=480,
        image_names=["frame_000000", "frame_000001"],
        camera_model=camera_model,
    )


def test_recon_to_arrays_reads_k_for_any_pinhole_model():
    """SIMPLE_PINHOLE has 3 params; unpacking 4 raised ValueError. Both models give one K."""
    _, K_pinhole, _, _, _ = _recon_to_arrays(_recon("PINHOLE"))
    _, K_simple, _, _, _ = _recon_to_arrays(_recon("SIMPLE_PINHOLE"))

    np.testing.assert_allclose(K_simple, K_pinhole)
    np.testing.assert_allclose(K_pinhole[0], [[500, 0, 330], [0, 500, 250], [0, 0, 1]])
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_verification.py -q
```

Expected: FAIL with `ValueError: not enough values to unpack (expected 4, got 3)`.
If it fails on import instead (`vismatch` missing), the import chain changed; check
`python -c "import evals.scripts.eval_verification"` and fix the import, not the assertion.

- [ ] **Step 3: Implement**

In `_recon_to_arrays`, replace

```python
        fx, fy, cx, cy = cam.params  # PINHOLE
        intr.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))
```

with

```python
        intr.append(np.asarray(cam.calibration_matrix(), dtype=np.float32))  # any camera model
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_verification.py -q
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add evals/scripts/eval_verification.py tests/evals/test_eval_verification.py && git commit -m "fix(evals): eval_verification reads K via calibration_matrix

Unpacking four params raised on SIMPLE_PINHOLE and misassigned longer
param vectors.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: JSON NaN leak

**Files:**
- Modify: `collab_splats/geometry/verification.py:432-448` (`clean_for_json`)
- Modify: `collab_splats/geometry/metrics.py:696-699` (report write)
- Test: `tests/geometry/test_verification.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_verification.py` (add `from collab_splats.geometry.verification import clean_for_json`
to the existing verification import line):

```python
def test_clean_for_json_makes_numpy_and_nonfinite_payloads_strict_json():
    """float32 NaN, inf, NaN in a tuple and in an ndarray all become null under allow_nan=False."""
    payload = {
        "f32": np.float32("nan"),
        "f64": np.float64("nan"),
        "inf": float("inf"),
        "neg_inf": np.float32("-inf"),
        "tup": (float("nan"), 1.0),
        "arr": np.array([np.nan, 2.0], dtype=np.float32),
        "int": np.int64(3),
        "ok": np.float32(0.5),
    }

    text = json.dumps(clean_for_json(payload), allow_nan=False)

    assert json.loads(text) == {
        "f32": None,
        "f64": None,
        "inf": None,
        "neg_inf": None,
        "tup": [None, 1.0],
        "arr": [None, 2.0],
        "int": 3,
        "ok": 0.5,
    }
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py -q -k strict_json
```

Expected: FAIL with `TypeError: Object of type float32 is not JSON serializable` (or
`ValueError: Out of range float values`).

- [ ] **Step 3: Implement `clean_for_json`**

Replace the function in `collab_splats/geometry/verification.py`:

```python
def clean_for_json(obj: object) -> object:
    """
    Recursively convert a payload to strict-JSON-safe builtins.

    - numpy scalars become Python scalars; ndarrays and tuples become lists
    - NaN and ±inf become None, so json.dumps(..., allow_nan=False) succeeds

    Args:
        obj: nested dicts, lists, tuples, ndarrays and scalars.

    Returns:
        The same structure built from dict, list, str, int, float, bool and None only.
    """
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    return obj
```

- [ ] **Step 4: Make the metrics write strict**

In `collab_splats/geometry/metrics.py` replace

```python
    # clean_for_json turns every nan into null. json.dumps otherwise writes a bare NaN, which no
    # strict JSON parser accepts; default= handles numpy scalars.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), indent=2, default=lambda o: o.item()))
```

with

```python
    # Strict JSON: clean_for_json nulls every non-finite value and converts numpy types
    # - allow_nan=False makes any future leak raise here instead of writing a bare NaN
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), indent=2, allow_nan=False))
```

- [ ] **Step 5: Run the geometry JSON tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_verification.py tests/geometry/test_metrics.py -q
```

Expected: all pass. A failure in a `test_metrics.py` end-to-end report test means the report holds
a type `clean_for_json` still misses (e.g. a `Path`); add that type to `clean_for_json` with a
case in the Step 1 test, never restore `default=`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/geometry/verification.py collab_splats/geometry/metrics.py tests/geometry/test_verification.py && git commit -m "fix(geometry): clean_for_json nulls numpy NaN and inf

A float32 NaN passed clean_for_json untouched and default=.item() wrote
it as a bare NaN. The metrics report now writes with allow_nan=False.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Rebase note for `clean/r4-lc`: it moves `clean_for_json` to `geometry/transforms.py:467` and the
report write to `reconstructor.py:1657`. On conflict, apply this body there and drop `default=`
at both write sites.

---

### Task 4: `rescale_intrinsics` in transforms; metrics adopts it

**Files:**
- Modify: `collab_splats/geometry/transforms.py` (new section after "Intrinsics estimation")
- Modify: `collab_splats/geometry/metrics.py:239-266` (delete `_scale_intrinsics_to_original`), `:342-347` (call site)
- Test: `tests/geometry/test_transforms.py` (append); `tests/geometry/test_metrics.py:17,1240-1244` (port)

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_transforms.py` (add `rescale_intrinsics` to its import block):

```python
def _project(K: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """Pinhole projection of camera-frame points to pixels."""
    uv = (K @ (pts_cam / pts_cam[:, 2:3]).T).T
    return uv[:, :2]


def test_rescale_intrinsics_maps_projections_through_the_crop():
    """Portrait frame, crop with nonzero top-left, off-centre principal point, non-square pixels."""
    orig_K = np.array([[900.0, 0, 470.0], [0, 880.0, 1010.0], [0, 0, 1]])
    box = np.array([0.0, 420.0, 1080.0, 1500.0, 1080.0, 1920.0])  # 1080x1080 crop, tl_y = 420
    model_hw = (518, 518)
    pts = np.array([[0.1, -0.2, 3.0], [-0.4, 0.3, 5.0], [0.0, 0.0, 2.0]])

    model_K = rescale_intrinsics(orig_K, box, model_hw, to_original=False)

    # Model pixel -> original pixel is undo-scale then add the crop origin
    s = np.array([518 / 1080, 518 / 1080])
    np.testing.assert_allclose(_project(model_K, pts) / s + box[:2], _project(orig_K, pts), atol=1e-9)


def test_rescale_intrinsics_round_trips_and_broadcasts():
    """to_original after to_model is identity on an (N, 3, 3) stack with per-frame boxes."""
    K = np.array([[[400.0, 0, 250.0], [0, 410.0, 140.0], [0, 0, 1]]] * 2)
    boxes = np.array([[11.0, 7.0, 59.0, 47.0, 64.0, 48.0], [0.0, 0.0, 64.0, 48.0, 64.0, 48.0]])

    there = rescale_intrinsics(K, boxes, (8, 12), to_original=False)
    back = rescale_intrinsics(there, boxes, (8, 12), to_original=True)

    np.testing.assert_allclose(back, K)
    assert not np.allclose(there[0], there[1])  # the two boxes really differ


@pytest.mark.parametrize("box", [[0, 0, 10, 10, 10], [5, 0, 5, 10, 10, 10], [0, 8, 10, 3, 10, 10]])
def test_rescale_intrinsics_rejects_bad_boxes(box):
    """Wrong length, zero width, negative height."""
    with pytest.raises(ValueError):
        rescale_intrinsics(np.eye(3), np.array(box, dtype=float), (4, 4), to_original=True)
```

In `tests/geometry/test_metrics.py`, remove `_scale_intrinsics_to_original,` from the import at line 17
and replace `test_scale_intrinsics_to_original_known_answer` (lines 1240-1244) with:

```python
def test_rescale_intrinsics_known_answer():
    """Crop (11,7)-(59,47) resized to 12x8: sx=0.25, sy=0.2."""
    K = np.array([[10.0, 0, 6.0], [0, 10.0, 4.0], [0, 0, 1]])
    box = np.array([11.0, 7.0, 59.0, 47.0, 64.0, 48.0])
    out = rescale_intrinsics(K, box, (8, 12), to_original=True)
    assert np.allclose(out, [[40.0, 0, 35.0], [0, 50.0, 27.0], [0, 0, 1]])
```

and add `from collab_splats.geometry.transforms import rescale_intrinsics` to its imports.

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py -q -k rescale_intrinsics
```

Expected: collection error `ImportError: cannot import name 'rescale_intrinsics'`.

- [ ] **Step 3: Implement `rescale_intrinsics`**

In `collab_splats/geometry/transforms.py`, insert before the "Point-set alignment" divider (line 245):

```python
########################################################################
########## Crop-aware intrinsics #######################################
########################################################################


def rescale_intrinsics(
    intrinsics: np.ndarray,
    crop_box: np.ndarray,
    model_hw: tuple[int, int],
    *,
    to_original: bool,
) -> np.ndarray:
    """
    Map K between the model grid and original-image pixels through a crop-then-resize.

    - crop box is an `original_coords` row [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], original pixels
    - per-axis scale is model / crop; the crop origin is added after the scale is undone
    - leading dims broadcast: K (..., 3, 3) pairs with crop_box (..., 6)

    Args:
        intrinsics: (..., 3, 3) K on the source grid.
        crop_box: (..., 6) crop rows; the first four values are read.
        model_hw: model grid as (height, width).
        to_original: True maps model -> original pixels, False original -> model.

    Returns:
        (..., 3, 3) float64 K on the target grid.

    Raises:
        ValueError: a crop row that is not 6 values, or has zero or negative width or height.
    """
    # Validate the crop rows before any division
    box = np.asarray(crop_box, dtype=np.float64)
    if box.shape[-1] != 6:
        raise ValueError(f"crop box needs 6 values [tl_x, tl_y, cr_x, cr_y, W, H], got shape {box.shape}")
    tl_x, tl_y, cr_x, cr_y = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    if np.any(cr_x <= tl_x) or np.any(cr_y <= tl_y):
        raise ValueError(f"crop box has zero or negative size: {box.tolist()}")

    # Per-axis model / crop scale, broadcast over leading dims
    model_h, model_w = model_hw
    sx = model_w / (cr_x - tl_x)
    sy = model_h / (cr_y - tl_y)

    # Undo scale then add origin (to original), or subtract origin then scale (to model)
    out = np.array(intrinsics, dtype=np.float64)
    if to_original:
        out[..., 0, 0] = out[..., 0, 0] / sx
        out[..., 1, 1] = out[..., 1, 1] / sy
        out[..., 0, 2] = out[..., 0, 2] / sx + tl_x
        out[..., 1, 2] = out[..., 1, 2] / sy + tl_y
    else:
        out[..., 0, 0] = out[..., 0, 0] * sx
        out[..., 1, 1] = out[..., 1, 1] * sy
        out[..., 0, 2] = (out[..., 0, 2] - tl_x) * sx
        out[..., 1, 2] = (out[..., 1, 2] - tl_y) * sy
    return out
```

- [ ] **Step 4: Adopt it in metrics.py**

Delete `_scale_intrinsics_to_original` (metrics.py lines 239-266, the whole function) and add
`from collab_splats.geometry.transforms import rescale_intrinsics` to the imports. Replace the loop

```python
        # Undo crop-then-resize on K: scale is model/crop, not model/canvas
        lifted_K = []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = lifted_d, np.stack(lifted_K)
```

with

```python
        # Undo crop-then-resize on K: scale is model/crop, not model/canvas
        lifted_K = rescale_intrinsics(intrinsics, original_coords, (model_h, model_w), to_original=True)
        depth, intrinsics = lifted_d, lifted_K
```

- [ ] **Step 5: Run geometry tests and the docstring contract**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py tests/geometry/test_metrics.py tests/test_docstring_contract.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/geometry/transforms.py collab_splats/geometry/metrics.py tests/geometry/test_transforms.py tests/geometry/test_metrics.py && git commit -m "feat(geometry): rescale_intrinsics maps K through the crop box

One crop-aware K mapping in both directions; metrics.py's private
_scale_intrinsics_to_original is replaced by it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Crop-aware `_rescale_reconstruction_to_original_dimensions`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:878-966`
- Test: `tests/pointcloud/test_feedforward_intrinsics.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_feedforward_intrinsics.py`:

```python
def _rescale_one(camera_model, params, box, model_wh, points2d=()):
    """Run the real rescale on one duck-typed camera; returns (params, width, height, points2D)."""
    camera = SimpleNamespace(
        model=SimpleNamespace(name=camera_model),
        params=np.array(params, dtype=np.float64),
        width=model_wh[0],
        height=model_wh[1],
    )
    pts = [SimpleNamespace(xy=np.array(xy, dtype=np.float64)) for xy in points2d]
    reconstruction = SimpleNamespace(
        images={1: SimpleNamespace(camera_id=1, name="0.png", points2D=pts)},
        cameras={1: camera},
    )
    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png")],
        np.array([box], dtype=np.float32),
        model_wh,
        shift_point2d_to_original_res=bool(points2d),
    )
    return camera.params, camera.width, camera.height, [p.xy for p in pts]


def test_rescale_maps_a_cropped_portrait_camera_to_original_pixels():
    """VGGT-X portrait: 1080x1920 centre-cropped to 1080x1080 (tl_y = 420), model 518x518."""
    box = [0, 420, 1080, 1500, 1080, 1920]
    s = 518 / 1080
    model_params = [900 * s, 880 * s, 470 * s, (1010 - 420) * s]  # fx, fy, cx, cy on the model grid

    params, width, height, _ = _rescale_one("PINHOLE", model_params, box, (518, 518))

    np.testing.assert_allclose(params, [900, 880, 470, 1010], rtol=1e-5)
    assert (width, height) == (1080, 1920)


def test_rescale_shifts_point2d_into_original_pixels():
    """A model-grid observation lands at xy / s + tl in the original frame."""
    box = [0, 420, 1080, 1500, 1080, 1920]
    s = 518 / 1080

    _, _, _, (xy,) = _rescale_one("PINHOLE", [400, 400, 259, 259], box, (518, 518), points2d=[(100.0, 50.0)])

    np.testing.assert_allclose(xy, [100 / s, 50 / s + 420], rtol=1e-5)


def test_rescale_keeps_distortion_params_for_opencv():
    """OPENCV params are fx, fy, cx, cy, k1, k2, p1, p2 — the tail is distortion, not cx/cy."""
    box = [0, 0, 1036, 1036, 1036, 1036]  # full frame, 2x
    params, _, _, _ = _rescale_one("OPENCV", [100, 100, 259, 259, 0.1, 0.2, 0.01, 0.02], box, (518, 518))

    np.testing.assert_allclose(params, [200, 200, 518, 518, 0.1, 0.2, 0.01, 0.02], rtol=1e-6)


def test_rescale_param_index_table_matches_pycolmap():
    """The model -> (focal, principal point) index table mirrors pycolmap's own."""
    import pycolmap

    from collab_splats.pointcloud.feedforward.base import _CAMERA_PARAM_IDXS

    for model, (focal, pp) in _CAMERA_PARAM_IDXS.items():
        cam = pycolmap.Camera(model=model, width=10, height=10)
        assert (list(cam.focal_length_idxs()), list(cam.principal_point_idxs())) == (focal, pp), model
```

(`pycolmap` is imported inside the last test only because the rest of the file is pycolmap-free
by design — see the `_rescaled_camera_params` docstring. If the file already imports pycolmap at
the top by the time this runs, move the import up.)

- [ ] **Step 2: Run to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py -q -k "cropped_portrait or point2d or distortion or index_table"
```

Expected: `cropped_portrait` FAILS (fx scaled by 1080/518 and 1920/518 per axis, no +420), `point2d`
FAILS (`(xy - tl) * scale`), `distortion` FAILS (p1, p2 multiplied by 2), `index_table` FAILS on
import (`_CAMERA_PARAM_IDXS` missing).

- [ ] **Step 3: Implement**

In `collab_splats/pointcloud/feedforward/base.py`, add `from collab_splats.geometry.transforms import rescale_intrinsics`
to the imports (check the existing `collab_splats.geometry` import line first and extend it). Add just
above `_rescale_reconstruction_to_original_dimensions`:

```python
# pycolmap param positions per camera model: (focal idxs, principal-point idxs)
# - mirrors Camera.focal_length_idxs() / principal_point_idxs(); a test pins the match
# - a table, not the pycolmap calls, because the rescale is duck-typed over the camera
_CAMERA_PARAM_IDXS: dict[str, tuple[list[int], list[int]]] = {
    "SIMPLE_PINHOLE": ([0], [1, 2]),
    "PINHOLE": ([0, 1], [2, 3]),
    "SIMPLE_RADIAL": ([0], [1, 2]),
    "RADIAL": ([0], [1, 2]),
    "OPENCV": ([0, 1], [2, 3]),
    "OPENCV_FISHEYE": ([0, 1], [2, 3]),
}
```

Replace the function body from `# Initialize per-loop shared-camera bookkeeping` through the end
of the point2D block with:

```python
    # Initialize per-loop shared-camera bookkeeping (used only when shared_camera=True)
    rescale_camera = True
    shared_intrinsics = None
    shared_width = None
    shared_height = None
    model_hw = (image_size[1], image_size[0])

    # Rescale intrinsics and image dimensions for each frame
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]

        pyimage.name = image_paths[pyimageid - 1].name

        box = original_image_sizes[pyimageid - 1]
        real_image_size = box[-2:]

        if rescale_camera and (not shared_camera or shared_intrinsics is None):
            # Model-grid K through the crop box to original pixels
            focal_idxs, pp_idxs = _CAMERA_PARAM_IDXS[pycamera.model.name]
            pred_params = copy.deepcopy(pycamera.params)
            K = np.eye(3)
            K[0, 0], K[1, 1] = pred_params[focal_idxs[0]], pred_params[focal_idxs[-1]]
            K[0, 2], K[1, 2] = pred_params[pp_idxs[0]], pred_params[pp_idxs[1]]
            K = rescale_intrinsics(K, box, model_hw, to_original=True)

            # Write K back; single-focal models keep the larger axis focal, as before
            if len(focal_idxs) == 1:
                pred_params[focal_idxs[0]] = max(K[0, 0], K[1, 1])
            else:
                pred_params[focal_idxs[0]], pred_params[focal_idxs[1]] = K[0, 0], K[1, 1]
            pred_params[pp_idxs[0]], pred_params[pp_idxs[1]] = K[0, 2], K[1, 2]

            if shared_camera:
                shared_intrinsics = pred_params
                shared_width = int(real_image_size[0])
                shared_height = int(real_image_size[1])

                pycamera.params = shared_intrinsics
                pycamera.width = shared_width
                pycamera.height = shared_height
            else:
                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])

        # Propagate shared intrinsics to subsequent frames when shared_camera=True
        if shared_camera and shared_intrinsics is not None:
            pycamera.params = shared_intrinsics
            pycamera.width = shared_width
            pycamera.height = shared_height

        # Shift Point2D observations: undo the model/crop scale, then add the crop origin
        if shift_point2d_to_original_res:
            tl = box[:2].astype(np.float64)
            scale = np.array([image_size[0] / (box[2] - box[0]), image_size[1] / (box[3] - box[1])])
            for point2D in pyimage.points2D:
                point2D.xy = point2D.xy / scale + tl
```

Also update the docstring's `original_image_sizes` entry to say
`(N, 6) [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h], crop in original pixels.`

Note the SIMPLE_PINHOLE rule: the old code multiplied f by `max(orig_w/model_w, orig_h/model_h)`;
`max(K[0,0], K[1,1])` after the crop-aware map is the same rule on a full-frame box.

- [ ] **Step 4: Run the whole intrinsics file, including the existing full-frame tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_feedforward_intrinsics.py tests/test_docstring_contract.py -q
```

Expected: all pass, the LoGeR round-trip tests unchanged.

- [ ] **Step 5: Run the pointcloud and wrapper gates (refine path calls this at `reconstructor.py:1195`)**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud tests/wrapper -q -rfE -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" 
```

Expected: the failure list is a subset of `/tmp/claude-0/consistency-control-failures.txt`.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_intrinsics.py && git commit -m "fix(pointcloud): crop-aware rescale of the COLMAP export to original pixels

Scale was orig/model with no crop origin, point2D subtracted an
original-pixel origin before scaling, and OPENCV/RADIAL distortion
params were scaled as if they were cx, cy.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: eval_splats K via `rescale_intrinsics`

**Files:**
- Modify: `evals/scripts/eval_splats.py:73-107` (`_native_images_and_intrinsics`)
- Test: `tests/evals/test_eval_splats.py` (append)

- [ ] **Step 1: Fix the fixture's box, then write the failing test**

`_write_ff_zarr` (test_eval_splats.py:30) records `[0, 0, w, h, orig_w, orig_h]`: a model-sized crop
at the top-left, which under the crop convention means scale 1. The existing native test
(`test_inputs_native_from_images_dir`, 2x frames, expects 2x K) only passes because the old code
ignored `[:4]`. Make the fixture say what it means — full frame, resized:

```python
        ("original_coords", np.tile(np.array([0, 0, orig_w, orig_h, orig_w, orig_h], np.float32), (n, 1))),
```

Then append:

```python
def test_native_intrinsics_undo_the_crop(tmp_path):
    """Cropped box: native K is K_model / s + tl, not K_model * orig / model."""
    model_w = model_h = 16
    orig_w, orig_h = 32, 64
    box = np.array([[0, 16, 32, 48, orig_w, orig_h]] * 2, dtype=np.float32)  # 32x32 crop, tl_y = 16
    K_model = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1]], dtype=np.float32)
    _write_images_dir(tmp_path / "images", 2, orig_h, orig_w)
    result = SimpleNamespace(
        image_paths=[Path("frame_000000.png"), Path("frame_000001.png")],
        original_coords=box,
        model_width=model_w,
        model_height=model_h,
        intrinsics=np.stack([K_model, K_model]),
    )

    _, K_native = _native_images_and_intrinsics(result, tmp_path / "images", tmp_path / "pointcloud.zarr")

    np.testing.assert_allclose(K_native[0], [[40, 0, 16], [0, 40, 32], [0, 0, 1]], rtol=1e-6)
```

Add to the imports: `from pathlib import Path`, `from types import SimpleNamespace`, and
`_native_images_and_intrinsics` on the existing `evals.scripts.eval_splats` import line.

- [ ] **Step 2: Run to verify it fails, and that the fixture edit alone breaks nothing**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_splats.py -q
```

Expected: only `test_native_intrinsics_undo_the_crop` FAILS — old code gives fy = 20 * 64/16 = 80
(expected 40). Every pre-existing test still passes (old code ignores `[:4]`).

- [ ] **Step 3: Implement**

In `evals/scripts/eval_splats.py`, add `from collab_splats.geometry.transforms import rescale_intrinsics`
to the imports. Replace

```python
    # Per-frame anisotropic rescale of fx, cx (x) and fy, cy (y) from model res to native
    scale_x = (orig_w / result.model_width).astype(np.float32)
    scale_y = (orig_h / result.model_height).astype(np.float32)
    intrinsics = result.intrinsics.astype(np.float32).copy()
    intrinsics[:, 0, 0] *= scale_x
    intrinsics[:, 0, 2] *= scale_x
    intrinsics[:, 1, 1] *= scale_y
    intrinsics[:, 1, 2] *= scale_y
    return images, intrinsics
```

with

```python
    # Model-res K through each frame's crop box to native pixels
    intrinsics = rescale_intrinsics(
        result.intrinsics, result.original_coords, (result.model_height, result.model_width), to_original=True
    ).astype(np.float32)
    return images, intrinsics
```

and change the docstring's first bullet to:

```python
    - Mirrors ``Reconstructor.splats()`` for the frames and ``rescale_intrinsics`` for K: each
      frame's crop box from ``original_coords`` maps model-res K to native pixels.
```

- [ ] **Step 4: Run the eval_splats tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/evals/test_eval_splats.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add evals/scripts/eval_splats.py tests/evals/test_eval_splats.py && git commit -m "fix(evals): eval_splats maps K through the crop box

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: MapAnything crop box in original pixels

`mapanything.py:226` writes `[0, 0, model_w, model_h, W, H]`. Consumers read `[:4]` as the crop in
original pixels: the mesh arm (`reconstructor.py:663`, `upsample_depths`) places model-res depth into
a model-sized box at the frame's top-left, and `metrics.py` derives scale 1. Upstream
`crop_resize_if_necessary` (facebookresearch/map-anything @ c845b8f, `mapanything/utils/cropping.py:231-240`
rescale, `:443-447` centred crop) scales by `max(target / size) + 1e-8`, floors, then centre-crops.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py:224-228`
- Test: `tests/pointcloud/feedforward/test_mapanything_creator.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.parametrize("hw", [(1000, 1000), (1080, 1920), (1920, 1080), (200, 300)])
def test_mapanything_crop_coords_match_upstream_crop(hw):
    """The box, cropped and resized with PIL, reproduces upstream's model-res image."""
    cropping = pytest.importorskip("mapanything.utils.cropping")
    from PIL import Image

    from collab_splats.pointcloud.feedforward.mapanything import _mapanything_crop_coords

    h, w = hw
    model_w, model_h = 518, 294
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) % 256)], axis=-1).astype(np.uint8)

    upstream = np.asarray(cropping.crop_resize_if_necessary(rgb, resolution=(model_w, model_h))[0], dtype=np.float32)
    (box,) = _mapanything_crop_coords([(h, w)], model_w, model_h)
    ours = np.asarray(
        Image.fromarray(rgb).crop(tuple(float(v) for v in box[:4])).resize((model_w, model_h), Image.LANCZOS),
        dtype=np.float32,
    )

    assert box[4:].tolist() == [w, h]
    assert np.abs(ours[..., :2] - upstream[..., :2]).mean() < 2.0
```

(Only the two monotone gradient channels are compared; the third wraps and aliases under resampling.
`(200, 300)` is smaller than the target: upstream upscales it (`force=True`), so the box must too.)

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_mapanything_creator.py -q -k match_upstream_crop
```

Expected: FAIL on import (`_mapanything_crop_coords` missing). Before implementing, sanity-check the
test can see the bug: temporarily define in the test `box = np.array([0, 0, model_w, model_h, w, h])`
instead of the helper call, run, and confirm the mean diff assertion fails for all three sizes. Revert.

- [ ] **Step 3: Implement**

In `mapanything.py`, add a module-level helper (above the creator class):

```python
def _mapanything_crop_coords(frame_hw: list[tuple[int, int]], model_w: int, model_h: int) -> np.ndarray:
    """
    Crop box of MapAnything's loader per frame, in original pixels.

    - upstream: facebookresearch/map-anything @ c845b8f, mapanything/utils/cropping.py:231-240
      (scale = max(target / size) + 1e-8, floored resize) and :440-446 (centred crop)
    - force=True upstream (cropping.py:193): smaller frames are upscaled, never left as-is

    Args:
        frame_hw: (height, width) of each original frame.
        model_w: model grid width.
        model_h: model grid height.

    Returns:
        (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    rows = []
    for h, w in frame_hw:
        # Resize so the image covers the target, as upstream does
        scale = max(model_w / w, model_h / h) + 1e-8
        rw, rh = int(np.floor(w * scale)), int(np.floor(h * scale))

        # Centred crop on the resized grid, mapped back to original pixels
        left, top = (rw - model_w) // 2, (rh - model_h) // 2
        rows.append([left / scale, top / scale, (left + model_w) / scale, (top + model_h) / scale, w, h])
    return np.array(rows, dtype=np.float32)
```

Replace the `original_coords = np.array(...)` block at lines 224-228 with:

```python
        # Crop box in original pixels, reproducing load_images' resize-then-centre-crop
        original_coords = _mapanything_crop_coords([f.shape[:2] for f in frames], model_w, model_h)
```

- [ ] **Step 4: Run the MapAnything tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/feedforward/test_mapanything_creator.py tests/test_docstring_contract.py -q
```

Expected: all pass (existing tests set `original_coords` to zeros and do not reach `_preprocess`).

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/feedforward/test_mapanything_creator.py && git commit -m "fix(pointcloud): MapAnything original_coords records its real crop

The box held the model size as the crop corner, so the mesh stage lifted
MapAnything depth into a model-sized patch at the frame's top-left and
metrics derived a scale of 1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Existing MapAnything `pointcloud.zarr` stores keep the old box; the fix applies to new runs. Say
so in the final report — any MapAnything mesh built before this commit is suspect.

---

### Task 8: Localization DB provenance from either config schema

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py:499-518` (`_stamp_db_provenance`)
- Test: `tests/dashboard/test_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/dashboard/test_pipeline.py` (add imports `yaml`, `zarr`, and
`from collab_splats.preproc import frames as fr` if missing):

```python
def test_stamp_db_provenance_reads_a_reconstructor_run_config(tmp_path):
    """batch.py writes run_config.yaml as the Reconstructor config; its backbone must be stamped."""
    (tmp_path / "run_config.yaml").write_text(
        yaml.safe_dump({"input_path": "/videos/GH010229.MP4", "pointcloud": {"method": "feedforward", "backend": "mapanything"}})
    )
    fr.write_frames(tmp_path / "images", np.zeros((2, 4, 4, 3), dtype=np.uint8), [{"frame_idx": 7}, {"frame_idx": 19}], {})
    zarr_path = tmp_path / "pointcloud.zarr"
    zarr.open(str(zarr_path), mode="w")

    pipeline._stamp_db_provenance(zarr_path, "loma", tmp_path)

    attrs = dict(zarr.open(str(zarr_path), mode="r")["local_features/loma"].attrs)
    assert attrs["backbone"] == "mapanything"
    assert attrs["video_ref"] == "/videos/GH010229.MP4"
    assert attrs["frame_indices"] == [7, 19]


def test_stamp_db_provenance_still_reads_a_dashboard_run_config(tmp_path):
    """The dashboard's own RunConfig schema keeps working."""
    RunConfig(env_model="vggtx", frame_indices=[1, 2], video_ref="gs://v.mp4").to_yaml(tmp_path / "run_config.yaml")
    zarr_path = tmp_path / "pointcloud.zarr"
    zarr.open(str(zarr_path), mode="w")

    pipeline._stamp_db_provenance(zarr_path, "loma", tmp_path)

    attrs = dict(zarr.open(str(zarr_path), mode="r")["local_features/loma"].attrs)
    assert (attrs["backbone"], attrs["frame_indices"], attrs["video_ref"]) == ("vggtx", [1, 2], "gs://v.mp4")
```

Match the module alias the file already uses for `collab_splats.dashboard.pipeline`.

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py -q -k stamp_db_provenance
```

Expected: the Reconstructor test FAILS with `'vggt_omega' == 'mapanything'`; the dashboard test passes.

- [ ] **Step 3: Implement**

Replace the body of `_stamp_db_provenance` down to the `store = zarr.open(...)` line with:

```python
    cfg_path = Path(out_dir) / "run_config.yaml"
    attrs: dict = {"extractor": extractor_name}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}

        # Two writers share this file name
        # - batch.py: the Reconstructor config, nested under pointcloud:
        # - the dashboard: a flat RunConfig
        if isinstance(data.get("pointcloud"), dict):
            images_dir = Path(out_dir) / "images"
            frame_paths = fr.frame_paths(images_dir) if images_dir.is_dir() else []
            attrs.update(
                {
                    "backbone": data["pointcloud"]["backend"],
                    "frame_indices": [fr.frame_idx_from_path(p) for p in frame_paths],
                    "video_ref": str(data.get("input_path", "")),
                }
            )
        else:
            run_cfg = RunConfig.from_yaml(cfg_path)
            attrs.update(
                {
                    "backbone": run_cfg.env_model,
                    "frame_indices": list(run_cfg.frame_indices),
                    "video_ref": run_cfg.video_ref,
                }
            )
```

(`yaml` and `fr` are already imported at `pipeline.py:15,31`.)

- [ ] **Step 4: Run the dashboard provenance tests and the config test**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_pipeline.py tests/dashboard/test_config.py tests/dashboard/test_run_localization.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/dashboard/pipeline.py tests/dashboard/test_pipeline.py && git commit -m "fix(dashboard): stamp DB provenance from a Reconstructor run_config

batch.py writes run_config.yaml in the Reconstructor schema; the
dashboard's RunConfig dropped every key, so processed scenes were stamped
with default backbone, frame_indices and video_ref.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Staleness check by stem; localization ids `.png`

**Files:**
- Modify: `collab_splats/localization/localizer.py:829-834`
- Modify: `collab_splats/wrapper/reconstructor.py:770-777`
- Test: `tests/localization/test_localization_cache.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/localization/test_localization_cache.py`:

```python
def test_cache_staleness_ignores_the_id_extension(tmp_path, caplog):
    """A DB built with frame_NNN.jpg ids is not stale for frame_NNN.png ids; other stems are."""
    pts3d, world_points, extrinsics, intrinsics = _make_scene()
    image_paths = _make_image_files(tmp_path / "imgs", n=3)  # frame_000.jpg ...
    localizer, _ = _build_localizer_with_mock(world_points, extrinsics, image_paths)
    zarr_path = _empty_zarr(tmp_path)
    localizer.save_index(zarr_path, "disk")
    result = _make_ff_result(world_points, extrinsics, image_paths)

    def stale_warned(ids):
        caplog.clear()
        with caplog.at_level("WARNING", logger="collab_splats.localization.localizer"):
            CameraLocalizer.from_feedforward(
                result, ids=ids, extractor=MagicMock(), zarr_path=zarr_path, extractor_name="disk"
            )
        return "stale" in caplog.text

    assert not stale_warned([f"frame_{i:03d}.png" for i in range(3)])
    assert stale_warned([f"frame_{i:03d}.png" for i in (0, 1, 5)])
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/localization/test_localization_cache.py -q -k ignores_the_id_extension
```

Expected: FAIL on the first assertion (whole-string compare warns for `.jpg` vs `.png`).

- [ ] **Step 3: Implement the stem compare**

In `localizer.py`, replace

```python
                    # Staleness check: compare cached labels against ids (else result.image_paths)
                    cached_paths = [str(p) for p in store[rec_key].attrs["image_paths"]]
                    expected = [str(x) for x in (ids if ids is not None else result.image_paths)]
```

with

```python
                    # Staleness check on stems: frame identity, not the label's extension
                    # - DBs on disk hold frame_NNNNNN.jpg ids; new builds write .png
                    cached_paths = [Path(str(p)).stem for p in store[rec_key].attrs["image_paths"]]
                    expected = [Path(str(x)).stem for x in (ids if ids is not None else result.image_paths)]
```

(Check `Path` is imported in `localizer.py`; add `from pathlib import Path` at the top if not.)

- [ ] **Step 4: Switch the reconstructor ids and delete the stale comment**

In `reconstructor.py`, delete the six-line comment starting
`# The .jpg suffix is deliberate and stays even though the store writes .png.` and change

```python
    ids = [f"frame_{int(fi):06d}.jpg" for fi in frame_indices]
```

to

```python
    # Localization ids name the store's own files
    ids = [f"frame_{int(fi):06d}.png" for fi in frame_indices]
```

- [ ] **Step 5: Run localization and wrapper tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/localization tests/wrapper -q -rfE -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed"
```

Expected: failures ⊆ control list. A new failure asserting `.jpg` ids from the localize stage is a
test pinning the old label: change its expected suffix to `.png`, nothing else.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/localization/localizer.py collab_splats/wrapper/reconstructor.py tests/localization/test_localization_cache.py && git commit -m "fix(localization): staleness compares stems; ids name the .png frames

Old .jpg-id DBs still match, so no migration.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Dashboard ref paths by stem; query frames as PNG

**Files:**
- Modify: `collab_splats/dashboard/pipeline.py:694` (query frame save), `:708-713` (ref path resolve)
- Test: `tests/dashboard/test_run_localization.py` (append)

- [ ] **Step 1: Read the `_run` / `wired` fixtures**

```bash
cd /workspace/collab-splats/.worktrees/consistency && sed -n 40,135p tests/dashboard/test_run_localization.py
```

Note how `_run(tmp_path, wired, append=...)` builds `out_dir = tmp_path / SCENE` and which ids the
fake localizer reports (`/orig/00000.jpg`, `/orig/00001.jpg`, `/orig/cam_f000007.jpg`).

- [ ] **Step 2: Write the failing tests**

Append:

```python
def test_ref_paths_resolve_jpg_ids_to_png_store_files(tmp_path, wired):
    """Reconstruction ids end in .jpg; images/ holds .png. The resolved path must exist."""
    images_dir = tmp_path / SCENE / "images"
    images_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(images_dir / "00000.png")

    out, _ = _run(tmp_path, wired)

    assert out.ref_image_paths[0] == images_dir / "00000.png"
    assert out.ref_image_paths[1] == images_dir / "00001.jpg"  # no file on disk: label kept


def test_appended_query_frame_is_lossless_png(tmp_path, wired, monkeypatch):
    """The query frame written for the DB decodes byte-identical to the array localized."""
    noise = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    monkeypatch.setattr(pipeline, "extract_frame", lambda video, idx: noise)

    _run(tmp_path, wired, append=True)

    (saved,) = (tmp_path / SCENE / "localized_frames").iterdir()
    assert saved.suffix == ".png"
    np.testing.assert_array_equal(np.asarray(Image.open(saved)), noise)
```

Noise, not the fixture's zeros: an all-zero JPEG decodes exactly, so zeros cannot see the codec.
If `localized_frames/` is not under `tmp_path / SCENE`, read `run_localization`'s `img_dir` and
fix the path only. Add `from PIL import Image` if missing.

- [ ] **Step 3: Run to verify they fail**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py -q -k "png_store_files or lossless_png"
```

Expected: both FAIL (`images/00000.jpg` returned; saved suffix `.jpg`).

- [ ] **Step 4: Implement**

In `pipeline.py`, add a helper next to `read_localized_group`:

```python
def _local_ref_path(out_dir: Path, label: str, source: str) -> Path:
    """
    Resolve a DB id to this machine's file: images/ by stem, localized_frames/ by name.

    Args:
        out_dir: scene output directory.
        label: DB id as recorded on the building machine.
        source: 'reconstruction' or 'localized'.

    Returns:
        The existing file with the id's stem, else the id's name under the expected directory.
    """
    if source != "reconstruction":
        return Path(out_dir) / "localized_frames" / Path(label).name

    # Ids may carry .jpg while the store writes .png; the stem is the frame identity
    images_dir = Path(out_dir) / "images"
    for ext in fr.IMAGE_EXTS:
        candidate = images_dir / f"{Path(label).stem}{ext}"
        if candidate.exists():
            return candidate
    return images_dir / Path(label).name
```

Replace

```python
            # DB ids are basenames recorded on the machine that built the DB; resolve each to a
            # local file — reconstruction frames live in images/, localized ones in localized_frames/.
            ref_image_paths = [
                out_dir / ("images" if src == "reconstruction" else "localized_frames") / Path(p).name
                for p, src in zip(localizer.image_paths, localizer.frame_sources)
            ]
```

with

```python
            # DB ids were recorded on the building machine; resolve each to a local file
            ref_image_paths = [
                _local_ref_path(out_dir, p, src) for p, src in zip(localizer.image_paths, localizer.frame_sources)
            ]
```

and change line 694's `.jpg` to `.png`:

```python
                img_path = img_dir / f"{Path(query_video).stem}_f{frame_idx:06d}.png"
```

- [ ] **Step 5: Run the dashboard localization tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/dashboard/test_run_localization.py tests/dashboard/test_localize_page.py tests/dashboard/test_pipeline.py -q
```

Expected: all pass, including the existing `test_ref_paths_remapped_to_local_images_dir`
(no file on disk → label kept).

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/dashboard/pipeline.py tests/dashboard/test_run_localization.py && git commit -m "fix(dashboard): resolve ref frames by stem; save query frames as PNG

Reconstruction ids end in .jpg while images/ holds .png, so every
reference path was missing. Query frames fed back into the DB were
JPEG q75.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: InstantSfM fallback frame name

**Files:**
- Modify: `collab_splats/pointcloud/sfm/instantsfm.py:292`

No behavioural test: the string is a label in a fallback branch taken only when upstream's image
container lacks `filenames`, and nothing reads its extension.

- [ ] **Step 1: Change the fallback**

```python
                filename = self.images.filenames[idx] if hasattr(self.images, "filenames") else f"{idx}.png"
```

- [ ] **Step 2: Run the sfm tests**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/sfm -q
```

Expected: pass (or failures ⊆ control list).

- [ ] **Step 3: Commit**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git add collab_splats/pointcloud/sfm/instantsfm.py && git commit -m "fix(pointcloud): instantsfm fallback image name is .png

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Gate, graph update, bookkeeping

- [ ] **Step 1: Sweep for leftover `.jpg` frame writes and old helpers**

```bash
cd /workspace/collab-splats/.worktrees/consistency && git grep -n '\.jpg"' -- collab_splats evals/scripts ':!*.ipynb'; git grep -n "_scale_intrinsics_to_original\|default=lambda o: o.item()" -- collab_splats evals tests
```

Expected: only `mesh/io.py` (texture atlas) and read-side extension lists in the first grep; the
second grep is empty.

- [ ] **Step 2: Per-package gate vs control**

```bash
cd /workspace/collab-splats/.worktrees/consistency && PYTHONPATH=. /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)" && for p in evals geometry pointcloud localization dashboard wrapper; do PYTHONPATH=. /opt/venv/reconstruction/bin/python -m pytest tests/$p -q -rfE -p no:cacheprovider 2>&1 | grep -E "^(FAILED|ERROR)"; done | sort > /tmp/claude-0/consistency-after-failures.txt; comm -13 /tmp/claude-0/consistency-control-failures.txt /tmp/claude-0/consistency-after-failures.txt
```

Expected: `comm` prints nothing (no new failures). Also run `tests/test_docstring_contract.py`.

- [ ] **Step 3: Format only the touched files**

```bash
cd /workspace/collab-splats/.worktrees/consistency && /opt/venv/reconstruction/bin/black $(git diff --name-only clean/final -- '*.py') && /opt/venv/reconstruction/bin/isort $(git diff --name-only clean/final -- '*.py') && git diff --stat
```

Never run repo-wide black. If formatting changed anything, re-run Step 2's gate for those
packages, then commit as `style: black/isort on phase-1 files`.

- [ ] **Step 4: Update the graph and the plan's fork point**

```bash
cd /workspace/collab-splats/.worktrees/consistency && graphify update . 
```

Fill in the fork SHA below, tick the completed checkboxes, and commit the plan on `clean/consistency`
(`git add -f docs/superpowers/plans/2026-09-26-consistency-phase1.md`).

- [ ] **Step 5: Report**

Report to the user: per-task outcome, the `comm` result, and that MapAnything meshes built before
Task 7 placed depth in the wrong region. Merging into `clean/final` is the user's call.

---

Fork point: `<fill in at Task 0>`
