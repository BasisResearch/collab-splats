# Pipeline Parity Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close structural divergence between our LC pipeline and VGGT-SLAM by fixing the 16 vs 17-frame VGGT inference window (H1), confirm other hypotheses resolved, then validate with solver dump comparison and Sim3 ATE eval.

**Architecture:** Add `_trim_forward_outputs` helper to `wrappers.py`; extend the sliding window passed to `_forward` by `overlap_frames` (=1) while keeping the Submap's stored metadata at `submap_size` frames. GPU validation uses existing `vggt_slam_solver_dump.py` / `our_solver_dump.py` / `compare_solver_internals.py`.

**Tech Stack:** Python 3.11, numpy, pytest, `/opt/conda/envs/reconstruction/bin/python`

---

## Pre-work: hypotheses already resolved

| H | Status | Evidence |
|---|---|---|
| H2 (preprocessing) | Ruled out | Both call `load_and_preprocess_images` for inference |
| H5 (GTSAM noise) | Ruled out | Both σ=0.05×np.ones(15) |
| H6 (extraction SL4) | Ruled out | `decompose_camera` divides by `P[-1,-1]` before RQ — handles `H[3,3]≠1` |
| H3 (T=inv(K)@K) | No code change | Use `scale_method=rotation_only` for SLAM parity; noted in Task 4 |
| H4 (world_points frame) | Deferred | Only non-none scale, only if gap remains after H1 |

**Active:** H1 (window size 16 vs 17) — primary fix in Tasks 1–2.

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/wrappers.py` | Add `_trim_forward_outputs`; extend window in `_run_lc_loop` |
| `tests/pointcloud/test_wrappers.py` | Tests for trim helper and window extension |
| `tests/pointcloud/test_pose_extraction.py` | Regression guard for H6 non-issue |
| `evals/results/parity_harness/findings_summary.md` | Updated with post-fix measurements |

---

## Task 1: `_trim_forward_outputs` — test then implement

**Files:**
- Modify: `tests/pointcloud/test_wrappers.py`
- Modify: `collab_splats/pointcloud/wrappers.py`

- [ ] **Step 1: Add failing tests to `tests/pointcloud/test_wrappers.py`**

Append after existing tests:

```python
import numpy as np
from collab_splats.pointcloud.wrappers import _trim_forward_outputs


def test_trim_forward_outputs_dict_trims_arrays():
    raw = {
        "extrinsic": np.zeros((7, 3, 4)),
        "intrinsics": np.zeros((7, 3, 3)),
        "depth": np.zeros((7, 64, 64, 1)),
        "depth_conf": np.zeros((7, 64, 64)),
        "world_points": np.zeros((7, 100, 3)),
        "world_points_conf": np.zeros((7, 100)),
        "images": np.zeros((7, 3, 64, 64)),
    }
    trimmed = _trim_forward_outputs(raw, 6)
    for key, val in trimmed.items():
        assert isinstance(val, np.ndarray)
        assert val.shape[0] == 6, f"{key}: expected 6, got {val.shape[0]}"


def test_trim_forward_outputs_list_trims_list():
    raw = [{"extrinsic": np.zeros((1, 3, 4))} for _ in range(7)]
    trimmed = _trim_forward_outputs(raw, 6)
    assert len(trimmed) == 6


def test_trim_forward_outputs_no_op_when_short():
    raw = {"extrinsic": np.zeros((5, 3, 4)), "scalar": 1.0}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["extrinsic"].shape[0] == 5
    assert trimmed["scalar"] == 1.0


def test_trim_forward_outputs_preserves_non_array_values():
    raw = {"extrinsic": np.zeros((7, 3, 4)), "label": "keep", "count": 42}
    trimmed = _trim_forward_outputs(raw, 6)
    assert trimmed["label"] == "keep"
    assert trimmed["count"] == 42
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_dict_trims_arrays \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_list_trims_list \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_no_op_when_short \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_preserves_non_array_values -v
```

Expected: `ImportError: cannot import name '_trim_forward_outputs'`

- [ ] **Step 3: Implement in `wrappers.py`**

Add after the `_assemble_precorrection_extrinsics` function, before the `LoopClosureWrapper` class:

```python
def _trim_forward_outputs(raw: "dict | list", k: int) -> "dict | list":
    """Trim per-frame predictions to first k entries.

    When _forward receives a K+overlap window for extra VGGT attention context,
    discards the extra overlap predictions so only K are stored per Submap.
    """
    if isinstance(raw, list):
        return raw[:k]
    trimmed = {}
    for key, val in raw.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > k:
            trimmed[key] = val[:k]
        else:
            trimmed[key] = val
    return trimmed
```

- [ ] **Step 4: Run to verify pass**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_dict_trims_arrays \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_list_trims_list \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_no_op_when_short \
  tests/pointcloud/test_wrappers.py::test_trim_forward_outputs_preserves_non_array_values -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers.py
git commit -m "feat(wrappers): add _trim_forward_outputs for K+1 window fix"
```

---

## Task 2: Window extension — test then implement

**Files:**
- Modify: `tests/pointcloud/test_wrappers.py`
- Modify: `collab_splats/pointcloud/wrappers.py`

- [ ] **Step 1: Add failing test to `tests/pointcloud/test_wrappers.py`**

```python
import torch
from unittest.mock import MagicMock, patch


def _make_raw(k: int) -> dict:
    return {
        "extrinsic": np.zeros((k, 3, 4), dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32)[None].repeat(k, axis=0),
        "depth": np.zeros((k, 4, 4, 1), dtype=np.float32),
        "depth_conf": np.zeros((k, 4, 4), dtype=np.float32),
    }


def test_lc_loop_passes_k_plus_overlap_to_forward():
    """_run_lc_loop must pass submap_size+overlap_frames frames to _forward."""
    from collab_splats.pointcloud.wrappers import LoopClosureWrapper
    from collab_splats.pointcloud.loop_closure.closure import LoopClosureConfig

    submap_size = 3
    overlap = 1
    n_frames = 9

    cfg = LoopClosureConfig(
        submap_size=submap_size,
        submap_overlap=overlap,
        min_submap_gap=0,
        max_loops_per_submap=0,
        lc_threshold_l2=999.0,
    )

    captured_sizes: list[int] = []

    def fake_forward(model, views, **kwargs):
        sz = views.shape[0] if hasattr(views, "shape") else len(views)
        captured_sizes.append(sz)
        return _make_raw(sz)

    base = MagicMock()
    base.views = torch.zeros(n_frames, 3, 4, 4)
    base.image_paths = [f"img_{i:03d}.png" for i in range(n_frames)]
    base._forward = fake_forward
    base._lc_collate_outputs = lambda r: r
    base._lc_retrieval = None

    wrapper = LoopClosureWrapper.__new__(LoopClosureWrapper)
    wrapper.base = base
    wrapper.config = cfg

    # Note: patch target may need adjustment based on the exact import path in wrappers.py.
    # Run `grep -n DinoSaladExtractor collab_splats/pointcloud/wrappers.py` to confirm.
    with patch("collab_splats.pointcloud.wrappers.DinoSaladExtractor") as mock_dino:
        mock_dino.return_value = lambda frames: torch.zeros(frames.shape[0], 128)
        wrapper._run_lc_loop()

    # Non-final full windows should be submap_size + overlap = 4
    non_final = captured_sizes[:-1]
    assert all(sz == submap_size + overlap for sz in non_final), (
        f"Expected windows of size {submap_size + overlap}, got {non_final}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_wrappers.py::test_lc_loop_passes_k_plus_overlap_to_forward -v
```

Expected: `AssertionError: Expected windows of size 4, got [3, 3, ...]`

- [ ] **Step 3: Implement window extension in `wrappers.py`**

In `_run_lc_loop`, find the sliding window loop body. The block to replace starts at `end = min(start + K, N)` and ends after `torch.cuda.empty_cache()`. Replace it:

```python
                end = min(start + K, N)
                window = views[start:end]
                k = window.shape[0] if hasattr(window, "shape") else len(window)

                # Feed K+O frames to VGGT for broader attention context (matches VGGT-SLAM
                # submap_size + overlapping_window_size window). Only first K predictions used.
                end_ctx = min(start + K + O, N)
                window_ctx = views[start:end_ctx]
                k_ctx = (
                    window_ctx.shape[0] if hasattr(window_ctx, "shape") else len(window_ctx)
                )

                with torch.no_grad():
                    raw = self.base._forward(self.base.model, window_ctx, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Discard extra O-frame predictions; only K predictions enter Submap
                if k_ctx > k:
                    raw = _trim_forward_outputs(raw, k)
```

`window` (not `window_ctx`) continues to be used for `frames_cpu = window.cpu()` and `image_paths[start:end]` — those stay K frames.

- [ ] **Step 4: Run full wrappers test suite**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/test_wrappers.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run full pointcloud suite for regressions**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/ -v --tb=short 2>&1 | tail -20
```

Expected: same pass/fail count as before (4 pre-existing failures per CLAUDE.md are acceptable).

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/wrappers.py tests/pointcloud/test_wrappers.py
git commit -m "fix(wrappers): extend VGGT inference window to K+overlap for H1 parity fix"
```

---

## Task 3: H6 regression guard — decompose_camera projective scale

Documents that H6 is not a bug. No production code change.

**Files:**
- Modify: `tests/pointcloud/test_pose_extraction.py`

- [ ] **Step 1: Add test to `tests/pointcloud/test_pose_extraction.py`**

```python
from collab_splats.pointcloud.loop_closure.graph import decompose_camera, normalize_to_sl4
import numpy as np


def test_decompose_camera_handles_sl4_projective_scale():
    """decompose_camera divides by H[-1,-1] — correct even when H[3,3]≠1 after SL(4) norm."""
    R_expected = np.array([
        [1., 0., 0.],
        [0., 0., -1.],
        [0., 1., 0.],
    ])
    t_expected = np.array([0.1, -0.2, 0.5])
    H = np.eye(4)
    H[:3, :3] = R_expected
    H[:3, 3] = t_expected

    H_sl4 = normalize_to_sl4(H)
    assert abs(H_sl4[3, 3] - 1.0) > 1e-6, "H[3,3] should differ from 1 after SL(4) norm"
    assert abs(np.linalg.det(H_sl4) - 1.0) < 1e-9

    _, R_out, t_out, _ = decompose_camera(H_sl4)

    np.testing.assert_allclose(R_out, R_expected, atol=1e-6)
    np.testing.assert_allclose(t_out, t_expected, atol=1e-6)
```

- [ ] **Step 2: Run to verify it passes without any code change**

```bash
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_pose_extraction.py::test_decompose_camera_handles_sl4_projective_scale -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/pointcloud/test_pose_extraction.py
git commit -m "test(loop_closure): regression guard for decompose_camera SL(4) projective scale (H6)"
```

---

## Task 4: GPU — re-run solver dumps and validate H1 fix

**Run in tmux** (heavy inference, OOM risk in notebooks).

- [ ] **Step 1: Re-run VGGT-SLAM baseline dump**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/vggt_slam_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --out_json evals/results/parity_harness/vggt_slam_internals.json \
  --max_frames 200 --submap_size 16
```

Expected: 12 boundaries, `scale ≈ 1.0` per boundary.

- [ ] **Step 2: Re-run our pipeline dump**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/our_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --out_json evals/results/parity_harness/our_internals.json \
  --out_tum  evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```

- [ ] **Step 3: Compare boundaries**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/compare_solver_internals.py \
  --slam_dump evals/results/parity_harness/vggt_slam_internals.json \
  --our_dump  evals/results/parity_harness/our_internals.json \
  --out_json  evals/results/parity_harness/boundary_diff.json \
  --flag_threshold 0.01
```

**Key metrics to check:**

| Metric | Before H1 fix | Target after fix |
|---|---|---|
| `delta_H_overlap_frob` boundary 0 | 0.010 | < 0.001 |
| `delta_T_frob` boundary 0 | 0.014 | < 0.005 |
| mean `delta_H_w` | 0.659 | < 0.05 |
| flagged boundaries | 12/12 | ≤ 2/12 |

- [ ] **Step 4: If `delta_H_overlap` boundary 0 still > 0.001 — print T matrices**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json, numpy as np
slam = json.load(open('evals/results/parity_harness/vggt_slam_internals.json'))
ours = json.load(open('evals/results/parity_harness/our_internals.json'))
b0s = slam['boundaries'][0]; b0o = ours['boundaries'][0]
print('SLAM T:'); print(np.array(b0s['T']))
print('OUR  T:'); print(np.array(b0o['T']))
print('SLAM T-I norm:', np.linalg.norm(np.array(b0s['T']) - np.eye(4), 'fro'))
print('OUR  T-I norm:', np.linalg.norm(np.array(b0o['T']) - np.eye(4), 'fro'))
print('delta_H_overlap:', np.linalg.norm(np.array(b0s['H_overlap']) - np.array(b0o['H_overlap']), 'fro'))
"
```

If SLAM T ≈ diagonal (not rotation) and our T ≈ rotation → H3 residual in non-none dump mode.
If SLAM T and our T both ≈ I → investigate further (H4 world_points frame).

- [ ] **Step 5: Record results in `evals/results/parity_harness/findings_summary.md`**

```markdown
## 2026-05-29 — After H1 fix (K+overlap window)

boundary 0 delta_H_overlap_frob: <value>
boundary 0 delta_T_frob: <value>
mean delta_H_w: <value>
flagged boundaries: <N>/12

Interpretation: <H1 fix sufficient / residual from H3 in non-none mode / other>
```

- [ ] **Step 6: Commit findings**

```bash
git add evals/results/parity_harness/findings_summary.md
git commit -m "docs(evals): record boundary diff after H1 window fix"
```

---

## Task 5: GPU — Sim3 ATE eval

- [ ] **Step 1: Run 29-frame Sim3 eval with scale=none**

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_spark \
  --conditions lc \
  --submap_size 16 \
  --scale_method none \
  --max_frames 29
```

Record `Sim3 ATE RMSE` from output.

- [ ] **Step 2: Run 200-frame SE3 baseline + LC comparison**

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_spark \
  --conditions baseline lc \
  --submap_size 16 \
  --max_frames 200
```

- [ ] **Step 3: Append to `evals/results/parity_harness/findings_summary.md`**

```markdown
## 2026-05-29 — Sim3 / SE3 ATE after H1 fix

29-frame scale=none Sim3 ATE: <value>  (target: ≤ 0.1m; VGGT-SLAM: 0.038m)
200-frame SE3 baseline ATE:   <value>
200-frame SE3 lc ATE:         <value>
```

- [ ] **Step 4: Commit**

```bash
git add evals/results/parity_harness/findings_summary.md
git commit -m "docs(evals): record Sim3/SE3 ATE after H1 window fix"
```

---

## Task 6 (conditional): H3 analysis if gap > 0.5m remains

Only run if 29-frame Sim3 ATE > 0.5m after H1 fix.

- [ ] **Step 1: Compare per-boundary scale values**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import json, numpy as np
slam = json.load(open('evals/results/parity_harness/vggt_slam_internals.json'))
ours = json.load(open('evals/results/parity_harness/our_internals.json'))
print(f'{'B':>3}  {'SLAM_scale':>12}  {'OUR_scale':>12}  {'delta':>10}')
for i, (sb, ob) in enumerate(zip(slam['boundaries'], ours['boundaries'])):
    d = abs(sb['scale'] - ob.get('scale', 1.0))
    print(f'{i:>3}  {sb[\"scale\"]:>12.6f}  {ob.get(\"scale\",1.0):>12.6f}  {d:>10.6f}')
"
```

If mean delta_scale > 0.05 → H3 is contributing. Use `scale_method=none` to eliminate T
entirely and re-run Sim3 eval. Compare ATE. If `scale_method=none` ATE < 0.1m, H3 is the
residual cause; document and close.

- [ ] **Step 2: Re-run Sim3 eval forcing scale=none if not already done**

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes \
  --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_spark \
  --conditions lc \
  --submap_size 16 \
  --scale_method none \
  --max_frames 29
```

- [ ] **Step 3: Document and commit conclusion**

```bash
git add evals/results/parity_harness/findings_summary.md
git commit -m "docs(evals): H3 scale analysis — record conclusion"
```

---

## Quick reference

```bash
# Tests
/opt/conda/envs/reconstruction/bin/python -m pytest tests/pointcloud/ -v --tb=short

# Key files modified
collab_splats/pointcloud/wrappers.py                        # H1 fix
tests/pointcloud/test_wrappers.py                           # trim + window tests
tests/pointcloud/test_pose_extraction.py                    # H6 regression guard

# GPU validation scripts (tmux only)
evals/runners/vggt_slam_solver_dump.py
evals/runners/our_solver_dump.py
evals/runners/compare_solver_internals.py
evals/eval_gt.py
```
