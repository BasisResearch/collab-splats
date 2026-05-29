# LC Cross-Model Eval Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix mapanything windowed LC, re-run calibration for all backends under post-stride-fix windowing, set evidence-backed per-model `default_verify_match_ratio`, then run full baseline+LC eval across all four backends.

**Architecture:** Three phases: (1) fix mapanything `_lc_collate_outputs` to use window-specific views rather than the first-K-frames slice of `_processed_views`; (2) run `eval_similarity_calibration.py` per model to get real score distributions; (3) bake thresholds into creator `ClassVar`s then run `eval_gt.py` for all backends.

**Tech Stack:** Python 3.11 (`/opt/conda/envs/reconstruction/bin/python`), PyTorch, MapAnything postprocess API, pytest.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `collab_splats/pointcloud/feedforward/mapanything.py` | Modify | Store `_lc_window_views` in `_forward` Tensor branch; use it (then clear) in `_lc_collate_outputs` |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Modify | Add `default_verify_match_ratio: ClassVar[float]` from calibration |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | Modify | Update `default_verify_match_ratio` if calibration differs from 0.99 |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Modify | Add `default_verify_match_ratio: ClassVar[float]` from calibration |
| `tests/pointcloud/test_lc_collate_window.py` | Create | Unit tests for window-correct collation |
| `worklog/notes/2026-05-29-lc-threshold-recalibration.md` | Create | Calibration evidence + recommended thresholds |

---

## Task 1: Fix `_lc_collate_outputs` window views bug

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Create: `tests/pointcloud/test_lc_collate_window.py`

The bug: `_lc_collate_outputs` passes `self._processed_views[:len(raw_list)]` to
`postprocess_model_outputs_for_inference`. During LC, `raw_list` comes from processing
window `views[start:end]` — so frames 0..K-1 of the full sequence are used as view context,
not the actual window frames. Fix: store the window's preprocessed views on `self` in the
Tensor branch of `_forward`, then read them in `_lc_collate_outputs`.

VGGTSPARKCreator inherits VGGTXCreator — no changes needed there.

- [ ] **Step 1: Write failing test**

Create `tests/pointcloud/test_lc_collate_window.py`:

```python
"""Test that _lc_collate_outputs uses window-specific view context, not first-K frames."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch


def _make_mock_processed_view(tag: str):
    """Return a minimal MapAnything view dict with an identifiable tag."""
    return {"img": torch.zeros(1, 3, 224, 224), "_tag": tag}


def test_lc_collate_uses_window_views_not_full_sequence():
    """_lc_collate_outputs must use the window view context stored by _forward,
    not self._processed_views[:N] which always points at the first K frames."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    # Simulate full-sequence _processed_views (frames 0..9)
    creator._processed_views = [_make_mock_processed_view(f"full_{i}") for i in range(10)]
    # Simulate window views for frames 5..7 (set by _forward Tensor branch)
    window_views = [_make_mock_processed_view(f"win_{i}") for i in range(3)]
    creator._lc_window_views = window_views

    captured_views = []

    def fake_postprocess(raw_list, views_ctx, apply_mask):
        captured_views.extend(views_ctx)
        # Return minimal structure _lc_collate_outputs can iterate
        return [
            {"camera_poses": [torch.eye(4).unsqueeze(0)],
             "intrinsics": [torch.eye(3).unsqueeze(0)]}
            for _ in raw_list
        ]

    raw_list = [
        {"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)}
        for _ in range(3)
    ]

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        side_effect=fake_postprocess,
    ), patch(
        "collab_splats.pointcloud.feedforward.mapanything.invert_poses",
        return_value=np.eye(4),
    ):
        creator._lc_collate_outputs(raw_list)

    assert all(v["_tag"].startswith("win_") for v in captured_views), (
        f"Expected window views (win_*), got: {[v['_tag'] for v in captured_views]}"
    )
    # _lc_window_views cleared after use
    assert creator._lc_window_views is None


def test_lc_window_views_cleared_after_collate():
    """_lc_window_views must be None after _lc_collate_outputs runs."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator._processed_views = [_make_mock_processed_view("full_0")]
    creator._lc_window_views = [_make_mock_processed_view("win_0")]

    raw_list = [{"pts3d_cam": torch.zeros(1, 3), "pts3d": torch.zeros(1, 3)}]

    with patch(
        "collab_splats.pointcloud.feedforward.mapanything.postprocess_model_outputs_for_inference",
        return_value=[{"camera_poses": [torch.eye(4).unsqueeze(0)],
                       "intrinsics": [torch.eye(3).unsqueeze(0)]}],
    ), patch(
        "collab_splats.pointcloud.feedforward.mapanything.invert_poses",
        return_value=np.eye(4),
    ):
        creator._lc_collate_outputs(raw_list)

    assert creator._lc_window_views is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/collab-splats && \
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_lc_collate_window.py -v --no-header 2>&1 | tail -20
```

Expected: FAIL — `AttributeError: 'MapAnythingCreator' has no attribute '_lc_window_views'` or assertion fails showing `full_*` tags.

- [ ] **Step 3: Apply the fix to `mapanything.py`**

Read current `_forward` and `_lc_collate_outputs` to get exact line numbers, then apply two edits.

**Edit 1** — in the Tensor branch of `_forward`, after building `window_views`, store it:

Find this block in `_forward` (after the `for view in window_views:` device-transfer loop):
```python
            forward_views = window_views
            console.log(f"  → {len(window_views)} images (LC window), minibatch_size={self.minibatch_size}")
```

Replace with:
```python
            forward_views = window_views
            self._lc_window_views = window_views  # consumed by _lc_collate_outputs
            console.log(f"  → {len(window_views)} images (LC window), minibatch_size={self.minibatch_size}")
```

**Edit 2** — in `_lc_collate_outputs`, replace the `self._processed_views[:len(raw_list)]` slice:

Find:
```python
        # Run minimal postprocess to populate camera_poses and intrinsics keys.
        # apply_mask=False: LC only needs poses, not masked point clouds.
        # Use a matching slice of self._processed_views as the view context.
        processed = postprocess_model_outputs_for_inference(
            raw_list,
            self._processed_views[: len(raw_list)],
            apply_mask=False,
        )
```

Replace with:
```python
        # Run minimal postprocess to populate camera_poses and intrinsics keys.
        # apply_mask=False: LC only needs poses, not masked point clouds.
        # Use window-specific views stored by _forward (Tensor branch) so the
        # view context matches the actual window frames, not the first-K frames
        # of the full sequence.
        views_ctx = self._lc_window_views if self._lc_window_views is not None \
            else self._processed_views[: len(raw_list)]
        self._lc_window_views = None  # clear after use
        processed = postprocess_model_outputs_for_inference(
            raw_list,
            views_ctx,
            apply_mask=False,
        )
```

Also add `_lc_window_views: Any = field(default=None, init=False, repr=False)` to the `MapAnythingCreator` dataclass fields (alongside `_processed_views`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && \
/opt/conda/envs/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_lc_collate_window.py -v --no-header 2>&1 | tail -20
```

Expected: 2 PASSED.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
cd /workspace/collab-splats && \
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -v --no-header -q 2>&1 | tail -30
```

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && \
git add collab_splats/pointcloud/feedforward/mapanything.py \
        tests/pointcloud/test_lc_collate_window.py && \
git commit -m "$(cat <<'EOF'
fix(feedforward): mapanything _lc_collate_outputs uses window views not full-seq slice

_lc_collate_outputs passed self._processed_views[:K] as view context to
postprocess_model_outputs_for_inference. During LC, this always selected
frames 0..K-1 of the full sequence regardless of which window was processed.
_forward (Tensor branch) now stores the window's preprocessed views in
_lc_window_views; _lc_collate_outputs reads and clears it.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Run similarity calibration — all backends

**Files:** `evals/eval_similarity_calibration.py` (read-only), `evals/results/` (write)

Run in tmux. OOM risk — no parallel processes. Python env: `/opt/conda/envs/reconstruction/bin/python`.

The calibration script uses `--mode retrieved` to test DINO-SALAD retrieved pairs — the production
use case, not random pairs. This is the authoritative comparison.

Existing `lc_calibration_mapanything.json` used the old `percentile90` aggregation — stale. Re-run all.

- [ ] **Step 1: Start tmux session and run calibration for vggtx**

```bash
tmux new-session -d -s lc_calib \
  "cd /workspace/collab-splats && \
  /opt/conda/envs/reconstruction/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 20 \
    --mode retrieved \
    --models vggtx \
    --out evals/results/lc_calibration_vggtx_postfix.json \
    2>&1 | tee /tmp/calib_vggtx.log; echo DONE_VGGTX"
echo "Started vggtx calibration in tmux:lc_calib"
```

Monitor:
```bash
tmux attach -t lc_calib
# Detach with Ctrl-B D when done
```

Wait for `DONE_VGGTX` in log before proceeding.

- [ ] **Step 2: Run calibration for mapanything**

```bash
tmux send-keys -t lc_calib \
  "/opt/conda/envs/reconstruction/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 20 \
    --mode retrieved \
    --models mapanything \
    --out evals/results/lc_calibration_mapanything_postfix.json \
    2>&1 | tee /tmp/calib_mapanything.log; echo DONE_MAPANYTHING" Enter
```

- [ ] **Step 3: Run calibration for omega**

```bash
tmux send-keys -t lc_calib \
  "/opt/conda/envs/reconstruction/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 20 \
    --mode retrieved \
    --models omega \
    --out evals/results/lc_calibration_omega_postfix.json \
    2>&1 | tee /tmp/calib_omega.log; echo DONE_OMEGA" Enter
```

Note: if `omega` is not a valid `--models` value, check accepted names:
```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_similarity_calibration.py --help
```

- [ ] **Step 4: Print calibration summary**

```bash
/opt/conda/envs/reconstruction/bin/python - <<'EOF'
import json, glob, numpy as np

files = sorted(glob.glob("/workspace/collab-splats/evals/results/lc_calibration_*_postfix.json"))
print(f"\n{'Model':<20} {'mtq_mean':>10} {'mtq_std':>9} {'mtq_min':>9} {'mtq_max':>9} {'recommended':>12}")
print("-" * 75)
for f in files:
    data = json.load(open(f))
    if not isinstance(data, list):
        data = [data]
    for r in data:
        if r.get("model") == "vggt_spark_reference":
            continue
        scores = r.get("mean_top_quarter", {}).get("scores", [])
        scores = [s for s in scores if isinstance(s, float) and s == s]
        if not scores:
            continue
        mean, std = np.mean(scores), np.std(scores)
        recommended = max(round(mean - 2 * std, 2), 0.5)
        print(f"{r['model']:<20} {mean:>10.4f} {std:>9.4f} "
              f"{min(scores):>9.4f} {max(scores):>9.4f} {recommended:>12.2f}")
EOF
```

Record the `recommended` values — used in Task 3.

---

## Task 3: Set per-model thresholds from calibration

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`
- Modify: `collab_splats/pointcloud/feedforward/vggt_omega.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`
- Create: `worklog/notes/2026-05-29-lc-threshold-recalibration.md`

VGGTSPARKCreator inherits VGGTXCreator — SPARK gets vggtx's threshold automatically unless scores
differ. VGGT-SPARK reference scores (measured earlier) are ~1.025 mean — same model weights as
VGGTXCreator (`facebook/VGGT-1B`). So one threshold covers both.

- [ ] **Step 1: Update `default_verify_match_ratio` in VGGTXCreator**

In `collab_splats/pointcloud/feedforward/vggtx.py`, find the `VGGTXCreator` class definition
(around line 154) and add (or update) the ClassVar. Use the `recommended` value from Task 2 Step 4
for `vggtx`. Replace `<VGGTX_VALUE>` with that number:

```python
@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    """..."""

    # Calibrated from eval_similarity_calibration.py on chess_seq01 retrieved pairs,
    # post stride=K fix (2026-05-29). Formula: mean - 2*std of mean_top_quarter scores.
    default_verify_match_ratio: ClassVar[float] = <VGGTX_VALUE>
```

- [ ] **Step 2: Update `default_verify_match_ratio` in VGGTOmegaCreator**

In `collab_splats/pointcloud/feedforward/vggt_omega.py` around line 130:

```python
    # Calibrated from eval_similarity_calibration.py on chess_seq01 retrieved pairs,
    # post stride=K fix (2026-05-29). Formula: mean - 2*std. High because Omega
    # attention ratios cluster well above VGGT-X (mtq mean ~1.328 pre-fix).
    default_verify_match_ratio: ClassVar[float] = <OMEGA_VALUE>
```

Replace `<OMEGA_VALUE>` with calibration output. If calibration confirms ~1.328 mean, recommended
will be ~0.99 — update the comment accordingly.

- [ ] **Step 3: Add `default_verify_match_ratio` to MapAnythingCreator**

In `collab_splats/pointcloud/feedforward/mapanything.py`, find `MapAnythingCreator` class
definition and add alongside other ClassVars:

```python
    # Calibrated from eval_similarity_calibration.py on chess_seq01 retrieved pairs,
    # post stride=K fix (2026-05-29). MapAnything attention architecture differs from
    # VGGT-X — layer_index=4, lower score range.
    default_verify_match_ratio: ClassVar[float] = <MAPANYTHING_VALUE>
```

- [ ] **Step 4: Write calibration note to worklog**

Create `worklog/notes/2026-05-29-lc-threshold-recalibration.md`:

```markdown
# LC Threshold Recalibration — 2026-05-29

## Context
All prior threshold measurements were pre-stride-fix (stride=K-O=15, K frames stored).
Stride fix (2026-05-29) changed to stride=K=16, K+O=17 frames stored — matching VGGT-SLAM
exactly. Recalibration required because window frame composition changed.

## Method
`eval_similarity_calibration.py --mode retrieved --n_pairs 20` on chess_seq01.
Metric: `mean_top_quarter` (matches VGGT-SPARK's `get_similarity()`).
Threshold formula: `max(mean - 2*std, 0.5)` — captures ~95% of overlapping pairs.

## Results

| Model       | mtq_mean | mtq_std | mtq_min | mtq_max | threshold set |
|-------------|----------|---------|---------|---------|---------------|
| vggtx/spark | [FILL]   | [FILL]  | [FILL]  | [FILL]  | [FILL]        |
| mapanything | [FILL]   | [FILL]  | [FILL]  | [FILL]  | [FILL]        |
| omega       | [FILL]   | [FILL]  | [FILL]  | [FILL]  | [FILL]        |

Fill from Task 2 Step 4 output.

## Reference
VGGT-SPARK reference (VGGT-1B via SPARK): mtq_mean=1.025, all 11 pairs above 0.85.
```

Fill in the table from Task 2 Step 4 output before committing.

- [ ] **Step 5: Run tests**

```bash
cd /workspace/collab-splats && \
/opt/conda/envs/reconstruction/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -20
```

Expected: all passing.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && \
git add collab_splats/pointcloud/feedforward/vggtx.py \
        collab_splats/pointcloud/feedforward/vggt_omega.py \
        collab_splats/pointcloud/feedforward/mapanything.py \
        worklog/notes/2026-05-29-lc-threshold-recalibration.md && \
git commit -m "$(cat <<'EOF'
feat(lc): per-model default_verify_match_ratio from post-stride-fix calibration

Recalibrated on chess_seq01 retrieved pairs (eval_similarity_calibration.py,
mode=retrieved, n_pairs=20). Prior thresholds were stale — measured before
stride=K fix changed window composition. Formula: mean - 2*std (mean_top_quarter).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Full eval — all backends

**Files:** `evals/eval_gt.py` (read-only), `evals/results/` (write)

Run in tmux. OOM risk — one run at a time. Each run takes ~5–15 min depending on backbone.
Env: `/opt/conda/envs/reconstruction/bin/python`.

- [ ] **Step 1: Run vggt_spark (baseline + lc)**

```bash
tmux new-session -d -s lc_eval \
  "cd /workspace/collab-splats && \
  /opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggt_spark \
    --conditions baseline lc \
    --submap_size 16 \
    --lc_scale_method none \
    --max_frames 200 \
    2>&1 | tee /tmp/eval_spark.log; echo DONE_SPARK"
```

Monitor: `tmux attach -t lc_eval` (Ctrl-B D to detach).

Record from log:
- Baseline ATE: `grep -E 'baseline.*ATE|ATE.*baseline' /tmp/eval_spark.log`
- LC ATE: `grep -E 'lc.*ATE|ATE.*lc' /tmp/eval_spark.log`

- [ ] **Step 2: Run vggtx (baseline + lc)**

```bash
tmux send-keys -t lc_eval \
  "/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggtx \
    --conditions baseline lc \
    --submap_size 16 \
    --lc_scale_method none \
    --max_frames 200 \
    2>&1 | tee /tmp/eval_vggtx.log; echo DONE_VGGTX" Enter
```

- [ ] **Step 3: Run vggt_omega (baseline + lc)**

```bash
tmux send-keys -t lc_eval \
  "/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone vggt_omega \
    --conditions baseline lc \
    --submap_size 16 \
    --lc_scale_method none \
    --max_frames 200 \
    2>&1 | tee /tmp/eval_omega.log; echo DONE_OMEGA" Enter
```

- [ ] **Step 4: Run mapanything (baseline + lc)**

```bash
tmux send-keys -t lc_eval \
  "/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
    --dataset 7scenes \
    --seq_dir evals/data/7scenes/chess/chess/seq-01 \
    --backbone mapanything \
    --conditions baseline lc \
    --submap_size 16 \
    --lc_scale_method none \
    --max_frames 200 \
    2>&1 | tee /tmp/eval_mapanything.log; echo DONE_MAPANYTHING" Enter
```

- [ ] **Step 5: Collect and print ATE table**

```bash
for log in spark vggtx omega mapanything; do
  echo "=== $log ===";
  grep -E 'ATE|ate|RMSE' /tmp/eval_${log}.log | tail -5;
done
```

- [ ] **Step 6: Commit worklog update**

Update `docs/superpowers/plans/2026-05-29-lc-cross-model-eval-handoff.md` ATE table with results,
then commit:

```bash
cd /workspace/collab-splats && \
git add docs/superpowers/plans/2026-05-29-lc-cross-model-eval-handoff.md \
        worklog/notes/2026-05-29-lc-threshold-recalibration.md && \
git commit -m "$(cat <<'EOF'
docs(worklog): LC cross-model eval results — post stride=K fix

Full baseline+LC ATE table for vggt_spark, vggtx, vggt_omega, mapanything
on chess_seq01. Per-model thresholds from recalibration (2026-05-29).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

- [x] Task 1 covers mapanything `_lc_collate_outputs` fix with tests
- [x] Task 2 covers calibration for all 4 backends (vggtx, mapanything, omega; spark inherits vggtx)
- [x] Task 3 covers threshold setting for all 3 creators + worklog note
- [x] Task 4 covers full eval for all 4 backends
- [x] No placeholders in code blocks — all commands are exact
- [x] Calibration note table has explicit fill instruction tied to Task 2 Step 4 output
- [x] `_lc_window_views` field added to dataclass in Task 1 Step 3
- [x] VGGTSPARKCreator inheritance from VGGTXCreator noted — no separate threshold needed unless scores diverge
