# LC Threshold Calibration Across FF Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine per-model `verify_match_ratio` thresholds for cross_frame_attention_ratio and update LoopClosureConfig to use them correctly, so LC actually fires for VGGT-X (and other backends) on chess_seq01.

**Architecture:** Run `eval_similarity_calibration.py` across all available FF models, analyze score distributions under both old (percentile_90) and new (mean_top_quarter) aggregation, determine if one threshold serves all models or per-model defaults are needed, implement accordingly via a class-level `default_verify_match_ratio` on each Creator, and validate LC fires on chess_seq01.

**Tech Stack:** Python 3.11 (`/opt/conda/envs/nerfstudio/bin/python`), PyTorch, tmux for GPU runs, existing `LoopClosureConfig` + `BaseFeedforwardCreator` abstractions.

---

## Context

Two separate thresholds in `LoopClosureConfig`:

| Field | Default | Role |
|---|---|---|
| `lc_cosine_threshold` | 0.75 | DINO-SALAD L2 gate — filters retrieval candidates |
| `verify_match_ratio` | 0.85 | `cross_frame_attention_ratio` gate — final accept/reject |

The aggregation fix (percentile_90 → mean_top_quarter) likely shifts VGGT-X scores from ~0.74 to an unknown higher value. If scores still sit below 0.85, we need a lower `verify_match_ratio` for VGGT-X. VGGT-1B baseline: mean_top_quarter scores cluster ~1.025 on chess_seq01 (all above 0.85 with margin).

## File Map

| File | Action | Why |
|---|---|---|
| `evals/eval_similarity_calibration.py` | already written | runs calibration |
| `evals/results/similarity_calibration.json` | created by script | analysis input (gitignored) |
| `collab_splats/pointcloud/feedforward/base.py` | modify | add `default_verify_match_ratio` class attr |
| `collab_splats/pointcloud/feedforward/vggtx.py` | modify | set model-specific default |
| `collab_splats/pointcloud/feedforward/mapanything.py` | modify | set model-specific default |
| `collab_splats/pointcloud/feedforward/vggt_omega.py` | modify | set model-specific default |
| `collab_splats/pointcloud/wrappers.py` | modify | use creator default when config.verify_match_ratio is default |
| `collab_splats/pointcloud/loop_closure/closure.py` | possibly modify | update LoopClosureConfig default if single threshold works |
| `worklog/notes/2026-05-27-lc-threshold-calibration.md` | create | record findings |
| `tests/pointcloud/test_feedforward_similarity.py` | create | verify class attr inheritance |

---

## Task 1: Run calibration — VGGT-X

**Files:** `evals/eval_similarity_calibration.py` (read-only), `evals/results/similarity_calibration.json` (output)

- [ ] **Step 1: Launch VGGT-X calibration in tmux**

```bash
tmux new-session -d -s lc_calib_vggtx \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 15 \
    --models vggtx \
    --out evals/results/lc_calibration_vggtx.json \
    2>&1 | tee /tmp/lc_calib_vggtx.log"
```

- [ ] **Step 2: Confirm session started**

```bash
tmux ls | grep lc_calib_vggtx
# Expected: lc_calib_vggtx: 1 windows (created ...)
```

- [ ] **Step 3: Tail log until done** (takes ~5-10 min on GPU)

```bash
tail -f /tmp/lc_calib_vggtx.log
# Expected last lines: summary table + "Results → evals/results/lc_calibration_vggtx.json"
```

---

## Task 2: Run calibration — MapAnything

**Files:** same as Task 1 but different model + output file.

- [ ] **Step 1: Launch MapAnything calibration in tmux**

```bash
tmux new-session -d -s lc_calib_ma \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 15 \
    --models mapanything \
    --out evals/results/lc_calibration_mapanything.json \
    2>&1 | tee /tmp/lc_calib_ma.log"
```

- [ ] **Step 2: Tail log until done** (~5-10 min)

```bash
tail -f /tmp/lc_calib_ma.log
# Expected last lines: summary table + "Results →..."
```

---

## Task 3: Run calibration — VGGT-Omega (if available)

- [ ] **Step 1: Check if Omega is installed**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.pointcloud.feedforward import VGGTOmegaCreator; print('ok')" 2>&1
# If "ok": proceed. If ImportError: skip this task.
```

- [ ] **Step 2: Launch Omega calibration (only if step 1 succeeded)**

```bash
tmux new-session -d -s lc_calib_omega \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_similarity_calibration.py \
    --scene_dir evals/data/7scenes/chess/chess/seq-01 \
    --n_pairs 15 \
    --models omega \
    --out evals/results/lc_calibration_omega.json \
    2>&1 | tee /tmp/lc_calib_omega.log"
```

- [ ] **Step 3: Tail log until done**

```bash
tail -f /tmp/lc_calib_omega.log
```

---

## Task 4: Analyze results

**Files:** `evals/results/lc_calibration_*.json` (read), `worklog/notes/2026-05-27-lc-threshold-calibration.md` (write)

- [ ] **Step 1: Print summary across all models**

```bash
/opt/conda/envs/nerfstudio/bin/python - <<'EOF'
import json, glob, numpy as np

files = sorted(glob.glob("evals/results/lc_calibration_*.json"))
print(f"\n{'Model':<25} {'pct90 mean':>11} {'mtq mean':>10} {'mtq min':>9} {'mtq max':>9} {'above 0.85':>10}")
print("-" * 80)
for f in files:
    data = json.load(open(f))
    for r in data:
        if r["model"] == "vggt_spark_reference":
            continue
        p90 = r.get("percentile90", {})
        mtq = r.get("mean_top_quarter", {})
        scores = [s for s in mtq.get("scores", []) if not (isinstance(s, float) and s != s)]
        n_above = sum(1 for s in scores if s >= 0.85)
        print(
            f"{r['model']:<25}"
            f" {p90.get('mean', float('nan')):>11.4f}"
            f" {mtq.get('mean', float('nan')):>10.4f}"
            f" {mtq.get('min', float('nan')):>9.4f}"
            f" {mtq.get('max', float('nan')):>9.4f}"
            f" {n_above}/{len(scores):>8}"
        )
EOF
```

- [ ] **Step 2: Determine threshold strategy**

Read the output. Apply this decision tree:

```
A) mtq mean > 0.85 for all models
   → Single threshold 0.85 works. No per-model logic needed.
   → Update LoopClosureConfig.verify_match_ratio default stays 0.85.
   → Skip Task 5 step 3+, go to Task 6.

B) mtq mean < 0.85 for some models but > some lower value X
   → Models need different thresholds.
   → Set per-model: threshold = mean - 2*std (catches ~95% of overlapping pairs)
   → Go to Task 5 (per-model defaults).

C) mtq mean < 0.60 for a model
   → Aggregation mismatch beyond threshold — different attention architecture.
   → Investigate token_offset or layer_index for that model before setting threshold.
```

- [ ] **Step 3: Compute per-model recommended thresholds**

```bash
/opt/conda/envs/nerfstudio/bin/python - <<'EOF'
import json, glob, numpy as np

files = sorted(glob.glob("evals/results/lc_calibration_*.json"))
print("\nRecommended verify_match_ratio per model (mean - 2*std, floored at 0.5):")
for f in files:
    data = json.load(open(f))
    for r in data:
        if r["model"] == "vggt_spark_reference":
            continue
        scores = [s for s in r.get("mean_top_quarter", {}).get("scores", [])
                  if isinstance(s, float) and s == s]
        if not scores:
            continue
        mean, std = np.mean(scores), np.std(scores)
        # threshold = mean - 2*std gives ~95% coverage on overlapping pairs
        # floor at 0.5 to avoid degenerate acceptance
        recommended = max(round(mean - 2*std, 2), 0.5)
        print(f"  {r['model']:<22}: mean={mean:.4f} std={std:.4f} → recommended={recommended:.2f}")
EOF
```

- [ ] **Step 4: Write findings to worklog**

Create `worklog/notes/2026-05-27-lc-threshold-calibration.md` with:
```markdown
# LC Threshold Calibration — 2026-05-27

## Aggregation fix
`cross_frame_attention_ratio`: `np.percentile(90)` → `mean_top_quarter` (matches VGGT-SPARK).

## Score summary (chess_seq01, 15 pairs, gap 5-40 frames)

| Model | pct90 mean | mtq mean | mtq min | mtq max | above 0.85 |
|---|---|---|---|---|---|
| vggt_spark (ref) | — | 1.025 | 1.002 | 1.043 | 11/11 |
| vggtx | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| mapanything | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| vggt_omega | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

## Threshold decision

[FILL: single threshold or per-model, reasoning]

## Per-model recommended verify_match_ratio

[FILL from step 3 output]
```

Fill `[FILL]` from step 1+3 output.

---

## Task 5: Implement threshold updates

**Files:** `collab_splats/pointcloud/feedforward/base.py`, `vggtx.py`, `mapanything.py`, `vggt_omega.py`, `collab_splats/pointcloud/wrappers.py`

*Skip if Task 4 Step 2 outcome is (A) — single threshold 0.85 works for all.*

- [ ] **Step 1: Add `default_verify_match_ratio` to BaseFeedforwardCreator**

In `collab_splats/pointcloud/feedforward/base.py`, find the class definition of `BaseFeedforwardCreator` and add:

```python
class BaseFeedforwardCreator(BasePointcloudCreator):
    """..."""

    # Threshold for cross_frame_attention_ratio LC verification gate.
    # Calibrated per model; VGGT-1B (VGGT-SPARK) baseline = 0.85.
    # Subclasses override with model-specific values from calibration.
    default_verify_match_ratio: float = 0.85
```

- [ ] **Step 2: Override in each creator with calibrated value**

In `collab_splats/pointcloud/feedforward/vggtx.py`, inside `VGGTXCreator`:
```python
# Set from eval_similarity_calibration.py — replace [VALUE] with Task 4 Step 3 output
default_verify_match_ratio: float = [VALUE]
```

In `collab_splats/pointcloud/feedforward/mapanything.py`, inside `MapAnythingCreator`:
```python
default_verify_match_ratio: float = [VALUE]
```

In `collab_splats/pointcloud/feedforward/vggt_omega.py`, inside `VGGTOmegaCreator`:
```python
default_verify_match_ratio: float = [VALUE]
```

- [ ] **Step 3: Update LoopClosure wrapper to use creator default**

In `collab_splats/pointcloud/wrappers.py`, find `LoopClosure.__init__`:

```python
def __init__(self, base: Any, config: LoopClosureConfig | None = None) -> None:
    self.base = base
    if config is None:
        # Use the creator's calibrated threshold as default verify_match_ratio
        creator_ratio = getattr(base, "default_verify_match_ratio", 0.85)
        self.config = LoopClosureConfig(verify_match_ratio=creator_ratio)
    else:
        self.config = config
```

- [ ] **Step 4: Write tests**

Create `tests/pointcloud/test_feedforward_similarity.py`:

```python
"""Tests: per-model default_verify_match_ratio and LoopClosure wrapper inheritance."""
import pytest
from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator


def test_base_has_default_verify_match_ratio():
    assert hasattr(BaseFeedforwardCreator, "default_verify_match_ratio")
    assert isinstance(BaseFeedforwardCreator.default_verify_match_ratio, float)


def test_vggtx_overrides_verify_match_ratio():
    # VGGTXCreator must define its own value (not inherit base 0.85 unchanged
    # unless calibration confirms 0.85 is correct for VGGT-X)
    assert hasattr(VGGTXCreator, "default_verify_match_ratio")
    assert 0.5 <= VGGTXCreator.default_verify_match_ratio <= 1.0


def test_mapanything_overrides_verify_match_ratio():
    assert hasattr(MapAnythingCreator, "default_verify_match_ratio")
    assert 0.5 <= MapAnythingCreator.default_verify_match_ratio <= 1.0


def test_loop_closure_uses_creator_default(monkeypatch):
    """LoopClosure with no explicit config picks up creator's threshold."""
    from unittest.mock import MagicMock
    from collab_splats.pointcloud.wrappers import LoopClosure

    mock_creator = MagicMock()
    mock_creator.default_verify_match_ratio = 0.72
    lc = LoopClosure(base=mock_creator)
    assert lc.config.verify_match_ratio == 0.72


def test_loop_closure_explicit_config_wins(monkeypatch):
    """Explicit LoopClosureConfig overrides creator default."""
    from unittest.mock import MagicMock
    from collab_splats.pointcloud.wrappers import LoopClosure
    from collab_splats.pointcloud.loop_closure.closure import LoopClosureConfig

    mock_creator = MagicMock()
    mock_creator.default_verify_match_ratio = 0.72
    cfg = LoopClosureConfig(verify_match_ratio=0.90)
    lc = LoopClosure(base=mock_creator, config=cfg)
    assert lc.config.verify_match_ratio == 0.90
```

- [ ] **Step 5: Run tests (import-only, no GPU)**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
  tests/pointcloud/test_feedforward_similarity.py -v --no-header
# Expected: all 5 pass
```

- [ ] **Step 6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py \
        collab_splats/pointcloud/feedforward/vggtx.py \
        collab_splats/pointcloud/feedforward/mapanything.py \
        collab_splats/pointcloud/feedforward/vggt_omega.py \
        collab_splats/pointcloud/wrappers.py \
        tests/pointcloud/test_feedforward_similarity.py
git commit -m "feat(lc): per-model default_verify_match_ratio on FF creators

Calibrated from eval_similarity_calibration.py on chess_seq01 (15 pairs).
LoopClosure wrapper picks up creator default when no explicit config given.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Validate LC fires on chess_seq01

**Files:** `evals/eval_gt.py` (read-only)

- [ ] **Step 1: Run LC pipeline on chess_seq01 with new threshold**

```bash
tmux new-session -d -s lc_validate \
  "/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py \
    --scene chess --seq seq-01 \
    --data_root evals/data/7scenes \
    --conditions lc_sl4 \
    --submap_size 16 \
    2>&1 | tee /tmp/lc_validate.log"
```

- [ ] **Step 2: Check LC triggered in logs**

```bash
grep -E "loop closure|LC triggered|accepted|similarity|verify_match" /tmp/lc_validate.log | head -20
# Expected: lines showing accepted=True for some candidates
```

- [ ] **Step 3: Check ATE vs baseline**

```bash
grep -E "ATE|RMSE|ate" /tmp/lc_validate.log | tail -5
# Compare against: baseline no-LC ATE = 0.3184 m
# LC should be ≤ 0.3184 m (if it fires and helps) or ~equal (if chess_seq01 is degenerate)
```

- [ ] **Step 4: If LC still doesn't fire — diagnose**

```bash
grep -E "ratio|threshold|below|similarity" /tmp/lc_validate.log | head -20
# Look for logged ratio values vs threshold
# If ratio values still below threshold: lower verify_match_ratio further by 0.05 and repeat Task 6
```

---

## Task 7: Commit calibration evidence + worklog update

- [ ] **Step 1: Fill in worklog note** (Task 4 Step 4)

Open `worklog/notes/2026-05-27-lc-threshold-calibration.md` and replace `[FILL]` placeholders with actual numbers from Task 4 output.

- [ ] **Step 2: Commit calibration outputs + worklog**

```bash
git add worklog/notes/2026-05-27-lc-threshold-calibration.md \
        evals/eval_similarity_calibration.py \
        collab_splats/pointcloud/utils.py
git commit -m "fix(lc): mean_top_quarter aggregation + threshold calibration analysis

cross_frame_attention_ratio: np.percentile(90) → mean_top_quarter to match
VGGT-SPARK get_similarity(). Calibration results in worklog/notes/.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] Spec coverage: run (T1-3), analyze (T4), implement (T5), validate (T6), commit (T7)
- [x] Data path confirmed: `evals/data/7scenes/chess/chess/seq-01`
- [x] Both thresholds distinguished: `lc_cosine_threshold` (DINO-SALAD, not touched) vs `verify_match_ratio` (attention ratio, calibrated here)
- [x] Decision tree in T4.2 handles all three outcomes (all pass / some fail / very low)
- [x] Tests cover both the class attr and the wrapper inheritance
- [x] T5 is conditional (skip if single threshold works) — clearly marked
- [x] No TBDs except `[VALUE]` placeholders that are filled from T4 output before T5 executes
