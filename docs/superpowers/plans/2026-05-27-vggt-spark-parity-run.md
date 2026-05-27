# VGGT-SPARK Parity Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `run_vggt_slam_lc.py` to use VGGT-SPARK's native `compute_similarity` support, run all five parity comparisons, and write a structured analysis of where our LC diverges from VGGT-SLAM's.

**Architecture:** Clone VGGT-SPARK (MIT-SPARK's VGGT-1B fork) as an uninstalled source tree; shadow the installed `vggt` package per-process via `sys.path.insert` in the runner only. All five comparisons run in the `reconstruction` env — GPU runs in tmux, CPU analysis inline. Results written to `worklog/notes/`.

**Tech Stack:** Python 3.11, `reconstruction` conda env, tmux for GPU runs, `compare_solver_internals.py` + `diagnose_lc_parity.py` (existing scripts).

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `third_party/vggt_spark/` | Create (clone) | VGGT-SPARK source tree — not pip-installed |
| `evals/runners/run_vggt_slam_lc.py` | Modify | sys.path shadow at top; delete `_VGGTCompatWrapper`; fix docstring |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` | Create (run) | run 3 output |
| `evals/results/parity_harness/boundary_diff.json` | Create (run) | comparison 2 output |
| `worklog/notes/2026-05-27-vggt-slam-parity-analysis.md` | Create | written analysis of all 5 comparisons |

---

### Task 1: Clone VGGT-SPARK

**Files:**
- Create: `third_party/vggt_spark/` (git clone, not tracked as submodule)

- [ ] **Step 1: Clone**

```bash
cd /workspace/collab-splats/third_party
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt_spark
cd ..
```

- [ ] **Step 2: Verify package structure**

```bash
ls third_party/vggt_spark/vggt/__init__.py
```

Expected: file exists. If missing, the repo structure differs from expected — stop and check `ls third_party/vggt_spark/` before proceeding.

- [ ] **Step 3: Verify compute_similarity is present**

```bash
grep -r "compute_similarity" third_party/vggt_spark/vggt/ | head -10
```

Expected: at least one match in `vggt/models/vggt.py`. If zero matches, VGGT-SPARK doesn't have the feature — stop and re-examine the repo.

- [ ] **Step 4: Add to .gitignore (don't track as submodule)**

```bash
grep -q "third_party/vggt_spark" .gitignore || echo "third_party/vggt_spark/" >> .gitignore
git add .gitignore
git commit -m "chore: gitignore third_party/vggt_spark (untracked eval dep)"
```

---

### Task 2: Fix `run_vggt_slam_lc.py`

**Files:**
- Modify: `evals/runners/run_vggt_slam_lc.py`

- [ ] **Step 1: Replace file header — add sys.path shadow before all imports**

The sys.path insertion MUST come before any `vggt` import. Replace the top of the file from `#!/usr/bin/env python` through the `from vggt.models.vggt import VGGT` line:

```python
#!/usr/bin/env python
"""Run VGGT-SLAM with loop closure on a 7-Scenes sequence, write dense TUM.

Usage:
    python evals/runners/run_vggt_slam_lc.py \\
        --seq_dir data/7scenes/chess/seq-01 \\
        --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \\
        --max_frames 200

Fixed pipeline args (matching VGGT-SLAM paper defaults):
    submap_size=16, overlapping_window_size=1, conf_threshold=25.0,
    max_loops=1, min_disparity=0

Note — min_disparity=0 (not 50): with min_disparity=50, chess_seq01 produces zero loop
closure candidates; setting 0 accepts all frames so LC is actually triggered. This is an
intentional deviation for evaluation purposes — both run_vggt_slam_lc.py and
our_solver_dump.py use the same setting so frame selection is consistent.
"""
from __future__ import annotations

# ── VGGT-SPARK shadow ─────────────────────────────────────────────────────────
# Shadow the installed vggt package (VGGT-X based) with VGGT-SPARK, which adds
# native compute_similarity=True support to VGGT.forward(). Must come before any
# vggt import. Affects this process only — reconstruction env is unaffected.
import sys
from pathlib import Path as _Path
_vggt_spark = str(_Path(__file__).resolve().parents[2] / "third_party" / "vggt_spark")
if _vggt_spark not in sys.path:
    sys.path.insert(0, _vggt_spark)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import glob
import logging
from pathlib import Path

import cv2
import torch
from tqdm.auto import tqdm

# Add repo root and VGGT-SLAM to path
_repo_root = str(Path(__file__).resolve().parents[2])
_slam_root = str(Path(__file__).resolve().parents[2] / "third_party" / "VGGT-SLAM")
for _p in (_repo_root, _slam_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT  # resolves to VGGT-SPARK version via sys.path shadow
```

- [ ] **Step 2: Delete `_VGGTCompatWrapper` and replace model assignment**

Find and delete the block from `# Wrap model to handle compute_similarity=True` through `model = _VGGTCompatWrapper(_vggt)`.

Replace with:

```python
    model = _vggt  # VGGT-SPARK natively handles compute_similarity=True
```

- [ ] **Step 3: Smoke-test import (no GPU needed)**

```bash
/opt/conda/envs/reconstruction/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('third_party/vggt_spark').resolve()))
from vggt.models.vggt import VGGT
import inspect
sig = inspect.signature(VGGT.forward)
assert 'compute_similarity' in sig.parameters, f'compute_similarity missing from VGGT.forward: {sig}'
print('OK — VGGT-SPARK VGGT.forward has compute_similarity param')
"
```

Expected: `OK — VGGT-SPARK VGGT.forward has compute_similarity param`

- [ ] **Step 4: Commit**

```bash
git add evals/runners/run_vggt_slam_lc.py
git commit -m "fix(evals): use VGGT-SPARK for compute_similarity in run_vggt_slam_lc

sys.path shadow third_party/vggt_spark before vggt import.
Delete _VGGTCompatWrapper (image_match_ratio=1.0 stub was invalid).
VGGT-SPARK adds native compute_similarity=True to VGGT.forward()
with same VGGT-1B weights — real image_match_ratio computed."
```

---

### Task 3: Run 3 — VGGT-SLAM with LC (GPU)

**Files:**
- Create: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum`

- [ ] **Step 1: Launch in tmux**

```bash
tmux new-session -d -s vggt_slam_run3
tmux send-keys -t vggt_slam_run3 \
  "/opt/conda/envs/reconstruction/bin/python evals/runners/run_vggt_slam_lc.py \
    --seq_dir data/7scenes/chess/seq-01 \
    --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
    --max_frames 200 2>&1 | tee /tmp/run3.log; echo RUN3_DONE" Enter
echo "Run 3 launched. Monitor: tmux attach -t vggt_slam_run3"
```

- [ ] **Step 2: Verify no crash on compute_similarity**

```bash
tmux attach -t vggt_slam_run3
# Watch for either: RUN3_DONE or TypeError
# Detach with Ctrl-B D
```

While running, check log for loop closure detection:

```bash
grep -E "detected_loops|loop closure|image_match_ratio|Loop" /tmp/run3.log | tail -20
```

Expected: at least one `detected_loops` line with a non-empty list, and `image_match_ratio` values that are NOT all 1.0.

- [ ] **Step 3: Verify output**

```bash
wc -l evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum
head -3 evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum
```

Expected: ~200 lines; each line `<timestamp> tx ty tz qx qy qz qw`.

- [ ] **Step 4: Commit TUM file**

```bash
git add evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum
git commit -m "feat(evals): add vggt_slam_lc.tum — run 3 with VGGT-SPARK

chess_seq01, 200 frames, submap_size=16, lc_thres=0.95,
min_disparity=0. LC triggered with real image_match_ratio."
```

---

### Task 4: Comparison 2 — Internal boundary diff

**Files:**
- Create: `evals/results/parity_harness/boundary_diff.json` (gitignored, local only)

Prerequisite: `vggt_slam_internals.json` and `our_internals.json` already exist (runs 1 & 2 ✅).

- [ ] **Step 1: Run compare_solver_internals**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/compare_solver_internals.py \
  2>&1 | tee /tmp/boundary_diff.log
```

Expected output: a table with columns `boundary_id | delta_T | delta_H_w | ratio`.

- [ ] **Step 2: Capture key findings**

```bash
# Boundaries with high divergence (delta_H_w > 0.1)
grep -E "^\s*[0-9]" /tmp/boundary_diff.log | awk '$3 > 0.1 {print "HIGH:", $0}'

# Boundaries where ratio >> 2 (solver logic gap, not model gap)
grep -E "^\s*[0-9]" /tmp/boundary_diff.log | awk '$4 > 2.0 {print "SOLVER GAP:", $0}'
```

---

### Task 5: Comparison 1 + 3 + 4 — ATE, loop detection, per-frame

**Files:**
- Uses: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` (Task 3)
- Uses: `evals/results/parity_harness/our_lc.tum` (run 2, exists ✅)

- [ ] **Step 1: Run diagnose_lc_parity**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16 \
  2>&1 | tee /tmp/diagnose_lc.log
```

- [ ] **Step 2: Extract ATE summary (Comparison 1)**

```bash
grep -E "ATE|ate|RMSE|rmse|our_baseline|our_lc|vggt_slam" /tmp/diagnose_lc.log
```

Record three values:
- `our_baseline_ATE` = 0.3184 m (known)
- `our_lc_ATE` = from output
- `vggt_slam_lc_ATE` = from output

Compute:
- LC algorithm contribution = `our_baseline_ATE − our_lc_ATE`
- Model quality gap = `our_lc_ATE − vggt_slam_lc_ATE`

- [ ] **Step 3: Extract loop detection summary (Comparison 3)**

```bash
grep -E "loop|Loop|detected|match_ratio" /tmp/diagnose_lc.log
grep -E "loop|Loop|detected|match_ratio" /tmp/run3.log
```

Compare: did both systems detect loops? Same submap pairs?

---

### Task 6: Comparison 5 — Null hypothesis

**Files:**
- Uses: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum` (run 1, exists ✅)

- [ ] **Step 1: Compute no-LC ATE**

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16 \
  2>&1 | grep -E "ATE|ate|RMSE|vggt_slam"
```

Record `vggt_slam_nolc_ATE`. Compare to `our_baseline_ATE` (0.3184 m):
- If `vggt_slam_nolc_ATE >> our_baseline_ATE`: VGGT-X is simply a stronger model — LC gap is partly model-driven.
- If approximately equal: baseline quality is similar; LC differences are algorithm-driven.

---

### Task 7: Write parity analysis note

**Files:**
- Create: `worklog/notes/2026-05-27-vggt-slam-parity-analysis.md`

- [ ] **Step 1: Write the analysis file**

Use all findings from Tasks 4–6. Template (fill in TBD values from run output):

```markdown
# VGGT-SLAM Parity Analysis — chess_seq01

Date: 2026-05-27  
Sequence: 7-Scenes chess seq-01, 200 frames, submap_size=16  
Config: min_disparity=0 (intentional deviation — see spec)

## ATE Summary (Comparison 1)

| Pipeline | ATE (m) |
|----------|---------|
| our_baseline (VGGT-X, no LC) | 0.3184 |
| our_lc (VGGT-X + our LC) | TBD |
| vggt_slam_nolc (VGGT-1B, no LC) | TBD |
| vggt_slam_lc (VGGT-1B + VGGT-SLAM LC) | TBD |

LC algorithm contribution (our_baseline − our_lc): TBD m  
Model quality gap (our_lc − vggt_slam_lc): TBD m  
VGGT-X advantage over VGGT-1B (our_baseline − vggt_slam_nolc): TBD m

## Boundary Diff Summary (Comparison 2)

Total boundaries: TBD  
Boundaries with delta_H_w > 0.1: TBD  
Boundaries with ratio > 2.0 (solver gap): TBD  

Parity status: [ACHIEVED / PARTIAL / NOT ACHIEVED]  
(Parity = delta_H_w < 0.05 for all boundaries)

High-divergence boundaries (delta_H_w > 0.1):
<!-- list boundary IDs, delta_H_w, ratio, interpretation -->

## Loop Detection (Comparison 3)

Our LC loop count: TBD  
VGGT-SLAM loop count: TBD  
Same submap pairs detected: TBD  

## Per-Frame Correction (Comparison 4)

<!-- describe where ATE drops post-LC correction: uniform vs localised -->

## Null Hypothesis (Comparison 5)

vggt_slam_nolc_ATE vs our_baseline_ATE:  
Interpretation: TBD

## Conclusion

<!-- 2-3 sentences: is our LC algorithm at parity? where does gap remain? -->

## Next Steps

<!-- list any boundaries flagged for follow-up investigation -->
```

- [ ] **Step 2: Fill in all TBD values from Task 4-6 output**

Replace every `TBD` with actual numbers from logs.

- [ ] **Step 3: Commit**

```bash
git add worklog/notes/2026-05-27-vggt-slam-parity-analysis.md
git add evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum  # if not yet committed
git commit -m "docs(worklog): vggt-slam parity analysis — chess_seq01

All 5 comparisons: ATE breakdown, boundary diff, loop detection,
per-frame correction, null hypothesis. Parity status: [fill in]."
```
