# VGGT-SPARK Integration + Full Parity Comparison Suite

**Amends:** `2026-05-27-vggt-slam-parity-verification-design.md` (Phase 1 runner) and  
`2026-05-27-parity-analysis-design.md` (analysis methodology)  
**Status:** Approved — supersedes the `_VGGTCompatWrapper` approach

---

## History

Our LC pipeline (`collab_splats/pointcloud/loop_closure/closure.py`) was built to mirror
VGGT-SLAM's `vggt_slam/solver.py`. Three algorithmic bugs were fixed along the way:
H_w inter-submap formula, pose extraction method, and confidence filtering
(see `2026-05-26-lc-vggtslam-parity-fixes.md`). However outputs were only compared
against GT — never directly against a live VGGT-SLAM run.

`2026-05-27-vggt-slam-parity-verification-design.md` defined a two-phase harness
(ATE comparison + internal state diff) and produced runs 1 & 2 (solver dumps).
Run 3 (`run_vggt_slam_lc.py`) crashed with
`TypeError: VGGT.forward() got an unexpected keyword argument 'compute_similarity'`
and was patched with `_VGGTCompatWrapper` (image_match_ratio=1.0 stub) — which silences
the crash but forces every LC candidate to pass, producing an invalid baseline TUM.

This spec fixes run 3 via VGGT-SPARK and defines the full comparison suite to execute
once a valid `vggt_slam_lc.tum` exists.

---

## Goal

Produce a valid `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` by running the
MIT-SPARK VGGT-SLAM pipeline with a model that natively supports `compute_similarity=True`,
then run all comparisons to characterise parity between our LC implementation and theirs.

---

## Problem with Current State

`run_vggt_slam_lc.py` contains `_VGGTCompatWrapper` which injects `image_match_ratio=1.0`
when `compute_similarity=True` is called. This is wrong: solver checks
`image_match_ratio > lc_thres` (default 0.95), so 1.0 forces **every** loop candidate to
pass regardless of actual frame similarity. Resulting TUM trajectory has false loop
closures → parity analysis is invalid.

---

## Decision: VGGT-SPARK via sys.path shadow

### Why VGGT-SPARK (not Option A QKV wrapper)

Option A wraps our VGGT-X model to compute similarity via attention QKV hooks. This
produces real similarity values but from VGGT-X, not VGGT-1B. VGGT-SLAM was designed and
tuned with VGGT-SPARK (MIT-SPARK's fork of VGGT-1B). Using VGGT-SPARK gives the oracle
baseline that matches published VGGT-SLAM results exactly.

VGGT-SPARK properties (confirmed):
- Same weights as VGGT-1B (`facebook/VGGT-1B` checkpoint loads directly)
- Same API — additive only: adds `compute_similarity=True` kwarg to `VGGT.forward()`
- Returns `predictions["image_match_ratio"]` (real attention-based similarity)
- Public repo: `https://github.com/MIT-SPARK/VGGT_SPARK`

### Why sys.path shadow (not separate conda env)

A separate conda env adds ~8-10 GB disk, requires activation discipline across multiple
runner scripts, and is unnecessary since VGGT-SLAM uses no custom CUDA extensions
(pure PyTorch — cu121/torch 2.5.1 confirmed compatible via runs 1 & 2).

The sys.path shadow is per-process and isolated. `run_vggt_slam_lc.py` is a standalone
script; inserting `third_party/vggt_spark/` at `sys.path[0]` before any `vggt` import
causes Python to load VGGT-SPARK for that process only. `sys.modules` cache propagates
to all VGGT-SLAM internal imports in the same process. `reconstruction` env's installed
`vggt` package (VGGT-X based) is completely unaffected.

---

## Setup

Clone VGGT-SPARK as an uninstalled source tree (no pip install):

```bash
cd third_party
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt_spark
cd ..
```

Verify package structure exists:

```bash
ls third_party/vggt_spark/vggt/__init__.py  # must exist
```

No changes to `setup/feedforward.sh` needed for the eval runner. If this becomes
part of the standard setup, add a `third_party/vggt_spark` clone step there.

---

## Changes to `run_vggt_slam_lc.py`

### 1. sys.path shadow — top of file, before all imports

```python
# Shadow installed vggt with VGGT-SPARK (MIT-SPARK fork that supports compute_similarity).
# This must come before any vggt import. Affects this process only — reconstruction env
# vggt package (VGGT-X) is unaffected.
import sys
from pathlib import Path
_vggt_spark = str(Path(__file__).resolve().parents[2] / "third_party" / "vggt_spark")
sys.path.insert(0, _vggt_spark)
```

### 2. Remove `_VGGTCompatWrapper`

Delete the entire class definition and the `model = _VGGTCompatWrapper(_vggt)` line.
Replace with direct assignment:

```python
model = _vggt  # VGGT-SPARK natively handles compute_similarity=True
```

### 3. Keep VGGT-1B weights

```python
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_vggt = VGGT()
_vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```

Same checkpoint — VGGT-SPARK's `VGGT` class loads VGGT-1B weights unchanged.

---

## min_disparity=0 — intentional deviation from paper defaults

VGGT-SLAM paper uses `min_disparity=50` (optical flow keyframe filter). On chess_seq01
with `--max_frames 200`, `min_disparity=50` rejects enough frames that **no loop closure
candidates are detected** — making the parity comparison vacuous.

Decision: both `run_vggt_slam_lc.py` and `our_solver_dump.py` use `min_disparity=0`
(accept all frames). This deviates from the paper's deployment defaults but is correct
for this evaluation: both systems see identical frame sets, loop closures are triggered,
and the comparison isolates algorithm parity rather than keyframe selection differences.

This deviation is intentional and must not be "fixed" back to 50.

---

## Correctness Check: runner vs main.py

| Aspect | VGGT-SLAM main.py | run_vggt_slam_lc.py | Status |
|--------|-------------------|---------------------|--------|
| optical flow | `use_optical_flow_downsample = True` hardcoded | `min_disparity <= 0.0` bypass | ✓ intentional (see above) |
| dtype | `torch.bfloat16` hardcoded | adaptive bfloat16/float16 | ✓ equivalent on A100 |
| image sort | `glob.glob` (no sort) | `glob.glob` (no sort) | ✓ consistent |
| solver init | `Solver(conf, lc_thres, vis_voxel_size)` | same | ✓ |
| `run_predictions` call | `(names, model, max_loops, clip_model, clip_preprocess)` | same, clip=None | ✓ |
| `add_points` + `optimize` order | sequential | same | ✓ |
| overlap reset | `image_names_subset[-overlapping_window_size:]` | same | ✓ |
| TUM output | `write_poses_to_file(..., kitti_format=False)` | same | ✓ |

---

## Full Comparison Suite

All five comparisons run after `vggt_slam_lc.tum` is produced. They answer different
questions about where our LC diverges from VGGT-SLAM's.

### Comparison 1 — ATE trajectory comparison (Phase 1)

**Script:** `diagnose_lc_parity.py`  
**Question:** How much does LC help, and how much gap remains vs VGGT-SLAM?

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| LC algorithm contribution | `our_baseline_ATE − our_lc_ATE` | improvement our LC delivers over raw feedforward |
| Model quality gap | `our_lc_ATE − vggt_slam_lc_ATE` | residual gap attributable to VGGT-X vs VGGT-1B |
| Parity gap | `our_lc_ATE − vggt_slam_lc_ATE` split by frame region | identifies if gap is uniform or concentrated in LC-corrected windows |

Known values: `our_baseline = 0.3184 m`. `our_lc` and `vggt_slam_lc` TBD from run.

If LC algorithm contribution ≈ 0, LC is not firing or not correcting effectively — debug
loop detection threshold or pose graph optimizer. If model quality gap dominates, our LC
algorithm is correct but limited by VGGT-X vs VGGT-1B feature quality.

### Comparison 2 — Internal solver state diff (Phase 2)

**Script:** `compare_solver_internals.py` over `vggt_slam_internals.json` + `our_internals.json`  
**Question:** At each submap boundary, do our intermediate world poses match VGGT-SLAM's?

Per-boundary metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `delta_T` | L2 distance of translation component of boundary pose | model-driven divergence (different model → different raw poses) |
| `delta_H_w` | Frobenius norm of H_w diff after LC correction | combined model + solver divergence |
| attribution ratio | `delta_H_w / (delta_T + 1e-6)` | ratio ≈ 1 → model drives the gap; ratio >> 2 → solver logic gap |

Flag threshold: `delta_H_w > 0.1` = high divergence regardless of attribution.  
Parity target: `delta_H_w < 0.05` for all boundaries.

If ratio >> 2 at a boundary: inspect `closure.py` `run_pose_graph_optimization` vs
`solver.py` around the same submap index. Prior known bugs: H_w formula
(inter-submap scale), pose extraction (`decompose_camera` vs direct), confidence filtering.

### Comparison 3 — Loop closure detection agreement

**Question:** Do both systems detect the same loop closure candidates on chess_seq01?

Extracted from solver logs / internals JSON:
- Total loop count: ours vs VGGT-SLAM
- Which submap pairs were matched
- `image_match_ratio` values (ours via `_extract_qkv_and_poses`; theirs via VGGT-SPARK)

If loop counts differ significantly: either retrieval (salad-based VLAD similarity threshold)
or disparity gating is causing different candidate sets. Since both use `min_disparity=0`
and `lc_thres=0.95`, differences point to retrieval vector quality.

### Comparison 4 — Per-frame ATE before/after LC window

**Question:** Does LC correction localise improvement to the right frames?

From `diagnose_lc_parity.py` per-frame output: plot ATE per frame for
`our_baseline` / `our_lc` / `vggt_slam_lc`. Expected: ATE drops sharply in the submap
containing the corrected loop and stays lower for remaining submaps. If correction is
global (all frames improve equally), the pose graph propagation is working. If only
the LC submap improves, propagation may not be spreading through the graph.

### Comparison 5 — Null hypothesis check (no-LC baseline)

**Question:** Does VGGT-SLAM without LC match our baseline?

`evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum` exists (run 1 ✅).
Compare its ATE to `our_baseline = 0.3184 m`. If VGGT-SLAM no-LC ATE >> ours, VGGT-X
is simply a stronger model. If approximately equal, differences in LC results are
algorithm-driven.

---

## Execution Order

GPU runs are sequential (OOM risk if parallel). All use `reconstruction` env.

```bash
PY=/opt/conda/envs/reconstruction/bin/python

# ── Step 1: Run 3 — VGGT-SLAM with LC (GPU, ~10-15 min in tmux) ──────────────
tmux new-session -d -s vggt_slam_run3
tmux send-keys -t vggt_slam_run3 \
  "$PY evals/runners/run_vggt_slam_lc.py \
    --seq_dir data/7scenes/chess/seq-01 \
    --out_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
    --max_frames 200 2>&1 | tee /tmp/run3.log; echo RUN3_DONE" Enter

# ── Step 2: CPU analysis (after run 3 completes) ──────────────────────────────

# Comparison 2: internal boundary diff table
$PY evals/runners/compare_solver_internals.py

# Comparisons 1 + 3 + 4: ATE table, per-frame, loop detection
$PY evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```

Runs 1 & 2 (solver dumps) are already complete — do not re-run.

---

## Output Artefacts

| File | Status | Used by |
|------|--------|---------|
| `evals/results/parity_harness/vggt_slam_internals.json` | ✅ done | Comparison 2 |
| `evals/results/parity_harness/our_internals.json` | ✅ done | Comparison 2 |
| `evals/results/parity_harness/our_lc.tum` | ✅ done | Comparisons 1, 4 |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum` | ✅ done | Comparison 5 |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` | ❌ blocked (run 3) | Comparisons 1, 3, 4 |
| `evals/results/parity_harness/boundary_diff.json` | ❌ depends on above | Comparison 2 |

---

## Success Criteria

| Check | Pass condition |
|-------|----------------|
| Run 3 completes | No `TypeError: compute_similarity`; TUM has ~200 entries |
| LC triggered | Solver log: loop closure count > 0; `image_match_ratio` values < 1.0 |
| Comparison 2 (boundary diff) | `delta_H_w < 0.05` for all boundaries = parity achieved |
| Comparison 1 (ATE) | `our_lc_ATE < our_baseline_ATE` (LC helps); gap to `vggt_slam_lc` attributed |
| Comparison 5 (null hypothesis) | VGGT-SLAM no-LC ATE documented and compared to `our_baseline` |

Boundaries with `delta_H_w > 0.1` are flagged for follow-up investigation in
`worklog/notes/2026-05-27-vggt-slam-parity-analysis.md`.
