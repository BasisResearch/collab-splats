# Handoff: VGGT-SLAM Parity Run 3 — Fix compute_similarity + Complete Analysis

## Status

Runs 1 & 2 complete. Run 3 crashed. Need to fix and re-run, then execute CPU analysis.

## Completed Outputs (do not re-run)

| File | Size | Status |
|------|------|--------|
| `evals/results/parity_harness/vggt_slam_internals.json` | 135 KB | ✅ done |
| `evals/results/parity_harness/our_internals.json` | 132 KB | ✅ done |
| `evals/results/parity_harness/our_lc.tum` | 19 KB | ✅ done |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_dense_nolc.tum` | 19 KB | ✅ done |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` | — | ❌ MISSING — run 3 failed |

## Run 3 Failure

**Script:** `evals/runners/run_vggt_slam_lc.py`
**Command:** `python evals/runners/run_vggt_slam_lc.py --seq_dir data/7scenes/chess/seq-01 --max_frames 200`

**Crash:**
```
File "third_party/VGGT-SLAM/vggt_slam/solver.py", line 347
    predictions_lc = model(lc_frames, compute_similarity=True)
TypeError: VGGT.forward() got an unexpected keyword argument 'compute_similarity'
```

**Root cause:** VGGT-SLAM's `solver.run_predictions()` detects a loop closure and calls
`model(lc_frames, compute_similarity=True)` to get `predictions_lc["image_match_ratio"]`.
The installed VGGT-1B model (forward sig: `images, query_points=None, verbose=False`)
does not support this kwarg. None of the installed models support it (VGGT-1B, VGGT-X,
vggt-omega all checked — `compute_similarity` appears nowhere except solver.py).

**Wrong fix already in file:** `run_vggt_slam_lc.py` currently has a `_VGGTCompatWrapper`
that injects `image_match_ratio=1.0`. This is NOT the right approach per the user.

## Correct Fix (user's instruction)

Two acceptable options — pick one:

### Option A: Use VGGTXCreator's built-in similarity support

`collab_splats/pointcloud/feedforward/vggtx.py` has `_extract_qkv_and_poses()` (around
line 370) which taps attention QKV hooks to compute real frame similarity. This is the
same mechanism as VGGT-SPARK. Wrap our VGGT-X model so that when called with
`compute_similarity=True`, it runs `_extract_qkv_and_poses` on the 2 LC frames and
computes a real `image_match_ratio` (e.g. cosine similarity of CLS tokens or key vectors)
then returns it alongside the normal predictions dict.

### Option B: Download VGGT-Spark checkpoint

VGGT-Spark is a variant of VGGT that natively supports `compute_similarity=True`.
If a checkpoint is available (HuggingFace or local), load it instead of VGGT-1B
in `run_vggt_slam_lc.py`. The VGGT-SLAM solver will work as-is.

**Remove the `_VGGTCompatWrapper` class** from `run_vggt_slam_lc.py` before applying
either fix — it's dead code now.

## After Fix: Re-run Run 3

```bash
/opt/conda/envs/reconstruction/bin/python evals/runners/run_vggt_slam_lc.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200
# Output: evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum
```

Run in tmux (heavy GPU run, ~10-20 min). Log to `/tmp/lc_run.log`.

## After Run 3: CPU Analysis

```bash
PY=/opt/conda/envs/reconstruction/bin/python

# Boundary diff table
$PY evals/runners/compare_solver_internals.py

# ATE breakdown
$PY evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```

## Analysis to Write

File: `worklog/notes/2026-05-27-vggt-slam-parity-analysis.md`

For each boundary in `boundary_diff.json`:
- `ratio = delta_H_w / (delta_T + 1e-6)`
- ratio ≈ 1 → model-driven difference (VGGT-1B vs VGGT-X input diff explains H_w diff)
- ratio >> 2 → residual solver logic gap (investigate `closure.py` vs `solver.py`)
- Flag any boundary with `delta_H_w > 0.1`

ATE breakdown:
- `our_baseline` = 0.3184 m (from prior eval_gt.py run)
- `our_lc` = from `diagnose_lc_parity.py`
- `vggt_slam_lc` = from `diagnose_lc_parity.py`
- LC algorithm contribution = `our_baseline − our_lc`
- Model quality gap = `our_lc − vggt_slam_lc`

## Key Files

| Path | Notes |
|------|-------|
| `evals/runners/run_vggt_slam_lc.py` | **Fix here** — remove `_VGGTCompatWrapper`, implement real compute_similarity |
| `evals/runners/compare_solver_internals.py` | Ready |
| `evals/runners/diagnose_lc_parity.py` | Ready |
| `collab_splats/pointcloud/feedforward/vggtx.py:370` | `_extract_qkv_and_poses` — similarity mechanism |
| `third_party/VGGT-SLAM/vggt_slam/solver.py:335-390` | LC logic — calls `model(lc_frames, compute_similarity=True)` |
| `docs/superpowers/specs/2026-05-27-parity-analysis-design.md` | Full design spec |
