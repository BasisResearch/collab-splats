# VGGT-SLAM Parity Analysis Design

## Goal

Run both pipelines on chess/seq-01, collect per-boundary solver internals, and determine whether ATE differences between our LC pipeline and VGGT-SLAM are caused by model quality (VGGT vs VGGT-X) or residual solver logic gaps.

## Execution Plan

Three GPU runs on chess/seq-01, max_frames=200, sequential on GPU 1:

1. `vggt_slam_solver_dump.py` — VGGT-1B, max_loops=0 → `evals/results/parity_harness/vggt_slam_internals.json`
2. `our_solver_dump.py` — VGGT-X, submap_size=16 → `evals/results/parity_harness/our_internals.json` + `our_lc.tum`
3. `run_vggt_slam_lc.py` — VGGT-1B, max_loops=1 → `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum`

CPU analysis runs after all three complete:
- `compare_solver_internals.py` → `boundary_diff.json` + printed table
- `diagnose_lc_parity.py` → per-frame ATE table (ours vs VGGT-SLAM vs GT)

## Analysis Methodology

### Per-Boundary Attribution

For each boundary entry in `boundary_diff.json`, compute:

```
ratio = delta_H_w / (delta_T + 1e-6)
```

Interpretation:
- ratio ≈ 1: H_w divergence fully explained by different T inputs (model difference, no solver bug)
- ratio >> 2: H_w diverges beyond T — residual solver logic gap; investigate that boundary in `closure.py` vs `solver.py`

Flag any boundary with `delta_H_w > 0.1` (high divergence regardless of ratio).

### ATE Breakdown

| Source | Value | How obtained |
|--------|-------|-------------|
| our_baseline | 0.3184 m | Previous eval_gt.py run |
| our_lc | TBD | diagnose_lc_parity.py with our_lc.tum vs GT |
| vggt_slam_lc | TBD | diagnose_lc_parity.py with vggt_slam_lc.tum vs GT |

Attribution:
- **LC algorithm contribution** = `our_baseline − our_lc` (improvement from our LC over raw feedforward)
- **Model quality gap** = `our_lc − vggt_slam_lc` (residual gap after LC, driven by VGGT vs VGGT-X)

### Code-Level Investigation (if needed)

If any boundary has ratio > 2:
- Read `closure.py:run_pose_graph_optimization` and `solver.py:add_edge` side-by-side
- Focus on H_w formula: `H_w = H_overlap @ T @ H_scale`
- Check scale extraction and any post-PGO correction steps

## Output

`worklog/notes/2026-05-27-vggt-slam-parity-analysis.md` containing:
- Full boundary diff table with ratio column
- ATE breakdown table
- Gap attribution conclusion (model-driven vs solver-logic)
- Next steps (fix residual bugs if found, or confirm parity achieved)

## Execution Commands

```bash
# GPU runs (sequential, each in tmux)
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/reconstruction/bin/python \
  evals/runners/vggt_slam_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200

CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/reconstruction/bin/python \
  evals/runners/our_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200

CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/reconstruction/bin/python \
  evals/runners/run_vggt_slam_lc.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200

# CPU analysis
/opt/conda/envs/reconstruction/bin/python evals/runners/compare_solver_internals.py
/opt/conda/envs/reconstruction/bin/python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```
