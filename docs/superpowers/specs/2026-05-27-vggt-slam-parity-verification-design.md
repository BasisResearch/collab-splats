# VGGT-SLAM Parity Verification Design

## Goal

Verify that our LC pipeline produces outputs that match VGGT-SLAM on a ground-truth dataset (7-Scenes chess/seq-01), at two levels: end-to-end trajectory ATE and per-boundary solver internals (scale, H_w, H_opt, extracted poses).

## Background

Our LC pipeline (`collab_splats/pointcloud/loop_closure/closure.py`) was built to mirror VGGT-SLAM's `vggt_slam/solver.py`. Three bugs were fixed (H_w formula, pose extraction, confidence filtering). However, we never directly compared outputs against a VGGT-SLAM run — only against GT. This spec defines how to close that gap.

## Architecture

Two phases, run sequentially:

**Phase 1 — End-to-end ATE comparison.** Run VGGT-SLAM on chess/seq-01 with LC enabled. Use existing `diagnose_lc_parity.py` to compare VGGT-SLAM trajectory vs ours vs GT per-frame. Establishes whether an end-to-end gap exists and where drift accumulates.

**Phase 2 — Per-boundary solver internals diff.** Run both pipelines on same images, capture intermediate values (scale, H_w, H_opt, extracted poses) per submap boundary from each, compute numerical diff table. Isolates exactly which boundary and which step diverges.

## Files

```
evals/
  baselines/vggt_slam/chess_seq01/
    vggt_slam_lc.tum                    ← Phase 1 output (committed; VGGT-SLAM with LC, dense)
  results/parity_harness/               ← Phase 2 outputs (gitignored via evals/results/)
    vggt_slam_internals.json
    our_internals.json
    our_lc.tum                          ← our pipeline's dense TUM (Phase 1 + 2)
    boundary_diff.json
  runners/
    diagnose_lc_parity.py               ← existing; add --ours_tum arg only
    run_vggt_slam_lc.py                 ← new: runs VGGT-SLAM with LC, saves TUM
    vggt_slam_solver_dump.py            ← new: monkey-patches Solver.add_edge, dumps internals
    our_solver_dump.py                  ← new: runs our LC pipeline with debug_out, dumps JSON + TUM
    compare_solver_internals.py         ← new: loads both dumps, prints diff table + writes JSON
```

## Phase 1 Detail

### `run_vggt_slam_lc.py`

Thin wrapper around `third_party/VGGT-SLAM/main.py`. Fixed args:
- `--submap_size 16`
- `--overlapping_window_size 1`
- `--conf_threshold 25.0`
- `--max_loops 1`
- `--log_results`
(No `--skip_dense_log` — we need the dense per-frame trajectory for ATE comparison)

CLI: `python evals/runners/run_vggt_slam_lc.py --seq_dir data/7scenes/chess/seq-01 --max_frames 200`

Output: `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` (TUM format, c2w, integer timestamps).

### Modification to `diagnose_lc_parity.py`

Add `--ours_tum` optional CLI argument (TUM file for our pipeline's output). When provided, loads it with `_load_tum()` and passes as `ours_w2c` to `diagnose()`. When absent, falls back to existing identity-placeholder behaviour. No other changes to the script.

### Running Phase 1 comparison

```bash
python evals/runners/run_vggt_slam_lc.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200

# our_solver_dump.py writes our_lc.tum as a side-effect
python evals/runners/our_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200

python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16
```

## Phase 2 Detail

### `vggt_slam_solver_dump.py`

Monkey-patches `vggt_slam.solver.Solver.add_edge` to intercept and capture per-boundary values before returning. Captured dict per boundary:

```python
{
    "submap_id": int,
    "scale": float,
    "H_w": [[4x4 float]],       # initial H for first node of new submap
    "H_overlap": [[4x4 float]], # graph H of prev overlap node at time of edge add
    "T": [[4x4 float]],         # inv(P_prev_ov) @ P_curr_ov
}
```

After the full pipeline runs (all submaps added + PGO optimized), also captures per-node `H_opt` (post-optimization) and extracted poses by calling `graph.get_homography(nid)` for each node.

CLI: `python evals/runners/vggt_slam_solver_dump.py --seq_dir data/7scenes/chess/seq-01 --max_frames 200`

Output: `evals/results/parity_harness/vggt_slam_internals.json`

### `our_solver_dump.py`

Runs our VGGT-X pipeline on same images with `debug_out=[]` passed to `run_pose_graph_optimization`. Serializes `debug_out` entries (already contain `scale`, `H_w`, `H_overlap`, `T`, `H_opt`, `corrected_proj` per boundary) to JSON. Also captures final extracted poses.

CLI: `python evals/runners/our_solver_dump.py --seq_dir data/7scenes/chess/seq-01 --max_frames 200`

Output: `evals/results/parity_harness/our_internals.json`

### `compare_solver_internals.py`

Loads both JSON files. For each boundary index present in both:

```
Boundary | delta_scale | delta_H_w(Frob) | delta_T(Frob) | delta_H_opt(Frob) | note
       0 |      0.0001 |          0.0023 |        0.0001 |            0.0041 |
       1 |      0.0032 |          0.0891 |        0.0003 |            0.2341 | ← diverges
```

- `delta_scale` = |scale_ours - scale_slam|
- `delta_H_w` = Frobenius norm of (H_w_ours - H_w_slam)
- `delta_T` = Frobenius norm of (T_ours - T_slam)
- `delta_H_opt` = Frobenius norm of (H_opt_ours - H_opt_slam), per boundary's first node

Prints table to stdout. Writes `evals/results/parity_harness/boundary_diff.json`.

A boundary flagged with `←` means delta_H_w > 0.01 (configurable threshold). This is the primary diagnostic signal.

## Intermediate Value Schema

Both dump files use the same schema:

```json
{
  "config": {"seq_dir": "...", "max_frames": 200, "submap_size": 16, "conf_threshold": 25.0},
  "boundaries": [
    {
      "boundary_idx": 0,
      "submap_id": 1,
      "scale": 1.0023,
      "H_w": [[...]],
      "H_overlap": [[...]],
      "T": [[...]],
      "H_opt": [[...]]
    }
  ],
  "final_poses": [[[4x4]], ...]
}
```

numpy arrays serialized as nested Python lists (json.dumps compatible). Load with `np.array(entry["H_w"])`.

## Success Criteria

1. Phase 1: `diagnose_lc_parity.py` runs without error and produces per-frame ATE table comparing VGGT-SLAM LC vs ours vs GT.
2. Phase 2: `compare_solver_internals.py` prints a boundary diff table. `delta_H_w < 0.05` for all boundaries = parity achieved. Any boundary with `delta_H_w > 0.1` = known divergence point, needs separate investigation.
3. All four runner scripts have `--help` and run end-to-end without error.

## Testing

No unit tests for the runner scripts themselves — they are diagnostic tools. The existing `tests/pointcloud/` suite covers the production LC code. Runners should be manually verified to produce non-empty JSON output.

## Execution Order

```bash
# Phase 1
python evals/runners/run_vggt_slam_lc.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200
python evals/runners/our_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200
python evals/runners/diagnose_lc_parity.py \
  --seq_dir data/7scenes/chess/seq-01 \
  --vggt_slam_tum evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum \
  --ours_tum evals/results/parity_harness/our_lc.tum \
  --max_frames 200 --submap_size 16

# Phase 2
python evals/runners/vggt_slam_solver_dump.py \
  --seq_dir data/7scenes/chess/seq-01 --max_frames 200
python evals/runners/compare_solver_internals.py \
  --slam_dump evals/results/parity_harness/vggt_slam_internals.json \
  --our_dump evals/results/parity_harness/our_internals.json
```

## Constraints

- `run_vggt_slam_lc.py` imports from `third_party/VGGT-SLAM/` — must add that path to `sys.path` (same pattern as `diagnose_lc_parity.py` repo-root guard).
- `vggt_slam_solver_dump.py` modifies VGGT-SLAM code via monkey-patching, not by editing `third_party/`. This avoids polluting the vendored code.
- `evals/results/parity_harness/` is gitignored (under `evals/results/`).
- `evals/baselines/vggt_slam/chess_seq01/vggt_slam_lc.tum` is committed to the repo as a reference artifact once produced.
- Python env: `/opt/conda/envs/reconstruction/bin/python` for all scripts.
