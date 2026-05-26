# Spec: Loop Closure Validation Against VGGT-SLAM

**Date:** 2026-05-26  
**Branch:** refactor/cu121  
**Status:** approved

## Goal

Internal validation that our loop closure (LC) produces better submap alignment and trajectory accuracy than the windowed-no-LC baseline, with VGGT-SLAM as an external reference bar. Not publication-quality — trustworthy enough to guide development decisions.

## Conditions

Six conditions, evaluated head-to-head:

| condition key | backbone | extra |
|---|---|---|
| `omega_baseline` | VGGT-Omega | none |
| `omega_ba` | VGGT-Omega | bundle adjustment |
| `omega_lc` | VGGT-Omega | our LC |
| `vggtx_baseline` | VGGT-X | none |
| `vggtx_lc` | VGGT-X | our LC |
| `vggt_slam` | VGGT-SLAM internal | their LC |

Key comparisons enabled:
- `vggtx_lc` vs `vggt_slam` → fair LC comparison (matched backbone era)
- `omega_lc` vs `vggtx_lc` → Omega backbone benefit
- `omega_lc` vs `omega_baseline` → our LC improvement on Omega

VGGT-SLAM uses the original VGGT backbone internally, not Omega. This is acceptable for internal validation.

## Dataset

**7-Scenes**, sequences:
- `chess_seq01` — already present at `data/7scenes/chess/`
- `fire_seq01` — download required
- `office_seq01` — download required

TUM is not available locally and is out of scope for this eval.

## Metrics

### Phase 1 — Trajectory (primary, matches VGGT-SLAM protocol)

All computed via `evo` library on TUM-format trajectory files, Sim3 alignment for monocular methods.

| metric | description |
|---|---|
| ATE RMSE | Absolute Trajectory Error, translation RMSE (metres) |
| ATE mean | Mean per-frame translation error |
| RPE trans RMSE | Relative Pose Error, translation component |
| RPE rot RMSE | Relative Pose Error, rotation component (degrees) |
| AUC@30 | Cumulative accuracy at 30° combined R+T threshold |

### Phase 2 — Submap Alignment (our differentiator)

Computed from internal LC state (`_lc_*` attributes on `LoopClosure.base`). Reported as before/after pose-graph optimization to show LC effect directly.

| metric | description |
|---|---|
| loop_residual_before | Mean camera-position distance across accepted loop match pairs, pre-PGO |
| loop_residual_after | Same, post-PGO (should be near zero for good LC) |
| boundary_gap_before | Mean camera-position gap at consecutive submap stitches, pre-PGO |
| boundary_gap_after | Same, post-PGO |
| chamfer_before | Mean Chamfer distance between matched-submap world_points, pre-PGO |
| chamfer_after | Same, post-PGO |

Only LC conditions emit alignment metrics. Baseline/BA/VGGT-SLAM alignment metrics are `null`.

## Components

### 1. `evals/download_7scenes.py` (new)

Downloads `fire_seq01` and `office_seq01` from Microsoft 7-Scenes CDN into `data/7scenes/`. No-op if target dir already exists and contains images. Verifies frame count after extraction.

### 2. `evals/eval_gt.py` (modify)

Add `--backbone {vggtx,vggt_omega}` CLI arg (default: `vggt_omega`).

`_make_creator(condition, backbone, submap_size)` — selects creator via `get_creator(backbone)()`. Condition names gain backbone prefix in output filenames: `--backbone vggt_omega --conditions baseline ba lc` emits `omega_baseline.tum`, `omega_ba.tum`, `omega_lc.tum`.

For LC conditions: after `run_inference()`, call `compute_alignment_metrics(creator)` and write `<condition>_alignment.json` alongside the `.tum` file.

### 3. `evals/runners/run_vggt_slam.py` (replace stub)

Remove `EnvBlocked` stub. Implement real subprocess call:

```python
cmd = [
    python,  # reconstruction env Python 3.11 (default: sys.executable)
    str(VGGTSLAM_DIR / "main.py"),
    "--image_folder", str(image_dir),
    "--max_loops", "1",
    "--min_disparity", "50",
    "--conf_threshold", "25",
    "--lc_thres", "0.95",
    "--submap_size", str(submap_size),
    "--log_results",
    "--log_path", str(output_tum),
]
subprocess.run(cmd, check=True, cwd=VGGTSLAM_DIR)
```

On success: remove corresponding `.pending` sentinel from `evals/baselines/vggt_slam/`. Copy output TUM file into `evals/results/<seq>/vggt_slam.tum`.

Params match `third_party/VGGT-SLAM/evals/eval_tum.sh` defaults.

### 4. `collab_splats/pointcloud/wrappers.py` (modify)

In `LoopClosure.run_inference()`, before calling `run_pose_graph_optimization`:

```python
self.base._lc_precorrection_extrinsics = _assemble_precorrection_extrinsics(submaps, N)
```

After calling it:

```python
self.base._lc_corrected_extrinsics = corrected_extrinsics
```

`_assemble_precorrection_extrinsics(submaps, N)` is a module-level helper that stitches per-submap `.poses` into an (N, 4, 4) array using the same overlap-dedup logic already present in `closure.py` (last-writer-wins on overlapping frames), but without pose-graph correction applied. Implementor: locate the existing overlap-dedup helper in `closure.py` and reuse or extract it.

Both attributes are stored on `self.base` alongside existing `_lc_submaps`, `_lc_loop_submaps`, `_lc_all_matches` (all marked inspection-only, not stable API).

### 5. `evals/reconstruction_quality.py` (new)

Three public functions:

```python
def loop_match_residual(
    matches: list[LoopMatch],
    submaps: list[Submap],
    pre_ext: np.ndarray,   # (N, 4, 4) pre-PGO world-to-cam
    post_ext: np.ndarray,  # (N, 4, 4) post-PGO world-to-cam
) -> dict:
    """Mean/max camera-position distance for accepted loop match pairs, before and after PGO."""

def submap_boundary_gap(
    submaps: list[Submap],
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
) -> dict:
    """Mean/max camera-position gap at consecutive submap boundaries, before and after PGO."""

def pointcloud_chamfer(
    matches: list[LoopMatch],
    submaps: list[Submap],
    pre_ext: np.ndarray,
    post_ext: np.ndarray,
    max_pairs: int = 20,
) -> dict:
    """Mean Chamfer distance between matched-submap world_points, before and after PGO.
    
    Transforms submap world_points into a common frame using pre/post extrinsics.
    Caps at max_pairs to bound compute time.
    """

def compute_alignment_metrics(lc_creator) -> dict:
    """Aggregate all three metrics from a post-run LoopClosure creator instance."""
```

All functions return JSON-serializable dicts. `compute_alignment_metrics` reads `_lc_submaps`, `_lc_all_matches`, `_lc_precorrection_extrinsics`, `_lc_corrected_extrinsics` from `lc_creator.base`.

### 6. `evals/eval_compare.py` (extend)

- Add AUC@30 column: add `compute_auc(pred_path, gt_path, align, max_threshold_deg=30)` to `evals/metrics.py`. It loads TUM files via `evo`, aligns, then calls `auc_at_threshold` from `collab_splats.pointcloud.loop_closure.eval` (which already implements the CO3Dv2/VGGSfM cumulative-accuracy protocol). `eval_compare.py` calls this alongside `compute_ate` / `compute_rpe`.
- Add alignment columns: for each method, if `<method>_alignment.json` exists in results dir, read it and populate `loop_residual_before`, `loop_residual_after`, `chamfer_ratio` (= `chamfer_before / chamfer_after`) columns. Missing = `null`.

### 7. `evals/eval_suite.sh` (new)

Orchestrator. Loops over:
- sequences: `chess_seq01`, `fire_seq01`, `office_seq01`
- backbones: `vggt_omega`, `vggtx`
- calls `eval_gt.py` with appropriate `--backbone` + `--conditions`
- calls `run_vggt_slam.py` once per sequence
- calls `eval_compare.py` to emit final `metrics.json` + Markdown table per sequence

Idempotent: skips any `.tum` that already exists.

## Data Flow

```
data/7scenes/<seq>/         images + groundtruth.txt (GT)
        ↓
eval_gt.py --backbone vggt_omega   →  results/<seq>/omega_{baseline,ba,lc}.tum
                                   →  results/<seq>/omega_lc_alignment.json
eval_gt.py --backbone vggtx        →  results/<seq>/vggtx_{baseline,lc}.tum
                                   →  results/<seq>/vggtx_lc_alignment.json
run_vggt_slam.py                   →  results/<seq>/vggt_slam.tum
        ↓
eval_compare.py                    →  results/<seq>/metrics.json
                                   →  stdout Markdown table
```

## Output Table (per sequence)

```
| method          | ATE RMSE | ATE mean | RPE trans | RPE rot° | AUC@30 | loop_residual↓ | chamfer_ratio↓ |
|-----------------|----------|----------|-----------|----------|--------|----------------|----------------|
| omega_baseline  | ...      | ...      | ...       | ...      | ...    | null           | null           |
| omega_ba        | ...      | ...      | ...       | ...      | ...    | null           | null           |
| omega_lc        | ...      | ...      | ...       | ...      | ...    | X→Y            | Z              |
| vggtx_baseline  | ...      | ...      | ...       | ...      | ...    | null           | null           |
| vggtx_lc        | ...      | ...      | ...       | ...      | ...    | X→Y            | Z              |
| vggt_slam       | ...      | ...      | ...       | ...      | ...    | null           | null           |
```

## Not in Scope

- TUM dataset (no data available)
- Point cloud vs. ground-truth mesh comparison (no GT mesh)
- Statistical significance across multiple runs
- VGGT-SLAM alignment metrics (black box, no submap access)
- AUC@threshold for reconstruction quality (deferred to Phase 3 if needed)
