# Eval Reorganization Design

**Date:** 2026-05-20  
**Status:** Approved

## Problem

- `eval_results/` at repo root has ad-hoc naming, not gitignored, grows without structure
- `ba_hightrack` condition is opaque — encodes two concrete parameters, not a concept
- Notebook cells use hardcoded globals (`DATASET = "co3dv2"`) instead of callable functions
- Two per-dataset notebooks (`eval_7scenes_gt.ipynb`, `eval_co3dv2_gt.ipynb`) duplicate structure and have no shared concepts/metrics explanation
- No documentation of what conditions mean, what metrics measure, or how to run an eval

## Approach

Option A (Notebook-first): single `ground-truth-evals.ipynb` is the entry point for running, loading, and understanding evals. `eval_gt.py` stays as the headless workhorse. `docs/pointcloud/README.md` is the short written doc.

## Changes

### 1. `evals/results/` — new results directory

- Move all new results to `evals/results/` (gitignored via `.gitignore`)
- Existing `eval_results/` at repo root: leave as-is (historical, not migrated)
- Nested structure: `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/`
  - `dataset` = e.g. `co3dv2`, `7scenes`, `tum`
  - `seq_name` = last component of `--seq_dir` (e.g. `apple`, `chess`, `freiburg1_desk`)
  - `run-{YYYYMMDD-HHMMSS}` = timestamp at eval start
- Auto-name when `--output_dir` is omitted: timestamp generated at script entry

### 2. `eval_gt.py` — condition naming + auto-naming

**Rename `ba_hightrack` → parameterized `ba_track-density-{N}`**

- Parse `ba_track-density-*` dynamically: extract N from condition string, pass as `max_query_pts` to `BundleAdjustmentConfig`
- Default BA (`ba`) keeps existing defaults: `max_query_pts=2048`, `query_frame_num=5`
- `ba_track-density-4096` = former `ba_hightrack`: `max_query_pts=4096`, `query_frame_num=8`
- `query_frame_num` scales with density: `max(5, N // 512)` — no separate knob needed
- Condition validation: accepts `baseline`, `ba`, `lc`, and any string matching `ba_track-density-\d+`

**`--output_dir` becomes optional** (`required=False, default=None`). When omitted, auto-named as below. Subprocess invocations inside `main()` always pass explicit `--output_dir`, so no change needed there.

**Auto-naming (when `--output_dir` omitted):**
```python
from datetime import datetime
seq_name = Path(args.seq_dir).name
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path("evals/results") / args.dataset / seq_name / f"run-{ts}"
```

### 3. `docs/pointcloud/ground-truth-evals.ipynb` — unified notebook

Replaces `eval_7scenes_gt.ipynb` and `eval_co3dv2_gt.ipynb` (both deleted).

**Structure — five sections:**

#### Section 1: Concepts

Narrative markdown cells explaining:

- **What we evaluate**: camera pose estimation accuracy. Given N frames, each pipeline predicts a camera pose (position + orientation) per frame. We compare against ground-truth poses from the dataset.
- **Metrics**:
  - `ATE RMSE` — Absolute Trajectory Error (root mean squared). "On average, how far off was the predicted camera position from the true position, across all frames?" After optimal rigid alignment between predicted and GT trajectories. Lower = better. Units: metres.
  - `RPE trans RMSE` — Relative Pose Error on translation. "Between each consecutive pair of frames, how much does the predicted motion differ from the true motion?" Captures local drift rather than global offset. Lower = better. Units: metres.
  - `AUC@30` — "What fraction of frames had position error under 30 cm?" Reported as a percentage (0–100). Higher = better. 100 = every frame within 30 cm of ground truth.
- **Conditions** (what pipeline produced the poses):

  | Condition | What it runs | When to use |
  |-----------|-------------|-------------|
  | `baseline` | VGGT-X feedforward only — predicts camera poses directly from images, no refinement | Starting point; fastest to run |
  | `ba` | `baseline` + bundle adjustment — refines poses by minimising reprojection error across 2D feature tracks | Default improvement step; moderate cost |
  | `ba_track-density-{N}` | Same as `ba` but extracts N feature tracks per frame (e.g. `ba_track-density-4096` uses 4096 vs default 2048) | When `ba` leaves residual drift; slower |
  | `lc` | Full loop closure — detects when the camera revisits a place and corrects accumulated drift globally via pose graph optimisation | Long sequences with loops (e.g. walking around an object) |

- **When to use `--submap_size`**: VGGT-X loads all frames into GPU at once. Sequences longer than ~200 frames exceed GPU memory. Pass `--submap_size 50` to process in overlapping windows of 50 frames. `baseline` and `ba` run per-window; `lc` stitches windows with loop closure.

- **What "error" means**: All metrics compare predicted camera positions to ground-truth positions recorded by the dataset (e.g. a motion capture system or RGB-D sensor). Lower error = the pipeline estimated where the camera was more accurately.

#### Section 2: Run an eval

Functions, not globals:

```python
def run_eval(
    dataset: str,
    seq_dir: str | Path,
    conditions: list[str] = ["baseline", "ba", "lc"],
    submap_size: int | None = None,
    max_frames: int = 500,
    results_root: str | Path = "evals/results",
) -> Path:
    """Run eval_gt.py and return the output directory path.
    results_root is relative to repo root — notebook must be run from repo root."""
    ...
```

Example call cell below the function:
```python
out = run_eval(
    dataset="co3dv2",
    seq_dir="/data/co3dv2/apple/110_13051_23361",
    conditions=["baseline", "ba", "ba_track-density-4096"],
    submap_size=50,
)
```

#### Section 3: Load + compare results

```python
def load_results(
    results_root: str | Path = "evals/results",
    dataset: str | None = None,
    seq_name: str | None = None,
) -> dict[str, dict]:
    """Discover all run-{YYYYMMDD-HHMMSS} dirs containing metrics.json.
    Returns {"{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}": metrics_dict}.
    Optionally filter by dataset and/or seq_name.
    results_root is relative to repo root — notebook must be run from repo root."""
    ...

def compare_runs(
    run_dirs: list[str | Path] | None = None,
    results_root: str | Path = "evals/results",
) -> "pd.DataFrame":
    """Cross-run + cross-condition summary table (ATE/RPE/AUC/runtime).
    If run_dirs=None, discovers all runs under results_root.
    Pass a single-element list to inspect one run: compare_runs([run_dir])."""
    ...
```

#### Section 4: Trajectory plots

```python
def plot_trajectories(
    run_dir: str | Path,
    conditions: list[str] | None = None,
) -> None:
    """3D trajectory comparison: GT (black) vs each condition."""
    ...
```

Loads `trajectories.npz` keys `gt`, `pred_{cond}`.

#### Section 5: Per-frame ATE

```python
def plot_ate_per_frame(
    run_dir: str | Path,
    conditions: list[str] | None = None,
) -> None:
    """Bar/line chart of per-frame ATE for each condition."""
    ...
```

Loads `ate_per_frame_{cond}` arrays from `trajectories.npz`.

### 4. `docs/pointcloud/README.md` — written overview

~1 page covering:
- What ground-truth eval measures (one paragraph)
- Supported datasets: 7scenes, TUM, CO3Dv2, KITTI, Waymo — download scripts in `evals/datasets/`
- How to run headlessly (one `eval_gt.py` invocation example)
- Pointer to `ground-truth-evals.ipynb` for interactive use

### 5. Files retired

| File | Action |
|------|--------|
| `docs/pointcloud/eval_7scenes_gt.ipynb` | Deleted |
| `docs/pointcloud/eval_co3dv2_gt.ipynb` | Deleted |

### 6. `.gitignore` addition

```
evals/results/
```

## Output format (unchanged)

Each `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/` contains:
```
metrics.json          # ATE/RPE/AUC/runtime per condition
trajectories.npz      # gt, pred_{cond}, ate_per_frame_{cond} arrays
plots/
  trajectory.png
  ate_per_frame.png
{condition}/
  transforms.json
  sparse_pc.ply
```

## Non-goals

- Migrating existing `eval_results/` to new location
- YAML eval configs
- CI/automated eval runs
- Supporting datasets beyond the existing five
