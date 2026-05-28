# Full-Sequence Keyframe-Parity Eval — Design Spec

**Date:** 2026-05-28  
**Status:** approved

## Context

Issue #43 on MIT-SPARK/VGGT-SLAM confirms chess/seq-01 ATE ~0.038m is achievable with
default VGGT-SLAM settings. Our current eval uses `max_frames=200` and `min_disparity=0`,
giving 0.2967m baseline — 8× worse due to compounding scale drift across ~13 submaps.

Root cause: `min_disparity=50` (VGGT-SLAM default) filters chess frames via RAFT optical
flow to ~30-50 keyframes → ~2 submaps → minimal inter-submap drift. We capped at 200 frames
with no disparity filter, creating many more boundaries.

Goal: reproduce ~0.038m on chess/seq-01, then run VGGT-X / Omega / MapAnything on the
exact same keyframe set for an apples-to-apples comparison (no LC in this first run).

## Goals

1. Reproduce VGGT-SLAM ~0.038m chess/seq-01 ATE (validates our runner against upstream).
2. Run all three feedforward models on the exact same keyframes.
3. Establish a clean baseline table before any LC investigation.

## Scope and non-goals

**In scope:**
- `run_vggt_slam_lc.py`: remove max_frames cap, fix min_disparity default, save keyframe list, add inline ATE
- `eval_gt.py`: add `--keyframe_list` arg
- One eval run: chess/seq-01, no LC, all three models

**Not in scope:**
- LC evaluation (separate follow-on)
- Other scenes / sequences
- Scale drift or Omega gate fixes (separate spec exists)

## Architecture

### 1. `evals/runners/run_vggt_slam_lc.py` changes

**`max_frames` → optional**

```python
def run_vggt_slam_lc(
    seq_dir: Path,
    out_tum: Path,
    max_frames: int | None = None,   # None = process all frames
    min_disparity: float = 50.0,     # VGGT-SLAM paper default
    ...
):
    all_images = sorted(all_images)
    if max_frames is not None:
        all_images = all_images[:max_frames]
```

CLI: `--max_frames` default changes to `None`; pass `--max_frames 200` to get old behavior.

**Save selected keyframes**

After filtering via `flow_tracker.compute_disparity`, write accepted paths to disk:

```python
# After frame-selection loop, before solver calls
if save_keyframes:
    kf_path = out_tum.parent / "selected_frames.txt"
    kf_path.write_text("\n".join(selected_image_paths))
    logger.info("Keyframes (%d) → %s", len(selected_image_paths), kf_path)
```

Add `--save_keyframes` CLI flag (default True when `min_disparity > 0`).

**Inline ATE computation**

After `write_poses_to_file`, load GT poses from `<seq_dir>/frame-*.pose.txt` and compute
ATE RMSE using Umeyama alignment (with scale). Log result to console and write to
`metrics.json` alongside the TUM file.

```python
ate_rmse = _compute_ate(out_tum, seq_dir)
logger.info("ATE RMSE: %.6f m", ate_rmse)
(out_tum.parent / "metrics.json").write_text(json.dumps({"ate_rmse": ate_rmse}))
```

Use `evo` library (`evo.tools.file_interface`, `evo.main_ape`) — already installed.

**New output path for full-seq runs**

Default `--out_tum` changes to:
`evals/baselines/vggt_slam/<scene>/vggt_slam_fullseq.tum`

Old default `vggt_slam_lc.tum` preserved via explicit CLI arg.

### 2. `evals/eval_gt.py` changes

Add `--keyframe_list` argument:

```python
parser.add_argument(
    "--keyframe_list",
    default=None,
    help="Path to selected_frames.txt from run_vggt_slam_lc.py. "
         "When provided, overrides frame loading to use exact same keyframes.",
)
```

When `--keyframe_list` is given, load the dataset normally then filter to only frames
whose image path appears in the keyframe list. GT poses come from dataset as usual — no
separate loading path needed.

```python
if args.keyframe_list:
    allowed = set(Path(args.keyframe_list).read_text().splitlines())
    frames = [f for f in frames if f.image_path in allowed]
```

### 3. ATE helper `_compute_ate`

```python
def _compute_ate(tum_path: Path, seq_dir: Path) -> float:
    """Umeyama-aligned ATE RMSE between TUM trajectory and 7-Scenes GT poses."""
    from evo.tools import file_interface
    from evo.core import metrics, sync
    import evo.main_ape as main_ape

    traj_est = file_interface.read_tum_trajectory_file(str(tum_path))
    traj_ref = _load_gt_tum(seq_dir)   # build TUM from frame-*.pose.txt
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est,
                          est_name="est",
                          pose_relation=metrics.PoseRelation.translation_part,
                          align=True, correct_scale=True)
    return float(result.stats["rmse"])
```

`_load_gt_tum` reads `frame-XXXXXX.pose.txt` (4×4 cam-to-world), converts to TUM format
with frame index as timestamp.

## File map

| File | Change |
|------|--------|
| `evals/runners/run_vggt_slam_lc.py` | max_frames optional, min_disparity=50 default, save_keyframes, inline ATE |
| `evals/eval_gt.py` | add `--keyframe_list` arg + `_load_frames_from_keyframe_list` |
| `evals/baselines/vggt_slam/chess_seq01/vggt_slam_fullseq.tum` | new output (gitignored) |
| `evals/baselines/vggt_slam/chess_seq01/selected_frames.txt` | new output (gitignored) |
| `evals/baselines/vggt_slam/chess_seq01/metrics.json` | new output (gitignored) |

## Success criteria

1. VGGT-SLAM full-seq run completes, writes TUM with ~(full-seq frame count) poses
2. Inline ATE RMSE ≈ 0.037–0.040m on chess/seq-01 (matches issue #43)
3. VGGT-X run with `--keyframe_list` completes and produces ATE
4. Omega and MapAnything runs complete (MapAnything single-pass, LC disabled)
5. Comparison table: VGGT-SLAM vs VGGT-X vs Omega vs MapAnything, same keyframes

## Implementation order

1. `_compute_ate` + `_load_gt_tum` helpers (needed by runner and eval_gt)
2. `run_vggt_slam_lc.py` changes (max_frames, min_disparity, save_keyframes, ATE)
3. Run VGGT-SLAM → confirm ~0.038m
4. `eval_gt.py` `--keyframe_list` support
5. Run VGGT-X, Omega, MapAnything with same keyframes
6. Record comparison table in worklog
