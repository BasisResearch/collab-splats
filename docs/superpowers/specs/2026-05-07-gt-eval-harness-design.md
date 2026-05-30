# Ground-Truth Evaluation Harness Design

**Date:** 2026-05-07  
**Status:** Approved  
**Scope:** Generic evaluation runner for BA / loop-closure pipelines against GT pose datasets

---

## Goal

End-to-end harness that:
1. Downloads a GT pose dataset (starting with 7-Scenes)
2. Runs a PCD creator under configurable conditions
3. Computes ATE + RPE against GT poses
4. Produces metrics JSON + trajectory plots

---

## Architecture

```
evals/
  download_7scenes.sh          # wget helper for 7-Scenes sequences
  datasets.py                  # get_dataset() registry + EvalDataset
  eval_gt.py                   # generic CLI runner
collab_splats/pointcloud/loop_closure/eval.py   # fill 3 existing stubs
docs/pointcloud/
  eval_7scenes_gt.ipynb        # visualization notebook (loads results, no compute)
```

---

## Data Flow

```
download_7scenes.sh
      ↓
/data/7scenes/{scene}/{seq}/
  color/frame-XXXXXX.color.png
  pose/frame-XXXXXX.pose.txt     ← cam-to-world 4×4
      ↓
get_dataset("7scenes")(seq_dir, max_frames=500)
  → EvalDataset(images=[...], gt_poses=(N,4,4) world-to-cam)
      ↓
eval_gt.py: 3 conditions
  "baseline" → VGGTXCreator(use_ba=False, enable_loop_closure=False)
  "ba"       → VGGTXCreator(use_ba=True,  enable_loop_closure=False)
  "lc"       → VGGTXCreator(use_ba=False, enable_loop_closure=True)
  each → creator.outputs.extrinsics  (N,4,4) world-to-cam
      ↓
eval.py: umeyama_align → ate_translation + rpe
      ↓
eval_results/
  metrics.json
  trajectories.npz
  plots/trajectory.png
  plots/ate_per_frame.png
```

---

## `evals/datasets.py`

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class EvalDataset:
    images: list[Path]
    gt_poses: np.ndarray   # (N, 4, 4) world-to-cam float32

def _load_7scenes(seq_dir: Path, max_frames: int = 500) -> EvalDataset:
    images = sorted((seq_dir / "color").glob("*.color.png"))[:max_frames]
    poses = np.stack([
        np.linalg.inv(np.loadtxt(seq_dir / "pose" / f"{p.stem.split('.')[0]}.pose.txt"))
        for p in images
    ])
    return EvalDataset(images=images, gt_poses=poses.astype(np.float32))

_REGISTRY = {"7scenes": _load_7scenes}

def get_dataset(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
```

Adding a new dataset: add one `_load_<name>` function + one registry entry. Zero runner changes.

**Pose convention:** 7-Scenes `.pose.txt` files are cam-to-world. `np.linalg.inv` flips to world-to-cam, matching `creator.outputs.extrinsics`.

---

## `eval.py` stubs (fill existing `NotImplementedError`)

Three stubs in `collab_splats/pointcloud/loop_closure/eval.py` already exist and document their contracts. Implementation:

### `umeyama_align(pred, gt)`
Calls existing `alignment.umeyama_se3` on camera positions (`[:, :3, 3]`). Returns `(aligned_pred, T_align)`.

### `ate_translation(pred, gt)`
1. `umeyama_align(pred, gt)` → `aligned`
2. Per-frame error: `‖aligned[:,:3,3] − gt[:,:3,3]‖₂`
3. Returns `{rmse, mean, median, max, per_frame}`.

### `rpe(pred, gt, delta=1)`
1. Relative transforms: `rel = inv(poses[:-δ]) @ poses[δ:]` for both pred and gt
2. Error: `inv(rel_gt) @ rel_pred`
3. Translation RMSE from `‖err[:,:3,3]‖`, rotation RMSE from `arccos((tr(R)−1)/2)` in degrees.
4. Returns `{trans_rmse, rot_rmse_deg}`.

---

## `evals/eval_gt.py`

### CLI

```
python evals/eval_gt.py \
  --dataset    7scenes \
  --seq_dir    /data/7scenes/chess/seq-01 \
  --output_dir ./eval_results/chess_seq01 \
  --max_frames 500 \
  --conditions baseline ba lc
```

### Condition → creator flags

| Condition  | `use_ba` | `enable_loop_closure` |
|------------|----------|-----------------------|
| `baseline` | False    | False                 |
| `ba`       | True     | False                 |
| `lc`       | False    | True                  |

### Per-condition logic

```python
creator = VGGTXCreator(use_ba=..., enable_loop_closure=..., loop_closure_config=...)
creator.run(image_dir=tmp_image_symlink_dir, output_dir=tmp_out)
pred = creator.outputs.extrinsics  # (N, 4, 4)
metrics[cond] = {
    "ate": ate_translation(pred, dataset.gt_poses),
    "rpe": rpe(pred, dataset.gt_poses),
}
```

Images are symlinked (or copied) into a temp dir so `VGGTXCreator` sees a standard image directory.

---

## Outputs

### `metrics.json`
```json
{
  "baseline": {"ate": {"rmse": 0.12, "mean": 0.10, ...}, "rpe": {"trans_rmse": 0.03, "rot_rmse_deg": 1.2}},
  "ba":       {"ate": {...}, "rpe": {...}},
  "lc":       {"ate": {...}, "rpe": {...}}
}
```

### `trajectories.npz`
Keys: `gt`, `pred_baseline`, `pred_ba`, `pred_lc` — each `(N, 4, 4)`.

### `plots/trajectory.png`
3D scatter of camera positions (translation component), all conditions + GT, coloured by condition.

### `plots/ate_per_frame.png`
Per-frame ATE error over frame index, one line per condition.

---

## Notebook `docs/pointcloud/eval_7scenes_gt.ipynb`

Load-only — no inference. Cells:
1. Load `metrics.json` → `pd.DataFrame` comparison table (ATE RMSE + RPE per condition)
2. Load `trajectories.npz` → 3D trajectory plot (matplotlib)
3. Load `trajectories.npz` → per-frame ATE plot
4. Commentary on what LC / BA gain

---

## Download

`evals/download_7scenes.sh <scene>` — wraps the MASt3R-SLAM download instructions:

```bash
# Usage: bash evals/download_7scenes.sh chess
# Downloads to ./data/7scenes/<scene>/
```

One scene at a time. Chess (~360MB) recommended for first eval run.

---

## Extending to other datasets

1. Add `evals/datasets.py`: `_load_tum(seq_dir, max_frames) -> EvalDataset`
2. Register: `_REGISTRY["tum"] = _load_tum`
3. Add `evals/download_tum.sh`

Runner, metrics, and plots unchanged.
