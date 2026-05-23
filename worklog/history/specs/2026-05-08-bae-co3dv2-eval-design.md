# BAE CO3Dv2 Evaluation Design

**Date:** 2026-05-08  
**Status:** Approved  
**Scope:** Extend existing GT eval harness to reproduce Table IV of the BAE paper (CO3Dv2, AUC@30), and validate BA improvement on longer 7-Scenes sequences.

## Goal

Show that VGGT-X + BAE improves camera pose accuracy over VGGT-X baseline, reproducing the trend from BAE paper Table IV:

| Condition | AUC@30 ↑ | Time ↓ |
|---|---|---|
| VGGT init only | 88.2 | ~0.2s |
| VGGT + BAE (paper) | 90.0 | ~0.9s |
| VGGT-X init only (ours) | _TBD_ | ~0.2s |
| VGGT-X + BAE (ours) | _TBD_ | _TBD_ |

Exact number match not expected (VGGT-X ≠ vanilla VGGT). Success = BA condition improves over baseline.

## Architecture

Extends existing infrastructure with minimal new code:

```
collab_splats/pointcloud/loop_closure/eval.py   ← add auc_at_threshold()
evals/datasets.py                               ← add _load_co3dv2(), update EvalDataset
evals/download_co3dv2.sh                        ← CO3Dv2 subset download helper
evals/eval_gt.py                                ← emit auc_30 + time_s in metrics.json
docs/pointcloud/eval_co3dv2_gt.ipynb            ← viz notebook (load-only)
```

## Data Flow

```
CO3Dv2 sequence dir
  └── images/            (JPEG frames)
  └── frame_annotations.jgz  (GT R, T per frame)

_load_co3dv2(seq_dir, max_frames)
  → EvalDataset(images, gt_poses, intrinsics)

eval_gt.py --dataset co3dv2 --conditions baseline ba ba_hightrack
  → per-condition: run VGGTXCreator → extract poses → compute metrics
  → metrics.json: {ate, rpe, auc_30, time_s}
```

## AUC@30 Metric

New function `auc_at_threshold(pred_poses, gt_poses, max_threshold_deg=30)` in `eval.py`.

Protocol (matches CO3Dv2 / VGGSfM / VGGT papers):

1. **Align**: Umeyama-align `pred_poses` to `gt_poses` (reuse existing `umeyama_align`)
2. **Rotation error per frame**: geodesic distance between R_pred and R_gt (degrees)
   - `err_R = arccos(clip((trace(R_rel) - 1) / 2, -1, 1))` where `R_rel = R_pred @ R_gt.T`
3. **Translation direction error per frame**: angle between camera center directions (degrees, scale-free)
   - Camera center in world: `c = -R.T @ t`
   - `err_T = arccos(clip(dot(c_pred_norm, c_gt_norm), -1, 1))`
4. **Per-frame combined error**: `err = max(err_R, err_T)`
5. **Accuracy at threshold t**: fraction of frames where `err < t`
6. **AUC**: `mean(accuracy at t for t in linspace(0, max_threshold_deg, 100))` × 100

Returns `{"auc_30": float, "per_frame_err": [...]}`.

## CO3Dv2 Loader

`_load_co3dv2(seq_dir: Path, max_frames: int) -> EvalDataset`

CO3Dv2 sequence format:
```
<category>/<sequence>/
  images/frame000001.jpg ...
  frame_annotations.jgz   (gzipped JSON array, one object per frame)
```

Each frame annotation:
```json
{
  "image": {"path": "...", "size": [H, W]},
  "viewpoint": {
    "R": [[...], [...], [...]],  // 3x3 world-to-cam rotation (row-major)
    "T": [tx, ty, tz],           // world-to-cam translation
    "focal_length": [fx, fy],
    "principal_point": [cx, cy]
  }
}
```

Loader:
1. Parse `frame_annotations.jgz` → list of frame dicts
2. Sort by frame number, take first `max_frames`
3. Build `gt_poses` as `(N, 4, 4)` float32 world-to-cam (R | T in top 3 rows)
4. Build `intrinsics` as `(N, 3, 3)` float32 K matrices
5. Return `EvalDataset(images, gt_poses, intrinsics)`

`EvalDataset` gets an optional `intrinsics: np.ndarray | None` field (default None; existing 7-Scenes loader unaffected).

Download subset: `evals/download_co3dv2.sh <category>` — wraps the CO3Dv2 download script for a single category.

For paper comparison use 10 standard categories: apple, ball, banana, bench, book, bottle, bowl, broccoli, car, chair.

## Conditions

| Condition | `use_ba` | Track params | Note |
|---|---|---|---|
| `baseline` | False | — | VGGT-X init only |
| `ba` | True | `max_query_pts=2048, query_frame_num=5` | Current defaults |
| `ba_hightrack` | True | `max_query_pts=4096, query_frame_num=8` | Paper defaults — isolates track-count vs model gap |

`ba_hightrack` is optional — skip if compute budget is tight. Useful for debugging why numbers differ from paper.

## Runner Update (`eval_gt.py`)

1. Add `co3dv2` to `--dataset` choices
2. Pass `intrinsics` from `EvalDataset` into creator when available
3. Time each condition with `time.perf_counter()` → emit `time_s` in metrics
4. Call `auc_at_threshold(pred, dataset.gt_poses)` → emit `auc_30` in metrics

Updated `metrics.json` shape:
```json
{
  "baseline": {"ate": {...}, "rpe": {...}, "auc_30": 87.4, "time_s": 0.21},
  "ba":       {"ate": {...}, "rpe": {...}, "auc_30": 89.6, "time_s": 1.10}
}
```

## 7-Scenes Extended Validation

Chess seq-01 (50 frames) was too short — no drift to correct. Run:
- `chess/seq-01` full: `--max_frames 1000`
- `fire/seq-01`: `--max_frames 500`
- `heads/seq-01`: `--max_frames 500`

Success on 7-Scenes = BA condition has lower ATE RMSE than baseline on at least 2 of 3 sequences. AUC@30 also computed for these runs.

## Known Differences from Paper

| Aspect | Paper (VGGT) | Our setup (VGGT-X) | Impact |
|---|---|---|---|
| Backbone | Vanilla VGGT | VGGT-X (depth/normal heads) | Different initial poses — expect different absolute numbers |
| `max_query_pts` | 4096 | 2048 (ba) / 4096 (ba_hightrack) | Fewer tracks → potentially weaker BA |
| `query_frame_num` | 8 | 5 (ba) / 8 (ba_hightrack) | Less cross-frame coverage |
| Image preprocessing | `load_and_preprocess_images_square` | `load_and_preprocess_images_ratio` | Aspect ratio handling differs |
| Camera model | SIMPLE_PINHOLE | SIMPLE_PINHOLE (confirmed `bundle_adjustment.py:200`) | Match — no action needed |

## Outputs

Per run: `metrics.json`, `trajectories.npz`, timing per condition.

Comparison table (written by notebook or script):

```
| Condition            | AUC@30 ↑ | ATE RMSE (m) | Time (s) |
|----------------------|----------|--------------|----------|
| VGGT-X baseline      |  XX.X    |    X.XXX     |   X.XX   |
| VGGT-X + BAE         |  XX.X    |    X.XXX     |   X.XX   |
| VGGT-X + BAE (high)  |  XX.X    |    X.XXX     |   X.XX   |
| VGGT + BAE (paper)   |  90.0    |    —         |   0.9    |
```

## Test Plan

- `test_auc_at_threshold.py`: identity poses → AUC = 100; large error → AUC near 0; known toy case with predictable AUC
- `test_co3dv2_loader.py`: mock `frame_annotations.jgz`, verify pose shape and dtype
- `nbconvert --execute eval_co3dv2_gt.ipynb` exits 0

## Extending to Other Sequences

Add to `datasets.py`: `_REGISTRY["co3dv2"] = _load_co3dv2`. Runner, metrics, and plots unchanged. Same pattern as existing 7-Scenes extension point.
