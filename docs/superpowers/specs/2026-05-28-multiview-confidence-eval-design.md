# Multiview Confidence Eval — Design Spec

**Date:** 2026-05-28
**Goal:** Empirically determine whether MapAnything's geometric multiview confidence
filter improves pose accuracy for VGGT-X and VGGTOmega, to decide if it warrants
a generalized postprocessing utility.

---

## Background

MapAnything runs `compute_multiview_depth_confidence` (from
`mapanything.utils.multiview_confidence`) during `_postprocess` by default
(`use_multiview_confidence=True`, `confidence_percentile=35.0`). This is a
**geometric** cross-view depth consistency check: for each pixel, it projects
the depth into all overlapping views and counts inlier/outlier matches.
Confidence = inliers / (inliers + outliers), in [0, 1].

VGGT-X and VGGTOmega use a **learned** per-pixel `depth_conf` from their dense
head (`1 + exp(logits)`, values ≥ 1), filtered at the 35th-percentile threshold.
These are complementary signals: learned conf captures model uncertainty; multiview
conf catches geometrically inconsistent predictions the model may still be confident
about.

The algorithm requires:
- `depth_z`: `List[(B, H, W, 1)]` — Z-depth per view
- `intrinsics`: `List[(B, 3, 3)]` — camera intrinsics per view
- `camera_poses`: `List[(B, 4, 4)]` — cam2world per view

All three are available in VGGT-X and VGGTOmega raw outputs after `_forward`.

---

## Eval Design

### Notebook

`evals/notebooks/multiview_conf_analysis.ipynb`

Run in tmux (heavy inference, ~3 forward passes). Uses
`/opt/conda/envs/reconstruction/bin/python` kernel. Dataset: 7-Scenes chess
`seq-01` at `/data/7scenes/chess/seq-01`.

---

### Section 1 — Inference + zarr save

Run each model on chess seq-01 once, save `FeedforwardResult` zarr. Sequential
to avoid OOM. Clear GPU memory between models with `torch.cuda.empty_cache()`.

| Model | Creator | Save path | Notes |
|---|---|---|---|
| VGGT-X | `VGGTXCreator` | `evals/results/mv_conf_eval/vggtx/` | default config |
| VGGTOmega | `VGGTOmegaCreator` | `evals/results/mv_conf_eval/vggt_omega/` | default config |
| MapAnything | `MapAnythingCreator(use_multiview_confidence=False)` | `evals/results/mv_conf_eval/mapanything/` | `mv_off` to preserve raw conf |

MapAnything runs with `use_multiview_confidence=False` so we hold raw model
outputs and apply filtering ourselves in later cells.

The `FeedforwardResult` zarr stores `confidence (N, H, W)`, `depth (N, H, W)`,
`extrinsics (N, 4, 4)`, `intrinsics (N, 3, 3)` — everything needed for both
filtering and ATE computation.

---

### Section 2 — Filtering variants

For each saved zarr, apply filtering variants without re-running inference.
Same percentile threshold (35.0) used in all variants for fair comparison.

**VGGT-X and VGGTOmega — 3 variants each:**

| Variant | Filter applied | Source |
|---|---|---|
| `learned_only` | `depth_conf >= percentile(depth_conf, 35)` | baseline (current behavior) |
| `mv_only` | `mv_conf >= percentile(mv_conf, 35)` | multiview conf replaces learned conf |
| `intersect` | both filters AND'd | stricter combined filter |

**MapAnything — 2 variants (sanity check):**

| Variant | Description |
|---|---|
| `mv_off` | raw `conf` from model only (no geometric filter) |
| `mv_on` | `compute_multiview_depth_confidence` applied (current default behavior) |

MapAnything `mv_on` replicates production behavior, confirming multiview conf
actually helps on this sequence before we draw conclusions for other models.

---

### Applying multiview confidence to VGGT-X / VGGTOmega

`compute_multiview_depth_confidence` is called directly from
`mapanything.utils.multiview_confidence` (already installed in reconstruction
env). Input preparation from `FeedforwardResult`:

```python
from mapanything.utils.multiview_confidence import compute_multiview_depth_confidence
from collab_splats.utils.geometry import invert_poses

# result: FeedforwardResult loaded from zarr
depth = torch.from_numpy(result.depth)          # (N, H, W)
depth_z = [depth[i].unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
           for i in range(len(depth))]

intrs = torch.from_numpy(result.intrinsics)     # (N, 3, 3)
intrinsics = [intrs[i].unsqueeze(0) for i in range(len(intrs))]  # List[(1,3,3)]

# extrinsics is world2cam (N,4,4); invert to cam2world for camera_poses arg
cam2world = invert_poses(result.extrinsics)     # (N, 4, 4)
camera_poses = [torch.from_numpy(cam2world[i]).unsqueeze(0)
                for i in range(len(cam2world))]  # List[(1,4,4)]

mv_conf_list = compute_multiview_depth_confidence(
    depth_z, intrinsics, camera_poses
)
mv_conf = torch.stack([c.squeeze(0) for c in mv_conf_list])  # (N, H, W), values in [0,1]
```

Filtering then mirrors `unproject_and_filter_points`: compute percentile
threshold, build boolean mask, index into `result.points` via `result.pixel_indices`.

---

### Section 3 — Point cloud quality metrics

Confidence filtering changes which **points** survive — camera poses come
directly from the model's pose head and are unaffected by filtering. ATE would
be identical across variants. The meaningful signals are point cloud quality:

- **Surviving point count** — density at the same 35th-percentile threshold
- **mv_conf score distribution** — histogram of multiview confidence values per
  model (shows how much signal exists; flat distribution = no useful signal)
- **Visual quality** — 3D scatter plot per variant for qualitative comparison

These directly answer whether multiview conf produces a cleaner/denser point
cloud, which is the upstream input to BA track extraction.

---

### Section 4 — Summary table + plots

Output per model:

```
Model       | Variant       | N points | mv_conf mean | Notes
------------|---------------|----------|--------------|-------
VGGT-X      | learned_only  | ...      | n/a          | baseline
VGGT-X      | mv_only       | ...      | ...          |
VGGT-X      | intersect     | ...      | ...          |
VGGTOmega   | learned_only  | ...      | n/a          | baseline
VGGTOmega   | mv_only       | ...      | ...          |
VGGTOmega   | intersect     | ...      | ...          |
MapAnything | mv_off        | ...      | n/a          | baseline
MapAnything | mv_on         | ...      | ...          | production default
```

Bar chart: point count by model × variant. Histogram: mv_conf score distribution
per model. 3D scatter: qualitative point cloud comparison for one representative
model × variant pair.

---

## Decision Gate

After running the notebook:

- **If mv_conf improves point density with no ATE regression** → proceed to
  implement `compute_multiview_depth_confidence` as a shared utility in
  `collab_splats/pointcloud/feedforward/` and add `use_multiview_confidence` /
  `confidence_percentile` params to `VGGTXCreator` and `VGGTOmegaCreator`.
- **If mv_conf degrades density with no quality gain** → discard; document
  finding in worklog.
- **If MapAnything sanity check shows mv_off ≈ mv_on** → treat as signal that
  the method doesn't generalize; investigate before proceeding.

---

## Files

| Path | Purpose |
|---|---|
| `evals/notebooks/multiview_conf_analysis.ipynb` | eval notebook (create) |
| `evals/results/mv_conf_eval/` | zarr stores + outputs (gitignored) |
