# Roadmap

## What's Built

- **Semantics**: ANN feature splatting, MaskCLIP + Talk2DINO, batched extraction, dashboard query API. General PyTorch utilities live in `collab_splats/utils/torch_utils.py` (`RegistryMixin`, `get_device`, `pytorch_gc`, `infer_batch_size`, `batch_iterator`). Retrieval (DINOv2-SALAD) lives in `pointcloud/localization.py` — Stage 1 of the camera localization pipeline (Stage 2: local feature matching; Stage 3: PnP reserved).
  Key ADRs: [009 — frame_sampling in utils](decisions/009-frame-sampling-in-utils.md).
- **Pointcloud**: creator registry (colmap / hloc / vggtx / mapanything), `CoordinateFrame` world transform, hloc direct call.
  Key ADRs: [010 — creator registry](decisions/010-pointcloud-creator-registry.md), [011 — CoordinateFrame](decisions/011-coordinate-frame-enum.md), [012 — hloc direct](decisions/012-hloc-direct-call.md).
- **Loop Closure**: Sim(3) pose graph, geometric overlap gate, 3-way noise split, Huber robustification.
  Key ADRs: [002](decisions/002-defer-sl4.md), [003](decisions/003-defer-graphmap.md), [004](decisions/004-defer-frametracker.md), [005](decisions/005-defer-per-backend-noise-tuning.md).
- **Mesh**: TSDF integration from `PointcloudResult`; feedforward → direct mesh path.
- **Dashboard**: integration test harness — full pipeline visual validation.
  Key ADRs: [007 — dashboard as harness](decisions/007-dashboard-as-integration-harness.md), [008 — MapAnything stub](decisions/008-mapanything-stub-pattern.md).
- **BA**: PyPose BAE bundle adjustment, CO3Dv2 eval baseline.
- **Branch strategy**: single working branch.
  Key ADR: [006](decisions/006-single-working-branch.md).

## Where We Are

- Sole working branch: `refactor/core-modules` (~82 commits ahead of `main`).
- Final PR before merge to `main`.
- Active in-flight: `gt-eval-harness`, `feedforward-import-cleanup`, `feedforward-mesh`.
- Phase 1 (core modules + loop closure + BA) near complete.

## What's Next

### Phase 2 — Dashboard Iteration
Goal: harden dashboard as the canonical integration surface. Add golden-image regression captures; auto-launch on PR with representative dataset; surface per-subsystem health indicators.

### Phase 3 — Feature Extension
Goal: extend semantic feature space beyond MaskCLIP + Talk2DINO. Candidate: DINOv3, SAM2 panoptic masks for grouping, learned per-Gaussian features distilled from open-vocabulary backbones.

### Candidate Features (unprioritized)
- DeCLIP integration (see [ADR 001](decisions/001-declip-integration.md)).
- Multi-view consistent feature distillation across submaps.
- Live capture → reconstruction loop (no offline preprocessing).

### Future Considerations
- **SL(4) submap loop closure (VGGT-SLAM-style).** Current LC uses windowed submap merge into a single global frame; not SL(4). If we adopt a homography-stitched-submap design, `vggt.utils.geometry.unproject_depth_map_to_point_map` must be switched to return **per-camera-local** points instead of first-cam-anchored world points — otherwise the per-submap SL(4) double-transforms an already-globalized point cloud, causing scale/alignment drift between loops.
  - Reference: [MIT-SPARK/VGGT-SLAM PR #39](https://github.com/MIT-SPARK/VGGT-SLAM/pull/39) + the canonical one-line fix in [`MIT-SPARK/VGGT_SPARK geometry.py@6e6e161`](https://github.com/MIT-SPARK/VGGT_SPARK/blob/6e6e161/vggt/utils/geometry.py#L15) (appends `cam_coords_points`, not `cur_world_points`).
  - Touch point if/when work begins: `collab_splats/pointcloud/feedforward/vggtx.py::unproject_and_filter_points` (currently flagged with a comment) + `collab_splats/pointcloud/wrappers.py::LoopClosure`.

### Parked Branches
- `tlb-grouping-segmentation` — Gaussian grouping + per-instance segmentation; separate plan TBD; waiting on stable feature splatting API.
- `tlb-improve-splatter` — Splatter wrapper rework; separate plan TBD; deferred until core refactor lands on `main`.
