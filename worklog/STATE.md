# State — last-updated 2026-05-23

## Active Branch
| Branch | Status | Ahead of main | Notes |
|---|---|---|---|
| `refactor/cu121` | active | ~83 commits | CUDA 12.1 + py3.11 migration branch; includes all core-modules work |

## In-Flight Work
- **gt-eval-harness** — [spec](specs/2026-05-07-gt-eval-harness-design.md) · [plan](plans/2026-05-07-gt-eval-harness.md) — status: wip
- **feedforward-import-cleanup** — [spec](specs/2026-05-08-feedforward-import-cleanup-design.md) · [plan](plans/2026-05-08-feedforward-import-cleanup.md) — status: wip
- **feedforward-mesh** — [spec](specs/2026-05-14-feedforward-mesh-design.md) · [plan](plans/2026-05-14-feedforward-mesh.md) — status: wip
- **docs-site** — [spec](specs/2026-05-20-docs-site-design.md) · [plan](plans/2026-05-20-docs-site.md) — status: wip (Sphinx docs site setup)
- **bae-vggt-parity** — [spec](specs/2026-05-20-bae-vggt-parity-design.md) — status: ready-for-handoff (verify our BA matches upstream `zitongzhan/vggt --implementation bae` on matched inputs)

## Recently Completed
- **vggtx-feedforward-refactor** (2026-05-23) — [spec](../docs/superpowers/specs/2026-05-23-vggtx-feedforward-refactor-design.md) · [plan](../docs/superpowers/plans/2026-05-23-vggtx-feedforward-refactor.md)
  - Deleted `_patch_vggtx_compute_similarity` (global class mutation, not thread-safe). Replaced with `extract_intermediate_features(frames, layer_index, **kwargs)` abstract on `BaseFeedforwardCreator` — per-call `register_forward_hook` on `aggregator.global_blocks[i].attn.qkv` (VGGTx) or `info_sharing.self_attention_blocks[i].attn.qkv` (MapAnything). Moved `cross_frame_attention_ratio` to `pointcloud/utils.py` as a pure function (port of VGGT-SPARK `get_similarity()`). `_verify_loop_candidate` is now concrete on base — uses `cross_frame_attention_ratio` + `features.get("poses")`. VGGTx decodes poses inside `extract_intermediate_features` and returns `"poses"` (2,4,4) numpy; MapAnything returns None. All lazy `import torch` moved to module top. Added structural docstrings to VGGTx methods matching MapAnything style.
- **sl4-loop-closure** (2026-05-23) — [spec](../docs/superpowers/specs/2026-05-22-sl4-loop-closure-design.md) · [plan](../docs/superpowers/plans/2026-05-22-sl4-loop-closure.md)
  - Replaced dead SE(3)/Sim3 pose graph stack with unified SL(4) `PoseGraph` ported from MIT-SPARK/VGGT-SLAM. Collapsed `pose_graph.py` + `retrieval.py` + `alignment.py` → `graph.py` + updated `closure.py`. New `graph.py`: `decompose_camera`, `normalize_to_sl4`, `estimate_scale_pairwise`, `PoseGraph` with per-frame SL(4) nodes (or SE(3) fallback via `manifold=`). `submap.py`: `+get_world_points(H)`, `+get_poses_world(H)`. `closure.py`: absorbed LoopMatch/LoopClosureConfig/LoopMatchQueue/find_loop_closures; new `run_pose_graph_optimization` mirrors vggt_slam/solver.py with `estimate_scale_pairwise` for inter-submap scale. `eval.py`: `_classify_edges` updated for `BetweenFactorSL4`. `wrappers.py`: `ImageRetrieval` dropped, `run_sim3` replaced. New `evals/eval_vggt_slam_comparison.py`: 7-Scenes ATE harness (baseline / lc_se3 / lc_sl4 / vggt_slam_oob). 234 tests passing.
- **localization-cleanup** (2026-05-21) — [spec](specs/2026-05-21-localization-cleanup-design.md) · [plan](plans/2026-05-21-localization-cleanup.md)
  - Extended `LocalizationResult` with `pts2d_ref` (M,2) + `ref_frame_indices` (M,) int32 — reference-frame 2D coords and frame index per correspondence, tracked in `localize()` matching loop. Added `plot_correspondences()` under new Visualization section: hloc-style side-by-side query/reference frame with green inlier / red outlier connecting lines. Added `test_localization_result_fields_on_success`. New `docs/pointcloud/localization.ipynb` tutorial: 17-cell self-contained notebook (score_all_frames keyframe extraction → VGGTXCreator → hold-out localization → correspondence viz → PyVista frustum view).
- **salad-package-migration** (2026-05-21)
  - Replaced `vendor/salad/` + sys.path hack with `pip install git+https://github.com/Dominic101/salad.git`. Updated imports in `localization.py`: `from salad.models_salad.aggregators.salad import SALAD` / `from salad.models_salad.backbones.dinov2 import DINOv2`. Deleted `vendor/salad/`. Bonus: `third_party/VGGT-SLAM/loop_closure.py` `from salad.eval import load_model` now resolves.
- **feedforward-tutorial** (2026-05-21) — [spec](specs/2026-05-21-feedforward-notebook-design.md) · [plan](plans/2026-05-21-feedforward-notebook.md)
  - `pointcloud_to_polydata(**point_data) → pv.PolyData` added to `collab_splats/utils/visualization.py` (copies pts3d, attaches arbitrary named scalar arrays). Rewrote `docs/pointcloud/feedforward_exploration.ipynb`: 21-cell tutorial running VGGT-X + MapAnything on C0043 keyframes via `score_all_frames` + step-by-step creator API. Per-model PyVista viewers, stats table (pts raw/filtered, conf mean/std, timing), camera trajectory overlay (blue/orange frustums). 10 tests passing.
- **camera-localization** (2026-05-21) — [spec](specs/2026-05-21-camera-localization-design.md) · [plan](plans/2026-05-21-camera-localization.md)
  - `localization.py` complete: `DiskExtractor` (DISK+LightGlue), `XFeatExtractor` (XFeat+MNN), `_build_frame_assignments` (torch.cdist NN), `CameraLocalizer` (exhaustive match + PnP). XFeat vendored at `vendor/xfeat/`. No global retrieval — matches all N frames to avoid false exclusions. `from_feedforward()` classmethod for feedforward pipeline.
- **inline-documentation-cleanup** (2026-05-21) — `pointcloud/utils.py`, `wrappers.py`, `bundle_adjustment.py`, `mesh/utils.py` — `########` dividers, inline block comments, `print()→logging`.
- **keyframe-extraction-tutorial** (2026-05-21) — [spec](specs/2026-05-20-keyframe-extraction-tutorial-design.md) · [plan](plans/2026-05-20-keyframe-extraction-tutorial.md)
  - `get_video_info` + `score_all_frames` added to `collab_splats/utils/frame_sampling.py`. Viz section behind `####` divider: `plot_frame_grid`, `plot_selection` (single/dual panel), `plot_frame_scores` (3-panel disparity/rotation/hist-sim), `plot_disparity_sensitivity` (re-thresholds precomputed scores). Notebook at `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` sources `C0043.MP4`. First entry in tutorial index under Preprocessing.
- **positional-debiasing** (2026-05-21) — [spec](specs/2026-05-20-positional-debiasing-design.md) · [plan](plans/2026-05-20-positional-debiasing.md)
  - SVD-based positional debiasing on `BaseFeatureExtractor`: `debias()` projects patch features onto orthogonal complement of positional subspace; `get_bias_visualization()` returns PCA heatmap. Basis lazy-cached per `(H_p, W_p)`. Validated for `DINOFeatureExtractor` + `Talk2DinoExtractor`; warns (does not raise) for others. All concrete `__init__` pass `**kwargs` for `svd_components` threading. 11 tests in `tests/semantics/test_positional_debiasing.py`.
- **feature-lifting** (2026-05-20) — [spec](specs/2026-05-20-feature-lifting-design.md) · [plan](plans/2026-05-20-feature-lifting.md)
  - `unproject_and_filter_points` → returns `pixel_indices (P, 3) int32` = `[frame_id, row, col]` per point via `np.where(conf_mask)`. `FeedforwardResult` carries `pixel_indices` + `features`; `BaseFeedforwardCreator` carries `extractor_name`. `lift_features` + `reproject_pixels` added to `pointcloud/utils.py`. `VGGTXCreator._postprocess` lifts features when `extractor_name` set. `BundleAdjustment` uses `reproject_pixels` (deterministic) when `pixel_indices` present, preserving color/feature alignment post-BA.
- **semantics-refactor** (2026-05-20) — [plan](plans/2026-05-20-semantics-refactor.md)
  - Hard imports for required deps; segmentation bugs fixed (empty-mask crash, uint16 truncation); stub backends removed; torch utilities extracted to `collab_splats/utils/torch_utils.py` (fixes cross-module import violation); retrieval moved to `pointcloud/localization.py` (stage 1 of localization pipeline); `RegistryMixin` deduplicates registry pattern; `print()` → `logging`; `########` section dividers throughout.
- **worklog-split** (2026-05-20) — [spec](history/specs/2026-05-20-worklog-split-design.md)
  - Split user docs (`docs/`) from internal tracking (`worklog/`); consolidated scattered READMEs into `docs/`; ADRs + history moved under `worklog/`.
- **eval-reorganization** (2026-05-20) — [spec](history/specs/2026-05-20-eval-reorganization-design.md) · [plan](history/plans/2026-05-20-eval-reorganization.md)
  - `ba_hightrack` → `ba_track-density-{N}`; results in `evals/results/` (gitignored); unified `docs/pointcloud/ground-truth-evals.ipynb`; old per-dataset notebooks deleted

## Open Blockers
- nerfstudio env: see [notes/2026-05-03-feedforward-env-debug](history/specs/2026-05-03-feedforward-env-debug.md) if relevant; check whether the symptom is current.
- Known test failures: see [known-test-failures.md](known-test-failures.md).

## Parked
| Item | Why parked | Pointer |
|---|---|---|
| `tlb-grouping-segmentation` | separate plan TBD | [ROADMAP](ROADMAP.md#parked-branches) |
| `tlb-improve-splatter` | separate plan TBD | [ROADMAP](ROADMAP.md#parked-branches) |

## Recent Decisions (last 30 days)
- [ADR 012 — hloc direct call](decisions/012-hloc-direct-call.md)
- [ADR 011 — CoordinateFrame enum](decisions/011-coordinate-frame-enum.md)
- [ADR 010 — pointcloud creator registry](decisions/010-pointcloud-creator-registry.md)
- [ADR 009 — frame_sampling in utils](decisions/009-frame-sampling-in-utils.md)
- [ADR 008 — MapAnything stub pattern](decisions/008-mapanything-stub-pattern.md)
- [ADR 007 — dashboard as integration harness](decisions/007-dashboard-as-integration-harness.md)
- [ADR 006 — single working branch](decisions/006-single-working-branch.md)
