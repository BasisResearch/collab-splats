# collab-splats Refactor Worklog

> **Append-only.** Newest entries at top of Session Log. See:
> - [STATE.md](STATE.md) for current state (branches, in-flight, blockers, parked)
> - [ROADMAP.md](ROADMAP.md) for future phases + architecture overview

### 2026-05-28 — LC calibration + ATE parity run 3

**VGGT-SLAM architecture confirmed identical** — two-gate pipeline: DINO-SALAD L2 retrieval → attention verify gate (0.85 threshold, same `get_similarity` algorithm). `compute_similarity=True` only in `third_party/vggt_spark` fork; installed `vggt` lacks it — hook-based `cross_frame_attention_ratio` is correct workaround. Score gap vs VGGT-SPARK reference (1.025) is pair distribution, not implementation: reference measured on DINO-SALAD retrieved pairs, our prior calibration used random temporal pairs.

**Calibration script extended** (`evals/eval_similarity_calibration.py`):
- `--mode retrieved` — DINO-SALAD nearest-neighbour pairs (authoritative vs VGGT-SPARK reference). Default remains `random` until validated.
- `--layer_index INT` — per-run override for layer sweep experiments; results filed as `similarity_calibration_retrieved_layer{N}.json`
- Layer index + block depth logged per model

**eval_gt.py** — added `mapanything` to backbone choices and `_BACKBONE_PREFIX`.

**Layer sweep results (retrieved mode, 20 DINO-SALAD pairs, chess_seq01):**

| Model | Depth | Layer | mtq mean | Notes |
|---|---|---|---|---|
| VGGT-X | 24 | 23 | 0.690 | last block |
| VGGT-X | 24 | 20 | 0.817 | VGGT-SPARK default |
| VGGT-X | 24 | 18 | 0.500 | local minimum |
| VGGT-X | 24 | 16 | 0.916 | above threshold |
| VGGT-X | 24 | **12** | **1.426** | calibration peak |
| Omega | 24 | 20 | 0.897 | false-positive LCs |
| Omega | 24 | **16** | **1.328** | optimal — set as default |
| Omega | 24 | 12 | 1.177 | also good |
| Omega | 24 | 8 | 1.112 | |
| MapAnything | 16 | 15 | 0.341 | last block = noise |
| MapAnything | 16 | **4** | **1.807** | optimal — set as default |

**Code changes from sweep:**
- `VGGTOmegaCreator._lc_layer_index = 16` (was inherited 20; removed mistaken `default_verify_match_ratio=0.59` placeholder — reverts to base 0.85)
- `MapAnythingCreator._lc_layer_index = 4` (was out-of-range 20; depth=16, peaks early ~25% unlike VGGT models)

**ATE results (chess_seq01, 200 frames):**

| Condition | ATE RMSE | Notes |
|---|---|---|
| VGGT-X baseline (s=16) | 0.3184m | |
| VGGT-X + LC, layer=20 | 0.2664m | −16.3% |
| VGGT-X + LC, layer=12 | 0.2664m | same — layer=20 kept |
| Omega baseline (s=16) | 0.3225m | |
| Omega + LC, layer=20 | 0.4959m | LC harmful |
| Omega + LC, layer=16 | 0.4958m | LC still harmful — open issue |
| MapAnything baseline (single-pass) | 0.1103m | 200 frames, no LC |
| VGGT-SLAM LC (vggt_spark model) | 0.0189m | 30 keyframes, scale-corrected — not comparable |

**Open issues (tracked):**
1. **Omega LC regression** — LC consistently hurts Omega (0.3225 → 0.496m) regardless of layer. Cause unknown; suspect wrong pose graph edges or Omega's coordinate conventions. Layer calibration alone doesn't fix it.
2. **MapAnything windowed LC incompatible** — `_forward` ignores `window` arg, always processes `self._processed_views` (all frames). Windowed submap approach broken. Single-pass baseline works (0.1103m). LC requires architectural rework of `_forward` to support windowed inference.
3. **VGGT-SLAM LC 0 closures** — `lc_thres=0.95` too strict or chess_seq01 first 200 frames has no revisits within VGGT-SLAM's keyframe subset (30 selected from 200).

**Infrastructure fixes:**
- `wrappers.py` — `_run_lc_loop` now calls `self.base._lc_collate_outputs(raw)` when `raw` is a list, enabling future MapAnything windowed support
- `BaseFeedforwardCreator._lc_collate_outputs` — no-op default
- `MapAnythingCreator._lc_collate_outputs` — aggregates list[dict] → flat dict (extrinsic + intrinsics from `camera_poses`); blocked on `camera_poses` not in raw model output (requires postprocess step)
- `eval_vggt_slam_comparison.py` — `run_vggt_slam_oob` now accepts `max_loops` param, injects `vggt_spark` PYTHONPATH so `compute_similarity=True` resolves, archives TUM to baselines dir; added `vggt_slam_lc` and `vggt_slam_nolc` conditions
- `docs/superpowers/specs/2026-05-28-parity-run3-design.md` — design doc for this session

## 2026-05-24

### ba-module-cleanup

Spec: `docs/superpowers/specs/2026-05-24-ba-module-cleanup-design.md`
Plan: `docs/superpowers/plans/2026-05-24-ba-module-cleanup.md`

- Dropped all lazy imports; moved `torch`, `torch.nn`, `pypose`, `bae.*`, `vggt.*` to hard imports at top of file
- Collapsed `_run_bundle_adjustment` (220 lines, called once) into `BundleAdjustment._optimize()` — reads `self.config` directly
- Lifted `ReprojNonBatched` inner class → module-level `_BAModel(nn.Module)` with injected `shared_camera` flag
- Lifted conditional same-name closures → module-level `_reproject_per_camera` / `_reproject_shared` decorated with `@map_transform`
- Removed dead `image_size` parameter; removed inner `rotate_quat` wrapper (inlined as `pp.SE3(...).Act(pts)`)
- Reordered module: public API (config + class) first, helpers below
- Updated tests: removed `image_size` from call sites, patched `_optimize` instead of `_run_bundle_adjustment`, removed stale `bae.utils.ba.rotate_quat` mock
- ~34% line reduction (438 → ~290 lines)

## 2026-05-23

### sl4-loop-closure — COMPLETE

Replaced the dead SE(3)/Sim3 pose graph stack with a unified SL(4) PoseGraph ported from MIT-SPARK/VGGT-SLAM. Squash commit on `refactor/cu121`.

**What changed:**
- `loop_closure/` collapsed from 6 files → 5: `pose_graph.py`, `retrieval.py`, `alignment.py` deleted; `graph.py` added
- `graph.py`: `decompose_camera` (RQ decomp + SVD orthogonality snap), `normalize_to_sl4` (H/|det H|^¼), `estimate_scale_pairwise` (median ratio), `PoseGraph` with per-frame SL(4) nodes mirroring `vggt_slam/solver.py:add_edge` — Huber-wrapped loop edges, LM optimization via `gtsam.SL4` + `BetweenFactorSL4`; SE(3) fallback via `manifold="se3"`
- `submap.py`: `+get_world_points(H)` + `+get_poses_world(H)` projective reprojection helpers
- `closure.py`: absorbed `LoopMatch`/`LoopClosureConfig`/`LoopMatchQueue`/`find_loop_closures` from `retrieval.py`; `dedup_overlap` from `alignment.py`; new `run_pose_graph_optimization` with `estimate_scale_pairwise` for inter-submap scale initialization; all dead `PoseGraph`(SE3)/`Sim3PoseGraph`/`ImageRetrieval`/`build_pose_graph`/`run_sim3_pose_graph_optimization` and their helpers deleted
- `eval.py`: `_classify_edges` updated for `BetweenFactorSL4` (returns `sequential/loop` schema)
- `wrappers.py`: `ImageRetrieval` → `BaseRetrievalExtractor.get("dino-salad")`; `run_sim3_pose_graph_optimization` → `run_pose_graph_optimization(manifold=cfg.manifold)`
- `evals/eval_vggt_slam_comparison.py` (new): 7-Scenes ATE comparison harness for baseline / lc_se3 / lc_sl4 / vggt_slam_oob conditions via `evo_ape`

**Test result:** 234 passed, 0 failures (2 pre-existing mapanything flakes excluded)

**Key decisions:**
- Per-frame SL(4) nodes (not per-submap) — mirrors VGGT-SLAM `solver.py`, not the old Sim3 per-submap approach
- `estimate_scale_pairwise` for inter-submap scale: median(‖Y‖/‖X‖) on overlapping world points
- `decompose_camera`: RQ decomp → SVD polar-decomp snap for R orthogonality; `t = inv(K) @ P[:,3]`
- SE(3) fallback kept (`manifold="se3"`) for ablation only — SL(4) is the default path
> - [decisions/](decisions/) for ADRs
>
> Entry template:
> ```
> ### YYYY-MM-DD <optional tag>
> - **Focus:** one line
> - **Did:** bullets
> - **Decided:** bullets (link ADRs: `→ ADR NNN`)
> - **Hit:** blockers / surprises / reversals
> - **Files:** key paths
> - **Next:** one line
> ```

## Session Log

### 2026-05-21 — localization-cleanup
- **Focus:** Surface ref-frame correspondence data in LocalizationResult; add hloc-style viz; tutorial notebook
- **Did:**
  - `LocalizationResult` extended with `pts2d_ref: np.ndarray | None` (M,2) + `ref_frame_indices: np.ndarray | None` (M,) int32
  - `localize()` matching loop now tracks `best_ref_2d` + `best_ref_frame` dicts alongside existing `best_2d`; all 3 return statements updated
  - `plot_correspondences(loc, query_image, image_paths, max_pairs=200)` added under new `############ Visualization` section — hloc-style side-by-side with connecting lines (lime=inlier, red=outlier), best ref frame by inlier count, guards for zero inliers + missing file + null mask
  - `test_localization_result_fields_on_success` added — asserts shape/dtype/length consistency of new fields
  - `docs/pointcloud/localization.ipynb` — 17-cell self-contained tutorial: score_all_frames keyframe extraction → VGGTXCreator → hold-out frame 10 → CameraLocalizer.from_feedforward + XFeatExtractor → plot_correspondences → PyVista (grey ref frustums + red localized) → consistency error vs feedforward reference
  - Symlink at `docs/source/tutorials/pointcloud/localization.ipynb`
- **Decided:** No GT comparison (C0043 has no ground-truth poses) — feedforward extrinsics[10] used as pseudo-reference, labeled as consistency check
- **Files:** `collab_splats/pointcloud/localization.py`, `tests/pointcloud/test_localization.py`, `docs/pointcloud/localization.ipynb`, `docs/source/tutorials/pointcloud/localization.ipynb`, `worklog/specs/2026-05-21-localization-cleanup-design.md`, `worklog/plans/2026-05-21-localization-cleanup.md`
- **Next:** —

### 2026-05-21 — salad-package-migration
- **Focus:** Replace vendor/salad sys.path hack with Dominic101/salad pip package
- **Did:**
  - `setup_feedforward.sh`: replaced `git clone serizba/salad → vendor/salad` with `pip install git+https://github.com/Dominic101/salad.git`
  - `localization.py`: removed `_SALAD_VENDOR_PATH` sys.path block; updated imports to `salad.models_salad.aggregators.salad.SALAD` + `salad.models_salad.backbones.dinov2.DINOv2`; updated `DinoSaladExtractor` docstring
  - `pyproject.toml`: updated comment
  - Deleted `vendor/salad/`
- **Decided:** keep manual SALAD+DINOv2 assembly (not `salad.eval.load_model`) — avoids pytorch_lightning dep, no device-hardcoding issue
- **Files:** `setup_feedforward.sh`, `collab_splats/pointcloud/localization.py`, `pyproject.toml`
- **Next:** —

### 2026-05-21 — keyframe-extraction-tutorial
- **Focus:** First preprocessing tutorial — keyframe extraction via FPS vs optical flow
- **Did:**
  - `get_video_info(video_path)` → `{total_frames, fps, duration_s}` with `cap.isOpened()` guard
  - `score_all_frames(video_path, ...)` → `list[dict]` with `frame_idx, disparity, rotation, histogram_similarity, score, selected`; first-frame edge case handled via `.get()` defaults
  - Viz section in `frame_sampling.py` behind `####` divider: `plot_frame_grid`, `plot_selection` (single/dual panel via `squeeze=False`), `plot_frame_scores` (3-panel sharex), `plot_disparity_sensitivity` (re-thresholds precomputed scores, avoids re-decode)
  - Notebook `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb` — 7 phases, no inline helpers, sources `C0043.MP4`
  - `docs/source/tutorials/index.rst` — Preprocessing toctree listed first
- **Files:** `collab_splats/utils/frame_sampling.py`, `tests/utils/test_frame_sampling.py`, `docs/source/tutorials/preprocessing/keyframe_extraction.ipynb`, `docs/source/tutorials/index.rst`

### 2026-05-21 — positional-debiasing
- **Focus:** SVD-based positional debiasing for `BaseFeatureExtractor` (INSID3 algorithm)
- **Did:**
  - Added `_DEBIAS_VALIDATED` module-level frozenset — `DINOFeatureExtractor` + `Talk2DinoExtractor` validated; others warn on use.
  - `BaseFeatureExtractor.__init__` gains `svd_components=500`, `_pos_basis_cache`, `_zero_feats_cache`.
  - `_build_positional_basis(H_p, W_p)` — forwards black PIL image through `self.forward()`, runs SVD on `(D, H_p*W_p)` centered feature matrix, stores top-K left singular vectors as positional subspace.
  - `_apply_debias(fmap)` — projects `(D, H_p*W_p)` features onto `P_perp = I - U @ U.T`, L2 re-normalizes.
  - `debias(features)` — public API; lazy-builds basis on first call per `(H_p, W_p)`, warns for unvalidated extractors.
  - `get_bias_visualization(H_p, W_p)` — PCA of zero-image features → top-3 PCs → uint8 RGB heatmap `(H_p, W_p, 3)`.
  - All concrete `__init__` (`DINOFeatureExtractor`, `Talk2DinoExtractor`, `MaskCLIPExtractor`) pass `**kwargs` to `super().__init__()` for `svd_components` threading.
- **Decided:** `forward()` untouched in all subclasses — debiasing is standalone post-processing step, not baked into forward pass. `patch_size` must be explicitly defined on extractor (no fallback) to catch misconfigurations early.
- **Files:**
  - `collab_splats/semantics/features.py` — all debiasing infrastructure
  - `tests/semantics/test_positional_debiasing.py` — 11 tests (state, forward unchanged, shape, output differs, unit-norm, caching, warning, viz shape/dtype/range, missing patch_size guard)
  - `worklog/specs/2026-05-20-positional-debiasing-design.md` — design spec
  - `worklog/plans/2026-05-20-positional-debiasing.md` — implementation plan
- **Next:** Thread `debias` through `lift_features()` (out of scope for this task — follow-on)

### 2026-05-20 — keyframe-extraction-tutorial brainstorm
- **Focus:** Design first preprocessing tutorial (keyframe extraction)
- **Did:** Brainstormed + wrote spec for `keyframe_extraction.ipynb`; established `preprocessing/` as new first tutorial section in docs; designed `score_all_frames()` driver + viz section in `frame_sampling.py`
- **Decided:** Single self-contained notebook; all plot logic in package (viz section behind `###` divider in `frame_sampling.py`); 7-phase tutorial structure (why → load → FPS → OF timeseries → OF overlay → comparison → sensitivity sweep → guidance)
- **Files:** `worklog/specs/2026-05-20-keyframe-extraction-tutorial-design.md`
- **Next:** Invoke writing-plans → implement `score_all_frames` + viz section + notebook

### 2026-05-20 feature-lifting
- **Focus:** Lift 2D features onto VGGT-X pointcloud — assign `BaseFeatureExtractor` patch vectors to each 3D point via its source pixel, surviving post-BA reprojection.
- **Did:**
  - `unproject_and_filter_points` now returns `(pts3d, colors, pixel_indices)` where `pixel_indices (P, 3) int32 = [frame_id, row, col]` per point, computed via `np.where(conf_mask)` after `randomly_limit_trues`.
  - `FeedforwardResult` gains `features: np.ndarray | None` and `pixel_indices: np.ndarray | None`; `BaseFeedforwardCreator` gains `extractor_name: str | None = None`.
  - `lift_features(images, pixel_indices, extractor_name, device) → (P, D) float32` added to `pointcloud/utils.py` — iterates frames, converts to PIL, calls extractor, scatter-assigns via patch-rounded coords. `_assign_frame_features` helper for single-frame scatter.
  - `reproject_pixels(depth, pixel_indices, extrinsics_3x4, intrinsics) → (P, 3) float32` added to `pointcloud/utils.py` — reprojects stored source pixels with new camera poses without re-running stochastic subsampling.
  - `VGGTXCreator._postprocess` calls `lift_features` when `extractor_name` is set; `features` stored in `FeedforwardResult`.
  - `BundleAdjustment` uses `reproject_pixels` when `pixel_indices` is present — deterministic, preserves colors/features alignment. Falls back to `_reproject_after_ba` when `pixel_indices` is None (MapAnything, older creators).
- **Hit:** Rogue background agents committed ~12 unrelated commits mid-session (semantics refactor, BA fixes, docs) and twice reverted `vggtx.py` and `test_feature_lifting.py`. Had to re-apply Task 2+3 changes twice.
- **Files:**
  - `collab_splats/pointcloud/feedforward/base.py` — `FeedforwardResult` + `BaseFeedforwardCreator` fields
  - `collab_splats/pointcloud/feedforward/vggtx.py` — `unproject_and_filter_points` + `_postprocess` wiring
  - `collab_splats/pointcloud/utils.py` — `lift_features`, `_assign_frame_features`, `reproject_pixels`
  - `collab_splats/pointcloud/wrappers.py` — BA `reproject_pixels` path
  - `tests/pointcloud/test_feature_lifting.py` — 9 tests (dataclass fields, unproject return, color alignment, reproject shape/identity, lift shape/frame-alignment)
  - `tests/pointcloud/test_wrappers.py` — BA `reproject_pixels` path test
  - `worklog/specs/2026-05-20-feature-lifting-design.md`, `worklog/plans/2026-05-20-feature-lifting.md`
- **Next:** Multi-view feature averaging (V2, deferred) or begin Phase 2 dashboard hardening.

### 2026-05-20 worklog-split
- **Focus:** Split user-facing docs (`docs/`) from internal progress tracking (`worklog/`); consolidate scattered READMEs.
- **Did:**
  - Created top-level `worklog/` with subfolders `{specs,plans,notes,decisions,history/{specs,plans,prs}}` and an index `README.md`.
  - `git mv` STATE.md, WORKLOG.md, known-test-failures.md, ROADMAP.md, decisions/ (12 ADRs), history/ (101 archived files), active specs/plans/notes from `docs/superpowers/` → `worklog/`.
  - Archived completed work into `worklog/history/`: `2026-05-20-eval-reorganization` spec+plan, `2026-05-20-docs-reorg` spec+plan (latter supersedes the partial earlier same-day spec).
  - Moved scattered files into proper trees: `EVAL_NOTES.md` → `worklog/notes/2026-05-08-co3dv2-eval-notes.md`; `archive/REFACTOR.md` (untracked) → `worklog/history/plans/2026-04-16-pointcloud-submodule-refactor.md`; `collab_splats/nerfstudio/README.md` → `docs/nerfstudio/README.md`; `evals/baselines/vggt_slam/README.md` → `docs/evals/baselines/vggt_slam/README.md`.
  - Deleted `stage/NOTEBOOK_IMPROVEMENTS.md` (pure scratch). Removed empty `archive/`. Staged pre-existing root `PROGRESS.md` + `REFACTOR.md` deletions.
  - Wrote `docs/README.md` (module-doc index) and `worklog/README.md` (subfolder purposes + workflow).
  - Mechanical cross-ref sweep across all tracked `.md` files (excluding `.worktrees/`, `third_party/`); patched active-trio paths and sibling relative links in STATE/WORKLOG.
  - Updated `CLAUDE.md` doc pointers.
- **Decided:**
  - `docs/` mirrors code tree; module READMEs live at `docs/<module>/README.md` not flat `docs/modules/*.md`.
  - `worklog/` is the single internal-tracking root; `decisions/` and `history/` are subfolders of it (not siblings under `docs/`).
  - Archived files inside `worklog/history/` are immutable record — broken refs left as-is.
- **Files:**
  - `docs/README.md`, `docs/nerfstudio/README.md`, `docs/evals/baselines/vggt_slam/README.md` — new locations.
  - `worklog/README.md`, `worklog/STATE.md`, `worklog/WORKLOG.md`, `worklog/ROADMAP.md`, `worklog/known-test-failures.md`.
  - `worklog/decisions/*.md` (12 ADRs), `worklog/history/{specs,plans,prs}/*.md` (~101 files).
  - `worklog/history/specs/2026-05-20-worklog-split-design.md` — this spec.
  - `worklog/history/specs/2026-05-20-docs-reorg-design.md` — supersession header added.
  - `CLAUDE.md` — doc pointers.
- **Next:** Verify on PR; merge into `main` with the rest of `refactor/core-modules`.

### 2026-05-20 eval-reorganization
- **Focus:** Consolidate eval structure, rename ba_hightrack, unify docs
- **Did:**
  - Renamed `ba_hightrack` condition → parameterised `ba_track-density-{N}` (parsed via regex in `_validate_condition` + `_make_creator`); `query_frame_num = max(5, N // 512)` scales automatically
  - Made `--output_dir` optional; auto-names as `evals/results/{dataset}/{seq_name}/run-{YYYYMMDD-HHMMSS}/`
  - Added `evals/results/` to `.gitignore`
  - Created `docs/pointcloud/README.md` — metrics table, conditions table, dataset download table, headless usage example
  - Created `docs/pointcloud/ground-truth-evals.ipynb` (5 sections): concepts, `run_eval()`, `load_results()` + `compare_runs()`, `plot_trajectories()`, `plot_ate_per_frame()`
  - Deleted `docs/pointcloud/eval_7scenes_gt.ipynb` and `eval_co3dv2_gt.ipynb` (retired)
  - Fixed pre-existing test failures: missing `"auc"` key in `_save_outputs` test fixtures
  - 13/13 tests pass in `tests/evals/test_eval_gt_helpers.py`
- **Decided:** Notebook-first (Option A) — single `ground-truth-evals.ipynb` is entry point; `eval_gt.py` stays headless workhorse; no YAML configs (YAGNI — adds indirection with no portability benefit)
- **Files:**
  - `evals/eval_gt.py` — condition naming, auto-naming, validation
  - `tests/evals/test_eval_gt_helpers.py` — 7 new tests + fixture fixes
  - `.gitignore` — `evals/results/`
  - `docs/pointcloud/README.md` — new
  - `docs/pointcloud/ground-truth-evals.ipynb` — new (replaces two old notebooks)
  - `worklog/history/specs/2026-05-20-eval-reorganization-design.md`
  - `worklog/history/plans/2026-05-20-eval-reorganization.md`
- **Next:** PR or merge `refactor/core-modules` → `main`

### 2026-04-20
- Brainstormed PR2 approach
- Decided: parallel stacked PRs, PR2 branches from `refactor/core-modules`
- Decided: dashboard = integration test harness, MapAnything stub pattern
- Retired `PROGRESS.md` + `plans/2026-04-19-phase1-pr2-dashboard-complete.md` → this doc
- Decided: absorb `refactor/semantics` (batching work) into PR1 — two big PRs only
- Decided: `refactor/nerfstudio-submodule` safe to delete now (0 unique commits)
- Deleted `refactor/nerfstudio-submodule` (worktree + branch)
- Cherry-picked batching commits from `refactor/semantics` → 25 pass, 1 skip, 0 new failures
- Deleted `refactor/semantics` (fully absorbed)
- **Next:** Start PR1 remaining tasks (pointcloud skeleton → NerfstudioSfmCreator → MapAnythingCreator → registry), then create PR2 branch
- Moved `frame_sampling` from `semantics/` → `utils/` (general preprocessing, not semantics-specific); re-exported from semantics for compat; test moved to `tests/utils/`

### 2026-04-20 (session 3)
- Designed feedforward pointcloud integration — MapAnythingCreator + VGGTXCreator
- Confirmed `tlb-improve-mesh` = source for stage/ feedforward utilities; `tlb-improve-splatter` = parked
- Key decisions: 4-creator registry (`colmap/hloc/mapanything/vggtx`), `BaseFeedforwardCreator` template, unified disk output contract (`colmap/sparse/0/*.bin` + `transforms.json`)
- Identified 3 bugs: wrong output path in `ColmapCreator`, missing world reorientation (transform B) in `_colmap_recon_to_result()`, no disk output from `MapAnythingCreator`
- Added `CoordinateFrame` enum + `world_transform` field to `PointcloudResult` for coordinate system metadata
- `HlocCreator` calls hloc directly — no nerfstudio dep (nerfstudio remains only for `ns-train`)
- Eliminated `stage/feedforward.py` (`Reconstructor`) — superseded by `BaseFeedforwardCreator`
- Spec written: `worklog/history/specs/2026-04-20-pointcloud-feedforward-design.md`
- **Next:** Review spec → invoke writing-plans → implement Tasks 5–9 on `refactor/core-modules`

### 2026-04-20 (session 4)
- Implemented Phase 2 feedforward integration (8 tasks) via subagent-driven development
- Copied `stage/mapanything_utils.py`, `stage/preproc_utils.py`, `stage/vggt_utils.py` from `tlb-improve-mesh` + nerfstudio fork
- Fixed `_colmap_recon_to_result()`: apply transform B (`_WORLD_TRANSFORM`), populate `frame=NERFSTUDIO` + `world_transform`
- `ColmapCreator` splits from `NerfstudioSfmCreator`; output path bug fixed (`colmap/sparse/0/`)
- `HlocCreator` calls hloc directly — no nerfstudio dep; hloc imports lazy inside `reconstruct()`
- `BaseFeedforwardCreator` template: `reconstruct()` calls `_run_inference()` then `_write_transforms()` once
- `MapAnythingCreator` refactored: `confidence_percentile=35.0` (not old `conf_threshold=1.5`); correct `stage/mapanything_utils` pipeline
- `VGGTXCreator` added: `use_global_alignment=False` default; calls `run_vggt(image_dir, colmap_dir=output_dir/"colmap")`
- Bug caught in review: `parents[3]` → `parents[2]` in `_add_stage_to_path()` (was inserting `/workspace` not repo root)
- Registry updated to `{"colmap", "hloc", "mapanything", "vggtx"}`; old `"sfm"/"feedforward"` keys removed
- **Next:** PR1 ready for review — open PR against `main`; then start PR2 (`refactor/dashboard-complete`)

### 2026-04-28
- Squash-merged `lc-01-refactor` → `refactor/core-modules` (commit `91b7c6b`)
- Spec 1 complete: loop closure sub-package split — `loop_closure.py`, `pose_graph.py`, `submap.py` → `loop_closure/` sub-package (6 files). Zero behavior change. 12 LC tests pass.
- Squash-merged `lc-02-bugs` → `refactor/core-modules` (commit `e1ae485`)
- Spec 2 complete: 6 silent bug fixes — F3 (dedup_overlap), F5 (verifier raises), F7 (assert_world_to_cam), F8 (cosine threshold), F9 (NMS), F10 (max_loops 1→5). 21 LC tests pass (88 total, 4 pre-existing failures unchanged).
- Squash-merged `lc-03-posegraph` → `refactor/core-modules`
- Spec 3 complete: 4 pose-graph rewires — F11 (Umeyama SE(3) + overlap_region_align), F2 (chained init via accumulated_T), F1 (inter-submap edges), F4 (verifier returns (bool, poses|None); VGGTXCreator extracts fresh pose_enc). 98 pointcloud tests pass (4 pre-existing failures unchanged). Submap grew world_points + world_points_conf fields.
- Squash-merged `lc-04-robustness` → `refactor/core-modules`
- Spec 4 complete: robustness layers — F6 (3-way noise: odom σ_t=0.05, inter σ_t=0.20, loop_intra σ_t=0.30, loop_inter σ_t=0.50), Huber kernel on loop edges (k=1.0), per-edge translation-magnitude down-weighting (α=0.1), translation-jump pre-add gate (max_jump_ratio=0.2). Orchestration extracted: closure.py now houses run_pose_graph_optimization, merge_submap_outputs, translation_jump_check. 104 pointcloud tests pass (4 pre-existing failures unchanged). ADRs 002-005 record deferred SL(4), GraphMap, FrameTracker, per-backend noise-tuning decisions.
- **LC sequence done.** Next: open PR `refactor/core-modules` → `main`

### 2026-05-08 (session 2) — BA eval script split

Spec: `worklog/history/specs/2026-05-08-ba-eval-script-split-design.md`
Plan: `/root/.claude/plans/please-look-at-this-partitioned-gadget.md`

- **Goal:** make BA condition runnable via `evals/eval_gt.py` (not via `bundle_adjustment.ipynb`); align with eval-harness design (script = compute, notebook = viz). Repurpose `bundle_adjustment.ipynb` as small API-docs notebook.
- **SIGKILL diagnosis:** previous SIGKILL 137 in `bundle_adjustment.ipynb` runs was real cgroup OOM, not Claude harness. `/sys/fs/cgroup/memory.max = 49999998976` (46.6 GB) — container cap, not 503 GB host. Earlier debug log misdiagnosed. `dmesg` denied in container so kernel kill events unreadable. Plan implication: keep frame count modest, run via tmux.
- **Dataset acquired:** `bash evals/download_7scenes.sh chess /workspace/collab-splats/data/7scenes` (~3 GB zip → 1000-frame seq-01.zip nested at `data/7scenes/chess/chess/`). Extracted seq-01 manually (download script doesn't recurse). Layout flat (`frame-XXXXXX.color.png` + `.pose.txt` side by side) as expected by `datasets.py:_load_7scenes`.
- **Bug surfaced + fixed: `BundleAdjustment` lacked `outputs`/`raw_outputs` proxy properties.** First `eval_gt.py --conditions baseline ba lc` run completed baseline (ATE 0.0553m matches prior session) and BA optimization itself (`Loss: 7.1e10 → 1.8e5` over 3 LM steps), then crashed at `eval_gt.py:66` with `AttributeError: 'BundleAdjustment' object has no attribute 'outputs'`. `LoopClosure` proxies `outputs`/`raw_outputs` to its `base` (wrappers.py:112-122); `BundleAdjustment` did not, so `creator.outputs.extrinsics` failed only for the BA condition. Fix: add `@property outputs` + setter and `@property raw_outputs` + setter to `BundleAdjustment` (`collab_splats/pointcloud/wrappers.py`). TDD: 2 regression tests added in `tests/pointcloud/test_wrappers.py` (`test_bundle_adjustment_outputs_proxies_to_base`, `test_bundle_adjustment_raw_outputs_proxies_to_base`); both fail before fix, pass after.
- **Stale tests in `test_bundle_adjustment.py` brought current.** `test_extract_tracks_vggsfm_shape` and `test_extract_tracks_vggsfm_conf_4d` mocked the pre-2026-05-08 `predict_tracks` API (returned per-query-frame lists). After session-1 fix, `extract_tracks_vggsfm` does `np.asarray(...)` on the now-concatenated single ndarray. Mocks rewritten to return single concatenated arrays. 30 BA + wrapper tests green.
- **End-to-end eval clean** (chess seq-01, 50 frames, all 3 conditions):

  | condition | ATE RMSE | RPE trans RMSE |
  |-----------|----------|----------------|
  | baseline  | 0.0553m  | 0.0069m        |
  | ba        | 0.0577m  | 0.0068m        |
  | lc        | 0.0548m  | 0.0073m        |

  BA marginally worse than baseline on this short sequence (no loop closures available to anchor the LM solver); LC marginally better. Numbers consistent with prior baseline; BA + LC behaviour expected for short seq-01.
- **Run #2 cgroup OOM at 50 frames** when running with concurrent RSS sampler in another bg shell — sampler loop ate cgroup budget and pushed BA over the 46.6 GB cap mid-track-extraction. Run #3 (no concurrent sampler) succeeded. Lesson: don't run side-shells while heavy eval runs.
- **Viz notebook refreshed.** `docs/pointcloud/eval_7scenes_gt.ipynb` — fixed `RESULTS_DIR` from stale `../../eval_results/chess_seq01` to `../../evals/results/chess_seq01`, re-executed in-place. All 3 conditions render in metrics table + 3D trajectory + per-frame ATE plots.
- **`docs/pointcloud/bundle_adjustment.ipynb` rewritten as small API-docs notebook.** Old version: 194-frame BA debugging artifact mixing compute + viz, killed by SIGKILL. New version: 5-frame canned chess subset, two cells (wrapper API + manual API), runs end-to-end <90s with no errors. Bug caught in rewrite: old notebook used `result.pts3d`/`result.extrinsics` but `PointcloudResult` exposes `points`/`camera_poses` — corrected.
- **Files modified this session:**
  - `collab_splats/pointcloud/wrappers.py` — `BundleAdjustment.{outputs,raw_outputs}` proxy properties
  - `tests/pointcloud/test_wrappers.py` — 2 new regression tests
  - `tests/pointcloud/test_bundle_adjustment.py` — 2 stale-mock fixes
  - `evals/datasets.py` (no change this session — already correct)
  - `evals/eval_gt.py` (no change this session — wiring already correct)
  - `docs/pointcloud/eval_7scenes_gt.ipynb` — RESULTS_DIR path fix + re-executed
  - `docs/pointcloud/bundle_adjustment.ipynb` — full rewrite as API how-to
  - `worklog/history/specs/2026-05-08-ba-eval-script-split-design.md` — new spec
  - `worklog/WORKLOG.md` — this entry

### 2026-05-08
- Fixed GT eval pipeline end-to-end: baseline + LC conditions run clean on 7-Scenes chess seq-01 (50 frames)
- **`evals/eval_gt.py`**: replaced `_CONDITIONS` constructor-kwargs pattern with `_make_creator` factory using `BundleAdjustment`/`LoopClosure` wrappers; argparse choices updated
- **`evals/datasets.py`**: fixed flat layout glob (`.color.png` / `.pose.txt` side by side, no `color/`/`pose/` subdirs); invert cam-to-world poses to world-to-cam
- **`setup_bundle_adjustment.sh`**: cloned bae at `0.2` (not `0.2.1`) — pypose does exact equality check `version == "0.2"`; added `import pypose` before `import bae` in smoke test to break circular init
- **`closure.py` → `merge_submap_outputs` (3 bugs):**
  1. `extrinsic` (size N=50) misaligned with depth/images (size M=58 with overlap) → rebuild from per-submap slices of `corrected_extrinsics` using `frame_start : frame_start + len(poses)` 
  2. `images` is torch tensor — numpy-only loop skipped it, `merged["images"]` stayed first-submap-only (size 20) → explicitly rebuild from `submap.frames` as numpy
  3. `corrected_extrinsics` is `(N, 4, 4)` but raw_outputs convention is `(K, 3, 4)` → slice `[:3, :]` to avoid `_raw_to_world_points` appending bottom row and producing `(K, 5, 4)`
  4. `FeedforwardResult.extrinsics` ended up size 58 (overlap-expanded) → added `extrinsic_global_4x4` key (N, 4, 4 deduped) in merge; `_postprocess` prefers it via `raw_outputs.get("extrinsic_global_4x4", extrinsic_4x4)`
- **Results** (chess seq-01, 50 frames, no loop closures detected — short sequence, no revisits):

  | condition | ATE RMSE | RPE trans RMSE |
  |-----------|----------|----------------|
  | baseline  | 0.0553m  | 0.0069m        |
  | lc        | 0.0559m  | 0.0073m        |

- BA condition still untested (requires `vggsfm` track extraction — separate investigation)

### 2026-04-23
- Squash-merged `feat/loop-closure` → `refactor/core-modules` (commit `d332a94`)
- Loop closure fully implemented: `Submap`, `LoopClosureConfig`, `ImageRetrieval`, `PoseGraph` (GTSAM SE(3)), `BaseFeedforwardCreator` integration
- `FeedforwardResult.extrinsics` now `(N,4,4)` homogeneous; COLMAP boundary slices to `(N,3,4)`
- Feature opt-in: `enable_loop_closure=False` default — zero impact on existing paths
- Also absorbed: rich/tqdm progress logging, pycolmap `cam_from_world` API fix, `clean_pointcloud` sentinel defaults
- Deleted `feat/loop-closure` branch + worktree
- **Next:** Open PR against `main`
