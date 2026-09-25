# Geometry release cleanup — design

Date: 2026-09-25 · Branch: `clean/geometry-release` off `clean/final` (`311ca5c8`) · Status: approved 2026-09-25

Rules: [017-release-cleanup-rules.md](../decisions/017-release-cleanup-rules.md) — binding,
not restated here. Reference runs: [preproc](2026-09-24-preproc-release-cleanup-design.md),
[semantics](2026-09-24-semantics-release-cleanup-design.md).

## Goal

Make `collab_splats/geometry/` release-ready: brief docs, no dead or deprecated code,
tunables as kwargs, evaluation code out of the library — **without changing what loop
closure computes**.

## Decisions (from brainstorm)

- **Loop closure output is frozen.** No change may alter LC poses, points or loop edges on
  any input that runs today. Two gates prove it (see Testing). Every LC fallback that
  degrades instead of failing stays as a fallback.
- **API breaks allowed** (017); callers, configs and tests change in the same commit.
- **`scale_method` keeps `rotation_only` and `none`** (user decision 2026-09-25, amends
  the earlier keep-all-four):
  - `se3` deleted: identical to `rotation_only` since `1372ac28` made `T` the K ratio
  - `pairwise_dist` deleted: fixed the old `se3` translation bias, which no longer exists;
    no config or preset selects it
  - `PoseGraph.add_submap` keeps its signature, `debug_out` included
- **Incremental BA stays** (`increment_size`, `_refine_incremental`); evals preset
  `incremental_ba-N` uses it.
- **Kept `except Exception` handlers on the LC path are narrowed**, not removed: each
  catches only the types its callee raises, logs a warning, same fallback. Success path is
  unchanged; an unexpected type now propagates.
- **One output contract.** `pointcloud.zarr` is the single dense contract for both
  methods: feedforward creators and sfm (`depth_align.result_from_reconstruction`) both
  write it through the `FeedforwardResult` type, with K at depth (model) resolution and
  `original_coords` mapping back to original pixels. There is no `feedforward.zarr`.
  Geometry docs describe that contract, not a "feedforward" one:
  - BA, verification and metrics read `pointcloud.zarr`; verification and metrics run for
    both methods, while BA refuses sfm at the refine stage
  - loop closure alone is feedforward-only (it wraps a creator's forward pass)
- **BA raises on original-resolution K** instead of rescaling it. Both writers of the
  contract store depth-res K; only a `pointcloud.zarr` from before the model-res fix hits
  the rescale. The error names the contract and says to re-run the pointcloud stage.
- **Metrics:** `{"available": False}` stays only when there is nothing to measure: no
  verification.json, no frame images, or no view pair that yields a residual or a
  correlation. Every other failure raises.
- **Tests:** preproc-style prune — delete tests of deleted code, drop history guards,
  merge duplicates; no readability rewrite.
- **Evaluation code moves to `evals/`**: trajectory metrics and pose-graph diagnostics.
- **Isolation:** worktree `.worktrees/geometry-release`, `third_party/*` symlinked in;
  every gate runs as `cd <wt> && PYTHONPATH=<wt> python ...` and prints
  `collab_splats.__file__`.
- **Hands-off:** notebooks on `clean/final` and `.worktrees/tutorial-rework`. Impact is
  listed under Tutorial impact, not edited.

## Starting point

Contract release checks on the untouched branch (geometry not yet in `PACKAGES`):

- `silent-fallback` ×8: `bundle_adjustment.py:138`, `loop_closure/graph.py:205`,
  `loop_closure/wrapper.py:465,477,499,522`, `metrics.py:495`, `metrics.py:599` (`or 0` ×2)
- `numeric-constant` ×2: `verification.py:34 DEFAULT_OVERLAP`, `graph.py:440 _MIN_CONF_POINTS`
- `banned-word` ×15: `metrics.py` (11), `transforms.py` (3), `verification.py` (1)
- `comment-cap` ×21 across `bundle_adjustment`, `graph`, `matching`, `wrapper`,
  `metrics`, `transforms`, `verification`

Everything else below comes from reading the files.

## Round 1 — prose only

No code change. Proof: AST equal after deleting every docstring statement on both sides
and stripping comments, plus one sanity mutation showing the check can fail.

### Per file

- Contract wording, every file: "FeedforwardResult" in prose becomes "the `pointcloud.zarr`
  result" wherever the code is method-agnostic; "feedforward" stays only for loop closure.
- Every `comment-cap` and `banned-word` hit in Starting point is fixed here, including
  runs not named below.
- `__init__.py` (package + `loop_closure/`) — module docstring to 2-4 bullets naming what
  each module holds.
- `transforms.py` — `estimate_intrinsics_from_points`: drop the LoGeR measurement lore
  (docstring + the 13-line comment at :185) to a one-line why.
- `bundle_adjustment.py` — class docstring (:89) says it refines a `pointcloud.zarr`
  result and that the refine stage refuses sfm, not "any FeedforwardResult regardless of
  source creator"; the `_extract_tracks_vggsfm` numpy-input comment (:680) blames a
  "windowed LC path" that no longer exists. The branch stays: sfm's
  `result_from_reconstruction` stores `images` as numpy float32. The comment names that
  source instead.
- `bundle_adjustment.py` — `BundleAdjustmentConfig` gets a docstring; per-field comments
  to one line each; drop the `increment_size` sweep results (:76-77);
  `_scale_intrinsics_to_model` docstring (:480-484) stops claiming stored K is at
  original resolution (Round 2 then cuts the function to the check); split the 6- and 5-line comment runs.
- `verification.py` — module docstring stops saying "feedforward reconstructions": it
  verifies any `pointcloud.zarr` reconstruction (`_write_frames` already handles
  sfm's shared camera); drop stale "Task 5 fills in" (:311); `PairStats` loses "Measured";
  comment runs split; the reconstruction copy gets a one-line why (pycolmap mutates its
  argument).
- `metrics.py` — module and `build_reconstruction_quality_report` docstrings state the
  report reads `pointcloud.zarr` for either method; the image-range comment (:337) names
  the contract, not two backends; every `measured` out of docstrings and comments;
  8-line comment runs to header + bullets.
- `loop_closure/graph.py` — `decompose_camera` drops "see closure.py" and the ATE-gap
  history; the comment at :307 stops claiming `se3` is the default; `add_loop_edge` drops
  "monolith lines ~581-624"; `PoseGraph` class and method docstrings to contract.
- `loop_closure/submap.py` — `Submap` gets a docstring (fields, world-to-cam convention);
  drop the `TODO(spec-2)` comment.
- `loop_closure/matching.py` — `LoopMatchQueue.push` / `get_matches` get docstrings;
  `LoopMatch.similarity_score` documented as an L2 distance (lower = more similar).
- `loop_closure/map.py` — class and method docstrings to contract.
- `loop_closure/wrapper.py` — `LoopClosureConfig` gets a docstring; one comment line per
  field; drop "4 = old default" (:53) and "fallback 0.85 (VGGT-SPARK calibration)" (:126);
  `LoopClosure` docstring to contract; 7-line run at :605 split.
- `docs/source/api/geometry.rst` — drop the nonexistent `loop_closure.closure` automodule
  (Round 2 drops `loop_closure.eval` with the move).

## Round 2 — code, one commit per logical change

Each row: `name | verdict | exact action`.

### transforms.py

| name | verdict | action |
|---|---|---|
| all public names | keep | — |
| `umeyama_se3`, `umeyama_sim3` degenerate input → identity | raise | `ValueError` on <3 points or zero total weight; not on the LC path (callers: BA `_carry_dropped_frames`, which guards <3, and the moved eval code); `test_umeyama_sim3_too_few_points_returns_identity` flips |
| `_compute_weighted_median` | keep | two in-file callers |

### global_alignment.py

| name | verdict | action |
|---|---|---|
| module | delete | parked, no caller; drop the lazy export in `__init__`, the commented-out block in `pointcloud/feedforward/vggtx.py:28,347`, and the `CLAUDE.md` architecture line |

### bundle_adjustment.py

| name | verdict | action |
|---|---|---|
| `_last_loss_history` | keep | rename to `loss_history` (public: outside callers read it); `wrapper/reconstructor.py:1225`, `evals/scripts/ba_start_at_gt.py:161`, tests in the same commit; `refine.json` key `loss_history` unchanged |
| `_UNSET`, `_optimize(max_reproj_error=, lm_steps=)` | delete | only tests passed them; tests set the config instead |
| `_optimize` early return (:313) when <2 frames or <2 points stay active after track filtering | raise | `ValueError` naming both counts; today it logs a warning and returns the input unrefined, so a scene with too few tracks now fails the refine stage |
| `_get_default_solver` | keep | the `(ImportError, RuntimeError)` → PCG fallback logs a warning |
| track-cache `except Exception` (:138) | raise | narrow to the errors an unreadable / stale zarr raises (`KeyError`, `ValueError`, `OSError`, zarr errors); anything else propagates |
| `_extract_tracks_vggsfm` | keep | drop parameter defaults; the only caller passes config values |
| `_scale_intrinsics_to_model` | raise | keep only the model-res check; K not at model resolution raises `ValueError` naming the re-run. The rescale and the rescale-back in `refine` go. `evals/scripts/ba_start_at_gt.py` uses the check. |
| `_scale_intrinsics_to_original` | merge | moves to `metrics.py`, its only caller |
| `increment_size`, `_refine_incremental`, `_refine_allonce` | keep | evals preset uses incremental |
| `shared_camera`, `_reproject_per_camera`, `_reproject_shared` | keep | evals per-camera conditions |
| `_carry_dropped_frames`, `_filter_observations`, `_compute_tracks_cache_key`, `_BAModel` | keep | — |
| upstream bae constants | keep | ported, cited |

### verification.py

| name | verdict | action |
|---|---|---|
| `DEFAULT_OVERLAP` | inline | becomes the kwarg default `overlap=10` |
| `_pair_pose_errors` | keep | rename to `pair_pose_errors` (public: evals imports it); `evals/scripts/eval_verification.py:29,180` in the same commit |
| `verify_reconstruction`, `PairStats`, `VerificationResult`, `clean_for_json`, `_distribution`, `_triangulate_and_summarize`, `_write_frames`, `_write_report` | keep | — |

### metrics.py

| name | verdict | action |
|---|---|---|
| `{"available": False}` at :148, :419, :478, :587 | keep | nothing to measure: no residual pair, no correlation pair, no frame images, no verification.json |
| `extract_photometric` `except Exception` (:495) | delete | a failure raises |
| `{"available": False}` running-error guard (:667) | raise | not an absent input |
| `s.get("num_matches") or 0`, `s.get("num_inliers") or 0` (:599) | raise | index directly; `verify` always writes both |
| `rel_thresh=0.05` (:570) | make-kwarg | on the function that uses it, threaded to `build_reconstruction_quality_report` |
| quantile grid (:188) | keep | report format |
| inline imports | keep | heavy optional deps only; others move to the top |
| `compute_depth_error`, `depth_error_in_pixels`, `read_quantiles`, `bounded_residual`, `residual_bin_edges`, `compute_photometric_ncc`, `build_reconstruction_quality_report` | keep | — |

### loop_closure/eval.py, loop_closure/edge_trace.py → evals/

| name | verdict | action |
|---|---|---|
| `ate_translation`, `rpe`, `auc_at_threshold`, `umeyama_align` | merge | move to `evals/trajectory_metrics.py` |
| `capture_pose_graph_loss`, `_classify_edges`, `_per_edge_error` | merge | move to `evals/pose_graph_diagnostics.py` |
| `compose_slam_chain`, `edge_divergence` | merge | move to `evals/pose_graph_diagnostics.py` |

Callers updated in the same commit: `evals/metrics.py:119`, `evals/scripts/eval.py:62`,
`evals/scripts/ba_start_at_gt.py:37`, `docs/source/api/geometry.rst`, and the tests below.
Tests `test_eval_metrics.py`, `test_auc_metric.py`, `test_loop_closure_eval.py` move to
`tests/evals/`; `test_loop_edge_chain.py`, `tests/evals/test_compare_loop_edges.py`,
`tests/evals/test_eval_gt_helpers.py` change their import path.

### loop_closure/graph.py

| name | verdict | action |
|---|---|---|
| `PoseGraph.optimize` `except Exception` | raise | catch `RuntimeError` (GTSAM's pybind translation); warning + keep initial values unchanged |
| `decompose_camera` `assert` | raise | `ValueError`; same condition |
| `estimate_scale_pairwise` 1.0 return | keep | LC output frozen |
| `_estimate_scale_pairwise_dist` | delete | only `pairwise_dist` used it |
| `scale_method` `se3`, `pairwise_dist` | delete | `se3` == `rotation_only`; `pairwise_dist` unselected; task T29b |
| `scale_method` `rotation_only`, `none`; `add_submap` (incl. `debug_out`) | keep | user decision |
| `umeyama_se3` / `umeyama_sim3` import | delete | only re-exported for `eval.py`, which moves |
| `_loop_chain_relatives` `K_*=None` defaults | delete | prod passes all; tests at `test_loop_edge_chain.py:233,247` pass them |
| `_MIN_CONF_POINTS` | make-kwarg | `PoseGraph(min_conf_points=100)`, read by `add_submap` and passed to `_lc_anchor_scale`; `add_submap` signature unchanged |
| `dedup_overlap`, `_resolve_frame_node`, `_cam_local_points`, `_lc_anchor_scale`, `extract_extrinsics`, `add_homography`, `add_between_factor`, `add_prior_factor`, `get_homography` | keep | — |

### loop_closure/map.py

| name | verdict | action |
|---|---|---|
| `get_corrected_extrinsics` | inline | pass-through; `wrapper.py:603` calls `graph.extract_extrinsics` |
| `get_latest_submap`, `get_largest_key`, `get_submap` | delete | tests only |
| `get_world_pointcloud`, `ordered_submaps_by_key`, `add_submap` | keep | — |

### loop_closure/submap.py

| name | verdict | action |
|---|---|---|
| `raw_outputs` field | delete | set at `wrapper.py:269`, freed at :298, never read |
| `get_world_points`, `get_poses_world` | delete | tests only (`test_submap_reprojection.py`) |
| `filter_data_by_confidence` | inline | one caller |
| `_CONF_PCT` | keep | the contract check does not flag it |
| `set_dense_points`, `get_points_in_world_frame`, `get_points_colors`, `get_all_poses_world`, `assert_world_to_cam` | keep | — |

### loop_closure/matching.py

| name | verdict | action |
|---|---|---|
| `translation_jump_check` | delete | config default `inf` always accepts and nothing sets it; drop the call in `wrapper.py:348`, `test_translation_jump.py`, and the patch in `tests/pointcloud/feedforward/test_verify_lc_data.py:277` |
| `find_loop_closures(nms_frame_distance=0)`, `LoopMatchQueue(nms_frame_distance=0)` | keep | drop the default; the caller passes the config value |
| `LoopMatch`, `LoopMatchQueue`, `find_loop_closures` | keep | — |

### loop_closure/wrapper.py

| name | verdict | action |
|---|---|---|
| `LoopClosureConfig.max_jump_ratio` | delete | with `translation_jump_check` |
| `LoopClosureConfig.lc_threshold_l2` | delete | alias; `find_loop_closures` reads `lc_retrieval_threshold` |
| `getattr(base, "default_verify_match_ratio", 0.85)` | raise | `base.default_verify_match_ratio`; every creator inherits it from `feedforward/base.py:1087`; `test_explicit_config_none_falls_back_to_085_without_attr` is deleted |
| `getattr(base, "model_height"/"model_width", 0)` (:632) | delete | no creator has either attribute, so both are always 0 and the fail-fast `ValueError` below fires; the branch goes and `model_height is None` joins that raise's condition. Same inputs raise, same message |
| identity-intrinsics `.get` default, `torch.zeros(k,3,1,1)` frames | raise | only if the plan proves no LC-capable creator reaches them; otherwise keep |
| `_enough_frames` fallback to plain inference | keep | LC behavior frozen |
| DINO-SALAD load `except Exception` (:522) | raise | `(ImportError, OSError, RuntimeError)`; warning + same fallback |
| `_viz_push_submap`, `_viz_draw_loops` `except Exception` | raise | types pinned in the plan by reading `Viewer`; warning + same no-op |
| `_viz_reupload_all` outer `except Exception` | delete | every push inside it is already guarded |
| `loops_found` / `verified` | merge | one counter |
| viewer `max_points=50000` | make-kwarg | `LoopClosureConfig.viz_max_points = 50000`; every caller builds `LoopClosure` from a config |
| `_trim_forward_outputs` | delete | tests only |
| `_camera_centers_from_poses` | keep | two callers |
| `_last_submaps`, `_last_lc_submaps`, proxy `__getattr__` | keep | — |

### Outside the package

| name | verdict | action |
|---|---|---|
| `VGGTOmegaCreator.default_max_jump_ratio` + its comment | delete | nothing reads it |
| `default_verify_match_ratio` ClassVars | keep | per-model calibration, read by the wrapper |
| `geometry/__init__` lazy `__getattr__` | keep | real cycle: `wrapper` → `pointcloud` → `geometry.transforms` |

## Caller sweep

Checked with `git grep` over `collab_splats configs scripts tests evals docs/source`.

| changed | outside callers | effect |
|---|---|---|
| `_last_loss_history` → `loss_history` | `wrapper/reconstructor.py:1225`, `evals/scripts/ba_start_at_gt.py:161` | updated; `refine.json` unchanged |
| BA raises on original-res K / <2 active frames | `wrapper/reconstructor.py` refine stage, `ba_start_at_gt.py` | old zarrs need a pointcloud re-run |
| `_pair_pose_errors` → `pair_pose_errors` | `evals/scripts/eval_verification.py` | updated |
| eval / edge_trace move | `evals/metrics.py`, `evals/scripts/eval.py`, `ba_start_at_gt.py`, `docs/source/api/geometry.rst` | updated |
| `global_alignment` delete | `vggtx.py` comments, `CLAUDE.md` | updated |
| `max_jump_ratio`, `lc_threshold_l2` delete | `vggt_omega.py` ClassVar; no config sets either | removed |
| metrics raise | `wrapper/reconstructor.py:1601` quality stage | a failing photometric pass now fails the stage |

No `configs/*.yaml` or `evals/configs/*.yaml` key is removed.

## Tutorial impact (not edited here)

- `clean/final` `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` reads
  `ba._last_loss_history` → `ba.loss_history`; passes `max_reproj_error` / `lm_steps` to
  the config (unchanged).
- `clean/final` `slam_loop_closure.ipynb`, `evals/ground_truth_evals.ipynb` import
  `ate_translation` / `rpe` / `umeyama_align` from `loop_closure.eval` → `evals.trajectory_metrics`.
- `tutorial-rework` `02_pointcloud/refinement.ipynb`: uses `invert_poses` and
  `refine.json["loss_history"]` — both unchanged. Its "silently falls back" prose about
  LC below `submap_size` stays true (`_enough_frames` kept).

## Deferred

- `_classify_edges` puts loop edges in the sequential group (assumes Robust noise; all
  edges use Diagonal) — a real bug in diagnostics, fixed separately after the move.
- `LoopMatch.similarity_score` → `l2_distance` rename (16 sites).
- `FeedforwardResult` names a method-agnostic contract (sfm writes it too). A rename
  (e.g. `DenseResult`) belongs to the `pointcloud` package, not this cleanup; geometry
  keeps importing the current name.
- LC silent fallbacks kept by decision: scale estimators' 1.0, `_enough_frames`.

## Testing

- **Baseline** on the untouched worktree before any edit: `tests/geometry tests/evals
  tests/pointcloud/feedforward tests/wrapper tests/test_docstring_contract.py`. Record
  pass/fail/skip; SKIP count compared on every later run.
- **Gate per commit:** same set, `__file__` proof line, no `| tail`, no `--tb=no`. No
  full-suite runs.
- **LC parity gate 1 — bit-exact synthetic.** Before any edit, a scratchpad script runs
  `LoopClosure.run_inference` on the `test_wrapper` fixtures (every `scale_method`,
  `loop_edge_timing` deferred and live; after T29b only `rotation_only` and `none` are
  compared, and the deleted `se3` case must equal the kept `rotation_only` baseline) and saves poses, points, colors and loop edges.
  After every Round 2 commit that touches `loop_closure/` or `transforms.py`, the same
  script must reproduce them with `np.array_equal`. No tolerance.
- **LC parity gate 2 — real scene.** `evals/scripts/eval.py` with
  `evals/configs/7scenes.yaml`, chess, `lc` condition, on the baseline and at the tip, in
  tmux, nothing else running. ATE, RPE and accepted-loop count must match.
- A failure at either gate reverts that commit; the gate is never loosened.
- Each behavior change gets a test: the new raises (BA <2 active frames/points, original-res K,
  `umeyama_*`, `decompose_camera`, metrics running-error), narrowed handlers still falling back on
  their named type and propagating another, `min_conf_points`, `rel_thresh`,
  `viz_max_points`, and LC with no dense submap still raising the fail-fast
  `ValueError`.
- Tests of deleted code are deleted.
- **End:** add `geometry` to `PACKAGES` and `RELEASED` in
  `tests/test_docstring_contract.py`; it must pass. The `CLAUDE.md` Code Style line that
  lists the enforced packages gains `geometry` in the same commit.

## Out of scope

- Readability rewrite of `tests/geometry/`.
- Any notebook; `configs/`; any package outside `geometry` except the named caller lines.

## Amendment 2026-09-25 — remove VGGT-SPARK and VGGT-SLAM

User decisions (2026-09-25):

- SPARK and VGGT-SLAM are parity references, never shipped; their only home is a new
  user-facing `docs/parity.md`.
- `evals/baselines`: SPARK/SLAM-produced runs deleted, key numbers summarized in
  `docs/parity.md`; our-backbone runs kept.
- `hloc` kept (out of scope).
- Unused clones `third_party/{bae,VGGT-X,vggt-omega,xfeat,vggt_spark,VGGT-SLAM}` removed
  from disk (done; patches + HEADs saved to the session scratchpad).

| Item | Verdict |
|---|---|
| `pointcloud/feedforward/vggt_spark_creator.py`, registry + `__all__` + `_SPARK_AVAILABLE` | delete |
| SPARK tests (`test_vggt_spark_native_similarity`, `test_spark_mv_subprocess`, `test_spark_load_guard`, `test_spark_overrides_vggtx_calibration`) | delete |
| SPARK/SLAM mentions in `collab_splats` comments/docstrings that explain *our* constants | reword to cite `docs/parity.md` |
| Port attributions (`MIT-SPARK/VGGT-SLAM` @ commit, file:line) in `loop_closure/*`, `README.md` | keep — attribution rule |
| `eval.py` `vggt_spark` backbone, SLAM overlay (`--selected_frames`, TUM overlay, `_COLORS["vggt_slam"]`) | delete |
| `eval_compare.py` `vggt_slam` method, `eval_similarity_calibration.py` SPARK reference row | delete |
| `pose_graph_diagnostics.py` `compose_slam_chain` / `edge_divergence` + `test_compare_loop_edges.py` | delete; `capture_pose_graph_loss` kept |
| `run_vggt_slam.py`, `test_run_vggt_slam.py`, `setup/vggt_slam.sh`, Dockerfile uv-for-vggt_slam line | delete |
| `cross_model_chess.yaml` `vggt_spark` backbone; `test_eval_config.py` sample backbone | drop / switch to `vggtx` |
| `evals/README.md` parity narrative (links 3 missing specs) | move to `docs/parity.md` |
| `third_party/README.md`, `pyproject.toml:115-118` comments, `docs/known-test-failures.md` SLAM rows | correct |
| `feedforward_methods.ipynb` SPARK row | tutorial-rework follow-up (no notebook edits) |

Gates: G′ (counts drop only by deleted tests; no new failures), P1 PARITY OK, astcmp on
comment-only `collab_splats` edits.
