# Geometry cleanup + sparse tracks + registry retrieval

Date: 2026-10-08 · Branch: `feat/rgbd-ba-cf` (worktree `.worktrees/rgbd-ba-cf`)

## Goal

- Make `collab_splats/geometry/` readable: remove dead code, single-use helpers, duplicate state
- Reuse the package's own functions instead of local copies
- Tracks become flat sparse observations so host RAM scales with observations, not frames × tracks
- Retrieval models are chosen by registry name everywhere; PE-CLIP removed, MegaLoc added

## Ground rules

- Behavior kept; every option stays (solver, refine_focal, track_source, seed_fraction, shared_camera,
  use_depth, use_photometric, increment_size, lm_tol/lm_patience, LC window BA, retrieval_k, depth_tol)
- No new classes except the MegaLoc registry backend; a new helper only when it replaces ≥2 copies
- No single-use helpers; long functions get numbered phase-block comments instead
- Never re-propose: tracks-free BA, photometric-only BA, per-frame depth grid, TSDF bands,
  multiview depth-stack filtering, depth_trunc
- Code style: the full contract (`tests/test_docstring_contract.py` incl. release + round-3 checks,
  `tests/test_import_style.py`) on every touched file; one-line block comments; US spelling;
  params over constants; no nested calls; flat tests; per-file `black -l 120` + `isort`, never repo-wide
- Risk tags: 🟢 bit-identical · 🟡 noise-level (measured at the parity gate)

## Order

1. core (transforms, projection, metrics, photometric, `__init__`)
2. tracks + BA (sparse format crosses the boundary, so they land together)
3. loop_closure
4. retrieval registry (localization/retrieval.py, tracks, LC, localizer)
5. tests, parity harness, docs

The gate runs after every step; step 2 also runs the parity check.

## 1. Core

### transforms.py
- KEEP `_umeyama`, `umeyama_se3`, `umeyama_sim3` (public API, user decision)
- CUT `rotation_angle_deg` (test-only; evals uses evo) and `OPENGL_TO_OPENCV` (test-only) 🟢
- INLINE `rotation_align_vectors` into `fit_dominant_plane`; drop its unreachable antiparallel branch
  (the normal is flipped to z ≥ 0 first) 🟢
- KEEP `extract_intrinsics` (user, 2026-10-08: useful; inline reverted)
- `rescale_intrinsics` / `shift_intrinsics`: drop the broadcast guard; in-place `*=` already raises 🟢
- Trim docstrings of `transform_points`, `decompose_camera`, `estimate_intrinsics_from_points` 🟢

### projection.py
- NEW `reprojection_error(points, w2c, K, px) -> (N,)`: pixel distance, `inf` behind camera or NaN.
  Replaces `tracks._transfer_err` and the batched error in BA `_filter_observations` 🟡
- Trim `unproject` / `depth_residual` docstrings 🟢

### metrics.py
- CUT `PairStats`; `_collect_pairs` fills a column dict 🟢
- MERGE `compute_depth_error` into `_collect_pairs` 🟢
- INLINE single-use `residual_bin_edges`, `bounded_residual`, `_histogram_counts`, `_frustum_overlap`
  (keep its empty guard), `_frame_medians` 🟢
- `compute_photometric_ncc`: hand-written unproject/project → `unproject` / `project` 🟡

### photometric.py / `__init__.py`
- `photometric.KEYS` tuple (sample order BA's `forward` unpacks); collapse the `to_numpy` pair 🟢
- `__init__.__getattr__` docstring 14 → 5 lines 🟢
- KEEP pixel-corner unproject (bae port), `_compute_weighted_median`, `multiview_depth_confidence`

## 2. Tracks + BA (sparse)

### Track format
```
extract_tracks(...) -> (frame (M,) int32, track (M,) int32, xy (M, 2) float32,
                        score (M,) float32, pts3d (P, 3) float32)
```
- One row per observation; replaces the dense (N, P) grid (1.3% filled at 294 frames)
- Matcher score is 1.0; VGGSfM's (N, P) output converts at its boundary (`vis > 0` kept;
  `vis_thresh` still applies in BA)
- Cache stores the same 5 arrays; cache key includes the retrieval name

### tracks.py (682 → ~340)
- `extract_tracks` absorbs `_extract_uncached` and `_cache_key` 🟢
- `extract_tracks_vggsfm` → private `_vggsfm_tracks` returning flat arrays 🟢
- `build_tracks` is one function with numbered phase blocks:
  1. extract (inlined `_extract_full_res`; reuse `preproc.frames.read_frames_chunked`)
  2. model grid + lift (inlined `_to_model_grid`, `_lift_keypoints`; pixel-center kept)
  3. pairs (inlined `_pairs`: sequential window + registry retrieval top-k with NMS)
  4. match + depth gate (`reprojection_error` both directions, max < `depth_tol`)
  5. verify → in-memory `pycolmap.CorrespondenceGraph`
  6. `_assemble_tracks` → flat arrays
- Verification (5): `ThreadPoolExecutor(num_threads)` maps `pycolmap.estimate_two_view_geometry` (uncalibrated, as the old `verify_matches`)
  over pairs; results added to the graph in pair order (deterministic); `num_threads: int = 8`
  keyword (container quota 7.65 CPUs: 8 threads 7.5×, 16 threads 4.8×); pairs < `min_matches` skipped.
  Replaces `_write_database`, `pairs.txt`, `verify_matches`, `DatabaseCache` 🟡
  - fallback if > +20 s vs today's 6.5 s verify at 294 frames: keep sqlite in one transaction,
    skip pairs < 50, `detect_watermark=False`
- `_assemble_tracks`: `_star_tracks` + `_assemble` merged; emits flat observations; star design kept 🟢
- `seed_fraction` → `seed_frames` conversion moves into `build_tracks` 🟢
- `timings` dict + log wall → one summary log line 🟢
- New keyword `retrieval: str = "dino-salad"` (registry name)
- Profile the match loop first; batch the depth gate on GPU only if it dominates

### bundle_adjustment.py (800 → ~650)
- `_filter_observations` on flat masks: `score > vis_thresh`, `reprojection_error < max_reproj_error`,
  `np.bincount(frame)` ≥ `min_inliers_per_frame`, `np.bincount(track)` ≥ 2 🟡
- `_optimize`: `xy[keep]`, `np.unique` for active frames/points; incremental windows `frame < k` 🟢
- `_BAModel.focal()` returns per-camera (K, 1) focal; replaces 3 shared-vs-per-camera branches
  (forward, photometric re-sample, K write-back) 🟢
- `forward` photometric unpack via `photometric.KEYS` 🟢
- INLINE `_lm_step` into the step loop; its tests move onto `refine` 🟢
- `__post_init__`: the 3 allowed-value checks in one loop 🟢
- `refine`: drop report resets duplicating `__init__`; `depth` required (both callers pass it) 🟢
- Log walls → 1–2 lines; drop unused `target_z`; `images`/`confidence` not Optional 🟢
- `_align_to_input_poses`: `min_spread` → literal 🟢
- New config field `track_retrieval: str = "dino-salad"` → `extract_tracks`
- `_optimize` phase blocks: 1 filter · 2 observation tensors + depth weights · 3 photometric pyramid ·
  4 solver · 5 LM/IRLS loop · 6 losses, poses, alignment, focal
- Config docstring grouped under `Tracks:` / `Filter:` / `Solve:` / `Runtime:` headers; rationale stays
  in `configs/base.yaml`
- KEEP: nearest-pixel depth lookup, `_align_to_input_poses` (not Umeyama), focal in pose column,
  `target=None` pin, ambient-grad env

## 3. Loop closure (1869 → ~1200)

- KEEP `map.py` GraphMap (user decision)
- CUT: `raw_outputs`, `Submap.is_lc_submap`, `LoopMatch.reject_reason`, `PoseGraph._frame_to_node`,
  `_node_ids`, `add_loop_edge(self_submaps=)`, carrier `frames` / zero `retrieval_vectors`,
  `base.n_loops_applied` (user decision) 🟢
- CUT ~12 guards that cannot fire (`dense_points is None`, `conf is None` ×3, carrier shape,
  `loc is None`, node-count ValueError, `0 <= g`, `estimate_scale_pairwise` shape check) 🟢
- MERGE `dedup_overlap` + `extract_extrinsics` into `_assemble_result` 🟢
- MERGE the 3 window-type branches into one `rgb01` per window; the 2 DINO-SALAD-failure fallbacks
  into one; the duplicate percentile line in `Submap.__post_init__` / `set_dense_points` 🟢
- INLINE: `_enough_frames`, `_attach_dense` (+ `partial`/`finish` plumbing), `_add_window_submap`,
  `_viz_draw_loops`, `estimate_scale_pairwise`, `add_prior_factor`, `_add_inner_chain`,
  `_conf_fallback_mask`, `_loop_chain_relatives`, `get_points_in_world_frame`, `get_points_colors` 🟢
- `LoopMatchQueue` → `sorted(...)[:max_loops]` + NMS loop (differs only on exact ties) 🟡
- `rich.Console` → `logger.info`; drop `console` params 🟢
- New config field `retrieval: str = "dino-salad"` replaces hard-coded `get("dino-salad")`
- Bug fix: `run_inference` resets `self.outputs` on the below-`submap_size` path (+ test)
- Trim `run_predictions`, `_run_lc_loop`, `LoopClosureConfig` docstrings; config grouped under
  `Windows:` / `Retrieval:` / `Graph:` / `Viz:` headers
- KEEP: `np.linalg.inv` on rigid poses (graph), L2 retrieval threshold, `find_loop_closures`

## 4. Retrieval registry

- Every site builds extractors via `BaseRetrievalExtractor.get(name)(device=...)`:
  tracks (`retrieval` kwarg ← BA `track_retrieval`), LC (`LoopClosureConfig.retrieval`),
  localizer (`CameraLocalizer(retrieval=...)`)
- CUT `PECLIPExtractor` (`"pe-clip"`, test-only) and its tests; drop `open_clip` import
- ADD `MegaLocExtractor` registered `"megaloc"`:
  - upstream gmberton/MegaLoc @ 5fe0dd69 (MIT), `torch.hub.load("gmberton/MegaLoc:<sha>",
    "get_trained_model")`; weights from HF `gberton/MegaLoc` (needs `huggingface_hub`, `safetensors`,
    both in venv)
  - preprocessing per upstream README: ImageNet normalize, resize 322 × 322
  - same `forward` contract as DinoSalad: PIL list or (N, 3, H, W) in [0, 1] → (N, 8448) L2-normalized, CPU
  - attribution once at the class
- Defaults stay `"dino-salad"` everywhere → bit-identical
- `lc_retrieval_threshold` is calibrated for DINO-SALAD L2; MegaLoc in LC needs its own sweep
  (recorded, not run here). Tracks top-k is rank-only, so any model works
- Tests: registry lookup, forward shape/norm on CPU (network-gated like DinoSalad's)

## 5. Tests, harness, docs

- ~15 test files reaching into cut/private names rewritten through public paths, flat functions;
  tolerances only on 🟡 items
- Parity harness `exh_old.py` patches `tracks_mod._pairs`: replace with `window=N, retrieval_k=0`
  (exhaustive). Stale `ancb_*.py` scripts left alone
- CLAUDE.md tree (retrieval line: DinoSalad/MegaLoc; tracks.py), `docs/` geometry page if present
- Stale `slam_loop_closure.ipynb` reference to `_lc_all_matches` recorded for tutorial-rework

## Acceptance

- Gate (real exit code, no pipe):
  `timeout 1800 env PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/
  tests/localization/ tests/test_docstring_contract.py tests/test_import_style.py
  tests/reconstructor/test_refine_stage.py -q -p no:randomly; echo EXIT=$?`
- Parity: `exh_old.py schur`, fresh cache, 294 frames: final loss ≈ 1.05396e7, extrinsics within 1e-6
  of `$P/exh_old/schur.npz`, same observation set; tracks host RAM scales with M
- One GPU job at a time; heavy runs in tmux; never write into GH010229 source/outputs

## Out of scope (follow-ups)

- Native-res photometric pyramid level; native-px verification thresholds
- Pixel-center (tracks) vs pixel-corner (localizer) convention
- MegaLoc LC threshold sweep and A/B vs DINO-SALAD
- `localizer._chunked` → `batch_iterator`; repo-wide `invert_poses` swaps
- Plumbing `retrieval_k` / `depth_tol` into config; GPU-side Schur memory
