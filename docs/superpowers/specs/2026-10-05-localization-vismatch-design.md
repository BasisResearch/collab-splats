# Localization on vismatch — cached-feature matching, matcher tracks, contract pass

**Date:** 2026-10-05
**Status:** design agreed in brainstorming 2026-10-05; plan to follow
**Branch:** `clean/localization`, worktree `.worktrees/localization`, created from `clean/final` **after** `feat/rgbd-ba-cf` lands
**Builds on:** [2026-10-03-vismatch-feature-matching-design.md](2026-10-03-vismatch-feature-matching-design.md) (pin `BasisResearch/vismatch@aa81830d`)

## Problem

- vismatch speedups landed in `LocalMatcher` (`extract(list)`, `match()`, uint8 upload, `skip_ransac`) but nothing in production calls them:
  - `CameraLocalizer.localize` routes every `LocalMatcher` to `match_images`, re-detecting both images per pair
  - index build calls `extract(rgb)` one frame at a time
- No matcher-based tracks exist; BA only has VGGSfM `predict_tracks`.
- Avoidable processing time: DINO-SALAD re-embeds all N frames on every load, serial PNG decode, O(N²) zarr appends.
- `from_feedforward` predates the `PointcloudResult` contract: duck-typed, reads private `_zarr_path`, requires `load_images=True` only for the pairwise path; its unused descriptor path maps pixels by size only (ignores the crop in `original_coords`).
- `localization` is not under the docstring contract.

## Goals

- Minimal change that puts every localization path on the vismatch fast path.
- Localization serves two consumers:
  - BA: matcher tracks (`xfeat`, `loma`) as an opt-in track source
  - query alignment: localize an external image against the scene, or against chosen scene frames (tutorial `ref_image`)
- Reuse existing package functions; delete what the updates make dead.

## Priorities and coordination

- BA branches land first; vismatch speedups second; this cleanup third.
- BA lands as `feat/rgbd-ba-cf` (@ `f196ecf4`, 58 commits on top of `clean/final` `c0bed552`, 0 behind); `clean/localization` branches from `clean/final` after it lands. State checked 2026-10-05:
  - `extractors.py` / `localizer.py` signatures identical to `clean/final`; no `match_batch` (the older `feat/rgbd-ba` `4ccf1438` copy is not carried, and stays out)
  - `retrieval.py`: `73877a40` tensor-path ImageNet normalize + `8e69804f` contract docstring on `DinoSaladExtractor.forward`; §5a depends on the normalize
  - BA: `extract_tracks(images, confidence, world_points, image_paths)` unchanged; `refine(..., image_paths=None, depth=None)` holds model-grid `images` (N, 3, H, W), `world_points`, model-res `intrinsics`; config gains `refine_focal`, `dtype`, `use_photometric`, `use_depth`, `depth_sigma` and a `__post_init__` check
  - `PointcloudResult` fields unchanged; `_unproject_frames` moves to `geometry.projection.unproject_frames`
  - reconstructor localization functions (`_localization_db_exists`, `_build_localization_db`, `store_rows`) unchanged
- `clean/dashboard-release`: public `CameraLocalizer` API keeps its signatures except the `from_feedforward` → `from_pointcloud` rename (one call site in `dashboard/pipeline.py`).
- VGGSfM tracks stay in `geometry/bundle_adjustment.py`.

## Design

### 1. `extractors.py` — xfeat + loma only

- `LocalMatcher.__init__` raises `ValueError` when vismatch `supports_batches` is False.
- Delete: `match_images`, `_probe_index_stability`, `_recover_indices`, `has_stable_indices`, `probe=`, both `_VISMATCH_*_BLOCKLIST`s, `_to_numpy` (→ `utils.torch_utils.to_numpy`).
- `MatchResult.idx_q` / `idx_db` always set (`match()` returns native indices).
- New keyword `max_num_keypoints: int = 2048` (vismatch `get_matcher`'s own default), forwarded unchanged.
- `import vismatch` moves to module top (pinned dep).

### 2. `localizer.py` — one match path

- `CameraLocalizer.from_pointcloud(result: PointcloudResult, *, zarr_path, images=None, ids=None, extractor=None, progress_callback=None, top_k=8)` replaces `from_feedforward`:
  - typed; `zarr_path` explicit, no `result._zarr_path` read
  - no `load_images=True` requirement; stale "load_world_points" message dropped
  - stale cache: `image_paths` compared exactly → rebuild (no stem / jpg-png legacy branches)
- Index build: `extract(list)` via `utils.torch_utils.batch_iterator`, chunk size a keyword default (`batch_size: int = 32`) on `from_pointcloud`.
- Device: vismatch `match()` already calls `torch.as_tensor(v, device=self.device)` on its inputs; the localizer moves cached features to the matcher device once (build and load) so that call becomes a no-op. No other device code.
- Callers keep passing a lazy image iterable: a cache hit reads zero frames (both callers rely on it today).
- `reconstructor._build_localization_db` drops `load_images=True` (only the pairwise path needed `ff.images`); it keeps deleting the group first (always-rebuild).
- `localize(query_image, query_intrinsics=None, refs=None)`:
  - `refs=None`: DINO-SALAD ranks reconstruction frames, top-K matched
  - `refs=[i, ...]`: exactly those frames (align to a chosen scene image)
  - per ref: cached features + `match()`; ref px → world_points grid (§4); `sample_world_points`; unchanged `_solve_pnp`
- Delete: `_localize_pairwise`, `_ref_images`, the descriptor-path loop's size-only rescale, image retention in `_build_pairwise_refs` (shrinks to global descriptors).
- `update_index` and the localized group persist `keypoints_normalized`; the localized group also stores per-frame `image_size` (queries vary in size). Today `load_index` rebuilds localized frames with neither, so loma cannot match them.
- Unchanged public API: `add_localized_frame`, `clear_localized_frames`, `save_index`, `load_index`, `update_index`, `image_paths`, `frame_sources`, `extrinsics`, `LocalizationResult` (incl. `ranked_ref_frames`), `read_localization_db`, `sample_world_points`, `seed_intrinsics`.
- `add_localized_frame` duplicate guard: exact id, not stem.

### 3. `tracks.py` (new) — matcher tracks for BA

Promotes the prototype that produced the measured xfeat BA runs (gh1kq60, gstar294r): `$S/match_tracks.py` with `MT_CHAIN=star`, `make_extract("xfeat", 10, "wp")`, where `$S` = session scratchpad `d91a918d…/scratchpad` (volatile `/tmp`: copy it into the worktree before the plan starts). Every `MT_*` env var becomes a keyword default with the measured value; the module-global `_state` / `set_poses` / `set_depth` monkeypatch goes away because the BA hook passes the arrays.

```python
def build_tracks(
    matcher: LocalMatcher,
    images: np.ndarray,        # (N, 3, H, W) RGB [0, 1], model grid (BA's `images`)
    world_points: np.ndarray,  # (N, H, W, 3), same grid
    extrinsics: np.ndarray,    # (N, 4, 4) w2c
    intrinsics: np.ndarray,    # (N, 3, 3) model-grid K
    *,
    window: int = 10,          # MT overlap
    retrieval_k: int = 20,     # MT_RETR_K
    retrieval_nms: int = 25,   # MT_RETR_NMS
    seed_frames: int = 30,     # MT_QFN (gh1kq60 used 60)
    min_matches: int = 50,     # MT_MIN_MATCHES
    depth_tol: float = 8.0,    # MT_DEPTH_TOL, px
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # tracks (N, P, 2), vis (N, P), pts3d (P, 3)
```

- Inputs are what `BundleAdjustment.refine` already holds (`feat/rgbd-ba-cf`: `images`, `world_points`, `extrinsics`, model-res `intrinsics`).
- Same return triple as `extract_tracks_vggsfm`, float32; `vis` 1.0 on observed, 0.0 elsewhere (prototype layout).
- Steps, prototype order; each step is an existing package / vismatch / pycolmap call:
  1. `utils.io.to_uint8_hwc(images, channels_first=True)` → `LocalMatcher.extract` per `batch_iterator` chunk (prototype: one frame at a time, 142–285 s on gh1k; this is the gate's extract lever)
  2. pairs: sequential `b - a ≤ window`, plus DINO-SALAD retrieval — `localization.retrieval.DinoSaladExtractor` on `images` in chunks, cosine top-`retrieval_k` per frame outside `|a - b| ≤ retrieval_nms`, deduped against the sequential set
  3. `LocalMatcher.match` per pair (native `idx_q` / `idx_db`)
  4. depth-consistency filter, symmetric: keypoints of `a` → `world_points[a]` at the nearest pixel → `geometry.projection.project` into `b` → px error vs the matched keypoint; same `b → a`; keep `max < depth_tol`. Replaces the prototype's `_transfer_err` (its own unproject math); `world_points` is already the unprojected depth
  5. temp-dir pycolmap `Database`: `write_camera` (PINHOLE from `intrinsics`, `use_camera_id=True`) / `write_image` / `write_keypoints` / `write_matches`; pairs txt; `pycolmap.verify_matches(db, pairs_txt)`
  6. star chaining: `pycolmap.DatabaseCache.create(db, DatabaseCacheOptions(min_num_matches=min_matches)).correspondence_graph`; `seed_frames` evenly spaced seeds (`np.linspace`), seeds absent from the graph skipped; per seed keypoint `extract_transitive_correspondences(seed, idx, 1)` → the seed plus its direct verified matches (a VGGSfM-style query); a frame holding two keypoints of one track is dropped from that track; ≥ 2 observations kept
  7. `pts3d`: `world_points` at the first observation's nearest pixel (prototype `"wp"`, VGGSfM semantics)
- Track assembly stays in pycolmap (verification, correspondence graph); the Python loop only reads `seed_frames × max_num_keypoints` star queries (prototype chain 7–23 s on gh1k). No union-find, no full transitive closure.
- Not carried from the prototype:
  - `"tri"` chain (`pycolmap.triangulate_points`): COLMAP's angle gates drop nearly every track on small window baselines; never the measured arm
  - `"graph"` chain (full transitive closure): measured and failed on chess — mismatches glued tracks into ~110 oversized components, nothing survived BA's filter after conflict removal
  - `pts_mode="tri"`, stats dict / prints (→ `logger.info` with per-phase timings), `MT_FULL_DIR` full-res extract (see open question)
- BA hook (after BA lands, ~5 lines in `bundle_adjustment.py`):
  - `BundleAdjustmentConfig.track_source: Literal["vggsfm", "xfeat", "loma"] = "vggsfm"`
  - `extract_tracks(images, confidence, world_points, image_paths)` gains `extrinsics`, `intrinsics` (refine passes its own); dispatches to `build_tracks` when not `vggsfm`; `confidence` unused there
  - `_compute_tracks_cache_key` includes `track_source`
- Open question (user): `MT_FULL_DIR` — gh1kq60 extracted on full-res 1080×1920 frames (2048 kp) and mapped keypoints to the model grid by size only. In scope (via the §4 crop-aware map and the frame store), or model grid only for now?

### 4. Pixel grid map — full-res → model grid

- Today: the live pairwise path matches against model-res `ff.images`, so ref px already sit on the world_points grid — correct. The descriptor path's size-only rescale ignores the crop, but no `LocalMatcher` reaches it.
- Needed now: refs become full-res store frames (§2), so ref px must map through the crop.
- Map: exact inverse of `PointcloudResult.__post_init__` — `px_model = (px - tl) * (W, H) / crop_wh` from `original_coords` (pixel-corner, as `rescale_intrinsics` / `shift_intrinsics`). Pixels outside the model grid are dropped.
- One private helper in `localizer.py`; `build_tracks` needs it only if the full-res question is answered yes.
- Test: cropped `original_coords` fixture; mapped px sample the world point of the matching model pixel.

### 5. Processing-time levers (profile first)

Profile on gh1k before fixing; a lever with no profile share is dropped from the plan.

| # | Today | Fix |
|---|---|---|
| a | DINO-SALAD re-embeds all N frames on every load, PIL per image | persist `global_desc (N, D)` in `local_features/<m>/reconstruction`; embed once, batched tensor path; missing array → cache rebuild (no legacy fallback) |
| b | serial `read_image` genexpr in `reconstructor._build_localization_db` and `dashboard/pipeline._build_localizer` | lazy generator of `preproc.frames.read_frames(dir, idxs, workers=8)` per 32-frame chunk; still zero reads on a hit |
| c | `update_index` resizes every single-chunk array once per frame (O(N²) IO) | concatenate new frames, one resize + write per array |
| d | single-chunk LZ4 feature arrays: one-threaded decode on load | row chunks; only if (a)'s profile shows read time |
| e | CPU features re-uploaded inside every vismatch `match()` | one upload per frame (§2); vismatch's own move becomes a no-op |
| f | `sample_world_points` per ref, full map through `grid_sample` | one call over the top-K maps; only if the localize profile shows it |

### 6. Reuse and contract pass

- `reconstructor._localization_db_exists` and `from_pointcloud`'s cache check → one localization helper.
- Frame rows: `reconstructor.store_rows` stays the one zarr-row → store-file lookup.
- `PointcloudResult.intrinsics` is typed `| None` but `__post_init__` always fills it; no None branch.
- `tqdm` inline import → top-level.
- Reused, not rewritten: `utils.torch_utils.to_numpy` / `batch_iterator`, `utils.io.to_uint8_hwc` / `open_valid`, `preproc.frames.read_frames`, `reconstructor.store_rows`, `localizer.sample_world_points`, `localization.retrieval.DinoSaladExtractor`, `geometry.projection.project`, vismatch `extract` / `match`, pycolmap `Database` / `verify_matches` / `DatabaseCache` / `CorrespondenceGraph`.
- Contract style on touched files: docstrings (summary line, bullets, `Args:`/`Returns:`), every parameter and return annotated, single-line block comments, one call per line (no nested calls), tunables as keyword defaults (`batch_size`, `window`, `max_num_keypoints`), absolute imports.
- Add `"localization"` to `PACKAGES` in `tests/test_docstring_contract.py`.
- Left alone: `seed_intrinsics` (no package equivalent), `viz.py` display rescale.

### 7. Tests

- Flat functions, `tests/localization/` mirrors the package.
- Delete: probe, `_recover_indices`, `match_images`, pairwise-path, descriptor-path stub tests.
- Add: `test_tracks.py` (synthetic posed scene: known star tracks recovered, depth filter drops a planted outlier, pts3d from world_points); parity test vs the copied prototype (`MT_CHAIN=star`) on a small fixture: same pairs, same track count; cropped pixel-map test; `refs=[i]` test; `update_index` and localized frames keep `keypoints_normalized` + `image_size`; `global_desc` round trip; non-batch model refused; cache hit reads zero frames.
- Update callers' tests: `tests/reconstructor`, `tests/dashboard` for `from_pointcloud`.

## Gates

| Gate | Pass |
|---|---|
| gh1k matcher tracks (984 frames, window 10 + retrieval 20 = 24,195 pairs, xfeat 2048 kpts) | extract ≤ 15 s, match ≤ 28 s; verify + chain time reported; tracks / observations within 5% of the prototype's gh1kq60 run (108,518 / 929,663 at `seed_frames=60`) |
| localize, LoMa, top-K 8, tutorial query | wall time reported before/after; inliers ≥ before |
| tutorial `ref_image`, `refs=[i]` | pose, ≥ 4 inliers |
| cache-hit `from_pointcloud` on gh1k | before/after reported; no DINO forward over N |
| `_build_localization_db` on gh1k | before/after reported |
| `update_index` 100 frames onto 1k | before/after reported |
| BA `track_source: vggsfm` | refine output bit-identical to pre-branch |
| suite | `tests/localization tests/dashboard tests/reconstructor tests/geometry tests/test_docstring_contract.py tests/test_import_style.py` green |

Heavy gates run in tmux on an idle GPU.

## Out of scope

- Matcher tracks as BA default; ATE A/B vs VGGSfM (later eval grid condition).
- vismatch `feat/colmap-export` and upstream PRs.
- Dashboard UI; tutorial notebooks — tutorial-rework owns them. It must switch off `FeedforwardResult`, `collab_splats.utils.visualization` and `from_feedforward`.
