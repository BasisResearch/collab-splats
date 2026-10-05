# Localization on vismatch — cached-feature matching, matcher tracks, contract pass

**Date:** 2026-10-05
**Status:** design agreed in brainstorming 2026-10-05; plan to follow
**Branch:** `clean/localization`, worktree `.worktrees/localization`, created from `clean/final` **after** the BA branches (`rgbd-ba` / `ba-speed`) land
**Builds on:** [2026-10-03-vismatch-feature-matching-design.md](2026-10-03-vismatch-feature-matching-design.md) (pin `BasisResearch/vismatch@aa81830d`)

## Problem

- vismatch speedups landed in `LocalMatcher` (`extract(list)`, `match()`, uint8 upload, `skip_ransac`) but nothing in production calls them:
  - `CameraLocalizer.localize` routes every `LocalMatcher` to `match_images`, re-detecting both images per pair
  - index build calls `extract(rgb)` one frame at a time
- No matcher-based tracks exist; BA only has VGGSfM `predict_tracks`.
- Avoidable processing time: DINO-SALAD re-embeds all N frames on every load, serial PNG decode, O(N²) zarr appends.
- `from_feedforward` predates the `PointcloudResult` contract: duck-typed, reads private `_zarr_path`, maps full-res pixels to the model grid by size only (ignores the crop in `original_coords`).
- `localization` is not under the docstring contract.

## Goals

- Minimal change that puts every localization path on the vismatch fast path.
- Localization serves two consumers:
  - BA: matcher tracks (`xfeat`, `loma`) as an opt-in track source
  - query alignment: localize an external image against the scene, or against chosen scene frames (tutorial `ref_image`)
- Reuse existing package functions; delete what the updates make dead.

## Priorities and coordination

- BA branches land first; vismatch speedups second; this cleanup third.
- Whatever BA lands in `localization/` is final; this branch adapts:
  - `rgbd-ba` uncommitted `LocalMatcher.match_batch`: deleted here (vismatch spec: equal to per-pair `match()` on GPU features)
  - `rgbd-ba` DinoSalad tensor-path ImageNet normalize: kept / moved here if not landed
- `clean/dashboard-release`: public `CameraLocalizer` API keeps its signatures except the `from_feedforward` → `from_pointcloud` rename (one call site in `dashboard/pipeline.py`).
- VGGSfM tracks stay in `geometry/bundle_adjustment.py`.

## Design

### 1. `extractors.py` — xfeat + loma only

- `LocalMatcher.__init__` raises `ValueError` when vismatch `supports_batches` is False.
- Delete: `match_images`, `_probe_index_stability`, `_recover_indices`, `has_stable_indices`, `probe=`, both `_VISMATCH_*_BLOCKLIST`s, `_to_numpy` (→ `utils.torch_utils.to_numpy`), `match_batch` if landed.
- `MatchResult.idx_q` / `idx_db` always set (`match()` returns native indices).
- New keyword `max_num_keypoints: int | None = None`, forwarded to the vismatch matcher.
- `import vismatch` moves to module top (pinned dep).

### 2. `localizer.py` — one match path

- `CameraLocalizer.from_pointcloud(result: PointcloudResult, *, zarr_path, images=None, ids=None, extractor=None, progress_callback=None, top_k=8)` replaces `from_feedforward`:
  - typed; `zarr_path` explicit, no `result._zarr_path` read
  - no `load_images=True` requirement; stale "load_world_points" message dropped
  - stale cache: `image_paths` compared exactly → rebuild (no stem / jpg-png legacy branches)
- Index build: `extract(list)` in chunks of 32 via `utils.torch_utils.batch_iterator`; features moved to the matcher device once, at build and at load.
- `localize(query_image, query_intrinsics=None, refs=None)`:
  - `refs=None`: DINO-SALAD ranks reconstruction frames, top-K matched
  - `refs=[i, ...]`: exactly those frames (align to a chosen scene image)
  - per ref: cached features + `match()`; ref px → world_points grid (§4); `sample_world_points`; unchanged `_solve_pnp`
- Delete: `_localize_pairwise`, `_ref_images`, the descriptor-path loop's size-only rescale, image retention in `_build_pairwise_refs` (shrinks to global descriptors).
- `update_index` and the localized group persist `keypoints_normalized` (loma breaks after an append today).
- Unchanged public API: `add_localized_frame`, `clear_localized_frames`, `save_index`, `load_index`, `update_index`, `image_paths`, `frame_sources`, `extrinsics`, `LocalizationResult` (incl. `ranked_ref_frames`), `read_localization_db`, `sample_world_points`, `seed_intrinsics`.
- `add_localized_frame` duplicate guard: exact id, not stem.

### 3. `tracks.py` (new) — matcher tracks for BA

```python
def build_tracks(
    matcher: LocalMatcher,
    images: np.ndarray,        # (N, H, W, 3) uint8, BA's grid
    world_points: np.ndarray,  # (N, Hw, Ww, 3)
    intrinsics: np.ndarray,    # (N, 3, 3) K of `images`
    world_intrinsics: np.ndarray,  # (N, 3, 3) K of the world_points grid
    *,
    window: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # tracks (N, P, 2), vis (N, P), pts3d (P, 3)
```

- Same return triple as `extract_tracks_vggsfm`.
- Steps:
  1. batched `extract`, features resident on device
  2. sequential-window pairs `(a, b)`, `b - a ≤ window`; `match()` per pair
  3. temp pycolmap `Database`: one PINHOLE camera per frame from `intrinsics`, keypoints, matches
  4. `pycolmap.verify_matches` (two-view geometry)
  5. `DatabaseCache` → `CorrespondenceGraph`; tracks from transitive correspondences
  6. `pts3d`: `sample_world_points` at each track's first observation, px mapped via §4; tracks without a valid sample dropped
- Tracks live in the pixel grid of `images`.
- No in-repo union-find (standing rule: tracks via pycolmap).
- BA hook (after BA lands, ~5 lines in `bundle_adjustment.py`):
  - `BundleAdjustmentConfig.track_source: Literal["vggsfm", "xfeat", "loma"] = "vggsfm"`
  - `extract_tracks` dispatches to `build_tracks` when not `vggsfm`
  - `_compute_tracks_cache_key` includes `track_source`

### 4. Pixel grid map — full-res → model grid

- Suspected bug: today's map scales by image size only; `original_coords` carries a crop box, so cropped preprocs sample the wrong world point.
- Fix: `px_model = K_model · K_full⁻¹ · [px, 1]` from `result.model_intrinsics` and `result.intrinsics` (both on the contract; encode crop and scale). Pixels outside the model grid are dropped.
- One private helper, used by `localize` and `build_tracks`.
- Plan task 1 writes the failing test (cropped `original_coords` fixture) before the fix; if it passes on today's code, the bug claim is withdrawn and only the reuse change stays.

### 5. Processing-time levers (profile first)

Profile on gh1k before fixing; a lever with no profile share is dropped from the plan.

| # | Today | Fix |
|---|---|---|
| a | DINO-SALAD re-embeds all N frames on every load, PIL per image | persist `global_desc (N, D)` in `local_features/<m>/reconstruction`; embed once, batched tensor path; missing array → cache rebuild (no legacy fallback) |
| b | serial `read_image` genexpr in `reconstructor._build_localization_db` and `dashboard/pipeline._build_localizer` | `preproc.frames.read_frames(dir, idxs, workers=8)` per 32-frame chunk |
| c | `update_index` resizes every single-chunk array once per frame (O(N²) IO) | concatenate new frames, one resize + write per array |
| d | single-chunk LZ4 feature arrays: one-threaded decode on load | row chunks; only if (a)'s profile shows read time |
| e | CPU features re-uploaded per `match()` | one upload per frame (§2) |
| f | `sample_world_points` per ref, full map through `grid_sample` | one call over the top-K maps; only if the localize profile shows it |

### 6. Reuse and contract pass

- `reconstructor._localization_db_exists` and `from_pointcloud`'s cache check → one localization helper.
- `tqdm` inline import → top-level.
- Contract style on touched files: docstrings (summary line, bullets, `Args:`/`Returns:`), every parameter and return annotated, single-line block comments, absolute imports.
- Add `"localization"` to `PACKAGES` in `tests/test_docstring_contract.py`.
- Left alone: `seed_intrinsics` (no package equivalent), `viz.py` display rescale.

### 7. Tests

- Flat functions, `tests/localization/` mirrors the package.
- Delete: probe, `_recover_indices`, `match_images`, pairwise-path, descriptor-path stub tests.
- Add: `test_tracks.py` (synthetic scene: known tracks recovered, pts3d from world_points); cropped pixel-map test; `refs=[i]` test; `update_index` keeps `keypoints_normalized`; `global_desc` round trip; non-batch model refused.
- Update callers' tests: `tests/reconstructor`, `tests/dashboard` for `from_pointcloud`.

## Gates

| Gate | Pass |
|---|---|
| gh1k matcher tracks (984 frames, window 25, xfeat 2048 kpts) | extract ≤ 15 s, match ≤ 28 s; verify + graph time reported |
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
