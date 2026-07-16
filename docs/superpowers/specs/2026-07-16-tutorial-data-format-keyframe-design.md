# Tutorial data-format alignment + keyframe tutorial rework — Design

**Date:** 2026-07-16
**Status:** Approved (brainstorming session)
**Scope:** `tutorial_config.py` + `01_preprocessing/keyframe_extraction.ipynb` + supporting `preproc` extension. Remaining notebooks deferred to a follow-up sweep (requirements recorded below).

## Context

Reconstruction outputs moved to the canonical layout `BASE/<session-date>/<video_stem>/` (e.g. `/workspace/outputs/2024_02_06/C0043/`) containing `C0043.MP4`, `frames/` (`%05d.jpg`, pipeline-written), `feedforward.zarr`, `mesh/mesh_tsdf.ply`, `semantics/`, `run_config.yaml` (see `docs/superpowers/plans/2026-07-14-localization-dashboard.md`). The tutorial config and notebooks predate this: stale `images/` path, wrong `run_config.yaml` key for video inference, and the keyframe tutorial writes scratch output into what is now the canonical data directory. Additionally, the keyframe tutorial describes the blur/exposure quality gate only in prose — it never shows example rejected frames, and the preproc API cannot distinguish gate-rejected frames from low-motion frames.

## Decisions

1. **Scope:** config + tutorial 01 now; other 14 notebooks in follow-up sweep passes.
2. **Preproc extension approved** — surface per-frame quality data publicly rather than duplicating gate logic in notebook cells.
3. **`IMAGES` renamed to `FRAMES`** (`OUTPUT_DIR / "frames"`), matching on-disk name. Notebooks updated as each is touched.
4. **Splats/mesh notebooks (03/06)** currently overriding `BASE_DIR` to `/workspace/fieldwork-data/` will migrate to `2024_02_06/C0043` — during the sweep, not this pass.
5. **Tutorial scratch output lives outside the data dir**: `TUTORIAL_CACHE = BASE_DIR / "tutorial_cache" / DATASET`. Canonical scene dir stays pristine (it gets rclone-pushed by the dashboard).
6. **VGGT-Omega is the primary method showcased across tutorials** (sweep requirement, see Deferred).

## Design

### 1. `docs/source/tutorials/tutorial_config.py`

- `IMAGES` → `FRAMES = OUTPUT_DIR / "frames"` (read-only pipeline output from the notebooks' perspective).
- Add `TUTORIAL_CACHE = BASE_DIR / "tutorial_cache" / DATASET` for all notebook scratch output.
- `_infer_video_path`: also accept the `video_ref` key (actual key written by `RunConfig.to_yaml`, value is an rclone-relative path like `reconstruction/2024_02_06/C0043/C0043.MP4`) — resolve its basename against `OUTPUT_DIR`. Fallback: glob `*.MP4` / `*.mp4` directly in `OUTPUT_DIR`.
- Drop dead commented `DATASET` options (old flat scene names, invalid under `<date>/<stem>` layout).
- `CACHE_DIR` alias kept pointing at `OUTPUT_DIR` (other notebooks still read canonical artifacts through it) — the sweep decides its final fate.

### 2. Preproc extension — `collab_splats/preproc/sampling.py`

- New public `check_frame_quality(gray, blur_threshold=_DEFAULT_BLUR_THRESHOLD, blur_score=None) -> tuple[bool, dict]`. Metrics dict: `blur_score`, `exposure_mean`, `exposure_std`, `reject_reason` (`None | "blur" | "exposure"`). Replaces private `_check_frame_quality`; `_iter_scored_frames` threads the metrics through its yield.
- `score_frames` records gain `exposure_mean`, `exposure_std`, `reject_reason`. Today gate-rejected frames are indistinguishable from low-motion frames (both `selected=False`, empty components); `reject_reason` fixes that.
- Export `check_frame_quality` from `collab_splats/preproc/__init__.py` `__all__`.

### 3. Viz addition — `collab_splats/preproc/viz.py`

- `plot_quality_examples(video_path, frame_scores, n_examples=4)` — grid with one row per category: accepted (sharp, well-exposed), blur-rejected, exposure-rejected. Pulls pixels via `load_frames`, annotates each frame with blur score / exposure mean. Rows with no matching frames are skipped with a note. Matches existing viz.py style; stays out of `__init__` re-exports (matplotlib isolation).

### 4. Notebook rework — `01_preprocessing/keyframe_extraction.ipynb`

Path changes:
- `IMAGES` → `FRAMES` treated as **read-only** pipeline output; no `FRAMES.mkdir`.
- All writes (`frame_scores.json`, demo-extracted keyframes) go to `TUTORIAL_CACHE`.

New content:
- New section "Quality gate: blur & exposure": run `score_frames` on C0043 → blur-score histogram with threshold line → `plot_quality_examples` showing real accepted / blur-rejected / exposure-rejected frames from the video, grouped via `reject_reason`.

Removals / fixes (stale-code audit):
- **§4 duplicate extraction cell removed** — the second extract cell defaults to `fps_indices` inside the OF section with a confusing "swap fps_indices → of_indices" comment, and its `if not any(...glob("*.jpg"))` guard silently skips when pipeline frames exist. Keep one clear OF extraction into `TUTORIAL_CACHE`.
- **§7 method-name drift fixed** — preproc uses `"uniform"/"optical_flow"`; drop the `SplatterConfig frame_selection="fps"` footnote and reference `RunConfig.sampling_method` (`"balanced" | "optical_flow"`, `collab_splats/dashboard/config.py:20`) as the pipeline-facing knob.
- Verify `load_frames` "(applies rotation automatically)" markdown claim against implementation; correct if false.

### 5. Tests

- `tests/preproc/test_sampling.py`: `check_frame_quality` on synthetic frames (sharp/well-exposed → accepted; blurry → `reject_reason="blur"`; dark/low-contrast → `reject_reason="exposure"`); `score_frames` records carry the new fields.
- `tests/preproc/test_viz.py`: `plot_quality_examples` smoke test (synthetic records + tiny video fixture already used by sampling tests, or monkeypatched `load_frames`).
- Flat test functions, per repo convention.

## Deferred to notebook sweep (follow-up spec/plan)

- Remaining 14 notebooks aligned to `FRAMES` / `TUTORIAL_CACHE` / canonical layout.
- **VGGT-Omega primary across tutorials**: `02_pointcloud/feedforward_methods` showcases `VGGTOmegaCreator` first; `bundle_adjustment` and `slam_loop_closure` switch primary backbone to omega (LC calibration: layer 13, threshold 1.55 — see `docs` LC calibration notes); other creators presented as alternatives.
- 03_splats / 06_mesh `fieldwork-data` overrides migrated to `2024_02_06/C0043` (verify splat training data availability first).
- `evals/ground_truth_evals.ipynb` hardcoded results path.
- Final disposition of `CACHE_DIR` alias.

## Verification

1. `/opt/venv/reconstruction/bin/python -m pytest tests/preproc/` — new + existing tests green.
2. Execute `keyframe_extraction.ipynb` top-to-bottom against `/workspace/outputs/2024_02_06/C0043` (papermill or jupyter execute): all cells run, quality-gate section renders example rejected frames, no writes land inside `OUTPUT_DIR`.
3. `ls /workspace/outputs/2024_02_06/C0043` before/after notebook run — directory contents unchanged.
4. `black . && isort .` clean on touched files.
