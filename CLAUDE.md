# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always follow:
- Use rtk tools /workspace/.claude/RTK.md
- At the start of every conversation always call /brainstorming -- use superpowers to accomplish tasks.
- Active spec/plan for in-flight work lives in docs/superpowers/specs/ and docs/superpowers/plans/.
- Architecture decisions: docs/superpowers/decisions/NNN-slug.md (sequential numbering).
- User-facing module docs live in docs/ (mirrors code tree).
- Superpowers specs/plans belong in docs/superpowers/specs/ and docs/superpowers/plans/.

## In-Flight Work

These tasks are started but not complete — do not assume their targets are done:

- **gt-eval-harness** — ground-truth ATE evaluation harness ([spec](docs/superpowers/specs/2026-05-07-gt-eval-harness-design.md) · [plan](docs/superpowers/plans/2026-05-07-gt-eval-harness.md))
- **feedforward-import-cleanup** — import hygiene pass ([spec](docs/superpowers/specs/2026-05-08-feedforward-import-cleanup-design.md) · [plan](docs/superpowers/plans/2026-05-08-feedforward-import-cleanup.md))
- **feedforward-mesh** — feedforward → TSDF mesh pipeline ([spec](docs/superpowers/specs/2026-05-14-feedforward-mesh-design.md) · [plan](docs/superpowers/plans/2026-05-14-feedforward-mesh.md))
- **docs-site** — Sphinx site setup ([spec](docs/superpowers/specs/2026-05-20-docs-site-design.md) · [plan](docs/superpowers/plans/2026-05-20-docs-site.md))
- **bae-vggt-parity** — verify BA matches upstream `zitongzhan/vggt --implementation bae` ([spec](docs/superpowers/specs/2026-05-20-bae-vggt-parity-design.md))
- **loma-matcher** — LoMa local matcher for localization ([spec](docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md) · [plan](docs/superpowers/plans/2026-07-08-loma-matcher-integration.md))

Recently completed (2026-08-13): **multiview-confidence-parity** — one geometric cross-view depth-consistency filter shared by every feedforward backbone. `compute_multiview_depth_confidence` (pointcloud/feedforward/base.py) now returns a `MultiviewConfidence` dataclass (`ratio`, `inlier_count`, `valid_count`, `judged`); `multiview_mask(mv, valid_depth, min_views=K)` is the only thresholding path. **`mv_conf_threshold` is gone from all four creators** — replaced by `min_views`, an inlier *count*, because the ratio is quantized with a huge atom at 1.0 and any percentile threshold on it collapses. `min_views=1` is provably identical to the old `threshold=0.0`, so MapAnything's shipping output is unchanged. Upstream *substitutes* mv for the learned confidence; **we AND them** (strictly more conservative). Sampling is nearest, not bilinear (bilinear across a depth discontinuity fabricates depth on no surface); occluded views leave the denominator while free-space violations stay as outliers; views with no overlapping partner keep all their pixels. New guard: the principal point must lie strictly inside the depth grid, which catches the model-res-depth-vs-original-res-K class that caused the 2026-08-11 mesh regression. `mv_ratio`/`mv_inlier_count`/`mv_valid_count` persist to `feedforward.zarr` when computed and are **absent, never zeros, when not** — **existing zarr stores have no `mv_*` arrays and are not backfilled**. Config surface is ONE boolean, `pointcloud.use_multiview_confidence`; tolerances stay creator fields. **Measured on 7-Scenes chess/seq-01 (60 frames, Step D of the report): mv never beat the learned confidence percentile at comparable retention, so the default stays `false` on base.yaml and on every non-metric creator.** Best case (omega, rel=0.01 K=16) cut the >10%-error pixel fraction 27% for 10% of pixels — real, ~2.6× more pixel-efficient than the learned filter, but small; the shipped rel=0.05/K=1 was *inert* (99.9% retention), hence recalibrated to **rel=0.01, K=2** on vggtx/omega/loger. MapAnything left at rel=0.02/abs=0.02/K=1 — it gains least (out10 −4% even at K=16) and parity is worth more. **`min_views` is an absolute count: K > N−1 would empty every judged view, so `multiview_mask` clamps per pixel (`required = min(min_views, valid_count)`) — an unreachable K degrades to "every partner you have must agree", never to "delete". K=1 is unchanged by the clamp, so MapAnything stays byte-identical; the corollary is that the K≥8 sweep rows are upper bounds, since on a short scene the clamp weakens them to "all agree".** VGGT-SPARK verified to inherit the path in a fresh subprocess; not wired into `creator_map`. Owed: sweep on a second scene/dataset, and downstream (cloud/mesh/ATE) effect is unmeasured ([spec](docs/superpowers/specs/2026-08-13-multiview-confidence-parity-design.md) · [report](docs/superpowers/specs/2026-08-13-multiview-confidence-measured-report.md) · [plan](docs/superpowers/plans/2026-08-13-multiview-confidence-parity.md)).

Recently completed (2026-08-13): **fps-frame-sampling** — `frame_selection` gained an `fps` method that samples at a constant wall-clock rate (`_sample_fps` + `_fps_targets`), joining `uniform`/`optical_flow` as a third sibling over a shared `_sample_positions` body (single ffmpeg `select=` pass — the 14fdc42 fast path is untouched). `max_frames` now has ONE meaning everywhere — the frame budget: the target COUNT for `uniform`, a ceiling for `fps`/`optical_flow`. `min_frames` is an `fps`-only floor; outside `[min_frames, max_frames]` fps targets are **re-spread over the whole video, never truncated** (the old `targets[:max_frames]` dropped the tail). `frame_proportion` **deleted** — it controlled no physical quantity. Passing another method's knob raises `ValueError`. `fps` joined `FrameStore._STALENESS_KEYS`, else changing the rate at fixed `max_frames` silently reuses the old `frames.zarr`. base.yaml defaults: `frame_selection: fps`, `fps: 1.0`, `min_frames: null`, `max_frames: 300` — verified on `data/tutorial/` (2388 frames @23.98fps → 100 frames spanning 0–2373). **Owed: LC calibrations and the 4-backbone ATE numbers were measured under `uniform` spacing; `loop_closure.yaml` now uses `fps: 4.0` + `min_frames: 300` and the numbers need re-verification.** `splatter.py` keeps its own unrelated `frame_proportion`/`Literal["fps", ...]` on the nerfstudio path — deliberately untouched, names now collide ([spec](docs/superpowers/specs/2026-08-13-fps-frame-sampling-design.md) · [plan](docs/superpowers/plans/2026-08-13-fps-frame-sampling.md)).

Recently completed (2026-08-11): **mesh-tsdf-adapter-convergence** — one TSDF adapter. `Reconstructor._run_tsdf_mesh` fused model-resolution depth from `feedforward.zarr` against original-resolution COLMAP intrinsics (`build_colmap` rescales the camera) and divided already-`[0,1]` RGB by 255 — measured 5.06M → 75k vertices and black vertex colours on `data/outputs/`. Both lines dated to 6bc9c81 and were invisible while `mesh.enabled: false`. `_feedforward_to_tsdf_inputs` (`mesh/utils.py`) now reads `result.depth`/`result.images` directly (world_points projection and PIL re-read/crop/`/255` deleted as measured-redundant) and is the single path for both `Reconstructor` and the dashboard. COLMAP stays the pose authority; guards on frame-count mismatch and on `[0, 255]` RGB. Post-fix baseline on `data/outputs/`: 5,060,214 verts at `depth_trunc=20`, 711,079 at the shipping `depth_trunc: 2.0` — **`depth_trunc` is now the dominant limit on mesh extent (86% of vertices), and depth is non-metric so `2.0` is not "2 metres"**. **Every `mesh.ply` on disk — local and under `environments-processed/` — was fused with the wrong K and needs `mesh(overwrite=True)`** ([spec](docs/superpowers/specs/2026-08-11-mesh-tsdf-adapter-convergence-design.md) · [plan](docs/superpowers/plans/2026-08-11-mesh-tsdf-adapter-convergence.md)).

Recently completed (2026-08-11): **stage-rerun-from-processed** — `--stages` naming only leaf stages (`mesh`, `semantics`, `localize` — derived from `Reconstructor._STAGE_DEPS`, exported as `LEAF_STAGES`) pulls a scene back out of `environments-processed` instead of rebuilding from curated video; anything including `preproc`/`pointcloud` rebuilds, so staleness is unreachable rather than handled. New `collab_splats/remote/rerun.py` (`prepare_scene`, `discover_scenes`); `Reconstructor` gained `_resolve_result()` (loads COLMAP off disk when `self.pointcloud is None`), leaf-stage markers in `_stage_output_exists`, and a refusal when a *named* stage's output already exists without `overwrite`. Nothing deletes a remote object — push stays `rclone copy`. Contract: `configs/README.md` → "Re-running one stage against a processed scene". **Owed: one live GCS run watched by a human** ([spec](docs/superpowers/specs/2026-08-11-stage-rerun-from-processed-design.md) · [plan](docs/superpowers/plans/2026-08-11-stage-rerun-from-processed.md)).

Recently completed (2026-07-30): **gcloud-remote-pipeline** — 12-task GCS remote processing pass: `environments-curated` → reconstruct → `environments-processed/<scene>/` with an `rclone check --one-way` verified push and post-push local cleanup (nothing is deleted until the push verifies); `sparse_pc.ply` now written binary; TSDF output unified on `mesh/mesh.ply` (no legacy `mesh_tsdf.ply` fallback — old scenes need `mesh(overwrite=True)` once); semantics converged on `features.zarr` + `autoencoder.pt` under `<backend>/semantics/<extractor>/`; `SessionSource` → `collab_splats/remote/SceneSource`; new driver `docs/examples/run_pipeline_remote.py` ([spec](docs/superpowers/specs/2026-07-29-gcloud-remote-pipeline-design.md) · [plan](docs/superpowers/plans/2026-07-29-gcloud-remote-pipeline.md)). Processed-scene output contract + known limitations: `configs/README.md`.

Recently completed (2026-07-17): **dashboard-ux-feedback** — 12-task responsiveness/visibility pass: global busy lock, step-level op-log timings, debounced frame preview, all blocking work off the IOLoop, fast-bind server (page up in ~3s, import progress streams to the page) ([spec](docs/superpowers/specs/2026-07-16-dashboard-ux-feedback-design.md) · [plan](docs/superpowers/plans/2026-07-16-dashboard-ux-feedback.md)). **Before committing any dashboard change, run the smoke gate:** `python -m collab_splats.dashboard --smoke` (must print SMOKE PASS). Manual browser checklist (plan Task 12 step 3) still owed. Browsing via IP needs `--websocket-origin`.

Recently completed (2026-07-16): **dashboard-loadtime** — 16-task perf pass (F1-F8, F10-F16; F9 zarr re-chunk deferred to a future decision doc) ([spec](docs/superpowers/specs/2026-07-16-dashboard-loadtime-design.md) · [plan](docs/superpowers/plans/2026-07-16-dashboard-loadtime.md)). Manual browser smoke checklist still owed.

Known test failures: `docs/known-test-failures.md`

## Installation and Setup

```bash
bash setup.sh                    # full install (uv sync — all deps incl. VGGT-X + MapAnything)
```

## Development Environment

- **Python env:** `python` in base shell may be py3.13 (wrong for this project). Always use `/opt/venv/reconstruction/bin/python` (py3.11), or activate the venv: `source /opt/venv/reconstruction/bin/activate`.
- **Memory:** container cgroup cap 46.6 GB. Heavy inference/eval → run in tmux, not notebooks.
- **Side shells:** don't run parallel processes during heavy eval runs (OOM risk).

## Architecture Overview

Core pipeline: video/images → pointcloud (VGGT-X or MapAnything) → optional BA → optional LC → mesh/features.

```
collab_splats/
  pointcloud/              # main reconstruction pipeline
    base.py                # BasePointcloudCreator, PointcloudResult
    feedforward/           # BaseFeedforwardCreator (5-step template method), VGGTXCreator, MapAnythingCreator, FeedforwardResult
    utils.py               # lift_features, reproject_pixels, colmap_reconstruction_to_result
    export.py              # write_pointcloud_ply: binary sparse_pc.ply writer
  geometry/                # pose/geometry backend: loop closure + bundle adjustment
    transforms.py          # extrinsics_to_homogeneous, invert_poses, OPENGL_TO_OPENCV (ex utils/geometry.py)
    bundle_adjustment.py   # Levenberg-Marquardt BA
    global_alignment.py    # VGGT-X native feature-match + joint BA (parked; not wired into any creator)
    loop_closure/          # submap pose graph (SL4/SE3), DINO-SALAD retrieval gate, LoopClosure wrapper
  localization/            # camera localization: query image → pose in known reconstruction
    retrieval.py           # Stage 1: BaseRetrievalExtractor, DinoSalad/PECLIP (also used by loop closure)
    extractors.py          # Stage 2: BaseLocalExtractor, Disk/XFeat/Loma/LomaG local matchers
    localizer.py           # Stage 3: CameraLocalizer, zarr feature cache
    viz.py                 # plot_correspondences
  semantics/               # 2D feature extraction
    features/              # BaseFeatureExtractor + RegistryMixin (base.py); registered
                           #   dinov2, maskclip, talk2dino — all ViT-width (384-1024D)
    compression.py         # FeatureAutoencoder: per-point encode/decode + recon_cosine
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3
  preproc/                 # video preprocessing: frame sampling + quality gate
    sampling.py            # ffmpeg-only decode, blur/exposure gate, sample_frames (uniform | optical_flow)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
  mesh/                    # TSDF + Poisson meshing (base, poisson, tsdf, utils)
  nerfstudio/              # nerfstudio method configs, models, datamanagers
  wrapper/                 # stage orchestration: Reconstructor (config-driven pipeline), batch drivers, Splatter
  remote/                  # rclone/GCS: SceneSource over environments-curated + environments-processed
  dashboard/               # interactive video/scene browser (reads the FLAT dashboard layout only)
  utils/
    torch_utils.py         # RegistryMixin, pytorch_gc, infer_batch_size, batch_iterator, get_device
evals/
  scripts/eval.py          # compute script — CLI/tmux only, never run in notebook
  datasets.py              # dataset loaders (7-Scenes, CO3Dv2)
  results/                 # gitignored
```

## Code Style

- **Imports at top:** all imports at the top of the file — no inline imports inside functions or methods (scripts and notebooks). Exception: optional heavy deps that would break the module on missing install may be imported inside the function that needs them, with a clear `ImportError` message.
- **Inline block comments:** each logical block of code gets a short comment explaining what it does. Comment at block level, not every line. Examples: `# Sort images by filename; reject non-image extensions`, `# Populate BA fields: subsampled world-point grid for track extraction`. Existing comments that meet this standard are kept; missing block comments are added.
- **Section dividers:** use `########`-style dividers to separate major sections in long files (constants, helpers, classes, etc.). Keeps files scannable without opening a doc.
- **Docstrings:** every public function and class gets a one-line summary docstring. Multi-line only when Args/Returns genuinely need it. No restating the function name. No padding.
- **Don't over-complicate:** prefer the simplest implementation that solves the problem. No premature abstractions, no dead branches for hypothetical future use, no wrapper layers that add no value. If a param is always default, ask whether it should exist.
- `logging` not `print()` — use `logger.debug()` / `logger.info()` throughout module code
- `RegistryMixin` for registry pattern (from `utils/torch_utils.py`)
- Template-method pattern for abstract pipelines (see `BaseFeedforwardCreator`)
- Typed `@dataclass` for pipeline outputs (`FeedforwardResult`, `PointcloudResult`)
- Hard imports — no stub backends; let missing deps raise `ImportError` at import time

## Testing

- Flat test functions — no class-based unless shared fixture state requires it
- `tests/` mirrors `collab_splats/` structure
- Run: `/opt/venv/reconstruction/bin/python -m pytest tests/`

## Evaluation

- `evals/scripts/eval.py` = compute (CLI/tmux only); notebooks in `docs/` = visualization only
- Results: `evals/results/` (gitignored); pass `--submap_size 50` for >100-frame sequences

## Commit Conventions

Conventional commits with scope: `feat(pointcloud):`, `fix(ba):`, `refactor(semantics):`, `docs(decisions):`

## Development Commands

```bash
black . && isort .                                                  # format
/opt/venv/reconstruction/bin/python -m pytest tests/               # test
/opt/venv/reconstruction/bin/python evals/scripts/eval.py --help   # eval
```
