# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always follow:
- Use rtk tools /workspace/.claude/RTK.md
- At the start of every conversation always call /brainstorming -- use superpowers to accomplish tasks.
- Active spec/plan for in-flight work lives in docs/superpowers/specs/ and docs/superpowers/plans/.
- Architecture decisions: docs/superpowers/decisions/NNN-slug.md (sequential numbering).
- User-facing module docs live in docs/ (mirrors code tree).
- Superpowers specs/plans belong in docs/superpowers/specs/ and docs/superpowers/plans/.
- Completed work is appended to docs/superpowers/CHANGELOG.md, never written into this
  file. This file lists in-flight work only and must stay well under 40,000 characters —
  Claude Code truncates past that, and a truncated CLAUDE.md silently drops the rules
  below. A PreToolUse hook (`.claude/hooks/claude-md-guard.py`) enforces both.

## In-Flight Work

These tasks are started but not complete — do not assume their targets are done:

- **gt-eval-harness** — ground-truth ATE evaluation harness ([spec](docs/superpowers/specs/2026-05-07-gt-eval-harness-design.md) · [plan](docs/superpowers/plans/2026-05-07-gt-eval-harness.md))
- **feedforward-import-cleanup** — import hygiene pass ([spec](docs/superpowers/specs/2026-05-08-feedforward-import-cleanup-design.md) · [plan](docs/superpowers/plans/2026-05-08-feedforward-import-cleanup.md))
- **feedforward-mesh** — feedforward → TSDF mesh pipeline ([spec](docs/superpowers/specs/2026-05-14-feedforward-mesh-design.md) · [plan](docs/superpowers/plans/2026-05-14-feedforward-mesh.md))
- **docs-site** — Sphinx site setup ([spec](docs/superpowers/specs/2026-05-20-docs-site-design.md) · [plan](docs/superpowers/plans/2026-05-20-docs-site.md))
- **bae-vggt-parity** — verify BA matches upstream `zitongzhan/vggt --implementation bae` ([spec](docs/superpowers/specs/2026-05-20-bae-vggt-parity-design.md))
- **loma-matcher** — LoMa local matcher for localization ([spec](docs/superpowers/specs/2026-07-08-loma-matcher-integration-design.md) · [plan](docs/superpowers/plans/2026-07-08-loma-matcher-integration.md))
- **clean-final** — integration branch for the five cleanup efforts; all five landed (preproc, semantics, pointcloud, splats, mesh), not yet merged to trunk ([spec](docs/superpowers/specs/2026-09-06-clean-final-integration-design.md) · [plan](docs/superpowers/plans/2026-09-06-clean-final-integration.md))
- **sky-mask** — ONNX sky segmentation as a `BaseSegmentation` backend, consumed by the mesh stage behind `mesh.mask_sky`; A/B against the meshing quality is the deliverable ([spec](docs/superpowers/specs/2026-09-07-sky-segmentation-design.md) · [plan](docs/superpowers/plans/2026-09-07-sky-segmentation.md))
- **vismatch-fork** — fork `BasisResearch/vismatch` at `/workspace/vismatch`: batch + COLMAP-export upstream PRs, `basis` integration branch for split/cache, collab-splats pins a basis SHA ([spec](docs/superpowers/specs/2026-09-25-vismatch-fork-design.md))
- **tutorial-rework** — rebuild the tutorial as nine self-contained notebooks on the clean/final API: no shared `data/outputs/` cache, each page builds its inputs into its own tempdir ([spec](docs/superpowers/specs/2026-09-09-tutorial-rework-design.md))
- **consistency** — dedup audit; phase 1 + 1b (convention bugs) and phase 2 (utils/io.py) squashed onto `clean/final` from `clean/consistency`; phase 3 (utils/colmap.py, dedup) not started ([spec](docs/superpowers/specs/2026-09-26-consistency-design.md) · [plan](docs/superpowers/plans/2026-09-26-consistency-phase1.md) · [phase 2](docs/superpowers/plans/2026-09-26-consistency-phase2.md))

## Recently Completed

Full entries live in [docs/superpowers/CHANGELOG.md](docs/superpowers/CHANGELOG.md) —
read it before assuming any subsystem below is unchanged. Five newest:

- **pointcloud-release** (2026-09-29)
- **mesh-release** (2026-09-29)
- **splats-release** (2026-09-28)
- **geometry-round3** (2026-09-27)
- **setup-prereq-gating** (2026-09-27)

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

Core pipeline: video/images → pointcloud (feedforward: VGGT-X / VGGT-Omega / MapAnything / LoGeR, or sfm: InstantSfM / COLMAP / hloc + VDA depth) → optional BA / optional LC (both feedforward only — sfm refuses them) → `pointcloud.zarr` → mesh/features/splats.

```
collab_splats/
  pointcloud/              # main reconstruction pipeline
    base.py                # BasePointcloudCreator, PointcloudResult (both Ks; COLMAP export-only via to_colmap)
    feedforward/           # BaseFeedforwardCreator (template method in _reconstruct), VGGTX, VGGTOmega, MapAnything, LoGeR creators
    sfm/                   # InstantSfMCreator (global SfM via upstream python API)
                           #   + ColmapCreator / HlocCreator (incremental; SFM_CREATORS dispatch; sift_db.py shared SIFT)
    depth.py               # estimate_depth (VDA metric depth per keyframe) + align_depth (COLMAP model
                           #   + VDA depth -> PointcloudResult at COLMAP scale)
    utils.py               # clean_pointcloud (result -> result), outlier_mask, confidence_mask, subsample_points,
                           #   cross_frame_attention_ratio
  geometry/                # pose/geometry backend: loop closure + bundle adjustment
    transforms.py          # extrinsics_to_homogeneous, invert_poses, OPENGL_TO_OPENCV, project_to_so3,
                           #   decompose_camera, intrinsics_4x4, rescale_intrinsics, shift_intrinsics
    projection.py          # unproject/project, depth_residual, depth_agreement, multiview_depth_confidence
    bundle_adjustment.py   # Levenberg-Marquardt BA: array-in refine, check_model_resolution
    metrics.py             # compute_reconstruction_quality -> reconstruction_quality_report.json
    loop_closure/          # submap pose graph (SL4/SE3), DINO-SALAD retrieval gate, LoopClosure wrapper
  localization/            # camera localization: query image → pose in known reconstruction
    retrieval.py           # Stage 1: BaseRetrievalExtractor, DinoSalad/PECLIP (also used by loop closure)
    extractors.py          # Stage 2: LocalMatcher over the vismatch model zoo (FEATURE_MATCH_MODELS fast paths)
    localizer.py           # Stage 3: CameraLocalizer, zarr feature cache
    viz.py                 # plot_correspondences
  semantics/               # 2D feature extraction
    features/              # BaseFeatureExtractor + RegistryMixin (base.py); registered
                           #   dinov2, maskclip, talk2dino — all ViT-width (384-1024D)
    lifting.py             # lift_features: per-pixel features -> points
    compression.py         # FeatureAutoencoder: per-point encode/decode + recon_cosine
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3,
                           #   skywater (sky masks for mesh.mask_sky)
    utils.py               # compute_semantic_contrast, interpolate_to_patch_size;
                           #   re-exports collab_splats.utils.torch_utils helpers
  preproc/                 # video preprocessing: measure (qa) then select (sampling)
    video.py               # PyAV decode: get_video_info, iter_frames, extract_frame
    qa.py                  # report-only capture quality: compute_video_quality, load_video_quality
    sampling.py            # sample_fps | sample_uniform | sample_optical_flow from filter_frame_quality's
                           #   eligible pool; fps rescue may keep an ineligible frame
    frames.py              # images/frame_NNNNNN.png + frames.json: the COLMAP-style keyframe store
    undistort.py           # calibrate_camera (pycolmap) + undistort_frames (pycolmap framing, cv2 pixels)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
  mesh/                    # TSDF meshing from arrays: tsdf, clean, texture, features
  splats/                  # gsplat training: trainer, checkpoint, gaussian, scaffold, losses, rendering, cameras, utils
  wrapper/                 # stage orchestration: Reconstructor (config-driven pipeline), batch drivers
  remote/                  # rclone/GCS: SceneSource over environments-curated + environments-processed
  dashboard/               # interactive video/scene browser (reads the FLAT dashboard layout only)
  utils/
    image.py               # open/resize_image, upsample_depths (guided), fill_missing_pixels
    colmap.py              # read/write_colmap_reconstruction
    io.py                  # to_json_safe + write_json: atomic, NaN -> null, numpy -> python
    torch_utils.py         # RegistryMixin, pytorch_gc, infer_batch_size, batch_iterator, get_device
evals/
  eval.py                  # GT grid runner: datasets x config-override conditions -> Reconstructor
  gt_metrics.py            # ATE / RPE (evo), pairwise AUC, GT depth error
  datasets.py              # dataset loaders (7-Scenes, TUM, CO3Dv2)
  configs/                 # grid YAMLs (base block + conditions)
  results/                 # gitignored
```

## Code Style

- **Imports at top:** all imports at the top of the file — no inline imports inside functions or methods (scripts and notebooks). Exception: optional heavy deps that would break the module on missing install may be imported inside the function that needs them, with a clear `ImportError` message.
- **Import style:** absolute `collab_splats.` imports only, never relative (`from .x`, `from ..x`). Four groups, blank line between: stdlib; general third-party (numpy, torch, cv2, ...); model/method upstreams (mapanything, vggt, instantsfm, gsplat, ... — `known_models` in pyproject `[tool.isort]`, add new ones there); our own code (`collab_splats`, `tests`, `evals`). `isort` applies it; `tests/test_import_style.py` enforces it.
- **Inline block comments:** each logical block of code gets a short comment explaining what it does. Comment at block level, not every line. Examples: `# Sort images by filename; reject non-image extensions`, `# Populate BA fields: subsampled world-point grid for track extraction`. Existing comments that meet this standard are kept; missing block comments are added.
- **Comment runs are a header then bullets:** any `#` run of three lines or more states the problem in 5-10 words on its own line, then gives the detail as `- ` bullets — never a wrapped paragraph. Bullets are fragments, not sentences.

  ```python
  # splatfacto's non-default DefaultStrategy args
  # - upstream: nerfstudio-project/nerfstudio @ 50e0e3c, splatfacto.py:264-280
  # - absgrad=False: 2dgs backward writes .absgrad on means2d, not gradient_2dgs
  # - grow_grad2d 2e-4: measured-good non-absgrad threshold
  ```
- **Section dividers:** use `########`-style dividers to separate major sections in long files (constants, helpers, classes, etc.). Keeps files scannable without opening a doc.
- **Docstrings follow the same shape.** Every module, public class and public function gets one. The `"""` open and close on their own lines — the summary starts on the line after the opening quotes, never on the same line, is one line of at most 100 chars, and does not restate the name. A blank line, then `- ` bullets, then the named sections. Never a prose paragraph.
- **Types live in the signature, meanings in the docstring.** Every parameter and return is annotated; `Args:` entries carry the name and what it means, never a duplicated type. A public function with parameters documents every one under `Args:`; one that returns something documents it under `Returns:`, or `Yields:` if it is a generator. Private (`_`-prefixed) defs keep the summary and optional bullets only — no `Args:`/`Returns:` block; a shape or unit fact moves to a bullet.

  ```python
  def sample_optical_flow(frames: np.ndarray, *, max_frames: int = 100) -> np.ndarray:
      """
      Keyframes chosen by cumulative optical-flow magnitude.

      - one frame per fixed flow budget, so slow pans yield fewer frames
      - budget is derived from max_frames, not a fixed stride

      Args:
          frames: decoded frames, (N, H, W, 3) uint8.
          max_frames: ceiling on the returned count.

      Returns:
          Indices into `frames`, ascending.
      """
  ```

  `tests/test_docstring_contract.py` enforces all of the above — docstrings, annotations and comment runs — for `preproc`, `semantics`, `pointcloud`, `geometry` and `mesh`. Add a package to its `PACKAGES` tuple as it is cleaned up.
- **Blank lines between blocks:** a run of statements that does one thing is separated from the next run, and every block comment gets a blank line above it. Walls of undifferentiated code are not human-readable.
- **Don't over-complicate:** prefer the simplest implementation that solves the problem. No premature abstractions, no dead branches for hypothetical future use, no wrapper layers that add no value. If a param is always default, ask whether it should exist.
- `logging` not `print()` — use `logger.debug()` / `logger.info()` throughout module code
- `RegistryMixin` for registry pattern (from `utils/torch_utils.py`)
- Template-method pattern for abstract pipelines (see `BaseFeedforwardCreator`)
- Typed `@dataclass` for pipeline outputs (`PointcloudResult`: `intrinsics` full-res, `model_intrinsics` model grid)
- Hard imports — no stub backends; let missing deps raise `ImportError` at import time

## Testing

- Flat test functions — no class-based unless shared fixture state requires it
- `tests/` mirrors `collab_splats/` structure
- Run: `/opt/venv/reconstruction/bin/python -m pytest tests/`

## Evaluation

- `python -m evals.eval --config evals/configs/<grid>.yaml` = compute (CLI/tmux only); notebooks in `docs/` = visualization only
- Results: `evals/results/` (gitignored); windowing for >100-frame sequences is `pointcloud.loop_closure: {submap_size: N, lc_retrieval_threshold: 0.0}` in the grid's `base`

## Commit Conventions

Conventional commits with scope: `feat(pointcloud):`, `fix(ba):`, `refactor(semantics):`, `docs(decisions):`

## Development Commands

```bash
black . && isort .                                                  # format
/opt/venv/reconstruction/bin/python -m pytest tests/               # test
/opt/venv/reconstruction/bin/python -m evals.eval --config evals/configs/7scenes.yaml --dry_run   # eval
```
