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
- **scaffold-gs** — Scaffold-GS anchors on gsplat ([spec](docs/superpowers/specs/2026-08-27-scaffold-gs-gsplat-design.md) · [plan](docs/superpowers/plans/2026-08-27-scaffold-gs-gsplat.md))
- **clean-final** — integration branch for the five cleanup efforts; preproc and semantics landed, pointcloud/splats/mesh pending ([spec](docs/superpowers/specs/2026-09-06-clean-final-integration-design.md) · [plan](docs/superpowers/plans/2026-09-06-clean-final-integration.md))

## Recently Completed

Full entries live in [docs/superpowers/CHANGELOG.md](docs/superpowers/CHANGELOG.md) —
read it before assuming any subsystem below is unchanged. Five newest:

- **pointcloud-cleanup** (2026-09-06)
- **depth-align-affine-2dgs-surface** (2026-08-27)
- **instantsfm-backend** (2026-08-23)
- **preproc-cleanup** (2026-08-22)
- **splats-module** (2026-08-22)

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

Core pipeline: video/images → pointcloud (feedforward: VGGT-X / VGGT-Omega / MapAnything / LoGeR, or sfm: InstantSfM + VDA depth) → optional BA / optional LC (both feedforward only — sfm refuses them) → `pointcloud.zarr` → mesh/features/splats.

```
collab_splats/
  pointcloud/              # main reconstruction pipeline
    base.py                # BasePointcloudCreator, PointcloudResult
    feedforward/           # BaseFeedforwardCreator (5-step template method), VGGTXCreator, MapAnythingCreator, FeedforwardResult
    sfm/                   # InstantSfMCreator (global SfM via upstream python API; the only wired backend)
                           #   + ColmapCreator / HlocCreator (incremental; not wired into _run_sfm)
    vda.py                 # generate_vda_depth: Video-Depth-Anything metric depth per keyframe
    depth_align.py         # result_from_reconstruction: COLMAP model + VDA depth -> FeedforwardResult at COLMAP scale
    utils.py               # lift_features, reproject_pixels, clean_pointcloud, confidence_mask, subsample_points
  geometry/                # pose/geometry backend: loop closure + bundle adjustment
    transforms.py          # extrinsics_to_homogeneous, invert_poses, OPENGL_TO_OPENCV (ex utils/geometry.py)
    bundle_adjustment.py   # Levenberg-Marquardt BA
    global_alignment.py    # VGGT-X native feature-match + joint BA (parked; not wired into any creator)
    loop_closure/          # submap pose graph (SL4/SE3), DINO-SALAD retrieval gate, LoopClosure wrapper
  localization/            # camera localization: query image → pose in known reconstruction
    retrieval.py           # Stage 1: BaseRetrievalExtractor, DinoSalad/PECLIP (also used by loop closure)
    extractors.py          # Stage 2: LocalMatcher over the vismatch model zoo (FEATURE_MATCH_MODELS fast paths)
    localizer.py           # Stage 3: CameraLocalizer, zarr feature cache
    viz.py                 # plot_correspondences
  semantics/               # 2D feature extraction
    features/              # BaseFeatureExtractor + RegistryMixin (base.py); registered
                           #   dinov2, maskclip, talk2dino — all ViT-width (384-1024D)
    compression.py         # FeatureAutoencoder: per-point encode/decode + recon_cosine
    segmentation/          # BaseSegmentation; registered insid3, mobilesamv2, sam3
    utils.py               # compute_semantic_contrast, interpolate_to_patch_size;
                           #   re-exports collab_splats.utils.torch_utils helpers
  preproc/                 # video preprocessing: measure (qa) then select (sampling)
    video.py               # PyAV decode: get_video_info, iter_frames, extract_frame
    qa.py                  # report-only capture quality: compute_video_quality, load_video_quality
    sampling.py            # context_indices + sample_fps | sample_uniform | sample_optical_flow, all from filter_frame_quality's eligible pool
    frames.py              # images/frame_NNNNNN.png + frames.json: the COLMAP-style keyframe store
    undistort.py           # calibrate_camera (pycolmap) + undistort_frames (pycolmap framing, cv2 pixels)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
  mesh/                    # TSDF + Poisson meshing (base, poisson, tsdf, utils)
  splats/                  # Gaussian-splat training on upstream gsplat: cameras, losses, rendering, trainer, outputs
  wrapper/                 # stage orchestration: Reconstructor (config-driven pipeline), batch drivers
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
- **Comment runs are a header then bullets:** any `#` run of three lines or more states the problem in 5-10 words on its own line, then gives the detail as `- ` bullets — never a wrapped paragraph. Bullets are fragments, not sentences.

  ```python
  # splatfacto's non-default DefaultStrategy args
  # - upstream: nerfstudio-project/nerfstudio @ 50e0e3c, splatfacto.py:264-280
  # - absgrad=False: 2dgs backward writes .absgrad on means2d, not gradient_2dgs
  # - grow_grad2d 2e-4: measured-good non-absgrad threshold
  ```
- **Section dividers:** use `########`-style dividers to separate major sections in long files (constants, helpers, classes, etc.). Keeps files scannable without opening a doc.
- **Docstrings follow the same shape.** Every module, public class and public function gets one. The `"""` open and close on their own lines — the summary starts on the line after the opening quotes, never on the same line, is one line of at most 100 chars, and does not restate the name. A blank line, then `- ` bullets, then the named sections. Never a prose paragraph.
- **Types live in the signature, meanings in the docstring.** Every parameter and return is annotated; `Args:` entries carry the name and what it means, never a duplicated type. A function with parameters documents every one under `Args:`; one that returns something documents it under `Returns:`, or `Yields:` if it is a generator.

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

  `tests/test_docstring_contract.py` enforces all of the above — docstrings, annotations and comment runs — for `preproc`, `semantics` and `pointcloud`. Add a package to its `PACKAGES` tuple as it is cleaned up.
- **Blank lines between blocks:** a run of statements that does one thing is separated from the next run, and every block comment gets a blank line above it. Walls of undifferentiated code are not human-readable.
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
