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
    features.py            # BaseFeatureExtractor + RegistryMixin; registered DINOv2/SAM extractors
  preproc/                 # video preprocessing: frame sampling + quality gate
    sampling.py            # ffmpeg-only decode, blur/exposure gate, sample_frames (uniform | optical_flow)
    viz.py                 # sampling analysis plots (notebook-only, not re-exported)
  mesh/                    # TSDF + Poisson meshing (base, poisson, tsdf, utils)
  nerfstudio/              # nerfstudio method configs, models, datamanagers
  dashboard/               # interactive video/scene browser
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
