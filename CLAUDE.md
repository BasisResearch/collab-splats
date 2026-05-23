# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always follow:
- Use rtk tools /workspace/.claude/RTK.md
- At the start of every conversation always call /brainstorming -- use superpowers to accomplish tasks.
- Before anything, read worklog/STATE.md (current state), then latest entries in worklog/WORKLOG.md, then worklog/decisions/NNN-*.md as referenced.
- Active spec/plan for in-flight work lives in worklog/{specs,plans}/. Completed work is archived to worklog/history/.
- Architecture decisions: worklog/decisions/NNN-slug.md (sequential numbering).
- User-facing module docs live in docs/ (mirrors code tree).
- Superpowers information belongs in worklog/ directory

## Installation and Setup

```bash
bash setup.sh                    # core install (nerfstudio env)
bash setup_feedforward.sh        # VGGT-X + MapAnything
```

## Development Environment

- **Python env:** `python` = base conda py3.13 (wrong for this project). Always use `/opt/conda/envs/nerfstudio/bin/python` (py3.11).
- **Memory:** container cgroup cap 46.6 GB. Heavy inference/eval → run in tmux, not notebooks.
- **Side shells:** don't run parallel processes during heavy eval runs (OOM risk).

## Architecture Overview

Core pipeline: video/images → pointcloud (VGGT-X or MapAnything) → optional BA → optional LC → mesh/features.

```
collab_splats/
  pointcloud/              # main reconstruction pipeline
    base.py                # BasePointcloudCreator, PointcloudResult
    feedforward/           # BaseFeedforwardCreator (5-step template method), VGGTXCreator, MapAnythingCreator, FeedforwardResult
    bundle_adjustment.py   # Levenberg-Marquardt BA
    wrappers.py            # BundleAdjustment + LoopClosure wrappers (proxy outputs/raw_outputs to base)
    loop_closure/          # pose graph + Sim3 alignment
    localization.py        # BaseRetrievalExtractor, DinoSaladExtractor (localization stage 1)
    utils.py               # lift_features, reproject_pixels, colmap_reconstruction_to_result
  semantics/               # 2D feature extraction
    features.py            # BaseFeatureExtractor + RegistryMixin; registered DINOv2/SAM extractors
    frame_sampling.py      # re-exported here; canonical at utils/frame_sampling.py
  mesh/                    # TSDF + Poisson meshing (base, poisson, tsdf, utils)
  nerfstudio/              # nerfstudio method configs, models, datamanagers
  dashboard/               # interactive video/scene browser
  utils/
    torch_utils.py         # RegistryMixin, pytorch_gc, infer_batch_size, batch_iterator, get_device
    frame_sampling.py      # optical-flow + FPS keyframe selection
evals/
  eval_gt.py               # compute script — CLI/tmux only, never run in notebook
  datasets.py              # dataset loaders (7-Scenes, CO3Dv2)
  results/                 # gitignored
```

## Code Style

- **Inline block comments:** each logical block of code gets a short comment explaining what it does. Comment at block level, not every line.
- `logging` not `print()` — use `logger.debug()` / `logger.info()` throughout module code
- `########`-style section dividers in long files
- `RegistryMixin` for registry pattern (from `utils/torch_utils.py`)
- Template-method pattern for abstract pipelines (see `BaseFeedforwardCreator`)
- Typed `@dataclass` for pipeline outputs (`FeedforwardResult`, `PointcloudResult`)
- Hard imports — no stub backends; let missing deps raise `ImportError` at import time

## Testing

- Flat test functions — no class-based unless shared fixture state requires it
- `tests/` mirrors `collab_splats/` structure
- Run: `/opt/conda/envs/nerfstudio/bin/python -m pytest tests/`

## Evaluation

- `evals/eval_gt.py` = compute (CLI/tmux only); notebooks in `docs/` = visualization only
- Results: `evals/results/` (gitignored); pass `--submap_size 50` for >100-frame sequences

## Commit Conventions

Conventional commits with scope: `feat(pointcloud):`, `fix(ba):`, `refactor(semantics):`, `docs(worklog):`

## Development Commands

```bash
black . && isort .                                                  # format
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/             # test
/opt/conda/envs/nerfstudio/bin/python evals/eval_gt.py --help      # eval
```
