# collab-splats

Video/image → 3D pointcloud → mesh + semantic features. Feedforward reconstruction using VGGT-X or MapAnything, with optional bundle adjustment, SL(4) loop closure, TSDF/Poisson meshing, semantic feature lifting, and camera localization.

## Capabilities

- **Feedforward reconstruction** — VGGT-X and MapAnything pointcloud creators
- **Bundle adjustment** — Levenberg-Marquardt refinement (bae backend)
- **Loop closure** — SL(4) pose graph from MIT-SPARK/VGGT-SLAM
- **Semantic lifting** — DINOv2/SAM features lifted into 3D pointcloud
- **Meshing** — TSDF and Poisson surface reconstruction
- **Camera localization** — SALAD retrieval + hloc matching

## Install

```bash
bash setup.sh               # full install (uv sync — all deps incl. VGGT-X + MapAnything)
```

Python env: always use `/opt/venv/reconstruction/bin/python` (py3.11).

## Getting Started

Tutorials in `docs/source/tutorials/`, numbered by pipeline stage:

| Stage | Topic |
|-------|-------|
| 01 · Preprocessing | Keyframe extraction |
| 02 · Pointcloud | Feedforward methods, bundle adjustment, loop closure, COLMAP |
| 03 · Splats | Derive splats, visualization |
| 04 · Semantics | Feature extraction, segmentation |
| 05 · Lifting | Semantic feature lifting |
| 06 · Mesh | Surface reconstruction |
| 07 · Localization | Camera localization |

## Evaluation

```bash
/opt/venv/reconstruction/bin/python evals/eval_gt.py --help
```

Results land in `evals/results/` (gitignored). See `docs/source/tutorials/evals/ground_truth_evals.ipynb` for visualization.
