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

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install it once:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: brew install uv
```

### 2. Install the package

One command does the full install — creates the venv, syncs all dependencies (including the VGGT-X + MapAnything feedforward stack), wires the CUDA build environment, and compiles the source extensions (`bae`, `gsplat-rade`):

```sh
bash setup.sh
```

`setup.sh` is the single source of truth — it runs at Docker build time and standalone, and is idempotent (`uv sync` installs only what's missing, skipping already-compiled extensions). Under the hood it runs `uv sync --all-extras` into `/opt/venv/reconstruction`.

For a lighter, selective install, sync individual extras instead:

```sh
uv sync                          # core deps only
uv sync --extra feedforward      # + VGGT-X, MapAnything, LightGlue, SALAD, CO3D
uv sync --extra gpu              # + CUDA build toolkit (nvcc) for compiling bae / gsplat
uv sync --extra dashboard        # + Panel dashboard
uv sync --extra dev              # + lint / test tooling
uv sync --all-extras             # everything (what setup.sh does)
```

> **Note:** `bae` and `gsplat-rade` are CUDA extensions built from source with `--no-build-isolation`. They need a CUDA toolkit (`nvcc`) and `build-essential` on the build host — `setup.sh` handles the `CUDA_HOME` / `PATH` wiring (system `/usr/local/cuda` if present, else the pip `cuda-toolkit` wheels from the `[gpu]` extra). Prefer `bash setup.sh` over a bare `uv sync` whenever the compiled extensions are involved.

### 3. Private dependency (collab-data)

`collab-data` is a private BasisResearch repo, kept out of the locked graph and installed post-sync (needs git credentials). `setup.sh` installs it best-effort; on a credential-less build it is skipped — re-run `setup.sh` at deploy, or install it directly:

```sh
uv pip install "git+https://github.com/BasisResearch/collab-data.git"
```

### 4. System requirements

- **NVIDIA driver + GPU** at runtime (model warmup loads CUDA kernels at import).
- **build-essential** (gcc/g++) + a CUDA toolkit at build time for the source extensions.
- Optional: `colmap`, `ffmpeg`, `rclone` for the COLMAP and data pipelines.

All subsequent commands assume the venv is active (`source /opt/venv/reconstruction/bin/activate`), or prefix them with `uv run`. The interpreter is always `/opt/venv/reconstruction/bin/python` (py3.11).

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
