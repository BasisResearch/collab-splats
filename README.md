# collab-splats

Video/image → 3D pointcloud → mesh + semantic features. Feedforward reconstruction using VGGT-X or MapAnything, with optional bundle adjustment, SL(4) loop closure, TSDF/Poisson meshing, semantic feature lifting, and camera localization.

## Capabilities

- **Feedforward reconstruction** — VGGT-X and MapAnything pointcloud creators
- **Bundle adjustment** — Levenberg-Marquardt refinement (bae backend)
- **Loop closure** — SL(4) pose graph from MIT-SPARK/VGGT-SLAM
- **Semantic lifting** — DINOv2/SAM features lifted into 3D pointcloud
- **Meshing** — TSDF and Poisson surface reconstruction
- **Camera localization** — SALAD retrieval + hloc matching
- **Live scene viewer** — browser-viewable viser scene for watching reconstructions build in real time

## Install

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install it once:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: brew install uv
```

### 2. Install the package

Clone and run `setup.sh`. It creates the venv at `/opt/venv/reconstruction`, syncs all
dependencies (incl. the VGGT-X + MapAnything feedforward stack), and compiles the CUDA
extensions (`bae`, `gsplat-rade`):

```sh
git clone https://github.com/BasisResearch/collab-splats.git
cd collab-splats
bash setup.sh
```

`setup.sh` is idempotent (re-run anytime; it builds only what's missing) and is the same
script used at Docker build. It runs `uv sync --all-extras`. For a lighter install, sync
extras individually:

```sh
uv sync                          # core deps only
uv sync --extra feedforward      # + VGGT-X, MapAnything, LightGlue, SALAD, CO3D
uv sync --extra gpu              # + CUDA build toolkit (nvcc) for compiling bae / gsplat
uv sync --extra dashboard        # + Panel dashboard
uv sync --extra dev              # + lint / test tooling
uv sync --all-extras             # everything (what setup.sh does)
```

> **Prefer `setup.sh` over a bare `uv sync` — and never `uv pip install ... @ git+...`.** Why:
> - The CUDA extensions (`bae`, `gsplat-rade`) build from source and need `nvcc` + `build-essential`. `setup.sh` wires `CUDA_HOME`/`PATH` (system `/usr/local/cuda`, else the `[gpu]` extra's `cuda-toolkit` wheels).
> - `uv pip install` ignores this project's `[tool.uv]` config, so it resolves wrong package sources and fails.

### 3. Private dependency (collab-data)

`collab-data` is a private BasisResearch repo, kept out of the locked graph and installed post-sync (needs git credentials). `setup.sh` installs it best-effort; on a credential-less build it is skipped — re-run `setup.sh` at deploy, or install it directly:

```sh
uv pip install "git+https://github.com/BasisResearch/collab-data.git"
```

**Data access (rclone remote).** The dashboard reads and writes scenes over an rclone
remote named `collab-data` (Google Cloud Storage). Set it up once.

Install `rclone` and `jq`:

```sh
brew install rclone jq                 # macOS
sudo apt install rclone jq             # Debian / Ubuntu
```

Save a GCS service-account key for the collab-data project to
`collab-data/config-local/collab-data.json`, then run the setup script from the collab-data
repo. It configures the `collab-data` remote and verifies access:

```sh
./scripts/setup_local_rclone.sh
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

## Dashboard

Browse scenes, reconstruct, mesh, lift features, and query the pointcloud or mesh by text.
Needs the dashboard extra (`uv sync --extra dashboard`).

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard
```

Then open `http://localhost:7860`. Override defaults with `--port`, `--host`, or
`--base-dir` (defaults: `7860`, `0.0.0.0`, `/workspace/outputs`). Restart the process to
pick up code changes — there is no autoreload.

Use the view toggle to switch the left pane between `pointcloud` and `mesh`. Type a query
to colour the right pane by similarity. Pointcloud and mesh share the same features, so
both panes answer the same query.

## Live Scene Viewer

`collab_splats.viewer.Viewer` serves a live 3D scene over websockets (viser) — push named
point clouds, camera frusta, and line segments from any running job (e.g. a tmux
reconstruction) and watch them land in the browser. No display/GL needed on the host.
Re-adding a node under the same name replaces it, so a scene can be refreshed in place.

```python
from collab_splats.viewer import Viewer
from collab_splats.pointcloud.utils import subsample_points

viewer = Viewer(port=8080)  # open http://<host>:8080
points, colors = subsample_points(points, colors, conf=conf, max_points=50_000)
viewer.add_points("submap_0", points, colors)
viewer.add_frustum("submap_0/cams/frame_0", pose_w2c, intrinsic)
```

Use `subsample_points` to cap each cloud to a fixed budget so multi-part scenes stay
balanced. GUI toggles: camera visibility and flat per-node coloring (shows part boundaries).

## Evaluation

```bash
/opt/venv/reconstruction/bin/python evals/scripts/eval.py --help
```

Results land in `evals/results/` (gitignored). See `docs/source/tutorials/evals/ground_truth_evals.ipynb` for visualization.