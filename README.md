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
extensions (`bae`, `gsplat`):

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
uv sync --extra gpu              # + CUDA runtime/header wheels used when compiling bae / gsplat
uv sync --extra dashboard        # + Panel dashboard
uv sync --extra dev              # + lint / test tooling
uv sync --all-extras             # everything (what setup.sh does)
```

> **Prefer `setup.sh` over a bare `uv sync` — and never `uv pip install ... @ git+...`.** Why:
> - The CUDA extensions (`bae`, `gsplat`) build from source and need `nvcc` + `build-essential`. `setup.sh` wires `CUDA_HOME`/`PATH` to the system `/usr/local/cuda` and fails fast with a micromamba recipe when nvcc is absent (no pip wheel ships nvcc).
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
- Optional: `ffmpeg`, `rclone` for the video and data pipelines.

### 5. Docker image

The image runs the same `setup.sh`, with every CUDA extension compiled ahead of time. Build it
from the collab-splats checkout, with `collab-data` cloned beside it (`../collab-data`); it is
a locked path dependency, handed to the build as a named context so no GitHub credentials are needed:

```sh
docker build --platform=linux/amd64 --progress=plain --build-context collab-data=../collab-data --build-arg MAX_JOBS=4 -t collab-splats:release .
docker run --gpus all -it collab-splats:release bash
```

- **First build takes ~2.5 h.** Most of it is gsplat's 3DGUT kernel (~2 h CPU even on native x86).
- **Rebuilds take minutes.** The CUDA compile is its own cached layer; it re-runs only when
  `pyproject.toml`, `uv.lock`, `setup.sh` or `../collab-data` change.
- **`MAX_JOBS`**: Docker memory in GB / 8 (one `cicc` peaks ~7.3 GB), at most 6.
- **GPUs**: native on Ampere/Ada (A100, A40, L40, RTX 30xx/40xx); Hopper (H100) via PTX JIT on
  first launch; Volta/Turing and Blackwell unsupported. Details in the `Dockerfile` header.

**Apple Silicon (Docker Desktop):**

- General: use the **Apple Virtualization framework** with **Rosetta for x86_64/amd64 emulation**
  on. Docker VMM has no Rosetta, falls back to QEMU, and uv segfaults under it.
- Resources: disk usage limit **≥ 250 GB** (the default 60 GB fills during the runtime stage).
- Docker Engine: raise the build-cache cap, or the 20 GB default evicts the compiled layer
  between builds:

  ```json
  "builder": { "gc": { "enabled": true, "defaultKeepStorage": "150GB" } }
  ```
- Never run `docker builder prune`, `docker system prune -a` or "Clean / Purge data": they
  delete the cached compile.

All subsequent commands assume the venv is active (`source /opt/venv/reconstruction/bin/activate`), or prefix them with `uv run`. The interpreter is always `/opt/venv/reconstruction/bin/python` (py3.11).

## Getting Started

Tutorials in `docs/source/tutorials/`, numbered by pipeline stage:

| Stage | Topic |
|-------|-------|
| 01 · Preprocessing | Keyframe extraction |
| 02 · Pointcloud | Feedforward methods, bundle adjustment, loop closure, COLMAP |
| 03 · Splats | [Gaussian-splat training](docs/source/tutorials/03_splats/train_splats.ipynb) (3DGS on upstream gsplat, 2DGS config diff) |
| 04 · Semantics | Feature extraction, segmentation |
| 05 · Lifting | Semantic feature lifting |
| 06 · Mesh | [Surface reconstruction](docs/source/tutorials/06_mesh/splats_mesh.ipynb) (TSDF from feedforward depth vs splat renders, semantic mesh query) |
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