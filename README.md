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
> - The CUDA extensions (`bae`, `gsplat`) build from source and need `nvcc` + `build-essential`. `setup.sh` wires `CUDA_HOME`/`PATH` to the toolkit (default `/usr/local/cuda`), warns when nvcc is absent, and prints the install recipe only if a build needs it (no pip wheel ships nvcc).
> - `uv pip install` ignores this project's `[tool.uv]` config, so it resolves wrong package sources and fails.

### 3. Private dependency (collab-data)

`collab-data` is a private BasisResearch repo, locked as a path dependency at `/workspace/collab-data` — `uv sync` cannot resolve without it. `setup.sh` clones it there if missing (needs GitHub credentials).

**Data access (rclone remote).** The dashboard reads and writes scenes over an rclone
remote named `collab-data` (Google Cloud Storage). Set it up once.

Install `rclone` and `jq`:

```sh
brew install rclone jq                 # macOS
curl https://rclone.org/install.sh | sudo bash && sudo apt install jq   # Linux (apt's rclone is 1.53, from 2020)
```

Save a GCS service-account key for the collab-data project to
`collab-data/config-local/collab-data.json`, then run the setup script from the collab-data
repo. It configures the `collab-data` remote and verifies access:

```sh
./scripts/setup_local_rclone.sh
```

**Curated environment videos.** `scripts/preprocess_gdrive_videos.py` turns the nested Drive
export into one folder per video and carries the capture metadata across: DaVinci Resolve
strips every timed data track on export, so the script pairs each export against its camera
original in `src/`, solves the trim offset from audio, and writes the metadata back as
static tags plus a retimed `gpmd` track inside the mp4, with a full-rate Parquet sidecar
beside it. `scripts/push_curated.sh` uploads that tree to the `environments-curated` bucket.
Both are re-runnable and skip work already done.

Only videos that have a `src/` counterpart are processed; an unedited camera original is
skipped and enters scope automatically once it is exported.

```sh
python scripts/preprocess_gdrive_videos.py --dry-run   # ../gdrive-src -> ../environments-curated
python scripts/preprocess_gdrive_videos.py
python scripts/preprocess_gdrive_videos.py --only GH010234   # one clip
python scripts/preprocess_gdrive_videos.py --index-only      # rebuild index.csv alone

./scripts/push_curated.sh --dry-run            # -> collab-data:environments-curated
./scripts/push_curated.sh
```

Each curated folder holds the video, a `_metadata.json` sidecar, and a `_telemetry.parquet`
sidecar when the camera recorded IMU. `environments-curated/index.csv` is a two-column
`unique_id,gps` table regenerated from the sidecars on every run.

`exiftool` is required alongside `ffmpeg` for this pipeline.

### 4. System requirements

You provide these; `setup.sh` never installs system packages.

- **NVIDIA driver + GPU** at runtime for the VGGT-X backend, splats (gsplat) and mesh texturing (nvdiffrast); without one, VGGT-X refuses at model load and texturing at the mesh stage. Other stages fall back to CPU, untested end to end and slow for model inference.
- **build-essential** (gcc/g++) + a CUDA toolkit at build time for the source extensions.
- Optional: `ffmpeg`, `exiftool`, `rclone` for the video and data pipelines.

All subsequent commands assume the venv is active (`source /opt/venv/reconstruction/bin/activate`), or prefix them with `uv run`. The interpreter is always `/opt/venv/reconstruction/bin/python` (py3.11).

### 5. Docker image

The image runs the same `setup.sh`, with every CUDA extension compiled ahead of time. The runtime
image has no nvcc by design; re-running `setup.sh` there is safe. Build it
from the collab-splats checkout, with `collab-data` cloned beside it (`../collab-data`); it is
a locked path dependency, handed to the build as a named context so no GitHub credentials are needed:

```sh
docker build --platform=linux/amd64 --progress=plain --build-context collab-data=../collab-data --build-arg MAX_JOBS=4 -t collab-splats:release .
docker run --gpus all -it collab-splats:release bash
```

- **First build takes ~2.5 h.** Most of it is gsplat's 3DGUT kernel (~2 h CPU even on native x86).
- **Rebuilds take minutes.** The CUDA compile is its own cached layer; it re-runs only when
  `pyproject.toml`, `uv.lock`, `LICENSE`, `setup.sh` or `../collab-data` change.
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

## Getting Started

```python
from collab_splats.reconstructor import Reconstructor

scene = Reconstructor({"input_path": "video.mp4", "output_path": "out/"})
scene.run()
print(scene.outputs)
```

Same run from the shell: `reconstruct local video.mp4 --output-root out/` (writes to `out/video/`).
Browse the result: `python -m collab_splats.viewer out/vggt_omega` (the CLI run: `out/video/vggt_omega`).

Tutorials in `docs/source/tutorials/` share one scene and build on each other:

| Stage | Tutorial |
|-------|----------|
| preproc | [Preprocessing](docs/source/tutorials/01_preprocessing/preprocessing.ipynb) |
| pointcloud, quality report | [Reconstruction](docs/source/tutorials/02_pointcloud/reconstruction.ipynb) |
| refine | [Refinement](docs/source/tutorials/02_pointcloud/refinement.ipynb) |
| splats | [Train splats](docs/source/tutorials/03_splats/train_splats.ipynb) |
| mesh | [Mesh](docs/source/tutorials/04_mesh/mesh.ipynb) |
| semantics | [Features](docs/source/tutorials/05_semantics/feature_extraction.ipynb) · [Segmentation](docs/source/tutorials/05_semantics/segmentation.ipynb) · [Lifting and query](docs/source/tutorials/05_semantics/lifting_and_query.ipynb) · [OCR lens](docs/source/tutorials/05_semantics/ocr_lens.ipynb) |
| localize | [Localization](docs/source/tutorials/06_localization/localization.ipynb) |

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
points, colors = subsample_points(points, colors, max_points=50_000)
viewer.add_points("submap_0", points, colors)
viewer.add_frustum("submap_0/cams/frame_0", pose_w2c, intrinsic)
```

Use `subsample_points` to cap each cloud to a fixed budget so multi-part scenes stay
balanced. GUI toggles: camera visibility and flat per-node coloring (shows part boundaries).

## Evaluation

```bash
/opt/venv/reconstruction/bin/python -m evals.eval --config evals/configs/7scenes.yaml --dry_run
```

Results land in `evals/results/` (gitignored); data download and grid configs: `evals/README.md`. See `docs/source/tutorials/evals/ground_truth_evals.ipynb` for visualization.