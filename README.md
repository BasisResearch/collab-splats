# collab-splats

This package provides an interface for 3D scene reconstruction & semantic feature extraction from 2D RGB videos. We can flexibly configure processing pipelines with a variety of reconstruction backends, mesh texturing, and semantic lifting.

## Quickstart

We run our package via a Docker image available at `tommybotch/collab-splats:release` (see [System requirements](#4-system-requirements)). Pull and start it:

```bash
# Pull the image (private repo: log in first)
docker login -u <DOCKERHUB_USER>
docker pull tommybotch/collab-splats:release

# Start a shell with your inputs, outputs and a model cache mounted
docker run --gpus all -it \
  -v /PATH/TO/INPUT:/PATH/TO/INPUT \
  -v /PATH/TO/OUTPUT:/PATH/TO/OUTPUT \
  -v /PATH/TO/MODELS:/workspace/models \
  tommybotch/collab-splats:release bash

# Inside the container (the venv is already active)
cd /workspace/collab-splats
```

Our pipeline runs from a `yaml` config. We provide `configs/base.yaml`, which reconstructs the scene with a feedforward model ([VGGT Omega](https://arxiv.org/abs/2605.15195)), creates a textured mesh, and lifts queryable semantics ([Talk2Dino](https://arxiv.org/abs/2411.19331) and OCR lens).

Run a single video:

```bash
reconstruct local --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

Run a batch of scenes, each an MP4 file or a directory of frame images (read in filename order):

```bash
reconstruct local --output-root /PATH/TO/OUTPUT/ \
  /PATH/TO/INPUT/INPUT.MP4 \
  /PATH/TO/INPUT/frames/
```

Change settings with your own config, which is merged over `base.yaml`:

```bash
reconstruct local --config configs/my_run.yaml --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

Every key is documented in [configs/README.md](configs/README.md).

## Package Organization

Each module is one stage of the pipeline; `reconstructor.py` chains them from the config:
- **preproc**: keyframe extraction, quality assessment (blur / exposure), frame filtering
- **pointcloud**: Structure-from-Motion (COLMAP, hloc, InstantSfM) or feedforward (VGGT-X, VGGT-Omega, MapAnything, LoGeR)
- **geometry**: bundle adjustment, loop closure, reconstruction quality report
- **mesh**: TSDF fusion, cleanup, texturing
- **splats**: Gaussian splatting (3DGS / 2DGS)
- **semantics**: VLM feature extraction, segmentation, 2D-to-3D lifting, text queries
- **localization**: locate a query image within a reconstruction

Supporting code: `utils/` (shared helpers), `viewer.py` + `dashboard/` (browser viewers), `remote.py` (cloud scene sync), `configs/` (pipeline configs), `evals/` (benchmarks).

## Output Structure

Each scene writes to `<output-root>/<name>/`. Frames and 2D caches are shared; each reconstruction backend gets its own folder:

```
<output-root>/<name>/
├── images/                         # keyframes
├── video_quality_report.json       # capture quality (blur, exposure, motion)
├── sky/                            # cached sky masks
├── semantics/                      # cached 2D features
└── vggt_omega/                     # one folder per backend
    ├── run_config.yaml             # config used for this run
    ├── pointcloud.zarr             # poses, intrinsics, depth, points
    ├── sparse_pc.ply               # pointcloud for viewing
    ├── colmap/sparse/0/            # COLMAP export
    ├── reconstruction_quality_report.json
    ├── mesh.ply                    # mesh
    ├── texture/                    # textured mesh (mesh.obj + albedo.png)
    ├── splats/ckpt.pt              # Gaussian splats (if enabled)
    └── semantics/<extractor>_lifted.zarr   # semantics on mesh vertices
```

View the mesh and query its semantics in the browser (`http://localhost:8080`, add `--textured` for the textured mesh):

```bash
python -m collab_splats.viewer /PATH/TO/OUTPUT/<name>/vggt_omega
```

Point it at `/PATH/TO/OUTPUT` instead to pick any meshed scene from a Scene dropdown.

## Install

### 1. Install uv

We use [uv](https://docs.astral.sh/uv/) for environment and dependency management. Install it once:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or: brew install uv
```

### 2. Install the package

We install everything with `setup.sh`. It creates the venv at `/opt/venv/reconstruction`, installs all dependencies, and compiles the CUDA extensions (`bae`, `gsplat`):

```sh
git clone https://github.com/BasisResearch/collab-splats.git
cd collab-splats
bash setup.sh
```

`setup.sh` is safe to re-run; it only builds what is missing. For a lighter install, sync extras individually:

```sh
uv sync                          # core deps only
uv sync --extra feedforward      # + feedforward models (VGGT-X, MapAnything, ...)
uv sync --extra gpu              # + CUDA wheels used to compile bae / gsplat
uv sync --extra dashboard        # + dashboard
uv sync --extra dev              # + lint / test tooling
uv sync --all-extras             # everything (what setup.sh does)
```

> **Use `setup.sh`, not a bare `uv sync`, and never `uv pip install ... @ git+...`.**
> - The CUDA extensions build from source and need `nvcc` + `build-essential`; `setup.sh` points the build at your CUDA toolkit and warns if `nvcc` is missing.
> - `uv pip install` ignores this project's `[tool.uv]` config and resolves the wrong package sources.

All later commands assume the venv is active (`source /opt/venv/reconstruction/bin/activate`), or prefix them with `uv run`.

### 3. Private dependency (collab-data)

`collab-data` is a private BasisResearch repo that must sit at `/workspace/collab-data`. `setup.sh` clones it there if missing (needs GitHub credentials).

The dashboard and `reconstruct remote` read and write scenes through an rclone remote named `collab-data` (Google Cloud Storage). Install `rclone` and `jq`:

```sh
brew install rclone jq                                                  # macOS
curl https://rclone.org/install.sh | sudo bash && sudo apt install jq   # Linux (apt's rclone is too old)
```

Save the GCS service-account key to `collab-data/config-local/collab-data.json`, then run the setup script from the collab-data repo:

```sh
./scripts/setup_local_rclone.sh
```

### 4. System requirements

You provide these; `setup.sh` never installs system packages.

- **NVIDIA GPU + driver** for VGGT-X, splats and mesh texturing. Other stages fall back to CPU, but slowly and untested.
- **build-essential** (gcc/g++) + a CUDA toolkit to compile the extensions.
- Optional: `ffmpeg`, `exiftool`, `rclone` for the video and data scripts.

### 5. Docker image

We build the image with the same `setup.sh`, with every CUDA extension compiled ahead of time. Build it from the collab-splats checkout, with `collab-data` cloned beside it (`../collab-data`):

```sh
docker build --platform=linux/amd64 --progress=plain --build-context collab-data=../collab-data --build-arg MAX_JOBS=4 -t collab-splats:release .
docker run --gpus all -it collab-splats:release bash
```

Push it to Docker Hub (keep the repo private: the image contains `collab-data`):

```sh
docker tag collab-splats:release tommybotch/collab-splats:release
docker push tommybotch/collab-splats:release
```

- **First build takes ~2.5 h**; later builds take minutes unless `pyproject.toml`, `uv.lock`, `setup.sh` or `collab-data` change.
- **`MAX_JOBS`**: Docker memory in GB / 8, at most 6.
- **GPUs**: Ampere/Ada (A100, A40, L40, RTX 30xx/40xx) and Hopper (H100). Not supported: Volta/Turing, Blackwell.

**Apple Silicon (Docker Desktop):**

- General: use the **Apple Virtualization framework** with **Rosetta** on (Docker VMM falls back to QEMU, which crashes the build).
- Resources: disk limit **≥ 250 GB**.
- Docker Engine: raise the build cache so the compiled layer is kept between builds:

  ```json
  "builder": { "gc": { "enabled": true, "defaultKeepStorage": "150GB" } }
  ```
- Never run `docker builder prune`, `docker system prune -a` or "Clean / Purge data": they delete the cached compile.

## Python API

The CLI wraps `Reconstructor`, which we can also run directly:

```python
from collab_splats.reconstructor import Reconstructor

scene = Reconstructor({"input_path": "video.mp4", "output_path": "out/"})
scene.run()
print(scene.outputs)
```

Our tutorials in `docs/source/tutorials/` walk through each stage on one shared scene:

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

We provide a browser dashboard to pick a remote scene and backend, run stages, view the pointcloud or mesh, and query semantics by text. It needs the dashboard extra (`uv sync --extra dashboard`) and the rclone remote:

```bash
python -m collab_splats.dashboard
```

Then open `http://localhost:7860` (change with `--port`, `--host`, `--base-dir`). Restart it to pick up code changes.

## Live Scene Viewer

`collab_splats.viewer.Viewer` streams point clouds and cameras from a running job to the browser (no display needed on the host). Re-adding a name replaces it:

```python
from collab_splats.viewer import Viewer
from collab_splats.pointcloud.utils import subsample_points

viewer = Viewer(port=8080)  # open http://<host>:8080
points, colors = subsample_points(points, colors, max_points=50_000)
viewer.add_points("submap_0", points, colors)
viewer.add_frustum("submap_0/cams/frame_0", pose_w2c, intrinsic)
```

## Curated Videos

We curate raw environment videos into the `environments-curated` bucket. `scripts/preprocess_gdrive_videos.py` makes one folder per video and restores the camera metadata (GPS, IMU) that editing strips; `scripts/push_curated.sh` uploads the result. Both need `ffmpeg` + `exiftool` and skip work already done:

```sh
python scripts/preprocess_gdrive_videos.py --dry-run   # ../gdrive-src -> ../environments-curated
python scripts/preprocess_gdrive_videos.py
python scripts/preprocess_gdrive_videos.py --only GH010234   # one clip

./scripts/push_curated.sh --dry-run
./scripts/push_curated.sh
```

## Evaluation

We benchmark against ground-truth datasets with `evals/eval.py`:

```bash
python -m evals.eval --config evals/configs/7scenes.yaml --dry_run
```

Results land in `evals/results/`; datasets and grid configs are described in [evals/README.md](evals/README.md).
