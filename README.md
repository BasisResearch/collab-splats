# collab-splats

This package provides an interface for 3D scene reconstruction & semantic feature extraction from 2D RGB videos. We can flexibly configure processing pipelines with a variety of reconstruction backends, mesh texturing, and semantic lifting.

## Quickstart

### Start the container

We run our package via a Docker image available at `tommybotch/collab-splats:release` (see [System requirements](docs/source/installation.md#system-requirements)). Pull and start it:

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

### Run on local videos

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

### Process scenes on GCloud

Setup (the image already has `rclone` and `jq`):

1. Clone the private [collab-data](https://github.com/BasisResearch/collab-data) repo to `/workspace/collab-data`.
2. Ask a maintainer for the GCS service-account key and save it as `/workspace/collab-data/config-local/collab-data.json`.
3. Create the `collab-data` rclone remote, then check it lists the curated scenes:

```bash
(cd /workspace/collab-data && ./scripts/setup_local_rclone.sh)
rclone lsd collab-data:environments-curated
```

Using `reconstruct remote` automatically links to the `environments-curated` bucket and requires ```collab-data``` access with the rclone remote (see [Private dependency](docs/source/installation.md#private-dependency-collab-data)). For each scene it pulls the video, reconstructs it, pushes the outputs to `environments-processed/<scene>/`, verifies the push, then deletes the local copy:

```bash
# Named scenes (the curated directory names)
reconstruct remote --output-root /workspace/outputs \
  2026_07_20-birds-C0043 2026_07_21-rats-C0100

# Everything in the bucket
reconstruct remote --output-root /workspace/outputs --all

# Keep the local copy for inspection
reconstruct remote --output-root /workspace/outputs --all --keep-local
```

Re-run one stage on scenes we already processed by naming only that stage; it pulls the scene back from `environments-processed` instead of rebuilding it:

```bash
reconstruct remote --output-root /workspace/outputs \
  --stages mesh --overwrite --config configs/my_run.yaml --all
```

- `--all` re-runs every curated scene (every processed scene when re-running one stage) and overwrites its processed outputs; nothing is skipped.
- Nothing local is deleted until the push verifies, and the curated video stays in the bucket.
- Exit codes, leaf stages and failure handling are in [configs/README.md](configs/README.md#remote-scenes).

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

### Viewing completed reconstructions

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

We install everything with `setup.sh`. It creates the venv at `/opt/venv/reconstruction`, installs all dependencies, and compiles the CUDA extensions (`bae`, `gsplat`, `fused-ssim`, `nvdiffrast`):

```sh
git clone https://github.com/BasisResearch/collab-splats.git
cd collab-splats
bash setup.sh
```

For extras, the semantics-only install and system requirements, see [Installation](docs/source/installation.md).

## Learn more

- [Installation](docs/source/installation.md): extras, semantics-only install, collab-data, system requirements
- [Docker image](docs/source/docker.md): build and push the image, Apple Silicon setup
- [Configuration](configs/README.md): every config key and stage
- [Usage](docs/source/usage.md): package organization, Python API and tutorials, dashboard, adding curated videos, evaluation
