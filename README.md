# collab-splats

This package provides an interface for 3D scene reconstruction & semantic feature extraction from 2D RGB videos. We can flexibly configure processing pipelines with a variety of reconstruction backends, mesh texturing, and semantic lifting.

## Quickstart

### Start the container

Our package runs via a Docker image available at `tommybotch/collab-splats:release` (see [System requirements](docs/source/installation.md#system-requirements)). Run the following code to pull and start the image:

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

### Local video processing

A pipeline runs using a `yaml` config — the default is `configs/base.yaml`. This reconstructs a video with [VGGT Omega](https://arxiv.org/abs/2605.15195), creates a textured mesh, and lifts queryable semantics ([Talk2Dino](https://arxiv.org/abs/2411.19331) and [OCR lens](https://ocr.baulab.info/)).

Single video or batch processing can be run with the following commands:

```bash
# run a single video at a path
reconstruct local --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4

# run a batch of scenes (either MP4 file or directory of images read in filename order)
reconstruct local --output-root /PATH/TO/OUTPUT/ \
  /PATH/TO/INPUT/INPUT.MP4 \
  /PATH/TO/INPUT/frames/

# override base.yaml with custom settings
reconstruct local --config configs/my_run.yaml --output-root /PATH/TO/OUTPUT/ /PATH/TO/INPUT/INPUT.MP4
```

For a complete list of configuration options, please see [configs/README.md](configs/README.md).

### Remote video processing

We also can use `collab-splats` for processing videos hosted in GCloud storage. However, this option requires (1) access to the `collab-data` repository and (2) GCS access (see [Private dependency](docs/source/installation.md#private-dependency-collab-data)). Please follow these steps to set up remote processing:

1. Clone the private [collab-data](https://github.com/BasisResearch/collab-data) repo to `/workspace/collab-data`.
2. Ask a maintainer for the GCS service-account key and save it as `/workspace/collab-data/config-local/collab-data.json`.
3. Create the `collab-data` rclone remote, then check it lists the curated scenes:

```bash
# ensure connection to GCS bucket and list directories in bucket
(cd /workspace/collab-data && ./scripts/setup_local_rclone.sh)
rclone lsd collab-data:environments-curated
```

After verifying the connection, we can use the `reconstruct remote` command to process files within the `environments-curated` bucket. The remote command automatically pulls files locally, processes through the pipeline, and pushes output to `environments-processed/<scene>/` — to complete the push it verifies every file uploaded before deleting local files.

```bash
# process two named scenes
reconstruct remote --output-root /workspace/outputs \
  2026_07_20-birds-C0043 2026_07_21-rats-C0100

# process all in the bucket (re-runs and overwrites every scene)
reconstruct remote --output-root /workspace/outputs --all

# process all files and keep a local copy of the results
reconstruct remote --output-root /workspace/outputs --all --keep-local

# if outputs exist, we can rerun a single stage of the pipeline without rebuilding the entire scene
reconstruct remote --output-root /workspace/outputs \
  --stages mesh --overwrite --config configs/NEW_CONFIG.yaml --all
```

Exit codes, leaf stages and failure handling are in [configs/README.md](configs/README.md#remote-scenes).

## Output Structure

Each scene writes to `<output-root>/<name>/`. Frames and 2D caches are shared; each reconstruction backend gets its own folder:

```
<output-root>/<name>/
├── images/                         # keyframes
├── video_quality_report.json       # capture quality (blur, exposure, motion)
├── sky/                            # cached sky masks
├── semantics/                      # cached 2D features
└── <backend>/                      # one folder per backend
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
python -m collab_splats.viewer /PATH/TO/OUTPUT/<name>/<backend>
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
