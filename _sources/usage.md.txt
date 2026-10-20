# Usage

## Package Organization

Each module is one stage of the pipeline; `reconstructor.py` chains them from the config:
- **preproc**: keyframe extraction, quality assessment (blur / exposure), frame filtering
- **pointcloud**: Structure-from-Motion (COLMAP, hloc, InstantSfM) or feedforward (VGGT-X, VGGT-Omega, MapAnything, LoGeR)
- **geometry**: bundle adjustment, loop closure, reconstruction quality report
- **mesh**: TSDF fusion, cleanup, texturing
- **splats**: Gaussian splatting (3DGS / 2DGS)
- **semantics**: image feature extraction (DINOv2, CLIP-aligned Talk2DINO / MaskCLIP), segmentation, 2D-to-3D lifting, text queries
- **localization**: locate a query image within a reconstruction

Supporting code: `utils/` (shared helpers), `viewer.py` + `dashboard/` (browser viewers), `remote.py` (cloud scene sync), `configs/` (pipeline configs), `evals/` (benchmarks).

## Python API

The `reconstruct` CLI wraps `Reconstructor`, which we can also run directly (CLI flags and config keys: [Configuration](configuration.md)):

```python
from collab_splats.reconstructor import Reconstructor

scene = Reconstructor({"input_path": "video.mp4", "output_path": "out/"})
scene.run()
print(scene.outputs)
```

Our tutorials in `docs/source/tutorials/` walk through each stage on one shared scene:

| Stage | Tutorial |
|-------|----------|
| preproc | [Preprocessing](tutorials/01_preprocessing/preprocessing.ipynb) |
| pointcloud, reconstruction_quality_report | [Reconstruction](tutorials/02_pointcloud/reconstruction.ipynb) |
| refine | [Refinement](tutorials/02_pointcloud/refinement.ipynb) |
| splats | [Training Splats](tutorials/03_splats/train_splats.ipynb) |
| mesh | [Mesh](tutorials/04_mesh/mesh.ipynb) |
| semantics | [Feature Extraction](tutorials/05_semantics/feature_extraction.ipynb) · [Segmentation](tutorials/05_semantics/segmentation.ipynb) · [Lifting and Query](tutorials/05_semantics/lifting_and_query.ipynb) · [OCR Lens](tutorials/05_semantics/ocr_lens.ipynb) |
| localize | [Localization](tutorials/06_localization/localization.ipynb) |

## Dashboard

We provide a browser dashboard to pick a remote scene and backend, run stages, view the pointcloud or mesh, and query semantics by text. It imports the whole pipeline, so it needs the full install (`bash setup.sh`), plus the rclone remote:

```bash
collab-dashboard --base-dir /PATH/TO/OUTPUT   # or: python -m collab_splats.dashboard --base-dir ...
```

Then open `http://localhost:7860` (change with `--port`, `--host`). When browsing via an IP or forwarded hostname, add `--websocket-origin <host>:<port>` (or `'*'`), otherwise the page loads but never renders. Restart it to pick up code changes.

## Adding Curated Videos

Raw video files are prepared for the `environments-curated` bucket with `scripts/preprocess_gdrive_videos.py`. This script writes one local directory per video and keeps the camera metadata (GPS, IMU) that video editing removes; `scripts/push_curated.sh` (or `--push`) uploads the result. The script requires `ffmpeg` + `exiftool`.

The processing pipeline is illustrated below:

```sh
python scripts/preprocess_gdrive_videos.py --dry-run   # ../gdrive-src -> ../environments-curated
python scripts/preprocess_gdrive_videos.py
python scripts/preprocess_gdrive_videos.py --only GH010234   # one clip

./scripts/push_curated.sh --dry-run
./scripts/push_curated.sh
```

## Evaluation

During design of the package modules, we used standard 3D reconstruction benchmarks to understand the benefits of given pipeline configurations. Ground-truth datasets can be benchmarked with `evals/eval.py`:

```bash
python -m evals.eval --config evals/configs/7scenes.yaml --dry-run
```

Results land in `evals/results/`; datasets and grid configs are described in [evals/README.md](https://github.com/BasisResearch/collab-splats/blob/main/evals/README.md).
