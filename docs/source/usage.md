# Usage

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
| preproc | [Preprocessing](tutorials/01_preprocessing/preprocessing.ipynb) |
| pointcloud, quality report | [Reconstruction](tutorials/02_pointcloud/reconstruction.ipynb) |
| refine | [Refinement](tutorials/02_pointcloud/refinement.ipynb) |
| splats | [Train splats](tutorials/03_splats/train_splats.ipynb) |
| mesh | [Mesh](tutorials/04_mesh/mesh.ipynb) |
| semantics | [Features](tutorials/05_semantics/feature_extraction.ipynb) · [Segmentation](tutorials/05_semantics/segmentation.ipynb) · [Lifting and query](tutorials/05_semantics/lifting_and_query.ipynb) · [OCR lens](tutorials/05_semantics/ocr_lens.ipynb) |
| localize | [Localization](tutorials/06_localization/localization.ipynb) |

## Dashboard

We provide a browser dashboard to pick a remote scene and backend, run stages, view the pointcloud or mesh, and query semantics by text. It needs the dashboard extra (`uv sync --extra dashboard`) and the rclone remote:

```bash
python -m collab_splats.dashboard
```

Then open `http://localhost:7860` (change with `--port`, `--host`, `--base-dir`). Restart it to pick up code changes.

## Adding Curated Videos

Raw video files are added to the `environments-curated` bucket via `scripts/preprocess_gdrive_videos.py`. This script creates one directory per video, and extracts the camera metadata (GPS, IMU) that video editing removes. All requires `ffmpeg` + `exiftool`.

The processing pipeline is illustrated below:

```sh
python scripts/preprocess_gdrive_videos.py --dry-run   # ../gdrive-src -> ../environments-curated
python scripts/preprocess_gdrive_videos.py
python scripts/preprocess_gdrive_videos.py --only GH010234   # one clip

./scripts/push_curated.sh --dry-run
./scripts/push_curated.sh
```

## Evaluation

During design of the package modules, we used standard 3D reconstruction benchmarks to understand the benefits of given pipeline configuraitons. Ground-truth datasets can be benchmarked with `evals/eval.py`:

```bash
python -m evals.eval --config evals/configs/7scenes.yaml --dry-run
```

Results land in `evals/results/`; datasets and grid configs are described in [evals/README.md](../../evals/README.md).
