# Getting Started

## Installation

collab-splats requires Python 3.10 and CUDA. Use the provided setup script:

```bash
git clone https://github.com/BasisResearch/collab-splats
cd collab-splats
bash setup.sh
```

This installs the package in development mode along with nerfstudio and all CUDA dependencies.

### Docker

A pre-built Docker image is available:

```bash
docker pull tommybotch/collab-splats:latest
```

## Quickstart

### Gaussian Splatting with depth and normals

```python
from collab_splats.wrapper.splatter import Splatter, SplatterConfig

config = SplatterConfig(
    file_path="path/to/video.mp4",
    method="rade-gs",
    output_path="path/to/output",
)
splatter = Splatter(config)
splatter.preprocess()
splatter.train()
```

### Semantic feature splatting

```python
config = SplatterConfig(
    file_path="path/to/video.mp4",
    method="rade-features",
)
splatter = Splatter(config)
splatter.preprocess()
splatter.train()
```

See the [Tutorials](tutorials/index) for full worked examples.
