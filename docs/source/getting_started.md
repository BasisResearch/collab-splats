# Getting Started

## Installation

collab-splats requires Python 3.10 and CUDA. Use the provided setup script:

```bash
git clone https://github.com/BasisResearch/collab-splats
cd collab-splats
bash setup.sh
```

This installs the package in development mode along with gsplat and all CUDA dependencies.

### Docker

A pre-built Docker image is available:

```bash
docker pull tommybotch/collab-splats:latest
```

## Quickstart

### Reconstruct a video

```bash
python docs/examples/run_pipeline.py data/tutorial/<video.mp4> --config configs/base.yaml
```

Stages run in order: frame sampling → feedforward pointcloud → (optional) splats, mesh,
semantics, localization. Enable Gaussian-splat training with `splats.enabled: true` in the
config; outputs land in `<output>/<backend>/splats/`.

See the [Tutorials](tutorials/index) for full worked examples.
