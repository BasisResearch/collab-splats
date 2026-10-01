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
reconstruct local data/tutorial/<video.mp4> --output-root data/outputs
```

Stages run in order: `preproc` → `pointcloud` → `refine` → `semantics` → `splats` → `mesh` →
`localize` → `reconstruction_quality_report`. `preproc`, `pointcloud` and
`reconstruction_quality_report` always run; the rest are enabled in the config. Enable Gaussian-splat training with `splats.enabled: true` in the
config; outputs land in `<output-root>/<video-stem>/<backend>/splats/`.

See the [Tutorials](tutorials/index) for full worked examples.
