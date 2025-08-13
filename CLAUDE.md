# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

This repository requires specific CUDA dependencies and nerfstudio integration. Use the provided setup script:

```bash
# Install the package in development mode
bash setup.sh
```

For Docker-based development:
```bash
# Uses docker image: tommybotch/collab-splats:latest
git clone https://github.com/BasisResearch/collab-splats/
cd collab-splats
bash setup.sh
```

## Architecture Overview

**collab-splats** is a nerfstudio extension that enables depth/normal derivation and meshing for Gaussian Splatting models. The codebase is structured around two main architectural patterns:

### Core Components

1. **Models** (`collab_splats/models/`):
   - `rade_gs_model.py`: Baseline depth/normal-enabled Gaussian splatting built on gsplat-rade
   - `rade_features_model.py`: Extended version supporting ANN feature space splatting

2. **Wrapper Interface** (`collab_splats/wrapper/splatter.py`):
   - `Splatter` class: High-level interface for preprocessing, training, and visualization
   - `SplatterConfig`: Configuration system for different splatting workflows
   - Supports methods: `splatfacto`, `feature-splatting`, `rade-gs`, `rade-features`

3. **Data Management** (`collab_splats/datamanagers/`):
   - `features_datamanager.py`: Handles feature-based data loading and processing

4. **Utilities** (`collab_splats/utils/`):
   - `mesh.py`: Post-processing meshing functionality
   - `segmentation.py` + `grouping.py`: Gaussian grouping and segmentation tools
   - `visualization.py`: PyVista-based 3D visualization
   - `camera_utils.py`: COLMAP camera integration

### NerfStudio Integration

The package registers two method configs with nerfstudio:
- `rade-gs`: Entry point in `collab_splats.configs.rade_gs_method:rade_gs_method`
- `rade-features`: Entry point in `collab_splats.configs.rade_features_method:rade_features_method`

### Dependencies

Key external dependencies:
- **gsplat-rade**: Custom CUDA kernels for depth/normal rasterization
- **meshlib**: 3D mesh processing (pinned to v3.0.6.229)
- **mobile_sam**: Segmentation backend
- **nerfstudio**: Base framework integration

## Development Commands

### Code Formatting
```bash
# Format code with black (line length: 120)
black .

# Sort imports
isort .
```

### Testing
Tests are primarily notebook-based in `tests/` directory:
- `test_rade_gs.ipynb`: Model testing
- `test_grouping.ipynb`: Gaussian grouping functionality
- `test_meshing.ipynb`: Mesh generation testing

Run Python tests directly:
```bash
python tests/test_grouping.py
```

### Example Workflows
Key examples in `examples/` directory:
- `derive_splats.ipynb`: Basic splatting pipeline
- `create_mesh.ipynb`: Mesh generation from splats
- `visualization.ipynb`: 3D visualization workflows
- `run_pipeline.py`: Batch processing script

## Key Configuration Patterns

The `SplatterConfig` TypedDict defines the main configuration interface:
- `file_path`: Input data path (video, images, etc.)
- `method`: Processing method selection
- `output_path`: Optional output directory (defaults to input parent)
- `frame_proportion`: Video frame sampling rate
- `overwrite`: Force reprocessing flag

## Troubleshooting

For visualization issues (plots not showing), check VSCode port forwarding settings as noted in the README.