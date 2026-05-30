# Documentation Site Design

**Date:** 2026-05-20  
**Status:** Approved

## Goal

Public-facing documentation site with two sections: browsable tutorials (Jupyter notebooks) and API reference (autodoc). Hosted on GitHub Pages. Flexible structure so notebooks can be reorganized without moving source files.

## Toolchain

- **Sphinx** — build engine
- **pydata-sphinx-theme** — modern theme (used by numpy/pandas/scikit-learn), wide content area, good notebook output rendering
- **nbsphinx** — notebook → HTML conversion, `nbsphinx_execute = "never"` (use committed outputs)
- **myst-parser** — Markdown support for prose pages
- **sphinx.ext.autodoc + napoleon** — API reference from docstrings

All deps except `pydata-sphinx-theme` already in `pyproject.toml [docs]`. `sphinx_rtd_theme` replaced by `pydata-sphinx-theme`.

## Directory Layout

```
docs/source/           ← new Sphinx root
  conf.py
  index.rst
  getting_started.md
  tutorials/
    index.rst
    splats/            ← symlinks to docs/splats/*.ipynb
    semantics/         ← symlinks to docs/semantics/*.ipynb
    pointcloud/        ← symlinks to docs/pointcloud/*.ipynb
    data/              ← symlinks to docs/data/*.ipynb
  api/
    index.rst
    wrapper.rst
    semantics.rst
    pointcloud.rst
    mesh.rst
```

Notebooks stay in their current locations (`docs/splats/`, `docs/semantics/`, etc.). Symlinks in `docs/source/tutorials/` point to them. Reorganizing = moving symlinks only.

## Navigation Structure

```
Getting Started
Tutorials
  ├── Splats
  │   ├── Derive Splats
  │   ├── Create Mesh
  │   ├── Visualization
  │   └── Compare MaskCLIP / Talk2DINO
  ├── Semantics
  │   ├── Feature Extraction
  │   ├── Segmentation
  │   └── MaskCLIP Reference Comparison
  ├── Point Cloud
  │   ├── Bundle Adjustment
  │   ├── Feedforward Exploration
  │   ├── Feedforward Mesh
  │   ├── Loop Closure Eval
  │   └── Ground Truth Evals
  └── Data
      └── GCloud Data Interface
API Reference
  ├── Wrapper
  ├── Semantics
  ├── Point Cloud
  └── Mesh
```

## API Reference Scope

| Section | Modules |
|---|---|
| Wrapper | `collab_splats/wrapper/splatter.py`, `config.py` |
| Semantics | `collab_splats/semantics/features.py`, `retrieval.py`, `segmentation.py` |
| Point Cloud | `collab_splats/pointcloud/base.py`, `bundle_adjustment.py`, `sfm.py`, `wrappers.py`, `feedforward/`, `loop_closure/` |
| Mesh | `collab_splats/mesh/base.py`, `poisson.py`, `tsdf.py`, `utils.py` |

`nerfstudio/` and `dashboard/` excluded — internal/integration code, not public API.

Each `.rst` uses `automodule` with `:members:` and `:undoc-members: False` so sparse docstrings don't pollute the reference.

## Build & CI/CD

**Local:**
```bash
make docs        # sphinx-build docs/source docs/_build/html
make docs-serve  # serve locally for preview
```

**GitHub Actions** (`.github/workflows/docs.yml`):
- Trigger: push to `main` → build + deploy; PRs → build only (no deploy)
- Install `.[docs]`, run `sphinx-build`, deploy to `gh-pages` branch via `peaceiris/actions-gh-pages`
- No GPU required — `nbsphinx_execute = "never"` uses committed notebook outputs

## Deployment

- Branch: `gh-pages` (auto-managed by Actions)
- URL: `https://basisresearch.github.io/collab-splats/`
- Future migration to ReadTheDocs: swap Actions workflow + add `.readthedocs.yaml`; Sphinx config unchanged

## Out of Scope

- Notebook re-execution in CI (GPU not available)
- Versioned docs (future ReadTheDocs migration)
- `nerfstudio/` and `dashboard/` API docs
