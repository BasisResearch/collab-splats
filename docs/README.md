# collab-splats — Documentation

User-facing documentation for the repository. For internal progress tracking, see [`/worklog`](../worklog/).

Layout mirrors the code tree: a module at `<source-path>/<module>/` has its docs at `docs/<module>/`. The `collab_splats/` prefix is dropped (the main package is implied); top-level dirs like `evals/` are preserved.

## Module Docs

- [semantics.md](semantics.md) — semantic feature splatting (top-level write-up)
- [nerfstudio/README.md](nerfstudio/README.md) — NerfStudio extension layer (mirrors `collab_splats/nerfstudio/`)
- [evals/baselines/vggt_slam/README.md](evals/baselines/vggt_slam/README.md) — VGGT-SLAM baseline setup (mirrors `evals/baselines/vggt_slam/`)

## Module Notebooks

- [pointcloud/](pointcloud/) — pointcloud + bundle adjustment + ground-truth evals
- [semantics/](semantics/) — feature extraction, MaskCLIP, Talk2DINO
- [splats/](splats/) — meshing, comparison, query
- [data/](data/) — data format references
