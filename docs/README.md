# collab-splats — Documentation

User-facing documentation for the repository.

Layout mirrors the code tree: a module at `<source-path>/<module>/` has its docs at `docs/<module>/`. The `collab_splats/` prefix is dropped; top-level dirs like `evals/` are preserved.

## Examples

- [examples/run_pipeline.py](examples/run_pipeline.py) — main driver: local videos / directories of videos
- [examples/run_pipeline_remote.py](examples/run_pipeline_remote.py) — same pipeline over `environments-curated` GCS scenes, pushing to `environments-processed/`
- [examples/reconstruct.py](examples/reconstruct.py) — re-run a saved `run_config.yaml`

Driver flags, output layout, and the processed-scene contract: [../configs/README.md](../configs/README.md).

## Module Notebooks

- [pointcloud/](pointcloud/) — pointcloud + bundle adjustment + ground-truth evals
- [semantics/](semantics/) — feature extraction, MaskCLIP, Talk2DINO

## Internal

- [known-test-failures.md](known-test-failures.md) — known test failures and workarounds
