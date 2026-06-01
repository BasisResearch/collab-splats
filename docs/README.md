# collab-splats — Documentation

User-facing documentation for the repository.

Layout mirrors the code tree: a module at `<source-path>/<module>/` has its docs at `docs/<module>/`. The `collab_splats/` prefix is dropped; top-level dirs like `evals/` are preserved.

## Examples

- [examples/reconstruct.py](examples/reconstruct.py) — CLI entry point for the Reconstructor pipeline
- [examples/run_all_datasets.sh](examples/run_all_datasets.sh) — batch reconstruction across all dataset configs

## Module Notebooks

- [pointcloud/](pointcloud/) — pointcloud + bundle adjustment + ground-truth evals
- [semantics/](semantics/) — feature extraction, MaskCLIP, Talk2DINO

## Internal

- [known-test-failures.md](known-test-failures.md) — known test failures and workarounds
