# collab-splats — Documentation

User-facing documentation for the repository.

Layout mirrors the code tree: a module at `<source-path>/<module>/` has its docs at `docs/<module>/`. The `collab_splats/` prefix is dropped; top-level dirs like `evals/` are preserved.

## Running the pipeline

- `reconstruct local VIDEO|DIR... --output-root R` — local videos or frame directories, one scene per input
- `reconstruct remote [SCENE...|--all] --output-root R` — same pipeline over `environments-curated` GCS scenes, pushing to `environments-processed/`
- `reconstruct local <input> --output-root R --config <scene>/run_config.yaml` — re-run a saved config

Flags, output layout, and the processed-scene contract: [../configs/README.md](../configs/README.md).

## Module docs

Module docs live in the API reference, prose above each autodoc block (`docs/source/api/<package>.rst`):

- [pointcloud.rst](source/api/pointcloud.rst) — depth alignment, bundle adjustment, backend config blocks
- [mesh.rst](source/api/mesh.rst) — TSDF fusion, cleaning, texturing, vertex features
- [splats.rst](source/api/splats.rst) — gsplat training

Tutorials and the API reference build with `make docs` from
`docs/source/`.
