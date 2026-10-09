# collab-splats — Documentation

User-facing documentation for the repository.

Layout mirrors the code tree: a module at `<source-path>/<module>/` has its docs at `docs/<module>/`. The `collab_splats/` prefix is dropped; top-level dirs like `evals/` are preserved.

## Running the pipeline

- `reconstruct local VIDEO|DIR... --output-root R` — local videos or frame directories, one scene per input
- `reconstruct remote [SCENE...|--all] --output-root R` — same pipeline over `environments-curated` GCS scenes, pushing to `environments-processed/`
- `reconstruct local <input> --output-root R --config <scene>/run_config.yaml` — re-run a saved config

Flags, output layout, and the processed-scene contract: [../configs/README.md](../configs/README.md).

## Module docs

- [pointcloud.md](pointcloud.md) — depth alignment, bundle adjustment, backend config blocks
- [mesh.md](mesh.md) — TSDF fusion, cleaning, texturing, vertex features

Splats docs live in the API reference (`docs/source/api/splats.rst`); loop-closure parity vs
VGGT-SLAM is decision 027. Tutorials and the API reference build with `make docs` from
`docs/source/`.
