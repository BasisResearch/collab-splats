# collab-splats — Documentation

User-facing documentation for the repository.

## Running the pipeline

- `reconstruct local VIDEO|DIR... --output-root R` — local videos or frame directories, one scene per input
- `reconstruct remote [SCENE...|--all] --output-root R` — same pipeline over `environments-curated` GCS scenes, pushing to `environments-processed/`
- `reconstruct local <input> --output-root R --config <scene>/<backend>/run_config.yaml` — re-run a saved config

Flags and config: [../configs/README.md](../configs/README.md). Output layout: [../README.md#output-structure](../README.md#output-structure).

## Module docs

Module docs live in the API reference, prose above each autodoc block (`docs/source/api/<package>.rst`):

- [reconstructor.rst](source/api/reconstructor.rst) — config-driven stage pipeline, `reconstruct` CLI
- [preproc.rst](source/api/preproc.rst) — keyframe store, video decode, quality report, frame sampling
- [semantics.rst](source/api/semantics.rst) — feature extraction, lifting, compression, segmentation
- [pointcloud.rst](source/api/pointcloud.rst) — feedforward and SfM backends, depth alignment
- [localization.rst](source/api/localization.rst) — query image to camera pose: retrieval, matching, PnP
- [geometry.rst](source/api/geometry.rst) — transforms, bundle adjustment, loop closure, quality metrics
- [mesh.rst](source/api/mesh.rst) — TSDF fusion, cleaning, texturing, vertex features
- [splats.rst](source/api/splats.rst) — gsplat training
- [utils.rst](source/api/utils.rst) — shared image, IO, COLMAP and torch helpers

Tutorials and the API reference build with `make docs` from the repo root.
