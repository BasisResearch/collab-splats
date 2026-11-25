"""
Low-level helpers shared across collab_splats; import the submodule you need.

- io: image decode, float-to-uint8, JSON reports, zarr codec and validity (torch-free)
- colmap: binary model write (stem names, atomic swap) and read (torch-free)
- torch_utils: device, GC, batching, registry
- image, notebook, progress, visualization: older helpers, imported by path
- no re-exports: a package-level import would pull torch into every torch-free caller
"""
