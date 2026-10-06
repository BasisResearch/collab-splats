# Texture CUDA guard

Date: 2026-10-06 · Status: approved (option 1 of the CPU-fallback audit)

## Problem

Audit of the mesh stage on a CPU-only machine (`CUDA_VISIBLE_DEVICES=""`):

- TSDF fusion (`mesh/tsdf.py`) picks `CUDA:0` or `CPU:0` from `o3c.cuda.is_available()` — runs on CPU
- clean / repair / prepare (`mesh/clean.py`, `mesh/utils.py`) are CPU-only code — run on CPU
- depth upsample (`get_device()`) and sky mask (ONNX provider filter) fall back to CPU
- texture (`mesh/texture.py`, `mesh.texture: true`, default off) hard-codes `.cuda()` and
  `dr.RasterizeCudaContext()`; nvdiffrast has no CPU rasterizer
- result: `tests/mesh` 48 passed / 10 failed, every failure in `test_texture.py`
  with `RuntimeError: No CUDA GPUs are available`, raised deep inside the bake

On a CPU-only run with `texture: true` the stage fuses and cleans for minutes, then dies
with a torch error that does not name the setting.

## Design

- `create_texture_mesh` raises `RuntimeError` first thing when `torch.cuda.is_available()`
  is false; the message names nvdiffrast and `mesh.texture: false`
- `Reconstructor.mesh` runs the same check at stage start, before any fusion, so the
  config error costs nothing
- RuntimeError, not ValueError: the arguments are fine, the hardware is missing
- no CPU texture path: rewriting the rasterizer for CPU is out of scope

## Tests

- `tests/mesh/test_texture.py`: a `cuda` skipif marker (the `tests/splats/` convention) on
  the ten GPU tests; a new CPU test monkeypatches `torch.cuda.is_available` to False and
  asserts `create_texture_mesh` raises before touching the views
- `tests/reconstructor/test_reconstructor.py`: `texture: true` with no CUDA raises before
  `create_tsdf_mesh` is called
- gate: `tests/mesh` and the mesh-stage reconstructor tests green both with and without
  `CUDA_VISIBLE_DEVICES=""`

## Docs

- `configs/base.yaml` texture comment and both docstrings' `Raises:` say CUDA-only

## Found while gating (not fixed here)

- `collab_splats.reconstructor` and `collab_splats.pointcloud` do not import on a CPU-only
  machine: VGGT-X (`Linketic/VGGT-X` @ `26d1b956`, `vggt/layers/mlp.py:33`) runs
  `warmup_gelu_fused()` at import, which allocates on `device="cuda"`
- so `tests/reconstructor/test_reconstructor.py` cannot be collected with
  `CUDA_VISIBLE_DEVICES=""`; the new reconstructor test is verified on the GPU run only
- `collab_splats.mesh` imports and runs on CPU

## Follow-up decision (2026-10-06): the pipeline requires a GPU

- options weighed: fork VGGT-X to make the warmup lazy (rejected: no fork), allow
  COLMAP-only on CPU (rejected: lazy vggt imports in vggtx, BA and the LC wrapper, and the
  sfm path still needs VDA depth, a ViT-L on CPU), require a GPU (chosen)
- README system requirements already say a GPU is needed at runtime
- `collab_splats/pointcloud/__init__.py` raises `ImportError("collab_splats.pointcloud needs
  a CUDA GPU; ...")` before the feedforward imports; it sits on the import path of
  `reconstructor`, the CLI and `evals`
- `collab_splats.mesh` stays importable on CPU, untested beyond import + tests/mesh
- `setup.sh`'s GPU-less build-stage check catches `Exception`, so it still warns, not fails
