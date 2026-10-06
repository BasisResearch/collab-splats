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
- resolved by the follow-up below
- `collab_splats.mesh` imports and runs on CPU

## Follow-up decision (2026-10-06): guard VGGT-X only

- options weighed: fork VGGT-X to make the warmup lazy (rejected: no fork); a package-wide
  `ImportError` in `pointcloud/__init__.py` (tried in `210547f3`, reverted: it also blocked
  sfm, depth, mesh via the CLI and localization, none of which crash on CPU); guard the
  VGGT-X backend only (chosen)
- measured: with `vggt.models.vggt` stubbed, `pointcloud`, `sfm`, `depth`, `reconstructor`,
  the CLI, `evals.eval` and `localization` all import on CPU; `vggt.utils.geometry`,
  `vggt.dependency.track_predict`, `vggt_omega.models` and `mapanything.models` import on CPU
- `VGGTXCreator._load_model` raises `RuntimeError("the vggtx backend needs a CUDA GPU; ...")`
  and only then imports `vggt.models.vggt` (the CLAUDE.md heavy-dependency exception; LoGeR
  defers its vendored import the same way)
- GPU-only remains: vggtx, splats (gsplat), mesh texture (nvdiffrast); other backends and
  stages import on CPU but are not tested end to end there
- CPU-only gate also found `preproc.undistort.calibrate_camera` crashing: the pycolmap CUDA
  wheel's default device errors with no GPU; it now passes `device=` chosen as in `sift_db.py`
