# PR1 — Core Modules

**Branch:** `refactor/core-modules` → `main`

## Title
refactor: consolidate core modules — semantics, nerfstudio, utils, pointcloud

## Body
Consolidates collab_splats into a clean module layout for Phase 1.

- Moves `collab_splats/semantics/` — features (MaskCLIP, DINO, Talk2DINO), segmentation, protocols, frame_sampling
- Moves `collab_splats/nerfstudio/` — models, method_configs, datamanagers (from top-level)
- Adds `collab_splats/utils/` — camera_utils, frame_sampling (general preprocessing, not semantics-specific)
- Adds `collab_splats/pointcloud/` — PointcloudResult, BasePointcloudCreator, NerfstudioSfmCreator, MapAnythingCreator, registry
- Adds batching protocol: `forward_batch`/`reshape_batch` on BaseFeatureExtractor + all extractors
- Adds `infer_batch_size()` VRAM-aware batch size utility
- Bug fix: `maskclip_onnx` bare import guard
- Bug fix: `pytorch_gc()` CPU crash guard
- All backwards-compat re-exports in place

Tests: 25 pass, 1 skip. Pre-existing env failures: 5 nerfstudio import failures + 1 GPU smoke test (not caused by this PR).
