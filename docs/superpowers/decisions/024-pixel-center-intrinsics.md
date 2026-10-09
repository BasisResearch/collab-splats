# 024 — PointcloudResult stores pixel-center intrinsics

Date: 2026-10-08 · Status: accepted · Branch: `feat/rgbd-ba-cf`

> Correction: Open3D TSDF is pixel-corner, not center; see [026](026-open3d-tsdf-corner-convention.md).

## Context

- two conventions in the package: pixel-center (pixel i at coordinate i: VGGT, `projection.py`, Open3D, nvdiffrast texture) and pixel-corner (pixel i at i + 0.5: COLMAP, gsplat)
- `PointcloudResult` mixed them: feedforward model K was used as center, but `__post_init__` rescaled it to full res as corner; the sfm path stored COLMAP's corner K beside a center-unprojected cloud
- result: full-res K off by 0.5 · (scale − 1) px (~0.9 px on GH010229 omega, scale 2.79) for TSDF and texture; photometric BA sampled pixel corners
- GH010229 cross-view NCC, no photometric term: corner-K gap20 0.0816 < center 0.0844; center-on-center ties corner-on-corner (G 0.4761 vs 0.4767 gap1)

## Decision

- every K `PointcloudResult` stores (`intrinsics`, `model_intrinsics`) is pixel-center
- center K rescales via the corner formula: shift +0.5, `rescale_intrinsics`, shift −0.5 (`__post_init__`, BA pyramid)
- corner consumers convert at their entry, +0.5: `to_colmap`, the gsplat trainer
- sfm `align_depth` stores both Ks shifted −0.5 from COLMAP's
- photometric BA samples at integer pixel indices (`grid_sample` `align_corners=True`)

## Consequences

- zarrs written before this keep the old full-res K; a leaf-stage re-run on one uses it; full re-run fixes it (no legacy guard)
- out of scope, still corner: `preproc/undistort.py` (cv2 with COLMAP K, sub-pixel effect), `plot_reprojection`
- the localizer is pixel-center end to end (2026-10-09): seed K, caller K, returned K and query px; the crop map inverts `__post_init__`
