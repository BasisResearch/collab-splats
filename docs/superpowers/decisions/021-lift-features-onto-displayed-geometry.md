# 021 — Lift features onto the geometry being displayed

Date: 2026-09-30 · Status: accepted · Branch: `feat/ocr-lens`

## Context

- 2026-06-04 (`specs/2026-06-04-mesh-feature-transfer-design.md`): mesh vertices got features
  second-hand — points lifted from the 2D cache, then k-NN (`transfer_features`) onto vertices
- that spec left one exit: revisit "if KNN transfer quality proves insufficient"
- OCR-lens label check on GH010229 (vggt_omega): mesh top-1 agrees with a multiview
  image→surface reference on 61-62% of vertices; only 52-76% of vertices get a label
- averaging compressed codes then decoding is not averaging word probabilities: a word a few
  close views see (hole) drowns in far views that see soil/tree
- lens patch footprint on the surface: p10 0.005, median 0.009, p90 0.033 world units; mesh
  edge median 0.0035; texel (8192 atlas) 0.0005 — vertices already resolve what the views can

## Decision

- features are lifted from the 2D cache straight onto whatever is displayed — pointcloud points
  or mesh vertices — with `lift_features`; no k-NN hop from points to vertices for display
- `lift_features` takes `result.pixel_indices=None`: points no view sees stay zero (unobserved);
  mesh vertices have no source pixel, so they use this
- unobserved stays unobserved: no inpainting; the viewer greys it
- `transfer_features` stays for smoothing features over neighbors in place
- no per-texel features: texels are 5-10x finer than any view resolves

## Consequences

- the viewer lifts per-frame word probabilities onto `mesh.ply` vertices itself; the
  pointcloud-level `<extractor>_lifted.zarr` is not its input
- invented mesh surface (hole fill, lids) mostly fails the depth test and shows unobserved
- `dashboard/viewer.py` still reads `vertex_features.npy`, which nothing writes (stale since
  mesh cleanup, 2026-09-06); dashboard refactor pending
- follow-up: labels over the photo-textured mesh — lift onto `texture/mesh.obj` vertices the
  same way and tint over the albedo; still per vertex, not per texel
