# 023 — Store mesh-vertex semantics in the lifted store

Date: 2026-10-07 · Status: accepted · Branch: `feat/scene-viewer`

## Context

- semantics-storage (option B, `c2c3a7df`) stores codes only: no vertex store, no word arrays;
  the viewer derived everything at launch
- measured launch, ocr_lens, GH010229 (294 frames, 545k vertices): 64 s
  - decoder load 28 s, decode codes → top-64 words 16-24 s, lift words onto vertices 9-12 s
- text mode (maskclip, talk2dino) would lift codes onto vertices at launch too
- measured cost of storing instead (file bytes, latent 128): vertex words +100 MB
  (ids int16 42 MB, probs fp16 58 MB); vertex codes +126 MB; 97% of vertices observed, so zeros
  barely compress

## Decision

- the semantics stage lifts onto `mesh.ply` vertices too, when the mesh exists, and writes the
  result into the same `<backend>/semantics/<extractor>_lifted.zarr`
- each extractor stores only what its viewer mode reads:
  - ocr_lens: `vertex_word_ids` (V, 64) int16 + `vertex_word_probs` (V, 64) fp16, vocab in attrs
  - queryable extractors (maskclip, talk2dino): `vertex_features` (V, latent) fp16 codes
  - others (dinov2): no vertex arrays
- semantics runs after mesh; the lifted store records `mesh_sha256` of the `mesh.ply` it lifted onto
- a store whose hash differs from the current `mesh.ply` is stale: the stage re-runs (no
  re-extraction, codes reused) and the viewer skips it
- stale stores are never deleted: push is `rclone copy` with a one-way check, so a local delete
  leaves the old store in the processed bucket and the next leaf re-run pulls it back
- lifting stays as decision 021 says: straight from the 2D codes onto vertices,
  `pixel_indices=None`; words decode then lift, codes lift then decode

## Consequences

- viewer launch reads arrays; no decoder, no lift, no `store_rows` in `viewer.py`
- lifted store roughly doubles per extractor (~+100 MB on GH010229)
- after a mesh rebuild only the configured extractor re-lifts; other extractors' stores stay
  stale (viewer skips them) until their own semantics run
- semantics stage time grows for ocr_lens by the decoder load + decode + vertex lift (~50 s)
- reverses option B's "no vertex store, no word arrays" for vertices only; points stay codes only
- a mesh from `mesh.source: splats` is lifted with `pointcloud.zarr` poses; pose-optimized splats
  shift slightly from them (accepted)
