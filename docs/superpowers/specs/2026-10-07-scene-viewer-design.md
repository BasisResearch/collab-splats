# Scene viewer — one viewer, optional semantics, stored vertex semantics

Date: 2026-10-07 · Status: rewritten for codes-only storage + decision 023 · Branch: `feat/scene-viewer`
(rebased onto `clean/final` `c2c3a7df`)

## Problem

- The only mesh query viewer is `docs/examples/ocr_lens_viewer.py`, hard-wired to `ocr_lens`.
- Since semantics-storage (option B) it rebuilds vertex words at every launch: 64 s on GH010229
  (decoder load 28 s, decode 16-24 s, vertex lift 9-12 s).
- Text-queryable extractors (maskclip, talk2dino) have no mesh query path.
- No viewer shows a mesh without semantics.

## Decisions (from brainstorm)

- **One viewer, in `collab_splats/viewer.py`.** `python -m collab_splats.viewer <scene>/<backend>`.
  `docs/examples/ocr_lens_viewer.py` is deleted.
- **Semantics are optional.** No lifted store with vertex arrays: the mesh shows, no semantics GUI.
- **No extractor flag.** A `Semantics` dropdown lists the extractors whose lifted store has vertex
  arrays; absent when there are none.
- **Default selection:** `ocr_lens` when listed, else the first listed extractor by name; `none`
  stays in the dropdown to clear the heat.
- **Vertex semantics are stored, not derived** ([decision 023](../decisions/023-store-vertex-semantics.md)):
  the semantics stage writes them into the existing lifted store; the viewer only reads.
- **Mode follows the store.**

| extractor | stored on vertices | query | label list | click probe |
|---|---|---|---|---|
| `none` selected | — | — | — | — |
| `ocr_lens` | `vertex_word_ids`, `vertex_word_probs` | summed word probabilities as heat | yes, mass-ranked | yes, top-10 words |
| queryable (maskclip, talk2dino) | `vertex_features` (codes) | `score_queries` on decoded codes as heat | no | no |
| other (dinov2) | nothing | not listed | — | — |

- queryable means `issubclass(BaseFeatureExtractor.get(name), BaseQueryableExtractor)`
- switching the dropdown clears heat and probe, removes the previous mode's GUI, builds the new one

## Storage side (semantics stage)

### Lifted store layout

`<backend>/semantics/<extractor>_lifted.zarr`, one store per extractor, written atomically as today:

| entry | shape / dtype | when |
|---|---|---|
| `features` | (P, latent) fp16 point codes | always (unchanged) |
| `autoencoder.pt` | | compressed (unchanged) |
| `vertex_word_ids` | (V, 64) int16 | ocr_lens, `mesh.ply` exists |
| `vertex_word_probs` | (V, 64) fp16 | ocr_lens, `mesh.ply` exists |
| `vertex_features` | (V, latent) fp16 codes | queryable extractor, `mesh.ply` exists |
| attrs | `input_dim`, `latent_dim` (unchanged); new `extractor`, `extractor_kwargs`, `mesh_sha256` (vertex arrays only); `words` (ocr_lens vocabulary) | |

- measured on GH010229 (latent 128): words +100 MB, codes +126 MB; store 128 → ~230-255 MB
- unobserved vertices (no view passes the depth test) are zero rows; ocr_lens: `vertex_word_probs[:, 0] == 0`

### Stage order and staleness

- `STAGES`: `semantics` moves after `mesh` (dependencies unchanged: `pointcloud`); the mesh stays optional
  - `LEAF_STAGES` unchanged: semantics and mesh both stay leaves
- the lifted store records `mesh_sha256`, the sha256 of the `mesh.ply` its vertex arrays were lifted
  onto (29 MB on GH010229, well under 1 s)
- `done("semantics")`: the lifted store exists and either there is no `mesh.ply`, the store records
  no hash (points only), or its hash matches; a mismatch re-runs the stage from the scene's
  `_codes.zarr` (no re-extraction)
- the viewer applies the same hash check and skips a stale store (warning: re-run semantics with
  that extractor)
- stale stores are never deleted (decision 023): push is `rclone copy` with a one-way check, so a
  local delete leaves the old store in the processed bucket and a leaf re-run pulls it back
- only the configured extractor re-lifts; others stay stale until their own run
- remote leaf re-runs pull every processed member, `mesh.ply` included, so the vertex lift runs remotely too

### Stage flow (`Reconstructor.semantics`), after today's step 6 (lift codes onto points)

7. `mesh.ply` exists and the extractor is ocr_lens: load the processor, vocabulary and decoder,
   `model_id = extractor_kwargs.get("model_id", <OCRLensExtractor's default>)` passed to both;
   inline in `semantics()`, no new function:
   - `vertex_cloud = dataclasses.replace(pointcloud, points=vertices, colors=..., pixel_indices=None)`:
     same cameras, points swapped for vertices; `colors` a placeholder (required field); no source pixel
   - decode: per frame `i`, existing `_load_frame(codes, rows, None, i)` (rows from the existing
     `store_rows` call) → existing `word_probabilities` → `topk(64)` per patch, an indexed (ids, probs)
     map each (64, H_p, W_p)
   - lift: one `lift_features(maps.__getitem__, vertex_cloud, num_classes=n_words)`, then `topk(64)`
     per vertex → (V, 64) ids + probs
   - no chunking: measured 2.4 s, 9.1 GiB peak at 545k vertices; memory grows ~17 KB per vertex
     (A40 ceiling ~2.4M vertices); a bigger mesh fails with CUDA OOM, not bad data
   - memory: a cache miss already frees the extractor (`del extractor; pytorch_gc()`) before the
     lift, so the decoder loads alone
8. `mesh.ply` exists and the extractor is queryable: one
   `lift_features(partial(_load_frame, codes, rows, None), vertex_cloud)` (`vertex_cloud` as in step 7),
   exactly as the points lift
   - no chunking: a (V, latent) accumulator is ~280 MB at 545k × 128; the points lift already runs
     500k in one call
9. write points + vertex arrays + `mesh_sha256` together with `write_point_features`

- words decode then lift; codes lift then decode at read (decision 021, storage spec measurements)

### Code moves

| from | to | why |
|---|---|---|
| example `_top_words` + `_lift_top_words` | inline in `Reconstructor.semantics()` (step 7) | no new function; reuses `word_probabilities` and `lift_features` |
| — | `semantics/store.py` `write_point_features(..., vertex_arrays=None, attrs=None)` | vertex arrays and attrs in the same atomic write |
| — | `semantics/store.py` `read_point_features(store_path, name="features")` | decodes `vertex_features` too |

- no new function in `lifting.py` or `ocr_lens.py`

## Viewer side (`collab_splats/viewer.py`)

- `Viewer` class unchanged, except:
  - click dispatcher registered once (flag), so mode switches never stack callbacks
  - `show_heat`: faces touching a NaN vertex are not drawn (no heat blended onto unobserved vertices)
  - `show_heat`: lift direction (whole-mesh normals × median edge) cached per mesh in `heat_shifts`; `add_mesh` drops it
- `main()`: CLI `backend_dir`, `--port`, `--textured`, `--texture_size`; builds a `Viewer`, calls
  `_build`, then `serve_forever()`
- `_build(viewer, backend_dir, textured, texture_size)`: mesh (`mesh.ply` + optional
  `texture/mesh.obj`, inline), dropdown, default selection, switching
- `_split(text)`: comma-separated query words; shared by both modes' query boxes
- `_find_stores(backend_dir) -> dict[str, Path]`: `<backend>/semantics/*_lifted.zarr`
  holding `vertex_word_ids` or `vertex_features`; name from attrs `extractor`; `mesh_sha256` mismatch
  skipped with a warning
- `_word_mode(viewer, store)`:
  - reads `vertex_word_ids`, `vertex_word_probs`, attrs `words`
  - query box (comma-separated words), `Query min p` floor, Search; unknown words noted
  - heat: per vertex, sum of `word_probs` where `word_ids` is a query word; unobserved NaN
  - label list: `np.bincount(word_ids, weights=word_probs, minlength=n_words)`, click = query
  - probe: clicked vertex's top-10 scene terms (`_chart`); unobserved vertices probe as unobserved;
    mesh colors unchanged
- `_text_mode(viewer, store)`:
  - `read_point_features(store, name="vertex_features")` → unit (V, D) float32; observed = raw codes'
    rows not all zero (decoded zero codes are not zero)
  - `Query`, `Negatives` (default `object`), `Query min score`, Search
  - extractor built on first Search: `BaseFeatureExtractor.get(attrs["extractor"])(**attrs["extractor_kwargs"])`; decoded features moved to its device once
  - heat: `score_queries` → (V,) in [0, 1]; unobserved set to NaN; `show_heat`
- switching away: `show_heat("mesh", None, 0.0)`; remove the mode's GUI handles; word mode also pops
  `label_lists["mesh"]`, `mesh_clicks["mesh"]` and removes `/probe`; `pytorch_gc()`
- import rule: `viewer.py` never imports `collab_splats.reconstructor`; it needs no decoder, no lift,
  no `store_rows`

## Testing

- storage:
  - `write_point_features` with vertex arrays: all arrays + attrs present, fp16/int16 dtypes, atomic
  - `read_point_features(name="vertex_features")`: float32 unit rows
  - stage: toy scene with `mesh.ply` → vertex arrays + `mesh_sha256` written; without → `features` only
  - stage, ocr_lens with a stub decoder: vertex ids + probs descending, unseen vertices zero
  - stage order: semantics after mesh; update `tests/reconstructor/test_reconstructor.py:923-924`
    (asserts the old order)
  - `done("semantics")`: false after `mesh.ply` changes, true when unchanged, no mesh, or no hash
- viewer (`tests/test_viewer.py`, mocked server):
  - `_find_stores`: no `semantics/` → empty; word store and queryable store listed; points-only store
    not listed; `mesh_sha256` mismatch skipped
  - `_build`: no stores → mesh only, no dropdown; ocr_lens present → default selected
  - word mode on a toy store: heat equals summed probs; label list by mass; probe charts top words
  - text mode with a stub queryable extractor: heat equals masked `score_queries`; built once
  - switch: heat cleared, GUI removed, label list and click handler gone
  - import check: `collab_splats.viewer` does not import `collab_splats.reconstructor`

Manual: GH010229 — re-run semantics (ocr_lens, then maskclip) after mesh; viewer launches in seconds;
switch modes, query both; no-semantics backend shows the mesh only.

## Files

| file | change |
|---|---|
| `collab_splats/reconstructor.py` | stage order; semantics steps 7-9; `done("semantics")` hash check |
| `collab_splats/semantics/store.py` | vertex arrays + attrs in `write_point_features`; `name` in `read_point_features` |
| `collab_splats/viewer.py` | `main`, `_build`, `_find_stores`, `_split`, `_word_mode`, `_text_mode`, `_chart` |
| `docs/examples/ocr_lens_viewer.py` | deleted |
| `docs/mesh.md`, `docs/semantics.md`, `configs/README.md` | viewer pointer, lifted store layout, stage order |
| `docs/superpowers/decisions/023-store-vertex-semantics.md` | new |
| tests | above |

## Out of scope

- word arrays on points (only vertices)
- re-lifting every extractor after a mesh rebuild (each re-runs on its own semantics run)
- deleting stale stores (local or remote)
- per-texel semantics
- dashboard (its imports are broken pending its own cleanup)
