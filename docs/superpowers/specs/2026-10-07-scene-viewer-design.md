# Scene viewer — one viewer, optional semantics

Date: 2026-10-07 · Status: approved in brainstorm, self-reviewed · Depends on: semantics-storage

## Problem

- The only mesh query viewer is `docs/examples/ocr_lens_viewer.py`, hard-wired to `ocr_lens`
  (word vocabulary, decoder, `.npy` word cache).
- Text-queryable extractors (maskclip, talk2dino) have no mesh query path; the Panel dashboard
  queries points only.
- No viewer shows a mesh without semantics: the OCR script refuses a backend with no word store.

## Decisions (from brainstorm)

- **One viewer, in `collab_splats/viewer.py`.** A `main()` there; launched with
  `python -m collab_splats.viewer <scene>/<backend>`. `docs/examples/ocr_lens_viewer.py` is deleted.
- **Semantics are optional.** No semantic store: the mesh shows, no semantics GUI.
- **No extractor flag.** A `Semantics` dropdown lists the vertex stores found under
  `<backend>/semantics/`; it is absent when there are none. Initial value `none`.
- **Mode follows the store.**

| store | query | label list | click probe |
|---|---|---|---|
| none / `none` selected | — | — | — |
| word arrays (`ocr_lens`) | summed word probabilities as heat | yes, mass-ranked | yes, top-10 words |
| codes of a queryable extractor (maskclip, talk2dino) | `score_queries` on decoded codes as heat | no | no |

- Non-queryable extractors (dinov2: no text encoder) are not listed: queryable means
  `issubclass(BaseFeatureExtractor.get(name), BaseQueryableExtractor)`.
- Switching the dropdown clears the heat and probe, removes the previous mode's GUI, builds the new one.

## Dependency: semantics-storage, extended

Built on the vertex store that `semantics-storage` defines
(`<backend>/semantics/<extractor>_vertices.zarr`), with one extension that spec does not have:

- **semantics writes a vertex store for queryable extractors too when `mesh.ply` exists**, not only ocr_lens
  - queryable (maskclip, talk2dino): `features` (V, latent) fp16 codes + `autoencoder.pt`, attrs
    `latent_dim`, `input_dim` (what `read_point_features` reads) and `extractor_kwargs` (to rebuild
    the text encoder)
  - ocr_lens: the word arrays only, as semantics-storage writes them
  - dinov2 and other non-queryable extractors: no vertex store (nothing could read the codes)
- semantics-storage owns that change; this spec lands after it. Lifting at viewer start-up
  (the path semantics-storage removes for OCR) is rejected.

## Design

### Module layout (`collab_splats/viewer.py`)

- `Viewer` class unchanged (widget layer; the reconstructor's loop-closure display keeps using it).
- New, below it:
  - `main()`: CLI `backend_dir`, `--port`, `--textured`, `--texture_size` (the OCR script's flags
    minus `--model_id`, `--chunk`); builds a `Viewer`, calls `_build`, then `serve_forever()`
  - `_build(viewer, backend_dir, textured, texture_size)`: mesh, dropdown, mode switching; what the
    tests drive (no serve loop)
  - `_load_mesh(backend_dir, textured, texture_size)`: `mesh.ply` + optional `texture/mesh.obj`,
    moved from the OCR script
  - `_find_stores(backend_dir, n_vertices)`: `{extractor: path}` for `*_vertices.zarr` that hold
    word arrays or a queryable extractor's codes; a store whose row count differs from `mesh.ply`
    is skipped with a warning (re-run semantics)
  - `_word_mode(viewer, store)` and `_text_mode(viewer, store)`: build that mode's GUI and handlers,
    return the GUI handles to remove on switch
- Switching away from a mode, in `_build` (same module, no new `Viewer` methods):
  - `viewer.show_heat("mesh", None, 0.0)`; remove the mode's GUI handles
  - word mode only: pop and remove `viewer.label_lists["mesh"]`; pop `viewer.mesh_clicks["mesh"]`
    (the dispatcher only picks meshes in `mesh_clicks`); remove the `/probe` marker
- Import rule: `viewer.py` never imports `collab_splats.reconstructor` (it imports `viewer.py`);
  `store_rows` is not needed once words come from the vertex store.

### Word mode (ocr_lens)

Storage plan Task 7's rewrite of the OCR script (vertex store, no decoder), moved in unchanged:

- query box (comma-separated words), `Query min p` floor, Search; unknown words noted
- heat: per vertex, sum of `word_probs` where `word_ids` is a query word
- label list: `np.bincount(word_ids, weights=word_probs, minlength=n_words)`, click = query
- probe: clicked vertex's top-10 scene terms in the HTML panel; unobserved vertices grey

### Text mode (queryable extractors)

- query box: comma-separated positives; `Negatives` box, default `object`
  (`score_queries`'s Talk2DINO convention); floor slider `Query min score`; Search
- on entering the mode: `read_point_features(store)` → unit (V, D) float32 features (decodes with the
  store's `autoencoder.pt`); observed = the raw stored codes' rows that are not all zero
  - decoded zero codes are not zero, so the mask must come from the raw codes
  - GH010229: 545k × D float32, ~1.1 GB at D = 512
- on first Search: `BaseFeatureExtractor.get(extractor)(**extractor_kwargs)` builds the extractor
- heat: `extractor.score_queries(torch.from_numpy(features), positives, negatives)` → (V,) in
  [0, 1]; unobserved vertices set to 0, so they never reach the floor; `show_heat`
- switching away drops the extractor and features, `pytorch_gc()`

### What moves, what goes

| item | verdict |
|---|---|
| `docs/examples/ocr_lens_viewer.py` | delete; `_chart` and the word-mode wiring move into `viewer.py` |
| `_probability_maps`, `_lift_onto`, `ocr_lens_vertices.npy` | already deleted by semantics-storage |
| `docs/mesh.md:281` | points at `python -m collab_splats.viewer` |
| `Viewer.show_heat`, `add_label_list`, `on_click` | reused unchanged |

## Testing (`tests/test_viewer.py`, mocked server)

- `_find_stores`: no `semantics/` → empty; word store and queryable codes store listed; dinov2-only
  codes store not listed; vertex-count mismatch skipped
- `_build` with no stores: mesh added, no dropdown
- dropdown switch: previous heat cleared, previous mode's GUI removed, new mode's GUI present;
  leaving word mode removes the label list and the click handler
- word mode on a toy store: query heat equals the summed `word_probs`; label list ranks by mass;
  probe charts top words
- text mode with a stub registered queryable extractor: heat equals `score_queries` on the
  `read_point_features` output; zero-code vertices score 0; extractor built once across two searches
- import check: `collab_splats.viewer` does not import `collab_splats.reconstructor`

Manual: GH010229 with ocr_lens and maskclip vertex stores — switch modes, query both;
no-semantics backend shows the mesh with no dropdown.

## Files

| file | change |
|---|---|
| `collab_splats/viewer.py` | `main`, `_build`, `_load_mesh`, `_find_stores`, `_word_mode`, `_text_mode`, `_chart` |
| `docs/examples/ocr_lens_viewer.py` | deleted |
| `docs/mesh.md` | viewer pointer |
| `tests/test_viewer.py` | tests above |
| semantics-storage scope | vertex store for queryable extractors (owned by that spec) |
