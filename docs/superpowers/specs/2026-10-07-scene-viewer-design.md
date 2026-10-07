# Scene viewer — one viewer, optional semantics

Date: 2026-10-07 · Status: approved in brainstorm, spec under review · Depends on: semantics-storage

## Problem

- The only mesh query viewer is `docs/examples/ocr_lens_viewer.py`, hard-wired to `ocr_lens`
  (word vocabulary, decoder, `.npy` word cache).
- Text-queryable extractors (maskclip, talk2dino) have no mesh query path; the Panel dashboard
  queries points only.
- Viewing a mesh with no semantics needs the OCR script anyway, or none.

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

- **semantics writes the vertex store for every extractor when `mesh.ply` exists**, not only ocr_lens
  - all extractors: `features` (V, latent) fp16 codes + `autoencoder.pt`, as the point store
  - ocr_lens additionally: the word arrays (`word_ids`, `word_probs`, `dropped_mass`, attrs `words`)
  - attrs `extractor`, `extractor_kwargs` (to rebuild the text encoder)
- semantics-storage owns that change; this spec lands after it. Lifting at viewer start-up
  (the path semantics-storage removes for OCR) is rejected.

## Design

### Module layout (`collab_splats/viewer.py`)

- `Viewer` class unchanged (widget layer; the reconstructor's loop-closure display keeps using it).
- New, below it:
  - `main()`: CLI `backend_dir`, `--port`, `--textured`, `--texture_size` (the OCR script's flags
    minus `--model_id`, `--chunk`)
  - `_load_mesh(backend_dir, textured, texture_size)`: `mesh.ply` + optional `texture/mesh.obj`,
    moved from the OCR script
  - `_find_stores(backend_dir, n_vertices)`: `{extractor: path}` for `*_vertices.zarr` that hold
    word arrays or a queryable extractor's codes; a store whose row count differs from `mesh.ply`
    is skipped with a warning (re-run semantics)
  - `_word_mode(viewer, store)` and `_text_mode(viewer, store)`: build that mode's GUI and handlers,
    return the GUI handles to remove on switch
- Import rule: `viewer.py` never imports `collab_splats.reconstructor` (it imports `viewer.py`);
  `store_rows` is not needed once words come from the vertex store.

### Word mode (ocr_lens)

The OCR script's behavior on the semantics-storage vertex store, unchanged otherwise:

- query box (comma-separated words), `Query min p` floor, Search; unknown words noted
- heat: per vertex, sum of `word_probs` where `word_ids` is a query word
- label list: `np.bincount(word_ids, weights=word_probs, minlength=n_words)`, click = query
- probe: clicked vertex's top-10 scene terms in the HTML panel; unobserved vertices grey

### Text mode (queryable extractors)

- query box: comma-separated positives; `Negatives` box, default `object`
  (`score_queries`'s Talk2DINO convention); floor slider `Query min score`; Search
- on first Search: `BaseFeatureExtractor.get(extractor)(**extractor_kwargs)` builds the extractor;
  the AE decodes the vertex codes once to unit (V, D) features, kept on the GPU for the mode's life
- heat: `extractor.score_queries(features, positives, negatives)` → (V,) in [0, 1], `show_heat`
- unobserved vertices (zero codes) score 0 explicitly, so they never reach the floor
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
- `main` with no stores: mesh added, no dropdown
- dropdown switch: previous heat cleared, previous mode's GUI removed, new mode's GUI present
- word mode on a toy store: query heat equals the summed `word_probs`; label list ranks by mass;
  probe charts top words
- text mode with a stub registered queryable extractor: heat equals `score_queries` on decoded
  codes; unobserved vertices 0; extractor built once across two searches
- import check: `collab_splats.viewer` does not import `collab_splats.reconstructor`

Manual: GH010229 with ocr_lens and maskclip vertex stores — switch modes, query both;
no-semantics backend shows the mesh with no dropdown.

## Files

| file | change |
|---|---|
| `collab_splats/viewer.py` | `main`, `_load_mesh`, `_find_stores`, `_word_mode`, `_text_mode`, `_chart` |
| `docs/examples/ocr_lens_viewer.py` | deleted |
| `docs/mesh.md` | viewer pointer |
| `tests/test_viewer.py` | tests above |
| semantics-storage scope | vertex store for every extractor (owned by that spec) |
