# Semantics storage — design

Date: 2026-10-07 · Branch: off `clean/final` · Status: approved in brainstorm; revised 2026-10-07 to option B (codes only)

## Goal

- Keep only what querying, segmentation, continuous maps and re-lifting need.
- Store codes only: fp16 per-frame codes and per-point codes, each with its AE. Words and
  mesh-vertex values are derived by the viewer at start-up, never stored.

## Measured today (GH010229, 1039 frames, vggt_omega, A40)

| store | shape | dtype | disk |
|---|---|---|---|
| `semantics/ocr_lens.zarr` | (1039, 4096, 28, 48) | fp16, zstd 0 | 16 G |
| `vggt_omega/semantics/ocr_lens_lifted.zarr` | (486k, 128) codes + AE | fp32 | 266 M |
| `vggt_omega/ocr_lens_vertices.npy` (viewer cache) | (545k, 3512) | fp16 | 3.8 G |

- The AE (4096 → 128) already exists; it is applied only at lift time, so the cache keeps full states.
- The AE trains on an evenly strided frame subset only because `load_features` caps fp32 samples at 8 GiB.
- The viewer loads `lm_head`, decodes every frame to the full 3,512-word vocabulary, lifts (V, 3512).

Measured for this design:
- decode codes → AE → `lm_head` → word probabilities: 15 ms per frame (~17 s for 1039 frames)
- `lift_features` of full-vocabulary maps, all 1039 frames onto 486k points:
  - today's lift, reading full-state frames: 264 s, 21.5 GiB peak GPU
  - visible-only `index_add_` (below), reading fp16 codes: 74 s, 12.2 GiB; top-64 equal to today's on
    every observed point (max p difference 1.2e-7)
  - boolean-mask `acc[vis] += s` instead: 61 s, 15.7 GiB; `index_add_` chosen for the memory
- where today's per-frame time goes (20 frames): read full-state frame 86 ms, decode 22 ms,
  visibility 2 ms, sample all points 65 ms, accumulate all points 175 ms; ~12% of points are
  visible in a median frame (max 38%)

- top-k mass kept per vertex (97k observed vertices of the `.npy`):

| k | median | p10 | worst |
|---|---|---|---|
| 32 | 0.944 | 0.845 | 0.48 |
| 64 | 0.975 | 0.912 | 0.59 |
| 128 | 0.988 | 0.946 | 0.70 |

- a median vertex has 12 words above 0.015; median top-1 p is 0.25
- scene probability of the 50 most common words, from top-64 vs full: 0.987 to 0.996

Measured for B (codes only), GH010229 `ocr_viewer`, 294 frames, 545k vertices, 500k points, A40:
- full states → fp16 codes: 22.9 s; codes store 101 MB (vs 4.5 GB full states)
- decode codes → per-patch top-64 words, all frames: 3.9 s
- indexed lift (below) of those words onto 545k vertices: 2.4 s, 9.1 GiB peak; onto points 3.6 s
  vs 66.8 s for the dense lift of full-vocabulary maps
- indexed vs dense lift, same top-64 patch inputs: top-1 equal on 99.9998% of vertices, 100% of
  points; top-10 sets equal on 99.999%; max p difference 0.0019
- codes lift onto vertices: 2.8 s
- per-patch top-64 truncation vs the full-vocabulary lift (indexed prototype, all 1039 frames of
  `2026_07_15` onto 486k points, full-state frames): top-1 identical on every point; top-10
  overlap 0.9996; top-64 L1 median 0.001, p99 0.007; 67 s, 9.3 GiB vs 264 s, 21.5 GiB
- per-frame top-64 held for the lift: ids int64 + probs fp32, ~1 MB per frame (~300 MB at 294
  frames, ~1 GB at 1039)
- lifting codes then decoding each vertex instead (route 2) is rejected: softmax of averaged codes
  is winner-take-all (mean max p 0.76 vs 0.31), loses minority words such as "line" (the tracks);
  top-1 agrees on 70% of vertices only

## Decisions (from brainstorm)

- **Codes only (option B).** On disk: the 2D codes store and the point store, both codes + AE.
  No word arrays, no vertex store.
  - the viewer decodes words and lifts onto mesh vertices at start-up, on the GPU (~7 s at 294
    frames, decoder load excluded); talk2dino text queries need the GPU anyway
  - storing per-frame top-64 words in the codes store later (B+) needs no migration
- **The 2D codes are pushed.** The viewer reads them, and a re-lift after a mesh or pointcloud
  re-run never re-extracts, local or remote.
- **The full-width states are temporary.** Extract, train the AE once on all frames, encode, delete.
  - changing `n_components` re-runs the extractor; accepted, the AE is not expected to be retrained
- **The AE trains on all frames.** `fit` streams frame blocks from the store instead of an 8 GiB sample.
- **Codes are fp16 on disk**, in both the 2D cache and the lifted stores.
- **Word probabilities: top-64 per patch, per frame, in the viewer.** Each patch keeps its own 64
  words; the indexed lift adds them at their id columns; each vertex then keeps its own top-64 in memory.
  - top-1 identical and top-10 overlap 0.9996 vs the full-vocabulary lift (measured above), so
    select-by-label and the probe chart are unchanged
  - every word with p > 1/65 per patch is kept; fainter words read as 0
- **Decode, then lift.** Word probabilities are computed per frame and then lifted, as the OCR-lens
  spec requires; lifting codes and decoding per vertex loses minority words (measured above).
- **Mesh vertices lift straight from the 2D codes** ([decision 021](../decisions/021-lift-features-onto-displayed-geometry.md)),
  in the viewer: no k-NN hop, `pixel_indices=None`, so vertices no view sees stay zero (unobserved).
  A mesh re-run cannot leave a vertex store stale: there is none.
- **Stage order unchanged.** Semantics no longer reads the mesh.
- **No legacy fallback.** Old `<extractor>.zarr` caches, `ocr_lens_vertices.npy` and `_ae.pt` go
  unread. The real run deletes them by hand once the new stores exist; an old `<extractor>.zarr`
  left on disk would otherwise be pushed.

## Layout

| store | path (spelled by `Reconstructor`) | contents |
|---|---|---|
| 2D codes | `<scene>/semantics/<extractor>_codes.zarr` | `features` (N, latent, H_p, W_p) fp16, `autoencoder.pt`; attrs `extractor`, `patch_size`, `n_frames`, `extractor_kwargs`, `latent_dim` |
| 2D states (temporary) | `<scene>/semantics/<extractor>_states.zarr` | full-width `features`; deleted once encoded |
| point store | `<scene>/<backend>/semantics/<extractor>_lifted.zarr` | `features` (P, latent) fp16, `autoencoder.pt` |

- every path is keyed by extractor: two extractors (e.g. ocr_lens, talk2dino) are two runs with
  `semantics.extractor` changed, each writing its own `_codes` and `_lifted` store with its own AE;
  the stage marker is the configured extractor's point store
- every backend lifts from the one scene-level codes store into its own `<backend>/semantics/`
- `n_components: null`: `<extractor>_codes.zarr` holds full-width features, `latent_dim` null, no AE,
  no temporary store; it is pushed at full width
- `PUSH_EXCLUDES`: `/semantics/**` becomes `/semantics/*_states.zarr/**`; codes and point stores are
  pushed, the temporary states never

Expected sizes, GH010229 (1039 frames): 2D codes ~360 MB; point store ~125 MB. Was ~20 GB.

## Stage flow (`Reconstructor.semantics`)

1. Cache check: `valid_feature_cache(codes_path, ...)` with `latent_dim`; a hit loads the codes
   store's `autoencoder.pt` and skips 2-5.
2. Extract every frame to the temporary states store: `write_feature_cache` over the extractor's
   per-frame maps (images decoded and batched in `semantics()`).
3. Train the AE on all frames: `FeatureAutoencoder.fit(states["features"], ...)` streams.
4. Encode every frame into the codes store: `write_feature_cache` over
   `partial(_load_frame, states, range(N), ae)`; save `autoencoder.pt`; validity attrs last.
5. Delete the states store.
6. Lift codes onto points → point store.

A crash between 2 and 4 leaves the states store; the next run overwrites it.

## API changes

Net: no public function added; one removed (`load_features`), one renamed and generalized
(`extract_feature_cache` → `write_feature_cache`), one sped up and given an indexed-map input
(`lift_features`).

### `collab_splats/semantics/store.py`

| name | verdict | change |
|---|---|---|
| `valid_feature_cache` | change | takes `store_path` and `latent_dim`; checks `latent_dim` too |
| `extract_feature_cache` | rename → `write_feature_cache` | `(store_path, maps, n_frames, attrs)`: writes an iterable of per-frame (D, H_p, W_p) maps as fp16, one chunk per frame, attrs last; serves both extraction and encoding; no extractor, no validity check, no `overwrite` (`Reconstructor` checks first) |
| `write_point_features` | change | casts codes to fp16 |
| `read_point_features` | keep | already casts to float32 and normalizes |

### `collab_splats/semantics/compression.py`

| name | verdict | change |
|---|---|---|
| `FeatureAutoencoder.fit` | change | takes any (N, D, ...) array (tensor, ndarray, zarr); reads axis-0 blocks of `read_gb`, flattens trailing axes to rows, shuffles blocks and rows within a block, every epoch |

### `collab_splats/semantics/lifting.py`

| name | verdict | change |
|---|---|---|
| `lift_features` | change | samples only visible points (`w > 0`) and accumulates them with `features_sum.index_add_`; weights multiply in place; the mean divides in place. Same output: invisible points carried weight 0. Every lift gets it, codes included |
| `lift_features(..., num_classes=)` | change | indexed maps: `frame_features(i)` may return `(ids, values)`, each (K, H_p, W_p), listing K entries per patch of a `num_classes`-long vector whose other entries are 0. `num_classes` is the id space and the output width, never K: nothing is truncated by it. Private `_add_indexed_bilinear` adds each visible point's bilinear blend of its 4 nearest patches' lists into a (P, num_classes) accumulator at the id columns, with `index_add_`. Same bilinear weights as the dense `grid_sample` (border, `align_corners=False`); output (P, num_classes) float32, as dense. Same visibility, weights and mean; no `pixel_indices` fallback for indexed maps. `ValueError` on indexed maps without `num_classes` or with `pixel_indices`, an id outside `[0, num_classes)` (it would land in another point's row), or `num_classes` with dense maps |

### `collab_splats/utils/torch_utils.py`

| name | verdict | why |
|---|---|---|
| `load_features` | delete | its one caller trains through `fit` now; its tests go with it |

### `collab_splats/semantics/features/ocr_lens.py`

No change: `word_probabilities`, `load_decoder` and `word_vocabulary` are reused as they are.

### `collab_splats/reconstructor.py`

| name | verdict | change |
|---|---|---|
| `semantics()` | change | flow above |
| `_load_frame` | keep | its AE branch now feeds the encode step; the lift reads stored codes with `ae=None` |

### `collab_splats/remote.py`

| name | verdict | change |
|---|---|---|
| `PUSH_EXCLUDES` | change | `/semantics/**` → `/semantics/*_states.zarr/**` |

### `docs/examples/ocr_lens_viewer.py`

Built on the viewer as `mesh-query-heat` left it (heat overlay, mass-ranked labels, no smoothing).

- start-up, on the GPU: reads `ocr_lens_codes.zarr` + its AE (None when `n_components` is null,
  passed through to `word_probabilities`), `pointcloud.zarr` (depth, poses)
  and `mesh.ply`; maps cloud frames to codes rows with `store_rows`, as the stage does; decodes
  each frame once: codes → `word_probabilities` → top-64 per patch, held on the GPU as indexed maps;
  `lift_features(..., num_classes=n_words)` onto `replace(cloud, points=vertices, pixel_indices=None)`, in
  vertex chunks of `chunk` rows, each its own `lift_features` call (a (V, 3512) fp32 accumulator
  is 7.6 GB at 545k); keeps vertex top-64
  (`word_ids`, `word_probs`) in memory
- the decoder and vocabulary load as today (`--model_id` stays)
- observed: `word_probs[:, 0] > 0`
- label-list mass: `np.bincount(word_ids, weights=word_probs, minlength=n_words)`
- query heat: per vertex, the sum of `word_probs` where `word_ids` is a query word; floor `min p` unchanged
- probe chart: on click, the vertex's stored entries that are scene terms, renormalized, top-10;
  scene terms stay the union of observed vertices' top-10 (columns 0-9); no (V_obs, n_terms) array
- `_probability_maps`, `_lift_onto`, the `.npy` cache: deleted

## Testing

- `lift_features`: on the toy scene, output equals the pre-change implementation's (kept as a
  reference in the test), with points visible in some frames only and with confidence weights.
- `write_feature_cache`: fp16, one chunk per frame, attrs written last (a crash mid-write reads
  invalid); after the stage the codes store holds (N, latent, H_p, W_p) and `autoencoder.pt`, and
  the states store is gone.
- `fit` streaming: every frame contributes (a fixture whose frames differ per frame); block
  shuffling runs on a zarr and a tensor; empty input still raises.
- validity: `latent_dim` mismatch, crash before the attrs, and wrong frame count each read invalid.
- `n_components: null`: full-width features in the codes store, no AE, no temporary store.
- indexed `lift_features`: on the toy scene, equals the dense lift of the same indexed maps
  scattered to `num_classes` channels (visible-in-some-frames points, confidence weights); indexed maps with `pixel_indices` raise; unobserved
  targets without `pixel_indices` stay zero. Each `num_classes` misuse raises `ValueError`.
- `write_point_features`: fp16 on disk; a failed write leaves no store; `read_point_features`
  returns float32 unit rows.
- `PUSH_EXCLUDES`: `semantics/x_states.zarr/...` excluded; `semantics/x_codes.zarr/...` and
  `<backend>/semantics/x_lifted.zarr/...` pushed.
- Removed with their code: `load_features` tests.
- Gate: `tests/semantics tests/reconstructor tests/utils tests/test_docstring_contract.py
  tests/test_import_style.py`, in the worktree, printing `collab_splats.__file__`.

Real run on `ocr_viewer/GH010229` (294 frames, has `mesh.ply`; tmux), numbers reported, no thresholds:
- disk size of each store
- stage time: extract, AE fit, encode, point lift
- peak GPU and RSS
- viewer start-up time: decode, vertex lift, peak GPU
- top-1 vertex label agreement vs the old `.npy` (cost of decoding from codes)
- then, asked first: delete the old `semantics/ocr_lens.zarr`, `<backend>/ocr_lens_vertices.npy`
  and `<backend>/semantics/ocr_lens_ae.pt` in `ocr_viewer/GH010229` and
  `2026_07_15-Goprosplat-GH010229` (the latter has no mesh; its semantics are re-run when needed)

## Docs

- `docs/semantics.md`: layout table (it already says fp32 and `_ae.pt`, both stale)
- `configs/README.md`: output tree (`<extractor>.zarr` → `_codes.zarr`) and the not-pushed list
  (`/semantics/**` → `/semantics/*_states.zarr/**`)
- `remote.py`: the `PUSH_EXCLUDES` comment (codes pushed, states not)
- `configs/base.yaml`: `n_components` comment (changing it re-extracts)
- CLAUDE.md in-flight entry: rewritten to B (codes only, words derived by the viewer, codes
  pushed); it still says words stored on points and vertices and semantics after mesh; CHANGELOG on
  completion

## Out of scope

- stored word probabilities, on points, vertices or per frame (B+); addable without migration
- a vertex store; the [scene viewer](2026-10-07-scene-viewer-design.md) spec, which reads
  `*_vertices.zarr`, moves to lifting codes at start-up — its owner updates it
- HDF5 (collab-data's `talk2dino_lifted.h5`); zarr stays, an exporter on their side if needed
- dashboard (its imports are already broken pending its own cleanup)
- one pretrained AE shared across scenes
