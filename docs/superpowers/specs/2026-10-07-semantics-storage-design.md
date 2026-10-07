# Semantics storage — design

Date: 2026-10-07 · Branch: off `clean/final` · Status: approved in brainstorm, spec under review

## Goal

- Keep only what querying, segmentation, continuous maps and re-lifting need.
- Store OCR-lens word probabilities so a viewer never decodes them.

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
- `lift_features` of full-vocabulary maps: 102 s per 131k-point chunk, ~6.5 min per 500k targets, 7.5 GiB GPU
- top-k mass kept per vertex (97k observed vertices of the `.npy`):

| k | median | p10 | worst |
|---|---|---|---|
| 32 | 0.944 | 0.845 | 0.48 |
| 64 | 0.975 | 0.912 | 0.59 |
| 128 | 0.988 | 0.946 | 0.70 |

- a median vertex has 12 words above 0.015; median top-1 p is 0.25
- scene probability of the 50 most common words, from top-64 vs full: 0.987 to 0.996

## Decisions (from brainstorm)

- **The 2D cache exists to re-lift.** Querying and word probabilities go through the lifted stores.
- **The full-width states are temporary.** Extract, train the AE once on all frames, encode, delete.
  - changing `n_components` re-runs the extractor; accepted, the AE is not expected to be retrained
- **The AE trains on all frames.** `fit` streams frame blocks from the store instead of an 8 GiB sample.
- **Codes are fp16 on disk**, in both the 2D cache and the lifted stores.
- **Word probabilities: per-target top-64**, each target its own 64 words.
  - exact for top-1 and top-10, so labels, select-by-label and the probe chart are unchanged
  - every word with p > 1/65 is kept; fainter words read as 0
  - `dropped_mass` records what top-64 left out, per target
- **Decode, then lift.** Word probabilities are computed per frame and then lifted, as the OCR-lens spec requires.
- **Points and mesh vertices both get stores** ([decision 021](../decisions/021-lift-features-onto-displayed-geometry.md)).
  - lifted from the 2D cache straight onto each, no k-NN hop for display
  - mesh vertices: `pixel_indices=None`, so vertices no view sees stay zero (unobserved)
- **Semantics runs after mesh.** `STAGES` order puts `mesh` before `semantics`; no new dependency.
  - semantics writes the vertex store when `mesh.ply` exists and the extractor is ocr_lens
  - a mesh re-run alone leaves the vertex store stale; the viewer refuses a vertex-count mismatch
- **No legacy fallback.** Old `<extractor>.zarr` caches and `ocr_lens_vertices.npy` go unread.

## Layout

| store | path (spelled by `Reconstructor`) | contents |
|---|---|---|
| 2D codes | `<scene>/semantics/<extractor>_codes.zarr` | `features` (N, latent, H_p, W_p) fp16, `autoencoder.pt`; attrs `extractor`, `patch_size`, `n_frames`, `extractor_kwargs`, `latent_dim` |
| 2D states (temporary) | `<scene>/semantics/<extractor>.zarr` | full-width `features`; deleted once encoded |
| point store | `<scene>/<backend>/semantics/<extractor>_lifted.zarr` | `features` (P, latent) fp16, `autoencoder.pt` |
| vertex store (ocr_lens only) | `<scene>/<backend>/semantics/<extractor>_vertices.zarr` | word arrays only; no codes, nothing reads them |

- word arrays, in the point store (ocr_lens) and the vertex store: `word_ids` (T, 64) uint16,
  `word_probs` (T, 64) fp16, `dropped_mass` (T,) fp16, attrs `words` (the vocabulary list)
- `n_components: null`: `<extractor>_codes.zarr` holds full-width features, `latent_dim` null, no AE,
  no temporary store.
- the 2D codes store stays local (`PUSH_EXCLUDES`); the point and vertex stores are pushed.

Expected sizes, GH010229: 2D codes ~360 MB; point store ~250 MB (codes + words); vertex store ~140 MB.
Was ~20 GB.

## Stage flow (`Reconstructor.semantics`)

1. Cache check: `valid_feature_cache(codes_path, ...)` with `latent_dim`; a hit skips 2-5.
2. Extract every frame to the temporary states store (`extract_feature_cache`).
3. Train the AE on all frames: `FeatureAutoencoder.fit(states["features"], ...)` streams.
4. Encode every frame into the codes store (`compress_feature_cache`); validity attrs last.
5. Delete the states store.
6. Lift codes onto points → point store.
7. ocr_lens only: `lift_word_probabilities` onto points, written in the point store's atomic write.
8. ocr_lens and `mesh.ply` exists: `lift_word_probabilities` onto its vertices → vertex store.

A crash between 2 and 4 leaves the states store; the next run overwrites it.

## API changes

Net: two public functions added, one public function removed.

### `collab_splats/semantics/store.py`

| name | verdict | change |
|---|---|---|
| `valid_feature_cache` | change | takes `store_path` and `latent_dim`; checks `latent_dim` too |
| `extract_feature_cache` | change | takes `store_path`; writes `latent_dim: None`; drops its own validity check and `overwrite` (`Reconstructor` checks first) |
| `compress_feature_cache` | **new** | `(states_path, codes_path, ae, batch_size)`: encode every frame into fp16 codes, save `autoencoder.pt`, copy attrs + `latent_dim` last |
| `write_point_features` | change | casts codes to fp16; `codes` may be None (vertex store); `arrays` and `attrs` keyword args go into the same atomic write |
| `read_point_features` | keep | already casts to float32 and normalizes |

### `collab_splats/semantics/compression.py`

| name | verdict | change |
|---|---|---|
| `FeatureAutoencoder.fit` | change | takes any (N, D, ...) array (tensor, ndarray, zarr); reads axis-0 blocks of `read_gb`, flattens trailing axes to rows, shuffles blocks and rows within a block, every epoch |

### `collab_splats/utils/torch_utils.py`

| name | verdict | why |
|---|---|---|
| `load_features` | delete | its one caller trains through `fit` now; its tests go with it |

### `collab_splats/semantics/features/ocr_lens.py`

| name | verdict | change |
|---|---|---|
| `lift_word_probabilities` | **new** | `(codes, rows, ae, cloud, decoder, vocab, *, k=64, chunk=131_072) -> (word_ids, word_probs, dropped_mass)`; per target chunk, `lift_features` over per-frame full-vocabulary maps, then top-k |
| `_word_map` | new, private | frame i's codes → AE → `word_probabilities` → (n_words, H_p, W_p); the `lift_features` callable via `partial`, like `_load_frame` |

### `collab_splats/reconstructor.py`

| name | verdict | change |
|---|---|---|
| `STAGES` | change | `mesh` before `semantics` |
| `semantics()` | change | flow above |
| `_load_frame` | change | AE branch goes; codes are read as stored |

### `docs/examples/ocr_lens_viewer.py`

- reads the vertex store: `word_ids`, `word_probs`, `words`; no decoder, no `--model-id`
- refuses when the store's row count differs from `mesh.ply`'s vertex count: re-run semantics
- scene terms: union of observed vertices' top-10; expanded to a (V_obs, n_terms) array over them only
- smoothing (`transfer_features` in place) and labels: unchanged
- query: per-vertex sum of the stored probabilities of the query words
- `_probability_maps`, `_lift_onto`, the `.npy` cache: deleted

## Testing

- `extract_feature_cache` + `compress_feature_cache`: codes store holds (N, latent, H_p, W_p) fp16 and
  `autoencoder.pt`; states store gone after the stage.
- `fit` streaming: every frame contributes (a fixture whose frames differ per frame); block
  shuffling runs on a zarr and a tensor; empty input still raises.
- validity: `latent_dim` mismatch, crash before the attrs, and wrong frame count each read invalid.
- `n_components: null`: full-width features in the codes store, no AE, no temporary store.
- `write_point_features`: fp16 on disk; `arrays`/`attrs` land in the same store; a failed write
  leaves no store; `read_point_features` returns float32 unit rows.
- `lift_word_probabilities` (toy decoder + vocabulary): matches a dense lift followed by top-k;
  `dropped_mass == observed sum - sum(top-k)`; k capped at the vocabulary size; unobserved
  targets have zero probabilities.
- `Reconstructor.semantics`: stage order (mesh before semantics); vertex store written only for
  ocr_lens with `mesh.ply`; word arrays only for ocr_lens.
- Removed with their code: `load_features` tests, the `_load_frame` AE test.
- Gate: `tests/semantics tests/reconstructor tests/utils tests/test_docstring_contract.py
  tests/test_import_style.py`, in the worktree, printing `collab_splats.__file__`.

Real run on GH010229 (tmux), numbers reported, no thresholds:
- disk size of each store
- stage time: extract, AE fit, encode, point lift, word probabilities (points, vertices)
- peak GPU and RSS
- top-1 vertex label agreement vs the old `.npy` (cost of decoding from codes)
- viewer start-up time

## Docs

- `docs/semantics.md`: layout table (it already says fp32 and `_ae.pt`, both stale)
- `configs/base.yaml`: `n_components` comment (changing it re-extracts)
- CLAUDE.md in-flight entry; CHANGELOG on completion

## Out of scope

- a sparse lift (lifting only the 64 ids; ~14x less work, unmeasured); follow-up if ~7 min per store is too slow
- dashboard (its imports are already broken pending its own cleanup)
- one pretrained AE shared across scenes
- deleting old caches on disk: done by hand
