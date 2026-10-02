# OCR verbalization lens + compress-before-lift semantics — design

- **Status:** draft, awaiting user approval (no repo code before approval)
- **Branch:** `feat/ocr-lens` off `clean/final` `16d0206d`, worktree `.worktrees/ocr-lens`
- **Prototype:** `scratch/ocr_lens/` (gitignored) — `ocr_lens_proto.py`, `score_heads.py`,
  `ae_word_fidelity.py`, `HANDOFF.md`
- **Upstream:** sfeucht/ocr @ 60868b7b (MIT); Feucht et al. 2026, arXiv 2609.18823

## Goal

A query-free per-pixel word decoder in `collab_splats.semantics`, lifted to per-point word labels,
runnable as the semantics stage (`semantics.extractor: ocr_lens`).

Two parts:

1. **Stage order change, all extractors:** compress per frame, then lift codes — replaces
   lift-then-compress, so neither RAM nor GPU ever holds all frames or a full-width `(P, D)`.
2. **`ocr_lens` extractor:** LLaVA-1.6 image-token states through the OCR-head verbalization lens,
   decodable to English lemmas.

Success criterion: `Reconstructor` semantics stage with `extractor: ocr_lens` on the tutorial
scene produces per-point codes that decode to scene words, within the memory envelope below.

## Evidence

### Backbone (prototype, `ref_image.jpg`)

| model | English mass, verbalization / logit lens |
|---|---|
| Qwen3-VL-2B | 0.24 / 0.24 |
| Qwen3-VL-8B | 0.10 / 0.06 |
| LLaVA-1.5-7B | 0.77 / 0.36 |
| LLaVA-1.6-Vicuna-7B | **0.95** / 0.36 |

LLaVA-1.6 gives clean regional words (sky, tree, log, ground), stable over layers 12-31. 98/102
OCR heads are shared with LLaVA-1.5, so the gain is AnyRes resolution, not head choice.

### Autoencoder word fidelity (`ae_word_fidelity.py`)

LLaVA-1.6, layer 17, RMS-normalized lens states. AE (`FeatureAutoencoder`, `target_cosine=0.95`,
≤100 epochs) trained on 8 tutorial-video frames (10,752 patches), tested on held-out
`ref_image.jpg` (1,344 patches, 4,781 lemmas). 3x3 patch blocks stand in for one point's views.
Reference for the block columns: words decoded from the full-width block mean.

| r | fit cos | test cos | patch top-1 | patch top-5 | lift→AE top-1 | AE→avg top-1 |
|---|---|---|---|---|---|---|
| 32 | 0.881 | 0.872 | 0.703 | 0.733 | 0.833 | 0.826 |
| 64 | 0.919 | 0.908 | 0.770 | 0.794 | 0.861 | 0.861 |
| 128 | 0.947 | 0.938 | 0.836 | 0.847 | 0.938 | 0.931 |
| 256 | 0.951 | 0.940 | 0.825 | 0.845 | 0.882 | 0.896 |

- r=256 stopped at 36 epochs on the cosine gate, so it is undertrained, not a ceiling
- AE→avg matches lift→AE within 0.01 at every r: averaging codes is as good as averaging
  full-width states, so compress-before-lift costs nothing measurable
- cosine is a weak proxy for words: 0.938 test cosine still flips 16% of per-patch top-1
- caveats: surrogate multi-view (image blocks), one test image, one seed

## Part 1 — semantics stage: compress before lift (all extractors)

### Current

```
extract_feature_cache -> load_feature_maps (all N maps in RAM) -> lift_features -> (P, D)
  -> FeatureAutoencoder.fit on (P, D) on GPU -> per_point_encode -> write_point_features
```

### New

```
extract_feature_cache(extractor, images_dir, semantics_cache_dir)   # fp16, one chunk per frame
ae.fit(store["features"], ...)                                       # streamed from the store
codes = lift_features(partial(_encode_frame, store, ae), result)    # (P, r)
write_point_features(lifted_dir, name, codes, ae)                   # format unchanged
```

### Changes

- **`extract_feature_cache`** (`semantics/utils.py`) — reused under its current name; only change:
  stored dtype float32 -> float16. Still one chunk per frame, one frame in RAM, validity attrs
  written last, valid hit skips extraction.
- validity also keys on the extractor's kwargs: `extract_feature_cache(..., extractor_kwargs)`
  stores them as a JSON attr and `open_valid` compares them, so a changed `layer` re-extracts
  instead of silently reusing another layer's cache
- **cache location** — unchanged: `Reconstructor.semantics_cache_dir` (`output_path/semantics`),
  passed down as `cache_dir`. No new config key; the functions take a plain path.
- **cache persists** — re-running semantics (after `refine`, or with a new `n_components`) hits
  the cache and skips extraction.
- **`lift_features(frame_features: Callable[[int], Tensor], result, *, depth_tol)`** —
  first argument becomes a per-frame loader instead of a list
  - main loop calls `frame_features(i)` once per frame; D comes from the first call
  - fallback (`_sample_at_source_pixels`) checks its per-frame point mask BEFORE loading, so it
    only loads frames that hold zero-weight points
  - algorithm otherwise unchanged: depth-consistent visibility, confidence weights, bilinear
    sample, weighted mean, source-frame fallback
  - list callers pass `maps.__getitem__`
- **`FeatureAutoencoder.fit(features: Tensor | zarr.Array, ...)`** — tensor input keeps the
  current in-memory path; a zarr `(N, D, H_p, W_p)` array streams:
  - each epoch shuffles frame order, reads one frame chunk, prefetches the next with a
    `ThreadPoolExecutor` future, shuffles that frame's patches into batches
  - fp16 chunks cast to float32 on read
  - trains on per-patch features (was per-point); fidelity table above covers the change
- **`_encode_frame(store, ae, i)`** (private, `wrapper/reconstructor.py`) — reads frame i, casts,
  `ae.encode` -> `(r, H_p, W_p)`; bound with `functools.partial`, no wrapper class
- **`_lift_and_save`** — body reordered as above
- **`n_components: null`** — no AE: lift the cached full-width maps through
  `lambda i: store[i]`-style loading; docs warn this is `P x 4096` for `ocr_lens`

### Memory envelope (300 frames, 1M points, r=128)

| | today | new |
|---|---|---|
| RAM | all N maps (ocr_lens ≈ 6-11 GB) | one frame + one prefetched chunk |
| GPU, lift | `P x D` accumulator (ocr_lens 16 GB) | `P x r` (0.5 GB) |
| GPU, AE fit | whole `(P, D)` resident (ocr_lens 16 GB) | one batch |
| cache on disk | fp32 | fp16 (ocr_lens ≈ 5.6 GB) |

### Deferred (unchanged in this effort)

- `load_feature_maps`, `cache_store_path` — kept
- dashboard `lift_point_features` (`dashboard/viewer.py`) and `_lift_and_compress`
  (`dashboard/pipeline.py`) — only change: pass `maps.__getitem__` to `lift_features`; they will
  see fp16 maps from the cache (`lift_features` casts to float32 on device)
- tutorials 05/06 — left to tutorial-rework
- `remote/sources.py` exclude rule — unchanged

## Part 2 — `ocr_lens` extractor

`collab_splats/semantics/features/ocr_lens.py`, `@BaseFeatureExtractor.register("ocr_lens")`,
module docstring opens with the attribution, once, at the top of the file:

```python
"""
OCR-head verbalization lens: LLaVA-1.6 image-token states decoded to English words.

- paper: Feucht, Krojer, Wang, Abrahamsen, Wallace, Bau (2026), "Using OCR Heads to Verbalize
  Image Semantics", arXiv 2609.18823 — https://ocr.baulab.info/
- code: ported from https://github.com/sfeucht/ocr @ 60868b7b (MIT):
  src/sec2__score_heads.py, src/sec2__ocr.py, src/sec3__object_detection.py, src/sec3__lens.ipynb
- ours, not upstream: LLaVA-1.6 AnyRes grid, WordNet vocabulary, RMS-normalized states
"""
```

- file list verified against the pin before landing; no per-function repeats of the credit

### Public surface

Everything lives in `collab_splats/semantics/features/ocr_lens.py`; one extractor class (the
registry needs it), the rest plain functions. No flags that switch modes.

| name | role |
|---|---|
| `OCRLensExtractor` | registered `ocr_lens`: image -> lens states per token-grid cell |
| `word_vocabulary(tokenizer, words=None)` | build the decode vocabulary (default or custom words) |
| `load_decoder(model_id)` | `norm + lm_head` only, ~260 MB, for decoding stored codes |
| `verbalize(features, decoder, vocab, *, k=10)` | states -> top-k words per row |
| `score_ocr_heads(model_id, *, n, batch_size, seed)` | recompute the per-head OCR scores |

Typical use:

```python
ext = OCRLensExtractor(layer=20)                        # or via config extractor_kwargs
maps = ext.forward(images)                              # [(4096, rows, cols)]
vocab = word_vocabulary(ext.processor.tokenizer, ["tree", "rock", "water", "sky"])
words, probs, mass = verbalize(point_states, ext.decoder, vocab)

scores = score_ocr_heads()                              # full 7B, run in tmux
ext = OCRLensExtractor(head_scores=scores)
```

### `OCRLensExtractor`

`OCRLensExtractor(model_id="llava-hf/llava-v1.6-vicuna-7b-hf", layer=17, n_heads=102,
head_scores=None, dtype="float16")`

- loads `LlavaNextForConditionalGeneration` + processor
- lens: sum of `W_O W_V` over the top-`n_heads` scored heads, `(4096, 4096)`, accumulated in
  fp32 on CPU (not model dtype)
  (GQA-aware indexing kept from the prototype, though Vicuna has no GQA)
- `layer` is a free parameter (default 17; the prototype saw stable words over 12-31):
  `0 <= layer < num_hidden_layers`, else `ValueError` at construction
- language model truncated to layers `0..layer`; the layer output is read with a forward hook
  (drops 14 of 32 layers, and no all-layer `output_hidden_states`)
- `head_scores`: `None` loads the packaged file, a path loads that file, a tensor is used as is
- attributes `decoder` (`nn.Sequential(norm, lm_head)`) and `vocab` (default `word_vocabulary`)
- `debias_validated = False`; `patch_size = 14` is nominal (recorded in cache attrs only)

`forward(images) -> list[Tensor]` overrides the base, since AnyRes has no fixed pixel patch:

- chat template with the image and an empty text turn (as upstream)
- keeps only the unpadded high-res token grid (drops the 24x24 overview and row-newline tokens),
  which covers the full image field of view, so `lift_features` grid sampling is valid
- returns `rms_normalize(lens @ h)` as `(4096, rows, cols)` float32 CPU per image
- decoding is NOT linear: the decoder's RMSNorm rescales an averaged state, giving
  `mean(logits) / rms(avg)`, sharpened where views disagree
- word probabilities therefore decode per frame, then lift (a mixture over views); lifted
  states are embedding averages, for similarity only

### Vocabulary: `word_vocabulary(tokenizer, words=None) -> WordVocab`

- `WordVocab`: typed dataclass `words: list[str]`, `token_ids: Tensor`, `word_index: Tensor`
  (token -> row in `words`)
- `words=None`: every word-initial (`▁`) ASCII-letter token whose WordNet base form exists,
  merged by lemma (≈4.8k lemmas for the Llama-2 tokenizer) — the default
- `words=[...]`: a custom label set; each word collects every word-initial token whose lemma
  matches it (`"trees"`, `"Tree"` -> `tree`); words with no single-token form are dropped with
  one warning listing them
- same function feeds `score_ocr_heads` (its single-token rendering words)
- new runtime dependency `nltk` (pyproject); `setup.sh` downloads the `wordnet` corpus; a missing
  corpus raises with the download command

### Decoding: `verbalize(features, decoder, vocab, *, k=10) -> (words, probs, vocab_mass)`

- `features`: `(P, 4096)` (e.g. decoded point codes) or `(4096, H, W)`
- `decoder -> softmax`, token probs summed per vocab word, renormalized, top-k
- `vocab_mass`: full-vocab probability landing on the vocabulary, per row (low = the lens has no
  word for it; with a custom vocab, low = none of your labels)
- `decoder` comes from the extractor or from `load_decoder(model_id)`, which reads only the
  `norm` and `lm_head` tensors from the checkpoint shards via the existing `hf_hub_download`
  helper
- query-free counterpart of `BaseQueryableExtractor.score_queries`

### Head scores

- packaged: `collab_splats/semantics/features/assets/llava16_vicuna7b_ocr_head_scores.pt` (5 KB,
  mean `(layers, heads)` float32, the same format `score_ocr_heads` returns); first package-data
  entry in `pyproject.toml`
- recompute: `score_ocr_heads(model_id=<default>, *, n=1024, batch_size=4, seed=177) -> Tensor`,
  ported from `scratch/ocr_lens/score_heads.py` (plain HF hooks, no nnsight)
  - loads its own full model (the extractor's is truncated at `layer`, so it cannot score heads
    above it)
  - renders `word_vocabulary` words onto mini-imagenet backgrounds (HF dataset shard via
    `hf_hub_download`), DejaVu font
  - returns mean OCR score per head, `(layers, heads)`; `torch.save` it and pass the path (or
    the tensor) as `head_scores`
  - port validated on Qwen3-VL-2B: 42/45 top heads match upstream's released scores
  - `batch_size=4`: 16 OOMs on LLaVA-1.6 in 46 GB

### Config

- `semantics.extractor: ocr_lens`; extractor defaults live in `__init__`
- new key `semantics.extractor_kwargs: {}` in `base.yaml`, passed through
  `_get_extractor(name, **kwargs)` to the extractor constructor — how a run changes the layer or
  points at rescored heads: `semantics: {extractor: ocr_lens, extractor_kwargs: {layer: 20}}`
  - generic, so any extractor's constructor args are reachable; values must be YAML/JSON scalars
    (they also key the cache, above)
  - `dtype` is therefore a string (`"float16"`) in the signature, resolved with `getattr(torch, ...)`
- docs recommend `n_components: 128` for `ocr_lens` (table above); global default stays 64

## Testing

Flat test functions, `tests/` mirrors the package.

- `tests/semantics/test_lifting.py`
  - callable input equals the old list result bit-for-bit on the existing fixture
  - counting loader: fallback loads only frames holding zero-weight points
- `tests/semantics/test_compression.py`
  - zarr-input fit reaches the tensor-input fit's cosine within tolerance on the same data
  - instrumented store: at most one read chunk + one prefetched chunk in flight
- `tests/semantics/test_semantics_utils.py` — `extract_feature_cache` writes float16; valid hit
  skips extraction; changed `extractor_kwargs` re-extracts; a mid-loop crash leaves an attr-less
  store that is re-extracted
- `tests/semantics/features/test_ocr_lens.py` (tiny random configs, no 7B download)
  - lens equals a hand-summed `W_O W_V` on a tiny Llama config
  - out-of-range `layer` raises; truncated model keeps exactly `layer + 1` decoder layers
  - AnyRes unpadded-grid extraction on a tiny `LlavaNext` config: grid shape, overview and
    newline tokens dropped
  - `word_vocabulary` on a synthetic tokenizer: default lemma merge; custom words collect
    inflected/cased tokens; unmatched words dropped with a warning
  - `verbalize` aggregation and `vocab_mass` on a synthetic vocabulary
  - `load_decoder` on a tiny saved checkpoint: same `verbalize` output as the extractor's decoder
- `score_ocr_heads` on a tiny config (same test file): output shape `(layers, heads)`, and the
  result round-trips as `head_scores` into the extractor
- GPU parity (skips without GPU or cached weights): `verbalize` top-1 map on `ref_image.jpg`
  equals the prototype's; prototype vs upstream `verb_ll` is already checked (notebook section 2)
- `tests/wrapper/test_reconstructor.py` — semantics stage in the new order; re-run hits the cache;
  `extractor_kwargs` reaches the extractor constructor
- `test_docstring_contract.py` and `test_import_style.py` already cover `semantics`

## Gates measured before landing

1. streamed AE fit wall-time on a ~300-frame scene vs today's in-memory fit
2. talk2dino on the new path: `score_queries` agreement on lifted points, compress-then-lift vs
   lift-then-compress
3. `ocr_lens` end to end on the tutorial scene: peak GPU and RAM, point-level word map

## Docs

- `docs/semantics.md` — `ocr_lens` section: what the lens is, fidelity table, r=128, decoding
  points to words (file stays untracked, as today)
- decision `docs/superpowers/decisions/020-compress-before-lift.md` — the stage order change
- `configs/base.yaml` comment: `ocr_lens` in the extractor list
- `docs/superpowers/CHANGELOG.md` entry on landing

## Risks

- cosine stop gate is a weak proxy for word fidelity; reported, not changed, in this effort
- fp16 cache is visible to the deferred dashboard readers (they get fp16 tensors)
- re-running semantics after deleting the cache re-runs the extractor (a 7B forward per frame
  for `ocr_lens`)
- new setup surface: `nltk` + the WordNet corpus

## Out of scope

- other backbones (Qwen3-VL, LLaVA-1.5): LLaVA-1.6 had the cleanest words (0.95 English mass)
- dashboard and tutorial migration off `load_feature_maps`
- word-agreement stop criterion for the AE
- mesh / splats consumption of word labels
