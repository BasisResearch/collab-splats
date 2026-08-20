# Per-image feature caching in LocalMatcher — design

**Date:** 2026-08-20 (revised same day: residual attribution added, tiers inverted, scope cuts)
**Status:** approved design, pre-implementation
**Branch:** `refactor/cu121-uv-migration`
**Supersedes:** `docs/superpowers/handoffs/2026-08-20-matcher-feature-cache-handoff.md` (investigation sketch)

## Problem

`verify()` on the 300-frame GoPro scene (`/workspace/outputs/2026_07_15-Goprosplat-GH010229`,
vggt_omega + loma) takes **2511.6 s**. Measured split (handoff §1, commit `755c22d` harness):

- loma pair matching: ~1853 s (73.8%) — 1,865 pairs × 993.8 ms
- of which `detect_and_describe`: ~1581 s (62.9%)
- residual: ~658 s **upper bound, unattributed** — DB export, `verify_matches`,
  `triangulate_points`, model cold start, inferred by subtraction across two different harnesses

Per-pair (gap-16, CUDA-synced, warm): `detect_and_describe` ×2 = 78.3%, `_recover_indices` = 8.4%,
learned match transformer = **3.5%**, MAGSAC (discarded) = 0.2%.

**1,865 pairs × 2 extractions for 300 unique images = 12.4× redundancy.** The expensive stage
(detection) is per-image and cacheable; the valuable stage (the learned matcher) is per-pair and
cheap.

### Time budget after this work

`verify()` already holds the extraction result before matching starts: it calls
`build_localization_db()` unconditionally (`reconstructor.py:1113`), loads the zarr-cached
features, and passes them into `verify_reconstruction` as `features` — the very keypoint
tables `_write_frames` exports to the COLMAP DB. So:

- **General-path models (descriptor-NN): extraction inside verify = 0.** Matching is
  mutual-NN over precomputed descriptors — seconds, and match rows ARE table indices
  (no `_recover_indices` round-trip).
- **Loma (learned matcher, shipping default): extraction inside verify = 0 too.** The
  zarr cache grows the pre-transform payload (`keypoints_normalized`), so loma matches
  from stored features exactly like the NN path — learned matcher 1,865 × 38 ms ≈ 71 s +
  H2D ≈ 25 s → **matching term ~1853 s → ~100 s** on payload-bearing caches. Old caches
  without the payload fall back to pairwise matching with an in-memory per-image encode
  cache (~250 s) until rebuilt.

The residual is NOT treated as immovable (earlier drafts did; wrong). pycolmap's own pairing
log completed 300/300 within ~2 min, so the C++ phases are unlikely to dominate it — the prime
suspects are our Python DB-export loops and one-off model cold start. This spec includes
attributing it and attacking the biggest term. Overall target: **sub-15 min guaranteed by the
matching fix alone; ~8–12 min expected once the residual's biggest term is addressed.**

## Decisions already settled (do not relitigate)

1. **Reach into vismatch internals where needed.** User directive, reaffirmed (handoff §10).
2. **Legacy Disk/XFeat/Loma extractor classes are gone** (commits `091ad53..c537803`); vismatch
   `LocalMatcher` is the sole provider. The stale CLAUDE.md architecture line still listing them
   gets fixed in this pass.
3. **pycolmap-native matching is the future, not the present** (see Exit strategy). Verified
   2026-08-20: colmap PR #4524 (LoMa-B, ONNX, with pycolmap bindings) merged to main 2026-08-18,
   but **no PyPI pycolmap wheel ships ONNX support** — 4.0.4 (installed) and 4.1.1 (latest) both
   compile it out (`ALIKED feature extraction requires ONNX support` abort, no bundled
   onnxruntime, `has_cuda=False`). Source builds default `ONNX_ENABLED=ON` with a CUDA execution
   provider (`onnx_utils.cc:77`), so a Docker rebuild unlocks it later.
4. **Learned matchers are the value of loma-class models.** Generic descriptor matching may only
   serve models whose own match stage *is* descriptor-NN, proven equivalent by a suite parity
   test — never silently substitute for a learned matcher.

## Design

Everything lives in `collab_splats/localization/extractors.py`, plus one dispatch tweak and
timing instrumentation in `geometry/verification.py` and one persisted array in
`localization/localizer.py`'s existing save/load pair. Both consumer seams **already exist**:
`LocalMatcher.match(query, db, hw)` is the reserved NotImplementedError seam
(`extractors.py:154`), and `verify_reconstruction`'s else-branch already calls
`matcher.match(features[i], features[j], hw)` on precomputed features (`verification.py:232`)
— kept for exactly this follow-on. The localizer's descriptor path is the same seam's second
consumer.

### 1. General path first — implement `match()` (the zoo-wide mechanism)

- Applies to any vismatch model whose `extract()` returns non-empty descriptors (~the sparse
  half of the 36 model files; the 18 detector-free models never qualify).
- `match()`: **`kornia.feature.match_mnn`** (kornia 0.8.2 already in the env) over
  L2-normalized descriptors — on unit vectors L2-mutual-NN ≡ cosine-mutual-NN, matching
  xfeat sparse's own `min_cossim=-1` stage. No hand-rolled NN math. Match rows ARE
  keypoint-table indices — no recovery.
- **No extraction and no cache on this path in verify()**: the features come in precomputed
  from the zarr cache the pipeline already builds. Localizer likewise holds zarr features.
- Dispatch in `verification.py`: prefer the `match()` path when
  `matcher.supports_descriptor_matching`; otherwise the existing pairwise `match_images`
  path, unchanged.
- Gating is a **static allowlist** (`_DESCRIPTOR_NN_MODELS = {"xfeat"}`), not a runtime
  probe: membership is licensed by a GPU parity test in the suite proving `match()` equals
  that model's own pair forward exactly. The env is pinned, so test-time enforcement is
  sound; a vismatch bump gets caught by the test. Learned-matcher models stay off the list —
  never silently degraded.

### 2. Loma split — feature-level matching for the learned matcher (byte-identical)

Loma is the shipping default (`localization.matcher: loma`) and its learned match stage
cannot take the NN path by design. Instead its pair forward is split into per-image and
per-pair halves, so features extract once and match many times — the same
extract-once/match-from-store shape as a COLMAP database. Activated statically for the
`LoMaMatcher` wrapper class (`self._split_loma_forward`, one boolean — no registry).

- **`LocalFeatures` gains ONE optional field: `keypoints_normalized`** (default None) — the
  pre-transform coordinates the learned matcher consumes (positional encoding runs on
  these; the pixel-frame `keypoints` cannot serve it, and inverting the affine chain is
  float-inexact). Descriptors are shared with the existing field. No shape metadata needed:
  matched pixel coords come from indexing the already-chained `keypoints` table — indexing
  before vs after an elementwise affine chain is byte-identical.
- **`_loma_detect_and_describe(image) -> LocalFeatures`** — per-image half:
  `matcher.preprocess` + `matcher.matcher.detect_and_describe(img, max_num_keypoints)`,
  then the wrapper's own coordinate chain (`to_pixel_coords`, `rescale_coords`, −0.5) over
  the full table. Run inside `ImportSandbox.get(type(self._matcher).__module__)` +
  `torch.inference_mode()` — vismatch's sandbox only wraps `__init__`/`_forward`
  (`base_matcher.py:27`), so reach-in calls must enter it themselves.
- **`_loma_match_features(f0, f1) -> MatchResult`** — per-pair half: replay the wrapper's
  match stage (`vismatch/im_models/loma.py:78-102`) — learned matcher on
  `keypoints_normalized` + descriptors, `filter_matches`, native indices
  (`torch.where(valid)[0]`, `m0[0][valid]`), pixel coords by indexing `keypoints`.
  Byte-identical to `_forward` by construction, enforced by the GPU parity tests.
  `_recover_indices` skipped. Covers all five LoMa archs (one wrapper class).
- **Routing** (all inside `LocalMatcher`): `extract()` routes to
  `_loma_detect_and_describe` (also faster — skips the self-pair match stage);
  `match()` serves loma when both feature sets carry `keypoints_normalized`;
  `match_images()` composes the two halves through the cache. A `can_match_features(f)`
  method is the single dispatch predicate: allowlisted NN model, or loma with
  payload-bearing features.
- **Persistence — the zarr feature cache grows one array.** `save_index` writes
  `keypoints_normalized` (CSR-aligned float32, same layout as `keypoints`) when present;
  `load_reconstruction_features` restores it. float32 → zarr → float32 is lossless, so the
  parity gate covers the roundtrip. `build_localization_db` already runs
  `detect_and_describe` on every image — this persists what is currently computed and
  thrown away. **Old caches without the array are not backfilled**: `can_match_features`
  returns False and loma falls back to pairwise `match_images` with the in-memory cache —
  still ~7× better than today, and a cache rebuild upgrades them.
- **In-memory cache** (`match_images` path only — localization query images, old caches):
  a plain dict keyed by array identity — `id(image)` lookup + `is` check against the stored
  reference (exact, no hashing; id reuse after GC is a miss, never a wrong hit). Cleared
  wholesale at 512 entries instead of LRU bookkeeping (~2–4 MB/image payloads, ~300/scene,
  worst case ~1–2 GB against the 46.6 GB cap).
- Further learned-matcher models (LightGlue family, sphereglue) get the same treatment
  later only if measurement shows demand; until then they simply run the pairwise path.
  Each vismatch wrapper's `_forward` is different code, so splits are inherently
  per-wrapper — the genuinely general fix is vismatch exposing encode/match halves
  upstream (see Exit strategy).

**Fallback.** Existing `match_images` pair forward, untouched, byte-identical — every model
without a split or allowlist entry, and all detector-free models. The pre-existing
`_probe_index_stability` (construction-time, unchanged by this work) keeps deciding whether
fallback models can serve verification at all.

### 3. Equivalence enforcement — test-time, not runtime

Fast paths are enabled statically (allowlist + the loma boolean) and proven by GPU parity
tests in the suite, on real models:

- loma extract: split `extract()` byte-identical to the self-pair extract it replaces.
- loma match: `match(extract(a), extract(b))` and the `match_images` composed path both
  byte-identical to the plain pair forward (coordinates and indices), including after a
  zarr save/load roundtrip.
- xfeat: `match(extract(a), extract(b))` equal to `match_images(a, b)` exactly.

The env pins vismatch, so an every-construction probe buys nothing over the suite; extending
either list requires a new passing parity test. No runtime demotion machinery.

## Residual attribution (the sub-15-minute work)

The ~658 s residual has never been measured directly (handoff §7.3). As part of this pass:

1. Instrument `verify()` with per-phase timings: matcher/model cold start, DB export
   (`_write_frames` + match writes), `verify_matches`, `triangulate_points`, report writing.
   One tmux run attributes it.
2. Attack the single biggest attributable term if it is ours — expected suspects:
   Python-loop DB export (vectorize or batch the pycolmap Database writes) and cold start
   (load once, not per stage). Upstream C++ phases (`verify_matches`, `triangulate_points`)
   are consumed as-is; if they dominate, record the number and stop — that is the
   pycolmap-native migration's problem, not ours.
3. Numbers append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`.

## Testing

Flat test functions in `tests/localization/test_extractors.py`:

1. **Parity fixtures** (load-bearing, GPU/slow-marked): real loma on the synthetic pair —
   split `extract()` vs self-pair extract, and `match()`/`match_images` vs the plain pair
   forward (flip `_split_loma_forward` for the reference), including a zarr save/load
   roundtrip. Real xfeat: `match(extract(a), extract(b))` equals `match_images(a, b)`
   exactly. These tests are what license the static fast paths — extending either requires
   a new passing parity test.
2. In-memory cache semantics: hit on same array object, miss on equal-content copy, clear
   at capacity.
3. `match()` on a model with no fast path raises NotImplementedError (existing test
   survives); empty-descriptor inputs return `_empty_match()`; loma features without
   `keypoints_normalized` fail `can_match_features` and keep the pairwise path.
4. `save_index`/`load_reconstruction_features` roundtrip `keypoints_normalized` and omit
   it cleanly when absent (mv_* precedent: absent, never zeros).
5. Existing extractor and verification tests pass unchanged (pairwise path is the old
   code); `MagicMock(spec=LocalMatcher)` sites pin `can_match_features.return_value = False`.

## Evidence plan (measured, not assumed)

1. 30-pair stratified harness (commit `755c22d` shape, warm, CUDA-synced) with caching on:
   per-pair ms vs 993.8 ms baseline, split by cache hit/miss.
2. Phase-instrumented full 300-frame `verify()` in tmux (serial, single-A40 rule) vs 2511.6 s:
   confirms the matching term ~250 s AND delivers the residual attribution in the same run.
3. Both append to the measured report.

## Implementation principles

- Reuse: `kornia.feature.match_mnn` for mutual-NN (no hand-rolled matcher); vismatch's
  `ImportSandbox.get` for the loma reach-in; existing `MatchResult`/`LocalFeatures`
  dataclasses. No new module, no new classes — `extractors.py` only (plus timing
  instrumentation in `verification.py`).
- Retire: `_recover_indices` becomes fallback-only; delete outright once no configured model
  needs it.
- Minimal: persistence rides the existing zarr feature cache (one added array — no new
  cache layer, no staleness machinery beyond what save_index already has), no batching,
  no byte-budget accounting, no detector-free reach-in, no vismatch fork.

## Further headroom (recorded, not in scope — implement only if measurement demands)

Stacked realistic floor of this architecture is **~4–6 min**, residual-dominated. The levers,
in payoff order:

1. **Batched pair matching** (loma): the learned matcher broadcasts over the batch dim
   (`einsum`/SDPA/`filter_matches`, handoff §2) — ~71 s → ~25 s. Was 3.5% of baseline;
   becomes ~30% of the post-cache loma matching term.
2. **Pair-count knob**: 1,865 pairs ≈ 6.2/frame is `overlap` config. Halving it halves the
   matcher, `verify_matches`, and match-write terms linearly. Quality trade-off, zero code.
3. **Residual floor**: after cold-start amortization + DB-export vectorization (in scope
   above), what remains is C++ `verify_matches` + `triangulate_points` — that floor belongs
   to the pycolmap-native migration.

Below ~4–6 min the answer is the Exit strategy (pycolmap-native ONNX+CUDA: extraction tens
of ms/image, LightGlue ms/pair → ~2–4 min territory) or fewer frames/pairs.

## Non-goals

- **Batched pair matching**: deferred to Further headroom above.
- **Detector-free adapters** (RoMa separable encoder, LoFTR backbone): large payloads,
  vendored-code surgery, unmeasured payoff (handoff §7.5–7.7).
- **Upstream vismatch PR** (per-model `encode(image)`/`match(enc0, enc1)` halves on
  `BaseMatcher`): the genuinely general fix — every wrapper-specific split here dies the
  day vismatch exposes it. Right long-term home, not our merge timeline; proven paths here
  become its reference implementations.

## Exit strategy

When a stable colmap/pycolmap release ships ONNX (LoMa-B landed in main 2026-08-18 via
PR #4524, wheels pending) — or we rebuild the Docker image's stale `colmap/colmap:20240213.23`
stage from source with `ONNX_ENABLED` + CUDA — `verify()` migrates to pycolmap-native
extract+match against `database.db` (features-once by construction, matcher loop and DB-export
code deleted). This cache then shrinks to localization-only or is deleted. Revisit at the next
colmap release.
