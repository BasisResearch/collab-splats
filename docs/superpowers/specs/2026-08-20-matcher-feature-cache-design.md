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
- **Loma (learned matcher, shipping default):** in-memory cache — 300 extractions
  × ~424 ms ≈ 127 s, learned matcher 1,865 × 38 ms ≈ 71 s, H2D ≈ 25 s →
  **matching term ~1853 s → ~250 s**.

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
   serve models whose own match stage *is* descriptor-NN, proven equivalent by probe — never
   silently substitute for a learned matcher.

## Design

Everything lives in `collab_splats/localization/extractors.py`, plus one dispatch tweak and
timing instrumentation in `geometry/verification.py`. Both consumer seams **already exist**:
`LocalMatcher.match(query, db, hw)` is the reserved NotImplementedError seam
(`extractors.py:154`), and `verify_reconstruction`'s else-branch already calls
`matcher.match(features[i], features[j], hw)` on precomputed features (`verification.py:232`)
— kept for exactly this follow-on. The localizer's descriptor path is the same seam's second
consumer.

### 1. General path first — implement `match()` (the zoo-wide mechanism)

- Applies to any vismatch model whose `extract()` returns non-empty descriptors (~the sparse
  half of the 36 model files; the 18 detector-free models never qualify).
- `match()`: GPU mutual-NN over the given descriptors (cosine, `min_cossim=-1` — mirrors
  xfeat sparse's own match stage). Match rows ARE keypoint-table indices — no recovery.
- **No extraction and no cache on this path in verify()**: the features come in precomputed
  from the zarr cache the pipeline already builds. Localizer likewise holds zarr features.
- Dispatch in `verification.py`: prefer the `match()` path when the probe proved the model
  descriptor-NN-equivalent (a capability flag on `LocalMatcher`); otherwise the existing
  pairwise `match_images` path, unchanged.
- **Enabled per model only when the construction-time probe proves it equivalent** to that
  model's own pair forward (below). NN-native models (xfeat sparse, handcrafted) pass;
  learned-matcher models fail and keep the pairwise path — correct, never silently degraded.

### 2. Per-image encode cache — learned-matcher adapters only

For models whose match stage is learned (probe-fails the general path), the win is caching
the per-image encode inside `match_images`:

- Keyed by **array identity**: `id(image)` lookup, then an `is` check against the stored array
  reference (exact, no hashing). Both consumers hold their images in stable in-memory lists for
  the duration of the loop (`verification.py:227`; localizer query/refs). An array that fails
  the `is` check (id reuse after GC) is a miss, never a wrong hit.
- Capped dict, evict oldest (default 512 entries). Sparse payloads are ~2–4 MB/image → worst
  case ~1–2 GB against the 46.6 GB container cap. No byte accounting.

**Specialization — loma adapter (byte-identical, learned matcher kept).**
Loma is the shipping default (`localization.matcher: loma`) and its learned match stage fails
the general path by design, so it gets the one model-specific adapter:
- Encode: `matcher.preprocess(img)` + `matcher.matcher.detect_and_describe(img,
  max_num_keypoints)`, wrapped in `sandboxed_method(fn, type(self._matcher).__module__)` —
  vismatch's sandbox only wraps `__init__`/`_forward` (`base_matcher.py:27`), so reach-in calls
  must enter it themselves. Payload is **pre-pixel-coords**: normalized kpts, desc, original
  shape, resized H×W.
- Match: replay the wrapper's own match stage (`vismatch/im_models/loma.py:78-102`) on two
  cached payloads — learned matcher, `filter_matches`, `to_pixel_coords`, `rescale_coords`,
  the −0.5 COLMAP-convention offset. Byte-identical to `_forward` by construction, enforced by
  the probe. Native indices (`torch.where(valid)[0]`, `m0[0][valid]`) — `_recover_indices`
  skipped. Covers all five LoMa archs (one wrapper class).
- Why not the zarr features: they store pixel-frame keypoints; the learned matcher consumes
  pre-transform normalized coordinates. Inverting the affine chain is float-inexact and would
  break byte-parity — the in-memory cache stores the pre-transform payload instead.
- Further learned-matcher models (LightGlue family, sphereglue) get the same treatment later
  only if measurement shows demand; the probe keeps them correct meanwhile.

**Fallback.** Existing `match_images` pair forward, untouched, byte-identical — any model the
probe rejects, and all detector-free models.

### 3. Construction-time equivalence probe

Extends `_probe_index_stability` (same synthetic shifted-copy fixture, one extra pair of
forwards at init):

- Run the selected fast path and the plain pair forward on the synthetic pair; compare
  `MatchResult` exactly (coordinates and indices).
- Mismatch → one `logger.warning`, permanent fallback for this instance. Probe failure costs
  speed, never correctness.
- The existing index-stability outcome is kept for fallback models; fast paths return native
  indices, bypassing `has_stable_indices` (retained — `verification.py` still consumes it for
  fallback models).

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

1. **Parity fixture** (load-bearing): real loma on the synthetic pair — cached-path
   `MatchResult` byte-identical to plain-path. For xfeat sparse: `match(extract(a),
   extract(b))` equals `match_images(a, b)` exactly (the general path). GPU/slow-marked
   per repo convention.
2. Probe demotion: stub matcher whose fast path diverges → fallback + warning, results correct.
3. Cache semantics: hit on same array object, miss on equal-content copy, eviction at capacity.
4. Empty-match / empty-descriptor paths fall back cleanly.
5. Existing extractor tests pass unchanged (fallback is the old code).

## Evidence plan (measured, not assumed)

1. 30-pair stratified harness (commit `755c22d` shape, warm, CUDA-synced) with caching on:
   per-pair ms vs 993.8 ms baseline, split by cache hit/miss.
2. Phase-instrumented full 300-frame `verify()` in tmux (serial, single-A40 rule) vs 2511.6 s:
   confirms the matching term ~250 s AND delivers the residual attribution in the same run.
3. Both append to the measured report.

## Implementation principles

- Reuse: `_probe_index_stability` fixture and shape; vismatch's `sandboxed_method`; existing
  `MatchResult`/`LocalFeatures` dataclasses. No new module — `extractors.py` only (plus timing
  instrumentation in `verification.py`).
- Retire: `_recover_indices` becomes fallback-only; delete outright once no configured model
  needs it.
- Minimal: no disk persistence, no batching, no `extract()` rerouting (zarr already caches
  extraction across runs), no byte-budget accounting, no detector-free reach-in, no vismatch
  fork.

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

- **Disk persistence** of the cache: staleness keys (model/weights/`max_num_keypoints`) are a
  separate decision; single-run redundancy is the measured pain.
- **Batched pair matching**: deferred to Further headroom above.
- **Detector-free adapters** (RoMa separable encoder, LoFTR backbone): large payloads,
  vendored-code surgery, unmeasured payoff (handoff §7.5–7.7).
- **Upstream vismatch PR** (`extract_features`/`match_extracted` on `BaseMatcher`): right
  long-term home, not our merge timeline. Proven paths here become its reference
  implementations.

## Exit strategy

When a stable colmap/pycolmap release ships ONNX (LoMa-B landed in main 2026-08-18 via
PR #4524, wheels pending) — or we rebuild the Docker image's stale `colmap/colmap:20240213.23`
stage from source with `ONNX_ENABLED` + CUDA — `verify()` migrates to pycolmap-native
extract+match against `database.db` (features-once by construction, matcher loop and DB-export
code deleted). This cache then shrinks to localization-only or is deleted. Revisit at the next
colmap release.
