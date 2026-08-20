# Per-image feature caching in LocalMatcher — design

**Date:** 2026-08-20
**Status:** approved design, pre-implementation
**Branch:** `refactor/cu121-uv-migration`
**Supersedes:** `docs/superpowers/handoffs/2026-08-20-matcher-feature-cache-handoff.md` (investigation sketch)

## Problem

`verify()` on the 300-frame GoPro scene (`/workspace/outputs/2026_07_15-Goprosplat-GH010229`,
vggt_omega + loma) takes **2511.6 s**. Measured split (handoff §1, commit `755c22d` harness):

- loma pair matching: ~1853 s (73.8%) — 1,865 pairs × 993.8 ms
- of which `detect_and_describe`: ~1581 s (62.9%)
- residual (DB export, `verify_matches`, `triangulate_points`, cold start): ~658 s upper bound

Per-pair (gap-16, CUDA-synced, warm): `detect_and_describe` ×2 = 78.3%, `_recover_indices` = 8.4%,
learned match transformer = **3.5%**, MAGSAC (discarded) = 0.2%.

**1,865 pairs × 2 extractions for 300 unique images = 12.4× redundancy.** The expensive stage
(detection) is per-image and cacheable; the valuable stage (the learned matcher) is per-pair and
cheap. Cache detection, keep the learned matcher.

Realistic ceiling **~2.4–2.8×** (42 min → ~15–17 min). The handoff's 3.4× included a 148 s
"decode once" term that does not apply: `verification.py:227` passes arrays from an in-memory
list, so the per-pair decode cost existed only in the file-loading harness. The ~658 s residual
does not move regardless.

## Decisions already settled (do not relitigate)

1. **Reach into vismatch internals.** User directive, reaffirmed twice (handoff §10).
2. **Legacy Disk/XFeat/Loma extractor classes are gone** (commits `091ad53..c537803`); vismatch
   `LocalMatcher` is the sole provider. The stale CLAUDE.md architecture line still listing them
   gets fixed in this pass.
3. **pycolmap-native matching is the future, not the present** (see Exit strategy). Verified
   2026-08-20: colmap PR #4524 (LoMa-B, ONNX, with pycolmap bindings) merged to main 2026-08-18,
   but **no PyPI pycolmap wheel ships ONNX support** — 4.0.4 (installed) and 4.1.1 (latest) both
   compile it out (`ALIKED feature extraction requires ONNX support` abort, no bundled
   onnxruntime, `has_cuda=False`). Source builds default `ONNX_ENABLED=ON` with a CUDA execution
   provider (`onnx_utils.cc:77`), so a Docker rebuild unlocks it later.
4. **Learned matchers are the value of loma-class models.** A generic NN match stage may only
   ever serve models whose own matcher *is* NN, proven by probe — never silently substitute for
   a learned matcher.

## Design

Everything lives in `collab_splats/localization/extractors.py`. **Zero call-site changes**:
the cache sits inside `LocalMatcher.match_images`, so `geometry/verification.py` and
`localization/localizer.py` benefit without edits.

### 1. Per-image encode cache

- Keyed by **array identity**: `id(image)` lookup, then an `is` check against the stored array
  reference (exact, no hashing). This works because both consumers hold their images in stable
  in-memory lists for the duration of the loop (`verification.py:227`; localizer query/refs).
  An array that fails the `is` check (id reuse after GC) is a miss, never a wrong hit.
- Bounded LRU by entry count (default 512). Sparse payloads are ~2–4 MB/image → worst case
  ~1–2 GB against the 46.6 GB container cap. Byte-budgeting is deferred until a large-payload
  (dense) adapter exists — none in this pass.
- Cache entries hold the encoded payload plus a reference to the source array (the identity
  anchor; no copy, no extra decode).

### 2. Three-tier dispatch in `match_images`

Ordered most-specific-first; **no tier matched → today's pair forward, byte-identical**.
Coverage grows monotonically; nothing can regress.

**Tier 1 — loma adapter (reach-in, byte-identical, learned matcher kept).**
- Encode: `matcher.preprocess(img)` + `matcher.matcher.detect_and_describe(img, max_num_keypoints)`,
  wrapped in `sandboxed_method(fn, self._matcher.__class__.__module__)` — vismatch's
  `ImportSandbox` only wraps `__init__`/`_forward` (`base_matcher.py:27`), so reach-in calls
  must enter the sandbox themselves. Payload is **pre-pixel-coords**: normalized kpts, desc,
  original shape, resized H×W.
- Match: replay the wrapper's own match stage (`vismatch/im_models/loma.py:78-102`) on two
  cached payloads — learned matcher, `filter_matches`, `to_pixel_coords`, `rescale_coords`,
  the −0.5 COLMAP-convention offset. Byte-identical to `_forward` by construction, enforced by
  the probe.
- Indices: `torch.where(valid)[0]` / `m0[0][valid]` **are** the keypoint-table indices —
  returned natively. `_recover_indices` (90.6 ms/pair) is skipped on this path.
- Covers all five LoMa archs (one wrapper class).

**Tier 2 — generic descriptor-NN adapter (opens the zoo).**
- Applies to any model whose `extract()` returns non-empty descriptors (~the sparse half of
  36 model files; the 18 detector-free models never match this tier).
- Encode: `BaseMatcher.extract()` — public API, runs inside the sandbox naturally, no reach-in.
  One self-pair forward per image, cached (amortizes the 913.4 ms/image cost 12.4×).
- Match: GPU mutual-NN over cached descriptors (cosine, no threshold — mirrors xfeat sparse's
  own `match(..., min_cossim=-1)`). Indices are table rows by construction.
- **Only enabled per model when the probe proves it byte-equivalent** to that model's own pair
  forward. NN-native models (xfeat sparse, handcrafted) pass; learned-matcher models
  (LightGlue family, sphereglue) fail and stay on the fallback until someone writes their
  Tier-1-style adapter (~15 lines each, documented recipe).

**Tier 3 — fallback.** Existing `match_images` pair forward, untouched.

### 3. Construction-time equivalence probe

Extends `_probe_index_stability` (same synthetic shifted-copy fixture, one extra pair of
forwards at init):

- Run the selected fast path and the plain pair forward on the synthetic pair; compare
  `MatchResult` (coordinates and indices) exactly.
- Mismatch → one `logger.warning`, permanent fallback to Tier 3 for this instance. The fast
  path can never silently change results — probe failure costs speed, not correctness.
- The existing index-stability outcome is kept for Tier-3 models; Tier 1/2 return native
  indices, so `has_stable_indices` is bypassed (not deleted — fallback models still need it,
  and `verification.py` still consumes it).

### 4. `LocalMatcher.extract()`

Routes through the same encode cache when a Tier 1/2 adapter is active (loma: convert cached
normalized kpts to pixel frame + descriptors; Tier 2: cached `extract()` output directly).
Fallback models keep the current self-pair `extract()`.

## Testing

Flat test functions in `tests/localization/test_extractors.py`:

1. **Parity fixture** (the load-bearing test): real loma on the synthetic pair — cached-path
   `MatchResult` byte-identical to plain-path (coordinates, indices, count). Same for xfeat
   sparse via Tier 2. Marked GPU/slow as repo convention dictates.
2. Probe behavior: a stub matcher whose fast path diverges → probe demotes to fallback, warning
   logged, results correct.
3. Cache: hit on same array object, miss on equal-content copy (identity semantics), LRU
   eviction at capacity, no cross-instance leakage.
4. Empty-match and empty-descriptor paths return `_empty_match()` / fall back cleanly.
5. Existing extractor tests keep passing unchanged (fallback tier is the old code).

## Evidence plan (measured, not assumed)

1. Re-run the 30-pair stratified harness (commit `755c22d` shape, warm, CUDA-synced) with
   caching on: per-pair ms vs the 993.8 ms baseline, split by hit/miss.
2. One full 300-frame `verify()` in tmux (serial, per the single-A40 rule) vs 2511.6 s.
   Target ~2.4–2.8×.
3. Report appends to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`.

## Implementation principles

- Reuse: `_probe_index_stability` fixture and shape; vismatch's `sandboxed_method`; existing
  `MatchResult`/`LocalFeatures` dataclasses. No new module — `extractors.py` only.
- Retire: `_recover_indices` becomes Tier-3-only; delete it outright if and when Tier 3 is the
  only consumer left and no configured model uses it.
- Minimal: no disk persistence, no batching, no detector-free (RoMa/LoFTR) reach-in, no
  byte-budget LRU, no vismatch fork. Each is listed under Non-goals with its reason.

## Non-goals

- **Disk persistence** of the cache: staleness keys (model/weights/`max_num_keypoints`) and
  footprint are a separate decision; the single-run redundancy is the measured pain.
- **Batched pair matching**: touches only the ~3.5% match term (handoff §2, loma has no
  one-to-many path).
- **Detector-free adapters** (RoMa separable encoder, LoFTR backbone): large payloads,
  per-architecture vendored-code surgery, unmeasured payoff (handoff §7.5–7.7).
- **Upstream vismatch PR** (`extract_features`/`match_extracted` on `BaseMatcher`): the right
  long-term home; not on our merge timeline. Proven adapters here become its reference
  implementations.

## Exit strategy

When a stable colmap/pycolmap release ships ONNX (LoMa-B landed in main 2026-08-18 via
PR #4524, wheels pending) — or we rebuild the Docker image's stale `colmap/colmap:20240213.23`
stage from source with `ONNX_ENABLED` + CUDA — `verify()` migrates to pycolmap-native
extract+match against `database.db` (features-once by construction, matcher loop deleted).
This cache then shrinks to localization-only or is deleted. Revisit at the next colmap release.
