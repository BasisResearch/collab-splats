# VisMatch as the Stage-2 Local Matcher — Design

**Date:** 2026-08-17
**Status:** Approved design, pending implementation plan
**Supersedes (on completion):** `2026-07-08-loma-matcher-integration-design.md` (in-flight LoMa port)

## Problem

`collab_splats/localization/extractors.py` hand-maintains five local-matcher wrappers
(`DiskExtractor`, `XFeatExtractor`, `XFeatStarExtractor`, `LomaExtractor`, `LomaGExtractor`),
each with its own model-loading, weight-fetching, and quirk handling. Every new matcher
(e.g. the in-flight LoMa port) is a fresh integration project.
[vismatch](https://github.com/gmberton/vismatch) (BSD-3 wrapper, actively maintained, 903★)
wraps 50+ matching models behind one interface and absorbs that maintenance upstream.

## Decision

vismatch becomes the **sole Stage-2 (local matching) model provider**. Stages 1 and 3 stay ours:

- **Stage 1 — retrieval** (`retrieval.py`: DinoSalad/PECLIP): unchanged. Also loop closure's
  retrieval gate; vismatch has no global-descriptor capability.
- **Stage 2 — local matching**: new single class `LocalMatcher` wrapping vismatch.
- **Stage 3 — pose** (`localizer.py`: world_points 2D→3D + pycolmap LO-RANSAC PnP): unchanged.
  vismatch outputs pixel-pair matches and a homography only — it has no pose/3D capability,
  which is why it cannot "replace the localization module" wholesale.

`LocalFeatures` and `MatchResult` dataclasses survive as the backend-neutral pipeline contract
(consumed by `localizer.py`, `geometry/verification.py`, `viz.py`, dashboard, evals).

## Install route (measured, 2026-08-17)

Stock `uv add vismatch` is unsatisfiable: every vismatch release hard-pins
`uniception==0.1.1` (for its `ufm` model) while we pin `uniception==0.1.7`
(MapAnything's network is built from it; 15 of the 42 symbols MapAnything imports do not
exist in 0.1.1 — verified by AST check against the 0.1.1 wheel). Fix, two lines in our
`pyproject.toml`:

```toml
[tool.uv]
override-dependencies = ["uniception==0.1.7", "lightning>=2.6"]
```

- Verified: resolves clean (vismatch 1.3.1, uniception 0.1.7, lightning 2.6.5, kornia 0.8.2).
- Semantics: we assert vismatch works with these versions. True for all models **except**
  `ufm` (needs uniception 0.1.1 API) and `edm` (needs lightning 2.3.3) → both go in the
  adapter blocklist with a clear error message.
- kornia settles at 0.8.2 (vismatch's `<0.8.3` ceiling; our tree already resolves 0.8.2 on
  other markers — LoFTR-family models stay usable).
- vismatch pinned to an exact version in pyproject (solo-maintainer upstream; upgrades are
  deliberate, not incidental).
- Non-blocking follow-up: upstream PR moving their `uniception`/`lightning` pins into
  optional extras (their tensorflow/sphereglue extras are the precedent); override shrinks
  when merged.

## Architecture

### `LocalMatcher` (extractors.py)

One concrete class — role-named, not vendor-named. No registry: model diversity lives in
vismatch's `get_matcher` namespace, not in our class hierarchy.

```python
matcher = LocalMatcher("superpoint-lightglue", device="cuda")
feats: LocalFeatures = matcher.extract(image)              # -> zarr cache, unchanged
match: MatchResult   = matcher.match(query_img, ref_img)   # pairwise, image-in
```

- **`extract(image)`** → vismatch `matcher.extract()` → `LocalFeatures(keypoints, descriptors)`.
  Feeds the existing `feedforward.zarr` `local_features/{model}/reconstruction` CSR cache
  unchanged (keypoints for verify + 2D→3D sampling). Note: vismatch `extract()` is
  implemented as `forward(img, img)` (self-match) — DB build for N refs costs N full pair
  inferences; acceptable, offline.
- **`match(query_img, ref_img)`** → vismatch `matcher(q, r)`, taking **pre-RANSAC**
  `matched_kpts0/1` → `MatchResult(query_px, ref_px)`. vismatch's homography RANSAC is
  skipped by design: homography inliers assume planar scenes / pure rotation — wrong filter
  for 3D localization. Our PnP LO-RANSAC is the geometric filter.
- **Index recovery for verify:** at init, probe the model with a bundled test pair — every
  matched keypoint must be an exact row of `all_kpts`. Pass → `match()` fills
  `idx_q`/`idx_db` by coordinate→row lookup (COLMAP keypoint-table indices). Fail →
  indices stay `None` (existing XFeatStar semantics; verify skips those pairs with no new
  code in `verification.py`).
- **Blocklists:** (a) dependency blocklist — `ufm`, `edm` raise with the reason;
  (b) license blocklist — models whose upstream license is non-commercial raise with the
  license name (table sourced from vismatch `docs/source/model_details.md`; basis.ai is a
  commercial user).
- **Coordinate-frame guard:** vismatch models resize internally per-model. `LocalMatcher`
  asserts returned keypoints are in the pixel frame of the *input* image (bounds check +
  a fixture test with a known correspondence per supported default model). This is the
  92f2e4a bug class (full-res index vs model-res grid) — the historically bug-prone spot.

### Localizer pairwise path (localizer.py)

vismatch has **no descriptor-level matching** — the only path is `forward(img0, img1)`.
So `CameraLocalizer.localize` matches the query image against each **top-K retrieved ref
image** (loaded from `frames.zarr`), K from the existing retrieval stage. Consequences:

- Per-query cost: K model forward passes (refs re-encoded each query). Accepted regression
  vs today's cached-descriptor LightGlue match; bounded by retrieval top-K (≈5–10).
- Matched ref pixels feed `sample_world_points` directly (pixel-based 2D→3D) — works even
  for dense matchers with no stable keypoints.
- The zarr feature cache remains for the verify stage's COLMAP DB export (keypoints +
  indices), not for match-time descriptor reuse; 2D→3D uses live matched pixels.

### Verify stage interaction

Config choice must not silently degrade a different stage: if
`pointcloud.geometric_verification: true` and the configured model fails the index probe,
**hard error at pipeline start** naming the model and the constraint — not a silent skip,
not a missing `verification.json` discovered later.

### Config surface

```yaml
localization:
  enabled: true
  matcher: superpoint-lightglue   # any non-blocklisted vismatch model name
```

`localization.extractor` is replaced by `localization.matcher`. One key. zarr cache path
keyed by model name (`local_features/{model}/…`); old `disk`/`xfeat`/`loma` cache entries
are orphaned — harmless (missing-model `KeyError` names the model; rebuild via `localize`
stage).

## Migration (staged, parity-gated)

1. **Land**: uv override + vismatch exact-pin PyPI dep; `LocalMatcher` added **alongside**
   existing extractors (registry untouched during transition).
2. **Parity gate** (compute, human-gated): benchmark `LocalMatcher` with
   `superpoint-lightglue`, `disk-lightglue`, `xfeat`, `loma` against in-repo baselines —
   disk 3314/2099, xfeat 2603/1211 correspondence/inlier counts, nb07 tutorial scene
   end-to-end. Also verify-stage e2e with one index-stable model.
3. **Retire** (only after parity): delete `DiskExtractor`, `XFeatExtractor`,
   `XFeatStarExtractor`, `LomaExtractor`, `LomaGExtractor`, and the `BaseLocalExtractor`
   ABC + registry (a registry with one member is dead weight). Type hints in
   `verification.py`/`localizer.py` move to `LocalMatcher`. Mark the loma-matcher spec
   superseded. **Caveat:** vismatch's `loma` is a different fork lineage than the spec'd
   port — "supersedes" holds only if the parity gate says so; loma calibration numbers
   reset regardless.
4. **Weights pre-fetch**: setup step downloads default-model checkpoints (vismatch fetches
   at first use via gdown/hf_hub — remote GCS runs and tmux evals must not hit the network
   mid-run).

## Testing

Flat test functions (no classes):

- `LocalMatcher` contract: `LocalFeatures`/`MatchResult` shapes and dtypes (mocked
  `get_matcher`).
- Index probe: positive (synthetic matcher returning exact rows), negative (perturbed
  coordinates → indices `None`).
- Blocklist: `ufm`/`edm`/non-commercial model names raise with reason.
- Localizer pairwise path with a mocked matcher; verify-skip on `idx=None`; hard error on
  `geometric_verification: true` + index-incapable model.
- Coordinate-frame fixture: known-correspondence image pair, assert pixel-frame keypoints
  for the shipping default model (real model, marked slow/GPU).
- Existing localization/verification tests pass unmodified until the retirement task.

Mocks cannot catch per-model coordinate bugs — that is what the parity gate (real compute)
is for.

## Risks accepted

- **Query latency** up (K forwards vs cached-descriptor match) — dashboard-visible.
- **~45 of 50+ models usable**: `ufm`/`edm` (deps), non-commercial licenses, dense models
  verify-incompatible (localization-only).
- **Solo-maintainer upstream** — exact-version pin; vendoring remains the escape hatch.
- **Determinism**: some matchers are non-deterministic; verification metrics may jitter.
- **Calibration reset**: all localization baselines re-measured under vismatch models.

## Implementation principles

Reuse `LocalFeatures`/`MatchResult`/zarr cache/verify contracts as-is; delete the five
in-repo wrappers and the registry once parity is proven; no new abstraction layers — one
class, one config key. Nothing in `retrieval.py`, `viz.py`, or Stage 3 PnP changes except
type hints.
