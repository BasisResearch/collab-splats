# vismatch fork — branch topology and PR ordering

**Date:** 2026-09-25
**Status:** approved design (umbrella), pre-implementation
**Upstream issue:** https://github.com/gmberton/vismatch/issues/74
**Upstream base:** `gmberton/vismatch` main @ `9d49b89` (v1.3.2); collab-splats pins `vismatch==1.3.1`

## Problem

collab-splats works around missing vismatch features locally:

- `collab_splats/localization/extractors.py` `_split_loma_forward`: reaches into
  `sys.modules[type(m).__module__]`, calls `m.matcher.detect_and_describe`, replays the coord
  chain, enters `ImportSandbox` by hand — env-pinned, only parity tests guard it
- `extract()` upstream = `forward(img, img)`: full self-pair forward, match stage wasted
- no match indices upstream → `_recover_indices` + `_probe_index_stability` rebuild COLMAP
  keypoint rows by exact float equality
- no batching → one pair per forward
- no COLMAP output → `geometry/verification.py` writes its own DB

Goal: move these into vismatch with the fewest deviations from upstream, while giving
collab-splats the most flexible integration.

## Maintainer stance (issue #74, alexstoken 2026-08-29)

| Item | Stance | Constraint |
|---|---|---|
| Batch inference | yes | via `forward()` only; `supports_batches` property; BaseMatcher loops for non-native models; caller never worries about input shape |
| COLMAP output | wanted | write first, not read; a stale branch exists (not public on upstream) |
| extract/match split | skeptical, asked for vision | detector-free models (LoFTR-style) break consistency |
| Feature caching | no plans | only if elegant and small |

Our 2026-08-31 reply (split limited to descriptor models, caching after batching) is unanswered
as of 2026-09-25.

## Branch topology

```
upstream/main ─┬─ feat/batch-forward ──► upstream PR  (1a base, then 1b per-model, stacked)
               ├─ feat/colmap-export ──► upstream PR  (parallel with batch)
               └─ basis  ◄── merges both, then #4 split and #5 cache/DB
                     ▲
            collab-splats pins a basis SHA
```

- repo: fork `BasisResearch/vismatch`, cloned `--recursive` at `/workspace/vismatch`
- remotes: `origin` = fork, `upstream` = gmberton
- feature branches cut from `upstream/main`, never from `basis` or each other (except 1b on 1a,
  #5 on #4)
- `basis`: merge-only, no direct feature commits; syncs by merging `upstream/main`, never
  rebased, so pinned SHAs stay reachable
- deviation from upstream = exactly the unmerged feature branches; shrinks as PRs land

## Order

### 0. Fork setup

- fork exists: https://github.com/BasisResearch/vismatch (even with upstream `9d49b89` on 2026-09-25)
- clone, remotes, record baseline gate on clean `upstream/main`

### 1. `feat/batch-forward` (upstream PR)

1a — base only:
- `BaseMatcher.forward()` accepts a single pair (as today) or a batch: `(B,3,H,W)` tensor/array,
  or lists of paths / PIL images
- unbatched in → dict (unchanged); batched in → `list[dict]` of length B, same keys
- pairwise only: B0 must equal B1, else ValueError
- `supports_batches: bool = False` class attribute; False → BaseMatcher loops `_forward` per pair
- no model file changes; existing tests pass unmodified
- new tests: shape normalization, loop output == per-pair output, B mismatch raises

1b — one stacked PR per model: loma, xfeat, lightglue (in that order)
- set `supports_batches = True`, batched `_forward`
- test: native batched output == looped output

### 2. `feat/colmap-export` (upstream PR, parallel with 1)

- new module writing COLMAP `database.db` via pycolmap; optional extra `vismatch[colmap]`
- per-image keypoint table + index-pair matches (+ optional two-view geometry)
- detector-based models: indices from `all_kpts`
- detector-free models: grid-quantized keypoint merge per image (hloc approach)
- consumes single-pair `forward()`; switching to batched forward is a follow-up only if
  maintainers want it
- before building: ask alexstoken on #74 for the stale branch, reuse over duplicate

### 3. `basis` + collab-splats pin

- `basis` = `upstream/main` + merge `feat/batch-forward` + merge `feat/colmap-export`
- collab-splats `pyproject.toml`:
  `vismatch @ git+https://github.com/BasisResearch/vismatch@<basis-sha>`
- local dev: editable install of `/workspace/vismatch`
- first check: uv git install pulls submodules recursively (vismatch vendors 30+ in
  `third_party/`); fallback = build and pin a wheel

### 4. `feat/extract-match-split` (off `basis`; upstream PR only after maintainer ack)

- opt-in `supports_feature_matching` property + `match(feats0, feats1)` for descriptor models
  (loma, xfeat first); all others keep the current path
- `extract()` skips the self-pair forward for opted-in models
- returns match indices natively
- collab-splats: delete `_split_loma_forward`, `_recover_indices`, `_probe_index_stability`;
  existing parity tests in `tests/localization/test_local_matcher.py` are the acceptance gate

### 5. `feat/feature-cache-db` (off `basis` after 4)

- match new images against keypoints + descriptors stored in a COLMAP DB (from #2)
- exhaustive matching over N images: O(N) extractions, not O(N²)
- stays on `basis` unless maintainers want it

## Gates

- every vismatch branch: upstream CI gate — `ruff check .`, `ruff format --check .`,
  `pytest tests -rs --timeout=300` — no new failures vs the step-0 baseline
- every `basis` SHA bump in collab-splats: `tests/localization` + `tests/geometry` pass, run
  with the `collab_splats.__file__` proof line (worktree PYTHONPATH trap)
- upstream style, not collab-splats style, inside the fork: ruff, py3.10, their docstring shape

## Out of scope

- reading COLMAP formats (maintainers asked for write first)
- exhaustive (all-vs-all) batch mode — pairwise only
- per-model native batching beyond loma / xfeat / lightglue

## Follow-ups

- one spec + plan per branch, starting with 1a
- post the ordering on #74 once 1a PR is open
