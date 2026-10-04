# vismatch feature matching — batched extract, lean match, collab-splats adoption

**Date:** 2026-10-03
**Status:** approved design; plan 2026-10-03-vismatch-feature-matching.md
**Supersedes:** steps 1b, 4 and 5 of [2026-09-25-vismatch-fork-design.md](2026-09-25-vismatch-fork-design.md)
**Upstream issue:** https://github.com/gmberton/vismatch/issues/74
**Upstream base:** `gmberton/vismatch` main @ `9d49b89` (v1.3.2); collab-splats runs site-packages `vismatch==1.3.1`

## Problem

- Track building and localization pay for work vismatch throws away:
  - `extract()` = `forward(img, img)`: 2 detects, a self-match, RANSAC on ~4k self-matches
  - one image per detect; no batched extract
  - input goes CPU float → `to_tensor_image` → `.to`; conversion is half the batched extract cost
- collab-splats works around it locally:
  - `_split_loma_forward`: reaches into the wrapper module, replays the coord chain, enters `ImportSandbox` by hand
  - `FEATURE_MATCH_MODELS` + kornia `match_mnn` for xfeat feature matching
  - `match_batch` (rgbd-ba worktree only, uncommitted)
- Owners' position on #74 (alexstoken, 2026-08-29):
  - batching: yes — `forward()` takes batched img0/img1, returns a list of output dicts; gated by `self.supports_batches`
  - caching: no plans
  - extract/match split: concern for detector-free models
  - COLMAP export: wanted, write first
- Our reply (2026-08-31): split only for descriptor models (LoMa, XFeat); others fall back.

## Goals

- Keep every measured gain (table below), with the least code.
- Write inside the fork in upstream style (ruff, py3.10, their docstrings, flat pytest).
- Contribute upstream where the owners agreed; never block collab-splats on review.
- Don't remove original vismatch code unless needed; add hooks, don't rewrite models.

## Measured levers

All on gh1k frames (3×384×688 uint8) unless noted; scripts in the session scratchpads.

| Lever | Before | After | Source |
|---|---|---|---|
| xfeat `extract` 640×480, B=32 | 132.5 ms/frame | 14.7 ms/frame | handoff `vm_prof2.py` |
| xfeat `extract` full-res GH | 40.3 ms/frame | 4.8 ms/frame | handoff `prof_xfeat_batch_forward.py` |
| xfeat `detectAndCompute` per-image vs B=32 | 6.12 ms/frame | 0.87 ms/frame (7.0×) | `bench_xfeat_extract.py` |
| input: CPU float + `.to` vs uint8 H2D + GPU convert | 8.34 ms/frame | 0.19 ms/frame | `bench_xfeat_extract.py` |
| LoMa pair: plain forward vs split match | 1033.5 ms/pair | 44.1 ms/pair | handoff |
| xfeat match: old `LocalMatcher.match` (kornia) | 5.23 ms/pair | 1.87 ms/pair (per-pair `xf.match`, GPU features) | `bench_match_paths.py` |

### Match batching is not a lever

Same run, same descriptors (32 pairs, 4096 kpts):

| Path | ms/pair |
|---|---|
| old `LocalMatcher.match` (CPU features, kornia `match_mnn`) | 5.23 |
| rgbd-ba `match_batch` (CPU features) | 3.12 |
| per-pair `xf.match` (CPU features) | 2.83 |
| per-pair `xf.match` (GPU features) | 1.87 |
| `match_batch` (GPU features) | 1.87 |

- `match_batch`'s 95 → 54 s came from dropping kornia overhead and per-pair syncs, not from batching.
- Features resident on GPU is the lever; per-pair `xf.match` is bit-exact and as fast.
- Padded batch variants (upstream `batch_match`, own bmm, cdist 1e3) measured: none worth the code.

## Design

### 1. Branch plan (fork `BasisResearch/vismatch`, `/workspace/vismatch`)

```
upstream/main
 ├─ feat/fast-input        ──► upstream PR (uint8 branch, 2 lines)
 └─ feat/batch-forward (1a) ──► upstream PR (supports_batches, loop)
     └─ feat/feature-matching ──► draft upstream PR (hooks, extract/match, xfeat + loma)
         └─ feat/colmap-export ──► upstream PR
basis  ◄── merge-only: all of the above; collab-splats pins a basis SHA
```

- `feat/batch-forward` @ `ceb0022` is brought to spec: add `self.supports_batches = False`; B mismatch stays an `assert` (decided).
- Post #74 with the measured numbers when the batch PR opens; the split goes up as a draft.
- Dropped: 1b LightGlue, step 5 (feature-cache DB — owners have no plans for caching), xfeat-star.

### 2. vismatch changes

| Change | Original code touched? | ~Lines |
|---|---|---|
| `self.supports_batches = False` in `BaseMatcher.__init__`; batched `forward`: `supports_batches` → `extract(list)` ×2 + per-pair `match()` with `all_*` merged back in, else 1a per-pair loop | no (postprocess not moved) | +11 |
| uint8 branch in `to_tensor_image` | no | +2 |
| `_extract_features` added to the sandbox-wrapped tuple; `_match_features` stays outside (entering the sandbox scans `sys.modules`: 21–36 ms per call vs ~0.3 ms xfeat match) | no | +2 |
| `extract(list)` uses the hooks when `supports_batches`; keys bit-exact with today's `extract`; adds `image_size` (W, H) and model extras | one branch | +8 |
| Lean `match(feats0, feats1)`: gather, valid mask, `compute_ransac`, `matched_idxs0/1`; same keys as `forward` minus `all_*`; single pair only; inputs via `torch.as_tensor(..., device=self.device)` | no | +18 |
| xfeat hooks: stack + one `detectAndCompute` when sizes match, else per image; per-pair `self.model.match(min_cossim=-1)`; `supports_batches = mode == "sparse"` | no | +12 |
| loma hooks: per-image `detect_and_describe`, model-grid kpts kept as an extra; per-pair matcher + `filter_matches`; `supports_batches = True` | no | +25 |

- Native batched `forward` reuses `match()` for valid mask + RANSAC → no original code moves; batched forward gets the batched-extract gain.
- 1a PR carries the attribute + loop only; the `supports_batches` branch lands in `feat/feature-matching` with `extract(list)` / `match()`.
- `match()` stays single-pair: batched matching measured no faster (see above).
- Model `_forward` stays untouched → single-pair contract kept for `EnsembleMatcher` (`base_matcher.py:280`) and `keypt2subpx.py:65`.
- Hook names avoid `rdd.py:201` `RDD_ThirdPartyMatcher._extract` (different contract).
- Audit (AST over `vismatch/`, 61 `BaseMatcher` subclasses): none defines `forward` / `extract` / `match` / `supports_batches` / the hook names; all call `super().__init__`.
- `image_size` is (W, H), matching internal use in xfeat-lighterglue, kornia, gim.
- LoMa kpts/desc are fp32 (forward's to_numpy would raise otherwise); only confidences get .float(), as in _forward
- PR notes:
  - batched xfeat extract is not bit-exact at B>1 (keypoint overlap ≥ 0.999; B=1 bit-exact)
  - LoMa `match()` applies the valid mask exactly as `forward` does (drops −0.5 kpts)

### 3. collab-splats adoption

| Change | Original code touched? | ~Lines |
|---|---|---|
| `pyproject.toml`: `vismatch @ git+https://github.com/BasisResearch/vismatch@<basis-sha>`; drop `lomatch` (and its only dependents tyro, typeguard from the lock); `setuptools<70` override (vismatch declares `>=83` for packaging only; maskclip_onnx needs `pkg_resources`) | dep lines | ±6 |
| `LocalMatcher.__init__`: `skip_ransac = True` | one line | +1 |
| `_to_tensor`: uint8 upload, no `t.max()` sync | small change | ~3 |
| `extract`: one image or a list → vismatch `extract`; delete `_split_loma_forward`; loma extra fills `keypoints_normalized`; store `image_size`; image_size at load from the zarr hw attr | replaces the split | −40 / +15 |
| `match`: vismatch `match()`, features moved to `self.device` (no-op when resident); delete `FEATURE_MATCH_MODELS` and kornia `match_mnn` | replaces | −30 / +10 |
| Keep `match_images`, `_recover_indices`, `_probe_index_stability`, zarr layout (+ one `image_size` attr) | none | 0 |
| Tests: drop `FEATURE_MATCH_MODELS` tests; keep the loma zarr round-trip parity test via vismatch; add list-`extract` test | tests | ~±30 |
| Docs: `docs/known-test-failures.md` (~:104–117, :332), localization page | docs | small |

- `LocalMatcher.match` has no production caller today: `localizer.py:909` routes every `LocalMatcher` to `match_images`. Users are the track prototype and tests; the zarr `keypoints_normalized` layout is kept to limit churn.
- `match_images` + `_recover_indices` stay: the localizer pairwise path needs them.
- `skip_ransac` is safe: no caller reads vismatch RANSAC output; the localizer uses pycolmap `num_inliers`.
- `lomatch`: collab-splats never imports it; vismatch loads its vendored `third_party/LoMa/src/loma` inside the sandbox, but a bare `import loma` resolves to pip `lomatch` — dead dep and a shadowing risk. Drop it, then re-run the loma round-trip parity test.
- Track builder keeps features on GPU and uploads once per frame (`bench_match_paths.py`: CPU-held features cost 2.83 vs 1.87 ms/pair).
- rgbd-ba worktree's uncommitted `match_batch` will conflict on rebase; user-owned, not touched.

## Gates

### vismatch (every fork PR)

- Upstream CI exactly: `ruff check .`, `ruff format --check .`, `pytest tests -vv -rs --timeout=300`; CPU-only, python 3.10.
- New tests, flat functions with upstream-style docstrings and the `device` fixture:

| Test | Models | Pass |
|---|---|---|
| `test_to_tensor_image_uint8` | — | bit-exact vs float path |
| mock _GridMatcher tests: extract/match/native forward/out-of-bounds | — | exact |
| `test_forward_batch_native` | xfeat, loma | match overlap ≥ 0.99 |
| `test_extract_matches_forward` | xfeat, loma | bit-exact vs old `extract` keys |
| `test_match_matches_forward` | xfeat, loma, `skip_ransac` | bit-exact; `matched_kpts == all_kpts[matched_idxs]` |
| `test_extract_batch` | xfeat, same-size `test_images` | B=1 bit-exact; B>1 overlap ≥ 0.99 |
| `test_match_features_not_sandboxed` | xfeat, loma | `_extract_features` wrapped, `_match_features` not |
| `test_match_not_implemented` | _CornerMatcher mock | raises `NotImplementedError` |

### Performance (public API, GPU, gh1k frames)

- input conversion ≤ 0.5 ms/frame
- xfeat `extract(list)` B=32: ≤ 16.2 ms/frame at 640×480, ≤ 5.3 ms/frame full-res
- xfeat `match()` with GPU features: ≤ 1× old `LocalMatcher.match` (5.23 ms/pair); measured 0.565 ms/pair, 1.45× the raw `xf.match` replica of `match_batch` (0.389)
  - original ≤ 1.1× `match_batch` gate was miscalibrated: it timed raw `xf.match`, not `match()`'s gather/valid-mask/result-dict work (~0.25 ms/pair); accepted 2026-10-04, the track-build gate decides
- loma `match()` with `skip_ransac`: ≤ 1.1× clean/final `LocalMatcher.match` loma split (handoff 44.1 ms/pair), same run
- ratios, not absolutes, for match: absolute ms drift between runs (0.77 vs 1.87 ms/pair same kernel)
- track build on gh1k: per-phase report; `extract` ≤ 15 s, `match.match` ≤ 28 s; filter / db / verify reported only

### collab-splats (every basis SHA bump)

- `tests/localization` + `tests/geometry`; no new failures beyond `docs/known-test-failures.md`
- print `direct_url.json` `commit_id` (the basis SHA) and `collab_splats.__file__` — `vismatch.__file__` does not prove the pin

## Out of scope

- promoting the track builder into repo code (own spec)
- `_recover_indices` cost on the localizer pairwise path (90.6 ms/pair)
- Docker image rebuild
- feature-cache DB (old step 5), LightGlue native batching (old 1b), xfeat-star
- padded batched matching
