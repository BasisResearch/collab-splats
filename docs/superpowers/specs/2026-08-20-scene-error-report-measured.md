# Scene error report — measured numbers

## Task 1: epipolar cost (verify)

- Scene / frames / backbone / matcher: `/workspace/outputs/2026_07_15-Goprosplat-GH010229`
  (GoPro `GH010229.mp4`, sampled at `fps: 2.0`) / **300 frames** / `vggt_omega` / `loma`
- Wall clock: **2511.6 s** (41.9 min) for `verify()` alone
- Peak rss: **9.26 GB** / 46.6 GB (baseline before the run 2.97 GB → **+6.29 GB** attributable)
- Pairs generated: **1,865** (`summary.n_pairs`, and `len(pair_stats)` agrees)
- Per-pair: **1346.7 ms**
- Extrapolated to 300 frames (~5,400 pairs at window=10): **not an extrapolation — this scene
  IS 300 frames, so 41.9 min / 1,865 pairs is the measured 300-frame number.** Both halves of
  the plan's estimate were wrong: the pair count is 1,865 not ~5,400, and the pairing is not a
  dense window (see "Pair structure" below).
- Image names in pair_stats look like: `frame_000002`, `frame_000043`
  (last pair in the file: `frame_013067`, `frame_013111`)
- Do those names sort in capture order? **Yes.** The digits are the *source video* frame index,
  not a 0..299 keyframe counter, but they are zero-padded to 6 digits, so lexicographic order
  equals numeric order equals capture order — verified directly
  (`[int(n.split('_')[1]) for n in sorted(names)] == sorted(...)` → True over all 300 names).

### Did verify complete?

**Yes.** First `verification.json` ever produced in this repo — Step 1's
`find -name verification.json` printed nothing beforehand, confirming the "never run" claim.
Written to
`/workspace/outputs/2026_07_15-Goprosplat-GH010229/vggt_omega/colmap/verification.json`.

### Cost breakdown

`verify()` calls `build_localization_db()` internally and skips when the cache is present. The
driver built it first so the two costs are separable:

| Phase | Seconds | Note |
| --- | ---: | --- |
| `build_localization_db` (loma extraction, 300 frames @ 1080p) | 345.7 | one-off per scene+extractor; cached in `feedforward.zarr :: local_features/loma` |
| `verify()` — DB export + loma pair matching + `verify_matches` + `triangulate_points` | 2511.6 | the measurement of record |
| **Cold total** | **2857.3** | 47.6 min |

The 2511.6 s is dominated by pairwise loma matching: pycolmap's pairing log finished at
300/300 within the first ~2 min, after which the log is silent for ~40 min with the GPU pinned
at 100% and `database.db` growing. Observed GPU memory mid-run: 13,057 MiB (single `nvidia-smi`
sample, not a tracked peak).

### Pair structure — matters for Task 2

`verification.py` uses `pycolmap.SequentialPairGenerator` with `DEFAULT_OVERLAP = 10`, and that
generator applies **quadratic overlap**: it emits power-of-two frame gaps, not a dense 1..10
window. Measured histogram over sorted-image-id positions:

| gap | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pairs | 299 | 298 | 296 | 289 | 267 | 225 | 107 | 56 | 28 |

Total 1,865. So `pair_stats` is **not** a sliding-window neighbourhood — 191 of the pairs span
64+ keyframes. Any per-frame aggregation in Task 2 has to decide explicitly whether long-baseline
pairs are included, because they carry systematically different error (a 256-gap pair sharing few
points is not evidence about either endpoint's local pose the way a gap-1 pair is).

`frame_stats` has 300 entries — one per image, keyed by the same `frame_XXXXXX` names.

### Tier 1 / Tier 2 distributions as measured (`summary`)

```
n_points                   61375
n_pairs                     1865
track_length               median 3.0    p90 6.0     p99 12.0
reproj_error_px            median 1.748  p90 2.773   p99 3.644
pair_inlier_ratio          median 0.838  p90 0.921   p99 0.946
pair_rot_error_deg         median 1.012  p90 21.932  p99 91.718
pair_t_direction_error_deg median 5.155  p90 27.945  p99 141.069
```

The rotation/translation tails are heavy — p99 rot error 91.7° and p99 t-direction 141.1°. Some
of that is real pose error and some is the long-baseline pairs above being ill-conditioned; the
report's whole point is to separate those, so this is the signal Task 2 onward consumes, not a
defect to fix here.

### Reproduction (Task 1)

Driver used (measurement only, no production code):
`<scratchpad>/run_verify.py` — loads the scene's own `run_config.yaml`, constructs
`Reconstructor(cfg, config_dir=configs/)`, times `build_localization_db()` then
`verify(overwrite=True)`, and samples `/sys/fs/cgroup/memory/memory.stat :: rss` on a background
thread. `configs/base.yaml` was **not** modified. Note that Python's stdout is block-buffered
through `tee`, so the timing markers only appear when the process exits — the run looks silent
for ~40 min while healthy.

## Matcher comparison for `verify`

Same scene, same 300 frames, same `vggt_omega` poses — only `localization.matcher` changes
(overridden in the driver's in-memory config dict; `configs/base.yaml` untouched). Three
matchers measured end-to-end, serially on the one A40.

### Which matchers are even eligible

`localization.matcher` is a **vismatch model name**, not a class in a local registry — the
candidate names in the task brief were approximate. Measured against `vismatch.available_models`
(71 models): `xfeat` and `xfeat-star` exist, **`disk` does not — the name is `disk-lightglue`**,
and **there is no `lomag`** (the nearest sibling is `loma-r`).

`verify` refuses any matcher that cannot address a stable COLMAP keypoint table. That is no
longer an XFeatStar-specific hard-code: `LocalMatcher._probe_index_stability()` runs one
synthetic pair at construction and requires BOTH (a) matched keypoints are exact rows of the
keypoint table (no per-pair refinement) and (b) `extract()` reproduces the same table across
calls. `verify_reconstruction` raises `ValueError` when `has_stable_indices` is False.
Probed directly:

| matcher | `has_stable_indices` | eligible for `verify` |
| --- | :--: | --- |
| `loma` | True | yes |
| `xfeat` | True | yes |
| `disk-lightglue` | True | yes |
| **`xfeat-star`** | **False** | **no — structural exclusion, not attempted** |

So the XFeatStar exclusion is confirmed by measurement rather than by reading: per-pair subpixel
refinement moves the same keypoint to different coordinates in different pairs, which COLMAP's
one-keypoint-table-per-image model cannot represent.

### Speed

| matcher | `build_localization_db` (s) | `verify()` (s) | cold total (s) | peak rss (GB) | ms / attempted pair | `verify` speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `loma` (baseline) | 345.7 | 2511.6 | 2857.3 | 9.26 | 1147.4 | 1.00× |
| `xfeat` | 74.1 | 1015.5 | 1089.6 | 8.58 | 463.9 | **2.47×** |
| `disk-lightglue` | 93.8 | **690.4** | **784.2** | 8.70 | 315.4 | **3.64×** |

Per-pair cost is normalised by **2,189 attempted pairs**, not by each run's reported pair count
— see "the pair sets are not identical" below. (Task 1's "1346.7 ms per pair" used loma's 1,865
*reported* pairs as the denominator; over the pairs actually attempted the figure is 1147.4 ms.)

The matcher really is the dominant cost. A 45-pair stratified microbenchmark (5 pairs at each
of the measured gaps 1…256, identical pairs and frames for every matcher, 3 warm-up pairs each
so CUDA init is not charged to whoever ran first) put loma's matching at 1069.9 ms/pair, i.e.
2342 s of the measured 2511.6 s. **The matcher-independent remainder — pycolmap
`verify_matches` + `triangulate_points` + DB IO — is only ~170 s, 6.8% of `verify`**, so the
ceiling for an infinitely fast matcher is ~14.8×, and the measured 3.64× is nowhere near it.

The microbenchmark predicted `disk-lightglue` within 7.7% (743.5 s predicted vs 690.4 s
measured) but **underestimated `xfeat` by 37.5%** (635.0 s vs 1015.5 s). The likely reason is
visible in the match counts: per-pair cost is not just the network forward. `_recover_indices`
is an O(matches × keypoints) exact-equality broadcast against a 2048-row table, and xfeat
returns by far the most raw matches, so it pays the most in index recovery and COLMAP DB writes.

### Quality — and why the headline medians mislead

**The pair sets are not identical, so the naive comparison is not apples-to-apples.** Each run
attempts the same 2,189 pairs (`SequentialPairGenerator` is deterministic: 300−gap pairs at each
quadratic gap), but only pairs that survive matching reach `pair_stats`:

| matcher | reported pairs | gap 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attempted | 2,189 | 299 | 298 | 296 | 292 | 284 | 268 | 236 | 172 | 44 |
| `xfeat` | 2,189 | 299 | 298 | 296 | 292 | 284 | 268 | 236 | 172 | 44 |
| `loma` | 1,865 | 299 | 298 | 296 | 289 | 267 | 225 | 107 | 56 | 28 |
| `disk-lightglue` | 1,498 | 299 | 298 | 289 | 231 | 159 | 123 | 48 | 33 | 18 |

`xfeat` reports every attempted pair; loma silently drops 324 and `disk-lightglue` drops 691,
overwhelmingly at long baselines. loma's set is a strict subset of xfeat's. On the 324 pairs
loma dropped, xfeat's own median rotation error is **89.1°** — they are genuinely bad pairs, so
dropping them *flatters the dropper's median*. Any comparison of each run's own median therefore
rewards a matcher for refusing hard pairs.

Restricting to the **1,455 pairs all three matchers reported** removes that bias:

| matcher | rot err med / p90 / p99 (°) | t-dir err med / p90 / p99 (°) | matches (med) | inliers (med) | inlier ratio (med) |
| --- | --- | --- | ---: | ---: | ---: |
| `loma` | **0.76** / **5.41** / **40.24** | **4.33** / **18.89** / **82.06** | 837 | 722 | 0.865 |
| `disk-lightglue` | 1.28 / 27.02 / 86.16 | 8.03 / 60.59 / 158.24 | 571 | 541 | **0.954** |
| `xfeat` | 4.18 / 46.28 / 149.30 | 21.19 / 111.89 / 167.86 | 660 | 248 | 0.381 |

For completeness, the same statistics over each run's own (unequal) pair set — these are the
numbers `verification.json :: summary` reports, and they are the misleading ones:

| matcher | pairs | rot med / p90 / p99 (°) | t-dir med / p90 / p99 (°) | inlier ratio (med) |
| --- | ---: | --- | --- | ---: |
| `loma` | 1,865 | 1.01 / 21.95 / 91.47 | 5.15 / 28.01 / 140.85 | 0.838 |
| `disk-lightglue` | 1,498 | 1.37 / 31.68 / 91.80 | 8.40 / 68.28 / 163.65 | 0.950 |
| `xfeat` | 2,189 | 14.31 / 127.96 / 175.06 | 32.81 / 135.96 / 171.35 | 0.185 |

### Tier 2 (triangulation)

| matcher | triangulated points | track survival (per-frame med) | `mean_reproj_error_px` (per-frame med) | keypoints/frame (med) |
| --- | ---: | ---: | ---: | ---: |
| `loma` | 61,375 | 0.3179 | 1.8088 | 2048 |
| `disk-lightglue` | 61,373 | 0.2917 | 1.7801 | 2048 |
| `xfeat` | 45,975 | 0.2107 | 1.7807 | 2048 |

Tier 2 separates the two challengers cleanly. `disk-lightglue` triangulates essentially the same
map as loma (61,373 vs 61,375 points — a 2-point difference) at 92% of loma's track survival,
while `xfeat` loses a quarter of the map (45,975 points) and a third of the track survival.
`mean_reproj_error_px` is flat across all three and is not a discriminator here — it is
conditioned on tracks that already survived, so a matcher that keeps fewer, easier tracks scores
the same or slightly better.

### Reading

**Nothing measured here is faster than loma without costing pose quality.** The honest summary
is a trade curve, not a winner:

- **`xfeat` — faster but clearly worse; not a viable substitute.** 2.47× faster on `verify`,
  but on identical pairs its rotation error median is 5.5× loma's (4.18° vs 0.76°), its
  translation-direction median 4.9× (21.19° vs 4.33°), and its inlier ratio collapses to 0.381
  from 0.865. It loses 25% of the triangulated map. The speedup is *not* explained by finding
  fewer correspondences — xfeat finds slightly more raw matches than disk and most of them are
  wrong. This is "faster but worse" in the sense the brief rules out.
- **`disk-lightglue` — the real trade, and the fastest of the three.** 3.64× faster on `verify`
  (690.4 s vs 2511.6 s) and 3.64× on cold total, for a **1.7× worse rotation-error median
  (1.28° vs 0.76°) and 1.9× worse translation-direction median (8.03° vs 4.33°)** on identical
  pairs. Its Tier 2 map is indistinguishable from loma's (61,373 vs 61,375 points) and its
  inlier ratio is actually *higher* (0.954 vs 0.865) — it is a precise matcher that returns
  fewer, cleaner correspondences.

The catch for `disk-lightglue` is coverage and tails, not the median. It reports only 1,498 of
2,189 attempted pairs against loma's 1,865, dropping long-baseline pairs loma still recovers
(gap 64: 48 vs 107; gap 128: 33 vs 56) — and its p90 rotation error on common pairs is 5× loma's
(27.02° vs 5.41°). For a report whose purpose is to *find* bad poses, losing 691 pairs of
evidence and having a heavier error tail is a direct cost to the thing being measured.

So the choice a human is being offered: **spend 41.9 min with loma for the most complete and
tightest-tailed evidence, or 11.5 min with `disk-lightglue` for the same triangulated map, a
median pose error under 2× worse, but a third fewer pairs of evidence and a noticeably heavier
tail.** If `verify` is ever run routinely rather than as a one-off diagnostic, `disk-lightglue`
is the defensible default and this is worth re-measuring on a second scene before switching. As
a one-off diagnostic on a scene you already suspect, loma's completeness is worth the 30 extra
minutes. `xfeat` is not a candidate at any budget.

Both conclusions rest on one scene; the ranking is consistent across every statistic measured
here, but no second scene has been run.

### Reproduction (matcher comparison)

- `<scratchpad>/probe_matchers.py` — constructs each `LocalMatcher` and prints
  `has_stable_indices`; this is what established the `xfeat-star` exclusion.
- `<scratchpad>/bench_matchers.py` — the 45-pair stratified microbenchmark and the
  matcher-independent-remainder / ceiling calculation. Note it hard-codes `TOTAL_PAIRS = 1865`
  (loma's *reported* count); the attempted count is 2,189, so its printed ceiling of 4.87× is
  understated — the corrected figure is ~14.8×, computed above.
- `<scratchpad>/run_verify_matcher.py <model-name>` — the Task 1 driver with a matcher override
  applied to the config dict in memory, plus per-run quality aggregation to
  `<scratchpad>/metrics_<matcher>.json`. Each run's `verification.json` is preserved as
  `<scratchpad>/verification_<matcher>.json`.

The scene was restored after the experiment: `colmap/{verification.json, database.db, verified/}`
put back from the loma backup, and the `local_features/{xfeat, disk-lightglue}` groups this
experiment added to `feedforward.zarr` removed, leaving only `local_features/loma` as found.

## Is loma's `verify` cost our wrapper, or loma itself?

Same scene / frames / poses. 30 pairs stratified over the same quadratic gaps 1…256, 3 warm-up
pairs, `torch.cuda.synchronize()` around every timed region, A40. Script:
`<scratchpad>/bench_loma.py`, results `<scratchpad>/bench_loma.json`. No production code changed;
nothing on disk touched (`find -newermt` over the scene after the run: empty, `local_features/`
still holds only `loma`).

### The path hypothesis is false — every matcher takes the pairwise path

`has_stable_indices` re-probed directly: **`loma` True, `disk-lightglue` True, `xfeat` True.**
The flag does *not* route matchers to different call paths. The deciding line is
`verification.py:221` — `if isinstance(matcher, LocalMatcher):` — and every vismatch model is a
`LocalMatcher`, so **all three go through `match_images()`, one forward pass per pair**. The
`else` branch (`matcher.match(...)`, the cached-descriptor path) is unreachable for vismatch:
`LocalMatcher.match()` raises `NotImplementedError` by construction. `has_stable_indices` gates
only index *recovery inside* `match_images` and eligibility for `verify` at all
(`verification.py:184`).

So loma is not penalised by a path disk-lightglue escapes. Both pay per-pair re-extraction.

### Wrapper vs direct loma (30 pairs)

(a) = `LocalMatcher.match_images()`, exactly what `verify` calls.
(b) = the third-party call stripped to the minimum that still yields correspondences —
`preprocess` + `LoMa.detect_and_describe` ×2 + `LoMa.__call__` + `filter_matches` +
`to_pixel_coords` + one D2H of the matched coords. Both start from the same numpy frames.

| path | median ms | p90 ms |
| --- | ---: | ---: |
| (a) our wrapper `match_images` | **993.8** | 1089.6 |
| (b) direct loma, incl. numpy→device | 898.4 | 933.2 |
| (b′) direct loma, inputs already on device | 877.9 | 911.3 |

**Overhead (a) − (b) = 95.4 ms = 9.6% of (a).** Nearly all of it is one line: `_recover_indices`
(90.6 ms), the O(matches × keypoints) exact-equality broadcast that maps matched coordinates back
to COLMAP keypoint-table rows — 446 matches × 2048 rows × 2 sides. This is the same effect that
made the earlier microbenchmark underestimate `xfeat` by 37.5%. It is *needed* work under the
current design, but it is `O(K·N)` where a sort/hash would be `O(K log N)`.

### Where a loma pair actually goes (one representative gap-16 pair)

| stage | ms | % |
| --- | ---: | ---: |
| image load from `frames.zarr` (2×) | 86.0 | 7.9 |
| numpy→device `_to_tensor` (2×) | 13.6 | 1.3 |
| `resize_to_divisible` | 1.0 | 0.1 |
| **`detect_and_describe` img0** | **423.5** | **39.1** |
| **`detect_and_describe` img1** | **424.1** | **39.2** |
| LoMa match transformer | 38.2 | 3.5 |
| `filter_matches` + `to_pixel_coords` | 1.3 | 0.1 |
| `rescale_coords` | 0.3 | 0.0 |
| device→numpy (kpts + descs) | 1.8 | 0.2 |
| out-of-bounds filter (vismatch) | 0.2 | 0.0 |
| `cv2.findHomography` MAGSAC (vismatch, result discarded) | 2.5 | 0.2 |
| our `_check_pixel_frame` (2×) | 0.1 | 0.0 |
| our `_recover_indices` (2×) | 90.6 | 8.4 |
| **total** | **1083.1** | 100 |

**Dominant stage: `detect_and_describe`, 847.6 ms = 78.3% of the pair** — loma's DINOv2 backbone,
run on both images of every pair. The whole matching stage that actually *uses* the pair is 38.2 ms.

### Counterfactual: what loma would cost with per-image caching

Measured inputs: `detect_and_describe` = **493.2 ms/image** (median, 12 images);
match-only from cached kpts/descs = **38.6 ms/pair** (median, same 30 pairs);
matcher-independent remainder of `verify` = **~170 s** (from the prior section).

| architecture | arithmetic | `verify` (s) |
| --- | --- | ---: |
| measured (pairwise) | 2189 × 993.8 ms + 170 s | **2511.6** (measured) |
| cached, features already on disk | 2189 × 38.6 ms + 170 s | **~255** |
| cached, counting a cold extraction pass | + 300 × 493.2 ms = 148 s | **~403** |

**~255 s vs the measured 2511.6 s — a 9.9× reduction, and faster than `disk-lightglue`'s
measured 690.4 s while keeping loma's 0.76°/5.41° accuracy.** 2189 pairs re-extract features
**4,378 times** for 300 distinct images: a 14.6× redundancy factor.

**Is the loma↔disk gap architectural?** Not *between* them — they share the identical pairwise
path, so the 4.05× per-pair gap (993.8 vs 245.4 ms) is loma's DINOv2 detector being that much
more expensive than DISK's CNN. What *is* architectural is that both pay it twice per pair
instead of once per image, which multiplies the model-cost gap by ~14.6 instead of amortising it.
Under caching, loma's *matching* (38.6 ms) would be in the same class as LightGlue's, and the
residual loma premium collapses to a one-off ~118 s of extra extraction (148 s vs ~30 s), not a
recurring 1,821 s.

### Bonus: `extract()` is the one place a direct call genuinely wins

`LocalMatcher.extract()` measures **913.4 ms/image** but a single `detect_and_describe` is
**493.2 ms** — because vismatch implements `BaseMatcher.extract()` as `forward(img, img)`, which
detects on the same image twice and runs a full self-match. Calling the feature stage directly
would roughly halve `build_localization_db` (345.7 s → ~190 s). That is a real, ~1.85× wrapper
inefficiency — on the extraction path, not on the pairwise path.

### Answer

**No — calling loma more directly does not meaningfully speed up `verify`: ~9.6% (95 ms/pair,
2511.6 s → ~2300 s), and 91 ms of that 95 is `_recover_indices`, which is required work.** The
10× win is architectural (cache per-image features and match from the cache), which is exactly
the already-designed but unimplemented `match_extracted` shim
(`docs/superpowers/specs/2026-08-17-descriptor-matching-shim-design.md`) — where loma is listed
**Deferred**, because its matcher wants normalized resized-resolution coordinates while
`extract()` caches original-resolution pixels −0.5. That coordinate conversion, not the model, is
what stands between `verify` and a ~10× loma speed-up.

Assumptions: 2,189 attempted pairs and the ~170 s matcher-independent remainder are carried over
from the sections above, not re-measured; the counterfactual assumes cached matching keeps
loma's match set (the match-only path was run on the same cached tensors the pairwise path
produces, so this is arithmetic, not extrapolation, but it was not verified match-for-match).
Image IO (86 ms/pair here) is charged in full to a cold pair; `verify` reads through a 32-frame
LRU, so it pays less in the sequential-pair loop.

---

## Task 6 Step 5: can the creators hand their multiview pass down to `build_report`?

`build_report` runs a dense multiview pass. When `pointcloud.use_multiview_confidence` is on, a
creator already ran that loop during reconstruction, so the report doubles it. Scoped here, **not
implemented** — Task 8 measures whether the duplication costs anything worth the plumbing.

### How many creators have a `collect` dict to hand down: zero

```bash
grep -rn "compute_multiview_depth_confidence" --include=*.py collab_splats/ | grep -v "def compute"
```

Four creators call it — `vggtx.py:354`, `vggt_omega.py:265`, `mapanything.py:446`,
`loger.py:440` — and **none of them passes `collect`**. `grep -rn "collect=" --include=*.py
collab_splats/` returns exactly one call site: `metrics.py:492`, inside `build_report`. So today
there is no dict to hand down; each creator would first have to opt in.

### Can the reconstruction path pass one through? Structurally yes, usefully no

Three findings, in increasing order of how much they cost:

1. **In the shipping configuration there is no second pass at all.** `use_multiview_confidence`
   is `false` on `configs/base.yaml:47`, so no creator runs the loop and the report's pass is the
   only one. The saving is zero by default and only appears on an opted-in scene.
2. **The dict is not persisted.** `collect` fills `pairs` / `rel_depth_error_counts` /
   `rel_depth_error_edges` in memory; nothing writes them to `feedforward.zarr`. A hand-down
   therefore helps only the inline `run_pipeline` path. `report` is a leaf stage
   (`_STAGE_DEPS["report"] == ["pointcloud"]`), so the `--stages report` disk re-run against a
   scene pulled from `environments-processed` could never see it and would run the pass anyway.
3. **The creators' pass does not measure the same thing, so reuse would change the numbers.**
   Tolerances are per-backend: `abs=0.0, rel=0.01` on vggtx/vggt_omega/loger and
   `abs=0.02, rel=0.02` on mapanything, against `build_report`'s fixed `abs_thresh=0.0,
   rel_thresh=0.05`. Those are not cosmetic. In `base.py`, `tol = abs_thresh + rel_thresh *
   expected_d.abs()` decides `occluded`, `counted = valid_ij & ~occluded`, and the collected rows
   are computed over `sel = counted & has_depth & (expected_d > 1e-6)` — so the tolerance selects
   the pixel population every per-pair median, IQR, parallax and depth is taken over, and the
   population the residual histogram accumulates. MapAnything additionally passes
   `depth_masks=combined_mask`, narrowing it further. Reusing the creator's dict would make the
   report's depth block a function of each backbone's own calibration, which is exactly the
   cross-backbone comparability that `abs_thresh=0.0` exists to guarantee.

**Reading:** the hand-down is not free reuse of an identical computation — it is a different
measurement that happens to share a loop. If Task 8 finds the duplicated pass expensive enough to
be worth removing, the honest version is for the creator to run the *report's* tolerances into a
second `collect` (or for the report to accept a backend-tolerance stamp in its output), not to
silently adopt whatever the creator happened to use.

---

## Task 8: report stage, measured

Two scenes, because no store on this machine carries all three measurements. Which store
carries which, and why, is recorded in the task itself.

All numbers below are readouts of `report.json` files written by `build_report` on this
machine. Nothing here grades a scene or names a cause; where a channel could not be measured,
that is stated as an outcome rather than filled in.

### Scene A — `data/outputs` (30 frames, model grid `384x688`, original 1080x1920)

Carries depth + photometric. Epipolar did not run — see below.

- Backend: `vggt_omega` (read from `configs/base.yaml` `pointcloud.backend`, a provenance
  label on the report, not a code path)
- Wall clock: **64.4 s** end-to-end in tmux (`REPORT_SECONDS`), of which ~23 s is interpreter
  start and imports. In-process `build_report` on a warm interpreter: **41.56 s**
- Peak rss: **6.40 GB** / 46.6 GB, sampled from `/sys/fs/cgroup/memory/memory.stat` `rss`
  across the whole process. The `(30, 1920, 1080, 3)` float32 RGB cast alone is 0.75 GB
- Measurements available: `['depth', 'photometric']`
- Pairs: 870 depth **directions** (`30·29`) + 57 photometric **pairs**; `report.json`
  **0.41 MB** (filter applied: **no**)
- Epipolar: **unmeasured** — see "Epipolar channel" below

**Timing split** (the two heavy calls wrapped at their import sites with `mock.patch`, so the
real `build_report` was timed; no production code changed):

| stage | seconds | share |
|---|---|---|
| multiview pass (`compute_multiview_depth_confidence`) | 3.63 | 8.7% |
| depth assemble (`compute_depth_error`) | 0.01 | 0.0% |
| photometric (`_run_photometric`) | **37.30** | **89.8%** |
| JSON serialise | 0.02 | 0.0% |
| JSON write | 0.02 | 0.0% |
| other (`load_zarr`, per-frame medians, ranks) | 0.59 | 1.4% |

Photometric split further, measured separately: `frames.zarr` open + `images()` +
`astype(float32)` is **0.53 s**, i.e. **0.8%** of the photometric cost. The remaining 99.2% is
the pair loop. `compute_photometric_ncc` defaults to `max_separation=2`, so it walks
`(i, i+1)` and `(i, i+2)` — 29 + 28 = **57 pairs at N=30**, `O(N·max_separation)`, not `O(N²)`.

One caveat on that second run: called standalone the pair loop measured **66.02 s**
(1.158 s/pair) against **37.30 s** (0.654 s/pair) inside `build_report`. The difference is
that `build_report` runs the multiview pass first and absorbs CUDA/warp context warm-up, which
the standalone loop pays itself. Both are recorded; the IO share (0.8%) is the robust part and
is what the parallelism question turns on.

**Depth, Scene A.** 61,818,426 residual samples of a possible `870·688·384 = 229,847,040`
(**26.9% retained** after the occlusion gate and the frustum/valid-depth checks); 1226 bins.

| quantile | `abs_quantiles` | `quantiles` (signed) |
|---|---|---|
| p50 | 1.1723% | +0.2908% |
| p90 | 41.9689% | +41.9689% |
| p99 | 217.7436% | +217.7436% |
| p999 | 421.7463% | +421.7463% |

Parallax p10/p50/p90: **7.1382 / 24.8407 / 49.1603 deg**. `depth_error_px` p50 **4.61 px**,
p90 **70.05 px**. Photometric `photometric_ncc` p10/p50/p90: **0.4361 / 0.6271 / 0.7875**.
`crop_coverage` is 1.0 on all 30 frames.

### Scene B — `evals/results/mv_*` (60 frames each, 640x480 original)

Depth only. `frames.zarr` and `verification.json` are absent, so photometric and epipolar
record `available: false` — the never-fails contract from Task 6, exercised on a real store for
the first time.

- Degraded cleanly on all three: **yes**. `measurements_available == ['depth']` on each, and
  both dead channels carry a `reason`:
  - `"no verification.json at evals/results/<store>/colmap/verification.json — set
    pointcloud.geometric_verification: true or run --stages verify"`
  - `"frames.zarr not found at evals/results/<store>/frames.zarr"`
- Wall clock: `mv_vggt_omega` **19.4 s**, `mv_mapanything` **10.6 s**, `mv_vggtx` **14.3 s**
  (one process, three stores in sequence). Peak rss over the whole run: **4.52 GB** / 46.6 GB
- `report.json`: **1.48 / 1.47 / 1.47 MB**

Model resolutions differ (`592x448` on vggt_omega, `518x392` on the other two), so the bin
counts differ by construction — 1958 against 1792. `mv_mapanything` also stamps
`crop_coverage` 0.6610 on every frame, against 1.0 on the other two.

### Sanity target (depth residual, `evals/results/mv_vggt_omega`)

| quantile | measured (`abs_quantiles`) | signed (`quantiles`) | prior (`depth_disagreement.py`) |
|---|---|---|---|
| median | **0.3667%** | +0.0217% | 0.37% |
| p90 | **1.5365%** | +0.9631% | 2.27% |
| p99 | **13.6333%** | +13.6333% | 25.67% |

Bin count for this scene (Rice, from `n_pair_directions·H·W = 3540·448·592 = 938,864,640`):
`2·n^(1/3) = 1958.4` → **1958**, which is the bin count in the file. Samples retained:
819,592,625 / 938,864,640 = **87.3%**.

**The median agrees to three significant figures. The two tail quantiles differ, and the
difference is by construction, not a discrepancy to fix.** The two numbers are taken over
different pixel populations:

- `depth_disagreement.py` gates on `src_valid & (expected_d > 0) & in-frustum & (sampled > 0)`
  and has **no occlusion gate at all** — its docstring says it "records the signed residual
  instead of thresholding it".
- `build_report:500` passes `rel_thresh=0.05`, so in `base.py` `tol = 0.05·|expected_d|`,
  `occluded = sampled_d < expected_d − tol`, and the collected population is
  `sel = counted & has_depth & (expected_d > 1e-6)` with `counted = valid_ij & ~occluded`.
  Pixels whose residual is more negative than −5% are classified occluded and leave the
  population entirely. Positive residuals (free-space violations) are kept at any magnitude.

That predicts exactly the observed signature — a one-sided truncation that cannot touch a
median sitting at 0.37%, but removes the negative half of the `abs` fold's tail. Measured on
the real counts:

- mass below `rel = −0.05`: **0.030%**; below `−0.10`: **0.000%**; below `−0.20`: **0.000%**.
  The negative tail stops at the gate, and the small residue is bin-edge granularity.
- mass above `rel = +0.05`: **2.66%**, retained. This is why signed p99 equals abs p99 — the
  top 1% of the folded distribution is entirely positive.

**The transform inversion is not implicated.** Re-running the Task 2 Step 1 round-trip against
the real bin edges: `bounded_residual(x)` then `u/(1−|u|)` over
`x ∈ {−1e6, −25.67, −0.2567, −0.05, 0, 0.0037, 0.1363, 25.67, 1e6}` returns a maximum relative
error of **5.0e-11**, and the shipped edges are finite, strictly monotone and span exactly
`[−1.0, 1.0]`.

Two smaller differences also separate the two numbers, both pushing the prior estimate the
other way and neither large enough to matter here: the prior script clamps residuals to
`[−0.5, +0.5]` before `torch.histc`, so any mass past ±50% piles into its edge bins and its own
p99 is a floor; and it reads bin **centres** off a 2000-bin uniform grid, against the bounded
non-uniform axis used here.

### Rank control (three backbones, same 60-frame scene, `data/7scenes/chess/seq-01`)

Model resolutions differ (`592x448` vs `518x392`), so bin counts differ by construction —
these are quantiles, never raw counts.

| backbone | hist p50 | p99 | pair-median |
|---|---|---|---|
| mv_mapanything | 0.00324 | 0.02003 | 0.00084 |
| mv_vggt_omega | 0.00367 | 0.13633 | 0.00096 |
| mv_vggtx | 0.00273 | 0.09588 | 0.00062 |

**Reference ordering, and its source: pose error is UNESTABLISHED.**
`evals/results/mv_sweep_{mapanything,vggt_omega,vggtx}.json` are the sweep outputs for these
exact stores, and their row columns are
`variant, rel_thresh, min_views, median_rel_err, p90, p95, frac_over_10pct, retention`.
**There is no pose-error column.** The 1.6× figure quoted in earlier revisions came from a
100-frame BA sweep, not from these 60-frame stores, so it is not a reference for them.

Ordered correctly: **unrankable against pose error.** No pose-error reference exists for these
stores, so the three numbers above are reported without a verdict.

One adjacent measurement does exist in the same files and is recorded separately, because it is
a *different quantity*: the sweeps' `unfiltered` rows carry `median_rel_err`, depth accuracy
against 7-Scenes ground-truth depth, where the report's depth channel measures cross-view
self-consistency with no ground truth at all.

| backbone | sweep `unfiltered` `median_rel_err` (vs GT depth) | p90 | frac > 10% |
|---|---|---|---|
| mv_mapanything | 0.01841 | 0.10891 | 0.10623 |
| mv_vggtx | 0.02200 | 0.21483 | 0.16219 |
| mv_vggt_omega | 0.02240 | 0.22015 | 0.15688 |

Against that ordering the report's **p99 column** is in the same order
(mapanything < vggtx < vggt_omega) and its **p50 columns** are not
(vggtx < mapanything < vggt_omega). Both facts are recorded; neither establishes the pose-error
ordering this control was designed to test, and a GT-depth ordering is not a substitute for it.

### Correlations

Per store, read against its own sample count.

| store | `error_vs_depth` | `error_vs_frame_separation` | `confidence_vs_error` (n_frames) | `ncc_vs_frame_separation` |
|---|---|---|---|---|
| Scene A `data/outputs` | **0.7842** | **0.6137** | **−0.7246** (30) | **−0.4394** (57 pairs) |
| mv_mapanything | −0.1003 | 0.0725 | −0.4622 (60) | null (channel absent) |
| mv_vggt_omega | 0.0700 | 0.7144 | 0.2370 (60) | null (channel absent) |
| mv_vggtx | −0.1424 | 0.0187 | 0.1946 (60) | null (channel absent) |

`error_vs_depth` and `error_vs_frame_separation` are over `n_pair_directions` = 870 (Scene A)
and 3540 (each mv store). The null the report ships beside `error_vs_depth` is
`sigma_Z ~ Z²/(f·B) => relative residual rises ~linearly in Z`, i.e. positive rho expected.
Scene A is 0.78 against that null; the three mv stores sit between −0.14 and +0.07. All four
`confidence_vs_error` rho are computed — every store carries a `confidence` array — and each
ships with its own `n_frames`, which is the only reason a raw scipy rho is publishable here.

### Disparity floor (derived, = 1 px)

Floor parallax is `rad2deg(1/focal_px)`; `depth_error_in_pixels` returns `None` below it.
Recomputing the count from the shipped `median_parallax_deg` column reproduces the shipped
count exactly in all four cases.

| store | focal (px) | floor (deg) | directions under 1 px | parallax p10 / p50 / p90 (deg) |
|---|---|---|---|---|
| Scene A `data/outputs` | 553.78 | 0.1035 | **0 / 870 (0.0%)** | 7.1382 / 24.8407 / 49.1603 |
| mv_mapanything | 427.70 | 0.1340 | 334 / 3540 (9.4%) | 0.1413 / 0.6994 / 1.6428 |
| mv_vggt_omega | 543.30 | 0.1055 | 313 / 3540 (8.8%) | 0.1267 / 0.8629 / 2.1563 |
| mv_vggtx | 449.61 | 0.1274 | 361 / 3540 (10.2%) | 0.1251 / 0.9097 / 2.3046 |

The two scenes sit two orders of magnitude apart on this axis: Scene A's median pair carries
~25 deg of parallax and no direction falls under the floor, while the chess sequence's median
pair carries ~0.7–0.9 deg and roughly one direction in ten cannot see depth at all.

**Do not read the parallax column alone as pose geometry.** Measured in Task 7 and pinned by
`test_control_depth_scale_moves_parallax_only_on_the_source_side`: world points are unprojected
through the *source* frame's depth, so a depth fault in frame *i* moves the parallax of every
pair `(i, ·)` — −9.05 to −9.15% for a ×1.1 depth scale, matching `1/1.1 − 1` — while pairs
`(·, i)` and pairs touching neither move by exactly zero. ρ still separates the two faults by
three orders (0.98 depth vs 634 pose), so attribution is unaffected.

### Running error

`running_error.depth` is the sequential-pair curve. Fitted as `cum ~ steps^p`, where `p = 1` is
linear accumulation, alongside the ratio of the observed endpoint to a linear extrapolation of
the first step.

| store | steps | first | last | obs/linear at end | log-log slope `p` | `error_vs_frame_separation` |
|---|---|---|---|---|---|---|
| Scene A `data/outputs` | 29 | 0.000838 | 0.123752 | **5.090** | **1.523** | 0.6137 |
| mv_mapanything | 59 | 0.001220 | 0.040274 | 0.560 | 0.975 | 0.0725 |
| mv_vggt_omega | 59 | 0.000816 | 0.015707 | 0.326 | 0.847 | 0.7144 |
| mv_vggtx | 59 | 0.000699 | 0.040153 | 0.973 | 0.942 | 0.0187 |

Scene A's curve rises faster than linearly (`p = 1.52`); the three mv stores do not
(`p = 0.85–0.98`). Read against `error_vs_frame_separation` before calling any of this
accumulation, per the note the report ships on `running_error`: frame index is a confounded
axis, and on Scene A both the running curve and the separation rho move together, while
`mv_vggt_omega` shows the highest separation rho (0.714) of the four on a curve that is
*sub*-linear. The report states both columns and makes no call between them.

### Epipolar channel — UNMEASURED

Recorded as an outcome, inside the ~20-minute time box the task sets for it.

Step 1c's command as written is `run_pipeline.py --config configs/base.yaml --stages verify
--overwrite`, which exits on `error: the following arguments are required: VIDEO|DIR,
--output-root`. Supplying those would not have helped: **`data/outputs/` contains no `colmap/`
directory at any depth** (`feedforward.zarr`, `frames.zarr`, `gopro-compare/`,
`tutorial_cache/` and nothing else), so `verify` has no reconstruction to load, and the store
is not in the `<scene>/<backend>/` layout the leaf-stage re-run path expects. Producing one
means a full pointcloud run from video, which exceeds the time box and would overwrite the
store every other number here was measured on. `run_pipeline.py` also writes to
`<output-root>/<video-stem>/`, so it could not have populated `data/outputs/colmap/` in any
case.

Consequence: no `verification.json` exists anywhere in this repo, the epipolar channel is
unmeasured on both scenes, and the three-way attribution (`ρ = measured_px / δd_equiv`) is
exercised only by the Task 7 controls, not by a real scene. The report emitted its other
channels and recorded the reason, which is the behaviour this stage was built for.

### Pair-table size

Measured, no filter applied. The 20 MB threshold in the task is not reached by either scene.

| store | total | `pair_directions` | rows | bytes/row | histogram | photometric rows |
|---|---|---|---|---|---|---|
| Scene A | 0.407 MB | 0.276 MB | 870 | 318 | 0.042 MB | 0.008 MB |
| mv_vggt_omega | 1.477 MB | 1.144 MB | 3540 | 323 | 0.065 MB | — |

The pair table is ~70–77% of the file and scales as `N·(N−1)`, while the histogram is
essentially fixed (Rice on the sample count, so `n^(1/3)`). **Extrapolated** from the measured
~320 bytes/row, a 300-frame scene would carry 89,700 directions ≈ **29 MB** of pair table,
which does cross the threshold. That is arithmetic on a measured row size, not a measurement;
the filter stays unbuilt until a 300-frame store is actually run.

### Compute follow-ups (decided by the split above, not in advance)

- **Duplicated multiview pass: 3.63 s of 41.56 s, 8.7%.** Worth handing the creator's
  `collect` dict down to `build_report`? **No.** The ceiling is 8.7% on the one scene that
  could be split, and the three objections already recorded in this document stand
  independently of cost: the pass does not run at all under the shipping
  `use_multiview_confidence: false`; the dict is never persisted, so the `--stages report`
  disk re-run could not see it; and the creators' per-backend tolerances select a different
  pixel population from `build_report`'s fixed `abs_thresh=0.0, rel_thresh=0.05`, so reuse
  would change the numbers. An 8.7% saving does not buy that.
- **Photometric share: 37.30 s, 89.8%.** Worth parallelising across pairs? **It is the only
  lever the split identifies** — everything else in the report is under 2% and would be noise.
  The cost is in the pair loop, not IO: `frames.zarr` read plus float32 cast is 0.53 s, 0.8% of
  the channel, and the loop is `O(N·max_separation)` (57 pairs at N=30, 597 at N=300), so the
  work is embarrassingly parallel across independent pairs. Whether to spend that complexity is
  a separate decision and is not taken here; what the measurement settles is that no other
  target is worth touching.
- **Anything large enough to justify streaming to zarr instead of JSON? No.** Serialise 0.02 s
  and write 0.02 s, 0.1% of the run combined, on files of 0.41–1.48 MB. Even the 29 MB
  extrapolation above is a size question about the pair table, not a throughput question about
  the writer.

### Prior baseline provenance (transcribed from `evals/scripts/depth_disagreement.py` before deletion)

The file was never tracked by git, so this is the only surviving record of what the
0.37 / 2.27 / 25.67% baseline was measured against.

```python
STORE = "evals/results/mv_vggt_omega/feedforward.zarr"
SDF_TRUNC = 0.01   # configs/base.yaml mesh.sdf_trunc
VOXEL = 0.0025     # configs/base.yaml mesh.voxel_size
CONF_PCT = 20.0    # configs/base.yaml mesh.conf_percentile
NB, LO, HI = 2000, -0.5, 0.5   # uniform histogram, residuals CLAMPED to [LO, HI]
```

Module docstring: *"Inter-view depth disagreement in depth units — the TSDF blur budget.
Mirrors compute_multiview_depth_confidence's projection exactly, but records the signed
residual (sampled_d - expected_d) instead of thresholding it."* It reported two masks,
`"all valid depth"` and `"conf > p20"` (via `confidence_mask(conf, CONF_PCT)`); the
0.37 / 2.27 / 25.67% row is the **`all valid depth`** row, and the `conf > p20` row is the one
that tightened p90 to 0.92%. Its `abs` fold sorted bin centres by absolute value and took a
weighted CDF, which is the same operation as `abs_quantiles`.

### Two corrections to the task's own command blocks, found by running them

Recorded so the next reader does not repeat them.

1. **Step 1c** is missing `VIDEO|DIR` and `--output-root`, and its target store has no
   `colmap/` reconstruction. Covered above.
2. **Step 4's readout reads `d['pairs']` from the depth block, which does not exist** — the
   depth block ships `pair_directions`, exactly as this plan's own Self-Review says it does
   ("Task 4 names every one of its keys for directions"). The block raises `KeyError: 'pairs'`
   rather than printing wrong numbers, so it fails loudly; the rank-control table above was
   produced with `d['pair_directions']` substituted and is otherwise the block as written.
