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
