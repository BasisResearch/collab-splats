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

### Reproduction

Driver used (measurement only, no production code):
`<scratchpad>/run_verify.py` — loads the scene's own `run_config.yaml`, constructs
`Reconstructor(cfg, config_dir=configs/)`, times `build_localization_db()` then
`verify(overwrite=True)`, and samples `/sys/fs/cgroup/memory/memory.stat :: rss` on a background
thread. `configs/base.yaml` was **not** modified. Note that Python's stdout is block-buffered
through `tee`, so the timing markers only appear when the process exits — the run looks silent
for ~40 min while healthy.
