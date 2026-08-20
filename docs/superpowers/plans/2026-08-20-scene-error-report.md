# Scene Error Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `report` leaf stage producing a reference-free `report.json` that locates reconstruction error in space and time, with no ground truth and no verdicts.

**Architecture:** One module, `geometry/metrics.py`, holding three measurement functions whose *dependencies deliberately differ* — depth cross-view (poses+depth), photometric NCC (poses+depth+appearance), and verify's epipolar rows (poses only). Reading them against each other is what attributes error. All three write into the **one** existing per-pair row type, `PairStats`, keyed by frame index, so the report is a single joined table. `depth_error_in_pixels` puts depth and pixel residuals on one axis.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, scipy 1.17.1, torch, zarr v3, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-scene-error-report-design.md` (commit `d5021b4`)

---

## Environment and Safety (read before Task 1)

- **Python:** always `/opt/venv/reconstruction/bin/python`. Bare `python` is 3.13 and wrong for this project.
- **Heavy runs:** tmux only, serially. Never two GPU jobs at once. Container cap 46.6 GB; read `rss` from `/sys/fs/cgroup/memory/memory.stat`, **not** `memory.usage_in_bytes`.
- **A concurrent session keeps files dirty** (`configs/base.yaml`, `pyproject.toml`, `collab_splats/remote/rerun.py`). **Never `git add -A`. Never repo-wide `black .`** — the venv's black 26.5.1 is newer than the repo's formatting. Stage named files only.
- **`git add -f`** is required for anything under `docs/superpowers/` (gitignored).
- **Pre-existing failures that are not yours:** 5 in `tests/wrapper/`, and `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius`.

## What this revision removed, and why

Every row below was verified against the repo or measured, not assumed.

| Removed | Replaced by | Verified reason |
|---|---|---|
| `PARALLAX_FLOOR_DEG` | `disparity_px = deg2rad(parallax)·focal < 1.0` | the floor is *derivable*: below one pixel of disparity the two views' rays differ by less than a pixel, so the pair cannot see depth at all. Per-scene, from the focal. The guard and the value became the same expression. |
| `REL_EDGES` range (`±0.5`) + clipping | bounded axis `u = r/(1+\|r\|)`, edges `linspace(-1,1)` | **measured**: on 200k residuals plus ±12 and ±40 outliers, **zero values dropped**, and recovered quantiles match `np.quantile` to 5 decimals through p99.9. Deletes the chosen range, the clip, and the saturating end bins. What remained after that was a bin *count* — killed in turn by the row below. |
| `RESIDUAL_BIN_EDGES` (the last constant) | `residual_bin_edges(n_samples)` — Rice's rule, `k = 2·n^(1/3)` | the bounded axis fixed the range, so only resolution was left, and resolution is a function of how many samples there are. **Measured** on a 3M-sample heavy-tailed population shaped like the real baseline: median recovery error 5.1% at 512 bins, 1.7% at 1024, **0.66% at 1560 (Rice, 60 frames)**, **0.08% at 4584 (Rice, 300 frames)**. Rice moves the right way — more pixels justify finer bins — and the caller that knows `N`, `H` and `W` computes it before the loop starts. `metrics.py` now has **zero constants** — and none moved into the tests either: `tests/geometry/test_metrics.py` derives its sample counts with `_n_samples(frames, side)`, production's own `n_pairs·H·W`, so the only literals on that side are the scene shapes a fixture has to name and the measured bin counts an assertion has to state. |
| `QUANTILE_GRID` | inline at its one remaining use | once every per-pair column ships raw, its quantiles are convenience a reader can compute. Only the histogram needs them, because raw is unavailable there. |
| `MIN_SAMPLES`, `PHOTOMETRIC_MAX_SEPARATION` | keyword args with defaults | tuning values belong at the call they tune. (`min_samples` still earns its existence: **measured**, `np.corrcoef` on 2 points returns exactly ±1.0 whatever the values.) |
| `rank_correlation()` | `stats.spearmanr(a, b, nan_policy="omit").statistic` | scipy's `nan_policy` was the whole wrapper body, and `verification.clean_for_json` (`verification.py:346`) already converts nan→null recursively at write time. **Measured**: `json.dumps` emits a bare `NaN`, which is invalid JSON — so the conversion is load-bearing, but it already exists. |
| `_index_from_name()` | `{iid: k for k, iid in enumerate(sorted(recon.images))}` | **the parser was a bad assumption and the repo already had the answer.** `sorted(recon.images)` order is the documented alignment contract (`verification.py:99, 136, 144, 150`) and this exact dict already exists twice (`:193`, `:314`). Digit parsing breaks on `IMG_2039.jpg`, on names with two number groups, and on any scene whose names do not sort in capture order. |
| `_verification_rows()` | `idx1`/`idx2` written by verify itself | `asdict(p)` at `verification.py:359` serialises whatever fields `PairStats` has, and Task 2 already edits `PairStats`. Put the shape at the source and the "merge" becomes `json.loads` plus one division. |
| `_photometric_original_res()` | shape mismatch handled inside `compute_photometric_ncc` | two functions for one measurement, split only by which grid it happened to run on. |
| `_cumulative()`, `_crop_coverage()`, `describe()`, `_by_depth()`, `_distribution` use, `SCHEMA_VERSION`, `normalized_residual()` | inline / scipy / numpy | a cumsum, three lines of arithmetic, `np.quantile`, a column correlation, a second quantile path, a version nothing reads, and a hand-rolled Pearson (**measured** identical to `sqrt(2−2·NCC)` to 8 dp). |

**Net surface of `metrics.py`: 8 functions, no constants.** Previous revision: 12 symbols and 5 constants. The one before that: 4 files and 3 classes.

**Naming is checked against the repo, not chosen.** Measured: `compute_` prefixes 8 functions in `collab_splats/`, `calculate_` prefixes **zero** — so the measurement functions are `compute_depth_error` / `compute_photometric_ncc`. The `_px` unit suffix on `focal_px` follows `mean_reproj_error_px` (`verification.py:326`). Loop locals reuse the mv loop's own names (`cam2world`, `pts_world`, `pts_cam_j`, `proj_j`, `in_front`, `expected_d`, `has_depth`, `counted`) so the two loops read alike. `n_pixels` follows the `n_keypoints` / `n_tracks` family in `frame_stats`; `PairStats`' own `num_matches` / `num_inliers` are pre-existing and left alone. The pass also caught a real bug rather than only cosmetics: `FrameStore` was imported from `preproc.sampling` (it lives in `preproc.frame_store`), called through a `read()` that does not exist, and indexed with `frame_indices()`, which holds **source-video** positions — pairing depth row `k` with whatever video frame happened to sit at that number. Corrected to `store.images()[:n]`.

**A name states the quantity, not its shape.** `median_rel` said "a median of something relative" and left the reader to work out what. It is renamed `median_rel_depth_error`, and its partner `iqr_rel_depth_error`, because both are statistics of one named quantity — the signed relative depth error `(d_sampled - d_expected) / d_expected`. That rename also answers the scale question directly: a uniform scale factor `s` between two frames' depth shows up in that quantity as exactly `s - 1`, so **the median IS the pairwise scale reading** and `s = 1 + median_rel_depth_error`. Scale does not get its own column — a derived column that restates a shipped one is the redundancy critique 1 point 4 already removed once. The identity is written into the `PairStats` field comments so the reader finds it where the field is. Same pass, same rule: the JSON keys `error_in_pixels` → `depth_error_px` (says which error, and its unit), `separation` → `frame_separation` (separation of what), `error_vs_separation` → `error_vs_frame_separation` and `ncc_vs_separation` → `ncc_vs_frame_separation`, the collect-dict keys `rel_edges` / `rel_counts` → `rel_depth_error_edges` / `rel_depth_error_counts`, and the locals `abs_rel` → `abs_rel_depth_error`, `seps` → `frame_seps`.

**Judgment call kept, flagged for pushback:** `bounded_residual` is one expression (`r/(1+|r|)`) but lives in `metrics.py` and is called from `base.py`, so inlining it would put the forward transform in one file and its inverse in another, where they can drift apart.

## Reuse Audit — what is NOT written here, and what supplies it

| Needed | Supplied by |
|---|---|
| Quantiles of a per-pair column | `np.quantile` — or the reader's own, since the raw column ships |
| Quantiles from accumulated per-pixel counts | `scipy.stats.rv_histogram((counts, edges)).ppf(q)` |
| Fraction below arbitrary X | the same object's `.cdf(x)` |
| Incremental accumulation over N² pairs | `counts += np.histogram(bounded_residual(v), bins=edges)[0]` |
| Bin count for a sample size | Rice's rule, `2·n^(1/3)` — numpy implements it as `np.histogram_bin_edges(a, bins="rice")`, but that needs the array in memory, which is the exact thing there is too much of. One line reimplements the rule; the private `np.lib.histograms._hist_bin_rice` is not public API. |
| Folding a signed histogram to \|r\| | `counts[k//2:] + counts[:k//2][::-1]` — symmetric edges, so bin `j` and bin `k−1−j` share \|u\| |
| Normalised patch agreement | `np.corrcoef(a, b)[0, 1]` — this IS the photometric measure |
| Any monotone correlation | `scipy.stats.spearmanr(a, b).statistic`, called **directly** at each site, with **no wrapper and no small-sample guard** — whatever scipy returns is what ships. **Measured correction (Task 5 quality pass):** an earlier revision routed all three through a private module-local helper, and a later one replaced the helper with a `len(rows) >= 3` floor that nulled the key below it. Both are gone. The helper was one function around one scipy call, existing only for one edge case; the floor was worse — this report makes no verdicts, and "your sample is too small to correlate" is a verdict. Every rho ships beside its own sample size (`n_pair_directions` for depth, `n_pairs` for photometric) and above the raw per-pair columns, so a reader seeing `0.9999999999999999` next to a count of 2 discounts it themselves. **Measured on scipy 1.17.1: plain `spearmanr` never raises at small n** — n=3 → 1.0, n=2 → 0.9999999999999999, n=1 → nan, no warning on any of them — and both functions early-return `available: False` on zero rows, so there is no reachable crash path. `nan_policy` is NOT used and is not needed: both producers already drop non-finite rows (`base.py` `sel.any()`, `compute_photometric_ncc`'s `isfinite`), scipy propagates anything that slips through as nan, and only `nan_policy="omit"` can raise. `clean_for_json` writes nan as null, so the JSON stays valid. |
| Rank of each frame | `scipy.stats.rankdata(v)` |
| nan → null so the JSON is valid | `verification.clean_for_json` (`verification.py:346`). **Measured correction (Task 2):** this audit read the name at its definition line and assumed module scope — it was in fact a private closure named `_clean`, **nested inside `_write_report`**, so importing it raised `ImportError` and every Task 2 test failed at collection. Task 2 promoted it to module level; the closure captured nothing but `np` and its own params, so the promotion is behaviour-preserving. Task 2's review then made it public — a leading-underscore name imported across module boundaries is the convention's own signal that it should not be private — so it is now **`clean_for_json`**, still in `verification.py` because that module owns the write. Every `import clean_for_json` in this plan works as written. |
| Frame index for a COLMAP image | `enumerate(sorted(recon.images))` — the existing alignment contract |
| Running accumulation | `np.cumsum` |
| Per-pair error row | `verification.PairStats` (`verification.py:41`) |
| Crop-to-original rescale of depth | `mesh.utils.guided_upsample_depth` (`mesh/utils.py:374`) |
| Epipolar inliers, relative pose, triangulated reprojection error | `pycolmap` via the existing `verify` stage — the report never re-derives them |

### Library sweep — what the installed stack does and does not supply

Probed in `/opt/venv/reconstruction`: **kornia 0.8.2, open3d 0.19.0, scikit-image 0.26.0, opencv 4.13.0, torchmetrics 1.9.0, scipy 1.17.1, pycolmap 4.0.4**. Two questions: is the photometric measure already implemented, and is the depth warp?

**Photometric agreement — nothing to adopt, and nothing hand-rolled either.** No NCC exists in `skimage.metrics`, `kornia.metrics`, or `torchmetrics.functional` (searched for `ncc` / `cross_corr` / `zncc`: zero hits). OpenCV has only `TM_CCOEFF_NORMED`, which is *sliding-window template matching* over an image, not a statistic on two paired vectors — the warp has already put the pixels in correspondence, so there is no window to slide. What the neighbouring libraries do offer is a different measurement: `skimage.metrics.structural_similarity`, `kornia.metrics.ssim`/`psnr`, `normalized_root_mse`. All of them need a **dense image**, and the warp yields a masked scatter of valid pixels, not a rectangle. The measure that fits paired scattered samples is ZNCC, and **ZNCC is `np.corrcoef`** — verified again here to 1e-16 (`mean(z_a·z_b) = 0.5791866664420522` vs `np.corrcoef = 0.5791866664420521`). So the line `np.corrcoef(a, b)[0, 1]` *is* the library implementation; there is no hand-rolled statistic left to replace.

**Depth warp — kornia can do it, and it was rejected on measurement, not taste.** `kornia.geometry.depth.warp_frame_depth` takes ONE `camera_matrix` for both frames, so it cannot express per-frame focals (11% fx spread on omega alone) — unusable. `DepthWarper(pinhole_dst, H, W, mode="nearest")` *can*: separate `PinholeCamera` per view, and `mode="nearest"` is accepted, which matters because bilinear across a depth discontinuity blends two surfaces into a colour present on neither. It fails on the mask:

- **Out-of-frame pixels come back as exact 0, with no mask returned.** Measured: warping a uniform grey image (`0.5` everywhere) through a baseline that pushes every pixel out of frame returns `fraction of output exactly 0 = 1.0`. Correlating that against the source reports disagreement where there is simply no overlap.
- **Behind-camera points are not rejected.** `project_points` divides by `z` with no sign test, so a point at `(-1, -0.5, -2)` and its mirror at `(1, 0.5, 2)` both land on pixel `(30.5, 23.0)` — measured identical. The warp grid cannot supply `in_front`; only the 3D points in frame j can.

Recovering `in_front` means unprojecting and transforming the points anyway, at which point projecting them is two more lines — so the kornia route is **strictly more code** (4×4 intrinsic padding, two `PinholeCamera` objects and a `DepthWarper` constructed per pair inside an O(N·separation) loop, plus torch↔numpy round-trips at original resolution) for the same result. Ten numpy lines stay.

**No batched angle-between-rays utility exists either** (`kornia.geometry`'s `angle` symbols are all rotation conversions; `scipy.spatial.distance.cosine` is 1-D, so it would need a Python loop over millions of pixels). The two-line vectorised `arccos` stands.

**Known duplication, accepted and stated:** `compute_multiview_depth_confidence` already unprojects, transforms, projects, samples nearest and builds `in_front`, and the photometric loop does the same geometry over colour instead of depth. It is not shared, because one runs in torch at model resolution and the other in numpy at original resolution, and unifying them means editing a hot production function with four callers for the benefit of a report. If a third consumer appears, extract then.

**Why one histogram survives, and why its bins are still not a constant:** per-pair per-pixel residuals are `N²·H·W` floats — 2.4e10 at 300 frames — so the depth residual must accumulate into bins fixed *before* the loop starts. That is the whole reason binning happens at all, and it rules out deriving the bins from the data itself: a first pass to find the range would double the most expensive stage in the report. But the bins need two things, a range and a resolution, and both are recoverable without looking at a single residual. The bounded axis fixes the range at `(−1, 1)` by construction. The resolution follows from the sample count, which the caller knows before the loop as `n_pairs · H · W`. So `residual_bin_edges(n_samples)` computes the edges up front, ships them in the collect dict beside the counts, and no constant survives. Every other quantity reduces to one scalar per pair (a few thousand floats) and ships as a raw column a reader can bin at any resolution they choose.

**Why not zarr, and why not parallel (Task 8 measures both).** The report's own output is per-pair scalars — a few thousand rows, kilobytes — plus a few-thousand-element int64 histogram. Streaming that to zarr adds IO and a storage concept to save nothing. The compute that *is* worth saving is different and real: `build_report` runs a **second** dense multiview pass, and when `pointcloud.use_multiview_confidence` is on the creator already ran that exact loop. Reuse is worth more than parallelism, and Task 6 Step 5 scopes it. Parallelising the photometric loop is deferred until Task 8 reports its share of wall clock — it is O(N·2) numpy pairs against an O(N²) GPU pass.

## File Structure

**Create:**
- `collab_splats/geometry/metrics.py`
- `tests/geometry/test_metrics.py`
- `tests/geometry/test_metrics_controls.py` — negative controls, separate because they are the load-bearing proof.

**Modify:**
- `collab_splats/geometry/verification.py:41-50, ~234-251` — `PairStats` re-keyed on frame index; the one construction site updated.
- `collab_splats/pointcloud/feedforward/base.py` — `compute_multiview_depth_confidence` gains a `collect` out-param. **No new class, return type unchanged**: it has four production callers (`vggtx.py:354`, `vggt_omega.py:265`, `mapanything.py:446`, `loger.py:440`).
- `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266` — register the `report` stage.
- `tests/wrapper/test_reconstructor.py`, `tests/wrapper/test_refine_stage.py` — six tests there mock the stage methods and pin the config-derived stage list, which an always-on `report` necessarily changes (Task 6). Four further tests added for the wiring the stage-graph test cannot see.
- `configs/README.md`, `tests/pointcloud/test_mv_conf.py`.

**Delete (Task 8):** `evals/scripts/depth_disagreement.py`.

---

### Task 1: Prove `verify` runs end-to-end and measure its cost

The epipolar rows are the only measurement that never touches depth, which is what makes attribution possible. **No `verification.json` exists anywhere in this repo** — `verify` has never completed here.

**Files:** Create `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`

- [ ] **Step 1: Confirm none exists**

```bash
find /workspace/collab-splats -name verification.json -not -path '*/.git/*' 2>/dev/null; echo "exit=$?"
```

Expected: nothing printed. If one IS found, read it, record its frame count, and correct the "never run" claim.

- [ ] **Step 2: Find a scene**

```bash
find /workspace/collab-splats/evals/results -maxdepth 3 -name feedforward.zarr 2>/dev/null | head
```

Needs `feedforward.zarr`, `colmap/sparse/0/` and a `frames.zarr`. Record it as `$SCENE`.

- [ ] **Step 3: Run it, timed, in tmux**

```bash
tmux new-session -d -s verify_measure \
  '/opt/venv/reconstruction/bin/python -c "
import logging, time, pathlib
logging.basicConfig(level=logging.INFO)
from collab_splats.wrapper.reconstructor import Reconstructor
r = Reconstructor.from_config_file(pathlib.Path(\"configs/base.yaml\"))
t0 = time.time()
r.verify(overwrite=True)
print(f\"VERIFY_SECONDS={time.time()-t0:.1f}\")
" 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/ee7cc0e1-beee-4d06-908d-0a6838558f0b/scratchpad/verify_measure.log'
```

The constructor call is indicative — use whatever the chosen scene needs (`docs/examples/run_pipeline_remote.py` has the driver pattern). The measurement is the deliverable.

Watch: `tmux attach -t verify_measure`. Memory: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

- [ ] **Step 4: Record it**

Write `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
# Scene error report — measured numbers

## Task 1: epipolar cost (verify)

- Scene / frames / backbone / matcher: <...>
- Wall clock: <VERIFY_SECONDS> s
- Peak rss: <GB> / 46.6 GB
- Pairs generated: <n from verification.json>
- Per-pair: <ms>
- Extrapolated to 300 frames: <min>. **Measured correction to this line:** pairs are NOT a sliding window. `pycolmap.SequentialPairingOptions` defaults to `quadratic_overlap=True` with `overlap=10` and `verification.py` never overrides it, so gaps are powers of two — measured on 300 frames: 1→299, 2→298, 4→296, 8→289, 16→267, 32→225, 64→107, 128→56, 256→28, **1,865 pairs total**, of which 191 span 64+ frames. Per-frame aggregation in Task 6 must decide explicitly whether a gap-256 pair is evidence about either endpoint's local pose.
- Image names in pair_stats look like: <paste two>
- Do those names sort in capture order? <yes/no — see Task 2 Step 3>

### Did verify complete?
<yes/no. If no: the exact traceback.>
```

The name lines matter as *evidence*, not as an input: Task 2 keys on sorted-image-id position precisely so nothing has to parse them. Record them to confirm that choice was necessary.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(specs): measured epipolar cost for the scene error report

First verification.json ever produced in this repo — the poses-only measurement's
only input was previously unproven. Records wall clock, pair count, per-pair cost
and the 300-frame extrapolation."
```

**If `verify` does not complete:** stop and report. Tasks 2-5 and 7-8 are independent of it, but the report loses its poses-only column and with it the ability to separate pose error from depth error.

---

### Task 2: Re-key `PairStats` on frame index, add the parallax bridge

Two fixes in one dataclass. **Frame index becomes the identity** — the depth path has integer indices and no filenames, verify has real filenames, and a report that joins them needs one key both can produce. Names become optional metadata. The separation gap the distance-vs-error axis needs is then `abs(idx1 - idx2)`, a subtraction, not a field.

The index comes from **position in `sorted(recon.images)`**, which is already this module's alignment contract (`verification.py:99, 136, 144, 150`) and already built as a dict twice (`:193`, `:314`). Nothing parses a filename.

**Files:**
- Modify: `collab_splats/geometry/verification.py:42-50` and the construction at 234-251
- Create: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_metrics.py`:

```python
"""Unit tests for reference-free scene error metrics."""

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import (
    bounded_residual,
    depth_error_in_pixels,
    residual_bin_edges,
)
from collab_splats.geometry.verification import PairStats


def test_pair_stats_is_keyed_on_frame_index():
    """Depth path has indices and no filenames; verify has filenames. Index is the join key."""
    p = PairStats(idx1=0, idx2=4)
    assert (p.idx1, p.idx2) == (0, 4)
    assert p.name1 is None and p.median_rel_depth_error is None


def test_separation_is_a_subtraction_not_a_field():
    """1->4 and 2->5 are both separation 3 — the distance-vs-error axis, derived not stored."""
    assert abs(PairStats(1, 4).idx1 - PairStats(1, 4).idx2) == 3
    assert abs(PairStats(2, 5).idx1 - PairStats(2, 5).idx2) == 3


def test_pair_stats_carries_epipolar_and_depth_together():
    """One row per pair. Two measurements fill different columns of it."""
    p = PairStats(0, 1, name1="f0.png", name2="f1.png", num_matches=500, num_inliers=450,
                  rot_error_deg=0.15, t_direction_error_deg=0.9, median_rel_depth_error=0.02)
    assert p.num_inliers == 450 and p.median_rel_depth_error == pytest.approx(0.02)


def test_depth_error_in_pixels_is_r_times_disparity():
    """delta_d = r * d, and d = f * parallax(rad) for small angles."""
    assert depth_error_in_pixels(0.1, 2.0, 500.0) == pytest.approx(0.1 * np.deg2rad(2.0) * 500.0)


def test_depth_error_in_pixels_scales_linearly_in_r():
    assert depth_error_in_pixels(0.10, 3.0, 500.0) == pytest.approx(
        2 * depth_error_in_pixels(0.05, 3.0, 500.0)
    )


def test_depth_error_in_pixels_shrinks_with_parallax():
    """The whole far-pixel asymmetry: same depth error, less parallax, fewer pixels moved."""
    assert depth_error_in_pixels(0.1, 1.0, 500.0) < depth_error_in_pixels(0.1, 6.0, 500.0)


def test_depth_error_in_pixels_is_none_below_one_pixel_of_disparity():
    """The floor is derived from the focal, not chosen: 1 px of disparity is the limit."""
    f = 500.0
    just_under = np.rad2deg(0.9 / f)  # 0.9 px of disparity
    just_over = np.rad2deg(1.1 / f)
    assert depth_error_in_pixels(0.1, just_under, f) is None
    assert depth_error_in_pixels(0.1, just_over, f) is not None


def test_the_disparity_floor_moves_with_the_focal_length():
    """A longer lens resolves depth at a smaller angle — so the floor cannot be a constant."""
    angle = np.rad2deg(1.5 / 500.0)  # 1.5 px at f=500, but only 0.3 px at f=100
    assert depth_error_in_pixels(0.1, angle, 500.0) is not None
    assert depth_error_in_pixels(0.1, angle, 100.0) is None


def test_depth_error_in_pixels_uses_magnitude_not_sign():
    assert depth_error_in_pixels(-0.1, 2.0, 500.0) == pytest.approx(
        depth_error_in_pixels(0.1, 2.0, 500.0)
    )


def test_ratio_against_a_measured_pixel_error_needs_no_second_function():
    """rho is a division at the call site, not an API — measured / equivalent."""
    equiv = depth_error_in_pixels(0.1, 2.0, 500.0)
    assert 5.0 * equiv / equiv == pytest.approx(5.0)


# Scene shapes as (frames, side), and the bin count Rice's rule gives each — measured, which
# is what a test asserts. The sample count here is the UNORDERED pair count, a lower bound on
# what production feeds the histogram (production's mv loop is ordered, N*(N-1)); these numbers
# exist to put the bin count at a realistic magnitude, not to mirror production's expression.
# Round-trip accuracy is a property of the BIN COUNT, not of array size, so the tests ask for a
# real scene's bin count and feed it a small array.
RICE_BINS = {(5, 518): 278, (60, 518): 1560, (300, 518): 4584}


def _n_samples(frames: int, side: int) -> int:
    return frames * (frames - 1) // 2 * side * side


def test_bounded_residual_is_monotone_and_never_leaves_the_bin_range():
    """No value can fall outside the histogram, so nothing is clipped and nothing is dropped."""
    edges = residual_bin_edges(_n_samples(60, 518))
    r = np.array([-1e6, -40.0, -0.3, 0.0, 0.3, 40.0, 1e6])
    u = bounded_residual(r)
    assert np.all(np.diff(u) > 0)
    assert u.min() > edges[0] and u.max() < edges[-1]


def test_bounded_residual_preserves_quantiles_through_the_histogram():
    """A monotone map commutes with quantiles — that is what makes the fixed range safe."""
    rng = np.random.default_rng(0)
    r = np.concatenate([rng.normal(0, 0.03, 200_000), [12.0, -40.0]])
    edges = residual_bin_edges(_n_samples(60, 518))
    counts, _ = np.histogram(bounded_residual(r), bins=edges)
    assert counts.sum() == r.size  # nothing dropped, unlike a clipped fixed range
    u = stats.rv_histogram((counts, edges)).ppf(0.99)
    assert u / (1.0 - abs(u)) == pytest.approx(np.quantile(r, 0.99), abs=1e-4)


def test_bin_edges_are_derived_from_the_sample_count_not_declared():
    """Resolution is the only thing the bounded axis left undetermined, and n determines it."""
    assert len(residual_bin_edges(10**9)) > len(residual_bin_edges(10**6))
    # Rice's rule, k = 2 * n**(1/3), across the scene sizes the report runs at.
    for (frames, side), k in RICE_BINS.items():
        assert len(residual_bin_edges(_n_samples(frames, side))) - 1 == k


def test_bin_edges_always_span_the_whole_bounded_axis():
    """Whatever n is, the range is (-1, 1) by construction — only resolution moves."""
    for n in (1, 10**3, 10**11):
        e = residual_bin_edges(n)
        assert e[0] == -1.0 and e[-1] == 1.0 and len(e) % 2 == 1  # even bin count, so it folds


def test_folding_a_signed_histogram_recovers_absolute_quantiles():
    """The prior depth_disagreement.py numbers are |rel| — a signed histogram must fold first."""
    rng = np.random.default_rng(1)
    r = rng.standard_t(df=1.6, size=300_000) * 0.0055
    edges = residual_bin_edges(_n_samples(60, 518))
    counts, _ = np.histogram(bounded_residual(r), bins=edges)
    half = (len(edges) - 1) // 2
    folded, fedges = counts[half:] + counts[:half][::-1], edges[half:]
    u = float(stats.rv_histogram((folded, fedges)).ppf(0.9))
    assert u / (1.0 - abs(u)) == pytest.approx(np.quantile(np.abs(r), 0.9), rel=0.02)


def test_scipy_supplies_the_correlation_directly():
    """The statistic IS scipy's, off the same columns that ship — no wrapper, no rewrite.

    The fixture is deliberately NOT monotone. A rho of exactly 1.0 is also what Pearson,
    Kendall and a hand-rolled rank difference all return, so a monotone fixture cannot tell
    which statistic actually ran; this one separates Spearman (0.83) from Pearson (0.89).
    """
    rels = [0.01, 0.05, 0.02, 0.08, 0.03, 0.09]
    depths = [2.0, 3.0, 1.0, 9.0, 4.0, 7.0]
    pairs = [_pair(k, k + 1, r, 3.0, depth=z) for k, (r, z) in enumerate(zip(rels, depths))]
    rho = compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_depth"]
    assert rho == stats.spearmanr(depths, [abs(r) for r in rels]).statistic
    # Anchors: a real intermediate rho, and one Pearson does NOT also produce.
    assert 0.0 < rho < 1.0
    assert rho != pytest.approx(float(np.corrcoef(depths, rels)[0, 1]))
```

**Quality-pass correction:** this test as originally written called no repo code at all — it
asserted on `stats.spearmanr(..., nan_policy="omit")`, a path this module deliberately does
not take. It now reads the shipped `correlations` value, which with no wrapper and no guard
between the two is a literal identity — which is the point. It belongs in the
`compute_depth_error` section, after the fixtures it uses.

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.geometry.metrics'`

- [ ] **Step 3: Re-key `PairStats`**

In `collab_splats/geometry/verification.py`, replace the `PairStats` body (lines 42-50):

```python
class PairStats:
    """Measured error for one image pair. Fields are optional per measurement.

    Keyed on FRAME INDEX rather than name. The depth cross-view pass has integer indices and
    no filenames, verify has COLMAP filenames, and the report joins the two — so both need a
    key both can produce. Names stay as metadata for the epipolar half.

    Frame separation (how far apart the two frames are) is abs(idx1 - idx2). It is a
    subtraction, not a field.
    """

    idx1: int
    idx2: int
    # Filled by verify_reconstruction (poses only — never reads depth)
    name1: str | None = None
    name2: str | None = None
    num_matches: int | None = None
    num_inliers: int | None = None
    rot_error_deg: float | None = None  # estimated-vs-model relative rotation, degrees
    t_direction_error_deg: float | None = None  # nan if degenerate
    # Filled by the depth cross-view pass
    n_pixels: int | None = None
    # Both are statistics of ONE quantity: the signed relative depth error
    # (d_sampled - d_expected) / d_expected, frame j's depth read against frame i's.
    # The median is the SCALE reading: a uniform scale factor s between the two frames'
    # depth appears here as exactly s - 1, so 0.02 means frame j is 2% deeper. The IQR is
    # the same population with that bias removed, i.e. the geometric noise. Scale is not a
    # separate column because it is this column: s = 1 + median_rel_depth_error.
    median_rel_depth_error: float | None = None  # signed; s - 1, the pairwise depth scale offset
    iqr_rel_depth_error: float | None = None  # spread with the bias removed — geometric noise
    median_parallax_deg: float | None = None  # how well this pair can see depth at all
    median_depth: float | None = None  # the "worse further away?" axis, as a column
    # Filled by the photometric pass
    photometric_ncc: float | None = None
```

Then update the single construction site. Before the `for pid, g in zip(...)` loop at line 235:

```python
    # Frame index = position in sorted image-id order. That ordering is this module's
    # alignment contract already (features and images are zipped against it above), so the
    # report joins on it instead of parsing digits out of a filename.
    id_to_idx = {iid: k for k, iid in enumerate(sorted(recon.images))}
```

and in the `PairStats(...)` call at line 248:

```python
            PairStats(
                idx1=id_to_idx[id1],
                idx2=id_to_idx[id2],
                name1=im1.name,
                name2=im2.name,
```

Nothing else changes: `asdict(p)` at line 359 picks the new fields up, so `verification.json` gains `idx1`/`idx2` with no serialiser edit.

- [ ] **Step 4: Write the bridge**

Create `collab_splats/geometry/metrics.py`. **Measured correction (Task 2 review):** an earlier draft of this header imported `json`, `Path`, `stats` and the JSON cleaner up front, for the tasks that use them later. `ruff check` reported four F401s on the file, four tasks before those imports were used. **Every task adds its own imports in the commit that first uses them**: Task 4 adds `from scipy import stats`, Task 5 needs nothing new, Task 6 adds `import json`, `from pathlib import Path` and `from collab_splats.geometry.verification import clean_for_json`.

**The lint gate is `ruff check <the files this task touched>`, NOT `bash scripts/lint.sh`** — measured at HEAD on 2026-08-20, the repo-wide script cannot pass and never could during this branch:

| stage | state at HEAD |
|---|---|
| `mypy -p collab_splats --follow-imports=skip` | 97 errors in 26 files — and `lint.sh` is `set -e`, so it aborts here and ruff never runs |
| `ruff check tests/ collab_splats/` | 129 errors, all pre-existing |
| `ruff format --diff tests/ collab_splats/` | 203 files would be reformatted — `pyproject.toml` has **no `[tool.ruff]` section**, so ruff assumes line-length 88 while the repo is black-formatted at 120 |

`.github/workflows/lint.yml` runs `make lint` only on push/PR to **`main`**, so nothing on `refactor/cu121-uv-migration` is gated by it today. Scoped `ruff check` on your own files is still required — the F401 above was a real defect and this branch does eventually reach `main` — but do not attempt to make the repo-wide script pass, and never run repo-wide `black`/`ruff format`: it would reformat 203 files, burying this feature's diff and colliding with the concurrent session.

Header as it stands:

```python
"""Reference-free scene error metrics: depth cross-view, photometric, and verify's epipolar rows.

Report-only. Nothing here feeds back into a reconstruction and nothing emits a verdict — the
output is distributions and how they vary, for a reader to interpret.

Every statistic comes from scipy or numpy. What lives here is the measurement those statistics
are computed over, not a reimplementation of them.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

########################################
# The residual histogram's axis
########################################

def residual_bin_edges(n_samples: int) -> np.ndarray:
    """Histogram edges for n_samples residuals. Nothing here is declared; both halves derive.

    Per-pixel depth residuals are N^2*H*W values (2.4e10 at 300 frames), far too many to hold,
    so they must accumulate into bins fixed BEFORE the loop starts. That rules out reading the
    bins off the data — a pre-pass to find the range would double the most expensive stage in
    the report. Bins need a range and a resolution, and both come from elsewhere:

      range       (-1, 1) by construction, because bounded_residual maps every possible
                  residual into it. No value can fall outside, so nothing is ever clipped.
      resolution  Rice's rule, k = 2 * n**(1/3) — the standard bin count for a sample of size
                  n. numpy implements it as np.histogram_bin_edges(a, bins="rice"), but that
                  wants the array in memory, which is the one thing there is too much of.

    Rice couples the two quantities the right way round: more pixels justify finer bins.
    Measured on a heavy-tailed population shaped like the real baseline, median recovery error
    is 5.1% at 512 bins, 1.7% at 1024, 0.66% at 1560 (a 60-frame scene) and 0.08% at 4584 (300
    frames). Below roughly 20 frames the bins do go coarse — 278 bins and 17% median error on
    a 5-frame scene — but only the PIXEL-level distribution loses resolution there. Per-pair
    medians ship as raw columns and are unaffected.

    The bin count is always even, so the histogram folds to |r| by adding the two halves.
    """
    k = 2 * max(1, int(round(max(int(n_samples), 1) ** (1.0 / 3.0))))
    return np.linspace(-1.0, 1.0, k + 1)


def bounded_residual(rel):
    """Map a relative depth residual onto (-1, 1) so a fixed histogram can never miss it.

    r / (1 + |r|) is monotone over all of R, so quantiles survive the map exactly: the qth
    quantile of the transformed values inverts back to the qth quantile of the originals.
    Invert with u / (1 - |u|).

    This exists so the histogram needs no chosen range and no clipping. A clipped range would
    silently pile the tail into the end bins, and np.histogram drops out-of-range values
    outright — either one makes a later "what fraction is above X" query quietly wrong.
    """
    r = np.asarray(rel, dtype=np.float64)
    return r / (1.0 + np.abs(r))


########################################
# Putting depth error and pixel error on one axis
########################################


def depth_error_in_pixels(rel_residual: float, parallax_deg: float, focal_px: float) -> float | None:
    """Express a relative depth residual in pixels, using this pair's own parallax.

    For a pair with perpendicular baseline B, disparity is d = f*B/Z, and a depth error dZ at
    depth Z moves the point in the image by f*B*dZ/Z^2. Substituting r = dZ/Z:

        delta_d = r * d,   d = f * alpha

    where alpha is the parallax angle in radians. Baseline cancels out of the relation; the
    focal reappears only to state the answer in pixels.

    Converting first and dividing second is the only fair way to compare a pixel error against
    a depth error. The 1/Z hiding inside d is exactly why distant pixels disagree less in
    pixel terms while disagreeing more in depth terms.

    Divide a measured pixel error by this at the call site — no second function needed:
      ~1   one underlying error, seen twice.
      >>1  pixels moved more than any depth error explains, so the excess is pose (pose error
           moves pixels while leaving depths mutually consistent) or appearance.
      <<1  depth disagrees more than pixels do, so the error lies along the ray where this
           pair's baseline cannot see it. Low observability, not necessarily bad depth.

    Returns None when the pair carries under one pixel of disparity, because then it cannot
    see depth at all. That floor is derived from the focal, not chosen: it is the same
    quantity the return value is built from.
    """
    disparity_px = np.deg2rad(parallax_deg) * focal_px
    if disparity_px < 1.0:
        return None
    return abs(rel_residual) * disparity_px
```

- [ ] **Step 5: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/geometry/test_verification.py -v
```

Expected: the 13 new tests pass. **`test_verification.py` may fail** where it constructs `PairStats` positionally or asserts on field order — the identity key changed on purpose. Update those constructions; do not add a compatibility shim.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/geometry/verification.py \
        tests/geometry/test_metrics.py tests/geometry/test_verification.py
git commit -m "feat(geometry): key PairStats on frame index, add the parallax bridge

The depth cross-view pass has integer indices and no filenames; verify has
COLMAP filenames. Joining them needs a key both can produce, so index becomes
the identity and names become metadata.

The index is the position in sorted(recon.images) — already this module's
alignment contract, and already built as a dict twice in the same file. The
previous draft parsed digits out of the filename instead, which assumed names
encode capture order: false for IMG_2039.jpg, for any name with two number
groups, and for any scene whose names do not sort in capture order.

Separation (1->4 and 2->5 are both 3) is abs(idx1-idx2): a subtraction, not a
field. median_depth joins the row so 'does error grow with depth' becomes a
column correlation rather than a binning routine. asdict() picks the new fields
up, so verification.json gains them with no serialiser change.

The bridge: delta_d = r * d with d = f*alpha. Baseline cancels; the focal
reappears only to state the answer in pixels. The 1/Z inside d is the entire
'far pixels disagree less in pixels, more in depth' effect.

Its floor is now derived rather than declared. Under one pixel of disparity the
two views' rays differ by less than a pixel, so the pair cannot see depth at
all — and disparity_px is the same quantity the return value is built from, so
the guard and the value are one expression. It moves with the focal, which a
constant could not.

bounded_residual maps the residual onto (-1,1) so the incremental histogram
needs no chosen range and no clipping. Monotone, so quantiles invert exactly:
measured on 200k residuals plus +-40 outliers, zero values dropped and recovered
quantiles match np.quantile to 5 decimals through p99.9."
```

---

### Task 3: Collect the residual and parallax the mv loop already computes

`compute_multiview_depth_confidence` computes `expected_d` and `sampled_d`, thresholds them to a boolean, and **discards the residual**. It also unprojects `pts_world`, from which the parallax angle is two dot products. Both are free; only the plumbing is new.

**No new class and no return-type change** — an optional dict is filled in place.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:488-661`
- Test: `tests/pointcloud/test_mv_conf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_mv_conf.py`:

```python
from collab_splats.geometry.metrics import residual_bin_edges


def _two_view(scale_j: float = 1.0):
    """Two cameras with a 0.2-unit sideways baseline viewing a constant-depth plane.

    scale_j multiplies frame 1's depth, injecting a known relative residual.
    """
    H = W = 16
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((H, W), 4.0, np.float32), np.full((H, W), 4.0 * scale_j, np.float32)])
    extr = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    extr[1, 0, 3] = -0.2  # world-to-cam translation => camera 1 sits at x=+0.2
    return depth, np.stack([K, K]), extr


def _collect(depth, K, extr, **kw):
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, **kw)
    return out


def test_collect_defaults_to_none_and_output_is_unchanged():
    """The four production creators must see byte-identical output."""
    depth, K, extr = _two_view()
    base = compute_multiview_depth_confidence(depth, K, extr, device="cpu")
    out = {}
    withc = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out)
    assert np.array_equal(base.ratio, withc.ratio)
    assert np.array_equal(base.inlier_count, withc.inlier_count)
    assert np.array_equal(base.valid_count, withc.valid_count)
    assert np.array_equal(base.judged, withc.judged)


def test_collect_fills_index_keyed_rows_and_the_one_histogram():
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    assert out["rel_depth_error_counts"].sum() > 0
    assert (out["pairs"][0].idx1, out["pairs"][0].idx2) == (0, 1)
    assert out["pairs"][0].name1 is None  # index is the key; no filenames invented
    # Edges travel with the counts: they are sized from this scene, so counts alone are unreadable.
    assert len(out["rel_depth_error_edges"]) == len(out["rel_depth_error_counts"]) + 1
    n, h, w = depth.shape
    # n*(n-1), not n*(n-1)//2: the mv loop is ordered and visits both (i,j) and (j,i), so every
    # pixel of every DIRECTION lands in this one histogram. Halving it undersizes Rice's rule.
    assert np.array_equal(out["rel_depth_error_edges"], residual_bin_edges(n * (n - 1) * h * w))


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 => median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view(scale_j=1.1)
    out = _collect(depth, K, extr, rel_thresh=0.5)
    row = next(r for r in out["pairs"] if (r.idx1, r.idx2) == (0, 1))
    assert row.median_rel_depth_error == pytest.approx(0.1, abs=0.02)


def test_signed_residual_is_zero_on_a_consistent_pair():
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_rel_depth_error == pytest.approx(0.0, abs=1e-3)


def test_parallax_angle_matches_geometry():
    """0.2 baseline at depth 4 => atan(0.2/4) ~ 2.86 deg at the principal ray."""
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_parallax_deg == pytest.approx(
        np.degrees(np.arctan(0.2 / 4.0)), abs=0.5
    )


def test_median_depth_lands_on_the_row():
    """The 'worse further away?' axis is a column, not a binning routine."""
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_depth == pytest.approx(4.0, abs=0.2)


def test_occluded_pixels_are_excluded_from_the_residual():
    """Occlusion is absent evidence, not disagreement — it must not pollute the scale bias."""
    depth, K, extr = _two_view()
    depth[1, :, :8] = 0.5  # a near occluder covering half of frame 1
    row = next(r for r in _collect(depth, K, extr)["pairs"] if r.idx1 == 0)
    assert row.median_rel_depth_error == pytest.approx(0.0, abs=1e-3)


def test_residual_is_scale_invariant():
    """Multiplying depth and translation by s must leave the relative residual unchanged."""
    depth, K, extr = _two_view(scale_j=1.1)
    a = _collect(depth, K, extr, rel_thresh=0.5)["pairs"][0]
    s = 7.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    b = _collect(depth * s, K, extr_s, rel_thresh=0.5)["pairs"][0]
    assert a.median_rel_depth_error == pytest.approx(b.median_rel_depth_error, abs=1e-4)
    assert a.median_parallax_deg == pytest.approx(b.median_parallax_deg, abs=1e-3)


def test_a_huge_residual_still_lands_in_the_histogram():
    """The bounded axis means no residual can miss the bins, however large."""
    depth, K, extr = _two_view(scale_j=60.0)
    out = _collect(depth, K, extr, rel_thresh=1e9)
    assert out["rel_depth_error_counts"].sum() > 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v -k "collect or residual or parallax or occluded or median_depth or huge"
```

Expected: FAIL — `TypeError: ... unexpected keyword argument 'collect'`

- [ ] **Step 3: Add the out-param**

In `collab_splats/pointcloud/feedforward/base.py`, add to the imports:

```python
from collab_splats.geometry.metrics import bounded_residual, residual_bin_edges
from collab_splats.geometry.verification import PairStats
```

Add to the signature, after `pair_gate`:

```python
    pair_gate: bool = True,
    collect: dict | None = None,
    device: str = "cuda",
```

Docstring Args addition:

```
        collect: Optional dict, filled IN PLACE with the signed residual and parallax angle
                 this loop already computes and would otherwise discard. Keys: "pairs"
                 (list[PairStats], keyed on frame index), "rel_depth_error_counts" (np.int64 counts) and
                 "rel_depth_error_edges" (the edges those counts are against, sized from this scene's own
                 sample count). The residual is the one per-pixel quantity, so it is the one
                 that has to bin rather than ship raw. The return value is the same either
                 way, so the four production creators are unaffected.
```

Before the `for i in range(N)` loop:

```python
    # Collection is opt-in and fills the caller's dict. The loop already holds everything
    # below; only the plumbing is new. The return contract does not move, because four
    # production creators depend on it.
    if collect is not None:
        # Bin resolution comes from how many residuals there will be, which is exact and known
        # here: every pair contributes at most one per pixel. Nothing is hardcoded, and nothing
        # needs a pre-pass over the data — the pre-pass is the cost the histogram exists to avoid.
        edges = residual_bin_edges(N * (N - 1) * H * W)
        collect["pairs"] = []
        collect["rel_depth_error_edges"] = edges
        collect["rel_depth_error_counts"] = np.zeros(len(edges) - 1, dtype=np.int64)
        cam_centers = cam2world[:, :3, 3]  # (N, 3) world-space camera positions
```

Inside the `for j in range(N)` loop, immediately **after** `valid_sum[i] += counted.reshape(H, W).float()`:

```python
            if collect is None:
                continue

            # Signed relative residual. The sign carries scale bias, the spread carries
            # geometric noise. Same pixels the ratio counts: occluded pixels are absent
            # evidence, and letting them in would drag the bias negative.
            # expected_d > 1e-6 excludes rather than clamps. expected_d is positive upstream, so
            # a clamp would never fire on bad input — it would only manufacture a huge rel from a
            # near-zero denominator, indistinguishable in the histogram from real disagreement.
            # It also rescues the parallax below: ||v_j|| >= expected_d, so bounding one bounds
            # the other, and cos_a can no longer collapse to a fabricated ~90 degrees.
            sel = counted & has_depth & (expected_d > 1e-6)
            if not bool(sel.any()):
                continue
            rel = (sampled_d_flat[sel] - expected_d[sel]) / expected_d[sel]

            # Parallax from the two ray directions, not from f*B/Z. The pinhole form needs a
            # focal length, and focal is exactly what is not comparable across backbones
            # (11% fx spread on omega alone). Ray directions are scale-free.
            pw = pts_world[sel]
            v_i = pw - cam_centers[i]
            v_j = pw - cam_centers[j]
            cos_a = (v_i * v_j).sum(-1) / (v_i.norm(dim=-1) * v_j.norm(dim=-1)).clamp(min=1e-12)
            par = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # The one histogram: too many per-pixel residuals to hold, so they accumulate
            # here. bounded_residual puts them on a finite axis first, so nothing is clipped
            # and nothing is dropped however large the residual.
            v = rel.detach().cpu().numpy()
            collect["rel_depth_error_counts"] += np.histogram(bounded_residual(v), bins=edges)[0]

            # Everything else is one number per pair, so it ships as a raw column instead.
            q = torch.quantile(rel, torch.tensor([0.25, 0.5, 0.75], device=rel.device))
            collect["pairs"].append(
                PairStats(
                    idx1=i,
                    idx2=j,
                    n_pixels=int(sel.sum()),
                    median_rel_depth_error=float(q[1]),
                    iqr_rel_depth_error=float(q[2] - q[0]),
                    median_parallax_deg=float(par.median()),
                    median_depth=float(expected_d[sel].median()),
                )
            )
```

The `return MultiviewConfidence(...)` at the end is **unchanged**.

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v
```

Expected: all pass — the 10 new plus every pre-existing test in the file.

- [ ] **Step 5: Prove the four production callers are untouched**

```bash
/opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_vggtx_creator.py \
  tests/pointcloud/test_vggt_omega_creator.py \
  tests/pointcloud/feedforward/test_mapanything_creator.py \
  tests/integration/test_pipeline_cu121.py -v
```

Expected: same counts as before the change. If any fail, the default-off contract is broken — fix rather than update the test.

- [ ] **Step 6: Check for a circular import**

`base.py` now imports from `geometry.metrics`, which imports from `geometry.verification`, which imports from `localization`.

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.pointcloud.feedforward.base; import collab_splats.geometry.metrics; print('imports clean')"
```

Expected: `imports clean`. **If it cycles**, move `residual_bin_edges` and `bounded_residual` into `base.py` and import them *from* `metrics.py` — neither has dependencies, so the edge always points one way.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_mv_conf.py
git commit -m "feat(pointcloud): collect the residual and parallax the mv loop discards

The loop computes expected_d and sampled_d, thresholds them to a boolean, and
throws the residual away. It also unprojects pts_world, from which parallax is
two dot products. Both are now collected into a caller-supplied dict.

No new class and no return-type change: MultiviewConfidence has four production
callers, so an out-param moves nothing about its contract.

Signed, because the sign separates scale bias (median) from geometric noise
(spread). Parallax from ray directions rather than f*B/Z — focal is exactly what
is not comparable across backbones.

One histogram, not four: per-pixel residuals are N^2*H*W (2.4e10 at 300 frames)
and must accumulate in place, on the bounded axis so no residual can miss the
bins. Parallax, depth and pixel-equivalent are one number per pair, so they ship
as raw columns a reader can bin however they like."
```

---

### Task 4: `compute_depth_error`

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import compute_depth_error


def _pair(i, j, rel, par, n=100, iqr=0.01, depth=4.0):
    return PairStats(i, j, n_pixels=n, median_rel_depth_error=rel, iqr_rel_depth_error=iqr,
                     median_parallax_deg=par, median_depth=depth)


def _collected(pairs):
    """The collect-dict shape compute_multiview_depth_confidence fills.

    Edges are sized for a real 60-frame scene, not for these few hundred fixture pixels:
    quantile recovery is a property of the bin count, so a fixture that derived its own
    coarse bins would be testing a resolution nothing ships at.
    """
    edges = residual_bin_edges(_n_samples(60, 518))
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for p in pairs:
        counts += np.histogram(
            bounded_residual(np.full(p.n_pixels, p.median_rel_depth_error)), bins=edges
        )[0]
    return {"pairs": pairs, "rel_depth_error_counts": counts, "rel_depth_error_edges": edges}


def test_depth_error_reports_grid_and_resolution():
    """Every block stamps its grid — model-res depth with original-res K is a known bug class."""
    m = compute_depth_error(_collected([_pair(0, 1, 0.0, 3.0)]), 500.0, "518x518")
    assert m["grid"] == "model" and m["resolution"] == "518x518"


def test_pair_rows_carry_separation():
    """4->1 and 2->5 both land at 3, so distance-vs-error is a column not a special case.

    One fixture is REVERSED on purpose. The producer's loop is ordered (base.py sets
    idx1=i, idx2=j for both directions), so rows with idx1 > idx2 genuinely ship. With two
    same-direction fixtures, idx2 - idx1 would also give [3, 3] — and that expression sends
    every reversed row negative, sign-flipping error_vs_frame_separation.
    """
    m = compute_depth_error(_collected([_pair(4, 1, 0.02, 3.0), _pair(2, 5, 0.03, 3.0)]), 500.0, "x")
    assert [r["frame_separation"] for r in m["pair_directions"]] == [3, 3]


def test_scale_bias_keeps_its_sign_on_the_row():
    """A pure scale error has a large median and a small spread; the sign must survive."""
    m = compute_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]), 500.0, "x")
    assert m["pair_directions"][0]["median_rel_depth_error"] == pytest.approx(-0.08)
    assert m["pair_directions"][0]["iqr_rel_depth_error"] == pytest.approx(0.005)


def test_pixel_equivalent_lands_on_each_pair_row():
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, 2.0)]), 500.0, "x")
    assert m["pair_directions"][0]["depth_error_px"] == pytest.approx(
        depth_error_in_pixels(0.1, 2.0, 500.0)
    )


def test_pairs_under_one_pixel_of_disparity_report_null_not_zero():
    tiny = np.rad2deg(0.5 / 500.0)  # half a pixel of disparity
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, tiny)]), 500.0, "x")
    assert m["pair_directions"][0]["depth_error_px"] is None
    assert m["pair_directions_under_one_pixel_disparity"] == 1


def test_per_pair_columns_ship_raw():
    """Raw: an exact value round-trips onto the row, so no rounding or binning happened here.

    Key presence alone does not test "raw" — round(x, 2) on every column survives it.
    """
    m = compute_depth_error(_collected([_pair(0, 1, 0.0123456789, 3.0, iqr=0.0098765432)]), 500.0, "x")
    row = m["pair_directions"][0]
    assert row["median_rel_depth_error"] == 0.0123456789  # exact, not approx
    assert row["iqr_rel_depth_error"] == 0.0098765432
    assert row["median_parallax_deg"] == 3.0 and row["median_depth"] == 4.0


def test_output_keys_are_the_contract_task_6_reads():
    """The report writer indexes these by name — a renamed or dropped key breaks it silently."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0) for k in range(4)]
    m = compute_depth_error(_collected(pairs), 500.0, "518x518")
    assert set(m) == {
        "available", "grid", "resolution", "units", "n_pair_directions",
        "residual_histogram", "pair_directions_under_one_pixel_disparity",
        "correlations", "pair_directions",
    }
    assert m["available"] is True
    assert m["n_pair_directions"] == len(pairs) == len(m["pair_directions"])
    h = m["residual_histogram"]
    assert h["total"] == int(np.asarray(h["counts"]).sum()) == sum(p.n_pixels for p in pairs)


def test_the_correlation_reads_residual_MAGNITUDE_not_signed_residual():
    """A growing NEGATIVE bias is growing disagreement — dropping abs() would call it shrinking."""
    pairs = [_pair(k, k + 1, -0.01 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    assert m["correlations"]["error_vs_depth"] > 0.9
    assert all(r["median_rel_depth_error"] < 0 for r in m["pair_directions"])


def test_a_tiny_sample_ships_scipys_answer_NEXT_TO_the_count_that_qualifies_it():
    """Two rows cannot support a rho, and the block publishes scipy's answer anyway.

    Report-only means no verdicts, and "this sample is too small to correlate" is a verdict.
    scipy answers 0.9999999999999999 off two rows by construction and never raises (measured
    on 1.17.1); what makes that safe to publish is that the sample size ships in the same dict
    and the raw rows ship below it, so the reader discounts it rather than inheriting a
    judgement. The value and the count are ONE contract, so both are asserted here.
    """
    m = compute_depth_error(
        _collected([_pair(0, 1, 0.01, 3.0, depth=1.0), _pair(1, 2, 0.02, 3.0, depth=2.0)]), 500.0, "x"
    )
    assert m["correlations"]["error_vs_depth"] == stats.spearmanr([1.0, 2.0], [0.01, 0.02]).statistic
    assert m["correlations"]["error_vs_depth"] == pytest.approx(1.0)  # the spurious perfect fit
    assert m["n_pair_directions"] == 2 == len(m["pair_directions"])  # what makes it readable
    # One row is the same case at the other end: scipy returns nan, still without raising.
    one = compute_depth_error(_collected([_pair(0, 1, 0.01, 3.0, depth=1.0)]), 500.0, "x")
    assert np.isnan(one["correlations"]["error_vs_depth"]) and one["n_pair_directions"] == 1


def test_nothing_in_the_output_grades_the_scene():
    """Report-only: distributions and how they vary, never a verdict for the reader to inherit."""
    pairs = [_pair(k, k + 1, 0.5 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    banned = {"verdict", "status", "grade", "quality", "pass", "passed", "failed", "ok", "healthy"}
    assert banned.isdisjoint(set(m) | set(m["correlations"]) | set(m["pair_directions"][0]))


def test_per_pixel_residual_ships_as_counts_and_edges():
    """The one quantity too large to hold — so any threshold query stays exact.

    0.3, not 0.1: the bounded axis and the residual axis only diverge far from zero. At 0.1
    the un-inverted bin value is 0.0904, inside abs=0.01 of 0.1, so the assertion could not
    see whether the inversion ran at all. At 0.3 it reads 0.2301 un-inverted against 0.2989
    inverted — a 7x margin. Do not lower it back.
    """
    m = compute_depth_error(_collected([_pair(0, 1, 0.3, 3.0)]), 500.0, "x")
    h = m["residual_histogram"]
    assert len(h["bin_edges"]) == len(h["counts"]) + 1 and h["total"] > 0
    assert h["quantiles"]["0.5"] == pytest.approx(0.3, abs=0.01)  # inverted back to a residual


def test_signed_and_folded_quantiles_both_ship():
    """Every prior |rel| number in this repo is absolute, so the signed axis alone is not
    comparable — a negative bias reads as a negative quantile until the histogram is folded.

    -0.3 for the same reason as the test above: at -0.1 the un-inverted bin value passes.
    """
    h = compute_depth_error(_collected([_pair(0, 1, -0.3, 3.0)]), 500.0, "x")["residual_histogram"]
    assert h["quantiles"]["0.5"] == pytest.approx(-0.3, abs=0.01)  # sign kept: scale bias
    assert h["abs_quantiles"]["0.5"] == pytest.approx(0.3, abs=0.01)  # folded: magnitude


def test_rising_residual_with_depth_shows_as_a_positive_correlation():
    """One number replaces the depth-strata routine — the raw columns are in the JSON."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    assert m["correlations"]["error_vs_depth"] > 0.9
    assert "verdict" not in m


def test_constant_depth_gives_nan_which_the_json_writer_turns_into_null():
    """scipy's answer, unwrapped — verification.clean_for_json does the nan -> null pass."""
    pairs = [_pair(k, k + 1, 0.01, 3.0, depth=4.0) for k in range(10)]
    assert np.isnan(compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_depth"])


def test_error_vs_frame_separation_is_reported():
    """Does disagreement grow with how far apart the two frames are?"""
    pairs = [_pair(0, k, 0.005 * k, 3.0) for k in range(1, 20)]
    assert compute_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_frame_separation"] > 0.9


def test_depth_error_is_unavailable_not_a_crash_when_empty():
    m = compute_depth_error(_collected([]), 500.0, "x")
    assert m["available"] is False and "reason" in m
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "depth_error or scale_bias or pixel_equiv or disparity or per_pair or per_pixel or rising or separation"
```

Expected: FAIL — `ImportError: cannot import name 'compute_depth_error'`

- [ ] **Step 3: Write the implementation**

First add this task's import to the top of `collab_splats/geometry/metrics.py` — imports land in the commit that first uses them, because an import written ahead of its use is an F401 under `ruff check` (see the gate note in Task 2 Step 4: check your own files, not `scripts/lint.sh`, which cannot pass at HEAD):

```python
from scipy import stats
```

Then append:

```python
########################################
# Depth cross-view error
########################################


def compute_depth_error(collected: dict, focal_px: float, resolution: str) -> dict:
    """How much the views disagree about depth: scale bias, geometric noise, parallax.

    Evaluated at MODEL resolution on purpose. Depth values are identical under nearest
    upsampling, so evaluating at original resolution returns the same number — but it would
    sample a guided-FILTERED depth map, reporting less disagreement than the model produced.
    That improvement belongs to the smoother, not the model.

    Args:
        collected:  the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px:   mean focal in pixels, used only to state the residual in pixel units.
        resolution: "WxH" of the grid, stamped into the output for the reader.
    """
    pairs = collected["pairs"]
    if not pairs:
        return {
            "available": False,
            "reason": "no overlapping view pairs produced depth residuals",
            "grid": "model",
            "resolution": resolution,
        }

    # One row per pair. Every column is raw, so the reader bins, thresholds and plots.
    rows = [
        {
            "idx1": p.idx1,
            "idx2": p.idx2,
            "frame_separation": abs(p.idx1 - p.idx2),  # how far apart the two frames are
            "n_pixels": p.n_pixels,
            # Signed, so scale reads straight off it: s = 1 + median_rel_depth_error.
            "median_rel_depth_error": p.median_rel_depth_error,
            "iqr_rel_depth_error": p.iqr_rel_depth_error,  # bias removed: geometric noise
            "median_parallax_deg": p.median_parallax_deg,
            "median_depth": p.median_depth,
            # None, not 0.0 — under a pixel of disparity a zero would read as "no error"
            # when it means "cannot tell".
            "depth_error_px": depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, focal_px),
        }
        for p in pairs
    ]

    abs_rel_depth_error = np.array([abs(p.median_rel_depth_error) for p in pairs])
    depths = np.array([p.median_depth for p in pairs], dtype=np.float64)
    frame_seps = np.array([abs(p.idx1 - p.idx2) for p in pairs], dtype=np.float64)
    under_1px = sum(1 for r in rows if r["depth_error_px"] is None)

    # Invert the bounded axis to read quantiles back as real residuals. Monotone, so the qth
    # quantile of the transformed values is the transform of the qth quantile.
    counts, edges = collected["rel_depth_error_counts"], collected["rel_depth_error_edges"]
    grid = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
    rv = stats.rv_histogram((counts, edges))
    quantiles = {}
    for q in grid:
        u = float(rv.ppf(q))
        quantiles[str(q)] = u / (1.0 - abs(u))

    # Same histogram folded to |r|. The edges are symmetric about zero and the bin count is
    # always even, so bin j and bin k-1-j share |u| and the fold is exact rather than a
    # re-binning. Signed quantiles answer "is there scale bias"; folded ones are the quantity
    # every prior |rel| measurement in this repo reports, so they are the comparable column.
    half = (len(edges) - 1) // 2
    rv_abs = stats.rv_histogram((counts[half:] + counts[:half][::-1], edges[half:]))
    abs_quantiles = {}
    for q in grid:
        u = float(rv_abs.ppf(q))
        abs_quantiles[str(q)] = u / (1.0 - abs(u))

    return {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; pixel equivalent in px",
        # DIRECTIONS, not pairs — which is why every key here says so. The mv loop is ordered:
        # (i,j) and (j,i) are separate rows with genuinely different values, because occlusion
        # is asymmetric — a pixel hidden looking one way is visible looking the other. The
        # photometric measurement and verify's epipolar block both count UNORDERED pairs under
        # the key "n_pairs", and a reader comparing the three would otherwise see a phantom 2x.
        "n_pair_directions": len(pairs),
        # The one pre-binned output, because it is the one per-pixel quantity. Counts plus
        # edges keeps threshold queries exact: rv_histogram(...).cdf(bounded_residual(x))
        # answers "what fraction of pixels fall below x" at any x.
        "residual_histogram": {
            "counts": counts.tolist(),
            "bin_edges": edges.tolist(),
            "total": int(counts.sum()),
            "quantiles": quantiles,
            "abs_quantiles": abs_quantiles,
            "axis": "bins are over r/(1+|r|); invert with u/(1-|u|)",
        },
        "pair_directions_under_one_pixel_disparity": under_1px,
        "correlations": correlations,
        "pair_directions": rows,
    }
```

`correlations` is built just above the return — direct scipy, no wrapper and no small-sample
guard:

```python
    # Two questions, one number each, straight from scipy — whatever it returns, unfiltered.
    # A small sample makes rho meaningless (measured on scipy 1.17.1: n=2 gives
    # 0.9999999999999999, n=1 gives nan, neither raises), which is publishable only because
    # "n_pair_directions" ships right beside these numbers: a reader sees rho ~ 1.0 next to a
    # count of 2 and discounts it. error_vs_depth has null_hypothesis below to read against;
    # positive rho is expected, and near 0 or near 1 are the interesting outcomes.
    correlations = {
        "error_vs_depth": float(stats.spearmanr(depths, abs_rel_depth_error).statistic),
        "error_vs_frame_separation": float(stats.spearmanr(frame_seps, abs_rel_depth_error).statistic),
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 34 passed — 18 already in the file from Tasks 2 and 3, plus this task's 16.
**Measured: 35.** Task 4's review added one more test than this plan text lists (nothing
pinned the success-path key set, so six mutations survived). Treat the count as a floor.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py
git commit -m "feat(geometry): compute_depth_error with scale/noise separation

Signed median is the scale bias, spread with the bias removed is geometric
noise — a pure scale error has a large median and small spread, a pose error the
reverse. Under one pixel of disparity the pixel equivalent is null, never 0.0: a
zero would read as 'no error' when it means 'cannot tell'.

'Does error grow with depth' and 'does error grow with frame separation' are one
stats.spearmanr call each over columns the pair rows already carry, replacing a
depth-stratification routine and a fixed bin count. The producers already ship finite
columns, and verification.clean_for_json turns the constant-column nan into null, so
there is no wrapper and no small-sample floor at all — scipy's answer ships as-is, next to
the n_pair_directions count that lets a reader size it. See the audit table at the top of
this plan. Raw columns ship too, so a reader who wants the binned shape can
build it at any resolution.

The residual histogram is the only pre-binned output, because it is the only
per-pixel quantity, and its quantiles invert back off the bounded axis.

Model resolution on purpose: depth is identical under nearest upsampling, so
original-res evaluation returns the same number while sampling a guided-FILTERED
map, reporting less disagreement than the model produced."
```

---

### Task 5: `compute_photometric_ncc`

The only measurement depending on appearance. **Zero-mean normalised cross-correlation** — verified to be exactly what the previous draft's hand-rolled `normalized_residual` computed (`residual == sqrt(2 − 2·NCC)` to 8 dp), so `np.corrcoef` replaces it. NCC absorbs both the `[0,255]` (VGGT) vs `[0,1]` (MapAnything) split and any exposure change; a raw difference would flag exposure as error.

**One function, both grids.** Depth arrives at model resolution and images at original resolution, so the function upsamples when their shapes disagree. A separate original-resolution wrapper would be a second function for one measurement, split only by which grid it happened to run on.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import compute_photometric_ncc


def _plane(n=2, hw=32, seed=0):
    """hw is a side length, or an (H, W) pair — a square fixture cannot see an H/W swap."""
    h, w = (hw, hw) if isinstance(hw, int) else hw
    rng = np.random.default_rng(seed)
    tex = rng.uniform(0, 255, size=(h, w, 3)).astype(np.float32)
    K = np.array([[40.0, 0, w / 2], [0, 40.0, h / 2], [0, 0, 1.0]], dtype=np.float32)
    return (
        np.stack([tex] * n),
        np.stack([np.full((h, w), 4.0, np.float32)] * n),
        np.stack([K] * n),
        np.stack([np.eye(4, dtype=np.float32)] * n),
    )


def test_identical_poses_and_depth_warp_to_ncc_one():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number."""
    img, d, K, e = _plane()
    a = compute_photometric_ncc(img, d, K, e, max_separation=1)["pairs"][0]
    b = compute_photometric_ncc(img / 255.0, d, K, e, max_separation=1)["pairs"][0]
    assert a["photometric_ncc"] == pytest.approx(b["photometric_ncc"], abs=1e-4)


def test_ncc_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry this measurement exists for."""
    img, d, K, e = _plane()
    shifted = img.copy()
    shifted[1] = shifted[1] * 1.4 + 20.0
    m = compute_photometric_ncc(shifted, d, K, e, max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = compute_photometric_ncc(img, d, K, e, max_separation=1)["pairs"][0]
    dirty = compute_photometric_ncc(noisy, d, K, e, max_separation=1)["pairs"][0]
    assert dirty["photometric_ncc"] < clean["photometric_ncc"]


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    """`available is False` alone does NOT discriminate — the isfinite check below the std
    guard drops the same rows. What the guard buys is that corrcoef is never CALLED on a
    zero-variance patch, so simplefilter("error") is the assertion that pins it."""
    img, d, K, e = _plane()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_photometric_ncc(np.full_like(img, 128.0), d, K, e, max_separation=1)
    assert m["available"] is False


def test_two_overlapping_pixels_do_not_count_as_a_correlation():
    """np.corrcoef on 2 points returns exactly +-1 whatever the values — hence min_samples."""
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, max_separation=1, min_samples=10**9)
    assert m["available"] is False


def test_min_samples_counts_pixels_not_the_ravelled_rgb_values():
    """A 32x32 identity pair overlaps in 1024 pixels and 3072 ravelled values; the floor
    admits it at 1024 and rejects it at 1025, which no value-count reading can produce."""
    img, d, K, e = _plane()
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1024)["pairs"][0]["n_pixels"] == 1024
    assert compute_photometric_ncc(img, d, K, e, max_separation=1,
                                   min_samples=1025)["available"] is False


def test_bounds_are_checked_against_the_right_axis_on_a_NON_SQUARE_frame():
    """W bounds u and H bounds v; on a square frame swapping them changes nothing, and on a
    1920x1080 frame it IndexErrors. Identical poses warp every pixel onto itself, so the
    swap silently drops the 8 rightmost columns — the pixel count discriminates."""
    img, d, K, e = _plane(hw=(24, 32))
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["n_pixels"] == 24 * 32
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_zero_depth_pixels_are_dropped_rather_than_warped_from_the_camera_centre():
    """depth == 0 means "no observation": unprojecting it puts the pixel at frame i's own
    camera centre. Identity poses hide this (the centre lands at z = 0 and in_front already
    drops it), so frame 1 is pulled back along z until the centre is in front of it."""
    img, d, K, e = _plane(n=2, hw=32)
    d = d.copy()
    d[0, :8, :] = 0.0
    e = e.copy()
    e[1, 2, 3] = 2.0
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["pairs"][0]["n_pixels"] == 32 * 32 - 8 * 32


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert all(r["frame_separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    assert compute_photometric_ncc(img, d, K, e, max_separation=1)["available"] is False


def test_photometric_upsamples_model_res_depth_and_lifts_its_K_with_it():
    """One function, both grids — and the K must ride the SAME transform as the depth.

    The crop is a strict sub-region (32x32 taken from a 64x64 canvas at (16, 8)), so the model
    -> original scale is crop_w / model_w = 2 and NOT canvas_w / model_w = 4. Using the canvas
    width doubles the focal, the warp lands 8 px out instead of 4, and NCC collapses — the
    2026-08-11 mesh-collapse bug class, caught here rather than in a mesh.
    """
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    # Model-res depth and K describing the crop only: 16x16 grid over a 32x32 crop.
    model_d = np.stack([np.full((16, 16), 4.0, np.float32)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))
    m = compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords,
                                max_separation=1)
    assert m["available"] is True and m["grid"] == "original"
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.02)


def test_the_upsample_guide_is_normalised_whatever_the_backbones_image_scale(monkeypatch):
    """guided_upsample_depth documents a uint8 guide and divides it by 255 internally.

    FeedforwardResult.images is [0, 255] on VGGT-X and [0, 1] on MapAnything, so an uncoerced
    guide is ~255x too flat on one backbone — measured by the reviewer at 0.398 max / 0.013
    mean depth shift on depths of 1-5 — and a float64 guide raises in OpenCV outright. The two
    scales must therefore lift the SAME depth.

    The fixture depth carries an EDGE, not the constant plane every other upsample test uses:
    the guided filter only consults the guide where depth varies, so a constant map returns the
    same answer under any guide at all and could not see this.
    """
    from collab_splats.mesh import utils as mesh_utils

    real = mesh_utils.guided_upsample_depth
    lifted = []

    def spy(depth, rgb_full, *args, **kwargs):
        out = real(depth, rgb_full, *args, **kwargs)
        lifted.append(out)
        return out

    monkeypatch.setattr(mesh_utils, "guided_upsample_depth", spy)
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.concatenate(
        [np.full((16, 8), 3.0, np.float32), np.full((16, 8), 5.0, np.float32)], axis=1)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    coords = np.tile(np.array([16, 8, 48, 40, 64, 64], dtype=np.float32), (2, 1))

    compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords, max_separation=1)
    compute_photometric_ncc(img / 255.0, model_d, model_K, e, original_coords=coords,
                            max_separation=1)
    assert len(lifted) == 4
    np.testing.assert_array_equal(lifted[0], lifted[2])
    np.testing.assert_array_equal(lifted[1], lifted[3])
    # Anchor: the guide really is load-bearing on this fixture, so the equality above is not
    # two runs of a filter that ignores its guide.
    flat_guide = real(model_d[0], np.zeros((64, 64, 3), np.uint8), (16, 8, 48, 40), (64, 64))
    assert not np.array_equal(lifted[0], flat_guide)


def test_images_that_are_not_the_canvas_the_crops_were_cut_from_are_a_refusal():
    """The crop boxes are in ORIGINAL pixels, so a different-resolution image set misplaces
    every one of them. Same check and same refusal as the native-resolution mesh path.

    Without it the frames.zarr-vs-reconstruction mismatch is silent: the crop still indexes
    (it is in range on the smaller canvas) and simply cuts the wrong region of every frame.
    """
    img, _, _, e = _translated_pair(shift_px=4, hw=64, f=40.0)
    model_d = np.stack([np.full((16, 16), 4.0, np.float32)] * 2)
    model_K = np.stack([np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], np.float32)] * 2)
    # original_coords claim a 128x128 canvas; the images are 64x64.
    coords = np.tile(np.array([16, 8, 48, 40, 128, 128], dtype=np.float32), (2, 1))
    with pytest.raises(ValueError, match="original_coords"):
        compute_photometric_ncc(img, model_d, model_K, e, original_coords=coords,
                                max_separation=1)


def test_photometric_resolution_is_derived_from_the_images_not_declared():
    """It used to be a caller-supplied string, and a 32x32 fixture round-tripped "1920x1080".

    Non-square on purpose: "32x24" also pins the ORDER, which a square frame cannot see.
    """
    img, d, K, e = _plane(hw=(24, 32))
    m = compute_photometric_ncc(img, d, K, e, max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "32x24"
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "ncc or photometric or flat_patch or two_overlapping"
```

Expected: FAIL — `ImportError: cannot import name 'compute_photometric_ncc'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Photometric agreement
########################################


def compute_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray | None = None,
    max_separation: int = 2,
    min_samples: int = 32,
) -> dict:
    """Warp each frame into its neighbours through pose+depth and correlate the RGB.

    The measure is zero-mean normalised cross-correlation: 1.0 is perfect agreement, 0.0 is
    none. np.corrcoef supplies it — NCC of two flattened patches IS their Pearson
    correlation, so there is nothing to write.

    Normalising buys two invariances a raw difference lacks: the [0, 255] (VGGT family) vs
    [0, 1] (MapAnything) image-scale split, so one number compares across backbones; and
    exposure or gain change, which would otherwise swamp the geometry being measured.

    This is the only measurement that reads appearance, so disagreement it sees that the depth
    and epipolar columns do not points at image formation rather than geometry.

    Runs at ORIGINAL resolution on purpose: RGB detail exists only there, and unlike depth
    this is a genuinely resolution-dependent quantity. When depth arrives on the smaller model
    grid it is upsampled here rather than in a separate wrapper.

    Rows are UNORDERED pairs, one per (i, j) with i < j, matching verification.py's epipolar
    block — hence "n_pairs"/"pairs" rather than the depth block's "pair_directions".

    The reported "resolution" is derived from `images` rather than passed in: a free-text
    argument can disagree with the grid the numbers were actually measured on.

    Args:
        images:          (N, H, W, 3) RGB, original resolution.
        depth:           (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics:      (N, 3, 3) K matching `depth`'s grid; rescaled here if depth is.
        extrinsics:      (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        max_separation:  pairs per frame. Appearance agreement between distant frames is
                         dominated by lighting and viewpoint change, not by the error measured
                         here, so this stays O(N*max_separation) rather than O(N^2).
        min_samples:     floor on overlapping PIXELS, which is the quantity `n_pixels` ships.
                         RGB is ravelled before correlating, so np.corrcoef actually sees
                         3x this many values. A floor is needed either way — corrcoef on two
                         values returns exactly +-1 whatever they are.
    """
    N = len(depth)
    ih, iw = images.shape[1:3]

    # Depth on the model grid, images on the original grid: lift depth and its K to match.
    # Pairing one grid's depth with the other grid's K is the 2026-08-11 mesh-collapse bug
    # class, so both move together or neither does.
    if depth.shape[1:] != (ih, iw):
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(ih, iw)}; "
                "original_coords is required to upsample"
            )
        # The crop boxes are in ORIGINAL pixels, so `images` has to be the canvas they were
        # computed against; a same-count set at another resolution misplaces every crop.
        # Same check, same refusal, as the native-resolution mesh path (mesh/utils.py).
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))
        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )
        # Imported here, not at module top: mesh.utils imports pointcloud.feedforward.base,
        # which imports this module for bounded_residual/residual_bin_edges. A top-level
        # import would close that cycle and fail at load time. bundle_adjustment joins it
        # because it pulls in bae/vggt/pypose, which the depth path must not pay for.
        from collab_splats.geometry.bundle_adjustment import _scale_intrinsics_to_original
        from collab_splats.mesh.utils import guided_upsample_depth

        model_h, model_w = depth.shape[1:]
        # The guide is documented uint8 and guided_upsample_depth divides it by 255 internally.
        # FeedforwardResult.images is [0, 255] on VGGT-X but [0, 1] on MapAnything, so an
        # uncoerced guide is ~255x too flat on one backbone (measured: 0.398 max depth shift)
        # and float64 raises in OpenCV outright. That [0, 255] vs [0, 1] split is a property of
        # the BACKBONE, not of a frame, so the scale is decided ONCE off the whole array and
        # only applied per frame. Deciding it per frame lets a nearly-black frame in a [0, 255]
        # scene — a dark room, a tunnel, a lens-capped shot, every pixel under 1.0 — read as
        # [0, 1] and get amplified 255x: measured 117.78/255 mean absolute guide error on such
        # a frame, black turned near-white, with guided_upsample_depth guided by it.
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0
        lifted_d, lifted_K = [], []
        for k in range(N):
            tlx, tly, crx, cry = (float(v) for v in original_coords[k][:4])
            guide = np.clip(np.asarray(images[k]) * rgb_scale, 0, 255).astype(np.uint8)
            # rgb_full is the original-res canvas the crop came from — images[k] already is
            # that, so no re-read. crop_box is original_coords[:4], out_hw the canvas size.
            lifted_d.append(
                guided_upsample_depth(depth[k], guide,
                                      (int(tlx), int(tly), int(crx), int(cry)), (ih, iw))
            )
            # The CROP was resized to the model grid, so the scale is model/crop, not
            # model/canvas, and the crop origin comes back onto the principal point.
            # The K arithmetic that undoes both is the forward's inverse, shared via
            # _scale_intrinsics_to_original; the scale itself is re-derived here because the
            # forward's principal-point guard returns sx = 1.0 on the model-res K we pass.
            sx, sy = model_w / (crx - tlx), model_h / (cry - tly)
            lifted_K.append(_scale_intrinsics_to_original(intrinsics[k], sx, sy, tlx, tly))
        depth, intrinsics = np.stack(lifted_d), np.stack(lifted_K)

    H, W = depth.shape[1:]
    # Derived, never declared: the grid the numbers below are actually measured on.
    resolution = f"{iw}x{ih}"
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))

    rows = []
    for i in range(N):
        # Unproject frame i's pixels to world through its own K and pose. Local names follow
        # the multiview loop in pointcloud/feedforward/base.py (cam2world, pts_world,
        # pts_cam_j, proj_j, in_front) so the two warps read as the same operation.
        pts_cam_i = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam_i, ones], axis=-1).T).T[:, :3]
        # Homogeneous once per i: the j loop re-projects the SAME world points.
        pts_world_h = np.concatenate([pts_world, ones], axis=-1)

        for j in range(i + 1, min(N, i + max_separation + 1)):
            # Project them into frame j and look up the colour that landed there. No occlusion
            # test, unlike the depth loop in pointcloud/feedforward/base.py: that one has
            # frame j's own depth map to compare against, and here there is nothing to test a
            # hidden pixel against. Occluded pixels stay in and read as disagreement.
            pts_cam_j = (extrinsics[j] @ pts_world_h.T).T[:, :3]
            proj_j = (intrinsics[j] @ pts_cam_j.T).T
            z = np.clip(proj_j[:, 2], 1e-6, None)
            # Nearest sampling, matching the depth pass: bilinear across a depth discontinuity
            # blends two surfaces into a colour present on neither.
            ui = np.round(proj_j[:, 0] / z).astype(np.int64)
            vi = np.round(proj_j[:, 1] / z).astype(np.int64)
            in_front = pts_cam_j[:, 2] > 0
            # depth == 0 is "no observation", not a surface 0 away: unprojecting it puts the
            # pixel at frame i's own camera centre, which can land somewhere real in frame j
            # and contribute a colour that pixel never saw. in_front does not cover it —
            # the centre is only behind frame j for some poses.
            ok = in_front & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < min_samples:
                continue
            a = images[i].reshape(-1, 3)[ok].ravel().astype(np.float64)
            b = images[j][vi[ok], ui[ok]].ravel().astype(np.float64)
            # A flat patch has no variance to correlate; corrcoef returns nan, which is
            # dropped rather than counted as agreement. The isfinite check below catches the
            # same rows, but only AFTER corrcoef has divided by zero — on a real scene with
            # sky or a blank wall that is one RuntimeWarning per pair.
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            ncc = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(ncc):
                continue
            rows.append({"idx1": i, "idx2": j, "frame_separation": j - i,
                         "photometric_ncc": ncc, "n_pixels": int(ok.sum())})

    if not rows:
        return {
            "available": False,
            "reason": "no view pairs produced a photometric correlation",
            "grid": "original",
            "resolution": resolution,
        }

    # Read the correlation columns back OFF the rows, so nothing can drift from what ships.
    ncc = np.array([r["photometric_ncc"] for r in rows], dtype=np.float64)
    frame_seps = np.array([r["frame_separation"] for r in rows], dtype=np.float64)

    # Straight from scipy, unfiltered, exactly as the depth block does it. A short scene or a
    # heavily skipped one reaches two rows easily and scipy answers +-1.0 there by
    # construction — which is safe to publish only because "n_pairs" below sits next to the
    # number and the raw rows sit under it. Suppressing it would be a verdict, and this report
    # does not make verdicts.
    correlations = {"ncc_vs_frame_separation": float(stats.spearmanr(frame_seps, ncc).statistic)}

    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalised cross-correlation; 1.0 = perfect agreement",
        # UNORDERED, like verify's epipolar block: one row per pair, and the rho's sample size.
        "n_pairs": len(rows),
        "correlations": correlations,
        "pairs": rows,
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 38 passed. **Measured: 51** — Task 4 ended at 35, not the 34 this plan predicted,
and Task 5 shipped 16 tests rather than the 10 listed here (the extra 6 pin behaviours this
task's own implementation body specifies but left untested — notably that the warp moves
pixels at all, since every one of the 10 tests below uses identity poses).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py
git commit -m "feat(geometry): compute_photometric_ncc via np.corrcoef, one grid-agnostic function

The previous draft hand-rolled an RMS-of-z-scored-difference and called it
normalized_residual. Measured, that value equals sqrt(2 - 2*NCC) to 8 decimals:
it WAS Pearson correlation, rewritten. np.corrcoef supplies it directly, and NCC
is the name the domain already uses.

Normalising buys invariance to the [0,255] vs [0,1] backbone image-scale split
AND to exposure change, which would otherwise swamp the geometry this measures.

One function for both grids. Depth arrives model-res and images original-res, so
the upsample and the matching K rescale happen here, guarded by a shape check.
The previous draft had a second function whose only job was that lift — one
measurement split in two by which grid it happened to run on. Depth and K move
together or not at all; pairing one grid's depth with the other's K is the
2026-08-11 mesh-collapse bug class.

min_samples and max_separation are keyword defaults rather than module
constants, at the call they tune. min_samples is not decoration: np.corrcoef on
2 points returns exactly +-1 whatever the values."
```

---

### Task 6: `build_report` and the leaf stage

`build_report` is a **function, not a class.** A `Report` class would add a constructor, attributes and a serialiser to a dict that is built once and written once — machinery with no behaviour behind it. Nothing mutates the report, nothing queries it in memory, and nothing subclasses it.

The epipolar half is a **load, not a measurement**: Task 2 made verify write `idx1`/`idx2`, so `verification.json` already has the report's row shape and only `inlier_ratio` is derived.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Modify: `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
import json

from collab_splats.geometry.metrics import _running_error, build_report
from collab_splats.wrapper.reconstructor import LEAF_STAGES, _STAGE_DEPS, _STAGE_ORDER


def test_verify_writes_the_index_keys_so_no_merge_code_is_needed():
    """asdict() serialises whatever fields PairStats has — the shape lives at the source."""
    from dataclasses import asdict
    row = asdict(PairStats(3, 11, name1="a.png", name2="b.png", num_matches=500, num_inliers=450))
    assert row["idx1"] == 3 and row["idx2"] == 11
    assert abs(row["idx1"] - row["idx2"]) == 8
    assert row["num_inliers"] / row["num_matches"] == pytest.approx(0.9)


def test_report_is_a_leaf_stage_depending_only_on_pointcloud():
    assert "report" in LEAF_STAGES
    assert _STAGE_DEPS["report"] == ["pointcloud"]
    assert _STAGE_ORDER.index("report") > _STAGE_ORDER.index("pointcloud")


def test_report_does_not_demote_any_existing_leaf():
    """A new dependency edge would silently break another stage's disk re-run."""
    for s in ("refine", "semantics", "mesh", "localize", "verify"):
        assert s in LEAF_STAGES


def test_running_error_is_sequential_pairs_and_absolute_steps():
    """Signed steps cancel and hide accumulation; a separation-5 pair is a revisit not a step."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.1},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "median_rel_depth_error": -0.1},
            {"idx1": 0, "idx2": 5, "frame_separation": 5, "median_rel_depth_error": 9.9}]
    out = _running_error(rows, "median_rel_depth_error")
    assert out["frame_index"] == [1, 2]  # separation-5 revisit excluded
    assert out["cumulative"] == pytest.approx([0.1, 0.2])  # |-0.1| added, not cancelled


def test_running_error_counts_an_ordered_pair_once():
    """The depth pass emits (i,j) AND (j,i); summing raw rows would double every step."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.10},
            {"idx1": 1, "idx2": 0, "frame_separation": 1, "median_rel_depth_error": -0.20},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "median_rel_depth_error": 0.30},
            {"idx1": 2, "idx2": 1, "frame_separation": 1, "median_rel_depth_error": 0.30}]
    out = _running_error(rows, "median_rel_depth_error")
    # Two steps, not four, and each frame index appears once.
    assert out["frame_index"] == [1, 2]
    # Step 0->1 is mean(|0.10|, |-0.20|) = 0.15, NOT the signed mean (-0.05) and not the sum.
    assert out["cumulative"] == pytest.approx([0.15, 0.45])


def test_running_error_on_unordered_rows_is_the_identity():
    """Epipolar rows are one per unordered pair, so grouping must not alter them."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "rot_error_deg": 0.4},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "rot_error_deg": 0.6}]
    out = _running_error(rows, "rot_error_deg")
    assert out["frame_index"] == [1, 2]
    assert out["cumulative"] == pytest.approx([0.4, 1.0])


def test_running_error_drops_non_finite_and_missing_values():
    """A dead measurement leaves None or nan in the column; neither may enter the cumsum."""
    rows = [{"idx1": 0, "idx2": 1, "frame_separation": 1, "rot_error_deg": 0.4},
            {"idx1": 1, "idx2": 2, "frame_separation": 1, "rot_error_deg": None},
            {"idx1": 2, "idx2": 3, "frame_separation": 1, "rot_error_deg": float("nan")},
            {"idx1": 3, "idx2": 4, "frame_separation": 1, "rot_error_deg": 0.6}]
    out = _running_error(rows, "rot_error_deg")
    assert out["frame_index"] == [1, 4]
    assert out["cumulative"] == pytest.approx([0.4, 1.0])


def test_report_json_is_valid_json_with_no_bare_nan():
    """json.dumps writes a bare NaN, which no strict parser accepts — clean_for_json prevents it."""
    from collab_splats.geometry.verification import clean_for_json
    text = json.dumps(clean_for_json({"rho": float("nan"), "nested": [float("nan"), 1.0]}))
    assert "NaN" not in text
    assert json.loads(text)["rho"] is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "verify_writes or leaf or demote or running_error or bare_nan"
```

Expected: FAIL at collection — `ImportError: cannot import name '_running_error' from 'collab_splats.geometry.metrics'`. `build_report` is missing from the same module, and `_STAGE_DEPS` has no `"report"` key; all three land once the import is satisfied.

- [ ] **Step 3: Write the stage entry point**

First add this task's imports to the top of `collab_splats/geometry/metrics.py` — imports land in the commit that first uses them, because an import written ahead of its use is an F401 under `ruff check` (see the gate note in Task 2 Step 4: check your own files, not `scripts/lint.sh`, which cannot pass at HEAD):

```python
import json
from pathlib import Path

from collab_splats.geometry.verification import clean_for_json
```

Then append:

```python
########################################
# Stage entry point
########################################


def _running_error(rows: list[dict], key: str) -> dict:
    """Cumulative |step| along the trajectory, one step per consecutive-frame pair.

    Sequential pairs only: a separation-5 pair is a revisit, not a step, and summing it would
    count the same ground twice. Absolute values, because signed steps cancel and would hide
    the accumulation this exists to show.

    Rows are grouped by UNORDERED pair before summing. The depth pass is an ordered loop, so
    it emits both (k, k+1) and (k+1, k) — separation 1 in both directions, with genuinely
    different values, because occlusion is asymmetric. Summing the raw rows would add every
    trajectory step twice and repeat every frame_index. Epipolar rows are already one per
    unordered pair, so their groups hold a single member and the mean is the identity: one
    expression serves both channels with no per-channel branch.

    The mean is over |value|, never over the signed value. +0.10 and -0.09 average to +0.005,
    which reads as agreement when the two directions in fact disagree.
    """
    grouped: dict[tuple[int, int], list[float]] = {}
    for x in rows:
        if x.get("frame_separation") != 1 or x.get(key) is None or not np.isfinite(x[key]):
            continue
        lo, hi = sorted((int(x["idx1"]), int(x["idx2"])))
        grouped.setdefault((lo, hi), []).append(abs(float(x[key])))

    steps = sorted(grouped.items())
    return {
        "frame_index": [hi for (_, hi), _ in steps],
        "cumulative": np.cumsum([float(np.mean(v)) for _, v in steps]).tolist(),
    }


def build_report(zarr_path: Path, verification_json: Path, frames_zarr: Path,
                 output_path: Path, backend: str) -> dict:
    """Run every measurement that can run and write report.json. Never raises on a dead one.

    Measurements are attempted independently: a missing confidence array, an absent
    verification.json or an unreadable frames.zarr each disable exactly one of them.

    Nothing here grades the scene, names a cause or flags a frame. Absolute thresholds that
    would justify a verdict are exactly what this stage exists to inform, so inventing them
    now would be a guess dressed as a finding.

    A function, not a class: the report is built once and written once. Nothing mutates it,
    queries it in memory or subclasses it, so a class would add a constructor, attributes and
    a serialiser with no behaviour behind them.
    """
    from collab_splats.pointcloud.feedforward.base import (
        FeedforwardResult,
        compute_multiview_depth_confidence,
    )

    r = FeedforwardResult.load_zarr(zarr_path)
    n = len(r.depth)
    model_res = f"{r.model_width}x{r.model_height}"
    focal_px = float(r.intrinsics[:, 0, 0].mean() + r.intrinsics[:, 1, 1].mean()) / 2.0

    # One dense pass yields the depth residual, the scale split, the parallax angles and the
    # per-pair depth. abs_thresh stays 0.0: scale invariance holds only there, and that is
    # what lets one function serve backbones whose depth scales differ completely.
    collected: dict = {}
    compute_multiview_depth_confidence(
        r.depth, r.intrinsics, r.extrinsics, abs_thresh=0.0, rel_thresh=0.05, collect=collected
    )
    depth_m = compute_depth_error(collected, focal_px, model_res)
    epipolar_m = _load_epipolar(verification_json, image_width=int(r.original_coords[0][4]))
    photometric_m = _run_photometric(r, frames_zarr, n)

    # Per-frame median |residual| — the column both the confidence check and the ranks read.
    per_frame = {}
    for k in range(n):
        v = [abs(p.median_rel_depth_error) for p in collected["pairs"] if k in (p.idx1, p.idx2)]
        if v:
            per_frame[k] = float(np.median(v))

    # Does the model know when it is wrong? Confidence is an INPUT being validated, not an
    # error source, so it gets one correlation rather than a measurement of its own. Absent on
    # older zarr stores, which are never backfilled.
    # scipy directly, unguarded, exactly like the two correlations in metrics.py: no wrapper
    # and no small-sample floor, because withholding a rho is a verdict and this report makes
    # none. Publishing an unguarded rho is only defensible because the count that qualifies it
    # ships beside it, the way `n_pair_directions` and `n_pairs` qualify the other two — so
    # `n_frames` is nested WITH the rho and cannot be read apart from it.
    # It is NOT recoverable from `frame_percentile_ranks`: `ranks` is {} when len(ks) <= 1
    # while this rho's sample is len(per_frame), so the two diverge at exactly the small
    # sample size where the reader needs the count most.
    # Both stay None when there is no confidence array: the rho was never computed, so there
    # is no sample to report — distinct from a computed rho over a tiny sample.
    conf_rho, conf_n = None, None
    if r.confidence is not None:
        conf = np.asarray(r.confidence)
        conf_n = len(per_frame)
        conf_rho = float(
            stats.spearmanr(
                np.array([float(np.median(conf[k])) for k in per_frame], dtype=np.float64),
                np.array(list(per_frame.values()), dtype=np.float64),
            ).statistic
        )

    # Where each frame sits in this scene's own distribution, 0..1. A NUMBER, never a label.
    # Within-scene ranks need no absolute threshold, which sidesteps the fact that pixel and
    # depth units are not comparable across backbones.
    ks = list(per_frame)
    ranks = {}
    if len(ks) > 1:
        rk = (stats.rankdata([per_frame[k] for k in ks]) - 1) / (len(ks) - 1)
        ranks = {int(k): float(x) for k, x in zip(ks, rk)}

    # Does disagreement build along the trajectory?
    # The row key differs by measurement: depth ships ORDERED directions under
    # "pair_directions", epipolar ships unordered pairs under "pairs". _running_error groups by
    # unordered key either way, so only the lookup name changes.
    running = {
        name: _running_error(m.get(rows_key, []), key)
        for name, rows_key, key, m in (
            ("depth", "pair_directions", "median_rel_depth_error", depth_m),
            ("epipolar", "pairs", "rot_error_deg", epipolar_m),
        )
    }

    report = {
        "scene": {"backend": backend, "n_frames": n, "model_resolution": model_res,
                  "zarr": str(zarr_path)},
        "measurements_available": sorted(
            k for k, m in (("epipolar", epipolar_m), ("depth", depth_m), ("photometric", photometric_m))
            if m.get("available")
        ),
        "measurements": {"epipolar": epipolar_m, "depth": depth_m, "photometric": photometric_m},
        "confidence_vs_error": {"spearman": conf_rho, "n_frames": conf_n},
        # Read running_error against error_vs_frame_separation before calling it drift: frame index
        # is a confounded axis, since scene content, motion speed and exposure all track it.
        "running_error": running,
        "frame_percentile_ranks": ranks,
        # Fraction of each ORIGINAL frame the model crop actually reconstructed. VGGTX resizes
        # width to 518 and centre-crops height to 518, so a 16:9 source loses a band with no
        # depth at all — and model-resolution evaluation is structurally blind to it, because
        # the model grid IS the crop.
        "crop_coverage": [
            {"index": k, "covered_fraction": float(
                max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9))}
            for k, c in enumerate(np.asarray(r.original_coords, dtype=np.float64))
        ],
        "notes": {
            "verdicts": "none by design — this describes distributions, it does not grade",
            "units": "scale-free or normalised throughout; 1 recon unit is NOT 1 metre",
            "attribution": "measurements differ in what they depend on; read them against each other",
        },
    }
    # clean_for_json turns every nan into null. json.dumps otherwise writes a bare NaN, which no
    # strict JSON parser accepts; default= handles numpy scalars.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(clean_for_json(report), indent=2, default=lambda o: o.item()))
    logger.info("Wrote %s (%d measurements available)", output_path, len(report["measurements_available"]))
    return report


def _load_epipolar(verification_json: Path, image_width: int) -> dict:
    """Load verify's tables. Not a measurement — verify made it; the matcher is never re-run.

    These rows are the only ones that never touch depth, which is why attribution works at
    all: something that moves here but not in the depth rows is a pose error. They are already
    original-resolution, since verify estimates from original-resolution keypoints.
    """
    p = Path(verification_json)
    if not p.exists():
        return {"available": False, "reason": f"no verification.json at {p} — set "
                "pointcloud.geometric_verification: true or run --stages verify",
                "grid": "original"}
    data = json.loads(p.read_text())

    rows = []
    for s in data.get("pair_stats", []):
        # Same expression verify already aggregates over at verification.py:363
        # (`p.num_inliers / p.num_matches ... if p.num_matches`) — per row here rather than
        # collapsed to a distribution, so it can be joined against the depth rows.
        n_m, n_i = s.get("num_matches") or 0, s.get("num_inliers") or 0
        rows.append({**s, "frame_separation": abs(s["idx1"] - s["idx2"]),
                     "inlier_ratio": (n_i / n_m) if n_m else None})

    frames = []
    for name, fs in sorted(data.get("frame_stats", {}).items()):
        px = fs.get("mean_reproj_error_px")
        # A bare pixel count is not comparable across backbones (a 518 crop against 448x592),
        # so the fraction ships alongside it.
        frames.append({**fs, "name": name,
                       "mean_reproj_error_frac_width": None if px is None else px / image_width})

    return {"available": True, "grid": "original", "resolution": f"width={image_width}",
            "units": "degrees; reprojection in px and as a fraction of image width",
            "source": str(p), "n_pairs": len(rows), "pairs": rows, "frames": frames}


def _run_photometric(r, frames_zarr: Path, n: int) -> dict:
    """Read original-resolution RGB out of frames.zarr and correlate. Never fatal."""
    if not Path(frames_zarr).exists():
        return {"available": False, "reason": f"frames.zarr not found at {frames_zarr}", "grid": "original"}
    try:
        from collab_splats.preproc.frame_store import FrameStore

        # images() returns the selected frames in row order, which is the order the
        # reconstruction indexes by. frame_indices() is NOT that — it holds source-video
        # positions, so using it to index would silently mispair depth with RGB.
        store = FrameStore.open(Path(frames_zarr))
        rgbs = store.images()[:n].astype(np.float32)
        m = len(rgbs)
        # No resolution argument: the function derives it from `rgbs` itself, so the stamped
        # grid cannot disagree with the grid the numbers were measured on.
        return compute_photometric_ncc(
            rgbs, r.depth[:m], r.intrinsics[:m], r.extrinsics[:m],
            original_coords=r.original_coords[:m],
        )
    except Exception as exc:  # noqa: BLE001 — a report must never fail a reconstruction
        logger.warning("photometric measurement failed: %s", exc, exc_info=True)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "grid": "original"}
```

**Both APIs were checked against the source, not written from memory** — `FrameStore` lives in `collab_splats/preproc/frame_store.py:25` (not `sampling.py`), exposes `open` / `__len__` / `image(i)` / `images(idxs=None)` / `records()` / `frame_indices()`, and has **no** `read()`. Re-confirm before running:

```bash
grep -n "def images\|def image\|def open\|def frame_indices" collab_splats/preproc/frame_store.py
grep -n "def guided_upsample_depth" collab_splats/mesh/utils.py
```

Expected: `frame_store.py` shows `images` and no `read`; `guided_upsample_depth` at `mesh/utils.py:374` with signature `(depth, rgb_full, crop_box, out_hw, radius=None, eps=1e-3)`.

- [ ] **Step 4: Register the stage**

`collab_splats/wrapper/reconstructor.py` line 48 — append `"report"`:

```python
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify", "report"]
```

In `_STAGE_DEPS`, after `"verify"`:

```python
    # report loads verification.json when verify has produced it and reports the epipolar
    # channel unavailable when it has not, so its only hard dependency is the reconstruction
    "report": ["pointcloud"],
```

In `_stage_output_exists`, after the `verify` branch:

```python
        if stage == "report":
            return (self.backend_dir / "report.json").exists()
```

In `run_pipeline`'s default stage list, after the `geometric_verification` branch:

```python
            # Always on, no config boolean. Every other diagnostic ships behind a
            # default-false flag, and the one boolean this would have had is the boolean that
            # keeps it off. Affordable because it runs no model and no matcher — it reads the
            # zarr the reconstruction just wrote — and because it never triggers verify: the
            # epipolar channel appears only when geometric_verification was already paid for.
            stages.append("report")
```

In the execution dispatch, after the `verify` branch:

```python
            elif stage == "report":
                self.report(overwrite=overwrite)
```

And after the `verify` method (ends line 1159):

```python
    def report(self, overwrite: bool = False) -> Path:
        """Reference-free error report: three measurements, one report.json. Reports only.

        Never fails a reconstruction — a measurement that cannot run records
        {"available": false, "reason": ...} and the rest still emit.
        """
        out_json = self.backend_dir / "report.json"
        if not overwrite and self._stage_output_exists("report"):
            logger.info("Report exists at %s, skipping", out_json)
            return out_json
        if self._resolve_result() is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # The epipolar rows are loaded when verify has produced them, and reported unavailable
        # when it has not. Building them here regardless would reach around an explicit
        # `geometric_verification: false` and charge every default run verify's cost (measured
        # 47.6 min and +6.29 GB RSS at 300 frames) for a stage that is always on. Degrading to
        # {"available": false, "reason": ...} is the report-only outcome, not a failure.
        verification_json = self.backend_dir / "colmap" / "verification.json"
        if self.config["pointcloud"]["geometric_verification"] and not verification_json.exists():
            try:
                self.verify()
            except Exception:  # noqa: BLE001 — a report must never fail a reconstruction
                logger.warning("verify failed; epipolar rows will be unavailable", exc_info=True)

        # Heavy deps inline so the module imports without GPU/model libs
        from collab_splats.geometry.metrics import build_report

        build_report(
            zarr_path=self.backend_dir / "feedforward.zarr",
            verification_json=verification_json,
            frames_zarr=self.frames_zarr,
            output_path=out_json,
            backend=self.config["pointcloud"]["backend"],
        )
        logger.info("Report written to %s", out_json)
        return out_json
```

**Also ship `source_frame_indices` at the top level of the report** (added after this plan was written; recorded here so a re-read does not drop it). Every per-frame block is keyed by reconstruction index 0..N-1, which is not the source video index once sampling skips frames, so without this map nothing keyed on the source video — a video-quality report, the frame store — can join to this one at all. Derive it from `r.image_paths`, not `FrameStore.frame_indices()`: `image_paths` is always on the result while `frames.zarr` is optional in this stage.

Check the stem against the documented `frame_{idx:06d}` shape **before** parsing it, and yield `None` when it does not match. `FrameStore.frame_idx_from_path` is `int(stem.split("_")[-1])`, which raises only on a non-numeric tail — it reads `IMG_1234` as `1234` and `00019` as `19`. A guessed index is worse than a missing one here, because a downstream join then pairs real frames with the wrong rows and nothing looks broken. Match the stem WHOLE, or a prefixed name slips through the same way. Do not change `frame_idx_from_path` itself — other callers depend on its behaviour. Keep `None` rather than raising; the report must never fail a reconstruction.

- [ ] **Step 5: Scope the duplicated multiview pass**

`build_report` runs a dense multiview pass. When `pointcloud.use_multiview_confidence` is on, the creator ran that same loop during reconstruction — so the report doubles it. This is the one real compute saving available, and it is worth more than parallelism.

```bash
grep -rn "compute_multiview_depth_confidence" --include=*.py collab_splats/ | grep -v "def compute"
```

Record in the measured report: how many of the four creators would have a `collect` dict to hand down, and whether the reconstruction path can pass one through to `build_report`. **Do not implement the hand-down yet** — the default is `use_multiview_confidence: false`, so the report's pass is usually the only one, and Task 8 measures whether the duplication costs anything worth the plumbing.

- [ ] **Step 6: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: `tests/geometry/` all pass; then `SMOKE PASS`.

`tests/wrapper/` will NOT come back untouched, and expecting it to is a defect in an earlier draft of this step. An always-on stage necessarily changes the config-derived stage list, and six tests pin that list while mocking the stage methods — the real `self.report()` then runs against a scene the mocks never wrote to disk and raises `ValueError: No PointcloudResult available`. Update those six (add `report` to the expected stage list, or patch it alongside the other stages); do NOT weaken the raise to route around them. The raise is unreachable in production: the inline path runs `pointcloud` immediately before, and a genuinely failed `pointcloud` raises in `run_pipeline` first.

After that, `tests/wrapper/` shows the **5 pre-existing failures** from the concurrent session's dirty `configs/base.yaml` (`test_init_fills_defaults_from_base_yaml`, `test_mesh_clean_repair_defaults_off`, `test_base_yaml_mesh_has_fidelity_keys`, `test_loger_block_reaches_run_feedforward_as_creator_kwargs`, `test_base_yaml_declares_the_loger_block`) **and no new ones**.

The stage-graph test only reads `_STAGE_ORDER` and `_STAGE_DEPS`, so it cannot see the append, the dispatch branch or the `report.json` marker — deleting any of the three leaves it green. Pin all three separately.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/wrapper/reconstructor.py \
        tests/geometry/test_metrics.py
git commit -m "feat(wrapper): register report as an always-on leaf stage

build_report is a function, not a class. The report is built once and written
once — nothing mutates it, queries it in memory or subclasses it, so a class
would add a constructor, attributes and a serialiser with no behaviour behind
them.

The epipolar half is a load, not a merge: Task 2 made verify write idx1/idx2, so
verification.json already has the report's row shape and only inlier_ratio is
derived. The previous draft renamed columns on the way in, which put the shape
in two places.

Running-error accumulation and crop coverage are written where they are used — a
cumsum and three lines of arithmetic did not need names. Both keep the comments
that made them non-obvious: sequential pairs only (a separation-5 pair is a
revisit, not a step), absolute values (signed steps cancel and hide the
accumulation), and crop coverage is invisible at model resolution because the
model grid IS the crop.

clean_for_json from verification.py does the nan -> null pass. json.dumps otherwise
writes a bare NaN, which no strict parser accepts.

_STAGE_DEPS['report'] == ['pointcloud'], so --stages report re-runs against a
scene pulled from environments-processed with no rerun.py change. Always on with
no config boolean, against repo precedent: the one boolean it would have had is
the boolean that keeps it off."
```

---

### Task 7: Negative controls — prove the measurements separate

A metric that does not move under an injected fault is decoration. **The depth-scale control is load-bearing**: it proves the measurements separate rather than moving together.

**Files:** Create `tests/geometry/test_metrics_controls.py`

**Status: SHIPPED 2026-08-20, then revised under quality review.** Three defects in the snippet this task originally specified were found and fixed before implementation; quality review then found two more Criticals in the shipped file, both confirmed by independent measurement. The snippet below is the file **as shipped after review**, checked byte-identical to `tests/geometry/test_metrics_controls.py` by a programmatic diff rather than by eye. 12 tests, all proved observable by mutation (table in Step 4b).

> **Design finding for Task 8 — parallax is depth-independent on the TARGET side only.**
> `median_parallax_deg` is computed from world points unprojected through the **source** frame's
> depth (`base.py:693`, `pw = pts_world[sel]`), so the source frame's depth selects which points
> the angle is measured to. A depth fault therefore **leaks into the parallax column on the
> source side**: measured, an x1.1 scale on frame 2 moves from-frame-2 parallax by −9.05 to
> −9.15% (the small-angle prediction is 1/1.1 − 1 = −9.09%), while into-frame-2 and untouched
> pairs move by **exactly 0.0**. The split is clean and by direction, and it belongs to the same
> ordered-pair story as `n_pair_directions`.
>
> **Task 8 must not read source-side parallax as a depth-independent quantity.** The separation
> still holds with room to spare — ρ is 0.98 for a pure depth fault against ~634 for a pure pose
> fault, three orders apart — so this does not threaten attribution; it means the parallax column
> is a *pose* quantity only when read on the target side.

- **Defect A — the exposure/depth control was vacuous.** `test_control_exposure_shift_is_invisible_to_the_depth_measurement` called `_pairs` twice on *identical* inputs and asserted equality: it measured determinism, not invariance, and injected no exposure shift anywhere. There is nowhere to inject one — `compute_multiview_depth_confidence` takes depth, intrinsics and extrinsics and no image argument at all — so the claim is structural, not runtime. Replaced by `test_control_depth_measurement_never_sees_appearance`, which asserts the signature via `inspect.signature`. That fails the moment someone gives the depth measurement an appearance channel, which is the event actually worth catching.
- **Defect B — the pose control injected no pose fault.** It never touched `extr`; it divided a hardcoded `5.0` by a computed `equiv`. Rewritten to perturb real extrinsics (camera 1 translated `dy=0.4` along world Y) and to *measure* the resulting pixel motion with a new `_reprojection_shift_px` helper (unproject frame i through the true geometry, project into frame j under both poses, difference). The Y axis is chosen deliberately: the plane `Z = Z0 + TILT*X` contains it, so the surface is invariant under the perturbation and the depth channel is structurally blind while pixels shift by `f*dy/Z`. Measured: shift 3.0150 px, depth residual −1.42e-3, equiv 0.00476 px, ρ ≈ 634.
- **Defect C — the fixture was silently gated out, and the load-bearing control had no data.** Reproducing the specified `_scene()` exactly: after `depth[2] *= 1.1`, `into_2` and `clean`-into-2 were both **empty** and `p[(1, 2)]` raised `KeyError` — only the 6 pairs not touching frame 2 survived. Mechanism confirmed against `_frustum_world_aabbs`: a constant depth map makes `near == far`, so frames 0/1/3 got a zero-thickness slab at z `[4.0000, 4.0000]` and frame 2 one at `[4.4000, 4.4000]`; `_aabbs_overlap` requires overlap on *every* axis, so every pair touching the scaled frame was rejected. This is exactly Step 3's "the fixture is degenerate" case. Fixed by tilting the plane (`Z = Z0 + 0.3*X`, depth rendered per camera in closed form), and the reason is recorded both in `_scene`'s docstring and in a dedicated test, `test_a_constant_depth_fixture_is_silently_gated_out`, so reverting the tilt fails loudly instead of going green on the empty set. Every test that loops now asserts its pair count first.
- **Two further vacuities fixed proactively.** (1) The forward-motion control asserted on `min(...)`, the *worst*-observed pair, which is trivially true of any scene — changed to `max(...)`, the best-observed pair, plus a strafing contrast that must clear the floor. (2) The photometric control as specified would pass on an NCC hardwired to a constant; a pose-fault sensitivity contrast was added (`pose_delta > 0.01` and `pose_delta > 1e6 * exposure_delta`).

**Quality review round (2026-08-20).** Every finding below was reproduced by independent measurement before being acted on; none was taken on assertion.

- **CRITICAL 1 (confirmed) — the parallax-stability test was decoration under a separation claim's name.** `test_control_depth_scale_does_not_move_the_parallax_angles` asserted only over `clean = [k for k in shared if 2 not in k]`, where the delta is **bit-identical zero** — a determinism check, not an invariance check. Worse, the docstring claim was *false* where it had content: parallax on pairs FROM frame 2 moves −9.05 to −9.15%. Rewritten as `test_control_depth_scale_moves_parallax_only_on_the_source_side`, three arms: untouched pairs (exact zero *by construction* — neither frame's depth changed, so the comment says so and the assertion is `==`), pairs INTO frame 2 (exact zero — **the arm with real content**, the target's depth genuinely does not enter), and pairs FROM frame 2 (must move, by the closed-form factor `1/DEPTH_FAULT` at `rel=0.01`; the worst pair sits 7e-4 from prediction, so the tolerance is measured, not guessed). Measured by direction:

  | pair | direction | before | after | delta |
  |---|---|---|---|---|
  | (0,1) (0,3) (1,0) (1,3) (3,0) (3,1) | untouched | — | — | **0.000e+00 exactly** |
  | (0,2) (1,2) (3,2) | INTO frame 2 | — | — | **0.000e+00 exactly** |
  | (2,0) | FROM frame 2 | 6.777373 | 6.163734 | **−9.054%** |
  | (2,1) | FROM frame 2 | 3.350767 | 3.044140 | **−9.151%** |
  | (2,3) | FROM frame 2 | 3.287833 | 2.990354 | **−9.048%** |

  The design consequence is recorded as the Task 8 finding at the top of this task, not only as a test fix. Plan mutation **M4 was replaced**: the old one coupled parallax to a global `depth_t.mean()`, which is shaped to the old test rather than to a bug class. The new M4 makes parallax blind to source depth by renormalising the ray to a fixed range — and it is **arm 3 that fails** (`Obtained: 1.002424344015356` against `0.909090909... ± 0.00909091`), with arms 1 and 2 still passing, which is what proves the rewrite has content rather than merely more assertions.
- **CRITICAL 2 (confirmed) — `rel_thresh=0.5` was an unexplained load-bearing constant.** At 0.5 the faulted scene yields **12** pairs; at the production default 0.05 it yields **9** — the occlusion branch (`sampled_d_flat < expected_d - tol` → `sel` empty → `continue`, no `PairStats`) deletes exactly the three from-frame-2 pairs, which are the ones CRITICAL 1's third arm is about. That is the identical silent-gating trap this file devotes a whole test to for `TILT`, reached by a second mechanism. Now named `CONTROL_REL_THRESH = 0.5` with the measurement in its comment, and **pinned by a test** — `test_the_controls_must_run_above_the_production_rel_thresh` asserts 12 at the control value, 9 at the default, zero from-frame-2 survivors at the default and three into-frame-2 survivors — so moving the controls to the production default fails loudly instead of quietly halving the evidence. Measured pair counts:

  | `rel_thresh` | clean scene | faulted scene | into_2 | from_2 |
  |---|---|---|---|---|
  | 0.05 (production default) | 12 | **9** | 3 | **0** |
  | 0.1 / 0.2 / 0.5 / 0.9 | 12 | 12 | 3 | 3 |

  The separation-axis scene was given the same treatment (its `0.9` became `CONTROL_REL_THRESH` with the count recorded: **24** pairs at 0.05, **30** at 0.2 and above). The forward-motion control keeps the bare default deliberately and now says why: it is a *floor* control, its scenes carry no depth fault, and at `FLOOR_STEP` all pairs survive at the default — so running it at the production value is evidence, not a hazard.
- **IMPORTANT 3 (confirmed) — the forward-motion control was confounded.** Forward step 0.05 against strafe 0.25 is 5x apart, so it compared magnitudes, not directions. At matched baseline the claim as written **dies**: at step 0.25 forward measures 1.078 px, above the 1.0 px floor. Measured at three steps (best pair, both directions):

  | step | forward best | strafe best |
  |---|---|---|
  | 0.05 | 0.2138 px | 0.7183 px (both under floor — no contrast) |
  | **0.10** | **0.4280 px** | **1.4331 px** (floor sits between them) |
  | 0.25 | 1.0778 px | 3.5700 px (both over floor — claim dies) |

  Kept the direction claim and matched the magnitudes at `FLOOR_STEP = 0.1`, where the floor genuinely separates the two directions; both are now read at their **best** pair, symmetric, and the test asserts both the ratio (`> 3.0`, measured 3.35x) and the bracket `fwd < 1.0 < strafe`. Renamed `test_control_forward_motion_is_the_direction_that_falls_under_the_floor`.
- **IMPORTANT 4 (confirmed) — nothing pinned `median_parallax_deg` to truth.** Multiplying parallax by 1.5 survived all ten shipped tests, because `predicted` and `measured` both consume the same value and self-normalise. Closed by `test_the_fixture_parallax_matches_closed_form_geometry`: an independent reimplementation (`_parallax_truth_deg`, sharing the fixture and nothing else — no mv code on its path) checked on four pairs at `rel=0.03` (measured rel error 5e-3 to 1.6e-2), plus one **absolute** pin, `pytest.approx(3.41, abs=0.05)` on pair (0,1). The x1.5 mutation is now M12 and kills two tests.
- **IMPORTANT 5 — `:294` zipped unguarded.** `faulted` had no `n_pairs` guard, so a dropped row would silently misalign and truncate the zip. Fixed: `assert faulted["n_pairs"] == clean["n_pairs"]`, same guard and same reason as everywhere else in the file.
- **IMPORTANT 6 — `_scene`'s docstring described pre-guard code.** It said the flat fixture "passes on the empty set"; that is what the *originally specified* file did. Corrected to say the pair-count guards turn it into a loud failure, and that the empty-set pass is what would happen without them.
- **IMPORTANT 7 — "and only those" at `:163` was false.** Pairs FROM frame 2 also move (−0.096, −0.090, −0.092) and were asserted nowhere. Both halves are now bucketed and asserted against their closed forms.
- **MINOR 8-11, all taken.** (8) Unused params removed: `_scene(tilt=…)` is now genuinely used — `np.full_like(depth, Z0)` was verified **byte-identical** to `_scene(n=4, tilt=0.0)[0]`, so the gating test calls the parameter instead of hand-rolling the array; `hw` and `_texture(hw=, seed=)` are gone in favour of the module constants. (9) `assert shift_px > 10.0 * equiv` was implied by the lines above it and ρ was pinned nowhere; ρ is now pinned directly, `rho == pytest.approx(634.0, rel=0.25)`. (10) The `-= 0.4` pose fault is named once, `POSE_FAULT_DY`, with the world-Y invariance argument attached to the constant rather than restated at each use. (11) One note on the constants block records that `FOCAL=30.0`, `HW=24` and `BASELINE=0.25` are tuned so clean pairs sit above the 1.0 px disparity floor.

**Consequence of pinning ρ, worth recording:** the pose-fault control is now sensitive to M3 (depth residual forced to zero). ρ is a ratio of two measurements, so `rel → 0` drives `equiv → 0` and `rho → inf`, failing the approx; under the old `shift_px > 10.0 * equiv` it passed. This changes the "correct survivor" story from the previous round — that test is no longer a pure pose probe, and that is honest: it reads a ratio, so it depends on both terms.

- [x] **Step 1: The tests, as shipped**

```python
"""Negative controls: each measurement must move under its own fault and stay still under others.

The report aggregates three measurements chosen because their DEPENDENCIES DIFFER — epipolar
reads poses, depth cross-view reads poses+depth, photometric reads poses+depth+appearance.
Attribution works only if that separation is real, so every claim below is a fault of known
magnitude and known location, asserted to land in exactly one channel.

Three things this file deliberately does NOT do:

  * It does not assert on an empty collection. Every test that loops asserts the pair count
    first, because two separate mechanisms delete pairs silently — the frustum gate (see
    ``test_a_constant_depth_fixture_is_silently_gated_out``) and the occlusion branch at the
    production tolerance (see ``test_the_controls_must_run_above_the_production_rel_thresh``).
    A loop over {} passes.
  * It does not divide an invented number by a measured one. The pose control perturbs real
    extrinsics and measures the pixel motion that perturbation actually produces.
  * It does not treat "the fault did not reach here" and "nothing could reach here" as the
    same evidence. Where an arm is exact-zero by construction it says so, and carries a
    second arm that has to move.
"""

import inspect

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import compute_photometric_ncc, depth_error_in_pixels
from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence

########################################
# Constants — every one of these is load-bearing, so none of them is a bare literal
########################################

# FOCAL, HW and BASELINE are jointly tuned against the one-pixel-of-disparity floor: at f=30 a
# 0.25 baseline at range ~4 gives the worst strafing pair ~1.75 px of disparity, clear of the
# floor, while FLOOR_STEP lands a forward pair under it. Moving any of them moves both sides of
# test_control_forward_motion_is_the_direction_that_falls_under_the_floor.
FOCAL = 30.0
HW = 24
BASELINE = 0.25

# The world surface is the plane Z = Z0 + TILT*X. TILT is the whole reason this fixture works;
# see _scene and test_a_constant_depth_fixture_is_silently_gated_out for why it is not zero.
Z0 = 4.0
TILT = 0.3

# The injected depth fault. x1.1 is deliberately larger than the production rel_thresh of 0.05,
# because a fault inside the tolerance is not a fault the measurement is supposed to report.
DEPTH_FAULT = 1.1

# Every control that injects a fault runs at this tolerance, NOT the production default of 0.05.
# At 0.05 the occlusion branch deletes the very pairs the fault produced: a from-frame-2 pair
# reads sampled < expected - tol, which the measurement classifies as OCCLUDED (absent evidence)
# and drops from the collection entirely. Measured: 12 ordered pairs here, 9 at the default,
# with all three from-frame-2 pairs gone. Pinned by
# test_the_controls_must_run_above_the_production_rel_thresh, so moving the controls "back to
# the default" fails loudly instead of quietly measuring six clean pairs.
CONTROL_REL_THRESH = 0.5

# The injected pose fault: camera 1's centre translated along world Y. That axis is chosen
# because the plane Z = Z0 + TILT*X contains it, so the SURFACE IS INVARIANT under the
# perturbation — the depth channel is structurally blind to this fault, which is precisely the
# claim under test — while every projection still shifts by f*dy/Z. An X or Z translation, or a
# rotation, would perturb depth on a slanted plane and confound the control.
POSE_FAULT_DY = 0.4

# Matched baseline for the forward-vs-strafe contrast. Both directions get the SAME magnitude,
# so the comparison is about direction and not about step size.
FLOOR_STEP = 0.1


########################################
# Fixture
########################################


def _scene(n=4, centers=None, tilt=TILT):
    """N cameras viewing the world plane Z = Z0 + tilt*X, depth rendered exactly per camera.

    A SLANTED plane, not a fronto-parallel one, and that is load-bearing rather than cosmetic.
    A constant depth map makes near == far, so ``_frustum_world_aabbs`` produces a zero-thickness
    slab and ``_aabbs_overlap`` — which needs overlap on every axis — rejects any pair whose
    slabs sit at different depths. Scaling one frame's depth is exactly such a displacement, so
    on a flat fixture the depth-scale control has no pairs to measure: measured, the flat version
    yields 6 ordered pairs instead of 12, with every pair touching the scaled frame gone. The
    pair-count guards in each test below turn that into a loud failure; without them the
    assertions would run on an empty set, which is how the originally specified version of this
    file passed. The tilt gives each frustum real depth extent, which is what a real scene has,
    and as a side effect spreads parallax over 3.3-10.1 deg instead of pinning every pixel at one
    angle.

    Rotations stay identity and the plane is independent of Y, which buys two exact properties
    the controls below rely on: depth is a function of the pixel's x ray-component alone, and
    the surface is invariant under camera translation along world Y.
    """
    K = np.array([[FOCAL, 0, HW / 2], [0, FOCAL, HW / 2], [0, 0, 1.0]], dtype=np.float32)
    if centers is None:
        centers = [(BASELINE * k, 0.0, 0.0) for k in range(n)]  # camera k strafing sideways

    # World-to-camera: identity rotation, so t = -C.
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k, c in enumerate(centers):
        extr[k, :3, 3] = -np.asarray(c, np.float32)

    # Ray-plane intersection in closed form. Ray through pixel u is (a, b, 1) with
    # a = (u - cx)/f; substituting C + s*(a, b, 1) into Z = Z0 + tilt*X gives
    # s*(1 - tilt*a) = Z0 + tilt*Cx - Cz, and Z-depth equals s because the ray's z is 1.
    a = (np.arange(HW) - HW / 2) / FOCAL
    depth = np.stack(
        [
            np.tile(((Z0 + tilt * cx - cz) / (1.0 - tilt * a))[None, :], (HW, 1)).astype(np.float32)
            for cx, _, cz in centers
        ]
    )
    return depth, np.stack([K] * n), extr


def _pairs(depth, K, extr, **kw):
    """{(i, j): PairStats} from one collected multiview pass. CPU so the controls need no GPU."""
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, **kw)
    return {(p.idx1, p.idx2): p for p in out["pairs"]}


def _faulted_scene(n=4):
    """The standard depth fault: frame 2 scaled, everything else exact."""
    depth, K, extr = _scene(n=n)
    depth[2] *= DEPTH_FAULT
    return depth, K, extr


def _texture(n):
    """Smooth RGB with a little noise: enough structure to correlate, no aliasing under warp."""
    rng = np.random.default_rng(0)
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    base = 120 + 60 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    imgs = np.stack([np.stack([base + 10 * c for c in range(3)], -1)] * n).astype(np.float32)
    return imgs + rng.normal(0, 2.0, imgs.shape)


def _parallax_truth_deg(depth, K, extr, i, j):
    """Median ray-to-ray angle over frame i's pixels, computed WITHOUT the measurement under test.

    An independent reimplementation, sharing the fixture and nothing else. It exists because
    every other assertion in this file consumes ``median_parallax_deg`` on both sides of a ratio
    and therefore self-normalises: a parallax scaled by a constant is invisible to all of them.
    """
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], -1)
    pts_cam = (np.linalg.inv(K[i]) @ pix.T).T * depth[i].reshape(-1, 1)
    c2w = np.linalg.inv(extr)
    pts_world = (c2w[i] @ np.concatenate([pts_cam, np.ones((H * W, 1))], -1).T).T[:, :3]
    v_i, v_j = pts_world - c2w[i][:3, 3], pts_world - c2w[j][:3, 3]
    cos_a = (v_i * v_j).sum(-1) / (np.linalg.norm(v_i, axis=-1) * np.linalg.norm(v_j, axis=-1))
    return float(np.median(np.rad2deg(np.arccos(np.clip(cos_a, -1.0, 1.0)))))


def _reprojection_shift_px(depth, K, extr_true, extr_faulty, i, j):
    """Median pixel motion in frame j when frame j's pose changes — the error a tracker sees.

    Measured, not assumed: frame i's pixels are unprojected once through the TRUE geometry, then
    projected into frame j under both poses and differenced. This is the only honest way to put
    a pose fault on the same pixel axis the bridge's output lives on.
    """
    H, W = depth.shape[1:]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))
    pts_cam_i = (np.linalg.inv(K[i]) @ pix.T).T * depth[i].reshape(-1, 1)
    pts_world = (np.linalg.inv(extr_true[i]) @ np.concatenate([pts_cam_i, ones], -1).T).T[:, :3]
    pts_world_h = np.concatenate([pts_world, ones], -1)

    def project(E):
        cam = (E[j] @ pts_world_h.T).T[:, :3]
        proj = (K[j] @ cam.T).T
        return proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None), cam[:, 2]

    uv_true, z_true = project(extr_true)
    uv_bad, _ = project(extr_faulty)
    in_front = z_true > 0
    return float(np.median(np.linalg.norm(uv_true[in_front] - uv_bad[in_front], axis=1)))


########################################
# The fixture itself is a control
########################################


def test_the_fixture_produces_every_pair_and_a_real_parallax_spread():
    """Nothing below means anything if a gate has quietly emptied the collection."""
    depth, K, extr = _scene(n=4)
    # No fault injected here, so this one runs at the PRODUCTION default: the clean fixture must
    # survive the shipping tolerance, and only a faulted one needs CONTROL_REL_THRESH.
    p = _pairs(depth, K, extr)
    # ORDERED directions: the mv loop runs (i, j) and (j, i) separately, so 4 frames give 12.
    assert len(p) == 12
    assert min(v.n_pixels for v in p.values()) > 100
    # Real depth extent, which is what keeps the frusta from degenerating to slabs.
    assert depth[0].max() / depth[0].min() > 1.2
    # And real parallax spread, so no control is secretly evaluated at a single angle.
    par = [v.median_parallax_deg for v in p.values()]
    assert max(par) / min(par) > 2.0


def test_the_fixture_parallax_matches_closed_form_geometry():
    """Pins median_parallax_deg to truth, which no ratio in this file can do.

    Every other parallax assertion here feeds the same number into both sides of a ratio, so a
    parallax scaled by a constant self-normalises and survives them all — measured, a x1.5
    scaling passes every other test in this file. This compares against an independent
    reimplementation instead. The residual disagreement is the pixel SET, not the angle: the
    measurement medians over the pixels that survive its own validity and occlusion filters,
    this helper medians over all of them, which is worth ~0.5-1.6% on this fixture.
    """
    depth, K, extr = _scene(n=4)
    p = _pairs(depth, K, extr)
    for i, j in [(0, 1), (1, 2), (0, 3), (2, 0)]:
        assert (i, j) in p
        truth = _parallax_truth_deg(depth, K, extr, i, j)
        assert p[(i, j)].median_parallax_deg == pytest.approx(truth, rel=0.03)
    # Absolute pin too, so a coordinated rescale of BOTH sides still fails.
    assert p[(0, 1)].median_parallax_deg == pytest.approx(3.41, abs=0.05)


def test_a_constant_depth_fixture_is_silently_gated_out():
    """Pins WHY _scene tilts the plane. A flat fixture loses pairs without raising anything.

    This is not a control; it is the recorded reason the fixture looks the way it does. If it
    ever fails the tilt may be dropped — until then, reverting _scene to a constant depth map
    would delete the depth-scale control's evidence while leaving it green.
    """
    flat, K, extr = _scene(n=4, tilt=0.0)
    assert len(_pairs(flat, K, extr, rel_thresh=CONTROL_REL_THRESH)) == 12  # flat alone is fine

    # ...until one frame's depth moves, which slides its zero-thickness frustum off the others.
    flat[2] *= DEPTH_FAULT
    gated = _pairs(flat, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(gated) == 6
    assert not [k for k in gated if 2 in k]  # every pair touching frame 2 is gone


def test_the_controls_must_run_above_the_production_rel_thresh():
    """Pins CONTROL_REL_THRESH the way the tilt is pinned: the default deletes the evidence.

    Same silent-deletion trap as the frustum gate, one layer down. A from-frame-2 pair carries
    sampled < expected - tol at the production tolerance, which the measurement reads as
    OCCLUDED — absent evidence, dropped from the denominator and from the collection — so the
    fault erases its own pairs. Anyone "restoring the default" here gets a red test rather than
    a quieter one.
    """
    depth, K, extr = _scene()
    depth[2] *= DEPTH_FAULT
    assert len(_pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)) == 12

    at_default = _pairs(depth, K, extr)  # production default, rel_thresh=0.05
    assert len(at_default) == 9
    assert not [k for k in at_default if k[0] == 2]  # every FROM-frame-2 pair deleted
    assert len([k for k in at_default if k[1] == 2]) == 3  # INTO-frame-2 pairs survive


########################################
# Depth fault
########################################


def test_control_depth_scale_moves_the_depth_measurement_on_both_sides_of_frame_2():
    """x1.1 on frame 2's depth moves BOTH directions touching frame 2, by different closed forms.

    Three buckets, all asserted, because the ordered-pair loop gives the fault two distinct
    signatures and reporting only one of them would be false:
      INTO frame 2 (i != 2, j == 2): sampled is scaled, expected is clean, so rel = +0.1 exactly.
      FROM frame 2 (i == 2):         the source point is pushed 1.1x along its ray, so expected
                                     is scaled and sampled is clean: rel = 1/1.1 - 1 = -0.0909.
      Neither:                       unreachable by the fault, so ~0.
    """
    depth, K, extr = _faulted_scene()
    p = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(p) == 12
    into_2 = [v.median_rel_depth_error for (i, j), v in p.items() if j == 2]
    from_2 = [v.median_rel_depth_error for (i, j), v in p.items() if i == 2]
    clean = [v.median_rel_depth_error for (i, j), v in p.items() if 2 not in (i, j)]
    assert len(into_2) == 3 and len(from_2) == 3 and len(clean) == 6

    # Tolerances from the measurement, not from what passes: worst element deviates 0.0023 on
    # the into side and 0.0049 on the from side, against the abs=0.01 asserted here.
    assert np.median(into_2) == pytest.approx(DEPTH_FAULT - 1.0, abs=0.01)
    assert np.median(from_2) == pytest.approx(1.0 / DEPTH_FAULT - 1.0, abs=0.01)
    assert np.max(np.abs(clean)) < 0.01  # measured 0.001593, pure resampling noise


def test_control_depth_scale_moves_parallax_only_on_the_source_side():
    """Parallax is pose geometry on the TARGET side only; on the source side a depth fault leaks.

    The measurement computes parallax from world points unprojected through the SOURCE frame's
    depth, so scaling that depth slides every point along its ray and changes the subtended
    angle. Scaling the TARGET frame's depth cannot reach it at all. Measured, for a x1.1 fault:
    into-frame-2 and untouched pairs move by exactly 0.0, from-frame-2 pairs move -9.05 to
    -9.15%, against the small-angle prediction 1/1.1 - 1 = -9.09%.

    This is a real property of the design, not a wart: it says a depth fault is NOT fully absent
    from the parallax column, so source-side parallax must not be read as depth-independent.
    """
    depth, K, extr = _scene()
    before = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    depth[2] *= DEPTH_FAULT
    after = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    shared = set(before) & set(after)
    # Assert the sample BEFORE looping over it: at the production tolerance this set is 9 and
    # the from-frame-2 arm below would silently have nothing in it.
    assert len(before) == len(after) == len(shared) == 12

    untouched = [k for k in shared if 2 not in k]
    into_2 = [k for k in shared if k[1] == 2]
    from_2 = [k for k in shared if k[0] == 2]
    assert len(untouched) == 6 and len(into_2) == 3 and len(from_2) == 3

    # Arm 1 — pairs the fault cannot reach. Exact-zero BY CONSTRUCTION (identical inputs to an
    # identical computation), so this arm pins determinism and nothing more. It is kept because
    # a non-zero here would mean the fault leaked across frames entirely, but it is not evidence
    # of separation on its own; arm 3 is what carries that.
    for key in untouched:
        assert after[key].median_parallax_deg == before[key].median_parallax_deg

    # Arm 2 — the arm with real content. The fault IS in frame 2 and these pairs read frame 2,
    # yet parallax never touches the target's depth, so they too are exactly unchanged.
    for key in into_2:
        assert after[key].median_parallax_deg == before[key].median_parallax_deg

    # Arm 3 — the leak. These MUST move, by the reciprocal of the injected scale.
    for key in from_2:
        ratio = after[key].median_parallax_deg / before[key].median_parallax_deg
        # rel=0.01 is justified by measurement: the worst pair sits 7e-4 from the prediction,
        # the small-angle approximation being exact only in the limit.
        assert ratio == pytest.approx(1.0 / DEPTH_FAULT, rel=0.01)


def test_control_injected_scale_has_a_closed_form_prediction():
    """r=0.1 predicts delta_d = 0.1*d exactly, and the ratio sits at 1 for a pure depth fault."""
    depth, K, extr = _faulted_scene()
    pairs = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert (1, 2) in pairs
    p = pairs[(1, 2)]
    predicted = depth_error_in_pixels(DEPTH_FAULT - 1.0, p.median_parallax_deg, FOCAL)
    measured = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    # Both must be real numbers: None means the pair carries under a pixel of disparity, which
    # would make the comparison below vacuous rather than passing.
    assert predicted is not None and measured is not None
    # Pin the closed form itself, delta_d = r * f * alpha. Without this line the ratio below is
    # invariant to the bridge's functional form — a bridge that ignored the residual entirely
    # would still divide to 1.0.
    assert predicted == pytest.approx(0.1 * np.deg2rad(p.median_parallax_deg) * FOCAL, rel=1e-6)
    assert measured == pytest.approx(predicted, rel=0.3)
    # A pure depth fault: the pixel motion IS the depth motion, so the ratio sits at 1.
    assert measured / predicted == pytest.approx(1.0, abs=0.3)


########################################
# Pose fault
########################################


def test_control_pose_fault_drives_the_ratio_far_above_one():
    """Pixels move while depths stay mutually consistent — the >>1 signature.

    The fault is a real translation of camera 1 along world Y (see POSE_FAULT_DY for why that
    axis): the surface is invariant under it, so the depth channel is structurally blind, while
    every projection still shifts by f*dy/Z.
    """
    depth, K, extr = _scene()
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= POSE_FAULT_DY  # camera 1's centre moves +dy in world Y

    # Measured pixel motion, and a check that it is the size the geometry says it is.
    shift_px = _reprojection_shift_px(depth, K, extr, extr_bad, 0, 1)
    expected_shift = FOCAL * POSE_FAULT_DY / float(np.median(depth[0]))
    assert shift_px == pytest.approx(expected_shift, rel=0.1)
    assert shift_px > 2.0  # a multi-pixel fault, not a rounding artefact

    pairs = _pairs(depth, K, extr_bad, rel_thresh=CONTROL_REL_THRESH)
    assert len(pairs) == 12
    p = pairs[(0, 1)]
    # The depth channel stays as quiet as an unperturbed pair: nothing here exceeds the
    # resampling floor the clean fixture already sits at.
    assert abs(p.median_rel_depth_error) < 0.01
    equiv = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    assert equiv is not None and equiv < 0.05

    # Pin rho itself, not just "it is big". This is the headline number the design rests on:
    # one formula reads 0.98 for a pure depth fault and ~634 here, three orders apart.
    rho = shift_px / equiv
    assert rho == pytest.approx(634.0, rel=0.25)


########################################
# Appearance fault
########################################


def test_control_depth_measurement_never_sees_appearance():
    """Appearance faults cannot reach a measurement that never reads appearance.

    Asserted structurally rather than by injecting an exposure shift, because there is nowhere
    to inject one: compute_multiview_depth_confidence takes depth, intrinsics and extrinsics and
    no image argument at all. A runtime version would have to call it twice on identical inputs
    and would therefore measure determinism, not invariance. This assertion fails the moment
    someone gives the depth measurement an appearance channel, which is the event worth catching.
    """
    params = set(inspect.signature(compute_multiview_depth_confidence).parameters)
    assert params == {
        "depth", "intrinsics", "extrinsics", "depth_masks",
        "abs_thresh", "rel_thresh", "pair_gate", "collect", "device",
    }
    assert not [p for p in params if any(w in p for w in ("image", "rgb", "color", "colour"))]


def test_control_exposure_shift_is_invisible_to_photometric_too():
    """NCC is what buys this — a raw difference would flag it as error."""
    depth, K, extr = _scene(n=4)
    images = _texture(4)
    clean = compute_photometric_ncc(images, depth, K, extr, max_separation=2)
    assert clean["available"] and clean["n_pairs"] == 5
    ncc_clean = [r["photometric_ncc"] for r in clean["pairs"]]
    # A correlation worth being invariant about: measured 0.884-0.956 on this fixture.
    assert min(ncc_clean) > 0.5

    # Gain AND offset on one frame only — the asymmetric case, since a global rescale would
    # also survive a merely shift-invariant measure.
    shifted = images.copy()
    shifted[1] = shifted[1] * 1.6 + 30.0
    after = compute_photometric_ncc(shifted, depth, K, extr, max_separation=2)
    assert after["n_pairs"] == clean["n_pairs"]  # else the zip below misaligns and truncates
    exposure_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, after["pairs"]))
    assert exposure_delta < 1e-9  # measured 3.3e-16

    # Without this half the test is decoration: an NCC hardwired to a constant would pass
    # everything above. A geometric fault of comparable size must move the same number.
    extr_bad = extr.copy()
    extr_bad[1, 1, 3] -= POSE_FAULT_DY
    faulted = compute_photometric_ncc(images, depth, K, extr_bad, max_separation=2)
    assert faulted["n_pairs"] == clean["n_pairs"]  # same guard, same reason
    pose_delta = max(abs(a - b["photometric_ncc"]) for a, b in zip(ncc_clean, faulted["pairs"]))
    assert pose_delta > 0.01  # measured 0.0277
    assert pose_delta > 1e6 * exposure_delta


########################################
# Observability floor and the separation axis
########################################


def test_control_forward_motion_is_the_direction_that_falls_under_the_floor():
    """At the SAME baseline magnitude, forward motion falls under the floor and strafing clears it.

    The magnitudes are matched deliberately. Comparing a small forward step against a large
    sideways one would only restate that a shorter baseline gives less parallax, which is true
    of any scene and says nothing about direction. Both scenes below use FLOOR_STEP, and both
    are read at their BEST-observed pair — asserting on the worst pair would be trivially
    satisfiable. Measured at step 0.1: forward best 0.428 px, strafe best 1.433 px, 3.3x apart
    with the floor sitting between them.
    """
    fwd_depth, fwd_K, fwd_extr = _scene(n=3, centers=[(0.0, 0.0, FLOOR_STEP * k) for k in range(3)])
    fwd = _pairs(fwd_depth, fwd_K, fwd_extr)
    assert len(fwd) == 6
    fwd_best = max(fwd.values(), key=lambda q: q.median_parallax_deg)
    fwd_px = np.deg2rad(fwd_best.median_parallax_deg) * FOCAL

    strafe_depth, strafe_K, strafe_extr = _scene(n=3, centers=[(FLOOR_STEP * k, 0.0, 0.0) for k in range(3)])
    strafe = _pairs(strafe_depth, strafe_K, strafe_extr)
    assert len(strafe) == 6
    strafe_best = max(strafe.values(), key=lambda q: q.median_parallax_deg)
    strafe_px = np.deg2rad(strafe_best.median_parallax_deg) * FOCAL

    # The direction claim: same baseline, several times the parallax.
    assert strafe_px / fwd_px > 3.0
    # The floor claim: the derived one-pixel-of-disparity threshold separates them, and the
    # bridge declines to answer on the forward side rather than emitting an infinity.
    assert fwd_px < 1.0 < strafe_px
    assert depth_error_in_pixels(0.05, fwd_best.median_parallax_deg, FOCAL) is None
    assert depth_error_in_pixels(0.05, strafe_best.median_parallax_deg, FOCAL) is not None


def test_control_separation_axis_has_teeth():
    """Injecting error that grows with frame gap must show as a positive rho."""
    depth, K, extr = _scene(n=6)
    for k in range(6):
        depth[k] *= 1.0 + 0.02 * k  # drift: each frame slightly more scaled than the last
    # Faults injected, so CONTROL_REL_THRESH: at the production default the occlusion branch
    # deletes the widest-gap pairs, which are exactly the ones carrying the signal (measured
    # 24 pairs instead of 30, and the ones lost are the most-drifted).
    p = _pairs(depth, K, extr, rel_thresh=CONTROL_REL_THRESH)
    assert len(p) == 30
    frame_seps = np.array([abs(i - j) for (i, j) in p], dtype=np.float64)
    errs = np.array([abs(v.median_rel_depth_error) for v in p.values()])
    assert stats.spearmanr(frame_seps, errs).statistic > 0.8
```

- [x] **Step 2: Run them**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics_controls.py -v
```

**A failure here is a finding, not a test bug.** If `test_control_depth_scale_moves_the_depth_measurement_by_the_injected_amount` fails, the measurements do not separate and the design's central claim is wrong — stop and report rather than adjusting tolerances.

Result: **10 passed** at first ship, **12 passed** after the review rewrite (two tests added: the closed-form parallax pin and the `rel_thresh` pin). The design's central claim survives. The depth-scale control did *not* fail on its merits — it failed only on the degenerate fixture, which is the Step 3 case below.

- [x] **Step 3: Fix whatever they expose**

No implementation is written speculatively. Fix the defect in `metrics.py` or `base.py`, then re-run. If a control cannot pass because the *fixture* is degenerate (a constant-depth plane gives every pixel nearly the same parallax), fix the fixture — but record why in a comment, because a degenerate fixture that silently passes is how an inert threshold ships.

Result: **no production defect was found; the fixture was the defect** (Defect C above). `collab_splats/geometry/metrics.py` and `collab_splats/pointcloud/feedforward/base.py` are UNCHANGED — the bridge, the pair gate and the NCC all behave as designed. The degeneracy was worse than this step anticipated: a constant-depth plane does not merely flatten parallax, it makes the pair gate *delete* the pairs the load-bearing control needs. Measured on the flat fixture: 6 of the 12 ordered pairs survive, and the 6 that vanish are exactly the ones touching frame 2 — the frame the fault is injected into — because `depth[2] *= 1.1` moves that frame's zero-thickness frustum slab from z=[4.0, 4.0] to [4.4, 4.4] and the AABB overlap test requires every axis.

**How the flat fixture actually fails, corrected.** An earlier revision of this paragraph (commit `93574367`) said the magnitude control fails via `np.median([])` → nan and that the closed-form control raises `KeyError` on `p[(1, 2)]`. That describes the code **before** the Defect C guards, i.e. the version this task originally specified. In the shipped file M1 (`TILT` → 0.0) trips the **pair-count guard first**: `assert len(p) == 12` fires with `6 == 12`, and `assert (1, 2) in pairs` fires before any subscript can raise. The nan and the `KeyError` are what *would* fire without the guards — which is precisely the failure mode the original task text would have hit, and precisely why the guards exist. The `clean max|rel| 0.00159` figure from that same commit is correct and stands (re-measured; a `0.00209` reported in the first round was wrong).

The distinction that made Defect C worth measuring rather than reasoning about survives the correction: on the flat fixture the two depth controls go red, but the parallax-stability control passes — quietly and meaninglessly, on the 6 surviving clean pairs, certifying nothing about the injected fault. The reason is recorded in `_scene`'s docstring and pinned by `test_a_constant_depth_fixture_is_silently_gated_out`.

Measured on the shipped slanted fixture: 12/12 ordered pairs survive clean and scaled, clean `max|rel|` 0.00159, `into_2` median 0.098587 against an injected +0.1, `from_2` median −0.092109 against the closed form `1/1.1 − 1 = −0.090909`, parallax spread 3.29–10.1 deg. Both ρ signatures come out of the one formula: **ρ = 0.98 for a pure depth fault, ρ = 633.85 for a pure pose fault** (reprojection shift 3.015000 px against the predicted `f·dy/Z` = 3.014925 px).

- [x] **Step 4: Full geometry suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
```

Expected: all pass. Result at first ship: **330 passed, 35 warnings in 31.35s.** After the review rewrite: **332 passed, 35 warnings in 28.30s** (+2 for the two added tests).

Full-suite gate (`tests/ -p no:randomly -q`), first ship: **5 failed, 1830 passed, 2 skipped, 1467 warnings in 714.51s** — 1837 collected, i.e. the pre-gate 1827 plus these 10 tests.

Full-suite gate after the review rewrite: **5 failed, 1832 passed, 2 skipped, 1467 warnings in 750.94s (0:12:30)** — 5 + 1832 + 2 = **1839 collected**, i.e. the 1837 above plus the two tests this round adds. Same 5 failures both times, the concurrent session's known dirty-`configs/base.yaml` set (`test_init_fills_defaults_from_base_yaml`, `test_mesh_clean_repair_defaults_off`, `test_base_yaml_mesh_has_fidelity_keys`, `test_loger_block_reaches_run_feedforward_as_creator_kwargs`, `test_base_yaml_declares_the_loger_block`). No sixth.

- [x] **Step 4b: Mutation table — a green suite is not evidence**

Every test was proved observable by breaking the thing it claims to detect and confirming it goes red. **Re-run in full after the review rewrite** — the tests changed, so the previous round's kills do not carry over and none is reported from memory. **13 mutations run, zero survivors, all 12 tests covered.** Each mutation was reverted immediately after; `git diff --stat` on the two production files is empty.

| # | Mutation | Result |
|---|---|---|
| M1 | `TILT` 0.3 → 0.0 (revert to the flat plane this task originally specified) | **7 FAILED**: fixture-integrity, rel-thresh-pin, depth-buckets, parallax-source-side, closed-form, pose-fault, separation-axis |
| M2 | `_aabbs_overlap` → `return True` | FAILED: `test_a_constant_depth_fixture_is_silently_gated_out` |
| M3 | depth residual forced to 0 (`rel = (expected_d[sel] - expected_d[sel]) / expected_d[sel]`) | **4 FAILED**: depth-buckets, closed-form, pose-fault, separation-axis |
| M4 | **(replaced)** parallax made blind to source depth — ray renormalised to a fixed range, `pw = cam_centers[i] + F.normalize(pts_world[sel] - cam_centers[i], dim=-1) * 4.0` | **2 FAILED**: parallax-truth-pin, parallax-source-side — and it is **arm 3** that fails (`Obtained: 1.002424344015356` vs `0.909090909… ± 0.00909091`); arms 1 and 2 pass |
| M5 | `depth_error_in_pixels` drops the residual factor (`return disparity_px`) | 2 FAILED: closed-form, pose-fault |
| M6 | one-pixel disparity floor 1.0 → 0.05 | FAILED: forward-motion |
| M7 | `POSE_FAULT_DY` → 0.0 | **2 FAILED**: pose-fault, photometric-exposure (the constant is now shared, so it reaches both) |
| M8 | add an `images=` param to `compute_multiview_depth_confidence` | FAILED: appearance-structural |
| M9 | NCC → raw mean-absolute-difference | FAILED: photometric-exposure |
| M10 | remove the injected drift (`*= 1.0`) | FAILED: separation-axis |
| M11 | NCC → constant `0.9` | FAILED: photometric-exposure |
| M12 | **(new)** parallax x 1.5 — *the review's survivor*: it passed all 10 tests of the previous round | **2 FAILED**: parallax-truth-pin, pose-fault |
| M13 | **(new)** `CONTROL_REL_THRESH` 0.5 → 0.05, i.e. move the controls onto the production default | **4 FAILED**: rel-thresh-pin, depth-buckets, parallax-source-side, separation-axis |

Two changes to the "correct survivor" story from the previous round, both consequences of the review and both recorded rather than smoothed over:

- **The pose-fault control no longer survives M3.** Pinning ρ (MINOR 9) makes it a ratio of two measurements, so forcing `rel → 0` drives `equiv → 0` and `rho → inf`, failing the approx. Under the old `shift_px > 10.0 * equiv` it passed. That is honest — the test now reads a ratio, so it depends on both terms — but it means the file no longer contains a test that is *provably* blind to the depth channel. The structural claim is instead carried by M8/appearance-structural and by the exact-zero arms of parallax-source-side.
- **M7 now kills two tests, not one.** `POSE_FAULT_DY` is named once (MINOR 10) and shared by the pose-fault and photometric controls, so zeroing it removes the fault from both.

One mutation was anticipated and pre-empted rather than reported as a survivor: the closed-form test's ratio assertion is invariant to the bridge's functional form (a bridge that ignored `rel_residual` entirely would still divide to 1.0), so the explicit `predicted == 0.1 * deg2rad(alpha) * FOCAL` line was added *before* M5 was run — M5 then killed it.

- [x] **Step 5: Commit**

```bash
# metrics.py and base.py are deliberately NOT staged: no production change was needed.
git add tests/geometry/test_metrics_controls.py
git add -f docs/superpowers/plans/2026-08-20-scene-error-report.md
git commit -m "test(geometry): negative controls proving the measurements separate

A metric that does not move under an injected fault is decoration. Each
measurement gets a fault whose magnitude and location are known, and must stay
still under the others'.

The depth-scale control is load-bearing: x1.1 on one frame's depth must move
median_rel_depth_error to ~0.1 on pairs into that frame ONLY, leaving parallax alone. The
bridge makes it quantitative — r=0.1 predicts delta_d = 0.1*d in closed form
with ratio ~1, while a pose fault drives the ratio >>1. Opposite signatures from
one formula.

The separation axis gets teeth too: injected drift that grows with frame gap
must show as a positive rank correlation, or the distance-vs-error column is
inert. The forward-motion control confirms the derived one-pixel-of-disparity
floor triggers on real geometry and that the bridge declines to answer rather
than emitting an infinity.

The fixture is itself a control. A constant-depth plane makes near == far, so
_frustum_world_aabbs yields a zero-thickness slab and the pair gate rejects
every pair touching a frame whose depth was scaled — the load-bearing control
then passes on an empty set. The plane is tilted for that reason, and a test
pins the reason so reverting the tilt fails loudly. Every test that loops
asserts its pair count first.

No production code changed: the bridge, the pair gate and the NCC all behave as
designed. Each control was proved observable by mutation — 11 mutations, zero
survivors."
```

Shipped as `e567b24`. The quality-review rewrite above is a **second commit on the same two files** — `tests/geometry/test_metrics_controls.py` and this plan — with production code again untouched and unstaged. Its mutation count is the 13-row table in Step 4b, not the 11 quoted in the message above, which is left as the historical record of the first commit.

---

### Task 8: Real-scene run, measured numbers, contract, retirement

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`, `configs/README.md`
- Delete: `evals/scripts/depth_disagreement.py`

> **Read before interpreting any parallax column — measured in Task 7, not assumed.**
> `median_parallax_deg` is **pose geometry on the target side only**. It is evaluated over
> world points unprojected through the *source* frame's depth, so a depth fault in frame *i*
> moves the parallax of every pair `(i, ·)` — measured −9.05 to −9.15% for an x1.1 depth
> scale, matching the small-angle prediction `1/1.1 − 1` — while pairs `(·, i)` and pairs
> touching neither frame move by **exactly zero**. Do not write "parallax is depth-independent"
> in the measured report, and do not read a source-side parallax shift as evidence of a pose
> error. ρ still separates the two faults by three orders (0.98 depth vs 634 pose), so
> attribution is unaffected; the caveat is about how the parallax column alone is read.
> Pinned by `test_control_depth_scale_moves_parallax_only_on_the_source_side`.

- [ ] **Step 1: Run it in tmux, timing each measurement**

```bash
tmux new-session -d -s scene_report \
  '/opt/venv/reconstruction/bin/python -c "
import logging, time, pathlib, json
logging.basicConfig(level=logging.INFO)
from collab_splats.geometry.metrics import build_report
root = pathlib.Path(\"evals/results/mv_vggt_omega\")
t0 = time.time()
rep = build_report(root / \"feedforward.zarr\", root / \"colmap\" / \"verification.json\",
                   root / \"frames.zarr\", root / \"report.json\", \"vggt_omega\")
print(f\"REPORT_SECONDS={time.time()-t0:.1f}\")
print(\"available:\", rep[\"measurements_available\"])
print(\"depth pair directions:\", rep[\"measurements\"][\"depth\"][\"n_pair_directions\"])
print(\"photometric pairs:\", rep[\"measurements\"][\"photometric\"][\"n_pairs\"])
print(\"json MB:\", round((root / \"report.json\").stat().st_size / 1e6, 2))
print(json.dumps(rep[\"measurements\"][\"depth\"][\"residual_histogram\"][\"quantiles\"], indent=2))
print(\"correlations:\", rep[\"measurements\"][\"depth\"][\"correlations\"])
" 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/ee7cc0e1-beee-4d06-908d-0a6838558f0b/scratchpad/scene_report.log'
```

Watch: `tmux attach -t scene_report`. Memory: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

Read the per-measurement split out of the INFO log timestamps — the multiview pass, the photometric loop and the JSON write are separately visible. **That split decides the parallelism question**, which is why it is measured here and not designed for in advance.

- [ ] **Step 2: Check the sanity target**

Measured baseline on this store: **median |rel| 0.37%, p90 2.27%, p99 25.67%**, tightening to p90 0.92% at conf>p20.

Those are **absolute** residuals, so the comparable column is `abs_quantiles`, not `quantiles` — the signed median cancels bias against spread and would read far lower for reasons that have nothing to do with agreement. Compare `abs_quantiles`, which is the same histogram folded at zero, hence the same quantity `depth_disagreement.py` measured, read back off the bounded axis.

```bash
/opt/venv/reconstruction/bin/python -c "
import json, pathlib
h = json.loads(pathlib.Path('evals/results/mv_vggt_omega/report.json').read_text())
h = h['measurements']['depth']['residual_histogram']
for q in ('0.5', '0.9', '0.99'):
    print(q, 'abs %.4f%%' % (100 * h['abs_quantiles'][q]), '| signed %.4f%%' % (100 * h['quantiles'][q]))
"
```

Expected: abs ≈ 0.37 / 2.27 / 25.67 %. **If they differ materially, explain the difference before proceeding.** Check first: residual population (`counted & has_depth` here, which the prior script may not have matched). A discrepancy at p99 specifically would implicate the transform inversion, so re-run the round-trip check from Task 2 Step 1 on the real counts.

- [ ] **Step 3: Check the pair-table size**

The report ships every gated pair as a raw row so a reader can re-bin any column. At 300 frames the mv loop's pair count is O(N²) before gating, so record `pairs` and `json MB` from Step 1.

**If `report.json` exceeds ~20 MB**, the fix is to keep the histogram and emit raw rows only for sequential pairs plus the worst 500 by `median_rel_depth_error` — one filter, no new concepts. Record the decision either way; do not add the filter pre-emptively.

- [ ] **Step 4: Rank control**

Run Step 1 against a `mapanything` store and a `vggt_omega` store of the same scene — they differ 1.6× in ATE on chess/seq-01.

```bash
/opt/venv/reconstruction/bin/python -c "
import json, pathlib, numpy as np
for name in ('mapanything', 'vggt_omega'):
    p = pathlib.Path(f'evals/results/{name}/report.json')
    if not p.exists():
        print(name, 'MISSING'); continue
    d = json.loads(p.read_text())['measurements']['depth']
    rel = [abs(r['median_rel_depth_error']) for r in d['pairs']]
    h = d['residual_histogram']['abs_quantiles']  # abs, to match the abs pair medians below
    print(name, 'hist p50', h['0.5'], 'p99', h['0.99'], 'pair p50', float(np.median(rel)))
"
```

**If the report cannot order those two, it will not separate anything.** Record the outcome either way — a null result here is the most important number in the task.

- [ ] **Step 5: Append the measurements**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
## Task 8: report stage, measured

- Scene: evals/results/mv_vggt_omega (60 frames, vggt_omega)
- Wall clock: <REPORT_SECONDS> s | Peak rss: <GB> / 46.6 GB
- Split: multiview <s> | photometric <s> | json write <s>
- Measurements available: <list>
- Pairs: <n> | report.json: <MB> MB  (filter applied: <yes/no>)

### Sanity target (depth residual)
| quantile | measured (`abs_quantiles`) | signed (`quantiles`) | prior (depth_disagreement.py) |
|---|---|---|---|
| median | <x>% | <x>% | 0.37% |
| p90 | <x>% | <x>% | 2.27% |
| p99 | <x>% | <x>% | 25.67% |

Bin count for this scene (Rice, from `n_pairs·H·W`): <k>.

<Agreement, or the explained difference.>

### Rank control (mapanything vs vggt_omega, 1.6x apart in ATE)
| backbone | hist p50 | p99 | pair-median |
|---|---|---|---|

Ordered correctly: <yes/no>. <If no: what that means for the design.>

### Correlations
- error_vs_depth: <rho> over <n_pair_directions> pair directions  (null: sigma_Z ~ Z^2/(f*B) => expect positive, ~linear)
- error_vs_frame_separation: <rho> over <n_pair_directions> pair directions
- confidence_vs_error: <`spearman` rho over `n_frames` frames; both null on stores with no
  confidence array. Read the rho against its own n — nothing here filters a small sample.>
- ncc_vs_frame_separation: <rho> over <n_pairs> pairs

### Disparity floor (derived, = 1 px)
- Focal used: <f> px => floor <deg> deg
- Pairs under one pixel of disparity: <n> / <total>
- Parallax quantiles: <p10 / p50 / p90>

### Running error
<Does the sequential-pair curve rise faster than linearly? Read against
error_vs_frame_separation before calling it accumulation.>

### Compute follow-ups (decided by the split above, not in advance)
- Duplicated multiview pass: <s> of the total. Worth handing the creator's
  collect dict down to build_report? <yes/no + why>
- Photometric share: <s>. Worth parallelising across pairs? <yes/no + why>
- Anything here large enough to justify streaming to zarr instead of JSON? <yes/no>
```

- [ ] **Step 6: Document the contract**

In `configs/README.md`, beside the existing `colmap/verification.json` entry:

```markdown
- `<backend>/report.json` — reference-free scene error report. One per-pair table
  (keyed on frame index, so epipolar and depth columns join), a per-frame table,
  per-frame percentile ranks, running-error curves along the trajectory, and rank
  correlations for error-vs-depth, error-vs-separation and
  confidence-vs-error. Written by the always-on `report` leaf stage; re-runnable
  with `--stages report --overwrite`.

  The stage runs no model and no matcher. It loads `colmap/verification.json`
  when it exists; in a full pipeline run verify is ordered ahead of `report`, so
  report reads verify's output rather than triggering it. With the shipping
  default (`geometric_verification: false`) no such file is produced, the
  epipolar block records `{"available": false, "reason": ...}`, and the depth and
  photometric channels still emit — `report` never reaches around an explicit
  opt-out to charge a default run for verify. To get the epipolar channel, set
  `pointcloud.geometric_verification: true` or run `--stages verify`. Note that
  `--stages report` on its own, with the flag on and no `verification.json`
  present, *will* run verify first and pay its cost.

  **Report-only: nothing here feeds back into the reconstruction.** No verdict,
  no grade, no cause — distributions and cumulative error only. Every block
  stamps its `grid` (`model` or `original`) and `resolution`; units are
  scale-free or normalised throughout, because 1 recon unit is not 1 metre and
  the factor differs per scene and per backbone. Pixel counts are not comparable
  across backbones, so reprojection is reported in px *and* as a fraction of
  image width.

  Per-pair columns ship as raw values, so any binning or threshold query is
  something the reader does. The one exception is the per-pixel depth residual,
  which is too large to hold (N²·H·W) and ships as `counts` + `bin_edges`. Those
  bins are over `u = r/(1+|r|)`, a monotone map onto (−1, 1): no residual can
  fall outside them however large, so nothing is clipped and nothing is dropped.
  Invert a bin edge or quantile with `u/(1−|u|)`, and query any threshold with
  `rv_histogram((counts, bin_edges)).cdf(x/(1+abs(x)))`.
```

- [ ] **Step 7: Retire the superseded script**

`evals/scripts/depth_disagreement.py` measured the signed residual as a one-off. That residual now lives in the refactored function, and two implementations of one quantity drift apart.

```bash
git rm evals/scripts/depth_disagreement.py
grep -rn "depth_disagreement" --include=*.py --include=*.md --include=*.ipynb . \
  | grep -v '\.git' | grep -v baseck | grep -v '\.worktrees'
```

Expected: only `docs/superpowers/` prose. If code references it, update the reference rather than keeping the file.

- [ ] **Step 8: Full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly 2>&1 | tail -20
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the 5 pre-existing `tests/wrapper/` failures and the pre-existing `test_view_transform_scales_to_target_radius` failure, **and nothing new**. Then `SMOKE PASS`.

- [ ] **Step 9: Commit**

```bash
git add configs/README.md
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(configs): report.json contract + measured scene error report

Records the first end-to-end run: wall clock with a per-measurement split, pair
count and JSON size, the depth-residual sanity target against the prior
depth_disagreement.py numbers, the mapanything-vs-vggt_omega rank control, the
four rank correlations, disparity-floor coverage, and the running-error curve.

The split decides the deferred compute questions — duplicated multiview pass,
photometric parallelism, zarr streaming — with numbers rather than in advance.

Retires evals/scripts/depth_disagreement.py — its signed residual now lives in
compute_multiview_depth_confidence, and two implementations of one quantity
drift apart."
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| Stage wiring (`report` leaf, always on, never fails) | 6 |
| Epipolar + reprojection | 1, 6 |
| Depth cross-view + scale | 3, 4 |
| Photometric | 5, 6 |
| Confidence validation | 6 (`confidence_vs_error.spearman` + its `n_frames`) |
| Resolution contract (per-measurement, grid stamped) | 4, 5, 6 |
| Units (scale-free / normalised) | 4, 5, 6 |
| Signed residual: scale vs noise | 3, 4 |
| Error vs depth ("worse further away?") | 3 (`median_depth`), 4 (`error_vs_depth`) |
| Error vs frame separation (1→4 vs 2→5) | 2, 4, 7 |
| Parallax bridge + disparity floor | 2, 4, 7 |
| Pair table + second-order axes | 4, 6 |
| Distributions + cumulative error | 4, 6 |
| Per-frame ranks, not calls | 6 |
| Exact threshold queries | 4 (histogram counts+edges), 6 (raw columns) |
| Coverage from `original_coords` | 6 |
| `report.json` | 6 |
| Runtime measurement | 1, 8 |
| Negative control per measurement | 7 |
| Sanity target + rank control | 8 |
| `configs/README.md` contract | 8 |
| Retire `depth_disagreement.py` | 8 |

**Gaps accepted and stated, not silently dropped:**
- **Optional GT block** (spec: "Ground truth — an optional block") has no task. Genuinely optional, adds a second input path, and every measurement computes identically without it. The spec's non-forking contract holds because nothing in Tasks 1-8 branches on GT.
- **Spatial pair distance `‖Cᵢ−Cⱼ‖/extent`** is not built. Frame separation is, and it carries the drift axis; the spatial axis needs camera-extent normalisation that only matters once a scene with real revisits is measured. Add it when Task 8 shows revisit pairs exist.
- **Binned views** of error-vs-depth and error-vs-confidence are replaced by one rank correlation each. The raw columns are in `report.json`, so the binned shape is recoverable at any resolution the reader picks — but this plan does not compute it.
- **Parallelism, zarr streaming, and reusing the creator's multiview pass** are deferred to a Task 8 measurement rather than designed in. Stated with the reasoning above, not omitted.

**Type consistency:** `PairStats` is the single per-pair row type, keyed `(idx1, idx2)` across `verification.py`, the mv loop and `compute_depth_error`; every measurement-specific field defaults to `None`. The `collect` dict has exactly three keys, `pairs`, `rel_depth_error_counts` and `rel_depth_error_edges`, written in Task 3 and read unchanged in Task 4 — the edges travel with the counts because they are now scene-dependent, and counts without their edges are unreadable. Frame index means one thing everywhere: position in `sorted(recon.images)` for verify, loop index `i` for the depth pass, and those two coincide by the alignment contract at `verification.py:99/136/144/150`. **Pair ordering is not uniform across the three channels, and each key says which it is.** The depth pass inherits the mv loop's `for i: for j: if i == j: continue`, so it emits `N*(N-1)` **ordered** rows — both `(i,j)` and `(j,i)`, carrying genuinely different values because occlusion is asymmetric (measured on the two-view fixture: `+0.1000` one way, `−0.0909` the other). `verification.py` canonicalises `id1 < id2` and emits one row per **unordered** pair; Task 5's photometric loop runs `for j in range(i+1, ...)` and is likewise unordered. Three consequences, each handled at its own site rather than by forcing the channels to agree: Task 3's histogram is sized `N*(N-1)*H*W`, not the unordered half, because every direction's pixels land in it; Task 4 names every one of its keys for directions — `n_pair_directions`, `pair_directions` and `pair_directions_under_one_pixel_disparity` — while Tasks 5 and 6 use `n_pairs`/`pairs`, so no reader compares the two and sees a phantom 2× (Task 6's `running` block therefore looks the row list up by a per-measurement key name); and `_running_error` groups by unordered key before summing, since `frame_separation` is `abs(idx1 - idx2)` and would otherwise match both directions and double every trajectory step. Task 6 never merges depth and epipolar rows into a single row — they stay in separate `measurements` blocks — so the 2:1 ratio needs no join logic anywhere.

`residual_bin_edges` and `bounded_residual` have exactly one definition, with Task 3 Step 6 guarding the import direction. The sample count has one definition per side and they are deliberately **not** the same expression: production computes the ordered `N*(N-1)*H*W` in Task 3, while `tests/geometry/test_metrics.py`'s `_n_samples(frames, side)` uses the unordered count as a realistic lower bound for exercising bin magnitudes — its comment says so, so the mismatch cannot be read as a bug and "fixed". Correlations are raw `float` (possibly nan) at every site, converted to null once by `clean_for_json` at write time. The critique-5 renames were applied to definitions and uses together and re-grepped: `median_rel_depth_error` / `iqr_rel_depth_error` are read in Task 4's pair rows, in Task 5's ranking, and in the Task 7 tests; `depth_error_px` and `frame_separation` are written in Task 4 and read in Task 5 and the Task 8 readout; `error_vs_frame_separation` and `ncc_vs_frame_separation` are each written at exactly one production site, asserted in Task 4's tests, and quoted in the Task 8 readout template under the same spelling. No old spelling survives anywhere in the plan.

**Placeholder scan:** no TBD/TODO. Three named unknowns with stated resolution paths, not hidden ones: Task 1 Step 3's `Reconstructor` construction (depends on the chosen scene), Task 2 Step 5's `test_verification.py` breakage (expected, with the fix stated), and Task 8 Step 3's JSON size (measured, with the fallback stated). The fourth is now closed: `FrameStore` and `guided_upsample_depth` were read from source rather than memory, which caught a wrong module (`preproc.sampling` → `preproc.frame_store`), a method that does not exist (`read`), and source-video indices being used as row positions.

**Overengineering audit — the complete surface of `metrics.py`:**

| Symbol | Call sites | Kept because |
|---|---|---|
| `residual_bin_edges` | 3, 4, tests | range is fixed by `bounded_residual`, resolution by the sample count — no constant left to declare |
| `bounded_residual` | 2, across 2 files | inlining would split the forward transform from its inverse |
| `depth_error_in_pixels` | 3 + controls | non-obvious math, independently tested, floor derived not declared |
| `compute_depth_error` | 1 | a measurement |
| `compute_photometric_ncc` | 1 | a measurement; both grids |
| `build_report` | 1 | the stage entry point |
| `_load_epipolar` | 1 | file IO plus one derived column |
| `_run_photometric` | 1 | frames.zarr IO and the never-fatal guard |
| `_running_error` | 2 (depth, epipolar) + tests | the ordered/unordered grouping is the one place the two channels' row semantics meet; inline it and the depth curve silently doubles |

Nine functions and **no module-level constants**: the parallax floor derives from the focal, the quantile grid is a local tuple, and the bin edges derive from the sample count. Nothing else exists. No histogram class, no `Report` class, no residual/stats dataclasses, no `MultiviewConfidence` change, and no hand-rolled Spearman, Pearson, rank, quantile, distribution, filename parser, cumulative sum, coverage routine, stratification routine, confidence-binning routine, JSON coercion, or schema stamp. **Measured correction (Task 5 quality pass):** an earlier revision had a tenth entry here, a private one-line wrapper over `stats.spearmanr` that floored small samples to nan. It is gone at all three call sites: `stats.spearmanr(a, b).statistic` is called directly and its answer ships unmodified, because every rho ships beside its own sample count and the raw columns it was computed from, and suppressing a number the reader can already discount is a verdict this report does not make.
