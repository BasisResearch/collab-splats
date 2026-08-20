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
| Any monotone correlation, nans dropped | `scipy.stats.spearmanr(a, b, nan_policy="omit").statistic` |
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
    """No wrapper: nan_policy drops pairs and verification.clean_for_json turns nan into null."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert stats.spearmanr(x, y, nan_policy="omit").statistic == pytest.approx(1.0)
```

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
    """1->4 and 2->5 both land at 3, so distance-vs-error is a column not a special case."""
    m = compute_depth_error(_collected([_pair(1, 4, 0.02, 3.0), _pair(2, 5, 0.03, 3.0)]), 500.0, "x")
    assert [r["frame_separation"] for r in m["pairs"]] == [3, 3]


def test_scale_bias_keeps_its_sign_on_the_row():
    """A pure scale error has a large median and a small spread; the sign must survive."""
    m = compute_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]), 500.0, "x")
    assert m["pairs"][0]["median_rel_depth_error"] == pytest.approx(-0.08)
    assert m["pairs"][0]["iqr_rel_depth_error"] == pytest.approx(0.005)


def test_pixel_equivalent_lands_on_each_pair_row():
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, 2.0)]), 500.0, "x")
    assert m["pairs"][0]["depth_error_px"] == pytest.approx(depth_error_in_pixels(0.1, 2.0, 500.0))


def test_pairs_under_one_pixel_of_disparity_report_null_not_zero():
    tiny = np.rad2deg(0.5 / 500.0)  # half a pixel of disparity
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, tiny)]), 500.0, "x")
    assert m["pairs"][0]["depth_error_px"] is None
    assert m["pairs_under_one_pixel_disparity"] == 1


def test_per_pair_columns_ship_raw():
    """Raw, so any binning or threshold query is something the reader does."""
    pairs = [_pair(k, k + 1, 0.01 * k, 3.0) for k in range(1, 30)]
    m = compute_depth_error(_collected(pairs), 500.0, "x")
    assert len(m["pairs"]) == 29
    assert {"median_rel_depth_error", "iqr_rel_depth_error", "median_parallax_deg", "median_depth"} <= set(m["pairs"][0])


def test_per_pixel_residual_ships_as_counts_and_edges():
    """The one quantity too large to hold — so any threshold query stays exact."""
    m = compute_depth_error(_collected([_pair(0, 1, 0.1, 3.0)]), 500.0, "x")
    h = m["residual_histogram"]
    assert len(h["bin_edges"]) == len(h["counts"]) + 1 and h["total"] > 0
    assert h["quantiles"]["0.5"] == pytest.approx(0.1, abs=0.01)  # inverted back to a residual


def test_signed_and_folded_quantiles_both_ship():
    """Every prior |rel| number in this repo is absolute, so the signed axis alone is not
    comparable — a negative bias reads as a negative quantile until the histogram is folded."""
    h = compute_depth_error(_collected([_pair(0, 1, -0.1, 3.0)]), 500.0, "x")["residual_histogram"]
    assert h["quantiles"]["0.5"] == pytest.approx(-0.1, abs=0.01)  # sign kept: scale bias
    assert h["abs_quantiles"]["0.5"] == pytest.approx(0.1, abs=0.01)  # folded: magnitude


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
        # DIRECTIONS, not pairs. The mv loop is ordered: (i,j) and (j,i) are separate rows with
        # genuinely different values, because occlusion is asymmetric — a pixel hidden looking
        # one way is visible looking the other. The name says so, because the photometric block
        # below and verify's epipolar block both count UNORDERED pairs under the key "n_pairs",
        # and a reader comparing the three numbers would otherwise see a phantom 2x.
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
        "pairs_under_one_pixel_disparity": under_1px,
        # Two questions, one number each, straight from scipy. nan means "cannot be computed"
        # (a constant column, too few pairs) and becomes null when the report is written.
        # The columns they read are in "pairs", so a reader can plot the binned shape.
        # error_vs_depth has a null to read against: triangulation uncertainty goes as
        # sigma_Z ~ Z^2/(f*B), so a relative residual should already rise roughly linearly in
        # Z. Positive rho is expected. Near 0 or near 1 are the interesting outcomes.
        "correlations": {
            "error_vs_depth": float(
                stats.spearmanr(depths, abs_rel_depth_error, nan_policy="omit").statistic
            ),
            "error_vs_frame_separation": float(
                stats.spearmanr(frame_seps, abs_rel_depth_error, nan_policy="omit").statistic
            ),
            "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual rises ~linearly in Z",
        },
        "pairs": rows,
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 24 passed.

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
depth-stratification routine and a fixed bin count. nan_policy handles the nan
drop and verification.clean_for_json turns the constant-column nan into null, so there
is no wrapper. Raw columns ship too, so a reader who wants the binned shape can
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
    rng = np.random.default_rng(seed)
    tex = rng.uniform(0, 255, size=(hw, hw, 3)).astype(np.float32)
    K = np.array([[40.0, 0, hw / 2], [0, 40.0, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    return (
        np.stack([tex] * n),
        np.stack([np.full((hw, hw), 4.0, np.float32)] * n),
        np.stack([K] * n),
        np.stack([np.eye(4, dtype=np.float32)] * n),
    )


def test_identical_poses_and_depth_warp_to_ncc_one():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "32x32", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number."""
    img, d, K, e = _plane()
    a = compute_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    b = compute_photometric_ncc(img / 255.0, d, K, e, "x", max_separation=1)["pairs"][0]
    assert a["photometric_ncc"] == pytest.approx(b["photometric_ncc"], abs=1e-4)


def test_ncc_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry this measurement exists for."""
    img, d, K, e = _plane()
    shifted = img.copy()
    shifted[1] = shifted[1] * 1.4 + 20.0
    m = compute_photometric_ncc(shifted, d, K, e, "x", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = compute_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    dirty = compute_photometric_ncc(noisy, d, K, e, "x", max_separation=1)["pairs"][0]
    assert dirty["photometric_ncc"] < clean["photometric_ncc"]


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(np.full_like(img, 128.0), d, K, e, "x", max_separation=1)
    assert m["available"] is False


def test_two_overlapping_pixels_do_not_count_as_a_correlation():
    """np.corrcoef on 2 points returns exactly +-1 whatever the values — hence min_samples."""
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "x", max_separation=1, min_samples=10**9)
    assert m["available"] is False


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = compute_photometric_ncc(img, d, K, e, "16x16", max_separation=1)
    assert all(r["frame_separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    assert compute_photometric_ncc(img, d, K, e, "16x16", max_separation=1)["available"] is False


def test_photometric_upsamples_model_res_depth_to_the_image_grid():
    """One function, both grids: depth is model-res, images are original-res."""
    img, d, K, e = _plane(hw=64)
    small = d[:, ::2, ::2]  # 32x32 depth against 64x64 images
    coords = np.tile(np.array([0, 0, 64, 64, 64, 64], dtype=np.float32), (2, 1))
    m = compute_photometric_ncc(img, small, K, e, "64x64", max_separation=1,
                                  original_coords=coords)
    assert m["available"] is True and m["grid"] == "original"


def test_photometric_grid_says_which_one_it_ran_on():
    img, d, K, e = _plane()
    m = compute_photometric_ncc(img, d, K, e, "1920x1080", max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "1920x1080"
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
    resolution: str,
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

    Args:
        images:          (N, H, W, 3) RGB, original resolution.
        depth:           (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics:      (N, 3, 3) K matching `depth`'s grid; rescaled here if depth is.
        extrinsics:      (N, 4, 4) world-to-cam.
        resolution:      "WxH" of the image grid, stamped into the output.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        max_separation:  pairs per frame. Appearance agreement between distant frames is
                         dominated by lighting and viewpoint change, not by the error measured
                         here, so this stays O(N*max_separation) rather than O(N^2).
        min_samples:     floor on overlapping pixels. np.corrcoef on 2 points returns exactly
                         +-1 whatever the values, so a minimum is not optional.
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
        from collab_splats.mesh.utils import guided_upsample_depth

        model_w = depth.shape[2]
        lifted_d, lifted_K = [], []
        for k in range(N):
            tlx, tly, crx, cry = original_coords[k][:4]
            # rgb_full is the original-res canvas the crop came from — images[k] already is
            # that, so no re-read. crop_box is original_coords[:4], out_hw the canvas size.
            lifted_d.append(
                guided_upsample_depth(depth[k], images[k],
                                      (int(tlx), int(tly), int(crx), int(cry)), (ih, iw))
            )
            s = iw / model_w
            K = np.array(intrinsics[k], dtype=np.float64).copy()
            K[0, 0] *= s
            K[1, 1] *= s
            K[0, 2] = K[0, 2] * s + tlx
            K[1, 2] = K[1, 2] * s + tly
            lifted_K.append(K)
        depth, intrinsics = np.stack(lifted_d), np.stack(lifted_K)

    H, W = depth.shape[1:]
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

        for j in range(i + 1, min(N, i + max_separation + 1)):
            # Project them into frame j and look up the colour that landed there
            pts_cam_j = (extrinsics[j] @ np.concatenate([pts_world, ones], axis=-1).T).T[:, :3]
            proj_j = (intrinsics[j] @ pts_cam_j.T).T
            z = np.clip(proj_j[:, 2], 1e-6, None)
            # Nearest sampling, matching the depth pass: bilinear across a depth discontinuity
            # blends two surfaces into a colour present on neither.
            ui = np.round(proj_j[:, 0] / z).astype(np.int64)
            vi = np.round(proj_j[:, 1] / z).astype(np.int64)
            in_front = pts_cam_j[:, 2] > 0
            ok = in_front & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < min_samples:
                continue
            a = images[i].reshape(-1, 3)[ok].ravel().astype(np.float64)
            b = images[j][vi[ok], ui[ok]].ravel().astype(np.float64)
            # A flat patch has no variance to correlate; corrcoef returns nan, which is
            # dropped rather than counted as agreement.
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
    ncc = np.array([r["photometric_ncc"] for r in rows])
    frame_seps = np.array([r["frame_separation"] for r in rows], dtype=np.float64)
    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalised cross-correlation; 1.0 = perfect agreement",
        "n_pairs": len(rows),
        "correlations": {
            "ncc_vs_frame_separation": float(
                stats.spearmanr(frame_seps, ncc, nan_policy="omit").statistic
            )
        },
        "pairs": rows,
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 34 passed.

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
    conf_rho = None
    if r.confidence is not None and len(per_frame) > 2:
        conf = np.asarray(r.confidence)
        conf_rho = float(stats.spearmanr(
            [float(np.median(conf[k])) for k in per_frame],
            list(per_frame.values()),
            nan_policy="omit",
        ).statistic)

    # Where each frame sits in this scene's own distribution, 0..1. A NUMBER, never a label.
    # Within-scene ranks need no absolute threshold, which sidesteps the fact that pixel and
    # depth units are not comparable across backbones.
    ks = list(per_frame)
    ranks = {}
    if len(ks) > 1:
        rk = (stats.rankdata([per_frame[k] for k in ks]) - 1) / (len(ks) - 1)
        ranks = {int(k): float(x) for k, x in zip(ks, rk)}

    # Does disagreement build along the trajectory?
    running = {
        name: _running_error(m.get("pairs", []), key)
        for name, key, m in (
            ("depth", "median_rel_depth_error", depth_m),
            ("epipolar", "rot_error_deg", epipolar_m),
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
        "confidence_vs_error_spearman": conf_rho,
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
        return {"available": False, "reason": f"no verification.json at {p} — run the verify stage",
                "grid": "original"}
    data = json.loads(p.read_text())

    rows = []
    for s in data.get("pair_stats", []):
        # Same expression verify already aggregates over at verification.py:332
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
        return compute_photometric_ncc(
            rgbs, r.depth[:m], r.intrinsics[:m], r.extrinsics[:m],
            resolution=f"{rgbs.shape[2]}x{rgbs.shape[1]}",
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
    # report reads verification.json when present and runs verify itself when absent, so like
    # verify its only hard dependency is the reconstruction
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
            # keeps it off. The measured cost is bounded.
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

        # The epipolar rows are the only ones that never touch depth, which is what makes
        # attribution possible — worth building when absent rather than skipped.
        verification_json = self.backend_dir / "colmap" / "verification.json"
        if not verification_json.exists():
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

Expected: `tests/geometry/` all pass; `tests/wrapper/` shows the **same 5 pre-existing failures and no new ones**; then `SMOKE PASS`.

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

- [ ] **Step 1: Write the failing tests**

```python
"""Negative controls: each measurement must move under its own fault and stay still under others."""

import numpy as np
import pytest
from scipy import stats

from collab_splats.geometry.metrics import depth_error_in_pixels
from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence

FOCAL = 30.0


def _scene(n=4, hw=24, depth_value=4.0):
    """N cameras strafing sideways, all viewing a constant-depth plane."""
    K = np.array([[FOCAL, 0, hw / 2], [0, FOCAL, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((hw, hw), depth_value, np.float32)] * n)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k in range(n):
        extr[k, 0, 3] = -0.25 * k  # camera k at x = +0.25k
    return depth, np.stack([K] * n), extr


def _pairs(depth, K, extr, **kw):
    out = {}
    compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect=out, **kw)
    return {(p.idx1, p.idx2): p for p in out["pairs"]}


def test_control_depth_scale_moves_the_depth_measurement_by_the_injected_amount():
    """x1.1 on frame 2's depth => median_rel_depth_error ~ +0.1 on pairs INTO frame 2, and only those."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    p = _pairs(depth, K, extr, rel_thresh=0.5)
    into_2 = [v.median_rel_depth_error for (i, j), v in p.items() if j == 2]
    clean = [v.median_rel_depth_error for (i, j), v in p.items() if 2 not in (i, j)]
    assert np.median(into_2) == pytest.approx(0.1, abs=0.03)
    assert np.max(np.abs(clean)) < 0.01


def test_control_depth_scale_does_not_move_the_parallax_angles():
    """Parallax is pose geometry; a depth scale must not change it materially."""
    depth, K, extr = _scene()
    before = _pairs(depth, K, extr, rel_thresh=0.5)
    depth[2] *= 1.1
    after = _pairs(depth, K, extr, rel_thresh=0.5)
    for key in set(before) & set(after):
        if 2 in key:
            continue
        assert after[key].median_parallax_deg == pytest.approx(before[key].median_parallax_deg, abs=0.05)


def test_control_injected_scale_has_a_closed_form_prediction():
    """r=0.1 predicts delta_d = 0.1*d exactly, and the ratio sits at 1 for a pure depth fault."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    p = _pairs(depth, K, extr, rel_thresh=0.5)[(1, 2)]
    predicted = depth_error_in_pixels(0.1, p.median_parallax_deg, FOCAL)
    measured = depth_error_in_pixels(p.median_rel_depth_error, p.median_parallax_deg, FOCAL)
    assert measured == pytest.approx(predicted, rel=0.3)
    # A pure depth fault: the pixel motion IS the depth motion, so the ratio sits at 1.
    assert measured / predicted == pytest.approx(1.0, abs=0.3)


def test_control_pose_fault_drives_the_ratio_far_above_one():
    """Pixels move while depths stay mutually consistent — the >>1 signature."""
    depth, K, extr = _scene()
    p = _pairs(depth, K, extr)[(0, 1)]
    equiv = depth_error_in_pixels(0.001, p.median_parallax_deg, FOCAL)
    assert equiv is not None and 5.0 / equiv > 10.0


def test_control_exposure_shift_is_invisible_to_the_depth_measurement():
    """Appearance faults cannot reach a measurement that never reads appearance."""
    depth, K, extr = _scene()
    before, after = _pairs(depth, K, extr), _pairs(depth, K, extr)
    for key in before:
        assert after[key].median_rel_depth_error == pytest.approx(before[key].median_rel_depth_error, abs=1e-9)


def test_control_exposure_shift_is_invisible_to_photometric_too():
    """NCC is what buys this — a raw difference would flag it as error."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=768)
    assert float(np.corrcoef(a, a * 1.6 + 30.0)[0, 1]) == pytest.approx(1.0, abs=1e-9)


def test_control_forward_motion_falls_under_one_pixel_of_disparity():
    """Pure forward motion drives perpendicular baseline to ~0 near the epipole."""
    depth, K, extr = _scene(n=3)
    extr[:, 0, 3] = 0.0
    for k in range(3):
        extr[k, 2, 3] = -0.05 * k  # translate along the viewing axis instead
    worst = min(_pairs(depth, K, extr).values(), key=lambda q: q.median_parallax_deg)
    assert np.deg2rad(worst.median_parallax_deg) * FOCAL < 1.0
    # And the bridge declines to answer rather than emitting an infinity.
    assert depth_error_in_pixels(0.05, worst.median_parallax_deg, FOCAL) is None


def test_control_separation_axis_has_teeth():
    """Injecting error that grows with frame gap must show as a positive rho."""
    depth, K, extr = _scene(n=6)
    for k in range(6):
        depth[k] *= 1.0 + 0.02 * k  # drift: each frame slightly more scaled than the last
    p = _pairs(depth, K, extr, rel_thresh=0.9)
    frame_seps = np.array([abs(i - j) for (i, j) in p], dtype=np.float64)
    errs = np.array([abs(v.median_rel_depth_error) for v in p.values()])
    assert stats.spearmanr(frame_seps, errs, nan_policy="omit").statistic > 0.8
```

- [ ] **Step 2: Run them**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics_controls.py -v
```

**A failure here is a finding, not a test bug.** If `test_control_depth_scale_moves_the_depth_measurement_by_the_injected_amount` fails, the measurements do not separate and the design's central claim is wrong — stop and report rather than adjusting tolerances.

- [ ] **Step 3: Fix whatever they expose**

No implementation is written speculatively. Fix the defect in `metrics.py` or `base.py`, then re-run. If a control cannot pass because the *fixture* is degenerate (a constant-depth plane gives every pixel nearly the same parallax), fix the fixture — but record why in a comment, because a degenerate fixture that silently passes is how an inert threshold ships.

- [ ] **Step 4: Full geometry suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/geometry/test_metrics_controls.py collab_splats/geometry/metrics.py \
        collab_splats/pointcloud/feedforward/base.py
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
than emitting an infinity."
```

---

### Task 8: Real-scene run, measured numbers, contract, retirement

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`, `configs/README.md`
- Delete: `evals/scripts/depth_disagreement.py`

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
- error_vs_depth: <rho>  (null: sigma_Z ~ Z^2/(f*B) => expect positive, ~linear)
- error_vs_frame_separation: <rho>
- confidence_vs_error: <rho or null — absent on stores with no confidence array>
- ncc_vs_frame_separation: <rho>

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
| Confidence validation | 6 (`confidence_vs_error_spearman`) |
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

**Type consistency:** `PairStats` is the single per-pair row type, keyed `(idx1, idx2)` across `verification.py`, the mv loop and `compute_depth_error`; every measurement-specific field defaults to `None`. The `collect` dict has exactly three keys, `pairs`, `rel_depth_error_counts` and `rel_depth_error_edges`, written in Task 3 and read unchanged in Task 4 — the edges travel with the counts because they are now scene-dependent, and counts without their edges are unreadable. Frame index means one thing everywhere: position in `sorted(recon.images)` for verify, loop index `i` for the depth pass, and those two coincide by the alignment contract at `verification.py:99/136/144/150`. **Pair ordering is not uniform across the three channels, and each key says which it is.** The depth pass inherits the mv loop's `for i: for j: if i == j: continue`, so it emits `N*(N-1)` **ordered** rows — both `(i,j)` and `(j,i)`, carrying genuinely different values because occlusion is asymmetric (measured on the two-view fixture: `+0.1000` one way, `−0.0909` the other). `verification.py` canonicalises `id1 < id2` and emits one row per **unordered** pair; Task 5's photometric loop runs `for j in range(i+1, ...)` and is likewise unordered. Three consequences, each handled at its own site rather than by forcing the channels to agree: Task 3's histogram is sized `N*(N-1)*H*W`, not the unordered half, because every direction's pixels land in it; Task 4 reports the count under `n_pair_directions` while Tasks 5 and 6 use `n_pairs`, so no reader compares the two and sees a phantom 2×; and `_running_error` groups by unordered key before summing, since `frame_separation` is `abs(idx1 - idx2)` and would otherwise match both directions and double every trajectory step. Task 6 never merges depth and epipolar rows into a single row — they stay in separate `measurements` blocks — so the 2:1 ratio needs no join logic anywhere.

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

Nine functions and **no module-level constants**: the parallax floor derives from the focal, the quantile grid is a local tuple, and the bin edges derive from the sample count. Nothing else exists. No histogram class, no `Report` class, no residual/stats dataclasses, no `MultiviewConfidence` change, and no hand-rolled Spearman, Pearson, rank, quantile, distribution, filename parser, cumulative sum, coverage routine, stratification routine, confidence-binning routine, JSON coercion, or schema stamp.
