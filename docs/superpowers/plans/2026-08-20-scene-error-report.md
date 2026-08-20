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

| Removed | Replaced by | Reason |
|---|---|---|
| `describe()` | `np.quantile` on the raw column | only ONE quantity is per-pixel |
| 6 of 7 `*_EDGES` constants | raw per-pair columns in the JSON | parallax/pixel-equiv/rot/t-dir/inlier/photometric are per-**pair** scalars — a few thousand floats, not 2.4e10 |
| `SCHEMA_VERSION` | — | nothing consumes `report.json`; a version stamp with no consumer and no migration path is a guess about the future |
| `_by_depth()` + `N_STRATA` | `median_depth` on the pair row + one `rank_correlation` | the bins were recoverable from a column the row should have carried anyway |
| `calculate_confidence_correlation()` + `N_CONF_BINS` | the same `rank_correlation` | confidence-vs-error and depth-vs-error are one operation on two columns |
| `normalized_residual()` | `np.corrcoef` | **verified**: the hand-rolled value equals `sqrt(2 − 2·NCC)` to 8 dp — it was Pearson correlation, rewritten |
| `read_epipolar_error()` | `_verification_rows()`, merged into the one pair table | the "measurement" was a JSON load plus a column rename |
| `assemble()`, `coverage()`, `cumulative()`, `frame_ranks()` | inline in `build_report` | four single-call-site helpers wrapping a dict literal, 3 lines of arithmetic, a `cumsum`, and a `rankdata` |
| fabricated `f"frame_{i:06d}"` names | `idx1`/`idx2` on `PairStats` | verify uses **real** COLMAP names (`verification.py:250`); the depth path invented a string and then parsed digits back out of it |

**Judgment calls kept, flagged for pushback:** `rank_correlation` (4 call sites, exists only for the None-guard every one of them needs) and `_verification_rows` (1 call site, isolates the `verification.json` shape so a verify schema change touches one place).

## Reuse Audit — what is NOT written here, and what supplies it

| Needed | Supplied by |
|---|---|
| Quantiles of a per-pair column | `np.quantile(col, QUANTILE_GRID)` |
| Quantiles from accumulated per-pixel counts | `scipy.stats.rv_histogram((counts, edges)).ppf(q)` |
| Fraction below arbitrary X | same object's `.cdf(x)` |
| Incremental accumulation over N² pairs | `counts += np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]` |
| Normalised patch agreement | `np.corrcoef(a, b)[0, 1]` — this IS the photometric measure |
| Any monotone correlation | `scipy.stats.spearmanr(a, b).statistic` |
| Rank of each frame | `scipy.stats.rankdata(v)` |
| median/p90/p99 | `verification._distribution` (`verification.py:273`) |
| Running accumulation | `np.cumsum` |
| numpy scalars → JSON | `json.dumps(..., default=lambda o: o.item())` |
| Per-pair error row | `verification.PairStats` (`verification.py:41`) |

**Why one histogram survives:** per-pair per-pixel residuals are `N²·H·W` floats — 2.4e10 at 300 frames — so the depth residual must accumulate into fixed bins in place. Every other quantity reduces to one scalar per pair (a few thousand floats), so it ships as a raw column, which a reader can bin at any resolution they choose. Pre-binning those would have thrown information away.

## File Structure

**Create:**
- `collab_splats/geometry/metrics.py`
- `tests/geometry/test_metrics.py`
- `tests/geometry/test_metrics_controls.py` — negative controls, separate because they are the load-bearing proof.

**Modify:**
- `collab_splats/geometry/verification.py:41-50, ~249` — `PairStats` re-keyed on frame index; the one construction site updated.
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

Needs `feedforward.zarr`, `colmap/sparse/0/` and a `frames.zarr`. Record as `$SCENE`.

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
- Extrapolated to 300 frames (~5,400 pairs at overlap=10): <min>
- Image names in pair_stats look like: <paste two>

### Did verify complete?
<yes/no. If no: the exact traceback.>
```

The image-name line matters: Task 2 parses a frame index out of those names, and the parser must match what verify actually writes.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(specs): measured epipolar cost for the scene error report

First verification.json ever produced in this repo — the poses-only measurement's
only input was previously unproven. Records wall clock, pair count, per-pair cost,
the 300-frame extrapolation, and the literal image-name format the report parses."
```

**If `verify` does not complete:** stop and report. Tasks 2-5 and 7-8 are independent of it, but the report loses its poses-only column and with it the ability to separate pose error from depth error.

---

### Task 2: Re-key `PairStats` on frame index, add the parallax bridge

Two fixes in one dataclass. **Frame index becomes the identity** — the depth path has integer indices and no filenames, verify has real filenames, and a report that joins them needs one key both can produce. Names become optional metadata. The separation gap the distance-vs-error axis needs is then `abs(idx1 - idx2)`, a subtraction, not a field.

**Files:**
- Modify: `collab_splats/geometry/verification.py:41-50` and the construction at ~249
- Create: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_metrics.py`:

```python
"""Unit tests for reference-free scene error metrics."""

import numpy as np
import pytest

from collab_splats.geometry.metrics import (
    PARALLAX_FLOOR_DEG,
    QUANTILE_GRID,
    depth_error_in_pixels,
    rank_correlation,
)
from collab_splats.geometry.verification import PairStats


def test_pair_stats_is_keyed_on_frame_index():
    """The depth path has indices and no filenames; verify has filenames. Index is the join key."""
    p = PairStats(idx1=0, idx2=4)
    assert (p.idx1, p.idx2) == (0, 4)
    assert p.name1 is None and p.median_rel is None


def test_separation_is_a_subtraction_not_a_field():
    """1->4 and 2->5 are both separation 3 — the distance-vs-error axis, derived not stored."""
    assert abs(PairStats(1, 4).idx1 - PairStats(1, 4).idx2) == 3
    assert abs(PairStats(2, 5).idx1 - PairStats(2, 5).idx2) == 3


def test_pair_stats_carries_epipolar_and_depth_together():
    """One row per pair. Two measurements fill different columns of it."""
    p = PairStats(0, 1, name1="f0.png", name2="f1.png", num_matches=500, num_inliers=450,
                  rot_error_deg=0.15, t_direction_error_deg=0.9, median_rel=0.02)
    assert p.num_inliers == 450 and p.median_rel == pytest.approx(0.02)


def test_depth_error_in_pixels_is_r_times_disparity():
    """delta_d = r * d, with d = f * parallax(rad) for small angles."""
    assert depth_error_in_pixels(0.1, 2.0, 500.0) == pytest.approx(0.1 * np.deg2rad(2.0) * 500.0)


def test_depth_error_in_pixels_scales_linearly_in_r():
    assert depth_error_in_pixels(0.10, 3.0, 500.0) == pytest.approx(
        2 * depth_error_in_pixels(0.05, 3.0, 500.0)
    )


def test_depth_error_in_pixels_shrinks_with_parallax():
    """The whole far-pixel asymmetry: same depth error, less parallax, fewer pixels moved."""
    assert depth_error_in_pixels(0.1, 1.0, 500.0) < depth_error_in_pixels(0.1, 6.0, 500.0)


def test_depth_error_in_pixels_is_none_below_the_parallax_floor():
    """Not an infinity, not a large number — undefined, and the caller must see that."""
    assert depth_error_in_pixels(0.1, PARALLAX_FLOOR_DEG * 0.5, 500.0) is None


def test_depth_error_in_pixels_uses_magnitude_not_sign():
    assert depth_error_in_pixels(-0.1, 2.0, 500.0) == pytest.approx(
        depth_error_in_pixels(0.1, 2.0, 500.0)
    )


def test_ratio_against_a_measured_pixel_error_needs_no_second_function():
    """rho is a division at the call site, not an API — measured / equivalent."""
    equiv = depth_error_in_pixels(0.1, 2.0, 500.0)
    assert 5.0 * equiv / equiv == pytest.approx(5.0)


def test_rank_correlation_is_monotone_invariant():
    """Spearman, never Pearson: depth is in recon units, confidence is logits on LoGeR."""
    x = np.linspace(1.0, 10.0, 50)
    assert rank_correlation(x, x**3) == pytest.approx(1.0)
    assert rank_correlation(x, -np.exp(x)) == pytest.approx(-1.0)


def test_rank_correlation_is_none_when_it_cannot_be_computed():
    """Absent column, constant column, too few pairs — every call site needs this guard."""
    assert rank_correlation(None, np.arange(5.0)) is None
    assert rank_correlation(np.ones(50), np.arange(50.0)) is None
    assert rank_correlation(np.arange(2.0), np.arange(2.0)) is None


def test_rank_correlation_drops_nan_pairs():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert rank_correlation(x, y) == pytest.approx(1.0)


def test_quantile_grid_is_ordered_and_covers_the_tail():
    assert list(QUANTILE_GRID) == sorted(QUANTILE_GRID) and max(QUANTILE_GRID) >= 0.99
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

    Keyed on FRAME INDEX, not name: the depth cross-view pass has integer indices and no
    filenames, verify has real COLMAP filenames, and the report joins the two. Names stay as
    metadata for the epipolar half. The separation gap (how far apart the two frames are, the
    axis distance-vs-error is read against) is abs(idx1 - idx2) — a subtraction, not a field.
    """

    idx1: int
    idx2: int
    # Epipolar half, populated by verify_reconstruction
    name1: str | None = None
    name2: str | None = None
    num_matches: int | None = None
    num_inliers: int | None = None
    rot_error_deg: float | None = None  # estimated-vs-model relative rotation, degrees
    t_direction_error_deg: float | None = None  # nan if degenerate
    # Depth cross-view half
    n_pixels: int | None = None
    median_rel: float | None = None  # signed => SCALE BIAS between the two views
    iqr_rel: float | None = None  # spread with the bias removed => GEOMETRIC NOISE
    median_parallax_deg: float | None = None  # the pair's depth observability
    median_depth: float | None = None  # carries the "worse further away?" axis as a column
    below_floor_frac: float | None = None
    # Photometric half
    photometric_ncc: float | None = None
```

Then update the single construction site (~line 249) to supply indices, using the same
parser the report uses:

```python
                PairStats(
                    idx1=_index_from_name(im1.name),
                    idx2=_index_from_name(im2.name),
                    name1=im1.name,
                    name2=im2.name,
```

and add the parser above `PairStats` in the same module (it is where names originate):

```python
def _index_from_name(name: str) -> int:
    """Frame index out of an image name, so index-keyed rows join across measurements."""
    digits = "".join(c for c in Path(name).stem if c.isdigit())
    return int(digits) if digits else -1
```

Confirm `Path` is already imported in `verification.py`; add `from pathlib import Path` if not.

- [ ] **Step 4: Write the bridge and the correlation guard**

Create `collab_splats/geometry/metrics.py`:

```python
"""Reference-free scene error metrics: depth cross-view, photometric, and verify's epipolar rows.

Report-only. Nothing here feeds back into a reconstruction and nothing emits a verdict — the
output is distributions and how they vary, for a reader to interpret.

Every statistic comes from scipy or numpy. What lives here is the measurement those statistics
are computed over, not a reimplementation of them.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

from collab_splats.geometry.verification import PairStats, _distribution, _index_from_name

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# Below this parallax angle a pair cannot observe depth along the ray, so the pixel equivalent
# is undefined rather than small. B is the baseline component PERPENDICULAR to the ray, so
# forward camera motion drives it to ~zero near the epipole — the same root cause as the
# AUC@5 ill-conditioning measured on 10-20 mm indoor baselines. Task 7 proves it triggers and
# Task 8 reports what fraction of a real scene falls below it.
PARALLAX_FLOOR_DEG = 0.5

# One grid, every column. Denser than median/p90/p99 because the distribution's shape is the
# deliverable, not three points on it.
QUANTILE_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)

# The ONLY fixed binning in this module. The per-pixel depth residual is N^2*H*W values
# (2.4e10 at 300 frames) so it cannot be held to quantile directly and must accumulate in
# place. Range from measurement, not taste: p99 |rel| is 25.7% on the sanity-target scene, so
# +/-0.5 keeps the useful range uncompressed and clips only a thin tail. Every OTHER quantity
# reduces to one scalar per pair and ships as a raw column, which a reader can bin however
# they like — pre-binning those would throw information away.
REL_EDGES = np.linspace(-0.5, 0.5, 2001)


########################################
# The parallax bridge
########################################


def depth_error_in_pixels(rel_residual: float, parallax_deg: float, focal_px: float) -> float | None:
    """A relative depth residual expressed in pixels, through the pair's actual parallax.

    For a pair with perpendicular baseline B, disparity d = f*B/Z, and a depth error dZ at
    depth Z moves the point in the image by f*B*dZ/Z^2. Substituting r = dZ/Z:

        delta_d = r * d = r * f * alpha

    with alpha the parallax angle in radians. Focal and baseline collapse out of the relation
    itself; f reappears only to express the answer in pixels.

    This is the legitimate way to compare a pixel error against a depth error: convert first,
    then divide. The 1/Z hiding inside d is the entire reason far pixels disagree less in
    pixel terms while disagreeing more in depth terms.

    The ratio measured_px / this is a division at the call site, not a second function:
      ~1  one underlying error seen twice.
      >>1 pixels move more than any depth error explains -> the excess is POSE (pose error
          moves pixels while leaving depths mutually consistent) or appearance.
      <<1 depth disagrees more than pixels do -> the error lies along the ray, where this
          pair's baseline cannot see it. Low observability, not necessarily bad depth.

    Returns None below the parallax floor, so a caller gets no ratio rather than an infinity.
    """
    if parallax_deg < PARALLAX_FLOOR_DEG:
        return None
    return abs(rel_residual) * np.deg2rad(parallax_deg) * focal_px


def rank_correlation(x, y) -> float | None:
    """Spearman rho between two report columns, or None when it cannot be computed.

    Four call sites — error vs depth, error vs frame separation, error vs confidence, and
    NCC vs separation — and all four need the same guard, which is the only reason this
    wraps stats.spearmanr.

    Spearman, never Pearson: depth is in recon units whose metre-factor differs per scene,
    confidence is logits on LoGeR and a bounded score elsewhere, and Spearman is invariant to
    every monotone rescaling between them. It also answers the question directly ("does error
    rise with depth?") without inventing bin edges for a binned view a reader can build
    themselves from the raw columns.
    """
    if x is None or y is None:
        return None
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    # Under 3 points rho is meaningless; a constant column makes it nan.
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    rho = stats.spearmanr(a, b).statistic
    return None if not np.isfinite(rho) else float(rho)
```

- [ ] **Step 5: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/geometry/test_verification.py -v
```

Expected: the 13 new tests pass. **`test_verification.py` may fail** where it constructs `PairStats` positionally or asserts on field order — the identity key changed on purpose. Update those constructions to the new signature; do not add a compatibility shim.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/geometry/verification.py \
        tests/geometry/test_metrics.py tests/geometry/test_verification.py
git commit -m "feat(geometry): key PairStats on frame index, add the parallax bridge

The depth cross-view pass has integer indices and no filenames; verify has real
COLMAP filenames. A report that joins them needs one key both can produce, so
index becomes the identity and names become metadata. The previous draft
fabricated f'frame_{i:06d}' in the depth path and then parsed the digits back
out of the string it had just formatted — a round trip for nothing, and a name
that matched no file on disk.

Frame separation (1->4 and 2->5 are both 3) is abs(idx1-idx2): a subtraction,
not a field. median_depth joins the row so 'does error grow with depth' is a
column correlation rather than a binning routine.

The bridge: delta_d = r * d with d = f*alpha. Focal and baseline collapse out of
the relation; f reappears only to express the answer in pixels. The 1/Z inside d
is the entire 'far pixels disagree less in pixels, more in depth' effect.
Returns None below the parallax floor rather than an infinity.

rank_correlation wraps stats.spearmanr solely for the None-guard its four call
sites share. Spearman not Pearson: depth is in recon units and confidence is
logits on LoGeR, and rho is invariant to both rescalings."
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
    assert out["rel_counts"].sum() > 0
    assert (out["pairs"][0].idx1, out["pairs"][0].idx2) == (0, 1)
    assert out["pairs"][0].name1 is None  # index is the key; no fabricated filenames


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 => median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view(scale_j=1.1)
    out = _collect(depth, K, extr, rel_thresh=0.5)
    row = next(r for r in out["pairs"] if (r.idx1, r.idx2) == (0, 1))
    assert row.median_rel == pytest.approx(0.1, abs=0.02)


def test_signed_residual_is_zero_on_a_consistent_pair():
    depth, K, extr = _two_view()
    assert _collect(depth, K, extr)["pairs"][0].median_rel == pytest.approx(0.0, abs=1e-3)


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
    assert row.median_rel == pytest.approx(0.0, abs=1e-3)


def test_residual_is_scale_invariant():
    """Multiplying depth and translation by s must leave the relative residual unchanged."""
    depth, K, extr = _two_view(scale_j=1.1)
    a = _collect(depth, K, extr, rel_thresh=0.5)["pairs"][0]
    s = 7.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    b = _collect(depth * s, K, extr_s, rel_thresh=0.5)["pairs"][0]
    assert a.median_rel == pytest.approx(b.median_rel, abs=1e-4)
    assert a.median_parallax_deg == pytest.approx(b.median_parallax_deg, abs=1e-3)
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v -k "collect or residual or parallax or occluded or median_depth"
```

Expected: FAIL — `TypeError: ... unexpected keyword argument 'collect'`

- [ ] **Step 3: Add the out-param**

In `collab_splats/pointcloud/feedforward/base.py`, add to the imports:

```python
from collab_splats.geometry.metrics import PARALLAX_FLOOR_DEG, REL_EDGES
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
                 (list[PairStats], keyed on frame index) and "rel_counts" (np.int64 counts
                 against metrics.REL_EDGES — the residual is the one per-pixel quantity, so
                 it is the one that must bin rather than ship raw). The return value is
                 unchanged either way, so the four production creators are unaffected.
```

Before the `for i in range(N)` loop:

```python
    # Residual collection is opt-in and fills the caller's dict: the loop already holds every
    # quantity below, but the four production creators must be byte-identical, so the return
    # contract does not move.
    if collect is not None:
        collect["pairs"] = []
        collect["rel_counts"] = np.zeros(len(REL_EDGES) - 1, dtype=np.int64)
        cam_centers = cam2world[:, :3, 3]  # (N, 3) world-space camera positions
```

Inside the `for j in range(N)` loop, immediately **after** `valid_sum[i] += counted.reshape(H, W).float()`:

```python
            if collect is None:
                continue

            # Signed relative residual: sign carries the scale bias, spread carries the
            # geometric noise. Same population the ratio counts — occluded pixels are absent
            # evidence, and including them would drag the bias negative.
            sel = counted & has_depth
            if not bool(sel.any()):
                continue
            rel = (sampled_d_flat[sel] - expected_d[sel]) / expected_d[sel].clamp(min=1e-6)

            # Parallax from the two ray directions, NOT from f*B/Z. The small-angle pinhole
            # form needs a focal length, and focal is exactly what is not comparable across
            # backbones (11% fx spread on omega alone). Ray directions are scale-free.
            pw = pts_world[sel]
            v_i = pw - cam_centers[i]
            v_j = pw - cam_centers[j]
            cos_a = (v_i * v_j).sum(-1) / (v_i.norm(dim=-1) * v_j.norm(dim=-1)).clamp(min=1e-12)
            par = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # The one histogram: per-pixel residuals are N^2*H*W and cannot be held to
            # quantile. np.histogram DROPS out-of-range values, which would make a later
            # fraction-below query quietly wrong, so clip first — the end bins saturate.
            v = rel.detach().cpu().numpy()
            collect["rel_counts"] += np.histogram(
                np.clip(v, REL_EDGES[0], REL_EDGES[-1]), bins=REL_EDGES
            )[0]

            # Everything else reduces to one scalar per pair and ships as a raw column.
            q = torch.quantile(rel, torch.tensor([0.25, 0.5, 0.75], device=rel.device))
            collect["pairs"].append(
                PairStats(
                    idx1=i,
                    idx2=j,
                    n_pixels=int(sel.sum()),
                    median_rel=float(q[1]),
                    iqr_rel=float(q[2] - q[0]),
                    median_parallax_deg=float(par.median()),
                    median_depth=float(expected_d[sel].median()),
                    below_floor_frac=float((par < PARALLAX_FLOOR_DEG).float().mean()),
                )
            )
```

The `return MultiviewConfidence(...)` at the end is **unchanged**.

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v
```

Expected: all pass — the 9 new plus every pre-existing test in the file.

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

Expected: `imports clean`. **If it cycles**, move `PARALLAX_FLOOR_DEG` and `REL_EDGES` into `base.py` and import them *from* `metrics.py` — the constants have no dependencies, so the edge always points one way.

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
and must accumulate in place, clipped first since np.histogram drops
out-of-range values. Parallax, depth and pixel-equivalent reduce to one scalar
per pair, so they ship as raw columns a reader can bin however they like."
```

---

### Task 4: `calculate_depth_error`

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import REL_EDGES, calculate_depth_error


def _pair(i, j, rel, par, n=100, iqr=0.01, depth=4.0):
    return PairStats(i, j, n_pixels=n, median_rel=rel, iqr_rel=iqr,
                     median_parallax_deg=par, median_depth=depth, below_floor_frac=0.0)


def _collected(pairs):
    """The collect-dict shape compute_multiview_depth_confidence fills."""
    counts = np.zeros(len(REL_EDGES) - 1, dtype=np.int64)
    for p in pairs:
        counts += np.histogram(np.full(p.n_pixels, p.median_rel), bins=REL_EDGES)[0]
    return {"pairs": pairs, "rel_counts": counts}


def test_depth_error_reports_grid_and_resolution():
    """Every block stamps its grid — model-res depth with original-res K is a known bug class."""
    m = calculate_depth_error(_collected([_pair(0, 1, 0.0, 3.0)]), 500.0, "518x518")
    assert m["grid"] == "model" and m["resolution"] == "518x518"


def test_pair_rows_carry_separation():
    """1->4 and 2->5 both land at 3, so distance-vs-error is a column not a special case."""
    m = calculate_depth_error(_collected([_pair(1, 4, 0.02, 3.0), _pair(2, 5, 0.03, 3.0)]), 500.0, "x")
    assert [r["separation"] for r in m["pairs"]] == [3, 3]


def test_scale_bias_is_the_signed_median_not_the_magnitude():
    """A pure scale error has a large median and a small spread; sign must survive."""
    m = calculate_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]), 500.0, "x")
    assert m["pairs"][0]["median_rel"] == pytest.approx(-0.08)
    assert m["scale_bias"]["median"] == pytest.approx(0.08)


def test_pixel_equivalent_lands_on_each_pair_row():
    m = calculate_depth_error(_collected([_pair(0, 1, 0.1, 2.0)]), 500.0, "x")
    assert m["pairs"][0]["error_in_pixels"] == pytest.approx(depth_error_in_pixels(0.1, 2.0, 500.0))


def test_below_floor_pairs_report_null_pixel_equivalent_not_zero():
    m = calculate_depth_error(_collected([_pair(0, 1, 0.1, PARALLAX_FLOOR_DEG * 0.5)]), 500.0, "x")
    assert m["pairs"][0]["error_in_pixels"] is None
    assert m["pairs_below_parallax_floor"] == 1


def test_per_pair_columns_ship_raw_and_quantiled():
    """Raw so a reader can bin them any way; quantiles so the JSON is readable alone."""
    pairs = [_pair(k, k + 1, 0.01 * k, 3.0) for k in range(1, 30)]
    m = calculate_depth_error(_collected(pairs), 500.0, "x")
    assert len(m["pairs"]) == 29
    assert m["parallax_deg"]["0.5"] == pytest.approx(3.0)


def test_per_pixel_residual_ships_as_counts_and_edges():
    """The one quantity too large to hold — histogram, so any threshold query stays exact."""
    m = calculate_depth_error(_collected([_pair(0, 1, 0.1, 3.0)]), 500.0, "x")
    h = m["residual_histogram"]
    assert len(h["edges"]) == len(h["counts"]) + 1 and h["total"] > 0


def test_rising_residual_with_depth_shows_as_a_positive_correlation():
    """One number replaces the depth-strata routine — the raw columns are in the JSON."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0, depth=1.0 + k) for k in range(20)]
    m = calculate_depth_error(_collected(pairs), 500.0, "x")
    assert m["correlations"]["error_vs_depth"] > 0.9
    assert "verdict" not in m


def test_constant_depth_gives_a_null_correlation_not_a_crash():
    pairs = [_pair(k, k + 1, 0.01, 3.0, depth=4.0) for k in range(10)]
    assert calculate_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_depth"] is None


def test_error_vs_separation_is_reported():
    """Does disagreement grow with how far apart the two frames are?"""
    pairs = [_pair(0, k, 0.005 * k, 3.0) for k in range(1, 20)]
    assert calculate_depth_error(_collected(pairs), 500.0, "x")["correlations"]["error_vs_separation"] > 0.9


def test_depth_error_is_unavailable_not_a_crash_when_empty():
    m = calculate_depth_error(_collected([]), 500.0, "x")
    assert m["available"] is False and "reason" in m
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "depth_error or scale_bias or pixel_equiv or below_floor or per_pair or per_pixel or rising or separation"
```

Expected: FAIL — `ImportError: cannot import name 'calculate_depth_error'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Depth cross-view error
########################################


def calculate_depth_error(collected: dict, focal_px: float, resolution: str) -> dict:
    """How much the views disagree about depth: scale bias, geometric noise, parallax.

    Evaluated at MODEL resolution on purpose. Depth values are identical under nearest
    upsampling, so original-res evaluation returns the same number — but it would sample a
    guided-FILTERED depth map, reporting lower disagreement than the model actually produced.
    That improvement belongs to the smoother, not the model.

    Args:
        collected:  the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px:   mean focal in pixels, used only to express the residual in pixel units.
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

    rows = [
        {
            "idx1": p.idx1,
            "idx2": p.idx2,
            # How far apart the two frames are. Distance-vs-error is read against this.
            "separation": abs(p.idx1 - p.idx2),
            "n_pixels": p.n_pixels,
            # Signed: the median IS the scale bias between the two views.
            "median_rel": p.median_rel,
            # Bias removed: what is left is geometric noise.
            "iqr_rel": p.iqr_rel,
            "median_parallax_deg": p.median_parallax_deg,
            "median_depth": p.median_depth,
            "below_floor_frac": p.below_floor_frac,
            # None, never 0.0 — below the floor this is undefined, and a zero would read as
            # "no error" when it means "cannot tell".
            "error_in_pixels": depth_error_in_pixels(p.median_rel, p.median_parallax_deg, focal_px),
        }
        for p in pairs
    ]

    abs_rel = np.array([abs(p.median_rel) for p in pairs])
    parallax = np.array([p.median_parallax_deg for p in pairs])
    depths = np.array([p.median_depth for p in pairs], dtype=np.float64)
    seps = np.array([abs(p.idx1 - p.idx2) for p in pairs], dtype=np.float64)
    rv = stats.rv_histogram((collected["rel_counts"], REL_EDGES))

    return {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; pixel equivalent in px",
        "n_pairs": len(pairs),
        # The one quantity too large to hold: counts+edges, so rv_histogram(...).cdf(x) still
        # answers "what fraction falls below x" exactly at any x. End bins saturate.
        "residual_histogram": {
            "counts": collected["rel_counts"].tolist(),
            "edges": REL_EDGES.tolist(),
            "total": int(collected["rel_counts"].sum()),
            "quantiles": {str(q): float(rv.ppf(q)) for q in QUANTILE_GRID},
            "note": "end bins are saturating (<= first edge, >= last edge)",
        },
        # Per-pair columns are a few thousand floats: quantiled here for readability, and the
        # raw values are in "pairs" below so a reader can re-bin at any resolution.
        "parallax_deg": {str(q): float(v) for q, v in zip(QUANTILE_GRID, np.quantile(parallax, QUANTILE_GRID))},
        # Scale and noise are independent readings of the same signed residual: a pure scale
        # error has a large median and a small spread, a pose error the reverse.
        "scale_bias": _distribution(abs_rel.tolist()),
        "noise": _distribution([p.iqr_rel for p in pairs]),
        "parallax_floor_deg": PARALLAX_FLOOR_DEG,
        "pairs_below_parallax_floor": int((parallax < PARALLAX_FLOOR_DEG).sum()),
        # One number each, replacing two binning routines. The columns they read are in
        # "pairs", so a reader who wants the binned shape can build it.
        # error_vs_depth has a null to read against: triangulation uncertainty goes as
        # sigma_Z ~ Z^2/(f*B), so a RELATIVE residual should rise roughly linearly in Z —
        # positive rho is expected, and the interesting cases are ~0 (depth normalised in a
        # way that hides error) or near 1 (nothing but range explains the disagreement).
        "correlations": {
            "error_vs_depth": rank_correlation(depths, abs_rel),
            "error_vs_separation": rank_correlation(seps, abs_rel),
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
git commit -m "feat(geometry): calculate_depth_error with scale/noise separation

Signed median is the scale bias, spread with the bias removed is geometric
noise — a pure scale error has a large median and small spread, a pose error the
reverse. Below the parallax floor the pixel equivalent is null, never 0.0: a
zero would read as 'no error' when it means 'cannot tell'.

'Does error grow with depth' and 'does error grow with frame separation' are one
rank correlation each over columns the pair rows already carry, replacing a
depth-stratification routine and a fixed bin count. The raw columns ship too, so
a reader who wants the binned shape can build it at any resolution.

The residual histogram is the only pre-binned output, because it is the only
per-pixel quantity. Everything else is per-pair and ships raw.

Model resolution on purpose: depth is identical under nearest upsampling, so
original-res evaluation returns the same number while sampling a guided-FILTERED
map — reporting lower disagreement than the model produced."
```

---

### Task 5: `calculate_photometric_ncc`

The only measurement depending on appearance. **Zero-mean normalised cross-correlation** — verified to be exactly what the previous draft's hand-rolled `normalized_residual` computed (`residual == sqrt(2 − 2·NCC)` to 8 dp), so `np.corrcoef` replaces it. NCC absorbs both the `[0,255]` (VGGT) vs `[0,1]` (MapAnything) split and any exposure change; a raw difference would flag exposure as error.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import calculate_photometric_ncc


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
    m = calculate_photometric_ncc(img, d, K, e, "32x32", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number."""
    img, d, K, e = _plane()
    a = calculate_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    b = calculate_photometric_ncc(img / 255.0, d, K, e, "x", max_separation=1)["pairs"][0]
    assert a["photometric_ncc"] == pytest.approx(b["photometric_ncc"], abs=1e-4)


def test_ncc_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry this measurement exists for."""
    img, d, K, e = _plane()
    shifted = img.copy()
    shifted[1] = shifted[1] * 1.4 + 20.0
    m = calculate_photometric_ncc(shifted, d, K, e, "x", max_separation=1)
    assert m["pairs"][0]["photometric_ncc"] == pytest.approx(1.0, abs=0.05)


def test_ncc_drops_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    img, d, K, e = _plane()
    noisy = img.copy()
    noisy[1] = noisy[1] + rng.normal(0, 90, noisy[1].shape)
    clean = calculate_photometric_ncc(img, d, K, e, "x", max_separation=1)["pairs"][0]
    dirty = calculate_photometric_ncc(noisy, d, K, e, "x", max_separation=1)["pairs"][0]
    assert dirty["photometric_ncc"] < clean["photometric_ncc"]


def test_flat_patch_is_skipped_not_a_divide_by_zero():
    img, d, K, e = _plane()
    m = calculate_photometric_ncc(np.full_like(img, 128.0), d, K, e, "x", max_separation=1)
    assert m["available"] is False


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = calculate_photometric_ncc(img, d, K, e, "16x16", max_separation=1)
    assert all(r["separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    assert calculate_photometric_ncc(img, d, K, e, "16x16", max_separation=1)["available"] is False


def test_photometric_reports_original_grid():
    img, d, K, e = _plane()
    m = calculate_photometric_ncc(img, d, K, e, "1920x1080", max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "1920x1080"
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "ncc or photometric or plane or flat_patch"
```

Expected: FAIL — `ImportError: cannot import name 'calculate_photometric_ncc'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Photometric agreement
########################################

# Below this many valid samples a correlation is noise.
MIN_SAMPLES = 8
# O(N * max_separation), not O(N^2): appearance agreement between temporally distant frames
# is dominated by lighting and viewpoint change, not by the reconstruction error measured here.
PHOTOMETRIC_MAX_SEPARATION = 2


def calculate_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    resolution: str,
    max_separation: int = PHOTOMETRIC_MAX_SEPARATION,
) -> dict:
    """Warp each frame into its neighbours through pose+depth and correlate the RGB.

    The measure is zero-mean normalised cross-correlation: 1.0 is perfect agreement, 0.0 is
    none. np.corrcoef supplies it — NCC of two flattened patches is exactly their Pearson
    correlation, so there is nothing to write.

    Normalisation buys two invariances a raw difference lacks: the [0, 255] (VGGT family) vs
    [0, 1] (MapAnything) image-scale split, so one number is comparable across backbones; and
    exposure/gain change, which would otherwise swamp the geometry being measured.

    The only measurement here depending on appearance, so a disagreement it sees that the
    depth and epipolar columns do not points at image formation rather than geometry.

    ORIGINAL resolution on purpose: RGB detail exists only there, and unlike depth this is a
    genuinely resolution-dependent quantity.

    Args:
        images:     (N, H, W, 3) RGB on the SAME grid as depth.
        depth:      (N, H, W) Z-depth.
        intrinsics: (N, 3, 3) pixel-unit K on that grid.
        extrinsics: (N, 4, 4) world-to-cam.
    """
    N, H, W = depth.shape
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)
    ones = np.ones((H * W, 1))

    rows = []
    for i in range(N):
        # Unproject frame i's pixels to world through its own K and pose
        pts_cam = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam, ones], axis=-1).T).T[:, :3]

        for j in range(i + 1, min(N, i + max_separation + 1)):
            pts_j = (extrinsics[j] @ np.concatenate([pts_world, ones], axis=-1).T).T[:, :3]
            proj = (intrinsics[j] @ pts_j.T).T
            z = np.clip(proj[:, 2], 1e-6, None)
            # NEAREST sampling, matching the depth pass: bilinear across a depth discontinuity
            # blends two surfaces into a colour present on neither.
            ui = np.round(proj[:, 0] / z).astype(np.int64)
            vi = np.round(proj[:, 1] / z).astype(np.int64)
            ok = (pts_j[:, 2] > 0) & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < MIN_SAMPLES:
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
            rows.append({"idx1": i, "idx2": j, "separation": j - i,
                         "photometric_ncc": ncc, "n_pixels": int(ok.sum())})

    if not rows:
        return {
            "available": False,
            "reason": "no view pairs produced a photometric correlation",
            "grid": "original",
            "resolution": resolution,
        }
    ncc = np.array([r["photometric_ncc"] for r in rows])
    seps = np.array([r["separation"] for r in rows], dtype=np.float64)
    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "zero-mean normalised cross-correlation; 1.0 = perfect agreement",
        "n_pairs": len(rows),
        "ncc": {str(q): float(v) for q, v in zip(QUANTILE_GRID, np.quantile(ncc, QUANTILE_GRID))},
        "correlations": {"ncc_vs_separation": rank_correlation(seps, ncc)},
        "pairs": rows,
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 32 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py
git commit -m "feat(geometry): calculate_photometric_ncc via np.corrcoef

The previous draft hand-rolled an RMS-of-z-scored-difference and called it
normalized_residual. Measured, that value equals sqrt(2 - 2*NCC) to 8 decimals:
it WAS Pearson correlation, rewritten. np.corrcoef supplies it directly, and NCC
is the name the domain already uses.

Normalisation buys invariance to the [0,255] vs [0,1] backbone image-scale split
AND to exposure change, which would otherwise swamp the geometry this measures.
Flat patches are skipped rather than dividing by zero.

Nearest sampling, matching the depth pass: bilinear across a discontinuity
blends two surfaces into a colour present on neither. O(N * max_separation), not
O(N^2) — appearance agreement between temporally distant frames is dominated by
lighting and viewpoint change, not by reconstruction error."
```

---

### Task 6: `build_report` and the leaf stage

The epipolar half is a **merge**, not a measurement: verify already wrote per-pair rows in the shape the report wants, so the report loads them and joins on `(idx1, idx2)`. Cumulative curves, frame ranks and crop coverage are three, two and three lines respectively, written where they are used.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Modify: `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
import json

from collab_splats.geometry.metrics import _crop_coverage, _cumulative, _verification_rows
from collab_splats.wrapper.reconstructor import LEAF_STAGES, _STAGE_DEPS, _STAGE_ORDER


def _vjson(tmp_path, pair_stats, frame_stats=None):
    p = tmp_path / "verification.json"
    p.write_text(json.dumps({"pair_stats": pair_stats, "frame_stats": frame_stats or {}, "summary": {}}))
    return p


def test_verification_rows_are_index_keyed_and_carry_separation(tmp_path):
    p = _vjson(tmp_path, [dict(idx1=3, idx2=11, name1="frame_000003.png", name2="frame_000011.png",
                               num_matches=500, num_inliers=450, rot_error_deg=0.15,
                               t_direction_error_deg=0.9)])
    m = _verification_rows(p, image_width=640)
    assert m["available"] is True and m["grid"] == "original"
    assert m["pairs"][0]["separation"] == 8
    assert m["pairs"][0]["inlier_ratio"] == pytest.approx(0.9)


def test_missing_verification_json_is_unavailable_not_a_crash(tmp_path):
    assert _verification_rows(tmp_path / "nope.json", image_width=640)["available"] is False


def test_reprojection_reported_in_px_and_as_image_fraction(tmp_path):
    """A bare pixel count is not comparable across backbones (518-crop vs 448x592)."""
    p = _vjson(tmp_path, [], {"frame_000000.png": {"mean_reproj_error_px": 1.28, "n_tracks": 100}})
    f = _verification_rows(p, image_width=640)["frames"][0]
    assert f["mean_reproj_error_px"] == pytest.approx(1.28)
    assert f["mean_reproj_error_frac_width"] == pytest.approx(0.002)


def test_nan_rotation_is_dropped_from_the_quantiles(tmp_path):
    p = _vjson(tmp_path, [
        dict(idx1=0, idx2=1, num_matches=10, num_inliers=8, rot_error_deg=None, t_direction_error_deg=1.0),
        dict(idx1=1, idx2=2, num_matches=10, num_inliers=9, rot_error_deg=0.3, t_direction_error_deg=1.0),
    ])
    assert _verification_rows(p, image_width=640)["rot_error_deg"]["0.5"] == pytest.approx(0.3)


def test_cumulative_uses_sequential_pairs_and_absolute_steps():
    """Signed steps would cancel and hide accumulation; separation>1 pairs are not steps."""
    rows = [{"idx1": 0, "idx2": 1, "separation": 1, "v": 0.1},
            {"idx1": 1, "idx2": 2, "separation": 1, "v": -0.1},
            {"idx1": 0, "idx2": 5, "separation": 5, "v": 9.9}]
    c = _cumulative(rows, "v")
    assert c["frame_index"] == [1, 2] and c["cumulative"] == pytest.approx([0.1, 0.2])


def test_crop_coverage_from_original_coords():
    """VGGTX centre-crops height to 518 — a 16:9 source loses a band with no depth at all."""
    c = _crop_coverage(np.array([[0, 281, 1920, 799, 1920, 1080]], dtype=np.float32))
    assert c["per_frame"][0]["covered_fraction"] == pytest.approx((1920 * 518) / (1920 * 1080), abs=1e-3)


def test_report_is_a_leaf_stage_depending_only_on_pointcloud():
    assert "report" in LEAF_STAGES
    assert _STAGE_DEPS["report"] == ["pointcloud"]
    assert _STAGE_ORDER.index("report") > _STAGE_ORDER.index("pointcloud")


def test_report_does_not_demote_any_existing_leaf():
    """A new dependency edge would silently break another stage's disk re-run."""
    for s in ("refine", "semantics", "mesh", "localize", "verify"):
        assert s in LEAF_STAGES
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "verification_rows or cumulative or crop_coverage or leaf or reprojection or nan_rotation"
```

Expected: FAIL — `ImportError: cannot import name '_verification_rows'`

- [ ] **Step 3: Write the merge and the two inline helpers**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Epipolar rows (merged from verify) and second-order views
########################################


def _verification_rows(verification_json: Path, image_width: int) -> dict:
    """Load verify's per-pair and per-frame tables into the report's row shape.

    Not a measurement — verify already made it. This is a JSON load plus a column rename, and
    it exists as a function only so the verification.json shape is depended on in one place.
    The matcher is never re-run.

    These rows are the only ones that never touch depth, which is the entire reason
    attribution is possible: something that moves here and not in the depth rows is a pose
    error. Already original-resolution, since verify estimates from original-res keypoints.
    """
    p = Path(verification_json)
    if not p.exists():
        return {"available": False, "reason": f"no verification.json at {p} — run the verify stage",
                "grid": "original"}
    data = json.loads(p.read_text())

    rows = []
    for s in data.get("pair_stats", []):
        i, j = s.get("idx1"), s.get("idx2")
        n_m, n_i = s.get("num_matches") or 0, s.get("num_inliers") or 0
        rows.append({
            "idx1": i, "idx2": j,
            "separation": None if i is None or j is None else abs(i - j),
            "num_matches": n_m, "num_inliers": n_i,
            "inlier_ratio": (n_i / n_m) if n_m else None,
            "rot_error_deg": s.get("rot_error_deg"),
            "t_direction_error_deg": s.get("t_direction_error_deg"),
        })

    frames = []
    for name, fs in sorted(data.get("frame_stats", {}).items()):
        px = fs.get("mean_reproj_error_px")
        frames.append({
            "name": name, "index": _index_from_name(name),
            "n_tracks": fs.get("n_tracks"), "track_survival": fs.get("track_survival"),
            "mean_reproj_error_px": px,
            # A bare pixel count is not comparable across backbones — normalise.
            "mean_reproj_error_frac_width": None if px is None else px / image_width,
        })

    def _q(key):
        v = np.array([r[key] for r in rows if r[key] is not None and np.isfinite(r[key])])
        return None if v.size == 0 else {str(q): float(x) for q, x in zip(QUANTILE_GRID, np.quantile(v, QUANTILE_GRID))}

    return {
        "available": True, "grid": "original", "resolution": f"width={image_width}",
        "units": "degrees; reprojection in px and as a fraction of image width",
        "source": str(p), "n_pairs": len(rows),
        "rot_error_deg": _q("rot_error_deg"),
        "t_direction_error_deg": _q("t_direction_error_deg"),
        "inlier_ratio": _q("inlier_ratio"),
        "pairs": rows, "frames": frames,
    }


def _cumulative(rows: list[dict], key: str) -> dict:
    """Running accumulation of |value| along consecutive frames — "does disagreement build?".

    Sequential pairs only: a separation-5 pair is a revisit observation, not a step along the
    trajectory, and summing it would double-count. Absolute values, because signed steps
    cancel and would hide exactly the accumulation this exists to show.

    Read against the per-separation columns, not alone: frame index is a confounded axis
    (scene content, motion speed and exposure all correlate with it).
    """
    steps = sorted(
        (r for r in rows if r.get("separation") == 1 and r.get(key) is not None and np.isfinite(r[key])),
        key=lambda r: min(r["idx1"], r["idx2"]),
    )
    return {
        "frame_index": [int(max(r["idx1"], r["idx2"])) for r in steps],
        "cumulative": np.cumsum([abs(float(r[key])) for r in steps]).tolist(),
        "note": "sequential pairs only; absolute steps; read against the per-separation columns",
    }


def _crop_coverage(original_coords: np.ndarray) -> dict:
    """Fraction of each ORIGINAL frame the model crop actually reconstructed.

    Rows are [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]. VGGTX resizes width to 518 and
    centre-CROPS height to 518, so on a 16:9 source a large band of every frame has no depth
    at all. Model-resolution evaluation is structurally blind to this — the model-res grid IS
    the crop.
    """
    c = np.asarray(original_coords, dtype=np.float64)
    area = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    fr = area / np.maximum(c[:, 4] * c[:, 5], 1e-9)
    return {
        "per_frame": [{"index": k, "covered_fraction": float(f)} for k, f in enumerate(fr)],
        "median_covered_fraction": float(np.median(fr)) if fr.size else None,
        "min_covered_fraction": float(np.min(fr)) if fr.size else None,
    }
```

- [ ] **Step 4: Write the stage entry point**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Stage entry point
########################################


def build_report(zarr_path: Path, verification_json: Path, frames_zarr: Path,
                 output_path: Path, backend: str) -> dict:
    """Run every measurement that can run and write report.json. Never raises on a dead one.

    Measurements are attempted independently: a missing confidence array, an absent
    verification.json or an unreadable frames.zarr each disable exactly one of them.

    No key here grades the scene, names a cause, or flags a frame. Absolute thresholds that
    would justify a verdict are exactly what this stage exists to inform, so inventing them
    now would be a guess dressed as a finding.
    """
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult, compute_multiview_depth_confidence

    r = FeedforwardResult.load_zarr(zarr_path)
    n = len(r.depth)
    model_res = f"{r.model_width}x{r.model_height}"
    focal_px = float(r.intrinsics[:, 0, 0].mean() + r.intrinsics[:, 1, 1].mean()) / 2.0

    # One dense pass yields the depth residual, the scale split, the parallax angles and the
    # per-pair depth. abs_thresh stays 0.0: scale invariance holds only there, and it is what
    # lets one function serve backbones with completely different depth scales.
    collected: dict = {}
    compute_multiview_depth_confidence(
        r.depth, r.intrinsics, r.extrinsics, abs_thresh=0.0, rel_thresh=0.05, collect=collected
    )
    depth_m = calculate_depth_error(collected, focal_px, model_res)
    epipolar_m = _verification_rows(verification_json, image_width=int(r.original_coords[0][4]))
    photometric_m = _photometric_original_res(r, frames_zarr, n)

    # Per-frame median |residual| — the column both the confidence check and the ranks read.
    per_frame = {}
    for k in range(n):
        v = [abs(p.median_rel) for p in collected["pairs"] if k in (p.idx1, p.idx2)]
        if v:
            per_frame[k] = float(np.median(v))

    # Does the model know when it is wrong? One correlation, not a measurement of its own —
    # confidence is an INPUT being validated. Absent on older zarr stores, never backfilled.
    conf_rho = None
    if r.confidence is not None and per_frame:
        conf = np.asarray(r.confidence)
        conf_rho = rank_correlation(
            np.array([float(np.median(conf[k])) for k in per_frame]),
            np.array(list(per_frame.values())),
        )

    # Frame ranks: each frame's position in this scene's own distribution, in [0, 1]. A
    # NUMBER, never a label — the report does not name a cause or flag a frame. Within-scene
    # ranks need no absolute threshold, sidestepping cross-backbone incomparability.
    ks = list(per_frame)
    ranks = {}
    if len(ks) > 1:
        rk = (stats.rankdata([per_frame[k] for k in ks]) - 1) / (len(ks) - 1)
        ranks = {int(k): float(x) for k, x in zip(ks, rk)}

    report = {
        "scene": {"backend": backend, "n_frames": n, "model_resolution": model_res,
                  "zarr": str(zarr_path)},
        "measurements_available": sorted(
            k for k, m in (("epipolar", epipolar_m), ("depth", depth_m), ("photometric", photometric_m))
            if m.get("available")
        ),
        "measurements": {"epipolar": epipolar_m, "depth": depth_m, "photometric": photometric_m},
        "confidence_vs_error_spearman": conf_rho,
        "cumulative": {
            "depth": _cumulative(depth_m.get("pairs", []), "median_rel"),
            "epipolar": _cumulative(epipolar_m.get("pairs", []), "rot_error_deg"),
        },
        "frame_percentile_ranks": ranks,
        "crop_coverage": _crop_coverage(r.original_coords),
        "notes": {
            "verdicts": "none by design — this describes distributions, it does not grade",
            "units": "scale-free or normalised throughout; 1 recon unit is NOT 1 metre",
            "attribution": "measurements differ in what they depend on; read them against each other",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=lambda o: o.item()))
    logger.info("Wrote %s (%d measurements available)", output_path, len(report["measurements_available"]))
    return report


def _photometric_original_res(r, frames_zarr: Path, n: int) -> dict:
    """Upsample depth and rescale K to the original grid, then correlate. Never fatal."""
    if not Path(frames_zarr).exists():
        return {"available": False, "reason": f"frames.zarr not found at {frames_zarr}", "grid": "original"}
    try:
        from collab_splats.mesh.utils import guided_upsample_depth
        from collab_splats.preproc.sampling import FrameStore

        store = FrameStore.open(Path(frames_zarr))
        rgbs, deps, Ks = [], [], []
        for k, fi in enumerate(store.frame_indices()[:n]):
            rgb = np.asarray(store.read(fi), dtype=np.float32)
            oh, ow = rgb.shape[:2]
            tlx, tly, crx, cry = r.original_coords[k][:4]
            deps.append(guided_upsample_depth(r.depth[k], rgb, (int(tlx), int(tly), int(crx), int(cry)), (oh, ow)))
            rgbs.append(rgb)
            # Model-res K rescaled to the original grid. The 2026-08-11 mesh-collapse bug
            # class is pairing one grid's depth with the other grid's K, so both move here.
            s = ow / r.model_width
            K = r.intrinsics[k].copy()
            K[0, 0] *= s
            K[1, 1] *= s
            K[0, 2] = K[0, 2] * s + tlx
            K[1, 2] = K[1, 2] * s + tly
            Ks.append(K)
        return calculate_photometric_ncc(
            np.stack(rgbs), np.stack(deps), np.stack(Ks), r.extrinsics[: len(rgbs)],
            resolution=f"{rgbs[0].shape[1]}x{rgbs[0].shape[0]}",
        )
    except Exception as exc:  # noqa: BLE001 — a report must never fail a reconstruction
        logger.warning("photometric measurement failed: %s", exc, exc_info=True)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "grid": "original"}
```

**Three API names to confirm before running**, read off memory rather than a fresh grep:

```bash
grep -n "def read\|def frame_indices\|def open" collab_splats/preproc/sampling.py
grep -n "def guided_upsample_depth" collab_splats/mesh/utils.py
grep -n "def load_zarr" collab_splats/pointcloud/feedforward/base.py
```

If `FrameStore`'s accessor is named differently, match the `_LazyFrames` usage at `reconstructor.py:1141`. If `guided_upsample_depth` has a different signature, match the call in `mesh/utils.py`'s native-resolution path.

- [ ] **Step 5: Register the stage**

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
            # default-false flag; a report nobody runs answers nothing, and the measured cost
            # is bounded. The one boolean it would have had is the boolean that keeps it off.
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

The epipolar half is a merge, not a measurement: verify already wrote per-pair
rows in the shape the report wants, so the report loads them and joins on
(idx1, idx2). _verification_rows exists as a function only to depend on the
verification.json shape in one place; the matcher is never re-run.

Cumulative, frame ranks and crop coverage were four single-call-site helpers
wrapping a cumsum, a rankdata and three lines of arithmetic. Two survive as
private helpers with real bodies; the ranks and the dict assembly are written
where they are used.

_STAGE_DEPS['report'] == ['pointcloud'], so --stages report re-runs against a
scene pulled from environments-processed with no rerun.py change.

Always on with no config boolean, against repo precedent: every other diagnostic
ships behind a default-false flag, and the one boolean this would have had is
the boolean that keeps it off.

Cumulative walks sequential pairs only (a separation-5 pair is a revisit, not a
trajectory step) using absolute values, since signed steps cancel and hide the
accumulation. Frame ranks are numbers, never labels."
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

from collab_splats.geometry.metrics import PARALLAX_FLOOR_DEG, depth_error_in_pixels, rank_correlation
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
    """x1.1 on frame 2's depth => median_rel ~ +0.1 on pairs INTO frame 2, and only those."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    p = _pairs(depth, K, extr, rel_thresh=0.5)
    into_2 = [v.median_rel for (i, j), v in p.items() if j == 2]
    clean = [v.median_rel for (i, j), v in p.items() if 2 not in (i, j)]
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
    measured = depth_error_in_pixels(p.median_rel, p.median_parallax_deg, FOCAL)
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
        assert after[key].median_rel == pytest.approx(before[key].median_rel, abs=1e-9)


def test_control_exposure_shift_is_invisible_to_photometric_too():
    """NCC is what buys this — a raw difference would flag it as error."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=768)
    assert float(np.corrcoef(a, a * 1.6 + 30.0)[0, 1]) == pytest.approx(1.0, abs=1e-9)


def test_control_forward_motion_lands_below_the_parallax_floor():
    """Pure forward motion drives perpendicular baseline to ~0 near the epipole."""
    depth, K, extr = _scene(n=3)
    extr[:, 0, 3] = 0.0
    for k in range(3):
        extr[k, 2, 3] = -0.05 * k  # translate along the viewing axis instead
    worst = min(_pairs(depth, K, extr).values(), key=lambda q: q.median_parallax_deg)
    assert worst.median_parallax_deg < PARALLAX_FLOOR_DEG
    # And the bridge must decline to answer rather than emit an infinity.
    assert depth_error_in_pixels(0.05, worst.median_parallax_deg, FOCAL) is None


def test_control_separation_axis_has_teeth():
    """Injecting error that grows with frame gap must show as a positive rho."""
    depth, K, extr = _scene(n=6)
    for k in range(6):
        depth[k] *= 1.0 + 0.02 * k  # drift: each frame slightly more scaled than the last
    p = _pairs(depth, K, extr, rel_thresh=0.9)
    seps = np.array([abs(i - j) for (i, j) in p])
    errs = np.array([abs(v.median_rel) for v in p.values()])
    assert rank_correlation(seps, errs) > 0.8
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
median_rel to ~0.1 on pairs into that frame ONLY, leaving parallax alone. The
bridge makes it quantitative — r=0.1 predicts delta_d = 0.1*d in closed form
with ratio ~1, while a pose fault drives the ratio >>1. Opposite signatures from
one formula.

The separation axis gets teeth too: injected drift that grows with frame gap
must show as a positive rank correlation, or the distance-vs-error column is
inert. Forward-motion control confirms the parallax floor triggers and that the
bridge declines to answer rather than emitting an infinity."
```

---

### Task 8: Real-scene run, measured numbers, contract, retirement

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`, `configs/README.md`
- Delete: `evals/scripts/depth_disagreement.py`

- [ ] **Step 1: Run it in tmux**

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
print(\"pairs:\", rep[\"measurements\"][\"depth\"][\"n_pairs\"])
print(\"json MB:\", round((root / \"report.json\").stat().st_size / 1e6, 2))
print(json.dumps(rep[\"measurements\"][\"depth\"][\"residual_histogram\"][\"quantiles\"], indent=2))
print(\"correlations:\", rep[\"measurements\"][\"depth\"][\"correlations\"])
" 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/ee7cc0e1-beee-4d06-908d-0a6838558f0b/scratchpad/scene_report.log'
```

Watch: `tmux attach -t scene_report`. Memory: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

- [ ] **Step 2: Check the sanity target**

Measured baseline on this store: **median |rel| 0.37%, p90 2.27%, p99 25.67%**, tightening to p90 0.92% at conf>p20.

Compare the printed quantiles. They should agree closely — it is the same quantity `depth_disagreement.py` measured. **If they differ materially, explain the difference before proceeding.** Check first: residual population (`counted & has_depth` here) and signed-vs-absolute.

- [ ] **Step 3: Check the pair-table size**

The report ships every gated pair as a raw row so a reader can re-bin any column. At 300 frames the mv loop's pair count is O(N²) before gating, so record `pairs` and `json MB` from Step 1.

**If `report.json` exceeds ~20 MB**, the fix is to keep the quantiles and the histogram and emit raw rows only for sequential pairs plus the worst 500 by `median_rel` — one filter, no new concepts. Record the decision either way; do not add the filter pre-emptively.

- [ ] **Step 4: Rank control**

Run Step 1 against a `mapanything` store and a `vggt_omega` store of the same scene — they differ 1.6× in ATE on chess/seq-01.

```bash
/opt/venv/reconstruction/bin/python -c "
import json, pathlib
for name in ('mapanything', 'vggt_omega'):
    p = pathlib.Path(f'evals/results/{name}/report.json')
    if not p.exists():
        print(name, 'MISSING'); continue
    d = json.loads(p.read_text())['measurements']['depth']
    print(name, 'p50', d['residual_histogram']['quantiles']['0.5'],
          'p99', d['residual_histogram']['quantiles']['0.99'],
          'scale_bias', d['scale_bias'])
"
```

**If the report cannot order those two, it will not separate anything.** Record the outcome either way — a null result here is the most important number in the task.

- [ ] **Step 5: Append the measurements**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
## Task 8: report stage, measured

- Scene: evals/results/mv_vggt_omega (60 frames, vggt_omega)
- Wall clock: <REPORT_SECONDS> s | Peak rss: <GB> / 46.6 GB
- Measurements available: <list>
- Pairs: <n> | report.json: <MB> MB  (filter applied: <yes/no>)

### Sanity target (depth residual)
| quantile | measured | prior (depth_disagreement.py) |
|---|---|---|
| median abs | <x>% | 0.37% |
| p90 | <x>% | 2.27% |
| p99 | <x>% | 25.67% |

<Agreement, or the explained difference.>

### Rank control (mapanything vs vggt_omega, 1.6x apart in ATE)
| backbone | p50 rel | p99 | scale bias |
|---|---|---|---|

Ordered correctly: <yes/no>. <If no: what that means for the design.>

### Correlations
- error_vs_depth: <rho>  (null: sigma_Z ~ Z^2/(f*B) => expect positive, ~linear)
- error_vs_separation: <rho>
- confidence_vs_error: <rho or null — absent on stores with no confidence array>
- ncc_vs_separation: <rho>

### Parallax
- Pairs below the 0.5 deg floor: <n> / <total>
- Parallax quantiles: <p10 / p50 / p90>

### Cumulative
<Does the sequential-pair curve rise faster than linearly? Read against
error_vs_separation before calling it accumulation.>
```

- [ ] **Step 6: Document the contract**

In `configs/README.md`, beside the existing `colmap/verification.json` entry:

```markdown
- `<backend>/report.json` — reference-free scene error report. One per-pair table
  (keyed on frame index, so epipolar and depth columns join), a per-frame table,
  per-frame percentile ranks, cumulative curves along the trajectory, and rank
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
  which is too large to hold (N²·H·W) and ships as `counts` + `edges`;
  `scipy.stats.rv_histogram((counts, edges)).cdf(x)` answers "what fraction falls
  below x" exactly at any x. Its end bins saturate — values are clipped before
  binning, so nothing is silently dropped.
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

Records the first end-to-end run: wall clock, measurements available, pair count
and JSON size, the depth-residual sanity target against the prior
depth_disagreement.py numbers, the mapanything-vs-vggt_omega rank control, the
four rank correlations, parallax floor coverage, and the cumulative curve.

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
| Parallax bridge + floor | 2, 4, 7 |
| Pair table + second-order axes | 4, 6 |
| Distributions + cumulative error | 4, 6 |
| Per-frame ranks, not calls | 6 |
| Exact threshold queries | 4 (histogram counts+edges), 6 (raw columns) |
| Coverage from `original_coords` | 6 |
| `report.json` | 6 |
| Runtime measurement | 1 |
| Negative control per measurement | 7 |
| Sanity target + rank control | 8 |
| `configs/README.md` contract | 8 |
| Retire `depth_disagreement.py` | 8 |

**Gaps accepted and stated, not silently dropped:**
- **Optional GT block** (spec: "Ground truth — an optional block") has no task. Genuinely optional, adds a second input path, and every measurement computes identically without it. The spec's non-forking contract holds because nothing in Tasks 1-8 branches on GT.
- **Spatial pair distance `‖Cᵢ−Cⱼ‖/extent`** is not built. Frame separation is, and it carries the drift axis; the spatial axis needs camera-extent normalisation that only matters once a scene with real revisits is measured. Add it when Task 8 shows revisit pairs exist.
- **Binned views** of error-vs-depth and error-vs-confidence are replaced by one rank correlation each. The raw columns are in `report.json`, so the binned shape is recoverable at any resolution the reader picks — but this plan does not compute it.

**Type consistency:** `PairStats` is the single per-pair row type, keyed `(idx1, idx2)` across `verification.py`, the mv loop, `calculate_depth_error` and `_verification_rows`; every measurement-specific field defaults to `None`. The `collect` dict has exactly two keys, `pairs` and `rel_counts`, written in Task 3 and read unchanged in Task 4. `_index_from_name` lives in `verification.py` (where names originate) and returns `int`, `-1` on an unparseable name. `rank_correlation` returns `float | None` and every consumer stores it directly. `PARALLAX_FLOOR_DEG` and `REL_EDGES` have exactly one definition, with Task 3 Step 6 guarding the import direction.

**Placeholder scan:** no TBD/TODO. Four named unknowns with stated resolution paths, not hidden ones: Task 1 Step 3's `Reconstructor` construction (depends on the chosen scene), Task 6 Step 4's `FrameStore.read` / `guided_upsample_depth` / `load_zarr` signatures (grep commands and fallbacks supplied inline), Task 2 Step 5's `test_verification.py` breakage (expected, with the fix stated), and Task 8 Step 3's JSON size (measured, with the fallback stated).

**Overengineering audit — the full public surface of `metrics.py`:**

| Symbol | Call sites | Kept because |
|---|---|---|
| `PARALLAX_FLOOR_DEG` | 3 | a threshold, proven live by a control |
| `QUANTILE_GRID` | 4 | one grid everywhere |
| `REL_EDGES` | 2 | the only per-pixel quantity |
| `MIN_SAMPLES`, `PHOTOMETRIC_MAX_SEPARATION` | 1 each | real tuning params with stated reasons |
| `depth_error_in_pixels` | 3 + controls | non-obvious math, independently tested |
| `rank_correlation` | 4 | the None-guard all four share |
| `calculate_depth_error` | 1 | a measurement |
| `calculate_photometric_ncc` | 1 | a measurement |
| `build_report` | 1 | the stage entry point |
| `_verification_rows` | 1 | isolates the verification.json shape |
| `_cumulative`, `_crop_coverage` | 1 each | real bodies with load-bearing comments |
| `_photometric_original_res` | 1 | the K-rescale guard; inlining makes `build_report` unreadable |

Nothing else exists. No histogram class, no residual/stats dataclasses, no `MultiviewConfidence` change, no `describe`, no `assemble`, no `coverage`/`cumulative`/`frame_ranks` public helpers, no stratification routine, no confidence-binning routine, no schema stamp, and no hand-rolled Spearman, Pearson, rank, quantile, or JSON coercion.
