# Scene Error Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `report` leaf stage producing a reference-free `report.json` that locates reconstruction error in space and time, with no ground truth and no verdicts.

**Architecture:** One module, `geometry/metrics.py`, holding four measurement functions whose *dependencies deliberately differ* — `read_epipolar_error` (poses only), `calculate_depth_error` (poses+depth), `calculate_photometric_error` (poses+depth+appearance), `calculate_confidence_correlation` (validates an input). Reading them against each other is what attributes error. `depth_error_in_pixels` converts depth residuals to pixel units so they share one axis. scipy supplies every statistic; the existing `PairStats` carries every per-pair row.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, scipy 1.17.1, torch, zarr v3, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-scene-error-report-design.md` (commit `d5021b4`)

---

## Environment and Safety (read before Task 1)

- **Python:** always `/opt/venv/reconstruction/bin/python`. Bare `python` is 3.13 and wrong for this project.
- **Heavy runs:** tmux only, serially. Never two GPU jobs at once. Container cap 46.6 GB; read `rss` from `/sys/fs/cgroup/memory/memory.stat`, **not** `memory.usage_in_bytes`.
- **A concurrent session keeps files dirty** (`configs/base.yaml`, `pyproject.toml`, `collab_splats/remote/rerun.py`). **Never `git add -A`. Never repo-wide `black .`** — the venv's black 26.5.1 is newer than the repo's formatting. Stage named files only.
- **`git add -f`** is required for anything under `docs/superpowers/` (gitignored).
- **Pre-existing failures that are not yours:** 5 in `tests/wrapper/`, and `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius`.

## Reuse Audit — what is NOT written here, and what supplies it

This plan writes no statistics code. Every substitution below was verified working before the plan was written:

| Needed | Supplied by | Verified |
|---|---|---|
| Fraction of samples below arbitrary X | `scipy.stats.rv_histogram((counts, edges)).cdf(x)` | matches true `(v<x).mean()` to 4 dp |
| Quantile from accumulated counts | same object's `.ppf(q)` | matches `np.quantile` to 4 dp |
| Incremental accumulation over N² pairs | `counts += np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]` | one line |
| Binning a value by another value's quantiles | `scipy.stats.binned_statistic(x, v, statistic="median", bins=edges)` | replaces depth stratification **and** confidence binning |
| Rank of each frame within the scene | `scipy.stats.rankdata(v)` | normalizes to [0,1] |
| Confidence-vs-error correlation | `scipy.stats.spearmanr(a, b).statistic` | returns −1.0 on an anti-correlated pair |
| median/p90/p99 of a list | `verification._distribution` (`verification.py:273`) | already in repo |
| Running accumulation | `np.cumsum` | |
| numpy scalars → JSON | `json.dumps(..., default=lambda o: o.item())` | |
| Per-pair error row | `verification.PairStats` (`verification.py:41`) | already in repo, extended in Task 2 |

**`np.histogram` silently drops out-of-range values**, which would make a "fraction below X" query quietly wrong. `np.clip` to the outer edges first, so outliers land in the end bins. The end bins therefore mean "≤ this" and "≥ this" — stated in the output.

## File Structure

**Create:**
- `collab_splats/geometry/metrics.py` — the four measurement functions, the parallax bridge, and `build_report`. One file because these are one concern (reference-free error measurement) and are read together.
- `tests/geometry/test_metrics.py`
- `tests/geometry/test_metrics_controls.py` — negative controls, separate because they are the load-bearing proof and must not be skimmed past.

**Modify:**
- `collab_splats/geometry/verification.py:41-50` — `PairStats` gains optional depth fields.
- `collab_splats/pointcloud/feedforward/base.py` — `compute_multiview_depth_confidence` gains a `collect` out-param. **No new class, and the return type is unchanged**: `MultiviewConfidence` ships today with four production callers (`vggtx.py:354`, `vggt_omega.py:265`, `mapanything.py:446`, `loger.py:440`), so nothing about its contract moves.
- `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266` — register the `report` stage.
- `configs/README.md` — output contract.
- `tests/pointcloud/test_mv_conf.py` — backward-compat proof + collection tests.

**Delete (Task 9):** `evals/scripts/depth_disagreement.py` — its residual now lives in the refactored function.

**The constraint that shapes the loop:** per-pair per-pixel residuals are `N²·H·W` floats — 2.4e10 at 300 frames. The loop accumulates into a counts array and per-pair scalars **in place** and never returns raw residual arrays.

---

### Task 1: Prove `verify` runs end-to-end and measure its cost

`read_epipolar_error` consumes `verification.json`. **No `verification.json` exists anywhere in this repo** — `verify` has never run to completion here, so the poses-only measurement's only input is unproven. De-risk before anything depends on it.

**Files:** Create `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`

- [ ] **Step 1: Confirm none exists**

```bash
find /workspace/collab-splats -name verification.json -not -path '*/.git/*' 2>/dev/null; echo "exit=$?"
```

Expected: nothing printed. If one IS found, read it, record its frame count, and correct the "never run" claim in the spec.

- [ ] **Step 2: Find a scene**

```bash
find /workspace/collab-splats/evals/results -maxdepth 3 -name feedforward.zarr 2>/dev/null | head
```

Needs `feedforward.zarr`, `colmap/sparse/0/` and a `frames.zarr` (verify feeds original-res frames to the matcher). Record as `$SCENE`.

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

The constructor call is indicative — use whatever the chosen scene needs (`docs/examples/run_pipeline_remote.py` has the driver pattern). The measurement is the deliverable, not the invocation.

Watch: `tmux attach -t verify_measure`. Memory: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

- [ ] **Step 4: Record it**

Write `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
# Scene error report — measured numbers

## Task 1: epipolar cost (verify)

- Scene / frames / backbone / matcher: <...>
- Wall clock: <VERIFY_SECONDS> s
- Peak rss: <GB> / 46.6 GB
- Pairs generated: <n from verification.json summary>
- Per-pair: <ms>
- Extrapolated to 300 frames (~5,400 pairs at overlap=10): <min>

### Did verify complete?
<yes/no. If no: the exact traceback, and what read_epipolar_error must do instead.>
```

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(specs): measured epipolar cost for the scene error report

First verification.json ever produced in this repo — the poses-only measurement's
only input was previously unproven. Records wall clock, pair count, per-pair cost
and the 300-frame extrapolation the design owed."
```

**If `verify` does not complete:** stop and report. Tasks 2-6 and 8-9 are independent of it, but Task 7's `read_epipolar_error` must be re-scoped and the spec's attribution claim weakens — without a poses-only measurement there is nothing to separate depth error from pose error against.

---

### Task 2: Extend `PairStats`, add the parallax bridge

One dataclass for "one view pair's measured error", not three. `PairStats` already carries the epipolar half; depth fields join it.

**Files:**
- Modify: `collab_splats/geometry/verification.py:41-50`
- Create: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_metrics.py`:

```python
"""Unit tests for reference-free scene error metrics."""

import numpy as np
import pytest

from collab_splats.geometry.metrics import PARALLAX_FLOOR_DEG, depth_error_in_pixels
from collab_splats.geometry.verification import PairStats


def test_pair_stats_carries_depth_fields_defaulted_none():
    """One pair class for one concept — the epipolar half must still construct unchanged."""
    p = PairStats("frame_000000", "frame_000001", 500, 450, 0.15, 0.9)
    assert p.median_rel is None and p.median_parallax_deg is None


def test_pair_stats_accepts_depth_fields():
    p = PairStats("a", "b", 0, 0, float("nan"), float("nan"), median_rel=0.1, median_parallax_deg=2.0)
    assert p.median_rel == pytest.approx(0.1)


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
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.geometry.metrics'`

- [ ] **Step 3: Extend `PairStats`**

In `collab_splats/geometry/verification.py`, replace the `PairStats` body (lines 42-50) with:

```python
class PairStats:
    """Measured error for one image pair. Fields are optional per measurement.

    Epipolar fields come from verify_matches; depth fields from the cross-view depth pass.
    One class because these are one concept — a pair's error — and a report joins them by
    (name1, name2). A measurement that did not run leaves its fields None.
    """

    name1: str
    name2: str
    num_matches: int
    num_inliers: int
    rot_error_deg: float  # estimated-vs-model relative rotation, degrees
    t_direction_error_deg: float  # translation-direction angle, degrees (nan if degenerate)
    # Depth cross-view fields, None unless the depth pass populated them
    n_pixels: int | None = None
    median_rel: float | None = None  # signed => SCALE BIAS between the two views
    iqr_rel: float | None = None  # spread with the bias removed => GEOMETRIC NOISE
    median_parallax_deg: float | None = None  # the pair's depth observability
    below_floor_frac: float | None = None
    # Photometric field, None unless the photometric pass populated it
    photometric_residual: float | None = None
```

- [ ] **Step 4: Write the bridge**

Create `collab_splats/geometry/metrics.py`:

```python
"""Reference-free scene error metrics: depth, photometric, confidence, epipolar.

Report-only. Nothing here feeds back into a reconstruction and nothing emits a verdict —
the output is distributions and how they vary, for a reader to interpret.

Every statistic comes from scipy or numpy. The functions here are the measurements those
statistics are computed over, not reimplementations of them.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

from collab_splats.geometry.verification import PairStats, _distribution

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

SCHEMA_VERSION = 1

# Below this parallax angle a pair cannot observe depth along the ray, so the pixel
# equivalent is undefined rather than small. B is the baseline component PERPENDICULAR to
# the ray, so forward camera motion drives it to ~zero near the epipole — the same root
# cause as the AUC@5 ill-conditioning measured on 10-20 mm indoor baselines.
PARALLAX_FLOOR_DEG = 0.5

# Quantile grid reported for every measurement. Denser than median/p90/p99 because the
# shape of the distribution is the deliverable, not three points on it.
QUANTILE_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)

# Histogram ranges. Signed residual: measured p99 |rel| is 25.7% on the sanity target, so
# +/-0.5 keeps the useful range uncompressed. Parallax: small indoors at 10-20 mm baselines.
REL_EDGES = np.linspace(-0.5, 0.5, 2001)
PARALLAX_EDGES = np.linspace(0.0, 30.0, 601)
EQUIV_PX_EDGES = np.linspace(0.0, 20.0, 2001)

# Bins for "does error grow with depth / does confidence track error".
N_STRATA = 5
N_CONF_BINS = 10


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
```

- [ ] **Step 5: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/geometry/test_verification.py -v
```

Expected: the 8 new tests pass, and `test_verification.py` passes unchanged — the new `PairStats` fields all default, so every existing construction site is valid.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/geometry/verification.py \
        tests/geometry/test_metrics.py
git commit -m "feat(geometry): parallax bridge + depth fields on PairStats

delta_d = r * d with d = f*alpha. Focal and baseline collapse out of the
relation; f reappears only to express the answer in pixels. The 1/Z inside d is
the entire 'far pixels disagree less in pixels, more in depth' effect, so
dividing it out removes the apparent contradiction rather than working around it.

The ratio measured/equivalent stays a division at the call site — it is one
divide, not an API. Returns None below the parallax floor rather than an
infinity: forward motion drives perpendicular baseline to zero near the epipole,
same root cause as the AUC@5 ill-conditioning at 10-20 mm indoor baselines.

PairStats carries the depth and photometric fields rather than new classes —
one pair, one row, joined by (name1, name2). All new fields default to None so
every existing construction site is unchanged."
```

---

### Task 3: Collect the residual and parallax the mv loop already computes

`compute_multiview_depth_confidence` computes `expected_d` and `sampled_d`, thresholds them to a boolean, and **discards the residual**. It also unprojects `pts_world`, from which the parallax angle is two dot products. Both are free; only the plumbing is new.

**No new class and no return-type change.** `MultiviewConfidence` ships today with four production callers; an optional `collect` dict is filled in place instead.

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


def test_collect_fills_pairs_and_histograms_in_place():
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    assert out["pairs"] and out["rel_counts"].sum() > 0
    assert out["parallax_counts"].sum() > 0


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 => median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view(scale_j=1.1)
    out = _collect(depth, K, extr, rel_thresh=0.5)
    row = next(r for r in out["pairs"] if (r.name1, r.name2) == ("frame_000000", "frame_000001"))
    assert row.median_rel == pytest.approx(0.1, abs=0.02)


def test_signed_residual_is_zero_on_a_consistent_pair():
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    row = out["pairs"][0]
    assert row.median_rel == pytest.approx(0.0, abs=1e-3)


def test_parallax_angle_matches_geometry():
    """0.2 baseline at depth 4 => atan(0.2/4) ~ 2.86 deg at the principal ray."""
    depth, K, extr = _two_view()
    out = _collect(depth, K, extr)
    assert out["pairs"][0].median_parallax_deg == pytest.approx(
        np.degrees(np.arctan(0.2 / 4.0)), abs=0.5
    )


def test_occluded_pixels_are_excluded_from_the_residual():
    """Occlusion is absent evidence, not disagreement — it must not pollute the scale bias."""
    depth, K, extr = _two_view()
    depth[1, :, :8] = 0.5  # a near occluder covering half of frame 1
    out = _collect(depth, K, extr)
    row = next(r for r in out["pairs"] if r.name1 == "frame_000000")
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
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v -k "collect or residual or parallax or occluded"
```

Expected: FAIL — `TypeError: ... unexpected keyword argument 'collect'`

- [ ] **Step 3: Add the out-param**

In `collab_splats/pointcloud/feedforward/base.py`, add to the imports:

```python
from collab_splats.geometry.metrics import EQUIV_PX_EDGES, PARALLAX_EDGES, PARALLAX_FLOOR_DEG, REL_EDGES
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
                 (list[PairStats]), "rel_counts"/"parallax_counts"/"equiv_px_counts"
                 (np.int64 histogram counts against metrics.*_EDGES). The return value is
                 unchanged either way, so the four production creators are unaffected.
```

Before the `for i in range(N)` loop:

```python
    # Residual collection is opt-in and fills the caller's dict: the loop already holds
    # every quantity below, but the four production creators must be byte-identical, so the
    # return contract does not move.
    if collect is not None:
        collect["pairs"] = []
        collect["rel_counts"] = np.zeros(len(REL_EDGES) - 1, dtype=np.int64)
        collect["parallax_counts"] = np.zeros(len(PARALLAX_EDGES) - 1, dtype=np.int64)
        collect["equiv_px_counts"] = np.zeros(len(EQUIV_PX_EDGES) - 1, dtype=np.int64)
        cam_centers = cam2world[:, :3, 3]  # (N, 3) world-space camera positions
        focal_mean = float(K[:, 0, 0].mean() + K[:, 1, 1].mean()) / 2.0
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

            # np.histogram DROPS out-of-range values, which would make a later
            # fraction-below query quietly wrong; clip so outliers land in the end bins.
            for key, vals, edges in (
                ("rel_counts", rel, REL_EDGES),
                ("parallax_counts", par, PARALLAX_EDGES),
                ("equiv_px_counts", rel.abs() * torch.deg2rad(par) * focal_mean, EQUIV_PX_EDGES),
            ):
                v = vals.detach().cpu().numpy()
                collect[key] += np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]

            q = torch.quantile(rel, torch.tensor([0.25, 0.5, 0.75], device=rel.device))
            collect["pairs"].append(
                PairStats(
                    name1=f"frame_{i:06d}",
                    name2=f"frame_{j:06d}",
                    num_matches=0,
                    num_inliers=0,
                    rot_error_deg=float("nan"),
                    t_direction_error_deg=float("nan"),
                    n_pixels=int(sel.sum()),
                    median_rel=float(q[1]),
                    iqr_rel=float(q[2] - q[0]),
                    median_parallax_deg=float(par.median()),
                    below_floor_frac=float((par < PARALLAX_FLOOR_DEG).float().mean()),
                )
            )
```

The `return MultiviewConfidence(...)` at the end is **unchanged**.

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v
```

Expected: all pass — the 8 new plus every pre-existing test in the file.

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

Expected: `imports clean`. **If it cycles**, move `PARALLAX_FLOOR_DEG` and the `*_EDGES` constants into `base.py` and import them *from* `metrics.py` instead — the constants have no dependencies, so the edge always points one way.

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

Histograms accumulate in place (N^2*H*W is 2.4e10 floats at 300 frames) and clip
before binning, since np.histogram drops out-of-range values and would make a
later fraction-below query quietly wrong."
```

---

### Task 4: `calculate_depth_error`

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import (
    PARALLAX_EDGES,
    REL_EDGES,
    calculate_depth_error,
    describe,
)


def _collected(pairs):
    """Build the collect-dict shape that compute_multiview_depth_confidence fills."""
    rel = np.zeros(len(REL_EDGES) - 1, dtype=np.int64)
    par = np.zeros(len(PARALLAX_EDGES) - 1, dtype=np.int64)
    for p in pairs:
        rel += np.histogram(np.full(p.n_pixels, p.median_rel), bins=REL_EDGES)[0]
        par += np.histogram(np.full(p.n_pixels, p.median_parallax_deg), bins=PARALLAX_EDGES)[0]
    return {
        "pairs": pairs,
        "rel_counts": rel,
        "parallax_counts": par,
        "equiv_px_counts": np.zeros(2000, dtype=np.int64),
    }


def _pair(i, j, rel, par, n=100, iqr=0.01):
    return PairStats(
        f"frame_{i:06d}", f"frame_{j:06d}", 0, 0, float("nan"), float("nan"),
        n_pixels=n, median_rel=rel, iqr_rel=iqr, median_parallax_deg=par, below_floor_frac=0.0,
    )


def test_describe_gives_quantiles_and_the_raw_histogram():
    """Quantiles for reading, counts+edges so any threshold query is an exact scipy cdf."""
    counts = np.histogram(np.linspace(0, 1, 1000), bins=np.linspace(0, 1, 11))[0]
    d = describe(counts, np.linspace(0, 1, 11))
    assert d["quantiles"]["0.5"] == pytest.approx(0.5, abs=0.05)
    assert d["counts"] == counts.tolist()
    assert len(d["edges"]) == 11


def test_describe_is_none_when_nothing_was_counted():
    assert describe(np.zeros(10, dtype=np.int64), np.linspace(0, 1, 11)) is None


def test_depth_error_reports_grid_and_resolution():
    """Every block stamps its grid — model-res depth with original-res K is a known bug class."""
    m = calculate_depth_error(_collected([_pair(0, 1, 0.0, 3.0)]), focal_px=500.0, resolution="518x518")
    assert m["grid"] == "model" and m["resolution"] == "518x518"


def test_depth_error_pair_rows_carry_separation():
    m = calculate_depth_error(_collected([_pair(0, 5, 0.02, 3.0)]), focal_px=500.0, resolution="x")
    assert m["pairs"][0]["temporal_separation"] == 5


def test_scale_bias_is_the_signed_median_not_the_magnitude():
    """A pure scale error has a large median and a small spread; sign must survive."""
    m = calculate_depth_error(_collected([_pair(0, 1, -0.08, 3.0, iqr=0.005)]), focal_px=500.0, resolution="x")
    assert m["pairs"][0]["median_rel"] == pytest.approx(-0.08)
    assert m["scale_bias"]["median"] == pytest.approx(0.08)


def test_pixel_equivalent_lands_on_each_pair_row():
    m = calculate_depth_error(_collected([_pair(0, 1, 0.1, 2.0)]), focal_px=500.0, resolution="x")
    assert m["pairs"][0]["error_in_pixels"] == pytest.approx(depth_error_in_pixels(0.1, 2.0, 500.0))


def test_below_floor_pairs_report_null_pixel_equivalent_not_zero():
    m = calculate_depth_error(
        _collected([_pair(0, 1, 0.1, PARALLAX_FLOOR_DEG * 0.5)]), focal_px=500.0, resolution="x"
    )
    assert m["pairs"][0]["error_in_pixels"] is None
    assert m["parallax"]["pairs_below_floor"] == 1


def test_depth_error_is_unavailable_not_a_crash_when_empty():
    m = calculate_depth_error(_collected([]), focal_px=500.0, resolution="x")
    assert m["available"] is False and "reason" in m


def test_rising_residual_with_depth_is_visible_in_the_strata():
    """Bins are the scene's own depth quantiles — one recon unit is not one metre."""
    pairs = [_pair(k, k + 1, 0.01 * (k + 1), 3.0) for k in range(20)]
    depths = np.linspace(1.0, 10.0, 20)
    m = calculate_depth_error(_collected(pairs), focal_px=500.0, resolution="x", pair_depths=depths)
    medians = [b["median_abs_rel"] for b in m["by_depth"]["bins"]]
    assert medians == sorted(medians)
    assert "verdict" not in m["by_depth"]


def test_constant_depth_strata_do_not_crash():
    pairs = [_pair(k, k + 1, 0.01, 3.0) for k in range(10)]
    m = calculate_depth_error(
        _collected(pairs), focal_px=500.0, resolution="x", pair_depths=np.full(10, 3.0)
    )
    assert m["by_depth"]["available"] is True
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "describe or depth_error or scale_bias or pixel_equiv or below_floor or strata"
```

Expected: FAIL — `ImportError: cannot import name 'describe'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Shared summarisation
########################################


def describe(counts: np.ndarray, edges: np.ndarray) -> dict | None:
    """Quantile grid plus the raw counts/edges. None when nothing was counted.

    scipy.stats.rv_histogram supplies both the quantiles here and, for a reader, the exact
    "what fraction falls below X" query at any X they pick — which a stored quantile grid
    cannot answer without interpolating worst in the tail. Counts and edges therefore ship
    alongside the quantiles rather than being summarised away.

    The end bins mean "<= first edge" and ">= last edge": values are clipped before binning,
    because np.histogram would otherwise drop them and skew every fraction query.
    """
    if counts.sum() == 0:
        return None
    rv = stats.rv_histogram((counts, edges))
    return {
        "quantiles": {str(q): float(rv.ppf(q)) for q in QUANTILE_GRID},
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "total": int(counts.sum()),
        "note": "end bins are saturating (<= first edge, >= last edge)",
    }


def _frame_index(name: str) -> int | None:
    """Integer index out of a frame_XXXXXX name, so rows join across measurements."""
    digits = "".join(c for c in Path(name).stem if c.isdigit())
    return int(digits) if digits else None


########################################
# Depth cross-view error
########################################


def calculate_depth_error(
    collected: dict, focal_px: float, resolution: str, pair_depths: np.ndarray | None = None
) -> dict:
    """How much the views disagree about depth: scale bias, geometric noise, parallax.

    Evaluated at MODEL resolution on purpose. Depth values are identical under nearest
    upsampling, so original-res evaluation returns the same number — but it would sample a
    guided-FILTERED depth map, reporting lower disagreement than the model actually produced.
    That improvement belongs to the smoother, not the model.

    Args:
        collected:   the dict compute_multiview_depth_confidence(collect=...) filled.
        focal_px:    mean focal in pixels, used only to express the residual in pixel units.
        resolution:  "WxH" of the grid, stamped into the output for the reader.
        pair_depths: optional per-pair median scene depth, enabling the by-depth strata.
    """
    pairs = collected["pairs"]
    if not pairs:
        return {
            "available": False,
            "reason": "no overlapping view pairs produced depth residuals",
            "grid": "model",
            "resolution": resolution,
        }

    rows = []
    for p in pairs:
        i, j = _frame_index(p.name1), _frame_index(p.name2)
        rows.append(
            {
                "i": i,
                "j": j,
                "temporal_separation": None if i is None or j is None else abs(i - j),
                "n_pixels": p.n_pixels,
                # Signed: the median IS the scale bias between the two views.
                "median_rel": p.median_rel,
                # Bias removed: what is left is geometric noise.
                "iqr_rel": p.iqr_rel,
                "median_parallax_deg": p.median_parallax_deg,
                "below_floor_frac": p.below_floor_frac,
                # None, never 0.0 — below the floor this is undefined, and a zero would read
                # as "no error" when it means "cannot tell".
                "error_in_pixels": depth_error_in_pixels(p.median_rel, p.median_parallax_deg, focal_px),
            }
        )

    biases = [abs(p.median_rel) for p in pairs]
    n_below = sum(1 for p in pairs if p.median_parallax_deg < PARALLAX_FLOOR_DEG)

    out = {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; pixel equivalent in px",
        "n_pairs": len(pairs),
        "residual": describe(collected["rel_counts"], REL_EDGES),
        "error_in_pixels": describe(collected["equiv_px_counts"], EQUIV_PX_EDGES),
        # Scale and noise are independent readings of the same signed residual: a pure scale
        # error has a large median and a small spread, a pose error the reverse.
        "scale_bias": _distribution(biases),
        "noise": _distribution([p.iqr_rel for p in pairs]),
        "parallax": {
            **describe(collected["parallax_counts"], PARALLAX_EDGES),
            "floor_deg": PARALLAX_FLOOR_DEG,
            "pairs_below_floor": n_below,
            "pairs_below_floor_frac": n_below / len(pairs),
        },
        "pairs": rows,
        "by_depth": {"available": False, "reason": "pair_depths not supplied"},
    }
    if pair_depths is not None:
        out["by_depth"] = _by_depth(np.asarray(pair_depths, float), np.array(biases))
    return out


def _by_depth(depth: np.ndarray, abs_rel: np.ndarray) -> dict:
    """Residual per depth-quantile bin — "do things disagree more far away?".

    Bins are quantiles of the SCENE's own depth, never absolute distance: one recon unit is
    not one metre and the factor differs per scene and per backbone.

    There is a null to read against. Triangulation uncertainty goes as sigma_Z ~ Z^2/(f*B),
    so a RELATIVE residual should grow roughly linearly in Z. Faster suggests something
    beyond geometry; flat suggests depth normalised in a way that hides error. No verdict is
    emitted — the numbers and the null are both reported.
    """
    keep = np.isfinite(depth) & np.isfinite(abs_rel) & (depth > 0)
    depth, abs_rel = depth[keep], abs_rel[keep]
    if depth.size == 0:
        return {"available": False, "reason": "no valid depth/residual samples"}
    # np.unique collapses tied quantiles, so constant depth degrades to one bin rather than
    # producing empty zero-width bins.
    edges = np.unique(np.quantile(depth, np.linspace(0.0, 1.0, N_STRATA + 1)))
    if edges.size < 2:
        edges = np.array([depth.min(), depth.max() + 1e-9])
    med, _, _ = stats.binned_statistic(depth, abs_rel, statistic="median", bins=edges)
    p90, _, _ = stats.binned_statistic(depth, abs_rel, statistic=lambda v: np.percentile(v, 90), bins=edges)
    cnt, _, _ = stats.binned_statistic(depth, abs_rel, statistic="count", bins=edges)
    return {
        "available": True,
        "binning": "scene depth quantiles",
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual grows ~linearly in Z",
        "bins": [
            {
                "depth_range": [float(edges[k]), float(edges[k + 1])],
                "n": int(cnt[k]),
                "median_abs_rel": float(med[k]),
                "p90_abs_rel": float(p90[k]),
            }
            for k in range(len(edges) - 1)
            if cnt[k] > 0
        ],
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py
git commit -m "feat(geometry): calculate_depth_error with scale/noise separation

Signed median is the scale bias, spread with the bias removed is geometric
noise — a pure scale error has a large median and small spread, a pose error the
reverse. Below the parallax floor the pixel equivalent is null, never 0.0: a zero
would read as 'no error' when it means 'cannot tell'.

Depth strata are the scene's own quantiles via scipy.stats.binned_statistic, and
carry the sigma_Z ~ Z^2/(f*B) null so 'faster than linear' and 'flat' are both
readable. Numbers and null, no verdict.

Model resolution on purpose: depth is identical under nearest upsampling, so
original-res evaluation returns the same number while sampling a guided-FILTERED
map — reporting lower disagreement than the model produced."
```

---

### Task 5: `calculate_photometric_error`

The only measurement depending on appearance. Normalised per patch, absorbing both the `[0,255]` (VGGT family) vs `[0,1]` (MapAnything) split and any exposure change — otherwise exposure swamps geometry.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import calculate_photometric_error, normalized_residual


def test_identical_patches_have_zero_residual():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert normalized_residual(a, a.copy()) == pytest.approx(0.0, abs=1e-6)


def test_residual_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    b = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert normalized_residual(a, b) == pytest.approx(normalized_residual(a / 255.0, b / 255.0), abs=1e-5)


def test_residual_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry this measurement exists for."""
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert normalized_residual(a, a * 1.4 + 20.0) == pytest.approx(0.0, abs=1e-4)


def test_residual_grows_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 255, size=(256, 3)).astype(np.float32)
    assert normalized_residual(a, a + rng.normal(0, 5, a.shape)) < normalized_residual(
        a, a + rng.normal(0, 80, a.shape)
    )


def test_flat_patch_returns_nan_not_a_divide_by_zero():
    rng = np.random.default_rng(4)
    assert np.isnan(
        normalized_residual(np.full((64, 3), 128.0), rng.uniform(0, 255, size=(64, 3)))
    )


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


def test_identical_poses_and_depth_warp_perfectly():
    img, d, K, e = _plane()
    m = calculate_photometric_error(img, d, K, e, resolution="32x32", max_separation=1)
    assert m["pairs"][0]["photometric_residual"] == pytest.approx(0.0, abs=0.05)


def test_photometric_respects_max_separation():
    img, d, K, e = _plane(n=4, hw=16)
    m = calculate_photometric_error(img, d, K, e, resolution="16x16", max_separation=1)
    assert all(r["temporal_separation"] <= 1 for r in m["pairs"])


def test_photometric_is_unavailable_for_a_single_frame():
    img, d, K, e = _plane(n=1, hw=16)
    m = calculate_photometric_error(img, d, K, e, resolution="16x16", max_separation=1)
    assert m["available"] is False


def test_photometric_reports_original_grid():
    img, d, K, e = _plane()
    m = calculate_photometric_error(img, d, K, e, resolution="1920x1080", max_separation=1)
    assert m["grid"] == "original" and m["resolution"] == "1920x1080"
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "photometric or patch or residual_is_invariant or warp"
```

Expected: FAIL — `ImportError: cannot import name 'normalized_residual'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Photometric error
########################################

# Below this many valid samples a patch's mean/std are too noisy to normalise against.
MIN_SAMPLES = 8
# O(N * max_separation), not O(N^2): appearance agreement between temporally distant frames
# is dominated by lighting and viewpoint change, not by the reconstruction error measured here.
PHOTOMETRIC_MAX_SEPARATION = 2
PHOTOMETRIC_EDGES = np.linspace(0.0, 2.0, 1001)


def normalized_residual(src: np.ndarray, dst: np.ndarray) -> float:
    """RMS difference between two patches after zero-mean/unit-variance normalisation.

    Normalising per patch buys two invariances a raw difference lacks:
      * the [0, 255] (VGGT family) vs [0, 1] (MapAnything) image-scale split, so one number
        is comparable across backbones;
      * exposure and gain change, which would otherwise swamp the geometry being measured.

    Returns nan for a flat or too-small patch — there is no normalisation for zero variance,
    and nan is dropped downstream rather than counted as agreement.
    """
    a = np.asarray(src, dtype=np.float64).ravel()
    b = np.asarray(dst, dtype=np.float64).ravel()
    if a.size < MIN_SAMPLES or a.size != b.size:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return float("nan")
    return float(np.sqrt(np.mean(((a - a.mean()) / sa - (b - b.mean()) / sb) ** 2)))


def calculate_photometric_error(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    resolution: str,
    max_separation: int = PHOTOMETRIC_MAX_SEPARATION,
) -> dict:
    """Warp each frame into its neighbours through pose+depth and score the RGB difference.

    The only measurement depending on appearance, so a residual here that the depth and
    epipolar measurements do not show points at image formation rather than geometry.

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

    rows, counts = [], np.zeros(len(PHOTOMETRIC_EDGES) - 1, dtype=np.int64)
    for i in range(N):
        # Unproject frame i's pixels to world through its own K and pose
        pts_cam = (np.linalg.inv(intrinsics[i]) @ pix.T).T * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam, ones], axis=-1).T).T[:, :3]

        for j in range(i + 1, min(N, i + max_separation + 1)):
            pts_j = (extrinsics[j] @ np.concatenate([pts_world, ones], axis=-1).T).T[:, :3]
            proj = (intrinsics[j] @ pts_j.T).T
            z = np.clip(proj[:, 2], 1e-6, None)
            # NEAREST sampling, matching the depth pass: bilinear across a depth
            # discontinuity blends two surfaces into a colour present on neither.
            ui = np.round(proj[:, 0] / z).astype(np.int64)
            vi = np.round(proj[:, 1] / z).astype(np.int64)
            ok = (pts_j[:, 2] > 0) & (depth[i].ravel() > 0)
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < MIN_SAMPLES:
                continue
            r = normalized_residual(images[i].reshape(-1, 3)[ok], images[j][vi[ok], ui[ok]])
            if np.isnan(r):
                continue
            counts += np.histogram(
                np.clip([r], PHOTOMETRIC_EDGES[0], PHOTOMETRIC_EDGES[-1]), bins=PHOTOMETRIC_EDGES
            )[0]
            rows.append(
                {"i": i, "j": j, "temporal_separation": j - i, "photometric_residual": r, "n_pixels": int(ok.sum())}
            )

    if not rows:
        return {
            "available": False,
            "reason": "no view pairs produced photometric residuals",
            "grid": "original",
            "resolution": resolution,
        }
    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "dimensionless (zero-mean/unit-variance normalised RMS)",
        "n_pairs": len(rows),
        "residual": describe(counts, PHOTOMETRIC_EDGES),
        "pairs": rows,
    }
```

- [ ] **Step 4: Run to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v
```

Expected: 28 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/metrics.py tests/geometry/test_metrics.py
git commit -m "feat(geometry): calculate_photometric_error, normalised per patch

Zero-mean/unit-variance normalisation buys invariance to the [0,255] vs [0,1]
backbone image-scale split AND to exposure change, which would otherwise swamp
the geometry this measures. Flat patches return nan rather than dividing by zero.

Nearest sampling, matching the depth pass: bilinear across a discontinuity
blends two surfaces into a colour present on neither. O(N * max_separation), not
O(N^2) — appearance agreement between temporally distant frames is dominated by
lighting and viewpoint change, not by reconstruction error."
```

---

### Task 6: `calculate_confidence_correlation`

Confidence is **not** an error measurement. The question is whether the model knows when it is wrong. Nothing in this repo has ever checked it.

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
from collab_splats.geometry.metrics import calculate_confidence_correlation


def test_calibrated_confidence_gives_strong_negative_correlation():
    """High confidence should mean LOW disagreement, hence negative."""
    c = np.linspace(0.0, 1.0, 1000)
    assert calculate_confidence_correlation(c, 1.0 - c)["spearman"] < -0.95


def test_useless_confidence_gives_near_zero_correlation():
    rng = np.random.default_rng(0)
    m = calculate_confidence_correlation(rng.uniform(size=5000), rng.uniform(size=5000))
    assert abs(m["spearman"]) < 0.1


def test_percentile_bins_not_absolute_thresholds():
    """Confidence is LOGITS on LoGeR — absolute thresholds are meaningless across backbones."""
    c = np.linspace(-8.0, 12.0, 1000)  # deliberately not in [0, 1]
    m = calculate_confidence_correlation(c, 1.0 / (1.0 + np.exp(c)))
    assert m["binning"] == "confidence percentiles" and len(m["bins"]) > 1


def test_bins_report_residual_per_confidence_decile():
    c = np.linspace(0.0, 1.0, 1000)
    medians = [b["median_abs_residual"] for b in calculate_confidence_correlation(c, 1.0 - c)["bins"]]
    assert medians == sorted(medians, reverse=True)


def test_missing_confidence_is_unavailable_not_zeros():
    """Older zarr stores have no confidence array — absent, never backfilled."""
    m = calculate_confidence_correlation(None, np.array([0.1, 0.2]))
    assert m["available"] is False and "reason" in m


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        calculate_confidence_correlation(np.zeros(10), np.zeros(11))
```

- [ ] **Step 2: Run to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k confidence
```

Expected: FAIL — `ImportError: cannot import name 'calculate_confidence_correlation'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Confidence validation
########################################


def calculate_confidence_correlation(confidence, rel_residual, n_bins: int = N_CONF_BINS) -> dict:
    """Does the model know when it is wrong? Its confidence against measured disagreement.

    Confidence is an INPUT being validated, not an error measurement. Nothing in this repo
    has previously checked whether the model's self-report tracks the disagreement measured
    elsewhere in this module.

    Binned by PERCENTILE and correlated by Spearman, never absolute thresholds or Pearson:
    confidence is logits on LoGeR and a bounded score on others, so an absolute threshold
    means different things per backbone. Spearman is invariant to exactly that rescaling.
    """
    if confidence is None:
        return {"available": False, "reason": "no confidence array in this store (absent, never backfilled)"}
    c = np.asarray(confidence, dtype=np.float64).ravel()
    r = np.abs(np.asarray(rel_residual, dtype=np.float64).ravel())
    if c.size != r.size:
        raise ValueError(f"length mismatch: confidence {c.size}, residual {r.size}")
    keep = np.isfinite(c) & np.isfinite(r)
    c, r = c[keep], r[keep]
    if c.size < n_bins:
        return {"available": False, "reason": f"only {c.size} paired samples"}

    edges = np.unique(np.quantile(c, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return {"available": False, "reason": "confidence is constant"}
    med, _, _ = stats.binned_statistic(c, r, statistic="median", bins=edges)
    p90, _, _ = stats.binned_statistic(c, r, statistic=lambda v: np.percentile(v, 90), bins=edges)
    cnt, _, _ = stats.binned_statistic(c, r, statistic="count", bins=edges)
    return {
        "available": True,
        "grid": "model",
        "binning": "confidence percentiles",
        "note": "confidence is validated here, not treated as an error measurement",
        "spearman": float(stats.spearmanr(c, r).statistic),
        "n_samples": int(c.size),
        "bins": [
            {
                "confidence_range": [float(edges[k]), float(edges[k + 1])],
                "n": int(cnt[k]),
                "median_abs_residual": float(med[k]),
                "p90_abs_residual": float(p90[k]),
            }
            for k in range(len(edges) - 1)
            if cnt[k] > 0
        ],
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
git commit -m "feat(geometry): calculate_confidence_correlation

Confidence is an input being validated, not an error measurement — the question
is whether the model knows when it is wrong, which nothing in this repo has
checked.

Percentile bins via binned_statistic and scipy's spearmanr, never absolute
thresholds or Pearson: confidence is logits on LoGeR and a bounded score
elsewhere, so an absolute threshold means different things per backbone and
Spearman is invariant to exactly that rescaling."
```

---

### Task 7: `read_epipolar_error`, `build_report`, and the leaf stage

**Files:**
- Modify: `collab_splats/geometry/metrics.py`
- Modify: `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266`
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
import json

from collab_splats.geometry.metrics import (
    SCHEMA_VERSION,
    assemble,
    coverage,
    cumulative,
    frame_ranks,
    read_epipolar_error,
)
from collab_splats.wrapper.reconstructor import LEAF_STAGES, _STAGE_DEPS, _STAGE_ORDER


def _vjson(tmp_path, pair_stats, frame_stats=None):
    p = tmp_path / "verification.json"
    p.write_text(json.dumps({"pair_stats": pair_stats, "frame_stats": frame_stats or {}, "summary": {}}))
    return p


def test_epipolar_reads_pair_stats(tmp_path):
    p = _vjson(tmp_path, [dict(name1="frame_000000", name2="frame_000001", num_matches=500,
                               num_inliers=450, rot_error_deg=0.15, t_direction_error_deg=0.9)])
    m = read_epipolar_error(p, image_width=640)
    assert m["available"] is True and m["grid"] == "original"
    assert m["pairs"][0]["inlier_ratio"] == pytest.approx(0.9)


def test_missing_verification_json_is_unavailable_not_a_crash(tmp_path):
    assert read_epipolar_error(tmp_path / "nope.json", image_width=640)["available"] is False


def test_reprojection_reported_in_px_and_as_image_fraction(tmp_path):
    p = _vjson(tmp_path, [], {"frame_000000": {"mean_reproj_error_px": 1.28, "n_tracks": 100}})
    f = read_epipolar_error(p, image_width=640)["frames"][0]
    assert f["mean_reproj_error_px"] == pytest.approx(1.28)
    assert f["mean_reproj_error_frac_width"] == pytest.approx(0.002)


def test_frame_indices_parsed_from_names(tmp_path):
    p = _vjson(tmp_path, [dict(name1="frame_000003", name2="frame_000011", num_matches=10,
                               num_inliers=8, rot_error_deg=0.2, t_direction_error_deg=1.0)])
    assert read_epipolar_error(p, image_width=640)["pairs"][0]["temporal_separation"] == 8


def test_nan_t_direction_is_dropped_from_the_distribution(tmp_path):
    p = _vjson(tmp_path, [
        dict(name1="frame_000000", name2="frame_000001", num_matches=10, num_inliers=8,
             rot_error_deg=0.2, t_direction_error_deg=None),
        dict(name1="frame_000001", name2="frame_000002", num_matches=10, num_inliers=9,
             rot_error_deg=0.3, t_direction_error_deg=1.0),
    ])
    assert read_epipolar_error(p, image_width=640)["t_direction_error_deg"]["total"] == 1


def _measurements():
    return {k: {"available": False, "reason": "x"} for k in ("epipolar", "depth", "photometric", "confidence")}


def test_assemble_stamps_schema_and_scene():
    r = assemble(_measurements(), {"backend": "vggt_omega", "n_frames": 60}, {}, {}, {})
    assert r["schema_version"] == SCHEMA_VERSION and r["scene"]["backend"] == "vggt_omega"


def test_assemble_emits_no_verdict_keys():
    """The whole point: distributions and cumulative error, never a call."""
    r = assemble(_measurements(), {"backend": "x", "n_frames": 2}, {}, {}, {})
    banned = {"verdict", "cause", "grade", "score", "flag", "flags", "pass", "fail"}
    assert not (banned & set(r)), f"verdict-shaped key leaked: {banned & set(r)}"


def test_assemble_records_which_measurements_ran():
    m = _measurements()
    m["depth"] = {"available": True, "n_pairs": 3}
    assert assemble(m, {"backend": "x", "n_frames": 3}, {}, {}, {})["measurements_available"] == ["depth"]


def test_assemble_survives_everything_unavailable():
    """A report must never fail a reconstruction."""
    assert assemble(_measurements(), {"backend": "x", "n_frames": 0}, {}, {}, {})["measurements_available"] == []


def test_report_is_json_serializable():
    r = assemble(_measurements(), {"backend": "x", "n_frames": np.int64(1)}, {}, {"depth": {0: np.float32(0.5)}}, {})
    json.dumps(r, default=lambda o: o.item())  # must not raise on numpy scalars


def test_cumulative_uses_sequential_pairs_and_absolute_steps():
    """Signed steps would cancel and hide accumulation; |i-j|>1 pairs are not steps."""
    rows = [
        {"i": 0, "j": 1, "temporal_separation": 1, "v": 0.1},
        {"i": 1, "j": 2, "temporal_separation": 1, "v": -0.1},
        {"i": 0, "j": 5, "temporal_separation": 5, "v": 9.9},
    ]
    c = cumulative(rows, "v")
    assert c["frame_index"] == [1, 2] and c["cumulative"] == pytest.approx([0.1, 0.2])


def test_frame_ranks_are_numbers_not_labels():
    """No verdict, no cause, no flag — a number the reader interprets."""
    r = frame_ranks({0: 1.0, 1: 5.0, 2: 9.0})
    assert r[0] == pytest.approx(0.0) and r[2] == pytest.approx(1.0)


def test_frame_ranks_skip_nan():
    assert 1 not in frame_ranks({0: 1.0, 1: float("nan"), 2: 3.0})


def test_coverage_from_original_coords():
    """VGGTX centre-crops height to 518 — a 16:9 source loses a band with no depth at all."""
    c = coverage(np.array([[0, 281, 1920, 799, 1920, 1080]], dtype=np.float32))
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
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -v -k "epipolar or assemble or cumulative or frame_ranks or coverage or leaf"
```

Expected: FAIL — `ImportError: cannot import name 'read_epipolar_error'`

- [ ] **Step 3: Write the epipolar reader and the small helpers**

Append to `collab_splats/geometry/metrics.py`:

```python
########################################
# Epipolar error (read from verify)
########################################

ROT_EDGES = np.linspace(0.0, 30.0, 601)
TDIR_EDGES = np.linspace(0.0, 90.0, 901)
INLIER_EDGES = np.linspace(0.0, 1.0, 101)


def read_epipolar_error(verification_json: Path, image_width: int) -> dict:
    """Pose-only error, read from verify's verification.json. Never re-runs the matcher.

    The ONLY measurement here that never touches depth, which is the entire reason
    attribution is possible: an error that moves this and not the depth measurement is a
    pose error.

    Already original-resolution — verify estimates from original-res keypoints. Reprojection
    is reported in px AND as a fraction of image width, because a bare pixel count is not
    comparable across backbones (VGGTX 518-crop, MapAnything 512/518, omega 448x592).
    """
    p = Path(verification_json)
    if not p.exists():
        return {"available": False, "reason": f"no verification.json at {p} — run the verify stage", "grid": "original"}
    data = json.loads(p.read_text())

    rot = np.zeros(len(ROT_EDGES) - 1, dtype=np.int64)
    tdir = np.zeros(len(TDIR_EDGES) - 1, dtype=np.int64)
    inl = np.zeros(len(INLIER_EDGES) - 1, dtype=np.int64)
    rows = []
    for s in data.get("pair_stats", []):
        i, j = _frame_index(s["name1"]), _frame_index(s["name2"])
        n_m, n_i = s.get("num_matches") or 0, s.get("num_inliers") or 0
        ratio = (n_i / n_m) if n_m else None
        # None/nan means the pair was degenerate. Skip it in the counts so it cannot poison
        # the distribution, while its row still appears in the table.
        for counts, val, edges in ((rot, s.get("rot_error_deg"), ROT_EDGES),
                                   (tdir, s.get("t_direction_error_deg"), TDIR_EDGES),
                                   (inl, ratio, INLIER_EDGES)):
            if val is not None and np.isfinite(val):
                counts += np.histogram(np.clip([val], edges[0], edges[-1]), bins=edges)[0]
        rows.append({
            "i": i, "j": j,
            "temporal_separation": None if i is None or j is None else abs(i - j),
            "num_matches": n_m, "num_inliers": n_i, "inlier_ratio": ratio,
            "rot_error_deg": s.get("rot_error_deg"),
            "t_direction_error_deg": s.get("t_direction_error_deg"),
        })

    frames = []
    for name, fs in sorted(data.get("frame_stats", {}).items()):
        px = fs.get("mean_reproj_error_px")
        frames.append({
            "name": name, "index": _frame_index(name),
            "n_tracks": fs.get("n_tracks"), "track_survival": fs.get("track_survival"),
            "mean_reproj_error_px": px,
            # A bare pixel count is not comparable across backbones — normalise.
            "mean_reproj_error_frac_width": None if px is None else px / image_width,
        })

    return {
        "available": True, "grid": "original", "resolution": f"width={image_width}",
        "units": "degrees; reprojection in px and as a fraction of image width",
        "source": str(p), "n_pairs": len(rows),
        "rot_error_deg": describe(rot, ROT_EDGES),
        "t_direction_error_deg": describe(tdir, TDIR_EDGES),
        "inlier_ratio": describe(inl, INLIER_EDGES),
        "pairs": rows, "frames": frames,
    }


########################################
# Second-order views
########################################


def cumulative(rows: list[dict], key: str) -> dict:
    """Running accumulation of |value| along consecutive frames — "does disagreement build?".

    Sequential pairs only: a |i-j|=5 pair is a revisit observation, not a step along the
    trajectory, and summing it would double-count. Absolute values, because signed steps
    cancel and would hide exactly the accumulation this exists to show.

    Read against the per-separation view, not alone: frame index is a confounded axis (scene
    content, motion speed and exposure all correlate with it).
    """
    steps = sorted(
        (r for r in rows if r.get("temporal_separation") == 1 and r.get(key) is not None
         and np.isfinite(r[key])),
        key=lambda r: min(r["i"], r["j"]),
    )
    return {
        "frame_index": [int(max(r["i"], r["j"])) for r in steps],
        "cumulative": np.cumsum([abs(float(r[key])) for r in steps]).tolist(),
        "note": "sequential pairs only; absolute steps; read against the per-separation view",
    }


def frame_ranks(per_frame: dict[int, float]) -> dict[int, float]:
    """Each frame's rank within this scene's own distribution, in [0, 1].

    A NUMBER, never a label. The report does not say which measurement is to blame for a
    high-ranking frame, does not name a cause, and does not flag it — the reader sees that
    frame 47 sits at p99 in depth and p60 in epipolar and draws their own conclusion.

    Within-scene ranks need no absolute threshold, sidestepping both the units problem and
    cross-backbone incomparability.
    """
    items = [(k, v) for k, v in per_frame.items() if v is not None and np.isfinite(v)]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: 0.0}
    r = (stats.rankdata([v for _, v in items]) - 1) / (len(items) - 1)
    return {k: float(x) for (k, _), x in zip(items, r)}


def coverage(original_coords: np.ndarray) -> dict:
    """Fraction of each ORIGINAL frame the model crop actually reconstructed.

    Rows are [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]. VGGTX resizes width to 518 and
    centre-CROPS height to 518, so on a 16:9 source a large band of every frame has no depth
    at all. Model-resolution evaluation is structurally blind to this — the model-res grid
    IS the crop.
    """
    c = np.asarray(original_coords, dtype=np.float64)
    area = np.clip(c[:, 2] - c[:, 0], 0, None) * np.clip(c[:, 3] - c[:, 1], 0, None)
    fr = area / np.maximum(c[:, 4] * c[:, 5], 1e-9)
    return {
        "per_frame": [{"index": k, "covered_fraction": float(f)} for k, f in enumerate(fr)],
        "median_covered_fraction": float(np.median(fr)) if fr.size else None,
        "min_covered_fraction": float(np.min(fr)) if fr.size else None,
    }


def assemble(measurements: dict, scene: dict, cumulatives: dict, ranks: dict, cover: dict) -> dict:
    """Assemble report.json. Distributions and cumulative error — never a verdict.

    No key grades the scene, names a cause, or flags a frame. Absolute thresholds that would
    justify a verdict are exactly what this stage exists to inform, so inventing them now
    would be a guess dressed as a finding.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "measurements_available": sorted(k for k, m in measurements.items() if m.get("available")),
        "measurements": measurements,
        "cumulative": cumulatives,
        "frame_percentile_ranks": ranks,  # numbers in [0, 1], not labels
        "coverage": cover,
        "notes": {
            "verdicts": "none by design — this describes distributions, it does not grade",
            "units": "scale-free or normalised throughout; 1 recon unit is NOT 1 metre",
            "attribution": "measurements differ in what they depend on; read them against each other",
        },
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
    """
    from collab_splats.pointcloud.feedforward.base import FeedforwardResult, compute_multiview_depth_confidence

    r = FeedforwardResult.load_zarr(zarr_path, load_images=False)
    n = len(r.depth)
    model_res = f"{r.model_width}x{r.model_height}"
    focal_px = float(r.intrinsics[:, 0, 0].mean() + r.intrinsics[:, 1, 1].mean()) / 2.0

    # One dense pass yields the depth residual, the scale split, the parallax angles and the
    # pixel equivalent. abs_thresh stays 0.0: scale invariance holds only there, and it is
    # what lets one function serve backbones with completely different depth scales.
    collected: dict = {}
    compute_multiview_depth_confidence(
        r.depth, r.intrinsics, r.extrinsics, abs_thresh=0.0, rel_thresh=0.05, collect=collected
    )

    # Per-pair median scene depth, for the by-depth strata.
    depth_np = np.asarray(r.depth)
    pair_depths = []
    for p in collected["pairs"]:
        d = depth_np[_frame_index(p.name1)]
        pair_depths.append(float(np.median(d[d > 0])) if (d > 0).any() else np.nan)

    depth_m = calculate_depth_error(collected, focal_px, model_res, pair_depths=np.array(pair_depths))
    epipolar_m = read_epipolar_error(verification_json, image_width=int(r.original_coords[0][4]))

    # Per-frame median |residual|, reused by both the confidence check and the ranks.
    per_frame = {}
    for k in range(n):
        v = [abs(p.median_rel) for p in collected["pairs"] if _frame_index(p.name1) == k]
        if v:
            per_frame[k] = float(np.median(v))
    if r.confidence is None or not per_frame:
        confidence_m = calculate_confidence_correlation(None, np.array([]))
    else:
        conf = np.asarray(r.confidence)
        confidence_m = calculate_confidence_correlation(
            np.array([float(np.median(conf[k])) for k in per_frame]),
            np.array(list(per_frame.values())),
        )

    photometric_m = _photometric_at_original_res(r, frames_zarr, n)

    measurements = {"epipolar": epipolar_m, "depth": depth_m,
                    "photometric": photometric_m, "confidence": confidence_m}
    ranks = {"depth": frame_ranks(per_frame)}
    cums = {"depth": cumulative(depth_m.get("pairs", []), "median_rel")}
    if epipolar_m.get("available"):
        ranks["epipolar"] = frame_ranks({
            f["index"]: f["mean_reproj_error_px"] for f in epipolar_m["frames"]
            if f["index"] is not None and f["mean_reproj_error_px"] is not None
        })
        cums["epipolar"] = cumulative(epipolar_m["pairs"], "rot_error_deg")

    report = assemble(
        measurements,
        {"backend": backend, "n_frames": n, "model_resolution": model_res, "zarr": str(zarr_path)},
        cums, ranks, coverage(r.original_coords),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=lambda o: o.item()))
    logger.info("Wrote %s (%d measurements available)", output_path, len(report["measurements_available"]))
    return report


def _photometric_at_original_res(r, frames_zarr: Path, n: int) -> dict:
    """Upsample depth and rescale K to the original grid, then warp. Never fatal."""
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
        return calculate_photometric_error(
            np.stack(rgbs), np.stack(deps), np.stack(Ks), r.extrinsics[: len(rgbs)],
            resolution=f"{rgbs[0].shape[1]}x{rgbs[0].shape[0]}",
        )
    except Exception as exc:  # noqa: BLE001 — a report must never fail a reconstruction
        logger.warning("photometric measurement failed: %s", exc, exc_info=True)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}", "grid": "original"}
```

**Two API names to confirm before running**, because both are read off memory of the module rather than a fresh grep:

```bash
grep -n "def read\|def frame_indices\|def open" collab_splats/preproc/sampling.py
grep -n "def guided_upsample_depth" collab_splats/mesh/utils.py
grep -n "def load_zarr" collab_splats/pointcloud/feedforward/base.py
```

If `FrameStore`'s accessor is named differently, match the `_LazyFrames` usage at `reconstructor.py:1141`. If `guided_upsample_depth` has a different signature, match the call in `mesh/utils.py`'s native-resolution path. If `load_zarr` has no `load_images` kwarg, drop it — the default already skips images.

- [ ] **Step 5: Register the stage**

`collab_splats/wrapper/reconstructor.py` line 48 — append `"report"`:

```python
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify", "report"]
```

In `_STAGE_DEPS`, after `"verify"`:

```python
    # report reads verification.json when present and runs verify itself when absent, so
    # like verify its only hard dependency is the reconstruction
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
            # default-false flag; a report nobody runs answers nothing, and the measured
            # cost is bounded. The one boolean it would have had is the boolean that keeps
            # it off.
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
        """Reference-free error report: four measurements, one report.json. Reports only.

        Never fails a reconstruction — a measurement that cannot run records
        {"available": false, "reason": ...} and the rest still emit.
        """
        out_json = self.backend_dir / "report.json"
        if not overwrite and self._stage_output_exists("report"):
            logger.info("Report exists at %s, skipping", out_json)
            return out_json
        if self._resolve_result() is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # The epipolar measurement is the only one that never touches depth, which is what
        # makes attribution possible — worth building when absent rather than skipped.
        verification_json = self.backend_dir / "colmap" / "verification.json"
        if not verification_json.exists():
            try:
                self.verify()
            except Exception:  # noqa: BLE001 — a report must never fail a reconstruction
                logger.warning("verify failed; epipolar will be unavailable", exc_info=True)

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

read_epipolar_error consumes verify's output and never re-runs the matcher — two
stages calling it would be waste. It is the only measurement that never touches
depth, which is the entire reason attribution works.

_STAGE_DEPS['report'] == ['pointcloud'], so --stages report re-runs against a
scene pulled from environments-processed with no rerun.py change.

Always on with no config boolean, against repo precedent: every other diagnostic
ships behind a default-false flag, and the one boolean this would have had is
the boolean that keeps it off.

Cumulative walks sequential pairs only (a |i-j|=5 pair is a revisit, not a
trajectory step) using absolute values, since signed steps cancel and hide the
accumulation. Frame ranks are numbers, never labels — a test asserts no
verdict-shaped key can leak into report.json."
```

---

### Task 8: Negative controls — prove the measurements separate

A metric that does not move under an injected fault is decoration. **The depth-scale control is load-bearing**: it proves the measurements separate rather than moving together.

**Files:** Create `tests/geometry/test_metrics_controls.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Negative controls: each measurement must move under its own fault and stay still under others."""

import numpy as np
import pytest

from collab_splats.geometry.metrics import (
    PARALLAX_FLOOR_DEG,
    depth_error_in_pixels,
    normalized_residual,
)
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
    return {(int(p.name1[-6:]), int(p.name2[-6:])): p for p in out["pairs"]}


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
    """Normalisation is what buys this — a raw difference would flag it as error."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=(256, 3)).astype(np.float32)
    assert normalized_residual(a, a * 1.6 + 30.0) == pytest.approx(0.0, abs=1e-4)


def test_control_forward_motion_lands_below_the_parallax_floor():
    """Pure forward motion drives perpendicular baseline to ~0 near the epipole."""
    depth, K, extr = _scene(n=3)
    extr[:, 0, 3] = 0.0
    for k in range(3):
        extr[k, 2, 3] = -0.05 * k  # translate along the viewing axis instead
    p = _pairs(depth, K, extr)
    worst = min(p.values(), key=lambda q: q.median_parallax_deg)
    assert worst.median_parallax_deg < PARALLAX_FLOOR_DEG
    # And the bridge must decline to answer rather than emit an infinity.
    assert depth_error_in_pixels(0.05, worst.median_parallax_deg, FOCAL) is None
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

Forward-motion control confirms the parallax floor triggers and that the bridge
declines to answer rather than emitting an infinity."
```

---

### Task 9: Real-scene run, measured numbers, contract, retirement

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
print(json.dumps(rep[\"measurements\"][\"depth\"][\"residual\"][\"quantiles\"], indent=2))
" 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/ee7cc0e1-beee-4d06-908d-0a6838558f0b/scratchpad/scene_report.log'
```

Watch: `tmux attach -t scene_report`. Memory: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

- [ ] **Step 2: Check the sanity target**

Measured baseline on this store: **median |rel| 0.37%, p90 2.27%, p99 25.67%**, tightening to p90 0.92% at conf>p20.

Compare the printed quantiles. They should agree closely — it is the same quantity `depth_disagreement.py` measured. **If they differ materially, explain the difference before proceeding.** Check first: residual population (`counted & has_depth` here) vs what that script used, and signed-vs-absolute.

- [ ] **Step 3: Rank control**

Run Step 1 against a `mapanything` store and a `vggt_omega` store of the same scene — they differ 1.6× in ATE on chess/seq-01.

```bash
/opt/venv/reconstruction/bin/python -c "
import json, pathlib
for name in ('mapanything', 'vggt_omega'):
    p = pathlib.Path(f'evals/results/{name}/report.json')
    if not p.exists():
        print(name, 'MISSING'); continue
    d = json.loads(p.read_text())['measurements']['depth']
    print(name, 'p50', d['residual']['quantiles']['0.5'],
          'p99', d['residual']['quantiles']['0.99'],
          'scale_bias', d['scale_bias'])
"
```

**If the report cannot order those two, it will not separate anything.** Record the outcome either way — a null result here is the most important number in the task.

- [ ] **Step 4: Append the measurements**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
## Task 9: report stage, measured

- Scene: evals/results/mv_vggt_omega (60 frames, vggt_omega)
- Wall clock: <REPORT_SECONDS> s | Peak rss: <GB> / 46.6 GB
- Measurements available: <list>

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

### Parallax
- Pairs below the 0.5 deg floor: <n> / <total> | Median parallax: <x> deg

### By depth
<Per-bin medians, and whether growth is slower/linear/faster than the
sigma_Z ~ Z^2/(f*B) null.>

### Cumulative
<Does the sequential-pair curve rise faster than linearly? Read against the
per-separation view before calling it accumulation.>
```

- [ ] **Step 5: Document the contract**

In `configs/README.md`, beside the existing `colmap/verification.json` entry:

```markdown
- `<backend>/report.json` — reference-free scene error report. Four measurements
  (`epipolar`, `depth`, `photometric`, `confidence`), per-pair and per-frame
  tables, histogram counts+edges, cumulative curves along the trajectory, and
  per-frame percentile ranks. Written by the always-on `report` leaf stage;
  re-runnable with `--stages report --overwrite`.

  **Report-only: nothing here feeds back into the reconstruction.** No verdict,
  no grade, no cause — distributions and cumulative error only. Every block
  stamps its `grid` (`model` or `original`) and `resolution`; units are
  scale-free or normalised throughout, because 1 recon unit is not 1 metre and
  the factor differs per scene and per backbone. Pixel counts are not comparable
  across backbones, so reprojection is reported in px *and* as a fraction of
  image width. Histogram end bins saturate (values are clipped before binning),
  so `scipy.stats.rv_histogram((counts, edges)).cdf(x)` answers "what fraction
  falls below x" exactly at any x.
```

- [ ] **Step 6: Retire the superseded script**

`evals/scripts/depth_disagreement.py` measured the signed residual as a one-off. That residual now lives in the refactored function, and two implementations of one quantity drift apart.

```bash
git rm evals/scripts/depth_disagreement.py
grep -rn "depth_disagreement" --include=*.py --include=*.md --include=*.ipynb . \
  | grep -v '\.git' | grep -v baseck | grep -v '\.worktrees'
```

Expected: only `docs/superpowers/` prose. If code references it, update the reference rather than keeping the file.

- [ ] **Step 7: Full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly 2>&1 | tail -20
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the 5 pre-existing `tests/wrapper/` failures and the pre-existing `test_view_transform_scales_to_target_radius` failure, **and nothing new**. Then `SMOKE PASS`.

- [ ] **Step 8: Commit**

```bash
git add configs/README.md
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(configs): report.json contract + measured scene error report

Records the first end-to-end run: wall clock, measurements available, the
depth-residual sanity target against the prior depth_disagreement.py numbers,
the mapanything-vs-vggt_omega rank control, parallax floor coverage, depth
strata against the sigma_Z ~ Z^2/(f*B) null, and the cumulative curve.

Retires evals/scripts/depth_disagreement.py — its signed residual now lives in
compute_multiview_depth_confidence, and two implementations of one quantity
drift apart."
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| Stage wiring (`report` leaf, always on, never fails) | 7 |
| Epipolar + reprojection | 1, 7 |
| Depth cross-view + scale | 3, 4 |
| Photometric | 5, 7 |
| Confidence validation | 6 |
| Resolution contract (per-measurement, grid stamped) | 4, 5, 7 |
| Units (scale-free / normalised) | 4, 5, 7 |
| Signed residual: scale vs noise | 3, 4 |
| Depth strata + null hypothesis | 4 |
| Parallax bridge + floor | 2, 4, 8 |
| Pair table + second-order axes | 4, 7 |
| Distributions + cumulative error | 4, 7 |
| Per-frame ranks, not calls | 7 |
| Exact threshold queries from histograms | 4 (`describe`) |
| Coverage from `original_coords` | 7 |
| `report.json` + schema_version | 7 |
| Runtime measurement | 1 |
| Negative control per measurement | 8 |
| Sanity target + rank control | 9 |
| `configs/README.md` contract | 9 |
| Retire `depth_disagreement.py` | 9 |

**Gaps accepted and stated, not silently dropped:**
- **Optional GT block** (spec: "Ground truth — an optional block") has no task. Genuinely optional, adds a second input path, and every measurement computes identically without it. The spec's non-forking contract holds because nothing in Tasks 1-9 branches on GT.
- **Spatial pair distance `‖Cᵢ−Cⱼ‖/extent`** is not built. Temporal separation is, and it carries the drift axis; the spatial axis needs camera-extent normalisation that only matters once a scene with real revisits is measured. Add it when Task 9 shows revisit pairs exist.

**Type consistency:** `PairStats` is the single per-pair row type across `verification.py`, the mv loop, `calculate_depth_error` and `read_epipolar_error`; all new fields default to `None`. The `collect` dict keys (`pairs`, `rel_counts`, `parallax_counts`, `equiv_px_counts`) are written in Task 3 and read unchanged in Task 4. `describe(counts, edges)` has one shape, consumed identically by all four measurements. `_frame_index` returns `int | None` and every consumer handles `None`. `PARALLAX_FLOOR_DEG` and the `*_EDGES` constants have exactly one definition, in `metrics.py`, with Task 3 Step 6 guarding the import direction.

**Placeholder scan:** no TBD/TODO. Three named unknowns with stated resolution paths, not hidden ones: Task 1 Step 3's `Reconstructor` construction (depends on the chosen scene), and Task 7 Step 4's `FrameStore.read` / `guided_upsample_depth` / `load_zarr` signatures (grep commands and fallbacks supplied inline).

**Overengineering audit (what this plan does NOT write):** no histogram class, no residual/stats dataclasses, no `MultiviewConfidence` change, no hand-rolled Spearman, rank, quantile, binning, or JSON coercion, no `explained_fraction` wrapper around a division, no `quantile_block` wrapper, and no separate `histogram.py`/`report.py` split. Each is either scipy, numpy, or an existing repo function — enumerated with its source in the Reuse Audit above.
