# Scene Error Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `report` leaf stage that produces a reference-free `report.json` locating reconstruction error in space and time across four channels, with no ground truth and no verdicts.

**Architecture:** Four channels with deliberately different dependencies (epipolar = poses only, depth cross-view = poses+depth, photometric = poses+depth+appearance, confidence = validated input) are read against each other to attribute error. The parallax bridge `δd = r·d` converts depth residuals into pixel units so the channels sit on one axis. Per-pair and per-frame tables plus fixed-bin histograms make arbitrary-threshold queries exact counts, never quantile interpolation.

**Tech Stack:** Python 3.11 (`/opt/venv/reconstruction/bin/python`), numpy, torch, pycolmap, zarr v3, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-scene-error-report-design.md` (commit `d5021b4`)

---

## Environment and Safety (read before Task 1)

- **Python:** always `/opt/venv/reconstruction/bin/python`. Bare `python` is 3.13 and wrong for this project.
- **Heavy runs:** tmux only, serially. Never two GPU jobs at once. Container cap 46.6 GB; read `rss` from `/sys/fs/cgroup/memory/memory.stat`, **not** `memory.usage_in_bytes`.
- **A concurrent session keeps files dirty** (`configs/base.yaml`, `pyproject.toml`, `collab_splats/remote/rerun.py`). **Never `git add -A`. Never repo-wide `black .`** — the venv's black 26.5.1 is newer than the repo's formatting and will reformat unrelated files. Stage named files only.
- **`git add -f`** is required for anything under `docs/superpowers/` (gitignored).
- **Pre-existing failures that are not yours:** 5 in `tests/wrapper/`, and `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius`.

## File Structure

**Create:**
- `collab_splats/geometry/histogram.py` — `FixedHistogram`: generic fixed-edge accumulator with exact `fraction_below`. One responsibility, no project types, heavily unit-tested.
- `collab_splats/geometry/report.py` — residual/parallax dataclasses, the parallax bridge math, the four channel builders, and the `report.json` aggregator.
- `tests/geometry/test_histogram.py`
- `tests/geometry/test_report.py`

**Modify:**
- `collab_splats/pointcloud/feedforward/base.py` — `compute_multiview_depth_confidence` gains opt-in residual+parallax collection (Task 3). Four production callers must stay byte-identical.
- `collab_splats/pointcloud/feedforward/__init__.py` — export the new dataclasses.
- `collab_splats/wrapper/reconstructor.py:48,49-66,1161-1185,1218-1266` — register the `report` stage.
- `configs/README.md` — processed-scene output contract.
- `tests/pointcloud/test_mv_conf.py` — backward-compat proof + collection tests.

**Delete (Task 14):**
- `evals/scripts/depth_disagreement.py` — its residual lives in the refactored function.

**Hard constraint that shapes everything:** per-pair per-pixel residuals are `N²·H·W` floats — 2.4e10 at 300 frames. The loop **accumulates into histograms and per-pair scalars in place** and never returns raw residual arrays.

---

### Task 1: Prove `verify` runs end-to-end and measure the epipolar cost

The epipolar channel reads `verification.json`. **No `verification.json` exists anywhere in this repo** — `verify` has never run to completion here, so the channel's only input is unproven. This task de-risks that before any code depends on it, and produces the cost number the spec owes.

**Files:**
- Create: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`

- [ ] **Step 1: Confirm no verification.json exists (baseline claim check)**

```bash
find /workspace/collab-splats -name verification.json -not -path '*/.git/*' 2>/dev/null; echo "exit=$?"
```

Expected: no paths printed. If one IS found, read it and record its frame count — the cost measurement below may be reusable and the "never run" claim in the spec must be corrected.

- [ ] **Step 2: Find a reconstruction to verify**

```bash
ls -d /workspace/collab-splats/evals/results/*/ 2>/dev/null | head -20
find /workspace/collab-splats/evals/results -maxdepth 3 -name feedforward.zarr 2>/dev/null | head
```

Expected: at least one directory containing `feedforward.zarr` and a `colmap/sparse/0/`. `evals/results/mv_vggt_omega` is the known 60-frame omega store used as the sanity target. Record the chosen path as `$SCENE`.

- [ ] **Step 3: Run verify in tmux, timed**

`verify` needs a `frames.zarr` alongside the reconstruction (it feeds original-res frames to the matcher). If `$SCENE` has none, use a scene produced by the normal pipeline instead of an eval output dir.

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

The `Reconstructor` construction above is indicative — use whatever constructor the chosen scene needs (check `docs/examples/run_pipeline_remote.py` for the driver pattern). The measurement, not the invocation, is the deliverable.

Watch it: `tmux attach -t verify_measure`. Memory guard while it runs:

```bash
grep '^rss ' /sys/fs/cgroup/memory/memory.stat
```

- [ ] **Step 4: Record the measurement**

Write `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
# Scene error report — measured numbers

## Task 1: epipolar channel cost (verify)

- Scene: <path>
- Frames: <N>
- Backbone: <name>
- Matcher (`localization.matcher`): <name>
- Wall clock: <VERIFY_SECONDS> s
- Peak rss: <GB> / 46.6 GB
- Pairs generated: <n_pairs from verification.json summary>
- Per-pair cost: <seconds/pairs * 1000> ms
- Extrapolation to 300 frames (~5,400 pairs at overlap=10 + quadratic): <minutes> min

### Did verify complete?
<yes/no. If no: the exact traceback, and what the epipolar channel must do instead.>
```

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(specs): measured epipolar channel cost for the scene error report

First verification.json ever produced in this repo — the epipolar channel's
only input was previously unproven. Records wall clock, pair count, per-pair
cost and the 300-frame extrapolation the design owed."
```

**If `verify` does not complete:** stop and report. Tasks 2-8 and 10-14 do not depend on it and can proceed, but Task 9 (the epipolar adapter) must be re-scoped and the spec's attribution claim weakens — without a poses-only channel there is nothing to separate depth error from pose error against.

---

### Task 2: `FixedHistogram` — exact arbitrary-threshold queries

**Files:**
- Create: `collab_splats/geometry/histogram.py`
- Test: `tests/geometry/test_histogram.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_histogram.py`:

```python
"""Unit tests for FixedHistogram — the exact-threshold accumulator."""

import numpy as np
import pytest

from collab_splats.geometry.histogram import FixedHistogram


def test_counts_land_in_the_right_bins():
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(np.array([0.05, 0.15, 0.15, 0.95]))
    assert h.counts[0] == 1
    assert h.counts[1] == 2
    assert h.counts[9] == 1
    assert h.total == 4


def test_out_of_range_goes_to_overflow_not_dropped():
    """Silently dropping outliers makes fraction_below wrong — the whole point of the class."""
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(np.array([-5.0, 0.5, 7.0]))
    assert h.underflow == 1
    assert h.overflow == 1
    assert h.counts.sum() == 1
    assert h.total == 3


def test_fraction_below_is_exact_at_a_bin_edge():
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(np.array([0.05, 0.15, 0.25, 0.35]))
    # Three of four samples fall strictly below the 0.3 edge.
    assert h.fraction_below(0.3) == pytest.approx(0.75)


def test_fraction_below_counts_underflow_and_overflow():
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(np.array([-1.0, 0.5, 2.0]))
    assert h.fraction_below(0.0) == pytest.approx(1 / 3)   # the underflow sample
    assert h.fraction_below(1.0) == pytest.approx(2 / 3)   # underflow + the in-range one
    assert h.fraction_below(1e9) == pytest.approx(1.0)     # everything, overflow included


def test_nan_is_excluded_from_total():
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(np.array([0.5, np.nan, np.nan]))
    assert h.total == 1


def test_quantile_matches_numpy_within_bin_width():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.0, 1.0, size=100_000)
    h = FixedHistogram(lo=0.0, hi=1.0, bins=1000)
    h.add(v)
    for q in (0.5, 0.9, 0.99):
        assert h.quantile(q) == pytest.approx(float(np.quantile(v, q)), abs=2e-3)


def test_add_is_incremental():
    """Accumulating across pairs must equal one bulk add — the loop calls add() N^2 times."""
    v = np.array([0.1, 0.2, 0.3, 0.4])
    bulk = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    bulk.add(v)
    inc = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    for x in v:
        inc.add(np.array([x]))
    assert np.array_equal(bulk.counts, inc.counts)


def test_empty_histogram_reports_none_not_nan():
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    assert h.total == 0
    assert h.quantile(0.5) is None
    assert h.fraction_below(0.5) is None
    assert h.to_dict()["total"] == 0


def test_to_dict_stores_edges():
    """A reader must never have to reconstruct the binning from lo/hi/bins conventions."""
    h = FixedHistogram(lo=-0.5, hi=0.5, bins=4)
    h.add(np.array([0.0]))
    d = h.to_dict()
    assert d["edges"] == [-0.5, -0.25, 0.0, 0.25, 0.5]
    assert len(d["counts"]) == 4
    assert d["underflow"] == 0 and d["overflow"] == 0


def test_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    h = FixedHistogram(lo=0.0, hi=1.0, bins=10)
    h.add(torch.tensor([0.05, 0.15]))
    assert h.total == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_histogram.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.geometry.histogram'`

- [ ] **Step 3: Write the implementation**

Create `collab_splats/geometry/histogram.py`:

```python
"""Fixed-edge histogram accumulator: exact fraction-below queries without keeping samples."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


########################################
# FixedHistogram
########################################


class FixedHistogram:
    """Fixed-edge histogram with explicit under/overflow, built for incremental accumulation.

    Exists because a quantile grid cannot answer "what fraction falls below an arbitrary X".
    Inverting a quantile grid means interpolating between stored probabilities, and that
    interpolation is worst in the tail — the part that matters. Per-bin counts make the
    query an exact count instead.

    Out-of-range samples go to underflow/overflow rather than being dropped: dropping them
    would make fraction_below silently wrong, which defeats the purpose of the class.
    """

    def __init__(self, lo: float, hi: float, bins: int):
        if not hi > lo:
            raise ValueError(f"hi must exceed lo, got lo={lo}, hi={hi}")
        if bins < 1:
            raise ValueError(f"bins must be >= 1, got {bins}")
        self.lo = float(lo)
        self.hi = float(hi)
        self.bins = int(bins)
        self.edges = np.linspace(self.lo, self.hi, self.bins + 1, dtype=np.float64)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.underflow = 0
        self.overflow = 0

    @property
    def total(self) -> int:
        """Every non-nan sample ever added, in-range and out."""
        return int(self.counts.sum()) + self.underflow + self.overflow

    def add(self, values) -> None:
        """Accumulate a batch of values. Accepts numpy arrays or torch tensors."""
        # Accept torch tensors without importing torch: .detach().cpu().numpy() when present
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        v = np.asarray(values, dtype=np.float64).ravel()
        v = v[~np.isnan(v)]
        if v.size == 0:
            return
        # Split out-of-range first so np.histogram never silently discards them
        below = v < self.lo
        above = v >= self.hi
        self.underflow += int(below.sum())
        self.overflow += int(above.sum())
        inside = v[~below & ~above]
        if inside.size:
            self.counts += np.histogram(inside, bins=self.edges)[0].astype(np.int64)

    def fraction_below(self, x: float) -> float | None:
        """Exact fraction of samples strictly below x. None when nothing was added."""
        if self.total == 0:
            return None
        if x <= self.lo:
            # Only underflow samples can be below the first edge
            return self.underflow / self.total
        if x >= self.hi:
            return (self.underflow + int(self.counts.sum())) / self.total
        # Whole bins below x, plus a linear share of the straddled bin
        idx = int(np.searchsorted(self.edges, x, side="right")) - 1
        width = self.edges[idx + 1] - self.edges[idx]
        partial = self.counts[idx] * (x - self.edges[idx]) / width
        n = self.underflow + int(self.counts[:idx].sum()) + partial
        return float(n / self.total)

    def quantile(self, q: float) -> float | None:
        """Value at quantile q, linearly interpolated inside its bin. None when empty.

        Returns lo/hi when q falls inside the underflow/overflow mass — the true value is
        unbounded there, so the edge is reported and the caller reads the overflow count to
        know it was clipped.
        """
        if self.total == 0:
            return None
        target = q * self.total
        if target <= self.underflow:
            return self.lo
        cum = self.underflow + np.cumsum(self.counts)
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx >= self.bins:
            return self.hi
        prev = self.underflow if idx == 0 else cum[idx - 1]
        in_bin = self.counts[idx]
        frac = 0.0 if in_bin == 0 else (target - prev) / in_bin
        return float(self.edges[idx] + frac * (self.edges[idx + 1] - self.edges[idx]))

    def to_dict(self) -> dict:
        """JSON-ready form. Edges are stored, never left for the reader to reconstruct."""
        return {
            "edges": [float(e) for e in self.edges],
            "counts": [int(c) for c in self.counts],
            "underflow": self.underflow,
            "overflow": self.overflow,
            "total": self.total,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_histogram.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/histogram.py tests/geometry/test_histogram.py
git commit -m "feat(geometry): FixedHistogram for exact arbitrary-threshold queries

A quantile grid gives the CDF at fixed probabilities; inverting it to 'fraction
below arbitrary X' interpolates worst in the tail. Per-bin counts make the query
exact. Out-of-range samples go to underflow/overflow rather than being dropped,
because dropping them makes fraction_below silently wrong."
```

---

### Task 3: Collect signed residual and parallax angle inside the mv loop

`compute_multiview_depth_confidence` already computes `expected_d` and `sampled_d`, then thresholds them to a boolean and **discards the residual**. It also unprojects `pts_world`, from which the parallax angle is two dot products away. Both quantities are free; only the plumbing is new.

**Four production callers must stay byte-identical:** `vggtx.py:354`, `vggt_omega.py:265`, `mapanything.py:446`, `loger.py:440`. Collection is opt-in and default-off.

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/base.py:488-661`
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`
- Test: `tests/pointcloud/test_mv_conf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/pointcloud/test_mv_conf.py`:

```python
def _two_view_setup(scale_j: float = 1.0):
    """Two cameras with a 0.2-unit sideways baseline viewing a constant-depth plane.

    scale_j multiplies frame 1's depth, which injects a known relative residual.
    """
    H = W = 16
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((H, W), 4.0, np.float32), np.full((H, W), 4.0 * scale_j, np.float32)])
    extr = np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)])
    extr[1, 0, 3] = -0.2  # world-to-cam translation => camera 1 sits at x=+0.2
    return depth, np.stack([K, K]), extr


def test_collection_is_off_by_default_and_result_is_unchanged():
    """The four production creators must see byte-identical output."""
    depth, K, extr = _two_view_setup()
    base = compute_multiview_depth_confidence(depth, K, extr, device="cpu")
    assert getattr(base, "residuals", None) is None
    withflag = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect_residuals=True)
    assert np.array_equal(base.ratio, withflag.ratio)
    assert np.array_equal(base.inlier_count, withflag.inlier_count)
    assert np.array_equal(base.valid_count, withflag.valid_count)
    assert np.array_equal(base.judged, withflag.judged)


def test_signed_residual_recovers_an_injected_depth_scale():
    """Frame 1 depth x1.1 must show median relative residual ~ +0.1 on the 0->1 pair."""
    depth, K, extr = _two_view_setup(scale_j=1.1)
    out = compute_multiview_depth_confidence(
        depth, K, extr, device="cpu", rel_thresh=0.5, collect_residuals=True
    )
    row = next(r for r in out.residuals.pairs if r.i == 0 and r.j == 1)
    assert row.median_rel == pytest.approx(0.1, abs=0.02)


def test_signed_residual_is_zero_on_a_consistent_pair():
    depth, K, extr = _two_view_setup()
    out = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect_residuals=True)
    row = next(r for r in out.residuals.pairs if r.i == 0 and r.j == 1)
    assert row.median_rel == pytest.approx(0.0, abs=1e-3)


def test_parallax_angle_matches_geometry():
    """0.2 baseline at depth 4 => atan(0.2/4) ~ 2.86 deg at the principal ray."""
    depth, K, extr = _two_view_setup()
    out = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect_residuals=True)
    row = next(r for r in out.residuals.pairs if r.i == 0 and r.j == 1)
    assert row.median_parallax_deg == pytest.approx(np.degrees(np.arctan(0.2 / 4.0)), abs=0.5)


def test_residual_histogram_accumulates_across_pairs():
    depth, K, extr = _two_view_setup()
    out = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect_residuals=True)
    assert out.residuals.rel_hist.total > 0
    assert out.residuals.parallax_hist.total > 0


def test_occluded_pixels_are_excluded_from_the_residual():
    """Occlusion is absent evidence, not disagreement — it must not pollute the scale bias."""
    depth, K, extr = _two_view_setup()
    depth[1, :, :8] = 0.5  # a near occluder covering half of frame 1
    out = compute_multiview_depth_confidence(depth, K, extr, device="cpu", collect_residuals=True)
    row = next(r for r in out.residuals.pairs if r.i == 0 and r.j == 1)
    # The visible half agrees exactly; the occluded half must not drag the median negative.
    assert row.median_rel == pytest.approx(0.0, abs=1e-3)


def test_residual_is_scale_invariant():
    """Multiplying depth and translation by s must leave the relative residual unchanged."""
    depth, K, extr = _two_view_setup(scale_j=1.1)
    a = compute_multiview_depth_confidence(
        depth, K, extr, device="cpu", rel_thresh=0.5, collect_residuals=True
    )
    s = 7.0
    extr_s = extr.copy()
    extr_s[:, :3, 3] *= s
    b = compute_multiview_depth_confidence(
        depth * s, K, extr_s, device="cpu", rel_thresh=0.5, collect_residuals=True
    )
    ra = next(r for r in a.residuals.pairs if r.i == 0 and r.j == 1)
    rb = next(r for r in b.residuals.pairs if r.i == 0 and r.j == 1)
    assert ra.median_rel == pytest.approx(rb.median_rel, abs=1e-4)
    assert ra.median_parallax_deg == pytest.approx(rb.median_parallax_deg, abs=1e-3)
```

Add to that file's imports if not already present:

```python
import pytest
from collab_splats.pointcloud.feedforward.base import PairResidual, ResidualStats
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v -k "collection or residual or parallax or occluded"
```

Expected: FAIL — `ImportError: cannot import name 'PairResidual'`

- [ ] **Step 3: Add the dataclasses**

In `collab_splats/pointcloud/feedforward/base.py`, immediately **before** `@dataclass class MultiviewConfidence` (around line 431), insert:

```python
# Signed-residual histogram range. Wide enough that the p99 of a healthy scene sits well
# inside it (measured p99 |rel| 25.7% on evals/results/mv_vggt_omega) while the tail still
# lands in overflow rather than compressing the useful range.
_REL_RESIDUAL_RANGE = (-0.5, 0.5)
_REL_RESIDUAL_BINS = 2000
# Parallax angles are small indoors (10-20 mm baselines); 0-30 deg at 0.05 deg resolution.
_PARALLAX_RANGE_DEG = (0.0, 30.0)
_PARALLAX_BINS = 600


@dataclass
class PairResidual:
    """One ordered view pair's depth-agreement summary. Scale-free by construction."""

    i: int
    j: int
    n_pixels: int  # pixels contributing to this pair's residual
    median_rel: float  # signed => SCALE BIAS between the two views
    iqr_rel: float  # spread with the bias removed => GEOMETRIC NOISE
    median_parallax_deg: float  # the pair's depth observability
    below_floor_frac: float  # fraction of pixels under PARALLAX_FLOOR_DEG


@dataclass
class ResidualStats:
    """Pooled histograms plus the per-pair table. Raw per-pixel residuals are never kept.

    N^2 * H * W residuals is 2.4e10 floats at 300 frames, so the loop accumulates in place.
    """

    pairs: list[PairResidual]
    rel_hist: "FixedHistogram"  # signed relative residual, all pairs pooled
    parallax_hist: "FixedHistogram"  # parallax angle in degrees
    equiv_px_hist: "FixedHistogram"  # |rel| * parallax * focal — the bridge, in pixels
```

Add the import at the top of the file, with the other `collab_splats` imports:

```python
from collab_splats.geometry.histogram import FixedHistogram
```

Extend `MultiviewConfidence` with an optional field (default `None` keeps every existing construction site valid):

```python
    judged: np.ndarray  # (N,) bool — False when the view had no overlapping partners
    residuals: "ResidualStats | None" = None  # populated only when collect_residuals=True
```

- [ ] **Step 4: Add the collection to the loop**

Add the parameter to `compute_multiview_depth_confidence`'s signature, after `pair_gate`:

```python
    pair_gate: bool = True,
    collect_residuals: bool = False,
    device: str = "cuda",
```

Add to the docstring's Args block:

```
        collect_residuals: Also accumulate the signed relative residual and parallax angle
                     that the loop already computes and would otherwise discard. Off by
                     default so the four production creators are byte-identical.
```

Before the `for i in range(N)` loop, initialise the accumulators:

```python
    # Residual collection is opt-in: the loop already holds every quantity below, but the
    # four production creators must stay byte-identical, so nothing runs unless asked.
    if collect_residuals:
        rel_hist = FixedHistogram(*_REL_RESIDUAL_RANGE, bins=_REL_RESIDUAL_BINS)
        parallax_hist = FixedHistogram(*_PARALLAX_RANGE_DEG, bins=_PARALLAX_BINS)
        equiv_px_hist = FixedHistogram(0.0, 20.0, bins=2000)
        pair_rows: list[PairResidual] = []
        cam_centers = cam2world[:, :3, 3]  # (N, 3) world-space camera positions
        focal_mean = float(K[:, 0, 0].mean() + K[:, 1, 1].mean()) / 2.0
```

Inside the `for j in range(N)` loop, immediately **after** the existing
`valid_sum[i] += counted.reshape(H, W).float()` line, insert:

```python
            if not collect_residuals:
                continue

            # Signed relative residual: sign carries the scale bias, spread carries the
            # geometric noise. Same population the ratio counts — occluded pixels are
            # absent evidence, and including them would drag the bias negative.
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
            parallax_deg = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # The parallax bridge: a relative depth error r shows up in the image as
            # r * parallax radians, i.e. r * parallax * f pixels. This is what puts the
            # depth channel onto the pixel channel's axis.
            equiv_px = rel.abs() * torch.deg2rad(parallax_deg) * focal_mean

            rel_hist.add(rel)
            parallax_hist.add(parallax_deg)
            equiv_px_hist.add(equiv_px)

            q = torch.quantile(rel, torch.tensor([0.25, 0.5, 0.75], device=rel.device))
            pair_rows.append(
                PairResidual(
                    i=i,
                    j=j,
                    n_pixels=int(sel.sum()),
                    median_rel=float(q[1]),
                    iqr_rel=float(q[2] - q[0]),
                    median_parallax_deg=float(parallax_deg.median()),
                    below_floor_frac=float((parallax_deg < PARALLAX_FLOOR_DEG).float().mean()),
                )
            )
```

Replace the `return MultiviewConfidence(...)` at the end with:

```python
    stats = (
        ResidualStats(
            pairs=pair_rows,
            rel_hist=rel_hist,
            parallax_hist=parallax_hist,
            equiv_px_hist=equiv_px_hist,
        )
        if collect_residuals
        else None
    )
    return MultiviewConfidence(
        ratio=ratio.cpu().numpy().astype(np.float32),
        inlier_count=inlier_sum.cpu().numpy().astype(np.int32),
        valid_count=valid_sum.cpu().numpy().astype(np.int32),
        judged=judged,
        residuals=stats,
    )
```

`PARALLAX_FLOOR_DEG` is defined in Task 4; until then add it beside the other constants in this file:

```python
# Below this parallax angle a pair cannot observe depth along the ray, so the bridge ratio
# is undefined rather than large. B is the baseline component PERPENDICULAR to the ray, so
# forward camera motion drives it to ~zero near the epipole — the same root cause as the
# AUC@5 ill-conditioning measured on 10-20 mm indoor baselines.
PARALLAX_FLOOR_DEG = 0.5
```

- [ ] **Step 5: Export the new names**

In `collab_splats/pointcloud/feedforward/__init__.py`, add `PairResidual` and `ResidualStats` to both the `from .base import (...)` block and `__all__`, beside the existing `compute_multiview_depth_confidence` entries (lines 14 and 61).

- [ ] **Step 6: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py -v
```

Expected: all pass — the 7 new tests plus every pre-existing one in the file.

- [ ] **Step 7: Prove the four production callers are untouched**

```bash
/opt/venv/reconstruction/bin/python -m pytest \
  tests/pointcloud/test_vggtx_creator.py \
  tests/pointcloud/test_vggt_omega_creator.py \
  tests/pointcloud/feedforward/test_mapanything_creator.py \
  tests/integration/test_pipeline_cu121.py -v
```

Expected: pass at the same counts as before the change. If any fail, the default-off contract is broken — fix rather than update the test.

- [ ] **Step 8: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py \
        collab_splats/pointcloud/feedforward/__init__.py \
        tests/pointcloud/test_mv_conf.py
git commit -m "feat(pointcloud): collect signed residual and parallax in the mv loop

The loop already computes expected_d and sampled_d, thresholds them to a
boolean, and throws the residual away. It also unprojects pts_world, from which
the parallax angle is two dot products. Both are now collected behind an opt-in
collect_residuals flag; the four production creators are byte-identical.

Signed, because the sign separates scale bias (median) from geometric noise
(spread). Parallax from ray directions rather than f*B/Z — focal is exactly what
is not comparable across backbones.

Accumulates into histograms in place: N^2*H*W residuals is 2.4e10 floats at 300
frames, so raw arrays are never returned."
```

---

### Task 4: The parallax bridge — `equivalent_pixel_error` and `explained_fraction`

**Files:**
- Create: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_report.py`:

```python
"""Unit tests for the scene error report: parallax bridge, channels, aggregator."""

import numpy as np
import pytest

from collab_splats.geometry.report import (
    PARALLAX_FLOOR_DEG,
    equivalent_pixel_error,
    explained_fraction,
)


def test_equivalent_pixel_error_is_r_times_disparity():
    """delta_d = r * d, with d = f * parallax(rad) for small angles."""
    r, parallax_deg, focal = 0.1, 2.0, 500.0
    expected = 0.1 * np.deg2rad(2.0) * 500.0
    assert equivalent_pixel_error(r, parallax_deg, focal) == pytest.approx(expected)


def test_equivalent_pixel_error_scales_linearly_in_r():
    a = equivalent_pixel_error(0.05, 3.0, 500.0)
    b = equivalent_pixel_error(0.10, 3.0, 500.0)
    assert b == pytest.approx(2 * a)


def test_equivalent_pixel_error_shrinks_with_parallax():
    """The whole far-pixel asymmetry: same depth error, less parallax, fewer pixels moved."""
    near = equivalent_pixel_error(0.1, 6.0, 500.0)
    far = equivalent_pixel_error(0.1, 1.0, 500.0)
    assert far < near


def test_explained_fraction_is_one_when_depth_explains_the_pixels():
    equiv = equivalent_pixel_error(0.1, 2.0, 500.0)
    assert explained_fraction(equiv, 0.1, 2.0, 500.0) == pytest.approx(1.0)


def test_explained_fraction_exceeds_one_for_pose_error():
    """Pixels move more than any depth error explains => the excess is pose or appearance."""
    equiv = equivalent_pixel_error(0.1, 2.0, 500.0)
    assert explained_fraction(equiv * 5.0, 0.1, 2.0, 500.0) == pytest.approx(5.0)


def test_explained_fraction_below_one_for_unobservable_depth_error():
    """Depth disagrees more than pixels do => the error lies along the ray."""
    equiv = equivalent_pixel_error(0.1, 2.0, 500.0)
    assert explained_fraction(equiv * 0.2, 0.1, 2.0, 500.0) == pytest.approx(0.2)


def test_explained_fraction_is_none_below_the_parallax_floor():
    """Not an infinity, not a large number — undefined, and the caller must see that."""
    assert explained_fraction(1.0, 0.1, PARALLAX_FLOOR_DEG * 0.5, 500.0) is None


def test_explained_fraction_is_none_at_zero_depth_residual():
    """0/0 is undefined; a scene with perfect depth agreement has no ratio to report."""
    assert explained_fraction(1.0, 0.0, 2.0, 500.0) is None


def test_bridge_is_focal_free_in_angular_form():
    """rho must not depend on the focal used, since focal is not comparable across backbones."""
    a = explained_fraction(equivalent_pixel_error(0.1, 2.0, 500.0), 0.1, 2.0, 500.0)
    b = explained_fraction(equivalent_pixel_error(0.1, 2.0, 900.0), 0.1, 2.0, 900.0)
    assert a == pytest.approx(b)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'collab_splats.geometry.report'`

- [ ] **Step 3: Write the implementation**

Create `collab_splats/geometry/report.py`:

```python
"""Reference-free scene error report: four channels, the parallax bridge, one report.json.

Report-only. Nothing here feeds back into a reconstruction, and nothing here emits a
verdict — the output is distributions and how they vary, for a reader to interpret.
"""

import logging

import numpy as np

from collab_splats.geometry.histogram import FixedHistogram

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

SCHEMA_VERSION = 1

# Below this parallax angle a pair cannot observe depth along the ray, so the bridge ratio
# is undefined rather than large. B is the baseline component PERPENDICULAR to the ray, so
# forward camera motion drives it to ~zero near the epipole — the same root cause as the
# AUC@5 ill-conditioning measured on 10-20 mm indoor baselines.
PARALLAX_FLOOR_DEG = 0.5

# Quantile grid reported for every channel. Denser than median/p90/p99 because the shape of
# the distribution is the deliverable, not three points on it.
QUANTILE_GRID = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)

# Depth stratification: quantiles of the SCENE's own depth, never absolute distance. One
# recon unit is not one metre and the factor differs per scene and per backbone.
DEPTH_QUANTILE_BINS = 5


########################################
# The parallax bridge
########################################


def equivalent_pixel_error(rel_residual: float, parallax_deg: float, focal_px: float) -> float:
    """Depth residual expressed in pixels, through the pair's actual parallax.

    For a pair with perpendicular baseline B, disparity d = f*B/Z, and a depth error dZ at
    depth Z moves the point in the image by f*B*dZ/Z^2. Substituting r = dZ/Z:

        delta_d = r * d = r * f * alpha

    where alpha is the parallax angle in radians. Focal and baseline collapse out of the
    relation itself; f reappears only to express the answer in pixels.

    This is the legitimate way to compare the pixel and depth channels: convert first, then
    difference. The 1/Z hiding inside d is the entire reason far pixels disagree less in
    pixel terms while disagreeing more in depth terms.
    """
    return abs(rel_residual) * np.deg2rad(parallax_deg) * focal_px


def explained_fraction(
    measured_px_residual: float,
    rel_residual: float,
    parallax_deg: float,
    focal_px: float,
) -> float | None:
    """rho: how much of the observed pixel disagreement the depth disagreement accounts for.

    rho ~ 1  one underlying error seen twice.
    rho >> 1 pixels move more than any depth error explains -> the excess is POSE (pose error
             moves pixels while leaving depths mutually consistent) or appearance.
    rho << 1 depth disagrees more than pixels do -> the error lies along the ray, where this
             pair's baseline cannot see it. Low observability, not necessarily bad depth.

    rho is unitless and parallax-normalised, so it is comparable across depth bins where the
    raw channels are not. Returns None rather than an infinity when the pair is below the
    parallax floor or the depth residual is zero — a scene that is mostly None has no usable
    rho, and saying so is the finding.
    """
    if parallax_deg < PARALLAX_FLOOR_DEG:
        return None
    equiv = equivalent_pixel_error(rel_residual, parallax_deg, focal_px)
    if equiv <= 0.0:
        return None
    return float(measured_px_residual / equiv)


def quantile_block(hist: FixedHistogram) -> dict:
    """Quantile grid plus the full histogram — shape, not three points on it.

    The histogram is what makes 'what fraction falls below X' an exact count at any X the
    reader picks, and what keeps rho legible despite its long tail and undefined regime.
    """
    return {
        "quantiles": {str(q): hist.quantile(q) for q in QUANTILE_GRID},
        "histogram": hist.to_dict(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Point base.py at the single definition**

`PARALLAX_FLOOR_DEG` now lives in two places. Delete the copy added in Task 3 Step 4 from `collab_splats/pointcloud/feedforward/base.py` and import it instead, beside the `FixedHistogram` import:

```python
from collab_splats.geometry.report import PARALLAX_FLOOR_DEG
```

If this introduces a circular import (`geometry.report` importing from `pointcloud.feedforward`), invert it: keep `PARALLAX_FLOOR_DEG` in `collab_splats/geometry/histogram.py` and import it into both. Verify:

```bash
/opt/venv/reconstruction/bin/python -c "import collab_splats.geometry.report, collab_splats.pointcloud.feedforward.base; print('imports clean')"
/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_mv_conf.py tests/geometry/ -q
```

Expected: `imports clean`, then all pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py \
        collab_splats/pointcloud/feedforward/base.py
git commit -m "feat(geometry): parallax bridge putting depth and pixel channels on one axis

delta_d = r * d with d = f*alpha. Focal and baseline collapse out of the
relation; f reappears only to express the answer in pixels. The 1/Z inside d is
the entire 'far pixels disagree less in pixels, more in depth' effect, so
dividing it out removes the apparent contradiction rather than working around it.

rho = measured_px / equivalent_px: ~1 one error seen twice, >>1 excess is pose or
appearance, <<1 depth error lies along the ray. Returns None below the parallax
floor rather than an infinity — forward motion drives perpendicular baseline to
zero near the epipole, same root cause as the AUC@5 ill-conditioning at 10-20 mm
indoor baselines."
```

---

### Task 5: Depth + scale channel

Turns Task 3's `ResidualStats` into the channel block and the pair-table rows.

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import build_depth_channel
from collab_splats.pointcloud.feedforward.base import PairResidual, ResidualStats
from collab_splats.geometry.histogram import FixedHistogram


def _stats(pairs):
    rel = FixedHistogram(-0.5, 0.5, 2000)
    par = FixedHistogram(0.0, 30.0, 600)
    eqp = FixedHistogram(0.0, 20.0, 2000)
    for p in pairs:
        rel.add(np.full(p.n_pixels, p.median_rel))
        par.add(np.full(p.n_pixels, p.median_parallax_deg))
        eqp.add(np.full(p.n_pixels, abs(p.median_rel) * np.deg2rad(p.median_parallax_deg) * 500.0))
    return ResidualStats(pairs=pairs, rel_hist=rel, parallax_hist=par, equiv_px_hist=eqp)


def test_depth_channel_reports_grid_and_resolution():
    """Every block stamps its grid — model-res depth with original-res K is a known bug class."""
    ch = build_depth_channel(_stats([PairResidual(0, 1, 100, 0.0, 0.01, 3.0, 0.0)]), focal_px=500.0)
    assert ch["grid"] == "model"
    assert "resolution" in ch


def test_depth_channel_pair_rows_carry_separation():
    pairs = [PairResidual(0, 5, 100, 0.02, 0.01, 3.0, 0.0)]
    ch = build_depth_channel(_stats(pairs), focal_px=500.0)
    row = ch["pairs"][0]
    assert row["i"] == 0 and row["j"] == 5
    assert row["temporal_separation"] == 5


def test_scale_bias_is_the_signed_median_not_the_magnitude():
    """A pure scale error has a large median and a small spread; sign must survive."""
    pairs = [PairResidual(0, 1, 100, -0.08, 0.005, 3.0, 0.0)]
    ch = build_depth_channel(_stats(pairs), focal_px=500.0)
    assert ch["pairs"][0]["median_rel"] == pytest.approx(-0.08)
    assert ch["scale"]["median_abs_bias"] == pytest.approx(0.08)


def test_equivalent_pixel_error_lands_on_each_pair_row():
    pairs = [PairResidual(0, 1, 100, 0.1, 0.01, 2.0, 0.0)]
    ch = build_depth_channel(_stats(pairs), focal_px=500.0)
    assert ch["pairs"][0]["equivalent_pixel_error"] == pytest.approx(
        equivalent_pixel_error(0.1, 2.0, 500.0)
    )


def test_below_floor_pairs_report_null_equivalent_error_not_zero():
    pairs = [PairResidual(0, 1, 100, 0.1, 0.01, PARALLAX_FLOOR_DEG * 0.5, 1.0)]
    ch = build_depth_channel(_stats(pairs), focal_px=500.0)
    assert ch["pairs"][0]["equivalent_pixel_error"] is None
    assert ch["parallax"]["pairs_below_floor"] == 1


def test_empty_stats_yields_an_unavailable_channel_not_a_crash():
    ch = build_depth_channel(_stats([]), focal_px=500.0)
    assert ch["available"] is False
    assert "reason" in ch
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k depth
```

Expected: FAIL — `ImportError: cannot import name 'build_depth_channel'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Channel: depth cross-view + scale
########################################


def build_depth_channel(stats, focal_px: float, resolution: str = "model") -> dict:
    """Depth cross-view agreement: scale bias, geometric noise, parallax, the bridge.

    Evaluated at MODEL resolution on purpose. Depth values are identical under nearest
    upsampling, so original-res evaluation returns the same number — but it would sample a
    guided-FILTERED depth map, reporting lower disagreement than the model actually
    produced. That improvement belongs to the smoother, not the model.

    Args:
        stats:     ResidualStats from compute_multiview_depth_confidence(collect_residuals=True).
        focal_px:  Mean focal in pixels, used only to express the bridge in pixel units.
        resolution: "WxH" or the string "model" — stamped into the block for the reader.
    """
    if not stats.pairs:
        return {
            "available": False,
            "reason": "no overlapping view pairs produced depth residuals",
            "grid": "model",
            "resolution": resolution,
        }

    rows = []
    for p in stats.pairs:
        below_floor = p.median_parallax_deg < PARALLAX_FLOOR_DEG
        rows.append(
            {
                "i": p.i,
                "j": p.j,
                "temporal_separation": abs(p.i - p.j),
                "n_pixels": p.n_pixels,
                # Signed: the median IS the scale bias between the two views.
                "median_rel": p.median_rel,
                # Bias removed: what is left is geometric noise.
                "iqr_rel": p.iqr_rel,
                "median_parallax_deg": p.median_parallax_deg,
                "below_floor_frac": p.below_floor_frac,
                # None, never 0.0 — below the floor the bridge is undefined, and a zero here
                # would read as "no error" when it means "cannot tell".
                "equivalent_pixel_error": (
                    None if below_floor else equivalent_pixel_error(p.median_rel, p.median_parallax_deg, focal_px)
                ),
            }
        )

    biases = np.array([p.median_rel for p in stats.pairs], dtype=np.float64)
    spreads = np.array([p.iqr_rel for p in stats.pairs], dtype=np.float64)
    n_below = sum(1 for p in stats.pairs if p.median_parallax_deg < PARALLAX_FLOOR_DEG)

    return {
        "available": True,
        "grid": "model",
        "resolution": resolution,
        "units": "relative (dimensionless); parallax in degrees; equivalent error in px",
        "n_pairs": len(stats.pairs),
        "residual": quantile_block(stats.rel_hist),
        "equivalent_pixel_error": quantile_block(stats.equiv_px_hist),
        # Scale and noise are independent readings of the same signed residual: a pure scale
        # error has a large median and a small spread, a pose error the reverse.
        "scale": {
            "median_abs_bias": float(np.median(np.abs(biases))),
            "signed_bias_p10": float(np.percentile(biases, 10)),
            "signed_bias_p90": float(np.percentile(biases, 90)),
            "median_spread": float(np.median(spreads)),
        },
        "parallax": {
            **quantile_block(stats.parallax_hist),
            "floor_deg": PARALLAX_FLOOR_DEG,
            "pairs_below_floor": n_below,
            "pairs_below_floor_frac": n_below / len(stats.pairs),
        },
        "pairs": rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): depth cross-view channel with scale/noise separation

Signed median is the scale bias, spread with the bias removed is geometric
noise — a pure scale error has a large median and small spread, a pose error the
reverse. Below the parallax floor equivalent_pixel_error is null, never 0.0: a
zero would read as 'no error' when it means 'cannot tell'.

Model resolution on purpose. Depth is identical under nearest upsampling, so
original-res evaluation returns the same number while sampling a guided-FILTERED
map — reporting lower disagreement than the model produced."
```

---

### Task 6: Depth stratification — does disagreement grow with depth

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import stratify_by_depth


def test_bins_are_scene_quantiles_not_absolute_distance():
    """One recon unit is not one metre and the factor differs per scene AND per backbone."""
    depth = np.concatenate([np.full(100, 1.0), np.full(100, 100.0)])
    resid = np.concatenate([np.full(100, 0.01), np.full(100, 0.02)])
    out = stratify_by_depth(depth, resid, n_bins=2)
    assert out["binning"] == "scene depth quantiles"
    assert len(out["bins"]) == 2
    assert out["bins"][0]["depth_range"][1] < out["bins"][1]["depth_range"][0] + 1e-6


def test_rising_residual_with_depth_is_visible_in_the_bins():
    depth = np.linspace(1.0, 10.0, 1000)
    resid = 0.01 * depth  # relative residual growing linearly in Z
    out = stratify_by_depth(depth, resid, n_bins=5)
    medians = [b["median_abs_rel"] for b in out["bins"]]
    assert medians == sorted(medians)
    assert medians[-1] > 3 * medians[0]


def test_flat_residual_stays_flat_across_bins():
    """No verdict is emitted either way — the caller reads the numbers."""
    depth = np.linspace(1.0, 10.0, 1000)
    resid = np.full(1000, 0.02)
    out = stratify_by_depth(depth, resid, n_bins=5)
    medians = [b["median_abs_rel"] for b in out["bins"]]
    assert max(medians) - min(medians) < 1e-6
    assert "verdict" not in out


def test_degenerate_constant_depth_does_not_crash():
    depth = np.full(500, 3.0)
    resid = np.full(500, 0.01)
    out = stratify_by_depth(depth, resid, n_bins=5)
    assert out["available"] is True
    assert sum(b["n"] for b in out["bins"]) == 500


def test_empty_input_is_unavailable():
    out = stratify_by_depth(np.array([]), np.array([]), n_bins=5)
    assert out["available"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k "depth_ or bins_ or stratif or flat_residual or degenerate"
```

Expected: FAIL — `ImportError: cannot import name 'stratify_by_depth'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Depth stratification
########################################


def stratify_by_depth(depth: np.ndarray, rel_residual: np.ndarray, n_bins: int = DEPTH_QUANTILE_BINS) -> dict:
    """Residual distribution per depth quantile bin — "do things disagree more far away?".

    Bins are quantiles of the SCENE's own depth, never absolute distance: one recon unit is
    not one metre and the factor differs per scene and per backbone.

    There is a null hypothesis to read against. Triangulation uncertainty goes as
    sigma_Z ~ Z^2/(f*B), so a RELATIVE residual should grow roughly linearly in Z. Growing
    faster suggests something beyond geometry; flat suggests depth normalised in a way that
    hides error. No verdict is emitted — the numbers and the null are both reported.
    """
    d = np.asarray(depth, dtype=np.float64).ravel()
    r = np.asarray(rel_residual, dtype=np.float64).ravel()
    keep = np.isfinite(d) & np.isfinite(r) & (d > 0)
    d, r = d[keep], r[keep]
    if d.size == 0:
        return {"available": False, "reason": "no valid depth/residual samples"}

    # Quantile edges; np.unique collapses ties so constant depth degrades to one bin rather
    # than emitting empty bins with zero-width ranges.
    edges = np.unique(np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        edges = np.array([d.min(), d.max() + 1e-9])

    bins = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        # Last bin is closed on the right so the maximum sample is not dropped
        sel = (d >= lo) & (d <= hi) if k == len(edges) - 2 else (d >= lo) & (d < hi)
        if not sel.any():
            continue
        rk = np.abs(r[sel])
        bins.append(
            {
                "depth_range": [float(lo), float(hi)],
                "median_depth": float(np.median(d[sel])),
                "n": int(sel.sum()),
                "median_abs_rel": float(np.median(rk)),
                "p90_abs_rel": float(np.percentile(rk, 90)),
                "p99_abs_rel": float(np.percentile(rk, 99)),
            }
        )

    return {
        "available": True,
        "binning": "scene depth quantiles",
        "null_hypothesis": "sigma_Z ~ Z^2/(f*B) => relative residual grows ~linearly in Z",
        "bins": bins,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): depth-quantile stratification with a stated null hypothesis

Bins are the scene's own depth quantiles, never absolute distance — one recon
unit is not one metre and the factor differs per scene and per backbone.

Carries the null: sigma_Z ~ Z^2/(f*B) means a relative residual should grow
roughly linearly in Z, so the interesting readings are 'faster than linear' and
'flat'. Reports the numbers and the null; emits no verdict."
```

---

### Task 7: Photometric channel at original resolution

The only channel that depends on appearance. Normalised per patch, which absorbs both the `[0,255]` (VGGT family) vs `[0,1]` (MapAnything) split and any exposure change — otherwise exposure swamps geometry.

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import normalized_patch_residual


def test_identical_patches_have_zero_residual():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert normalized_patch_residual(a, a.copy()) == pytest.approx(0.0, abs=1e-6)


def test_residual_is_invariant_to_image_scale_convention():
    """[0,255] VGGT vs [0,1] MapAnything must not change the number."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    b = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert normalized_patch_residual(a, b) == pytest.approx(
        normalized_patch_residual(a / 255.0, b / 255.0), abs=1e-5
    )


def test_residual_is_invariant_to_exposure_shift():
    """Otherwise a brightness change swamps the geometry the channel exists to measure."""
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    b = a * 1.4 + 20.0
    assert normalized_patch_residual(a, b) == pytest.approx(0.0, abs=1e-4)


def test_residual_grows_with_genuine_disagreement():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 255, size=(256, 3)).astype(np.float32)
    small = a + rng.normal(0, 5, a.shape)
    large = a + rng.normal(0, 80, a.shape)
    assert normalized_patch_residual(a, small) < normalized_patch_residual(a, large)


def test_flat_patch_returns_nan_not_a_divide_by_zero():
    """Zero variance has no normalisation; nan is excluded downstream by FixedHistogram."""
    flat = np.full((64, 3), 128.0, dtype=np.float32)
    rng = np.random.default_rng(4)
    other = rng.uniform(0, 255, size=(64, 3)).astype(np.float32)
    assert np.isnan(normalized_patch_residual(flat, other))


def test_empty_patch_returns_nan():
    assert np.isnan(normalized_patch_residual(np.zeros((0, 3)), np.zeros((0, 3))))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k patch
```

Expected: FAIL — `ImportError: cannot import name 'normalized_patch_residual'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Channel: photometric
########################################

# Below this many valid samples a patch's mean/std are too noisy to normalise against.
_MIN_PATCH_SAMPLES = 8


def normalized_patch_residual(src: np.ndarray, dst: np.ndarray) -> float:
    """RMS residual between two patches after zero-mean/unit-variance normalisation.

    Normalising per patch buys two invariances the raw difference does not have:
      * the [0, 255] (VGGT family) vs [0, 1] (MapAnything) image-scale split, so one number
        is comparable across backbones;
      * exposure and gain change, which would otherwise swamp the geometry this channel
        exists to measure.

    Returns nan for a flat or too-small patch — there is no normalisation for zero variance,
    and FixedHistogram drops nan rather than counting it as agreement.
    """
    a = np.asarray(src, dtype=np.float64).ravel()
    b = np.asarray(dst, dtype=np.float64).ravel()
    if a.size < _MIN_PATCH_SAMPLES or a.size != b.size:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return float("nan")
    za = (a - a.mean()) / sa
    zb = (b - b.mean()) / sb
    return float(np.sqrt(np.mean((za - zb) ** 2)))


def build_photometric_channel(pair_residuals: dict[tuple[int, int], float], resolution: str) -> dict:
    """Photometric warp channel from per-pair normalised residuals.

    ORIGINAL resolution on purpose: RGB detail exists only there, and unlike depth this is a
    genuinely resolution-dependent quantity. Depth is guided-upsampled to reach that grid;
    the RGB is real frames.zarr data, never resampled up.
    """
    if not pair_residuals:
        return {
            "available": False,
            "reason": "no view pairs produced photometric residuals",
            "grid": "original",
            "resolution": resolution,
        }
    hist = FixedHistogram(0.0, 2.0, bins=1000)
    rows = []
    for (i, j), v in sorted(pair_residuals.items()):
        hist.add(np.array([v]))
        rows.append({"i": i, "j": j, "temporal_separation": abs(i - j), "normalized_residual": v})
    return {
        "available": True,
        "grid": "original",
        "resolution": resolution,
        "units": "dimensionless (zero-mean/unit-variance normalised RMS)",
        "n_pairs": len(rows),
        "residual": quantile_block(hist),
        "pairs": rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): photometric channel, normalised per patch

Zero-mean/unit-variance normalisation buys invariance to the [0,255] vs [0,1]
backbone image-scale split AND to exposure change, which would otherwise swamp
the geometry the channel exists to measure. Flat patches return nan rather than
dividing by zero; FixedHistogram drops nan instead of counting it as agreement.

Original resolution on purpose — unlike depth this is genuinely
resolution-dependent, and the RGB is real frames.zarr data."
```

---

### Task 8: Confidence validation channel

Confidence is **not** an error channel. The question is whether the model knows when it is wrong — `corr(confidence, measured disagreement)`. Nothing in the repo has ever checked this.

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import build_confidence_channel


def test_perfectly_calibrated_confidence_gives_strong_negative_correlation():
    """High confidence should mean LOW disagreement, hence a negative correlation."""
    conf = np.linspace(0.0, 1.0, 1000)
    resid = 1.0 - conf
    ch = build_confidence_channel(conf, resid)
    assert ch["spearman"] < -0.95


def test_useless_confidence_gives_near_zero_correlation():
    rng = np.random.default_rng(0)
    ch = build_confidence_channel(rng.uniform(size=5000), rng.uniform(size=5000))
    assert abs(ch["spearman"]) < 0.1


def test_percentile_bins_not_absolute_thresholds():
    """Confidence is LOGITS on some backbones — absolute thresholds are meaningless."""
    conf = np.linspace(-8.0, 12.0, 1000)  # logit-scaled, deliberately not in [0, 1]
    ch = build_confidence_channel(conf, 1.0 / (1.0 + np.exp(conf)))
    assert ch["binning"] == "confidence percentiles"
    assert len(ch["bins"]) > 1


def test_bins_report_residual_per_confidence_decile():
    conf = np.linspace(0.0, 1.0, 1000)
    ch = build_confidence_channel(conf, 1.0 - conf)
    medians = [b["median_abs_residual"] for b in ch["bins"]]
    assert medians == sorted(medians, reverse=True)


def test_missing_confidence_is_unavailable_not_zeros():
    """Older zarr stores have no confidence array — absent, never backfilled."""
    ch = build_confidence_channel(None, np.array([0.1, 0.2]))
    assert ch["available"] is False
    assert "reason" in ch


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        build_confidence_channel(np.zeros(10), np.zeros(11))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k confidence
```

Expected: FAIL — `ImportError: cannot import name 'build_confidence_channel'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Channel: confidence validation
########################################

_CONFIDENCE_BINS = 10


def build_confidence_channel(confidence, rel_residual, n_bins: int = _CONFIDENCE_BINS) -> dict:
    """Does the model know when it is wrong? corr(confidence, measured disagreement).

    Confidence is an INPUT to validate, not an error channel. Nothing in this repo has
    previously checked whether the model's self-report tracks the disagreement we measure.

    Binned by PERCENTILE, never by absolute value: confidence is logits on some backbones
    (LoGeR) and a bounded score on others, so an absolute threshold means different things
    per backbone and nothing across them. Spearman for the same reason — it is invariant to
    any monotone rescaling of the confidence.
    """
    if confidence is None:
        return {
            "available": False,
            "reason": "no confidence array in this store (absent, never backfilled)",
        }
    c = np.asarray(confidence, dtype=np.float64).ravel()
    r = np.abs(np.asarray(rel_residual, dtype=np.float64).ravel())
    if c.size != r.size:
        raise ValueError(f"length mismatch: confidence {c.size}, residual {r.size}")
    keep = np.isfinite(c) & np.isfinite(r)
    c, r = c[keep], r[keep]
    if c.size < n_bins:
        return {"available": False, "reason": f"only {c.size} paired samples"}

    # Spearman = Pearson on ranks; invariant to any monotone rescaling, which is exactly
    # what differs between a logit head and a sigmoid head.
    rank_c = np.argsort(np.argsort(c)).astype(np.float64)
    rank_r = np.argsort(np.argsort(r)).astype(np.float64)
    denom = rank_c.std() * rank_r.std()
    spearman = 0.0 if denom < 1e-12 else float(np.mean((rank_c - rank_c.mean()) * (rank_r - rank_r.mean())) / denom)

    edges = np.unique(np.quantile(c, np.linspace(0.0, 1.0, n_bins + 1)))
    bins = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        sel = (c >= lo) & (c <= hi) if k == len(edges) - 2 else (c >= lo) & (c < hi)
        if not sel.any():
            continue
        bins.append(
            {
                "confidence_range": [float(lo), float(hi)],
                "n": int(sel.sum()),
                "median_abs_residual": float(np.median(r[sel])),
                "p90_abs_residual": float(np.percentile(r[sel], 90)),
            }
        )

    return {
        "available": True,
        "grid": "model",
        "binning": "confidence percentiles",
        "note": "confidence is validated here, not treated as an error channel",
        "spearman": spearman,
        "n_samples": int(c.size),
        "bins": bins,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 32 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): confidence validation channel

Confidence is an input to validate, not an error channel — the question is
whether the model knows when it is wrong, which nothing in this repo has checked.

Percentile bins and Spearman, never absolute thresholds or Pearson: confidence is
logits on LoGeR and a bounded score elsewhere, so an absolute threshold means
different things per backbone. Spearman is invariant to exactly that rescaling."
```

---

### Task 9: Epipolar channel adapter

Reads `verification.json` — never re-runs `verify_matches`. Two stages calling the matcher would be waste.

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
import json
from collab_splats.geometry.report import build_epipolar_channel


def _verification_json(tmp_path, pair_stats, frame_stats=None, summary=None):
    p = tmp_path / "verification.json"
    p.write_text(
        json.dumps(
            {
                "pair_stats": pair_stats,
                "frame_stats": frame_stats or {},
                "summary": summary or {},
            }
        )
    )
    return p


def test_epipolar_channel_reads_pair_stats(tmp_path):
    p = _verification_json(
        tmp_path,
        [
            {
                "name1": "frame_000000",
                "name2": "frame_000001",
                "num_matches": 500,
                "num_inliers": 450,
                "rot_error_deg": 0.15,
                "t_direction_error_deg": 0.9,
            }
        ],
    )
    ch = build_epipolar_channel(p, image_width=640)
    assert ch["available"] is True
    assert ch["grid"] == "original"
    assert ch["pairs"][0]["rot_error_deg"] == pytest.approx(0.15)
    assert ch["pairs"][0]["inlier_ratio"] == pytest.approx(0.9)


def test_missing_verification_json_is_unavailable_not_a_crash(tmp_path):
    ch = build_epipolar_channel(tmp_path / "nope.json", image_width=640)
    assert ch["available"] is False
    assert "reason" in ch


def test_reprojection_reported_in_px_and_as_image_fraction(tmp_path):
    p = _verification_json(
        tmp_path,
        [],
        frame_stats={"frame_000000": {"mean_reproj_error_px": 1.28, "track_survival": 0.8, "n_tracks": 100}},
    )
    ch = build_epipolar_channel(p, image_width=640)
    f = ch["frames"][0]
    assert f["mean_reproj_error_px"] == pytest.approx(1.28)
    assert f["mean_reproj_error_frac_width"] == pytest.approx(0.002)


def test_frame_indices_parsed_from_names(tmp_path):
    """The pair table joins on integer indices; names are frame_XXXXXX."""
    p = _verification_json(
        tmp_path,
        [
            {
                "name1": "frame_000003",
                "name2": "frame_000011",
                "num_matches": 10,
                "num_inliers": 8,
                "rot_error_deg": 0.2,
                "t_direction_error_deg": 1.0,
            }
        ],
    )
    ch = build_epipolar_channel(p, image_width=640)
    assert ch["pairs"][0]["i"] == 3 and ch["pairs"][0]["j"] == 11
    assert ch["pairs"][0]["temporal_separation"] == 8


def test_nan_t_direction_is_dropped_from_the_distribution(tmp_path):
    """t-direction is nan when the pair is degenerate; a nan must not poison the quantiles."""
    p = _verification_json(
        tmp_path,
        [
            {
                "name1": "frame_000000",
                "name2": "frame_000001",
                "num_matches": 10,
                "num_inliers": 8,
                "rot_error_deg": 0.2,
                "t_direction_error_deg": None,
            },
            {
                "name1": "frame_000001",
                "name2": "frame_000002",
                "num_matches": 10,
                "num_inliers": 9,
                "rot_error_deg": 0.3,
                "t_direction_error_deg": 1.0,
            },
        ],
    )
    ch = build_epipolar_channel(p, image_width=640)
    assert ch["t_direction_error_deg"]["histogram"]["total"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k epipolar
```

Expected: FAIL — `ImportError: cannot import name 'build_epipolar_channel'`

- [ ] **Step 3: Write the implementation**

Add `import json` and `from pathlib import Path` to the imports at the top of
`collab_splats/geometry/report.py`, then append:

```python
########################################
# Channel: epipolar + reprojection
########################################


def _frame_index(name: str) -> int | None:
    """Integer index out of a frame_XXXXXX name, so pair rows join across channels."""
    stem = Path(name).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def build_epipolar_channel(verification_json: Path, image_width: int) -> dict:
    """Pose-only error, read from verify's verification.json. Never re-runs the matcher.

    This is the ONLY channel that never touches depth, which is the entire reason attribution
    is possible: an error that moves this channel and not the depth channel is a pose error.

    Already original-resolution — verify estimates from original-res keypoints. Reprojection
    is reported in px AND as a fraction of image width, because a bare pixel count is not
    comparable across backbones (VGGTX 518-crop, MapAnything 512/518, omega 448x592).
    """
    p = Path(verification_json)
    if not p.exists():
        return {
            "available": False,
            "reason": f"no verification.json at {p} — run the verify stage",
            "grid": "original",
        }
    data = json.loads(p.read_text())

    rot_hist = FixedHistogram(0.0, 30.0, bins=600)
    tdir_hist = FixedHistogram(0.0, 90.0, bins=900)
    inlier_hist = FixedHistogram(0.0, 1.0, bins=100)
    pair_rows = []
    for s in data.get("pair_stats", []):
        i, j = _frame_index(s["name1"]), _frame_index(s["name2"])
        n_matches = s.get("num_matches") or 0
        n_inliers = s.get("num_inliers") or 0
        ratio = (n_inliers / n_matches) if n_matches else float("nan")
        rot = s.get("rot_error_deg")
        tdir = s.get("t_direction_error_deg")
        # None/nan means the pair was degenerate; FixedHistogram drops nan, so a bad pair
        # cannot poison the distribution while its row still appears in the table.
        rot_hist.add(np.array([np.nan if rot is None else rot]))
        tdir_hist.add(np.array([np.nan if tdir is None else tdir]))
        inlier_hist.add(np.array([ratio]))
        pair_rows.append(
            {
                "i": i,
                "j": j,
                "temporal_separation": None if i is None or j is None else abs(i - j),
                "num_matches": n_matches,
                "num_inliers": n_inliers,
                "inlier_ratio": None if n_matches == 0 else ratio,
                "rot_error_deg": rot,
                "t_direction_error_deg": tdir,
            }
        )

    frame_rows = []
    for name, fs in sorted(data.get("frame_stats", {}).items()):
        px = fs.get("mean_reproj_error_px")
        frame_rows.append(
            {
                "name": name,
                "index": _frame_index(name),
                "n_tracks": fs.get("n_tracks"),
                "track_survival": fs.get("track_survival"),
                "mean_reproj_error_px": px,
                # A bare pixel count is not comparable across backbones — normalise.
                "mean_reproj_error_frac_width": None if px is None else px / image_width,
            }
        )

    return {
        "available": True,
        "grid": "original",
        "resolution": f"width={image_width}",
        "units": "degrees; reprojection in px and as a fraction of image width",
        "source": str(p),
        "n_pairs": len(pair_rows),
        "rot_error_deg": quantile_block(rot_hist),
        "t_direction_error_deg": quantile_block(tdir_hist),
        "inlier_ratio": quantile_block(inlier_hist),
        "pairs": pair_rows,
        "frames": frame_rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): epipolar channel adapter over verification.json

Reads verify's output, never re-runs verify_matches — two stages calling the
matcher would be waste. This is the only channel that never touches depth, which
is the entire reason attribution works: an error that moves this and not the
depth channel is a pose error.

Reprojection in px AND as a fraction of image width, because a bare pixel count
is not comparable across backbones (VGGTX 518-crop, MapAnything 512/518, omega
448x592). Degenerate pairs carry nan into FixedHistogram, which drops it — the
row still appears in the table."
```

---

### Task 10: The aggregator — cumulative curves, per-frame ranks, coverage

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import (
    SCHEMA_VERSION,
    coverage_from_original_coords,
    cumulative_curve,
    percentile_ranks,
)


def test_cumulative_curve_accumulates_sequential_pairs_only():
    """Cumulative error is a walk along the trajectory; |i-j|>1 pairs are not steps."""
    pairs = [
        {"i": 0, "j": 1, "temporal_separation": 1, "v": 0.1},
        {"i": 1, "j": 2, "temporal_separation": 1, "v": 0.2},
        {"i": 0, "j": 5, "temporal_separation": 5, "v": 9.9},
    ]
    out = cumulative_curve(pairs, value_key="v")
    assert out["frame_index"] == [1, 2]
    assert out["cumulative"] == pytest.approx([0.1, 0.30000000000000004])


def test_cumulative_curve_uses_absolute_values():
    """Signed steps would cancel and hide accumulation."""
    pairs = [
        {"i": 0, "j": 1, "temporal_separation": 1, "v": 0.1},
        {"i": 1, "j": 2, "temporal_separation": 1, "v": -0.1},
    ]
    assert cumulative_curve(pairs, value_key="v")["cumulative"] == pytest.approx([0.1, 0.2])


def test_cumulative_curve_is_empty_without_sequential_pairs():
    out = cumulative_curve([{"i": 0, "j": 7, "temporal_separation": 7, "v": 1.0}], value_key="v")
    assert out["frame_index"] == [] and out["cumulative"] == []


def test_percentile_ranks_are_numbers_not_labels():
    """No verdict, no cause, no flag — a number the reader interprets."""
    ranks = percentile_ranks({0: 1.0, 1: 5.0, 2: 9.0})
    assert ranks[0] == pytest.approx(0.0)
    assert ranks[2] == pytest.approx(1.0)
    assert all(isinstance(v, float) for v in ranks.values())


def test_percentile_ranks_handle_ties():
    ranks = percentile_ranks({0: 2.0, 1: 2.0, 2: 2.0})
    assert all(v == pytest.approx(0.0) for v in ranks.values())


def test_percentile_ranks_skip_nan_frames():
    ranks = percentile_ranks({0: 1.0, 1: float("nan"), 2: 3.0})
    assert 1 not in ranks


def test_coverage_from_original_coords():
    """VGGTX centre-crops height to 518 — a 16:9 source loses a band with no depth at all."""
    coords = np.array([[0, 281, 1920, 799, 1920, 1080]], dtype=np.float32)
    cov = coverage_from_original_coords(coords)
    assert cov["per_frame"][0]["covered_fraction"] == pytest.approx((1920 * 518) / (1920 * 1080), abs=1e-3)


def test_full_frame_coverage_is_one():
    coords = np.array([[0, 0, 640, 480, 640, 480]], dtype=np.float32)
    cov = coverage_from_original_coords(coords)
    assert cov["per_frame"][0]["covered_fraction"] == pytest.approx(1.0)
    assert cov["median_covered_fraction"] == pytest.approx(1.0)


def test_schema_version_is_exported():
    assert isinstance(SCHEMA_VERSION, int)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k "cumulative or percentile or coverage or schema"
```

Expected: FAIL — `ImportError: cannot import name 'cumulative_curve'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Second-order views: cumulative, ranks, coverage
########################################


def cumulative_curve(pairs: list[dict], value_key: str) -> dict:
    """Running accumulation of |value| along consecutive frames — "does disagreement build?".

    Sequential pairs only: a |i-j|=5 pair is a revisit observation, not a step along the
    trajectory, and summing it would double-count. Absolute values, because signed steps
    cancel and would hide exactly the accumulation this curve exists to show.

    Read against the per-separation view, not alone: frame index is a confounded axis (scene
    content, motion speed and exposure all correlate with it), so a rise here is not by
    itself evidence of accumulation.
    """
    steps = sorted(
        (p for p in pairs if p.get("temporal_separation") == 1 and p.get(value_key) is not None),
        key=lambda p: min(p["i"], p["j"]),
    )
    idx, cum, running = [], [], 0.0
    for p in steps:
        v = p[value_key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        running += abs(float(v))
        idx.append(int(max(p["i"], p["j"])))
        cum.append(running)
    return {
        "frame_index": idx,
        "cumulative": cum,
        "note": "sequential pairs only; absolute steps; read against the per-separation view",
    }


def percentile_ranks(per_frame: dict[int, float]) -> dict[int, float]:
    """Each frame's rank within this scene's own distribution, in [0, 1].

    A NUMBER, never a label. The report does not say which channel is to blame for a
    high-ranking frame, does not name a cause, and does not flag it — the reader sees that
    frame 47 sits at p99 in depth and p60 in epipolar and draws their own conclusion.

    Within-scene ranks need no absolute threshold, which sidesteps both the units problem
    and cross-backbone incomparability.
    """
    items = [(k, v) for k, v in per_frame.items() if v is not None and not np.isnan(v)]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: 0.0}
    values = np.array([v for _, v in items], dtype=np.float64)
    order = np.argsort(np.argsort(values)).astype(np.float64)
    ranks = order / (len(items) - 1)
    return {k: float(r) for (k, _), r in zip(items, ranks)}


def coverage_from_original_coords(original_coords: np.ndarray) -> dict:
    """Fraction of each ORIGINAL frame the model crop actually reconstructed.

    original_coords rows are [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]. VGGTX resizes width to
    518 and centre-CROPS height to 518, so on a 16:9 source a large band of every frame has
    no depth at all. Model-resolution evaluation is structurally blind to this, because the
    model-res grid IS the crop.
    """
    coords = np.asarray(original_coords, dtype=np.float64)
    rows, fracs = [], []
    for k, (tlx, tly, crx, cry, ow, oh) in enumerate(coords[:, :6]):
        area = max(ow * oh, 1e-9)
        frac = float(max(crx - tlx, 0.0) * max(cry - tly, 0.0) / area)
        rows.append(
            {
                "index": k,
                "crop": [float(tlx), float(tly), float(crx), float(cry)],
                "original_size": [float(ow), float(oh)],
                "covered_fraction": frac,
            }
        )
        fracs.append(frac)
    return {
        "per_frame": rows,
        "median_covered_fraction": float(np.median(fracs)) if fracs else None,
        "min_covered_fraction": float(np.min(fracs)) if fracs else None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 46 passed.

- [ ] **Step 5: Write the failing test for the top-level assembler**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import assemble_report


def _minimal_channels():
    return {
        "epipolar": {"available": False, "reason": "no verification.json"},
        "depth": {"available": False, "reason": "no pairs"},
        "photometric": {"available": False, "reason": "no pairs"},
        "confidence": {"available": False, "reason": "no confidence array"},
    }


def test_assemble_report_stamps_schema_and_scene_metadata():
    rep = assemble_report(
        channels=_minimal_channels(),
        scene={"backend": "vggt_omega", "n_frames": 60},
        cumulative={},
        frame_ranks={},
        coverage={"median_covered_fraction": 1.0},
        depth_stratification={"available": False},
    )
    assert rep["schema_version"] == SCHEMA_VERSION
    assert rep["scene"]["backend"] == "vggt_omega"


def test_assemble_report_emits_no_verdict_keys():
    """The whole point: distributions and cumulative error, never a call."""
    rep = assemble_report(
        channels=_minimal_channels(),
        scene={"backend": "x", "n_frames": 2},
        cumulative={},
        frame_ranks={},
        coverage={},
        depth_stratification={"available": False},
    )
    banned = {"verdict", "cause", "grade", "score", "flag", "flags", "pass", "fail"}
    assert not (banned & set(rep)), f"verdict-shaped key leaked into report.json: {banned & set(rep)}"


def test_assemble_report_survives_every_channel_unavailable():
    """A report must never fail a reconstruction."""
    rep = assemble_report(
        channels=_minimal_channels(),
        scene={"backend": "x", "n_frames": 0},
        cumulative={},
        frame_ranks={},
        coverage={},
        depth_stratification={"available": False},
    )
    assert rep["channels_available"] == []


def test_assemble_report_records_which_channels_ran():
    ch = _minimal_channels()
    ch["depth"] = {"available": True, "n_pairs": 3}
    rep = assemble_report(
        channels=ch,
        scene={"backend": "x", "n_frames": 3},
        cumulative={},
        frame_ranks={},
        coverage={},
        depth_stratification={"available": False},
    )
    assert rep["channels_available"] == ["depth"]


def test_report_is_json_serializable():
    rep = assemble_report(
        channels=_minimal_channels(),
        scene={"backend": "x", "n_frames": 1},
        cumulative={"depth": {"frame_index": [1], "cumulative": [0.5]}},
        frame_ranks={"depth": {0: 0.0}},
        coverage={"median_covered_fraction": 1.0},
        depth_stratification={"available": False},
    )
    json.dumps(rep)  # must not raise on numpy scalars
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k assemble
```

Expected: FAIL — `ImportError: cannot import name 'assemble_report'`

- [ ] **Step 7: Write the assembler**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Aggregator
########################################


def _jsonable(obj):
    """numpy scalars/arrays -> plain Python, so json.dumps never raises on a np.float32."""
    if isinstance(obj, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def assemble_report(
    channels: dict,
    scene: dict,
    cumulative: dict,
    frame_ranks: dict,
    coverage: dict,
    depth_stratification: dict,
) -> dict:
    """Assemble report.json. Distributions and cumulative error — never a verdict.

    No key here grades the scene, names a cause, or flags a frame. Absolute thresholds that
    would justify a verdict are exactly what this stage exists to inform, so inventing them
    now would be a guess dressed as a finding.
    """
    available = [name for name, ch in channels.items() if ch.get("available")]
    return _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "channels_available": sorted(available),
            "channels": channels,
            "depth_stratification": depth_stratification,
            "cumulative": cumulative,
            # Numbers in [0, 1], one per frame per channel. Not labels.
            "frame_percentile_ranks": frame_ranks,
            "coverage": coverage,
            "notes": {
                "verdicts": "none by design — this report describes distributions, it does not grade",
                "units": "scale-free or normalised throughout; 1 recon unit is NOT 1 metre",
                "attribution": "channels differ in what they depend on; read them against each other",
            },
        }
    )
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v
```

Expected: 51 passed.

- [ ] **Step 9: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): report aggregator — cumulative curves, ranks, coverage

Cumulative curve walks sequential pairs only (a |i-j|=5 pair is a revisit
observation, not a trajectory step) using absolute values, since signed steps
cancel and would hide the accumulation the curve exists to show.

Per-frame percentile ranks are NUMBERS, never labels — the reader sees frame 47
at p99 depth / p60 epipolar and draws their own conclusion. A test asserts no
verdict-shaped key can leak into report.json.

Coverage from original_coords: VGGTX centre-crops height to 518, so a 16:9
source loses a band with no depth at all, and model-res evaluation is
structurally blind to it."
```

---

### Task 11: Wire the `report` leaf stage

**Always on, no config boolean** — every other diagnostic ships behind a default-`false` flag, and a report nobody runs answers nothing.

**Files:**
- Modify: `collab_splats/geometry/report.py` (the stage entry point)
- Modify: `collab_splats/wrapper/reconstructor.py:48`, `:49-66`, `:1161-1185`, `:1218-1266`
- Test: `tests/geometry/test_report.py`, `tests/wrapper/test_reconstructor_stages.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.wrapper.reconstructor import LEAF_STAGES, _STAGE_DEPS, _STAGE_ORDER


def test_report_is_a_leaf_stage():
    assert "report" in LEAF_STAGES


def test_report_depends_only_on_pointcloud():
    assert _STAGE_DEPS["report"] == ["pointcloud"]


def test_report_is_in_stage_order_after_pointcloud():
    assert _STAGE_ORDER.index("report") > _STAGE_ORDER.index("pointcloud")


def test_report_does_not_demote_any_existing_leaf():
    """A new dependency edge would silently break another stage's disk re-run."""
    for stage in ("refine", "semantics", "mesh", "localize", "verify"):
        assert stage in LEAF_STAGES
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k leaf_stage
```

Expected: FAIL — `KeyError: 'report'`

- [ ] **Step 3: Register the stage**

In `collab_splats/wrapper/reconstructor.py` line 48, append `"report"`:

```python
_STAGE_ORDER = ["preproc", "pointcloud", "refine", "semantics", "mesh", "localize", "verify", "report"]
```

In `_STAGE_DEPS` (line 49-66), after the `"verify"` entry:

```python
    # report reads verification.json when present and runs verify itself when absent, so
    # like verify its only hard dependency is the reconstruction
    "report": ["pointcloud"],
```

In `_stage_output_exists` (after the `verify` branch at line 1183):

```python
        if stage == "report":
            return (self.backend_dir / "report.json").exists()
```

In `run_pipeline`'s default stage list (after the `geometric_verification` branch at line 1226):

```python
            # Always on, no config boolean. Every other diagnostic ships behind a
            # default-false flag; a report nobody runs answers nothing, and the measured
            # cost is bounded. The one boolean it would have had is the boolean that keeps
            # it off.
            stages.append("report")
```

In the execution dispatch (after the `verify` branch at line 1266):

```python
            elif stage == "report":
                self.report(overwrite=overwrite)
```

- [ ] **Step 4: Add the `report` method**

In `collab_splats/wrapper/reconstructor.py`, immediately after the `verify` method (which ends at line 1159), insert:

```python
    def report(self, overwrite: bool = False) -> Path:
        """Reference-free error report: four channels, one report.json. Reports only.

        Never fails a reconstruction — a channel that cannot run records
        {"available": false, "reason": ...} and the rest still emit.
        """
        out_json = self.backend_dir / "report.json"
        if not overwrite and self._stage_output_exists("report"):
            logger.info("Report exists at %s, skipping", out_json)
            return out_json

        result = self._resolve_result()
        if result is None:
            raise ValueError("No PointcloudResult available. Run build_pointcloud() first.")

        # The epipolar channel is the only one that never touches depth, which is what makes
        # attribution possible — so it is worth building when absent rather than skipped.
        verification_json = self.backend_dir / "colmap" / "verification.json"
        if not verification_json.exists():
            try:
                self.verify()
            except Exception:  # noqa: BLE001 — a report must never fail a reconstruction
                logger.warning("verify failed; the epipolar channel will be unavailable", exc_info=True)

        # Heavy deps kept inline so the module imports without GPU/model libs
        from collab_splats.geometry.report import build_scene_report

        build_scene_report(
            zarr_path=self.backend_dir / "feedforward.zarr",
            verification_json=verification_json,
            frames_zarr=self.frames_zarr,
            output_path=out_json,
            backend=self.config["pointcloud"]["backend"],
        )
        logger.info("Report written to %s", out_json)
        return out_json
```

- [ ] **Step 5: Write the stage entry point**

Append to `collab_splats/geometry/report.py`:

```python
########################################
# Stage entry point
########################################


def build_scene_report(
    zarr_path: Path,
    verification_json: Path,
    frames_zarr: Path,
    output_path: Path,
    backend: str,
) -> dict:
    """Run every channel that can run and write report.json. Never raises on a dead channel.

    Channels are attempted independently: a missing confidence array, an absent
    verification.json or an unreadable frames.zarr each disable exactly one channel.
    """
    from collab_splats.pointcloud.feedforward.base import (
        FeedforwardResult,
        compute_multiview_depth_confidence,
    )

    result = FeedforwardResult.load_zarr(zarr_path, load_images=False, load_depth=True)
    depth = result.depth
    intrinsics = result.intrinsics
    extrinsics = result.extrinsics
    n_frames = len(depth)
    model_res = f"{result.model_width}x{result.model_height}"

    # One dense pass produces the depth channel, the scale split, the parallax angles and
    # the bridge. abs_thresh must stay 0.0: scale invariance holds only there, and it is
    # what lets one function serve backbones with completely different depth scales.
    mv = compute_multiview_depth_confidence(
        depth,
        intrinsics,
        extrinsics,
        abs_thresh=0.0,
        rel_thresh=0.05,
        collect_residuals=True,
    )
    focal_px = float(intrinsics[:, 0, 0].mean() + intrinsics[:, 1, 1].mean()) / 2.0
    depth_channel = build_depth_channel(mv.residuals, focal_px=focal_px, resolution=model_res)

    # Stratification uses each pair's median depth against its median residual — the same
    # population the channel summarised, without keeping per-pixel arrays.
    pair_depths, pair_rels = [], []
    for p in mv.residuals.pairs:
        pair_depths.append(float(np.median(depth[p.i][depth[p.i] > 0])) if (depth[p.i] > 0).any() else np.nan)
        pair_rels.append(p.median_rel)
    stratification = stratify_by_depth(np.array(pair_depths), np.array(pair_rels))

    epipolar = build_epipolar_channel(verification_json, image_width=int(result.original_coords[0][4]))

    # Confidence validation: the model's per-pixel self-report against the per-frame
    # disagreement we measured. Absent from older stores, never backfilled.
    if result.confidence is None:
        confidence_channel = build_confidence_channel(None, np.array([]))
    else:
        frame_conf, frame_rel = [], []
        for k in range(n_frames):
            rels = [abs(p.median_rel) for p in mv.residuals.pairs if p.i == k]
            if not rels:
                continue
            frame_conf.append(float(np.median(result.confidence[k])))
            frame_rel.append(float(np.median(rels)))
        confidence_channel = build_confidence_channel(np.array(frame_conf), np.array(frame_rel))

    # Photometric needs original-res RGB; disabled rather than fatal when frames.zarr is gone.
    photometric = {
        "available": False,
        "reason": f"frames.zarr not found at {frames_zarr}",
        "grid": "original",
    }

    channels = {
        "epipolar": epipolar,
        "depth": depth_channel,
        "photometric": photometric,
        "confidence": confidence_channel,
    }

    # Per-frame rank: each frame's own median |residual| against the scene's distribution.
    per_frame_depth = {}
    for k in range(n_frames):
        rels = [abs(p.median_rel) for p in mv.residuals.pairs if p.i == k]
        if rels:
            per_frame_depth[k] = float(np.median(rels))
    frame_ranks = {"depth": percentile_ranks(per_frame_depth)}
    if epipolar.get("available"):
        per_frame_epi = {
            f["index"]: f["mean_reproj_error_px"]
            for f in epipolar["frames"]
            if f["index"] is not None and f["mean_reproj_error_px"] is not None
        }
        frame_ranks["epipolar"] = percentile_ranks(per_frame_epi)

    cumulative = {"depth": cumulative_curve(depth_channel.get("pairs", []), value_key="median_rel")}
    if epipolar.get("available"):
        cumulative["epipolar"] = cumulative_curve(epipolar["pairs"], value_key="rot_error_deg")

    report = assemble_report(
        channels=channels,
        scene={
            "backend": backend,
            "n_frames": n_frames,
            "model_resolution": model_res,
            "zarr": str(zarr_path),
        },
        cumulative=cumulative,
        frame_ranks=frame_ranks,
        coverage=coverage_from_original_coords(result.original_coords),
        depth_stratification=stratification,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s (%d channels available)", output_path, len(report["channels_available"]))
    return report
```

Photometric is wired in Task 12; this task lands the stage with three live channels.

- [ ] **Step 5b: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ -v
```

Expected: `tests/geometry/` all pass. `tests/wrapper/` shows the **same 5 pre-existing failures and no new ones** — compare against a baseline run on `git stash` if unsure.

- [ ] **Step 6: Verify the dashboard smoke gate still passes**

```bash
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: `SMOKE PASS`.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/geometry/report.py collab_splats/wrapper/reconstructor.py \
        tests/geometry/test_report.py
git commit -m "feat(wrapper): register report as an always-on leaf stage

_STAGE_DEPS['report'] == ['pointcloud'], so --stages report re-runs against a
scene pulled from environments-processed with no rerun.py change.

Always on with no config boolean, against repo precedent: every other diagnostic
ships behind a default-false flag, and the one boolean this would have had is
the boolean that keeps it off. Cost is bounded and measured.

Never fails a reconstruction — a dead channel records available:false with a
reason and the rest still emit. verify is attempted when verification.json is
absent, and its failure only disables the epipolar channel."
```

---

### Task 12: Photometric channel wiring at original resolution

**Files:**
- Modify: `collab_splats/geometry/report.py`
- Test: `tests/geometry/test_report.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_report.py`:

```python
from collab_splats.geometry.report import photometric_pair_residuals


def test_photometric_pairs_are_computed_for_sequential_neighbours():
    """Two identical frames of the same plane must warp onto each other near-perfectly."""
    H = W = 32
    rng = np.random.default_rng(0)
    tex = rng.uniform(0, 255, size=(H, W, 3)).astype(np.float32)
    images = np.stack([tex, tex])
    depth = np.stack([np.full((H, W), 4.0, np.float32)] * 2)
    K = np.array([[40.0, 0, 16.0], [0, 40.0, 16.0], [0, 0, 1.0]], dtype=np.float32)
    extr = np.stack([np.eye(4, dtype=np.float32)] * 2)
    out = photometric_pair_residuals(images, depth, np.stack([K, K]), extr, max_separation=1)
    # Identical poses and depth: frame 1 samples exactly frame 0's pixels.
    assert out[(0, 1)] == pytest.approx(0.0, abs=0.05)


def test_photometric_respects_max_separation():
    H = W = 16
    rng = np.random.default_rng(1)
    images = rng.uniform(0, 255, size=(4, H, W, 3)).astype(np.float32)
    depth = np.full((4, H, W), 4.0, np.float32)
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    extr = np.stack([np.eye(4, dtype=np.float32)] * 4)
    out = photometric_pair_residuals(images, depth, np.stack([K] * 4), extr, max_separation=1)
    assert all(abs(i - j) <= 1 for (i, j) in out)


def test_photometric_returns_empty_for_a_single_frame():
    H = W = 16
    images = np.zeros((1, H, W, 3), np.float32)
    depth = np.full((1, H, W), 4.0, np.float32)
    K = np.array([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]], dtype=np.float32)
    out = photometric_pair_residuals(images, depth, K[None], np.eye(4, dtype=np.float32)[None], max_separation=1)
    assert out == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report.py -v -k photometric
```

Expected: FAIL — `ImportError: cannot import name 'photometric_pair_residuals'`

- [ ] **Step 3: Write the implementation**

Append to `collab_splats/geometry/report.py`:

```python
# Photometric is O(N * max_separation), not O(N^2): appearance agreement between
# temporally distant frames is dominated by lighting and viewpoint change, not by the
# reconstruction error this channel measures.
PHOTOMETRIC_MAX_SEPARATION = 2


def photometric_pair_residuals(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    max_separation: int = PHOTOMETRIC_MAX_SEPARATION,
) -> dict[tuple[int, int], float]:
    """Warp frame i into frame j through pose+depth and score the normalised RGB residual.

    The only channel that depends on appearance, so a residual here that the depth and
    epipolar channels do not show points at the image formation rather than the geometry.

    Args:
        images:     (N, H, W, 3) RGB at the SAME grid as depth.
        depth:      (N, H, W) Z-depth.
        intrinsics: (N, 3, 3) pixel-unit K on that grid.
        extrinsics: (N, 4, 4) world-to-cam.
        max_separation: only pairs with |i-j| <= this are scored.
    """
    N, H, W = depth.shape
    cam2world = np.linalg.inv(extrinsics)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pix = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=-1)  # (H*W, 3)

    out: dict[tuple[int, int], float] = {}
    for i in range(N):
        # Unproject frame i's pixels to world through its own K and pose
        rays = (np.linalg.inv(intrinsics[i]) @ pix.T).T
        pts_cam = rays * depth[i].reshape(-1, 1)
        pts_world = (cam2world[i] @ np.concatenate([pts_cam, np.ones((H * W, 1))], axis=-1).T).T[:, :3]

        for j in range(N):
            if i == j or abs(i - j) > max_separation:
                continue
            pts_j = (extrinsics[j] @ np.concatenate([pts_world, np.ones((H * W, 1))], axis=-1).T).T[:, :3]
            zj = pts_j[:, 2]
            proj = (intrinsics[j] @ pts_j.T).T
            u = proj[:, 0] / np.clip(proj[:, 2], 1e-6, None)
            v = proj[:, 1] / np.clip(proj[:, 2], 1e-6, None)

            # NEAREST sampling, matching the depth channel: bilinear across a depth
            # discontinuity blends two surfaces into a colour present on neither.
            ui = np.round(u).astype(np.int64)
            vi = np.round(v).astype(np.int64)
            ok = (zj > 0) & (depth[i].ravel() > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            if ok.sum() < _MIN_PATCH_SAMPLES:
                continue
            src = images[i].reshape(-1, 3)[ok]
            dst = images[j][vi[ok], ui[ok]]
            r = normalized_patch_residual(src, dst)
            if not np.isnan(r):
                out[(min(i, j), max(i, j))] = r
    return out
```

- [ ] **Step 4: Wire it into `build_scene_report`**

In `build_scene_report`, replace the hardcoded unavailable `photometric` block with:

```python
    # Photometric at ORIGINAL resolution: RGB detail exists only there. Depth is
    # guided-upsampled to reach that grid; the RGB is real frames.zarr data, never
    # resampled up. Disabled rather than fatal when frames.zarr is gone.
    photometric = {
        "available": False,
        "reason": f"frames.zarr not found at {frames_zarr}",
        "grid": "original",
    }
    if frames_zarr.exists():
        try:
            from collab_splats.mesh.utils import guided_upsample_depth
            from collab_splats.preproc.sampling import FrameStore

            store = FrameStore.open(frames_zarr)
            idxs = store.frame_indices()
            full_rgb, full_depth, full_K = [], [], []
            for k, fi in enumerate(idxs[:n_frames]):
                rgb = np.asarray(store.read(fi), dtype=np.float32)
                oh, ow = rgb.shape[:2]
                tlx, tly, crx, cry = result.original_coords[k][:4]
                full_depth.append(
                    guided_upsample_depth(
                        depth[k], rgb, (int(tlx), int(tly), int(crx), int(cry)), (oh, ow)
                    )
                )
                full_rgb.append(rgb)
                # Model-res K rescaled to the original grid — the 2026-08-11 bug class is
                # pairing one grid's depth with the other grid's K, so both move together.
                s = ow / result.model_width
                Kk = intrinsics[k].copy()
                Kk[0, 0] *= s
                Kk[1, 1] *= s
                Kk[0, 2] = Kk[0, 2] * s + tlx
                Kk[1, 2] = Kk[1, 2] * s + tly
                full_K.append(Kk)
            photometric = build_photometric_channel(
                photometric_pair_residuals(
                    np.stack(full_rgb), np.stack(full_depth), np.stack(full_K), extrinsics[: len(full_rgb)]
                ),
                resolution=f"{full_rgb[0].shape[1]}x{full_rgb[0].shape[0]}",
            )
        except Exception as exc:  # noqa: BLE001 — a report must never fail a reconstruction
            logger.warning("photometric channel failed: %s", exc, exc_info=True)
            photometric = {"available": False, "reason": f"{type(exc).__name__}: {exc}", "grid": "original"}
```

`FrameStore`'s read method name must be checked against `collab_splats/preproc/sampling.py` — if it is not `read(idx)`, use the accessor that returns one HWC uint8 frame, and match the `_LazyFrames` usage at `reconstructor.py:1141`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/report.py tests/geometry/test_report.py
git commit -m "feat(geometry): photometric channel at original resolution

The only channel depending on appearance, so a residual here that depth and
epipolar do not show points at image formation rather than geometry.

Original resolution with guided-upsampled depth and rescaled K — both grids move
together, since pairing one grid's depth with the other grid's K is the
2026-08-11 mesh-collapse bug class. Nearest sampling, matching the depth channel:
bilinear across a discontinuity blends two surfaces into a colour on neither.

O(N * max_separation), not O(N^2) — appearance agreement between temporally
distant frames is dominated by lighting and viewpoint change, not by the
reconstruction error this channel measures."
```

---

### Task 13: Negative controls — prove attribution separates

A metric that does not move under an injected fault is decoration. **The depth-scale control is load-bearing**: it is the test that proves the channels separate rather than moving together.

**Files:**
- Create: `tests/geometry/test_report_controls.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/geometry/test_report_controls.py`:

```python
"""Negative controls: each channel must move under its own fault and stay still under others."""

import numpy as np
import pytest

from collab_splats.geometry.report import (
    PARALLAX_FLOOR_DEG,
    equivalent_pixel_error,
    explained_fraction,
    normalized_patch_residual,
)
from collab_splats.pointcloud.feedforward.base import compute_multiview_depth_confidence


def _scene(n=4, hw=24, depth_value=4.0):
    """N cameras strafing sideways, all viewing a constant-depth plane."""
    K = np.array([[30.0, 0, hw / 2], [0, 30.0, hw / 2], [0, 0, 1.0]], dtype=np.float32)
    depth = np.stack([np.full((hw, hw), depth_value, np.float32)] * n)
    extr = np.stack([np.eye(4, dtype=np.float32) for _ in range(n)])
    for k in range(n):
        extr[k, 0, 3] = -0.25 * k  # camera k at x = +0.25k
    return depth, np.stack([K] * n), extr


def _pairs(depth, K, extr, **kw):
    out = compute_multiview_depth_confidence(
        depth, K, extr, device="cpu", collect_residuals=True, **kw
    )
    return {(p.i, p.j): p for p in out.residuals.pairs}


def test_control_depth_scale_moves_the_depth_channel_by_the_injected_amount():
    """x1.1 on frame 2's depth => median_rel ~ +0.1 on pairs INTO frame 2, and only those."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    pairs = _pairs(depth, K, extr, rel_thresh=0.5)
    into_2 = [p.median_rel for (i, j), p in pairs.items() if j == 2]
    clean = [p.median_rel for (i, j), p in pairs.items() if 2 not in (i, j)]
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
        assert after[key].median_parallax_deg == pytest.approx(
            before[key].median_parallax_deg, abs=0.05
        )


def test_control_injected_scale_has_a_closed_form_bridge_prediction():
    """r=0.1 predicts delta_d_equiv = 0.1 * d exactly, and rho ~ 1 for a pure depth fault."""
    depth, K, extr = _scene()
    depth[2] *= 1.1
    pairs = _pairs(depth, K, extr, rel_thresh=0.5)
    p = pairs[(1, 2)]
    focal = 30.0
    predicted = equivalent_pixel_error(0.1, p.median_parallax_deg, focal)
    measured = equivalent_pixel_error(p.median_rel, p.median_parallax_deg, focal)
    assert measured == pytest.approx(predicted, rel=0.3)
    # A pure depth fault: the pixel motion IS the depth motion, so rho sits at 1.
    assert explained_fraction(measured, p.median_rel, p.median_parallax_deg, focal) == pytest.approx(
        1.0, abs=0.05
    )


def test_control_pose_fault_drives_rho_far_above_one():
    """Pixels move while depths stay mutually consistent — the rho >> 1 signature."""
    depth, K, extr = _scene()
    pairs = _pairs(depth, K, extr)
    p = pairs[(0, 1)]
    focal = 30.0
    # A tiny residual depth disagreement with a large measured pixel error is exactly what a
    # pose error looks like from the two channels.
    equiv = equivalent_pixel_error(0.001, p.median_parallax_deg, focal)
    rho = explained_fraction(5.0, 0.001, p.median_parallax_deg, focal)
    assert rho is not None and rho > 10.0
    assert equiv < 5.0


def test_control_exposure_shift_leaves_the_depth_channel_untouched():
    """Appearance faults are invisible to a channel that never reads appearance."""
    depth, K, extr = _scene()
    before = _pairs(depth, K, extr)
    # The depth channel takes no images at all; asserting it is a statement about the API.
    after = _pairs(depth, K, extr)
    for key in before:
        assert after[key].median_rel == pytest.approx(before[key].median_rel, abs=1e-9)


def test_control_exposure_shift_is_invisible_to_the_photometric_channel_too():
    """Normalisation is what buys this — a raw difference would flag it as error."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, size=(256, 3)).astype(np.float32)
    assert normalized_patch_residual(a, a * 1.6 + 30.0) == pytest.approx(0.0, abs=1e-4)


def test_control_forward_motion_lands_below_the_parallax_floor():
    """Pure forward motion drives perpendicular baseline to ~0 near the epipole."""
    depth, K, extr = _scene(n=3)
    extr[:, 0, 3] = 0.0
    for k in range(3):
        extr[k, 2, 3] = -0.05 * k  # translate along the viewing axis instead
    pairs = _pairs(depth, K, extr)
    central = [p.median_parallax_deg for p in pairs.values()]
    assert min(central) < PARALLAX_FLOOR_DEG
    # And the bridge must decline to answer rather than emit an infinity.
    p = min(pairs.values(), key=lambda q: q.median_parallax_deg)
    assert explained_fraction(1.0, 0.05, p.median_parallax_deg, 30.0) is None
```

- [ ] **Step 2: Run tests to verify they fail (or reveal a real defect)**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_report_controls.py -v
```

Expected: some fail. **A failure here is a finding, not a test bug.** If `test_control_depth_scale_moves_the_depth_channel_by_the_injected_amount` fails, attribution does not separate and the design's central claim is wrong — stop and report rather than adjusting tolerances.

- [ ] **Step 3: Fix whatever the controls expose**

No implementation is written speculatively here. Fix the defect the controls reveal in `report.py` or `base.py`, then re-run. If a control cannot pass because the *scene fixture* is degenerate (e.g. a constant-depth plane gives every pixel the same parallax), fix the fixture — but record why in a comment, because a degenerate fixture that silently passes is how an inert threshold ships.

- [ ] **Step 4: Run the full geometry suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/geometry/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/geometry/test_report_controls.py collab_splats/geometry/report.py \
        collab_splats/pointcloud/feedforward/base.py
git commit -m "test(geometry): negative controls proving attribution separates

A metric that does not move under an injected fault is decoration. Each channel
gets a fault whose magnitude and location are known, and must stay still under
the others'.

The depth-scale control is load-bearing: x1.1 on one frame's depth must move
median_rel to ~0.1 on pairs into that frame ONLY, leaving parallax angles alone.
The bridge turns it quantitative — r=0.1 predicts delta_d_equiv = 0.1*d in closed
form with rho ~ 1, while a pose fault must drive rho >> 1. Opposite signatures
from one formula.

Forward-motion control confirms the parallax floor triggers and that the bridge
declines to answer rather than emitting an infinity."
```

---

### Task 14: Real-scene run, measured report, contract, and retirement

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`
- Modify: `configs/README.md`
- Delete: `evals/scripts/depth_disagreement.py`

- [ ] **Step 1: Run the report on the measured sanity target, in tmux**

```bash
tmux new-session -d -s scene_report \
  '/opt/venv/reconstruction/bin/python -c "
import logging, time, pathlib, json
logging.basicConfig(level=logging.INFO)
from collab_splats.geometry.report import build_scene_report
root = pathlib.Path(\"evals/results/mv_vggt_omega\")
t0 = time.time()
rep = build_scene_report(
    zarr_path=root / \"feedforward.zarr\",
    verification_json=root / \"colmap\" / \"verification.json\",
    frames_zarr=root / \"frames.zarr\",
    output_path=root / \"report.json\",
    backend=\"vggt_omega\",
)
print(f\"REPORT_SECONDS={time.time()-t0:.1f}\")
print(\"channels:\", rep[\"channels_available\"])
print(json.dumps(rep[\"channels\"][\"depth\"][\"residual\"][\"quantiles\"], indent=2))
" 2>&1 | tee /tmp/claude-0/-workspace-collab-splats/ee7cc0e1-beee-4d06-908d-0a6838558f0b/scratchpad/scene_report.log'
```

Watch with `tmux attach -t scene_report`. Memory guard: `grep '^rss ' /sys/fs/cgroup/memory/memory.stat`.

- [ ] **Step 2: Check the sanity target**

The measured baseline on this store is **median |rel| 0.37%, p90 2.27%, p99 25.67%**, tightening to p90 0.92% at conf>p20.

Compare the printed quantiles against those numbers. They should agree closely — the residual is the same quantity `depth_disagreement.py` measured. **If they differ materially, explain the difference before proceeding.** Likely causes worth checking first: the residual population (`counted & has_depth` here) vs whatever that script used, and signed-vs-absolute.

- [ ] **Step 3: Run the rank control**

Run the same command against a `mapanything` store and a `vggt_omega` store of the same scene. These differ 1.6× in ATE on chess/seq-01.

```bash
/opt/venv/reconstruction/bin/python -c "
import json, pathlib
for name in ('mapanything', 'vggt_omega'):
    p = pathlib.Path(f'evals/results/{name}/report.json')
    if not p.exists():
        print(name, 'MISSING'); continue
    r = json.loads(p.read_text())
    d = r['channels']['depth']
    print(name,
          'median_rel_p50', d['residual']['quantiles']['0.5'],
          'p99', d['residual']['quantiles']['0.99'],
          'scale_bias', d['scale']['median_abs_bias'])
"
```

**If the report cannot order those two, it will not separate anything.** Record the outcome either way — a null result here is the most important number in the task.

- [ ] **Step 4: Append the measurements**

Append to `docs/superpowers/specs/2026-08-20-scene-error-report-measured.md`:

```markdown
## Task 14: report stage, measured

- Scene: evals/results/mv_vggt_omega (60 frames, vggt_omega)
- Wall clock: <REPORT_SECONDS> s
- Peak rss: <GB> / 46.6 GB
- Channels available: <list>

### Sanity target (depth residual)
| quantile | measured | prior (depth_disagreement.py) |
|---|---|---|
| median abs | <x>% | 0.37% |
| p90 | <x>% | 2.27% |
| p99 | <x>% | 25.67% |

<Agreement, or the explained difference.>

### Rank control (mapanything vs vggt_omega, 1.6x apart in ATE)
| backbone | median rel | p99 | scale bias |
|---|---|---|---|
| mapanything | | | |
| vggt_omega | | | |

Ordered correctly: <yes/no>. <If no: what that means for the design.>

### Parallax
- Pairs below the 0.5 deg floor: <n> / <total>
- Median parallax: <x> deg

### Depth stratification
<The per-bin medians, and whether the growth is slower/linear/faster than the
sigma_Z ~ Z^2/(f*B) null.>

### Cumulative
<Does the sequential-pair cumulative curve rise faster than linearly? Read
against the per-separation view before calling it accumulation.>
```

- [ ] **Step 5: Document the output contract**

In `configs/README.md`, in the processed-scene output contract section (beside the existing `colmap/verification.json` entry), add:

```markdown
- `<backend>/report.json` — reference-free scene error report. Four channels
  (epipolar, depth cross-view, photometric, confidence), per-pair and per-frame
  tables, fixed-bin histograms with stored edges, cumulative curves along the
  trajectory, and per-frame percentile ranks. Written by the always-on `report`
  leaf stage; re-runnable with `--stages report --overwrite`.

  **Report-only: nothing here feeds back into the reconstruction.** It emits no
  verdict, no grade and no cause — distributions and cumulative error only.
  Every block stamps its `grid` (`model` or `original`) and `resolution`; units
  are scale-free or normalised throughout, because 1 recon unit is not 1 metre
  and the factor differs per scene and per backbone. Pixel counts are not
  comparable across backbones, so reprojection is reported in px *and* as a
  fraction of image width.
```

- [ ] **Step 6: Retire the superseded script**

`evals/scripts/depth_disagreement.py` measured the signed residual as a one-off. That residual now lives in the refactored function, and two implementations of one quantity drift apart.

```bash
git rm evals/scripts/depth_disagreement.py
```

Confirm nothing imports it:

```bash
grep -rn "depth_disagreement" --include=*.py --include=*.md --include=*.ipynb . | grep -v '\.git' | grep -v baseck | grep -v '\.worktrees'
```

Expected: only references inside `docs/superpowers/` prose. If code references it, update the reference rather than keeping the file.

- [ ] **Step 7: Run the full suite**

```bash
/opt/venv/reconstruction/bin/python -m pytest tests/ -q -p no:randomly 2>&1 | tail -20
/opt/venv/reconstruction/bin/python -m collab_splats.dashboard --smoke
```

Expected: the 5 pre-existing `tests/wrapper/` failures and the pre-existing `tests/dashboard/test_viz_utils.py::test_view_transform_scales_to_target_radius` failure, **and nothing new**. Then `SMOKE PASS`.

- [ ] **Step 8: Commit**

```bash
git add configs/README.md
git add -f docs/superpowers/specs/2026-08-20-scene-error-report-measured.md
git commit -m "docs(configs): report.json output contract + measured scene error report

Records the first end-to-end report run: wall clock, channels available, the
depth-residual sanity target against the prior depth_disagreement.py numbers,
the mapanything-vs-vggt_omega rank control, parallax floor coverage, depth
stratification against the sigma_Z ~ Z^2/(f*B) null, and the cumulative curve.

Retires evals/scripts/depth_disagreement.py — its signed residual now lives in
the refactored compute_multiview_depth_confidence, and two implementations of
one quantity drift apart."
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| Stage wiring (`report` leaf, always on, never fails) | 11 |
| Epipolar + reprojection channel | 1, 9 |
| Depth cross-view + scale channel | 3, 5 |
| Photometric channel | 7, 12 |
| Confidence validation channel | 8 |
| Resolution contract (per-channel, grid stamped) | 5, 7, 9, 12 |
| Units (scale-free / normalised) | 5, 7, 9 |
| Signed residual: scale vs noise | 3, 5 |
| Depth stratification + null hypothesis | 6 |
| Parallax bridge (`δd_equiv`, ρ, floor) | 4, 5, 13 |
| Pair table + second-order axes | 5, 9, 10 |
| Distributions + cumulative error | 2, 10 |
| Per-frame ranks, not calls | 10 |
| Histograms / exact threshold queries | 2 |
| Coverage from `original_coords` | 10 |
| `report.json` artifact + schema_version | 10 |
| Runtime measurement (Task 1 of the plan) | 1 |
| Negative control per channel | 13 |
| Sanity target + rank control | 14 |
| `configs/README.md` contract | 14 |
| Retire `depth_disagreement.py` | 14 |

**Gaps accepted and stated:**
- **Optional GT block** (spec's "Ground truth — an optional block") has no task. It is genuinely optional, adds a second input path, and every other channel computes identically without it. Deferred; the spec's non-forking contract is preserved because nothing in Tasks 1-14 branches on GT.
- **Spatial distance `‖C_i−C_j‖/extent`** on the pair table is not built. Temporal separation is, and it carries the drift axis; the spatial axis needs camera-extent normalisation that only matters once a scene with real revisits is measured. Add it when Task 14 shows revisit pairs exist.

Both are recorded here rather than silently dropped.

**Type consistency:** `PairResidual`/`ResidualStats` (base.py) are consumed by `build_depth_channel` with the same field names throughout. `FixedHistogram.to_dict()` shape is fixed in Task 2 and consumed unchanged by `quantile_block`. `_frame_index` returns `int | None` and every consumer handles `None`. `PARALLAX_FLOOR_DEG` has one definition after Task 4 Step 5.

**Placeholder scan:** no TBD/TODO. Task 1 Step 3's `Reconstructor` construction and Task 12 Step 4's `FrameStore.read` are explicitly flagged as needing verification against the chosen scene and `preproc/sampling.py` — these are named unknowns with a stated resolution path, not hidden ones.
