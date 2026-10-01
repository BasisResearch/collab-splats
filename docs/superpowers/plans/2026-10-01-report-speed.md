# Reconstruction Quality Report Speed-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `Reconstructor.reconstruction_quality_report` wall time (40 min at 1k frames) with report values unchanged by default, plus one opt-in pair-pruning knob.

**Architecture:** The O(N²) cross-view pass in `geometry/metrics._collect_pairs` moves its histogram onto the GPU (`torch.bucketize` + `bincount`), stops syncing per pair, and batches target views through `depth_residual` generalized to a leading pose batch. Per-frame medians become one O(pairs) pass. The photometric pass streams one frame's guided upsample at a time (kornia `guided_blur` on GPU) and projects in float32 on GPU. A frustum-overlap pre-pass behind `reconstruction_quality_report.min_pair_overlap` (default 0.0 = off) prunes non-overlapping pairs.

**Tech Stack:** torch (CUDA, A40), kornia 0.8.2 (`kornia.filters.guided_blur`), numpy 2.1.3, OpenCV, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-report-speed-design.md`

---

## Ground rules for every task

- **Worktree only.** All code changes happen in `/workspace/collab-splats/.worktrees/report-speed`
  (branch `perf/report-speed`). Never edit `/workspace/collab-splats` itself except the final
  docs task, which says so explicitly.
- **Test the worktree, not the main checkout.** Every pytest command is
  `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest ...`.
  If unsure, run `PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"`;
  it must print a path under `.worktrees/report-speed`.
- **Commit with `git commit --only <paths>`.** The index is shared with other sessions.
- **Never pipe pytest into `tail`/`head`.** That hides the exit code. Use `-q` and read the summary.
- **Code style** (CLAUDE.md + user memory):
  - Block comments are ONE plain line saying what the code does. No header+bullet runs.
  - Docstrings: `"""` on their own lines, one-line summary, then `- ` bullets. Public functions
    get `Args:`/`Returns:`; private `_` helpers get summary + bullets only.
  - Absolute `collab_splats.` imports, all at file top. Tunables are keyword defaults, not module constants.
  - Blank line above every block comment and around every `for`/`if`/`with` block.
  - US spelling.
- **Heavy runs** (the harness on GH010229) go in tmux; nothing else runs alongside them.
- **Parity contract (Tasks 3-8):** integer columns (`idx1`, `idx2`, `n_pixels`, `frame_idx`) and
  histogram `counts`/`bin_edges` bit-identical; float columns `atol=1e-6`. Tasks 9-10 change
  photometric values and use the tolerance mode.

## File map

| File | Change |
|---|---|
| `collab_splats/geometry/metrics.py` | GPU histogram, sync reduction, batched targets, O(pairs) medians, streaming photometric, `min_pair_overlap` |
| `collab_splats/geometry/projection.py` | `project` / `depth_residual` take a leading pose batch; `depth_agreement` returns `expected` too |
| `collab_splats/geometry/transforms.py` | `transform_points` takes a leading pose batch |
| `collab_splats/utils/image.py` | kornia `guided_blur` replaces `_box` / `_guided_filter` |
| `collab_splats/wrapper/reconstructor.py` | uint8 images, `min_pair_overlap` from config, recorded in `scene` |
| `configs/base.yaml` | `reconstruction_quality_report: {min_pair_overlap: 0.0}` |
| `tests/geometry/test_metrics.py` | histogram, medians, batching, pruning tests; upsample-spy test updated |
| `tests/geometry/test_projection.py` | batched `project` / `depth_residual` tests |
| `tests/geometry/test_transforms.py` | batched `transform_points` test |
| `tests/utils/test_image.py` | cv2 `BORDER_REFLECT_101` oracle parity test |
| `tests/wrapper/test_reconstructor.py` | `scene.min_pair_overlap` assertion |
| `scratch/report_speed/run_report.py`, `compare_reports.py` | measurement harness (gitignored `scratch/`, never committed) |

---

### Task 0: Worktree

**Files:** none (git plumbing)

- [ ] **Step 1: Create the worktree from `clean/final`**

```bash
cd /workspace/collab-splats
git worktree add .worktrees/report-speed -b perf/report-speed clean/final
```

Expected: `Preparing worktree (new branch 'perf/report-speed')`.

- [ ] **Step 2: Symlink the untracked `third_party/*` checkouts**

Only `third_party/README.md` is tracked; tests skip without the rest.

```bash
cd /workspace/collab-splats/.worktrees/report-speed
for d in /workspace/collab-splats/third_party/*/ /workspace/collab-splats/third_party/.vda_fetch_done; do
  name=$(basename "$d"); [ -e "third_party/$name" ] || ln -s "$d" "third_party/$name"
done
ls -la third_party
```

Expected: symlinks `LoGeR`, `Video-Depth-Anything`, `hloc`, `.vda_fetch_done`.

- [ ] **Step 3: Verify imports resolve to the worktree and the geometry suite is green**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats, kornia, gsplat; print(collab_splats.__file__, kornia.__version__, gsplat.__version__)"
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry tests/utils -q -x
```

Expected: path under `.worktrees/report-speed`, kornia `0.8.2`; all pass. Record the pass/skip counts
in your report as the pre-change gate.

---

### Task 1: Measurement harness and baseline

**Files:**
- Create: `scratch/report_speed/run_report.py` (gitignored, not committed)
- Create: `scratch/report_speed/compare_reports.py` (gitignored, not committed)

- [ ] **Step 1: Write `run_report.py`**

It builds a scratch scene that symlinks the real inputs, so the stage never writes into
`/workspace/outputs`. It times each phase by wrapping module functions.

```python
"""
Run reconstruction_quality_report on GH010229 with phase timers; scratch harness, not shipped.
"""

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import torch

from collab_splats.geometry import metrics
from collab_splats.wrapper import reconstructor as rmod
from collab_splats.wrapper.reconstructor import Reconstructor

SCENE = Path("/workspace/outputs/ocr_viewer/GH010229")


def _timed(module, name, timings):
    """Replace module.name with a wrapper that adds its wall time to timings[name]."""
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = real(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - t0
        return out

    setattr(module, name, wrapper)


def main() -> None:
    """Build the scratch scene, run the stage, write <label>.json and <label>.timing.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--min_pair_overlap", type=float, default=None)
    ap.add_argument("--out", default="scratch/report_speed/out")
    args = ap.parse_args()

    # Scratch scene: real images/ and pointcloud.zarr symlinked, report written here
    root = Path(args.out) / "scene"
    backend = root / "vggt_omega"
    backend.mkdir(parents=True, exist_ok=True)
    for link, target in ((root / "images", SCENE / "images"), (backend / "pointcloud.zarr", SCENE / "vggt_omega" / "pointcloud.zarr")):
        if not link.exists():
            link.symlink_to(target)

    # Feedforward vggt_omega config pointed at the scratch scene
    config = {
        "input_path": str(root / "video.mp4"),
        "output_path": str(root),
        "pointcloud": {
            "method": "feedforward",
            "backend": "vggt_omega",
            "bundle_adjustment": False,
            "loop_closure": False,
            "clean": {"enabled": False},
        },
        "semantics": {"enabled": False},
        "mesh": {"enabled": False},
        "localization": {"enabled": False},
    }
    if args.min_pair_overlap is not None:
        config["reconstruction_quality_report"] = {"min_pair_overlap": args.min_pair_overlap}
    rec = Reconstructor(config)
    rec._resolve_result = lambda: object()

    # Phase timers on every function the stage reaches
    timings: dict[str, float] = {}
    for name in ("_collect_pairs", "compute_photometric_ncc", "upsample_depths"):
        _timed(metrics, name, timings)
    if hasattr(metrics, "_frame_medians"):
        _timed(metrics, "_frame_medians", timings)
    _timed(rmod, "_scene_frames", timings)

    # The stage itself, timed end to end
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out_json = rec.reconstruction_quality_report(overwrite=True)
    timings["total"] = time.perf_counter() - t0
    timings["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    timings["cuda_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3

    # Report and timings side by side under the label
    dest = Path(args.out)
    shutil.copy(out_json, dest / f"{args.label}.json")
    (dest / f"{args.label}.timing.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
```

Note: `upsample_depths` is timed inside `metrics` because metrics imports it by name; the
wrapper sees every call from `compute_photometric_ncc`. If `Reconstructor(config)` rejects the
minimal config, copy the missing keys from `tests/wrapper/test_reconstructor.py::_make_config`.

- [ ] **Step 2: Write `compare_reports.py`**

```python
"""
Compare two reconstruction quality reports under the report-speed parity contract.
"""

import argparse
import json
import sys

import numpy as np

EXACT = {"idx1", "idx2", "n_pixels", "frame_idx"}


def _column_delta(a: list, b: list) -> float:
    """Max |a - b| over a float column; None must match None."""
    if [x is None for x in a] != [x is None for x in b]:
        return float("inf")
    pa = np.array([x for x in a if x is not None], np.float64)
    pb = np.array([x for x in b if x is not None], np.float64)
    return float(np.abs(pa - pb).max()) if len(pa) else 0.0


def main() -> None:
    """Exit 1 when an exact column or the histogram differs, or a float column exceeds atol."""
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--photometric", choices=["exact", "report"], default="exact")
    args = ap.parse_args()
    a, b = json.load(open(args.a)), json.load(open(args.b))
    failed = False

    # frames and depth_pairs: exact integer columns, atol float columns
    for table in ("frames", "depth_pairs"):
        for col in a[table]:
            if col in EXACT:
                ok = a[table][col] == b[table][col]
                print(f"{table}.{col}: {'identical' if ok else 'DIFFERS'}")
            else:
                d = _column_delta(a[table][col], b[table][col])
                ok = d <= args.atol
                print(f"{table}.{col}: max |delta| {d:.3e}")
            failed |= not ok

    # Histogram: bit-identical counts and edges
    for key in ("counts", "bin_edges"):
        ok = a["depth_residual_histogram"][key] == b["depth_residual_histogram"][key]
        print(f"histogram.{key}: {'identical' if ok else 'DIFFERS'}")
        failed |= not ok

    # Photometric: exact, or report the pair-set and value deltas only
    pa, pb = a["photometric_pairs"], b["photometric_pairs"]
    keys_a = list(zip(pa["idx1"], pa["idx2"]))
    keys_b = list(zip(pb["idx1"], pb["idx2"]))
    common = sorted(set(keys_a) & set(keys_b))
    ia = {k: n for n, k in enumerate(keys_a)}
    ib = {k: n for n, k in enumerate(keys_b)}
    d_ncc = max((abs(pa["photometric_ncc"][ia[k]] - pb["photometric_ncc"][ib[k]]) for k in common), default=0.0)
    d_n = [pb["n_pixels"][ib[k]] - pa["n_pixels"][ia[k]] for k in common]
    print(f"photometric: {len(keys_a)} vs {len(keys_b)} pairs, {len(common)} common")
    print(f"photometric_ncc max |delta| {d_ncc:.3e}; n_pixels delta min {min(d_n, default=0)} max {max(d_n, default=0)}")
    if args.photometric == "exact":
        failed |= pa != pb

    print("FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Record the baseline in tmux**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
mkdir -p scratch/report_speed/out
tmux new -d -s report-speed "cd $PWD && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python scratch/report_speed/run_report.py baseline > scratch/report_speed/out/baseline.log 2>&1; echo EXIT \$? >> scratch/report_speed/out/baseline.log"
```

Wait for `EXIT` in the log (Monitor with an until-loop on `grep -q EXIT`, never a `pgrep` loop).
Expected: `EXIT 0`, `baseline.json` and `baseline.timing.json` written. Run a second time as
`baseline2` and confirm `compare_reports.py baseline.json baseline2.json` prints PASS. That proves
the stage is deterministic before any parity claim leans on it.

- [ ] **Step 4: Report the baseline**

Report the per-phase seconds, peak RSS, CUDA peak and pair count. Nothing is committed; `scratch/` is gitignored.

---

### Task 2: GPU histogram

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (`bounded_residual`, new `_histogram_counts`, `_collect_pairs`)
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append after `test_bounded_residual_preserves_quantiles_through_the_histogram`; add `import torch`
to the third-party import group.

```python
def test_bounded_residual_on_a_tensor_is_bit_identical_to_numpy():
    """The GPU histogram's map must match the numpy map bit for bit, so float64 in torch too."""
    rng = np.random.default_rng(0)
    rel = rng.standard_normal(10_000).astype(np.float32) * 3
    u_np = bounded_residual(rel)
    u_t = bounded_residual(torch.as_tensor(rel))
    assert u_t.dtype == torch.float64
    np.testing.assert_array_equal(u_t.numpy(), u_np)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA"))])
def test_histogram_counts_match_np_histogram_including_the_edges(device):
    """bucketize + bincount bins exactly as np.histogram, both closed ends included."""
    edges = residual_bin_edges(_n_samples(300, 518))
    rng = np.random.default_rng(1)
    u = bounded_residual(rng.standard_normal(200_000) * 2)

    # Every edge value, plus both ends, which np.histogram closes into the outer bins
    u = np.concatenate([u, edges, [-1.0, 1.0, 1.0]])
    got = metrics._histogram_counts(torch.as_tensor(u, device=device), torch.as_tensor(edges, device=device))
    np.testing.assert_array_equal(got.cpu().numpy(), np.histogram(u, bins=edges)[0])
```

- [ ] **Step 2: Run them to see them fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -q -k "bit_identical_to_numpy or histogram_counts_match"`
Expected: FAIL. The tensor test raises inside `np.asarray`, and the others fail with `AttributeError: ... '_histogram_counts'`.

- [ ] **Step 3: Implement**

Replace `bounded_residual` and add `_histogram_counts` right after it (`Tensor` import: add
`from torch import Tensor` to the third-party group):

```python
def bounded_residual(rel: np.ndarray | Tensor | float) -> np.ndarray | Tensor:
    """
    Map a relative depth residual onto (-1, 1) so a fixed histogram can never miss it.

    - r / (1 + |r|) is monotone, so quantiles survive the map exactly
    - invert with u / (1 - |u|)
    - no chosen range: clipping piles the tail into end bins, and np.histogram drops out-of-range values
    - a tensor maps in float64 on its own device, bit-identical to the numpy path

    Args:
        rel: relative depth residuals, any shape; numpy, scalar or torch.

    Returns:
        Float64 array, or float64 tensor for a tensor input, of the same shape, in (-1, 1).
    """
    if isinstance(rel, Tensor):
        r = rel.double()
        return r / (1.0 + r.abs())
    r = np.asarray(rel, dtype=np.float64)
    return r / (1.0 + np.abs(r))


def _histogram_counts(u: Tensor, edges: Tensor) -> Tensor:
    """
    np.histogram(u, bins=edges)[0] on u's device: bucketize then bincount.

    - float64 edges: float32 histc misbins against these edges
    - right=True and the clamp close both outer bins, as np.histogram does
    """
    k = len(edges) - 1
    idx = (torch.bucketize(u, edges, right=True) - 1).clamp_(0, k - 1)
    return torch.bincount(idx, minlength=k)
```

In `_collect_pairs`, keep the numpy `edges` for `collected` but accumulate on the device:

```python
    # One histogram over every ordered pair's residuals: n*(n-1)*h*w at most
    edges = residual_bin_edges(n * (n - 1) * h * w)
    edges_t = torch.as_tensor(edges, device=device)
    counts = torch.zeros(len(edges) - 1, dtype=torch.int64, device=device)
    collected = {"pairs": [], "rel_depth_error_edges": edges}
    agreement = []
```

Replace the two histogram lines inside the `j` loop:

```python
            # Per-pixel residuals into the one device histogram; one PairStats row per direction
            counts += _histogram_counts(bounded_residual(rel_sel), edges_t)
```

And before `return collected, agreement`:

```python
    collected["rel_depth_error_counts"] = counts.cpu().numpy()
    return collected, agreement
```

- [ ] **Step 4: Run the metrics suite**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/geometry/test_metrics_controls.py -q`
Expected: all pass.

- [ ] **Step 5: Parity on GH010229**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
tmux new -d -s report-speed "cd $PWD && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python scratch/report_speed/run_report.py t2 > scratch/report_speed/out/t2.log 2>&1; echo EXIT \$? >> scratch/report_speed/out/t2.log"
# after EXIT 0:
/opt/venv/reconstruction/bin/python scratch/report_speed/compare_reports.py scratch/report_speed/out/baseline.json scratch/report_speed/out/t2.json
```

Expected: `PASS`. Report `_collect_pairs` seconds against the baseline.

- [ ] **Step 6: Commit**

```bash
git add tests/geometry/test_metrics.py collab_splats/geometry/metrics.py
git commit --only tests/geometry/test_metrics.py collab_splats/geometry/metrics.py -m "perf(metrics): residual histogram on the GPU via bucketize + bincount

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `depth_agreement` returns `expected`; drop the redundant transform

**Files:**
- Modify: `collab_splats/geometry/projection.py:143-173` (`depth_agreement`), `:235` (`multiview_depth_confidence`)
- Modify: `collab_splats/geometry/metrics.py` (`_collect_pairs`, imports)
- Test: `tests/geometry/test_projection.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/geometry/test_projection.py` (reuse its existing imports; add `depth_agreement`
and `project` to the `collab_splats.geometry.projection` import if missing):

```python
def test_depth_agreement_returns_the_camera_depth_project_computes():
    """expected is project's camera z, so callers need no second transform."""
    rng = np.random.default_rng(3)
    points = torch.as_tensor(rng.uniform(-1, 1, (500, 3)) + [0, 0, 4], dtype=torch.float32)
    w2c = torch.eye(4)
    w2c[0, 3] = 0.3
    K = torch.tensor([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]])
    depth = torch.full((16, 16), 4.0)

    *_, expected = depth_agreement(points, w2c, K, depth, 0.05)
    _, points_cam = project(points, w2c, K)
    assert torch.equal(expected, points_cam[:, 2])
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_projection.py -q -k returns_the_camera_depth`
Expected: FAIL. `expected` binds to the 3rd element (`rel`), so `torch.equal` is False.

- [ ] **Step 3: Implement**

In `depth_agreement`, change the return annotation to `tuple[Tensor, Tensor, Tensor, Tensor]`,
the Returns block to:

```
    Returns:
        (agree, seen, rel_residual, expected), each (P,)
        - rel_residual: (sampled - expected) / expected; exactly -1 where sampled is 0
        - expected: the point's z in the camera
```

and the last line to:

```python
    return agree, seen, (sampled - expected) / expected, expected
```

In `multiview_depth_confidence`:

```python
            agree_ij, seen_ij, _, _ = depth_agreement(points, extrinsics_t[j], intrinsics_t[j], depth_t[j], rel_thresh)
```

In `_collect_pairs`:

```python
            agree, seen, rel, z = depth_agreement(points, extrinsics_t[j], intrinsics_t[j], depth_t[j], rel_thresh)
```

and delete the line `z = transform_points(points, extrinsics_t[j])[:, 2]`. Remove
`transform_points` from the `collab_splats.geometry.transforms` import in metrics.py (keep
`invert_poses`).

- [ ] **Step 4: Run geometry tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry -q`
Expected: all pass (same counts as Task 0 plus the new tests).

- [ ] **Step 5: Parity on GH010229**

Same tmux command as Task 2 Step 5 with label `t3`; compare `baseline.json` vs `t3.json`. Expected: `PASS`.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/projection.py collab_splats/geometry/metrics.py tests/geometry/test_projection.py
git commit --only collab_splats/geometry/projection.py collab_splats/geometry/metrics.py tests/geometry/test_projection.py -m "perf(metrics): reuse depth_agreement's camera depth instead of re-transforming

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: One sync per pair, one transfer per source frame

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (`_collect_pairs`)

No new test: behavior is unchanged. The existing `_collect_pairs` tests plus the GH010229
parity run are the gate.

- [ ] **Step 1: Restructure the inner loop**

Replace the body from `sel = seen & ...` through the `collected["pairs"].append(...)` block, and
the agreement lines after the `j` loop, with:

```python
            # Residuals over seen pixels with a sampled depth, clear of camera j's center
            # - rel > -1 keeps sampled > 0 only: sampled <= 0 is no measurement
            # - near-zero z: the quotient and the parallax both degenerate
            sel = seen & (rel > -1) & (z > 1e-6)
            n_sel = int(sel.sum())
            if n_sel == 0:
                continue

            rel_sel = rel[sel]

            # Parallax from ray directions: scale-free, needs no focal length
            ray_i = points[sel] - centers[i]
            ray_j = points[sel] - centers[j]
            cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
            parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # Per-pixel residuals into the one device histogram; pair stats stay on the device
            counts += _histogram_counts(bounded_residual(rel_sel), edges_t)
            q = torch.quantile(rel_sel, quantiles)
            rows.append(torch.stack([q[1], q[2] - q[0], parallax.median(), z[sel].median()]))
            row_keys.append((j, n_sel))

        # One transfer per source frame for its pair rows and its agreement counts
        if rows:
            stats = torch.stack(rows).tolist()
            for (j, n_sel), (q50, iqr, par, med_z) in zip(row_keys, stats):
                collected["pairs"].append(
                    PairStats(
                        idx1=i,
                        idx2=j,
                        n_pixels=n_sel,
                        median_rel_depth_error=q50,
                        iqr_rel_depth_error=iqr,
                        median_parallax_deg=par,
                        median_depth=med_z,
                    )
                )

        # Share of this frame's seen pixels that any other view agrees with
        n_agree, n_seen = torch.stack([any_agree.sum(), any_seen.sum()]).tolist()
        agreement.append(n_agree / n_seen if n_seen else None)
```

At the top of the `i` loop body (after `any_seen = ...`) add:

```python
        rows: list[Tensor] = []
        row_keys: list[tuple[int, int]] = []
```

And before the `i` loop (after `agreement = []`):

```python
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)
```

`.tolist()` on a float32 tensor yields the same Python floats as `float(q[1])`; `q[2] - q[0]` is
still the float32 subtraction on the device. Both keep the report bit-identical.

- [ ] **Step 2: Run tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/geometry/test_metrics_controls.py -q`
Expected: all pass.

- [ ] **Step 3: Parity on GH010229**

Same tmux command as Task 2 Step 5 with label `t4`; compare `baseline.json` vs `t4.json`. Expected: `PASS`.

- [ ] **Step 4: Commit**

```bash
git commit --only collab_splats/geometry/metrics.py -m "perf(metrics): one GPU sync per pair, pair rows transferred once per source frame

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: O(pairs) per-frame medians

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (new `_frame_medians`, `compute_reconstruction_quality`)
- Test: `tests/geometry/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Append near the other `compute_reconstruction_quality` tests:

```python
def test_frame_medians_equal_the_per_frame_scan():
    """One bucketing pass gives exactly what scanning every pair per frame gave."""
    rng = np.random.default_rng(2)
    n = 12
    pairs = [
        _pair(int(i), int(j), float(rng.normal(0, 0.05)), 5.0)
        for i, j in rng.integers(0, n, size=(80, 2))
        if i != j
    ]

    # The pre-change O(N * pairs) scan, frame 11 may well be untouched
    scan = []
    for k in range(n):
        v = [abs(p.median_rel_depth_error) for p in pairs if k in (p.idx1, p.idx2)]
        scan.append(float(np.median(v)) if v else None)

    assert metrics._frame_medians(pairs, n) == scan
    assert metrics._frame_medians([], 3) == [None, None, None]
```

- [ ] **Step 2: Run to see it fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -q -k frame_medians_equal`
Expected: FAIL with `AttributeError: ... '_frame_medians'`.

- [ ] **Step 3: Implement**

Add above `compute_reconstruction_quality` (in the "Per frame + assembly" section):

```python
def _frame_medians(pairs: list[PairStats], n: int) -> list[float | None]:
    """
    Per frame, the median |median_rel_depth_error| over the pairs touching it.

    - one pass buckets every pair under both its frames: O(pairs), not O(N * pairs)
    - None for a frame no pair touches
    """
    touching: list[list[float]] = [[] for _ in range(n)]

    for p in pairs:
        v = abs(p.median_rel_depth_error)
        touching[p.idx1].append(v)
        touching[p.idx2].append(v)

    return [float(np.median(v)) if v else None for v in touching]
```

In `compute_reconstruction_quality`, replace the per-frame medians block (comment, `median_abs = []`
and the `tqdm` loop) with:

```python
    # Per-frame median |residual| over the depth pairs touching each frame
    median_abs = _frame_medians(collected["pairs"], n)
```

- [ ] **Step 4: Run tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -q`
Expected: all pass, including `test_median_abs_rel_depth_error_is_...` (it monkeypatches
`_collect_pairs`, which still feeds `_frame_medians`).

- [ ] **Step 5: Parity on GH010229**

Label `t5`; compare `baseline.json` vs `t5.json`. Expected: `PASS`.

- [ ] **Step 6: Commit**

```bash
git add tests/geometry/test_metrics.py collab_splats/geometry/metrics.py
git commit --only tests/geometry/test_metrics.py collab_splats/geometry/metrics.py -m "perf(metrics): per-frame medians in one O(pairs) pass

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Leading pose batch in `transform_points`, `project`, `depth_residual`

**Files:**
- Modify: `collab_splats/geometry/transforms.py:74-89`
- Modify: `collab_splats/geometry/projection.py:65-140`
- Test: `tests/geometry/test_transforms.py`, `tests/geometry/test_projection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_transforms.py` (ensure `torch` and `transform_points` are imported):

```python
def test_transform_points_maps_one_point_set_through_a_pose_batch():
    """(B, 4, 4) poses over (P, 3) points give (B, P, 3), each slice the 2-D result."""
    rng = np.random.default_rng(0)
    points = torch.as_tensor(rng.standard_normal((50, 3)), dtype=torch.float32)
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, 3] = torch.as_tensor(rng.standard_normal((3, 3)), dtype=torch.float32)
    poses[1, :3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

    out = transform_points(points, poses)
    assert out.shape == (3, 50, 3)
    for b in range(3):
        torch.testing.assert_close(out[b], transform_points(points, poses[b]), rtol=0, atol=1e-6)
```

Append to `tests/geometry/test_projection.py` (add `depth_residual` to the import if missing):

```python
def _views(b=4, hw=(12, 16), seed=0):
    """b cameras around a noisy depth field: poses, K, depth, and world points to project."""
    rng = np.random.default_rng(seed)
    h, w = hw
    K = torch.tensor([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1.0]]).repeat(b, 1, 1)
    w2c = torch.eye(4).repeat(b, 1, 1)
    w2c[:, :3, 3] = torch.as_tensor(rng.uniform(-0.3, 0.3, (b, 3)), dtype=torch.float32)
    depth = torch.as_tensor(rng.uniform(3, 5, (b, h, w)), dtype=torch.float32)
    depth[:, 0, :3] = 0.0
    points = torch.as_tensor(rng.uniform(-1, 1, (400, 3)) + [0, 0, 4], dtype=torch.float32)
    return points, w2c, K, depth


def test_project_over_a_pose_batch_matches_one_camera_at_a_time():
    points, w2c, K, _ = _views()
    pixels, cam = project(points, w2c, K)
    assert pixels.shape == (4, 400, 2) and cam.shape == (4, 400, 3)
    for b in range(4):
        p1, c1 = project(points, w2c[b], K[b])
        torch.testing.assert_close(pixels[b], p1, rtol=0, atol=1e-5)
        torch.testing.assert_close(cam[b], c1, rtol=0, atol=1e-6)


def test_depth_residual_over_a_pose_batch_matches_one_view_at_a_time():
    points, w2c, K, depth = _views()
    residual, expected, sampled, valid, pixels = depth_residual(points, w2c, K, depth)
    assert residual.shape == (4, 400) and pixels.shape == (4, 400, 2)
    for b in range(4):
        r1, e1, s1, v1, _ = depth_residual(points, w2c[b], K[b], depth[b])
        torch.testing.assert_close(expected[b], e1, rtol=0, atol=1e-6)
        torch.testing.assert_close(sampled[b], s1, rtol=0, atol=0)
        assert torch.equal(valid[b], v1)
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_transforms.py tests/geometry/test_projection.py -q -k "pose_batch"`
Expected: FAIL. `T[:3, :3]` slices the batch axis, giving shape errors.

- [ ] **Step 3: Implement `transform_points`**

The 2-D path keeps its exact expression, so existing callers are bit-identical.

```python
def transform_points(points: np.ndarray | Tensor, T: np.ndarray | Tensor) -> np.ndarray | Tensor:
    """
    Points moved by a rigid transform: R @ p + t.

    - reads only the 3x4 block, so a (4, 4) bottom row must be [0, 0, 0, 1]
    - numpy or torch, not mixed; dtype follows the library's promotion of the inputs
    - torch: differentiable in both arguments
    - a (B, 4, 4) batch maps (P, 3) or (B, P, 3) points to (B, P, 3)

    Args:
        points: (..., 3) points, or (P, 3) / (B, P, 3) with a pose batch.
        T: (4, 4) or (3, 4) rigid transform, or a (B, 4, 4) / (B, 3, 4) batch of them.

    Returns:
        (..., 3) transformed points; (B, P, 3) for a pose batch.
    """
    if T.ndim == 2:
        return points @ T[:3, :3].T + T[:3, 3]

    # Pose batch: one matmul per pose, translation broadcast over the points
    return points @ T[..., :3, :3].swapaxes(-1, -2) + T[..., None, :3, 3]
```

- [ ] **Step 4: Implement `project` and `depth_residual`**

In `project`, update Args (`world_to_cam: (4, 4) or (3, 4) w2c, or a (B, 4, 4) batch.`,
`intrinsics: (3, 3) camera matrix, or a (B, 3, 3) batch.`), Returns
(`(pixels (..., 2), camera-frame points (..., 3)); (B, P, ...) for a batch.`), and the divide:

```python
    # Clamped divide: unclamped, a mask's 0 * inf poisons the backward
    depth = points_cam[..., 2].clamp(min=min_depth)
    fx, fy = intrinsics[..., 0, 0, None], intrinsics[..., 1, 1, None]
    cx, cy = intrinsics[..., 0, 2, None], intrinsics[..., 1, 2, None]
    u = points_cam[..., 0] * fx / depth + cx
    v = points_cam[..., 1] * fy / depth + cy
    return torch.stack([u, v], dim=-1), points_cam
```

In `depth_residual`, update Args (`world_to_cam: (4, 4) w2c ..., or a (B, 4, 4) batch.`,
`intrinsics: ... or a (B, 3, 3) batch.`, `depth: (H, W) ..., or a (B, H, W) batch.`) and add a
docstring bullet `- a batch of B views returns each output with a leading B`. Body:

```python
    height, width = depth.shape[-2:]

    # World -> pixel; project's clamped divide keeps points behind the camera finite
    pixels, points_cam = project(points_world, world_to_cam, intrinsics)
    expected = points_cam[..., 2]
    in_front = expected > 0

    # Pixels -> grid_sample's [-1, 1] frame, corners on the outer pixel centers
    grid_u = pixels[..., 0] / (width - 1) * 2 - 1
    grid_v = pixels[..., 1] / (height - 1) * 2 - 1
    grid = torch.stack([grid_u, grid_v], dim=-1)
    in_bounds = (grid[..., 0] >= -1) & (grid[..., 0] <= 1) & (grid[..., 1] >= -1) & (grid[..., 1] <= 1)

    # Nearest read of each view's depth map at its projected pixels
    depth_map = depth.reshape(-1, 1, height, width)
    grid = grid.reshape(depth_map.shape[0], 1, -1, 2)
    sampled = F.grid_sample(depth_map, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(expected.shape)

    residual = expected - sampled
    return residual, expected, sampled, in_front & in_bounds, pixels
```

Update `depth_agreement`'s Args the same way (it is elementwise, so no body change).

- [ ] **Step 5: Run geometry tests, then prove the 2-D path is bit-identical**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry -q`
Expected: all pass.

Parity on GH010229, label `t6`; compare `baseline.json` vs `t6.json`. Expected: `PASS`. `_collect_pairs`
still calls the 2-D path, so this proves the 2-D rewrite of `project` / `depth_residual` changed nothing.

- [ ] **Step 6: Commit**

```bash
git add collab_splats/geometry/transforms.py collab_splats/geometry/projection.py tests/geometry/test_transforms.py tests/geometry/test_projection.py
git commit --only collab_splats/geometry/transforms.py collab_splats/geometry/projection.py tests/geometry/test_transforms.py tests/geometry/test_projection.py -m "feat(geometry): transform_points, project and depth_residual take a leading pose batch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Batched target views in `_collect_pairs` (gated)

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (`_collect_pairs`, imports)
- Test: `tests/geometry/test_metrics.py`

**Gate:** the GH010229 parity run must be `PASS` (bit-identical integer columns and histogram,
floats within 1e-6). If it is not, `git checkout -- collab_splats/geometry/metrics.py tests/geometry/test_metrics.py`
and skip to Task 8. Record in the report which column differed and by how much, plus the
`_collect_pairs` seconds batched vs `t6`. Task 11 writes that into the spec.

- [ ] **Step 1: Write the failing test**

```python
def test_collect_pairs_is_identical_whatever_the_target_batch():
    """Chunking target views changes the kernel shapes, never a pair row or a histogram count."""
    depth, K, extr = _two_view(scale_j=1.03)
    depth = np.concatenate([depth, depth[:1] * 1.01])
    K = np.concatenate([K, K[:1]])
    extr = np.concatenate([extr, extr[1:]])
    extr[2, 1, 3] = 0.1

    one, agree_one = metrics._collect_pairs(depth, K, extr, 0.05, target_batch=1)
    many, agree_many = metrics._collect_pairs(depth, K, extr, 0.05, target_batch=2)
    assert agree_one == agree_many
    np.testing.assert_array_equal(one["rel_depth_error_counts"], many["rel_depth_error_counts"])
    assert one["pairs"] == many["pairs"]
```

`PairStats` is a dataclass, so `==` compares every field exactly.

- [ ] **Step 2: Run to see it fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -q -k whatever_the_target_batch`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'target_batch'`.

- [ ] **Step 3: Implement**

Add `from collab_splats.utils.torch_utils import get_device, infer_batch_size` (replacing the
`get_device`-only import). New signature and docstring bullet:

```python
def _collect_pairs(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    rel_thresh: float,
    target_batch: int | None = None,
    bytes_per_point: int = 160,
) -> tuple[dict, list[float | None]]:
    """
    Per-pair depth stats, the residual histogram and per-frame agreement, in one O(N^2) pass.

    - collected: "pairs" (list[PairStats]), "rel_depth_error_counts", "rel_depth_error_edges"
    - agreement: share of seen pixels with at least one agreeing view; None when nothing is seen
    - residuals only where seen: occlusion is absent evidence, not disagreement
    - target views run target_batch at a time; None sizes the batch from VRAM at bytes_per_point
    """
```

After `agreement = []` / `quantiles = ...`:

```python
    # Target views per batch: VRAM-sized unless given, never more than the other frames
    if target_batch is None:
        target_batch = infer_batch_size(h * w * bytes_per_point / 1024**3)
    target_batch = max(1, min(target_batch, n - 1))
```

Replace the whole `for j in range(n):` loop with:

```python
        targets = [j for j in range(n) if j != i]

        for start in range(0, len(targets), target_batch):
            js = targets[start : start + target_batch]
            agree, seen, rel, z = depth_agreement(
                points, extrinsics_t[js], intrinsics_t[js], depth_t[js], rel_thresh
            )
            seen &= has_source
            any_agree |= (agree & has_source).any(0)
            any_seen |= seen.any(0)

            # Residuals over seen pixels with a sampled depth, clear of camera j's center
            # - rel > -1 keeps sampled > 0 only: sampled <= 0 is no measurement
            # - near-zero z: the quotient and the parallax both degenerate
            sel = seen & (rel > -1) & (z > 1e-6)
            n_sel = sel.sum(1).tolist()

            for b, j in enumerate(js):
                if n_sel[b] == 0:
                    continue

                sel_b = sel[b]
                rel_sel = rel[b][sel_b]

                # Parallax from ray directions: scale-free, needs no focal length
                ray_i = points[sel_b] - centers[i]
                ray_j = points[sel_b] - centers[j]
                cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
                parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

                # Per-pixel residuals into the one device histogram; pair stats stay on the device
                counts += _histogram_counts(bounded_residual(rel_sel), edges_t)
                q = torch.quantile(rel_sel, quantiles)
                rows.append(torch.stack([q[1], q[2] - q[0], parallax.median(), z[b][sel_b].median()]))
                row_keys.append((j, n_sel[b]))
```

(`depth_agreement` now hands back `(B, P)` tensors; the rest of the `i` loop body from Task 4 is unchanged.)

- [ ] **Step 4: Run tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry -q`
Expected: all pass.

- [ ] **Step 5: Gate on GH010229**

Label `t7`; compare `baseline.json` vs `t7.json`. `PASS` means commit. A failure means revert and record, as described above.

- [ ] **Step 6: Commit (only on PASS)**

```bash
git add tests/geometry/test_metrics.py collab_splats/geometry/metrics.py
git commit --only tests/geometry/test_metrics.py collab_splats/geometry/metrics.py -m "perf(metrics): batch target views through depth_residual

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: kornia guided filter in `upsample_depths`

**Files:**
- Modify: `collab_splats/utils/image.py` (module docstring, imports, depth-upsampling section)
- Test: `tests/utils/test_image.py`

- [ ] **Step 1: Confirm kornia's channel contract**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "
import torch; from kornia.filters import guided_blur
g = torch.rand(1, 1, 40, 50); x = torch.rand(1, 2, 40, 50)
y = guided_blur(g, x, 7, 1e-3)
y0 = guided_blur(g, x[:, :1], 7, 1e-3)
print(y.shape, torch.equal(y[:, :1], y0))"
```

Expected: `torch.Size([1, 2, 40, 50]) True`. That means a gray guide filters a 2-channel input
channel by channel. If it prints `False` or raises, call `guided_blur` twice (num, then den) in Step 4.

- [ ] **Step 2: Write the failing parity test**

Append to `tests/utils/test_image.py` (add `import cv2` to the third-party group):

```python
def _oracle_guided_filter(guide, src, radius, eps):
    """He et al. gray-guide guided filter on cv2 box means, BORDER_REFLECT_101 like torch reflect."""

    def box(x):
        return cv2.boxFilter(x, -1, (2 * radius + 1,) * 2, normalize=True, borderType=cv2.BORDER_REFLECT_101)

    mean_g, mean_s = box(guide), box(src)
    a = (box(guide * src) - mean_g * mean_s) / (box(guide * guide) - mean_g * mean_g + eps)
    b = mean_s - a * mean_g
    return box(a) * guide + box(b)


def test_upsample_depths_matches_a_cv2_guided_filter_oracle():
    """The kornia filter is the He et al. filter; a kornia upgrade that changes it fails here."""
    rng = np.random.default_rng(0)
    H, W, h, w = 48, 64, 12, 16
    rgb = rng.integers(0, 256, (1, H, W, 3), dtype=np.uint8)
    depth = np.where(np.arange(w)[None, :] < w // 2, 3.0, 5.0).astype(np.float32)[None].repeat(h, 1)
    depth[0, 2:4, 2:5] = 0.0
    box = np.array([[0, 0, W, H]])

    got = upsample_depths(depth, rgb, box)[0]

    # Same validity-weighted recipe as the module, with the cv2 oracle as the filter
    depth_nn = cv2.resize(depth[0], (W, H), interpolation=cv2.INTER_NEAREST)
    valid = (depth_nn > 0).astype(np.float32)
    guide = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    radius = int(np.ceil(2 * W / w))
    num = _oracle_guided_filter(guide, depth_nn * valid, radius, 1e-3)
    den = _oracle_guided_filter(guide, valid, radius, 1e-3)
    want = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    want[valid == 0] = 0.0
    np.maximum(want, 0.0, out=want)

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-3)
```

- [ ] **Step 3: Run to see it fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/utils/test_image.py -q -k cv2_guided_filter_oracle`
Expected: FAIL in the 2r border band (the current filter uses `BORDER_REFLECT`), max |Δ| about 0.1.

- [ ] **Step 4: Implement**

Module docstring first line becomes `Image helpers: PIL coercion, guided depth upsampling, hole filling.`;
the `upsample_depths` bullet becomes `- upsample_depths: model-res depth onto the original-res RGB grid (kornia guided filter)`.
Imports (third-party group, then ours):

```python
import cv2
import numpy as np
import torch
from kornia.filters import guided_blur
from PIL import Image

from collab_splats.utils.torch_utils import get_device
```

Delete `_box` and `_guided_filter`. Replace `_guided_upsample_depth` and `upsample_depths`:

```python
def _guided_upsample_depth(
    depth: np.ndarray,
    rgb_full: np.ndarray,
    crop_box: np.ndarray,
    device: torch.device,
    radius: int | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Upsample one model-res depth map into its crop region of the original-res RGB canvas.

    - crop_box (tl_x, tl_y, cr_x, cr_y) in original pixels; radius None = ~2 × the upsample factor
    - masked pixels stay 0, canvas outside the crop is 0
    - kornia guided_blur on device: He et al., gray guide, reflect-101 border
    """
    H, W = rgb_full.shape[:2]
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y
    if cw <= 0 or ch <= 0:
        raise ValueError(f"Degenerate crop box {crop_box} — original_coords are corrupt")
    if tl_x < 0 or tl_y < 0 or cr_x > W or cr_y > H:
        raise ValueError(f"Crop box {crop_box} lies outside the {H}x{W} canvas")

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: depth and validity as two channels of one guided filter
    guide_t = torch.as_tensor(guide, device=device)[None, None]
    src_t = torch.as_tensor(np.stack([depth_nn * valid_nn, valid_nn]), device=device)[None]
    num, den = guided_blur(guide_t, src_t, 2 * radius + 1, eps)[0]
    filtered = torch.where(den > 1e-6, num / den.clamp(min=1e-6), 0.0)

    # The guide must never resurrect deleted depth, and depth must stay non-negative
    filtered[src_t[0, 1] == 0] = 0.0
    filtered = filtered.clamp(min=0.0).cpu().numpy()

    canvas = np.zeros((H, W), dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas


def upsample_depths(depths: np.ndarray, rgbs: np.ndarray, crop_boxes: np.ndarray) -> np.ndarray:
    """
    Guided-filter upsample model-res depth maps onto their original-res RGB frames.

    - filters on get_device(), one frame at a time

    Args:
        depths: (N, h, w) model-res depth, 0 = no observation.
        rgbs: (N, H, W, 3) uint8 original-res frames; H, W set the output size.
        crop_boxes: (N, 4) [tl_x, tl_y, cr_x, cr_y] model crops in original pixels
            (original_coords[:, :4]).

    Returns:
        (N, H, W) float32 depth at frame resolution.
    """
    depths = np.asarray(depths)
    rgbs = np.asarray(rgbs)
    crop_boxes = np.asarray(crop_boxes)
    if not (len(depths) == len(rgbs) == len(crop_boxes)):
        raise ValueError(f"{len(depths)} depths, {len(rgbs)} rgbs, {len(crop_boxes)} crop boxes")

    # One guided upsample per frame into a preallocated stack
    device = torch.device(get_device())
    n, H, W = len(depths), rgbs.shape[1], rgbs.shape[2]
    out = np.zeros((n, H, W), dtype=np.float32)

    for i in range(n):
        out[i] = _guided_upsample_depth(np.asarray(depths[i], dtype=np.float32), rgbs[i], crop_boxes[i], device)

    return out
```

- [ ] **Step 5: Run every `upsample_depths` consumer's tests**

Run:

```bash
cd /workspace/collab-splats/.worktrees/report-speed
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/utils tests/geometry tests/wrapper/test_reconstructor.py tests/wrapper/test_splats_stage.py tests/wrapper/test_sfm_stage.py tests/wrapper/test_mask_sky.py tests/wrapper/test_absent_confidence.py tests/mesh -q
```

Expected: all pass. A failure that compares an upsample against hard-coded numbers inside the 2r
border band is the accepted border change (spec option a). Update the number, and quote the old
and new values in the report. Any failure away from the border is a bug. Stop and report it.

- [ ] **Step 6: Photometric delta on GH010229**

Label `t8`; run `compare_reports.py <last-PASS label>.json t8.json --photometric report`. Expected:
depth columns and histogram identical (PASS). Report the photometric `ncc` max |Δ| and the
`n_pixels` delta range, plus the `upsample_depths` seconds against the previous run.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/utils/image.py tests/utils/test_image.py
git commit --only collab_splats/utils/image.py tests/utils/test_image.py -m "perf(utils): guided depth upsample via kornia guided_blur on the GPU

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

(If Step 5 forced fixture-number updates in other test files, add those paths to both commands.)

---

### Task 9: Streaming photometric pass, uint8 images

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (`compute_photometric_ncc`)
- Modify: `collab_splats/wrapper/reconstructor.py:1463`
- Test: `tests/geometry/test_metrics.py` (spy test update + new uint8 test)

- [ ] **Step 1: Update the upsample-spy test and add a uint8 test**

In `test_the_upsample_guide_is_normalized_whatever_the_backbones_image_scale`, replace the two
assertion lines after the two `compute_photometric_ncc` calls with:

```python
    # One call per compute, lifting only frame 0: frame 1 has no partner at max_separation=1
    assert len(lifted) == 2
    assert lifted[0].shape == (1, 64, 64) and lifted[1].shape == (1, 64, 64)
```

(The `assert_array_equal` and the flat-guide anchor that follow stay as they are.)

Append:

```python
def test_photometric_ncc_takes_uint8_images_as_is():
    """The stage hands over uint8 frames; the result equals the float [0, 255] input's."""
    img, depth, K, e = _translated_pair(shift_px=4, hw=32)
    as_float = compute_photometric_ncc(img.round(), depth, K, e, max_separation=1)
    as_uint8 = compute_photometric_ncc(img.round().astype(np.uint8), depth, K, e, max_separation=1)
    assert as_uint8["n_pixels"] == as_float["n_pixels"]
    np.testing.assert_allclose(as_uint8["photometric_ncc"], as_float["photometric_ncc"], rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run to see the spy test fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py -q -k "upsample_guide_is_normalized or uint8_images"`
Expected: the spy test FAILS (`(2, 64, 64) != (1, 64, 64)`); the uint8 test may already pass.

- [ ] **Step 3: Rewrite `compute_photometric_ncc`'s body**

Docstring changes:
- `images: (N, H, W, 3) RGB at original resolution; uint8 or float, [0, 255] or [0, 1].`
- `- zero-mean NCC via torch.corrcoef in float64: 1.0 is perfect agreement, 0.0 is none`
- add `- streams: frame i's depth is upsampled only when it has a partner, one frame at a time`
- add `- float32 projection on get_device(); round-to-nearest pixels, bounds [0, W) x [0, H)`

Body from `t0 = ...` to the end:

```python
    t0 = time.perf_counter()
    N = len(depth)
    ih, iw = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, max_separation=%d", N, iw, ih, max_separation)

    # Model-grid depth is lifted to the image grid frame by frame below
    lift = depth.shape[1:] != (ih, iw)
    if lift:
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(ih, iw)}; " "original_coords is required to upsample"
            )

        # Crop boxes are in original pixels, so `images` must be that canvas
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))
        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )

        # Guide scale decided once over the whole array, never per frame
        # - a per-frame decision would amplify a dark [0, 255] frame 255x
        rgb_scale = 255.0 if images.max() <= 1.0 else 1.0

    # Poses and K in float32 on the device for unproject / project
    device = torch.device(get_device())
    world_to_cam = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    H, W = ih, iw

    # Closed-form pair count, so the bar states the real unit of work
    n_pairs_expected = sum(min(N, i + max_separation + 1) - (i + 1) for i in range(N))
    cols: dict[str, list] = {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}

    for i in tqdm(range(N), desc=f"Photometric NCC ({n_pairs_expected} pairs)", unit="frame"):
        j_end = min(N, i + max_separation + 1)
        if j_end == i + 1:
            continue

        # Frame i's depth on the image grid; the guide must be uint8 [0, 255]
        depth_i = depth[i]
        if lift:
            guide = np.clip(np.asarray(images[i]) * rgb_scale, 0, 255).astype(np.uint8)
            depth_i = upsample_depths(depth[i : i + 1], guide[None], original_coords[i : i + 1, :4])[0]

        # Unproject frame i's pixels to world; depth == 0 is no observation
        depth_t = torch.as_tensor(depth_i, dtype=torch.float32, device=device)
        pts_world = unproject(depth_t, world_to_cam[i], intrinsics_t[i]).reshape(-1, 3)
        has_depth = depth_t.reshape(-1) > 0

        # Frame i and its partners uploaded once per source frame
        window = torch.as_tensor(np.asarray(images[i:j_end]), device=device)
        colors_i = window[0].reshape(-1, 3)

        for j in range(i + 1, j_end):
            # Project into frame j; no occlusion test, occluded pixels read as disagreement
            pixels, pts_cam_j = project(pts_world, world_to_cam[j], intrinsics_t[j])

            # Nearest sampling, matching the depth pass; round half to even like np.round
            ui = torch.round(pixels[:, 0]).long()
            vi = torch.round(pixels[:, 1]).long()
            ok = (pts_cam_j[:, 2] > 0) & has_depth
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            n_ok = int(ok.sum())
            if n_ok < min_samples:
                continue

            # Paired RGB samples: frame i's pixel against where it landed in frame j
            a = colors_i[ok].reshape(-1).double()
            b = window[j - i][vi[ok], ui[ok]].reshape(-1).double()

            # Skip flat patches: no variance to correlate
            if a.std(correction=0) < 1e-8 or b.std(correction=0) < 1e-8:
                continue

            ncc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            if not np.isfinite(ncc):
                continue

            for col, v in zip(cols, (i, j, ncc, n_ok)):
                cols[col].append(v)

    logger.info(
        "Photometric NCC: %d pairs correlated in %.2fs", len(cols["photometric_ncc"]), time.perf_counter() - t0
    )
    return cols
```

- [ ] **Step 4: Stage passes uint8**

In `collab_splats/wrapper/reconstructor.py`, `reconstruction_quality_report`:

```python
            images = _scene_frames(self.images_dir)[_store_rows(self.images_dir, ff.image_paths)]
```

`_scene_frames` returns `frames.read_frames(images_dir)`, documented (N, H, W, 3) uint8; no cast needed.

- [ ] **Step 5: Run tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry tests/wrapper/test_reconstructor.py -q`
Expected: all pass. A photometric test that compares exact pixel counts and now differs by a few
pixels points at float32 rounding at .5 boundaries. Report it with the numbers; do not loosen a
test silently.

- [ ] **Step 6: Photometric delta and memory on GH010229**

Label `t9`; `compare_reports.py t8.json t9.json --photometric report`. Expected: depth columns and
histogram PASS; report `ncc` max |Δ| (expect ≤1e-3) and the `n_pixels` delta range, plus
`compute_photometric_ncc` seconds and `peak_rss_gb` against `t8`.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/wrapper/reconstructor.py tests/geometry/test_metrics.py
git commit --only collab_splats/geometry/metrics.py collab_splats/wrapper/reconstructor.py tests/geometry/test_metrics.py -m "perf(metrics): stream the photometric pass on the GPU, uint8 frames end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `min_pair_overlap` pruning knob (default off)

**Files:**
- Modify: `collab_splats/geometry/metrics.py` (new `_frustum_overlap`, `_collect_pairs`, `compute_reconstruction_quality`)
- Modify: `collab_splats/wrapper/reconstructor.py` (`reconstruction_quality_report`)
- Modify: `configs/base.yaml`
- Test: `tests/geometry/test_metrics.py`, `tests/wrapper/test_reconstructor.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/geometry/test_metrics.py`:

```python
def _three_views_one_facing_away():
    """Frames 0 and 1 share a view of a plane; frame 2 sits at the origin facing -z."""
    depth, K, extr = _two_view(scale_j=1.02)
    away = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32)
    return (np.concatenate([depth, depth[:1]]), np.concatenate([K, K[:1]]), np.concatenate([extr, away[None]]))


def test_frustum_overlap_is_one_for_the_same_camera_and_zero_facing_away():
    depth, K, extr = _three_views_one_facing_away()
    d, k, e = (torch.as_tensor(x) for x in (depth, K, extr))
    points = metrics.unproject(d[0], e[0], k[0]).reshape(-1, 3)
    overlap = metrics._frustum_overlap(points, e, k, *depth.shape[1:])
    assert overlap[0] == 1.0
    assert 0.0 < overlap[1] <= 1.0
    assert overlap[2] == 0.0


def test_min_pair_overlap_zero_is_the_unpruned_pass():
    depth, K, extr = _three_views_one_facing_away()
    a, agree_a = metrics._collect_pairs(depth, K, extr, 0.05)
    b, agree_b = metrics._collect_pairs(depth, K, extr, 0.05, min_pair_overlap=0.0)
    assert a["pairs"] == b["pairs"] and agree_a == agree_b
    np.testing.assert_array_equal(a["rel_depth_error_counts"], b["rel_depth_error_counts"])


def test_min_pair_overlap_drops_only_pairs_that_share_no_view():
    """Frame 2 sees nothing of 0 or 1, so pruning it changes no row, count or agreement."""
    depth, K, extr = _three_views_one_facing_away()
    full, agree_full = metrics._collect_pairs(depth, K, extr, 0.05)
    pruned, agree_pruned = metrics._collect_pairs(depth, K, extr, 0.05, min_pair_overlap=0.01)
    assert pruned["pairs"] == full["pairs"] and agree_pruned == agree_full
    np.testing.assert_array_equal(pruned["rel_depth_error_counts"], full["rel_depth_error_counts"])

    # Anchor: the unpruned pass did visit frame 2's pairs; pruning had something to skip
    assert {(p.idx1, p.idx2) for p in full["pairs"]} == {(0, 1), (1, 0)}


def test_min_pair_overlap_above_a_pairs_overlap_drops_that_pair():
    depth, K, extr = _three_views_one_facing_away()
    pruned, _ = metrics._collect_pairs(depth, K, extr, 0.05, min_pair_overlap=1.01)
    assert pruned["pairs"] == []
```

`torch` is already in the top import group (Task 2); `metrics.unproject` resolves because metrics imports it at module level.

Append to `tests/wrapper/test_reconstructor.py`, near `test_report_json_is_the_columnar_contract`:

```python
def test_report_scene_block_records_min_pair_overlap(tmp_path):
    """The pruning knob is part of how the numbers were made, so the report carries it."""
    rec = Reconstructor(_make_config(tmp_path, {"reconstruction_quality_report": {"min_pair_overlap": 0.05}}))
    rec._resolve_result = lambda: object()
    _save_tiny_zarr(rec, with_confidence=False)

    report = json.loads(rec.reconstruction_quality_report().read_text())
    assert report["scene"]["min_pair_overlap"] == 0.05
```

Check how `_make_config` merges `overrides` first: `rtk proxy sed -n 101,135p tests/wrapper/test_reconstructor.py`.
If it shallow-updates, the dict above is correct as written.

- [ ] **Step 2: Run to see them fail**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry/test_metrics.py tests/wrapper/test_reconstructor.py -q -k "frustum_overlap or min_pair_overlap"`
Expected: FAIL (`_frustum_overlap` missing, unexpected keyword `min_pair_overlap`, `KeyError: 'min_pair_overlap'`).

- [ ] **Step 3: Implement `_frustum_overlap`**

Add above `_collect_pairs`:

```python
def _frustum_overlap(points: Tensor, extrinsics: Tensor, intrinsics: Tensor, h: int, w: int) -> Tensor:
    """
    Share of points inside each camera's frustum, one batched projection into all N cameras.

    - in front and inside the pixel-center span [0, w-1] x [0, h-1], as depth_residual bounds it
    - ignores occlusion, so it over-estimates overlap: pruning on it keeps extra pairs
    - (N,) float; 0 for an empty point set
    """
    if len(points) == 0:
        return torch.zeros(len(extrinsics), device=points.device)

    pixels, points_cam = project(points, extrinsics, intrinsics)
    u, v = pixels[..., 0], pixels[..., 1]
    inside = (points_cam[..., 2] > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return inside.float().mean(-1)
```

- [ ] **Step 4: Wire the knob through `_collect_pairs`**

Add the keyword `min_pair_overlap: float = 0.0` and `overlap_stride: int = 8` after `rel_thresh`
(before `target_batch` if Task 7 landed). Add docstring bullet:
`- min_pair_overlap > 0 skips targets whose frustum holds less than that share of frame i's points (stride-subsampled)`.
Replace `targets = [j for j in range(n) if j != i]` (or, if Task 7 was reverted, the
`for j in range(n): if i == j: continue` header, changed to `for j in targets:`) with:

```python
        targets = [j for j in range(n) if j != i]

        # Optional pruning: drop targets whose frustum barely holds frame i's subsampled points
        if min_pair_overlap > 0:
            sub = points.reshape(h, w, 3)[::overlap_stride, ::overlap_stride].reshape(-1, 3)
            sub = sub[has_source.reshape(h, w)[::overlap_stride, ::overlap_stride].reshape(-1)]
            overlap = _frustum_overlap(sub, extrinsics_t, intrinsics_t, h, w).tolist()
            targets = [j for j in targets if overlap[j] >= min_pair_overlap]
```

In `compute_reconstruction_quality`, add keyword `min_pair_overlap: float = 0.0` after `rel_thresh`,
with Args entry `min_pair_overlap: skip pairs whose frustum overlap is below this share; 0 keeps every pair.`,
and pass it on: `_collect_pairs(depth, model_intrinsics, extrinsics, rel_thresh, min_pair_overlap=min_pair_overlap)`.

- [ ] **Step 5: Config and stage**

Append to `configs/base.yaml` (after `localization`):

```yaml

reconstruction_quality_report:
  min_pair_overlap: 0.0   # 0 = every ordered pair; >0 skips pairs whose frustum overlap is below it
```

In `reconstruction_quality_report`:

```python
        # Optional pair pruning; 0.0 (base.yaml) keeps every ordered pair
        min_pair_overlap = self.config["reconstruction_quality_report"]["min_pair_overlap"]

        tables = compute_reconstruction_quality(
            ff.depth,
            ff.model_intrinsics,
            ff.intrinsics,
            ff.extrinsics,
            ff.original_coords,
            [Path(str(p)).name for p in ff.image_paths],
            ff.confidence,
            images,
            min_pair_overlap=min_pair_overlap,
        )
```

and add `"min_pair_overlap": min_pair_overlap,` to the `scene` dict after `"zarr"`.

- [ ] **Step 6: Run tests**

Run: `cd /workspace/collab-splats/.worktrees/report-speed && PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/geometry tests/wrapper/test_reconstructor.py -q`
Expected: all pass. `test_report_json_is_the_columnar_contract` asserts top-level keys only, so the new `scene` key does not break it.

- [ ] **Step 7: Default parity on GH010229**

Label `t10`; `compare_reports.py t9.json t10.json`. Expected: `PASS` (the default 0.0 changes
nothing; the photometric pass is untouched since t9, so exact mode holds).

- [ ] **Step 8: Commit**

```bash
git add collab_splats/geometry/metrics.py collab_splats/wrapper/reconstructor.py configs/base.yaml tests/geometry/test_metrics.py tests/wrapper/test_reconstructor.py
git commit --only collab_splats/geometry/metrics.py collab_splats/wrapper/reconstructor.py configs/base.yaml tests/geometry/test_metrics.py tests/wrapper/test_reconstructor.py -m "feat(metrics): optional min_pair_overlap frustum pruning, default off

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: After-profile, A/B sweep, measured section in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-10-01-report-speed-design.md` (in the worktree)

- [ ] **Step 1: Sweep in tmux, one run at a time**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
tmux new -d -s report-speed "cd $PWD && for v in 0.0 0.01 0.05 0.1; do PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python scratch/report_speed/run_report.py ab_\$v --min_pair_overlap \$v > scratch/report_speed/out/ab_\$v.log 2>&1; done; echo EXIT \$? >> scratch/report_speed/out/ab_0.1.log"
```

- [ ] **Step 2: Tabulate**

For each `ab_<v>.json` against `ab_0.0.json` compute:
- pairs kept
- per-frame `multiview_agreement` max |Δ|
- per-frame `median_abs_rel_depth_error` max |Δ|
- histogram total-variation distance `0.5 * sum|p - q|` over normalized counts
- `total` seconds

Do it in a short inline python script. Expected: v=0.0 reproduces `t10` exactly.

- [ ] **Step 3: Write "## Measured (2026-10-01, GH010229, 294 frames)" into the spec**

Insert before "## Out of scope". Include:
- baseline vs final per-phase seconds, peak RSS, CUDA peak
- each commit's phase delta (`t2`..`t10`)
- the Task 7 gate outcome (kept, or reverted with the differing column and cost)
- the photometric ncc / n_pixels deltas from Tasks 8-9
- the A/B table
- 1k-frame extrapolation, labeled as an N² extrapolation: depth pass × (1000·999)/(294·293); photometric × 1000/294

No recommendation of a knob value; the user picks.

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-10-01-report-speed-design.md
git commit --only docs/superpowers/specs/2026-10-01-report-speed-design.md -m "docs(specs): report-speed measured results and min_pair_overlap A/B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Full gate, changelog, graph

**Files:**
- Modify: `docs/superpowers/CHANGELOG.md`, `CLAUDE.md` (both in the worktree)

- [ ] **Step 1: Format check on touched files only**

Never run repo-wide black (venv black is newer than the repo's):

```bash
cd /workspace/collab-splats/.worktrees/report-speed
/opt/venv/reconstruction/bin/python -m isort --check-only collab_splats/geometry/metrics.py collab_splats/geometry/projection.py collab_splats/geometry/transforms.py collab_splats/utils/image.py collab_splats/wrapper/reconstructor.py
```

Expected: no output. Fix any reported file with `isort <file>` and amend nothing. Make a new
`style:` commit if needed.

- [ ] **Step 2: Full suite in the worktree**

```bash
cd /workspace/collab-splats/.worktrees/report-speed
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -c "import collab_splats; print(collab_splats.__file__)"
PYTHONPATH=$PWD /opt/venv/reconstruction/bin/python -m pytest tests/ -q
```

Run in tmux if it exceeds 10 minutes. Expected: the only failures are those listed in
`docs/known-test-failures.md`. Diff the failure list against that file and report counts. A new failure blocks.

- [ ] **Step 3: CHANGELOG entry and In-Flight removal**

Append to `docs/superpowers/CHANGELOG.md`, following the format of the newest entry there (read
its top first). Use the measured before/after seconds from Task 11, the commit range, and the
knob default. In `CLAUDE.md`, delete the `- **report-speed** — ...` In-Flight bullet and add
`- **report-speed** (2026-10-01)` at the top of "Recently Completed", dropping the oldest of the
five so it stays five.

```bash
git add -f docs/superpowers/CHANGELOG.md
git commit --only docs/superpowers/CHANGELOG.md CLAUDE.md -m "docs(changelog): report-speed landed on perf/report-speed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Refresh the graph**

```bash
cd /workspace/collab-splats/.worktrees/report-speed && graphify update .
```

- [ ] **Step 5: Report**

Report the branch tip SHA, the commit list, before/after wall time, and the A/B table. Merging
`perf/report-speed` into `clean/final` is the user's call; do not merge.
