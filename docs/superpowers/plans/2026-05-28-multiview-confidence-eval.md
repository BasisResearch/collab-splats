# Multiview Confidence Eval Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval notebook that empirically measures whether MapAnything's geometric multiview confidence filter improves point cloud quality for VGGT-X and VGGTOmega.

**Architecture:** Notebook in `evals/notebooks/`. Inference runs once per model and results are saved to zarr to avoid re-running. Filtering variants are then applied in-memory from the loaded zarr. MapAnything sanity check runs inference twice (mv_on/mv_off). Metrics: surviving point count + mv_conf score distribution + 3D scatter visual.

**Tech Stack:** Python 3.11 (`/opt/conda/envs/reconstruction` kernel), `mapanything.utils.multiview_confidence.compute_multiview_depth_confidence`, `collab_splats.pointcloud.feedforward.*Creator`, `FeedforwardResult.save_zarr/load_zarr`, `matplotlib`, `torch`, `numpy`

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `evals/notebooks/multiview_conf_analysis.ipynb` | Create | Eval notebook — all cells |
| `evals/results/mv_conf_eval/vggtx/` | Created at runtime | VGGT-X zarr store |
| `evals/results/mv_conf_eval/vggt_omega/` | Created at runtime | VGGTOmega zarr store |

---

## Task 1: Notebook skeleton + setup cell

**Files:**
- Create: `evals/notebooks/multiview_conf_analysis.ipynb`

- [ ] **Step 1: Create notebook with setup cell**

Create `evals/notebooks/multiview_conf_analysis.ipynb` as a Jupyter notebook (JSON format). The first code cell imports everything and sets constants.

Cell 1 content:

```python
import sys, os
sys.path.insert(0, "/workspace/collab-splats")
sys.path.insert(0, "/workspace/collab-splats/evals")
os.chdir("/workspace/collab-splats")

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collab_splats.pointcloud.feedforward import VGGTXCreator, VGGTOmegaCreator, MapAnythingCreator
from collab_splats.pointcloud.feedforward.base import FeedforwardResult
from collab_splats.utils.geometry import invert_poses
from mapanything.utils.multiview_confidence import compute_multiview_depth_confidence
from datasets import get_dataset

SEQ_DIR   = Path("/data/7scenes/chess/seq-01")
ZARR_BASE = Path("evals/results/mv_conf_eval")
PERCENTILE = 35.0
MAX_FRAMES = 50   # subset for tractable mv_conf compute (O(N^2) pairwise)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ZARR_BASE.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE}")
```

Cell 2 — load dataset and assert frames exist:

```python
dataset = get_dataset("7scenes")(SEQ_DIR, max_frames=MAX_FRAMES)
image_paths = dataset.images[:MAX_FRAMES]
print(f"Loaded {len(image_paths)} frames from {SEQ_DIR}")
assert len(image_paths) > 0, "No images found — check SEQ_DIR"

# Write image paths to a temp dir (creators expect an image_dir)
import tempfile, shutil
_tmp_image_dir = Path(tempfile.mkdtemp())
for i, src in enumerate(image_paths):
    dst = _tmp_image_dir / f"{i:06d}{src.suffix}"
    shutil.copy(src, dst)
print(f"Temp image dir: {_tmp_image_dir} ({len(list(_tmp_image_dir.iterdir()))} files)")
```

- [ ] **Step 2: Verify notebook file exists**

```bash
ls evals/notebooks/multiview_conf_analysis.ipynb
```

Expected: file present.

---

## Task 2: VGGT-X inference cell

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cell)

- [ ] **Step 1: Add inference cell — runs in tmux**

Add notebook cell:

```python
# ── VGGT-X inference ──────────────────────────────────────────────────────────
# Run in tmux: heavy inference. Zarr saved to avoid re-running.
VGGTX_ZARR = ZARR_BASE / "vggtx"

if not VGGTX_ZARR.exists():
    creator = VGGTXCreator(conf_threshold=PERCENTILE, max_points=500_000)
    creator.run(_tmp_image_dir)
    result_vggtx = creator.result
    result_vggtx.save_zarr(VGGTX_ZARR)
    print(f"Saved VGGT-X zarr → {VGGTX_ZARR}")
    del creator
    torch.cuda.empty_cache()
else:
    print(f"VGGT-X zarr exists, skipping inference: {VGGTX_ZARR}")

result_vggtx = FeedforwardResult.load_zarr(VGGTX_ZARR)
print(f"VGGT-X: {result_vggtx.points.shape[0]:,} points | "
      f"depth {result_vggtx.depth.shape} | "
      f"conf {result_vggtx.confidence.shape} | "
      f"world_points {result_vggtx.world_points.shape}")
```

- [ ] **Step 2: Assert required fields are populated**

Add assertion cell:

```python
assert result_vggtx.depth is not None,        "depth missing from zarr"
assert result_vggtx.confidence is not None,    "confidence missing from zarr"
assert result_vggtx.world_points is not None,  "world_points missing from zarr"
assert result_vggtx.intrinsics is not None,    "intrinsics missing from zarr"
assert result_vggtx.extrinsics is not None,    "extrinsics missing from zarr"
N = result_vggtx.depth.shape[0]
H, W = result_vggtx.depth.shape[1], result_vggtx.depth.shape[2]
assert result_vggtx.confidence.shape == (N, H, W), \
    f"conf shape mismatch: {result_vggtx.confidence.shape} vs ({N},{H},{W})"
assert result_vggtx.world_points.shape == (N, H, W, 3), \
    f"world_points shape mismatch: {result_vggtx.world_points.shape}"
print(f"VGGT-X assertions passed. N={N}, H={H}, W={W}")
```

---

## Task 3: VGGTOmega inference cell

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cell)

- [ ] **Step 1: Add VGGTOmega inference cell**

```python
# ── VGGTOmega inference ───────────────────────────────────────────────────────
OMEGA_ZARR = ZARR_BASE / "vggt_omega"

if not OMEGA_ZARR.exists():
    creator = VGGTOmegaCreator(conf_threshold=PERCENTILE, max_points=500_000)
    creator.run(_tmp_image_dir)
    result_omega = creator.result
    result_omega.save_zarr(OMEGA_ZARR)
    print(f"Saved VGGTOmega zarr → {OMEGA_ZARR}")
    del creator
    torch.cuda.empty_cache()
else:
    print(f"VGGTOmega zarr exists, skipping inference: {OMEGA_ZARR}")

result_omega = FeedforwardResult.load_zarr(OMEGA_ZARR)
print(f"VGGTOmega: {result_omega.points.shape[0]:,} points | "
      f"depth {result_omega.depth.shape} | conf {result_omega.confidence.shape}")
```

- [ ] **Step 2: Assert VGGTOmega fields**

```python
assert result_omega.depth is not None and result_omega.confidence is not None
assert result_omega.world_points is not None and result_omega.intrinsics is not None
N2, H2, W2 = result_omega.depth.shape
assert result_omega.world_points.shape == (N2, H2, W2, 3)
print(f"VGGTOmega assertions passed. N={N2}, H={H2}, W={W2}")
```

---

## Task 4: MapAnything inference — mv_off and mv_on

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cells)

MapAnything sanity check: run inference twice with `use_multiview_confidence=True/False`.
Compare surviving point counts directly — no zarr needed here since these are two
separate runs and point counts are the only metric.

- [ ] **Step 1: Add MapAnything mv_off cell**

```python
# ── MapAnything inference — mv_off ────────────────────────────────────────────
creator_ma_off = MapAnythingCreator(
    use_multiview_confidence=False,
    confidence_percentile=PERCENTILE,
    max_points=500_000,
)
creator_ma_off.run(_tmp_image_dir)
result_ma_off = creator_ma_off.result
n_points_ma_off = result_ma_off.points.shape[0]
print(f"MapAnything mv_off: {n_points_ma_off:,} points")
del creator_ma_off
torch.cuda.empty_cache()
```

- [ ] **Step 2: Add MapAnything mv_on cell**

```python
# ── MapAnything inference — mv_on ─────────────────────────────────────────────
creator_ma_on = MapAnythingCreator(
    use_multiview_confidence=True,
    confidence_percentile=PERCENTILE,
    max_points=500_000,
)
creator_ma_on.run(_tmp_image_dir)
result_ma_on = creator_ma_on.result
n_points_ma_on = result_ma_on.points.shape[0]
print(f"MapAnything mv_on:  {n_points_ma_on:,} points")
del creator_ma_on
torch.cuda.empty_cache()

print(f"\nMapAnything sanity check: {n_points_ma_off:,} → {n_points_ma_on:,} "
      f"({'↓' if n_points_ma_on < n_points_ma_off else '↑'} "
      f"{abs(n_points_ma_on - n_points_ma_off):,} points)")
```

---

## Task 5: Multiview confidence helper cell

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cell)

This helper converts a `FeedforwardResult` (loaded from zarr) into inputs for
`compute_multiview_depth_confidence`, runs it, and returns the `(N, H, W)` numpy
confidence array.

- [ ] **Step 1: Add helper cell**

```python
# ── Multiview confidence helper ───────────────────────────────────────────────

def compute_mv_conf(result: FeedforwardResult, device: str = DEVICE) -> np.ndarray:
    """Compute geometric multiview confidence from a FeedforwardResult.

    Returns (N, H, W) float32 numpy array, values in [0, 1].
    Requires result.depth, result.intrinsics, result.extrinsics to be populated.
    """
    depth_np = result.depth.astype(np.float32)          # (N, H, W)
    N, H, W  = depth_np.shape

    # compute_multiview_depth_confidence expects (B, H, W, 1) depth per view
    depth_z = [
        torch.from_numpy(depth_np[i]).unsqueeze(0).unsqueeze(-1).to(device)  # (1, H, W, 1)
        for i in range(N)
    ]

    # intrinsics: (N, 3, 3) numpy → list of (1, 3, 3) tensors
    intrs_np = result.intrinsics.astype(np.float32)     # (N, 3, 3)
    intrinsics = [
        torch.from_numpy(intrs_np[i]).unsqueeze(0).to(device)  # (1, 3, 3)
        for i in range(N)
    ]

    # extrinsics: world2cam (N, 4, 4) → cam2world via invert_poses
    cam2world_np = invert_poses(result.extrinsics.astype(np.float32))   # (N, 4, 4)
    camera_poses = [
        torch.from_numpy(cam2world_np[i]).unsqueeze(0).to(device)  # (1, 4, 4)
        for i in range(N)
    ]

    with torch.no_grad():
        mv_conf_list = compute_multiview_depth_confidence(
            depth_z, intrinsics, camera_poses
        )

    # Stack and move to CPU numpy
    mv_conf = torch.stack([c.squeeze(0).cpu() for c in mv_conf_list]).numpy()  # (N, H, W)
    return mv_conf.astype(np.float32)


def apply_percentile_mask(conf: np.ndarray, percentile: float = PERCENTILE) -> np.ndarray:
    """Return (N, H, W) bool mask: True where conf >= percentile threshold."""
    thresh = float(np.percentile(conf, percentile))
    return conf >= thresh


def count_valid_points(world_points: np.ndarray, mask: np.ndarray) -> int:
    """Count surviving points given a (N, H, W) bool mask over (N, H, W, 3) world_points."""
    return int(mask.sum())


print("Helpers defined.")
```

- [ ] **Step 2: Smoke-test helper on VGGT-X (just verify shapes)**

```python
print("Computing VGGT-X mv_conf (may take ~30s)...")
mv_conf_vggtx = compute_mv_conf(result_vggtx)
print(f"mv_conf_vggtx: shape={mv_conf_vggtx.shape}, "
      f"min={mv_conf_vggtx.min():.3f}, max={mv_conf_vggtx.max():.3f}, "
      f"mean={mv_conf_vggtx.mean():.3f}")
assert mv_conf_vggtx.shape == result_vggtx.depth.shape, \
    f"mv_conf shape {mv_conf_vggtx.shape} != depth shape {result_vggtx.depth.shape}"
assert 0.0 <= mv_conf_vggtx.min() and mv_conf_vggtx.max() <= 1.0, \
    "mv_conf values out of [0,1] range"
print("Shape/range assertions passed.")
```

---

## Task 6: VGGT-X filtering variants

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cell)

- [ ] **Step 1: Add VGGT-X variants cell**

```python
# ── VGGT-X filtering variants ─────────────────────────────────────────────────

wp_vggtx = result_vggtx.world_points   # (N, H, W, 3) full grid

# Variant 1: learned_only — existing depth_conf percentile (already applied in zarr)
# result_vggtx.points are already filtered; count from world_points for apples-to-apples
learned_conf_vggtx = result_vggtx.confidence    # (N, H, W) depth_conf
mask_learned_vggtx = apply_percentile_mask(learned_conf_vggtx, PERCENTILE)
n_learned_vggtx = count_valid_points(wp_vggtx, mask_learned_vggtx)

# Variant 2: mv_only — replace depth_conf with geometric mv_conf
# mv_conf_vggtx computed in Task 5
mask_mv_vggtx = apply_percentile_mask(mv_conf_vggtx, PERCENTILE)
n_mv_vggtx = count_valid_points(wp_vggtx, mask_mv_vggtx)

# Variant 3: intersect — both filters AND'd
mask_intersect_vggtx = mask_learned_vggtx & mask_mv_vggtx
n_intersect_vggtx = count_valid_points(wp_vggtx, mask_intersect_vggtx)

print("VGGT-X filtering variants:")
print(f"  learned_only : {n_learned_vggtx:>8,} points")
print(f"  mv_only      : {n_mv_vggtx:>8,} points")
print(f"  intersect    : {n_intersect_vggtx:>8,} points")
```

---

## Task 7: VGGTOmega filtering variants

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cell)

- [ ] **Step 1: Add VGGTOmega variants cell**

```python
# ── VGGTOmega filtering variants ──────────────────────────────────────────────

print("Computing VGGTOmega mv_conf...")
mv_conf_omega = compute_mv_conf(result_omega)
print(f"mv_conf_omega: mean={mv_conf_omega.mean():.3f}")

wp_omega = result_omega.world_points   # (N, H, W, 3)

learned_conf_omega = result_omega.confidence
mask_learned_omega = apply_percentile_mask(learned_conf_omega, PERCENTILE)
n_learned_omega = count_valid_points(wp_omega, mask_learned_omega)

mask_mv_omega = apply_percentile_mask(mv_conf_omega, PERCENTILE)
n_mv_omega = count_valid_points(wp_omega, mask_mv_omega)

mask_intersect_omega = mask_learned_omega & mask_mv_omega
n_intersect_omega = count_valid_points(wp_omega, mask_intersect_omega)

print("VGGTOmega filtering variants:")
print(f"  learned_only : {n_learned_omega:>8,} points")
print(f"  mv_only      : {n_mv_omega:>8,} points")
print(f"  intersect    : {n_intersect_omega:>8,} points")
```

---

## Task 8: Results table + point count bar chart

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cells)

- [ ] **Step 1: Add summary table cell**

```python
# ── Summary table ─────────────────────────────────────────────────────────────

import pandas as pd

rows = [
    ("VGGT-X",    "learned_only",  n_learned_vggtx),
    ("VGGT-X",    "mv_only",       n_mv_vggtx),
    ("VGGT-X",    "intersect",     n_intersect_vggtx),
    ("VGGTOmega", "learned_only",  n_learned_omega),
    ("VGGTOmega", "mv_only",       n_mv_omega),
    ("VGGTOmega", "intersect",     n_intersect_omega),
    ("MapAnything", "mv_off",      n_points_ma_off),
    ("MapAnything", "mv_on",       n_points_ma_on),
]
df = pd.DataFrame(rows, columns=["Model", "Variant", "N_points"])
df["mv_conf_mean"] = [
    "n/a", mv_conf_vggtx.mean(), mv_conf_vggtx.mean(),
    "n/a", mv_conf_omega.mean(), mv_conf_omega.mean(),
    "n/a", "n/a",
]
print(df.to_string(index=False))
```

- [ ] **Step 2: Add point count bar chart cell**

```python
# ── Bar chart: point count by model × variant ─────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 5))
colors = {"learned_only": "#4477AA", "mv_only": "#EE7733", "intersect": "#AA3377",
          "mv_off": "#4477AA", "mv_on": "#EE7733"}

x_labels = [f"{r[0]}\n{r[1]}" for r in rows]
y_vals   = [r[2] for r in rows]
bar_colors = [colors.get(r[1], "#888888") for r in rows]

bars = ax.bar(x_labels, y_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", padding=3, fontsize=8)
ax.set_ylabel("Surviving points")
ax.set_title("Point cloud density by model × filtering variant (chess seq-01, 50 frames)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ZARR_BASE / "point_count_comparison.png", dpi=150)
plt.show()
print(f"Saved → {ZARR_BASE / 'point_count_comparison.png'}")
```

---

## Task 9: mv_conf distribution histograms + 3D scatter

**Files:**
- Modify: `evals/notebooks/multiview_conf_analysis.ipynb` (add cells)

- [ ] **Step 1: Add mv_conf histogram cell**

Histograms show whether mv_conf has a meaningful bimodal distribution (good signal)
or is flat/uniform (no signal). A peaked distribution near 0 and 1 = strong signal.

```python
# ── mv_conf score distributions ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (label, mv_conf) in zip(axes, [("VGGT-X", mv_conf_vggtx),
                                        ("VGGTOmega", mv_conf_omega)]):
    ax.hist(mv_conf.ravel(), bins=50, color="#4477AA", alpha=0.8, edgecolor="white")
    thresh = float(np.percentile(mv_conf, PERCENTILE))
    ax.axvline(thresh, color="red", linestyle="--", label=f"p{int(PERCENTILE)} = {thresh:.3f}")
    ax.set_xlabel("mv_conf score")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"{label} — mv_conf distribution")
    ax.legend()

plt.tight_layout()
plt.savefig(ZARR_BASE / "mv_conf_distributions.png", dpi=150)
plt.show()
print(f"Saved → {ZARR_BASE / 'mv_conf_distributions.png'}")
```

- [ ] **Step 2: Add 3D scatter — VGGT-X learned_only vs mv_only**

Subsample to 5k points per variant for rendering speed.

```python
# ── 3D scatter: VGGT-X learned_only vs mv_only ────────────────────────────────

rng = np.random.default_rng(42)
SCATTER_N = 5_000

wp = result_vggtx.world_points   # (N, H, W, 3)

def _sample_points(world_points, mask, n):
    idx = np.stack(np.where(mask), axis=1)  # (P, 3)
    if len(idx) > n:
        idx = idx[rng.choice(len(idx), n, replace=False)]
    pts = world_points[idx[:, 0], idx[:, 1], idx[:, 2]]  # (n, 3)
    return pts

pts_learned  = _sample_points(wp, mask_learned_vggtx,  SCATTER_N)
pts_mv       = _sample_points(wp, mask_mv_vggtx,       SCATTER_N)

fig = plt.figure(figsize=(14, 6))
for i, (pts, title) in enumerate([(pts_learned, f"learned_only ({n_learned_vggtx:,} pts)"),
                                   (pts_mv,      f"mv_only      ({n_mv_vggtx:,} pts)")], 1):
    ax = fig.add_subplot(1, 2, i, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.3, alpha=0.5, c=pts[:, 2],
               cmap="viridis")
    ax.set_title(f"VGGT-X {title}")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

plt.tight_layout()
plt.savefig(ZARR_BASE / "scatter_vggtx_comparison.png", dpi=150)
plt.show()
print(f"Saved → {ZARR_BASE / 'scatter_vggtx_comparison.png'}")
```

---

## Self-Review

### Spec Coverage

| Spec section | Covered by task |
|---|---|
| Run inference once per model, save zarr | Task 2, 3 |
| MapAnything mv_on/mv_off comparison | Task 4 |
| Filtering variants: learned_only / mv_only / intersect | Task 6, 7 |
| `compute_multiview_depth_confidence` input prep | Task 5 |
| Metrics: point count, mv_conf distribution, visual quality | Task 8, 9 |
| Run in tmux (heavy inference) | Noted in Task 2 cell comments |

All spec sections covered. No gaps.

### Placeholder scan

No TBDs. All code blocks complete with exact shapes and assertions.

### Type consistency

- `result_vggtx.depth`: `(N, H, W)` np.ndarray — used consistently in Task 5 (reshape to list of `(1, H, W, 1)`)
- `result_vggtx.intrinsics`: `(N, 3, 3)` np.ndarray — consistent
- `result_vggtx.extrinsics`: `(N, 4, 4)` np.ndarray world2cam — `invert_poses` takes numpy, returns numpy
- `invert_poses` — numpy-only, confirmed from source; torch conversion happens inside `compute_mv_conf`
- `compute_multiview_depth_confidence` returns `List[Tensor (B, H, W)]` with B=1 — squeezed correctly in Task 5
- `world_points`: `(N, H, W, 3)` np.ndarray — `_sample_points` and `count_valid_points` use consistent indexing

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-28-multiview-confidence-eval.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
