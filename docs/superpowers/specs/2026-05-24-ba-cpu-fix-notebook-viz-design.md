# Design: BA CPU Fix + Notebook Visualizations
**Date:** 2026-05-24  
**Branch:** refactor/cu121

---

## Problem

Two issues in the bundle adjustment pipeline:

1. **Track prediction runs on CPU** when `result.images` is already a CPU `torch.Tensor`. Only the numpy path (`isinstance(images, np.ndarray)`) moves images to `target_device`; a CPU tensor is left as-is. Since `predict_tracks` uses `images.device` to place the tracker model, the entire VGGSfM tracker runs on CPU.

2. **`bundle_adjustment.ipynb` lacks diagnostic visualizations**: no way to visually verify BA improved camera poses, and no loss curve to confirm optimizer convergence.

---

## Changes

### 1. Bug fix — `_extract_tracks_vggsfm` (bundle_adjustment.py)

After the numpy-to-tensor conversion, unconditionally move images to `target_device`:

```python
if isinstance(images, np.ndarray):
    images = torch.from_numpy(images)
images = images.to(target_device)   # ← always; fixes CPU-tensor case
```

One line added. No behavior change when images are already on CUDA.

---

### 2. Loss history capture — `BundleAdjustmentConfig` + `BundleAdjustment._optimize`

Add `capture_loss_history: bool = False` to `BundleAdjustmentConfig`.

When True, `_optimize` replaces `scheduler.optimize(input=...)` with a manual loop:

```python
if cfg.capture_loss_history:
    loss_history = []
    for _ in range(n_steps):
        loss = optimizer.step(input=input_dict)
        loss_history.append(float(loss))
    self._last_loss_history = loss_history
else:
    scheduler.optimize(input=input_dict)
```

`bae.optim.LM.step(input)` returns the scalar loss (confirmed from source). Manual loop skips `StopOnPlateau`'s patience-based early stopping — acceptable for visualization; default behavior (StopOnPlateau) unchanged.

`self._last_loss_history` is a side-channel attribute set only when `capture_loss_history=True`. Not part of public API.

---

### 3. Notebook cells — `bundle_adjustment.ipynb`

Add a new `## §Visualizations` section after the BA run cells. Two plots:

**A — Pre/post camera positions (matplotlib 3D)**

Extract camera centers from extrinsics before and after `ba.refine()`. Camera center (world coords) = `-R.T @ t` where `[R | t]` is the `(3, 4)` extrinsic. Plot both trajectories as 3D scatter+polyline with connecting lines per frame (before=blue, after=orange). This reuses `extrinsics_to_c2w` already imported in the feedforward notebook, or inline numpy math.

**B — Loss curve (matplotlib)**

Instantiate `BundleAdjustment(BundleAdjustmentConfig(capture_loss_history=True))`, call `refine()`, then plot `ba._last_loss_history` vs. step index (semilogy). Add horizontal reference at final loss.

---

## Scope

- `collab_splats/pointcloud/bundle_adjustment.py`: bug fix (1 line) + config field + manual loop branch (~15 lines)
- `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`: 2 new notebook sections (~3 cells each)
- No changes to public API (`refine()` signature unchanged)
- No new tests needed for the config flag (existing BA tests cover `_optimize`; manual loop uses same tensors)

---

## Out of scope

- Exposing `_last_loss_history` in `refine()` return value (not needed for notebook use)
- StopOnPlateau patience logic in capture mode (notebook viz doesn't need early stopping)
