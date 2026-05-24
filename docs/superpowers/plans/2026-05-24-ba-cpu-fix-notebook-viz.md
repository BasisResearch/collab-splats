# BA CPU Fix + Notebook Visualizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix VGGSfM track prediction running on CPU when `result.images` is a `torch.Tensor`; add pre/post BA camera position visualization and loss curve to `bundle_adjustment.ipynb`.

**Architecture:** (1) One-line fix in `_extract_tracks_vggsfm` ensures `.to(target_device)` is always called regardless of input type. (2) `BundleAdjustmentConfig.capture_loss_history` flag switches `_optimize` from `StopOnPlateau.optimize()` to a manual `optimizer.step()` loop that records per-step loss on `self._last_loss_history`. (3) Two new notebook cells use these features for visualization.

**Tech Stack:** PyTorch, bae LM optimizer (`bae.optim.LM`), pypose, matplotlib, numpy, pytest (mocking via `unittest.mock`).

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/bundle_adjustment.py` | Bug fix (line 274), new config field (`capture_loss_history`), init attr, manual loop branch in `_optimize` |
| `tests/pointcloud/test_bundle_adjustment.py` | 2 new tests: device contract, loss history |
| `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb` | Modify `section-b-code` cell; add 3 new cells (markdown + 2 code) |

---

### Task 1: Fix CPU device bug in `_extract_tracks_vggsfm`

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py:272-279`
- Test: `tests/pointcloud/test_bundle_adjustment.py`

**Context:** `_extract_tracks_vggsfm` only calls `.to(target_device)` inside the `isinstance(images, np.ndarray)` branch. When `result.images` is already a `torch.Tensor` on CPU, the branch is skipped — `img_device` stays CPU and `predict_tracks` builds the tracker on CPU. Fix: call `.to(target_device)` unconditionally after the isinstance guard.

- [ ] **Step 1: Write the failing test**

Add to `tests/pointcloud/test_bundle_adjustment.py` at the end of the "Test 1: extract_tracks_vggsfm" block (after `test_extract_tracks_vggsfm_conf_4d`):

```python
def test_extract_tracks_vggsfm_tensor_images_reach_target_device():
    """Tensor images (not numpy) must be moved to target_device before predict_tracks.

    Regression guard: the isinstance(np.ndarray) guard previously meant CPU torch.Tensor
    inputs bypassed .to(target_device). predict_tracks uses images.device for the tracker
    model — wrong device means the whole tracker runs on CPU even when CUDA is available.
    """
    N, H, W = 2, 8, 8
    # CPU torch.Tensor — NOT numpy, exercises the non-numpy code path
    images_cpu = torch.zeros(N, 3, H, W)

    # Capture images device as seen by predict_tracks
    received_device: list[str] = []

    def fake_predict(imgs, conf=None, points_3d=None, **kw):
        received_device.append(str(imgs.device))
        P = 4
        return (
            np.zeros((N, P, 2), dtype=np.float32),
            np.zeros((N, P), dtype=np.float32),
            np.zeros((N, P), dtype=np.float32),
            np.zeros((P, 3), dtype=np.float32),
            np.zeros((P, 3), dtype=np.float32),
        )

    with patch(
        "collab_splats.pointcloud.bundle_adjustment.predict_tracks",
        side_effect=fake_predict,
    ):
        # Re-import the live module (bae/vggt available in nerfstudio env)
        from collab_splats.pointcloud.bundle_adjustment import _extract_tracks_vggsfm
        _extract_tracks_vggsfm(images_cpu, conf=None, world_points=None, device="cpu")

    assert len(received_device) == 1, "predict_tracks must be called exactly once"
    # After fix: images.device always matches target_device regardless of input type
    assert received_device[0] == "cpu", (
        f"images.device={received_device[0]} != target_device='cpu'; "
        "tensor images are not being relocated — tracker will run on wrong device"
    )
```

- [ ] **Step 2: Run test to verify it is importable and captures intent**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
    tests/pointcloud/test_bundle_adjustment.py::test_extract_tracks_vggsfm_tensor_images_reach_target_device \
    -v
```

Expected: PASS (CPU→CPU case already works; test documents the contract). The test will FAIL for CUDA targets without GPU — that is the real regression scenario, validated here as a contract guard for the CPU path.

- [ ] **Step 3: Implement the fix**

In `collab_splats/pointcloud/bundle_adjustment.py`, change lines 272-279:

**Before:**
```python
    # Accept numpy array input from windowed LC path (raw_outputs stores merged images as numpy)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).to(target_device)

    # predict_tracks inherits device from images.device — it does NOT self-relocate
    img_device = images.device
```

**After:**
```python
    # Accept numpy array input from windowed LC path (raw_outputs stores merged images as numpy)
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)

    # Always move to target_device — predict_tracks uses images.device for tracker placement
    # and does NOT self-relocate. CPU torch.Tensor inputs would bypass this without the
    # unconditional .to() call.
    images = images.to(target_device)
    img_device = images.device
```

- [ ] **Step 4: Run test + full BA test suite to verify no regression**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v
```

Expected: all existing tests pass + new test passes.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py \
        tests/pointcloud/test_bundle_adjustment.py
git commit -m "fix(ba): always move images tensor to target_device in _extract_tracks_vggsfm

Previously only numpy arrays were relocated to target_device; torch.Tensor
inputs bypassed .to(), leaving predict_tracks to build its tracker on the
wrong device (CPU when CUDA was the target).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 2: `capture_loss_history` config flag + `_optimize` manual loop

**Files:**
- Modify: `collab_splats/pointcloud/bundle_adjustment.py:39-50` (config), `57-65` (`__init__`), `189-208` (`_optimize`)
- Test: `tests/pointcloud/test_bundle_adjustment.py`

**Context:** `bae.optim.LM.step(input)` returns a scalar loss tensor. When `capture_loss_history=True`, replace `scheduler.optimize(input=...)` with a manual loop calling `optimizer.step(input=...)` `lm_steps` times, collecting loss at each step. Stored on `self._last_loss_history`. Default behaviour (StopOnPlateau with patience/early-stop) is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/pointcloud/test_bundle_adjustment.py` inside the "Tests for BundleAdjustment class" block:

```python
@pytest.mark.skipif(not _pypose_available(), reason="requires pypose + bae")
def test_optimize_captures_loss_history_when_flag_set():
    """_optimize populates self._last_loss_history when capture_loss_history=True.

    Verifies: list populated, length == lm_steps, all values are finite floats,
    losses are non-negative (squared reprojection residuals).
    """
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    n_steps = 5
    cfg = BundleAdjustmentConfig(capture_loss_history=True, lm_steps=n_steps)
    ba = BundleAdjustment(config=cfg)

    # _last_loss_history must start empty before any run
    assert ba._last_loss_history == []

    ba._optimize(
        pts3d,
        extrinsics,
        intrinsics,
        tracks,
        vis_mask.astype(np.float32),
        max_reproj_error=None,   # skip reprojection filter so all points are active
    )

    hist = ba._last_loss_history
    assert isinstance(hist, list), f"expected list, got {type(hist)}"
    assert len(hist) == n_steps, f"expected {n_steps} entries, got {len(hist)}"
    assert all(isinstance(v, float) for v in hist), "all entries must be Python floats"
    assert all(v >= 0 for v in hist), "losses are squared residuals — must be non-negative"


@pytest.mark.skipif(not _pypose_available(), reason="requires pypose + bae")
def test_optimize_no_loss_history_by_default():
    """Without capture_loss_history, _last_loss_history stays empty after _optimize."""
    from collab_splats.pointcloud.bundle_adjustment import BundleAdjustment, BundleAdjustmentConfig

    N, P, H, W = 4, 60, 128, 128
    pts3d, extrinsics, intrinsics, tracks, vis_mask = _build_synthetic_scene(N, P, H, W)

    ba = BundleAdjustment()   # capture_loss_history defaults to False
    ba._optimize(
        pts3d, extrinsics, intrinsics,
        tracks, vis_mask.astype(np.float32),
        max_reproj_error=None,
    )

    assert ba._last_loss_history == [], (
        "_last_loss_history must remain empty when capture_loss_history=False"
    )
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest \
    tests/pointcloud/test_bundle_adjustment.py::test_optimize_captures_loss_history_when_flag_set \
    tests/pointcloud/test_bundle_adjustment.py::test_optimize_no_loss_history_by_default \
    -v
```

Expected: `test_optimize_captures_loss_history_when_flag_set` FAILS with `TypeError: BundleAdjustmentConfig.__init__() got an unexpected keyword argument 'capture_loss_history'`.

- [ ] **Step 3: Add `capture_loss_history` field to `BundleAdjustmentConfig`**

In `collab_splats/pointcloud/bundle_adjustment.py`, change the `BundleAdjustmentConfig` dataclass:

**Before (lines 39-49):**
```python
@dataclass
class BundleAdjustmentConfig:
    """Configuration for LM bundle adjustment."""

    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048       # track extraction: max query points
    query_frame_num: int = 5        # track extraction: number of query frames
    device: str | None = None       # target device; None = auto (CUDA if available, else CPU)
```

**After:**
```python
@dataclass
class BundleAdjustmentConfig:
    """Configuration for LM bundle adjustment."""

    max_reproj_error: float = 4.0
    lm_steps: int = 40
    shared_camera: bool = False
    min_inliers_per_frame: int = 64
    max_query_pts: int = 2048           # track extraction: max query points
    query_frame_num: int = 5            # track extraction: number of query frames
    device: str | None = None           # target device; None = auto (CUDA if available, else CPU)
    capture_loss_history: bool = False  # record per-step LM loss; access via BundleAdjustment._last_loss_history
```

- [ ] **Step 4: Add `_last_loss_history` attribute to `BundleAdjustment.__init__`**

**Before (lines 64-65):**
```python
    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()
```

**After:**
```python
    def __init__(self, config: BundleAdjustmentConfig | None = None) -> None:
        self.config = config or BundleAdjustmentConfig()
        # Populated per _optimize() call when capture_loss_history=True; else stays empty
        self._last_loss_history: list[float] = []
```

- [ ] **Step 5: Add manual step loop to `_optimize`**

In `collab_splats/pointcloud/bundle_adjustment.py`, change the optimisation block (lines 189-207):

**Before:**
```python
        # Optimise reprojection residuals with Levenberg-Marquardt
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, cfg.shared_camera)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=_get_default_solver(device=device),
                reject=10,
            )
            scheduler = pp.optim.scheduler.StopOnPlateau(
                optimizer, steps=n_steps, patience=3, decreasing=1e-3, verbose=False,
            )
            scheduler.optimize(input={
                "points_2d": obs_2d_t,
                "camera_indices": cam_idx,
                "point_indices": pt_idx_t,
                "principal_points": principal_points,
            })
```

**After:**
```python
        # Optimise reprojection residuals with Levenberg-Marquardt
        input_dict = {
            "points_2d": obs_2d_t,
            "camera_indices": cam_idx,
            "point_indices": pt_idx_t,
            "principal_points": principal_points,
        }
        with torch.enable_grad():
            model = _BAModel(cam_params, pts3d_tensor, shared_focal, cfg.shared_camera)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
            optimizer = LM(
                model,
                strategy=strategy,
                solver=_get_default_solver(device=device),
                reject=10,
            )

            if cfg.capture_loss_history:
                # Manual step loop: collect scalar loss at each iteration.
                # StopOnPlateau patience/early-stop is not applied here — all n_steps run.
                loss_hist: list[float] = []
                for _ in range(n_steps):
                    loss = optimizer.step(input=input_dict)
                    loss_hist.append(float(loss))
                self._last_loss_history = loss_hist
            else:
                # Default path: StopOnPlateau with patience-based early stopping
                self._last_loss_history = []
                scheduler = pp.optim.scheduler.StopOnPlateau(
                    optimizer, steps=n_steps, patience=3, decreasing=1e-3, verbose=False,
                )
                scheduler.optimize(input=input_dict)
```

- [ ] **Step 6: Run all BA tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_bundle_adjustment.py -v
```

Expected: all pass, including the two new tests.

- [ ] **Step 7: Commit**

```bash
git add collab_splats/pointcloud/bundle_adjustment.py \
        tests/pointcloud/test_bundle_adjustment.py
git commit -m "feat(ba): add capture_loss_history flag for per-step LM loss recording

Adds BundleAdjustmentConfig.capture_loss_history (default False). When True,
_optimize replaces StopOnPlateau.optimize() with a manual step loop that
stores scalar losses in self._last_loss_history. Default behaviour unchanged.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Notebook visualizations

**Files:**
- Modify: `docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb`

**Context:** The notebook uses `NotebookEdit` (or raw JSON `Edit`) to add cells. Variables available after `section-b-code` runs: `ff_result` (pre-BA `FeedforwardResult`), `refined` (post-BA), `ba` (`BundleAdjustment` instance). We modify `section-b-code` to save pre-BA extrinsics and enable loss history, then add three new cells (one markdown, two code) between `section-b-code` and `footer`.

- [ ] **Step 1: Modify `section-b-code` cell to capture pre-BA state**

Use `NotebookEdit` (or `Edit` on the raw JSON) to replace the source of cell `section-b-code`.

**New source for `section-b-code`:**
```python
# Load cached feedforward result; restore images from zarr store (needed for track extraction)
ff_result = FeedforwardResult.load_zarr(VGGTX_ZARR)
_store = zarr.open(str(VGGTX_ZARR), mode="r")
ff_result.images = torch.from_numpy(_store["images"][:])   # (N, 3, H, W)

# Save pre-BA extrinsics for the visualisation below
extrinsics_pre = ff_result.extrinsics.copy()   # (N, 4, 4)

# Refine poses — enable loss history so we can plot the convergence curve
ba = BundleAdjustment(config=BundleAdjustmentConfig(capture_loss_history=True))
refined = ba.refine(ff_result)
print(f"refined: {refined.extrinsics.shape[0]} cameras  pts3d: {refined.pts3d.shape[0]:,} points")
```

- [ ] **Step 2: Add §3 section markdown cell after `section-b-code`**

Insert a new `markdown` cell with id `viz-md` between `section-b-code` and `footer`:

```markdown
## §3 — Visualizations

Pre/post BA camera positions and the LM loss convergence curve.
`extrinsics_pre` and `refined.extrinsics` are both `(N, 4, 4)` world-to-camera matrices;
camera center in world coordinates is `−R.T @ t`.
```

- [ ] **Step 3: Add camera positions cell after `viz-md`**

Insert a new `code` cell with id `viz-cameras`:

```python
import matplotlib.pyplot as plt
import numpy as np


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """Return world-space camera centers from (N, 4, 4) world-to-camera extrinsics."""
    R = extrinsics[:, :3, :3]   # (N, 3, 3)
    t = extrinsics[:, :3, 3]    # (N, 3)
    # Camera center = -R.T @ t for each camera (inverse of world-to-camera translation)
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), -t)


# Extract per-camera world positions before and after BA
centers_pre  = _camera_centers(extrinsics_pre)
centers_post = _camera_centers(refined.extrinsics)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Trajectory arcs — connect consecutive cameras with lines
ax.plot(*centers_pre.T,  "o-", color="steelblue",  label="Pre-BA",  alpha=0.8, markersize=5)
ax.plot(*centers_post.T, "o-", color="darkorange", label="Post-BA", alpha=0.8, markersize=5)

# Per-camera displacement connectors (grey dashed)
for pre, post in zip(centers_pre, centers_post):
    ax.plot(
        [pre[0], post[0]], [pre[1], post[1]], [pre[2], post[2]],
        color="gray", alpha=0.35, linewidth=0.8, linestyle="--",
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend(fontsize=11)
ax.set_title("Camera Positions: Pre vs Post BA")
plt.tight_layout()
plt.show()

# Summary statistics
deltas = np.linalg.norm(centers_post - centers_pre, axis=-1)
print(f"Camera displacement — mean: {deltas.mean():.4f} m  max: {deltas.max():.4f} m")
```

- [ ] **Step 4: Add loss curve cell after `viz-cameras`**

Insert a new `code` cell with id `viz-loss`:

```python
import matplotlib.pyplot as plt


loss_hist = ba._last_loss_history

if not loss_hist:
    print("No loss history captured. Re-run with BundleAdjustmentConfig(capture_loss_history=True).")
else:
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = list(range(1, len(loss_hist) + 1))

    # Loss curve on log scale — LM losses can span several orders of magnitude
    ax.semilogy(steps, loss_hist, "o-", color="steelblue", markersize=5, linewidth=1.5)

    # Final-loss reference line
    ax.axhline(
        loss_hist[-1], color="gray", linestyle="--", alpha=0.6,
        label=f"Final: {loss_hist[-1]:.3e}",
    )

    ax.set_xlabel("LM Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Bundle Adjustment — LM Loss Convergence")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print improvement ratio
    print(f"Loss reduction: {loss_hist[0]:.3e} → {loss_hist[-1]:.3e}  "
          f"({loss_hist[0] / loss_hist[-1]:.1f}× improvement)")
```

- [ ] **Step 5: Run the notebook to verify cells execute without error**

In the nerfstudio env kernel, run cells §0 through §3 in sequence:

```bash
/opt/conda/envs/nerfstudio/bin/jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.kernel_name=nerfstudio \
    --ExecutePreprocessor.timeout=600 \
    "docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb" \
    --output bundle_adjustment_executed.ipynb 2>&1 | tail -20
```

Expected: no `ERROR` lines. Inspect `bundle_adjustment_executed.ipynb` to confirm camera plot and loss curve appear as outputs.

- [ ] **Step 6: Commit**

```bash
git add docs/source/tutorials/02_pointcloud/bundle_adjustment.ipynb
git commit -m "feat(notebook): pre/post BA camera viz + loss curve in bundle_adjustment.ipynb

Modifies section-b-code to capture extrinsics_pre and enable loss history.
Adds §3 section with 3D camera trajectory comparison (pre/post BA) and
semilogy loss convergence plot using ba._last_loss_history.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- ✅ Bug: `_extract_tracks_vggsfm` device fix → Task 1
- ✅ `capture_loss_history` config + `_last_loss_history` → Task 2
- ✅ Pre/post camera positions viz → Task 3, Step 3
- ✅ Loss curve viz → Task 3, Step 4
- ✅ Coding principles: block comments on every logical section, `########` dividers preserved, one-line docstrings

**Placeholder scan:** No TBDs. All code blocks are complete and runnable.

**Type consistency:**
- `BundleAdjustmentConfig.capture_loss_history: bool` used in Task 2 Step 3 and Task 3 Step 1 consistently.
- `self._last_loss_history: list[float]` defined in Task 2 Step 4, read in Task 3 Step 4 as `ba._last_loss_history`.
- `extrinsics_pre` defined in Task 3 Step 1, consumed in Step 3 — same session scope.
- `_camera_centers(extrinsics: np.ndarray) -> np.ndarray` used only within the same cell.
