# Mesh Native-Resolution TSDF Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confidence-masked depth pre-integration + opt-in native-resolution TSDF fusion (original-res RGB from frames.zarr, guided-filter-upsampled depth, COLMAP original-res K).

**Architecture:** Both features live in the feedforward→TSDF adapter (`_feedforward_to_tsdf_inputs`); `Open3DTSDFFusion.create` stays resolution-agnostic, gaining only uint8 RGB passthrough and a principal-point guard. Two config keys, both default-off — shipping output byte-identical.

**Tech Stack:** numpy, cv2 (`boxFilter`/`resize` only — no ximgproc), open3d, zarr FrameStore. Python: `/opt/venv/reconstruction/bin/python`.

**Spec:** `docs/superpowers/specs/2026-08-18-mesh-native-res-fusion-design.md`

---

## File Map

- Modify: `collab_splats/mesh/utils.py` — new `guided_upsample_depth` + `_guided_filter`; extend `_feedforward_to_tsdf_inputs` and `pointcloud_to_mesh`
- Modify: `collab_splats/pointcloud/utils.py` — extract `confidence_mask` from `subsample_points`; both it and the mesh adapter use it (same filter rule as the pointcloud)
- Modify: `collab_splats/mesh/tsdf.py` — uint8 passthrough + principal-point guard in `create`
- Modify: `collab_splats/wrapper/reconstructor.py` — `_run_tsdf_mesh` + `mesh()` wiring
- Modify: `configs/base.yaml` — `mesh.conf_percentile`, `mesh.native_resolution`
- Test: `tests/mesh/test_utils.py`, `tests/mesh/test_tsdf.py`

---

### Task 1: `guided_upsample_depth`

**Files:** Modify `collab_splats/mesh/utils.py`, Test `tests/mesh/test_utils.py`

- [x] **Step 1: Write the failing tests** (append to tests/mesh/test_utils.py under a new `######## guided_upsample_depth` divider)

```python
def _step_scene(factor=4):
    """Model-res depth with a vertical step edge + RGB guide whose edge aligns with it."""
    h, w = 32, 32
    depth = np.full((h, w), 1.0, dtype=np.float32)
    depth[:, w // 2 :] = 2.0
    H, W = h * factor, w * factor
    rgb = np.full((H, W, 3), 40, dtype=np.uint8)
    rgb[:, W // 2 :] = 200
    return depth, rgb


def test_guided_upsample_depth_places_crop():
    """Output canvas is zero outside the crop box and populated inside it."""
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=2)
    canvas_hw = (100, 120)  # bigger than the 64x64 crop
    full_rgb = np.zeros((*canvas_hw, 3), dtype=np.uint8)
    full_rgb[10:74, 20:84] = rgb
    out = guided_upsample_depth(depth, full_rgb, crop_box=(20, 10, 84, 74), out_hw=canvas_hw)

    assert out.shape == canvas_hw
    assert out.dtype == np.float32
    assert np.all(out[:10] == 0) and np.all(out[74:] == 0)
    assert np.all(out[:, :20] == 0) and np.all(out[:, 84:] == 0)
    assert (out[10:74, 20:84] > 0).mean() > 0.99


def test_guided_upsample_depth_masked_pixels_stay_zero():
    """Depth==0 (masked / no observation) must never be resurrected by the filter."""
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=4)
    depth[8:16, 8:16] = 0.0  # masked block
    H, W = rgb.shape[:2]
    out = guided_upsample_depth(depth, rgb, crop_box=(0, 0, W, H), out_hw=(H, W))

    assert np.all(out[32:64, 32:64] == 0)  # the masked block, upsampled 4x
    valid = out[out > 0]
    assert valid.min() >= 1.0 - 1e-3 and valid.max() <= 2.0 + 1e-3  # no overshoot


def test_guided_upsample_depth_step_edge_stays_sharp():
    """The anti-bilinear property: an aligned guide edge keeps the depth step sharp.

    Bilinear at 4x smears intermediates across the whole kernel; the guided filter with a
    matching guide edge confines them to a thin transition band.
    """
    from collab_splats.mesh.utils import guided_upsample_depth

    depth, rgb = _step_scene(factor=4)
    H, W = rgb.shape[:2]
    out = guided_upsample_depth(depth, rgb, crop_box=(0, 0, W, H), out_hw=(H, W))

    interior = out[:, np.r_[0 : W // 2 - 8, W // 2 + 8 : W]]  # away from the edge band
    fabricated = (interior > 1.1) & (interior < 1.9)
    assert fabricated.mean() < 0.01
```

- [x] **Step 2: Run tests, verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -q -k guided_upsample`
Expected: 3 FAIL with `ImportError: cannot import name 'guided_upsample_depth'`

- [x] **Step 3: Implement** (in `collab_splats/mesh/utils.py`, new section after the clean_repair section; `import cv2` at top of file — cv2 is already an installed dep of the venv)

```python
########
# Guided depth upsampling — native-resolution TSDF fusion
########


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    """Normalized box filter, the O(1) primitive of the guided filter."""
    k = 2 * radius + 1
    return cv2.boxFilter(x, -1, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. gray-guide guided filter: edge-preserving smoothing of src steered by guide."""
    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    var_g = _box(guide * guide, radius) - mean_g * mean_g
    cov_gs = _box(guide * src, radius) - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


def guided_upsample_depth(
    depth: np.ndarray,
    rgb_full: np.ndarray,
    crop_box: tuple[int, int, int, int],
    out_hw: tuple[int, int],
    radius: int | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """Upsample a model-res depth map into its crop region of an original-res canvas.

    Nearest-neighbour resize (never fabricates depth), then a validity-weighted guided filter
    with the original-res RGB as guide snaps depth edges to image edges. Pixels that were 0
    (masked / no observation) in the source stay exactly 0. Canvas outside the crop box is 0.

    Args:
        depth:    (h, w) float32 model-res depth, 0 = no observation
        rgb_full: (H, W, 3) uint8 original-res frame (the guide)
        crop_box: (tl_x, tl_y, cr_x, cr_y) model crop in original pixels (original_coords[:4])
        out_hw:   (H, W) output canvas size (original_coords[4:6] reversed)
    """
    tl_x, tl_y, cr_x, cr_y = (int(round(v)) for v in crop_box)
    cw, ch = cr_x - tl_x, cr_y - tl_y

    # Nearest resize of depth and validity to crop size — blocky but never invents values
    depth_nn = cv2.resize(depth, (cw, ch), interpolation=cv2.INTER_NEAREST)
    valid_nn = (depth_nn > 0).astype(np.float32)

    # Gray guide in [0, 1] from the original-res crop; radius spans ~2x the upsample factor
    guide = cv2.cvtColor(rgb_full[tl_y:cr_y, tl_x:cr_x], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if radius is None:
        radius = max(1, int(np.ceil(2 * cw / depth.shape[1])))

    # Validity-weighted filtering: masked pixels contribute nothing to their neighbours
    num = _guided_filter(guide, depth_nn * valid_nn, radius, eps)
    den = _guided_filter(guide, valid_nn, radius, eps)
    filtered = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)

    # The guide must never resurrect deleted depth
    filtered[valid_nn == 0] = 0.0

    canvas = np.zeros(out_hw, dtype=np.float32)
    canvas[tl_y:cr_y, tl_x:cr_x] = filtered
    return canvas
```

- [x] **Step 4: Run tests, verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -q -k guided_upsample`
Expected: 3 passed. If the step-edge assertion fails marginally, tune `eps` down (sharper) — do not loosen the assertion past 0.02.

- [x] **Step 5: Commit**

```bash
git add collab_splats/mesh/utils.py tests/mesh/test_utils.py
git commit -m "feat(mesh): guided depth upsampling for native-resolution fusion"
```

---

### Task 2: Confidence masking + native-res adapter

**Files:** Modify `collab_splats/pointcloud/utils.py` (extract `confidence_mask`), `collab_splats/mesh/utils.py` (`_feedforward_to_tsdf_inputs`, `pointcloud_to_mesh`), Test `tests/pointcloud/test_utils.py`, `tests/mesh/test_utils.py`

- [x] **Step 0: Write the failing helper tests** (append to tests/pointcloud/test_utils.py, flat functions)

```python
def test_confidence_mask_global_percentile_strict():
    """Keep = strictly above the global cutoff — the subsample_points rule, shape-agnostic."""
    from collab_splats.pointcloud.utils import confidence_mask

    conf = np.array([[0.0, 0.0], [1.0, 1.0]])  # p50 cutoff = 0.5
    keep = confidence_mask(conf, 50.0)
    np.testing.assert_array_equal(keep, [[False, False], [True, True]])


def test_confidence_mask_uniform_confidence_keeps_all():
    """Uniform conf: nothing is strictly above the cutoff → keep everything, never delete-all."""
    from collab_splats.pointcloud.utils import confidence_mask

    keep = confidence_mask(np.full((3, 4), 0.7), 50.0)
    assert keep.all() and keep.shape == (3, 4)


def test_subsample_points_conf_filter_unchanged():
    """The refactor onto confidence_mask keeps subsample_points' output identical."""
    from collab_splats.pointcloud.utils import subsample_points

    rng = np.random.default_rng(0)
    pts = rng.random((100, 3))
    conf = np.concatenate([np.zeros(50), np.ones(50)])
    out, _ = subsample_points(pts, None, conf, max_points=1000, conf_percentile=50.0)
    np.testing.assert_array_equal(out, pts[50:])  # strict >: only the conf==1 half survives
```

Then implement in `collab_splats/pointcloud/utils.py` (new function above `subsample_points`), and refactor `subsample_points`' filter block to use it:

```python
def confidence_mask(conf: np.ndarray, percentile: float) -> np.ndarray:
    """Boolean keep-mask: conf strictly above the global percentile cutoff; all-True if none is.

    Strict > so a cutoff equal to the minimum still filters, while uniform conf (nothing
    above the cutoff) keeps everything rather than deleting everything. Shape-agnostic —
    the pointcloud path calls it on (P,) point confidences, the mesh path on (N, H, W) maps.
    """
    cutoff = np.percentile(conf, percentile)
    above = conf > cutoff
    return above if above.any() else np.ones(conf.shape, dtype=bool)
```

```python
    # Drop points at/below the conf cutoff (see confidence_mask for the edge-case semantics)
    if conf is not None and len(conf) > 0:
        above = confidence_mask(conf, conf_percentile)
        points = points[above]
        colors = colors[above] if colors is not None else None
```

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/pointcloud/test_utils.py -q`
Expected: new tests pass, existing tests untouched (behavior-preserving refactor).

- [x] **Step 1: Write the failing tests**

```python
def _tiny_ff_result(with_confidence=True):
    """Minimal FeedforwardResult for adapter tests: 2 frames, 8x8 model res, 16x16 original."""
    import torch

    from collab_splats.pointcloud.feedforward.base import FeedforwardResult

    n, h, w = 2, 8, 8
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 2.0, (n, h, w)).astype(np.float32)
    conf = np.zeros((n, h, w), dtype=np.float32)
    conf[:, :, : w // 2] = 1.0  # right half low-confidence
    ext = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    K = np.tile(np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]], np.float32), (n, 1, 1))
    return FeedforwardResult(
        points=np.zeros((1, 3), np.float32),
        colors=np.zeros((1, 3), np.uint8),
        extrinsics=ext,
        intrinsics=K,
        image_paths=[Path(f"f{i}.jpg") for i in range(n)],
        original_coords=np.tile(np.array([0, 0, 16, 16, 16, 16], np.float32), (n, 1)),
        model_width=w,
        model_height=h,
        images=torch.rand(n, 3, h, w),
        confidence=torch.from_numpy(conf) if with_confidence else None,
        depth=depth,
    )


def test_tsdf_inputs_conf_percentile_zeroes_low_confidence_depth():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result()
    depths, _, _, _ = _feedforward_to_tsdf_inputs(ff, conf_percentile=50.0)
    assert np.all(depths[:, :, 4:] == 0)  # low-confidence half masked
    assert np.all(depths[:, :, :4] > 0)  # high-confidence half untouched


def test_tsdf_inputs_conf_percentile_without_confidence_raises():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result(with_confidence=False)
    with pytest.raises(ValueError, match="confidence"):
        _feedforward_to_tsdf_inputs(ff, conf_percentile=50.0)


def test_tsdf_inputs_defaults_unchanged():
    """Options off → byte-identical to the pre-change adapter output."""
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    ff = _tiny_ff_result()
    depths, rgbs, c2w, K = _feedforward_to_tsdf_inputs(ff)
    np.testing.assert_array_equal(depths, ff.depth)
    assert rgbs.shape == (2, 8, 8, 3) and rgbs.dtype == np.float32
    np.testing.assert_array_equal(K, ff.intrinsics)


def test_tsdf_inputs_native_resolution_uses_store_rgb_and_upsampled_depth():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    class FakeStore:  # FrameStore duck-type: len + images()
        def __len__(self):
            return 2

        def images(self):
            return np.full((2, 16, 16, 3), 128, dtype=np.uint8)

    ff = _tiny_ff_result()
    native_K = ff.intrinsics * np.array([[2, 1, 2], [1, 2, 2], [1, 1, 1]], np.float32)
    depths, rgbs, _, K = _feedforward_to_tsdf_inputs(
        ff, frame_store=FakeStore(), native_intrinsics=native_K
    )
    assert depths.shape == (2, 16, 16)
    assert rgbs.dtype == np.uint8 and rgbs.shape == (2, 16, 16, 3)
    np.testing.assert_array_equal(K, native_K)
    assert (depths > 0).all()  # full-frame crop, no masking → fully populated


def test_tsdf_inputs_native_resolution_frame_count_mismatch_raises():
    from collab_splats.mesh.utils import _feedforward_to_tsdf_inputs

    class ShortStore:
        def __len__(self):
            return 1

        def images(self):
            return np.zeros((1, 16, 16, 3), dtype=np.uint8)

    ff = _tiny_ff_result()
    with pytest.raises(ValueError, match="[Ff]rame"):
        _feedforward_to_tsdf_inputs(ff, frame_store=ShortStore(), native_intrinsics=ff.intrinsics)
```

(`from pathlib import Path` joins the imports at top of the test file if not already there.)

- [x] **Step 2: Run tests, verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -q -k tsdf_inputs`
Expected: FAIL — `_feedforward_to_tsdf_inputs() got an unexpected keyword argument`

- [x] **Step 3: Implement.** Add `from collab_splats.pointcloud.utils import confidence_mask` to the imports at the top of `mesh/utils.py` (`pointcloud/utils.py` does not import from `mesh/` — no cycle). Replace `_feedforward_to_tsdf_inputs` body (keep the two existing None-guards verbatim at the top) and thread the options through `pointcloud_to_mesh`:

```python
def _feedforward_to_tsdf_inputs(
    result: FeedforwardResult,
    conf_percentile: float | None = None,
    frame_store=None,
    native_intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a FeedforwardResult into (depths, rgbs, c2w, intrinsics) for TSDF fusion.

    Default: everything at model resolution, mutually pixel-aligned, straight off the forward
    pass. `conf_percentile` zeroes depth below that per-frame confidence percentile (0 = no
    observation to Open3D). `frame_store` + `native_intrinsics` switch to native resolution:
    original-res uint8 RGB from frames.zarr, depth guided-upsampled into the model crop region
    (see guided_upsample_depth), and the caller's original-res K (the COLMAP camera).

    Raises:
        ValueError: depth/images missing; conf_percentile set but confidence absent;
                    frame_store length != frame count; frame_store without native_intrinsics.
    """
    # [existing result.depth / result.images None-guards stay here unchanged]

    depths = np.ascontiguousarray(result.depth, dtype=np.float32).copy()  # (N, H, W)

    # Confidence gate BEFORE any upsampling — never amplify pixels about to be deleted.
    # Same global-percentile rule as the pointcloud path (shared confidence_mask helper),
    # so the mesh inherits exactly the filter that makes the sparse cloud look clean.
    if conf_percentile is not None:
        if result.confidence is None:
            raise ValueError(
                "conf_percentile is set but this result has no confidence — re-run the "
                "pointcloud stage, or unset mesh.conf_percentile."
            )
        conf = result.confidence
        if hasattr(conf, "numpy"):
            conf = conf.detach().cpu().numpy()
        depths[~confidence_mask(conf, conf_percentile)] = 0.0
        logger.info(
            "Confidence mask (p%.0f): %.1f%% of depth pixels dropped",
            conf_percentile,
            100.0 * float((depths == 0).mean()),
        )

    c2w = invert_poses(result.extrinsics).astype(np.float32)

    # Native-resolution path: original-res RGB + guided-upsampled depth + caller's K
    if frame_store is not None:
        if native_intrinsics is None:
            raise ValueError("frame_store requires native_intrinsics (the original-res COLMAP K)")
        n = depths.shape[0]
        if len(frame_store) != n:
            raise ValueError(
                f"Frame-count mismatch: frames.zarr has {len(frame_store)} frames but the "
                f"reconstruction has {n} — they are from different runs."
            )
        rgbs = np.ascontiguousarray(frame_store.images())  # (N, H, W, 3) uint8
        out_hw = rgbs.shape[1:3]
        native_depths = np.zeros((n, *out_hw), dtype=np.float32)
        for i in tqdm(range(n), desc="Upsampling depth to native resolution"):
            native_depths[i] = guided_upsample_depth(
                depths[i], rgbs[i], crop_box=tuple(result.original_coords[i, :4]), out_hw=out_hw
            )
        return native_depths, rgbs, c2w, native_intrinsics.copy()

    # Model-resolution path (default): images is (N, 3, H, W) in [0, 1]
    imgs = result.images
    if hasattr(imgs, "numpy"):
        imgs = imgs.detach().cpu().numpy()
    rgbs = np.ascontiguousarray(imgs.transpose(0, 2, 3, 1), dtype=np.float32)  # (N, H, W, 3)

    return depths, rgbs, c2w, result.intrinsics.copy()
```

In `pointcloud_to_mesh`, add the three kwargs and forward them:

```python
def pointcloud_to_mesh(
    result: FeedforwardResult,
    output_dir: Path,
    method: str = "open3d_tsdf",
    conf_percentile: float | None = None,
    frame_store=None,
    native_intrinsics: np.ndarray | None = None,
    **mesher_kwargs,
) -> MeshResult:
    ...
    depths, rgbs, c2w, intrinsics = _feedforward_to_tsdf_inputs(
        result,
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
    )
```

(Docstring gains the three args; body otherwise unchanged.)

- [x] **Step 4: Run the full mesh test file**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_utils.py -q`
Expected: all pass (new + pre-existing — defaults path byte-identical).

- [x] **Step 5: Commit**

```bash
git add collab_splats/pointcloud/utils.py collab_splats/mesh/utils.py tests/pointcloud/test_utils.py tests/mesh/test_utils.py
git commit -m "feat(mesh): confidence masking + native-resolution TSDF adapter path

Masking reuses the pointcloud path's own filter rule via a shared
confidence_mask helper extracted from subsample_points."
```

---

### Task 3: `Open3DTSDFFusion.create` — uint8 passthrough + principal-point guard

**Files:** Modify `collab_splats/mesh/tsdf.py`, Test `tests/mesh/test_tsdf.py`

- [x] **Step 1: Write the failing tests** (append to tests/mesh/test_tsdf.py; reuse that file's existing synthetic-scene helpers for depths/c2w/K — if none fit, build a 2-frame flat-plane scene inline as below)

```python
def _flat_scene(h=32, w=32):
    """Two identity-pose frames looking at a flat plane at depth 1."""
    depths = np.ones((2, h, w), dtype=np.float32)
    c2w = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    K = np.tile(
        np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32), (2, 1, 1)
    )
    return depths, c2w, K


def test_create_uint8_rgb_matches_float(tmp_path):
    """uint8 RGB fuses to the same mesh as the equivalent [0,1] float RGB."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    depths, c2w, K = _flat_scene()
    rgb_u8 = np.full((2, 32, 32, 3), 200, dtype=np.uint8)
    rgb_f = rgb_u8.astype(np.float32) / 255.0

    m1 = Open3DTSDFFusion(output_dir=tmp_path / "a", voxel_size=0.05, sdf_trunc=0.2)
    m2 = Open3DTSDFFusion(output_dir=tmp_path / "b", voxel_size=0.05, sdf_trunc=0.2)
    p1 = m1.create(depths, rgb_u8, c2w, K).mesh_path
    p2 = m2.create(depths, rgb_f, c2w, K).mesh_path
    assert p1.read_bytes() == p2.read_bytes()


def test_create_principal_point_outside_grid_raises(tmp_path):
    """Original-res K paired with model-res depth must fail loudly, not fuse a collapsed mesh."""
    from collab_splats.mesh.tsdf import Open3DTSDFFusion

    depths, c2w, K = _flat_scene()
    K = K.copy()
    K[:, 0, 2] = 500.0  # cx far outside the 32-px grid
    rgbs = np.zeros((2, 32, 32, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="[Pp]rincipal point"):
        Open3DTSDFFusion(output_dir=tmp_path, voxel_size=0.05, sdf_trunc=0.2).create(
            depths, rgbs, c2w, K
        )
```

- [x] **Step 2: Run tests, verify they fail**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_tsdf.py -q`
Expected: uint8 test FAIL inside the `rgbs > 1.5` guard (`ValueError: rgbs must be in [0, 1]`); guard test FAIL with `DID NOT RAISE`.

- [x] **Step 3: Implement in `create()`**. Replace the `[0, 1]` guard block with:

```python
        # uint8 passes through untouched; float must be [0, 1] — [0, 255] float would wrap
        # to a black mesh instead of failing, which is how the Reconstructor path shipped
        # black meshes unnoticed
        is_uint8 = rgbs.dtype == np.uint8
        if not is_uint8 and rgbs.size and float(np.nanmax(rgbs)) > 1.5:
            raise ValueError(
                f"rgbs must be uint8 or float in [0, 1], got float max {float(np.nanmax(rgbs)):.3f}. "
                "Pass FeedforwardResult.images directly — it is already normalised."
            )
```

After `N, H, W = depths.shape`, add:

```python
        # The principal point must land inside the depth grid — the mismatched-resolution
        # pairing (original-res K with model-res depth, or vice versa) fuses a collapsed
        # mesh silently otherwise (the 2026-08-11 regression class)
        cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
        if not (np.all(cx > 0) and np.all(cx < W) and np.all(cy > 0) and np.all(cy < H)):
            raise ValueError(
                f"Principal point outside the {W}x{H} depth grid "
                f"(cx range [{cx.min():.0f}, {cx.max():.0f}], cy range [{cy.min():.0f}, {cy.max():.0f}]) "
                "— intrinsics and depth are at different resolutions."
            )
```

In the integration loop, the conversion line becomes:

```python
            rgb_u8 = (
                np.ascontiguousarray(rgbs[i])
                if is_uint8
                else (np.ascontiguousarray(rgbs[i]) * 255).astype(np.uint8)
            )
```

- [x] **Step 4: Run tests, verify they pass**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/mesh/test_tsdf.py -q`
Expected: all pass (existing 6 + new 2).

- [x] **Step 5: Commit**

```bash
git add collab_splats/mesh/tsdf.py tests/mesh/test_tsdf.py
git commit -m "feat(mesh): uint8 RGB passthrough + principal-point guard in TSDF fusion"
```

---

### Task 4: Wiring — `_run_tsdf_mesh`, `mesh()`, base.yaml

**Files:** Modify `collab_splats/wrapper/reconstructor.py:397-437` and `mesh()` (~line 909), `configs/base.yaml:83-88`, Test `tests/wrapper/test_reconstructor.py`

- [x] **Step 1: Write the failing test** (append to tests/wrapper/test_reconstructor.py, matching that file's existing style for config-default tests)

```python
def test_base_yaml_mesh_has_fidelity_keys():
    """New mesh keys exist and default OFF — shipping output stays byte-identical."""
    import yaml

    cfg = yaml.safe_load((Path(__file__).parents[2] / "configs" / "base.yaml").read_text())
    assert cfg["mesh"]["conf_percentile"] is None
    assert cfg["mesh"]["native_resolution"] is False
```

- [x] **Step 2: Run test, verify it fails**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/test_reconstructor.py -q -k fidelity_keys`
Expected: FAIL with `KeyError: 'conf_percentile'`

- [x] **Step 3: Add config keys** to `configs/base.yaml` mesh block:

```yaml
mesh:
  enabled: true
  voxel_size: 0.0025         # dashboard-parity TSDF defaults
  sdf_trunc: 0.01
  depth_trunc: 1.5          # drop depth samples beyond this range (metres)
  clean_repair: true       # drop out-of-bounds components + fill small holes, rewriting mesh.ply
  conf_percentile: null    # mask depth below this per-frame confidence percentile (null = off)
  native_resolution: false # fuse at original frame resolution (frames.zarr RGB + upsampled depth)
```

- [x] **Step 4: Wire `_run_tsdf_mesh`.** New signature and body changes:

```python
def _run_tsdf_mesh(
    result: "PointcloudResult",
    feedforward_zarr: Path,
    output_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clean_repair: bool = False,
    conf_percentile: float | None = None,
    native_resolution: bool = False,
    frames_zarr: Path | None = None,
) -> Path:
    """Fuse depth + RGB from feedforward.zarr into a TSDF mesh, using COLMAP poses."""
```

The model-res path loads images; the native path skips that load (frames.zarr supplies RGB) —
change the `load_zarr` line to:

```python
    ff = FeedforwardResult.load_zarr(
        feedforward_zarr, load_images=not native_resolution, load_world_points=False
    )
```

After `ff.extrinsics = result.extrinsics`, build the native-path arguments:

```python
    # Native path: original-res RGB from frames.zarr, COLMAP's original-res K as intrinsics
    frame_store = None
    native_intrinsics = None
    if native_resolution:
        from collab_splats.preproc.frame_store import FrameStore

        if frames_zarr is None or not frames_zarr.exists():
            raise FileNotFoundError(
                f"native_resolution requires frames.zarr (looked at {frames_zarr})"
            )
        frame_store = FrameStore.open(frames_zarr)
        native_intrinsics = result.intrinsics  # original-res by contract (build_colmap)
```

and pass through:

```python
    mesh_result = pointcloud_to_mesh(
        ff,
        output_dir,
        method="open3d_tsdf",
        conf_percentile=conf_percentile,
        frame_store=frame_store,
        native_intrinsics=native_intrinsics,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        clean_repair=clean_repair,
    )
```

- [x] **Step 5: Wire `mesh()`.** The `_run_tsdf_mesh` call gains:

```python
            conf_percentile=mesh_cfg["conf_percentile"],
            native_resolution=mesh_cfg["native_resolution"],
            frames_zarr=self.frames_zarr,
```

- [x] **Step 6: Run wrapper + mesh suites**

Run: `/opt/venv/reconstruction/bin/python -m pytest tests/wrapper/ tests/mesh/ -q`
Expected: all pass except the pre-existing `test_init_fills_defaults_from_base_yaml`
(`assert 2.0 == 1.0` — the user's uncommitted `fps: 2.0` edit, unrelated). If that test also
asserts the mesh block verbatim, update its expected dict for the two new keys — that change
IS in scope.

- [x] **Step 7: Commit**

```bash
git add collab_splats/wrapper/reconstructor.py configs/base.yaml tests/wrapper/test_reconstructor.py
git commit -m "feat(mesh): wire conf_percentile + native_resolution through the mesh stage"
```

---

### Task 5: Verification on the reference scene

**Files:** none (scratchpad scripts + measurements; results land in the plan checkboxes and final report)

Scene: `/workspace/outputs/2026_07_15-Goprosplat-GH010229/vggt_omega` (the clean_repair reference — 7.1M-face mesh territory). Run each in background/tmux, never in parallel (46.6 GB cap). Drive via a scratchpad script that calls `pointcloud_to_mesh` directly with `clean_repair=False` so component counts are measured pre-cleanup.

- [x] **Step 1: Regression — defaults off.** Re-fuse with `conf_percentile=None, native_resolution=False` into a scratch dir; compare against a same-code re-run from the pre-change commit (or the existing verify_mesh.ply baseline). Expected: byte-identical mesh.ply.
  **MEASURED 2026-08-18:** HEAD (7ca6ffa) vs pre-change (c3390d5, imported via PYTHONPATH worktree, provenance line in log): **byte-identical** (222,585,778 bytes). Both: 4,814,993 verts / 7,121,592 faces / 240,371 components; fuse 127 s (HEAD) vs 114 s (pre), peak RSS 12.5 GB. Params: voxel 0.0025, sdf_trunc 0.01, depth_trunc 1.5, clean_repair=False.
- [x] **Step 2: Masking A/B.** `conf_percentile ∈ {20, 40}`: record component count (`cluster_connected_triangles`), vertex count, wall-clock, peak RSS. Success: component count drops well below the 240k baseline at moderate vertex loss.
  **MEASURED 2026-08-18** (baseline p=null: 240,371 components / 4,814,993 verts / 127 s / 12.5 GB):
  | conf_percentile | components | verts | fuse s | peak RSS GB |
  |---|---|---|---|---|
  | 20 | 36,732 (−85%) | 1,822,336 (−62%) | 48.2 | 6.5 |
  | 40 | 5,918 (−97.5%) | 659,852 (−86%) | 30.7 | 5.4 |
  Renders: p40 removes floater speckle and closes ground holes; loss concentrates in low-confidence distant/peripheral content, main surfaces intact. Success criterion met.
- [x] **Step 3: Native-res run.** `native_resolution=True` (model-res baseline params otherwise): record wall-clock, peak RSS (cap 46.6 GB), vertex count; save side-by-side renders of a texture-rich crop (scratchpad, user eyeballs).
  **MEASURED 2026-08-18:** native alone: fuse 195 s, peak RSS 16.8 GB (under 46.6 GB cap), 4,938,494 verts / 428,853 components — denser rays integrate *more* low-conf speckle, so masking is the intended companion. Combined native + p40: fuse 99.5 s, 7.5 GB, 663,481 verts / **5,768 components** — cleanest run of the set. Renders (scratchpad `render_run_*.png`, camera-0 pose): native+p40 has visibly crisper grass/bark/curb texture vs model-res p40; awaiting user eyeball.
- [x] **Step 4: Update this plan's checkboxes with measured numbers; commit plan + memory update.**

---

## Self-review notes

- Spec coverage: masking (T2), guided upsample (T1), uint8+guards (T3), config/wiring (T4), verification (T5). Frames-order guard = count check (T2) — selection order is ascending frame_idx on both sides by construction.
- Types consistent: adapter returns the same 4-tuple everywhere; `frame_store` duck-typed (`__len__` + `images()`) so tests need no zarr fixture.
- No placeholders; every code step is complete.
