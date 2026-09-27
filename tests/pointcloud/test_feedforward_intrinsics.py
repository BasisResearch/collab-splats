"""Tests: result.intrinsics always at model resolution after _postprocess / reproject()."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pycolmap
import pytest
import torch
from PIL import Image as PILImage

from collab_splats.pointcloud.feedforward.base import (
    _CAMERA_PARAM_IDXS,
    FeedforwardResult,
    _rescale_reconstruction_to_original_dimensions,
)
from collab_splats.pointcloud.feedforward.loger import _compute_target_size
from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_raw_omega(model_h: int = 16, model_w: int = 8) -> dict:
    """Minimal raw_outputs dict as fixed VGGTOmega._forward returns.

    Uses small dims for speed; invariants are dimension-independent.
    model-res intrinsics: cx < model_W, cy < model_H.
    """
    N = 2
    rng = np.random.default_rng(0)
    intr = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intr[:, 0, 0] = 50.0
    intr[:, 1, 1] = 50.0
    intr[:, 0, 2] = model_w / 2  # cx well inside W
    intr[:, 1, 2] = model_h / 2  # cy well inside H
    return {
        "images": torch.zeros(N, 3, model_h, model_w),
        "extrinsic": np.eye(3, 4, dtype=np.float32)[None].repeat(N, axis=0),
        "intrinsics": intr,
        "intrinsics_downsampled": intr,
        "depth": rng.random((N, model_h, model_w, 1)).astype(np.float32) + 0.1,
        "depth_conf": rng.random((N, model_h, model_w)).astype(np.float32),
    }


def _make_omega_creator(N: int = 2) -> VGGTOmegaCreator:
    """Build a VGGTOmegaCreator with minimal config for unit tests (no model loaded)."""
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.conf_threshold = 50.0
    creator.max_points = 500_000
    creator.image_paths = [MagicMock() for _ in range(N)]
    # VGGTOmega-style original_coords: full image, no crop (AR in supported range)
    creator.original_coords = np.array([[0, 0, 1080, 1920, 1080, 1920]] * N, dtype=np.float32)
    return creator


# ── VGGTOmega _postprocess invariant tests ─────────────────────────────────────


def test_omega_result_intrinsics_cx_inside_model_width():
    """After _postprocess, result.intrinsics cx must be < model_width."""
    raw = _make_raw_omega()
    creator = _make_omega_creator(N=2)
    result = creator._postprocess(raw)
    model_w = result.model_width
    for i in range(result.intrinsics.shape[0]):
        cx = result.intrinsics[i, 0, 2]
        assert cx < model_w, (
            f"frame {i}: cx={cx:.1f} >= model_width={model_w} "
            "— intrinsics stored at original-res instead of model-res"
        )


def test_omega_result_intrinsics_cy_inside_model_height():
    """After _postprocess, result.intrinsics cy must be < model_height."""
    raw = _make_raw_omega()
    creator = _make_omega_creator(N=2)
    result = creator._postprocess(raw)
    model_h = result.model_height
    for i in range(result.intrinsics.shape[0]):
        cy = result.intrinsics[i, 1, 2]
        assert cy < model_h, (
            f"frame {i}: cy={cy:.1f} >= model_height={model_h} "
            "— intrinsics stored at original-res instead of model-res"
        )


# ── reproject() scaling test ───────────────────────────────────────────────────


def test_reproject_uses_intrinsics_directly_no_scaling():
    """reproject() must use self.intrinsics without applying original_coords scaling.

    Pixel at the principal point (u=cx, v=cy) with unit depth should unproject to
    (x≈0, y≈0, z=1) under identity extrinsics.  If original_coords scaling is
    (incorrectly) applied, cx_eff ≈ 0 and fx_eff ≈ 0.37 instead of 50 — the
    unprojected x is then ~11, not 0, so the tight bound < 0.01 catches the bug.
    """
    N, H, W = 2, 8, 8
    fx = 50.0
    cx, cy = 4.0, 4.0

    depth = np.ones((N, H, W), dtype=np.float32)
    # One point per frame, both at the principal point (col=cx, row=cy)
    pixel_indices = np.array([[0, int(cy), int(cx)], [1, int(cy), int(cx)]], dtype=np.int32)

    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fx
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # VGGTOmega-style: cr_x=1080 >> W=8; triggers broken scaling if still present
    original_coords = np.array([[0, 0, 1080, 1920, 1080, 1920]] * N, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity W2C
    images = torch.zeros(N, 3, H, W)

    result = FeedforwardResult(
        points=np.zeros((2, 3), dtype=np.float32),
        colors=np.zeros((2, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_paths=[],
        original_coords=original_coords,
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
        images=images,
    )
    reprojected = result.reproject()

    assert np.all(np.isfinite(reprojected.points)), "reproject() returned non-finite points"
    # Principal-point pixel + identity W2C → world x ≈ 0, y ≈ 0
    # Broken scaling gives fx_eff ≈ 0.37, x ≈ (4 - 0.030)/0.37 * 1.0 ≈ 10.7
    x_vals = reprojected.points[:, 0]
    assert np.all(np.abs(x_vals) < 0.01), (
        f"x={x_vals} — expected ~0 for principal-point pixels under identity W2C; "
        "large values indicate original_coords scaling was incorrectly applied"
    )


# ── _compute_vggtx_crop_coords tests (added in Task 6) ─────────────────────────


def test_vggtx_crop_coords_portrait():
    """Portrait 1080×1920: height crop expected, cr_x = orig_w."""
    from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords

    coords = _compute_vggtx_crop_coords([(1080, 1920)], target_size=518)

    assert coords.shape == (1, 6)
    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_x == 0.0
    assert cr_x == 1080.0  # full width always kept
    assert orig_w == 1080.0
    assert orig_h == 1920.0
    assert tl_y > 0  # portrait → height crop applied
    assert cr_y < orig_h  # crop ends before image bottom
    assert cr_y > tl_y  # non-empty crop


def test_vggtx_crop_coords_landscape_no_crop():
    """Landscape 1920×1080: no height crop (height stays ≤ 518 after width resize)."""
    from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords

    coords = _compute_vggtx_crop_coords([(1920, 1080)], target_size=518)

    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_y == 0.0
    assert cr_y == orig_h  # no crop


def test_vggtx_crop_coords_cr_x_gt_target_size():
    """cr_x must always equal orig_w > target_size so the TSDF heuristic fires."""
    from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords

    coords = _compute_vggtx_crop_coords([(1080, 1920)], target_size=518)

    assert coords[0, 2] > 518  # cr_x = 1080 > model_W = 518


# ── LoGeR PINHOLE round-trip tests (added in Task 11) ──────────────────────────


def _rescaled_camera_params(camera_model, params, model_wh, orig_wh):
    """Run the real rescale over a single camera and return its original-res params.

    _rescale_reconstruction_to_original_dimensions
    (collab_splats/pointcloud/feedforward/base.py) is duck-typed
    over pycolmap — it touches only .images/.cameras, .model.name, .params, .width,
    .height, .name and .camera_id (to look the camera up).  A SimpleNamespace stands in
    because constructing a real pycolmap.Reconstruction needs a Frame binding (`Check failed: image.HasFrameId()`)
    that is pure ceremony for a camera-only assertion.
    """
    model_w, model_h = model_wh
    orig_w, orig_h = orig_wh
    camera = SimpleNamespace(
        model=SimpleNamespace(name=camera_model),
        params=np.array(params, dtype=np.float64),
        width=model_w,
        height=model_h,
    )
    reconstruction = SimpleNamespace(
        images={1: SimpleNamespace(camera_id=1, name="0.png")},
        cameras={1: camera},
    )
    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png")],
        # A full-frame box gives s = model / orig and tl = (0, 0) on both axes.
        np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]], dtype=np.float32),
        (model_w, model_h),
    )
    return reconstruction.cameras[1].params


def test_loger_pinhole_k_round_trips_to_original_resolution():
    """LoGeR's model-res K must rescale back to original resolution with fx != fy intact.

    _compute_target_size rounds each axis to a multiple of 14 independently, so a
    square-pixel physical camera genuinely produces fx != fy at model resolution.
    PINHOLE carries both focals through build_pycolmap_reconstruction's camera-params
    branch, and the per-axis rescale in _rescale_reconstruction_to_original_dimensions
    recovers the true focal exactly on both axes (both in
    collab_splats/pointcloud/feedforward/base.py).
    """
    orig_w, orig_h = 640, 480
    # 255_000 is LoGeR's shipping pixel budget (the LoGeRCreator.pixel_limit field default
    # in collab_splats/pointcloud/feedforward/loger.py), inlined rather than imported so
    # these tests stay independent of
    # the creator — the camera model below is likewise passed as a literal.
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)

    # A square-pixel physical camera: one true focal, f = 1600 px at original resolution
    f = 1600.0
    scale_x, scale_y = model_w / orig_w, model_h / orig_h
    fx_model, fy_model = f * scale_x, f * scale_y

    # The anisotropy is real, not a rounding artifact — guard the premise of the test
    assert fx_model != pytest.approx(fy_model, rel=1e-3)

    # build_colmap's PINHOLE branch keeps both focals
    params = _rescaled_camera_params(
        "PINHOLE",
        [fx_model, fy_model, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )
    # params[1] is the load-bearing assertion.  The rescale scales by orig/model — the
    # RECIPROCAL of this test's local scale_x/scale_y — so its factors are 640/574 =
    # 1.1150 on x and 480/434 = 1.1060 on y.  SIMPLE_PINHOLE's max() therefore picks the
    # x factor, which is exactly what inverts fx_model = f * scale_x, so params[0]
    # round-trips under BOTH camera models.  Keep params[0] (it pins that x round-trips
    # at all) but do not delete params[1] believing params[0] covers the model choice.
    # rel=1e-6: original_image_sizes is float32, so the scales carry ~2.4e-8 relative
    # error (measured); 1e-8 flakes.
    assert params[0] == pytest.approx(f, rel=1e-6)
    assert params[1] == pytest.approx(f, rel=1e-6)


def test_simple_pinhole_would_lose_the_focal_loger_keeps():
    """Why LoGeRCreator.camera_model is PINHOLE: SIMPLE_PINHOLE costs 6.5 px here.

    Two production stages compose.  build_pycolmap_reconstruction averages fx and fy
    into one param on its SIMPLE_PINHOLE branch, then
    _rescale_reconstruction_to_original_dimensions multiplies that single param by
    max(scale_x, scale_y) on its SIMPLE_PINHOLE branch rather than per axis as it does
    for PINHOLE (both in collab_splats/pointcloud/feedforward/base.py).  Neither stage
    is lossy alone; together they do not round-trip.
    """
    orig_w, orig_h = 640, 480
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)
    f = 1600.0
    fx_model = f * (model_w / orig_w)
    fy_model = f * (model_h / orig_h)

    params = _rescaled_camera_params(
        "SIMPLE_PINHOLE",
        [(fx_model + fy_model) / 2.0, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )

    # Measured: 1606.5041 against a true 1600.0 — +6.504 px, +0.41%
    assert params[0] == pytest.approx(1606.5041, abs=1e-3)
    assert params[0] != pytest.approx(f, rel=1e-3)


def _rescale_one(camera_model, params, box, model_wh):
    """Run the real rescale on one duck-typed camera; returns (params, width, height)."""
    camera = SimpleNamespace(
        model=SimpleNamespace(name=camera_model),
        params=np.array(params, dtype=np.float64),
        width=model_wh[0],
        height=model_wh[1],
    )
    reconstruction = SimpleNamespace(
        images={1: SimpleNamespace(camera_id=1, name="0.png")},
        cameras={1: camera},
    )
    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png")],
        np.array([box], dtype=np.float32),
        model_wh,
    )
    return camera.params, camera.width, camera.height


def test_rescale_maps_a_cropped_portrait_camera_to_original_pixels():
    """VGGT-X portrait: 1080x1920 center-cropped to 1080x1080 (tl_y = 420), model 518x518."""
    box = [0, 420, 1080, 1500, 1080, 1920]
    s = 518 / 1080
    model_params = [900 * s, 880 * s, 470 * s, (1010 - 420) * s]  # fx, fy, cx, cy on the model grid

    params, width, height = _rescale_one("PINHOLE", model_params, box, (518, 518))

    np.testing.assert_allclose(params, [900, 880, 470, 1010], rtol=1e-5)
    assert (width, height) == (1080, 1920)


def test_rescale_keeps_distortion_params_for_opencv():
    """OPENCV params are fx, fy, cx, cy, k1, k2, p1, p2 — the tail is distortion, not cx/cy."""
    box = [0, 0, 1036, 1036, 1036, 1036]  # full frame, 2x
    params, _, _ = _rescale_one("OPENCV", [100, 100, 259, 259, 0.1, 0.2, 0.01, 0.02], box, (518, 518))

    np.testing.assert_allclose(params, [200, 200, 518, 518, 0.1, 0.2, 0.01, 0.02], rtol=1e-6)


def test_rescale_non_square_crop_with_nonzero_tl_does_not_swap_axes():
    """A non-square crop with tl_x, tl_y both nonzero pins per-axis, not per-image, scale.

    Crop [100, 50, 900, 650] out of a 1000x700 original is 800x600, resized onto the
    non-square 518x392 model grid: sx = 518/800 = 0.6475 != sy = 392/600 = 0.6533...
    A square crop (sx == sy) hides an sx/sy swap in the K update; this box makes the
    two axes numerically distinct.
    """
    box = [100, 50, 900, 650, 1000, 700]
    tl_x, tl_y = 100.0, 50.0
    sx, sy = 518 / 800, 392 / 600

    # Pick original-pixel intrinsics by hand, then project them onto the model grid
    fx, fy, cx, cy = 900.0, 850.0, 500.0, 350.0
    model_params = [fx * sx, fy * sy, (cx - tl_x) * sx, (cy - tl_y) * sy]

    params, width, height = _rescale_one("PINHOLE", model_params, box, (518, 392))

    # rtol=1e-5: original_image_sizes is float32, so the box carries ~2.5e-8
    # relative error through the divisions above (measured).
    np.testing.assert_allclose(params, [fx, fy, cx, cy], rtol=1e-5)
    assert (width, height) == (1000, 700)


def test_rescale_uses_each_images_own_crop_box():
    """Two images with different crop boxes must each rescale by their OWN box.

    Image 1 is full-frame (1000x1000, s = 0.518); image 2 is the same portrait
    crop used elsewhere in this file (1080x1920 cropped to 1080x1080, s = 518/1080).
    Using original_image_sizes[0] for every image would rescale image 2 with
    image 1's box, giving the wrong focal and principal point for it.
    """
    box1 = [0, 0, 1000, 1000, 1000, 1000]
    s1 = 518 / 1000
    box2 = [0, 420, 1080, 1500, 1080, 1920]
    s2 = 518 / 1080

    camera1 = SimpleNamespace(
        model=SimpleNamespace(name="PINHOLE"),
        params=np.array([1000 * s1, 1000 * s1, 500 * s1, 500 * s1], dtype=np.float64),
        width=518,
        height=518,
    )
    camera2 = SimpleNamespace(
        model=SimpleNamespace(name="PINHOLE"),
        params=np.array([900 * s2, 880 * s2, 470 * s2, (1010 - 420) * s2], dtype=np.float64),
        width=518,
        height=518,
    )
    reconstruction = SimpleNamespace(
        images={
            1: SimpleNamespace(camera_id=1, name="0.png"),
            2: SimpleNamespace(camera_id=2, name="1.png"),
        },
        cameras={1: camera1, 2: camera2},
    )

    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png"), Path("1.png")],
        np.array([box1, box2], dtype=np.float32),
        (518, 518),
    )

    np.testing.assert_allclose(camera1.params, [1000, 1000, 500, 500], rtol=1e-5)
    assert (camera1.width, camera1.height) == (1000, 1000)

    np.testing.assert_allclose(camera2.params, [900, 880, 470, 1010], rtol=1e-5)
    assert (camera2.width, camera2.height) == (1080, 1920)


def test_rescale_param_index_table_matches_pycolmap():
    """The model -> (focal, principal point) index table mirrors pycolmap's own."""
    # Guard against a vacuously empty table before the per-model loop below
    assert {"PINHOLE", "SIMPLE_PINHOLE"} <= _CAMERA_PARAM_IDXS.keys()

    for model, (focal, pp) in _CAMERA_PARAM_IDXS.items():
        cam = pycolmap.Camera(model=model, width=10, height=10)
        assert (list(cam.focal_length_idxs()), list(cam.principal_point_idxs())) == (focal, pp), model


# ── VGGT-X crop box describes upstream crop mode ───────────────────────────────


def _upstream_crop_grid(orig_w: int, orig_h: int) -> tuple[float, float, int, int]:
    """
    Upstream crop mode's (sx, sy, start_y, model_h) for one original size.

    - Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:237-251
    - resize to (518, round(h*518/w/14)*14), then center-crop a taller height to 518 rows
    """
    new_h = round(orig_h * 518 / orig_w / 14) * 14
    start_y = (new_h - 518) // 2 if new_h > 518 else 0
    return 518 / orig_w, new_h / orig_h, start_y, min(new_h, 518)


@pytest.mark.parametrize(
    "orig_wh",
    [
        (1080, 1920),  # portrait: new_h 924, y scale 924/1920, not 518/1080
        (1000, 1040),  # near-square portrait, still cropped: new_h rounds 538.7 down to 532 > 518
        (1920, 1080),  # landscape: no crop, height resized to 294
        (1000, 1000),  # square: no crop
    ],
)
def test_vggtx_crop_box_round_trips_the_model_k(orig_wh):
    """
    A K moved onto the upstream model grid comes back through the box unchanged.
    """
    from collab_splats.geometry.transforms import rescale_intrinsics, shift_intrinsics
    from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords

    orig_w, orig_h = orig_wh
    box = _compute_vggtx_crop_coords([orig_wh])[0]

    # Model K from the true upstream transform: per-axis scale, then the crop shifts cy
    sx, sy, start_y, model_h = _upstream_crop_grid(orig_w, orig_h)
    K = np.array([[1000.0, 0.0, orig_w / 2], [0.0, 1000.0, orig_h / 2], [0.0, 0.0, 1.0]])
    K_model = np.array(
        [[1000.0 * sx, 0.0, orig_w / 2 * sx], [0.0, 1000.0 * sy, orig_h / 2 * sy - start_y], [0.0, 0.0, 1.0]]
    )

    crop_hw = (box[3] - box[1], box[2] - box[0])
    back = rescale_intrinsics(K_model, (model_h, 518), crop_hw)
    back = shift_intrinsics(back, box[:2])
    np.testing.assert_allclose(back, K, atol=1e-2)


@pytest.mark.parametrize(
    "orig_wh",
    [
        (100, 104),  # near-square portrait, cropped: new_h 532; old box off 1.61 levels
        (108, 192),  # 9:16 portrait, cropped: new_h 924; old box off 0.49 levels
        (100, 101),  # near-square portrait, no crop: new_h 518, full-frame box under both formulas
        (192, 108),  # landscape: new_h 294, no crop
        (100, 100),  # square: new_h 518, no crop
    ],
)
def test_vggtx_crop_box_maps_loader_rows_to_their_source_rows(orig_wh):
    """
    Each row the real upstream loader outputs holds the source row the box maps it to.

    - frame is a vertical gray ramp, so a pixel's value encodes its source y
    - the old 518/orig_w y scale is off by mean 1.61 levels at 100x104, 0.49 at 108x192
    - tolerance 0.4 level: the correct box measures 0.27-0.32 (uint8 ramp quantization)
    """
    from vggt.utils.load_fn import load_and_preprocess_images

    from collab_splats.pointcloud.feedforward.base import frames_as_pil_source
    from collab_splats.pointcloud.feedforward.vggtx import _compute_vggtx_crop_coords

    orig_w, orig_h = orig_wh
    box = _compute_vggtx_crop_coords([orig_wh])[0]

    # Vertical ramp: row y holds 255 * (y + 0.5) / orig_h, its pixel-center coordinate
    ramp = np.round((np.arange(orig_h) + 0.5) / orig_h * 255).astype(np.uint8)
    frame = np.broadcast_to(ramp[:, None, None], (orig_h, orig_w, 3)).copy()
    with frames_as_pil_source([frame]):
        out = load_and_preprocess_images(["frame.png"], mode="crop")[0, 0].numpy() * 255

    # Loader grid must be the grid the box is paired with downstream
    _, _, _, model_h = _upstream_crop_grid(orig_w, orig_h)
    assert out.shape == (model_h, 518)

    # Model row centers through the box into source y; skip 3 rows of bicubic edge clamping
    rows = np.arange(model_h) + 0.5
    src_y = box[1] + rows * (box[3] - box[1]) / model_h
    err = np.abs(out[:, 259] - src_y / orig_h * 255)[3:-3]
    assert err.mean() < 0.4, f"mean row error {err.mean():.3f} gray levels"
