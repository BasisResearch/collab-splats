"""Tests: result.model_intrinsics on the model grid, result.intrinsics its full-res undo."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vggt.utils.load_fn import load_and_preprocess_images

from collab_splats.geometry.transforms import rescale_intrinsics, shift_intrinsics
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward.loger import _compute_target_size
from collab_splats.pointcloud.feedforward.vggt_omega import VGGTOmegaCreator
from tests.pointcloud.conftest import _frame_files, _vggtx_boxes

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
        "depth": rng.random((N, model_h, model_w, 1)).astype(np.float32) + 0.1,
        "depth_conf": rng.random((N, model_h, model_w)).astype(np.float32),
    }


def _make_omega_creator(N: int = 2) -> VGGTOmegaCreator:
    """Build a VGGTOmegaCreator with minimal config for unit tests (no model loaded)."""
    creator = VGGTOmegaCreator.__new__(VGGTOmegaCreator)
    creator.conf_percentile = 50.0
    creator.max_points = 500_000
    creator.image_paths = [MagicMock() for _ in range(N)]
    # VGGTOmega-style original_coords: full image, no crop (AR in supported range)
    creator.original_coords = np.array(
        [[0, 0, 1080, 1920, 1080, 1920]] * N, dtype=np.float32
    )
    return creator


# ── VGGTOmega _postprocess invariant tests ─────────────────────────────────────


def test_omega_result_intrinsics_cx_inside_model_width():
    """After _postprocess, result.model_intrinsics cx must be < model_width."""
    raw = _make_raw_omega()
    creator = _make_omega_creator(N=2)
    result = creator._postprocess(raw)
    model_w = result.model_width
    for i in range(result.model_intrinsics.shape[0]):
        cx = result.model_intrinsics[i, 0, 2]
        assert cx < model_w, (
            f"frame {i}: cx={cx:.1f} >= model_width={model_w} "
            "— intrinsics stored at original-res instead of model-res"
        )


def test_omega_result_intrinsics_cy_inside_model_height():
    """After _postprocess, result.model_intrinsics cy must be < model_height."""
    raw = _make_raw_omega()
    creator = _make_omega_creator(N=2)
    result = creator._postprocess(raw)
    model_h = result.model_height
    for i in range(result.model_intrinsics.shape[0]):
        cy = result.model_intrinsics[i, 1, 2]
        assert cy < model_h, (
            f"frame {i}: cy={cy:.1f} >= model_height={model_h} "
            "— intrinsics stored at original-res instead of model-res"
        )


# ── reproject() scaling test ───────────────────────────────────────────────────


def test_reproject_uses_intrinsics_directly_no_scaling():
    """reproject() must use self.model_intrinsics without applying original_coords scaling.

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
    pixel_indices = np.array(
        [[0, int(cy), int(cx)], [1, int(cy), int(cx)]], dtype=np.int32
    )

    intrinsics = np.eye(3, dtype=np.float32)[None].repeat(N, axis=0)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fx
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # VGGTOmega-style: cr_x=1080 >> W=8; triggers broken scaling if still present
    original_coords = np.array([[0, 0, 1080, 1920, 1080, 1920]] * N, dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)  # identity W2C
    images = torch.zeros(N, 3, H, W)

    result = PointcloudResult(
        points=np.zeros((2, 3), dtype=np.float32),
        colors=np.zeros((2, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=None,
        model_intrinsics=intrinsics,
        image_paths=[],
        original_coords=original_coords,
        model_width=W,
        model_height=H,
        depth=depth,
        pixel_indices=pixel_indices,
        images=images,
    )
    reprojected = result.reproject()

    assert np.all(np.isfinite(reprojected.points)), (
        "reproject() returned non-finite points"
    )
    # Principal-point pixel + identity W2C → world x ≈ 0, y ≈ 0
    # Broken scaling gives fx_eff ≈ 0.37, x ≈ (4 - 0.030)/0.37 * 1.0 ≈ 10.7
    x_vals = reprojected.points[:, 0]
    assert np.all(np.abs(x_vals) < 0.01), (
        f"x={x_vals} — expected ~0 for principal-point pixels under identity W2C; "
        "large values indicate original_coords scaling was incorrectly applied"
    )


# ── VGGT-X crop boxes (VGGTXCreator._preprocess) ───────────────────────────────────────


def test_vggtx_crop_coords_portrait():
    """Portrait 1080×1920: height crop expected, cr_x = orig_w."""
    coords = _vggtx_boxes([(1080, 1920)])

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
    coords = _vggtx_boxes([(1920, 1080)])

    tl_x, tl_y, cr_x, cr_y, orig_w, orig_h = coords[0]
    assert tl_y == 0.0
    assert cr_y == orig_h  # no crop


def test_vggtx_crop_coords_cr_x_gt_target_size():
    """cr_x must always equal orig_w > target_size so the TSDF heuristic fires."""
    coords = _vggtx_boxes([(1080, 1920)])

    assert coords[0, 2] > 518  # cr_x = 1080 > model_W = 518


# ── LoGeR PINHOLE round-trip tests (added in Task 11) ──────────────────────────


def _full_res_k(
    model_intrinsics: np.ndarray, boxes: np.ndarray, model_wh: tuple[int, int]
) -> np.ndarray:
    """
    The full-res K PointcloudResult.__post_init__ derives from a model-grid K and crop boxes.
    """
    n = len(boxes)
    result = PointcloudResult(
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=np.eye(4, dtype=np.float32)[None].repeat(n, axis=0),
        intrinsics=None,
        model_intrinsics=model_intrinsics,
        image_paths=[Path(f"{i}.png") for i in range(n)],
        original_coords=np.asarray(boxes, dtype=np.float32),
        model_width=model_wh[0],
        model_height=model_wh[1],
    )
    return result.intrinsics


def _rescaled_camera_params(params, model_wh, orig_wh):
    """
    One PINHOLE camera's [fx, fy, cx, cy] undone from the model grid to a full-frame box.
    """
    orig_w, orig_h = orig_wh
    fx, fy, cx, cy = params
    K = np.array([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])

    # Rows are [x0, y0, x1, y1, W, H]: a full-frame crop box, so only the resize is undone
    box = np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]], dtype=np.float32)
    K = _full_res_k(K, box, model_wh)[0]
    return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]


def test_loger_pinhole_k_round_trips_to_original_resolution():
    """LoGeR's model-res K must rescale back to original resolution with fx != fy intact.

    _compute_target_size rounds each axis to a multiple of 14 independently, so a
    square-pixel physical camera genuinely produces fx != fy at model resolution.
    The per-axis rescale in PointcloudResult.__post_init__ (collab_splats/pointcloud/base.py)
    recovers the true focal on both axes.
    """
    orig_w, orig_h = 640, 480
    # 255_000 is LoGeR's shipping pixel budget (the LoGeRCreator.pixel_limit field default
    # in collab_splats/pointcloud/feedforward/loger.py), inlined rather than imported so
    # these tests stay independent of the creator.
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)

    # A square-pixel physical camera: one true focal, f = 1600 px at original resolution
    f = 1600.0
    scale_x, scale_y = model_w / orig_w, model_h / orig_h
    fx_model, fy_model = f * scale_x, f * scale_y

    # The anisotropy is real, not a rounding artifact — guard the premise of the test
    assert fx_model != pytest.approx(fy_model, rel=1e-3)

    # The full-res K keeps both focals
    params = _rescaled_camera_params(
        [fx_model, fy_model, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )
    # rel=1e-6: the full-res K is stored float32 (~6e-8 relative); 1e-8 flakes.
    assert params[0] == pytest.approx(f, rel=1e-6)
    assert params[1] == pytest.approx(f, rel=1e-6)


def test_full_res_k_applies_the_crop_origin():
    """
    A cropped frame's full-res K lands in source pixels: resize undone, crop origin added.
    """
    K = np.array([[[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]])
    coords = np.array([[11, 7, 59, 47, 70, 50]], dtype=np.float32)
    out = _full_res_k(K, coords, (12, 8))[0]

    # Pixel-center principal point (ADR 024): (c + 0.5) * scale - 0.5 + origin
    np.testing.assert_allclose(
        [out[0, 0], out[1, 1], out[0, 2], out[1, 2]], [40, 50, 36.5, 29]
    )


def test_full_res_k_uses_each_images_own_crop_box():
    """
    Two images with different crop boxes each rescale by their OWN box, not image 1's.
    """
    s1, s2 = 518 / 1000, 518 / 1080

    # Model-grid K from source K by the pixel-center rescale (ADR 024): (c + 0.5) * s - 0.5
    params = [
        [1000 * s1, 1000 * s1, 500.5 * s1 - 0.5, 500.5 * s1 - 0.5],
        [900 * s2, 880 * s2, 470.5 * s2 - 0.5, 590.5 * s2 - 0.5],
    ]
    K = np.array(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]] for fx, fy, cx, cy in params]
    )
    boxes = np.array(
        [[0, 0, 1000, 1000, 1000, 1000], [0, 420, 1080, 1500, 1080, 1920]],
        dtype=np.float32,
    )

    out = _full_res_k(K, boxes, (518, 518))

    np.testing.assert_allclose(
        [out[0, 0, 0], out[0, 1, 1], out[0, 0, 2], out[0, 1, 2]],
        [1000, 1000, 500, 500],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        [out[1, 0, 0], out[1, 1, 1], out[1, 0, 2], out[1, 1, 2]],
        [900, 880, 470, 1010],
        rtol=1e-5,
    )


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
        (
            1000,
            1040,
        ),  # near-square portrait, still cropped: new_h rounds 538.7 down to 532 > 518
        (1920, 1080),  # landscape: no crop, height resized to 294
        (1000, 1000),  # square: no crop
    ],
)
def test_vggtx_crop_box_round_trips_the_model_k(orig_wh):
    """
    A K moved onto the upstream model grid comes back through the box unchanged.
    """
    orig_w, orig_h = orig_wh
    box = _vggtx_boxes([orig_wh])[0]

    # Model K from the true upstream transform: per-axis scale, then the crop shifts cy
    sx, sy, start_y, model_h = _upstream_crop_grid(orig_w, orig_h)
    K = np.array(
        [[1000.0, 0.0, orig_w / 2], [0.0, 1000.0, orig_h / 2], [0.0, 0.0, 1.0]]
    )
    K_model = np.array(
        [
            [1000.0 * sx, 0.0, orig_w / 2 * sx],
            [0.0, 1000.0 * sy, orig_h / 2 * sy - start_y],
            [0.0, 0.0, 1.0],
        ]
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
        (
            100,
            101,
        ),  # near-square portrait, no crop: new_h 518, full-frame box under both formulas
        (192, 108),  # landscape: new_h 294, no crop
        (100, 100),  # square: new_h 518, no crop
    ],
)
def test_vggtx_crop_box_maps_loader_rows_to_their_source_rows(orig_wh, tmp_path):
    """
    Each row the real upstream loader outputs holds the source row the box maps it to.

    - frame is a vertical gray ramp, so a pixel's value encodes its source y
    - the old 518/orig_w y scale is off by mean 1.61 levels at 100x104, 0.49 at 108x192
    - tolerance 0.4 level: the correct box measures 0.27-0.32 (uint8 ramp quantization)
    """
    orig_w, orig_h = orig_wh
    box = _vggtx_boxes([orig_wh])[0]

    # Vertical ramp: row y holds 255 * (y + 0.5) / orig_h, its pixel-center coordinate
    ramp = np.round((np.arange(orig_h) + 0.5) / orig_h * 255).astype(np.uint8)
    frame = np.broadcast_to(ramp[:, None, None], (orig_h, orig_w, 3)).copy()
    paths = _frame_files([frame], tmp_path)
    out = load_and_preprocess_images([str(paths[0])], mode="crop")[0, 0].numpy() * 255

    # Loader grid must be the grid the box is paired with downstream
    _, _, _, model_h = _upstream_crop_grid(orig_w, orig_h)
    assert out.shape == (model_h, 518)

    # Model row centers through the box into source y; skip 3 rows of bicubic edge clamping
    rows = np.arange(model_h) + 0.5
    src_y = box[1] + rows * (box[3] - box[1]) / model_h
    err = np.abs(out[:, 259] - src_y / orig_h * 255)[3:-3]
    assert err.mean() < 0.4, f"mean row error {err.mean():.3f} gray levels"
