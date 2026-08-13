"""Tests: result.intrinsics always at model resolution after _postprocess / reproject()."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage

from collab_splats.pointcloud.feedforward.base import (
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

    _rescale_reconstruction_to_original_dimensions (base.py:748-840) is duck-typed
    over pycolmap — it touches only .images/.cameras, .model.name, .params, .width,
    .height, .name, .camera_id (to look the camera up) and .points2D (iterated only
    when shift_point2d_to_original_res=True, which this helper leaves at its default
    False, so an empty list suffices).  A SimpleNamespace stands in because constructing a real
    pycolmap.Reconstruction needs a Frame binding (`Check failed: image.HasFrameId()`)
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
        images={1: SimpleNamespace(camera_id=1, name="0.png", points2D=[])},
        cameras={1: camera},
    )
    _rescale_reconstruction_to_original_dimensions(
        reconstruction,
        [Path("0.png")],
        # Rows are [x0, y0, x1, y1, W, H]; the rescale reads only the last two (base.py:793).
        np.array([[0, 0, orig_w, orig_h, orig_w, orig_h]], dtype=np.float32),
        (model_w, model_h),
    )
    return reconstruction.cameras[1].params


def test_loger_pinhole_k_round_trips_to_original_resolution():
    """LoGeR's model-res K must rescale back to original resolution with fx != fy intact.

    _compute_target_size rounds each axis to a multiple of 14 independently, so a
    square-pixel physical camera genuinely produces fx != fy at model resolution.
    PINHOLE carries both focals through (base.py:719-720) and the per-axis rescale
    (base.py:803-805) recovers the true focal exactly on both axes.
    """
    orig_w, orig_h = 640, 480
    # 255_000 is LoGeR's shipping pixel budget (LoGeRCreator.pixel_limit default,
    # loger.py:196), inlined rather than imported so these tests stay independent of
    # the creator — the camera model below is likewise passed as a literal.
    model_w, model_h = _compute_target_size(orig_w, orig_h, 255_000)

    # A square-pixel physical camera: one true focal, f = 1600 px at original resolution
    f = 1600.0
    scale_x, scale_y = model_w / orig_w, model_h / orig_h
    fx_model, fy_model = f * scale_x, f * scale_y

    # The anisotropy is real, not a rounding artefact — guard the premise of the test
    assert fx_model != pytest.approx(fy_model, rel=1e-3)

    # build_colmap's PINHOLE branch keeps both focals
    params = _rescaled_camera_params(
        "PINHOLE",
        [fx_model, fy_model, (model_w - 1) / 2.0, (model_h - 1) / 2.0],
        (model_w, model_h),
        (orig_w, orig_h),
    )
    # params[1] is the load-bearing assertion.  At 640x480 -> 574x434 scale_x > scale_y,
    # so SIMPLE_PINHOLE's max(scale_x, scale_y) IS scale_x and params[0] round-trips
    # under both camera models.  Keep params[0] (it pins that x round-trips at all) but
    # do not delete params[1] believing params[0] covers the model choice — it does not.
    # rel=1e-6: original_image_sizes is float32, so the scales carry ~2.4e-8 relative
    # error (measured); 1e-8 flakes.
    assert params[0] == pytest.approx(f, rel=1e-6)
    assert params[1] == pytest.approx(f, rel=1e-6)


def test_simple_pinhole_would_lose_the_focal_loger_keeps():
    """Why LoGeRCreator.camera_model is PINHOLE: SIMPLE_PINHOLE costs 6.5 px here.

    Two production stages compose.  build_colmap averages fx and fy into one param
    for SIMPLE_PINHOLE (base.py:721-722), then the rescale multiplies that single
    param by max(scale_x, scale_y) (base.py:801-802) rather than per axis
    (base.py:803-805).  Neither stage is lossy alone; together they do not round-trip.
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
