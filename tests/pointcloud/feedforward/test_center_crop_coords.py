"""center_crop_coords: shared crop-box helper and the three backend rules that call it."""

import numpy as np
import pytest

from mapanything.utils.image import RESOLUTION_MAPPINGS

from collab_splats.pointcloud.feedforward.base import center_crop_coords
from tests.pointcloud.conftest import _omega_boxes, _vggtx_boxes
from tests.pointcloud.feedforward.conftest import _mapanything_boxes

########################################################################
########## Frozen oracles ##############################################
########################################################################

# Pre-refactor crop-box functions, verbatim, as test-only oracles
# - source: collab_splats/pointcloud/feedforward/{vggtx,vggt_omega,mapanything}.py @ 1142edf8
# - replaced in production by center_crop_coords plus each backend's resize rule


def _old_vggtx(sizes: list[tuple[int, int]]) -> np.ndarray:
    """
    Crop window of upstream VGGT-X crop mode, in original-image pixels.

    - crop mode resizes width to 518, then center-crops a taller height to 518
    - the window lets the TSDF RGB loader and COLMAP rescale invert the transform

    Args:
        sizes: per-image (orig_w, orig_h).

    Returns:
        (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h]; cr_x is always orig_w.
    """
    # Upstream crop-mode target width, px
    # - Linketic/VGGT-X @ 26d1b95, vggt/utils/load_fn.py:211 (target_size = 518)
    # - crop mode resizes width to it (:238-242) and center-crops a taller height to it (:249-251)
    target_size = 518

    coords = []
    for orig_w, orig_h in sizes:
        # Upstream: resize width → target_size, maintain AR; round height to div-by-14
        scale = target_size / orig_w
        new_h_raw = orig_h * scale
        new_h = round(new_h_raw / 14) * 14  # divisible-by-14 rounding used upstream

        if new_h > target_size:
            # Height crop applied; the resized height is new_h, so y scales by new_h / orig_h
            sy = new_h / orig_h
            start_y_resized = (new_h - target_size) // 2
            tl_y = start_y_resized / sy
            cr_y = (start_y_resized + target_size) / sy
        else:
            tl_y = 0.0
            cr_y = float(orig_h)

        coords.append([0.0, tl_y, float(orig_w), cr_y, float(orig_w), float(orig_h)])

    return np.array(coords, dtype=np.float32)


def _old_omega(sizes: list[tuple[int, int]]) -> np.ndarray:
    """
    Crop window of Omega's aspect-ratio center crop, in original-image pixels.

    Args:
        sizes: per-image (orig_w, orig_h).

    Returns:
        (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    coords = []
    for width, height in sizes:
        aspect_ratio = height / max(width, 1)
        left, top, crop_width, crop_height = 0, 0, width, height

        # Center crop to the aspect band, integer offsets as the loader crops
        # - port: facebookresearch/vggt-omega @ 39a0cb8, vggt_omega/utils/load_fn.py:68-82
        if aspect_ratio < 0.5:
            crop_width = min(width, max(1, int(round(height / 0.5))))
            left = max((width - crop_width) // 2, 0)
        elif aspect_ratio > 2.0:
            crop_height = min(height, max(1, int(round(width * 2.0))))
            top = max((height - crop_height) // 2, 0)

        coords.append([left, top, left + crop_width, top + crop_height, width, height])

    return np.array(coords, dtype=np.float32)


def _old_mapanything(
    frame_hw: list[tuple[int, int]], model_w: int, model_h: int
) -> np.ndarray:
    """
    Crop box of MapAnything's loader per frame, in original pixels.

    - upstream: facebookresearch/map-anything @ c845b8f, mapanything/utils/cropping.py:231-240
      (scale = max(target / size) + 1e-8, floored resize) and :441-447 (centered crop)
    - force=True upstream (cropping.py:193): smaller frames are upscaled, never left as-is
    - box arithmetic follows upstream's intrinsics-derived scale (max(target / size) + 1e-8,
      the same scale `camera_matrix_of_crop` uses), not PIL's per-axis pixel ratio rw / w —
      the two differ by at most a few px at 4K

    Args:
        frame_hw: (height, width) of each original frame.
        model_w: model grid width.
        model_h: model grid height.

    Returns:
        (N, 6) float32 [tl_x, tl_y, cr_x, cr_y, orig_w, orig_h].
    """
    rows = []
    for h, w in frame_hw:
        # Resize so the image covers the target, as upstream does
        scale = max(model_w / w, model_h / h) + 1e-8
        rw, rh = int(np.floor(w * scale)), int(np.floor(h * scale))

        # Centered crop on the resized grid, mapped back to original pixels
        left, top = (rw - model_w) // 2, (rh - model_h) // 2
        rows.append(
            [
                left / scale,
                top / scale,
                (left + model_w) / scale,
                (top + model_h) / scale,
                w,
                h,
            ]
        )
    return np.array(rows, dtype=np.float32)


########################################################################
########## Sweep: new == old, bit for bit ##############################
########################################################################

# ~3.9k frame sizes, landscape and portrait, plus common capture sizes
_STEPS = (range(64, 4097, 97), range(64, 4097, 89))
_GRID = {(w, h) for a, b in (_STEPS, _STEPS[::-1]) for w in a for h in b}
_COMMON = {(1920, 1080), (1080, 1920), (3840, 2160), (640, 480), (518, 518)}
SIZES = sorted(_GRID | _COMMON)

# Every model grid MapAnything's fixed_mapping mode can pick (518 / 512 / 504 sets)
MAPANYTHING_GRIDS = sorted(
    {wh for table in RESOLUTION_MAPPINGS.values() for wh in table.values()}
)


def test_vggtx_boxes_equal_old_function():
    new = _vggtx_boxes(SIZES)
    old = _old_vggtx(SIZES)
    assert new.dtype == old.dtype == np.float32
    assert np.array_equal(new, old)


def test_omega_boxes_equal_old_function():
    new = _omega_boxes(SIZES)
    old = _old_omega(SIZES)
    assert new.dtype == old.dtype == np.float32
    assert np.array_equal(new, old)


@pytest.mark.parametrize("model_wh", MAPANYTHING_GRIDS)
def test_mapanything_boxes_equal_old_function(model_wh):
    new = _mapanything_boxes(SIZES, *model_wh)
    frame_hw = [(h, w) for w, h in SIZES]
    old = _old_mapanything(frame_hw, *model_wh)
    assert new.dtype == old.dtype == np.float32
    assert np.array_equal(new, old)


########################################################################
########## center_crop_coords ##########################################
########################################################################


def test_no_crop_is_full_frame():
    box = center_crop_coords((1920, 1080), (1920, 1080), (1920, 1080), (1, 1))
    assert box == [0, 0, 1920, 1080, 1920, 1080]


def test_crop_offsets_are_integer_in_the_resized_grid():
    # 1000x800 -> 300x240 at s = 0.3, crop 279x224: left (300 - 279) // 2 = 10, not 10.5
    s = 0.3
    box = center_crop_coords((1000, 800), (300, 240), (279, 224), (s, s))
    np.testing.assert_allclose(box, [10 / s, 8 / s, 289 / s, 232 / s, 1000, 800])
