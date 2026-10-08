"""
Depth module: VDA metric depth estimation and its alignment to an sfm model.

- estimate: the depth_vda/ cache and the stubbed inference path
- align: sparse-point pixel provenance, and align_depth's correspondence filtering,
  per-frame median fit, global-median fallback and PointcloudResult builder
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
import pytest
import torch

from collab_splats.geometry.transforms import shift_intrinsics
from collab_splats.pointcloud import depth as depth_mod
from tests.pointcloud._stubs import make_recon

_NAMES = ["frame_000000.jpg", "frame_000001.jpg"]


########################################################
########## Sparse-point pixel provenance ###############
########################################################


def _recon_with_keypoint(xy):
    """
    Minimal stand-in for pycolmap objects: one point3D (id 7) observed once in
    image 1 ("frame_000000.jpg") at original-res keypoint `xy`.
    """

    class _Elem:
        image_id, point2D_idx = 1, 0

    class _Track:
        elements = [_Elem()]

    class _P3D:
        track = _Track()

    class _Pt2D:
        pass

    _Pt2D.xy = np.asarray(xy, dtype=np.float64)

    class _Img:
        name = "frame_000000.jpg"
        points2D = [_Pt2D()]

    class _Recon:
        points3D = {7: _P3D()}
        images = {1: _Img()}

    return _Recon()


def test_empty_track_points_are_ignored():
    # InstantSfM exports sub-min-track-length points with empty tracks — the result
    # tail must exclude them (pixel_indices reads track.elements[0])
    recon, depths, images, names = _scene_inputs(n=1)
    recon.add_point3D(
        np.array([1.0, 1.0, 5.0]), pycolmap.Track(), np.array([1, 2, 3], np.uint8)
    )

    out, _ = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    assert out.points.shape == (1, 3)
    assert len(out.pixel_indices) == 1


def test_pixel_indices_from_reconstruction_scales_to_depth_res():
    # Original 800x600 -> depth 80x60 = scale 0.1; keypoint (200, 100) -> (row 10, col 20)
    idx = depth_mod._pixel_indices_from_reconstruction(
        _recon_with_keypoint((200.0, 100.0)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert idx.shape == (1, 3)
    assert idx.dtype == np.int32
    assert idx.tolist() == [[0, 10, 20]]  # [frame_row, row=y*0.1, col=x*0.1]


def test_pixel_indices_clamped_to_grid():
    # Edge keypoint -> scaled index must stay in-grid
    idx = depth_mod._pixel_indices_from_reconstruction(
        _recon_with_keypoint((799.9, 599.9)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert 0 <= idx[0, 1] < 60 and 0 <= idx[0, 2] < 80


def test_pixel_indices_floor_negative_subpixel_clamps_to_zero():
    # floor-then-clip reproduces int-then-clamp exactly, not just clamps
    # - pre-clip: col = -5.0 * 0.1 = -0.5, row = -3.0 * 0.1 = -0.3; floor gives (-1, -1)
    # - old int() truncation gave (0, 0) instead; both then clamp to the same cell (0, 0)
    idx = depth_mod._pixel_indices_from_reconstruction(
        _recon_with_keypoint((-5.0, -3.0)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert idx.tolist() == [[0, 0, 0]]


########################################################
########## align_depth #################################
########################################################

ORIG_W, ORIG_H = 64, 48  # original frame resolution (images/)
# VDA depth grid. The x and y ratios differ on purpose (1/4 vs 1/6): at a common ratio a
# swapped scale_x / scale_y is invisible in every assertion below.
DEPTH_W, DEPTH_H = 16, 8


def _scene_inputs(n=2):
    """
    (recon, depths, images, names) for an n-frame scene at constant VDA depth 2.0.
    """
    names = [f"frame_{i:06d}.jpg" for i in range(n)]
    recon = make_recon([Path(x).stem for x in names], cam_w=ORIG_W, cam_h=ORIG_H)
    depths = np.full((n, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.stack(
        [np.full((ORIG_H, ORIG_W, 3), 40 * (i + 1), dtype=np.uint8) for i in range(n)]
    )
    return recon, depths, images, names


def _recon_with_frames(frame_points):
    """
    Multi-point recon: each frame's own list of (world xyz, keypoint xy) observations.

    - one shared PINHOLE camera at (ORIG_W, ORIG_H); identity poses, so camera-space z equals world z
    - each pair becomes its own point3D, observed once, in the image at its list position
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(
        model="PINHOLE",
        width=ORIG_W,
        height=ORIG_H,
        params=[50.0, 50.0, 32.0, 24.0],
        camera_id=1,
    )
    recon.add_camera_with_trivial_rig(cam)
    pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.zeros(3))

    for i, points in enumerate(frame_points):
        im = pycolmap.Image(name=f"frame_{i:06d}", camera_id=1, image_id=i + 1)
        im.points2D = [
            pycolmap.Point2D(np.asarray(kp, dtype=np.float64)) for _, kp in points
        ]
        recon.add_image_with_trivial_frame(im, pose)

        for idx, (xyz, _) in enumerate(points):
            track = pycolmap.Track()
            track.add_element(i + 1, idx)
            recon.add_point3D(
                np.asarray(xyz, dtype=np.float64),
                track,
                np.array([10, 20, 30], dtype=np.uint8),
            )

    return recon


def _cell_keypoint(row, col):
    """
    Original-res keypoint that floor-samples to depth-grid (row, col) exactly, mid-cell.
    """
    return (col * (ORIG_W / DEPTH_W) + 2, row * (ORIG_H / DEPTH_H) + 2)


def test_align_depth_fallback_frame_gets_the_global_median_scale():
    """
    A frame under min_obs inherits the fitted frames' MEDIAN scale, not their mean.

    - three fitted frames at scales 1, 2, 9 median to 2.0; a mean would give 4.0
    - the thin frame is named in the fallback-frames attr
    """
    cells = [(4, 2), (4, 4), (4, 6)]
    vda = [
        2.0,
        3.0,
        4.0,
    ]  # shared by every fitted frame; colmap depth = scale * vda gives that scale as ratio

    fitted_frames = [
        [((0.0, 0.0, scale * v), _cell_keypoint(r, c)) for v, (r, c) in zip(vda, cells)]
        for scale in (1.0, 2.0, 9.0)
    ]
    # Thin frame: one observation only — below min_obs=3, so its own 100x ratio never fits
    thin_frame = [((0.0, 0.0, 100.0), _cell_keypoint(4, 2))]
    recon = _recon_with_frames(fitted_frames + [thin_frame])
    names = [f"frame_{i:06d}.jpg" for i in range(4)]

    depths = np.zeros((4, DEPTH_H, DEPTH_W), dtype=np.float32)
    for v, (r, c) in zip(vda, cells):
        depths[:3, r, c] = v
    depths[3, 4, 2] = (
        1.0  # thin frame's own ratio would be 100 / 1 = 100, but it never fits
    )
    images = np.zeros((4, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    out, attrs = depth_mod.align_depth(recon, depths, images, names, min_obs=3)

    np.testing.assert_allclose(attrs["depth_scales"], [1.0, 2.0, 9.0, 2.0], rtol=1e-6)
    assert attrs["depth_scale_fallback_frames"] == ["frame_000003"]
    np.testing.assert_allclose(
        out.depth[3, 4, 2], 2.0, rtol=1e-6
    )  # median 2.0, not mean 4.0 or its own 100x


def test_align_depth_refuses_when_no_frame_reaches_min_obs():
    """
    Every frame under min_obs is unrecoverable — refuse rather than silently skip alignment.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    with pytest.raises(ValueError, match="too sparse"):
        depth_mod.align_depth(recon, depths, images, names, min_obs=2)


def test_align_depth_filters_invalid_observations_before_fitting():
    """
    A minority of invalid observations doesn't move the fit — neither does a majority.

    - zero-VDA and behind-camera observations are each the majority in their own case, so a
      dropped filter pulls the median toward inf / negative instead of the lone valid ratio
    - off-grid stays a minority case: its filter is structural (in_bounds indexing), not a
      comparison a mutation could silently drop
    """
    # Each case: (name, [(world z, depth-grid cell)] for its 3 obs, vda-depth overrides, min_obs)
    # First obs is always ratio 5 / 2.0 = 2.5; a None cell means the keypoint is off the grid
    cases = [
        ("zero-vda", [(5.0, (4, 2)), (5.0, (4, 4)), (5.0, (4, 6))], {(4, 2): 2.0}, 1),
        (
            "behind-camera",
            [(5.0, (4, 2)), (-5.0, (4, 4)), (-3.0, (4, 6))],
            {(4, 2): 2.0, (4, 4): 1.0, (4, 6): 1.0},
            1,
        ),
        (
            "off-grid",
            [(5.0, (4, 2)), (10.0, (4, 4)), (5.0, None)],
            {(4, 2): 2.0, (4, 4): 4.0},
            2,
        ),
    ]

    off_grid_xy = (1000.0, _cell_keypoint(4, 6)[1])
    for name, obs, depth_cells, min_obs in cases:
        points = [
            ((0.0, 0.0, z), _cell_keypoint(*cell) if cell else off_grid_xy)
            for z, cell in obs
        ]
        recon = _recon_with_frames([points])
        depths = np.zeros((1, DEPTH_H, DEPTH_W), dtype=np.float32)
        for (row, col), value in depth_cells.items():
            depths[0, row, col] = value
        images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

        _, attrs = depth_mod.align_depth(
            recon, depths, images, ["frame_000000.jpg"], min_obs=min_obs
        )

        np.testing.assert_allclose(
            attrs["depth_scales"], [2.5], rtol=1e-6, err_msg=name
        )


def test_align_depth_scale_is_the_median_not_the_mean():
    """
    Three observations agree on a ratio, two are wild outliers — the median ignores them.
    """
    points = [
        ((0.0, 0.0, 4.0), _cell_keypoint(4, 2)),  # ratio 2.0
        ((0.0, 0.0, 6.0), _cell_keypoint(4, 4)),  # ratio 2.0
        ((0.0, 0.0, 8.0), _cell_keypoint(4, 6)),  # ratio 2.0
        ((0.0, 0.0, 1.0), _cell_keypoint(4, 8)),  # ratio 0.01, outlier low
        ((0.0, 0.0, 100.0), _cell_keypoint(4, 10)),  # ratio 100.0, outlier high
    ]
    recon = _recon_with_frames([points])
    names = ["frame_000000.jpg"]

    depths = np.zeros((1, DEPTH_H, DEPTH_W), dtype=np.float32)
    depths[0, 4, 2] = 2.0
    depths[0, 4, 4] = 3.0
    depths[0, 4, 6] = 4.0
    depths[0, 4, 8] = 100.0
    depths[0, 4, 10] = 1.0
    images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    _, attrs = depth_mod.align_depth(recon, depths, images, names, min_obs=3)

    np.testing.assert_allclose(attrs["depth_scales"], [2.0], rtol=1e-6)


def test_align_depth_shapes_and_k_rescaling():
    """
    Rows follow `names`; K is rescaled from camera res to the depth grid; images land at depth res.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    out, attrs = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    assert out.depth.shape == (2, DEPTH_H, DEPTH_W)
    assert out.images.shape == (2, 3, DEPTH_H, DEPTH_W)
    assert out.world_points.shape == (2, DEPTH_H, DEPTH_W, 3)
    assert out.model_width == DEPTH_W and out.model_height == DEPTH_H
    assert out.image_paths == [Path("frame_000000"), Path("frame_000001")]
    assert out.confidence is None
    np.testing.assert_allclose(
        out.original_coords[0], [0, 0, ORIG_W, ORIG_H, ORIG_W, ORIG_H]
    )

    # The WHOLE K is rescaled, principal point included: fx was 50 at width 64 and the depth
    # grid is 16 wide -> 50 * 16/64. An original-res cx/cy against a model-res depth grid is
    # the 2026-08-11 mesh-collapse class, and world_points cannot catch it (depth is unaffected)
    np.testing.assert_allclose(
        out.model_intrinsics[0][0, 0], 50.0 * DEPTH_W / ORIG_W, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.model_intrinsics[0][1, 1], 50.0 * DEPTH_H / ORIG_H, rtol=1e-5
    )
    # Principal point lands pixel-center: COLMAP's corner cx scaled, then 0.5 off
    np.testing.assert_allclose(
        out.model_intrinsics[0][0, 2], 32.0 * DEPTH_W / ORIG_W - 0.5, rtol=1e-5
    )
    np.testing.assert_allclose(
        out.model_intrinsics[0][1, 2], 24.0 * DEPTH_H / ORIG_H - 0.5, rtol=1e-5
    )

    # Poses are homogeneous w2c, one per frame in row order. unproject reads [:, :3, :] only,
    # so a broken bottom row would reach save_zarr with nothing to stop it
    assert out.extrinsics.shape == (2, 4, 4)
    np.testing.assert_allclose(out.extrinsics[:, 3], [[0.0, 0.0, 0.0, 1.0]] * 2)
    assert out.extrinsics[1][2, 3] == pytest.approx(1.0)

    # RGB carried into the [0, 1] float convention: frame i is a constant 40 * (i + 1)
    assert out.images.dtype == torch.float32
    np.testing.assert_allclose(out.images[1], 80 / 255.0, atol=1e-3)

    # Sparse tail: the one point3D, its color, and the depth-grid pixel its first observation
    # lands on — the end-to-end check that the builder hands the helper the right sx/sy and row
    np.testing.assert_allclose(out.points, [[0.0, 0.0, 5.0]])
    assert out.points.dtype == np.float32
    assert out.colors.tolist() == [[10, 20, 30]] and out.colors.dtype == np.uint8
    assert out.pixel_indices.dtype == np.int32
    assert out.pixel_indices.tolist() == [
        [0, int(20 * DEPTH_H / ORIG_H), int(40 * DEPTH_W / ORIG_W)]
    ]


def test_align_depth_rescales_depth_to_the_colmap_world():
    """
    VDA depth 2.0 against a COLMAP depth of 5.0 -> scale 2.5, stamped in the attrs.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, attrs = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    np.testing.assert_allclose(out.depth, 5.0, rtol=1e-4)
    assert attrs["depth_scale"] == "colmap"
    np.testing.assert_allclose(attrs["depth_scales"], [2.5], rtol=1e-4)
    assert attrs["depth_scale_fallback_frames"] == []
    assert "depth_align_model" not in attrs


def test_align_depth_reads_depth_through_cam_from_world():
    """
    Per-frame scales follow the w2c pose: frame 1 sits 1 unit further from the point.
    """
    # Point at world z=5 seen from two w2c poses
    # - translations z=0 and z=1: camera-frame depths 5 and 6
    # - against VDA depth 2.0: scales 2.5 and 3.0
    # - the inverse pose gives frame 1 depth 4: scale 2.0
    recon, depths, images, names = _scene_inputs(n=2)

    _, attrs = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    np.testing.assert_allclose(attrs["depth_scales"], [2.5, 3.0], rtol=1e-6)


def test_align_depth_reunprojects_world_points_from_aligned_depth():
    """
    world_points are re-derived from the ALIGNED depth, not scaled from the VDA-metric ones.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, _ = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    # identity pose for frame 0: the world z of every unprojected pixel is the aligned depth
    np.testing.assert_allclose(out.world_points[0, :, :, 2], 5.0, rtol=1e-4)


def test_align_depth_world_points_sit_on_colmap_pixel_centers():
    """
    Each depth pixel's world point projects to that pixel's COLMAP center, not its corner.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, _ = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    # Project every world point with the full-res COLMAP camera; identity pose for frame 0
    camera = next(iter(recon.cameras.values()))
    xy = camera.img_from_cam(out.world_points[0].reshape(-1, 3).astype(np.float64))

    # Depth pixel (row, col) covers full-res [col, col + 1) * sx, so its center is (col + 0.5) * sx
    rows, cols = np.mgrid[:DEPTH_H, :DEPTH_W]
    centers = np.stack(
        [(cols + 0.5) * ORIG_W / DEPTH_W, (rows + 0.5) * ORIG_H / DEPTH_H], axis=-1
    )
    np.testing.assert_allclose(xy, centers.reshape(-1, 2), atol=1e-3)


def test_align_depth_refuses_camera_resolution_mismatch():
    """
    COLMAP cameras at a different resolution than the frames mean a stale staged set / SIFT db.
    """
    names = ["frame_000000.jpg"]
    recon = make_recon(["frame_000000"], cam_w=128, cam_h=96)
    depths = np.full((1, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="camera resolution"):
        depth_mod.align_depth(recon, depths, images, names, min_obs=1)


def test_align_depth_refuses_partial_registration():
    """
    Fewer registered images than frames leaves rows without poses — refuse rather than pad.
    """
    recon, depths, images, names = _scene_inputs(n=2)
    names.append("frame_000002.jpg")
    depths = np.concatenate([depths, depths[:1]])
    images = np.concatenate([images, images[:1]])

    with pytest.raises(RuntimeError, match="subset to the registered frames"):
        depth_mod.align_depth(recon, depths, images, names, min_obs=1)


def test_align_depth_refuses_name_mismatch():
    """
    Registered names that are not the requested stems mean two different runs.
    """
    recon, depths, images, _ = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="do not match"):
        depth_mod.align_depth(
            recon, depths, images, ["frame_000000.jpg", "frame_000007.jpg"], min_obs=1
        )


def test_names_depths_mismatch_raises_before_alignment():
    """
    One name short of the depth stack is a row misalignment — refuse before aligning.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="2 image names for 1 depth maps"):
        depth_mod.align_depth(recon, depths[:1], images, names, min_obs=1)


def test_align_depth_refuses_a_frames_to_depth_row_mismatch():
    """
    One frame short of the depth stack is an opaque IndexError out of the resize, one over is
    a silent truncation — both are row misalignment, so refuse.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="1 frames for 2 depth maps"):
        depth_mod.align_depth(recon, depths, images[:1], names, min_obs=1)

    with pytest.raises(ValueError, match="3 frames for 2 depth maps"):
        depth_mod.align_depth(
            recon, depths, np.concatenate([images, images[:1]]), names, min_obs=1
        )


def test_align_depth_simple_radial_keeps_linear_k_per_grid():
    """
    A SIMPLE_RADIAL camera stores its linear K twice, pixel-center: keyframe grid and depth grid, k1 dropped.
    """
    # SIMPLE_RADIAL params (f, cx, cy, k1) on the 64x48 keyframe grid
    names = ["frame_000000.jpg"]
    recon = make_recon(
        ["frame_000000"], model="SIMPLE_RADIAL", params=(60.0, 28.0, 18.0, 0.1)
    )
    depths = np.full((1, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    out, _ = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    # Full-res K: calibration_matrix of the SIMPLE_RADIAL camera, one f for both axes, principal point -0.5
    np.testing.assert_array_equal(
        out.intrinsics[0], [[60.0, 0.0, 27.5], [0.0, 60.0, 17.5], [0.0, 0.0, 1.0]]
    )

    # Depth-grid K by hand: x by 16 / 64 = 1/4, y by 8 / 48 = 1/6, then -0.5
    # - fx 60 -> 15, cx 28 -> 6.5; fy 60 -> 10, cy 18 -> 2.5
    np.testing.assert_allclose(
        out.model_intrinsics[0],
        [[15.0, 0.0, 6.5], [0.0, 10.0, 2.5], [0.0, 0.0, 1.0]],
        rtol=1e-6,
    )


def test_align_depth_k_maps_back_through_its_box():
    """
    The stored model K re-derives the stored full-res K as feedforward results do: one convention, not two.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    out, _ = depth_mod.align_depth(recon, depths, images, names, min_obs=1)

    # PointcloudResult derives full-res K from the model K when intrinsics is None
    derived = replace(out, intrinsics=None).intrinsics
    assert out.model_intrinsics.dtype == np.float32
    np.testing.assert_allclose(derived, out.intrinsics, rtol=1e-6)

    # The full-res K is the COLMAP K shifted to pixel-center, stored float32
    K_center = shift_intrinsics(recon.cameras[1].calibration_matrix(), (-0.5, -0.5))
    assert out.intrinsics.dtype == np.float32
    np.testing.assert_array_equal(
        out.intrinsics,
        np.broadcast_to(K_center, out.intrinsics.shape).astype(np.float32),
    )


########################################################################
# _depth_cache_complete: the exact stem-set cache gate
########################################################################


def test_depth_cache_complete_is_an_exact_stem_set_check(tmp_path):
    """
    The gate compares stem sets: no dir, partial and leftover-extra are all incomplete.
    """
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert not depth_mod._depth_cache_complete(npy_dir, _NAMES)  # no dir

    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))
    assert not depth_mod._depth_cache_complete(npy_dir, _NAMES)  # partial

    np.save(npy_dir / "frame_000001.npy", np.ones((4, 4), dtype=np.float32))
    assert depth_mod._depth_cache_complete(npy_dir, _NAMES)  # exact

    np.save(npy_dir / "frame_000009.npy", np.ones((4, 4), dtype=np.float32))
    assert not depth_mod._depth_cache_complete(npy_dir, _NAMES)  # leftover extra file


########################################################################
# estimate_depth
########################################################################


def test_missing_clone_raises_actionable_import_error(tmp_path, monkeypatch):
    """
    No third_party clone -> ImportError naming setup.sh, before any weight download.
    """
    monkeypatch.setattr(depth_mod, "VDA_ROOT", tmp_path / "nope")

    # The ~1.5 GB fetch must not start when the clone that consumes it is absent
    monkeypatch.setattr(
        depth_mod,
        "load_hf_weights",
        lambda *a, **kw: pytest.fail("downloaded before the clone guard"),
    )
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)

    with pytest.raises(ImportError, match="setup.sh"):
        depth_mod.estimate_depth(frames, tmp_path, _NAMES)


def test_existing_depth_set_is_loaded_not_recomputed(tmp_path, monkeypatch):
    """
    An exact per-stem npy set short-circuits inference and comes back as the stacked array.
    """
    monkeypatch.setattr(
        depth_mod, "VDA_ROOT", tmp_path / "nope"
    )  # inference would raise
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for i, name in enumerate(_NAMES):
        np.save(
            npy_dir / f"{name[:-4]}.npy",
            np.full((4, 6), float(i + 1), dtype=np.float32),
        )

    depths = depth_mod.estimate_depth(
        np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES
    )

    assert depths.shape == (2, 4, 6)
    assert depths.dtype == np.float32
    np.testing.assert_allclose(depths[0], 1.0)
    np.testing.assert_allclose(depths[1], 2.0)


def test_partial_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    One map for two names is treated as missing -> the inference path -> missing clone raises.
    """
    monkeypatch.setattr(depth_mod, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        depth_mod.estimate_depth(
            np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES
        )


def test_wrong_named_depth_set_does_not_short_circuit(tmp_path, monkeypatch):
    """
    Right count, wrong stems (a stale selection): the gate compares stem sets, not counts.
    """
    monkeypatch.setattr(depth_mod, "VDA_ROOT", tmp_path / "nope")
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    for stem in ("frame_000000", "frame_000007"):
        np.save(npy_dir / f"{stem}.npy", np.ones((4, 4), dtype=np.float32))

    with pytest.raises(ImportError, match="setup.sh"):
        depth_mod.estimate_depth(
            np.zeros((2, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES
        )


def test_partial_depth_cache_is_wiped_and_regenerated(tmp_path, monkeypatch):
    """
    One stem missing plus a stale stem: depth_vda/ alone is dropped, then every map is rewritten.
    """

    class _StubModel:
        def infer_video_depth(self, frames, target_fps, input_size, device, fp32):
            n, h, w = frames.shape[:3]
            return np.full((n, h, w), 3.0, dtype=np.float32), target_fps

    monkeypatch.setattr(depth_mod, "_load_vda_model", lambda device: _StubModel())

    # A cached map for one requested stem, a stale one from another run, and a sibling file
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    npy_dir.mkdir(parents=True)
    np.save(npy_dir / "frame_000000.npy", np.full((4, 4), 9.0, dtype=np.float32))
    np.save(npy_dir / "frame_000007.npy", np.ones((4, 4), dtype=np.float32))
    (tmp_path / "depth_vda" / "leftover.txt").write_text("stale")

    # A sibling under out_dir (the mapper's SIFT DB in the sfm stage) is outside the wipe
    sibling = tmp_path / "colmap" / "keep.db"
    sibling.parent.mkdir()
    sibling.write_text("db")

    depths = depth_mod.estimate_depth(
        np.zeros((2, 40, 80, 3), dtype=np.uint8), tmp_path, _NAMES, depth_width=20
    )

    # Only the requested stems survive, all freshly inferred — the cached 9.0 map included
    assert sorted(p.name for p in npy_dir.glob("*.npy")) == [
        "frame_000000.npy",
        "frame_000001.npy",
    ]
    assert not (tmp_path / "depth_vda" / "leftover.txt").exists()
    assert sibling.read_text() == "db"
    np.testing.assert_array_equal(depths, 3.0)
    np.testing.assert_array_equal(np.load(npy_dir / "frame_000000.npy"), 3.0)


def test_names_and_frames_must_align(tmp_path):
    """
    Names are consumed positionally against frames — a length mismatch is a hard error.
    """
    with pytest.raises(ValueError, match="one-to-one"):
        depth_mod.estimate_depth(
            np.zeros((3, 32, 32, 3), dtype=np.uint8), tmp_path, _NAMES
        )


def test_inference_result_is_resized_and_written(tmp_path, monkeypatch):
    """
    A stubbed model's (N, H, W) output is nearest-resized to depth_width and written per stem.
    """

    class _StubModel:
        def infer_video_depth(self, frames, target_fps, input_size, device, fp32):
            # One step edge per frame, mirrored between them: a two-valued map makes the
            # interpolation mode observable (bilinear would emit a blended third value), and
            # the mirroring makes the two frames distinguishable so stem/content pairing is
            # checkable. The edge sits off-center on purpose: a centered one lands between two
            # same-valued bilinear taps, so even INTER_LINEAR would blend nothing.
            n, h, w = frames.shape[:3]
            maps = np.full((n, h, w), 1.0, dtype=np.float32)
            for i in range(n):
                if i % 2 == 0:
                    maps[i, :, 38:] = 100.0
                else:
                    maps[i, :, :42] = 100.0
            return maps, target_fps

    monkeypatch.setattr(depth_mod, "_load_vda_model", lambda device: _StubModel())
    frames = np.zeros((2, 40, 80, 3), dtype=np.uint8)

    depths = depth_mod.estimate_depth(frames, tmp_path, _NAMES, depth_width=20)

    assert depths.shape == (2, 10, 20)  # 80x40 -> 20x10 keeps the aspect ratio

    # 80 -> 20 is a 4x nearest decimation: dst col d samples src col 4d, so frame 0's edge at
    # src 38 lands between dst 9 and 10 (bilinear would blend 50.5 into dst 9). Pins the
    # sampling positions AND which frame's map came back under which stem.
    np.testing.assert_array_equal(
        depths[0], np.tile(np.where(np.arange(20) >= 10, 100.0, 1.0), (10, 1))
    )
    np.testing.assert_array_equal(
        depths[1], np.tile(np.where(np.arange(20) >= 11, 1.0, 100.0), (10, 1))
    )

    # Each map lands under its own frame's stem — content, not just the filename set
    npy_dir = tmp_path / "depth_vda" / "images" / "npy"
    assert sorted(p.name for p in npy_dir.glob("*.npy")) == [
        "frame_000000.npy",
        "frame_000001.npy",
    ]
    for i, name in enumerate(_NAMES):
        np.testing.assert_array_equal(np.load(npy_dir / f"{name[:-4]}.npy"), depths[i])


def test_estimate_depth_forwards_fp32(tmp_path, monkeypatch):
    """
    fp32 reaches infer_video_depth as a keyword.
    """
    seen = {}

    class _StubModel:
        def infer_video_depth(self, frames, target_fps, input_size, device, fp32):
            seen["fp32"] = fp32
            n, h, w = frames.shape[:3]
            return np.zeros((n, h, w), dtype=np.float32), target_fps

    monkeypatch.setattr(depth_mod, "_load_vda_model", lambda device: _StubModel())
    frames = np.zeros((2, 40, 80, 3), dtype=np.uint8)

    depth_mod.estimate_depth(frames, tmp_path, _NAMES, fp32=True)

    assert seen["fp32"] is True
