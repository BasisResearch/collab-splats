"""
Depth alignment: track-observation correspondences, the per-frame scale fit, and the
COLMAP-scale FeedforwardResult builder that consumes them.
"""

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from collab_splats.pointcloud import depth_align

GRID_H, GRID_W = 16, 32
CAM_W, CAM_H = 64, 32


########################################################
########## Correspondences and the scale fit ###########
########################################################


def _fake_reconstruction(per_image_depths):
    """
    Duck-typed pycolmap stand-in: one image per entry, observations at fixed pixels.

    - per_image_depths maps "frame_NNNNNN.jpg" -> list of (u_grid, v_grid, d_colmap).
      Pixels are given in DEPTH-GRID coordinates and scaled up to camera resolution here,
      so a test states where in the depth map an observation lands.
    """

    class _Camera:
        def __init__(self):
            self.width, self.height = CAM_W, CAM_H

    class _Point2D:
        def __init__(self, xy, point3D_id):
            self.xy = np.asarray(xy, dtype=np.float64)
            self.point3D_id = point3D_id

        def has_point3D(self):
            return self.point3D_id is not None

    class _Point3D:
        def __init__(self, xyz):
            self.xyz = np.asarray(xyz, dtype=np.float64)

    class _Image:
        def __init__(self, name, points2D):
            self.name = name
            self.camera_id = 1
            self.points2D = points2D

        def cam_from_world(self):
            # Identity pose: a point's world xyz IS its camera-frame xyz, so xyz[2] = d_colmap
            class _Pose:
                def matrix(self_inner):
                    return np.eye(4)

            return _Pose()

    points3D, images = {}, {}
    next_id = 1
    for image_id, (name, observations) in enumerate(per_image_depths.items(), start=1):
        points2D = []
        for u_grid, v_grid, d_colmap in observations:
            points3D[next_id] = _Point3D([0.0, 0.0, d_colmap])
            points2D.append(_Point2D([u_grid * CAM_W / GRID_W, v_grid * CAM_H / GRID_H], next_id))
            next_id += 1
        images[image_id] = _Image(name, points2D)

    class _Recon:
        pass

    recon = _Recon()
    recon.points3D = points3D
    recon.images = images
    recon.cameras = {1: _Camera()}
    return recon


def test_correspondences_pair_track_depth_with_sampled_vda_depth():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0), (7, 8, 20.0)]})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0
    depth[0, 8, 7] = 8.0

    pairs = depth_align._depth_correspondences(recon, ["frame_000000.jpg"], depth)
    assert len(pairs) == 1
    d_colmap, d_vda = pairs[0]
    np.testing.assert_allclose(sorted(d_colmap), [10.0, 20.0])
    np.testing.assert_allclose(sorted(d_vda), [5.0, 8.0])


def test_correspondences_drop_zero_out_of_bounds_and_behind_camera_samples():
    recon = _fake_reconstruction(
        {
            "frame_000000.jpg": [
                (3, 4, 10.0),  # kept
                (5, 5, 20.0),  # VDA depth left at 0 -> dropped
                (GRID_W + 8, 4, 10.0),  # pixel off the right edge of the depth grid -> dropped
                (7, 8, -10.0),  # point behind the camera, on a pixel that HAS depth -> dropped
            ]
        }
    )
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    depth[0, 4, 3] = 5.0
    depth[0, 8, 7] = 9.0

    d_colmap, d_vda = depth_align._depth_correspondences(recon, ["frame_000000.jpg"], depth)[0]
    np.testing.assert_allclose(d_colmap, [10.0])
    np.testing.assert_allclose(d_vda, [5.0])


def test_correspondences_raise_on_unregistered_name():
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0)]})
    depth = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)
    with pytest.raises(ValueError, match="not in reconstruction"):
        depth_align._depth_correspondences(recon, ["frame_000000.jpg", "frame_000009.jpg"], depth)


def test_scale_alignment_recovers_a_constant_ratio():
    # 30 observations at exactly 2x, three of them 100x wrong: the median holds at 2.0 while
    # the mean of the same ratios is 21.8
    observations = [(i % GRID_W, i % GRID_H, 2.0 * (i + 1)) for i in range(30)]
    observations[:3] = [(u, v, d * 100.0) for u, v, d in observations[:3]]
    recon = _fake_reconstruction({"frame_000000.jpg": observations})
    depth = np.zeros((1, GRID_H, GRID_W), dtype=np.float32)
    for i in range(30):
        depth[0, i % GRID_H, i % GRID_W] = float(i + 1)

    scales, stats = depth_align._fit_depth_scales(recon, ["frame_000000.jpg"], depth, min_obs=20)
    assert scales[0] == pytest.approx(2.0, rel=1e-6)
    assert stats["n_fallback"] == 0


def test_alignment_rejects_a_names_to_depth_row_mismatch():
    # One name short of the depth stack used to walk off the end of image_names silently,
    # aligning row i of the depth with frame i of a different list
    recon = _fake_reconstruction({"frame_000000.jpg": [(3, 4, 10.0)]})
    two_rows = np.zeros((2, GRID_H, GRID_W), dtype=np.float32)

    with pytest.raises(ValueError, match="rows would misalign"):
        depth_align._fit_depth_scales(recon, ["frame_000000.jpg"], two_rows, min_obs=20)


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


def test_tracked_point3d_ids_drops_observationless_points():
    # InstantSfM exports sub-min-track-length points with empty tracks — the result
    # tail must exclude them (pixel_indices reads track.elements[0])
    class _EmptyTrack:
        elements = []

    class _EmptyP3D:
        track = _EmptyTrack()

    recon = _recon_with_keypoint((1.0, 2.0))
    recon.points3D[3] = _EmptyP3D()
    assert depth_align._tracked_point3d_ids(recon) == [7]


def test_pixel_indices_from_reconstruction_scales_to_depth_res():
    # Original 800x600 -> depth 80x60 = scale 0.1; keypoint (200, 100) -> (row 10, col 20)
    idx = depth_align._pixel_indices_from_reconstruction(
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
    idx = depth_align._pixel_indices_from_reconstruction(
        _recon_with_keypoint((799.9, 599.9)),
        point3d_ids=[7],
        name_to_row={"frame_000000.jpg": 0},
        scale_x=0.1,
        scale_y=0.1,
        depth_hw=(60, 80),
    )
    assert 0 <= idx[0, 1] < 60 and 0 <= idx[0, 2] < 80


########################################################
########## result_from_reconstruction ##################
########################################################

ORIG_W, ORIG_H = 64, 48  # frames.zarr resolution
# VDA depth grid. The x and y ratios differ on purpose (1/4 vs 1/6): at a common ratio a
# swapped scale_x / scale_y is invisible in every assertion below.
DEPTH_W, DEPTH_H = 16, 8
K_PARAMS = [50.0, 50.0, 32.0, 24.0]  # fx, fy, cx, cy at ORIG res


def _pycolmap_scene(names, cam_w=ORIG_W, cam_h=ORIG_H):
    """
    One PINHOLE camera at cam_w x cam_h, one image per name, one point3D seen in every image.
    """
    recon = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="PINHOLE", width=cam_w, height=cam_h, params=K_PARAMS, camera_id=1)
    recon.add_camera_with_trivial_rig(cam)
    track = pycolmap.Track()
    for i, name in enumerate(names):
        im = pycolmap.Image(name=name, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([40.0, 20.0]))]
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([0.0, 0.0, float(i)]))
        recon.add_image_with_trivial_frame(im, pose)
        track.add_element(i + 1, 0)
    recon.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], dtype=np.uint8))
    return recon


def _scene_inputs(n=2):
    """
    (recon, depths, images, names) for an n-frame scene at constant VDA depth 2.0.
    """
    names = [f"frame_{i:06d}.jpg" for i in range(n)]
    recon = _pycolmap_scene([Path(x).stem for x in names])
    depths = np.full((n, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.stack([np.full((ORIG_H, ORIG_W, 3), 40 * (i + 1), dtype=np.uint8) for i in range(n)])
    return recon, depths, images, names


def test_result_from_reconstruction_shapes_and_k_rescaling():
    """
    Rows follow `names`; K is rescaled from camera res to the depth grid; images land at depth res.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    out, attrs = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    assert out.depth.shape == (2, DEPTH_H, DEPTH_W)
    assert out.images.shape == (2, 3, DEPTH_H, DEPTH_W)
    assert out.world_points.shape == (2, DEPTH_H, DEPTH_W, 3)
    assert out.model_width == DEPTH_W and out.model_height == DEPTH_H
    assert out.image_paths == [Path("frame_000000"), Path("frame_000001")]
    assert out.confidence is None
    np.testing.assert_allclose(out.original_coords[0], [0, 0, ORIG_W, ORIG_H, ORIG_W, ORIG_H])

    # The WHOLE K is rescaled, principal point included: fx was 50 at width 64 and the depth
    # grid is 16 wide -> 50 * 16/64. An original-res cx/cy against a model-res depth grid is
    # the 2026-08-11 mesh-collapse class, and world_points cannot catch it (depth is unaffected)
    np.testing.assert_allclose(out.intrinsics[0][0, 0], 50.0 * DEPTH_W / ORIG_W, rtol=1e-5)
    np.testing.assert_allclose(out.intrinsics[0][1, 1], 50.0 * DEPTH_H / ORIG_H, rtol=1e-5)
    np.testing.assert_allclose(out.intrinsics[0][0, 2], 32.0 * DEPTH_W / ORIG_W, rtol=1e-5)
    np.testing.assert_allclose(out.intrinsics[0][1, 2], 24.0 * DEPTH_H / ORIG_H, rtol=1e-5)

    # Poses are homogeneous w2c, one per frame in row order. unproject reads [:, :3, :] only,
    # so a broken bottom row would reach save_zarr with nothing to stop it
    assert out.extrinsics.shape == (2, 4, 4)
    np.testing.assert_allclose(out.extrinsics[:, 3], [[0.0, 0.0, 0.0, 1.0]] * 2)
    assert out.extrinsics[1][2, 3] == pytest.approx(1.0)

    # RGB carried into the [0, 1] float convention: frame i is a constant 40 * (i + 1)
    assert out.images.dtype == np.float32
    np.testing.assert_allclose(out.images[1], 80 / 255.0, atol=1e-3)

    # Sparse tail: the one point3D, its color, and the depth-grid pixel its first observation
    # lands on — the end-to-end check that the builder hands the helper the right sx/sy and row
    np.testing.assert_allclose(out.points, [[0.0, 0.0, 5.0]])
    assert out.points.dtype == np.float32
    assert out.colors.tolist() == [[10, 20, 30]] and out.colors.dtype == np.uint8
    assert out.pixel_indices.dtype == np.int32
    assert out.pixel_indices.tolist() == [[0, int(20 * DEPTH_H / ORIG_H), int(40 * DEPTH_W / ORIG_W)]]


def test_result_from_reconstruction_rescales_depth_to_the_colmap_world():
    """
    VDA depth 2.0 against a COLMAP depth of 5.0 -> scale 2.5, stamped in the attrs.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, attrs = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    np.testing.assert_allclose(out.depth, 5.0, rtol=1e-4)
    assert attrs["depth_scale"] == "colmap"
    np.testing.assert_allclose(attrs["depth_scales"], [2.5], rtol=1e-4)
    assert attrs["depth_scale_fallback_frames"] == []
    assert "depth_align_model" not in attrs


def test_result_from_reconstruction_reunprojects_world_points_from_aligned_depth():
    """
    world_points are re-derived from the ALIGNED depth, not scaled from the VDA-metric ones.
    """
    recon, depths, images, names = _scene_inputs(n=1)

    out, _ = depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)

    # identity pose for frame 0: the world z of every unprojected pixel is the aligned depth
    np.testing.assert_allclose(out.world_points[0, :, :, 2], 5.0, rtol=1e-4)


def test_result_from_reconstruction_refuses_camera_resolution_mismatch():
    """
    COLMAP cameras at a different resolution than the frames mean a stale staged set / SIFT db.
    """
    names = ["frame_000000.jpg"]
    recon = _pycolmap_scene(["frame_000000"], cam_w=128, cam_h=96)
    depths = np.full((1, DEPTH_H, DEPTH_W), 2.0, dtype=np.float32)
    images = np.zeros((1, ORIG_H, ORIG_W, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="camera resolution"):
        depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)


def test_result_from_reconstruction_refuses_partial_registration():
    """
    Fewer registered images than frames leaves rows without poses — refuse rather than pad.
    """
    recon, depths, images, names = _scene_inputs(n=2)
    names.append("frame_000002.jpg")
    depths = np.concatenate([depths, depths[:1]])
    images = np.concatenate([images, images[:1]])

    with pytest.raises(RuntimeError, match="partial"):
        depth_align.result_from_reconstruction(recon, depths, images, names, min_obs=1)


def test_result_from_reconstruction_refuses_name_mismatch():
    """
    Registered names that are not the requested stems mean two different runs.
    """
    recon, depths, images, _ = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="do not match"):
        depth_align.result_from_reconstruction(
            recon, depths, images, ["frame_000000.jpg", "frame_000007.jpg"], min_obs=1
        )


def test_result_from_reconstruction_refuses_a_frames_to_depth_row_mismatch():
    """
    One frame short of the depth stack is an opaque IndexError out of the resize, one over is
    a silent truncation — both are row misalignment, so refuse.
    """
    recon, depths, images, names = _scene_inputs(n=2)

    with pytest.raises(ValueError, match="1 frames for 2 depth maps"):
        depth_align.result_from_reconstruction(recon, depths, images[:1], names, min_obs=1)

    with pytest.raises(ValueError, match="3 frames for 2 depth maps"):
        depth_align.result_from_reconstruction(
            recon, depths, np.concatenate([images, images[:1]]), names, min_obs=1
        )
