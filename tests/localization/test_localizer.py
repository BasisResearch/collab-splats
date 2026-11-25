import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from collab_splats.geometry.transforms import rescale_intrinsics, shift_intrinsics
from collab_splats.localization import (
    LocalFeatures,
    LocalizationResult,
    extractors,
)
from collab_splats.localization import localizer as localizer_mod
from collab_splats.localization import (
    retrieval,
    viz,
)
from collab_splats.localization.localizer import (
    CameraLocalizer,
    read_localization_db,
    seed_intrinsics,
)


def test_submodules_have_logger():
    for mod in (extractors, localizer_mod, retrieval, viz):
        assert hasattr(mod, "logger")
        assert isinstance(mod.logger, logging.Logger)


def _make_synthetic_scene(
    n_pts: int = 30, n_frames: int = 4, H: int = 480, W: int = 640
):
    """Planar synthetic scene: n_pts points at z=5 plus dense per-frame world maps.

    A fronto-parallel plane keeps the dense world map linear in pixel coords, so
    bilinear sampling in sample_world_points is exact at any subpixel location.
    """
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    pts3d[:, 2] = 5.0  # planar scene at z=5

    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))

    for i in range(n_frames):
        extrinsics[i, 0, 3] = i * 0.1  # slight lateral shift

    # Dense per-frame world maps: backproject each pixel at plane depth z=5, undo pose (R=I)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    cam = np.stack(
        [
            (xs - K[0, 2]) / K[0, 0] * 5.0,
            (ys - K[1, 2]) / K[1, 1] * 5.0,
            np.full(xs.shape, 5.0),
        ],
        -1,
    )
    world_points = np.stack(
        [cam - extrinsics[i, :3, 3] for i in range(n_frames)]
    ).astype(np.float32)

    return pts3d, world_points, extrinsics, K


def _projecting_matcher(stub_matcher, pts3d, extrinsics, K):
    """
    StubMatcher whose keypoints are pts3d projected through the frame id stored in pixel [0, 0, 0].
    """

    def keypoints(image):
        pose = extrinsics[int(image[0, 0, 0])]
        pts_cam = pts3d @ pose[:3, :3].T + pose[:3, 3]
        px = pts_cam @ K.T
        return px[:, :2] / px[:, 2:3]

    return stub_matcher(keypoints)


def _frame_images(n: int, H: int = 480, W: int = 640) -> list[np.ndarray]:
    """
    One image per frame, filled with its frame index so the matcher knows which pose to project.
    """
    return [np.full((H, W, 3), i, dtype=np.uint8) for i in range(n)]


def test_from_pointcloud_ids_default_to_result_image_paths(tmp_path, stub_matcher):
    """ids=None labels each frame by str() of result.image_paths."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    n = len(extrinsics)
    H, W = world_points.shape[1:3]
    result = SimpleNamespace(
        world_points=world_points,
        extrinsics=extrinsics,
        image_paths=[Path(f"images/frame_{i:06d}.png") for i in range(n)],
        original_coords=None,
    )
    images = _frame_images(n, H, W)

    loc = CameraLocalizer.from_pointcloud(
        result,
        zarr_path=tmp_path / "pc.zarr",
        images=images,
        extractor=_projecting_matcher(stub_matcher, pts3d, extrinsics, K),
    )

    assert loc.image_paths == [str(p) for p in result.image_paths]


def test_retrieval_name_selects_registry_model_and_keys_db(
    tmp_path, monkeypatch, stub_matcher, fake_salad
):
    """retrieval="megaloc" builds the retrieval model by registry name and stamps it on the saved DB."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    images = _frame_images(len(extrinsics))
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]
    requested = []

    # Registry stub records the requested name, returns the CPU stand-in
    registry = SimpleNamespace(get=lambda name: requested.append(name) or fake_salad)
    monkeypatch.setattr(localizer_mod, "BaseRetrievalExtractor", registry)

    # Construction embeds the references, which builds the retrieval model
    extractor = _projecting_matcher(stub_matcher, pts3d, extrinsics, K)
    loc = CameraLocalizer(
        world_points,
        extrinsics,
        images=images,
        ids=ids,
        extractor=extractor,
        retrieval="megaloc",
    )
    loc.save_index(tmp_path / "pc.zarr", "stub")

    group = zarr.open_group(str(tmp_path / "pc.zarr"), mode="r")[
        "local_features/stub/reconstruction"
    ]
    assert requested == ["megaloc"]
    assert group.attrs["retrieval"] == "megaloc"


def test_localization_db_exists_false_without_commit_marker(tmp_path):
    """A reconstruction group lacking the image_paths marker is a crashed build, not a DB."""
    zarr.open_group(str(tmp_path / "pc.zarr"), mode="w").require_group(
        "local_features/mock/reconstruction"
    )

    assert not localizer_mod.localization_db_exists(tmp_path / "pc.zarr", "mock")


def test_reference_frame_size_must_match_original_coords(stub_matcher):
    """Frames whose (w, h) differ from original_coords' (orig_w, orig_h) raise ValueError."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    n = len(extrinsics)
    H, W = world_points.shape[1:3]
    coords = np.tile(np.array([0, 0, 2 * W, 2 * H, 2 * W, 2 * H], np.float32), (n, 1))
    images = _frame_images(n, H, W)
    ids = [f"frame_{i:06d}" for i in range(n)]

    with pytest.raises(ValueError, match="original_coords"):
        CameraLocalizer(
            world_points,
            extrinsics,
            images=images,
            ids=ids,
            extractor=_projecting_matcher(stub_matcher, pts3d, extrinsics, K),
            original_coords=coords,
        )


def test_later_frame_size_mismatch_fails_at_its_chunk(stub_matcher):
    """A later frame whose (w, h) differs from its own original_coords row raises before later chunks extract."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()
    n = len(extrinsics)
    H, W = world_points.shape[1:3]
    coords = np.tile(np.array([0, 0, W, H, W, H], np.float32), (n, 1))
    images = _frame_images(n, H, W)
    images[2] = np.full((H // 2, W // 2, 3), 2, dtype=np.uint8)
    ids = [f"frame_{i:06d}" for i in range(n)]
    extractor = _projecting_matcher(stub_matcher, pts3d, extrinsics, K)

    with pytest.raises(ValueError, match="frame 2"):
        CameraLocalizer(
            world_points,
            extrinsics,
            images=images,
            ids=ids,
            extractor=extractor,
            original_coords=coords,
            batch_size=1,
        )

    assert extractor.n_extract == 2


def test_camera_localizer_recovers_known_pose(stub_matcher):
    """CameraLocalizer should recover the identity pose for a camera at extrinsics[0]."""
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()

    images = _frame_images(len(extrinsics))
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _projecting_matcher(stub_matcher, pts3d, extrinsics, K)
    loc = CameraLocalizer(
        world_points, extrinsics, images=images, ids=ids, extractor=extractor
    )

    # Query = camera 0 (identity extrinsic)
    query_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = loc.localize(query_image, K)

    assert isinstance(result, LocalizationResult)
    assert result.pose is not None, "localize() returned None pose — pycolmap failed"
    assert result.pose.shape == (4, 4)
    np.testing.assert_allclose(result.pose[:3, :3], extrinsics[0, :3, :3], atol=0.05)
    np.testing.assert_allclose(result.pose[:3, 3], extrinsics[0, :3, 3], atol=0.05)
    assert result.n_inliers > 0
    assert result.inlier_mask is not None
    assert result.inlier_mask.dtype == bool
    assert len(result.inlier_mask) == result.n_correspondences


def test_localization_result_fields_on_success(stub_matcher):
    pts3d, world_points, extrinsics, K = _make_synthetic_scene()

    images = _frame_images(len(extrinsics))
    ids = [f"frame_{i:04d}.png" for i in range(len(extrinsics))]

    extractor = _projecting_matcher(stub_matcher, pts3d, extrinsics, K)
    loc = CameraLocalizer(
        world_points, extrinsics, images=images, ids=ids, extractor=extractor
    )
    result = loc.localize(np.zeros((480, 640, 3), dtype=np.uint8), K)

    assert isinstance(result, LocalizationResult)
    assert result.pose is not None
    assert result.inlier_mask is not None
    assert result.inlier_mask.dtype == bool
    assert len(result.inlier_mask) == result.n_correspondences
    assert result.n_inliers > 0
    assert result.pts2d_ref is not None
    assert result.pts2d_ref.shape == result.pts2d.shape
    assert result.ref_frame_indices is not None
    assert len(result.ref_frame_indices) == result.n_correspondences


def test_localize_via_depth_lookup(stub_matcher):
    """Query identical to ref frame 0 localizes at ref 0's pose via world_points sampling."""
    wp, K = _plane_scene()
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [img],
        ["frame_0"],
        extractor=stub_matcher(_grid_keypoints),
    )

    res = loc.localize(img, query_intrinsics=K)

    assert res.pose is not None
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)
    assert res.n_correspondences >= 10
    assert res.ref_hw == (64, 64)


def _norm_feats(n=5, d=8, with_norm=True):
    return LocalFeatures(
        keypoints=torch.rand(n, 2) * 50,
        descriptors=torch.rand(n, d),
        keypoints_normalized=torch.rand(n, 2) * 2 - 1 if with_norm else None,
        image_size=(8, 8),
    )


def _localizer_replaying(feats, replay_matcher):
    """CameraLocalizer whose extractor replays the given per-frame features."""
    n = len(feats)
    extractor = replay_matcher(feats)
    return CameraLocalizer(
        world_points=np.zeros((n, 8, 8, 3), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        images=[np.zeros((8, 8, 3), dtype=np.uint8)] * n,
        ids=[f"f{i}" for i in range(n)],
        extractor=extractor,
    )


def test_save_load_roundtrips_keypoints_normalized(tmp_path, replay_matcher):
    feats = [_norm_feats() for _ in range(3)]
    _localizer_replaying(feats, replay_matcher).save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = read_localization_db(tmp_path / "ff.zarr", "loma")

    for orig, got in zip(feats, loaded):
        assert got.keypoints_normalized is not None
        np.testing.assert_array_equal(
            got.keypoints_normalized.numpy(), orig.keypoints_normalized.numpy()
        )


def test_save_omits_keypoints_normalized_when_any_frame_lacks_it(
    tmp_path, replay_matcher
):
    # A zero-filled normalized table is wrong data; write it only when every frame has it
    feats = [_norm_feats(), _norm_feats(with_norm=False), _norm_feats()]
    _localizer_replaying(feats, replay_matcher).save_index(tmp_path / "ff.zarr", "loma")
    loaded, _, _ = read_localization_db(tmp_path / "ff.zarr", "loma")
    assert all(f.keypoints_normalized is None for f in loaded)


def _plane_scene(H: int = 64, W: int = 64, f: float = 50.0, z: float = 2.0):
    """
    Fronto-parallel plane at depth z under a pinhole K, as one (1, H, W, 3) world map.
    """
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    wp = np.stack(
        [(xs - W / 2) / f * z, (ys - H / 2) / f * z, np.full(xs.shape, z)], -1
    ).astype(np.float32)
    return wp[None], K


def _grid_keypoints(image: np.ndarray) -> np.ndarray:
    """
    4x3 non-collinear keypoint grid scaled to the image.
    """
    h, w = image.shape[:2]
    return np.array(
        [
            [w * (0.125 + 0.22 * (i % 4)), h * (0.125 + 0.28 * (i // 4))]
            for i in range(12)
        ]
    )


def test_localize_refs_uses_only_chosen_frames(stub_matcher):
    wp, K = _plane_scene()
    wp3 = np.repeat(wp, 3, 0)
    img = np.zeros((64, 64, 3), np.uint8)
    loc = CameraLocalizer(
        wp3,
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        [img] * 3,
        ["a", "b", "c"],
        extractor=stub_matcher(_grid_keypoints),
    )

    res = loc.localize(img, query_intrinsics=K, refs=[2])

    assert res.pose is not None
    assert set(res.ref_frame_indices.tolist()) == {2}
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)


def test_localize_refs_refuses_non_reconstruction_frame(stub_matcher):
    wp, K = _plane_scene()
    img = np.zeros((64, 64, 3), np.uint8)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [img],
        ["a"],
        extractor=stub_matcher(_grid_keypoints),
    )

    with pytest.raises(ValueError, match="reconstruction"):
        loc.localize(img, query_intrinsics=K, refs=[1])


def test_crop_map_samples_matching_model_pixel():
    # Pixel-center map: model px m sits at full px (m + 0.5) / s - 0.5 + tl, s = 50 / 1000
    box = np.array([100, 0, 1100, 1000, 1200, 1000], np.float32)
    px_full = np.array([[100 + 20.5 * 20 - 0.5, 10.5 * 20 - 0.5]], np.float32)

    px_model = localizer_mod._crop_to_model_grid(px_full, box, (50, 50))

    np.testing.assert_allclose(px_model, [[20.0, 10.0]], atol=1e-5)


def _cropped_keypoints(image: np.ndarray) -> np.ndarray:
    """
    The model-grid keypoint grid placed where it lands in a 128x96 frame center-cropped to 96x96.
    """
    grid = _grid_keypoints(np.zeros((64, 64, 3)))
    return (grid + 0.5) * (96 / 64) - 0.5 + np.array([16, 0])


def test_localize_fullres_cropped_refs(stub_matcher):
    # Model grid 64x64; refs are 128x96 full-res frames whose center 96x96 crop became the grid
    wp, K_model = _plane_scene()
    box = np.array([[16, 0, 112, 96, 128, 96]], np.float32)
    K_full = shift_intrinsics(K_model, (0.5, 0.5))
    K_full = rescale_intrinsics(K_full, (64, 64), (96, 96))
    K_full = shift_intrinsics(K_full, (16 - 0.5, -0.5))
    ref = np.zeros((96, 128, 3), np.uint8)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [ref],
        ["a"],
        extractor=stub_matcher(_cropped_keypoints),
        original_coords=box,
    )

    res = loc.localize(ref, query_intrinsics=K_full, refs=[0])

    assert res.pose is not None
    np.testing.assert_allclose(res.pose, np.eye(4), atol=1e-2)
    assert res.ref_hw == (96, 128)


def _anisotropic_keypoints(image: np.ndarray) -> np.ndarray:
    """
    Model-grid pixels of a 64x48 grid placed in a 200x100 frame through crop box [20, 4, 180, 100].
    """
    return (_model_px() + 0.5) / np.array([64 / 160, 48 / 96]) - 0.5 + np.array([20, 4])


def _model_px() -> np.ndarray:
    """
    Six spread integer pixels on a 64x48 model grid.
    """
    return np.array(
        [[3, 5], [60, 7], [10, 40], [50, 44], [31, 22], [17, 33]], np.float64
    )


def test_localize_lookup_follows_crop_axes(stub_matcher):
    # Non-square grid, non-square crop, distinct x / y scales: an axis swap or scale slip moves the lookup
    H, W = 48, 64
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    wp = np.stack([xs * 0.1, ys * 0.3 + 1.0, 2.0 + xs * 0.01 + ys * 0.02], -1).astype(
        np.float32
    )[None]
    box = np.array([[20, 4, 180, 100, 200, 100]], np.float32)
    ref = np.zeros((100, 200, 3), np.uint8)
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [ref],
        ["a"],
        extractor=stub_matcher(_anisotropic_keypoints),
        original_coords=box,
        config={"refinement": {"refine_focal_length": False}},
    )

    res = loc.localize(ref, refs=[0])

    model_px = _model_px().astype(int)
    expected = wp[0, model_px[:, 1], model_px[:, 0]]
    np.testing.assert_allclose(res.pts3d_matched, expected, atol=1e-4)
    np.testing.assert_allclose(res.pts2d_ref, _anisotropic_keypoints(ref), atol=1e-4)
    assert res.ref_hw == (100, 200)


def _pnp_scene(f: float = 800.0, H: int = 480, W: int = 640, n: int = 200) -> tuple:
    """
    Random 3D points in front of a known world-to-camera pose, projected through a centered K.
    """
    rng = np.random.default_rng(3)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [[0.9950042, 0, 0.0998334], [0, 1, 0], [-0.0998334, 0, 0.9950042]]
    )
    pose[:3, 3] = [0.2, -0.1, 0.5]

    # Points in camera space inside the image, then back to world
    px = rng.uniform([20, 20], [W - 20, H - 20], (n, 2))
    depth = rng.uniform(3.0, 9.0, n)
    rays = np.column_stack([(px[:, 0] - cx) / f, (px[:, 1] - cy) / f, np.ones(n)])
    pts_cam = rays * depth[:, None]
    pts_world = (pts_cam - pose[:3, 3]) @ pose[:3, :3]
    return px.astype(np.float32), pts_world.astype(np.float32), pose, K


def _pnp_localizer(stub_matcher, config: dict | None = None) -> CameraLocalizer:
    """
    One-frame plane localizer, only used to reach _solve_pnp.
    """
    wp, _ = _plane_scene()
    img = np.zeros((64, 64, 3), np.uint8)
    return CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [img],
        ["a"],
        extractor=stub_matcher(_grid_keypoints),
        config=config,
    )


def test_solve_pnp_returns_refined_focal(stub_matcher):
    # Seed focal 1.2 * W is wrong; the returned K must be the one the pose was refined with
    px, pts, pose_true, K_true = _pnp_scene()
    seed = seed_intrinsics(480, 640)
    loc = _pnp_localizer(stub_matcher)
    frame = np.zeros(len(px), np.int32)

    res = loc._solve_pnp([px], [pts], [px], [frame], (480, 640), seed)

    assert res.pose is not None
    K = res.query_intrinsics
    assert K.shape == (3, 3) and K.dtype == np.float32
    assert K[0, 0] == K[1, 1]
    assert abs(K[0, 0] - 800.0) / 800.0 < 0.01
    np.testing.assert_allclose(K[:2, 2], [319.5, 239.5])
    np.testing.assert_allclose(res.pose, pose_true, atol=1e-2)
    assert seed[0, 0] == np.float32(1.2 * 640)


def test_solve_pnp_too_few_correspondences_keeps_seed(stub_matcher):
    px, pts, _, _ = _pnp_scene(n=3)
    seed = seed_intrinsics(480, 640)
    loc = _pnp_localizer(stub_matcher)

    res = loc._solve_pnp([px], [pts], [px], [np.zeros(3, np.int32)], (480, 640), seed)

    assert res.pose is None
    np.testing.assert_array_equal(res.query_intrinsics, seed)


def test_solve_pnp_ransac_failure_keeps_seed(stub_matcher):
    # Shuffled 3D points under a sub-pixel threshold leave RANSAC no consistent pose
    px, pts, _, _ = _pnp_scene(n=12)
    pts = pts[np.random.default_rng(1).permutation(len(pts))]
    seed = seed_intrinsics(480, 640)
    loc = _pnp_localizer(
        stub_matcher, config={"estimation": {"ransac": {"max_error": 0.01}}}
    )

    res = loc._solve_pnp(
        [px], [pts], [px], [np.zeros(len(px), np.int32)], (480, 640), seed
    )

    assert res.pose is None
    np.testing.assert_array_equal(res.query_intrinsics, seed)


def _recording_grid(shapes: list) -> Callable[[np.ndarray], np.ndarray]:
    """
    _grid_keypoints that also records each extracted image's shape.
    """

    def keypoints(image):
        shapes.append(image.shape[:2])
        return _grid_keypoints(image)

    return keypoints


@pytest.mark.parametrize(
    "query_hw, small_hw", [((128, 128), (64, 64)), ((100, 150), (43, 64))]
)
def test_localize_shrinks_larger_query_to_reference_long_side(
    stub_matcher, query_hw, small_hw
):
    # 64x64 refs: extraction runs on the per-axis rounded shrink, K and px come back on the query grid
    wp, _ = _plane_scene()
    ref = np.zeros((64, 64, 3), np.uint8)
    shapes = []
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [ref],
        ["a"],
        extractor=stub_matcher(_recording_grid(shapes)),
        config={"refinement": {"refine_focal_length": False}},
    )
    query = np.zeros((*query_hw, 3), np.uint8)
    K_small = np.array(
        [[50, 0, small_hw[1] / 2], [0, 50, small_hw[0] / 2], [0, 0, 1]],
        dtype=np.float32,
    )

    # Pixel-center K and px: shift to corner, scale, shift back
    scale = np.array([query_hw[1] / small_hw[1], query_hw[0] / small_hw[0]])
    K_full = shift_intrinsics(K_small, (0.5, 0.5))
    K_full = rescale_intrinsics(K_full, small_hw, query_hw)
    K_full = shift_intrinsics(K_full, (-0.5, -0.5))
    small_px = _grid_keypoints(np.zeros((*small_hw, 3)))
    expected_px = (small_px + 0.5) * scale - 0.5

    res = loc.localize(query, query_intrinsics=K_full)

    assert shapes[-1] == small_hw
    np.testing.assert_allclose(res.query_intrinsics, K_full, atol=1e-4)
    np.testing.assert_allclose(res.pts2d, expected_px, atol=1e-4)
    assert res.pose is not None


def test_localize_never_upscales_smaller_query(stub_matcher):
    wp, _ = _plane_scene()
    ref = np.zeros((64, 64, 3), np.uint8)
    shapes = []
    loc = CameraLocalizer(
        wp,
        np.eye(4, dtype=np.float32)[None],
        [ref],
        ["a"],
        extractor=stub_matcher(_recording_grid(shapes)),
    )
    query = np.zeros((32, 32, 3), np.uint8)

    res = loc.localize(query)

    assert shapes[-1] == (32, 32)
    np.testing.assert_allclose(res.pts2d, _grid_keypoints(query), atol=1e-4)


########################################################################
########## seed_intrinsics #############################################
########################################################################


def test_seed_landscape_focal_and_center():
    K = seed_intrinsics(480, 640)  # H, W
    f = 1.2 * 640  # 1.2 * max(W, H)
    assert K.shape == (3, 3)
    assert np.isclose(K[0, 0], f)  # fx
    assert np.isclose(K[1, 1], f)  # fy (square pixels)
    assert np.isclose(K[0, 2], 319.5)  # cx = (W - 1) / 2, pixel-center
    assert np.isclose(K[1, 2], 239.5)  # cy = (H - 1) / 2
    assert K.dtype == np.float32


def test_seed_portrait_uses_max_dimension():
    K = seed_intrinsics(800, 600)  # H > W
    f = 1.2 * 800
    assert np.isclose(K[0, 0], f)
    assert np.isclose(K[1, 1], f)
    assert np.isclose(K[0, 2], 299.5)
    assert np.isclose(K[1, 2], 399.5)


def test_localizationresult_carries_intrinsics_field():
    K = seed_intrinsics(480, 640)
    r = LocalizationResult(
        pose=None,
        n_correspondences=0,
        n_inliers=0,
        pts2d=None,
        pts3d_matched=None,
        inlier_mask=None,
        query_intrinsics=K,
    )
    assert np.allclose(r.query_intrinsics, K)


########################################################################
########## LocalizationResult.ranked_ref_frames ########################
########################################################################


def _make(ref_frame_indices, inlier_mask):
    return LocalizationResult(
        pose=None,
        n_inliers=int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        n_correspondences=len(ref_frame_indices)
        if ref_frame_indices is not None
        else 0,
        pts2d=None,
        pts3d_matched=None,
        pts2d_ref=None,
        inlier_mask=inlier_mask,
        ref_frame_indices=ref_frame_indices,
    )


def test_ranked_orders_by_inlier_count_desc():
    ref = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)
    mask = np.array([True, True, True, True, True, True])
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [2, 0, 1]


def test_ranked_excludes_zero_inlier_frames():
    ref = np.array([0, 1, 1], dtype=np.int32)
    mask = np.array([False, True, True])
    res = _make(ref, mask)
    assert res.ranked_ref_frames == [1]


def test_ranked_empty_when_no_result():
    assert _make(None, None).ranked_ref_frames == []
    ref = np.array([0, 1], dtype=np.int32)
    assert _make(ref, None).ranked_ref_frames == []


def test_ranked_empty_when_no_inliers():
    ref = np.array([0, 1, 2], dtype=np.int32)
    mask = np.array([False, False, False])
    assert _make(ref, mask).ranked_ref_frames == []


def test_ranked_breaks_ties_by_lowest_index():
    ref = np.array([2, 0, 1], dtype=np.int32)  # one inlier each
    mask = np.array([True, True, True])
    assert _make(ref, mask).ranked_ref_frames == [0, 1, 2]
