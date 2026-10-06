"""
Matcher tracks: synthetic posed scene, helper units, star-chain parity with the prototype.
"""

import importlib.machinery
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from collab_splats.geometry import tracks as tracks_mod
from collab_splats.localization.extractors import LocalFeatures, MatchResult
from collab_splats.preproc.frames import write_frames

PROTOTYPE = Path(__file__).parent / "data" / "match_tracks_prototype.py.txt"


########################################
# Synthetic scene
########################################


def _surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Smooth non-planar height field z = f(x, y), world units.
    """
    return 5.0 + 0.3 * np.sin(2.0 * x) + 0.3 * np.cos(2.0 * y)


def _scene(n_frames: int = 12, height: int = 60, width: int = 80, upscale: int = 2, n_points: int = 600) -> dict:
    """
    Posed cameras over the surface: dense world points, depth, and per-frame keypoint tables.

    - keypoints are exact projections of shared surface points, restricted to [0, W-1] x [0, H-1]
    - full-res keypoints invert the size map (kp + 0.5) * W / w - 0.5
    """
    rng = np.random.default_rng(0)
    K = np.array([[70.0, 0, width / 2], [0, 70.0, height / 2], [0, 0, 1]])
    intrinsics = np.repeat(K[None], n_frames, 0).astype(np.float32)

    # Cameras translate along x and yaw slightly
    extrinsics = np.tile(np.eye(4), (n_frames, 1, 1))

    for i in range(n_frames):
        yaw = 0.01 * i
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        centre = np.array([0.1 * i, 0.0, 0.0])
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = -R @ centre

    # Dense world points: per-pixel ray / surface intersection by fixed-point iteration
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    rays_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    world_points = np.zeros((n_frames, height, width, 3))
    depth = np.zeros((n_frames, height, width))

    for i in range(n_frames):
        R = extrinsics[i, :3, :3]
        centre = -R.T @ extrinsics[i, :3, 3]
        rays = rays_cam @ R
        s = np.full((height, width), 5.0)

        for _ in range(50):
            pts = centre + rays * s[..., None]
            s = (_surface(pts[..., 0], pts[..., 1]) - centre[2]) / rays[..., 2]

        world_points[i] = centre + rays * s[..., None]
        depth[i] = s

    # Shared surface points, then their projections per frame
    xy = rng.uniform([-1.5, -1.5], [2.5, 1.5], (n_points, 2))
    points = np.column_stack([xy, _surface(xy[:, 0], xy[:, 1])])
    kp_model, kp_ids = [], []

    for i in range(n_frames):
        cam = points @ extrinsics[i, :3, :3].T + extrinsics[i, :3, 3]
        px = cam @ K.T
        px = px[:, :2] / px[:, 2:]
        inside = (px[:, 0] >= 0) & (px[:, 0] <= width - 1) & (px[:, 1] >= 0) & (px[:, 1] <= height - 1)
        kp_model.append(px[inside].astype(np.float32))
        kp_ids.append(np.flatnonzero(inside))

    kp_full = [(kp + 0.5) * upscale - 0.5 for kp in kp_model]
    return {
        "K": intrinsics,
        "extrinsics": extrinsics.astype(np.float32),
        "world_points": world_points.astype(np.float32),
        "depth": depth.astype(np.float32),
        "kp_full": kp_full,
        "kp_ids": kp_ids,
        "hw": (height, width),
        "full_hw": (height * upscale, width * upscale),
    }


def _write_store(tmp_path: Path, scene: dict) -> list[Path]:
    """
    Full-res frame store: frame i is a flat image whose pixel (0, 0) holds i.
    """
    H, W = scene["full_hw"]
    n = len(scene["extrinsics"])
    frames = np.full((n, H, W, 3), 200, np.uint8)
    frames[:, 0, 0, 0] = np.arange(n)
    return write_frames(tmp_path / "images", frames, list(range(n)))


def _model_images(scene: dict) -> np.ndarray:
    """
    (N, 3, H, W) model-grid images in [0, 1]; pixel (0, 0) of channel 0 encodes the frame.
    """
    H, W = scene["hw"]
    n = len(scene["extrinsics"])
    images = np.full((n, 3, H, W), 200 / 255, np.float32)
    images[:, 0, 0, 0] = np.arange(n) / 255
    return images


class _FakeMatcher:
    """
    Matcher over the scene's keypoint tables; descriptors carry the surface point id.
    """

    def __init__(self, scene: dict) -> None:
        self._scene = scene
        self._matcher = types.SimpleNamespace(max_num_keypoints=2048)

    def _one(self, image: np.ndarray) -> LocalFeatures:
        i = int(image[0, 0, 0])
        kp = torch.from_numpy(self._scene["kp_full"][i].copy())
        ids = torch.from_numpy(self._scene["kp_ids"][i].astype(np.float32))
        return LocalFeatures(keypoints=kp, descriptors=ids[:, None], image_size=(image.shape[1], image.shape[0]))

    def extract(self, images):
        if isinstance(images, list):
            return [self._one(im) for im in images]
        return self._one(images)

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        return features

    def match(self, q: LocalFeatures, db: LocalFeatures) -> MatchResult:
        q_ids = q.descriptors[:, 0].numpy()
        db_ids = db.descriptors[:, 0].numpy()
        _, idx_q, idx_db = np.intersect1d(q_ids, db_ids, return_indices=True)
        return MatchResult(
            query_px=q.keypoints.numpy()[idx_q],
            ref_px=db.keypoints.numpy()[idx_db],
            idx_q=idx_q.astype(np.int64),
            idx_db=idx_db.astype(np.int64),
        )


class _FakeSalad(torch.nn.Module):
    """
    Global descriptor seeded by the frame index in pixel (0, 0); deterministic, CPU.
    """

    def __init__(self, device: str | None = None) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        idx = (images[:, 0, 0, 0] * 255).round().long()
        desc = torch.stack([torch.from_numpy(np.random.default_rng(int(i)).normal(size=16)) for i in idx])
        return torch.nn.functional.normalize(desc.float(), dim=-1)


@pytest.fixture
def scene() -> dict:
    return _scene()


@pytest.fixture(autouse=True)
def fake_salad(monkeypatch):
    monkeypatch.setattr(tracks_mod, "DinoSaladExtractor", _FakeSalad)


########################################
# Helpers
########################################


def test_to_model_grid_size_map():
    feats = LocalFeatures(
        keypoints=torch.tensor([[0.0, 0.0], [159.0, 119.0]]), descriptors=torch.zeros(2, 1), image_size=(160, 120)
    )

    out = tracks_mod._to_model_grid(feats, (60, 80))

    np.testing.assert_allclose(out.keypoints.numpy(), [[-0.25, -0.25], [79.25, 59.25]])


@pytest.mark.parametrize("full_wh", [(1920, 1080), (2704, 1520)])
def test_to_model_grid_accepts_multiple_of_14_stretch(full_wh):
    w, h = full_wh
    feats = LocalFeatures(
        keypoints=torch.tensor([[0.0, 0.0], [w - 1.0, h - 1.0]]), descriptors=torch.zeros(2, 1), image_size=(w, h)
    )

    # VGGT-X / MapAnything resize: width 518, height rounded to a multiple of 14 (294 for both)
    out = tracks_mod._to_model_grid(feats, (294, 518))

    expected = [[0.5 * 518 / w - 0.5, 0.5 * 294 / h - 0.5], [518 - 0.5 * 518 / w - 0.5, 294 - 0.5 * 294 / h - 0.5]]
    np.testing.assert_allclose(out.keypoints.numpy(), expected, rtol=1e-6)
    assert out.image_size == (518, 294)


def test_to_model_grid_refuses_cropped_aspect():
    feats = LocalFeatures(keypoints=torch.zeros(1, 2), descriptors=torch.zeros(1, 1), image_size=(1920, 1080))

    with pytest.raises(ValueError, match="aspect"):
        tracks_mod._to_model_grid(feats, (518, 518))


def test_extract_full_res_refuses_mixed_dirs(tmp_path, scene):
    paths = _write_store(tmp_path, scene)
    stray = tmp_path / "other" / paths[0].name

    with pytest.raises(ValueError, match="one directory"):
        tracks_mod._extract_full_res(_FakeMatcher(scene), [stray, *paths[1:]], batch_size=4)


def test_extract_full_res_chunks_preserve_order(tmp_path, scene):
    paths = _write_store(tmp_path, scene)

    feats = tracks_mod._extract_full_res(_FakeMatcher(scene), paths, batch_size=5)

    assert len(feats) == len(paths)

    for i, f in enumerate(feats):
        np.testing.assert_array_equal(f.keypoints.numpy(), scene["kp_full"][i])


def test_pairs_sequential_plus_retrieval(scene):
    pairs = tracks_mod._pairs(_model_images(scene), window=3, retrieval_k=2, retrieval_nms=4, batch_size=5)

    seq = [(a, b) for a in range(12) for b in range(a + 1, min(12, a + 4))]
    assert pairs[: len(seq)] == seq
    extra = pairs[len(seq) :]
    assert extra
    assert extra == sorted(extra) and all(b - a > 4 for a, b in extra)
    assert len(set(pairs)) == len(pairs)


def test_pairs_skip_suppressed_candidates_when_few_frames(scene):
    images = _model_images(scene)[:6]

    pairs = tracks_mod._pairs(images, window=1, retrieval_k=2, retrieval_nms=4, batch_size=5)

    # Only frames 0 and 5 lie outside each other's nms band; the prototype would add (a, a) self pairs
    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]


def _wide_pair(scene: dict) -> tuple:
    """
    Frames 0 and 6 (0.6 world-unit baseline): model-grid keypoints, lifts, and shared-point rows.
    """
    kps = [(kp + 0.5) / 2 - 0.5 for kp in scene["kp_full"]]
    lifted = tracks_mod._lift_keypoints(scene["world_points"], kps)
    _, ia, ib = np.intersect1d(scene["kp_ids"][0], scene["kp_ids"][6], return_indices=True)
    return kps, lifted, ia, ib


def test_depth_filter_drops_planted_outlier(scene):
    kps, lifted, ia, ib = _wide_pair(scene)

    # Swap one match's reference keypoint for a far one
    ib_bad = ib.copy()
    ib_bad[0] = ib[-1]

    ok = tracks_mod._depth_filter(
        lifted[0][ia],
        lifted[6][ib_bad],
        kps[0][ia],
        kps[6][ib_bad],
        scene["extrinsics"][[0, 6]],
        scene["K"][[0, 6]],
        depth_tol=0.5,
    )

    assert not ok[0] and ok[1:].all()


def test_depth_filter_rejects_one_direction_failure(scene):
    kps, lifted, ia, ib = _wide_pair(scene)
    pts_a = lifted[0][ia]
    pts_b = lifted[6][ib]

    # Push one frame-0 world point along x: only the a -> b transfer breaks
    pts_a[0, 0] += 0.5
    err_ab = tracks_mod._transfer_err(pts_a, kps[6][ib], scene["extrinsics"][6], scene["K"][6])
    err_ba = tracks_mod._transfer_err(pts_b, kps[0][ia], scene["extrinsics"][0], scene["K"][0])

    ok = tracks_mod._depth_filter(
        pts_a, pts_b, kps[0][ia], kps[6][ib], scene["extrinsics"][[0, 6]], scene["K"][[0, 6]], depth_tol=0.5
    )

    assert err_ab[0] > 0.5 and err_ba[0] < 0.5
    assert not ok[0] and ok[1:].all()


class _StarGraph:
    """
    Fake correspondence graph: keypoint k of frame i matches keypoint k of frame i + 1 (cyclic).
    """

    def __init__(self, n_frames: int) -> None:
        self._n_frames = n_frames

    def exists_image(self, image_id: int) -> bool:
        return True

    def extract_transitive_correspondences(self, image_id: int, idx: int, depth: int) -> list:
        other = image_id % self._n_frames + 1
        return [types.SimpleNamespace(image_id=other, point2D_idx=idx)]


def test_star_tracks_dedupes_seeds_when_more_seeds_than_frames():
    kps = [np.zeros((2, 2), np.float32) for _ in range(3)]

    tracks = tracks_mod._star_tracks(_StarGraph(3), kps, seed_frames=10)

    assert len(tracks) == 6
    assert len({tuple(t) for t in tracks}) == len(tracks)


def test_pairs_refuses_negative_nms(scene):
    with pytest.raises(ValueError, match="retrieval_nms"):
        tracks_mod._pairs(_model_images(scene), window=1, retrieval_k=2, retrieval_nms=-1, batch_size=5)


def _refuse_salad(*args, **kwargs) -> None:
    """
    Stand-in DINO-SALAD constructor that fails the test if called.
    """
    raise AssertionError("DINO-SALAD constructed with retrieval_k=0")


def test_pairs_without_retrieval_never_loads_salad(scene, monkeypatch):
    monkeypatch.setattr(tracks_mod, "DinoSaladExtractor", _refuse_salad)

    pairs = tracks_mod._pairs(_model_images(scene)[:4], window=1, retrieval_k=0, retrieval_nms=4, batch_size=5)

    assert pairs == [(0, 1), (1, 2), (2, 3)]


def test_extract_full_res_refuses_empty(scene):
    with pytest.raises(ValueError, match="empty"):
        tracks_mod._extract_full_res(_FakeMatcher(scene), [], batch_size=4)


########################################
# End to end
########################################


# Small-scene build_tracks keywords, shared by the end-to-end tests and the prototype arm
SMALL_SCENE = {
    "window": 3,
    "retrieval_k": 2,
    "retrieval_nms": 4,
    "seed_frames": 4,
    "min_matches": 15,
    "depth_tol": 2.0,
    "batch_size": 5,
}


def _build(tmp_path: Path, scene: dict, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build_tracks over the synthetic scene with small-scene keyword values.
    """
    paths = _write_store(tmp_path, scene)
    options = {**SMALL_SCENE, **kwargs}
    images = _model_images(scene)
    matcher = _FakeMatcher(scene)
    return tracks_mod.build_tracks(
        matcher, images, paths, scene["world_points"], scene["extrinsics"], scene["K"], **options
    )


def test_build_tracks_recovers_star_tracks(tmp_path, scene):
    tracks, vis, pts3d = _build(tmp_path, scene)

    assert tracks.dtype == vis.dtype == pts3d.dtype == np.float32
    assert tracks.shape[1] == vis.shape[1] == len(pts3d) > 50
    assert (vis.sum(axis=0) >= 2).all()

    # Every observation reprojects onto its keypoint under the true poses
    for i in range(len(vis)):
        seen = vis[i] > 0
        cam = pts3d[seen] @ scene["extrinsics"][i, :3, :3].T + scene["extrinsics"][i, :3, 3]
        px = cam @ scene["K"][i].T
        px = px[:, :2] / px[:, 2:]
        np.testing.assert_allclose(px, tracks[i, seen], atol=0.5)


def test_build_tracks_pts3d_from_world_points(tmp_path, scene):
    tracks, vis, pts3d = _build(tmp_path, scene)
    first = vis.argmax(axis=0)

    for j in range(0, len(pts3d), 25):
        i = first[j]
        expect, ok = tracks_mod.sample_world_points(scene["world_points"][i], tracks[i, j][None])
        assert ok[0]
        np.testing.assert_allclose(pts3d[j], expect[0], atol=1e-5)


########################################
# Prototype parity
########################################


def _load_prototype() -> types.ModuleType:
    """
    The vendored star-chain prototype as a module.
    """
    source = str(PROTOTYPE)
    loader = importlib.machinery.SourceFileLoader("match_tracks_prototype", source)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    return module


def test_build_tracks_matches_prototype(tmp_path, scene, monkeypatch):
    proto = _load_prototype()
    fake = _FakeMatcher(scene)
    paths = _write_store(tmp_path, scene)
    full_dir = str(paths[0].parent)
    model_images = _model_images(scene)
    images = torch.from_numpy(model_images)

    # Prototype arm of the gh1k runs: MT_CHAIN=star, MT_FULL_DIR set, depth filter on
    monkeypatch.setattr(proto, "LocalMatcher", lambda name, probe=False: fake)
    monkeypatch.setattr(proto, "DinoSaladExtractor", _FakeSalad)
    monkeypatch.setattr(proto, "CHAIN", "star")
    monkeypatch.setattr(proto, "QUERY_FRAMES", SMALL_SCENE["seed_frames"])
    monkeypatch.setattr(proto, "RETR_K", SMALL_SCENE["retrieval_k"])
    monkeypatch.setattr(proto, "RETR_NMS", SMALL_SCENE["retrieval_nms"])
    monkeypatch.setattr(proto, "MIN_MATCHES", SMALL_SCENE["min_matches"])
    monkeypatch.setattr(proto, "DEPTH_TOL", SMALL_SCENE["depth_tol"])
    monkeypatch.setattr(proto, "FULL_DIR", full_dir)

    # Env-read knobs pinned so a stray MT_* export cannot skew parity
    monkeypatch.setattr(proto, "MAX_KP", fake._matcher.max_num_keypoints)
    monkeypatch.setattr(proto, "MIN_ANGLE", 0.1)

    proto.set_poses(scene["extrinsics"], scene["K"])
    proto.set_depth(scene["depth"])
    extract = proto.make_extract("xfeat", SMALL_SCENE["window"], "wp")
    p_tracks, p_vis, p_pts3d = extract(images, None, scene["world_points"], None)

    tracks, vis, pts3d = _build(tmp_path / "ours", scene)

    # Same tracks, same observations; pts3d differ only by bilinear vs nearest lookup (observed max ~0.04)
    assert tracks.shape == p_tracks.shape
    np.testing.assert_array_equal(vis, p_vis)
    np.testing.assert_allclose(tracks, p_tracks, atol=1e-5)
    np.testing.assert_allclose(pts3d, p_pts3d, atol=0.05)

    # Same pair set as the prototype
    pairs = tracks_mod._pairs(
        model_images,
        window=SMALL_SCENE["window"],
        retrieval_k=SMALL_SCENE["retrieval_k"],
        retrieval_nms=SMALL_SCENE["retrieval_nms"],
        batch_size=SMALL_SCENE["batch_size"],
    )
    assert proto._state["stats"][-1]["pairs"] == len(pairs)
