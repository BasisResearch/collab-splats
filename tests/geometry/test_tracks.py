"""
Matcher tracks: synthetic posed scene, helper units, star-chain parity with the prototype.
"""

import importlib.machinery
import inspect
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from collab_splats.geometry import tracks as tracks_mod
from collab_splats.geometry.projection import reprojection_error, sample_world_points
from collab_splats.localization.extractors import LocalFeatures, MatchResult
from collab_splats.localization.retrieval import BaseRetrievalExtractor
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
        center = np.array([0.1 * i, 0.0, 0.0])
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = -R @ center

    # Dense world points: per-pixel ray / surface intersection by fixed-point iteration
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    rays_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1)
    world_points = np.zeros((n_frames, height, width, 3))
    depth = np.zeros((n_frames, height, width))

    for i in range(n_frames):
        R = extrinsics[i, :3, :3]
        center = -R.T @ extrinsics[i, :3, 3]
        rays = rays_cam @ R
        s = np.full((height, width), 5.0)

        for _ in range(50):
            pts = center + rays * s[..., None]
            s = (_surface(pts[..., 0], pts[..., 1]) - center[2]) / rays[..., 2]

        world_points[i] = center + rays * s[..., None]
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
    Full-res frame store: frame i is a flat image whose channel 0 holds i, so any resize keeps it.
    """
    H, W = scene["full_hw"]
    n = len(scene["extrinsics"])
    frames = np.full((n, H, W, 3), 200, np.uint8)
    frames[..., 0] = np.arange(n)[:, None, None]
    return write_frames(tmp_path / "images", frames, list(range(n)))


def _model_images(scene: dict) -> np.ndarray:
    """
    (N, 3, H, W) model-grid images in [0, 1]; channel 0 encodes the frame.
    """
    H, W = scene["hw"]
    n = len(scene["extrinsics"])
    images = np.full((n, 3, H, W), 200 / 255, np.float32)
    images[:, 0] = np.arange(n)[:, None, None] / 255
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

    def match_batch(self, pairs: list[tuple[LocalFeatures, LocalFeatures]]) -> list[MatchResult]:
        return [self.match(q, db) for q, db in pairs]


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
def retrieval_names(monkeypatch) -> list:
    """
    Stand-in retrieval registry: records every looked-up name and hands back _FakeSalad.
    """
    names = []

    def get(name: str) -> type:
        names.append(name)
        return _FakeSalad

    monkeypatch.setattr(tracks_mod, "BaseRetrievalExtractor", types.SimpleNamespace(get=get))
    return names


def _dense(out: tuple, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flat tracks back on the dense (N, P) grid: tracks, vis and pts3d.
    """
    frame, track, xy, score, pts3d = out
    tracks = np.zeros((n_frames, len(pts3d), 2), np.float32)
    vis = np.zeros((n_frames, len(pts3d)), np.float32)
    tracks[frame, track] = xy
    vis[frame, track] = score
    return tracks, vis, pts3d


class _RecordingMatcher(_FakeMatcher):
    """
    Fake matcher that records model-grid features in frame order and every matched pair.

    - features: one LocalFeatures returned for every frame instead of the scene's keypoints
    - corrupt: pair whose first match gets the last match's reference keypoint
    """

    def __init__(self, scene: dict, features: LocalFeatures | None = None, corrupt: tuple | None = None) -> None:
        super().__init__(scene)
        self._features = features
        self._corrupt = corrupt
        self.feats = []
        self.pairs = []

    def extract(self, images):
        if self._features is None:
            return super().extract(images)
        return [self._features for _ in images]

    def to_device(self, features: LocalFeatures) -> LocalFeatures:
        self.feats.append(features)
        return features

    def match(self, q: LocalFeatures, db: LocalFeatures) -> MatchResult:
        pair = (self._index(q), self._index(db))
        self.pairs.append(pair)
        m = super().match(q, db)

        if pair == self._corrupt:
            m.idx_db[0] = m.idx_db[-1]

        return m

    def _index(self, features: LocalFeatures) -> int:
        return next(i for i, f in enumerate(self.feats) if f is features)


def _run(tmp_path: Path, scene: dict, matcher: _FakeMatcher, **kwargs) -> tuple:
    """
    build_tracks over a scene's frame store with the given matcher; flat output.

    - sequential+retrieval unless the test names a pairing, so window / retrieval_k take effect
    """
    paths = _write_store(tmp_path, scene)
    kwargs.setdefault("pairing", "sequential+retrieval")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tracks_mod, "LocalMatcher", lambda source: matcher)
        return tracks_mod.build_tracks("xfeat", paths, scene["world_points"], scene["extrinsics"], scene["K"], **kwargs)


def _grid_run(tmp_path: Path, features: LocalFeatures, model_hw: tuple[int, int]) -> list[LocalFeatures]:
    """
    build_tracks on two frames whose features are all `features`; the model-grid features it produced.
    """
    scene = _scene(n_frames=2)
    H, W = model_hw
    scene["world_points"] = np.zeros((2, H, W, 3), np.float32)
    matcher = _RecordingMatcher(scene, features=features)
    _run(tmp_path, scene, matcher, retrieval_k=0)
    return matcher.feats


def _verified_matches(monkeypatch) -> MagicMock:
    """
    Spy on two-view verification; each call's args carry both cameras and the match index table.
    """
    spy = MagicMock(wraps=tracks_mod.pycolmap.estimate_two_view_geometry)
    monkeypatch.setattr(tracks_mod.pycolmap, "estimate_two_view_geometry", spy)
    return spy


def _pair_matches(spy: MagicMock, a: int, b: int) -> np.ndarray:
    """
    Match index table the spy saw for frame pair (a, b).
    """
    return next(c.args[4] for c in spy.call_args_list if (c.args[0].camera_id, c.args[2].camera_id) == (a + 1, b + 1))


########################################
# Helpers
########################################


def test_to_model_grid_size_map(tmp_path):
    feats = LocalFeatures(
        keypoints=torch.tensor([[0.0, 0.0], [159.0, 119.0]]), descriptors=torch.zeros(2, 1), image_size=(160, 120)
    )

    out = _grid_run(tmp_path, feats, (60, 80))

    np.testing.assert_allclose(out[0].keypoints.numpy(), [[-0.25, -0.25], [79.25, 59.25]])


@pytest.mark.parametrize("full_wh", [(1920, 1080), (2704, 1520)])
def test_to_model_grid_accepts_multiple_of_14_stretch(tmp_path, full_wh):
    w, h = full_wh
    feats = LocalFeatures(
        keypoints=torch.tensor([[0.0, 0.0], [w - 1.0, h - 1.0]]), descriptors=torch.zeros(2, 1), image_size=(w, h)
    )

    # VGGT-X / MapAnything resize: width 518, height rounded to a multiple of 14 (294 for both)
    out = _grid_run(tmp_path, feats, (294, 518))

    expected = [[0.5 * 518 / w - 0.5, 0.5 * 294 / h - 0.5], [518 - 0.5 * 518 / w - 0.5, 294 - 0.5 * 294 / h - 0.5]]
    np.testing.assert_allclose(out[0].keypoints.numpy(), expected, rtol=1e-6)
    assert out[0].image_size == (518, 294)


def test_to_model_grid_refuses_cropped_aspect(tmp_path):
    feats = LocalFeatures(keypoints=torch.zeros(1, 2), descriptors=torch.zeros(1, 1), image_size=(1920, 1080))

    with pytest.raises(ValueError, match="aspect"):
        _grid_run(tmp_path, feats, (518, 518))


def test_extract_reads_frames_by_path(tmp_path, scene, monkeypatch):
    paths = _write_store(tmp_path, scene)
    stray = tmp_path / "other" / paths[0].name
    stray.parent.mkdir()
    paths[0].rename(stray)
    matcher = _RecordingMatcher(scene)
    monkeypatch.setattr(tracks_mod, "LocalMatcher", lambda source: matcher)

    # Frame 0 sits in another directory; it is still read for frame 0
    tracks_mod.build_tracks(
        "xfeat", [stray, *paths[1:]], scene["world_points"], scene["extrinsics"], scene["K"], retrieval_k=0
    )

    np.testing.assert_allclose(matcher.feats[0].keypoints.numpy(), (scene["kp_full"][0] + 0.5) / 2 - 0.5, rtol=1e-6)


def test_extract_preserves_frame_order(tmp_path, scene):
    matcher = _RecordingMatcher(scene)

    _run(tmp_path, scene, matcher, retrieval_k=0)

    assert len(matcher.feats) == 12

    for i, f in enumerate(matcher.feats):
        np.testing.assert_allclose(f.keypoints.numpy(), (scene["kp_full"][i] + 0.5) / 2 - 0.5, rtol=1e-6)


def test_pairs_sequential_plus_retrieval(tmp_path, scene):
    matcher = _RecordingMatcher(scene)

    _run(tmp_path, scene, matcher, window=3, retrieval_k=2, retrieval_nms=4)

    pairs = matcher.pairs
    seq = [(a, b) for a in range(12) for b in range(a + 1, min(12, a + 4))]
    assert pairs[: len(seq)] == seq
    extra = pairs[len(seq) :]
    assert extra
    assert extra == sorted(extra) and all(b - a > 4 for a, b in extra)
    assert len(set(pairs)) == len(pairs)


def test_pairs_exhaustive_is_every_pair_without_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(tracks_mod, "BaseRetrievalExtractor", types.SimpleNamespace(get=lambda name: _refuse_salad))
    scene = _scene(n_frames=6)
    matcher = _RecordingMatcher(scene)

    _run(tmp_path, scene, matcher, pairing="exhaustive", window=1, retrieval_k=2)

    assert matcher.pairs == [(a, b) for a in range(6) for b in range(a + 1, 6)]


def test_pairs_default_is_exhaustive():
    assert inspect.signature(tracks_mod.build_tracks).parameters["pairing"].default == "exhaustive"


def test_pairs_refuses_unknown_pairing(tmp_path, scene):
    with pytest.raises(ValueError, match="pairing"):
        _run(tmp_path, scene, _FakeMatcher(scene), pairing="sequential")


def test_pairs_skip_suppressed_candidates_when_few_frames(tmp_path):
    scene = _scene(n_frames=6)
    matcher = _RecordingMatcher(scene)

    _run(tmp_path, scene, matcher, window=1, retrieval_k=2, retrieval_nms=4)

    # Only frames 0 and 5 lie outside each other's nms band; the prototype would add (a, a) self pairs
    assert matcher.pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]


def test_pairs_refuses_negative_nms(tmp_path, scene):
    with pytest.raises(ValueError, match="retrieval_nms"):
        _run(tmp_path, scene, _FakeMatcher(scene), window=1, retrieval_k=2, retrieval_nms=-1)


def _refuse_salad(*args, **kwargs) -> None:
    """
    Stand-in DINO-SALAD constructor that fails the test if called.
    """
    raise AssertionError("DINO-SALAD constructed with retrieval_k=0")


def test_pairs_without_retrieval_never_loads_salad(tmp_path, monkeypatch):
    monkeypatch.setattr(tracks_mod, "BaseRetrievalExtractor", types.SimpleNamespace(get=lambda name: _refuse_salad))
    scene = _scene(n_frames=4)
    matcher = _RecordingMatcher(scene)

    _run(tmp_path, scene, matcher, window=1, retrieval_k=0, retrieval_nms=4)

    assert matcher.pairs == [(0, 1), (1, 2), (2, 3)]


def test_retrieval_name_reaches_the_registry(tmp_path, retrieval_names):
    scene = _scene(n_frames=6)

    _run(tmp_path, scene, _RecordingMatcher(scene), window=1, retrieval="megaloc", retrieval_k=2, retrieval_nms=4)

    assert retrieval_names == ["megaloc"]


def test_unknown_retrieval_name_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(tracks_mod, "BaseRetrievalExtractor", BaseRetrievalExtractor)
    scene = _scene(n_frames=4)

    # Resolved before any model load, so retrieval_k=0 still refuses it
    with pytest.raises(ValueError, match="no-such-retrieval"):
        _run(tmp_path, scene, _RecordingMatcher(scene), retrieval="no-such-retrieval", retrieval_k=0)


def test_depth_filter_drops_planted_outlier(tmp_path, scene, monkeypatch):
    seen = _verified_matches(monkeypatch)
    _, ia, ib = np.intersect1d(scene["kp_ids"][0], scene["kp_ids"][6], return_indices=True)

    # Frame pair (0, 6) gets its first match's reference keypoint swapped for a far one
    matcher = _RecordingMatcher(scene, corrupt=(0, 6))
    _run(tmp_path, scene, matcher, window=6, retrieval_k=0, depth_tol=0.5)

    expected = np.stack([ia[1:], ib[1:]], axis=1)
    np.testing.assert_array_equal(_pair_matches(seen, 0, 6), expected)


def test_depth_filter_rejects_one_direction_failure(tmp_path, scene, monkeypatch):
    seen = _verified_matches(monkeypatch)
    _, ia, ib = np.intersect1d(scene["kp_ids"][0], scene["kp_ids"][6], return_indices=True)
    kp = (scene["kp_full"][0][ia[0]] + 0.5) / 2 - 0.5

    # Push frame 0's world points around one keypoint along x: only the 0 -> 6 transfer breaks
    kps_a = (scene["kp_full"][0][ia] + 0.5) / 2 - 0.5
    before, _ = sample_world_points(scene["world_points"][0], kps_a)
    x, y = int(kp[0]), int(kp[1])
    scene["world_points"][0, y : y + 2, x : x + 2, 0] += 0.5
    after, _ = sample_world_points(scene["world_points"][0], kps_a)
    _run(tmp_path, scene, _FakeMatcher(scene), window=6, retrieval_k=0, depth_tol=0.5)

    # The 6 -> 0 transfer of the same match still passes
    kp_b = (scene["kp_full"][6][ib[:1]] + 0.5) / 2 - 0.5
    pts_b, _ = sample_world_points(scene["world_points"][6], kp_b)
    pts_b = torch.as_tensor(pts_b, dtype=torch.float64)
    w2c = torch.as_tensor(scene["extrinsics"][0], dtype=torch.float64)
    K = torch.as_tensor(scene["K"][0], dtype=torch.float64)
    px = torch.as_tensor(kp[None], dtype=torch.float64)
    err_ba = reprojection_error(pts_b, w2c, K, px)
    assert err_ba[0] < 0.5

    # The pushed match is dropped; every other drop shares the pushed cells
    kept = _pair_matches(seen, 0, 6)
    dropped = ~np.isin(ia, kept[:, 0])
    moved = (before != after).any(axis=1)
    assert dropped[0]
    assert not (dropped & ~moved).any()


class _StarGraph:
    """
    Fake correspondence graph: keypoint k of frame i matches keypoint k of frame i + 1 (cyclic).
    """

    def __init__(self, n_frames: int) -> None:
        self._n_frames = n_frames

    def exists_image(self, image_id: int) -> bool:
        return True

    def extract_correspondences(self, image_id: int, idx: int) -> list:
        other = image_id % self._n_frames + 1
        return [types.SimpleNamespace(image_id=other, point2D_idx=idx)]


def test_assemble_tracks_dedupes_seeds_when_more_seeds_than_frames():
    kps = [np.zeros((2, 2), np.float32) for _ in range(3)]
    kp_world = [np.zeros((2, 3), np.float32) for _ in range(3)]

    frame, track, _, _, pts3d = tracks_mod._assemble_tracks(_StarGraph(3), kps, kp_world, seed_frames=10)

    assert len(pts3d) == 6
    members = {tuple(frame[track == j]) for j in range(6)}
    assert len(members) == 3


class _AmbiguousGraph:
    """
    Fake correspondence graph: keypoint 0 of frame 0 matches two keypoints in frame 1 and one, listed twice, in frame 2.
    """

    def exists_image(self, image_id: int) -> bool:
        return True

    def extract_correspondences(self, image_id: int, idx: int) -> list:
        if (image_id, idx) != (1, 0):
            return []

        members = [(2, 0), (2, 1), (3, 0), (3, 0)]
        return [types.SimpleNamespace(image_id=i, point2D_idx=k) for i, k in members]


def test_assemble_tracks_drops_ambiguous_frame_and_keeps_duplicated_one():
    kps = [np.zeros((2, 2), np.float32) for _ in range(3)]
    kp_world = [np.ones((2, 3), np.float32) for _ in range(3)]

    frame, track, _, _, pts3d = tracks_mod._assemble_tracks(_AmbiguousGraph(), kps, kp_world, seed_frames=2)

    # Frame 1 holds two keypoints and leaves the track; frame 2's repeated entry is one keypoint
    assert len(pts3d) == 1
    assert frame.tolist() == [0, 2]
    assert track.tolist() == [0, 0]


def test_assemble_tracks_flat_rows_match_dense_grid():
    kps = [np.arange(6, dtype=np.float32).reshape(3, 2) + 10 * i for i in range(4)]
    kp_world = [np.arange(9, dtype=np.float32).reshape(3, 3) + 100 * i for i in range(4)]

    # Seed frame 0 keypoint 1 has no world point: its track is dropped
    kp_world[0][1] = np.nan
    out = tracks_mod._assemble_tracks(_StarGraph(4), kps, kp_world, seed_frames=2)
    frame, track, xy, score, pts3d = out
    tracks, vis, _ = _dense(out, 4)

    # Rows frame-major, then by track; every score 1
    order = np.lexsort((track, frame))
    np.testing.assert_array_equal(order, np.arange(len(frame)))
    assert (score == 1).all() and frame.dtype == track.dtype == np.int32

    # Seeds 0 and 3 both start at frame 0, so keypoint 1 of either has no world point
    assert len(pts3d) == 4
    expected_frames = [(0, 1), (0, 1), (0, 3), (0, 3)]
    assert [tuple(np.flatnonzero(vis[:, j])) for j in range(4)] == expected_frames
    kp_of_track = [0, 2, 0, 2]

    # Observed cells hold the matched keypoint; pts3d is the world point at the first (lowest) frame
    for j, k in enumerate(kp_of_track):
        for i in np.flatnonzero(vis[:, j]):
            np.testing.assert_array_equal(tracks[i, j], kps[i][k])

        np.testing.assert_array_equal(pts3d[j], kp_world[0][k])


########################################
# End to end
########################################


# Small-scene build_tracks keywords, shared by the end-to-end tests and the prototype arm
SMALL_SCENE = {
    "window": 3,
    "retrieval_k": 2,
    "retrieval_nms": 4,
    "seed_fraction": 4 / 12,
    "min_matches": 15,
    "depth_tol": 2.0,
}


def _build(tmp_path: Path, scene: dict, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    build_tracks over the synthetic scene with small-scene keyword values, on the dense grid.
    """
    options = {**SMALL_SCENE, **kwargs}
    out = _run(tmp_path, scene, _FakeMatcher(scene), **options)
    return _dense(out, len(scene["extrinsics"]))


def test_build_tracks_flat_format(tmp_path, scene):
    frame, track, xy, score, pts3d = _run(tmp_path, scene, _FakeMatcher(scene), **SMALL_SCENE)

    assert frame.dtype == track.dtype == np.int32
    assert xy.dtype == score.dtype == pts3d.dtype == np.float32
    assert xy.shape == (len(frame), 2) and score.shape == (len(frame),)
    np.testing.assert_array_equal(np.lexsort((track, frame)), np.arange(len(frame)))
    np.testing.assert_array_equal(np.unique(track), np.arange(len(pts3d)))


def test_build_tracks_seeds_at_least_two_frames(tmp_path, scene, monkeypatch):
    seen = []
    real = tracks_mod._assemble_tracks
    monkeypatch.setattr(tracks_mod, "_assemble_tracks", lambda g, k, l, n: seen.append(n) or real(g, k, l, n))

    _build(tmp_path, scene, seed_fraction=0.01)

    assert seen == [2]


def test_build_tracks_repeat_runs_match(tmp_path, scene):
    # Thread scheduling differs run to run; the seeded verify must not
    first = _run(tmp_path / "first", scene, _FakeMatcher(scene), **SMALL_SCENE)
    second = _run(tmp_path / "second", scene, _FakeMatcher(scene), **SMALL_SCENE)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


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
    monkeypatch.setattr(proto, "QUERY_FRAMES", round(SMALL_SCENE["seed_fraction"] * 12))
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

    matcher = _RecordingMatcher(scene)
    out = _run(tmp_path / "ours", scene, matcher, **SMALL_SCENE)
    tracks, vis, pts3d = _dense(out, 12)

    # Same tracks, same observations; pts3d differ only by bilinear vs nearest lookup (observed max ~0.04)
    assert tracks.shape == p_tracks.shape
    np.testing.assert_array_equal(vis, p_vis)
    np.testing.assert_allclose(tracks, p_tracks, atol=1e-5)
    np.testing.assert_allclose(pts3d, p_pts3d, atol=0.05)

    # Same pair count as the prototype
    assert proto._state["stats"][-1]["pairs"] == len(matcher.pairs)


########################################
# VGGSfM source
########################################


def _fake_predict(calls: list, n_frames: int = 2, n_points: int = 4):
    """
    predict_tracks stand-in that records (images, kwargs) and returns float64 arrays.
    """

    def predict(images, **kwargs):
        calls.append((images, kwargs))
        return (
            np.random.rand(n_frames, n_points, 2),
            np.random.rand(n_frames, n_points),
            np.ones((n_frames, n_points)),
            np.random.rand(n_points, 3),
            np.ones((n_points, 3)),
        )

    return predict


def test_vggsfm_conversion_is_frame_major_and_drops_zero_scores(monkeypatch):
    tracks = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    vis = np.array([[0.9, 0.0, 0.5], [0.0, 0.7, 0.3]], np.float32)
    pts3d = np.ones((3, 3), np.float32)
    monkeypatch.setattr(tracks_mod, "predict_tracks", lambda *a, **k: (tracks, vis, None, pts3d, None))

    conf = np.ones((2, 4, 4), np.float32)
    wp = np.zeros((2, 4, 4, 3), np.float32)
    frame, track, xy, score, p3 = tracks_mod._vggsfm_tracks(
        np.zeros((2, 3, 4, 4), np.float32), conf, wp, max_query_pts=4096, query_frame_num=8, fine_tracking=False
    )

    assert frame.tolist() == [0, 0, 1, 1]
    assert track.tolist() == [0, 2, 1, 2]
    assert np.array_equal(xy, tracks[frame, track])
    assert np.allclose(score, [0.9, 0.5, 0.7, 0.3])
    assert p3.shape == (3, 3)


def test_vggsfm_tracks_shapes_and_dtype(monkeypatch):
    calls = []
    monkeypatch.setattr(tracks_mod, "predict_tracks", _fake_predict(calls, n_frames=6, n_points=50))

    wp = np.zeros((6, 64, 64, 3), np.float32)
    frame, track, xy, score, pts3d = tracks_mod._vggsfm_tracks(
        torch.zeros(6, 3, 64, 64), torch.ones(6, 64, 64), wp, max_query_pts=512, query_frame_num=2, fine_tracking=False
    )

    assert frame.dtype == track.dtype == np.int32
    assert (xy.shape, score.shape, pts3d.shape) == ((len(frame), 2), (len(frame),), (50, 3))
    assert {a.dtype for a in (xy, score, pts3d)} == {np.dtype(np.float32)}
    assert calls[0][1]["max_query_pts"] == 512
    assert calls[0][1]["query_frame_num"] == 2


def test_vggsfm_tracks_moves_images_to_get_device(monkeypatch):
    calls = []
    monkeypatch.setattr(tracks_mod, "predict_tracks", _fake_predict(calls))
    devices = []
    monkeypatch.setattr(tracks_mod, "get_device", lambda: devices.append("cpu") or "cpu")

    tracks_mod._vggsfm_tracks(
        np.zeros((2, 3, 8, 8), np.float32),
        np.ones((2, 8, 8)),
        np.zeros((2, 8, 8, 3)),
        max_query_pts=4096,
        query_frame_num=8,
        fine_tracking=False,
    )

    assert devices == ["cpu"]
    assert calls[0][0].device.type == "cpu"
    assert calls[0][0].dtype == torch.float32


def test_vggsfm_tracks_pads_non_square_to_square(monkeypatch):
    calls = []
    monkeypatch.setattr(tracks_mod, "predict_tracks", _fake_predict(calls))

    tracks_mod._vggsfm_tracks(
        torch.ones(2, 3, 6, 8),
        torch.ones(2, 6, 8),
        np.ones((2, 6, 8, 3), np.float32),
        max_query_pts=4096,
        query_frame_num=8,
        fine_tracking=False,
    )

    images, kwargs = calls[0]
    assert images.shape == (2, 3, 8, 8) and kwargs["conf"].shape == (2, 8, 8)
    assert kwargs["points_3d"].shape == (2, 8, 8, 3)
    assert (images[:, :, 6:] == 0).all() and (images[:, :, :6] == 1).all()


########################################
# Cache and dispatch
########################################


def _counted_vggsfm(monkeypatch, outputs: list | None = None) -> list:
    """
    Replace _vggsfm_tracks with a counter returning outputs in turn (a 3-frame flat set by default).
    """
    calls = []

    def fake(images, conf, world_points, **kwargs):
        calls.append(kwargs)

        if outputs:
            return outputs.pop(0)

        frame = np.repeat(np.arange(3, dtype=np.int32), 5)
        track = np.tile(np.arange(5, dtype=np.int32), 3)
        return frame, track, np.ones((15, 2), np.float32), np.ones(15, np.float32), np.ones((5, 3), np.float32)

    monkeypatch.setattr(tracks_mod, "_vggsfm_tracks", fake)
    return calls


def _cached(tmp_path: Path, **kwargs) -> tuple:
    """
    extract_tracks on a 3-frame vggsfm input, cached under tmp_path.
    """
    wp = np.zeros((3, 8, 8, 3), np.float32)
    paths = [f"frame_{i:06d}.png" for i in range(3)]
    return tracks_mod.extract_tracks(
        torch.zeros(3, 3, 8, 8),
        torch.ones(3, 8, 8),
        wp,
        frame_paths=paths,
        source="vggsfm",
        cache_dir=tmp_path,
        **kwargs,
    )


def test_cache_saves_then_hits(tmp_path, monkeypatch):
    calls = _counted_vggsfm(monkeypatch)

    first = _cached(tmp_path)
    second = _cached(tmp_path)

    assert len(calls) == 1

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("knob", [{"max_query_pts": 7}, {"query_frame_num": 5}, {"fine_tracking": True}])
def test_cache_knob_change_re_extracts(tmp_path, monkeypatch, knob):
    calls = _counted_vggsfm(monkeypatch)

    _cached(tmp_path)
    _cached(tmp_path, **knob)

    assert len(calls) == 2


def test_cache_key_ignores_path_order_but_covers_world_points(tmp_path, monkeypatch):
    calls = _counted_vggsfm(monkeypatch)
    wp = np.zeros((3, 8, 8, 3), np.float32)
    paths = [f"frame_{i:06d}.png" for i in range(3)]
    wp2 = wp.copy()
    wp2[0, 0, 0, 0] = 1.0

    # Same paths reordered hit; changed world points miss
    for points, order in [(wp, paths), (wp.copy(), paths[::-1]), (wp2, paths)]:
        tracks_mod.extract_tracks(
            torch.zeros(3, 3, 8, 8), None, points, frame_paths=order, source="vggsfm", cache_dir=tmp_path
        )

    assert len(calls) == 2


def test_keyless_cache_is_re_extracted(tmp_path, monkeypatch):
    calls = _counted_vggsfm(monkeypatch)

    # A write that crashed before the key stamp leaves a keyless store
    tracks_mod.zarr.open(str(tmp_path / "tracks.zarr"), mode="w")

    _cached(tmp_path)

    assert len(calls) == 1


def test_unexpected_cache_error_propagates(tmp_path, monkeypatch):
    calls = _counted_vggsfm(monkeypatch)
    (tmp_path / "tracks.zarr").mkdir()

    def broken_open(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(tracks_mod.zarr, "open", broken_open)

    with pytest.raises(TypeError, match="bug"):
        _cached(tmp_path)

    # zarr.open also backs the cache write, so only a zero extract count proves the read raised
    assert calls == []


def test_cache_without_frame_paths_keys_on_world_points(tmp_path, monkeypatch):
    calls = _counted_vggsfm(monkeypatch)
    wp = np.zeros((3, 8, 8, 3), np.float32)

    for _ in range(2):
        tracks_mod.extract_tracks(torch.zeros(3, 3, 8, 8), None, wp, source="vggsfm", cache_dir=tmp_path)

    assert len(calls) == 1
    assert (tmp_path / "tracks.zarr").exists()


def test_no_cache_without_cache_dir(monkeypatch):
    calls = _counted_vggsfm(monkeypatch)
    wp = np.zeros((3, 8, 8, 3), np.float32)
    paths = [f"frame_{i:06d}.png" for i in range(3)]

    for _ in range(2):
        tracks_mod.extract_tracks(torch.zeros(3, 3, 8, 8), None, wp, source="vggsfm", frame_paths=paths)

    assert len(calls) == 2


def test_vggsfm_source_threads_knobs(monkeypatch):
    calls = _counted_vggsfm(monkeypatch)

    tracks_mod.extract_tracks(
        "imgs", "conf", "wp", source="vggsfm", max_query_pts=7, query_frame_num=3, fine_tracking=True
    )

    assert calls == [{"max_query_pts": 7, "query_frame_num": 3, "fine_tracking": True}]


def test_matcher_source_dispatches(monkeypatch):
    seen = {}

    def fake_build(source, frame_paths, world_points, extrinsics, intrinsics, retrieval, seed_fraction):
        seen.update(source=source, frame_paths=frame_paths, retrieval=retrieval, seed_fraction=seed_fraction)
        return "frame", "track", "xy", "score", "pts3d"

    monkeypatch.setattr(tracks_mod, "build_tracks", fake_build)
    paths = ["a.png", "b.png", "c.png", "d.png"]

    out = tracks_mod.extract_tracks(
        "imgs",
        "conf",
        "wp",
        source="xfeat",
        extrinsics="ext",
        intrinsics="K",
        frame_paths=paths,
        retrieval="megaloc",
        seed_fraction=0.5,
    )

    assert out == ("frame", "track", "xy", "score", "pts3d")
    assert seen == {"source": "xfeat", "frame_paths": paths, "retrieval": "megaloc", "seed_fraction": 0.5}


def test_unknown_source_kwarg_raises(monkeypatch):
    monkeypatch.setattr(tracks_mod, "predict_tracks", _fake_predict([]))

    with pytest.raises(TypeError, match="seed_fraction"):
        tracks_mod.extract_tracks(
            torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8), np.zeros((2, 8, 8, 3)), source="vggsfm", seed_fraction=0.5
        )


def test_vggsfm_defaults_are_upstream_demo_values():
    params = inspect.signature(tracks_mod._vggsfm_tracks).parameters

    assert params["max_query_pts"].default == 4096
    assert params["query_frame_num"].default == 8
    assert params["fine_tracking"].default is False


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_build_tracks_rejects_seed_fraction_out_of_range(fraction):
    with pytest.raises(ValueError, match="seed_fraction"):
        tracks_mod.build_tracks(
            "xfeat",
            ["a.png", "b.png"],
            np.zeros((2, 8, 8, 3), np.float32),
            np.tile(np.eye(4), (2, 1, 1)),
            np.tile(np.eye(3), (2, 1, 1)),
            seed_fraction=fraction,
        )


def test_matcher_source_needs_frame_paths():
    with pytest.raises(ValueError, match="frame_paths"):
        tracks_mod.extract_tracks("imgs", "conf", "wp", source="xfeat", extrinsics="ext", intrinsics="K")
