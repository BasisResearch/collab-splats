"""
Matcher tracks for bundle adjustment: star queries over verified cached-feature matches.

- full-res keypoints mapped to the model grid; pairs = sequential window + DINO-SALAD retrieval
- pycolmap verifies and builds the correspondence graph; each seed keypoint's direct matches form a track
- same (tracks, vis, pts3d) triple as extract_tracks_vggsfm
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pycolmap
import torch

from collab_splats.geometry.projection import project, sample_world_points
from collab_splats.localization.extractors import LocalFeatures, LocalMatcher
from collab_splats.localization.retrieval import DinoSaladExtractor
from collab_splats.preproc.frames import frame_idx_from_path, read_frames
from collab_splats.utils.torch_utils import batch_iterator, pytorch_gc, to_numpy

logger = logging.getLogger(__name__)


########################################
# Public API
########################################


def build_tracks(
    matcher: LocalMatcher,
    images: np.ndarray | torch.Tensor,
    frame_paths: list[Path],
    world_points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    window: int = 10,
    retrieval_k: int = 20,
    retrieval_nms: int = 25,
    seed_frames: int = 30,
    min_matches: int = 50,
    depth_tol: float = 8.0,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Star tracks from verified matcher correspondences, on the model grid.

    - pairs: sequential window plus DINO-SALAD top-k outside the nms band
    - matches gated by symmetric reprojection under the prior poses, then pycolmap-verified
    - one track per seed keypoint and its direct verified matches, >= 2 observations

    Args:
        matcher: xfeat or loma LocalMatcher; its max_num_keypoints caps keypoints per frame.
        images: (N, 3, H, W) RGB in [0, 1] on the model grid; retrieval only.
        frame_paths: full-res store frame per model frame, all in one directory.
        world_points: (N, H, W, 3) model-grid world points.
        extrinsics: (N, 4, 4) world-to-camera.
        intrinsics: (N, 3, 3) model-grid K.
        window: sequential pair span, frames.
        retrieval_k: retrieval pairs per frame.
        retrieval_nms: frame gap at or under which retrieval pairs are suppressed.
        seed_frames: evenly spaced frames whose keypoints seed tracks.
        min_matches: verified inliers a pair needs to join the correspondence graph.
        depth_tol: symmetric transfer-error gate, model px.
        batch_size: frames per extract and retrieval chunk.

    Returns:
        tracks (N, P, 2) model px, vis (N, P) 1.0 where observed, pts3d (P, 3) world; float32.

    Raises:
        ValueError: frame_paths span directories, or a frame's aspect differs from the model grid's.
    """
    world_points = np.asarray(world_points)
    N, H, W = world_points.shape[:3]
    timings = {}

    # Full-res features mapped onto the model grid, then moved to the matcher once
    t = time.perf_counter()
    feats = _extract_full_res(matcher, frame_paths, batch_size)
    feats = [_to_model_grid(f, (H, W)) for f in feats]
    kps = [to_numpy(f.keypoints) for f in feats]
    feats = [matcher.to_device(f) for f in feats]
    lifted = _lift_keypoints(world_points, kps)
    timings["extract"] = time.perf_counter() - t

    # Candidate pairs
    t = time.perf_counter()
    pairs = _pairs(images, window, retrieval_k, retrieval_nms, batch_size)
    timings["pairs"] = time.perf_counter() - t

    with tempfile.TemporaryDirectory(prefix="tracks_") as tmp:
        db_path = Path(tmp) / "database.db"

        # Match every pair, depth-filter, and write the database
        t = time.perf_counter()
        n_raw, n_kept = _write_database(
            db_path, matcher, feats, kps, lifted, pairs, extrinsics, intrinsics, (H, W), depth_tol
        )
        timings["match"] = time.perf_counter() - t

        # Epipolar verification over exactly the matched pairs
        t = time.perf_counter()
        pairs_path = Path(tmp) / "pairs.txt"
        lines = [f"{a:05d}.png {b:05d}.png" for a, b in pairs]
        pairs_path.write_text("\n".join(lines))
        pycolmap.verify_matches(str(db_path), str(pairs_path))
        timings["verify"] = time.perf_counter() - t

        # Correspondence graph of pairs with at least min_matches inliers
        t = time.perf_counter()
        options = pycolmap.DatabaseCacheOptions()
        options.min_num_matches = min_matches
        db = pycolmap.Database.open(str(db_path))

        try:
            cache = pycolmap.DatabaseCache.create(db, options)
            graph = cache.correspondence_graph
        finally:
            db.close()

        observations = _star_tracks(graph, kps, seed_frames)
        timings["chain"] = time.perf_counter() - t

    tracks, vis, pts3d = _assemble(observations, kps, lifted, N)
    logger.info(
        "tracks: %d frames, %d pairs, %d/%d matches kept by depth, %d tracks, %d observations; timings %s",
        N,
        len(pairs),
        n_kept,
        n_raw,
        tracks.shape[1],
        int(vis.sum()),
        {k: round(v, 1) for k, v in timings.items()},
    )
    return tracks, vis, pts3d


########################################
# Steps
########################################


def _extract_full_res(matcher: LocalMatcher, frame_paths: list[Path], batch_size: int) -> list[LocalFeatures]:
    """
    Features of every full-res store frame, read and extracted one chunk at a time.

    - chunked so the full-res stack is never resident
    """
    # An empty store has nothing to extract or to locate a directory from
    if not frame_paths:
        raise ValueError("build_tracks: frame_paths is empty")

    frames_dir = Path(frame_paths[0]).parent

    # read_frames serves one directory by source index
    if any(Path(p).parent != frames_dir for p in frame_paths):
        raise ValueError(f"build_tracks: frame_paths must sit in one directory, got more than {frames_dir}")

    idxs = [frame_idx_from_path(p) for p in frame_paths]
    feats = []

    for (chunk,) in batch_iterator(batch_size, idxs):
        frames = read_frames(frames_dir, chunk)
        feats += matcher.extract(list(frames))

    return feats


def _to_model_grid(features: LocalFeatures, model_hw: tuple[int, int], *, aspect_tol: float = 0.02) -> LocalFeatures:
    """
    Full-res keypoints on the model grid, pixel-center size map.

    - separate x / y scales absorb the preprocess's multiple-of-14 height rounding (a slight stretch)
    - a frame whose aspect ratio differs from the grid's by more than aspect_tol (relative) is refused
    """
    H, W = model_hw
    w, h = features.image_size

    # Size-only mapping would misplace every keypoint of a cropped preprocess
    if abs((w / h) / (W / H) - 1) > aspect_tol:
        raise ValueError(f"build_tracks: frame aspect {w}x{h} differs from model grid {W}x{H} (cropped preprocess)")

    scale = torch.tensor([W / w, H / h])
    keypoints = (features.keypoints + 0.5) * scale - 0.5
    return replace(features, keypoints=keypoints, image_size=(W, H))


def _pairs(
    images: np.ndarray | torch.Tensor, window: int, retrieval_k: int, retrieval_nms: int, batch_size: int
) -> list[tuple[int, int]]:
    """
    Sequential pairs within window, then sorted DINO-SALAD pairs outside the nms band.

    - retrieval: cosine top-k per frame, |a - b| <= nms suppressed, deduped against the sequential set
    - retrieval_k == 0 never loads DINO-SALAD
    """
    # A negative band leaves the diagonal unsuppressed, so self pairs would reach the database
    if retrieval_nms < 0:
        raise ValueError(f"build_tracks: retrieval_nms must be >= 0, got {retrieval_nms}")

    N = len(images)
    pairs = [(a, b) for a in range(N) for b in range(a + 1, min(N, a + window + 1))]

    # Sequential pairs only: skip the retrieval model entirely
    if retrieval_k == 0:
        return pairs

    # Global descriptors per chunk of model-grid images
    salad = DinoSaladExtractor()
    images = torch.as_tensor(images)
    descs = []

    for start in range(0, N, batch_size):
        chunk = images[start : start + batch_size]
        chunk = chunk.float()
        descs.append(salad(chunk))

    desc = torch.cat(descs)
    del salad
    pytorch_gc()

    # Cosine similarity with the near-diagonal band suppressed
    sim = desc @ desc.T
    sim = sim.numpy()
    frame = np.arange(N)
    sim[np.abs(frame[:, None] - frame[None, :]) <= retrieval_nms] = -np.inf

    # Top-k per frame, as (min, max) pairs not already sequential
    extra = set()

    for a in range(N):
        order = np.argsort(-sim[a])
        order = order[:retrieval_k]

        # Fewer than k candidates outside the band: argsort reaches suppressed -inf entries, self included
        for b in order:
            if not np.isfinite(sim[a, b]):
                continue

            extra.add((min(a, int(b)), max(a, int(b))))

    extra -= set(pairs)
    return pairs + sorted(extra)


def _lift_keypoints(world_points: np.ndarray, kps: list[np.ndarray]) -> list[np.ndarray]:
    """
    World point under every keypoint of every frame; NaN where the lookup is invalid.
    """
    lifted = []

    for i, kp in enumerate(kps):
        pts, valid = sample_world_points(world_points[i], kp)
        pts[~valid] = np.nan
        lifted.append(pts)

    return lifted


def _transfer_err(points: np.ndarray, px: np.ndarray, world_to_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Pixel error of world points projected into a camera vs their matched keypoints; inf if invalid.
    """
    points = points.astype(np.float64)
    world_to_cam = world_to_cam.astype(np.float64)
    K = K.astype(np.float64)
    px = px.astype(np.float64)
    points_t = torch.from_numpy(points)
    w2c_t = torch.from_numpy(world_to_cam)
    K_t = torch.from_numpy(K)
    px_t = torch.from_numpy(px)
    pixels, points_cam = project(points_t, w2c_t, K_t)
    err = torch.linalg.norm(pixels - px_t, dim=1)

    # NaN lifts and points behind the camera never pass
    err[~(points_cam[:, 2] > 0)] = torch.inf
    err[torch.isnan(err)] = torch.inf
    return err.numpy()


def _depth_filter(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    px_a: np.ndarray,
    px_b: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsics: np.ndarray,
    depth_tol: float,
) -> np.ndarray:
    """
    Matches whose symmetric transfer error stays under depth_tol model px.

    - world_to_cam / intrinsics stack frame a then frame b
    """
    err_ab = _transfer_err(pts_a, px_b, world_to_cam[1], intrinsics[1])
    err_ba = _transfer_err(pts_b, px_a, world_to_cam[0], intrinsics[0])
    return np.maximum(err_ab, err_ba) < depth_tol


def _write_database(
    db_path: Path,
    matcher: LocalMatcher,
    feats: list[LocalFeatures],
    kps: list[np.ndarray],
    lifted: list[np.ndarray],
    pairs: list[tuple[int, int]],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    model_hw: tuple[int, int],
    depth_tol: float,
) -> tuple[int, int]:
    """
    pycolmap database of cameras, images, keypoints and depth-filtered matches; (raw, kept) counts.

    - image i is image_id i + 1, named f"{i:05d}.png", one PINHOLE camera per frame
    """
    H, W = model_hw
    db = pycolmap.Database.open(str(db_path))

    try:
        # One camera and image row per frame, with its model-grid keypoints
        for i in range(len(kps)):
            K = intrinsics[i]
            params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
            camera = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=params, camera_id=i + 1)
            db.write_camera(camera, use_camera_id=True)
            row = pycolmap.Image(name=f"{i:05d}.png", camera_id=i + 1)
            row.image_id = i + 1
            db.write_image(row, use_image_id=True)
            kp = kps[i].astype(np.float64)
            db.write_keypoints(i + 1, kp)

        n_raw = n_kept = 0

        # Match each pair; keep depth-consistent matches only
        for a, b in pairs:
            m = matcher.match(feats[a], feats[b])
            n_raw += len(m)

            if len(m) == 0:
                continue

            ok = _depth_filter(
                lifted[a][m.idx_q],
                lifted[b][m.idx_db],
                kps[a][m.idx_q],
                kps[b][m.idx_db],
                extrinsics[[a, b]],
                intrinsics[[a, b]],
                depth_tol,
            )

            if not ok.any():
                continue

            idx = np.stack([m.idx_q[ok], m.idx_db[ok]], axis=1)
            idx = idx.astype(np.uint32)
            db.write_matches(a + 1, b + 1, idx)
            n_kept += int(ok.sum())
    finally:
        db.close()

    return n_raw, n_kept


def _star_tracks(
    graph: pycolmap.CorrespondenceGraph, kps: list[np.ndarray], seed_frames: int
) -> list[list[tuple[int, int]]]:
    """
    One track per seed keypoint: itself plus its direct verified matches, as (frame, keypoint) lists.

    - seeds: seed_frames evenly spaced frames; frames absent from the graph are skipped
    - a frame contributing two keypoints to one track is dropped from it; >= 2 observations kept
    """
    N = len(kps)
    seeds = np.linspace(0, N - 1, seed_frames).round().astype(int)

    # More seeds than frames would repeat frames and duplicate their tracks
    seeds = np.unique(seeds)
    tracks = []
    n_conflict = 0

    for seed in seeds:
        image_id = int(seed) + 1

        # A seed with no verified matches is absent from the graph
        if not graph.exists_image(image_id):
            logger.info("tracks: seed frame %d has no verified matches, skipped", seed)
            continue

        for idx in range(len(kps[seed])):
            corrs = graph.extract_transitive_correspondences(image_id, idx, 1)
            members = {(c.image_id, c.point2D_idx) for c in corrs}
            members.add((image_id, idx))

            # Group by frame; an ambiguous frame is dropped from the track
            per_image = defaultdict(list)

            for img_id, kp in members:
                per_image[img_id].append(kp)

            obs = []

            for img_id, kp_list in sorted(per_image.items()):
                if len(kp_list) > 1:
                    n_conflict += 1
                    continue

                obs.append((img_id - 1, kp_list[0]))

            if len(obs) >= 2:
                tracks.append(obs)

    logger.info("tracks: %d star tracks, %d ambiguous frame drops", len(tracks), n_conflict)
    return tracks


def _assemble(
    observations: list[list[tuple[int, int]]], kps: list[np.ndarray], lifted: list[np.ndarray], n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense (N, P) layout; pts3d is the world point at the first observation, invalid tracks dropped.
    """
    P = len(observations)
    tracks = np.zeros((n_frames, P, 2), np.float32)
    vis = np.zeros((n_frames, P), np.float32)
    pts3d = np.zeros((P, 3), np.float32)

    for j, obs in enumerate(observations):
        for i, k in obs:
            tracks[i, j] = kps[i][k]
            vis[i, j] = 1.0

        i0, k0 = obs[0]
        pts3d[j] = lifted[i0][k0]

    # A first observation with no world point has nothing to lift the track from
    valid = ~np.isnan(pts3d).any(axis=1)
    return tracks[:, valid], vis[:, valid], pts3d[valid]
