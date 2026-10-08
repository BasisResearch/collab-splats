"""
BA tracks: a cached extraction from VGGSfM or matcher star tracks, on the model grid.

- extract_tracks: zarr cache + source dispatch, the one entry bundle_adjustment.refine calls
- build_tracks: star tracks over verified matcher correspondences
- tracks are flat observations: frame, track, xy, score rows sorted by (frame, track), plus pts3d
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import repeat
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import zarr
from zarr.codecs import BloscCodec

from vggt.dependency.track_predict import predict_tracks

from collab_splats.geometry.projection import reprojection_error, sample_world_points
from collab_splats.localization.extractors import LocalMatcher
from collab_splats.localization.retrieval import BaseRetrievalExtractor
from collab_splats.utils.io import read_image
from collab_splats.utils.torch_utils import get_device, pytorch_gc, to_numpy

logger = logging.getLogger(__name__)


########################################################################
# Public API
########################################################################


def extract_tracks(
    images: np.ndarray | torch.Tensor,
    confidence: np.ndarray | torch.Tensor,
    world_points: np.ndarray,
    *,
    source: Literal["vggsfm", "xfeat", "loma"],
    frame_paths: list[Path] | None = None,
    cache_dir: Path | None = None,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    **source_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tracks from the chosen source; cached as cache_dir/tracks.zarr when cache_dir is set.

    - cache key: sorted frame paths, a world_points digest, source and source_kwargs; a mismatch is rebuilt
    - the key is stamped last, so a half-written store has no key and is rebuilt
    - matcher sources need poses, model-grid K and full-res frame paths
    - source_kwargs go to _vggsfm_tracks (vggsfm) or build_tracks (matchers); unknown keys raise TypeError

    Args:
        images: (N, 3, H, W) model-grid frames in [0, 1]; VGGSfM only.
        confidence: (N, H, W) per-pixel confidence; VGGSfM only.
        world_points: (N, H, W, 3) model-grid points the tracks take their 3D points from.
        source: "vggsfm" predicts tracks; "xfeat" / "loma" build matcher star tracks.
        frame_paths: full-res store frame per model frame; cache key and matcher input.
        cache_dir: track-cache dir; None skips the cache.
        extrinsics: (N, 4, 4) world-to-camera; matcher sources only.
        intrinsics: (N, 3, 3) model-grid K; matcher sources only.
        source_kwargs: source-specific settings, e.g. max_query_pts (vggsfm) or seed_fraction (matchers).

    Returns:
        frame (M,) int32, track (M,) int32, xy (M, 2) model px, score (M,) and pts3d (P, 3) world; rows
        sorted by (frame, track).

    Raises:
        ValueError: a matcher source without extrinsics, intrinsics or frame_paths, or any build_tracks error.
        TypeError: a source_kwargs key the chosen source does not take.
    """
    arrays = ("frame", "track", "xy", "score", "pts3d")

    # Cache hit: a store stamped with this call's key (sorted frame paths, world_points digest, knobs)
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "tracks.zarr"
        knobs = {"track_source": source, **source_kwargs}
        digest = np.ascontiguousarray(world_points)
        digest = hashlib.sha256(digest.tobytes()).hexdigest()
        paths = sorted(str(p) for p in frame_paths or ())
        meta = {"frame_paths": paths, "world_points": digest, **knobs}
        meta = json.dumps(meta, sort_keys=True)
        key = hashlib.sha256(meta.encode()).hexdigest()

        # A stale or half-written store (key stamped last) is deleted and rebuilt
        if cache_path.exists():
            store = zarr.open(str(cache_path), mode="r")

            if store.attrs.get("cache_key") == key:
                logger.info("Track cache hit (skipping extraction): %s", cache_path)
                return tuple(store[k][:] for k in arrays)

            logger.warning("Track cache key mismatch, re-extracting: %s", cache_path)
            shutil.rmtree(cache_path)

    # VGGSfM predicts tracks; matcher sources chain verified matches over the prior poses
    if source == "vggsfm":
        out = _vggsfm_tracks(images, confidence, world_points, **source_kwargs)
    else:
        if extrinsics is None or intrinsics is None or not frame_paths:
            raise ValueError(f"extract_tracks: source {source!r} needs extrinsics, intrinsics and frame_paths")

        out = build_tracks(source, frame_paths, world_points, extrinsics, intrinsics, **source_kwargs)

    # Write the cache: one lz4 chunk per array, key stamped last
    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lz4 = BloscCodec(cname="lz4")
        store = zarr.open(str(cache_path), mode="w")

        for name, arr in zip(arrays, out):
            store.create_array(name, data=arr, chunks=arr.shape, compressors=lz4)

        store.attrs["cache_key"] = key
        logger.debug("Track cache saved: %s", cache_path)

    return out


def build_tracks(
    source: Literal["xfeat", "loma"],
    frame_paths: list[Path],
    world_points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    window: int = 10,
    retrieval: str = "dino-salad",
    retrieval_k: int = 20,
    retrieval_nms: int = 25,
    seed_fraction: float = 0.34,
    min_matches: int = 50,
    depth_tol: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Star tracks from verified matcher correspondences, on the model grid.

    - pairs: sequential window plus retrieval top-k on model-grid frames, outside the nms band
    - matches gated by symmetric reprojection under the prior poses, then pycolmap-verified in memory
    - one track per seed keypoint and its direct verified matches, >= 2 observations
    - full-res keypoints map to the model grid by separate x / y scales (multiple-of-14 height stretch)

    Args:
        source: LocalMatcher model, "xfeat" or "loma"; its max_num_keypoints caps keypoints per frame.
        frame_paths: full-res store frame per model frame.
        world_points: (N, H, W, 3) model-grid world points.
        extrinsics: (N, 4, 4) world-to-camera.
        intrinsics: (N, 3, 3) model-grid K.
        window: sequential pair span, frames.
        retrieval: BaseRetrievalExtractor registry name, e.g. "dino-salad" or "megaloc".
        retrieval_k: retrieval pairs per frame; 0 loads no retrieval model.
        retrieval_nms: frame gap at or under which retrieval pairs are suppressed.
        seed_fraction: share of frames, evenly spaced (at least 2), whose keypoints seed tracks.
        min_matches: verified inliers a pair needs to join the correspondence graph.
        depth_tol: symmetric transfer-error gate, model px.

    Returns:
        frame (M,) int32, track (M,) int32, xy (M, 2) model px, score (M,) all 1.0 and pts3d (P, 3) world;
        rows sorted by (frame, track).

    Raises:
        ValueError: seed_fraction outside (0, 1], a frame's aspect differs from the model grid's,
            retrieval_nms < 0, or an unknown retrieval name.
    """
    # Model grid shape and per-phase wall times
    world_points = np.asarray(world_points)
    N, H, W = world_points.shape[:3]
    timings = {}

    # Seeds are a share of the frames
    if not 0 < seed_fraction <= 1:
        raise ValueError(f"build_tracks: seed_fraction must be in (0, 1], got {seed_fraction!r}")

    # Seed count scales with the scene; at least 2 so every scene seeds tracks
    seed_frames = round(seed_fraction * N)
    seed_frames = max(2, seed_frames)

    # A negative band leaves the diagonal unsuppressed, so self pairs would reach verification
    if retrieval_nms < 0:
        raise ValueError(f"build_tracks: retrieval_nms must be >= 0, got {retrieval_nms}")

    # Unknown retrieval name raises here, even with retrieval_k == 0
    retriever_cls = BaseRetrievalExtractor.get(retrieval)

    # 1. Extract: full-res frames per matcher call, plus model-grid retrieval descriptors unless retrieval_k == 0
    t = time.perf_counter()
    matcher = LocalMatcher(source)
    retriever = retriever_cls() if retrieval_k > 0 else None
    chunk = 32
    feats, descs = [], []

    # One decode pool for every chunk; cv2 releases the GIL
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(frame_paths), chunk):
            # Decode the chunk in parallel
            frames = pool.map(read_image, frame_paths[start : start + chunk])
            frames = list(frames)

            # One batched extract per chunk: per-frame calls are 4.6x slower and pick other keypoints
            feats += matcher.extract(frames)

            # Retrieval sees the model grid, as the feedforward images: antialiased resize, one call per chunk
            if retriever is None:
                continue

            images = np.stack(frames)
            images = torch.from_numpy(images)
            images = images.permute(0, 3, 1, 2).float() / 255
            images = F.interpolate(images, size=(H, W), mode="bilinear", antialias=True)
            descs.append(retriever(images))

    del retriever
    pytorch_gc()

    # 2. Model grid + world points per keypoint: pixel-center size map; a cropped preprocess is refused
    kps, kp_world = [], []

    for i, f in enumerate(feats):
        w, h = f.image_size

        if abs((w / h) / (W / H) - 1) > 0.02:
            raise ValueError(f"build_tracks: frame aspect {w}x{h} differs from model grid {W}x{H} (cropped preprocess)")

        scale = torch.tensor([W / w, H / h])
        keypoints = (f.keypoints + 0.5) * scale - 0.5
        feats[i] = replace(f, keypoints=keypoints, image_size=(W, H))
        feats[i] = matcher.to_device(feats[i])
        kp = to_numpy(keypoints)
        pts, valid = sample_world_points(world_points[i], kp)
        pts[~valid] = np.nan
        kps.append(kp)
        kp_world.append(pts)

    timings["extract"] = time.perf_counter() - t

    # 3. Pairs: sequential within window, then retrieval top-k outside the nms band
    t = time.perf_counter()
    pairs = [(a, b) for a in range(N) for b in range(a + 1, min(N, a + window + 1))]

    if retrieval_k > 0:
        desc = torch.cat(descs)

        # Cosine similarity with the near-diagonal band suppressed
        sim = desc @ desc.T
        sim = sim.numpy()
        frame = np.arange(N)
        gap = np.abs(frame[:, None] - frame[None, :])
        sim[gap <= retrieval_nms] = -np.inf

        # Top-k per frame; under k candidates, argsort reaches the -inf band (self included), so drop those
        top = np.argsort(-sim, axis=1)
        top = top[:, :retrieval_k]
        rows = np.broadcast_to(frame[:, None], top.shape)
        keep = np.isfinite(sim[rows, top])

        # As (min, max) pairs not already sequential
        lo = np.minimum(rows[keep], top[keep])
        hi = np.maximum(rows[keep], top[keep])
        lo = lo.tolist()
        hi = hi.tolist()
        extra = zip(lo, hi)
        extra = set(extra)
        extra -= set(pairs)
        pairs += sorted(extra)

    timings["pairs"] = time.perf_counter() - t

    # 4. Match + depth gate per chunk of pairs: symmetric transfer error under the prior poses; NaN world points give inf
    t = time.perf_counter()

    # Gate tensors on the GPU: CPU torch oversubscribes the container's cores and runs ~10x slower
    device = get_device()
    w2c = torch.as_tensor(extrinsics, dtype=torch.float64, device=device)
    K = torch.as_tensor(intrinsics, dtype=torch.float64, device=device)
    offsets = np.cumsum([0] + [len(kp) for kp in kps[:-1]])
    kps_t = torch.as_tensor(np.concatenate(kps), dtype=torch.float64, device=device)
    kp_world_t = torch.as_tensor(np.concatenate(kp_world), dtype=torch.float64, device=device)
    matched = []
    n_raw = n_kept = 0

    for start in range(0, len(pairs), chunk):
        batch = pairs[start : start + chunk]
        results = matcher.match_batch([(feats[a], feats[b]) for a, b in batch])

        # Every match of the chunk as flat rows: each side's frame and its row in the concatenated keypoints
        counts = [len(m) for m in results]
        n_raw += sum(counts)
        frame_a = np.repeat([a for a, _ in batch], counts)
        frame_b = np.repeat([b for _, b in batch], counts)
        rows_a = offsets[frame_a] + np.concatenate([m.idx_q for m in results])
        rows_b = offsets[frame_b] + np.concatenate([m.idx_db for m in results])
        frame_a, frame_b, rows_a, rows_b = (
            torch.as_tensor(x, device=device) for x in (frame_a, frame_b, rows_a, rows_b)
        )

        # One per-match-camera transfer error each way for the whole chunk
        err_ab = reprojection_error(kp_world_t[rows_a], w2c[frame_b], K[frame_b], kps_t[rows_b])
        err_ba = reprojection_error(kp_world_t[rows_b], w2c[frame_a], K[frame_a], kps_t[rows_a])
        ok = torch.maximum(err_ab, err_ba) < depth_tol
        ok = to_numpy(ok)
        ok = np.split(ok, np.cumsum(counts)[:-1])

        # A pair with no depth-consistent match has nothing to verify
        for (a, b), m, ok_pair in zip(batch, results, ok):
            if not ok_pair.any():
                continue

            idx = np.stack([m.idx_q[ok_pair], m.idx_db[ok_pair]], axis=1)
            idx = idx.astype(np.uint32)
            matched.append(((a, b), idx))
            n_kept += int(ok_pair.sum())

    # Free the matcher and its device features before the bae solve claims the GPU
    del matcher, feats
    pytorch_gc()
    timings["match"] = time.perf_counter() - t

    # 5. Verify: one PINHOLE camera per frame with no prior focal
    t = time.perf_counter()
    cameras = []

    for i in range(N):
        K_i = intrinsics[i]
        params = [float(K_i[0, 0]), float(K_i[1, 1]), float(K_i[0, 2]), float(K_i[1, 2])]
        camera = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=params, camera_id=i + 1)
        cameras.append(camera)

    # Fixed RANSAC seed: thread scheduling must not change the inliers
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.random_seed = 0
    kps64 = [kp.astype(np.float64) for kp in kps]

    # Per-pair argument lists for the verify map
    cams_a = [cameras[a] for (a, _), _ in matched]
    kps_a = [kps64[a] for (a, _), _ in matched]
    cams_b = [cameras[b] for (_, b), _ in matched]
    kps_b = [kps64[b] for (_, b), _ in matched]
    pair_matches = [idx for _, idx in matched]

    # Verify pairs in parallel; pycolmap releases the GIL
    with ThreadPoolExecutor(8) as pool:
        geometries = pool.map(
            pycolmap.estimate_two_view_geometry, cams_a, kps_a, cams_b, kps_b, pair_matches, repeat(options)
        )
        geometries = list(geometries)

    timings["verify"] = time.perf_counter() - t

    # 6. Tracks: graph over every frame; pairs in input order keep it deterministic
    t = time.perf_counter()
    graph = pycolmap.CorrespondenceGraph()

    for i in range(N):
        graph.add_image(i + 1, len(kps[i]))

    # A pair under min_matches verified inliers stays out of the graph
    for ((a, b), _), geometry in zip(matched, geometries):
        if len(geometry.inlier_matches) >= min_matches:
            graph.add_two_view_geometry(a + 1, b + 1, geometry)

    # Freeze the graph, then read star tracks off it
    graph.finalize()
    out = _assemble_tracks(graph, kps, kp_world, seed_frames)
    timings["chain"] = time.perf_counter() - t

    logger.info(
        "tracks: %d frames, %d pairs, %d/%d matches kept by depth, %d tracks, %d observations; timings %s",
        N,
        len(pairs),
        n_kept,
        n_raw,
        len(out[4]),
        len(out[0]),
        {k: round(v, 1) for k, v in timings.items()},
    )
    return out


########################################################################
# Steps
########################################################################


def _vggsfm_tracks(
    images: np.ndarray | torch.Tensor,
    conf: np.ndarray | torch.Tensor,
    world_points: np.ndarray,
    *,
    max_query_pts: int = 4096,
    query_frame_num: int = 8,
    fine_tracking: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-frame 2D tracks predicted by VGGSfM (ALIKED+SP keypoints), uncached, as flat observations.

    - conf: (N, H, W); world_points: (N, H, W, 3); a non-square grid is padded to square
    - max_query_pts / query_frame_num: upstream demo defaults
    - fine_tracking: sharper tracks at a large memory cost
    - zero-score cells are not observations; rows frame-major
    """
    logger.info(
        "Extracting VGGSfM tracks: %d frames, max_query_pts=%d, query_frame_num=%d, fine_tracking=%s (slow step)",
        len(images),
        max_query_pts,
        query_frame_num,
        fine_tracking,
    )

    # Images on get_device() in float32; the tracker's grid_sample lacks CUDA BFloat16
    device = get_device()
    images = torch.as_tensor(images)
    images = images.to(device).float()

    # Confidence and world points on CPU (VGGSfM mixes in CPU numpy indexing)
    conf = torch.as_tensor(conf).cpu()
    points = torch.tensor(world_points, dtype=torch.float32)

    # VGGSfM asserts H == W when given conf and world points; pad the shorter dim
    H, W = images.shape[-2:]

    if H != W:
        pad_h, pad_w = max(H, W) - H, max(H, W) - W
        images = F.pad(images, (0, pad_w, 0, pad_h))
        conf = F.pad(conf, (0, pad_w, 0, pad_h))
        points = F.pad(points, (0, 0, 0, pad_w, 0, pad_h))

    # no_grad so the upstream .numpy() calls work
    with torch.no_grad():
        pred_tracks, pred_vis_scores, _pred_confs, pred_pts3d, _pred_colors = predict_tracks(
            images,
            conf=conf,
            points_3d=points,
            max_query_pts=max_query_pts,
            query_frame_num=query_frame_num,
            fine_tracking=fine_tracking,
        )

    # Upstream returns numpy; pin float32
    tracks = np.asarray(pred_tracks, dtype=np.float32)
    vis_scores = np.asarray(pred_vis_scores, dtype=np.float32)
    pts3d = np.asarray(pred_pts3d, dtype=np.float32)

    # Flat observations, frame-major; zero-score cells are not observations
    frame, track = np.nonzero(vis_scores > 0)
    xy = tracks[frame, track]
    score = vis_scores[frame, track]
    return frame.astype(np.int32), track.astype(np.int32), xy, score, pts3d


def _assemble_tracks(
    graph: pycolmap.CorrespondenceGraph, kps: list[np.ndarray], kp_world: list[np.ndarray], seed_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Star tracks as flat observations: each seed keypoint plus its direct verified matches.

    - seeds: seed_frames evenly spaced frames; every frame is in the graph, matched or not
    - a frame contributing two keypoints to one track is dropped from it; >= 2 observations kept
    - pts3d is the world point at the first observation; tracks without one are dropped
    - rows sorted by (frame, track)
    """
    # Evenly spaced seed frames
    N = len(kps)
    seeds = np.linspace(0, N - 1, seed_frames).round().astype(int)

    # More seeds than frames would repeat frames and duplicate their tracks
    seeds = np.unique(seeds)
    frames, tracks, xy, pts3d = [], [], [], []
    n_conflict = 0

    for seed in seeds:
        image_id = int(seed) + 1

        for idx in range(len(kps[seed])):
            corrs = graph.extract_correspondences(image_id, idx)

            # Unmatched keypoint: no track
            if not corrs:
                continue

            # Distinct (frame, keypoint) members: a repeated entry is one keypoint, not an ambiguity
            members = {(c.image_id, c.point2D_idx) for c in corrs}
            members.add((image_id, idx))

            # An ambiguous frame (two keypoints in the track) is dropped from the track
            counts = Counter(img_id for img_id, _ in members)
            n_conflict += sum(n > 1 for n in counts.values())
            obs = sorted((img_id - 1, kp) for img_id, kp in members if counts[img_id] == 1)

            # Too short, or no world point at the first observation
            if len(obs) < 2:
                continue

            i0, k0 = obs[0]
            point = kp_world[i0][k0]

            if np.isnan(point).any():
                continue

            # Emit the track's observations under the next track id
            track_id = len(pts3d)
            pts3d.append(point)

            for i, k in obs:
                frames.append(i)
                tracks.append(track_id)
                xy.append(kps[i][k])

    logger.info("tracks: %d star tracks, %d ambiguous frame drops", len(pts3d), n_conflict)

    # Frame-major rows, the order BA's observation arrays expect
    frame = np.asarray(frames, np.int32)
    track = np.asarray(tracks, np.int32)
    order = np.lexsort((track, frame))
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    pts3d = np.asarray(pts3d, np.float32).reshape(-1, 3)
    score = np.ones(len(frame), np.float32)
    return frame[order], track[order], xy[order], score, pts3d
