"""
Report-only, scale-free cross-view reconstruction quality, no ground truth.

- compute_reconstruction_quality: every report table from arrays; columns in docs/source/api/geometry.rst
- depth_pairs + pooled residual histogram: depth error where two views overlap
- photometric_pairs: NCC of pixel colors after warping one view into another through its depth, at index gaps 1-20
- frames: per-frame median depth error, multiview agreement, covered fraction, confidence median
- confidence_median: not comparable across backbones
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from collab_splats.geometry.projection import depth_agreement, project, unproject
from collab_splats.geometry.transforms import invert_poses
from collab_splats.preproc import frames
from collab_splats.utils.image import upsample_depths
from collab_splats.utils.torch_utils import (
    full_fp32_matmul,
    get_device,
    infer_batch_size,
)

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

# Keyframe stem contract frame_{idx:06d}: frame_idx_from_path would read any numeric tail
_FRAME_STEM_RE = re.compile(r"frame_\d{6,}$")

########################################################################
# Per pair: depth
########################################################################


def _collect_pairs(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    rel_thresh: float,
    *,
    min_pair_overlap: float = 0.0,
    overlap_stride: int = 8,
    target_batch: int | None = None,
    bytes_per_point: int = 160,
) -> tuple[dict, dict, list[float | None]]:
    """
    Per-pair depth stats, the residual histogram and per-frame agreement, in one O(N^2) pass.

    - returns (depth_pairs, histogram, agreement); agreement is one share per frame, None when nothing is seen
    - depth_pairs: columnar {idx1, idx2, n_pixels, median_rel_depth_error, iqr_rel_depth_error,
      median_parallax_deg, median_depth}, one row per ordered direction, (d_sampled - d_expected) / d_expected
    - histogram: {counts, bin_edges} of pooled pixel residuals, only where seen (occlusion is not disagreement)
    - target views run target_batch at a time; None sizes it from total VRAM at bytes_per_point
    - min_pair_overlap > 0 skips targets whose frustum holds less than that share of frame i's points
    - overlap comes from an overlap_stride pixel subsample; 0 skips the pre-pass entirely
    """
    # Depth, cameras and camera centers on the device
    n, h, w = depth.shape
    device = torch.device(get_device())
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    cam_to_world = invert_poses(extrinsics)
    centers = torch.as_tensor(cam_to_world[:, :3, 3], dtype=torch.float32, device=device)

    # One histogram over every ordered pair's residuals, n*(n-1)*h*w at most, plus a NaN overflow bin
    k = 2 * max(1, int(round(max(n * (n - 1) * h * w, 1) ** (1.0 / 3.0))))
    edges = np.linspace(-1.0, 1.0, k + 1)
    edges_t = torch.as_tensor(edges, device=device)
    counts = torch.zeros(k + 1, dtype=torch.int64, device=device)
    cols: dict[str, list] = {
        "idx1": [],
        "idx2": [],
        "n_pixels": [],
        "median_rel_depth_error": [],
        "iqr_rel_depth_error": [],
        "median_parallax_deg": [],
        "median_depth": [],
    }
    agreement = []

    # Pair quantile levels, on the device so quantile never transfers
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)

    # Target views per batch: VRAM-sized unless given, never more than the other frames
    if target_batch is None:
        target_batch = infer_batch_size(h * w * bytes_per_point / 1024**3)

    target_batch = max(1, min(target_batch, n - 1))
    frame_ids = torch.arange(n, device=device)

    # Progress-bar label counting every ordered pair direction
    desc = f"Cross-view depth check ({n * (n - 1)} pair directions)"

    # Every source frame against every other frame, one bar step per source
    for i in tqdm(range(n), desc=desc, unit="frame", leave=False):
        # Frame i's world points and its per-pixel agreement accumulators
        points = unproject(depth_t[i], extrinsics_t[i], intrinsics_t[i]).reshape(-1, 3)
        has_source = depth_t[i].reshape(-1) > 0
        any_agree = torch.zeros_like(has_source)
        any_seen = torch.zeros_like(has_source)
        rows: list[Tensor] = []
        row_keys: list[tuple[int, int]] = []

        # Every other frame as a target, indexed on the device: a host index list syncs
        targets = [j for j in range(n) if j != i]
        targets_t = torch.cat([frame_ids[:i], frame_ids[i + 1 :]])

        # Optional pruning: drop targets whose frustum barely holds frame i's subsampled points
        if min_pair_overlap > 0:
            sub = points.reshape(h, w, 3)[::overlap_stride, ::overlap_stride].reshape(-1, 3)
            sub = sub[has_source.reshape(h, w)[::overlap_stride, ::overlap_stride].reshape(-1)]

            # Share of points in front of and inside each camera's frustum; occlusion ignored, so it over-counts
            if len(sub) == 0:
                overlap = [0.0] * n
            else:
                pixels, points_cam = project(sub, extrinsics_t, intrinsics_t)
                u, v = pixels[..., 0], pixels[..., 1]
                inside = (points_cam[..., 2] > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
                overlap = inside.float().mean(-1).tolist()

            targets = [j for j in targets if overlap[j] >= min_pair_overlap]
            targets_t = torch.as_tensor(targets, dtype=torch.long, device=device)

        # Depth agreement against the targets, target_batch views at a time
        for start in range(0, len(targets), target_batch):
            js = targets[start : start + target_batch]
            js_t = targets_t[start : start + target_batch]
            agree, seen, rel, z = depth_agreement(
                points, extrinsics_t[js_t], intrinsics_t[js_t], depth_t[js_t], rel_thresh
            )
            seen &= has_source
            any_agree |= (agree & has_source).any(0)
            any_seen |= seen.any(0)

            # Residuals over seen pixels with a sampled depth (rel > -1), clear of camera j's center
            sel = seen & (rel > -1) & (z > 1e-6)

            # The batch's two GPU syncs: per-view counts, then row-major indices, ascending per view
            n_sel_t = sel.sum(1)
            n_sel = n_sel_t.tolist()
            view, idx = sel.nonzero().unbind(1)

            if not idx.numel():
                continue

            # Selected residuals, depths and target centers
            rel_sel = rel[view, idx]
            z_sel = z[view, idx]
            centers_b = centers[js_t]

            # Parallax from ray directions: scale-free, needs no focal length
            ray_i = points[idx] - centers[i]
            ray_j = points[idx] - centers_b[view]
            cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
            parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # Residuals mapped into (-1, 1) by r / (1 + |r|), monotone and so quantile-preserving, then binned
            r = rel_sel.double()
            bounded = r / (1.0 + r.abs())
            bins = (torch.bucketize(bounded, edges_t, right=True) - 1).clamp_(0, k - 1)

            # NaN (an inf depth's residual) goes to the overflow bin k, which the output drops
            bins = torch.where(bounded.isnan(), k, bins)
            counts.index_add_(0, bins, torch.ones_like(bins))

            # Sort each quantity within its view's segment: by value, then stably by view
            seg_sorted = []

            for vals in (rel_sel, parallax, z_sel):
                val, p = vals.sort(stable=True)
                seg_sorted.append(val[view[p].sort(stable=True).indices])

            rel_s, par_s, z_s = seg_sorted

            # Quartiles by torch.quantile's linear rule, medians at torch.median's lower middle
            last = len(idx) - 1
            starts = torch.cumsum(n_sel_t, 0) - n_sel_t
            span = (n_sel_t - 1).clamp(min=0)
            ranks = quantiles[None, :] * span[:, None]
            below = (starts[:, None] + ranks.long()).clamp(max=last)
            above = (starts[:, None] + ranks.ceil().long()).clamp(max=last)
            q = torch.lerp(rel_s[below], rel_s[above], ranks - ranks.long())
            mid = (starts + span // 2).clamp(max=last)

            # One row per view, empty views included as clamped dummies the host drops
            rows.append(torch.stack([q[:, 1], q[:, 2] - q[:, 0], par_s[mid], z_s[mid]], 1))
            row_keys.extend(zip(js, n_sel))

        # One transfer per source frame for its pair rows and its agreement counts
        if rows:
            stats = torch.cat(rows).tolist()

            for (j, n_px), (q50, iqr, par, med_z) in zip(row_keys, stats):
                if n_px == 0:
                    continue

                for col, v in zip(cols, (i, j, n_px, q50, iqr, par, med_z)):
                    cols[col].append(v)

        # Share of this frame's seen pixels that any other view agrees with
        n_agree, n_seen = torch.stack([any_agree.sum(), any_seen.sum()]).tolist()
        agreement.append(n_agree / n_seen if n_seen else None)

    # Log the pair count; the pooled histogram crosses to the host once, overflow bin dropped
    logger.info("Depth error: %d pair directions", len(cols["idx1"]))
    histogram = {"counts": counts[:-1].cpu().numpy().tolist(), "bin_edges": edges.tolist()}
    return cols, histogram, agreement


########################################################################
# Per pair: photometric
########################################################################


def compute_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray | None = None,
    separations: tuple[int, ...] = (1, 2, 5, 10, 20),
    min_samples: int = 32,
) -> dict:
    """
    Warp each frame into the frames at fixed index gaps through pose + depth and correlate the RGB.

    - zero-mean Pearson NCC in float64: 1.0 is perfect agreement, 0.0 is none
    - normalizing cancels the [0, 255] vs [0, 1] image-scale split and exposure or gain change
    - the only appearance metric: disagreement seen only here points at image formation
    - original resolution; model-grid depth is upsampled one frame at a time, only frames with a partner
    - rows are unordered pairs (i < j)
    - float32 camera-local warp on get_device(), float64 relative poses

    Args:
        images: (N, H, W, 3) RGB at original resolution; uint8 [0, 255], or float [0, 255] or [0, 1].
        depth: (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics: (N, 3, 3) K on the images' pixel grid.
        extrinsics: (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        separations: index gaps j - i scored per frame; wide gaps expose pose drift that
            neighbors hide, and the cost stays O(N * len(separations)) rather than O(N^2).
        min_samples: floor on overlapping pixels (the `n_pixels` column); Pearson NCC on two
            values is exactly ±1 whatever they are.

    Returns:
        {idx1, idx2, photometric_ncc, n_pixels}, columnar; empty lists when no pair correlates.

    Raises:
        ValueError: depth needs upsampling but original_coords is missing or describes
            another resolution.
    """
    # Start the timer and log the pass size
    t0 = time.perf_counter()
    N = len(depth)
    H, W = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, separations=%s", N, W, H, separations)

    # Model-grid depth is lifted to the image grid frame by frame below
    lift = depth.shape[1:] != (H, W)

    if lift:
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(H, W)}; original_coords is required to upsample"
            )

        # Crop boxes are in original pixels, so `images` must be that canvas
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))

        if (H, W) != expected_hw:
            raise ValueError(
                f"images are {(H, W)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )

        # Float guide scale decided once over the whole array; per frame would amplify a dark frame 255x
        rgb_scale = 255.0 if images.dtype != np.uint8 and images.max() <= 1.0 else 1.0

    # Poses stay float64 on the host; only camera-local quantities go to the device in float32
    device = torch.device(get_device())
    world_to_cam = np.asarray(extrinsics, dtype=np.float64)
    cam_to_world = invert_poses(world_to_cam)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)

    # Identity pose, so frame i's depth unprojects into its own camera frame
    identity = torch.eye(4, dtype=torch.float32, device=device)

    # Closed-form pair count, so the bar states the real unit of work, and the columnar output
    gaps = sorted(set(separations))
    n_pairs_expected = sum(max(N - g, 0) for g in gaps)
    cols: dict[str, list] = {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}

    # Each frame against the frames at its forward gaps
    for i in tqdm(range(N), desc=f"Photometric NCC ({n_pairs_expected} pairs)", unit="frame"):
        # Forward partners inside the sequence; the last frame has none
        partners = [i + g for g in gaps if i + g < N]

        if not partners:
            continue

        # Frame i's depth on the image grid; the guide must be uint8 [0, 255]
        depth_i = depth[i]

        if lift:
            guide = np.asarray(images[i])

            # Float frames are scaled and clipped to uint8; uint8 frames are the guide as-is
            if guide.dtype != np.uint8:
                guide = np.clip(guide * rgb_scale, 0, 255).astype(np.uint8)

            depth_i = upsample_depths(depth[i : i + 1], guide[None], original_coords[i : i + 1, :4])[0]

        # Camera-frame points; world points far from the origin would lose float32 precision
        depth_t = torch.as_tensor(np.asarray(depth_i), dtype=torch.float32, device=device)

        # unproject's matmul would run in TF32 under a global allow_tf32
        with full_fp32_matmul():
            pts_cam_i = unproject(depth_t, identity, intrinsics_t[i]).reshape(-1, 3)

        # Pixels with a valid depth sample
        has_depth = depth_t.reshape(-1) > 0

        # Camera i -> camera j for every partner, composed in float64 then uploaded once
        rel_poses = world_to_cam[partners] @ cam_to_world[i]
        rel_poses = torch.as_tensor(rel_poses, dtype=torch.float32, device=device)

        # Frame i and its partners uploaded once per source frame, in their own dtype
        window = torch.tensor(np.asarray(images[[i, *partners]]), device=device)
        colors_i = window[0].reshape(-1, 3)

        for k, j in enumerate(partners):
            # Project camera i's points into frame j; no occlusion test
            pixels, pts_cam_j = project(pts_cam_i, rel_poses[k], intrinsics_t[j])

            # Nearest sampling, matching the depth pass; round half to even like np.round
            ui = torch.round(pixels[:, 0]).long()
            vi = torch.round(pixels[:, 1]).long()
            ok = (pts_cam_j[:, 2] > 0) & has_depth
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idx = ok.nonzero().squeeze(1)
            n_ok = idx.numel()

            if n_ok < min_samples:
                continue

            # Paired RGB samples: frame i's pixel against where it landed in frame j
            a = colors_i[idx].reshape(-1).double()
            b = window[k + 1][vi[idx], ui[idx]].reshape(-1).double()

            # Float64 Pearson NCC on the device; one host transfer per pair below
            a0, b0 = a - a.mean(), b - b.mean()
            std_a, std_b = a0.square().mean().sqrt(), b0.square().mean().sqrt()
            ncc_t = ((a0 * b0).mean() / (std_a * std_b)).clamp(-1.0, 1.0)

            # Both stds and the NCC come back to the host in one transfer
            std_a, std_b, ncc = torch.stack([std_a, std_b, ncc_t]).tolist()

            # Skip flat patches: no variance to correlate
            if std_a < 1e-8 or std_b < 1e-8 or not np.isfinite(ncc):
                continue

            # Append the pair row
            for col, v in zip(cols, (i, j, ncc, n_ok)):
                cols[col].append(v)

    logger.info("Photometric NCC: %d pairs correlated in %.2fs", len(cols["photometric_ncc"]), time.perf_counter() - t0)
    return cols


########################################################################
# Per frame + assembly
########################################################################


def compute_reconstruction_quality(
    depth: np.ndarray,
    model_intrinsics: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray,
    image_names: list[str],
    confidence: np.ndarray | None,
    images: np.ndarray | None,
    rel_thresh: float = 0.05,
    min_pair_overlap: float = 0.0,
) -> dict:
    """
    Every table the report holds, from arrays; the Reconstructor stage owns all IO.

    - a missing optional input nulls exactly its table or columns; a failing measurement raises
    - column meanings: docs/source/api/geometry.rst

    Args:
        depth: (N, h, w) Z-depth on the model grid.
        model_intrinsics: (N, 3, 3) K on the model grid.
        intrinsics: (N, 3, 3) K on the original images' pixel grid.
        extrinsics: (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows: x0, y0, x1, y1, original width, original height.
        image_names: per-frame image file names, reconstruction order.
        confidence: (N, h, w) per-pixel confidence, or None.
        images: (M, H, W, 3) original-resolution RGB, M <= N, or None.
        rel_thresh: cross-view occlusion and agreement band, as a fraction of depth; looser than
            the creators' filter so disagreement stays in the stats.
        min_pair_overlap: skip pairs whose frustum overlap is below this share; 0 keeps every pair.

    Returns:
        {"frames", "depth_pairs", "depth_residual_histogram", "photometric_pairs"}
        - frames.multiview_agreement: per frame, None when no other view sees it
        - photometric_pairs None without images
    """
    # Frame count and float64 crop rows
    n = len(depth)
    coords = np.asarray(original_coords, dtype=np.float64)

    # One cross-view pass feeds the pair tables and the per-frame agreement
    depth_pairs, histogram, agreement = _collect_pairs(
        depth, model_intrinsics, extrinsics, rel_thresh, min_pair_overlap=min_pair_overlap
    )

    # Photometric over the frames that have images; read_frames order is reconstruction order
    photometric_pairs = None

    if images is not None:
        m = len(images)
        photometric_pairs = compute_photometric_ncc(
            images, depth[:m], intrinsics[:m], extrinsics[:m], original_coords=original_coords[:m]
        )

    # Per-frame median |residual| over the depth pairs touching each frame; None when none touch
    touching: list[list[float]] = [[] for _ in range(n)]

    for i1, i2, err in zip(depth_pairs["idx1"], depth_pairs["idx2"], depth_pairs["median_rel_depth_error"]):
        touching[i1].append(abs(err))
        touching[i2].append(abs(err))

    median_abs = [float(np.median(v)) if v else None for v in touching]

    # Source frame index only where the stem is on the frame_{idx:06d} contract
    frame_idx = [
        frames.frame_idx_from_path(name) if _FRAME_STEM_RE.fullmatch(Path(name).stem) else None for name in image_names
    ]

    # Fraction of each original frame the model crop kept; no model-grid table sees a cropped band
    covered = [float(max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9)) for c in coords]

    # Per-frame table, then the full report
    frames_table = {
        "frame_idx": frame_idx,
        "covered_fraction": covered,
        "median_abs_rel_depth_error": median_abs,
        "multiview_agreement": agreement,
        "confidence_median": [None] * n if confidence is None else [float(np.median(c)) for c in confidence],
    }
    return {
        "frames": frames_table,
        "depth_pairs": depth_pairs,
        "depth_residual_histogram": histogram,
        "photometric_pairs": photometric_pairs,
    }
