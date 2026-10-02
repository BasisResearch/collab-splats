"""
Report-only, scale-free cross-view quality, no ground truth; columns: docs/source/api/geometry.rst

- depth_pairs: depth error where two views overlap, n_pixels shared; plus a pooled residual histogram
- photometric_pairs: NCC of pixel colors after warping one view into another through its depth
- frames.median_abs_rel_depth_error: per frame, median |depth error| over the pairs touching it
- frames.multiview_agreement: per frame, share of seen pixels another view agrees with
- frames.covered_fraction: share of each original frame that survived the model's crop
- frames.confidence_median: median backbone confidence; not comparable across backbones
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from collab_splats.geometry.projection import depth_agreement, project, unproject
from collab_splats.geometry.transforms import invert_poses
from collab_splats.preproc import frames
from collab_splats.utils.image import upsample_depths
from collab_splats.utils.torch_utils import get_device, infer_batch_size

logger = logging.getLogger(__name__)

########################################################################
# Constants
########################################################################

# Keyframe stem contract (frame_{idx:06d}) guarding the source-frame join
# - frame_idx_from_path reads any numeric tail: IMG_1234 -> 1234, 00019 -> 19
# - a wrong index silently pairs real frames with the wrong rows, worse than none
# - six or more digits, so the contract holds past a million frames
_FRAME_STEM_RE = re.compile(r"frame_\d{6,}$")

########################################################################
# Results
########################################################################


@dataclass
class PairStats:
    """
    Depth agreement of one ordered frame pair, one depth_pairs row.

    - filled by the report's cross-view depth pass, _collect_pairs
    - depth error quantity: (d_sampled - d_expected) / d_expected, frame j against frame i
    - frame separation is abs(idx1 - idx2), not a field

    Args:
        idx1: first frame index.
        idx2: second frame index.
        n_pixels: pixels the depth pass compared.
        median_rel_depth_error: signed scale offset s - 1; 0.02 means frame j is 2% deeper.
        iqr_rel_depth_error: spread with the bias removed, i.e. geometric noise.
        median_parallax_deg: how well this pair can see depth at all.
        median_depth: the "worse further away?" axis, as a column.
    """

    idx1: int
    idx2: int
    n_pixels: int | None = None
    median_rel_depth_error: float | None = None
    iqr_rel_depth_error: float | None = None
    median_parallax_deg: float | None = None
    median_depth: float | None = None


########################################################################
# Shared: residual axis
########################################################################


def residual_bin_edges(n_samples: int) -> np.ndarray:
    """
    Histogram edges for n_samples bounded residuals, fixed before any residual is seen.

    - per-pixel residuals number N^2*H*W: too many to hold, and a range pre-pass would double the cost
    - range (-1, 1) by construction: bounded_residual maps every residual into it, none clip
    - bin count from Rice's rule, k = 2 * n**(1/3): more pixels justify finer bins
    - below roughly 20 frames the pixel-level bins go coarse; per-pair medians are unaffected
    - the bin count is always even, so the histogram folds to |r| by adding the two halves

    Args:
        n_samples: residual count the histogram will hold.

    Returns:
        k + 1 evenly spaced edges over [-1, 1], k even.
    """
    k = 2 * max(1, int(round(max(int(n_samples), 1) ** (1.0 / 3.0))))
    return np.linspace(-1.0, 1.0, k + 1)


def bounded_residual(rel: np.ndarray | Tensor | float) -> np.ndarray | Tensor:
    """
    Map a relative depth residual onto (-1, 1) so a fixed histogram can never miss it.

    - r / (1 + |r|) is monotone, so quantiles survive the map exactly
    - invert with u / (1 - |u|)
    - no chosen range: clipping piles the tail into end bins, and np.histogram drops out-of-range values
    - a tensor maps in float64 on its own device, bit-identical to the numpy path

    Args:
        rel: relative depth residuals, any shape; numpy, scalar or torch.

    Returns:
        Float64 array, or float64 tensor for a tensor input, of the same shape, in (-1, 1).
    """
    if isinstance(rel, Tensor):
        r = rel.double()
        return r / (1.0 + r.abs())

    r = np.asarray(rel, dtype=np.float64)
    return r / (1.0 + np.abs(r))


def _histogram_counts(u: Tensor, edges: Tensor) -> Tensor:
    """
    np.histogram(u, bins=edges)[0] on u's device, plus one overflow bin for NaN.

    - float64 edges: float32 histc misbins against these edges
    - right=True and the clamp close both outer bins, as np.histogram does
    - NaN (an inf depth's residual) lands in bin k, which callers drop as np.histogram drops NaN
    - returns k + 1 counts; slice [:k] for the histogram
    - index_add_, not bincount: bincount syncs on CUDA, integer atomics are exact
    """
    k = len(edges) - 1
    idx = (torch.bucketize(u, edges, right=True) - 1).clamp_(0, k - 1)
    idx = torch.where(u.isnan(), k, idx)
    counts = torch.zeros(k + 1, dtype=torch.int64, device=u.device)
    return counts.index_add_(0, idx, torch.ones_like(idx))


########################################################################
# Per pair: depth
########################################################################


def _frustum_overlap(points: Tensor, extrinsics: Tensor, intrinsics: Tensor, h: int, w: int) -> Tensor:
    """
    Share of points inside each camera's frustum, one batched projection into all N cameras.

    - in front and inside the pixel-center span [0, w-1] x [0, h-1], as depth_residual bounds it
    - ignores occlusion, so it over-estimates overlap: pruning on it keeps extra pairs
    - (N,) float; 0 for an empty point set
    """
    if len(points) == 0:
        return torch.zeros(len(extrinsics), device=points.device)

    pixels, points_cam = project(points, extrinsics, intrinsics)
    u, v = pixels[..., 0], pixels[..., 1]
    inside = (points_cam[..., 2] > 0) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    return inside.float().mean(-1)


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
) -> tuple[dict, list[float | None]]:
    """
    Per-pair depth stats, the residual histogram and per-frame agreement, in one O(N^2) pass.

    - collected: "pairs" (list[PairStats]), "rel_depth_error_counts", "rel_depth_error_edges"
    - agreement: share of seen pixels with at least one agreeing view; None when nothing is seen
    - residuals only where seen: occlusion is absent evidence, not disagreement
    - target views run target_batch at a time; None sizes it from total VRAM (infer_batch_size, not free) at bytes_per_point
    - min_pair_overlap > 0 skips targets whose frustum holds less than that share of frame i's points
    - overlap comes from an overlap_stride pixel subsample; min_pair_overlap == 0 skips the pre-pass entirely
    """
    n, h, w = depth.shape
    device = torch.device(get_device())
    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    extrinsics_t = torch.as_tensor(extrinsics, dtype=torch.float32, device=device)
    cam_to_world = invert_poses(extrinsics)
    centers = torch.as_tensor(cam_to_world[:, :3, 3], dtype=torch.float32, device=device)

    # One histogram over every ordered pair's residuals, n*(n-1)*h*w at most, plus a NaN overflow bin
    edges = residual_bin_edges(n * (n - 1) * h * w)
    edges_t = torch.as_tensor(edges, device=device)
    counts = torch.zeros(len(edges), dtype=torch.int64, device=device)
    collected = {"pairs": [], "rel_depth_error_edges": edges}
    agreement = []

    # Pair quantile levels, on the device so quantile never transfers
    quantiles = torch.tensor([0.25, 0.5, 0.75], device=device)

    # Target views per batch: VRAM-sized unless given, never more than the other frames
    if target_batch is None:
        target_batch = infer_batch_size(h * w * bytes_per_point / 1024**3)

    target_batch = max(1, min(target_batch, n - 1))
    frame_ids = torch.arange(n, device=device)

    # Every source frame against every other frame, one bar step per source
    desc = f"Cross-view depth check ({n * (n - 1)} pair directions)"
    for i in tqdm(range(n), desc=desc, unit="frame", leave=False):
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
            overlap = _frustum_overlap(sub, extrinsics_t, intrinsics_t, h, w).tolist()
            targets = [j for j in targets if overlap[j] >= min_pair_overlap]
            targets_t = torch.as_tensor(targets, dtype=torch.long, device=device)

        for start in range(0, len(targets), target_batch):
            js = targets[start : start + target_batch]
            js_t = targets_t[start : start + target_batch]
            agree, seen, rel, z = depth_agreement(
                points, extrinsics_t[js_t], intrinsics_t[js_t], depth_t[js_t], rel_thresh
            )
            seen &= has_source
            any_agree |= (agree & has_source).any(0)
            any_seen |= seen.any(0)

            # Residuals over seen pixels with a sampled depth, clear of camera j's center
            # - rel > -1 keeps sampled > 0 only: sampled <= 0 is no measurement
            # - near-zero z: the quotient and the parallax both degenerate
            sel = seen & (rel > -1) & (z > 1e-6)

            # The batch's two GPU syncs: per-view counts, then row-major indices, ascending per view
            n_sel_t = sel.sum(1)
            n_sel = n_sel_t.tolist()
            view, idx = sel.nonzero().unbind(1)
            if not idx.numel():
                continue

            rel_sel = rel[view, idx]
            z_sel = z[view, idx]
            centers_b = centers[js_t]

            # Parallax from ray directions: scale-free, needs no focal length
            ray_i = points[idx] - centers[i]
            ray_j = points[idx] - centers_b[view]
            cos_a = (ray_i * ray_j).sum(-1) / (ray_i.norm(dim=-1) * ray_j.norm(dim=-1)).clamp(min=1e-12)
            parallax = torch.rad2deg(torch.arccos(cos_a.clamp(-1.0, 1.0)))

            # Per-pixel residuals into the one device histogram
            counts += _histogram_counts(bounded_residual(rel_sel), edges_t)

            # Sort each quantity within its view's segment: by value, then stably by view
            seg_sorted = []
            for vals in (rel_sel, parallax, z_sel):
                v, p = vals.sort(stable=True)
                seg_sorted.append(v[view[p].sort(stable=True).indices])

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

                collected["pairs"].append(
                    PairStats(
                        idx1=i,
                        idx2=j,
                        n_pixels=n_px,
                        median_rel_depth_error=q50,
                        iqr_rel_depth_error=iqr,
                        median_parallax_deg=par,
                        median_depth=med_z,
                    )
                )

        # Share of this frame's seen pixels that any other view agrees with
        n_agree, n_seen = torch.stack([any_agree.sum(), any_seen.sum()]).tolist()
        agreement.append(n_agree / n_seen if n_seen else None)

    # The pooled histogram crosses to the host once
    collected["rel_depth_error_counts"] = counts[:-1].cpu().numpy()
    return collected, agreement


def compute_depth_error(collected: dict) -> tuple[dict, dict]:
    """
    Per-direction depth disagreement and the pixel residual histogram, both columnar.

    - model resolution: original resolution would sample guided-filtered depth instead
    - rows are ordered pair directions: (i, j) and (j, i) differ, occlusion is asymmetric

    Args:
        collected: the dict _collect_pairs filled.

    Returns:
        (depth_pairs, depth_residual_histogram)
        - depth_pairs: {idx1, idx2, n_pixels, median_rel_depth_error, iqr_rel_depth_error,
          median_parallax_deg, median_depth}, one entry per pair direction
        - depth_residual_histogram: {counts, bin_edges}
    """
    pairs = collected["pairs"]
    logger.info("Depth error: %d pair directions", len(pairs))

    # One entry per pair direction in every column, raw for the reader to bin
    depth_pairs = {
        "idx1": [p.idx1 for p in pairs],
        "idx2": [p.idx2 for p in pairs],
        "n_pixels": [p.n_pixels for p in pairs],
        "median_rel_depth_error": [p.median_rel_depth_error for p in pairs],
        "iqr_rel_depth_error": [p.iqr_rel_depth_error for p in pairs],
        "median_parallax_deg": [p.median_parallax_deg for p in pairs],
        "median_depth": [p.median_depth for p in pairs],
    }

    # The one pre-binned output, for the one per-pixel quantity
    histogram = {
        "counts": collected["rel_depth_error_counts"].tolist(),
        "bin_edges": collected["rel_depth_error_edges"].tolist(),
    }
    return depth_pairs, histogram


########################################################################
# Per pair: photometric
########################################################################


def compute_photometric_ncc(
    images: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    original_coords: np.ndarray | None = None,
    max_separation: int = 2,
    min_samples: int = 32,
) -> dict:
    """
    Warp each frame into its neighbors through pose + depth and correlate the RGB.

    - zero-mean Pearson NCC in float64: 1.0 is perfect agreement, 0.0 is none
    - normalizing cancels the [0, 255] vs [0, 1] image-scale split and exposure or gain change
    - the only appearance metric: disagreement seen only here points at image formation
    - original resolution; model-grid depth is upsampled one frame at a time, only frames with a partner
    - rows are unordered pairs (i < j)
    - float32 camera-local warp on get_device(), no matmul (TF32-safe), float64 relative poses

    Args:
        images: (N, H, W, 3) RGB at original resolution; uint8 [0, 255], or float [0, 255] or [0, 1].
        depth: (N, h, w) Z-depth on the model grid, or on the image grid already.
        intrinsics: (N, 3, 3) K on the images' pixel grid.
        extrinsics: (N, 4, 4) world-to-cam.
        original_coords: (N, 6) crop rows, required only when depth needs upsampling.
        max_separation: pairs per frame; distant frames differ mostly by lighting and
            viewpoint, so the cost stays O(N * max_separation) rather than O(N^2).
        min_samples: floor on overlapping pixels (the `n_pixels` column); Pearson NCC on two
            values is exactly ±1 whatever they are.

    Returns:
        {idx1, idx2, photometric_ncc, n_pixels}, columnar; empty lists when no pair correlates.

    Raises:
        ValueError: depth needs upsampling but original_coords is missing or describes
            another resolution.
    """
    t0 = time.perf_counter()
    N = len(depth)
    ih, iw = images.shape[1:3]
    logger.info("Photometric NCC: %d frames at %dx%d, max_separation=%d", N, iw, ih, max_separation)

    # Model-grid depth is lifted to the image grid frame by frame below
    lift = depth.shape[1:] != (ih, iw)

    if lift:
        if original_coords is None:
            raise ValueError(
                f"depth is {depth.shape[1:]} but images are {(ih, iw)}; " "original_coords is required to upsample"
            )

        # Crop boxes are in original pixels, so `images` must be that canvas
        expected_hw = (int(original_coords[0, 5]), int(original_coords[0, 4]))

        if (ih, iw) != expected_hw:
            raise ValueError(
                f"images are {(ih, iw)} but original_coords say the original resolution is "
                f"{expected_hw} — they are from different preprocessing runs."
            )

        # Float guide scale decided once over the whole array; per frame would amplify a dark frame 255x
        rgb_scale = 255.0 if images.dtype != np.uint8 and images.max() <= 1.0 else 1.0

    # Poses stay float64 on the host; only camera-local quantities go to the device in float32
    device = torch.device(get_device())
    world_to_cam = np.asarray(extrinsics, dtype=np.float64)
    cam_to_world = invert_poses(world_to_cam)
    intrinsics_t = torch.as_tensor(np.asarray(intrinsics), dtype=torch.float32, device=device)
    H, W = ih, iw

    # Flattened pixel grid of the image canvas, shared by every source frame
    grid_v, grid_u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_u, grid_v = grid_u.reshape(-1), grid_v.reshape(-1)

    # Closed-form pair count, so the bar states the real unit of work
    n_pairs_expected = sum(min(N, i + max_separation + 1) - (i + 1) for i in range(N))
    cols: dict[str, list] = {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}

    for i in tqdm(range(N), desc=f"Photometric NCC ({n_pairs_expected} pairs)", unit="frame"):
        j_end = min(N, i + max_separation + 1)

        if j_end == i + 1:
            continue

        # Frame i's depth on the image grid; the guide must be uint8 [0, 255]
        depth_i = depth[i]

        if lift:
            guide = np.asarray(images[i])

            # Float frames are scaled and clipped to uint8; uint8 frames are the guide as-is
            if guide.dtype != np.uint8:
                guide = np.clip(guide * rgb_scale, 0, 255).astype(np.uint8)

            depth_i = upsample_depths(depth[i : i + 1], guide[None], original_coords[i : i + 1, :4])[0]

        # Camera-frame points from K-rays times depth; world points far from the origin lose float32 precision
        depth_t = torch.as_tensor(np.asarray(depth_i), dtype=torch.float32, device=device)
        z_i = depth_t.reshape(-1)
        fx, fy, cx, cy = intrinsics_t[i, 0, 0], intrinsics_t[i, 1, 1], intrinsics_t[i, 0, 2], intrinsics_t[i, 1, 2]
        pts_cam_i = torch.stack([(grid_u - cx) / fx * z_i, (grid_v - cy) / fy * z_i, z_i], dim=-1)
        has_depth = z_i > 0

        # Camera i -> camera j for every partner, composed in float64 then uploaded once
        rel_poses = world_to_cam[i + 1 : j_end] @ cam_to_world[i]
        rel_poses = torch.as_tensor(rel_poses, dtype=torch.float32, device=device)

        # Frame i and its partners uploaded once per source frame, in their own dtype
        window = torch.tensor(np.asarray(images[i:j_end]), device=device)
        colors_i = window[0].reshape(-1, 3)

        for j in range(i + 1, j_end):
            # Move into camera j elementwise: a CUDA matmul under global TF32 truncates to 10 mantissa bits
            rel = rel_poses[j - i - 1]
            pts_cam_j = (pts_cam_i[:, None, :] * rel[:3, :3]).sum(-1) + rel[:3, 3]

            # Project into frame j with project()'s clamped divide; no occlusion test
            z_j = pts_cam_j[:, 2].clamp(min=1e-6)
            u_j = pts_cam_j[:, 0] * intrinsics_t[j, 0, 0] / z_j + intrinsics_t[j, 0, 2]
            v_j = pts_cam_j[:, 1] * intrinsics_t[j, 1, 1] / z_j + intrinsics_t[j, 1, 2]

            # Nearest sampling, matching the depth pass; round half to even like np.round
            ui = torch.round(u_j).long()
            vi = torch.round(v_j).long()
            ok = (pts_cam_j[:, 2] > 0) & has_depth
            ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idx = ok.nonzero().squeeze(1)
            n_ok = idx.numel()

            if n_ok < min_samples:
                continue

            # Paired RGB samples: frame i's pixel against where it landed in frame j
            a = colors_i[idx].reshape(-1).double()
            b = window[j - i][vi[idx], ui[idx]].reshape(-1).double()

            # Float64 Pearson NCC on the device; one host transfer per pair below
            a0, b0 = a - a.mean(), b - b.mean()
            std_a, std_b = a0.square().mean().sqrt(), b0.square().mean().sqrt()
            ncc_t = ((a0 * b0).mean() / (std_a * std_b)).clamp(-1.0, 1.0)

            # Both stds and the NCC come back to the host in one transfer
            std_a, std_b, ncc = torch.stack([std_a, std_b, ncc_t]).tolist()

            # Skip flat patches: no variance to correlate
            if std_a < 1e-8 or std_b < 1e-8 or not np.isfinite(ncc):
                continue

            for col, v in zip(cols, (i, j, ncc, n_ok)):
                cols[col].append(v)

    logger.info(
        "Photometric NCC: %d pairs correlated in %.2fs", len(cols["photometric_ncc"]), time.perf_counter() - t0
    )
    return cols


########################################################################
# Per frame + assembly
########################################################################


def _frame_medians(pairs: list[PairStats], n: int) -> list[float | None]:
    """
    Per frame, the median |median_rel_depth_error| over the pairs touching it.

    - one pass buckets every pair under both its frames: O(pairs), not O(N * pairs)
    - None for a frame no pair touches
    """
    touching: list[list[float]] = [[] for _ in range(n)]

    for p in pairs:
        v = abs(p.median_rel_depth_error)
        touching[p.idx1].append(v)
        touching[p.idx2].append(v)

    return [float(np.median(v)) if v else None for v in touching]


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
    n = len(depth)
    coords = np.asarray(original_coords, dtype=np.float64)

    # One cross-view pass feeds the pair tables and the per-frame agreement
    collected, agreement = _collect_pairs(
        depth, model_intrinsics, extrinsics, rel_thresh, min_pair_overlap=min_pair_overlap
    )
    depth_pairs, histogram = compute_depth_error(collected)

    # Photometric over the frames that have images; read_frames order is reconstruction order
    photometric_pairs = None
    if images is not None:
        m = len(images)
        photometric_pairs = compute_photometric_ncc(
            images, depth[:m], intrinsics[:m], extrinsics[:m], original_coords=original_coords[:m]
        )

    # Per-frame median |residual| over the depth pairs touching each frame
    median_abs = _frame_medians(collected["pairs"], n)

    # Source frame index only where the stem is on the frame_{idx:06d} contract
    frame_idx = [
        frames.frame_idx_from_path(name) if _FRAME_STEM_RE.fullmatch(Path(name).stem) else None
        for name in image_names
    ]

    # Fraction of each original frame the model crop reconstructed
    # - a center crop of a wide source loses a band no model-grid table can see
    covered = [float(max(c[2] - c[0], 0) * max(c[3] - c[1], 0) / max(c[4] * c[5], 1e-9)) for c in coords]

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
