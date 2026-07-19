"""Pose-graph output merging and overlap deduplication.

Reconstructs per-frame trajectories from per-submap corrected poses
(dedup_overlap) and assembles the unified raw_outputs dict (merge_submap_outputs).
"""

from __future__ import annotations

import numpy as np

from .submap import Submap


def dedup_overlap(
    submap_ids: list[int],
    submap_starts: list[int],
    corrected: dict[int, np.ndarray],
    total_frames: int,
) -> np.ndarray:
    """Reconstruct (total_frames, 4, 4) from per-submap corrected poses, deduplicating overlap.

    First-writer-wins: the overlap frame belongs to two adjacent submaps; we keep
    submap-0's estimate (processed first). VGGT-SLAM does NOT dedup — its
    write_poses_to_file (map.py:142-162) emits every submap's frames, so the
    shared overlap frame appears twice in its TUM (a duplicate timestamp). evo
    associates by timestamp and keeps the first occurrence — also submap-0's — so
    the two pipelines agree on the boundary pose despite SLAM's duplicate row.
    """
    out = np.tile(np.eye(4, dtype=np.float32), (total_frames, 1, 1))
    assigned = np.zeros(total_frames, dtype=bool)
    for sid, start in zip(submap_ids, submap_starts):
        poses = corrected[sid]  # (K_i, 4, 4)
        k = poses.shape[0]
        for local_i in range(k):
            global_i = start + local_i
            if 0 <= global_i < total_frames and not assigned[global_i]:
                out[global_i] = poses[local_i]
                assigned[global_i] = True
    return out


def _resolve_frame_node(
    frame_to_node: dict[tuple[int, int], int],
    submaps: list[Submap],
    image_path,
) -> tuple[int, Submap, int] | None:
    """Resolve an image path to (node_id, submap, local frame index)."""
    for submap in submaps:
        for local_i, p in enumerate(submap.image_paths):
            if p == image_path:
                nid = frame_to_node.get((submap.submap_id, local_i))
                if nid is not None:
                    return nid, submap, local_i
    return None


def merge_submap_outputs(
    submaps: list[Submap],
    corrected_extrinsics: np.ndarray,
    graph: "PoseGraph | None" = None,
) -> dict:
    """Assemble a unified raw_outputs dict from per-submap outputs with corrected poses.

    Args:
        graph: optimized PoseGraph (from graph.py) whose get_homography() returns the
            SL(4) homography for any node_id. When provided, each submap's world_points
            are reprojected via submap.get_world_points(H=graph.get_homography(frame_start)).
            Without this the concatenated world_points are in mixed submap-local frames.
    """
    merged: dict = {}
    if not submaps or submaps[0].raw_outputs is None:
        return {"extrinsic": corrected_extrinsics}

    # Handle backends (e.g. MapAnything) whose _forward returns list[dict] per frame.
    # Deduplicate overlap frames: each submap contributes only its non-overlap frames
    # (first K frames, where K = len(s.poses) - overlap), except the last submap which
    # contributes all its frames. This produces exactly N unique frames aligned with
    # corrected_extrinsics (N, 4, 4).
    if isinstance(submaps[0].raw_outputs, list):
        merged_list: list = []
        n_submaps = len(submaps)
        for idx, s in enumerate(submaps):
            if not s.raw_outputs:
                continue
            raw_list = s.raw_outputs
            if idx < n_submaps - 1:
                # Infer overlap from next submap's frame_start vs this submap's end
                next_start = submaps[idx + 1].frame_start
                this_end = s.frame_start + len(raw_list)
                overlap = max(0, this_end - next_start)
                keep = len(raw_list) - overlap
                merged_list.extend(raw_list[:keep])
            else:
                merged_list.extend(raw_list)
        merged_list_out: dict = {
            "_raw_list": merged_list,
            "extrinsic": corrected_extrinsics[:, :3, :],
            "extrinsic_global_4x4": corrected_extrinsics,
        }
        return merged_list_out

    sample = submaps[0].raw_outputs
    for key, val in sample.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1:
            try:
                merged[key] = np.concatenate(
                    [s.raw_outputs[key] for s in submaps if s.raw_outputs and key in s.raw_outputs],
                    axis=0,
                )
            except Exception:
                merged[key] = val
        else:
            merged[key] = val

    # raw_outputs["images"] is a torch tensor — generic loop above falls through to
    # merged["images"] = sample["images"] (only first submap, size K not N*K).
    # Rebuild from submap.frames (already CPU) converting to numpy so downstream
    # code (unproject_and_filter_points) receives an array aligned with depth/conf.
    frame_chunks = []
    for s in submaps:
        if s.frames is None:
            continue
        f = s.frames
        if hasattr(f, "cpu"):
            f = f.cpu().float().numpy()
        else:
            f = np.asarray(f, dtype=np.float32)
        frame_chunks.append(f)
    if frame_chunks:
        merged["images"] = np.concatenate(frame_chunks, axis=0)

    # world_points live on Submap.world_points (not in raw_outputs).  Merge them
    # here so downstream code gets a single global-frame point cloud.
    # When graph is provided, reproject each submap's world_points via SL(4)
    # get_world_points(H=graph.get_homography(frame_start)). Without it points
    # are concatenated in their mixed submap-local frames.
    wp_chunks = []
    for s in submaps:
        if s.world_points is None:
            continue
        if graph is not None:
            pts = s.get_world_points(H=graph.get_homography(s.frame_start))
        else:
            pts = s.world_points.reshape(-1, 3).astype(np.float32)
        wp_chunks.append(pts)
    if wp_chunks:
        merged["world_points"] = np.concatenate(wp_chunks, axis=0)

    # corrected_extrinsics is indexed by global frame index (size N).
    # All other keys were concatenated across submaps including overlapping frames,
    # so extrinsic must follow the same scheme: repeat frames from each submap's
    # local window (frame_start : frame_start + len(poses)) to stay aligned.
    # corrected_extrinsics is (N, 4, 4); raw_outputs["extrinsic"] convention is (K, 3, 4).
    # Slice to 3x4 so _raw_to_world_points can append the bottom row without producing (K,5,4).
    # corrected_extrinsics is (N, 4, 4); raw_outputs["extrinsic"] convention is (K, 3, 4).
    # Slice to 3x4 so _raw_to_world_points can append the bottom row without producing (K,5,4).
    # Uses the overlap-expanded scheme (size M >= N) so depth/images stay aligned.
    merged["extrinsic"] = np.concatenate(
        [corrected_extrinsics[s.frame_start : s.frame_start + len(s.poses), :3, :] for s in submaps],
        axis=0,
    )
    # Store the deduped global poses (N, 4, 4) separately so _postprocess can expose exactly
    # N extrinsics in FeedforwardResult (not the overlap-expanded M).
    merged["extrinsic_global_4x4"] = corrected_extrinsics

    # Build a dedup-index array: for each global frame g in 0..N-1, record the first
    # row in the M-expanded concatenated arrays that corresponds to frame g.
    # Used by BundleAdjustment._apply_ba to align intrinsics/images/conf/world_points
    # (all M-expanded) down to N unique frames before running track extraction + BA.
    N_global = corrected_extrinsics.shape[0]
    dedup_rows = np.full(N_global, -1, dtype=np.int64)
    row = 0
    for s in submaps:
        for li in range(len(s.poses)):
            g = s.frame_start + li
            if g < N_global and dedup_rows[g] < 0:
                dedup_rows[g] = row
            row += 1
    merged["_dedup_rows"] = dedup_rows  # (N,) int64 — index into M-expanded arrays

    return merged
