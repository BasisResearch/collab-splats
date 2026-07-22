"""LoopClosure wrapper around pointcloud creators.

Quickstart:
    from collab_splats.pointcloud import get_creator
    from collab_splats.geometry.loop_closure import LoopClosure, LoopClosureConfig

    base = get_creator("vggtx")()
    lc = LoopClosure(base, config=LoopClosureConfig())
    result = lc.reconstruct(image_dir="path/to/images", output_dir="path/to/out")

See docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb for the full
walkthrough (candidate matching, plots).

This module deliberately depends on pointcloud result types (PointcloudResult,
FeedforwardResult, _raw_to_world_points) because it wraps feedforward creators.
That reverse dependency (geometry -> pointcloud) is why LoopClosure is lazily
exported via __getattr__ from the geometry and geometry.loop_closure __init__s.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from rich.console import Console
from tqdm.auto import tqdm
from vggt.utils.geometry import unproject_depth_map_to_point_map

from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses
from collab_splats.localization import BaseRetrievalExtractor
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import FeedforwardResult, _raw_to_world_points
from collab_splats.pointcloud.utils import subsample_points
from collab_splats.preproc.frame_store import FrameStore

from .graph import PoseGraph
from .map import GraphMap
from .matching import find_loop_closures, translation_jump_check
from .submap import Submap, assert_world_to_cam

logger = logging.getLogger(__name__)

__all__ = ["LoopClosure"]


@dataclasses.dataclass
class LoopClosureConfig:
    submap_size: int = 20
    submap_overlap: int = 1  # 1 = VGGT-SLAM parity; 4 = old default
    # DINO-SALAD retrieval gate: accept candidate if L2(q, ref) < lc_retrieval_threshold.
    # L2 distance on unit-norm DINO-SALAD embeddings (range [0, 2]; typical good matches < 0.5).
    # 0.95 matches VGGT-SLAM main.py default (lc_thres=0.95). 0.0 disables retrieval.
    lc_retrieval_threshold: float = 0.95
    max_loops_per_submap: int = 5
    # None → resolve to the creator's default_verify_match_ratio at LoopClosure
    # wrapper init (fallback 0.85); an explicit float always wins.
    verify_match_ratio: float | None = None
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    # Inter-submap scale estimation method.
    # "rotation_only" — VGGT-SLAM default: T[:3,:3] applied to curr_pts (rotation only)
    # "se3"           — full SE3 T applied before norm ratio
    # "pairwise_dist" — pairwise distance ratio, translation-invariant
    # "none"          — skip scale estimation entirely; always use scale=1.0
    scale_method: Literal["se3", "rotation_only", "pairwise_dist", "none"] = "rotation_only"
    max_jump_ratio: float = math.inf  # reject loops where ‖ΔT.t‖/path_length > this; math.inf disables
    conf_threshold: float = 25.0  # confidence gate for scale estimation; matches VGGT-SLAM --conf_threshold 25
    # deferred = all loop edges + final solve after the window loop (repo default); live = insert each loop edge during the window loop (VGGT-SLAM style).
    loop_edge_timing: Literal["deferred", "live"] = "deferred"

    @property
    def lc_threshold_l2(self) -> float:
        """L2 threshold passed to find_loop_closures. Alias for lc_retrieval_threshold."""
        return self.lc_retrieval_threshold


def _camera_centers_from_poses(poses: np.ndarray) -> np.ndarray:
    """World camera centers (S, 3) from (S, 4, 4) world-to-cam poses: C = -R.T @ t."""
    poses = np.asarray(poses, dtype=np.float64)
    centers = np.empty((poses.shape[0], 3), dtype=np.float32)
    for i in range(poses.shape[0]):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        centers[i] = -R.T @ t
    return centers


def _trim_forward_outputs(raw: "dict | list", k: int) -> "dict | list":
    """Trim per-frame predictions to first k entries.

    When _forward receives a K+overlap window for extra VGGT attention context,
    discards the extra overlap predictions so only K are stored per Submap.
    """
    if isinstance(raw, list):
        return raw[:k]
    trimmed = {}
    for key, val in raw.items():
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > k:
            trimmed[key] = val[:k]
        else:
            trimmed[key] = val
    return trimmed


########################################################
########## LoopClosure wrapper ########################
########################################################


class LoopClosure:
    """Proxy wrapper that runs the submap loop-closure pipeline around any feedforward creator.

    Forwards all non-inference methods to ``base`` and overrides ``run_inference()``
    with the full LC loop.
    """

    def __init__(self, base: Any, config: LoopClosureConfig | None = None) -> None:
        self.base = base
        self.config = config if config is not None else LoopClosureConfig()
        # Resolve verify_match_ratio=None to the creator's per-model calibrated
        # threshold on EVERY construction path (explicit config included), so
        # per-model calibrations aren't shadowed by the dataclass default.
        # An explicit float always wins; fallback 0.85 (VGGT-SPARK calibration).
        if self.config.verify_match_ratio is None:
            self.config = dataclasses.replace(
                self.config,
                verify_match_ratio=getattr(base, "default_verify_match_ratio", 0.85),
            )
        # Scene state (VGGT-SLAM Solver structure): the submap collection + the pose graph.
        self.map = GraphMap()
        self.graph = PoseGraph()
        # Optional live scene Viewer, set externally by the driver (P6.3). When None
        # (default) every viewer hook below is a guarded no-op — the LC path is
        # byte-identical to a run without visualization.
        self.viz = None
        # Test-support stash: the submap lists that drove self.graph on the last run.
        # Initialized empty so early-return paths / pre-run reads never hit AttributeError.
        self._last_submaps: list[Submap] = []
        self._last_lc_submaps: list[Submap] = []
        # True once _run_lc_loop has assembled base.outputs from the GraphMap dense
        # cloud; makes postprocess() a no-op so base._postprocess doesn't overwrite it.
        self._lc_assembled: bool = False

    ######################################################
    ########## Delegation — proxy to self.base ##########
    ######################################################

    def __getattr__(self, name: str) -> Any:
        # Delegate any attribute not defined on the wrapper to the wrapped creator.
        # __getattr__ only fires for missing names, so the explicit overrides below
        # (run_inference, run, reconstruct, outputs/raw_outputs properties) still win.
        return getattr(self.base, name)

    @property
    def outputs(self) -> Any:
        return self.base.outputs

    @outputs.setter
    def outputs(self, value: Any) -> None:
        self.base.outputs = value

    @property
    def raw_outputs(self) -> Any:
        return self.base.raw_outputs

    def reconstruct(self, source: FrameStore | Path, output_dir: Path) -> PointcloudResult:
        # source is a FrameStore (canonical keyframe store) or a legacy image dir
        output_dir = Path(output_dir)
        self.load_model()
        self.setup_inference(source)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

    def run(self, source: FrameStore | Path) -> FeedforwardResult:
        """Run inference pipeline without COLMAP; return FeedforwardResult.

        For the LC path the output is assembled directly from the GraphMap dense
        cloud (N-frame), so no overlap-row deduplication is needed here.
        """
        self.load_model()
        self.setup_inference(source)
        self.run_inference()
        self.postprocess()
        return self.base.outputs

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to base unless the LC path already assembled outputs from the GraphMap."""
        if self._lc_assembled:
            return None
        return self.base.postprocess(*args, **kwargs)

    ######################################################
    ########## Inference — LC loop override ###########
    ######################################################

    def run_inference(self, **kwargs: Any) -> None:
        if self._enough_frames():
            self._run_lc_loop(**kwargs)
        else:
            self.base.run_inference(**kwargs)

    def _enough_frames(self) -> bool:
        views = self.base.views
        n = views.shape[0] if hasattr(views, "shape") else len(views)
        return n >= self.config.submap_size

    def run_predictions(
        self,
        window: Any,
        wi: int,
        start: int,
        submaps: list[Submap],
        lc_submaps: list[Submap],
        retrieval_extractor: Any,
        console: Console,
        **kwargs: Any,
    ) -> "tuple[Submap, list[Submap], list]":
        """Forward-pass one window, build its Submap, detect + verify loop candidates.

        Returns (submap, lc_submaps, loop_matches): the window's Submap, the list of
        verified loop-closure submaps (each 2 frames), and all post-NMS candidates
        (accepted + rejected). Not a singular match — max_loops_per_submap defaults to 5.
        """
        cfg = self.config
        k = window.shape[0] if hasattr(window, "shape") else len(window)
        end = start + k  # window == views[start:start+k]; matches the driver's slice bound

        with torch.no_grad():
            raw = self.base._forward(self.base.model, window, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Models that return list[dict] (e.g. MapAnything) must aggregate to a flat
        # dict for the LC loop. raw_lc is used for LC metadata; raw is stored in the
        # Submap so _postprocess can use the original per-frame structure.
        raw_lc = self.base._lc_collate_outputs(raw) if isinstance(raw, list) else raw

        ext_3x4 = raw_lc["extrinsic"]  # (k, 3, 4)
        poses_4x4 = extrinsics_to_homogeneous(ext_3x4)  # (k, 4, 4)

        assert_world_to_cam(poses_4x4)

        intr_key = "intrinsics" if "intrinsics" in raw_lc else "intrinsic"
        intrinsics = raw_lc.get(intr_key, np.tile(np.eye(3), (k, 1, 1)).astype(np.float32))

        if hasattr(window, "cpu"):
            # Tensor window (e.g. VGGT-X, Omega): shape (K, C, H, W)
            frames_cpu = window.cpu()
        elif isinstance(window, list) and window and isinstance(window[0], dict) and "img" in window[0]:
            # List-of-dicts window (e.g. MapAnything): extract img tensors and stack to (K, C, H, W)
            frames_cpu = torch.cat([v["img"].cpu() for v in window], dim=0)
        else:
            frames_cpu = torch.zeros(k, 3, 1, 1)
        ret_vecs = retrieval_extractor(frames_cpu)  # (k, D)

        wp, wp_conf = _raw_to_world_points(raw_lc)
        submap = Submap(
            submap_id=wi,
            frames=frames_cpu,
            poses=poses_4x4,
            intrinsics=intrinsics,
            retrieval_vectors=ret_vecs,
            image_paths=list(self.base.image_paths[start:end]),
            raw_outputs=raw,
            frame_start=start,
            world_points=wp,
            world_points_conf=wp_conf,
        )

        # Populate the window submap's DENSE data (fat Submap, VGGT-SLAM data path):
        # full-res per-pixel points from depth unprojection, per-pixel RGB, and conf.
        if "depth" in raw_lc and "depth_conf" in raw_lc:
            dense_points = unproject_depth_map_to_point_map(raw_lc["depth"], ext_3x4, intrinsics).astype(np.float32)
            # (k, C, H, W) -> (k, H, W, 3). VGGT-SLAM scales [0, 1] frames by 255;
            # guard against already-[0, 255] frames (backend preprocessing varies).
            frames_hw3 = frames_cpu.float().numpy().transpose(0, 2, 3, 1)
            if frames_hw3.size and frames_hw3.max() > 1.0:
                dense_colors = frames_hw3.astype(np.uint8)
            else:
                dense_colors = (frames_hw3 * 255.0).astype(np.uint8)
            submap.set_dense_points(dense_points, dense_colors, raw_lc["depth_conf"].astype(np.float32))

        # Free the per-submap raw prediction dict now that points/colors/conf have been
        # unprojected into the submap's dense fields. raw_outputs (full depth+images+conf
        # per frame) is the OOM driver in long streaming runs (VGGT-SLAM memory profile);
        # nothing below this line reads raw/raw_lc. frames stays resident — loop verify
        # (below, and future submaps' retrieval) needs it.
        submap.raw_outputs = None

        # Query retrieval index for loop candidates against prior submaps
        past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
        loop_matches = find_loop_closures(
            submap,
            past_for_lc,
            cfg.lc_threshold_l2,
            cfg.max_loops_per_submap,
            nms_frame_distance=cfg.nms_frame_distance,
        )

        window_lc_submaps: list[Submap] = []
        for match in loop_matches:
            q_frame = frames_cpu[match.query_frame_idx]
            d_submap = submaps[match.detected_submap_id]
            d_frame = d_submap.frames[match.detected_frame_idx]
            verify_ok, lc_data = self.base._verify_loop_candidate(
                q_frame, d_frame, verify_match_ratio=cfg.verify_match_ratio
            )
            if not verify_ok:
                console.log(
                    f"  ✗ Loop rejected (verify ratio): "
                    f"submap {match.query_submap_id} → {match.detected_submap_id}"
                    f"  dist={match.similarity_score:.3f}"
                )
                match.reject_reason = "verify_ratio"
            if verify_ok and lc_data is None:
                # Defensive guard: the verify contract requires every accepting
                # backend to return lc_data with joint poses — this firing means
                # a backend contract violation, not an expected reject path.
                logger.error(
                    "LC verify accepted but returned no lc_data (backend contract " "violation): submap %s → %s",
                    match.query_submap_id,
                    match.detected_submap_id,
                )
                verify_ok = False
                match.reject_reason = "no_joint_poses"
            if verify_ok:
                lc_poses = lc_data["poses"]
                lc_rel = (invert_poses(lc_poses[1].astype(np.float64)) @ lc_poses[0].astype(np.float64)).astype(
                    np.float32
                )
                jump_ok, jump_ratio = translation_jump_check(
                    submaps + [submap],
                    query_idx=len(submaps),
                    query_frame=match.query_frame_idx,
                    detected_idx=match.detected_submap_id,
                    detected_frame=match.detected_frame_idx,
                    lc_relative_pose=lc_rel,
                    max_jump_ratio=cfg.max_jump_ratio,
                )
                if not jump_ok:
                    console.log(
                        f"  ✗ Loop rejected (jump ratio={jump_ratio:.2f}): "
                        f"submap {match.query_submap_id} → {match.detected_submap_id}"
                    )
                    match.reject_reason = "jump_ratio"
                else:
                    match.accepted = True
                    match.reject_reason = None
                    console.log(
                        f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                        f"  dist={match.similarity_score:.3f} jump={jump_ratio:.2f}"
                    )
                    # Reshape LC geometry to Submap's (K, P, 3)/(K, P) convention:
                    # (2, H, W, 3) → (2, H·W, 3) and (2, H, W) → (2, H·W).
                    lc_wp = lc_data.get("world_points")
                    lc_wp = lc_wp.reshape(2, -1, 3) if lc_wp is not None else None
                    lc_conf = lc_data.get("conf")
                    lc_conf = lc_conf.reshape(2, -1) if lc_conf is not None else None
                    # submap_id mirrors the original in-place accumulator growth:
                    # len(submaps) (current submap not yet appended) + all prior lc
                    # submaps + those already appended for this window.
                    window_lc_submaps.append(
                        Submap(
                            submap_id=len(submaps) + len(lc_submaps) + len(window_lc_submaps),
                            frames=torch.stack([q_frame, d_frame]),
                            poses=lc_poses,
                            intrinsics=np.stack(
                                [
                                    submap.intrinsics[match.query_frame_idx],
                                    d_submap.intrinsics[match.detected_frame_idx],
                                ]
                            ),
                            retrieval_vectors=torch.zeros(2, ret_vecs.shape[-1]),
                            image_paths=[
                                submap.image_paths[match.query_frame_idx],
                                d_submap.image_paths[match.detected_frame_idx],
                            ],
                            is_lc_submap=True,
                            world_points=lc_wp,
                            world_points_conf=lc_conf,
                        )
                    )

        return submap, window_lc_submaps, loop_matches

    def add_points(
        self,
        submap: Submap,
        lc_submaps: list[Submap],
        submaps: list[Submap],
        lc_submaps_acc: list[Submap],
    ) -> None:
        """Append the window's submap + verified loop submaps to the driver's lists.

        Bookkeeping only — PGO is deferred to the batch call after the sweep.
        """
        submaps.append(submap)
        lc_submaps_acc.extend(lc_submaps)
        # Register the finalized window submap (dense-data carrier) in the scene map.
        self.map.add_submap(submap)
        # Drive the pose graph incrementally, mirroring the batch shim's per-submap
        # cadence (add each submap's sequential edges, then optimize). Loop edges are
        # deferred to after the window sweep (in _run_lc_loop), matching the shim.
        self.graph.add_submap(
            submap,
            self.config.submap_overlap,
            conf_threshold=self.config.conf_threshold,
            scale_method=self.config.scale_method,
        )
        self.graph.optimize()

        # Hook 1: push the just-appended submap (latest-only) to the live viewer.
        self._viz_push_submap(submap)

    def _add_loop_edge(self, lc: Submap, submaps: list[Submap]) -> None:
        """Add one verified loop submap's loop edge to self.graph (always explicit scale/conf)."""
        self.graph.add_loop_edge(
            lc,
            submaps,
            conf_threshold=self.config.conf_threshold,
            scale_method=self.config.scale_method,
        )

    ######################################################
    ########## Live viewer hooks (guarded) ##############
    ######################################################

    def _viz_push_submap(self, submap: Submap) -> None:
        """Guarded push of one submap's corrected points + per-frame frusta to the viewer."""
        if self.viz is None:
            return
        # A viewer failure must NEVER break reconstruction.
        try:
            # LC carrier / degraded submaps have no dense cloud — nothing to draw.
            if submap.is_lc_submap or submap.points is None:
                return
            # Skip this submap's leading overlap frames: they duplicate the previous
            # submap's trailing frames (dedup_overlap assigns overlap to the earlier
            # submap). Without this, every seam draws O frames' points + frusta twice.
            skip = self.config.submap_overlap if submap.frame_start > 0 else 0
            world_pts = submap.get_points_in_world_frame(self.graph, skip_first=skip)
            # A pure-overlap tail submap trims to nothing — leave the scene untouched.
            if world_pts.shape[0] == 0:
                return
            pts, cols = subsample_points(
                world_pts,
                submap.get_points_colors(skip_first=skip),
                max_points=50000,
            )
            self.viz.add_points(f"submap_{submap.submap_id}", pts, cols)
            # Per-frame frusta at the corrected world-to-cam poses (overlap frames skipped).
            poses = submap.get_all_poses_world(self.graph)
            for i in range(skip, poses.shape[0]):
                self.viz.add_frustum(f"submap_{submap.submap_id}/cam_{i}", poses[i], submap.intrinsics[i])
        except Exception as e:
            logger.warning("viewer submap push failed (submap %s): %s", submap.submap_id, e)

    def _viz_reupload_all(self) -> None:
        """Guarded full re-upload of every non-LC submap (scene snaps to corrected poses)."""
        if self.viz is None:
            return
        # Belt-and-suspenders: the enumeration + graph reads are wrapped too, so
        # NOTHING viewer-related can propagate into _run_lc_loop.
        try:
            for s in self.map.ordered_submaps_by_key():
                self._viz_push_submap(s)
        except Exception as e:
            logger.warning("viewer reupload failed: %s", e)

    def _viz_draw_loops(self, loop_matches: list, submaps: list[Submap]) -> None:
        """Guarded loop-edge lines between graph-corrected query + detected camera centers.

        Endpoints are re-sourced from the query/detected submaps' get_all_poses_world
        (the SAME graph-corrected world-to-cam convention/frame as the frusta) — NOT
        from the LC submap's raw verify poses, which live in the loop pair's local frame.
        """
        if self.viz is None:
            return
        try:
            for m in loop_matches:
                if not getattr(m, "accepted", False):
                    continue
                q = submaps[m.query_submap_id]
                d = submaps[m.detected_submap_id]
                q_center = _camera_centers_from_poses(q.get_all_poses_world(self.graph)[m.query_frame_idx][None])[0]
                d_center = _camera_centers_from_poses(d.get_all_poses_world(self.graph)[m.detected_frame_idx][None])[0]
                segment = np.stack([q_center, d_center])[None]  # (1, 2, 3)
                self.viz.add_lines(f"loop_{m.query_submap_id}_{m.detected_submap_id}", segment)
        except Exception as e:
            logger.warning("viewer loop-line failed: %s", e)

    def _run_lc_loop(self, **kwargs: Any) -> None:
        console = Console()

        # Reset the scene map + pose graph so a re-run doesn't accumulate stale state.
        self.map = GraphMap()
        self.graph = PoseGraph()
        # Cleared until the GraphMap output is assembled at the end of this run.
        self._lc_assembled = False

        cfg = self.config
        K, O = cfg.submap_size, cfg.submap_overlap
        step = max(1, K)  # stride = submap_size, matching VGGT-SLAM (not K-O)
        views = self.base.views
        N = views.shape[0] if hasattr(views, "shape") else len(views)
        device = str(next(self.base.model.parameters()).device)

        # Load DINO-SALAD retrieval extractor; fall back to full-sequence inference if unavailable
        try:
            retrieval_cls = BaseRetrievalExtractor.get("dino-salad")
            retrieval_extractor = retrieval_cls(device=device)
        except Exception as e:
            logger.warning("DINO-SALAD failed to load (%s) — skipping loop closure", e)
            self.base.raw_outputs = self.base._forward(self.base.model, views, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []
        n_submaps = math.ceil(max(1, N - O) / step)
        loops_found = 0
        verified = 0

        console.log(f"Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})")

        # Slide submap window across frames (stride = submap_size, overlap frames stored per submap)
        with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
            for wi, start in enumerate(range(0, N, step)):
                # Each window = K+O frames, matching VGGT-SLAM's submap_size+overlapping_window_size.
                # The overlap frame (index K) is kept as the boundary/carry frame for the next submap.
                end = min(start + K + O, N)
                window = views[start:end]

                # run_predictions == VGGT-SLAM's per-frame forward+detect; add_points defers PGO to the batch step.
                submap, window_lc_submaps, loop_matches = self.run_predictions(
                    window, wi, start, submaps, lc_submaps, retrieval_extractor, console, **kwargs
                )
                self.add_points(submap, window_lc_submaps, submaps, lc_submaps)

                # live timing: insert THIS window's verified loop edges immediately (right
                # after the submap that detected them) + re-solve — VGGT-SLAM solver.py style.
                if cfg.loop_edge_timing == "live" and window_lc_submaps:
                    for lc in window_lc_submaps:
                        self._add_loop_edge(lc, submaps)
                    self.graph.optimize()

                # Hook 2: on accepted loop(s), draw the loop line(s) and re-upload the
                # whole scene (a loop redistributes error across all prior submaps).
                if window_lc_submaps:
                    self._viz_draw_loops(loop_matches, submaps)
                    self._viz_reupload_all()

                loops_found += len(window_lc_submaps)
                verified += len(window_lc_submaps)
                pbar.update(1)
                pbar.set_postfix(loops=loops_found, verified=verified)
                if end >= N:
                    break

        # Number of accepted loop-closure submaps applied (user-visible summary).
        self.base.n_loops_applied = len(lc_submaps)

        # Finish driving self.graph: in deferred mode, add all loop-closure chains now
        # (live mode already inserted each edge during the window loop), then one final
        # solve — the batch per-submap cadence's tail. Output assembly reads directly
        # off self.graph (correction-at-read) via the GraphMap below.
        if cfg.loop_edge_timing == "deferred":
            for lc in lc_submaps:
                self._add_loop_edge(lc, submaps)
        self.graph.optimize()

        # Hook 3: final PGO done — re-upload every submap so the scene snaps to the
        # fully corrected (loop-closed) poses.
        self._viz_reupload_all()

        # Stash the driven submap lists so parity tests can compare self.graph to a
        # manual incremental drive of the same submaps.
        self._last_submaps = submaps
        self._last_lc_submaps = lc_submaps

        # Assemble the final FeedforwardResult directly from the GraphMap dense cloud
        # (VGGT-SLAM style — points already unprojected per-submap, corrected at read
        # via self.graph). Sets base.outputs so postprocess()/run()/build_colmap consume
        # it; the _lc_assembled flag makes wrapper.postprocess() a no-op.
        self.base.outputs = self._assemble_result(N)
        self._lc_assembled = True

    def _assemble_result(self, n_frames: int) -> FeedforwardResult:
        """Build FeedforwardResult from the GraphMap dense cloud (VGGT-SLAM correction-at-read)."""
        # Dense world cloud + corrected extrinsics come straight from the graph-corrected map.
        points, colors = self.map.get_world_pointcloud(self.graph)
        extrinsics = self.map.get_corrected_extrinsics(self.graph, n_frames)

        # Cap the assembled cloud to the creator's point budget before it flows into
        # build_colmap. get_world_pointcloud concatenates every submap's dense per-pixel
        # points (~50M for a 500-frame scene); materializing that as pycolmap Point3D
        # objects blows the 50 GB cgroup and SIGKILLs. The old thin _postprocess applied
        # this same max_points cap — the LC path (postprocess no-op) must too. ATE-neutral:
        # trajectory error reads extrinsics, not points. conf already masked at read.
        points, colors = subsample_points(points, colors, max_points=self.base.max_points)

        # Dedup per-frame intrinsics to N unique frames — first occurrence per global
        # frame, the same overlap-assignment rule dedup_overlap uses for poses. Model
        # dims come from a dense point grid (points were (S, H, W, 3) before flatten).
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1))
        assigned = np.zeros(n_frames, dtype=bool)
        model_height = model_width = None
        for s in self.map.ordered_submaps_by_key():
            if s.is_lc_submap:
                continue
            if model_height is None and s.points is not None:
                model_height, model_width = int(s.points.shape[1]), int(s.points.shape[2])
            for local_i in range(s.intrinsics.shape[0]):
                g = s.frame_start + local_i
                if 0 <= g < n_frames and not assigned[g]:
                    intrinsics[g] = s.intrinsics[local_i].astype(np.float32)
                    assigned[g] = True
        # Fall back to the creator's model dims if no submap carried a dense grid.
        if model_height is None:
            model_height = int(getattr(self.base, "model_height", 0))
            model_width = int(getattr(self.base, "model_width", 0))

        # Fail fast: an empty cloud or zero model dims means every non-LC submap
        # lacked dense points (fully-degraded backend) or was fully conf-masked.
        # Building a 0-point/0-dim FeedforwardResult only crashes cryptically later
        # in build_colmap's intrinsic rescale — name the cause here instead.
        if points.shape[0] == 0 or model_width == 0 or model_height == 0:
            raise ValueError(
                "LC produced an empty point cloud — all submaps lacked dense points "
                "or were confidence-masked out (no geometry to assemble a result from)."
            )

        return FeedforwardResult(
            points=points,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_paths=list(self.base.image_paths),
            original_coords=self.base.original_coords,
            model_width=model_width,
            model_height=model_height,
        )
