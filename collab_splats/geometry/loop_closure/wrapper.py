"""
Submap loop closure around a feedforward creator's forward pass.

- walkthrough: docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
- imports pointcloud result types, so geometry exports LoopClosure lazily via __getattr__
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

from .graph import PoseGraph, check_scale_method
from .map import GraphMap
from .matching import find_loop_closures
from .submap import Submap, assert_world_to_cam

logger = logging.getLogger(__name__)

__all__ = ["LoopClosure"]


@dataclasses.dataclass
class LoopClosureConfig:
    """
    Loop-closure settings, grouped by pipeline step.

    - windowing: submap_size, submap_overlap
    - retrieval: lc_retrieval_threshold, max_loops_per_submap, nms_frame_distance, min_submap_gap
    - verification: verify_match_ratio
    - scale estimation: scale_method, conf_threshold
    - pose graph: loop_edge_timing
    - viewer: viz_max_points
    """

    # window stride; each submap holds this plus submap_overlap frames
    submap_size: int = 20
    # frames each window adds past submap_size, shared with the next; 1 = VGGT-SLAM parity
    submap_overlap: int = 1
    # max L2 distance between unit DINO-SALAD descriptors, range [0, 2]; 0.0 disables retrieval
    lc_retrieval_threshold: float = 0.95
    # candidates kept per query submap
    max_loops_per_submap: int = 5
    # None: the creator's default_verify_match_ratio; an explicit float wins
    verify_match_ratio: float | None = None
    # suppression radius in frames between candidates on the same detected submap
    nms_frame_distance: int = 25
    # newest submaps left out of the retrieval search
    min_submap_gap: int = 1
    # inter-submap scale: rotation_only (VGGT-SLAM) or none (scale 1.0)
    scale_method: Literal["rotation_only", "none"] = "rotation_only"
    conf_threshold: float = 25.0  # confidence gate for scale estimation; matches VGGT-SLAM --conf_threshold 25
    # deferred: all loop edges after the window loop; live: each edge as found (VGGT-SLAM)
    loop_edge_timing: Literal["deferred", "live"] = "deferred"
    viz_max_points: int = 50000  # viewer per-submap point cap

    def __post_init__(self) -> None:
        """
        Fail at construction on an unknown scale_method, before any model loads.

        Raises:
            ValueError: `scale_method` is not "rotation_only" or "none".
        """
        check_scale_method(self.scale_method)


def _camera_centers_from_poses(poses: np.ndarray) -> np.ndarray:
    """World camera centers (S, 3) from (S, 4, 4) world-to-cam poses: C = -R.T @ t."""
    poses = np.asarray(poses, dtype=np.float64)
    centers = np.empty((poses.shape[0], 3), dtype=np.float32)
    for i in range(poses.shape[0]):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]
        centers[i] = -R.T @ t
    return centers


########################################################
########## LoopClosure wrapper ########################
########################################################


class LoopClosure:
    """
    Proxy around a feedforward creator that replaces its inference with the submap LC loop.

    - `run_inference()` is the LC entry point; `reconstruct()` and `run()` call it
    - every other attribute is forwarded to `base`
    - fewer than submap_size frames, or DINO-SALAD failing to load, falls back to the
      creator's own forward pass

    Quickstart:
        from pathlib import Path
        from collab_splats.geometry import LoopClosure, LoopClosureConfig
        from collab_splats.pointcloud import get_creator
        lc = LoopClosure(get_creator("vggtx")(), config=LoopClosureConfig())
        lc.load_model()
        lc.setup_inference(Path("scene/images"))
        lc.run_inference()
        lc.postprocess()
        result = lc.outputs  # FeedforwardResult
    """

    def __init__(self, base: Any, config: LoopClosureConfig | None = None) -> None:
        self.base = base
        self.config = config if config is not None else LoopClosureConfig()
        # None resolves to the creator's per-model calibration
        if self.config.verify_match_ratio is None:
            self.config = dataclasses.replace(
                self.config,
                verify_match_ratio=base.default_verify_match_ratio,
            )
        # Scene state (VGGT-SLAM Solver structure): the submap collection + the pose graph.
        self.map = GraphMap()
        self.graph = PoseGraph()
        # Optional live Viewer, set by the driver
        # - None: every viewer hook is a no-op and output is unchanged
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
        # Delegate attributes the wrapper lacks to the wrapped creator
        # - fires only for missing names, so the overrides below still win
        return getattr(self.base, name)

    @property
    def outputs(self) -> Any:
        """
        The wrapped creator's outputs.

        Returns:
            base.outputs.
        """
        return self.base.outputs

    @outputs.setter
    def outputs(self, value: Any) -> None:
        """
        Set the wrapped creator's outputs.

        Args:
            value: new base.outputs.
        """
        self.base.outputs = value

    @property
    def raw_outputs(self) -> Any:
        """
        The wrapped creator's raw forward outputs.

        Returns:
            base.raw_outputs.
        """
        return self.base.raw_outputs

    def reconstruct(self, source: Path, output_dir: Path) -> PointcloudResult:
        """
        Full pipeline with loop closure, ending in a COLMAP model.

        Args:
            source: the scene's images/ directory.
            output_dir: where build_colmap writes its model.

        Returns:
            The creator's PointcloudResult.
        """
        # source is the scene's images/ directory — the one keyframe store
        output_dir = Path(output_dir)
        self.load_model()
        self.setup_inference(source)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

    def run(self, source: Path) -> FeedforwardResult:
        """
        Pipeline with loop closure, without COLMAP.

        - the LC output is already one row per frame, so no overlap dedup is needed

        Args:
            source: the scene's images/ directory.

        Returns:
            The assembled FeedforwardResult.
        """
        self.load_model()
        self.setup_inference(source)
        self.run_inference()
        self.postprocess()
        return self.base.outputs

    def postprocess(self, *args: Any, **kwargs: Any) -> Any:
        """
        Creator postprocess, skipped once the LC loop has assembled outputs.

        Args:
            args: forwarded to base.postprocess.
            kwargs: forwarded to base.postprocess.

        Returns:
            None after LC assembly; otherwise base.postprocess's return value.
        """
        if self._lc_assembled:
            return None
        return self.base.postprocess(*args, **kwargs)

    ######################################################
    ########## Inference — LC loop override ###########
    ######################################################

    def run_inference(self, **kwargs: Any) -> None:
        """
        Run the LC loop, or fall back to the creator's own forward pass.

        - LC loop: sets base.outputs to the assembled FeedforwardResult
        - fewer than submap_size frames: the creator's run_inference
        - DINO-SALAD fails to load: sets base.raw_outputs only; outputs is not assembled

        Args:
            kwargs: forwarded to the creator's forward pass.
        """
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
        """
        Forward one window, build its Submap, then find and verify loop candidates.

        Args:
            window: the window's model inputs (tensor or list of view dicts).
            wi: window index, used as the new submap_id.
            start: global index of the window's first frame.
            submaps: window submaps built so far.
            lc_submaps: loop submaps accepted so far.
            retrieval_extractor: DINO-SALAD extractor for the window's descriptors.
            console: console for loop accept/reject lines.
            kwargs: forwarded to the creator's forward pass.

        Returns:
            (submap, window_lc_submaps, loop_matches): the window's Submap, its verified
            2-frame loop submaps, and every post-NMS candidate, accepted or rejected.
        """
        cfg = self.config
        k = window.shape[0] if hasattr(window, "shape") else len(window)
        end = start + k  # window == views[start:start+k]; matches the driver's slice bound

        with torch.no_grad():
            raw = self.base._forward(self.base.model, window, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Collate list[dict] outputs (e.g. MapAnything) into one flat dict
        # - raw_lc feeds the LC loop; raw keeps the per-frame structure
        raw_lc = self.base._lc_collate_outputs(raw) if isinstance(raw, list) else raw

        ext_3x4 = raw_lc["extrinsic"]  # (k, 3, 4)
        poses_4x4 = extrinsics_to_homogeneous(ext_3x4)  # (k, 4, 4)

        assert_world_to_cam(poses_4x4)

        intrinsics = raw_lc["intrinsics"]

        # Stack the window's frames to (K, C, H, W)
        # - tensor window: VGGT-X, VGGT-Omega, LoGeR
        # - list of {"img": ...} view dicts: MapAnything
        if hasattr(window, "cpu"):
            frames_cpu = window.cpu()
        elif isinstance(window, list) and window and isinstance(window[0], dict) and "img" in window[0]:
            frames_cpu = torch.cat([v["img"].cpu() for v in window], dim=0)
        else:
            raise TypeError(
                f"LC window must be a tensor or a list of {{'img': ...}} dicts, got {type(window).__name__}"
            )
        ret_vecs = retrieval_extractor(frames_cpu)  # (k, D)

        wp, wp_conf = _raw_to_world_points(raw_lc)
        submap = Submap(
            submap_id=wi,
            frames=frames_cpu,
            poses=poses_4x4,
            intrinsics=intrinsics,
            retrieval_vectors=ret_vecs,
            image_paths=list(self.base.image_paths[start:end]),
            frame_start=start,
            world_points=wp,
            world_points_conf=wp_conf,
        )

        # Populate the window submap's DENSE data (fat Submap, VGGT-SLAM data path):
        # full-res per-pixel points from depth unprojection, per-pixel RGB, and conf.
        if "depth" in raw_lc and "depth_conf" in raw_lc:
            dense_points = unproject_depth_map_to_point_map(raw_lc["depth"], ext_3x4, intrinsics).astype(np.float32)
            if "colors" in raw_lc:
                # Backend supplied denormalized (k, H, W, 3) uint8 RGB (e.g. MapAnything,
                # whose frame tensors are dinov2-normalized and unusable as color).
                dense_colors = raw_lc["colors"].astype(np.uint8)
            else:
                # (k, C, H, W) -> (k, H, W, 3). VGGT-SLAM scales [0, 1] frames by 255;
                # guard against already-[0, 255] frames (backend preprocessing varies).
                frames_hw3 = frames_cpu.float().numpy().transpose(0, 2, 3, 1)
                if frames_hw3.size and frames_hw3.max() > 1.0:
                    dense_colors = frames_hw3.astype(np.uint8)
                else:
                    dense_colors = (frames_hw3 * 255.0).astype(np.uint8)
            submap.set_dense_points(dense_points, dense_colors, raw_lc["depth_conf"].astype(np.float32))

        # Query retrieval index for loop candidates against prior submaps
        past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
        loop_matches = find_loop_closures(
            submap,
            past_for_lc,
            cfg.lc_retrieval_threshold,
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
                # Accept without lc_data breaks the verify contract
                # - a backend bug, not an expected reject path
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

                # Reject a loop whose relative pose is not finite
                if not np.isfinite(lc_rel).all():
                    console.log(
                        f"  ✗ Loop rejected (non-finite pose): "
                        f"submap {match.query_submap_id} → {match.detected_submap_id}"
                    )
                    match.reject_reason = "non_finite_pose"
                else:
                    match.accepted = True
                    match.reject_reason = None
                    console.log(
                        f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                        f"  dist={match.similarity_score:.3f}"
                    )
                    # Reshape LC geometry to Submap's (K, P, 3)/(K, P) convention:
                    # (2, H, W, 3) → (2, H·W, 3) and (2, H, W) → (2, H·W).
                    lc_wp = lc_data.get("world_points")
                    lc_wp = lc_wp.reshape(2, -1, 3) if lc_wp is not None else None
                    lc_conf = lc_data.get("conf")
                    lc_conf = lc_conf.reshape(2, -1) if lc_conf is not None else None
                    # Next free submap_id after windows and loop submaps
                    # - submaps excludes the current window, which is not yet appended
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
        """
        Record a window's submaps, then add its sequential edges to the pose graph.

        - loop edges are added later, per loop_edge_timing

        Args:
            submap: the window's submap.
            lc_submaps: the window's verified loop submaps.
            submaps: driver's window submaps, appended in place.
            lc_submaps_acc: driver's loop submaps, extended in place.
        """
        submaps.append(submap)
        lc_submaps_acc.extend(lc_submaps)
        # Register the finalized window submap (dense-data carrier) in the scene map.
        self.map.add_submap(submap)
        # Add this submap's sequential edges, then optimize
        # - loop edges come later, in _run_lc_loop
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
        # Viewer I/O, runtime and bad-data failures are logged; programming errors propagate
        try:
            # LC carrier / degraded submaps have no dense cloud — nothing to draw.
            if submap.is_lc_submap or submap.points is None:
                return
            # Skip leading overlap frames the previous submap already drew
            # - otherwise every seam draws its points and frusta twice
            skip = self.config.submap_overlap if submap.frame_start > 0 else 0
            world_pts = submap.get_points_in_world_frame(self.graph, skip_first=skip)
            # A pure-overlap tail submap trims to nothing — leave the scene untouched.
            if world_pts.shape[0] == 0:
                return
            pts, cols = subsample_points(
                world_pts,
                submap.get_points_colors(skip_first=skip),
                max_points=self.config.viz_max_points,
            )
            self.viz.add_points(f"submap_{submap.submap_id}", pts, cols)
            # Per-frame frusta at the corrected world-to-cam poses (overlap frames skipped).
            poses = submap.get_all_poses_world(self.graph)
            for i in range(skip, poses.shape[0]):
                self.viz.add_frustum(f"submap_{submap.submap_id}/cam_{i}", poses[i], submap.intrinsics[i])
        except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:
            logger.warning("viewer submap push failed (submap %s): %s", submap.submap_id, e)

    def _viz_reupload_all(self) -> None:
        """Guarded full re-upload of every non-LC submap (scene snaps to corrected poses)."""
        if self.viz is None:
            return
        for s in self.map.ordered_submaps_by_key():
            self._viz_push_submap(s)

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
                if not m.accepted:
                    continue
                q = submaps[m.query_submap_id]
                d = submaps[m.detected_submap_id]
                q_center = _camera_centers_from_poses(q.get_all_poses_world(self.graph)[m.query_frame_idx][None])[0]
                d_center = _camera_centers_from_poses(d.get_all_poses_world(self.graph)[m.detected_frame_idx][None])[0]
                segment = np.stack([q_center, d_center])[None]  # (1, 2, 3)
                self.viz.add_lines(f"loop_{m.query_submap_id}_{m.detected_submap_id}", segment)
        except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:
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
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("DINO-SALAD failed to load (%s) — skipping loop closure", e)
            self.base.raw_outputs = self.base._forward(self.base.model, views, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []
        n_submaps = math.ceil(max(1, N - O) / step)
        loops = 0

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

                loops += len(window_lc_submaps)
                pbar.update(1)
                pbar.set_postfix(loops=loops)
                if end >= N:
                    break

        # Number of accepted loop-closure submaps applied (user-visible summary).
        self.base.n_loops_applied = len(lc_submaps)

        # Add deferred loop edges, then the final solve
        # - live mode already added each edge in the window loop
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

        # Assemble outputs from the graph-corrected GraphMap cloud
        # - _lc_assembled makes postprocess() a no-op, so nothing overwrites it
        self.base.outputs = self._assemble_result(N)
        self._lc_assembled = True

    def _assemble_result(self, n_frames: int) -> FeedforwardResult:
        """Build FeedforwardResult from the GraphMap dense cloud (VGGT-SLAM correction-at-read)."""
        # Dense world cloud + corrected extrinsics come straight from the graph-corrected map.
        points, colors = self.map.get_world_pointcloud(self.graph, overlap=self.config.submap_overlap)
        extrinsics = self.graph.extract_extrinsics(n_frames)

        # Cap the dense cloud to the creator's max_points
        # - build_colmap turns every point into a pycolmap Point3D and runs out of memory
        # - postprocess is a no-op here, so this is the only cap
        points, colors = subsample_points(points, colors, max_points=self.base.max_points)

        # One intrinsics row per global frame; model dims from a dense grid
        # - first submap to cover a frame wins, as for poses
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

        # Fail fast on an empty cloud or zero model dims
        # - every submap lacked dense points or was fully confidence-masked
        # - otherwise build_colmap's intrinsic rescale fails later, less clearly
        if model_height is None or points.shape[0] == 0 or model_width == 0 or model_height == 0:
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
