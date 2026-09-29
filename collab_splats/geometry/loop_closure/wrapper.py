"""
Submap loop closure around a feedforward creator's forward pass.

- walkthrough: docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
- imports pointcloud result types, so geometry exports LoopClosure lazily via __getattr__
- adapted from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): main.py (window loop),
  vggt_slam/solver.py (Solver.run_predictions, Solver.add_points)
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

from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.map import GraphMap
from collab_splats.geometry.loop_closure.matching import find_loop_closures
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    transform_points,
)
from collab_splats.localization import BaseRetrievalExtractor
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.utils import subsample_points

logger = logging.getLogger(__name__)

__all__ = ["LoopClosure", "LoopClosureConfig"]


@dataclasses.dataclass
class LoopClosureConfig:
    """
    Loop-closure settings, grouped by pipeline step.

    - windowing: submap_size, submap_overlap
    - retrieval: lc_retrieval_threshold, max_loops_per_submap, nms_frame_distance, min_submap_gap
    - verification: verify_match_ratio
    - scale estimation: conf_percentile
    - pose graph: loop_edge_timing
    - viewer: viz_max_points

    Args:
        submap_size: window stride; each submap holds this plus submap_overlap frames.
        submap_overlap: frames shared by consecutive windows; 1 = VGGT-SLAM parity.
            Owned by the earlier submap: dense reads and viewer pushes skip a non-first
            submap's leading overlap frames, else every seam counts twice.
        lc_retrieval_threshold: max L2 distance between unit DINO-SALAD descriptors, range
            [0, 2]; 0.0 disables retrieval.
        max_loops_per_submap: candidates kept per query submap.
        verify_match_ratio: None takes the creator's default_verify_match_ratio; an
            explicit float wins.
        nms_frame_distance: suppression radius in frames between candidates on the same
            detected submap.
        min_submap_gap: newest submaps left out of the retrieval search.
        conf_percentile: percentile of each submap's dense depth_conf that gates scale points;
            VGGT-SLAM's --conf_threshold 25 (MIT-SPARK/VGGT-SLAM @ fd3fd218, main.py:32,
            vggt_slam/submap.py:40).
        loop_edge_timing: deferred adds all loop edges after the window loop; live adds
            each edge as found (VGGT-SLAM).
        viz_max_points: viewer per-submap point cap.
    """

    submap_size: int = 20
    submap_overlap: int = 1
    lc_retrieval_threshold: float = 0.95
    max_loops_per_submap: int = 5
    verify_match_ratio: float | None = None
    nms_frame_distance: int = 25
    min_submap_gap: int = 1
    conf_percentile: float = 25.0
    loop_edge_timing: Literal["deferred", "live"] = "deferred"
    viz_max_points: int = 50000


########################################################################
# LoopClosure wrapper
########################################################################


class LoopClosure:
    """
    Proxy around a feedforward creator that replaces its inference with the submap LC loop.

    - `run_inference()` is the LC entry point; `create_pointcloud()` reaches it via _reconstruct
    - every other attribute is forwarded to `base`
    - fewer than submap_size frames, or DINO-SALAD failing to load, falls back to the
      creator's own forward pass

    Quickstart:
        from pathlib import Path
        from collab_splats.geometry import LoopClosure, LoopClosureConfig
        from collab_splats.pointcloud import get_creator
        lc = LoopClosure(get_creator("vggtx")(), config=LoopClosureConfig())
        result = lc.create_pointcloud(Path("scene/images"), Path("scene/out"))
    """

    def __init__(self, base: Any, config: LoopClosureConfig | None = None) -> None:
        """
        Wrap a creator, resolving the config against its per-model defaults.

        Args:
            base: feedforward creator whose forward pass the LC loop drives.
            config: loop-closure settings; None uses LoopClosureConfig().
        """
        self.base = base
        self.config = config if config is not None else LoopClosureConfig()

        # None resolves to the creator's per-model calibration
        if self.config.verify_match_ratio is None:
            if base.default_verify_match_ratio is None:
                raise NotImplementedError(f"{type(base).__name__} sets no default_verify_match_ratio; LC unsupported")
            self.config = dataclasses.replace(self.config, verify_match_ratio=base.default_verify_match_ratio)

        # Scene state (VGGT-SLAM Solver structure): the submap collection + the pose graph
        self.map = GraphMap()
        self.graph = PoseGraph()

        # Optional live Viewer, set by the driver
        # - None: every viewer hook is a no-op and output is unchanged
        self.viz = None

        # True once _run_lc_loop has assembled base.outputs from the GraphMap dense cloud
        # - _reconstruct then returns it as is, so base._postprocess does not overwrite it
        # - cleared at the start of each _run_lc_loop
        self._lc_assembled: bool = False

    ####################################################################
    # Delegation to self.base
    ####################################################################

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attributes the wrapper lacks to the wrapped creator.

        - fires only for missing names, so the overrides below still win

        Args:
            name: attribute missing on the wrapper.

        Returns:
            getattr(base, name).
        """
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

    # The base creator's template: postprocess and export around _reconstruct
    create_pointcloud = BasePointcloudCreator.create_pointcloud

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        The base creator's steps with the windowed LC forward; the assembled result when LC ran.
        """
        self.base.load_model()
        self.base.setup_inference(paths)
        self.run_inference()
        if self._lc_assembled:
            return self.base.outputs
        return self.base._postprocess(self.base.raw_outputs)

    ####################################################################
    # Inference: LC loop override
    ####################################################################

    def run_inference(self) -> None:
        """
        Run the LC loop, or fall back to the creator's own forward pass.

        - LC loop: sets base.outputs to the assembled PointcloudResult
        - fewer than submap_size frames: the creator's run_inference
        - DINO-SALAD fails to load: sets base.raw_outputs only; outputs is not assembled
        """
        if self._enough_frames():
            self._run_lc_loop()
        else:
            self.base.run_inference()

    def _n_views(self) -> int:
        """
        Frame count of the creator's views, a tensor or a list of view dicts.
        """
        views = self.base.views
        return views.shape[0] if hasattr(views, "shape") else len(views)

    def _enough_frames(self) -> bool:
        """
        Whether the creator holds at least submap_size frames.

        - True means the LC loop can run
        """
        return self._n_views() >= self.config.submap_size

    def run_predictions(
        self,
        window: Any,
        wi: int,
        start: int,
        submaps: list[Submap],
        lc_submaps: list[Submap],
        retrieval_extractor: Any,
        console: Console,
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

        Returns:
            (submap, window_lc_submaps, loop_matches): the window's Submap, its verified
            loop carriers (see _run_lc_loop), and every post-NMS candidate.
        """
        # Forward the window, then free the cached activations
        cfg = self.config
        k = window.shape[0] if hasattr(window, "shape") else len(window)
        end = start + k  # window == views[start:start+k]; matches the driver's slice bound

        with torch.no_grad():
            raw = self.base._forward(self.base.model, window)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Window poses; the graph reads each submap's local frame as frame 0's camera
        ext_3x4 = raw["extrinsic"]  # (k, 3, 4)
        poses_4x4 = extrinsics_to_homogeneous(ext_3x4)  # (k, 4, 4)
        if not np.allclose(poses_4x4[0], np.eye(4), atol=0.1):
            raise ValueError(f"Window poses not normalized to frame 0: expected poses[0] ≈ eye(4), got\n{poses_4x4[0]}")

        intrinsics = raw["intrinsics"]

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

        # Retrieval descriptors, then the window's Submap
        ret_vecs = retrieval_extractor(frames_cpu)  # (k, D)
        submap = Submap(
            submap_id=wi,
            frames=frames_cpu,
            poses=poses_4x4,
            intrinsics=intrinsics,
            retrieval_vectors=ret_vecs,
            image_paths=list(self.base.image_paths[start:end]),
            frame_start=start,
            conf_percentile=cfg.conf_percentile,
        )

        # Dense per-pixel data (fat Submap, VGGT-SLAM data path)
        # - points: full-res depth unprojection
        # - colors and conf: per pixel
        if "depth" in raw and "depth_conf" in raw:
            dense_points = unproject_depth_map_to_point_map(raw["depth"], ext_3x4, intrinsics).astype(np.float32)

            # Frames (k, C, H, W) -> (k, H, W, 3) uint8 RGB
            # - VGGT-SLAM scales [0, 1] frames by 255
            # - MapAnything view dicts are dinov2-normalized: its raw [0, 1] images instead
            # - guard: some backends already give [0, 255]
            if hasattr(window, "cpu"):
                frames_hw3 = frames_cpu.float().numpy().transpose(0, 2, 3, 1)
            else:
                frames_hw3 = raw["images"].transpose(0, 2, 3, 1)
            if frames_hw3.size and frames_hw3.max() > 1.0:
                dense_colors = frames_hw3.astype(np.uint8)
            else:
                dense_colors = (frames_hw3 * 255.0).astype(np.uint8)
            submap.set_dense_points(dense_points, dense_colors, raw["depth_conf"].astype(np.float32))

        # Query retrieval index for loop candidates against prior submaps
        past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
        loop_matches = find_loop_closures(
            submap,
            past_for_lc,
            cfg.lc_retrieval_threshold,
            cfg.max_loops_per_submap,
            nms_frame_distance=cfg.nms_frame_distance,
        )

        # Verify each candidate on the query and detected frames
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

                    # Pair's dense points (2, H, W, 3) in its frame-0 camera, and conf (2, H, W)
                    # - VGGT-SLAM unprojects depth_lc the same way (solver.py:268)
                    lc_pts = np.asarray(lc_data["world_points"], dtype=np.float32)
                    lc_conf = np.asarray(lc_data["conf"], dtype=np.float32)

                    # Build the loop carrier; see _run_lc_loop
                    # - next free submap_id after windows and loop submaps
                    # - submaps excludes the current window, not yet appended
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
                            points=lc_pts,
                            conf=lc_conf,
                            conf_percentile=cfg.conf_percentile,
                        )
                    )

        return submap, window_lc_submaps, loop_matches

    def _add_window_submap(
        self,
        submap: Submap,
        lc_submaps: list[Submap],
        submaps: list[Submap],
        lc_submaps_acc: list[Submap],
    ) -> None:
        """
        Record a window's submaps, then add its sequential edges to the pose graph.

        - loop edges are added later, per loop_edge_timing
        - driver accumulators: submaps appended, lc_submaps_acc extended, both in place
        """
        submaps.append(submap)
        lc_submaps_acc.extend(lc_submaps)

        # Register the finalized window submap (dense-data carrier) in the scene map
        self.map.add_submap(submap)

        # Add this submap's sequential edges, then optimize
        # - loop edges come later, in _run_lc_loop
        self.graph.add_submap(submap, self.config.submap_overlap)
        self.graph.optimize()

        # Hook 1: push the just-appended submap (latest-only) to the live viewer
        self._viz_push_submap(submap)

    def _add_loop_edge(self, lc: Submap, submaps: list[Submap]) -> None:
        """
        Add one verified loop carrier's edge to self.graph.

        - lc: verified loop carrier (see _run_lc_loop) whose frames come from submaps
        """
        self.graph.add_loop_edge(lc, submaps)

    ####################################################################
    # Live viewer hooks (guarded)
    ####################################################################

    def _viz_push_submap(self, submap: Submap) -> None:
        """
        Push one submap's corrected points and per-frame frusta to the viewer.

        - no-op without a viewer
        - skips loop carriers and submaps without dense points
        """
        if self.viz is None:
            return

        # Viewer I/O, runtime and bad-data failures are logged; programming errors propagate
        try:
            # No dense cloud to draw on loop carriers or degraded submaps
            if submap.is_lc_submap or submap.points is None:
                return

            # Skip overlap frames the previous submap drew; see LoopClosureConfig
            skip = self.config.submap_overlap if submap.frame_start > 0 else 0
            world_pts = submap.get_points_in_world_frame(self.graph, skip_first=skip)

            # A pure-overlap tail submap trims to nothing; leave the scene untouched
            if world_pts.shape[0] == 0:
                return

            # Subsampled point cloud
            all_points = np.ones(len(world_pts), dtype=bool)
            keep = subsample_points(all_points, self.config.viz_max_points)
            cols = submap.get_points_colors(skip_first=skip)
            self.viz.add_points(f"submap_{submap.submap_id}", world_pts[keep], cols[keep])

            # Per-frame frusta at the corrected world-to-cam poses (overlap frames skipped)
            poses = submap.get_all_poses_world(self.graph)
            for i in range(skip, poses.shape[0]):
                self.viz.add_frustum(f"submap_{submap.submap_id}/cam_{i}", poses[i], submap.intrinsics[i])
        except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:
            logger.warning("viewer submap push failed (submap %s): %s", submap.submap_id, e)

    def _viz_reupload_all(self) -> None:
        """
        Re-upload every submap so the scene snaps to the corrected poses.

        - no-op without a viewer; _viz_push_submap skips loop carriers
        """
        if self.viz is None:
            return
        for s in self.map.ordered_submaps_by_key():
            self._viz_push_submap(s)

    def _viz_draw_loops(self, loop_matches: list, submaps: list[Submap]) -> None:
        """
        Draw lines between graph-corrected query and detected camera centers.

        - no-op without a viewer; failures are logged
        - endpoints from the window submaps' get_all_poses_world, the frusta's frame
        - not the carrier's verify poses, which live in the loop pair's local frame
        - only accepted loop_matches are drawn; submaps indexed by submap_id
        """
        if self.viz is None:
            return
        try:
            for m in loop_matches:
                if not m.accepted:
                    continue
                q = submaps[m.query_submap_id]
                d = submaps[m.detected_submap_id]
                q_center = invert_poses(q.get_all_poses_world(self.graph)[m.query_frame_idx])[:3, 3]
                d_center = invert_poses(d.get_all_poses_world(self.graph)[m.detected_frame_idx])[:3, 3]
                segment = np.stack([q_center, d_center])[None]  # (1, 2, 3)
                self.viz.add_lines(f"loop_{m.query_submap_id}_{m.detected_submap_id}", segment)
        except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:
            logger.warning("viewer loop-line failed: %s", e)

    def _run_lc_loop(self) -> None:
        """
        Slide windows over the frames, close loops, then assemble base.outputs.

        - loop carrier: a 2-frame Submap (is_lc_submap) per accepted loop candidate
        - carrier frames: the query frame and its detected frame, jointly forwarded by
          the creator's verify step; poses in the pair's local frame
        - a carrier only adds a loop edge to the pose graph; not added to GraphMap, no cloud
        - carrier ids follow the window submaps; loop_edge_timing sets when edges land
        - DINO-SALAD failing to load sets base.raw_outputs from one full forward pass
        """
        console = Console()

        # Reset the scene map + pose graph so a re-run does not accumulate stale state
        self.map = GraphMap()
        self.graph = PoseGraph()

        # see _lc_assembled in __init__
        self._lc_assembled = False

        # Window geometry: stride = submap_size, matching VGGT-SLAM main.py:109-130 (not K-O)
        cfg = self.config
        K, O = cfg.submap_size, cfg.submap_overlap
        step = max(1, K)
        views = self.base.views
        N = self._n_views()
        device = str(next(self.base.model.parameters()).device)

        # Load DINO-SALAD retrieval extractor; fall back to full-sequence inference if unavailable
        try:
            retrieval_cls = BaseRetrievalExtractor.get("dino-salad")
            retrieval_extractor = retrieval_cls(device=device)
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("DINO-SALAD failed to load (%s) — skipping loop closure", e)
            self.base.raw_outputs = self.base._forward(self.base.model, views)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        # Driver state: window submaps, loop carriers, loop count
        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []
        n_submaps = math.ceil(max(1, N - O) / step)
        loops = 0

        console.log(f"Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})")

        # Slide the window across frames; overlap: see LoopClosureConfig
        with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
            for wi, start in enumerate(range(0, N, step)):
                # Each window = K+O frames, as VGGT-SLAM main.py:109 submap_size+overlapping_window_size
                end = min(start + K + O, N)
                window = views[start:end]

                # Forward + detect (vggt_slam/solver.py:Solver.run_predictions), then record the submaps
                submap, window_lc_submaps, loop_matches = self.run_predictions(
                    window, wi, start, submaps, lc_submaps, retrieval_extractor, console
                )
                self._add_window_submap(submap, window_lc_submaps, submaps, lc_submaps)

                # Live timing: add this window's loop edges now and re-solve
                # - as VGGT-SLAM: loop edges in vggt_slam/solver.py:255-287, then a solve at main.py:117-120
                if cfg.loop_edge_timing == "live" and window_lc_submaps:
                    for lc in window_lc_submaps:
                        self._add_loop_edge(lc, submaps)
                    self.graph.optimize()

                # Hook 2: draw accepted loops and re-upload the whole scene
                # - a loop redistributes error across all prior submaps
                if window_lc_submaps:
                    self._viz_draw_loops(loop_matches, submaps)
                    self._viz_reupload_all()

                # Progress
                loops += len(window_lc_submaps)
                pbar.update(1)
                pbar.set_postfix(loops=loops)
                if end >= N:
                    break

        # Number of accepted loop-closure submaps applied (user-visible summary)
        self.base.n_loops_applied = len(lc_submaps)

        # Add deferred loop edges, then the final solve
        # - live mode already added each edge in the window loop
        if cfg.loop_edge_timing == "deferred":
            for lc in lc_submaps:
                self._add_loop_edge(lc, submaps)
        self.graph.optimize()

        # Hook 3: final PGO done; re-upload so the scene snaps to loop-closed poses
        self._viz_reupload_all()

        # Assemble outputs from the graph-corrected GraphMap cloud; see _lc_assembled in __init__
        self.base.outputs = self._assemble_result(N)
        self._lc_assembled = True

    def _assemble_result(self, n_frames: int) -> PointcloudResult:
        """
        Build a PointcloudResult from the GraphMap dense cloud.

        - VGGT-SLAM correction-at-read: poses and points from the optimized graph
        - one pose and intrinsics row per frame of the full n_frames sequence
        - raises ValueError on an empty cloud or zero model dims
        """
        # Dense world cloud + corrected extrinsics come straight from the graph-corrected map
        points, colors = self.map.get_world_pointcloud(self.graph, overlap=self.config.submap_overlap)
        extrinsics = self.graph.extract_extrinsics(n_frames)

        # Cap the dense cloud to the creator's max_points
        # - to_colmap turns every point into a pycolmap Point3D and runs out of memory
        # - the create_pointcloud cap then binds on nothing
        # - cap before the clean, unlike creator postprocess: SOR over ~1e8 dense points would OOM
        all_points = np.ones(len(points), dtype=bool)
        keep = subsample_points(all_points, self.base.max_points)
        points = points[keep]
        colors = colors[keep]

        # One intrinsics row per global frame; model dims from a dense grid
        # - first submap to cover a frame wins, as for poses
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1))
        assigned = np.zeros(n_frames, dtype=bool)
        model_height = model_width = None
        world_points = depth = confidence = None
        for s in self.map.ordered_submaps_by_key():
            if s.is_lc_submap:
                continue

            # Size the per-pixel arrays from the first dense grid
            if model_height is None and s.points is not None:
                model_height, model_width = int(s.points.shape[1]), int(s.points.shape[2])
                world_points = np.zeros((n_frames, model_height, model_width, 3), dtype=np.float32)
                depth = np.zeros((n_frames, model_height, model_width), dtype=np.float32)
                confidence = np.zeros((n_frames, model_height, model_width), dtype=np.float32)

            grid = s.get_world_grid(self.graph) if s.points is not None else None
            for local_i in range(s.intrinsics.shape[0]):
                g = s.frame_start + local_i
                if not 0 <= g < n_frames or assigned[g]:
                    continue

                intrinsics[g] = s.intrinsics[local_i].astype(np.float32)
                assigned[g] = True

                # Per-pixel world points, depth under the corrected pose, and confidence
                if grid is not None:
                    world_points[g] = grid[local_i]
                    depth[g] = transform_points(grid[local_i], extrinsics[g])[..., 2]
                    confidence[g] = s.conf[local_i]

        # Fail fast on an empty cloud or zero model dims
        # - every submap lacked dense points or was fully confidence-masked
        # - otherwise PointcloudResult's full-res K rescale fails later, less clearly
        if model_height is None or points.shape[0] == 0 or model_width == 0 or model_height == 0:
            raise ValueError(
                "LC produced an empty point cloud — all submaps lacked dense points "
                "or were confidence-masked out (no geometry to assemble a result from)."
            )

        return PointcloudResult(
            points=points,
            colors=colors,
            extrinsics=extrinsics,
            intrinsics=None,
            model_intrinsics=intrinsics,
            image_paths=list(self.base.image_paths),
            original_coords=self.base.original_coords,
            model_width=model_width,
            model_height=model_height,
            confidence=torch.from_numpy(confidence),
            world_points=world_points,
            depth=depth,
        )
