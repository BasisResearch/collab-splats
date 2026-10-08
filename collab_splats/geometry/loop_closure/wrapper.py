"""
Submap loop closure around a feedforward creator's forward pass.

- walkthrough: docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb
- imports pointcloud result types, so geometry exports LoopClosure lazily via __getattr__
- adapted from MIT-SPARK/VGGT-SLAM @ fd3fd218 (BSD-2-Clause): main.py, vggt_slam/solver.py
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

from collab_splats.geometry.bundle_adjustment import (
    BundleAdjustment,
    BundleAdjustmentConfig,
)
from collab_splats.geometry.loop_closure.graph import PoseGraph
from collab_splats.geometry.loop_closure.map import GraphMap
from collab_splats.geometry.loop_closure.matching import LoopMatch, find_loop_closures
from collab_splats.geometry.loop_closure.submap import Submap
from collab_splats.geometry.projection import unproject_frames
from collab_splats.geometry.transforms import (
    extrinsics_to_homogeneous,
    invert_poses,
    transform_points,
)
from collab_splats.localization import BaseRetrievalExtractor
from collab_splats.pointcloud.base import BasePointcloudCreator, PointcloudResult
from collab_splats.pointcloud.utils import subsample_points
from collab_splats.utils.torch_utils import hold_matmul_precision, pytorch_gc, to_numpy

logger = logging.getLogger(__name__)

__all__ = ["LoopClosure", "LoopClosureConfig"]


@dataclasses.dataclass
class LoopClosureConfig:
    """
    Loop-closure settings, grouped by pipeline step.

    - fields are grouped as windows, retrieval, graph and viz

    Args:
        submap_size: window stride; each submap holds this plus submap_overlap frames.
        submap_overlap: frames shared by consecutive windows (1 = VGGT-SLAM); the earlier submap owns them.
        retrieval: retrieval registry name, e.g. "dino-salad" or "megaloc"; megaloc needs its own threshold.
        lc_retrieval_threshold: max L2 between unit descriptors, [0, 2], tuned for dino-salad; <= 0 skips retrieval.
        max_loops_per_submap: candidates kept per query submap, before NMS.
        verify_match_ratio: loop verify threshold; None takes the creator's default_verify_match_ratio.
        nms_frame_distance: suppression radius in frames between candidates on the same detected submap.
        min_submap_gap: newest submaps left out of the retrieval search.
        conf_percentile: depth_conf percentile gating scale points; VGGT-SLAM --conf_threshold 25 (main.py:32).
        loop_edge_timing: deferred adds all loop edges after the window loop; live adds each as found.
        viz_max_points: viewer per-submap point cap.
    """

    # Windows
    submap_size: int = 20
    submap_overlap: int = 1

    # Retrieval
    retrieval: str = "dino-salad"
    lc_retrieval_threshold: float = 0.95
    max_loops_per_submap: int = 5
    verify_match_ratio: float | None = None
    nms_frame_distance: int = 25
    min_submap_gap: int = 1

    # Graph
    conf_percentile: float = 25.0
    loop_edge_timing: Literal["deferred", "live"] = "deferred"

    # Viz
    viz_max_points: int = 50000


########################################################################
# LoopClosure wrapper
########################################################################


class LoopClosure:
    """
    Proxy around a feedforward creator that replaces its inference with the submap LC loop.

    - `run_inference()` is the LC entry point; `create_pointcloud()` reaches it via _reconstruct
    - every other attribute is forwarded to `base`
    - fewer than submap_size frames, or the retrieval model failing to load, falls back to the
      creator's own forward pass

    Quickstart:
        from pathlib import Path
        from collab_splats.geometry import LoopClosure, LoopClosureConfig
        from collab_splats.pointcloud import get_creator
        lc = LoopClosure(get_creator("vggtx")(), config=LoopClosureConfig())
        result = lc.create_pointcloud(Path("scene/images"), Path("scene/out"))
    """

    def __init__(
        self, base: Any, config: LoopClosureConfig | None = None, ba: BundleAdjustmentConfig | None = None
    ) -> None:
        """
        Wrap a creator, resolving the config against its per-model defaults.

        Args:
            base: feedforward creator whose forward pass the LC loop drives.
            config: loop-closure settings; None uses LoopClosureConfig().
            ba: bundle adjustment run inside each window after its forward; None runs none.
        """
        self.base = base
        self.config = config if config is not None else LoopClosureConfig()
        self.ba = ba

        # None resolves to the creator's per-model calibration
        if self.config.verify_match_ratio is None:
            if base.default_verify_match_ratio is None:
                raise NotImplementedError(f"{type(base).__name__} sets no default_verify_match_ratio; LC unsupported")

            self.config = dataclasses.replace(self.config, verify_match_ratio=base.default_verify_match_ratio)

        # Scene state (VGGT-SLAM Solver structure): the submap collection + the pose graph
        self.map = GraphMap()
        self.graph = PoseGraph()

        # Optional live Viewer, set by the driver; None makes every viewer hook a no-op
        self.viz = None

        # Result _run_lc_loop assembles; _reconstruct returns it as is, so base._postprocess cannot overwrite it
        self.outputs: PointcloudResult | None = None

        # Per-window BA records and the focal later windows hold; reset by run_inference
        self.window_ba: list[dict] = []
        self._ba_focal: float | None = None

        # Full-res frame paths for matcher tracks in window BA; set by _reconstruct
        self.frame_paths: list[Path] = []

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

    # The base creator's template: postprocess and export around _reconstruct
    create_pointcloud = BasePointcloudCreator.create_pointcloud

    def _reconstruct(self, paths: list[Path], out_dir: Path) -> PointcloudResult:
        """
        The base creator's steps with the windowed LC forward; the assembled result when LC ran.
        """
        # Base load and setup, then the LC or fallback forward
        self.frame_paths = list(paths)
        self.base.load_model()
        self.base.setup_inference(paths)
        self.run_inference()

        if self.outputs is not None:
            return self.outputs

        return self.base._postprocess(self.base.raw_outputs)

    ####################################################################
    # Inference: LC loop override
    ####################################################################

    def run_inference(self) -> None:
        """
        Run the LC loop, or fall back to one whole-scene forward.

        - LC loop: sets outputs to the assembled PointcloudResult
        - under submap_size frames, or the retrieval model failing to load: one window, refined as window 0
        - the fallback sets base.raw_outputs only and leaves outputs None
        """
        # Fresh per-run state: window BA records, held focal, assembled result
        self.window_ba = []
        self._ba_focal = None
        self.outputs = None
        cfg = self.config

        # LC needs at least submap_size frames; views are a tensor or view dicts, len counts frames for both
        run_lc = len(self.base.views) >= cfg.submap_size

        # Load the retrieval model only when a candidate can pass; dist < threshold admits none at threshold <= 0
        retrieval_extractor = None

        if run_lc and cfg.lc_retrieval_threshold > 0:
            device = str(next(self.base.model.parameters()).device)

            try:
                retrieval_cls = BaseRetrievalExtractor.get(cfg.retrieval)
                retrieval_extractor = retrieval_cls(device=device)
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning("Retrieval %s failed to load (%s) — skipping loop closure", cfg.retrieval, e)
                run_lc = False

        # LC loop, else the whole-scene fallback below
        if run_lc:
            self._run_lc_loop(retrieval_extractor)
            return

        # Fallback: one whole-scene forward
        self.base.raw_outputs, _ = self._forward_window(self.base.views, 0)
        pytorch_gc()

    def run_predictions(
        self,
        window: Any,
        raw: dict,
        rgb01: np.ndarray,
        wi: int,
        start: int,
        submaps: list[Submap],
        lc_submaps: list[Submap],
        retrieval_extractor: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[Submap, list[Submap], list[LoopMatch], np.ndarray]:
        """
        Build one forwarded window's Submap, then find and verify loop candidates.

        - dense points are unprojected here; the caller attaches them with colors and conf

        Args:
            window: the window's model inputs (tensor or list of view dicts).
            raw: the window's forward outputs.
            rgb01: the window's model-grid RGB in [0, 1], (k, 3, H, W); see _forward_window.
            wi: window index, used as the new submap_id.
            start: global index of the window's first frame.
            submaps: window submaps built so far.
            lc_submaps: loop submaps accepted so far.
            retrieval_extractor: retrieval extractor for the window's descriptors; None skips retrieval.

        Returns:
            (submap, window_lc_submaps, loop_matches, dense_points): the window's Submap, its
            verified loop carriers (see _run_lc_loop), every post-NMS candidate, and its
            (k, H, W, 3) frame-0 points, to attach before recording the submap.
        """
        # Window frame range: window == views[start:end]
        cfg = self.config
        end = start + len(window)

        # Window (k, 4, 4) poses; the graph reads each submap's local frame as frame 0's camera
        ext_3x4 = raw["extrinsic"]
        poses_4x4 = extrinsics_to_homogeneous(ext_3x4)
        pose0 = poses_4x4[0]

        if not np.allclose(pose0, np.eye(4), atol=0.1):
            raise ValueError(f"Window poses not normalized to frame 0: expected poses[0] ≈ eye(4), got\n{pose0}")

        intrinsics = raw["intrinsics"]

        # Model inputs for loop verify: the window tensor, or MapAnything's dinov2-normalized view dicts
        if isinstance(window, torch.Tensor):
            frames_cpu = window.cpu()
        else:
            frames_cpu = torch.cat([v["img"].cpu() for v in window], dim=0)

        # Drop raw's reference to the forward's images; rgb01 holds what the window needs
        raw.pop("images", None)

        # Retrieval descriptors on [0, 1] RGB, then the window's Submap; None when retrieval is off
        retrieval_frames = torch.from_numpy(rgb01)
        ret_vecs = retrieval_extractor(retrieval_frames) if retrieval_extractor is not None else None  # (k, D)
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

        # Dense per-pixel points (fat Submap, VGGT-SLAM data path): full-res depth unprojection
        depth = raw["depth"].reshape(raw["depth"].shape[:3])
        dense_points = unproject_frames(depth, ext_3x4, intrinsics)

        # Query retrieval index for loop candidates against prior submaps; none when retrieval is off
        loop_matches = []

        if retrieval_extractor is not None:
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
            pair = f"submap {match.query_submap_id} → {match.detected_submap_id}"
            q_frame = frames_cpu[match.query_frame_idx]
            d_submap = submaps[match.detected_submap_id]
            d_frame = d_submap.frames[match.detected_frame_idx]
            verify_ok, lc_data = self.base._verify_loop_candidate(
                q_frame, d_frame, verify_match_ratio=cfg.verify_match_ratio
            )

            if not verify_ok:
                logger.info("Loop rejected (verify ratio): %s dist=%.3f", pair, match.similarity_score)
                continue

            # Reject a loop whose relative pose is not finite
            lc_poses = lc_data["poses"]
            q_pose = lc_poses[0].astype(np.float64)
            d_pose = lc_poses[1].astype(np.float64)
            lc_rel = invert_poses(d_pose) @ q_pose
            lc_rel_f32 = lc_rel.astype(np.float32)

            if not np.isfinite(lc_rel_f32).all():
                logger.info("Loop rejected (non-finite pose): %s", pair)
                continue

            match.accepted = True
            logger.info("Loop accepted: %s dist=%.3f", pair, match.similarity_score)

            # Loop carrier with the pair's (2, H, W, 3) frame-0 points and conf (solver.py:268); see _run_lc_loop
            q_i, d_i = match.query_frame_idx, match.detected_frame_idx
            window_lc_submaps.append(
                Submap(
                    submap_id=len(submaps) + len(lc_submaps) + len(window_lc_submaps),
                    poses=lc_poses,
                    intrinsics=np.stack([submap.intrinsics[q_i], d_submap.intrinsics[d_i]]),
                    image_paths=[submap.image_paths[q_i], d_submap.image_paths[d_i]],
                    points=np.asarray(lc_data["world_points"], dtype=np.float32),
                    conf=np.asarray(lc_data["conf"], dtype=np.float32),
                    conf_percentile=cfg.conf_percentile,
                )
            )

        return submap, window_lc_submaps, loop_matches, dense_points

    def _forward_window(self, window: Any, start: int) -> tuple[dict, np.ndarray]:
        """
        One window's forward under no_grad, then its bundle adjustment when ba is set.

        - returns (raw, rgb01): rgb01 is the model-grid RGB in [0, 1], (k, 3, H, W) float32
        - rgb01: the window tensor, or MapAnything's raw images (its views are dinov2-normalized)
        - runs on the pipeline worker thread; no_grad is thread-local, so it is entered here
        - BA enters enable_grad itself; worker order makes window 0's focal known before window 1's solve
        - the first refined window's focal is held fixed in every later window (refine_focal only)
        - a solve raising ValueError keeps the feedforward poses; its record has ok False
        """
        # Forward under the precision lock and no_grad
        with hold_matmul_precision(), torch.no_grad():
            raw = self.base._forward(self.base.model, window)

            # Model-grid RGB in [0, 1] for BA, retrieval and colors
            if isinstance(window, torch.Tensor):
                window_f32 = window.float()
                rgb01 = to_numpy(window_f32)
            else:
                rgb01 = raw["images"]

            # No BA configured: forward only
            if self.ba is None:
                return raw, rgb01

            # Peak GPU memory counted from here
            t0 = time.perf_counter()
            k = raw["extrinsic"].shape[0]

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Hold the first refined window's focal
            intrinsics = raw["intrinsics"].copy()
            cfg = self.ba

            if self._ba_focal is not None:
                intrinsics[:, 0, 0] = self._ba_focal
                intrinsics[:, 1, 1] = self._ba_focal
                cfg = dataclasses.replace(cfg, refine_focal=False)

            # One track cache per window
            if cfg.tracks_cache_dir is not None:
                cache_dir = Path(cfg.tracks_cache_dir) / f"w{start:06d}"
                cfg = dataclasses.replace(cfg, tracks_cache_dir=cache_dir)

            # Solve inputs: (k, H, W) depth, world points under the solve's K, 4x4 poses
            depth = raw["depth"].reshape(raw["depth"].shape[:3])
            world_points = unproject_frames(depth, raw["extrinsic"], intrinsics)
            poses = extrinsics_to_homogeneous(raw["extrinsic"])
            confidence = torch.from_numpy(raw["depth_conf"])

            # A matcher track source without frames is a config error, not a failed window
            if cfg.track_source != "vggsfm" and not self.frame_paths:
                raise ValueError(f"window BA: track_source {cfg.track_source!r} needs frame_paths")

            # Solve; a ValueError keeps the feedforward poses
            ba = BundleAdjustment(cfg)
            ok = True

            try:
                refined, refined_intrinsics = ba.refine(
                    rgb01,
                    confidence,
                    world_points,
                    poses,
                    intrinsics,
                    depth=depth,
                    frame_paths=self.frame_paths[start : start + k] or None,
                )
            except ValueError as e:
                logger.warning("Window BA at frame %d failed (%s); keeping feedforward poses", start, e)
                ok = False

            # Back to frame 0's camera, as run_predictions expects
            if ok:
                refined = refined.astype(np.float64)
                frame0_inv = invert_poses(refined[:1])[0]
                local = refined @ frame0_inv
                raw["extrinsic"] = local[:, :3, :].astype(np.float32)
                raw["intrinsics"] = refined_intrinsics.astype(np.float32)

                # Window 0's refined focal is held by every later window
                if self._ba_focal is None and self.ba.refine_focal:
                    self._ba_focal = float(refined_intrinsics[0, 0, 0])

            # Window record
            record = {
                "start": start,
                "n_frames": k,
                "ok": ok,
                "alignment_scale": ba.alignment_scale,
                "loss_final": ba.loss_history[-1][-1] if ba.loss_history and ba.loss_history[-1] else None,
                "focal": float(raw["intrinsics"][0, 0, 0]),
                "seconds": time.perf_counter() - t0,
                "gpu_max_mib": torch.cuda.max_memory_allocated() >> 20 if torch.cuda.is_available() else None,
            }
            self.window_ba.append(record)
            logger.info(
                "Window BA at frame %d: ok %s, focal %.1f, %.0f s", start, ok, record["focal"], record["seconds"]
            )
            return raw, rgb01

    ####################################################################
    # Live viewer hooks (guarded)
    ####################################################################

    def _viz_push_submap(self, submap: Submap) -> None:
        """
        Push one submap's corrected points and per-frame frusta to the viewer.

        - no-op without a viewer; called on window submaps only, which always carry dense points
        """
        if self.viz is None:
            return

        # Viewer I/O, runtime and bad-data failures are logged; programming errors propagate
        try:
            # Skip overlap frames the previous submap drew; see LoopClosureConfig
            skip = self.config.submap_overlap if submap.frame_start > 0 else 0

            # Graph-corrected world points and colors past the overlap, masked by confidence
            grid = submap.get_world_grid(self.graph)[skip:]
            mask = submap.conf[skip:] > submap.conf_threshold
            world_pts = grid[mask]
            cols = submap.colors[skip:][mask]

            # A pure-overlap tail submap trims to nothing; leave the scene untouched
            if world_pts.shape[0] == 0:
                return

            # Subsampled point cloud
            all_points = np.ones(len(world_pts), dtype=bool)
            keep = subsample_points(all_points, self.config.viz_max_points)
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

        - no-op without a viewer; the map holds window submaps only, never loop carriers
        """
        if self.viz is None:
            return

        # Push every window submap
        for s in self.map.ordered_submaps_by_key():
            self._viz_push_submap(s)

    def _run_lc_loop(self, retrieval_extractor: Callable[[torch.Tensor], torch.Tensor] | None) -> None:
        """
        Slide windows over the frames, close loops, then assemble outputs.

        - loop carrier: the verified (query, detected) frame pair as a 2-frame Submap; adds a loop edge only
        - carrier ids follow the window submaps; loop_edge_timing sets when edges land
        - with a viewer, each push waits for the running forward (precision lock)
        """
        # Reset the scene map + pose graph so a re-run does not accumulate stale state
        self.map = GraphMap()
        self.graph = PoseGraph()

        # Window geometry: stride = submap_size, matching VGGT-SLAM main.py:109-130 (not K-O)
        cfg = self.config
        K, O = cfg.submap_size, cfg.submap_overlap
        views = self.base.views
        N = len(views)

        # Driver state: window submaps and loop carriers
        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []

        # Window bounds: K+O frames each, as VGGT-SLAM main.py:109; overlap: see LoopClosureConfig
        bounds = []

        for start in range(0, N, K):
            end = min(start + K + O, N)
            bounds.append((start, end))

            if end >= N:
                break

        # Submap count for the log and the bar
        n_submaps = len(bounds)
        logger.info("Loop closure: %d frames → %d submaps (size=%d, overlap=%d)", N, n_submaps, K, O)

        # Slice each window once; the forward submits and the loop share it
        windows = [views[s:e] for s, e in bounds]

        # Window k+1's forward runs on one worker while window k's CPU post runs here
        with ThreadPoolExecutor(1) as pool, tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
            pending = pool.submit(self._forward_window, windows[0], bounds[0][0])

            for wi, ((start, _), window) in enumerate(zip(bounds, windows)):
                raw, rgb01 = pending.result()

                # GPU post before launch: unproject takes the precision lock the next forward holds
                submap, window_lc_submaps, loop_matches, dense_points = self.run_predictions(
                    window, raw, rgb01, wi, start, submaps, lc_submaps, retrieval_extractor
                )

                # Launch the next window's forward, then the CPU post overlaps it
                if wi + 1 < len(windows):
                    pending = pool.submit(self._forward_window, windows[wi + 1], bounds[wi + 1][0])

                # Dense uint8 colors: VGGT-SLAM scales [0, 1] frames by 255; some backends already give [0, 255]
                frames_hw3 = rgb01.transpose(0, 2, 3, 1)

                if frames_hw3.size and frames_hw3.max() > 1.0:
                    dense_colors = frames_hw3.astype(np.uint8)
                else:
                    dense_colors = (frames_hw3 * 255.0).astype(np.uint8)

                submap.set_dense_points(dense_points, dense_colors, raw["depth_conf"].astype(np.float32))

                # Record the submaps, add the sequential edges and optimize; loop edges land per loop_edge_timing
                submaps.append(submap)
                lc_submaps.extend(window_lc_submaps)
                self.map.add_submap(submap)
                self.graph.add_submap(submap, cfg.submap_overlap)
                self.graph.optimize()

                # Hook 1: push the just-appended submap to the live viewer
                self._viz_push_submap(submap)

                # Live timing: add this window's loop edges and re-solve, as VGGT-SLAM solver.py:255-287
                if cfg.loop_edge_timing == "live" and window_lc_submaps:
                    for lc in window_lc_submaps:
                        self.graph.add_loop_edge(lc)

                    self.graph.optimize()

                # Progress
                pbar.update(1)
                pbar.set_postfix(loops=len(lc_submaps))

                # Hook 2 runs only on a window that closed a loop, with a viewer attached
                if not (window_lc_submaps and self.viz is not None):
                    continue

                # Loop lines between graph-corrected camera centers of each accepted match; failures logged
                try:
                    for m in loop_matches:
                        if not m.accepted:
                            continue

                        q_pose = submaps[m.query_submap_id].get_all_poses_world(self.graph)[m.query_frame_idx]
                        d_pose = submaps[m.detected_submap_id].get_all_poses_world(self.graph)[m.detected_frame_idx]
                        q_center = invert_poses(q_pose)[:3, 3]
                        d_center = invert_poses(d_pose)[:3, 3]
                        segment = np.stack([q_center, d_center])[None]  # (1, 2, 3)
                        self.viz.add_lines(f"loop_{m.query_submap_id}_{m.detected_submap_id}", segment)
                except (ValueError, ZeroDivisionError, RuntimeError, OSError) as e:
                    logger.warning("viewer loop-line failed: %s", e)

                # A loop moves every prior submap, so re-upload the whole scene
                self._viz_reupload_all()

        # One cache release after the loop
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Add deferred loop edges (live mode added them in the loop), then the final solve
        if cfg.loop_edge_timing == "deferred":
            for lc in lc_submaps:
                self.graph.add_loop_edge(lc)

        self.graph.optimize()

        # Hook 3: final PGO done; re-upload so the scene snaps to loop-closed poses
        self._viz_reupload_all()

        # Assemble outputs from the graph-corrected GraphMap cloud; see outputs in __init__
        self.outputs = self._assemble_result(N)

    def _assemble_result(self, n_frames: int) -> PointcloudResult:
        """
        Build a PointcloudResult from the GraphMap dense cloud.

        - VGGT-SLAM correction-at-read: poses and points from the optimized graph
        - one pose and intrinsics row per frame of the full n_frames sequence
        - each submap is lifted once; the cloud keeps only the capped draw, never the dense stack
        - world_points is None: the lean store re-derives it from depth
        - raises ValueError on an empty cloud or zero model dims
        """
        # Overlap skip and per-submap kept-pixel table
        overlap = self.config.submap_overlap
        kept = {}

        # Confidence-kept pixels of each window submap, as _viz_push_submap masks them
        for s in self.map.ordered_submaps_by_key():
            skip = overlap if s.frame_start > 0 else 0
            mask = s.conf[skip:] > s.conf_threshold
            kept[s.submap_id] = (skip, mask, int(mask.sum()))

        # Draw the max_points cap over the whole cloud before lifting; to_colmap and SOR OOM on it
        total = sum(count for _, _, count in kept.values())
        all_kept = np.ones(total, dtype=bool)
        keep = subsample_points(all_kept, self.base.max_points)

        # Split the draw into each submap's share, in submap order
        share = {}
        a = 0

        for submap_id, (skip, mask, count) in kept.items():
            b = a + count
            share[submap_id] = (skip, mask, keep[a:b])
            a = b

        # One pose and intrinsics row per global frame, identity where no submap covers it
        extrinsics = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
        intrinsics = np.tile(np.eye(3, dtype=np.float32), (n_frames, 1, 1))
        assigned = np.zeros(n_frames, dtype=bool)
        model_height = model_width = None
        depth = confidence = None
        pts_chunks, col_chunks = [], []

        for s in self.map.ordered_submaps_by_key():
            # Lift each submap once; keep only its share of the capped draw
            skip, mask, sub_keep = share[s.submap_id]
            grid = s.get_world_grid(self.graph)

            # Flat pixel index of each kept point: one gather, not a mask copy then a subsample copy
            idx = np.flatnonzero(mask)[sub_keep]
            grid_flat = grid[skip:].reshape(-1, 3)
            colors_flat = s.colors[skip:].reshape(-1, 3)
            pts_chunks.append(grid_flat[idx])
            col_chunks.append(colors_flat[idx])

            # Size the per-pixel arrays from the first grid
            if model_height is None:
                model_height, model_width = int(s.points.shape[1]), int(s.points.shape[2])
                depth = np.zeros((n_frames, model_height, model_width), dtype=np.float32)
                confidence = np.zeros((n_frames, model_height, model_width), dtype=np.float32)

            # First submap to cover a frame wins: the overlap frame keeps the earlier submap's pose, as evo
            poses_world = s.get_all_poses_world(self.graph)

            for local_i in range(s.intrinsics.shape[0]):
                g = s.frame_start + local_i

                if assigned[g]:
                    continue

                extrinsics[g] = poses_world[local_i]
                intrinsics[g] = s.intrinsics[local_i].astype(np.float32)
                assigned[g] = True

                # Depth under the corrected pose, and confidence
                depth[g] = transform_points(grid[local_i], extrinsics[g])[..., 2]
                confidence[g] = s.conf[local_i]

        # Stack the kept points; an empty map gives empty arrays
        points = np.vstack(pts_chunks) if pts_chunks else np.zeros((0, 3), dtype=np.float32)
        colors = np.vstack(col_chunks) if col_chunks else np.zeros((0, 3), dtype=np.uint8)

        # Fail fast on an empty cloud or zero model dims, before PointcloudResult's K rescale fails less clearly
        if model_height is None or points.shape[0] == 0 or model_width == 0 or model_height == 0:
            raise ValueError(
                "LC produced an empty point cloud — every pixel was confidence-masked out "
                "(no geometry to assemble a result from)."
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
            world_points=None,
            depth=depth,
        )
