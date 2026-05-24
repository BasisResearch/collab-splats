from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BasePointcloudCreator, PointcloudResult
from .bundle_adjustment import BundleAdjustmentConfig, _extract_tracks_vggsfm, _run_bundle_adjustment
from .feedforward import FeedforwardResult
from .loop_closure import LoopClosureConfig

__all__ = ["BundleAdjustment", "LoopClosure"]


class BundleAdjustment(BasePointcloudCreator):
    """Wrapper that runs bundle adjustment after any feedforward creator (or LoopClosure).

    Expects the wrapped creator to populate FeedforwardResult.images/conf/world_points.
    """

    def __init__(
        self,
        base: Any,  # BaseFeedforwardCreator | LoopClosure — duck-typed
        config: BundleAdjustmentConfig | None = None,
    ) -> None:
        self.base = base
        self.config = config or BundleAdjustmentConfig()

    def refine(self, result: FeedforwardResult, output_dir: Path) -> PointcloudResult:
        """Run BA on a pre-computed FeedforwardResult without re-running inference.

        Works for any backend (VGGTX, VGGTOmega, MapAnything). Requires
        result.images, result.conf, result.world_points, and result.pixel_indices
        to be populated — all are stored by save_zarr() and restored by load_zarr()
        (images must be loaded separately from the zarr store and set on the result).

        Args:
            result:     Pre-computed result with images, conf, world_points populated.
            output_dir: Directory to write the COLMAP reconstruction.

        Returns:
            PointcloudResult with bundle-adjusted poses and point cloud.
        """
        output_dir = Path(output_dir)
        # Set result on base so _apply_ba and build_colmap can access it.
        # raw_outputs is not needed by _apply_ba (backend-agnostic reprojection uses
        # world_points), but the dedup-rows LC path reads it as a dict — provide empty dict.
        self.base.outputs = result
        self.base.raw_outputs = {}
        self._apply_ba()
        return self.base.build_colmap(output_dir)

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        import torch

        image_dir, output_dir = Path(image_dir), Path(output_dir)
        self.base.load_model()
        self.base.setup_inference(image_dir)
        self.base.run_inference()
        self.base.postprocess()
        # Offload the backbone model from GPU before running track extraction + BA.
        # Both BA stages load their own GPU models (ALIKED/SP keypoints + Warp
        # tracker) and the backbone (~8 GB for VGGT-1B) would push the combined
        # footprint past the container memory cap.
        # _reproject_ba only uses raw_outputs (CPU numpy), not self.model,
        # so the backbone need not be on GPU during _apply_ba.
        # When self.base is LoopClosure, the actual model lives at self.base.base.
        _backbone = getattr(self.base, "model", None) or getattr(
            getattr(self.base, "base", None), "model", None
        )
        _backbone_owner = None
        if _backbone is not None and torch.cuda.is_available():
            _backbone_owner = (
                self.base if hasattr(self.base, "model") else self.base.base
            )
            _backbone_owner.model = _backbone.to("cpu")
            torch.cuda.empty_cache()
        self._apply_ba()
        return self.base.build_colmap(output_dir)

    @property
    def outputs(self) -> Any:
        return self.base.outputs

    @outputs.setter
    def outputs(self, value: Any) -> None:
        self.base.outputs = value

    @property
    def raw_outputs(self) -> Any:
        return self.base.raw_outputs

    @raw_outputs.setter
    def raw_outputs(self, value: Any) -> None:
        self.base.raw_outputs = value

    def _apply_ba(self) -> None:
        result: FeedforwardResult = self.base.outputs
        if result.images is None:
            raise ValueError(
                f"{type(self.base).__name__} did not populate FeedforwardResult.images. "
                "Creator's _postprocess() must always set images/conf/world_points."
            )

        N = result.extrinsics.shape[0]
        images       = result.images
        conf         = result.conf
        world_points = result.world_points
        intrinsics   = result.intrinsics

        # When base is LoopClosure (windowed), merge_submap_outputs produces M-expanded
        # arrays (M >= N due to overlap frames).  extrinsics is already deduped to N via
        # extrinsic_global_4x4, but images/conf/world_points/intrinsics still have M rows.
        # _dedup_rows[g] = first M-row index corresponding to global frame g → slice to N.
        _raw = getattr(self.base, "raw_outputs", None) or getattr(
            getattr(self.base, "base", None), "raw_outputs", None
        )
        dedup = _raw.get("_dedup_rows") if isinstance(_raw, dict) else None
        if dedup is not None:
            if intrinsics is not None and intrinsics.shape[0] != N:
                intrinsics = intrinsics[dedup]
            if images is not None and hasattr(images, "__len__") and len(images) != N:
                images = images[dedup]
            if conf is not None and hasattr(conf, "__len__") and len(conf) != N:
                conf = conf[dedup]
            if world_points is not None and world_points.shape[0] != N:
                world_points = world_points[dedup]

        cfg_dict = dataclasses.asdict(self.config)
        track_params = {
            "max_query_pts": cfg_dict.pop("max_query_pts"),
            "query_frame_num": cfg_dict.pop("query_frame_num"),
        }

        # Extract 2D tracks across frames — VGGSfM tracker requires square input (padded in extract_tracks_vggsfm)
        tracks, vis_scores, pts3d_kp = _extract_tracks_vggsfm(
            images, conf, world_points,
            **track_params,
        )
        extrinsics_3x4 = result.extrinsics[:, :3, :]  # (N, 3, 4)

        # Run Levenberg-Marquardt BA to refine poses and 3D points
        _, refined_ext_3x4, refined_intr = _run_bundle_adjustment(
            pts3d_kp,
            extrinsics_3x4,
            intrinsics,
            tracks,
            vis_scores,
            image_size=(result.model_height, result.model_width),
            **cfg_dict,
        )

        if result.pixel_indices is not None and result.world_points is not None:
            # Backend-agnostic reprojection: world_points → camera frame (old poses) →
            # world frame (new poses).  Preserves pixel_indices so the point set stays
            # index-aligned with pre-BA colors and features.
            fi = result.pixel_indices[:, 0]
            ri = result.pixel_indices[:, 1]
            ci = result.pixel_indices[:, 2]
            old_pts = result.world_points[fi, ri, ci].astype(np.float64)   # (P, 3)
            ext0 = result.extrinsics[:, :3, :]                              # (N, 3, 4)
            R0 = ext0[fi, :, :3].astype(np.float64)                        # (P, 3, 3)
            t0 = ext0[fi, :, 3].astype(np.float64)                         # (P, 3)
            pts3d_cam = np.einsum("pij,pj->pi", R0, old_pts) + t0          # (P, 3)
            R1 = refined_ext_3x4[fi, :, :3].astype(np.float64)             # (P, 3, 3)
            t1 = refined_ext_3x4[fi, :, 3].astype(np.float64)              # (P, 3)
            pts3d = np.einsum("pij,pj->pi", R1.transpose(0, 2, 1), pts3d_cam - t1).astype(np.float32)
            colors = result.colors
        else:
            pts3d, colors = self.base._reproject(
                self.base.raw_outputs, refined_ext_3x4, refined_intr
            )

        # Write refined poses and intrinsics back into the feedforward result
        n = refined_ext_3x4.shape[0]
        bottom = np.tile([[0, 0, 0, 1]], (n, 1, 1)).astype(np.float32)
        refined_ext_4x4 = np.concatenate([refined_ext_3x4, bottom], axis=1)

        self.base.outputs = dataclasses.replace(
            result,
            pts3d=pts3d,
            colors=colors,
            extrinsics=refined_ext_4x4,
            intrinsics=refined_intr,
        )


########################################################
########## LoopClosure wrapper ########################
########################################################

class LoopClosure:
    """Proxy wrapper that runs the submap loop-closure pipeline around any feedforward creator.

    Forwards all non-inference methods to ``base`` and overrides ``run_inference()``
    with the full LC loop. ``BundleAdjustment`` can wrap ``LoopClosure`` transparently
    because both duck-type the same ``BaseFeedforwardCreator`` interface.
    """

    def __init__(self, base: Any, config: LoopClosureConfig | None = None) -> None:
        self.base = base
        self.config = config or LoopClosureConfig()

    ######################################################
    ########## Delegation — proxy to self.base ##########
    ######################################################

    def load_model(self) -> None:
        self.base.load_model()

    def setup_inference(self, image_dir: Path) -> None:
        self.base.setup_inference(image_dir)

    def postprocess(self, **kwargs: Any) -> None:
        self.base.postprocess(**kwargs)

    def build_colmap(self, output_dir: Path) -> PointcloudResult:
        return self.base.build_colmap(output_dir)

    def _reproject(self, raw_outputs: Any, ext: Any, intr: Any) -> Any:
        """Delegate _reproject to base creator."""
        return self.base._reproject(raw_outputs, ext, intr)

    @property
    def outputs(self) -> Any:
        return self.base.outputs

    @outputs.setter
    def outputs(self, value: Any) -> None:
        self.base.outputs = value

    @property
    def raw_outputs(self) -> Any:
        return self.base.raw_outputs

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

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

    def _run_lc_loop(self, **kwargs: Any) -> None:
        import logging

        import torch
        from rich.console import Console
        from tqdm.auto import tqdm

        from collab_splats.pointcloud.loop_closure import Submap
        from collab_splats.pointcloud.loop_closure.closure import find_loop_closures
        from collab_splats.pointcloud.loop_closure.submap import assert_world_to_cam

        console = Console()

        cfg = self.config
        K, O = cfg.submap_size, cfg.submap_overlap
        step = max(1, K - O)
        views = self.base.views
        N = views.shape[0] if hasattr(views, "shape") else len(views)
        device = str(next(self.base.model.parameters()).device)

        # Helper used inside the loop (mirrors feedforward._raw_to_world_points)
        def _raw_to_world_points(raw: dict) -> tuple:
            from collab_splats.pointcloud.feedforward import _raw_to_world_points as _fn
            return _fn(raw)

        # Load DINO-SALAD retrieval extractor; fall back to full-sequence inference if unavailable
        try:
            from collab_splats.pointcloud.localization import BaseRetrievalExtractor
            retrieval_cls = BaseRetrievalExtractor.get("dino-salad")
            retrieval_extractor = retrieval_cls(device=device)
            self.base._lc_retrieval = retrieval_extractor
        except Exception as e:
            logging.getLogger(__name__).warning(
                "DINO-SALAD failed to load (%s) — skipping loop closure", e
            )
            self.base.raw_outputs = self.base._forward(self.base.model, views, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        submaps: list[Submap] = []
        lc_submaps: list[Submap] = []
        all_loop_candidates: list = []
        n_submaps = math.ceil(max(1, N - O) / step)
        loops_found = 0
        verified = 0

        console.log(f"Loop closure: {N} frames → {n_submaps} submaps (size={K}, overlap={O})")

        # Slide submap window across frames (stride = submap_size - overlap)
        with tqdm(total=n_submaps, desc="Loop closure", unit="submap") as pbar:
            for wi, start in enumerate(range(0, N, step)):
                end = min(start + K, N)
                window = views[start:end]
                k = window.shape[0] if hasattr(window, "shape") else len(window)

                # Run feedforward inference on this submap's frames
                with torch.no_grad():
                    raw = self.base._forward(self.base.model, window, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                ext_3x4 = raw["extrinsic"]                                               # (k, 3, 4)
                bottom = np.tile([0, 0, 0, 1], (k, 1)).reshape(k, 1, 4).astype(np.float32)
                poses_4x4 = np.concatenate([ext_3x4, bottom], axis=1)                    # (k, 4, 4)

                assert_world_to_cam(poses_4x4)

                intr_key = "intrinsics" if "intrinsics" in raw else "intrinsic"
                intrinsics = raw.get(intr_key, np.tile(np.eye(3), (k, 1, 1)).astype(np.float32))

                frames_cpu = window.cpu() if hasattr(window, "cpu") else torch.zeros(k, 3, 1, 1)
                ret_vecs = retrieval_extractor(frames_cpu)                                # (k, D)

                wp, wp_conf = _raw_to_world_points(raw)
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

                # Query retrieval index for loop candidates against prior submaps
                past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
                loop_matches = find_loop_closures(
                    submap, past_for_lc, cfg.lc_threshold_l2, cfg.max_loops_per_submap,
                    nms_frame_distance=cfg.nms_frame_distance,
                )

                for match in loop_matches:
                    q_frame = frames_cpu[match.query_frame_idx]
                    d_submap = submaps[match.detected_submap_id]
                    d_frame = d_submap.frames[match.detected_frame_idx]
                    verify_ok, lc_poses = self.base._verify_loop_candidate(
                        q_frame, d_frame, verify_match_ratio=cfg.verify_match_ratio
                    )
                    if not verify_ok:
                        console.log(
                            f"  ✗ Loop rejected (verify ratio): "
                            f"submap {match.query_submap_id} → {match.detected_submap_id}"
                            f"  dist={match.similarity_score:.3f}"
                        )
                    if verify_ok and lc_poses is None:
                        console.log(
                            f"  ✗ Loop skipped (no joint poses): "
                            f"submap {match.query_submap_id} → {match.detected_submap_id}"
                        )
                        verify_ok = False
                    if verify_ok:
                        from collab_splats.pointcloud.loop_closure.closure import translation_jump_check
                        lc_rel = (
                            np.linalg.inv(lc_poses[1].astype(np.float64))
                            @ lc_poses[0].astype(np.float64)
                        ).astype(np.float32)
                        jump_ok, jump_ratio = translation_jump_check(
                            submaps + [submap],
                            query_idx=len(submaps),
                            query_frame=match.query_frame_idx,
                            detected_idx=match.detected_submap_id,
                            detected_frame=match.detected_frame_idx,
                            lc_relative_pose=lc_rel,
                        )
                        if not jump_ok:
                            console.log(
                                f"  ✗ Loop rejected (jump ratio={jump_ratio:.2f}): "
                                f"submap {match.query_submap_id} → {match.detected_submap_id}"
                            )
                        else:
                            match.accepted = True
                            verified += 1
                            loops_found += 1
                            console.log(
                                f"  ↩ Loop: submap {match.query_submap_id} → {match.detected_submap_id}"
                                f"  dist={match.similarity_score:.3f} jump={jump_ratio:.2f}"
                            )
                            lc_submaps.append(Submap(
                                submap_id=len(submaps) + len(lc_submaps),
                                frames=torch.stack([q_frame, d_frame]),
                                poses=lc_poses,
                                intrinsics=np.stack([
                                    submap.intrinsics[match.query_frame_idx],
                                    d_submap.intrinsics[match.detected_frame_idx],
                                ]),
                                retrieval_vectors=torch.zeros(2, ret_vecs.shape[-1]),
                                image_paths=[
                                    submap.image_paths[match.query_frame_idx],
                                    d_submap.image_paths[match.detected_frame_idx],
                                ],
                                is_lc_submap=True,
                            ))
                    all_loop_candidates.append(match)

                submaps.append(submap)
                pbar.update(1)
                pbar.set_postfix(loops=loops_found, verified=verified)
                if end >= N:
                    break

        # Inspection-only state (not stable API — see feedforward.py docstring)
        self.base._lc_submaps = submaps
        self.base._lc_loop_submaps = lc_submaps
        self.base._lc_overlap_frames = O
        self.base._lc_all_matches = all_loop_candidates

        # Merge per-submap world_points and poses into unified outputs
        t0_pg = time.perf_counter()
        from collab_splats.pointcloud.loop_closure.closure import (
            merge_submap_outputs,
            run_pose_graph_optimization,
        )
        corrected_extrinsics = run_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=cfg.submap_overlap,
        )
        console.log(
            f"  Pose graph: {N} frames, {len(lc_submaps)} loop edges → "
            f"{time.perf_counter() - t0_pg:.1f}s"
        )
        self.base.raw_outputs = merge_submap_outputs(
            submaps, corrected_extrinsics,
        )
