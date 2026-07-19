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
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from rich.console import Console
from tqdm.auto import tqdm

from collab_splats.geometry.transforms import extrinsics_to_homogeneous, invert_poses
from collab_splats.localization import BaseRetrievalExtractor
from collab_splats.pointcloud.base import PointcloudResult
from collab_splats.pointcloud.feedforward import FeedforwardResult, _raw_to_world_points

from .graph import run_pose_graph_optimization
from .matching import find_loop_closures, translation_jump_check
from .merge import merge_submap_outputs
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

    @property
    def lc_threshold_l2(self) -> float:
        """L2 threshold passed to find_loop_closures. Alias for lc_retrieval_threshold."""
        return self.lc_retrieval_threshold


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

    def reconstruct(self, image_dir: Path, output_dir: Path) -> PointcloudResult:
        image_dir, output_dir = Path(image_dir), Path(output_dir)
        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        return self.build_colmap(output_dir)

    def run(self, image_dir: Path) -> FeedforwardResult:
        """Run inference pipeline without COLMAP; return FeedforwardResult."""
        image_dir = Path(image_dir)
        self.load_model()
        self.setup_inference(image_dir)
        self.run_inference()
        self.postprocess()
        result = self.base.outputs

        # LC merge_submap_outputs may produce M-row merged arrays where M != N unique frames.
        # _dedup_rows maps merged rows → unique frame indices so BA sees consistent shapes.
        raw = self.base.raw_outputs
        dedup = raw.get("_dedup_rows") if isinstance(raw, dict) else None
        if dedup is not None:
            N = result.extrinsics.shape[0]
            kwargs: dict = {}
            if result.images is not None and len(result.images) != N:
                kwargs["images"] = result.images[dedup]
            if result.confidence is not None and hasattr(result.confidence, "__len__") and len(result.confidence) != N:
                kwargs["confidence"] = result.confidence[dedup]
            if result.world_points is not None and result.world_points.shape[0] != N:
                kwargs["world_points"] = result.world_points[dedup]
            if result.intrinsics is not None and result.intrinsics.shape[0] != N:
                kwargs["intrinsics"] = result.intrinsics[dedup]
            if kwargs:
                result = dataclasses.replace(result, **kwargs)
            self.base.outputs = result

        return result

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
        console = Console()

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
                k = window.shape[0] if hasattr(window, "shape") else len(window)

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

                # Query retrieval index for loop candidates against prior submaps
                past_for_lc = submaps[: max(0, len(submaps) - cfg.min_submap_gap)]
                loop_matches = find_loop_closures(
                    submap,
                    past_for_lc,
                    cfg.lc_threshold_l2,
                    cfg.max_loops_per_submap,
                    nms_frame_distance=cfg.nms_frame_distance,
                )

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
                            "LC verify accepted but returned no lc_data (backend contract "
                            "violation): submap %s → %s",
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
                            verified += 1
                            loops_found += 1
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
                            lc_submaps.append(
                                Submap(
                                    submap_id=len(submaps) + len(lc_submaps),
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

                submaps.append(submap)
                pbar.update(1)
                pbar.set_postfix(loops=loops_found, verified=verified)
                if end >= N:
                    break

        # Number of accepted loop-closure submaps applied (user-visible summary).
        self.base.n_loops_applied = len(lc_submaps)

        # Merge per-submap world_points and poses into unified outputs
        t0_pg = time.perf_counter()
        corrected_extrinsics = run_pose_graph_optimization(
            submaps,
            lc_submaps,
            total_frames=N,
            overlap_frames=cfg.submap_overlap,
            conf_threshold=cfg.conf_threshold,
            scale_method=cfg.scale_method,
        )
        console.log(f"  Pose graph: {N} frames, {len(lc_submaps)} loop edges → " f"{time.perf_counter() - t0_pg:.1f}s")
        self.base.raw_outputs = merge_submap_outputs(
            submaps,
            corrected_extrinsics,
        )
