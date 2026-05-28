from __future__ import annotations

import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from tqdm.auto import tqdm

from collab_splats.utils.geometry import extrinsics_to_homogeneous, invert_poses

from .base import PointcloudResult
from .feedforward import FeedforwardResult, _raw_to_world_points
from .localization import BaseRetrievalExtractor
from .loop_closure import LoopClosureConfig, Submap
from .loop_closure.closure import (
    find_loop_closures,
    merge_submap_outputs,
    run_pose_graph_optimization,
    translation_jump_check,
)
from .loop_closure.submap import assert_world_to_cam

__all__ = ["LoopClosure"]


def _assemble_precorrection_extrinsics(submaps: list, total_frames: int) -> np.ndarray:
    """Stitch raw per-submap poses into (total_frames, 4, 4) without PGO correction.

    Uses the same first-writer-wins overlap dedup as dedup_overlap in closure.py.
    """
    from .loop_closure.closure import dedup_overlap
    corrected_raw = {s.submap_id: s.poses for s in submaps}
    return dedup_overlap(
        submap_ids=[s.submap_id for s in submaps],
        submap_starts=[s.frame_start for s in submaps],
        corrected=corrected_raw,
        total_frames=total_frames,
    )


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
        if config is None:
            # Pick up per-model calibrated threshold when no explicit config given.
            # Falls back to LoopClosureConfig default (0.85) if creator lacks the attr.
            model_ratio = getattr(base, "default_verify_match_ratio", None)
            self.config = (
                LoopClosureConfig(verify_match_ratio=model_ratio)
                if model_ratio is not None
                else LoopClosureConfig()
            )
        else:
            self.config = config

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

    def reproject(self, result: FeedforwardResult) -> FeedforwardResult:
        """Re-extract pts3d/colors using refined poses."""
        return self.base.reproject(result)

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
        step = max(1, K - O)
        views = self.base.views
        N = views.shape[0] if hasattr(views, "shape") else len(views)
        device = str(next(self.base.model.parameters()).device)

        # Load DINO-SALAD retrieval extractor; fall back to full-sequence inference if unavailable
        try:
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

                # Models that return list[dict] (e.g. MapAnything) must aggregate to a flat
                # dict for the LC loop. raw_lc is used for LC metadata; raw is stored in the
                # Submap so _postprocess can use the original per-frame structure.
                raw_lc = self.base._lc_collate_outputs(raw) if isinstance(raw, list) else raw

                ext_3x4 = raw_lc["extrinsic"]                                            # (k, 3, 4)
                poses_4x4 = extrinsics_to_homogeneous(ext_3x4)                           # (k, 4, 4)

                assert_world_to_cam(poses_4x4)

                intr_key = "intrinsics" if "intrinsics" in raw_lc else "intrinsic"
                intrinsics = raw_lc.get(intr_key, np.tile(np.eye(3), (k, 1, 1)).astype(np.float32))

                frames_cpu = window.cpu() if hasattr(window, "cpu") else torch.zeros(k, 3, 1, 1)
                ret_vecs = retrieval_extractor(frames_cpu)                                # (k, D)

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
                        lc_rel = (
                            invert_poses(lc_poses[1].astype(np.float64))
                            @ lc_poses[0].astype(np.float64)
                        ).astype(np.float32)
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
        self.base._lc_precorrection_extrinsics = _assemble_precorrection_extrinsics(submaps, N)

        # Merge per-submap world_points and poses into unified outputs
        t0_pg = time.perf_counter()
        corrected_extrinsics = run_pose_graph_optimization(
            submaps, lc_submaps, total_frames=N,
            overlap_frames=cfg.submap_overlap,
            conf_threshold=cfg.conf_threshold,
        )
        console.log(
            f"  Pose graph: {N} frames, {len(lc_submaps)} loop edges → "
            f"{time.perf_counter() - t0_pg:.1f}s"
        )
        self.base._lc_corrected_extrinsics = corrected_extrinsics
        self.base.raw_outputs = merge_submap_outputs(
            submaps, corrected_extrinsics,
        )
