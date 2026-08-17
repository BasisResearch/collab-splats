"""Geometric verification of feedforward reconstructions via pycolmap.

Layer on top of any backbone: takes the final COLMAP-frame reconstruction (pose/camera
authority) plus the localization extractor's per-frame features, and produces
  - Tier 1: per-pair epipolar inlier counts and estimated-vs-model relative-pose errors,
  - Tier 2: a triangulated sparse cloud with real feature tracks, filtered at COLMAP's
    defaults (4.0 px reprojection, 1.5 deg triangulation angle), with per-frame stats.
Same import-features -> verify_matches -> triangulate_points flow as hloc's
triangulation module — the validated reference for this pattern.
Spec: docs/superpowers/specs/2026-08-14-geometric-verification-design.md.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from collab_splats.geometry.transforms import rotation_angle_deg
from collab_splats.localization.extractors import BaseLocalExtractor, LocalFeatures, LocalMatcher

logger = logging.getLogger(__name__)

########################################
# Constants
########################################

# Sequential pairing window, forwarded to pycolmap's SequentialPairingOptions.overlap.
# Module default on purpose — repo precedent (mv confidence) keeps tuning knobs out of config.
DEFAULT_OVERLAP = 10


########################################
# Results
########################################


@dataclass
class PairStats:
    """Epipolar verification of one image pair against the model's relative pose."""

    name1: str
    name2: str
    num_matches: int
    num_inliers: int
    rot_error_deg: float  # estimated-vs-model relative rotation, degrees
    t_direction_error_deg: float  # translation-direction angle, degrees (nan if degenerate)


@dataclass
class VerificationResult:
    """Verified reconstruction + Tier 1/2 statistics."""

    reconstruction: pycolmap.Reconstruction  # model poses + triangulated tracked points
    pair_stats: list[PairStats]
    frame_stats: dict[str, dict]  # per image name: n_keypoints, n_tracks, mean_reproj_error_px
    summary: dict = field(default_factory=dict)


########################################
# Pose-error helper
########################################


def _pair_pose_errors(estimated: pycolmap.Rigid3d, model_rel: pycolmap.Rigid3d) -> tuple[float, float]:
    """(rotation error deg, translation-direction error deg) of estimated vs model relative pose.

    The epipolar estimate fixes translation only up to scale, so the direction angle is
    the honest comparison; a near-zero baseline on either side makes it undefined (nan).
    The sign of the direction is kept (verify_matches resolves it by cheirality), unlike
    the AUC protocol's arccos(|cos|) in loop_closure/eval.py — a flipped translation IS
    a pose error here.
    """
    rot_err = rotation_angle_deg(estimated.rotation.matrix() @ model_rel.rotation.matrix().T)
    t_est, t_mod = estimated.translation, model_rel.translation
    n_est, n_mod = np.linalg.norm(t_est), np.linalg.norm(t_mod)
    if n_est < 1e-9 or n_mod < 1e-9:
        return rot_err, float("nan")
    cos = np.clip(np.dot(t_est / n_est, t_mod / n_mod), -1.0, 1.0)
    return rot_err, float(np.degrees(np.arccos(cos)))


########################################
# COLMAP database export
########################################


def _write_frames(db: pycolmap.Database, recon: pycolmap.Reconstruction, features: list[LocalFeatures]) -> None:
    """Write cameras/images/keypoints into an open COLMAP database.

    Ids mirror `recon` exactly (one camera per image, camera_id == image_id) —
    triangulate_points joins DB rows to reconstruction frames by id, and a shared DB
    camera against per-image trivial rigs fails COLMAP's RigId check. Matches are
    written later by the caller (pair generation needs the images in the DB first).
    """
    for image_id, feats in zip(sorted(recon.images), features):
        image = recon.images[image_id]
        camera = recon.cameras[image.camera_id]
        kpts = feats.keypoints.numpy().astype(np.float64)
        # Bounds guard: catches a feature cache built at a different resolution than the
        # reconstruction's cameras (the 2026-08-11 model-res-vs-original-res class)
        if len(kpts) and (kpts.min() < 0.0 or kpts[:, 0].max() >= camera.width or kpts[:, 1].max() >= camera.height):
            raise ValueError(
                f"Keypoints for {image.name} exceed camera bounds ({camera.width}x{camera.height}) "
                "— the feature cache and the reconstruction disagree on image resolution."
            )
        db.write_camera(camera, use_camera_id=True)
        # Set image_id via the property, not the ctor (the ctor kwarg is unverified in
        # this pycolmap build; the property + use_image_id=True path is the documented one)
        image_row = pycolmap.Image(name=image.name, camera_id=image.camera_id)
        image_row.image_id = image_id
        db.write_image(image_row, use_image_id=True)
        db.write_keypoints(image_id, kpts)


########################################
# Entry point
########################################


def verify_reconstruction(
    recon: pycolmap.Reconstruction,
    features: list[LocalFeatures],
    matcher: BaseLocalExtractor,
    output_dir: str | Path,
    overlap: int = DEFAULT_OVERLAP,
    images: list[np.ndarray] | None = None,
) -> VerificationResult:
    """Triangulate and epipolar-verify a reconstruction's poses with independent features.

    Args:
        recon: pose/camera authority (original-resolution K); its points are ignored.
        features: per-image LocalFeatures, aligned with sorted(recon.images) order.
        matcher: a BaseLocalExtractor whose match() exposes keypoint indices, or an
            index-stable LocalMatcher (matched pairwise over `images`).
        output_dir: writes database.db, verified/ (COLMAP model), verification.json.
        overlap: sequential pairing window (pycolmap SequentialPairingOptions.overlap).
        images: pairwise matchers (LocalMatcher) only — the exact RGB frames `features`
            was extracted from, aligned with sorted(recon.images) like `features`
            (index recovery lands on the extract-time keypoint tables only for
            identical inputs).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(recon.images)
    if len(features) != len(image_ids):
        raise ValueError(
            f"{len(features)} feature frames vs {len(image_ids)} reconstruction images — "
            "the cache and the reconstruction describe different runs."
        )

    # Pairwise matchers must prove index stability up front — a silent skip here would
    # surface later as a missing verification.json with no explanation.
    if isinstance(matcher, LocalMatcher):
        if not matcher.has_stable_indices:
            raise ValueError(
                f"matcher '{matcher.model_name}' cannot provide stable keypoint indices "
                "(failed the index-stability probe) — geometric verification requires them. "
                "Choose a sparse index-stable model or disable pointcloud.geometric_verification."
            )
        if images is None or len(images) != len(image_ids):
            raise ValueError(
                f"pairwise matcher requires `images` aligned with recon frames "
                f"(got {'none' if images is None else len(images)} for {len(image_ids)} frames)"
            )

    # ── Fresh DB with frames written first: pair generation reads images off the DB ──
    db_path = output_dir / "database.db"
    if db_path.exists():
        db_path.unlink()  # stale DBs accumulate duplicate rows; always start fresh
    db = pycolmap.Database.open(str(db_path))
    try:
        _write_frames(db, recon, features)

        # ── Sequential pairs from pycolmap's own generator (COLMAP's video pairing;
        # quadratic_overlap adds power-of-two longer-baseline pairs). loop_detection stays
        # off: it needs a SIFT vocab tree, unusable with learned descriptors — retrieval-
        # based loop pairs are a follow-on when a caller needs them. ──
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pairs = pycolmap.SequentialPairGenerator(pairing, db).all_pairs()

        # ── Match every pair with our extractor; index pairs are COLMAP's match format ──
        cam0 = recon.cameras[recon.images[image_ids[0]].camera_id]
        # Feedforward output is uniform-resolution across all frames, so camera 0's (H, W)
        # is reused for every pair below — no per-pair guard needed.
        hw = (cam0.height, cam0.width)
        id_to_pos = {iid: k for k, iid in enumerate(image_ids)}
        matches: dict[tuple[int, int], np.ndarray] = {}
        for id1, id2 in pairs:
            if isinstance(matcher, LocalMatcher):
                # Pairwise path: match the raw images; idx_q/idx_db are recovered rows of
                # the extract-time keypoint tables — the very tables _write_frames just
                # exported. Recovery can still fail per pair despite the passed probe
                # (match_images downgrades idx to None): skip that pair with a warning
                # rather than aborting the whole report for one degenerate pair.
                m = matcher.match_images(images[id_to_pos[id1]], images[id_to_pos[id2]])
                if m.idx_q is None or m.idx_db is None:
                    logger.warning("Verification: pair (%d, %d) lost index recovery — skipping", id1, id2)
                    continue
            else:
                m = matcher.match(features[id_to_pos[id1]], features[id_to_pos[id2]], hw)
                if m.idx_q is None or m.idx_db is None:
                    raise ValueError(
                        f"{type(matcher).__name__} does not expose keypoint indices "
                        "(per-pair refined matchers cannot feed COLMAP tracks) — use disk/xfeat/loma."
                    )
            if len(m) == 0:
                continue
            matches[(id1, id2)] = np.stack([m.idx_q, m.idx_db], axis=1).astype(np.uint32)
            db.write_matches(id1, id2, matches[(id1, id2)])
    finally:
        db.close()
    logger.info("Verification: %d/%d pairs matched", len(matches), len(pairs))

    # ── Epipolar verification (Tier 1) over the matched pairs ──
    pairs_path = output_dir / "pairs.txt"
    pairs_path.write_text("\n".join(f"{recon.images[a].name} {recon.images[b].name}" for a, b in matches))
    tvg_options = pycolmap.TwoViewGeometryOptions()
    tvg_options.compute_relative_pose = True  # cam2_from_cam1 stays None without this
    pycolmap.verify_matches(str(db_path), str(pairs_path), options=tvg_options)
    pairs_path.unlink()  # scratch input to verify_matches only — keep colmap/ at its documented contract

    db = pycolmap.Database.open(str(db_path))
    try:
        pair_ids, geoms = db.read_two_view_geometries()
    finally:
        db.close()
    pair_stats = []
    for pid, g in zip(pair_ids, geoms):
        id1, id2 = pycolmap.pair_id_to_image_pair(int(pid))
        # pair ids are stored canonically (id1 < id2); our matches dict is keyed by the
        # generator's order, which may be swapped. The key is always present: verify_matches
        # was run against pairs.txt, which we wrote only from `matches`, so every geometry
        # read back here corresponds to a pair we ourselves fed it.
        key = (id1, id2) if (id1, id2) in matches else (id2, id1)
        im1, im2 = recon.images[id1], recon.images[id2]
        model_rel = im2.cam_from_world() * im1.cam_from_world().inverse()
        if g.cam2_from_cam1 is not None:
            rot_err, tdir_err = _pair_pose_errors(g.cam2_from_cam1, model_rel)
        else:
            rot_err = tdir_err = float("nan")  # too few inliers for a pose estimate
        pair_stats.append(
            PairStats(
                name1=im1.name,
                name2=im2.name,
                num_matches=int(matches[key].shape[0]),
                num_inliers=len(g.inlier_matches),
                rot_error_deg=rot_err,
                t_direction_error_deg=tdir_err,
            )
        )

    # ── Tier 2: known-pose triangulation (Task 5 fills in from here) ──
    verified, frame_stats, summary = _triangulate_and_summarize(recon, db_path, output_dir, pair_stats, features)
    result = VerificationResult(
        reconstruction=verified, pair_stats=pair_stats, frame_stats=frame_stats, summary=summary
    )
    _write_report(result, output_dir / "verification.json")
    return result


########################################
# Tier 2: triangulation + summaries
########################################


def _distribution(values) -> dict | None:
    """median/p90/p99 of a value list — never median alone. None when empty/all-nan."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    return {
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p99": float(np.percentile(v, 99)),
    }


def _triangulate_and_summarize(
    recon: pycolmap.Reconstruction,
    db_path: Path,
    output_dir: Path,
    pair_stats: list[PairStats],
    features: list[LocalFeatures],
) -> tuple[pycolmap.Reconstruction, dict, dict]:
    """Run known-pose triangulation and derive per-frame and scene-level statistics.

    triangulate_points clears the model's points (clear_points=True, the default), chains
    tracks from the DB matches, and filters at COLMAP defaults (4.0 px reprojection, 1.5 deg
    angle) inside its own pipeline — no extra filter call belongs here. It always rewrites
    the full binary model (cameras/images/points3D.bin) at output_path, so a stale dir from
    a prior run leaves no dangling state; mkdir(exist_ok=True) below is sufficient.
    """
    verified_dir = output_dir / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    # triangulate_points mutates its reconstruction argument in place and returns the same
    # object — copy first so the caller's model (its points included) stays untouched
    recon = pycolmap.Reconstruction(recon)
    # image dir is unused (keypoints live in the DB) but must exist
    verified = pycolmap.triangulate_points(recon, str(db_path), str(output_dir), str(verified_dir))
    logger.info("Verification: triangulated %d points", verified.num_points3D())

    # Per-frame survival + reprojection error via each frame's track observations
    frame_stats: dict[str, dict] = {}
    # features is aligned with sorted(recon.images) by contract — map from recon, not
    # verified, so a frame dropped by triangulation could never shift the alignment
    id_to_pos = {iid: k for k, iid in enumerate(sorted(recon.images))}
    for image_id in sorted(verified.images):
        image = verified.images[image_id]
        errors = [
            verified.points3D[p2d.point3D_id].error
            for p2d in image.points2D
            if p2d.has_point3D()
        ]
        n_kpts = len(features[id_to_pos[image_id]].keypoints)
        frame_stats[image.name] = {
            "n_keypoints": int(n_kpts),
            "n_tracks": len(errors),
            "track_survival": (len(errors) / n_kpts) if n_kpts else 0.0,
            "mean_reproj_error_px": float(np.mean(errors)) if errors else None,
        }

    track_lengths = [p.track.length() for p in verified.points3D.values()]
    reproj_errors = [p.error for p in verified.points3D.values()]
    inlier_ratios = [p.num_inliers / p.num_matches for p in pair_stats if p.num_matches]
    summary = {
        "n_points": int(verified.num_points3D()),
        "n_pairs": len(pair_stats),
        "track_length": _distribution(track_lengths),
        "reproj_error_px": _distribution(reproj_errors),
        "pair_inlier_ratio": _distribution(inlier_ratios),
        "pair_rot_error_deg": _distribution(p.rot_error_deg for p in pair_stats),
        "pair_t_direction_error_deg": _distribution(p.t_direction_error_deg for p in pair_stats),
    }
    return verified, frame_stats, summary


def _write_report(result: VerificationResult, path: Path) -> None:
    """Serialize pair/frame/summary stats to verification.json (nan -> null)."""

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    payload = _clean(
        {
            "pair_stats": [asdict(p) for p in result.pair_stats],
            "frame_stats": result.frame_stats,
            "summary": result.summary,
        }
    )
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Verification report written to %s", path)
