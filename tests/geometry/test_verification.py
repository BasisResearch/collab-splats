"""Tests for geometric verification: COLMAP DB export + Tier 1 epipolar pose verification."""

import json

import numpy as np
import pytest
import torch

from collab_splats.geometry.verification import verify_reconstruction
from collab_splats.localization.extractors import (
    BaseLocalExtractor,
    LocalFeatures,
    MatchResult,
)
from collab_splats.pointcloud.feedforward.base import build_pycolmap_reconstruction

W, H = 640, 480
_K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float32)


def _synthetic_scene(n_cams: int = 3, n_pts: int = 60, seed: int = 0):
    """World points + lateral camera array + exact projections.

    Returns (pts_w (P,3), extrinsics (N,3,4) w2c, per-frame keypoints list of (P,2))."""
    rng = np.random.default_rng(seed)
    pts_w = np.stack(
        [rng.uniform(-1, 1, n_pts), rng.uniform(-0.8, 0.8, n_pts), rng.uniform(3.0, 5.0, n_pts)],
        axis=1,
    ).astype(np.float32)
    extrinsics = []
    for i in range(n_cams):
        E = np.eye(4, dtype=np.float32)
        E[0, 3] = -0.3 * i  # camera at x = +0.3*i; w2c translation is the negative
        extrinsics.append(E[:3])
    extrinsics = np.stack(extrinsics)
    kps = []
    for E in extrinsics:
        pc = (E[:3, :3] @ pts_w.T + E[:3, 3:4]).T
        uv = (_K @ (pc / pc[:, 2:3]).T).T[:, :2]
        kps.append(uv.astype(np.float32))
    return pts_w, extrinsics, kps


def _make_recon(extrinsics):
    """Poses-only pycolmap reconstruction over the synthetic cameras."""
    n = len(extrinsics)
    return build_pycolmap_reconstruction(
        pts3d=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        extrinsics=extrinsics,
        intrinsics=np.stack([_K] * n),
        image_width=W,
        image_height=H,
        image_names=[f"frame_{i:05d}" for i in range(n)],
    )


def _features_from_keypoints(kps):
    """Wrap projected keypoints as LocalFeatures (descriptors unused by the stub matcher)."""
    return [LocalFeatures(keypoints=torch.from_numpy(k), descriptors=torch.zeros(len(k), 4)) for k in kps]


class _IdentityMatcher(BaseLocalExtractor):
    """Stub matcher: keypoint i in every frame observes world point i (ground-truth tracks)."""

    def extract(self, image):  # pragma: no cover - never called in these tests
        raise NotImplementedError

    def match(self, query, db, image_hw):
        n = min(len(query.keypoints), len(db.keypoints))
        idx = np.arange(n, dtype=np.int64)
        return MatchResult(
            query_px=query.keypoints.numpy()[:n],
            ref_px=db.keypoints.numpy()[:n],
            idx_q=idx,
            idx_db=idx.copy(),
        )


class _NoIndexMatcher(_IdentityMatcher):
    """Stub matcher mimicking XFeatStar: pixels only, no table indices."""

    def match(self, query, db, image_hw):
        m = super().match(query, db, image_hw)
        return MatchResult(query_px=m.query_px, ref_px=m.ref_px)


def test_tier1_pair_stats_on_clean_scene(tmp_path):
    """verify_matches recovers each pair's relative pose to within a fraction of a degree."""
    _, extrinsics, kps = _synthetic_scene()
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    assert len(result.pair_stats) == 3  # overlap=10 window covers all pairs of 3 frames
    for p in result.pair_stats:
        assert p.num_inliers >= 55
        assert p.rot_error_deg < 0.1
        assert p.t_direction_error_deg < 1.0


def test_no_index_matcher_rejected(tmp_path):
    """A matcher without keypoint indices (XFeatStar) is rejected with a clear error."""
    _, extrinsics, kps = _synthetic_scene()
    with pytest.raises(ValueError, match="indices"):
        verify_reconstruction(
            recon=_make_recon(extrinsics),
            features=_features_from_keypoints(kps),
            matcher=_NoIndexMatcher(),
            output_dir=tmp_path,
        )


def test_keypoint_bounds_guard(tmp_path):
    """Keypoints outside the camera grid abort the export (resolution-mismatch class)."""
    _, extrinsics, kps = _synthetic_scene()
    kps[1][0] = [W * 2.0, H * 2.0]  # simulate a cache built at a different resolution
    with pytest.raises(ValueError, match="bounds"):
        verify_reconstruction(
            recon=_make_recon(extrinsics),
            features=_features_from_keypoints(kps),
            matcher=_IdentityMatcher(),
            output_dir=tmp_path,
        )


def test_triangulation_recovers_scene(tmp_path):
    """Tier 2 triangulates the synthetic scene: full yield, full tracks, ~zero error."""
    pts_w, extrinsics, kps = _synthetic_scene()
    result = verify_reconstruction(
        recon=_make_recon(extrinsics),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    verified = result.reconstruction
    assert verified.num_points3D() >= 55  # of 60; COLMAP may drop boundary cases
    # Every surviving point carries a real (non-empty) track and lies on a GT point
    for p in verified.points3D.values():
        assert p.track.length() == 3
        assert np.linalg.norm(pts_w - p.xyz, axis=1).min() < 1e-3
    # Per-frame stats populated for all frames; reprojection error is sub-pixel
    assert set(result.frame_stats) == {f"frame_{i:05d}" for i in range(3)}
    for s in result.frame_stats.values():
        assert s["n_tracks"] >= 55
        assert s["mean_reproj_error_px"] < 0.5
    # Summary distributions present (median/p90/p99 — never median alone)
    assert result.summary["n_points"] >= 55
    assert set(result.summary["track_length"]) == {"median", "p90", "p99"}
    # Report written and loadable
    report = json.loads((tmp_path / "verification.json").read_text())
    assert report["summary"]["n_points"] == result.summary["n_points"]
    assert len(report["pair_stats"]) == 3
    # COLMAP model on disk for downstream tooling
    assert (tmp_path / "verified" / "points3D.bin").exists()


def _rot_x(deg: float) -> np.ndarray:
    """Rotation about x, degrees."""
    a = np.radians(deg)
    return np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]], dtype=np.float32
    )


def test_negative_control_perturbed_pose_flagged(tmp_path):
    """+2 deg rotation on one camera shows up in exactly that camera's pair errors and survival."""
    _, extrinsics, kps = _synthetic_scene(n_cams=5)
    bad = 2
    perturbed = extrinsics.copy()
    perturbed[bad, :3, :3] = _rot_x(2.0) @ perturbed[bad, :3, :3]

    # Features come from the TRUE geometry; only the model's pose for frame `bad` lies.
    result = verify_reconstruction(
        recon=_make_recon(perturbed),
        features=_features_from_keypoints(kps),
        matcher=_IdentityMatcher(),
        output_dir=tmp_path,
    )
    bad_name = f"frame_{bad:05d}"
    for p in result.pair_stats:
        involved = bad_name in (p.name1, p.name2)
        if involved:
            # The epipolar estimate follows the matches (truth), so it disagrees with the
            # model's perturbed relative pose by ~the injected 2 degrees.
            assert p.rot_error_deg > 1.0, f"{p.name1}-{p.name2} not flagged: {p.rot_error_deg}"
        else:
            assert p.rot_error_deg < 0.2, f"clean pair {p.name1}-{p.name2}: {p.rot_error_deg}"
    # Tier 2: reprojection through the wrong pose kills that frame's observations
    clean_survival = [
        s["track_survival"] for n, s in result.frame_stats.items() if n != bad_name
    ]
    assert result.frame_stats[bad_name]["track_survival"] < min(clean_survival)
