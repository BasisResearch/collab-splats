"""Tests for translation-jump gate (item 15)."""
import numpy as np
import pytest
import torch
from pathlib import Path

from collab_splats.geometry.loop_closure import Submap
from collab_splats.geometry.loop_closure.closure import translation_jump_check


def _linear_submap(submap_id: int, k: int, dx: float, x_offset: float = 0.0) -> Submap:
    """Submap with cameras spaced `dx` along +x. W2C: t = -x."""
    poses = np.tile(np.eye(4), (k, 1, 1)).astype(np.float32)
    for i in range(k):
        poses[i, 0, 3] = -(x_offset + i * dx)
    return Submap(
        submap_id=submap_id,
        frames=torch.zeros(k, 3, 4, 4),
        poses=poses,
        intrinsics=np.tile(np.eye(3), (k, 1, 1)).astype(np.float32),
        retrieval_vectors=torch.zeros(k, 16),
        image_paths=[Path(f"s{submap_id}_f{i}.jpg") for i in range(k)],
    )


def test_translation_jump_accepts_consistent_loop():
    """Loop pose matches odom-implied pose → ratio ≈ 0 → accepted."""
    s0 = _linear_submap(0, k=5, dx=1.0, x_offset=0.0)
    s1 = _linear_submap(1, k=5, dx=1.0, x_offset=5.0)
    submaps = [s0, s1]

    # Odom path length 0→0 (s0 frame 0) to 1→4 (s1 frame 4): 9 unit steps + boundary = 9.
    # Odom relative pose: T = inv(pose_det) @ pose_query (W2C convention)
    pose_det = s0.poses[0]
    pose_query = s1.poses[4]
    T_odom = np.linalg.inv(pose_det.astype(np.float64)) @ pose_query.astype(np.float64)
    lc_rel = T_odom.astype(np.float32)  # exactly matching odom

    accept, ratio = translation_jump_check(submaps, 1, 4, 0, 0, lc_rel, max_jump_ratio=0.2)
    assert accept, f"Should accept ratio={ratio}"
    assert ratio < 0.05


def test_translation_jump_rejects_wild_loop():
    """Loop pose claims 50m offset; odom path ~9m → ratio > 0.2 → rejected."""
    s0 = _linear_submap(0, k=5, dx=1.0, x_offset=0.0)
    s1 = _linear_submap(1, k=5, dx=1.0, x_offset=5.0)
    submaps = [s0, s1]

    # Build a wildly inconsistent LC relative pose (50m off).
    lc_rel = np.eye(4, dtype=np.float32)
    lc_rel[0, 3] = 50.0
    accept, ratio = translation_jump_check(submaps, 1, 4, 0, 0, lc_rel, max_jump_ratio=0.2)
    assert not accept, f"Should reject ratio={ratio}"
    assert ratio > 1.0


def test_translation_jump_intra_submap():
    """Loop within same submap — uses intra-submap path length."""
    s0 = _linear_submap(0, k=10, dx=1.0)
    submaps = [s0]
    pose_det = s0.poses[0]
    pose_query = s0.poses[5]
    T_odom = np.linalg.inv(pose_det.astype(np.float64)) @ pose_query.astype(np.float64)
    lc_rel = T_odom.astype(np.float32)
    accept, ratio = translation_jump_check(submaps, 0, 5, 0, 0, lc_rel, max_jump_ratio=0.2)
    assert accept and ratio < 0.05
