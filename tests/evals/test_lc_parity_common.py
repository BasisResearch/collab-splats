"""Tests for LC parity harness helpers."""

import sys
from pathlib import Path

import pytest

# Local `evals/` is shadowed by an installed `evals` pip package; insert the
# evals dir on sys.path and import the module directly (sibling-test convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from runners.lc_parity_common import (
    SCENES,
    check_gates,
    check_lc_harmless,
    check_scaling_gate,
    collect_frames,
    filter_images_to_list,
    list_scene_images,
    slam_max_frames_for_prefix,
    slice_keyframes,
    write_tum_allowed_frames,
)


def make_fake_tum_seq(root: Path) -> Path:
    """Minimal TUM seq: 3 rgb frames, GT covers only the first two (third is a GT gap)."""
    seq = root / "tum" / "rgbd_dataset_freiburg3_long_office_household"
    (seq / "rgb").mkdir(parents=True)
    rgb_lines = ["# color images"]
    for ts in ("10.000000", "10.050000", "99.000000"):
        (seq / "rgb" / f"{ts}.png").touch()
        rgb_lines.append(f"{ts} rgb/{ts}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines))
    gt_lines = ["# gt"] + [f"{ts} 0 0 0 0 0 0 1" for ts in ("10.001", "10.049")]
    (seq / "groundtruth.txt").write_text("\n".join(gt_lines))
    return seq


def test_scene_registry_complete():
    # 7 7-Scenes + 4 TUM, each with dataset type and long-scene flag
    assert len(SCENES) == 11
    assert SCENES["7s_chess"].dataset == "7scenes"
    assert SCENES["tum_fr1_room"].dataset == "tum"
    long_scenes = [k for k, s in SCENES.items() if s.scaling_sweep]
    assert sorted(long_scenes) == ["7s_office", "7s_redkitchen", "tum_fr1_room", "tum_fr3_office"]


def test_slice_keyframes_writes_prefix(tmp_path):
    kf = tmp_path / "selected_frames.txt"
    kf.write_text("\n".join(f"/data/frame-{i:06d}.color.png" for i in range(100)))
    out = slice_keyframes(kf, 0.25, tmp_path / "prefix_25.txt")
    lines = out.read_text().splitlines()
    assert len(lines) == 25
    assert lines[0].endswith("frame-000000.color.png")
    assert lines[-1].endswith("frame-000024.color.png")


def test_slice_keyframes_full_is_identity(tmp_path):
    kf = tmp_path / "selected_frames.txt"
    kf.write_text("\n".join(f"img_{i}.png" for i in range(10)))
    out = slice_keyframes(kf, 1.0, tmp_path / "prefix_100.txt")
    assert out.read_text() == kf.read_text()


def test_list_scene_images_7scenes_layout(tmp_path):
    # 7-Scenes: color frames directly in seq dir
    for i in range(3):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    (tmp_path / "frame-000000.pose.txt").touch()  # must be excluded
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 3
    assert all(p.suffix == ".png" and "color" in p.name for p in imgs)


def test_list_scene_images_tum_layout(tmp_path):
    # TUM: images under rgb/
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for ts in ("1305031102.175304", "1305031102.211214"):
        (rgb / f"{ts}.png").touch()
    (tmp_path / "groundtruth.txt").touch()
    imgs = list_scene_images(tmp_path)
    assert len(imgs) == 2
    assert imgs[0].parent.name == "rgb"
    assert imgs == sorted(imgs)


def test_slam_max_frames_for_prefix(tmp_path):
    # 10 source frames, keyframes are frames 0,3,6,9. 50% prefix = kf 0,3
    # → SLAM must process source frames up to index 3 → max_frames=4.
    for i in range(10):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("\n".join(str(tmp_path / f"frame-{i:06d}.color.png") for i in (0, 3)))
    assert slam_max_frames_for_prefix(prefix, tmp_path) == 4


def test_filter_images_to_list_keeps_only_listed(tmp_path):
    # Matching is by basename: absolute frame paths vs a bare-basename allow list
    frames = [str(tmp_path / "rgb" / f"frame-{i}.png") for i in range(5)]
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("frame-0.png\nframe-2.png\nframe-4.png\n")
    kept = filter_images_to_list(frames, allow)
    assert [Path(p).name for p in kept] == ["frame-0.png", "frame-2.png", "frame-4.png"]


def test_write_tum_allowed_frames_drops_gt_gap(tmp_path):
    # The GT-gap frame (no groundtruth within 0.02 s) must not appear in the allow list
    seq = make_fake_tum_seq(tmp_path)
    out = tmp_path / "allowed_frames.txt"
    write_tum_allowed_frames(seq, out)
    assert out.read_text().splitlines() == ["10.000000.png", "10.050000.png"]


def test_collect_frames_no_list_is_raw_sorted_listing(tmp_path):
    # Without image_list the frame universe is the raw dir listing (7-Scenes path)
    for i in range(4):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    frames = collect_frames(tmp_path)
    assert [Path(p).name for p in frames] == [f"frame-{i:06d}.color.png" for i in range(4)]


def test_collect_frames_respects_image_list(tmp_path):
    # image_list restricts by basename; order of survivors is preserved
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    for name in ("10.0.png", "10.5.png", "99.0.png"):
        (rgb / name).touch()
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("10.0.png\n10.5.png")
    frames = collect_frames(tmp_path, image_list=allow)
    assert [Path(p).name for p in frames] == ["10.0.png", "10.5.png"]


def test_collect_frames_max_frames_applies_after_filter(tmp_path):
    # max_frames caps the FILTERED sequence (matches slam_max_frames_for_prefix indexing)
    for i in range(6):
        (tmp_path / f"f{i}.png").touch()
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("\n".join(f"f{i}.png" for i in (0, 2, 4)))
    frames = collect_frames(tmp_path, image_list=allow, max_frames=2)
    assert [Path(p).name for p in frames] == ["f0.png", "f2.png"]


def test_slam_max_frames_for_prefix_with_image_list(tmp_path):
    # Allowed frames = even indices (0,2,4,6,8); prefix ends at frame-4, whose
    # index in the FILTERED sequence is 2 → max_frames=3 (runner filters first).
    for i in range(10):
        (tmp_path / f"frame-{i:06d}.color.png").touch()
    allow = tmp_path / "allowed_frames.txt"
    allow.write_text("\n".join(f"frame-{i:06d}.color.png" for i in (0, 2, 4, 6, 8)))
    prefix = tmp_path / "prefix.txt"
    prefix.write_text("\n".join(str(tmp_path / f"frame-{i:06d}.color.png") for i in (0, 4)))
    assert slam_max_frames_for_prefix(prefix, tmp_path, image_list=allow) == 3


def test_check_gates_pass():
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.104, "loops_applied": 5}
    g = check_gates(slam, ours)
    assert g["ate_pass"] and g["loops_pass"] and g["all_pass"]
    assert g["ate_delta"] == pytest.approx(0.004)


def test_check_gates_ate_absolute_fallback():
    # tiny ATEs: 5% of 20mm = 1mm, but 5mm absolute tolerance applies
    slam = {"ate_rmse": 0.020, "loop_closures": 0}
    ours = {"ate_rmse": 0.024, "loops_applied": 0}
    assert check_gates(slam, ours)["ate_pass"]


def test_check_gates_fail_on_loop_mismatch():
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.100, "loops_applied": 0}
    g = check_gates(slam, ours)
    assert not g["loops_pass"] and not g["all_pass"]


def test_check_gates_missing_ate():
    g = check_gates({"ate_rmse": None, "loop_closures": 1}, {"ate_rmse": 0.1, "loops_applied": 1})
    assert not g["ate_pass"] and g["ate_delta"] is None


def test_check_gates_check_loops_false_ignores_loop_mismatch():
    # Baseline condition: LC never runs, so its loop count (0) shouldn't be compared
    # against SLAM's — only the ATE gate should determine all_pass.
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.102, "loops_applied": 0}
    g = check_gates(slam, ours, check_loops=False)
    assert g["loops_pass"] is True
    assert g["ate_pass"] and g["all_pass"]


def test_check_gates_check_loops_false_still_fails_on_ate():
    slam = {"ate_rmse": 0.100, "loop_closures": 5}
    ours = {"ate_rmse": 0.500, "loops_applied": 0}
    g = check_gates(slam, ours, check_loops=False)
    assert g["loops_pass"] is True
    assert not g["ate_pass"] and not g["all_pass"]


def test_check_scaling_gate_ok():
    # ATE(100%) <= 1.5x ATE(25%): 0.025 <= 1.5*0.02 = 0.03
    assert check_scaling_gate(0.02, 0.025) is True


def test_check_scaling_gate_blowup():
    # 0.05 > 1.5*0.02 = 0.03
    assert check_scaling_gate(0.02, 0.05) is False


def test_check_scaling_gate_boundary_is_ok():
    assert check_scaling_gate(0.02, 0.03) is True  # exactly 1.5x passes (<=)


def test_check_scaling_gate_missing_values():
    assert check_scaling_gate(None, 0.05) is None
    assert check_scaling_gate(0.02, None) is None
    assert check_scaling_gate(None, None) is None


def test_check_lc_harmless_true_when_lc_improves_or_ties():
    assert check_lc_harmless(0.100, 0.100) is True
    assert check_lc_harmless(0.100, 0.090) is True


def test_check_lc_harmless_true_within_relative_tolerance():
    # 5% of 0.100 = 0.005 → 0.104 <= 0.105
    assert check_lc_harmless(0.100, 0.104) is True


def test_check_lc_harmless_false_beyond_relative_tolerance():
    # 5% of 0.100 = 0.005 → 0.110 > 0.105
    assert check_lc_harmless(0.100, 0.110) is False


def test_check_lc_harmless_tiny_ate_uses_absolute_tolerance():
    # 5% of 0.010 = 0.0005, but the 5mm absolute floor applies: 0.010 + 0.005 = 0.015
    assert check_lc_harmless(0.010, 0.014) is True
    assert check_lc_harmless(0.010, 0.016) is False


def test_check_lc_harmless_missing_values():
    assert check_lc_harmless(None, 0.05) is None
    assert check_lc_harmless(0.02, None) is None
    assert check_lc_harmless(None, None) is None
