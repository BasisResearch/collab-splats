"""Driver argv construction for the LC parity matrix (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

# `run_lc_parity.py` does a plain same-dir `from lc_parity_common import ...`, so the
# runners dir itself (not `evals/`) must be on sys.path for both modules to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))

from run_lc_parity import (
    DEFAULT_BACKBONES,
    _done,
    build_prefix_commands,
    build_scene_commands,
)


def test_build_scene_commands_levels_0_and_1(tmp_path):
    data_root = tmp_path / "data"
    out_root = tmp_path / "lc_parity"
    seq = data_root / "7scenes/chess/chess/seq-01"
    seq.mkdir(parents=True)
    cmds = build_scene_commands("7s_chess", data_root, out_root, DEFAULT_BACKBONES)
    # One shared Level 0 run first: run_vggt_slam_lc with LC on, full sequence, paper config
    assert len(cmds) == 1 + len(DEFAULT_BACKBONES)
    slam = cmds[0]
    assert "run_vggt_slam_lc.py" in slam[1]
    assert "--max_loops" in slam and slam[slam.index("--max_loops") + 1] == "1"
    assert "--min_disparity" in slam and slam[slam.index("--min_disparity") + 1] == "50"
    assert "--lc_thres" in slam and slam[slam.index("--lc_thres") + 1] == "0.95"
    assert "--max_frames" not in slam  # full sequence — no cap
    # Level 1: one eval_gt run per backbone, all consuming the same Level-0 keyframe list
    for backbone, ours in zip(DEFAULT_BACKBONES, cmds[1:]):
        assert "eval.py" in ours[1]
        assert ours[ours.index("--backbone") + 1] == backbone
        assert ours[ours.index("--max_loops_per_submap") + 1] == "1"  # upstream parity cap
        kf = Path(ours[ours.index("--keyframe_list") + 1])
        assert kf == out_root / "7s_chess/slam/selected_frames.txt"
        assert "baseline" in ours and "lc" in ours
        # Per-backbone output dir (replaces the single `ours/`)
        out_dir = Path(ours[ours.index("--output_dir") + 1])
        assert out_dir == out_root / f"7s_chess/ours_{backbone}"
        # Trajectory-overlay plot gets the Level-0 SLAM reference (gap 1)
        slam_tum = Path(ours[ours.index("--slam_tum") + 1])
        assert slam_tum == out_root / "7s_chess/slam/slam.tum"


def test_build_scene_commands_single_backbone_matches_old_behavior(tmp_path):
    # --backbones vggt_spark alone reproduces the pre-sweep shape: 1 slam + 1 ours.
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    (data_root / "7scenes/chess/chess/seq-01").mkdir(parents=True)
    cmds = build_scene_commands("7s_chess", data_root, out_root, ["vggt_spark"])
    assert len(cmds) == 2
    assert cmds[1][cmds[1].index("--backbone") + 1] == "vggt_spark"


def test_prefix_commands_only_for_sweep_scenes(tmp_path):
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    (data_root / "7scenes/office/office/seq-01").mkdir(parents=True)
    # non-sweep scene → no prefix runs
    assert build_prefix_commands("7s_chess", data_root, out_root, DEFAULT_BACKBONES) == []
    # sweep scene → 2 prefixes (25, 50) × (1 slam + len(DEFAULT_BACKBONES) ours) = 8
    cmds = build_prefix_commands("7s_office", data_root, out_root, DEFAULT_BACKBONES, write_files=False)
    assert len(cmds) == 2 * (1 + len(DEFAULT_BACKBONES)) == 8
    # write_files=False (dry run) must not create prefix keyframes.txt on disk
    assert not out_root.exists() or not list(out_root.rglob("keyframes.txt"))
    # First fraction's group: 1 slam cmd + one ours cmd per backbone, each pointing
    # --slam_tum at that same prefix's own slam dir and its own ours_<backbone> dir
    group = cmds[: 1 + len(DEFAULT_BACKBONES)]
    for backbone, ours in zip(DEFAULT_BACKBONES, group[1:]):
        slam_tum = Path(ours[ours.index("--slam_tum") + 1])
        assert slam_tum == out_root / "7s_office/prefix_25/slam/slam.tum"
        out_dir = Path(ours[ours.index("--output_dir") + 1])
        assert out_dir == out_root / f"7s_office/prefix_25/ours_{backbone}"


def _fake_tum_seq(data_root: Path) -> Path:
    """Minimal TUM seq matching tum_fr3_office's rel path; frame 99.* has a GT gap."""
    seq = data_root / "tum/rgbd_dataset_freiburg3_long_office_household"
    (seq / "rgb").mkdir(parents=True)
    rgb_lines = []
    for ts in ("10.000000", "10.050000", "99.000000"):
        (seq / "rgb" / f"{ts}.png").touch()
        rgb_lines.append(f"{ts} rgb/{ts}.png")
    (seq / "rgb.txt").write_text("\n".join(rgb_lines))
    (seq / "groundtruth.txt").write_text("10.001 0 0 0 0 0 0 1\n10.049 0 0 0 0 0 0 1")
    return seq


def test_build_scene_commands_tum_generates_image_list(tmp_path):
    # TUM scenes: driver writes the GT-filtered allow list and restricts the SLAM
    # ref to it (--image_list); ours-side eval_gt applies the same filter itself.
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    _fake_tum_seq(data_root)
    cmds = build_scene_commands("tum_fr3_office", data_root, out_root, ["vggt_spark"])
    slam = cmds[0]
    allow = out_root / "tum_fr3_office/allowed_frames.txt"
    assert slam[slam.index("--image_list") + 1] == str(allow)
    # GT-gap frame excluded, survivors listed by basename
    assert allow.read_text().splitlines() == ["10.000000.png", "10.050000.png"]
    # ours cmds are untouched — eval_gt's own loader already GT-filters
    assert all("--image_list" not in ours for ours in cmds[1:])


def test_build_scene_commands_tum_write_files_false_skips_disk(tmp_path):
    # Dry run: argv still carries --image_list, but nothing is written
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    cmds = build_scene_commands("tum_fr3_office", data_root, out_root, ["vggt_spark"],
                                write_files=False)
    assert "--image_list" in cmds[0]
    assert not (out_root / "tum_fr3_office/allowed_frames.txt").exists()


def test_build_scene_commands_7scenes_no_image_list(tmp_path):
    # 7-Scenes has no GT-gap filter — behavior must stay byte-identical (no flag, no file)
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    (data_root / "7scenes/chess/chess/seq-01").mkdir(parents=True)
    cmds = build_scene_commands("7s_chess", data_root, out_root, ["vggt_spark"])
    assert all("--image_list" not in c for c in cmds)
    assert not list(out_root.rglob("allowed_frames.txt")) if out_root.exists() else True


def test_build_prefix_commands_tum_slam_carries_image_list(tmp_path):
    # Prefix SLAM re-runs must see the same restricted frame universe as the main run
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    cmds = build_prefix_commands("tum_fr3_office", data_root, out_root, ["vggt_spark"],
                                 write_files=False)
    assert len(cmds) == 4  # 2 fractions x (1 slam + 1 ours)
    allow = str(out_root / "tum_fr3_office/allowed_frames.txt")
    for slam in (cmds[0], cmds[2]):
        assert slam[slam.index("--image_list") + 1] == allow
    for ours in (cmds[1], cmds[3]):
        assert "--image_list" not in ours


def test_build_prefix_commands_7scenes_no_image_list(tmp_path):
    data_root, out_root = tmp_path / "data", tmp_path / "lc_parity"
    cmds = build_prefix_commands("7s_office", data_root, out_root, ["vggt_spark"],
                                 write_files=False)
    assert cmds and all("--image_list" not in c for c in cmds)


def test_done_skip_logic(tmp_path):
    # eval_gt.py convention: metrics.json lands directly inside --output_dir
    out_dir = tmp_path / "ours_vggt_spark"
    cmd = ["python", "eval.py", "--output_dir", str(out_dir)]
    assert _done(cmd) is False
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text("{}")
    assert _done(cmd) is True

    # run_vggt_slam_lc.py convention: metrics.json lands in --out_tum's parent dir
    slam_dir = tmp_path / "slam"
    cmd2 = ["python", "run_vggt_slam_lc.py", "--out_tum", str(slam_dir / "slam.tum")]
    assert _done(cmd2) is False
    slam_dir.mkdir(parents=True)
    (slam_dir / "metrics.json").write_text("{}")
    assert _done(cmd2) is True


def test_run_one_pins_subprocess_imports_to_repo(monkeypatch):
    # Regression: the venv's editable collab_splats install points at the primary
    # checkout, so a worktree probe would silently exercise the wrong code unless
    # _run_one pins PYTHONPATH to its own repo root (+ xfeat for localization).
    import os

    import run_lc_parity as rlp

    captured = {}

    def fake_run(cmd, check, env=None):
        captured["env"] = env

    monkeypatch.setattr(rlp.subprocess, "run", fake_run)
    rlp._run_one([rlp.PY, "eval.py", "--output_dir", "/nonexistent-parity-out"], dry_run=False)
    env = captured["env"]
    assert env is not None and "PYTHONPATH" in env
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(rlp.REPO_ROOT)
    assert parts[1] == str(rlp.REPO_ROOT / "third_party" / "xfeat")
