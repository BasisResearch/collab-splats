import sys
from pathlib import Path

# Local `evals/` is shadowed by an installed `evals` pip package; insert the
# evals dir on sys.path and import the runner module directly (sibling-test convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from runners.run_cross_model_benchmark import RunSpec, build_commands


def test_build_commands_one_per_backbone_frameset():
    specs = [
        RunSpec(backbone="vggt_spark", frameset="slam_d10", submap_size=16,
                conditions=("baseline", "lc"), keyframe_list=Path("kf.txt")),
        RunSpec(backbone="vggtx", frameset="slam_d10_single", submap_size=None,
                conditions=("baseline",), keyframe_list=Path("kf.txt")),
    ]
    cmds = build_commands(specs, seq_dir=Path("/seq"), out_root=Path("/out"))
    assert len(cmds) == 2
    # windowed spark run includes submap flag and both conditions
    c0 = " ".join(cmds[0])
    assert "--backbone vggt_spark" in c0
    assert "--submap_size 16" in c0
    assert "--conditions baseline lc" in c0
    assert "--keyframe_list kf.txt" in c0
    # single-pass run omits --submap_size entirely
    c1 = " ".join(cmds[1])
    assert "--submap_size" not in c1
    assert "--backbone vggtx" in c1
