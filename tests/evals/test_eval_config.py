import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from eval import load_eval_config, build_grid  # noqa: E402


def test_load_config_flat(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(
        "name: t\n"
        "datasets:\n  - {name: 7scenes, seq_dir: /d/seq, keyframe_list: null}\n"
        "backbones: [vggt_omega, vggt_spark]\n"
        "conditions: [baseline, lc]\n"
        "submap_size: 50\n"
        "max_frames: 200\n"
        "output_dir: /out\n"
    )
    cfg = load_eval_config(cfg_file)
    assert cfg.name == "t"
    assert cfg.backbones == ["vggt_omega", "vggt_spark"]
    assert cfg.submap_size == 50


def test_build_grid_product(tmp_path):
    cfg_file = tmp_path / "exp.yaml"
    cfg_file.write_text(
        "name: t\n"
        "datasets:\n  - {name: 7scenes, seq_dir: /d/seq, keyframe_list: null}\n"
        "backbones: [vggt_omega, vggt_spark]\n"
        "conditions: [baseline, lc]\n"
        "output_dir: /out\n"
    )
    cfg = load_eval_config(cfg_file)
    grid = build_grid(cfg)
    assert len(grid) == 4
    assert {c.backbone for c in grid} == {"vggt_omega", "vggt_spark"}
