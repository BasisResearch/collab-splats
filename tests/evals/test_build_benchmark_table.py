import sys
from pathlib import Path

# Local `evals/` is shadowed by an installed `evals` pip package; insert the
# evals dir on sys.path and import the module directly (sibling-test convention).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from runners.build_benchmark_table import assemble_rows


def test_assemble_rows_computes_deltas():
    runs = {
        ("vggt_spark", "slam_d10", "baseline"): {"ate": {"rmse": 0.017},
            "rpe": {"trans_rmse": 0.01, "rot_rmse_deg": 1.0},
            "auc": {"auc_5": 50, "auc_15": 80, "auc_30": 90}},
        ("vggtx", "slam_d10", "baseline"): {"ate": {"rmse": 0.027},
            "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 2.0},
            "auc": {"auc_5": 40, "auc_15": 70, "auc_30": 85}},
    }
    slam_ate = {"slam_d10": 0.0176}
    rows = assemble_rows(runs, slam_ate, reference_backbone="vggt_spark")
    vggtx = [r for r in rows if r["backbone"] == "vggtx"][0]
    assert abs(vggtx["delta_vs_spark"] - 0.010) < 1e-6   # 0.027 - 0.017
    assert abs(vggtx["delta_vs_slam"] - (0.027 - 0.0176)) < 1e-6
