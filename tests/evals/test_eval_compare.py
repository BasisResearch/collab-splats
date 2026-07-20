"""Tests for evals/scripts/eval_compare.py — phase-2 unified comparison runner.

Seeds fixtures via `evals.trajectory_io.write_tum`, runs `eval_compare.main`,
and inspects the resulting `metrics.json` plus per-method alignment choices.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "scripts"))

pytest.importorskip("evo")


def _random_w2c(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rots = R.from_rotvec(rng.normal(size=(n, 3)) * 0.2).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, :3, :3] = rots
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return poses


def _scale_w2c_camera_positions(poses_w2c: np.ndarray, s: float) -> np.ndarray:
    out = poses_w2c.copy()
    out[:, :3, 3] = poses_w2c[:, :3, 3] * s
    return out


def _seed_dir(tmp_path: Path, gt: np.ndarray, methods: dict[str, np.ndarray]) -> Path:
    from trajectory_io import write_tum

    results_dir = tmp_path / "chess_seq01"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_tum(results_dir / "gt.tum", gt)
    for name, poses in methods.items():
        write_tum(results_dir / f"{name}.tum", poses)
    return results_dir


def _run_main(argv: list[str]) -> None:
    from eval_compare import main

    saved = sys.argv
    sys.argv = ["eval_compare.py", *argv]
    try:
        main()
    finally:
        sys.argv = saved


# --- happy path --------------------------------------------------------------


def test_compare_emits_metrics_json(tmp_path):
    gt = _random_w2c(20, seed=1)
    pred1 = gt.copy()
    pred1[:, :3, 3] += 0.01
    pred2 = gt.copy()
    pred2[:, :3, 3] += 0.02
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": pred1, "ours_ba": pred2})

    _run_main(["--results-dir", str(results_dir)])

    out_path = results_dir / "metrics.json"
    assert out_path.exists(), "metrics.json must be written"
    payload = json.loads(out_path.read_text())
    assert "methods" in payload
    methods = payload["methods"]
    assert set(methods.keys()) == {"ours_baseline", "ours_ba"}
    for name, body in methods.items():
        assert body["status"] == "ok"
        # All monocular feedforward methods default to sim3 (scale ambiguity); see eval_compare _DEFAULT_ALIGN.
        assert body["align"] == "sim3"
        assert "ate" in body and "rmse" in body["ate"]
        assert "rpe" in body and "trans_rmse" in body["rpe"]


def test_compare_records_pending_status(tmp_path):
    gt = _random_w2c(10, seed=2)
    pred = gt.copy()
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": pred})
    (results_dir / "vggt_slam.pending").write_text("queued\n")

    _run_main(["--results-dir", str(results_dir)])
    payload = json.loads((results_dir / "metrics.json").read_text())
    assert payload["methods"]["vggt_slam"] == {"status": "pending"}
    assert payload["methods"]["ours_baseline"]["status"] == "ok"


def test_compare_default_alignment_picked_per_method(tmp_path):
    gt = _random_w2c(15, seed=3)
    pred = gt.copy()
    pred[:, :3, 3] += 0.01
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": pred, "ours_lc": pred})

    _run_main(["--results-dir", str(results_dir)])
    payload = json.loads((results_dir / "metrics.json").read_text())
    # Default alignment is resolved per method; all monocular conditions now map to sim3.
    assert payload["methods"]["ours_baseline"]["align"] == "sim3"
    assert payload["methods"]["ours_lc"]["align"] == "sim3"


def test_compare_align_override_cli(tmp_path):
    gt = _random_w2c(20, seed=4)
    pred = _scale_w2c_camera_positions(gt, s=2.0)
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": pred})

    _run_main(
        [
            "--results-dir",
            str(results_dir),
            "--align-overrides",
            "ours_baseline=sim3",
        ]
    )
    payload = json.loads((results_dir / "metrics.json").read_text())
    body = payload["methods"]["ours_baseline"]
    assert body["align"] == "sim3", "override should switch alignment to sim3"
    # sim3 should absorb the 2x scale → small ATE; se3 default would have been large.
    assert body["ate"]["rmse"] < 1e-2


def test_compare_unknown_method_warns_defaults_to_sim3(tmp_path, caplog):
    gt = _random_w2c(10, seed=5)
    pred = gt.copy()
    results_dir = _seed_dir(tmp_path, gt, {"random_method": pred})

    with caplog.at_level(logging.WARNING):
        _run_main(["--results-dir", str(results_dir)])

    payload = json.loads((results_dir / "metrics.json").read_text())
    assert payload["methods"]["random_method"]["align"] == "sim3"
    assert any(
        "random_method" in rec.message for rec in caplog.records
    ), f"expected warning mentioning random_method; got: {[r.message for r in caplog.records]}"


def test_compare_raises_on_unexpected_file_type(tmp_path):
    gt = _random_w2c(5, seed=6)
    pred = gt.copy()
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": pred})
    (results_dir / "notes.txt").write_text("stray file\n")

    with pytest.raises(ValueError, match="notes.txt|unexpected"):
        _run_main(["--results-dir", str(results_dir)])


def test_compare_missing_gt_raises(tmp_path):
    from trajectory_io import write_tum

    results_dir = tmp_path / "no_gt"
    results_dir.mkdir()
    write_tum(results_dir / "ours_baseline.tum", _random_w2c(5, seed=7))

    with pytest.raises((FileNotFoundError, ValueError), match="gt"):
        _run_main(["--results-dir", str(results_dir)])


# --- public helper names + grid aggregation ---------------------------------


def test_public_scan_and_format_names(tmp_path):
    """scan_results_dir / format_markdown are the public (un-prefixed) names."""
    from eval_compare import scan_results_dir, format_markdown

    gt = _random_w2c(5, seed=8)
    results_dir = _seed_dir(tmp_path, gt, {"ours_baseline": gt.copy()})
    methods, pending = scan_results_dir(results_dir, results_dir / "gt.tum")
    assert "ours_baseline" in methods
    md = format_markdown({"ours_baseline": {"status": "pending"}})
    assert "ours_baseline" in md


def _seed_cell(root: Path, cell: str, cond: str, ate_rmse: float) -> None:
    cell_dir = root / cell
    cell_dir.mkdir(parents=True)
    (cell_dir / "metrics.json").write_text(
        json.dumps(
            {
                "_config": {"backbone": "vggt_omega", "dataset": "7scenes"},
                cond: {
                    "ate": {"rmse": ate_rmse},
                    "rpe": {"trans_rmse": 0.01, "rot_rmse_deg": 0.5},
                    "auc": {"auc_30": 80.0},
                    "time_s": 1.0,
                },
            }
        )
    )


def test_collect_grid_metrics_and_format(tmp_path):
    """Two <cell>/metrics.json -> 2 rows; markdown renders 2 data lines."""
    from eval_compare import collect_grid_metrics, format_markdown_rows

    _seed_cell(tmp_path, "chess__vggt_omega__baseline", "baseline", 0.10)
    _seed_cell(tmp_path, "chess__vggt_omega__lc", "lc", 0.05)

    rows = collect_grid_metrics(tmp_path)
    assert len(rows) == 2
    assert {r["_cell"] for r in rows} == {
        "chess__vggt_omega__baseline",
        "chess__vggt_omega__lc",
    }

    md = format_markdown_rows(rows)
    data_lines = [ln for ln in md.splitlines() if ln.startswith("|") and "---" not in ln and "ATE RMSE" not in ln]
    assert len(data_lines) == 2
