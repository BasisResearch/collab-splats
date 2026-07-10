"""Parity table aggregation from per-scene metrics.json fixtures."""

import json
import sys
from pathlib import Path

import pytest

# `run_lc_parity.py`-style same-dir import: put evals/runners itself on sys.path
# (mirrors tests/evals/test_run_lc_parity.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals" / "runners"))

from build_parity_table import build_table, max_translation_deviation


def _write(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _write_tum_lines(path: Path, rows: list[tuple[float, tuple[float, float, float]]]) -> None:
    """Minimal TUM writer for synthetic fixtures: identity rotation, given translation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 0.000000 0.000000 0.000000 1.000000"
        for ts, t in rows
    ]
    path.write_text("\n".join(lines) + "\n")


def _cells(row: str) -> list[str]:
    """Split a markdown table row into trimmed cells (drops the leading/trailing empties)."""
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_build_table_pass_and_fail_rows(tmp_path):
    # scene A: parity holds
    _write(
        tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 100, "submaps": 7}
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 3, "candidates": 4}},
    )
    # scene B: LC blow-up
    _write(
        tmp_path / "7s_office/slam/metrics.json",
        {"ate_rmse": 0.030, "loop_closures": 10, "keyframes": 300, "submaps": 20},
    )
    _write(
        tmp_path / "7s_office/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.600}, "loops_applied": 4, "candidates": 9}},
    )
    table = build_table(tmp_path)
    assert "7s_chess" in table and "7s_office" in table
    chess_row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    office_row = next(l for l in table.splitlines() if l.startswith("| 7s_office [vggt_spark] |"))
    # No baseline key in either fixture → base gate is still pending.
    assert _cells(chess_row)[-1] == "PASS"  # lc gate
    assert _cells(chess_row)[-2] == "pending"  # base gate
    assert _cells(office_row)[-1] == "FAIL"  # lc gate: ATE blow-up


def test_build_table_skips_missing_scenes(tmp_path):
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0})
    # ours_vggt_spark/ dir exists (a run started) but no metrics.json yet — row pending
    (tmp_path / "7s_chess/ours_vggt_spark").mkdir(parents=True)
    table = build_table(tmp_path)
    assert "pending" in table


def test_build_table_no_backbone_dirs_yields_no_rows(tmp_path):
    # SLAM ran but no ours_<backbone> dir exists at all yet — nothing to discover.
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0})
    table = build_table(tmp_path)
    assert "7s_chess" not in table


def test_main_creates_root_dir(tmp_path, monkeypatch, capsys):
    # CLI must not crash when --root doesn't exist yet (writes header-only table)
    from build_parity_table import main

    root = tmp_path / "not_yet_created"
    monkeypatch.setattr(sys, "argv", ["build_parity_table.py", "--root", str(root)])
    assert main() == 0
    assert (root / "_parity_table.md").exists()


def test_build_table_prefix_sweep_rows(tmp_path):
    # 7s_office is a scaling_sweep scene — a prefix_25 run should surface as its
    # own labeled row alongside the main 100% row.
    _write(
        tmp_path / "7s_office/slam/metrics.json",
        {"ate_rmse": 0.030, "loop_closures": 10, "keyframes": 300, "submaps": 20},
    )
    _write(
        tmp_path / "7s_office/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.031}, "loops_applied": 10, "candidates": 12}},
    )
    _write(
        tmp_path / "7s_office/prefix_25/slam/metrics.json",
        {"ate_rmse": 0.015, "loop_closures": 2, "keyframes": 75, "submaps": 5},
    )
    _write(
        tmp_path / "7s_office/prefix_25/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.016}, "loops_applied": 2, "candidates": 3}},
    )
    table = build_table(tmp_path)
    assert "7s_office@25% [vggt_spark]" in table
    prefix_row = next(l for l in table.splitlines() if "7s_office@25% [vggt_spark]" in l)
    assert _cells(prefix_row)[-1] == "PASS"
    # Scaling suffix belongs on the main row only, never on a prefix row.
    assert "scale:" not in prefix_row


def test_build_table_header_has_base_and_lc_columns(tmp_path):
    table = build_table(tmp_path)
    header = table.splitlines()[0]
    assert "ours ATE (base)" in header
    assert "ours ATE (lc)" in header
    assert "base gate" in header
    assert "lc gate" in header
    assert "max Δt" in header
    assert "| gates |" not in header  # old single-verdict column is gone


def test_build_table_baseline_passes_lc_fails(tmp_path):
    # The pre-fix expectation: baseline (LC off) parity holds, LC-on does not.
    # A single combined verdict would hide this — base gate and lc gate must differ.
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 50, "submaps": 4},
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_spark/metrics.json",
        {
            "baseline": {"ate": {"rmse": 0.0205}},
            "lc": {"ate": {"rmse": 0.500}, "loops_applied": 3, "candidates": 5},
        },
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    cells = _cells(row)
    assert cells[-2] == "PASS"  # base gate: baseline ATE close to SLAM
    assert cells[-1] == "FAIL"  # lc gate: LC ATE blow-up despite matching loop count


def test_build_table_omega_row_harmless_gate(tmp_path):
    # Cross-model backbone: no upstream reference, so base gate is "—" (ungated) and
    # the lc gate is HARMLESS/HARMFUL vs the backbone's OWN baseline, not vs SLAM.
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 50, "submaps": 4},
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_omega/metrics.json",
        {
            "baseline": {"ate": {"rmse": 0.100}},
            "lc": {"ate": {"rmse": 0.102}, "loops_applied": 3, "candidates": 5},  # within 5%/5mm of baseline
        },
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_omega] |"))
    cells = _cells(row)
    assert cells[-2] == "—"  # base gate: reported, not gated
    assert cells[-1] == "HARMLESS"  # lc gate: LC did not degrade omega's own baseline


def test_build_table_omega_row_harmful_gate(tmp_path):
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 50, "submaps": 4},
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_omega/metrics.json",
        {
            "baseline": {"ate": {"rmse": 0.100}},
            "lc": {"ate": {"rmse": 0.300}, "loops_applied": 3, "candidates": 5},  # far worse than baseline
        },
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_omega] |"))
    cells = _cells(row)
    assert cells[-2] == "—"
    assert cells[-1] == "HARMFUL"


def test_build_table_spark_and_omega_rows_coexist(tmp_path):
    # Same scene, two backbones present — each gets its own row with its own gate rules.
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.020, "loop_closures": 3, "keyframes": 50, "submaps": 4},
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 3, "candidates": 4}},
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_omega/metrics.json",
        {"baseline": {"ate": {"rmse": 0.100}}, "lc": {"ate": {"rmse": 0.101}, "loops_applied": 3, "candidates": 4}},
    )
    table = build_table(tmp_path)
    spark_row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    omega_row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_omega] |"))
    assert _cells(spark_row)[-1] == "PASS"
    assert _cells(omega_row)[-1] == "HARMLESS"


def test_build_table_scaling_gate_ok_suffix(tmp_path):
    _write(
        tmp_path / "7s_office/slam/metrics.json",
        {"ate_rmse": 0.030, "loop_closures": 10, "keyframes": 300, "submaps": 20},
    )
    _write(
        tmp_path / "7s_office/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.030}, "loops_applied": 10, "candidates": 12}},
    )
    _write(tmp_path / "7s_office/prefix_25/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 2})
    _write(
        tmp_path / "7s_office/prefix_25/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.022}, "loops_applied": 2, "candidates": 3}},
    )
    table = build_table(tmp_path)
    main_row = next(l for l in table.splitlines() if l.startswith("| 7s_office [vggt_spark] |"))
    prefix_row = next(l for l in table.splitlines() if "7s_office@25% [vggt_spark]" in l)
    assert "scale:OK" in main_row  # 0.030 <= 1.5 * 0.022
    assert "scale:" not in prefix_row


def test_build_table_scaling_gate_blowup_suffix(tmp_path):
    _write(
        tmp_path / "7s_office/slam/metrics.json",
        {"ate_rmse": 0.030, "loop_closures": 10, "keyframes": 300, "submaps": 20},
    )
    _write(
        tmp_path / "7s_office/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.600}, "loops_applied": 10, "candidates": 12}},
    )
    _write(tmp_path / "7s_office/prefix_25/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 2})
    _write(
        tmp_path / "7s_office/prefix_25/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.020}, "loops_applied": 2, "candidates": 3}},
    )
    table = build_table(tmp_path)
    main_row = next(l for l in table.splitlines() if l.startswith("| 7s_office [vggt_spark] |"))
    assert "scale:BLOWUP" in main_row  # 0.600 > 1.5 * 0.020


def test_build_table_non_sweep_scene_never_gets_scale_suffix(tmp_path):
    # 7s_chess is not a scaling_sweep scene — no suffix even if a stray prefix_25
    # dir somehow exists on disk.
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0, "keyframes": 10, "submaps": 1})
    _write(
        tmp_path / "7s_chess/ours_vggt_spark/metrics.json",
        {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 0, "candidates": 0}},
    )
    _write(
        tmp_path / "7s_chess/prefix_25/ours_vggt_spark/metrics.json", {"lc": {"ate": {"rmse": 0.001}, "loops_applied": 0}}
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    assert "scale:" not in row


def test_max_translation_deviation_missing_file_returns_none(tmp_path):
    assert max_translation_deviation(tmp_path / "no_slam.tum", tmp_path / "no_ours.tum") is None


def test_max_translation_deviation_positional_pairing(tmp_path):
    # SLAM timestamps are the source-video frame index (0, 51, 95); ours are the
    # sequential position in the filtered keyframe set (0, 1, 2) — the two
    # conventions diverge past frame 0, so pairing must be positional, not by the
    # literal timestamp value.
    slam_tum = tmp_path / "slam.tum"
    ours_tum = tmp_path / "ours.tum"
    _write_tum_lines(slam_tum, [(0.0, (0, 0, 0)), (51.0, (0.03, 0, 0)), (95.0, (0.0, 0, 0))])
    _write_tum_lines(ours_tum, [(0.0, (0, 0, 0)), (1.0, (0.0, 0, 0)), (2.0, (0.10, 0, 0))])
    dev = max_translation_deviation(slam_tum, ours_tum)
    assert dev == pytest.approx(0.10, abs=1e-6)


def test_max_translation_deviation_drops_duplicate_slam_row(tmp_path):
    # Upstream's solver can emit one duplicate row at a submap boundary.
    slam_tum = tmp_path / "slam.tum"
    ours_tum = tmp_path / "ours.tum"
    _write_tum_lines(
        slam_tum, [(0.0, (0, 0, 0)), (51.0, (0.01, 0, 0)), (51.0, (0.01, 0, 0)), (95.0, (0.05, 0, 0))]
    )
    _write_tum_lines(ours_tum, [(0.0, (0, 0, 0)), (1.0, (0.01, 0, 0)), (2.0, (0.05, 0, 0))])
    dev = max_translation_deviation(slam_tum, ours_tum)
    assert dev == pytest.approx(0.0, abs=1e-6)


def test_build_table_max_dt_column_with_real_tum_files(tmp_path):
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.02, "loop_closures": 0, "keyframes": 3, "submaps": 1},
    )
    _write_tum_lines(
        tmp_path / "7s_chess/slam/slam.tum", [(0.0, (0, 0, 0)), (5.0, (0.02, 0, 0)), (9.0, (0.04, 0, 0))]
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_spark/metrics.json",
        {"baseline": {"ate": {"rmse": 0.021}}, "lc": {"ate": {"rmse": 0.022}, "loops_applied": 0, "candidates": 0}},
    )
    _write_tum_lines(
        tmp_path / "7s_chess/ours_vggt_spark/spark_lc.tum", [(0.0, (0, 0, 0)), (1.0, (0.02, 0, 0)), (2.0, (0.05, 0, 0))]
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    max_dt_cell = _cells(row)[9]  # scene,kf,submaps,SLAM ATE,SLAM loops,base,lc,loops,ΔATE,max Δt,...
    assert max_dt_cell != "—"
    assert float(max_dt_cell) == pytest.approx(0.01, abs=1e-6)


def test_build_table_max_dt_column_uses_backbone_specific_tum_prefix(tmp_path):
    # omega's LC tum is named omega_lc.tum, not spark_lc.tum — the glob in
    # _lc_tum_path must find it without hardcoding the "spark" prefix.
    _write(
        tmp_path / "7s_chess/slam/metrics.json",
        {"ate_rmse": 0.02, "loop_closures": 0, "keyframes": 3, "submaps": 1},
    )
    _write_tum_lines(
        tmp_path / "7s_chess/slam/slam.tum", [(0.0, (0, 0, 0)), (5.0, (0.02, 0, 0)), (9.0, (0.04, 0, 0))]
    )
    _write(
        tmp_path / "7s_chess/ours_vggt_omega/metrics.json",
        {"baseline": {"ate": {"rmse": 0.021}}, "lc": {"ate": {"rmse": 0.022}, "loops_applied": 0, "candidates": 0}},
    )
    _write_tum_lines(
        tmp_path / "7s_chess/ours_vggt_omega/omega_lc.tum", [(0.0, (0, 0, 0)), (1.0, (0.02, 0, 0)), (2.0, (0.05, 0, 0))]
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_omega] |"))
    max_dt_cell = _cells(row)[9]
    assert max_dt_cell != "—"
    assert float(max_dt_cell) == pytest.approx(0.01, abs=1e-6)


def test_build_table_header_has_loop_pr_columns(tmp_path):
    table = build_table(tmp_path)
    header = table.splitlines()[0]
    assert "loop P" in header and "loop R" in header


def test_build_table_loop_pr_em_dash_when_artifacts_missing(tmp_path):
    # No gt.tum / lc_decisions_lc.json in the arm dir → loop P/R cells are em-dashes.
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0, "keyframes": 3, "submaps": 1})
    _write(tmp_path / "7s_chess/ours_vggt_spark/metrics.json", {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 0}})
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    cells = _cells(row)
    assert cells[10] == "—" and cells[11] == "—"  # loop P, loop R


def test_build_table_loop_pr_from_gt_verified_revisit(tmp_path):
    # Full synthetic arm: 3 submaps of 16 frames; submap 2 revisits submap 0 and the
    # single accepted loop links them → loop P = 1.00 and loop R = 1.00 in the row.
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 1, "keyframes": 48, "submaps": 3})
    arm = tmp_path / "7s_chess/ours_vggt_spark"
    _write(
        arm / "metrics.json",
        {
            "_config": {"submap_size": 16},
            "baseline": {"ate": {"rmse": 0.021}},
            "lc": {"ate": {"rmse": 0.020}, "loops_applied": 1, "candidates": 1},
        },
    )
    # Clusters at x=0, x=100, x=0 — submaps 0 and 2 co-located, submap 1 far away.
    xs = [c + 0.1 * k for c in (0.0, 100.0, 0.0) for k in range(16)]
    _write_tum_lines(arm / "gt.tum", [(float(i), (x, 0.0, 0.0)) for i, x in enumerate(xs)])
    _write(
        arm / "lc_decisions_lc.json",
        [{"query_submap": 2, "query_frame": 0, "detected_submap": 0, "detected_frame": 0,
          "accepted": True, "reject_reason": None, "l2_score": 0.5}],
    )
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    cells = _cells(row)
    assert cells[10] == "1.00"  # loop P
    assert cells[11] == "1.00"  # loop R


def test_build_table_max_dt_em_dash_when_tum_missing(tmp_path):
    _write(tmp_path / "7s_chess/slam/metrics.json", {"ate_rmse": 0.02, "loop_closures": 0, "keyframes": 3, "submaps": 1})
    _write(tmp_path / "7s_chess/ours_vggt_spark/metrics.json", {"lc": {"ate": {"rmse": 0.021}, "loops_applied": 0}})
    table = build_table(tmp_path)
    row = next(l for l in table.splitlines() if l.startswith("| 7s_chess [vggt_spark] |"))
    assert _cells(row)[9] == "—"
