"""Tests for the evals grid runner; Reconstructor and zarr loading are stubbed."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from evals import eval as ev
from evals.datasets import EvalDataset


def _grid_yaml(tmp_path, **overrides):
    """Write a two-condition grid YAML over one 7-Scenes dataset."""
    raw = {
        "name": "t",
        "output_dir": str(tmp_path / "out"),
        "base": {"semantics": {"enabled": False}},
        "datasets": [
            {"name": "chess", "type": "7scenes", "seq_dir": "x", "max_frames": 5}
        ],
        "conditions": {
            "omega": {"pointcloud": {"backend": "vggt_omega"}},
            "lc": {"pointcloud": {"loop_closure": True}},
        },
    }
    raw.update(overrides)
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_build_grid_merges_base_and_override(tmp_path):
    cells = ev.build_grid(ev.load_eval_config(_grid_yaml(tmp_path)))
    assert [c.output_dir.name for c in cells] == ["chess__omega", "chess__lc"]
    assert cells[1].config == {
        "semantics": {"enabled": False},
        "pointcloud": {"loop_closure": True},
    }


def test_load_eval_config_rejects_empty_conditions(tmp_path):
    with pytest.raises(ValueError, match="condition"):
        ev.load_eval_config(_grid_yaml(tmp_path, conditions={}))


def test_load_eval_config_rejects_unknown_dataset(tmp_path):
    ds = [{"name": "k", "type": "kitti", "seq_dir": "x", "max_frames": 5}]
    with pytest.raises(KeyError, match="kitti"):
        ev.load_eval_config(_grid_yaml(tmp_path, datasets=ds))


def test_dry_run_prints_cells_and_writes_nothing(tmp_path, capsys):
    ev.run_grid(_grid_yaml(tmp_path), dry_run=True)
    assert capsys.readouterr().out.count("--cell") == 2
    assert not (tmp_path / "out").exists()


def test_stage_frames_gt_order(tmp_path):
    srcs = []
    for name in ["b.png", "a.png"]:
        (tmp_path / name).write_bytes(b"x")
        srcs.append(tmp_path / name)
    ev._stage_frames(srcs, tmp_path / "in")
    staged = sorted((tmp_path / "in").iterdir())
    assert [p.name for p in staged] == ["000000.png", "000001.png"]
    assert staged[0].resolve() == (tmp_path / "b.png").resolve()


def test_gt_match_reorders():
    paths = [Path("frame_000002"), Path("frame_000000"), Path("frame_000001")]
    rows, gt_idx = ev._gt_match(paths, 3)
    assert rows.tolist() == [1, 2, 0] and gt_idx.tolist() == [0, 1, 2]


def test_gt_match_skips_unregistered_frame():
    rows, gt_idx = ev._gt_match([Path("frame_000002"), Path("frame_000000")], 3)
    assert rows.tolist() == [1, 0] and gt_idx.tolist() == [0, 2]


@pytest.mark.parametrize("names", [["frame_000000", "frame_000000"], ["frame_000003"]])
def test_gt_match_rejects_duplicate_or_out_of_range(names):
    with pytest.raises(ValueError, match="GT"):
        ev._gt_match([Path(n) for n in names], 3)


def test_pred_depth_at_gt_res_places_crop():
    out = ev._pred_depth_at_gt_res(
        np.ones((2, 2), np.float32), np.array([2, 1, 6, 5]), (6, 8)
    )
    assert out[1:5, 2:6].min() == 1.0 and out.sum() == 16.0


def _fake_run(tmp_path, monkeypatch, n=6, write_report=True, unregistered=()):
    """Stub the dataset, Reconstructor and zarr loader for one cell; `unregistered` GT frames get no pose."""
    gt_c2w = np.tile(np.eye(4), (n, 1, 1))
    gt_c2w[:, 0, 3] = np.arange(n) * 0.1
    gt_c2w[:, 1, 3] = np.sin(np.arange(n))
    imgs = []
    for i in range(n):
        (tmp_path / f"{i}.png").write_bytes(b"x")
        imgs.append(tmp_path / f"{i}.png")
    ds = EvalDataset(images=imgs, gt_poses=np.linalg.inv(gt_c2w).astype(np.float32))
    monkeypatch.setattr(ev, "get_dataset", lambda _t: lambda *_a, **_k: ds)

    class FakeRecon:
        def __init__(self, config):
            self.config = {"splats": {"enabled": False}, **config}
            self.out = Path(config["output_path"])
            self.pointcloud_zarr = self.out / "vggt_omega" / "pointcloud.zarr"

        def run(self):
            if write_report:
                (self.out / "vggt_omega").mkdir(parents=True)
                (
                    self.out / "vggt_omega" / "reconstruction_quality_report.json"
                ).write_text('{"ok": 1}')

    # Reconstruction returns frames reversed; the runner must undo that
    rev = np.array([i for i in range(n)[::-1] if i not in unregistered])
    ff = SimpleNamespace(
        extrinsics=np.linalg.inv(gt_c2w[rev]).astype(np.float32),
        image_paths=[Path(f"frame_{i:06d}") for i in rev],
        depth=None,
        original_coords=None,
    )
    monkeypatch.setattr(ev, "Reconstructor", FakeRecon)
    monkeypatch.setattr(
        ev.PointcloudResult, "load_zarr", classmethod(lambda cls, *a, **k: ff)
    )
    return ev.EvalCell("chess", "7scenes", Path("x"), n, "omega", {}, tmp_path / "cell")


def test_run_cell_scores_reordered_poses(tmp_path, monkeypatch):
    cell = _fake_run(tmp_path, monkeypatch)
    m = ev.run_cell(cell)
    assert m["ate"]["rmse"] < 1e-6
    assert m["reports"]["reconstruction_quality_report"] == {"ok": 1}
    assert (
        json.loads((cell.output_dir / "eval_metrics.json").read_text())["condition"]
        == "omega"
    )
    assert (cell.output_dir / "plots" / "trajectory.png").exists()


def test_run_cell_scores_registered_frames_and_penalizes_auc(tmp_path, monkeypatch):
    m = ev.run_cell(_fake_run(tmp_path, monkeypatch, unregistered=(2,)))

    # ATE over the 5 registered frames stays exact; the missing one is reported, not scored
    assert m["ate"]["rmse"] < 1e-6
    assert (m["registered_frames"], m["n_frames"]) == (5, 6)

    # AUC: the 10 of 30 directed pairs touching frame 2 are failures
    assert m["auc"]["auc_30"] == pytest.approx(100.0 * 20 / 30)


def test_run_cell_missing_report_raises(tmp_path, monkeypatch):
    cell = _fake_run(tmp_path, monkeypatch, write_report=False)
    with pytest.raises(ValueError, match="reconstruction_quality_report"):
        ev.run_cell(cell)


def test_run_grid_skips_done_cells_and_aggregates(tmp_path, monkeypatch):
    path = _grid_yaml(tmp_path)
    cells = ev.build_grid(ev.load_eval_config(path))
    done = cells[0].output_dir
    done.mkdir(parents=True)
    metrics = {
        "n_frames": 500,
        "registered_frames": 498,
        "ate": {"rmse": 0.01},
        "rpe": {"trans_rmse": 0.02, "rot_rmse_deg": 0.3},
        "auc": {"auc_30": 90.0},
        "depth": None,
    }
    (done / "eval_metrics.json").write_text(json.dumps(metrics))

    calls = []
    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, check: calls.append(cmd))
    ev.run_grid(path)

    assert len(calls) == 1 and calls[0][-1] == "chess__lc"
    table = (tmp_path / "out" / "comparison.md").read_text()
    assert "chess__omega" in table and "498/500" in table and "0.0100" in table
