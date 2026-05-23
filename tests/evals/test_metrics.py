"""evo wrapper tests for evals/metrics.py.

Numerical correctness verified against evo's Python API. The Sim3 vs SE3
alignment switch must change behaviour when scales differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))


pytest.importorskip("evo")


def _random_w2c(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rots = R.from_rotvec(rng.normal(size=(n, 3)) * 0.2).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    poses[:, :3, :3] = rots
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    return poses


def _scale_w2c_camera_positions(poses_w2c: np.ndarray, s: float) -> np.ndarray:
    """Return new w2c poses whose camera positions in world are scaled by s.

    For w2c T = [R | t]: camera position p_world = -R^T t. Scaling p by s gives
    new t' = -R @ (s * p) = s * t.
    """
    out = poses_w2c.copy()
    out[:, :3, 3] = poses_w2c[:, :3, 3] * s
    return out


def _write(tmp_path: Path, name: str, poses: np.ndarray, ts: np.ndarray | None = None) -> Path:
    from trajectory_io import write_tum

    p = tmp_path / name
    write_tum(p, poses, timestamps=ts)
    return p


# --- ATE ---------------------------------------------------------------------


def test_ate_identical_trajectories_is_zero(tmp_path):
    from metrics import compute_ate

    poses = _random_w2c(10)
    p = _write(tmp_path, "pred.tum", poses)
    g = _write(tmp_path, "gt.tum", poses)
    out = compute_ate(p, g, align="se3")
    assert out["rmse"] < 1e-6
    assert out["mean"] < 1e-6


def test_ate_returns_required_keys(tmp_path):
    from metrics import compute_ate

    poses = _random_w2c(10)
    p = _write(tmp_path, "pred.tum", poses)
    g = _write(tmp_path, "gt.tum", poses)
    out = compute_ate(p, g, align="se3")
    for k in ("rmse", "mean", "median", "max", "std"):
        assert k in out, f"missing key {k}"
        assert isinstance(out[k], float)


def test_ate_se3_offset_aligns_to_zero(tmp_path):
    """SE3 alignment removes a rigid SE(3) world-frame offset → rmse ≈ 0.

    World-frame transform M applied to GT: new w2c = old w2c @ M^-1.
    """
    from metrics import compute_ate

    gt = _random_w2c(20)
    M = np.eye(4)
    M[:3, :3] = R.from_rotvec([0.4, -0.2, 0.1]).as_matrix()
    M[:3, 3] = [1.0, 2.0, -0.5]
    pred = gt @ np.linalg.inv(M)
    g = _write(tmp_path, "gt.tum", gt)
    p = _write(tmp_path, "pred.tum", pred)
    out = compute_ate(p, g, align="se3")
    assert out["rmse"] < 1e-3, f"se3 align should remove SE3 offset, got {out['rmse']}"


def test_ate_scaled_sim3_vs_se3(tmp_path):
    """Sim3 align removes scale; SE3 does not."""
    from metrics import compute_ate

    gt = _random_w2c(30, seed=11)
    pred = _scale_w2c_camera_positions(gt, s=2.0)
    g = _write(tmp_path, "gt.tum", gt)
    p = _write(tmp_path, "pred.tum", pred)

    out_sim3 = compute_ate(p, g, align="sim3")
    out_se3 = compute_ate(p, g, align="se3")

    assert out_sim3["rmse"] < 1e-3, f"sim3 should absorb scale, got {out_sim3['rmse']}"
    assert out_se3["rmse"] > 0.1, f"se3 should NOT absorb scale, got {out_se3['rmse']}"


# --- RPE ---------------------------------------------------------------------


def test_rpe_identical_trajectories_is_zero(tmp_path):
    from metrics import compute_rpe

    poses = _random_w2c(10)
    p = _write(tmp_path, "pred.tum", poses)
    g = _write(tmp_path, "gt.tum", poses)
    out = compute_rpe(p, g, align="se3", delta=1)
    assert out["trans_rmse"] < 1e-6


def test_rpe_returns_trans_and_rot_keys(tmp_path):
    from metrics import compute_rpe

    poses = _random_w2c(10)
    p = _write(tmp_path, "pred.tum", poses)
    g = _write(tmp_path, "gt.tum", poses)
    out = compute_rpe(p, g, align="se3", delta=1)
    for k in ("trans_rmse", "rot_rmse_deg"):
        assert k in out
        assert isinstance(out[k], float)


# --- align knob --------------------------------------------------------------


def test_align_choice_validation(tmp_path):
    from metrics import compute_ate

    poses = _random_w2c(5)
    p = _write(tmp_path, "pred.tum", poses)
    g = _write(tmp_path, "gt.tum", poses)
    with pytest.raises(ValueError, match="align"):
        compute_ate(p, g, align="bogus")


# --- CLI parity --------------------------------------------------------------


def test_compute_ate_matches_evo_ape_cli(tmp_path):
    """Spec acceptance criterion 3: agreement with `evo_ape … -as` CLI within 1e-9."""
    import shutil
    import subprocess

    if shutil.which("evo_ape") is None:
        pytest.skip("evo_ape CLI not installed")

    from metrics import compute_ate

    rng = np.random.default_rng(123)
    gt = _random_w2c(30, seed=99)
    pred = gt.copy()
    pred[:, :3, 3] += rng.normal(scale=0.05, size=(30, 3))

    g = _write(tmp_path, "gt.tum", gt)
    p = _write(tmp_path, "pred.tum", pred)

    py_out = compute_ate(p, g, align="sim3")

    cli = subprocess.run(
        ["evo_ape", "tum", str(g), str(p), "-as", "--no_warnings"],
        check=True, capture_output=True, text=True,
    )
    # evo_ape stdout has a "rmse  X.XXXXX" line in its summary table.
    rmse_cli = None
    for ln in cli.stdout.splitlines():
        toks = ln.strip().split()
        if len(toks) >= 2 and toks[0] == "rmse":
            rmse_cli = float(toks[1])
            break
    assert rmse_cli is not None, f"could not parse rmse from evo_ape stdout:\n{cli.stdout}"
    # evo_ape stdout prints rmse to 6 decimals; round-off bounds the agreement.
    assert abs(py_out["rmse"] - rmse_cli) < 1e-5, f"py {py_out['rmse']} vs cli {rmse_cli}"
