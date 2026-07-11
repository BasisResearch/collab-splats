"""loop_ablation.json writing: schema, delta arithmetic, loop identity wiring."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))

from collab_splats.geometry.loop_closure.eval import ate_translation


def _fake_gt(N=6):
    """Straight-line GT trajectory along x."""
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, 0, 3] = np.arange(N, dtype=np.float32)
    return poses


def _perturbed(gt, seed, scale=0.2):
    """GT with deterministic translation noise — a fake ablated trajectory."""
    rng = np.random.default_rng(seed)
    out = gt.copy()
    out[:, :3, 3] += scale * rng.standard_normal((gt.shape[0], 3)).astype(np.float32)
    return out


def _fake_lc_stats(gt):
    """lc_stats dict shaped like _subprocess_mode output: 2 accepted + 1 rejected candidate."""
    decisions = [
        {"query_submap": 3, "detected_submap": 0, "query_frame": 2, "detected_frame": 1,
         "accepted": True, "reject_reason": None, "l2_score": 0.3},
        {"query_submap": 4, "detected_submap": 1, "query_frame": 0, "detected_frame": 3,
         "accepted": False, "reject_reason": "verify_ratio", "l2_score": 0.9},
        {"query_submap": 5, "detected_submap": 2, "query_frame": 1, "detected_frame": 0,
         "accepted": True, "reject_reason": None, "l2_score": 0.4},
    ]
    ablations = [_perturbed(gt, seed) for seed in (1, 2)]
    return {
        "loops_applied": 2,
        "candidates": 3,
        "decisions": decisions,
        "ablation_extrinsics": [a.tolist() for a in ablations],
    }, ablations


def test_write_loop_ablation_schema_and_deltas(tmp_path, capsys):
    from eval_gt import _write_loop_ablation

    gt = _fake_gt()
    lc_stats, ablations = _fake_lc_stats(gt)
    ate_full = 0.01
    payload = _write_loop_ablation(tmp_path, lc_stats, ate_full, gt, ate_baseline=0.5)

    # File written and identical to the returned payload
    out_path = tmp_path / "loop_ablation.json"
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == payload

    # Top-level schema
    assert payload["ate_full_lc"] == ate_full
    assert payload["ate_baseline"] == 0.5
    assert len(payload["loops"]) == 2

    # Per-loop schema, delta arithmetic, and identity from accepted decisions in order
    accepted = [d for d in lc_stats["decisions"] if d["accepted"]]
    for k, loop in enumerate(payload["loops"]):
        expected_ate = ate_translation(ablations[k], gt)["rmse"]
        assert loop["index"] == k
        assert loop["query_submap"] == accepted[k]["query_submap"]
        assert loop["detected_submap"] == accepted[k]["detected_submap"]
        assert loop["query_frame"] == accepted[k]["query_frame"]
        assert loop["detected_frame"] == accepted[k]["detected_frame"]
        assert loop["ate_without"] == expected_ate
        assert loop["delta_vs_full"] == expected_ate - ate_full

    # 2-line console summary: most helpful + most harmful loop
    out = capsys.readouterr().out
    assert "most helpful" in out
    assert "most harmful" in out


def test_write_loop_ablation_noop_without_ablations(tmp_path):
    from eval_gt import _write_loop_ablation

    gt = _fake_gt()
    lc_stats = {"loops_applied": 0, "candidates": 1, "decisions": []}
    assert _write_loop_ablation(tmp_path, lc_stats, 0.1, gt) is None
    assert not (tmp_path / "loop_ablation.json").exists()


def test_write_loop_ablation_baseline_optional(tmp_path):
    from eval_gt import _write_loop_ablation

    gt = _fake_gt()
    lc_stats, _ = _fake_lc_stats(gt)
    payload = _write_loop_ablation(tmp_path, lc_stats, 0.01, gt)
    assert payload["ate_baseline"] is None
