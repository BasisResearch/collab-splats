# VGGT-SLAM baseline trajectories

VGGT-SLAM (MIT-SPARK) requires Python 3.11 / SL(4) manifold support. The
project env currently runs Python 3.10, so this directory holds **`.pending`
sentinel files** rather than real `.tum` trajectories.

The phase-2 comparison runner (`evals/eval_compare.py`) treats `.pending`
files as `{"status": "pending"}` in `metrics.json` and skips metric
computation cleanly.

When the env upgrade lands:

1. Activate the `vggt-slam` Python 3.11 env (see `evals/runners/run_vggt_slam.py`).
2. Run `third_party/VGGT-SLAM/evals/eval_tum.sh` (or the per-seq invocation).
3. Copy each emitted TUM trajectory into this directory as `<seq>.tum`.
4. Delete the corresponding `<seq>.pending` sentinel.
