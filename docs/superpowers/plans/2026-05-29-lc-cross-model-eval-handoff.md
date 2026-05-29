# LC Cross-Model Evaluation Handoff

**Date:** 2026-05-29
**Branch:** refactor/cu121
**Goal:** Evaluate loop closure performance with the corrected windowing (stride=K, K+O frames per
submap — now matches VGGT-SLAM exactly) across all feedforward backends.

---

## What Changed Today

`wrappers.py` `_run_lc_loop` now matches VGGT-SLAM's window convention:

| | Before | After (today) |
|---|---|---|
| Stride | `K - O = 15` | `K = 16` |
| Frames fed to VGGT | `K+O = 17` (trimmed to K=16 stored) | `K+O = 17` (all 17 stored) |
| Carry/boundary frame | frame 15 | frame 16 (VGGT-SLAM exact) |

Commits: `813ece1`, `c6825df`, latest (stride fix).

VGGT-SPARK result with this fix: **0.0186m Sim3 ATE** on chess_seq01 29-frame keyframe set
(VGGT-SLAM target: 0.038m).

---

## Per-Backend Status

| Backbone | `_lc_layer_index` | Windowed LC | Known blocker before eval |
|---|---|---|---|
| `vggt_spark` | 20 | ✓ working | `verify_match_ratio` threshold needs recalibration (0 loops fired at 0.85) |
| `vggtx` | 20 | ✓ working (same code path) | Same threshold issue |
| `vggt_omega` | 16 | ✓ code path works | `verify_match_ratio` effectively off (Omega mtq=1.328 >> 0.85); Omega LC was harmful before (0.3225→0.4958m) — raise threshold to 0.99 first |
| `mapanything` | 4 | ✗ architecturally broken | `_forward` ignores window slice, processes all frames — windowed LC impossible without fix |

---

## Pre-Eval Fixes Required

### 1. Recalibrate `verify_match_ratio` per backbone (REQUIRED for vggtx, vggt_omega)

The `verify_match_ratio` in `LoopClosureConfig` (default 0.85) is the `cross_frame_attention_ratio`
gate. Scores differ per model:
- `vggt_spark` / `vggtx`: scores cluster ~0.80–0.85; 0.85 threshold fires 0 loops
- `vggt_omega`: scores ~1.328 (all above 0.85 — false positives; was 0.496m ATE)
- `mapanything`: blocked by `_forward` issue

**For `vggtx` / `vggt_spark`:** lower threshold to ~0.65–0.70 (per 2026-05-27 calibration finding).
Override in eval call via `--lc_verify_match_ratio 0.70` (check `eval_gt.py --help` for exact flag name).
Or set `LoopClosureConfig.verify_match_ratio` default per-backbone in the creator's `default_lc_config`.

**For `vggt_omega`:** raise threshold to 0.99 to block false-positive LCs. This was noted as
"not yet implemented" in the 2026-05-28 parity run 3 handoff.

### 2. Fix MapAnything `_forward` windowing (REQUIRED for mapanything LC)

`collab_splats/pointcloud/feedforward/mapanything.py` `_forward` ignores its `views` argument
and processes `self._processed_views` (all frames). This means windowed submap inference
always uses all frames — the window slice is ignored. Fix: refactor `_forward` to accept and
use the `views` tensor argument directly, not `self._processed_views`.

Reference: `docs/superpowers/specs/2026-05-28-lc-diagnostics-and-model-parity-design.md` §69.

---

## Eval Commands

Run in tmux. Python env: `/opt/conda/envs/reconstruction/bin/python`. OOM risk — no parallel runs.

### VGGT-SPARK (already done today, re-run for record)

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_spark --conditions baseline lc \
  --submap_size 16 --lc_scale_method none --max_frames 200
```

### VGGT-X

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggtx --conditions baseline lc \
  --submap_size 16 --max_frames 200
```

Note: if 0 loops fire, re-run with lower `verify_match_ratio` (see pre-eval fix 1).

### VGGT-Omega

**First:** ensure `verify_match_ratio=0.99` for Omega (to prevent false-positive LCs). Check
`collab_splats/pointcloud/feedforward/vggt_omega.py` for `default_lc_config` or pass via CLI.

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --backbone vggt_omega --conditions baseline lc \
  --submap_size 16 --max_frames 200
```

Expected: LC should no longer be harmful once verify_match_ratio=0.99 blocks false loops.
If still harmful, investigate pose graph edge convention (open issue from parity run 3).

### MapAnything

MapAnything single-pass (no LC) baseline: **0.1103m** — already strong, no LC needed for quality.
LC eval blocked until `_forward` windowing is fixed. For now, only run baseline:

```bash
/opt/conda/envs/reconstruction/bin/python evals/eval_gt.py \
  --dataset 7scenes --seq_dir data/7scenes/chess/seq-01 \
  --backbone mapanything --conditions baseline \
  --max_frames 200
```

---

## Target ATE Table

Fill in after runs. Previous results (before today's stride fix, from parity run 3):

| Backbone | Baseline SE3 ATE | LC SE3 ATE | LC delta | Notes |
|---|---|---|---|---|
| vggt_spark | 0.2724m (today) | 0.3768m (today) | **+38% worse** | threshold not calibrated |
| vggtx | 0.3184m | 0.2664m | −16.3% | from parity run 3 — pre stride fix |
| vggt_omega | 0.3225m | 0.4958m | **+54% worse** | verify gate not raised; pre stride fix |
| mapanything | 0.1103m | — | — | windowed LC broken |

After this eval (fill in):

| Backbone | Baseline SE3 ATE | LC SE3 ATE | LC delta | Notes |
|---|---|---|---|---|
| vggt_spark | | | | stride=K, recalibrated threshold |
| vggtx | | | | stride=K, recalibrated threshold |
| vggt_omega | | | | verify=0.99 |
| mapanything | | | | baseline only |

---

## Open Issues After This Eval

1. **Omega LC regression root cause** — even with verify=0.99, if LC is still harmful, the issue
   is in the pose graph edge convention for Omega. Prior hypothesis: Omega's per-submap normalized
   coordinates differ from VGGT-X in a way that makes inter-submap T estimation wrong.

2. **MapAnything windowed LC** — architectural fix to `_forward` needed before any windowed
   eval is possible. Until fixed, MapAnything can only run single-pass (all frames at once).

3. **Multi-scene validation** — chess_seq01 is low-disparity. All results above are single-scene.
   Should replicate on fire_seq01, office_seq01 before drawing conclusions.

4. **verify_match_ratio tuning** — after recalibration, update `default_lc_config` in each
   backend's creator dataclass so the threshold is baked in and doesn't need per-run CLI flags.

---

## Python env

```bash
/opt/conda/envs/reconstruction/bin/python
```
