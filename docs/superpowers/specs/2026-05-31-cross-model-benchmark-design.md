# Design: Cross-model LC benchmark on chess

**Date:** 2026-05-31 · **Branch:** refactor/cu121
**Prereq:** LC pose-extraction fix landed (`1372ac2`); `vggt_spark` is at VGGT-SLAM parity and is the **reference**.
**Read first:** `2026-05-31-cross-model-benchmark-plan.md`, `2026-05-31-lc-parity-benchmarking-handoff.md`, `2026-05-31-vggt-spark-stage-parity-findings.md`.

---

## Goal

Benchmark every feedforward backbone (`vggt_spark`, `vggtx`, `vggt_omega`, `mapanything`) on 7-Scenes chess, ranking each `(model, condition, params)` by ATE/RPE against ground truth, with `vggt_spark` as the reference column. Two distinct questions drive the matrix:

1. **Windowing cost** — does splitting a sequence into submaps + stitching lose accuracy vs running all frames through the model at once (single forward pass), independent of loop closure?
2. **Loop-closure benefit** — over a long trajectory that revisits earlier viewpoints, does LC remove accumulated drift?

These are separate axes needing different frame sets and conditions; the benchmark treats them as such.

---

## Backbones and LC layers

| backbone | LC layer (default) | notes |
|---|---|---|
| `vggt_spark` | 20 | **reference** (VGGT-SLAM parity). Needs `_forward` override (in repo). |
| `vggtx` | 20 | VGGT-X |
| `vggt_omega` | 16 | prior note: LC harmful — **re-measure post-fix** |
| `mapanything` | 4 | prior note: windowed LC broken — **re-measure post-fix** |

Layer handling: **defaults + adaptive sweep.** Run each backbone at its documented layer. For any backbone where `lc` ATE ≥ `baseline` (LC hurts) or where ATE diverges from spark at matched params, sweep ±a few nearby layers to see whether the post-fix optimum moved, and record the curve. Do not sweep layers where the default already wins.

---

## Frame sets (chess/seq-01)

| set | construction | fits single-pass? | purpose |
|---|---|---|---|
| **short** | existing `slam_d10` (26 frames, 2 submaps), `slam_d20` (12 frames, 1 submap) | yes | goal 1 (windowing cost) |
| **long** | `max_frames` raised toward 1000 at low disparity, **gated on the loop probe** | no | goal 2 (drift + real loops) |

The existing baselines (d10/d20/d30/d50, all `max_frames=200`) close **0 loops**. Lower disparity packs more frames into the *same* path span → more submaps but no new revisits; loops require a *longer* span (higher `max_frames`).

---

## Conditions per backbone

| condition | flags | isolates |
|---|---|---|
| `single_pass` | `submap_size=None` (full-batch forward) | accuracy ceiling — **new condition**, short sets only (memory-bound) |
| `baseline` | windowed `submap_size=16`, `lc_retrieval_threshold=0.0` | windowing cost = `baseline − single_pass` |
| `lc` | windowed + loops (`lc_scale_method=rotation_only`) | LC benefit = `lc − baseline` — meaningful only on the long set |
| `ba` | bundle adjustment | expansion |

Single-pass memory ceiling is empirical: probe upward from the short sets, back off on OOM, record the max frame count that fits per backbone.

---

## Execution plan

All heavy runs in **tmux, one model at a time** (46 GB cgroup cap, no parallel GPU jobs). Autonomous through full expansion — no mid-run checkpoint.

### Step 0 — Sanity gate (blocks everything)
Run `vggt_spark` baseline at d10. Assert: (a) printed loaded-module path is the real SPARK dir, **not** VGGT-X (import-cache trap — `eval_gt --backbone vggt_spark` can silently load VGGT-X); (b) ATE ≈ 0.017 m vs SLAM 0.0176. Either fails → stop and report; rest of table is untrustworthy. Loaded-model path is recorded as a column for **every** run.

### Step 1 — Loop probe (gates the goal-2 arm)
Run SLAM-LC (`run_vggt_slam_lc.py --max_frames 1000 --min_disparity <low> --max_loops 1`) and read `get_num_loops()`.
- `> 0` → freeze as `slam_dN_long` (selected_frames + metrics via `run_disparity_sweep.py`); proceed with the long-set matrix.
- `== 0` → chess seq-01 does not revisit. **Report LC-on-chess as structurally untestable, stop the goal-2 arm**, deliver goal-1 + windowing results only, and recommend a looping sequence as explicit follow-up. (Do not fabricate loops.)

### Step 2 — Core matrix
`{4 backbones} × {single_pass (short only), baseline, lc} × submap_size 16 × lc_scale_method rotation_only × {short sets, long set if probe passed}`.
Each run: `--keyframe_list` for frame parity, `--output_ate` JSON saved.

### Step 3 — Expansion (autonomous)
Add `ba` for all backbones; add intermediate disparities (d20/d30) to the reference column. Same serial discipline.

### Step 4 — Divergence localization
Any backbone whose ATE stays far from spark at matched params → `parity_trace.py --side {slam,ours,diff} --min_disparity D` to pin the diverging stage (forward / trajectory / boundary scale / homographies). Report the stage, not just the number.

---

## Comparison method

Metrics (all already in `collab_splats/pointcloud/loop_closure/eval.py`):
- **ATE = Sim3-aligned** (`correct_scale`) RMSE — headline. Never raw pose diff: SLAM stores homographies bf16, ours float64, ~1.5 mm floor over a multi-submap chain.
- **RPE** — both `trans_rmse` (m) and `rot_rmse_deg` (°). Local drift rate; rotation catches what translation misses.
- **Pose AUC @ {5, 15, 30}°** — relative/pairwise, **alignment-free**. `auc_at_threshold` already wired at 30°; add 5° and 15° (one-line — call the same fn at more thresholds). AUC@5 discriminates high-accuracy backbones (@30 saturates near 100).

**Diagnostic use:** AUC is alignment-free, so it isolates *local pose quality* from *global drift*. Windowing cost and LC benefit are global (ATE) effects — if ATE moves while AUC stays flat, the effect is pure stitching/drift, not model pose quality; if AUC moves too, the local poses degraded. This separation drives the "where LC helps vs hurts" analysis.

Table columns: ATE · RPE(t) · RPE(r) · AUC@5 · AUC@15 · AUC@30 · Δ-vs-spark (ATE) · Δ-vs-SLAM (ATE) · loaded-model.

---

## Deliverables

1. `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md` — full table (`backbone × condition × params` → ATE/RPE(t,r)/AUC@{5,15,30}/Δ-vs-spark/Δ-vs-SLAM/loaded-model) + analysis:
   - best model per condition;
   - windowing cost per backbone (`baseline − single_pass`);
   - where LC helps vs hurts per backbone (long set), or the untestable finding;
   - any backbone still diverging from spark, with the traced stage.
2. Raw `--output_ate` JSONs committed under `evals/baselines/cross_model/`.
3. Commit on `refactor/cu121`.

---

## Definition of done

- Sanity gate passed (spark ≈ 0.017 m at d10, real SPARK loaded and logged).
- Core matrix run; ATE/RPE tabulated with loaded-model verified per row.
- Goal 1 answered: windowing cost quantified per backbone (`single_pass` vs `baseline`).
- Goal 2 answered: LC benefit quantified on the long set, **or** chess documented as non-looping with the goal-2 arm stopped and a follow-up recommended.
- Adaptive layer sweep run wherever LC hurt or diverged; results recorded.
- Findings written; any divergence traced to a stage.
