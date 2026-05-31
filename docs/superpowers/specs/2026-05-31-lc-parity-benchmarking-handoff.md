# Handoff: Loop Closure parity fixed → cross-model benchmarking

**Date:** 2026-05-31 · **Branch:** refactor/cu121
**For:** the agent benchmarking ATE across backbones now that LC works.

---

## TL;DR

Loop closure now matches VGGT-SLAM numerically. The 17× ATE gap was a **single pose-extraction convention bug** (`R` vs `Rᵀ`) in `closure.py`, not the model, scale, stitching, or optimizer.

| chess/seq-01 baseline (no loops) | before | after | SLAM ref |
|---|---|---|---|
| d10 ATE (2 submaps) | 0.308 m | **0.0171 m** | 0.0176 m |
| d50/d30/d20 (1 submap) | already exact | exact | — |

The fix lives in shared LC code (`closure.py` pose extraction) → **applies to every backbone** that uses windowed loop closure.

---

## What was wrong

`graph.py:decompose_camera` ports VGGT-SLAM's `slam_utils.decompose_camera` but uses the `no_inverse=True` branch (`t = inv(K)@P[:,3]`). SLAM's default is `t = -R @ inv(K)@P[:,3]`, i.e. `R` is camera→world and the camera centre is `C = -R·t`. Our `closure.py` stored `R` as the world→cam rotation, so the trajectory used `C = -Rᵀ·t`.

`R` vs `Rᵀ` (= inverse rotation). For near-identity rotations (single submap, anchored at frame 0) the two agree → single-submap matched at 1e-4. Once a 2nd submap makes the homographies genuinely projective with real rotations, they diverge → smooth bend growing with accumulated rotation (0.53 m at d10).

**Fix:** `closure.py` pose extraction stores `R.T`. One-line, single caller (no BA/mesh blast radius). See `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md` for the full elimination trail.

---

## The parity harness (use this to verify any backbone)

`evals/runners/parity_trace.py --side {slam,ours,diff} --min_disparity N`

- Runs **both real pipelines** with monkey-patched dump hooks (no re-derived math), writes per-stage `.npy` to `/tmp/parity_dN/`, prints a PASS/DIVERGE table.
- Run `slam` then `ours` then `diff` (one model at a time — 46 GB cgroup cap).
- `selected_frames.txt` under `evals/baselines/disparity_sweep/slam_dN/` guarantees identical frames across pipelines.
- Stages: 1 preprocess · 2 forward (ext/intr/depth/conf) · 4 trajectory · 6 boundary scale/H_w · per-node homographies.

Tolerances: forward bf16 ≤1e-2 (observed 0), single-submap trajectory ≤1e-3, **multi-submap ≤5e-3** (bf16 floor, see below).

---

## CRITICAL gotchas before trusting any benchmark number

1. **`eval_gt --backbone vggt_spark` may silently run VGGT-X, not SPARK.** `VGGTSPARKCreator._load_model` inserts the SPARK path then imports `vggt`, but if `vggt` is already cached (VGGT-X imported earlier) the insert is shadowed → you get VGGT-X weights/code under the "vggt_spark" label. The harness forces real SPARK via `sys.path` order. **Verify which model actually loads** before reporting SPARK numbers.

2. **SPARK needs its `_forward` override.** `vggt_spark_creator.py:_forward` runs bf16 with **no outer autocast** (mirrors `solver.run_predictions`). The inherited VGGTX `_forward` wraps in `torch.autocast`, which crashes SPARK's bf16 `camera_head.token_norm` (`expected Float but found BFloat16`). Other backbones keep the autocast path.

3. **Re-verify per-model LC layer calibration — old numbers used the buggy pose extraction.** Prior optimal layers: VGGT-X=20, Omega=16, MapAnything=4 (`project_lc_layer_calibration`). Those multi-submap ATEs were computed with the `R`/`Rᵀ` bug, so they likely shifted. Known prior issues to recheck: Omega LC was *harmful*; MapAnything windowed LC was *broken*. Both may change now.

4. **bf16 floor.** SLAM stores homographies in bf16 (quantized to 1/256); ours float64. ~1.5 mm accumulates over a multi-submap chain. Not a bug — compare via **ATE** (Sim3-aligned) or use a ≥5 mm raw tolerance for multi-submap.

5. **Frame parity.** Always pass `--keyframe_list .../slam_dN/selected_frames.txt` so all models run identical frames. Frame 0 **is** included (not dropped — confirmed). d10's SLAM TUM has a benign overlap-frame duplicate (un-deduped per-submap write; evo keeps first; our `dedup_overlap` keeps submap-0's).

---

## Config used for the validated runs

`submap_size=16`, `submap_overlap=1`, `conf_threshold=25`, `scale_method=none` (baseline) / `rotation_only`. Disparity sweep 50/30/20/10 → frame counts 5/8/12/26 (submaps 1/1/1/2). `max_loops=0` (baseline, no actual loop closures yet).

---

## NOT yet verified (next work, in priority order)

1. **LC with loops** (`max_loops>0`): the similarity gate (`cross_frame_attention_ratio` vs SLAM's `image_match_ratio`) is unverified. All parity above is baseline (no loops closed). This is the next parity target.
2. **Other backbones end-to-end** (vggtx / omega / mapanything) through the corrected pipeline — the fix is backbone-agnostic but only `vggt_spark` is harness-verified.
3. **Lower disparity** (d<10 → 3+ submaps) and **other sequences/datasets**.
4. Stage-5: confirm `compute_ate` dedups the overlap-frame duplicate timestamp.

---

## Key files

- `collab_splats/pointcloud/loop_closure/closure.py` — the fix (`R.T` at pose extraction, ~line 575).
- `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` — SPARK `_forward` override.
- `evals/runners/parity_trace.py` — parity harness.
- `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-design.md` — design + method.
- `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md` — full investigation + root cause.
- `evals/baselines/disparity_sweep/` — frozen SLAM baselines + `selected_frames.txt` per level.
