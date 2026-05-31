# Design: vggt_spark ↔ VGGT-SLAM Stage-by-Stage Parity

**Date:** 2026-05-31
**Branch:** refactor/cu121
**Goal:** Confirm our `vggt_spark` loop-closure pipeline produces numerically identical intermediates to VGGT-SLAM at every stage, on identical keyframes. Start at `min_disparity=50`, locate first divergence, then escalate level-by-level explaining each deviation.

Supersedes the inline-re-derive approach of `compare_slam_ours.py` (which re-implements SLAM math and risks drifting from the real `solver.py`).

---

## Background

Prior work (`2026-05-30-vggt-spark-parity-sweep-design.md`) established frame parity via `selected_frames.txt` and flagged a similarity-score gap. This spec drills into **per-stage numerical parity** so any ATE gap is attributed to a specific stage, not guessed.

Two facts verified during brainstorming:

1. **Frame 0 is NOT dropped.** VGGT-SLAM's `frame_overlap.py:24-26` returns `True` on the first `compute_disparity` call (seeds the keyframe). Our runner shares that tracker, and `frame-000000.color.png` is line 1 of every `selected_frames.txt`. The earlier "skipping frame 0" symptom is in **downstream matching**, not selection.

2. **d10 has a 26-vs-27 mismatch — duplicate is an un-deduped byproduct, not intentional.** d10 keyframes = 26; VGGT-SLAM TUM = 27. The runner keeps the overlap frame for continuity (`run_vggt_slam_lc.py:156`, `[-overlap:]`), so the boundary frame is a real member of both submaps. `write_poses_to_file` (`map.py:142-162`) writes every submap's `frame_ids` with **no dedup** → the shared frame appears twice (submap-0 coords, then submap-1 coords). Our `dedup_overlap` (`closure.py:291`) is **first-writer-wins**, keeping submap-0's estimate. evo associates by timestamp and keeps the first occurrence (also submap-0's), so the two pipelines are **semantically equivalent** *iff* the ATE tool dedups the duplicate timestamp. **Stage-5 must verify `compute_ate` does not double-count the dup line** — if it does, SLAM RMSE is biased by one frame.

---

## Escalation Ladder (driven by submap structure)

`submap_size=16`, `overlap=1` → window = 17 frames.

| level | keyframes | submaps | first exercises |
|---|---|---|---|
| d50 | 5 | 1 | preprocess, VGGT forward, world-points, pose extraction, ATE align |
| d30 | 8 | 1 | (same, more frames) |
| d20 | 12 | 1 | (same) |
| d10 | 26 | 2 | boundary scale + T + H_w, cross-submap PGO, overlap dedup |

**d50 first**: single submap isolates the forward/preprocess/extraction path with no graph math. Only once d50 matches do we move to d10, where boundary + PGO logic is the *only* new variable.

---

## Stages Compared

| # | Stage | Ours (real code) | VGGT-SLAM (real code) | Compare |
|---|---|---|---|---|
| 1 | Frame set + preprocessed image tensor | eval_gt matching → creator preprocess | flow_tracker → `load_and_preprocess_images` | path list equal; pixel tensor `max|Δ|` |
| 2 | VGGT forward: depth, extrinsic, intrinsic, depth_conf | `creator._forward` | `solver.run_predictions` | per-frame `max|Δ|` each field |
| 3 | Point cloud | `_raw_to_world_points` (world/cam0) | `add_points` (camera-local) | per-frame, after common frame transform |
| 4 | Pose extraction R,t | `decompose_camera` | `vggt_slam.decompose_camera` | per-frame `max|Δ|` |
| 5 | ATE vs 7-Scenes GT | `compute_ate` | `compute_ate` | RMSE, alignment matrix |
| 6 | *(d10+)* boundary scale, T, H_w | `closure.run_pose_graph_optimization` | `solver.add_submap` | scalar scale, H diag, det |
| 7 | *(d10+)* PGO optimize + extract | `graph.PoseGraph.optimize` | GTSAM graph optimize | per-node H, final R,t |
| 8 | *(d10+)* overlap dedup | `dedup_overlap` | `write_poses_to_file` | final frame count + per-frame pose |

---

## Instrumentation: Monkey-Patch Hooks

No edits to vendored VGGT-SLAM source or our pipeline source. The harness wraps real methods at runtime, calls the original, and dumps inputs/outputs to `.npy`. Reversible; restores originals in `finally`.

Patched (ours): `closure.run_pose_graph_optimization`, `graph.estimate_scale_pairwise`, `graph.decompose_camera`, `feedforward.base._raw_to_world_points`.
Patched (SLAM): `Solver.run_predictions`, `Solver.add_points`, `Solver.add_submap`, `Map.write_poses_to_file`, `decompose_camera`.

Each patch writes `{pipeline}/stage{N}_{field}_{frame}.npy` under the run dir.

---

## Deliverables

| # | File | Purpose |
|---|---|---|
| 1 | `evals/runners/parity_trace.py` | Harness: `--min_disparity`, runs both pipelines with dump hooks, writes per-stage `.npy`, prints diff table per stage with first-divergence flag |
| 2 | `docs/superpowers/specs/2026-05-31-vggt-spark-stage-parity-findings.md` | Running log: per level, which stage first diverges and the root cause |

Output layout: `/tmp/parity_d<N>/{ours,slam}/stage<N>_*.npy` + `diff_report.json`.

---

## Diff Protocol

For each stage, in order:
1. Load ours + slam arrays for the stage.
2. Align by frame path (handle the d10 overlap-dup: match by basename, dedup SLAM side).
3. Report `max|Δ|`, `mean|Δ|`, and shape. Flag divergence if `max|Δ| > 1e-4` (tunable per stage; pose tol looser for bf16 forward).
4. **Stop at first diverging stage**, record root cause in findings doc, fix or document, re-run that level.

---

## Stage Report (reported back per run)

After every level, the harness emits a per-stage report — printed table + `diff_report.json` — and I relay it. One row per stage, in order:

```
level d50  | frames=5 submaps=1
stage 1 preprocess     PASS  max|Δ|=0.0e+00  shape=(5,3,294,518)
stage 2 vggt_forward   PASS  max|Δ|=3.1e-03  (bf16 tol 1e-2) depth/ext/intr/conf
stage 3 pointcloud     PASS  max|Δ|=4.0e-05
stage 4 pose_extract   PASS  max|Δ|=2.2e-06
stage 5 ate_align      PASS  ours=0.0176 slam=0.0176 Δ=0.0000  dup_check=ok
-> first divergence: none
```

Each row: stage name, PASS/DIVERGE, `max|Δ|`, tolerance used, and fields covered. On DIVERGE the report stops at that stage and records the offending frame/value + suspected cause. The findings doc accumulates one block per level.

## Phases

1. **d50 baseline parity** (single submap): stages 1–5. Establish forward/preprocess/extraction parity. *Hard gate.*
2. **d30, d20** (single submap): confirm d50 fix holds with more frames.
3. **d10** (2 submaps): stages 6–8. Boundary scale, PGO, overlap dedup. Resolve the 26-vs-27 matching.
4. **Lower levels / LC verification** (similarity gate): only after geometry parity confirmed. Out of scope until phases 1–3 pass.

---

## Out of Scope

- Other sequences/datasets (chess/seq-01 only).
- Other backbones (vggt_x, mapanything) until vggt_spark parity confirmed.
- LC similarity-gate parity (`cross_frame_attention_ratio` vs `image_match_ratio`) — deferred to phase 4.
- Retrieval-stage (SALAD vs DinoSalad) parity.

---

## Success Criteria

- Every stage at d50/d30/d20 matches within tolerance (forward bf16 tol, geometry `1e-4`).
- d10 boundary scale, H_w, PGO poses match; overlap-dedup discrepancy explained and reconciled.
- `diff_report.json` shows no flagged divergence at d50–d10, OR each remaining divergence has a documented, justified root cause in the findings doc.
