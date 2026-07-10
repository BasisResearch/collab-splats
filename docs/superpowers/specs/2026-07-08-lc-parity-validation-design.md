# LC Parity Validation Harness — Design

**Date:** 2026-07-08 · **Status:** approved-pending-review
**Goal:** Validate our loop-closure pipeline against upstream VGGT-SLAM 2.0 to establish result parity **before** changing or extending anything.

## Context

Our LC pipeline (`collab_splats/pointcloud/loop_closure/` + `wrappers.py`) reimplements VGGT-SLAM 2.0 (vendored at `third_party/VGGT-SLAM`, includes the `vggt-slam2` update; one local ablation commit `604efe8` adds the `VGGT_SLAM_SCALE_SE3` env flag). The 2026-05-31 cross-model benchmark showed LC never helps on chess/seq-01: no-op for `vggt_spark`/`mapanything`, catastrophic (17–23× ATE blow-up) for `vggtx`/`vggt_omega`.

A 2026-07-08 divergence audit (code-confirmed, both sides read end-to-end) identified root causes:

1. **No-op cause (spark, mapanything) — confirmed in code.** `wrappers.py:283-287` hard-drops any verified loop whose `_verify_loop_candidate` returns `poses=None`. Spark's native-verify override (`vggt_spark_creator.py:181`, uses `compute_similarity=True` / `image_match_ratio` — correct upstream parity) and MapAnything's verify both return `poses=None` by contract ("caller uses submap poses") — so every accepted loop is discarded at the "no joint poses" check.
2. **Catastrophic cause (vggtx, omega) — confirmed statically + numerically.** Loop edge `closure.py:559-563` is `inv(lc.poses[0]) @ lc.poses[1]`:
   - **Inverted.** gtsam `BetweenFactorSL4` numerical test (2026-07-08): with `lc.poses[0] ≈ I` (guaranteed by `assert_world_to_cam`), the correct constraint in the graph's own `H_inner = P_{i-1} @ inv(P_i)` convention has factor error 0.0; ours has error ~178. Our edge is the exact inverse.
   - **Unscaled.** Upstream routes every loop through `add_edge` (`solver.py:118-169`) which estimates per-pixel scale between identical-image pointclouds and composes `H_scale` into the relative homography. Our loop edge has no scale reconciliation (our sequential edges do).
   - **Topology.** Upstream inserts a dedicated 2-frame LC submap as graph nodes with 3 constraints (query→LC0 scaled, LC0→LC1 inner, LC1→detected scaled). Ours adds 1 direct query↔detected edge.
   - **Count.** Upstream `max_loops=1` **per submap** (verified: `main.py:113` → `solver.py:347` queue cap per `run_predictions` call; `detected_loops[0]` hardcoded). Ours `max_loops_per_submap=5`.

Confirmed-parity components (no action needed): retrieval (DINO-SALAD L2 argmin, `<0.95` accept, both sides), solver backend (gtsam SL(4), `BetweenFactorSL4`, LM), spark native verify gate, sequential-edge scale estimation, windowed baseline (matches VGGT-SLAM baseline ATE on chess).

**This spec covers the validation harness only.** Fixes are applied after the harness empirically pins the failures (Phase D), and are validated by re-running the same harness.

## Design

Three levels, each gating the next. Backbone: `vggt_spark` only — same weights as upstream, the only apples-to-apples configuration. Other backbones come after spark parity is green.

### Level 0 — Reference matrix (upstream, untouched)

Run vendored VGGT-SLAM via `evals/runners/run_vggt_slam_lc.py` (exists) with paper config (`submap_size=16`, `min_disparity=50`, `conf_threshold=25`, `lc_thres=0.95`, `max_loops=1`) on:

- **7-Scenes** (seq-01 each): chess, fire, heads, office, pumpkin, redkitchen, stairs — loader `_load_7scenes` exists; only chess downloaded.
- **TUM RGB-D**: fr1/desk, fr1/room, fr2/xyz, fr3/long_office_household — loader `_load_tum` exists; upstream ships `evals/eval_tum.sh` as config reference.

**Full sequences, no frame cap.** Every scene runs over all frames, with keyframes selected by upstream's optical-flow disparity tracker (`frame_overlap.py`, `min_disparity=50`) — i.e. the natural upstream configuration, not the truncated framesets (26/200/384 frames) used in prior benchmarks. This is the scale-stress condition: loop count and submap count grow with sequence length, and LC must not degrade as they do.

Per scene, persist: TUM trajectory file, ATE (Sim3-aligned RMSE), **closed-loop count** (`solver.graph.get_num_loops()`), **keyframe list**, and a **3D trajectory plot vs ground truth**. Output under `evals/baselines/lc_parity/<scene>/slam/`.

### Level 1 — End-to-end parity (ours, spark backbone) + cross-model sweep

Run our pipeline (`LoopClosure` wrapper, spark creator) on the **exact keyframe list from the Level-0 run** (eliminates keyframe selection as a confounder) with matched config: `submap_size=16`, `submap_overlap=1`, `max_loops_per_submap=1`, `conf_threshold=25`, `lc_retrieval_threshold=0.95`. Two conditions per scene: baseline (LC off) and LC on.

**Cross-model sweep (added 2026-07-09, user request).** In addition to `vggt_spark`, run the same Level-1 protocol (same keyframes, same config, baseline + lc) for `vggt_omega` and `mapanything` (`vggtx` available behind the same `--backbones` flag). Outputs land under `<scene>/ours_<backbone>/`. Gate semantics differ by backbone:

- `vggt_spark` — full parity gates (ATE ≤ 5%/5 mm vs SLAM, loop counts equal). Same weights as upstream; any gap = pipeline divergence.
- `vggt_omega` / `mapanything` — no upstream reference exists for these weights, so ATE-vs-SLAM and loop counts are **reported, not gated**. Gated instead: **LC-harmless** — `ate_lc ≤ max(1.05 × ate_baseline, ate_baseline + 0.005)` (LC must never make a backbone worse than its own baseline), plus the scaling gate on sweep scenes. This is the cross-model comparison goal: which backbone has the best baseline, and whether LC helps, no-ops, or harms each.

Per scene, compare against Level 0:

| Metric | Gate |
|---|---|
| ATE delta | \|Δ\| ≤ 5% relative or ≤ 5 mm absolute |
| Closed-loop count | equal |
| Max per-frame SE3 deviation | report (no hard gate initially) |
| Trajectory overlay (ours + SLAM + GT, 3D) | visual artifact per scene |

Baseline condition must pass everywhere (already passes on chess). LC-on is expected to **fail** pre-fix — the harness quantifies the failure; that is its purpose.

**Scaling condition (frame-count stress).** Because ours consumes the Level-0 keyframe list, optical-flow differences between subsequent frames are already accounted for — both pipelines see identical frames at every length. On the two longest sequences per dataset (7-Scenes: office, redkitchen; TUM: fr1/room, fr3/long_office_household), additionally run both pipelines at progressive keyframe prefixes — 25%, 50%, 100% of the full keyframe list — and record ATE, loop count, and submap count at each point. Gates:

| Metric | Gate |
|---|---|
| ATE vs frame count (LC on) | bounded — no blow-up as loops accumulate; ATE(100%) ≤ ~1.5× ATE(25%) after Sim3 alignment, and never exceeds the baseline-condition ATE by more than the Level-1 gate |
| ATE delta vs SLAM at each prefix | same ≤ 5% / 5 mm gate |
| Loop count at each prefix | equal to SLAM at same prefix |

This directly tests the user-facing failure mode: LC quality must hold as more and more frames (and therefore more loop closures) are added, not just on short clips.

### Level 2 — Stage-trace on divergent scenes

For each Level-1 failure, dump per-stage artifacts from both pipelines to JSON and diff. Extend existing tooling (`parity_trace.py`, `our_solver_dump.py`, `vggt_slam_solver_dump.py`, `diagnose_lc_parity.py`) rather than writing new runners.

1. **Retrieval:** candidate set (query frame, detected submap/frame, L2 score) → set difference.
2. **Verify:** `image_match_ratio` per candidate → should be bit-close (both spark-native). Also log every gate decision (accept / reject-ratio / reject-jump / **drop-no-poses**) — the benchmark runs persisted no console logs, so the no-op path is code-confirmed but not yet log-confirmed.
3. **Loop edges:** the constraint fed to the graph. Compare ours against upstream's *composed* query→detected chain (`H_relA @ H_innerLC @ H_relB`) — directly quantifies the inversion + missing-scale defects.
4. **Graph:** factor dump (count, types, noise sigmas, initial values), initial/final total error, optimized per-node homographies.
5. **Final poses:** per-frame SE3 divergence after extraction.

First stage whose artifacts diverge = pinned failure. Expected pre-fix result: retrieval ✓, verify ✓, loop-edge ✗.

### Phases

- **A.** Download missing scenes (7-Scenes ×6, TUM ×4; store via collab-data rclone). Run Level 0 serially in tmux (46.6 GB cgroup cap — no parallel heavy runs). Extend `eval_gt.py` plotting with the 3D trajectory-vs-GT overlay if absent.
- **B.** Level 1 sweep → per-scene pass/fail table + trajectory overlays.
- **C.** Level 2 on failures → per-stage divergence report.
- **D.** (Out of scope for this spec) Apply the identified fixes; re-run harness to green; then extend to vggtx / vggt_omega / mapanything.

## Error handling

- A scene where upstream itself crashes or closes 0 loops is recorded as-is in the reference matrix (still useful for baseline parity) and excluded from LC-gate comparison.
- Keyframe-list transfer must be exact (path list, order); harness asserts frame counts match before running.
- All runs emit structured logs (loop decisions per gate) so no-op vs reject is always distinguishable post-hoc.

## Testing

- Unit: keyframe-list round-trip (SLAM run → file → our pipeline consumes identical list); gate-decision logging; loop-edge composition helper (upstream chain composition) against a hand-built 2-submap fixture.
- Integration: chess/seq-01 end-to-end through Levels 0–2 (data already local) before scaling to the full matrix; the chess run also exercises the prefix mechanism (25/50/100%) once before the long-sequence sweeps.

## Non-goals

- No changes to `closure.py` / `wrappers.py` LC logic in Phases A–C.
- No non-spark backbones until spark parity is green.
- No CO3Dv2 / KITTI / Waymo — validated indoor SLAM scenes only (7-Scenes, TUM).

## References

- Divergence audit: this spec's Context section (2026-07-08 session).
- Prior benchmark: `docs/superpowers/specs/2026-05-31-cross-model-benchmark-results.md`.
- Numerical convention test: gtsam `BetweenFactorSL4` experiment (scratchpad `sl4_convention_test.py`, reproduced in Context).
- Upstream: `third_party/VGGT-SLAM` @ `604efe8` (MIT-SPARK/VGGT-SLAM + vggt-slam2).
