# Results: LC parity probe — chess d5, cross-model

**Date:** 2026-07-09 · **Branch:** feat/lc-parity-validation @ 8b2141c
**Spec:** `2026-07-08-lc-parity-validation-design.md` · **Plan:** `../plans/2026-07-08-lc-parity-validation.md`
**Config:** chess/seq-01, full sequence, `min_disparity=5` (loop-rich probe), `submap_size=16`, `max_loops=1/submap`, `lc_thres=0.95`, `conf=25`, spark keyframes shared across all backbones.

## TL;DR

1. **Reference reproduced.** Upstream VGGT-SLAM at d5: 384 keyframes / 45 submaps / **21 loops** / ATE 0.0455 m — matches the 2026-05-31 benchmark exactly.
2. **Retrieval + verify are at upstream parity on every backbone.** All three backbones surface exactly **21 loop candidates** — the same count upstream closes.
3. **Loop application is the sole failure locus**, in the two modes predicted by the 2026-07-08 code audit, now confirmed live with per-candidate logs:
   - **spark, mapanything — no-op:** all 21 accepted candidates dropped at `wrappers.py:284` "no joint poses" (`_verify_loop_candidate` returns `poses=None` by contract; caller discards instead of using submap poses). `lc_decisions_lc.json`: 21 × `reject_reason: "no_joint_poses"`.
   - **vggt_omega — catastrophic:** verify returns poses → 20/21 edges applied → ATE 0.0300 → **0.6255 m** (21×). Inverted + unscaled loop edges (audit defects a/b) wreck the SL(4) graph.
4. **Baselines:** omega best (0.0300), spark 0.0442 (PASS vs SLAM, Δ1.4 mm), mapanything weak (0.1927). Windowed-no-LC (spark 0.0442) again beats SLAM-with-LC (0.0455).

## Gate table

| scene | kf | submaps | SLAM ATE | SLAM loops | ours base | ours lc | ours loops | ΔATE | max Δt | base gate | lc gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7s_chess [vggt_spark] | 384 | 45 | 0.0455 | 21 | 0.0442 | 0.0442 | 0 | 0.0014 | 0.0391 | **PASS** | **FAIL** (0≠21 loops) |
| 7s_chess [vggt_omega] | 384 | 45 | 0.0455 | 21 | **0.0300** | 0.6255 | 20 | 0.5800 | 1.9234 | — | **HARMFUL** |
| 7s_chess [mapanything] | 384 | 45 | 0.0455 | 21 | 0.1927 | 0.1927 | 0 | 0.1472 | 1.9893 | — | HARMLESS (no-op) |

Artifacts: `evals/baselines/lc_parity_d5/7s_chess/{slam,ours_<backbone>}/` (metrics.json + lc_decisions committed; TUM/plots/npz local). Paper-config smoke (d50, 0 loops): `evals/baselines/lc_parity/`, gates PASS.

## Conclusions

- The parity harness works end-to-end and pins the LC failure to loop-edge application with per-candidate evidence. Nothing upstream of it (keyframing, retrieval, verify, baselines, pose-graph plumbing) diverges materially.
- **Fix order (Phase D):** (1) `wrappers.py` poses-None drop — use submap poses per the spark creator's documented contract; unblocks spark + mapanything. (2) Loop-edge construction — mirror upstream `add_edge` chain (scale-reconciled anchors, correct direction); unblocks omega and makes applied edges correct everywhere. (3) Re-run this exact probe; spark lc gate must flip to PASS (21/21 loops, ATE ≈ 0.0455 or better), omega lc must beat or match 0.0300 baseline (HARMLESS at worst).
- Only after the probe is green: paper-config multi-scene matrix (7-Scenes + TUM, data already downloaded) and the scaling sweeps.

## Reproduce

```bash
PY=/opt/venv/reconstruction/bin/python
$PY evals/runners/run_lc_parity.py --scenes 7s_chess --min_disparity 5 \
    --out_root evals/baselines/lc_parity_d5 --data_root evals/data     # tmux, ~3 h
$PY evals/runners/build_parity_table.py --root evals/baselines/lc_parity_d5
```

---

# Post-fix results (Phase D + E, 2026-07-09)

Fixes applied since the pre-fix baseline above: verify contract returns lc_data
with geometry (46953ae); scale-reconciled 3-edge loop chain replacing the
inverted/unscaled direct edge (ee2e4d9, 9bb00a2); parity driver pins subprocess
imports to its own checkout (90186e1 — the first "post-fix" probe silently ran
primary-checkout code); omega/vggtx verify emit world_points+conf from their
existing forwards (9bb9c34, dec9356); mapanything submaps carry depth for the
shared grid path (ea224f3, a0aad8c).

## Gate table (chess d5, `evals/baselines/lc_parity_d5_postfix/_parity_table.md`)

| backbone | base ATE | lc ATE | loops | pre-fix lc | verdict |
|---|---|---|---|---|---|
| vggt_spark | 0.0442 | **0.0421** | 21/21 | no-op (0/21) | PASS — LC improves, beats SLAM ref 0.0455 |
| vggt_omega | 0.0300 | **0.0186** | 20/21 | 0.6255 HARMFUL | best gain (−38%); scale was the missing piece |
| mapanything | 0.1467 | **0.0555** | 21/21 | no-op | biggest ratio (2.6×); baseline itself improved 0.1927→0.1467 (sequential-edge scale now live too) |
| vggtx | 0.0422 | 0.0442 | 4/21 | not measured | HARMLESS; verify gate accepts only 4/21 at layer 20 / ratio 0.85 — threshold calibration follow-up |

Uniformity gate: **zero** "anchor scale 1.0" fallback warnings across all four
arms — every backbone feeds the shared `_lc_anchor_scale` estimator with real
geometry, so LC gains are directly comparable.

## Conclusions

- The pre-fix omega catastrophe decomposed into two independent errors: edge
  direction (fixed by the 3-edge chain: 0.6255→0.1198) and per-forward scale
  normalization (fixed by real anchor scales: 0.1198→0.0186).
- Scale reconciliation matters exactly where a backbone's 2-frame verify scale
  drifts from its submap scale: omega (large drift, LC flipped harmful→best-gain)
  vs spark (near-1 ratios, LC fine either way).
- MapAnything's geometry supply fix improved its no-LC baseline as well —
  sequential inter-submap scale estimation shares the same submap world_points.
- Open follow-ups: vggtx verify-threshold calibration (4/21 accepted);
  per-candidate `lc_decisions` serialization captures retrieval-time state only
  when run against pre-fix code (post-fix artifacts verified correct).

## Reproduce

    /opt/venv/reconstruction/bin/python evals/runners/run_lc_parity.py \
      --scenes 7s_chess --min_disparity 5 \
      --out_root evals/baselines/lc_parity_d5_postfix \
      --data_root /workspace/collab-splats/evals/data \
      --backbones vggt_spark vggt_omega mapanything vggtx
    /opt/venv/reconstruction/bin/python evals/runners/build_parity_table.py \
      --root evals/baselines/lc_parity_d5_postfix

## Threshold recalibration addendum (2026-07-10)

Per-model verify thresholds recalculated with the clean-negative sweep (21
SLAM-confirmed positives vs 20 GT-pose-verified negatives, seed 42) after the
plumbing fix made per-model defaults live (252757d, b0aaf3f):

| backbone | layer | threshold | AUC | note |
|---|---|---|---|---|
| vggt_spark | native head | 0.95 | 0.92 | kept — overlap vs clean negatives documented; stricter drops true loops |
| vggtx | 10 | 1.17 | 1.00 | supersedes layer 20 (AUC 0.42) |
| vggt_omega | **13** | **1.55** | 1.00 | supersedes 16/1.16 (AUC 0.82, rejected 6 true loops) |
| mapanything | 4 (confirmed) | **1.46** | 1.00 | supersedes 1.65 (rejected 9 true loops, LC ATE 2× worse) |

Final d5 table — every arm 21/21 loops, zero anchor-scale fallbacks:
spark 0.0442→0.0421 PASS · omega 0.0300→0.0189 · mapanything 0.1467→0.0555 ·
vggtx 0.0422→0.0417. Commits: b0aaf3f (calibration), a64e0aa (re-runs+table).

## Closing: 4-scene matrix summary (2026-07-10)

Full matrix (`evals/baselines/lc_parity_matrix/_parity_table.md`, commit
7b3bba7) — 7s_chess, 7s_office, 7s_redkitchen, tum_fr3_office × 4 backbones,
main runs + 25%/50% prefix negatives.

**TUM fr3_office (75 kf, 7 submaps — the longest scene; GT-filtered SLAM ref
0.0319, 2 loops).** LC improves every backbone, all with 2/2 loops applied and
`scale:OK` where geometry is live:

| backbone | base ATE | lc ATE | note |
|---|---|---|---|
| vggt_spark | 0.0450 | **0.0320** | matches SLAM (0.0319) to 1e-4; PASS gate |
| vggt_omega | 0.0510 | **0.0377** | |
| mapanything | 0.1296 | **0.0623** | 2.1× reduction |
| vggtx | 0.0427 | **0.0303** | beats SLAM ref |

**7-Scenes matrix rows.** office (58 kf, 6 submaps): LC improves all four
mains (e.g. omega 0.0274→0.0231, vggtx 0.1113→0.1043); redkitchen (43 kf):
improves spark/vggtx, omega ~flat (0.0145→0.0152, HARMLESS), mapanything
HARMFUL (0.0501→0.0740 — see below); chess matrix arm (29 kf, 2 submaps) has
no loop candidates at this sampling — the loop-rich chess evidence is the d5
probe above. All prefix (25%/50%) negative arms: zero false loops, HARMLESS.

**Loop precision/recall (GT-verified, post-hoc from decisions + gt.tum).**
Precision **1.00 everywhere** — no false loop was ever accepted, on any
backbone, any scene, any prefix arm. Recall: 0.22 on chess d5 (dense,
loop-rich), 0.67 on office and fr3_office, 1.0 on redkitchen. The gate is
conservative by design; recall headroom is a tuning axis, not a correctness
problem.

**Redkitchen finding.** The single harmful case (mapanything, 0.0501→0.0740)
is NOT a retrieval failure: the loop is GT-true (P=1.0, R=1.0 on that scene).
It is a correction-quality issue — mapanything's scale estimate blows up on
that pair (`scale:BLOWUP` flag) and the applied Sim3 overcorrects. n=1;
tracked as future correction-quality work, not a gate regression.

## Deferred / future work

- **Dense d5 long-scene arms** — approved but not run (office/redkitchen/fr3
  at d5 sampling; compute-heavy).
- **Per-loop ablation live numbers** — metric committed @ 45da10b; every
  future LC run now emits per-loop delta-ATE attribution, but no dedicated
  ablation campaign has been run yet.
- **Jump-guard ablation** — deprioritized: precision is 1.0 across the whole
  matrix, no false loops observed, so the guard has nothing to reject.
- **KITTI generalization tier** — outdoor/driving loop closure untested.
- **Production-sampler eval arm** — matrix uses parity keyframe lists;
  an arm with the production optical-flow/FPS sampler is still open.
