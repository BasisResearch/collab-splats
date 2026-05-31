# Handoff: cross-model benchmark → reproduce VGGT-SLAM defaults first

**Date:** 2026-05-31 · **Branch:** refactor/cu121
**For:** the next agent. Read this before extending the benchmark.

---

## Why this handoff exists (the confusion)

The cross-model benchmark in this session used **custom sweep parameters**, NOT VGGT-SLAM's default config. So its numbers are internally consistent but **not comparable to VGGT-SLAM's published 7-Scenes results** or to GitHub issue [MIT-SPARK/VGGT-SLAM#43](https://github.com/MIT-SPARK/VGGT-SLAM/issues/43).

| param | VGGT-SLAM `main.py` default | what this session ran |
|---|---|---|
| `min_disparity` | **50** | 5 (d5_long), 10 (d10), 20 |
| `max_frames` | none (full folder) | 200 / 1000 caps |
| `submap_size` | 16 | 16 ✓ |
| `max_loops` | 1 | 1 (lc) / 0 (baseline) ✓ |
| `conf_threshold` | 25.0 | 25.0 ✓ |
| `lc_thres` (retrieval) | 0.95 | 0.95 ✓ |

`min_disparity` is the big divergence. Default **50** keeps few, well-separated keyframes; this session ran **5–20** (much denser → more submaps, more loop edges). Issue #43's chess ATE **0.6968 m** (VGGT-SLAM 2.0, TUM config, full seq, LC on) sits in the **same catastrophic-LC regime** as this session's `lc` results (vggtx 0.86 m, omega 0.58 m on d5_long), **not** the baselines (0.017–0.044 m). LC-with-many-loops blows chess up to ~0.6–0.9 m; no-LC baseline is sub-10cm.

**Takeaway:** before any cross-model / cross-dataset comparison, **reproduce VGGT-SLAM exactly with its own defaults** so there's a trusted, published-matching anchor. Only then vary the backbone.

---

## DIRECTIVE for next agent

1. **Reproduce VGGT-SLAM defaults exactly, with VGGT-SPARK.** Run `third_party/VGGT-SLAM/main.py` with **all defaults** (`min_disparity=50`, `submap_size=16`, `max_loops=1`, `conf_threshold=25`, `lc_thres=0.95`, no `max_frames` cap) on the **full** 7-Scenes chess image folder. Confirm the ATE matches VGGT-SLAM's reported chess number. SPARK is VGGT-SLAM's native backbone, so this is the ground-truth anchor.
   - Watch the same trap this session hit: `eval_gt --backbone vggt_spark` can silently run VGGT-X. The load-guard (committed) now raises if the wrong module loads; `main.py` uses real SPARK directly. Confirm the loaded module path.
2. **Then swap the backbone on identical config.** Run vggtx / vggt_omega / mapanything through the SAME pipeline at the SAME default params (`min_disparity=50`, full folder). That is the fair comparison — same frames, same LC config, only the model changes.
3. **Then extend to all 7-Scenes (and other datasets)** at defaults: fire, heads, office, pumpkin, redkitchen, stairs. Compare each backbone vs the SPARK/SLAM anchor per scene.
4. **Do NOT reuse this session's `slam_d5_long` / `slam_d10` framesets for the headline comparison** — they are off-default. They are still useful for the windowing-cost and LC-failure analysis below, but not for matching published numbers.

---

## What this session built (all committed on refactor/cu121)

Commits `c9142f1 … a2b56e0` (13 total):

**Code / tooling (reusable):**
- `c9142f1` multi-threshold pose AUC (`auc_at_threshold(thresholds=...)`, dynamic `auc_{t}` keys) + fixed 3 stale tests.
- `8d2797f` eval_gt reports RPE-rotation + AUC@{5,15,30} in `metrics.json`.
- `2891810` **load-guard**: `vggt_spark` raises if it loads VGGT-X (was silent).
- `e624e82` **bug fix**: `_VGGT_SPARK_ROOT` was `parents[4]` (`/workspace`) → nonexistent → fell through to site-packages VGGT-X. Now `parents[3]` (`<repo>/third_party/vggt_spark`). **This is the root cause of the documented "eval_gt vggt_spark silently runs VGGT-X" warning.**
- `c4b4ac8` **bug fix**: LC-with-loops crashed (`tensors on cuda:0 and cpu`) — `_verify_loop_candidate` fed CPU frames to the CUDA model. Both verify paths now move to model device.
- `b3a0dd2` `evals/runners/run_cross_model_benchmark.py` — serial eval_gt matrix runner.
- `008c2d2` `evals/runners/build_benchmark_table.py` — aggregates metrics → markdown.
- `b2fe9c1` `--lc_layer` override for layer sweeps.

**Data / artifacts:** `evals/baselines/cross_model/` (lean: metrics.json/ate.json/TUM/table; heavy COLMAP/ply/plots/npz gitignored), `slam_d5_long` long SLAM ref, `_layersweep/`, `_gate/`.

**Docs:** design spec, plan, **results** (`2026-05-31-cross-model-benchmark-results.md`), this handoff.

---

## Findings that still hold (config-independent)

- **Windowing costs nothing** (goal 1): submap windowing ≈ or slightly beats single-pass full-batch inference on every backbone. Submaps are free.
- **LC-with-real-loops is broken** (goal 2): on a 21-loop set, LC is a **no-op** (spark, mapanything apply 0 loop edges) or **catastrophic** (vggtx 0.86 m, omega 0.58 m). Layer sweep {8,12,16,24} all catastrophic → failure is in **loop-edge application / pose-graph correction**, NOT the verify layer. This is the real open bug and it gates any LC-on result.
- **vggt_omega** had the best baseline ATE + by far the best AUC@5 (~67 vs ~46) in this session's config — re-confirm under defaults.

These were measured at off-default `min_disparity`, so treat the *magnitudes* as config-specific, but the *qualitative* conclusions (windowing free; LC-with-loops broken) are mechanism-level and should reproduce.

---

## Open bugs / next work (priority order)

1. **Reproduce VGGT-SLAM defaults with SPARK** (directive #1) — the anchor everything else needs.
2. **LC-with-loops corruption** — `collab_splats/pointcloud/wrappers.py:_run_lc_loop` loop-edge construction + the SL(4)/Sim3 correction. Why do applied loops wreck the graph (vggtx/omega), and why do spark/mapanything register 0 loops? Compare our retrieval/verify gate to SLAM's native `image_match_ratio` on the known loops. LC is unusable until this is fixed.
3. **mapanything long-range degradation** — separate model-quality issue (baseline 0.19 m on the long set, AUC@5 2.8).

## Key files
- `evals/eval_gt.py` — compute (CLI/tmux). Now: `--lc_layer`, AUC@{5,15,30}, RPE-rot in metrics.json.
- `evals/runners/run_cross_model_benchmark.py` / `build_benchmark_table.py` — matrix runner + table.
- `evals/runners/run_vggt_slam_lc.py` — SLAM-LC runner (used for the long ref / loop probe).
- `third_party/VGGT-SLAM/main.py` — **upstream entry point; use its defaults for the anchor.**
- `collab_splats/pointcloud/feedforward/vggt_spark_creator.py` — SPARK loader (fixed root + load-guard).
- `collab_splats/pointcloud/wrappers.py` — LC loop (`_run_lc_loop`); the LC-with-loops bug lives here.
