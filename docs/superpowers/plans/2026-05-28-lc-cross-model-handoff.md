# LC Cross-Model Parity — Handoff Prompt for Next Agent

**Date:** 2026-05-28  
**Branch:** `refactor/cu121`

---

## Orientation

Read in order:
1. `worklog/STATE.md`
2. `docs/superpowers/plans/2026-05-28-parity-run3-results.md` (full ATE table + open issues)
3. `worklog/WORKLOG.md` top entry (2026-05-28 session)

---

## What Was Established This Session

### Architecture

LC pipeline is a two-gate system **identical to VGGT-SLAM (VGGT-SPARK)** — confirmed by source audit:

1. **DINO-SALAD retrieval gate** — `lc_retrieval_threshold=0.95` (L2 distance on unit-norm embeddings; aligned with VGGT-SLAM `main.py` default this session)
2. **Attention verify gate** — `verify_match_ratio=0.85` (`cross_frame_attention_ratio`, `mean_top_quarter` aggregation, exact port of VGGT-SPARK `get_similarity`)

Pose graph: SL4 (`manifold="sl4"` in `LoopClosureConfig`), same algorithm as VGGT-SLAM's solver.

### What Works: VGGT-X + SL4 LC ✓

chess_seq01, 200 frames, `submap_size=16`: ATE **0.3184m → 0.2664m (−16.3%)**. Proven. Layer=20 is the calibrated default matching VGGT-SPARK (`global_blocks`, depth=24). Layer=12 peaks on calibration score (1.426 mtq) but produces identical ATE — keep layer=20.

### Calibrated Layer Indices (committed to source)

| Model | Block path | Depth | `_lc_layer_index` | mtq (DINO-SALAD retrieved pairs) |
|---|---|---|---|---|
| VGGT-X | `aggregator.global_blocks` | 24 | **20** | 0.817 |
| VGGT-Omega | `aggregator.inter_frame_blocks` | 24 | **16** | 1.328 |
| MapAnything | `info_sharing.self_attention_blocks` | 16 | **4** | 1.807 |

Calibration script: `evals/eval_similarity_calibration.py --mode retrieved --layer_index N`

---

## Open Issues

### Issue 1: Omega LC regression (priority: high)

LC **worsens** Omega ATE: 0.3225m → 0.496m (+54%), regardless of layer index. Layer=16 (calibration peak 1.328) does not fix it.

**Best hypothesis:** Coordinate convention mismatch. Omega's raw extrinsics may be cam-to-world where VGGT-X outputs world-to-cam. `assert_world_to_cam` in `_run_lc_loop` (`wrappers.py`) may pass silently on wrong-convention matrices, inserting inverted loop edges into the SL4 pose graph.

**Debugging path:**
1. Print `poses_4x4` for first Omega submap — verify sign convention matches VGGT-X
2. If inverted: add `_invert_extrinsics_for_lc: ClassVar[bool]` to `VGGTOmegaCreator`, invert in collation
3. Alternatively: log LC closure pairs and compare raw Omega vs VGGT-X poses for same scene
4. Check `estimate_scale_pairwise` in `run_pose_graph_optimization` — Omega scale may diverge from VGGT-X

**Key files:** `collab_splats/pointcloud/feedforward/vggt_omega.py`, `collab_splats/pointcloud/wrappers.py:_run_lc_loop`, `collab_splats/pointcloud/loop_closure/closure.py:run_pose_graph_optimization`

---

### Issue 2: MapAnything windowed LC (priority: medium)

MapAnything single-pass baseline: **0.1103m** — already excellent, no LC needed for baseline quality. But windowed LC is architecturally broken.

**Root cause:** `MapAnythingCreator._forward(model, views)` ignores the `views` argument entirely; always processes `self._processed_views` (all frames, set during `_preprocess`). The LC loop's window slice is silently discarded.

**Secondary blocker:** `_lc_collate_outputs` (already stubbed in `MapAnythingCreator`) needs `camera_poses` key — this only appears after `postprocess_model_outputs_for_inference`, not in raw `model.forward()` output. Fix requires running minimal postprocess inside `_lc_collate_outputs`.

**Fix path:**
1. Override `_forward` in `MapAnythingCreator` to accept a window of frames:
   - Takes `views` tensor (k, C, H, W) from LC loop window slice
   - Calls `preprocess_input_views_for_inference` on those k frames (not `self._processed_views`)
   - Calls `model.forward()` on the preprocessed window
2. In `_lc_collate_outputs`, run `postprocess_model_outputs_for_inference` on the raw list to extract `camera_poses`, then aggregate extrinsics + intrinsics
3. Store original raw list as `raw_outputs` in Submap (for postprocessing); store aggregated dict as the LC-loop-facing format

**Key files:** `collab_splats/pointcloud/feedforward/mapanything.py` (`_forward`, `_lc_collate_outputs`), `collab_splats/pointcloud/wrappers.py` (`_run_lc_loop`)

---

## Eval Commands

```bash
# After Omega fix — compare baseline vs LC
python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone vggt_omega --submap_size 16 --max_frames 200 --conditions baseline lc

# After MapAnything windowed fix
python evals/eval_gt.py \
  --dataset 7scenes --seq_dir evals/data/7scenes/chess/chess/seq-01 \
  --backbone mapanything --submap_size 16 --max_frames 200 --conditions baseline lc

# Calibration validation (confirm retrieved-mode scores match expectations)
python evals/eval_similarity_calibration.py \
  --scene_dir evals/data/7scenes/chess/chess/seq-01 \
  --mode retrieved --n_pairs 20 --models vggtx mapanything omega
```

---

## Summary of Code State

| File | Key change |
|---|---|
| `loop_closure/closure.py` | `lc_retrieval_threshold=0.95` (L2, replaces `lc_cosine_threshold`); `lc_cosine_threshold` deprecated with migration |
| `feedforward/vggt_omega.py` | `_lc_layer_index=16` (was 20); removed wrong `default_verify_match_ratio=0.59` placeholder |
| `feedforward/mapanything.py` | `_lc_layer_index=4` (was out-of-range 20); `_lc_collate_outputs` stub (needs `camera_poses` fix) |
| `feedforward/base.py` | `_lc_collate_outputs` no-op default; `_lc_layer_index`, `_lc_token_offset` class vars |
| `wrappers.py` | `_run_lc_loop` calls `_lc_collate_outputs` when `_forward` returns list |
| `evals/eval_gt.py` | `mapanything` added to backbone choices |
| `evals/eval_similarity_calibration.py` | `--mode retrieved`, `--layer_index` flags |
