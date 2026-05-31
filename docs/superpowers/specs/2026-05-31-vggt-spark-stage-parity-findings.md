# Findings: vggt_spark ↔ VGGT-SLAM Stage Parity

Running log from `evals/runners/parity_trace.py`. One block per level. Tolerances: forward bf16 ≤1e-2 (observed 0), geometry ≤1e-4, trajectory(raw, pre-align) ≤1e-3.

---

## Harness bugs fixed before parity could run

1. **SLAM import collision** — `import evals.runners...` resolved to the installed `evals` package. Fixed: importlib file-load by path.
2. **SPARK forward crash (real pipeline bug)** — `VGGTSPARKCreator` inherited `VGGTXCreator._forward`, which wraps the model call in `torch.autocast`. SPARK's `VGGT.forward` casts to bf16 and runs its heads under `torch.cuda.amp.autocast(enabled=False)`; the outer autocast leaked fp32 aggregator tokens into SPARK's bf16 `camera_head.token_norm` (which, unlike VGGT-X's camera_head, does not cast its input) → `expected Float but found BFloat16`. Fixed with a `VGGTSPARKCreator._forward` override mirroring `solver.run_predictions` (bf16 in, no outer autocast).
   - **Side note:** eval_gt's `--backbone vggt_spark` likely runs **VGGT-X** code, because `vggt` is already imported/cached when `_load_model` inserts the SPARK path → the insert is shadowed. The harness loads real SPARK (what VGGT-SLAM uses). Existing `vggt_spark` sweep numbers may actually be VGGT-X.

---

## d50 / d30 / d20 — single submap (LC loop skipped, n < submap_size=16)

**Exact parity.** Forward bitwise-identical; trajectory sub-mm.

| level | frames | stage1 | stage2 ext/intr/depth/conf | stage4 traj(raw) |
|---|---|---|---|---|
| d50 | 5 | 0.000e0 | 0.000e0 (all) | 4 PASS 2.58e-4 |
| d30 | 8 | 0.000e0 | 0.000e0 (all) | PASS 4.27e-4 |
| d20 | 12 | 0.000e0 | 0.000e0 (all) | PASS 1.34e-4 |

→ Preprocess + VGGT forward + single-submap pose extraction are **identical** between pipelines. Forward eliminated as a parity variable.

---

## d10 — 2 submaps (first multi-submap case) — **DIVERGES at stage 4**

| stage | result | max\|Δ\| |
|---|---|---|
| 1 preprocess (submap0, 17f) | PASS | 0.000e0 |
| 2 ext/intr/depth/conf (submap0) | PASS | 0.000e0 |
| 4 trajectory(raw) | **DIVERGE** | **0.528 m** (26 frames; slam_dup_frames=1) |

**Forward still bitwise-identical** at d10 → divergence is purely **submap-1 stitching**, not the model.

### Quantified
- ours ATE-vs-GT = **0.308 m**; slam ATE-vs-GT = **0.0176 m** (17×).
- ours-vs-slam Sim3-**aligned** RMSE = **0.070 m**, scale **0.844** → not a pure similarity transform; genuine shape difference.

### Divergence shape
Per-frame |Δpos| grows **smoothly and monotonically from frame 0** (anchor), reaching 0.66 m by frame 197. **No jump at the submap boundary (frame 140).** Both trajectories anchored at origin; ours bends in −Y, SLAM stays ~0 Y.

### Root-cause hypothesis
Our **boundary scale = 0.4658** (`stage6_boundary0_scale`; `H_w` diag ≈ 0.468, `T` diag ≈ [1.023, 1.027, …]). This single submap-1 rescale, applied to the SL(4) PGO inter-submap edge, is inconsistent with the per-frame sequential chain. With only a prior on node 0 and no loop edges (baseline, max_loops=0), the optimizer distributes the inconsistency across **all** nodes → smooth global drift, even within submap-0. Consistent with the historical Sim3 LC scale bug (`project_sim3_lc_bugfix`).

### Overlap-duplicate confirmation
`slam_dup_frames=1` — VGGT-SLAM's TUM emits the boundary frame twice (un-deduped per-submap write); our `dedup_overlap` collapses it. Both reduce to submap-0's estimate. Benign for ATE *iff* the ATE tool dedups (still to verify in compute_ate).

---

> **NOTE — the boundary-scale hypothesis above is DISPROVEN.** SLAM's scale = 0.4628 ≈ ours 0.4658; the graph nodes match to bf16. Real root cause below.

## ROOT CAUSE (confirmed) — pose-extraction convention in `decompose_camera`

Exhaustively eliminated (all match SLAM): forward (Δ=0), preprocess (Δ=0), scale (0.4658 vs 0.4628), `T`, inner `H_inner`, `H_w`, `proj_mats=K`, graph nodes (bottom/projective row identical to **1e-15**, full H to bf16 ~3e-3), PGO optimize (proven **no-op**: pre==post, RMSE 0). Vendored SLAM is faithful to upstream for baseline. Divergence is **purely the graph→pose readout**.

Decoding SLAM's own graph homographies:
- with **SLAM's** `decompose_camera` (default `no_inverse=False`: `t = -R @ inv(K) @ P[:,3]`) → reproduces `slam_TUM` **exactly (max|Δ|=0.00000)**.
- with **our** convention (`graph.py:decompose_camera` returns `t = inv(K) @ P[:,3]` — the `no_inverse=True` branch, missing `-R @` — stored as w2c then `-Rᵀt`) → the bent trajectory (**max|Δ|=0.664**, = our 0.308 ATE).

`R` vs `Rᵀ`. Our `decompose_camera` docstring claims *"Source: VGGT-SLAM slam_utils.py"* but ported the wrong branch. Single-submap matched (d20/d30) because near-identity rotations give `R ≈ Rᵀ`; the 2nd submap + scale make the homographies genuinely projective, so `-R` vs `-Rᵀ` diverges. Transitively: fixing the convention → ours = `slam_TUM` → ATE 0.308 → ~0.0176.

## Fix (applied + validated)

`closure.py` pose extraction now stores `R.T` (correct world-to-cam rotation) so the camera centre is `-R @ t`, matching SLAM. `decompose_camera` has a single caller, so blast radius is contained.

**Validation (chess/seq-01, raw pre-align trajectory vs SLAM):**

| level | submaps | before | after |
|---|---|---|---|
| d50 | 1 | 2.6e-4 | 2.6e-4 |
| d30 | 1 | 4.3e-4 | 4.3e-4 |
| d20 | 1 | 1.3e-4 | 1.3e-4 |
| d10 | 2 | **0.528 m** | **1.6e-3 m** |

d10 residual (1.6mm) is the bf16 floor: SLAM stores homographies in bf16 (1/256 quantized), ours float64; ~1.5mm accumulates over the chain. Expected ATE-vs-GT: 0.308 → ~0.018 (SLAM parity).

## Remaining
- Stage-5 ATE dup-handling check in `compute_ate` (overlap-frame duplicate).
- After fix: lower disparity (d<10), then LC (max_loops>0) similarity gate.
