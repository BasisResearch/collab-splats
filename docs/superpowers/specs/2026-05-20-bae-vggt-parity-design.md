# Spec: Upstream BAE + VGGT Parity Verification

**Status:** Approved (post-brainstorm 2026-05-20)
**Date:** 2026-05-20
**Tags:** ba, vggt, evaluation, parity

> **Read as a self-contained task brief.** The agent executing this spec will not have prior conversation context. All required context is below.

---

## Context

`collab-splats` runs a post-VGGT-X bundle adjustment pipeline built on `pypose/bae` (LM + PCG, `TrackingTensor` + `map_transform` API). Reference for this BA path is `zitongzhan/vggt/demo_colmap.py --use_ba --implementation bae`, which uses **vanilla VGGT** plus the same `bae` solver.

Goal: verify our BA wiring replicates upstream's BAE solver output on matched inputs. Two upstreams are gold standards:

- `https://github.com/pypose/bae` (branch `release`) — the LM/PCG/TrustRegion solver itself.
- `https://github.com/zitongzhan/vggt/blob/main/demo_colmap.py` — the canonical caller (`prepare_bae` + `ReprojNonBatched`).

Prior A/B (`worklog/notes/2026-05-08-co3dv2-eval-gap-investigation.md`) identified preprocessing as the dominant lever: `load_and_preprocess_images_square` (upstream) vs `load_and_preprocess_images_ratio` (our default). This spec freezes preprocessing to square on both sides to isolate the BA solver.

**Open question this spec resolves:** given identical preprocessing + identical seed, does our `collab-splats` BA produce the same refined poses as `zitongzhan/vggt/demo_colmap.py --use_ba --implementation bae`?

---

## Upstream gold-standard knobs (verified against `demo_colmap.py` source)

| Knob | Upstream default | Use for parity |
|---|---|---|
| `--use_ba` | False (must be set) | **required** |
| `--implementation` | `pycolmap` | `bae` |
| `--max_reproj_error` | 8.0 | 8.0 |
| `--vis_thresh` | 0.2 | 0.2 |
| `--query_frame_num` | 8 | 8 |
| `--max_query_pts` | 4096 | 4096 |
| `--fine_tracking` | True | True |
| `--camera_type` | `SIMPLE_PINHOLE` | `SIMPLE_PINHOLE` |
| `--shared_camera` | False | False |
| `--seed` | 42 | 42 |
| Preprocessing | `load_and_preprocess_images_square(paths, 518)` | square, 518 |
| LM strategy | `TrustRegion(up=2.0, down=0.5**4)` | identical |
| LM optimizer | `LM(model, strategy, solver=PCG(), reject=10)` | identical |
| Scheduler | `StopOnPlateau(steps=40, patience=3, decreasing=1e-3)` | identical |
| Behind-cam filter | `proj2d[proj_cam[:,-1]<=0] = 1e6` then `< max_reproj_error` | identical (committed this session) |
| cudnn.deterministic | False | **True (controlled divergence — see Determinism)** |

---

## Goal

Run upstream `demo_colmap.py --use_ba --implementation bae` and our `collab-splats` BA pipeline on the same scene with the same preprocessing + seed. Report:

1. Per-frame pose delta (translation L2, rotation angle) between upstream and ours.
2. ATE/RPE/AUC@30 for each pipeline vs ground truth.
3. Whether the deltas come from (a) model init (VGGT vs VGGT-X), (b) tracks, or (c) BA solver itself.

---

## Three pipelines

- **Pipeline A — gold standard.** Subprocess upstream `demo_colmap.py --use_ba --implementation bae` on cloned `zitongzhan/vggt`. Vanilla VGGT + upstream BAE call. Output: pycolmap `Reconstruction` written to `sparse/0/`.
- **Pipeline B — ours as-is.** Existing collab-splats VGGT-X + BA path. Square preprocessing forked into the driver (no source-code patch). Diagnoses end-to-end drift, including the VGGT-X vs VGGT init gap.
- **Pipeline C — diagnostic.** Vanilla `VGGT.from_pretrained("facebook/VGGT-1B")` instantiated directly in the driver. Raw outputs fed into our `extract_tracks_vggsfm` + `run_bundle_adjustment`. Bypasses `VGGTXCreator` entirely. **This is the BA-solver-only comparison.**

If Pipeline C agrees with Pipeline A within the thresholds in §Verdict, the BA solver is at parity. The B-vs-A gap is then attributable to VGGT-X vs vanilla VGGT model init, not BA wiring.

---

## Success criteria

- All three pipelines run end-to-end on at least one CO3Dv2 sequence and one 7scenes sequence at 50 frames.
- Pose-delta distributions saved per scene.
- Decomposition (Pipeline C) executed.
- Final report at `evals/results/_bae_parity/report.md` includes verdict against tiered thresholds below.

---

## Verdict thresholds

**C-vs-A** (BA-solver diagnostic, raw per-frame deltas; also report SE(3)-Umeyama-aligned deltas for diagnostic, but verdict is on raw):

| Tier | t_err median | t_err p95 | r_err median | r_err p95 | Metric agreement (AUC@30, rel) | Verdict |
|---|---|---|---|---|---|---|
| Parity | <1e-2 m | <5e-2 m | <0.5° | <2° | within 5% | BA solver at parity |
| Drift | 1e-2–1e-1 m | 5e-2–5e-1 m | 0.5–5° | 2–10° | — | minor numeric drift, investigate |
| Bug | >1e-1 m | >5e-1 m | >5° | >10° | — | real bug in BA wiring |

**Sanity floor:** C-vs-A median < 1e-4 m AND < 1e-3° → halt, flag as output-mixup (e.g., wrong file read, cached numpy). Re-run with explicit per-pipeline output dirs.

**B-vs-A:** SE(3)-Umeyama-aligned deltas; report only, no pass/fail. Expected: B-vs-A worse than C-vs-A by the VGGT-X vs vanilla VGGT init gap.

**vs GT** (A, B, C): SE(3)-Umeyama-aligned per existing `loop_closure.eval`. Reported as a row per pipeline.

---

## Required environment

`/opt/conda/envs/nerfstudio/bin/python` is the project env (Python 3.10, torch, pypose, bae, vggt installed). Use it. Do not create a new env.

Verify upfront:
```bash
/opt/conda/envs/nerfstudio/bin/python -c "import pypose, bae, vggt, pycolmap; print('ok')"
```

If `pycolmap` missing, install via `/opt/conda/envs/nerfstudio/bin/pip install pycolmap`. Pin to upstream's expected version if conflicts surface.

---

## Setup steps

### 1. Clone upstream reference

```bash
mkdir -p /workspace/_parity
cd /workspace/_parity
git clone https://github.com/zitongzhan/vggt.git zitongzhan-vggt
cd zitongzhan-vggt
git log -1 --oneline   # record commit SHA in the final report
```

Do **not** pip-install zitongzhan-vggt — its `vggt` package would shadow the conda-env `vggt` (which is what our codebase imports). Instead, run `demo_colmap.py` from the cloned directory using `PYTHONPATH=/workspace/_parity/zitongzhan-vggt` so the script picks up its own `vggt` for the demo and our codebase keeps using the conda-env one.

### 2. Stage matched inputs

For each test scene, produce a directory with frames named `000000.png, 000001.png, ...` (zero-padded). Use symlinks.

**Test scenes (50 frames each, single-pass):**
- **co3dv2-apple-seq1**: `/workspace/collab-splats/evals/data/co3dv2/apple/110_13051_23361/images/frame*.jpg` → first 50.
- **7scenes-chess**: `/workspace/collab-splats/evals/data/7scenes/chess/chess/seq-01/*.color.png` → first 50.

If `co3dv2/apple/110_13051_23361` is missing, fall back to `co3dv2/apple/540_79043_153212` and note in the report.

GT poses: load via `evals/datasets.py::get_dataset("co3dv2"|"7scenes")` (same pattern as `evals/eval_gt.py:36-37`).

### 3. Matched preprocessing — fork in driver

Pipeline A: upstream `demo_colmap.py` already calls `load_and_preprocess_images_square` internally.

Pipelines B and C: driver calls `vggt.utils.load_fn.load_and_preprocess_images_square(paths, 518)` directly → `(images, original_coords)` tuple (same return shape as `_ratio`, verified against upstream source). Driver builds raw_outputs dict by:

1. Loading the model (VGGT-X for B; vanilla `VGGT.from_pretrained("facebook/VGGT-1B")` for C).
2. Running inference on the square-preprocessed tensor (518×518).
3. Skipping the `VGGTXCreator._postprocess` intrinsic-rescale path (which expects ratio's `original_coords`). All downstream operations stay in 518×518 model space, matching upstream.

**Do not monkey-patch `vggt.utils.load_fn.load_and_preprocess_images_ratio`** — superseded by this fork. Do not add a `use_square` flag to `VGGTXCreator` — keep source clean.

### 4. Determinism

Driver sets, **before any model load or inference**, in every pipeline:

```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: upstream `demo_colmap.py` sets `cudnn.deterministic = False`. The driver intentionally diverges to maximize per-run reproducibility for the parity test. Documented in the report. The remaining nondeterminism (e.g., atomicAdds in CUDA reductions) means track-set comparison drops byte-identity in favor of per-frame Jaccard overlap on quantized pixel coords (round to 1px).

---

## Run plan

### Pipeline A — upstream reference (vanilla VGGT + BAE)

Subprocess invocation from driver:

```python
subprocess.run(
    [sys.executable, str(_PARITY_VGGT_DIR / "demo_colmap.py"),
     "--scene_dir", str(scene_dir),
     "--use_ba",
     "--implementation", "bae",
     "--seed", "42",
     "--query_frame_num", "8",
     "--max_query_pts", "4096",
     "--camera_type", "SIMPLE_PINHOLE",
     "--max_reproj_error", "8.0",
     "--vis_thresh", "0.2",
     "--fine_tracking"],
    env={**os.environ, "PYTHONPATH": str(_PARITY_VGGT_DIR)},
    check=True,
    timeout=600,
)
```

After completion, `pycolmap.Reconstruction(scene_dir / "sparse/0")` → sort images by name → build `(N, 3, 4)` world-to-cam extrinsics → promote to `(N, 4, 4)`. Capture `Reconstruction.summary()` (num 3d points, mean reproj error, num inliers) for the report.

### Pipeline B — ours, VGGT-X init

In-process. Driver:

1. `creator = VGGTXCreator(image_paths=staged_paths, ...)` — load model, hold reference.
2. Bypass `_preprocess`; feed pre-computed square-preprocessed images directly into the model.
3. Capture raw_outputs (`images, extrinsic, intrinsics, depth, depth_conf`) in 518×518 space.
4. `extract_tracks_vggsfm(images, depth_conf, world_points, max_query_pts=4096, query_frame_num=8, fine_tracking=True)`.
5. `run_bundle_adjustment(pts3d_kp, extrinsic, intrinsic, tracks, vis>=0.2, image_size=(518,518), max_reproj_error=8.0, lm_steps=40, shared_camera=False)`.
6. Capture LM iteration count, final loss, runtime, peak GPU memory.

### Pipeline C — diagnostic, vanilla VGGT init

In-process. Driver:

1. `model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()` — **no `VGGTXCreator`**. No VGGT-X patches loaded.
2. Same inference call shape as Pipeline B (model forward on square-preprocessed tensor).
3. Build raw_outputs dict with same keys as B (`images, extrinsic, intrinsics, depth, depth_conf`). Verify `image_match_ratio` is NOT in the dict (sanity check — that key is VGGT-X-specific).
4. Same `extract_tracks_vggsfm` + `run_bundle_adjustment` call as B.
5. Capture LM trace identically to B.

If `from_pretrained("facebook/VGGT-1B")` fails to download, run `huggingface-cli download facebook/VGGT-1B` first.

---

## Comparison protocol

For each scene, driver computes:

1. **Pose delta C vs A — raw (no alignment)** — per-frame `‖t_C − t_A‖` and `angle(R_C, R_A)`. Report median + p95. **This is the parity gate.**

2. **Pose delta C vs A — SE(3)-Umeyama-aligned** — diagnostic. If raw delta is large but aligned delta is small → frame-convention bug (e.g., one pipeline anchors to frame 0, the other doesn't). Reuse `collab_splats.pointcloud.loop_closure.eval.umeyama_align`.

3. **Pose delta B vs A — SE(3)-Umeyama-aligned** — report only. No gate (VGGT-X vs VGGT model gap dominates).

4. **Metrics vs GT** (A, B, C): `ate_translation`, `rpe`, `auc_at_threshold` from `collab_splats.pointcloud.loop_closure.eval`. Side-by-side table.

5. **BA convergence trace** (B, C): LM iter count, final loss, runtime. Reference A's `Reconstruction.summary()`.

6. **Track-set overlap** (A vs B, A vs C): per-frame Jaccard on quantized pixel coords (round to 1px). Byte-identity not required.

---

## Deliverable

All artifacts written to `evals/results/_bae_parity/` (leading underscore = scoped separately from general eval results).

```
evals/results/_bae_parity/
  report.md                          # narrative + tables + verdict per scene
  <scene>/extrinsics_A.npy           # (N,4,4) float64 world-to-cam
  <scene>/extrinsics_B.npy
  <scene>/extrinsics_C.npy
  <scene>/intrinsics_A.npy           # (N,3,3)
  <scene>/intrinsics_B.npy
  <scene>/intrinsics_C.npy
  <scene>/tracks_A.npy + tracks_B.npy + tracks_C.npy
  <scene>/vis_A.npy + vis_B.npy + vis_C.npy
  <scene>/ba_trace_B.json            # {lm_iters, final_loss, runtime_s, peak_gb}
  <scene>/ba_trace_C.json
  <scene>/colmap_summary_A.json
  <scene>/tracks_overlap.json
  <scene>/pose_delta_B_vs_A.png
  <scene>/pose_delta_C_vs_A.png
```

`report.md` includes:

- Upstream `zitongzhan-vggt` commit SHA + the exact `demo_colmap.py` command line used.
- `pypose/bae` commit SHA (from `import bae; bae.__version__` or git in the installed package dir).
- Table per scene: A/B/C × {ATE, RPE_t, RPE_rot, AUC@30}.
- Table per scene: C-vs-A raw + aligned pose-delta median + p95.
- Table per scene: B-vs-A aligned pose-delta median + p95.
- Per-stage timing (preproc / inference / track predict / BA).
- Verdict per scene against tiered thresholds.
- Caveats: track-overlap Jaccard (not byte-identity), cudnn.deterministic divergence from upstream, any pipeline that failed end-to-end.

---

## Isolation guarantees

Per explicit user requirement: this work stays separate from general evaluations.

- Driver path: `evals/_bae_parity_driver.py` (leading underscore, matches `_ba_finding_eval.py` convention).
- Output dir: `evals/results/_bae_parity/` (leading underscore — distinct from `ground-truth-evals`, `ba_track-density-*`, etc.).
- No edits to `collab_splats/` source code. No edits to `evals/eval_gt.py`, `evals/runners/`, ground-truth notebooks.
- Upstream clone lives at `/workspace/_parity/zitongzhan-vggt` — outside the repo tree.
- If driver grows past ~400 lines, factor into `evals/_bae_parity/{pipeline_a.py, pipeline_b.py, pipeline_c.py, compare.py}` — single-purpose, not imported by anything else.
- Commit the driver itself (small, reproducible). Gitignore `evals/results/_bae_parity/` artifacts (large, regenerable).

---

## Critical files (existing — read before writing new code)

- `collab_splats/pointcloud/bundle_adjustment.py` — `extract_tracks_vggsfm`, `run_bundle_adjustment`. Already matches upstream BAE path structurally; behind-camera filter committed this session.
- `collab_splats/pointcloud/feedforward/vggtx.py` — `VGGTXCreator` + `unproject_and_filter_points`. Pipeline B holds a reference but skips its `_postprocess`.
- `evals/_ba_finding_eval.py` — existing A/B driver. **Do not extend.** This parity driver is separate.
- `evals/datasets.py` — `get_dataset("co3dv2"|"7scenes")` loaders. Use as-is.
- `collab_splats/pointcloud/loop_closure/eval.py` — `umeyama_align`, `ate_translation`, `rpe`, `auc_at_threshold`. Reuse.
- `worklog/notes/2026-05-08-co3dv2-eval-gap-investigation.md` — prior preprocessing findings. Cite in report.
- `worklog/history/specs/2026-05-06-ba-bae-bundle-adjustment-design.md` — original BA design rationale.

---

## Error handling + fallbacks

| Failure | Behavior |
|---|---|
| `pycolmap` missing | `pip install pycolmap`; pin if conflict |
| Upstream clone fails | Hard fail with instructions; no auto-clone |
| co3dv2 seq1 path missing | Try `110_13051_23361` → `540_79043_153212` → fail with explicit message |
| Pipeline A subprocess timeout (10 min) | Log + skip scene, continue |
| Pipeline B/C `OutOfMemoryError` | Mark scene failed, continue |
| Vanilla VGGT checkpoint missing | `huggingface-cli download facebook/VGGT-1B`; or let `from_pretrained` download |
| Sanity floor tripped (C-vs-A < 1e-4 m) | Halt, flag output-mixup, re-run with explicit dirs |

---

## Testing

- Smoke test: `--frames 10 --scenes 7scenes:chess` end-to-end. Verify all three pipelines produce non-empty extrinsics + non-NaN metrics. ~5 min wall time.
- No unit tests — this spec IS the verification deliverable, not a library feature.

---

## Out of scope

- Implementing square preprocessing as a default (separate spec).
- Multi-scene scaling beyond co3dv2-seq1 + 7scenes-chess (separate spec).
- Loop closure / SL(4) considerations (see roadmap `Future Considerations`).
- Pycolmap-path BA (`--implementation pycolmap`) — only `bae` path in scope.
- Any change to general evaluation entry points or notebooks.

---

## Sunset

After verdict reached + report committed, driver remains in repo as a regression check. If gold-standard upstream changes (new commit SHA), re-run with new SHA. Otherwise treat as frozen.
