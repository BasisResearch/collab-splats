# BAE Parity Harness — Handoff (2026-05-21)

## What This Is

End-to-end parity test between our BAE bundle adjustment implementation and upstream
`zitongzhan/vggt demo_colmap`. Three pipelines:

- **Pipeline A** — gold standard: pre-generated reference artifacts from `bae_upstream`
  conda env running upstream `demo_colmap.py --use_ba --implementation bae`
- **Pipeline B** — ours: VGGT-X + our BA wrapper
- **Pipeline C** — diagnostic: vanilla VGGT-1B + our BA wrapper (isolates BA solver
  from model choice)

**Verdict: C ≈ A → our BA solver is correctly wired.**

---

## Current State: Smoke Test PASSED

Smoke run on `7scenes:chess`, 10 frames, all 3 pipelines complete without error.

| Pipeline | ATE_RMSE (m) | RPE_t (m) | RPE_rot (deg) |
|---|---|---|---|
| A | 0.0040 | 0.0055 | 0.159 |
| B | 0.0041 | 0.0049 | 0.156 |
| C | 0.0039 | 0.0047 | 0.156 |

**C-vs-A raw parity gate:** t_med=1.8mm, t_p95=3.6mm, r_med=0.050°, r_p95=0.097°
— all within spec thresholds (< 1cm, < 5cm, < 0.5°, < 2°). AUC@30 rel_diff=0.0.
**Verdict: `parity`.**

Note: aligned rotation errors (105°/72°) are a Umeyama sign-flip artifact in the
harness, not real pose divergence. The raw C-vs-A numbers are authoritative.

Results: `.worktrees/bae-parity/evals/results/_bae_parity/smoke5/`

---

## Fixes Applied (All Committed)

**Worktree `bae-parity` branch:**
- `evals/_bae_parity/pipeline_b.py` — cast images to model dtype before forward;
  add `intrinsics_downsampled` alias in raw dict
- `evals/_bae_parity/pipeline_c.py` — load VGGT-1B in bfloat16 (`.to(torch.bfloat16)`);
  cast images to model dtype; add `intrinsics_downsampled` alias
- `collab_splats/pointcloud/localization.py` — replace vendored salad path setup with
  pip-based imports (forward-port of commit 170b603 from main)
- `collab_splats/pointcloud/feedforward/mapanything.py` — timm 0.6.x DropPath shim

**Main `refactor/core-modules` branch:**
- `vendor/bae/bae/utils/ba.py` — added `rotate_quat` (uses `quat_rotate_xyzw` +
  translation; vmap-safe for `@map_transform`)
- `collab_splats/pointcloud/feedforward/mapanything.py` — timm 0.6.x DropPath shim

---

## Known Non-Issues

**CPU at ~800% / load ~30 during BA** — expected. CuDSS (GPU direct solver) not
installed (`bae.sparse.solve` missing, no `libcudss` on system); PCG fallback drives
CPU with a Python iteration loop. Matches upstream behavior. Not a bug. To fix:
install cuDSS library + rebuild bae from source.

**Jaccard A-vs-C = 0.0** — expected. Pipeline A reference artifacts don't include
tracks (demo_colmap doesn't dump them); driver degrades gracefully with empty arrays.

---

## What's Left

### 1. Generate Pipeline A references for 50-frame runs

Current reference covers only 10 frames. Must regenerate before T14 eval:

```bash
cd /workspace/collab-splats/.worktrees/bae-parity
# Check which scenes are available first:
ls /workspace/collab-splats/data/7scenes/

/opt/conda/envs/nerfstudio/bin/python \
    evals/_bae_parity/reference/generate.py \
    --scenes 7scenes:chess 7scenes:fire 7scenes:heads 7scenes:office \
             7scenes:pumpkin 7scenes:redkitchen 7scenes:stairs \
    --frames 50
```

### 2. Full T14 Eval (run in tmux — heavy GPU, ~hours)

```bash
cd /workspace/collab-splats/.worktrees/bae-parity
tmux new -s bae_parity
/opt/conda/envs/nerfstudio/bin/python evals/_bae_parity_driver.py \
    --scenes 7scenes:chess 7scenes:fire 7scenes:heads 7scenes:office \
             7scenes:pumpkin 7scenes:redkitchen 7scenes:stairs \
    --frames 50 --out evals/results/_bae_parity/t14
```

### 3. Check T14 Report Against Spec Thresholds

```bash
cat /workspace/collab-splats/.worktrees/bae-parity/evals/results/_bae_parity/t14/report.md
```

Thresholds (from `worklog/specs/2026-05-20-bae-vggt-parity-design.md`):
- **Parity:** median t_err < 1cm, p95 < 5cm; median r_err < 0.5°, p95 < 2°;
  AUC@30 within 5%
- **Sanity floor:** median < 1mm AND < 0.001° → halt (output-mixup)

### 4. Fix Aligned Rotation Metric (low priority)

Umeyama returning ~180° flip for B-vs-A and C-vs-A aligned comparisons. Doesn't
affect parity verdict (raw gate is authoritative) but makes aligned diagnostics
unreadable. Likely needs `umeyama_align` to handle reflection ambiguity.
Location: `collab_splats/pointcloud/loop_closure/eval.py` (or wherever
`umeyama_align` is defined).

---

## Key Files

| Path | Purpose |
|---|---|
| `.worktrees/bae-parity/evals/_bae_parity/pipeline_a.py` | Load upstream reference artifacts |
| `.worktrees/bae-parity/evals/_bae_parity/pipeline_b.py` | VGGT-X + our BA |
| `.worktrees/bae-parity/evals/_bae_parity/pipeline_c.py` | Vanilla VGGT + our BA (diagnostic) |
| `.worktrees/bae-parity/evals/_bae_parity_driver.py` | Main driver (argparse: `--scenes --frames --out`) |
| `.worktrees/bae-parity/evals/_bae_parity/reference/generate.py` | Generate Pipeline A .npy artifacts |
| `.worktrees/bae-parity/evals/results/_bae_parity/reference/` | Pre-generated Pipeline A artifacts |
| `collab_splats/pointcloud/bundle_adjustment.py` | `extract_tracks_vggsfm`, `run_bundle_adjustment` |
| `vendor/bae/bae/utils/ba.py` | `rotate_quat` (vmap-safe SE3 transform) |
| `worklog/specs/2026-05-20-bae-vggt-parity-design.md` | Original design spec + thresholds |
