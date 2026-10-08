# Splats train speed — Scaffold-2DGS step cost

Branch `perf/splats-speed` (from clean/final `e7ab168b`).

## Goal

Cut wall-clock per training step for Scaffold-2DGS on the tutorial scene without changing results,
then measure two levers that do change numerics and decide on them from data.

Config under test (tutorial `SCENE_CONFIG`): scaffold / 2dgs, `max_steps: 3000`,
`resolution_schedule: 300`, `num_downscales: 2`, `pose_opt: true`, `appearance_opt: false`,
losses depth 0.01, normal_consistency 0.05 @700, distortion 0.01 @300, opacity_reg 0, scale_reg 0.
96 portrait 1080x1920 views, ~400k anchors, ~0.9M decoded Gaussians per view.

## Levers

| # | Lever | Class | Where |
|---|-------|-------|-------|
| 2 | Per-factor GPU target cache: each view's image + depth target built once per downscale factor, kept on the GPU; a new factor drops the old one | bit-identical | `splats/utils.py` `cached_target`, `splats/trainer.py` step loop |
| 1 | Host syncs per step: loss values stay 0-d tensors until logged / reported; one index gather instead of five boolean gathers in the decode; branch-free "none open" fallback; branch-free accumulate masks; GPU camera-id table | bit-identical | `splats/losses.py`, `splats/scaffold.py`, `splats/trainer.py` |
| 3 | MLP decode precision: TF32 (bench-global flag) vs bf16 autocast of the three heads (`scaffold.mlp_bf16`) | numeric change | `splats/scaffold.py` `ScaffoldMLPs.forward` |
| 5 | `scaffold.n_offsets` 10 vs 5 | numeric + capacity change | config only (`splats.scaffold.n_offsets`) |

Left as is (data-dependent shapes or not provably bit-identical):

- `torch.nonzero(visible)` in `Scaffold.decode`: anchor gather feeds the rasterizer, shape is data-dependent
- the one remaining nonzero over the keep mask in `_offsets_to_gaussians`: same reason
- `rendered_depth[has_target]` in `depth_loss`: a `torch.where` + sum / count mean reduces in another order
- `distloss=True` before the distortion start step: gsplat's backward still reads a zero
  `v_render_distort`; gating it needs step plumbing through both models and a GPU proof

## Bit-identity argument (levers 1 + 2)

- cache: same `downscale_image` + `prepare_target` calls on the same inputs, so identical tensors;
  CPU unit test compares against the per-step path for every factor
- index gather vs boolean mask: PyTorch lowers a bool index to `nonzero` + integer index, so values and
  gradients match; the integer ids are computed once instead of five times
- "none open" fallback: setting the argmax slot unconditionally is a no-op when any slot is open
  (the max is then > 0, so it is already kept)
- accumulate: `decode_index` is unique per step, so adding `0.0` / `0` for unrendered slots leaves
  every sum unchanged; replaces two boolean gathers
- loss values: `.detach()` of the same scalars, converted with `.item()` only when logged and at the end

## Measurement protocol

Bench script (scratch, not in repo) trains the tutorial scene through `Reconstructor.splats()` into a
fresh scratch output dir and writes JSON: train seconds, ms/step over [0, 600) and [600, max_steps)
(one `cuda.synchronize` at each boundary), mean PSNR / SSIM from `splats_quality_report.json`,
peak GPU memory, sha256 of the final parameters.

- seed: `torch.manual_seed`, `np.random.seed`, `random.seed` before training (the trainer seeds only
  the view order; background color and MLP init are unseeded)
- determinism probe: same code, same seed, twice; equal hashes → identity is checkable by hash
- gsplat backward and `index_add_` use CUDA atomics, so hashes may differ run to run; then identity is
  judged by max-abs parameter difference vs the same-code run-to-run spread (`bench.py --compare`)
- one GPU job at a time; nothing else on the GPU

## Acceptance

- levers 1 + 2: CPU equivalence tests green; GPU check at 1600 steps (covers all three factors,
  accumulate from step 501, both late-start losses) shows hash equality, or a diff no larger than the
  same-code spread
- levers 3 / 5: reported as train seconds and mean PSNR / SSIM deltas vs the levers-1+2 baseline over
  the full 3000 steps; a default changes only on a measured win with no visible quality loss
- the `scaffold.mlp_bf16` knob is removed or made the default once measured

## Results (2026-10-08, tutorial scene, 3000 steps, seed 0, shared GPU)

| arm | ms/step 600-end | train s | PSNR | SSIM | decoded/step |
|---|---|---|---|---|---|
| clean/final | 101.7 | 274 | 15.80 | 0.531 | 710k |
| levers 1 + 2 | 96.5 | 262 | 15.76 | 0.533 | 664k |
| + TF32 (bench flag) | 67.1 | 189 | 15.96 | 0.534 | 710k |
| + `mlp_bf16` | 65.7 | 186 | 15.82 | 0.532 | 729k |
| + `n_offsets` 5 | 72.6 | 201 | 15.71 | 0.483 | 292k |

- GPU shared with an external job (~12 GB, ~95% util); arm-to-arm timing carries that noise
- `mlp_bf16` made the default: TF32-level speed, PSNR/SSIM within spread; knob kept for float32
- TF32 not landed: process-global flag, no gain over bf16
- `n_offsets` stays 10: 5 costs 0.05 SSIM

