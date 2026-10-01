# Reconstruction quality report speed-up — design

Status: approved in brainstorming 2026-10-01. Implementation on worktree `.worktrees/report-speed`,
branch `perf/report-speed`, forked from `clean/final`.

## Problem

`Reconstructor.reconstruction_quality_report` takes ~40 min on a 1k-frame scene. Static audit of
`collab_splats/geometry/metrics.py` and the stage (`wrapper/reconstructor.py`):

| # | Hotspot | Where | Cost at N=1000 |
|---|---|---|---|
| H1 | per-pair `.cpu()` + CPU `np.histogram` | `_collect_pairs` | measured 2.02 ms/pair x 999k pairs ≈ 33 min |
| H2 | ~8 GPU->CPU syncs per pair (`sel.any()`, `float(q[..])`, medians, `sel.sum()`) | `_collect_pairs` | per-pair latency, unmeasured |
| H3 | `transform_points` recomputes camera z that `depth_residual` already returned as `expected` | `metrics.py:167` | one extra (P,3) transform per pair |
| H4 | per-frame medians scan every pair per frame: O(N x pairs) Python | `compute_reconstruction_quality` | ~1e9 comparisons |
| H5 | images cast to float32 (~25 GB), guides uint8 (6 GB), upsampled depth (8 GB), float64 torch copy (16 GB) | stage + `compute_photometric_ncc` | past the 46.6 GB cgroup cap |
| H6 | guided upsample of every frame on CPU, then float64 CPU projection | `upsample_depths`, photometric loop | 33.6 ms/frame upsample, measured |
| H7 | every ordered pair projected at full res, including pairs that share no view | `_collect_pairs` | O(N^2) floor |

## Measurements behind the decisions

Scratch scripts in the session scratchpad; A40, torch CUDA.

**Histogram** (120k residuals/pair, 10,676 bins = `residual_bin_edges` at N=1000, 294x518):

| Method | ms/pair | Counts vs current |
|---|---|---|
| current: `.cpu()` + `np.histogram(edges)` | 2.02 | — |
| `np.histogram(bins=k, range=(-1, 1))` | 1.71 | identical |
| GPU `torch.bucketize` (f64 edges) + `bincount` | 0.18 | identical |
| GPU `torch.histc` (f32) | 0.05 | 1,736 counts misbinned — rejected |

**Guided filter** (`kornia.filters.guided_blur` 0.8.2 vs `utils/image._guided_filter`, 1080x1440, r=6):
- same algorithm (He et al., gray guide, box means of a and b)
- float64 interior beyond 2r: max |Δ| 1.3e-13
- the only difference is the border: kornia `reflect` = torch reflect = cv2 `BORDER_REFLECT_101`;
  ours uses cv2 `BORDER_REFLECT`
- with ours switched to `BORDER_REFLECT_101`: float32 max |Δ| 1.4e-4 on values up to 2.8, all pixels
- border switch alone moves 3.8% of pixels (the 2r band), max |Δ| 0.13
- speed: 33.6 ms/frame CPU vs 5.0 ms/frame GPU (batch 16)

## Decisions

- Scope: the `reconstruction_quality_report` stage only.
- Report values: exact by default; one approximation behind a knob, default off.
- Keep our `multiview_depth_confidence`; do not swap it for the mapanything upstream.
- Histogram via `torch.bucketize` + `bincount`, not `histc`.
- `utils/image.py` gains torch + kornia; only its docstring promised torch-free. The enforced
  torch-free surface is `import collab_splats.utils` and `collab_splats.utils.io`, both untouched;
  every importer of `utils.image` already imports torch. Import paths unchanged.
- Target-view batching reuses `transform_points` / `project` / `depth_residual`, generalized to a
  leading pose batch; the 2-D call path stays bit-identical.
- Guided filter via `kornia.filters.guided_blur`, replacing our CPU filter; the 2r border band
  follows kornia's `BORDER_REFLECT_101`. Accepted border-band change (option a), pinned by a parity
  test against a cv2 oracle.

## Section 1 — exact speed-ups (default path)

All in `collab_splats/geometry/metrics.py` unless noted. Each item is its own commit with its own
parity gate.

1. **GPU histogram.** `bounded_residual` in torch float64 -> `torch.bucketize(u, edges, right=True) - 1`
   -> `bincount(minlength=k)` into one device int64 accumulator; one `.cpu()` at the end.
   Values at u == 1.0 clamp into the last bin, as `np.histogram` does.
2. **Fewer syncs.** Per kept pair, stack (n_pixels, q25, q50, q75, parallax median, depth median)
   into one device row; empty pairs masked, not branched on. Rows transfer once per source frame i.
3. **No redundant transform.** `depth_agreement` also returns `expected` (camera z); `_collect_pairs`
   reads it instead of calling `transform_points`. `multiview_depth_confidence` ignores the extra
   output; its behavior is unchanged.
4. **Batched targets.** For a source frame i, the kept j's run through `depth_residual` in chunks of
   B views x P points; B from `utils/torch_utils.infer_batch_size`. Gate: integer columns and
   histogram bit-identical; if batched matmul rounding breaks that, the item is dropped and the
   measured cost recorded here.
5. **O(pairs) per-frame medians.** One pass buckets `|median_rel_depth_error|` under idx1 and idx2,
   then takes each frame's median. Output identical to the O(N x pairs) scan.
6. **Photometric memory and speed.**
   - stage passes uint8 images (no float32 cast); `compute_photometric_ncc` takes uint8
   - `utils/image.upsample_depths` filters with `kornia.filters.guided_blur` on GPU, one call per frame
     with depth and validity as two channels;
     `_box` and `_guided_filter` are deleted (reuse/retire rule); the parity test carries its own
     cv2 `boxFilter` oracle with `BORDER_REFLECT_101`
   - streaming: only frame i's depth is ever lifted (frame j contributes color only), so it is
     upsampled one frame at a time, and only when frame i has a partner; the float64 full-stack
     torch copy is removed
   - projection in float32 on GPU; the round + bounds semantics are kept as is (depth_residual's
     continuous [0, W-1] bound differs at the edge column, so it is not reused here)
   - side effect: mesh stage and splats depth targets share `upsample_depths`, so they speed up and
     take the same border-band change

## Section 2 — optional pair pruning (default off)

New config block in `configs/base.yaml`:

```yaml
reconstruction_quality_report:
  min_pair_overlap: 0.0   # 0 = every ordered pair (current behavior)
```

Passed as a keyword default: `compute_reconstruction_quality(..., min_pair_overlap=0.0)`.

**Pre-pass, per source frame i** (skipped entirely when `min_pair_overlap == 0`):
- unproject frame i on a stride-8 pixel subsample
- project into all N cameras in one batched op: in front, in bounds, source depth > 0
- overlap(i, j) = share of valid subsampled points inside camera j's frustum
- full-resolution pass runs only over j with overlap >= `min_pair_overlap`

Frustum-only ignores occlusion, so it over-estimates overlap: it keeps extra pairs, never drops a
visible one beyond stride-8 aliasing at tiny overlaps.

**When on:** pruned pairs vanish from `depth_pairs`; their pixels leave the histogram;
`multiview_agreement` and per-frame medians move only where a pruned pair contributed. The report's
`scene` block records `min_pair_overlap`.

**A/B before recommending a value:** sweep `min_pair_overlap` in {0, 0.01, 0.05, 0.1} on the
measurement scene; record per-frame `multiview_agreement` max |Δ|, per-frame median max |Δ|,
histogram total-variation distance, pairs kept, wall time. Default stays 0.0; the user picks a value
after the table.

## Section 3 — verification

**Measurement scene.** `/workspace/outputs/ocr_viewer/GH010229/vggt_omega/pointcloud.zarr`
(294 frames, 384x688 model grid; user choice 2026-10-01). No bucket pull. 1k-frame figures are N^2
extrapolations from it and are labeled as such.

**Baseline first.** Profile the current stage before any change; wall time per phase
(`_collect_pairs`, per-frame medians, image load, upsample, photometric) and peak RSS. Run in tmux,
no side processes.

**Parity gates.**
- Section 1 items 1-5: report JSON equal before vs after on the fixture scene and the measurement
  scene; integer columns and histogram counts bit-identical, float columns `atol=1e-6`.
- Item 6: tolerance gate; record `photometric_ncc` max |Δ| and `n_pixels` delta on both scenes.
- New test: `upsample_depths` filter vs a cv2 `boxFilter` (`BORDER_REFLECT_101`) guided-filter
  oracle within 1e-3, so a kornia upgrade that changes the filter fails loudly.

**Memory gate.** Peak RSS of the stage on the measurement scene, scaled to 1k x 1080p, well under
46.6 GB.

**Tests** (`tests/geometry/test_metrics.py`, `tests/utils/`):
- existing tests stay green
- O(pairs) medians equal the old scan on a random pair set
- GPU histogram equals `np.histogram` on random residuals including u = ±1 edge values
- `min_pair_overlap=0` equals the unpruned path; a positive value drops only zero-overlap pairs on a
  synthetic scene

**Done.** Before/after wall time on the measurement scene; the Section 2 A/B table; full
`tests/` gate in the worktree (`cd <wt> && PYTHONPATH=<wt>`, print `collab_splats.__file__`);
CHANGELOG entry; In-Flight entry removed from CLAUDE.md.

## Out of scope

- Other stages (preproc, pointcloud, mesh, splats) except the shared `upsample_depths` speed-up.
- Pixel subsampling of the full-resolution depth pass.
- Replacing `multiview_depth_confidence`.
