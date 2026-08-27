# Scaffold-GS on gsplat — Design

**Date:** 2026-08-27
**Status:** Approved
**Scope decision:** Stage 1 only — the anchor representation feeding the *existing* `rasterization`
(3dgs) and `rasterization_2dgs` (2dgs) paths. PGSR is stage 3 and has its own spec. Octree-GS is
rejected for now (see "Octree-GS: rejected, not blocked").

## Goal

Add Scaffold-GS as an alternative scene representation in `collab_splats/splats/`, selected by a new
`splats.representation` axis crossed with the existing `splats.primitive`. Anchors carry a feature
vector; three MLP heads decode `n_offsets` neural Gaussians per anchor per view; the decoded
Gaussians go through the gsplat rasterizer we already use.

Three outcomes wanted, in one representation:

1. **Surface/mesh quality** — via the eventual Scaffold-PGSR (stage 3); this stage lays the anchor layer.
2. **Fewer primitives / lower memory** — one feature vector per anchor instead of per-Gaussian SH.
3. **View-dependent robustness** — MLP-decoded colour and opacity conditioned on view direction.

## Feasibility verdict (why gsplat can host this)

Scaffold-GS is not a rasterizer change. `gsplat.rasterization` is a pure function of
`(means, quats, scales, opacities, colors)` and makes no assumption that those are leaf
`nn.Parameter`s, so MLP outputs rasterize and gradients flow back into the heads. Everything
Scaffold needs beyond that exists in the pinned gsplat (`d2f5c0f`):

- `gsplat.strategy.base.Strategy` is a two-method interface (`step_pre_backward`,
  `step_post_backward`) — an anchor densifier subclasses it without inheritance tax.
- `gsplat.strategy.ops._update_param_with_optimizer` / `remove` are generic over any
  `ParameterDict` + per-parameter Adam, so anchor grow/prune reuses upstream optimizer-state surgery.
- `info["means2d"]` with `packed=False` supplies the screen-space gradients Scaffold accumulates
  (upstream calls this `viewspace_point_tensor.grad`).

What must be written: the anchor field + decode, the three MLP heads, and an `AnchorStrategy`.
No CUDA.

### Licensing posture — reimplement, never vendor

- **GS-SR** (`yanxian-ll/GS-SR`, the repo that motivated this) has **no LICENSE file** — GitHub's
  license API returns 404. All rights reserved; it cannot be vendored or copied.
- **Scaffold-GS** (`city-super/Scaffold-GS`) and **Octree-GS** (`city-super/Octree-GS`) are under the
  Inria/MPII Gaussian-Splatting License — research/non-commercial.
- **PGSR** (`zju3dv/PGSR`) is a ZJU custom licence: non-commercial, and any modification must be
  open-sourced.

We pin gsplat (Apache-2.0) precisely to stay clear of the Inria-licensed CUDA. So: **reimplement from
the papers**, cite `repo + commit + file + line` at each ported site as elsewhere in this codebase, and
copy no source. Same posture as the TALO decision.

## Octree-GS: rejected, not blocked

Octree-GS is Scaffold-GS plus an LOD layer — the same anchor + MLP core, with anchors additionally
assigned to octree levels and a per-view level chosen by camera distance (`dist2level`), plus optional
progressive coarse-to-fine level unlocking. It is a strict superset, so this stage is not throwaway work.

Rejected for now because:

- **LOD pays off on camera-distance spread.** Its wins are MatrixCity (aerial) and Mip-NeRF360
  (unbounded); GS-SR reaches for it because it targets aerial/satellite tiles with orthophoto and DSM
  output. Our scenes are handheld walkthroughs at near-constant standoff — `dist2level` would collapse
  to one or two active levels.
- **Nine more knobs** (`fork`, `base_layer`, `levels`, `visible_threshold`, `update_ratio`, `dist2level`,
  `init_level`, `extra_ratio`, `extra_up`) stacked on top of anchor grow thresholds that already need
  recalibration for gsplat. Our track record on uncalibrated thresholds is -1.26 dB (`grow_grad2d`).
- **Its headline win is storage** (-64.87% on Mip-NeRF360). Our pressure is training-time
  densification, which anchors already attack.
- Same non-commercial licence, so no relief there either.

**Design implication:** keep the boundary `anchor set -> decode -> Gaussians` clean, so an LOD filter can
slot in front of `decode` later. Costs nothing now. Revisit if we take on a scene with real scale spread
(drone / large outdoor) or if memory still binds after this stage.

## Architecture

New surface is one file, mirroring `gssr/gaussian/scaffold_gaussian.py` so it stays diffable against
upstream while porting:

```
collab_splats/splats/
  scaffold.py      # NEW: config, MLP heads, AnchorField, AnchorStrategy (######## dividers)
  rendering.py     # EDIT: split "produce Gaussians for this view" from "rasterize"
  trainer.py       # EDIT: representation branch, MLP optimizer, ckpt contents
  outputs.py       # EDIT: baked ply for scaffold runs
```

`scaffold.py` is expected around 700 lines. If it passes ~900 during implementation, `AnchorStrategy`
splits into `scaffold_strategy.py`; not before.

### The render seam

`render_view` currently activates raw parameters inline (`exp(scales)`, `sigmoid(opacities)`, SH concat)
and rasterizes in the same function. Split those:

- **`VanillaSource`** — today's activation, byte-identical behaviour.
- **`ScaffoldSource`** — `AnchorField.decode(cam_to_world)`.

`render_view` takes the produced Gaussian dict and rasterizes. `tests/splats/test_rendering.py` is the
guard that the vanilla path is unchanged. This seam is also where an octree LOD filter would go.

### AnchorField

Parameters:

| Tensor | Shape | Role |
|---|---|---|
| `anchors` | `[A, 3]` | anchor positions |
| `offsets` | `[A, K, 3]` | per-anchor offset directions, `K = n_offsets` |
| `anchor_feat` | `[A, F]` | anchor feature, `F = feat_dim` |
| `scaling` | `[A, 6]` | first 3 scale the offsets, last 3 scale the decoded Gaussians |
| `rotation` | `[A, 4]` | anchor quaternion |
| `opacities` | `[A, 1]` | anchor-level opacity; feeds pruning statistics only |

MLP heads (`mlps` section of the same file): `opacity: [F + view_dim] -> K`,
`cov: [F + view_dim] -> 7K` (3 scale + 4 quat per offset), `colour: [F + view_dim (+ appearance_dim)] -> 3K`.

**Init.** Voxelize the seed points at `voxel_size = median kNN spacing x voxel_multiplier`, reusing the
`NearestNeighbors` call already in `init_gaussians_from_points` (`trainer.py:255`). This is scale-free by
construction and therefore behaves identically under `normalize_scene` true or false — deliberately not
Scaffold's absolute `voxel_size: 0.001`, which is calibrated for normalized COLMAP scenes and would be
off by a large factor in our world-unit frame.

**`decode(cam_to_world)`** per view:

1. Frustum-filter anchors.
2. Compute the unit view direction from each surviving anchor to the camera centre. The distance is
   computed only to normalize it, and is NOT concatenated into the head input: upstream's
   `add_opacity_dist` / `add_cov_dist` / `add_color_dist` all default to `False`
   (`arguments/__init__.py`), so the shipped heads read `[anchor_feat, ob_view]` and
   `generate_neural_gaussians` takes the `cat_local_view_wodist` branch. This is also what keeps the
   heads scale-free, which our `normalize_scene` path requires: outputs are written after
   `denormalize_anchors`, so a world-unit head input would render a different model than the one that
   trained (measured: 62.7x distance shift saturated `mlp_opacity`'s tanh at -1 for 100% of offsets).
3. Run the three heads.
4. Drop offsets whose decoded opacity is <= 0.
5. `means = anchor + offset * scaling[:, :3]`; scales/quats from the cov head times `scaling[:, 3:6]`.
6. Return the Gaussian dict **and `decode_index`** — the `(anchor, offset)` slot each emitted Gaussian
   came from. Densification is built entirely on this index.

Under `primitive: 2dgs` the cov head's third scale channel is unused: `rasterization_2dgs` reads
`scales[..., :2]`.

### AnchorStrategy

Subclasses `gsplat.strategy.base.Strategy`. `step_post_backward` scatters the per-Gaussian screen-space
gradient norm back through `decode_index` into per-slot accumulators (gradient sum, opacity sum, denom).
Every `refine_every` steps inside `[update_from, update_until]`:

- **Grow**: multi-level voxel candidate selection with `scatter_max` dedup against occupied cells; new
  anchors initialise with zero offsets and aggregated features (Scaffold's `anchor_growing`).
- **Prune**: anchors whose accumulated opacity stays below `min_opacity` after enough visits.

All tensor surgery goes through `gsplat.strategy.ops._update_param_with_optimizer`, which grows and
prunes Adam `exp_avg` / `exp_avg_sq` alongside the parameters. MLP parameters are excluded from the dict
handed to the strategy (fixed size, nothing to grow), so `check_sanity` runs against the anchor subset only.

Two traps, recorded now:

1. **Gradient key differs by primitive.** `DefaultStrategy` reads `gradient_2dgs` on the 2dgs path, and
   gsplat's 2dgs backward writes `.absgrad` on `means2d` only, never on that tensor
   (`trainer.py:333`). `AnchorStrategy` must select the matching key per primitive or it silently
   accumulates zeros and never grows.
2. **Threshold units.** The raw `means2d.grad` is pixel-space, but `DefaultStrategy` renormalises it to
   [-1, 1] screen space before thresholding (`strategy/default.py:243-249`:
   `grads[..., 0] *= width / 2 * n_cameras`, likewise for height). `AnchorStrategy` copies those exact
   lines, after which Scaffold's published `grad_threshold` transfers directly with no conversion, and
   the per-slot mean (`grad2d / count`, Scaffold's `offset_gradient_accum / offset_denom`) matches too.
   The live subtlety is coarse-to-fine: `width` / `height` are the *downscaled* dims, so the effective
   threshold shifts each time the resolution doubles. Precedent for getting this wrong: `grow_grad2d`
   8e-4 (absgrad-calibrated) starved densification by 1.26 dB.

## Config

`representation` is a new axis, orthogonal to `primitive`. Every existing config stays valid and untouched.

```yaml
splats:
  primitive: 2dgs            # rasterizer: unchanged
  representation: scaffold   # NEW: vanilla | scaffold
  appearance_opt: true       # our per-image affine module: unchanged, independent
  scaffold:
    n_offsets: 10
    feat_dim: 32
    voxel_multiplier: 1.0    # x median kNN spacing of the seed points
    update_from: 1500
    update_until: 15000
    refine_every: 100
    grad_threshold: 2e-4        # Scaffold's published value; screen space matches gsplat after renormalisation
    min_opacity: 0.005
    appearance_dim: 0        # Scaffold's own per-image embedding; 0 = off
    mlp_lr: 2e-3
```

### Colour and appearance — both mechanisms ship

Scaffold's `mlp_colour` emits RGB directly, so `sh_degree` and `sh_degree_interval` go inert under
scaffold; config validation rejects them rather than silently ignoring them.

Scaffold's own `embedding_appearance` (`scaffold.appearance_dim > 0`, concatenated into the colour MLP
input) and our existing `AppearanceModule` (`appearance_opt`, per-image affine, train views only) are
**independent toggles**. Neither is retired. Which one wins, and whether they are additive, is measured —
see the sweep. This matters because every lever pair measured on this codebase so far has been
anti-additive.

### Other knock-ons

- **`scale_reg`** currently penalises log-scale *parameters*. Under scaffold it reads the **decoded**
  scales from the render instead. Same loss name, different source.
- **`pose_opt`** keeps working and gains a second gradient path: view direction feeds the MLPs, so pose
  deltas now move colour and opacity too. Harmless; gets a comment at the site.
- **`num_downscales`** (coarse-to-fine) stays at its default. Flagged, not solved: c2f already OOM'd 2dgs
  under `DefaultStrategy`, and its behaviour under anchor growing is unmeasured.
- MCMC-only regularisers (`opacity_reg`) are simply absent from scaffold defaults.

## Artifacts

Scaffold has no static per-Gaussian parameters — they exist only after a view-dependent decode.

- **`ckpt.pt`**: anchors, offsets, features, scaling, rotation, opacities, the three MLP `state_dict`s,
  and the config. Exact restore.
- **`splats.ply`**: **baked**. One pass over the training cameras records each anchor's mean observed view
  direction; each anchor is decoded once at that direction; the resulting Gaussians are written through
  the existing writer with RGB in the degree-0 SH band and `shN` zeroed. Provenance attribute
  `ply_baked=true` marks it lossy. Viewer-loadable, which the alternative (no ply) is not.
- **`splats.zarr`**: unchanged. It stores *rendered views*, not Gaussian parameters, so every downstream
  consumer — TSDF mesh (`mesh/utils.py:626`), `evals/scripts/eval_splats.py`,
  `evals/scripts/analyze_splats.py`, the dashboard — is unaffected by the view-dependent decode.

## Testing

`tests/splats/test_scaffold.py`, flat functions:

- decode output shapes for both primitives;
- `decode_index` round-trips to the correct `(anchor, offset)` slot;
- opacity mask drops offsets below threshold;
- a synthetic high-gradient slot grows an anchor;
- low-opacity anchors prune;
- Adam `exp_avg` shapes track the parameters through both grow and prune;
- anchor count invariant when the same scene is scaled 10x (voxel size is spacing-derived);
- the gradient key selected matches the primitive;
- baked ply loads and has the expected field set.

`tests/splats/test_rendering.py` (existing) guards that the vanilla source split changed nothing.

## Measurement

**Baselines are not rerun.** Both vanilla numbers are already logged in
`docs/superpowers/specs/2026-08-26-psnr-levers-design.md` (§Measured, §Combined runs) and are the
comparison targets as-is. Scaffold runs must reproduce each baseline's protocol exactly — same scene,
same config file, same step count — or the comparison is void.

| primitive | scene | config | PSNR | SSIM | gaussians | train s |
|---|---|---|---|---|---|---|
| 3dgs | `GH010229` (300 fr, undistort, dense-aligned VDA) | `gopro_3dgs_c2f_overrides.yaml` + `appearance_opt: true` | 21.606 | 0.7169 | 1 000 000 | 601 |
| 2dgs | `GH010229_undist_r7` | `gopro_lever_combined_2dgs.yaml` | 20.800 | 0.6780 | 1 875 613 | 803 |

Protocol carried from those runs: 12k steps (**not** 30k), `cap_max` 1M for the vanilla 3dgs reference,
`normalize_scene`, `pose_opt`, c2f on for 3dgs and **off** for 2dgs (`DefaultStrategy` still OOMs with it),
`semantics.enabled: false` per the standing rule.

Note the two baselines sit on **different frame sets** — the 3dgs appearance-only run is on the base
scene, the 2dgs high-water mark is on the radius-7 scene. Each scaffold run therefore uses the scene its
own primitive's baseline used. Do not cross them.

New runs — 8 total, scaffold only:

| # | primitive | scene | scaffold.appearance_dim | appearance_opt |
|---|---|---|---|---|
| 1-4 | 3dgs | `GH010229` | 0 / on | off / on |
| 5-8 | 2dgs | `GH010229_undist_r7` | 0 / on | off / on |

Reported per run: PSNR, SSIM, anchor count, decoded-Gaussian count, peak memory, wall time.

**Acceptance bar:** scaffold reaches its primitive's logged PSNR within noise at materially fewer
primitives, and the 2x2 appearance question is answered (which exposure model wins; whether they stack).
Mesh quality (main-component fraction — current best 2dgs 0.634) is deferred to the PGSR stage.

**Run hygiene:** send splat outputs through the `/tmp` symlink — `/workspace` quota is ~8.7 GB and
`zarr mode="w"` rmtree's a symlinked `splats.zarr` (the dir-level symlink is the safe form).

## Out of scope

- **PGSR** — stage 3, separate spec. Feasibility already checked and it does not need CUDA work:
  gsplat's `extra_signals` carries PGSR's `input_all_map` (normal 3 + distance 1 = exactly the 4-channel
  headroom we already use for 3dgs normals; the `1.0` channel is alpha, returned separately), and
  `plane_depth = d / (n . K^-1 p)` is closed-form per pixel in torch. Two gotchas to carry forward:
  `rasterization_2dgs` has **no** `extra_signals`, so PGSR rides the 3DGS `rasterization` path (which is
  what PGSR is — flattened 3DGS); and the compiled kernel takes rgb+depth+4 = 8 channels but rejects 7,
  so PGSR must render `RGB+ED`, not `RGB`. The multi-view NCC / homography / reprojection losses are pure
  torch, and are the bulk of that stage's work.
- **Octree-GS LOD** — see above.
- **Mesh main-fraction measurement** — deferred to the PGSR stage.

## Staging

1. **This spec** — anchors + `AnchorStrategy` on the existing 3dgs / 2dgs rasterizers.
2. **Scaffold + 2DGS tuning** — threshold and `n_offsets` calibration on the surface primitive.
3. **Scaffold + PGSR** — planar primitive, unbiased depth, single- and multi-view geometric losses.
