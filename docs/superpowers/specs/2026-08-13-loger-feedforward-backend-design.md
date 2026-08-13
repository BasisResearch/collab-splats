# LoGeR as a feedforward backend

**Date:** 2026-08-13
**Status:** design approved, plan not yet written
**Scope:** add `loger` as a fourth `Reconstructor`-reachable feedforward backend

## Goal

Make LoGeR usable exactly like `vggtx`, `mapanything`, and `vggt_omega`: set
`pointcloud.backend: loger` and the rest of the pipeline is unchanged. Minimal additions —
every piece of new code below is justified against "why can't an existing function do this".

## What LoGeR is

Pi3 backbone (DINOv2-L `dinov2_vitl14_reg` encoder, patch size 14, 36-layer decoder) plus a
**TTT (test-time training) fast-weight memory** (`FastWeightGluMLPMultihead` / `TTTOperator`,
inserted at even decoder layers, updating K and V in place with learned per-head learning rates
and an optional Muon optimizer), run with **sliding-window inference and overlap stitching**.

It is purpose-built for long sequences. The window keeps model memory bounded regardless of
sequence length, and `reset_every` hard-resets the fast weights for extreme lengths. That is the
entire reason to add it: the VGGT family is set-based and OOMs past a few hundred frames.

Two checkpoints on HF `Junyi42/LoGeR`, as raw `latest.pt` files under per-variant
subdirectories: `LoGeR` and `LoGeR_star` (LoGeR* uses SE(3)). Each ships an
`original_config.yaml` in the repo itself, so model kwargs need no download.

### LoGeR is a restricted trained model, not a flexible architecture

`Pi3.__init__` exposes `pi3x` and `pi3x_metric`, which look like runtime options. They are not —
they are training-time architecture switches, and **neither shipped config sets them**, so both
checkpoints run `pi3x=False`.

`pi3x=True` swaps `self.point_head` for a different `ConvHead` (`num_features=4`,
`dim_out=[2, 1]`), and `pi3x_metric` additionally creates `self.metric_token`,
`self.metric_decoder`, and `self.metric_head`. Those are differently-shaped and additional
parameters, so `load_state_dict(strict=True)` against a LoGeR checkpoint fails. There is no
drop-in upgrade path to the Pi3* architecture without Pi3*-trained weights.

One consequence worth recording: the metric branch (`metric_head(...).exp()` scaling camera
translation, `pi3.py:792`) is exactly what would produce metric depth, and it is off. **LoGeR
depth is non-metric, like every other backend we run.** No `depth_trunc` win here.

The two variants also differ from each other architecturally — `ttt_pre_norm: true` on `LoGeR`,
absent on `LoGeR_star`; `se3: true` on `LoGeR_star` only — so `variant` selects a
config-plus-weights pair, not merely a weight file. Both set `ttt_inter_multi: 4` where the
`Pi3` default is `2`, which is why the shipped yaml is authoritative and constructor defaults
must never be relied on.

## Source selection: which fork

**Neither fork is a superset.** PolyCam forked `Junyi42/LoGeR` at `85cf441` (2026-03-10) and
added `5d7c1a7 run_loger.py` (2026-04-14) plus a CLAUDE.md. Upstream then added three commits
PolyCam does not have: `296cb54` (vbr preprocess), `13d78a0` (vbr eval fix), and
`7685b7a [FEAT] load images to cpu to reduce VRAM usage` (2026-04-27). `git diff up/main..main`
is 7 files, +816/-392.

**Decision: vendor upstream `Junyi42/LoGeR` at `7685b7a`.**

The deciding commit is `7685b7a`. Our container is capped at 46.6 GB and long sequences are the
whole point of this backend, so the VRAM fix is directly load-bearing. PolyCam's unique
contribution is `run_loger.py` — a headless CLI whose job our creator takes over anyway. We port
the one algorithm inside it that we need (the focal estimator) and discard the rest.

Vendored to `third_party/LoGeR/`. `third_party/*` is gitignored (`.gitignore:7`); only
`third_party/README.md` is tracked, and the pinned commit is recorded there. `sys.path.insert`
happens inside `_load_model` and is removed in a `finally`, following
`vggt_spark_creator.py:34,126`.

**Zero new pip dependencies.** Verified against `/opt/venv/reconstruction/bin/python`
(torch 2.5.1+cu121, py 3.11.15): `huggingface_hub`, `natsort`, `plyfile`, `einops`,
`safetensors`, `roma`, `yaml` are all already installed.

## Intrinsics: how LoGeR differs from every other backend

This is the substantive difference and the main source of implementation risk.

| Backend | Where K comes from | Per-frame? |
|---|---|---|
| VGGT-X / VGGT-Omega / VGGT-SPARK | `pose_encoding_to_extri_intri(pose_enc, hw)` — direct network output, FOV-parameterised, regressed jointly with pose | yes |
| MapAnything | `p["intrinsics"]` after `postprocess_model_outputs_for_inference` — derived from predicted ray directions | yes |
| **LoGeR** | **nothing predicts it.** `CameraHead` returns pose only. K is **solved** by least-squares pinhole fit to the predicted pointmap | **shared across frames** (our choice) |

The failure modes are inverted, which matters for how we guard each:

- A **regressed** K can be geometrically invalid. `vggt_omega.py:220` carries a comment naming
  exactly this: it is the "root cause of `cx > model_W` in `result.intrinsics`". The network is
  free to emit a principal point outside the image.
- A **fitted** K is centre-principal by construction, so `cx`/`cy` cannot land outside the
  image and that class of bug is unreachable. It fails differently: it can be
  plausibly-but-globally-wrong if the pointmap is scaled oddly or the confidence mask is
  unrepresentative. That is a *silent* failure, so it needs a numeric check, not a range guard.

Note it is **not** square-pixel by construction. `_snap_square_pixels(fx, fy, tol=0.02)`
averages the two only when they already agree within 2%; otherwise they stay distinct. That
matters — see the aspect-ratio discussion under `_preprocess`.

### What LoGeR's pointmap actually is

On the non-`pi3x` path — which is the path both shipped checkpoints take — the point head emits
three channels which are split and recombined (`pi3.py:772-775`):

```python
ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, Nw, H, W, -1)
xy, z = ret.split([2, 1], dim=-1)
z = torch.exp(z)
local_points = torch.cat([xy * z, z], dim=-1)
```

So `X = xy₀·Z`, `Y = xy₁·Z`, `Z = z`: `xy` is a free per-pixel, per-frame **ray-direction
field**. A pinhole K requires `xy₀ = (u − cx)/fx`, a fixed function of pixel coordinate identical
across frames. Nothing constrains the network to that, so the native pointmap can represent
things no single K can reproduce: lens distortion (worst at the periphery), per-frame ray
variation, and any other learned non-pinhole deviation.

This is structurally the same situation as MapAnything, which also predicts ray directions and
also collapses them to a K — precedent that the pinhole approximation is acceptable in practice.
The K-unprojected cloud is an approximation of the native one, and the parity test below
measures exactly how good an approximation it is.

### The fit

`_estimate_shared_focal(local_points, conf)`, ported from `run_loger.py`'s
`estimate_focal_lengths` / `_focal_from_frame` / `_snap_square_pixels`.

`local_points[..., 2]` **is** depth, per the construction above. So each valid pixel gives one
equation of the pinhole model, and a confidence-weighted least squares over all frames yields
`fx` and `fy`. **`fx` and `fy` are fitted independently** and only merged by
`_snap_square_pixels` when within 2% — this is load-bearing, see `_preprocess`.

Shared across frames is correct for our inputs: `frames.zarr` comes from a single video, one
physical camera, and the config exposes no zoom.

Roughly 40 lines. Justified because there is no alternative — the model emits no K.

**The port drops one branch.** `estimate_focal_lengths` falls back to `max(W, H) * 1.2` when the
fit fails (the same heuristic as `localization/localizer.py:24 seed_intrinsics`). We delete it
and raise instead: a wrong-but-plausible K against real depth is precisely the `be24be2`
mesh-intrinsics regression.

### When to choose LoGeR over the alternatives

**Choose it for:** sequences past the ~300-frame ceiling where VGGT-Omega OOMs; long captures
where drift accumulates, since the TTT memory is designed to carry state across the sequence.

**Avoid it for:** short sequences (<100 frames), where the set-based VGGT models see every frame
jointly and LoGeR's windowing buys nothing; anything needing loop closure (see below);
captures where intrinsics genuinely vary (zoom), which the shared-K fit cannot represent;
**unordered image collections** — LoGeR's windows are sequential, whereas the VGGT family is
set-based and has no ordering requirement.

## Footprint

3 new files, 7 modified, zero new pip deps.

| | Path | Why |
|---|---|---|
| new | `third_party/LoGeR/` | vendored upstream @ `7685b7a`, gitignored, pinned in tracked `third_party/README.md` |
| new | `collab_splats/pointcloud/feedforward/loger.py` | `LoGeRCreator` + two module-level helpers |
| new | `tests/pointcloud/test_loger_creator.py` | mirrors `test_vggt_omega_creator.py` |
| mod | `collab_splats/pointcloud/feedforward/__init__.py` | guarded export, `try/except ImportError`, same shape as `VGGTOmegaCreator` |
| mod | `collab_splats/pointcloud/__init__.py` | `_LOGER_AVAILABLE` guard + `_REGISTRY["loger"]`, mirroring `_OMEGA_AVAILABLE` at lines 8–28 |
| mod | `collab_splats/wrapper/reconstructor.py` | `_FEEDFORWARD_BACKENDS` (L43), `creator_map` (L189), per-backend kwargs passthrough (L195), LC refusal, `max_frames` warning |
| mod | `configs/base.yaml` | backend comment + `pointcloud.loger:` kwargs block (see below) |
| mod | `configs/README.md` | backend row, intrinsics note, when-to-choose, `max_frames` guidance |
| mod | `docs/source/api/pointcloud.rst` | one `automodule` stanza for `feedforward.loger` |
| mod | `docs/source/conf.py` | `"loger"` added to `autodoc_mock_imports` (line 37) — the vendored package is not importable on a docs build |

### Reused unchanged — no new code

`unproject_and_filter_points` (`vggtx.py:92`), `invert_poses` and `extrinsics_to_homogeneous`
(`geometry/transforms.py`), `build_pycolmap_reconstruction`,
`_rescale_reconstruction_to_original_dimensions`, `frames_as_pil_source`, `FeedforwardResult`
and its zarr IO, `compute_multiview_depth_confidence`, and the whole 5-step template.

### Genuinely new, and why

1. `_estimate_shared_focal(local_points, conf)` — justified above. ~40 lines.
2. `_loger_original_coords(sizes, model_hw)` — ~15 lines. LoGeR resizes without cropping, so
   each row is `[0, 0, orig_w, orig_h, orig_w, orig_h]`. This is *strictly simpler* than
   `_compute_omega_original_coords`'s crop arithmetic and cannot be replaced by it, because that
   function computes a crop that LoGeR never performs. Needed so
   `_rescale_reconstruction_to_original_dimensions` can invert the resize.
3. `LoGeRCreator` itself.

### Deliberately not implemented

- `extract_intermediate_features` raises `NotImplementedError` naming LoGeR's native windowed
  TTT memory as the reason.
- No loop closure in the first cut.

### Pre-existing gaps, noted not fixed

- `vggt_spark` is in `_REGISTRY` (`pointcloud/__init__.py:28`) but absent from
  `_FEEDFORWARD_BACKENDS` (`reconstructor.py:43`) and `creator_map` (L189), so it is
  unreachable from `Reconstructor`. `loger` is wired into both, since the requirement is that it
  work like any other backend.
- Neither `vggt_omega` nor `vggt_spark` appears in `docs/source/api/pointcloud.rst`, which
  documents only `base`, `sfm`, `feedforward.base`, `vggtx`, and `mapanything`. Optional
  vendored backends have simply never been added. `loger` gets its stanza; back-filling the
  other two is a separate, unrelated change.

## Data flow: the five template methods

### `_load_model(device)`

`sys.path.insert(0, third_party/LoGeR)` → `from loger.models.pi3 import Pi3` → remove in
`finally`. Read `ckpts/{variant}/original_config.yaml` from the vendored tree.

**The yaml's `model:` block mixes constructor kwargs and forward kwargs, so routing is
three-way, not a filter.** `LoGeR_star` sets `se3: true`, but `se3` is not a `Pi3.__init__`
parameter — it is popped inside `forward` (`pi3.py:589`, mutually exclusive with `sim3`). A
naive `inspect.signature(Pi3.__init__)` filter would silently discard it and run LoGeR* in the
wrong alignment mode. So:

- key in `inspect.signature(Pi3.__init__)` → constructor kwarg
- otherwise, key in the known forward-kwarg set (`se3`, `sim3`, `window_size`, `overlap_size`,
  `reset_every`, `num_iterations`) → forward kwarg
- otherwise → **raise**, naming the key. Silent drops are how LoGeR* would degrade invisibly.

Re-parse `ttt_insert_after` / `attn_insert_after` when they arrive as `"[4,8]"` strings. Weights
via `hf_hub_download("Junyi42/LoGeR", f"{variant}/latest.pt")`, `torch.load(map_location="cpu")`,
unwrap `model_state_dict`, strip any `module.` prefix, `load_state_dict(strict=True)`,
`.eval().to(device)`. Constructor arguments override the yaml; the yaml overrides `Pi3`
defaults, which are wrong for both variants (`ttt_inter_multi` is 4 in both configs, 2 in code).

### `_preprocess(frames, frame_idxs) -> (views, image_paths, original_coords)`

Synthetic `frame_{idx:06d}` labels, the VGGT-Omega convention — the frame store is the only IO
path and there are no real filenames. Target size from LoGeR's own rule:
`scale = sqrt(pixel_limit / (W * H))`, round each axis to a multiple of 14, shrink the longer
axis until under budget. PIL LANCZOS → `(N, 3, H, W)` float in `[0, 1]`, fed through
`frames_as_pil_source` so nothing touches disk.

LoGeR derives the target size from frame 0 alone. Rather than inherit that silent assumption we
assert frame-size uniformity and raise otherwise.

`original_coords` is where LoGeR is simpler than Omega: pure resize, no crop, so each row is
`[0, 0, orig_w, orig_h, orig_w, orig_h]`.

#### The resize is mildly anisotropic, and that is fine — conditionally

`k = round(W_t/14)` and `m = round(H_t/14)` round **independently**, and the shrink loop can
remove a whole 14 px from one axis. So aspect ratio is not exactly preserved; the image is
stretched by up to a few percent on one axis.

This does not distort the pointcloud, for two reasons that both depend on *not* forcing square
pixels:

1. `_estimate_shared_focal` fits `fx` and `fy` **separately**, so the anisotropy is absorbed into
   the K rather than corrupting the geometry. `_snap_square_pixels` merges them only when they
   already agree within 2%.
2. `_rescale_reconstruction_to_original_dimensions` scales with separate `scale_x` and `scale_y`,
   so the original-resolution K recovers the true aspect.

If either of those became isotropic — forcing `fx == fy`, or scaling K by a single factor — the
error would be baked in. This is recorded because the correctness is non-obvious and easy to
"simplify" away. The model is also trained with this exact preprocessing, so the stretch is
in-distribution.

Cropping to a multiple of 14 instead would be worse: it discards field of view, and
`original_coords` would need Omega's crop arithmetic back.

### `_forward(model, views)`

| LoGeR output | shape | our key | transform |
|---|---|---|---|
| `local_points[..., 2:3]` | (N,H,W,1) | `depth` | already depth; model builds `cat([xy*z, z])` |
| `sigmoid(conf)` | (N,H,W,1) | `depth_conf` | squeeze trailing axis |
| `camera_poses` | (N,4,4) **c2w** | `extrinsic` | `invert_poses(...)[:, :3, :]` → w2c (N,3,4) |
| `points` | (N,H,W,3) | `native_points` | already world-space, kept free |
| — | — | `intrinsics` | fitted K broadcast to (N,3,3) |

`camera_poses` is camera-to-world. `pi3.py:807` proves it:
`points = einsum('bnij,bnhwj->bnhwi', camera_poses, homogenize_points(local_points))`. Our
`FeedforwardResult.extrinsics` is world-to-camera, so the inversion is mandatory.

bf16 autocast when CUDA capability ≥ 8, else fp16. `views` stays CPU-resident per upstream
`7685b7a` — that behaviour is precisely why we chose that fork.

The returned dict uses **the same keys every other backend emits** (`images`, `extrinsic`,
`intrinsics`, `intrinsics_downsampled`, `depth`, `depth_conf`). That is deliberate:
`_raw_to_world_points`, `_reproject`, and the BA wrapper then need zero LoGeR-specific
adaptation.

### `_postprocess(raw)`

Near-identical to Omega's: optional `compute_multiview_depth_confidence` mask →
`unproject_and_filter_points(...)` → `FeedforwardResult`. `depth` is stored as (N,H,W) —
trailing axis squeezed — to match the field contract.

#### Which cloud lands in `world_points`

All three wired backends populate the field: `vggtx.py:365` and `vggt_omega.py:278` derive it via
`_raw_to_world_points(raw_outputs, subsample=1)` inside a try/except that falls back to `None`;
`mapanything.py:461` uses `stacked_pts3d`. LoGeR has a second candidate the others lack — its
native `points`, free and already world-space.

They are not the same thing. Per the ray-field analysis above, the native cloud can encode
non-pinhole geometry that K-unprojection cannot reproduce, so it is potentially *more* faithful.
But `world_points` feeding BA and LC alongside a K that cannot reproduce it means BA spends its
first iterations fighting the model.

**Decision: use `_raw_to_world_points`, matching vggtx and vggt_omega.** Native `points` is
retained in the raw dict solely as the parity-test reference. Consistency wins because the
residual between the two is measurable, and the parity test measures precisely it:

- Residual small (≲0.5% of scene scale) → the model is effectively pinhole and the two clouds
  are interchangeable; consistency is free.
- Residual large → the model is meaningfully non-pinhole, which outranks this decision entirely,
  because the shared-K fit would then also be lossy for the mesh and BA paths. Revisit as its
  own piece of work.

The plan records the measured residual as a number rather than assuming which case holds.

### `_reproject(raw, extrinsics_3x4, intrinsics)`

Byte-for-byte VGGT-Omega's implementation, ~8 lines: delegate to `unproject_and_filter_points`
with the refined poses.

## Error handling

Fail loudly, no silent fallbacks — consistent with "hard imports, no stub backends".

| Condition | Where | Behaviour |
|---|---|---|
| `third_party/LoGeR` absent | `feedforward/__init__.py` | `ImportError` swallowed by the guarded export, so `loger` never enters `_REGISTRY` and `make_creator("loger")` raises unknown-backend. Same as Omega/SPARK. |
| `loop_closure` truthy with `backend: loger` | `_run_feedforward` | `ValueError`. LC thresholds are per-backbone and uncalibrated here — see below. |
| Frame sizes non-uniform | `_preprocess` | `ValueError`. LoGeR sizes from frame 0 only; refuse rather than silently mis-resize the rest. |
| Focal fit degenerate (empty conf mask, non-finite, `f <= 0`) | `_estimate_shared_focal` | `RuntimeError` naming frame count and mask survivors. **No `1.2 * max(W,H)` fallback** — see the intrinsics section. |
| RGB outside `[0, 1]` | `_forward` | `AssertionError`. Guards the `a157421` `[0,255]` bug class at the source. LoGeR's `ToTensor` gives `[0,1]`, matching VGGT-X, but this is asserted rather than assumed. |
| `extract_intermediate_features` called | creator | `NotImplementedError` naming LoGeR's windowed TTT memory. |

### Why no loop closure in the first cut

LoGeR's TTT plus windowed memory overlaps functionally with our `LoopClosure` wrapper, and every
backbone so far has needed its own retrieval-layer and threshold sweep: SPARK 0.95 native,
VGGT-X L10/1.17, Omega L13/1.55, MapAnything L4/1.46. Shipping uncalibrated thresholds would
produce quietly worse trajectories. Refuse instead. Calibration is a separate piece of work with
its own sweep.

### Multiview confidence belongs to an existing design, not this one

`use_multiview_confidence` is already a per-creator flag — MapAnything defaults `True`, VGGT-X
and VGGT-Omega default `False` — and `docs/superpowers/specs/2026-08-12-multiview-confidence-all-models-design.md`
(commit `cbaacc5`) already owns bringing every backbone to MapAnything's default-on standard,
including the calibration methodology.

So `LoGeRCreator` gets the flag with the same plumbing as the others and defaults it `False`,
matching VGGT-X and Omega today. Calibrating LoGeR's threshold is scoped to that spec, which
should name `loger` in its backbone list. Duplicating the sweep here would produce two
calibration procedures for one mechanism.

## Configuration

Backends are mutually exclusive — `pointcloud.backend` is one scalar, and selecting `loger`
replaces the backbone entirely.

```yaml
pointcloud:
  backend: loger    # vggt_omega | vggtx | mapanything | loger  (feedforward only)
  loger:            # per-backend creator kwargs; read only when backend matches
    variant: LoGeR_star
    window_size: 32
    overlap_size: 8
```

### Per-backend kwargs passthrough

`reconstructor.py:195` becomes:

```python
creator = creator_map[backend](max_points=max_points, **pc_cfg.get(backend, {}))
```

This is **generic, not LoGeR-specific** — every backend gains a config-file surface for its
constructor kwargs, which is more consistent than today, where a creator's tunables are
reachable only by constructing it in Python.

The nesting is the mechanism, not decoration. `pointcloud:` also holds pipeline-level keys —
`method`, `backend`, `loop_closure`, `max_points`, `export_max_points`, `viz`,
`bundle_adjustment`, `clean` — and a flat namespace would pass all of them to the creator
constructor and `TypeError`. Flattening would require a maintained exclusion list that silently
breaks whenever a pipeline-level key is added. Nesting separates "pipeline knob" from "model
knob" structurally. It also suits `base.yaml`'s stated role as "the SINGLE SOURCE OF DEFAULTS":
every backend's defaults can be documented there simultaneously, and switching `backend:` does
not require rewriting settings.

A flat `backend_args:` block was considered — no name duplication — but it can only describe the
active backend, so `base.yaml` loses the ability to document the others.

Two guards: `max_points` is already passed explicitly, so it must be rejected from the block
rather than producing a duplicate-kwarg `TypeError`; and unknown keys surface as the
constructor's own `TypeError`, which is the correct failure.

LoGeR's remaining knobs (`reset_every`, `pixel_limit`, `use_multiview_confidence`) are
`LoGeRCreator.__init__` parameters, defaulted from the vendored `original_config.yaml`, and
therefore also settable from the block.

### `preprocessing.max_frames`

`max_frames: 300` is a VGGT-Omega GPU property living in the preproc stage. Preproc runs first
and writes `frames.zarr`; pointcloud only reads it. So 300 is a hard ceiling: flip to
`backend: loger` at defaults and it silently processes 300 frames — exactly matching Omega — and
LoGeR appears to buy nothing. (That line's comment already says "cap at 200" beside a value of
300; stale, worth fixing while we are there.)

**The default stays 300.** Raising it now would encode a guess, and the binding constraint is
probably not LoGeR's:

> LoGeR's windowing bounds the **model**, not our result. `FeedforwardResult` holds `images`,
> `world_points`, `depth`, and `confidence`, all dense per-frame and all resident. At the 255k
> pixel budget that is roughly 8 MB/frame, so 1000 frames is about 8 GB in the dataclass before
> the zarr write, plus LoGeR's native `points`. Against a 46.6 GB cgroup that is the real
> ceiling, and it is *our* code. Same shape as the LC P7.3 result, where 1k frames hit 50/50 GB
> on resident fat-submap accumulation rather than model memory.

So: keep 300, add a warning, measure, then write the measured number down.

- **Warning** in `_run_feedforward`, ~3 lines: `backend == "loger"` and the store holds ≤ 300
  frames → `logger.warning` that LoGeR is windowed and this cap is Omega's, not its own. No
  behaviour change; kills the silent-underdelivery case.
- **Measurement task** in the plan: sweep frame count to OOM on our box, with `reset_every` on
  and off, recording peak RSS and VRAM. That number lands in `configs/README.md` and the
  `max_frames` comment.

**Rejected for this work — LoGeR's result stays dense and identical in shape to every other
backend's.** `LoGeRCreator` populates `world_points`, `images`, `depth`, and `confidence` just
as `vggtx`, `mapanything`, and `vggt_omega` do.

The alternative considered and rejected was a sparse result: drop `world_points` (recomputable
from `depth`, K, and pose) and stream frames to zarr rather than holding them. It would likely
move the ceiling substantially. It is rejected here for two reasons, the second decisive:

1. It touches `FeedforwardResult` and the shared base template, well past "minimal additions".
2. **Consistency.** Doing it for `loger` alone would make one backend's result a different
   shape from the other three, and every downstream consumer — the mesh path, the dashboard,
   the zarr readers, localization — would have to learn which backend produced the scene. If
   sparse results are worth having, they are worth having for all four backends, in their own
   design doc.

The ceiling is named here only so it is not mistaken for a LoGeR limitation. It is our result
dataclass, it affects every backend equally, and LoGeR is merely the first backend able to feed
it enough frames to matter.

## Testing

`tests/pointcloud/test_loger_creator.py`, flat functions, mirroring
`test_vggt_omega_creator.py`. Every fast test uses a fake model — no weights, no GPU, no network.

**Unit, against a synthetic pinhole scene:**

1. **Focal fit recovers a known focal.** Build `local_points` from a synthetic depth map and a
   known `f`; assert `_estimate_shared_focal` returns it within tolerance. Ground truth is exact
   here because no model is involved.
2. **Confidence weighting bites.** Corrupt 30% of points and give them low confidence; assert
   the fit still recovers `f`. Proves the weighting is not decorative.
3. **Degenerate input raises** rather than falling back: empty mask, NaN, negative depth.
4. **`_loger_original_coords`** emits `[0,0,w,h,w,h]` and round-trips through
   `_rescale_reconstruction_to_original_dimensions` to original-resolution K.
5. **c2w → w2c inversion.** Fake `camera_poses`; assert `result.extrinsics` is world-to-camera,
   not the raw model output. The single easiest thing to get backwards.
6. **`FeedforwardResult` field contract** — shapes and dtypes, `depth` is (N,H,W) not
   (N,H,W,1), colors uint8.
7. **Refusals**: `extract_intermediate_features` raises `NotImplementedError`; LC plus `loger`
   raises `ValueError`.
8. **Registry**: `"loger" in list_creators()` when the vendored tree is present.

**Integration — the parity check.** Marked `@pytest.mark.slow`, skipped without weights:

Unproject `depth` with the fitted K and `invert_poses(camera_poses)`, and compare against
LoGeR's native `points`. Assert median per-point error under a scene-scale-relative tolerance.

This one assertion simultaneously proves the focal fit, the pose inversion, the depth
extraction, and that `unproject_and_filter_points` is a legitimate reuse — because LoGeR itself
computes `points = camera_poses @ homogenize(local_points)`, so any of those four being wrong
breaks it. It is the reason reusing the existing unprojection is both the minimal choice and the
verifiable one: zero new production code carries the validation.

**Smoke test, first implementation task, gates everything else:** import `Pi3` under torch
2.5.1+cu121 and run one forward pass on 8 frames. LoGeR pins torch 2.6.0 and we run 2.5.1; the
versions are close but the gap is unverified, and nothing else in the plan is worth starting
until it passes.

## Open items carried into the plan

1. Torch 2.5.1 vs LoGeR's pinned 2.6.0 — smoke test gates the work.
2. The `max_frames` ceiling for `loger` is unmeasured. Sweep, then document.
3. **The pinhole residual is unmeasured.** The parity test must report it as a number. It
   decides whether the fitted-K approximation is tight, and if it is not, the finding affects
   the mesh and BA paths, not just `world_points`.
4. Loop closure calibration for the LoGeR backbone — separate work, refused until then.
5. Multiview confidence calibration — owned by
   `2026-08-12-multiview-confidence-all-models-design.md`; `loger` should be added to its scope.
6. Sparse `FeedforwardResult` for long sequences — rejected here for consistency; would need
   its own design covering all four backends.
7. Pi3* / `pi3x` — no drop-in path without Pi3*-trained weights. If such a checkpoint is
   published, the metric branch would give metric depth and is worth revisiting.
