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

### The shipped yaml is authoritative

`variant` selects a config-plus-weights pair, not merely a weight file: the two configs differ
from each other (`ttt_pre_norm: true` on `LoGeR`, absent on `LoGeR_star`; `se3: true` on
`LoGeR_star` only), and both set `ttt_inter_multi: 4` where the `Pi3` constructor default is `2`.

So the vendored `original_config.yaml` is read and applied verbatim, and `Pi3` constructor
defaults are never relied on. Scope note: we run LoGeR as trained and shipped — no architecture
variants, no alternative backbones. **LoGeR depth is non-metric**, like every other backend we
run, so there is no `depth_trunc` win here.

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
| **LoGeR** | **nothing predicts it.** `CameraHead` returns pose only. K is **solved** by a confidence-weighted median pinhole fit to the predicted pointmap | **shared across frames** (our choice) |

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

The point head emits three channels, split and recombined (`pi3.py:772-775`):

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
`estimate_focal_lengths` / `_focal_from_frame` / `_weighted_median` / `_snap_square_pixels`.

**It is a weighted median, not a least-squares fit.** Each valid pixel yields its own focal
estimate directly by inverting the pinhole model — `fx_pp = uu * Z / X`, `fy_pp = vv * Z / Y`,
where `local_points[..., 2]` is `Z` per the construction above — and the estimate is the
confidence-weighted median over all of them. That is a deliberate robustness property: a median
is unmoved by the outliers a squared-error fit would chase, which matters because the ray field
is unconstrained at depth discontinuities.

Three behaviours to preserve, all in `_focal_from_frame`:

- validity mask `Z > 1e-3`, `|X| > 1e-6`, `|Y| > 1e-6`, `conf > 0.1` — the last is a hard
  threshold, not a weight
- sanity bounds `W*0.1 < fx < W*10` and `H*0.1 < fy < H*10` applied before the median
- `_weighted_median` subsamples above 50 000 samples with a seeded RNG (`default_rng(42)`), so
  the result is deterministic

**`fx` and `fy` are estimated independently** and merged by `_snap_square_pixels` only when
within 2% — load-bearing, see `_preprocess`.

Shared across frames is correct for our inputs: `frames.zarr` comes from a single video, one
physical camera, and the config exposes no zoom.

**Two branches of the original are dropped**, both because they are dead or wrong for us:

1. `estimate_focal_lengths(shared=False)` — the per-frame path. We always want shared, and a
   parameter that is always its default should not exist.
2. The `max(W, H) * 1.2` fallback on fit failure (the same heuristic as
   `localization/localizer.py:24 seed_intrinsics`). We raise instead: a wrong-but-plausible K
   against real depth is precisely the `be24be2` mesh-intrinsics regression.

Roughly 40 lines after those cuts. Justified because there is no alternative — the model emits
no K.

### When to choose LoGeR over the alternatives

**Choose it for:** sequences past the ~300-frame ceiling where VGGT-Omega OOMs; long captures
where drift accumulates, since the TTT memory is designed to carry state across the sequence.

**Avoid it for:** short sequences (<100 frames), where the set-based VGGT models see every frame
jointly and LoGeR's windowing buys nothing; anything needing loop closure (see below);
captures where intrinsics genuinely vary (zoom), which the shared-K fit cannot represent;
**unordered image collections** — LoGeR's windows are sequential, whereas the VGGT family is
set-based and has no ordering requirement.

## Footprint

**2 new files, 7 modified in the repo**, zero new pip deps, plus one environment step.

The environment step is the vendored tree: `third_party/LoGeR/` = upstream @ `7685b7a`. It is
gitignored (`.gitignore:7`), so it is a setup action, not a repo change; the pinned commit is
recorded in the tracked `third_party/README.md`.

| | Path | Why |
|---|---|---|
| new | `collab_splats/pointcloud/feedforward/loger.py` | `LoGeRCreator` + one module-level helper |
| new | `tests/pointcloud/test_loger_creator.py` | mirrors `test_vggt_omega_creator.py` |
| mod | `collab_splats/pointcloud/feedforward/__init__.py` | guarded export, `try/except ImportError`, same shape as `VGGTOmegaCreator` |
| mod | `collab_splats/pointcloud/__init__.py` | `_LOGER_AVAILABLE` guard + `_REGISTRY["loger"]`, mirroring `_OMEGA_AVAILABLE` at lines 8–28 |
| mod | `collab_splats/wrapper/reconstructor.py` | `_FEEDFORWARD_BACKENDS` (L43), `creator_map` (L189), per-backend kwargs passthrough (L195), LC refusal, `max_frames` warning |
| mod | `configs/base.yaml` | backend comment + `pointcloud.loger:` kwargs block (see below) |
| mod | `configs/README.md` | backend row, intrinsics note, when-to-choose, `max_frames` guidance |
| mod | `docs/source/api/pointcloud.rst` | one `automodule` stanza for `feedforward.loger` |
| mod | `docs/source/conf.py` | `"loger"` added to `autodoc_mock_imports` (line 37) — the vendored package is not importable on a docs build |

### Reused unchanged — no new code

`unproject_and_filter_points` (`vggtx.py:92`), `invert_poses` (`geometry/transforms.py`),
`build_pycolmap_reconstruction`, `_rescale_reconstruction_to_original_dimensions`,
`frames_as_pil_source`, `FeedforwardResult` and its zarr IO, `_raw_to_world_points`,
`compute_multiview_depth_confidence`, and the whole 5-step template.

### Genuinely new, and why

1. `_estimate_shared_focal(local_points, conf)` — justified above. ~40 lines.
2. `LoGeRCreator` itself.

**No `_loger_original_coords` helper.** An earlier draft proposed one. It is unnecessary: LoGeR
crops nothing, so every row is `[0, 0, orig_w, orig_h, orig_w, orig_h]` and the whole thing is
two lines inline in `_preprocess`. A named function for `np.tile` of a constant row is the kind
of premature abstraction the code style rules out. (Contrast `_compute_omega_original_coords`,
which exists because Omega's crop arithmetic is genuinely non-trivial.)

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

**`se3` lives in the `model:` block but is a forward kwarg.** `LoGeR_star` sets `se3: true`
under `model:`, yet `se3` is not a `Pi3.__init__` parameter — it is popped inside `forward`
(`pi3.py:589`, mutually exclusive with `sim3`). A naive `inspect.signature(Pi3.__init__)` filter
would silently discard it and run LoGeR* in the wrong alignment mode.

Verified against `run_loger.py`'s `build_forward_kwargs`, `se3` is the **only** such intruder —
the window knobs (`window_size`, `overlap_size`, `reset_every`, `num_iterations`) come from the
yaml's `training_settings` block, not `model:`. So the rule is deliberately narrow rather than a
maintained allowlist that would drift against upstream:

- pull `se3` out of `model:` explicitly, route it to forward kwargs
- every remaining `model:` key must be in `inspect.signature(Pi3.__init__)`, else **raise**,
  naming the key

Raising rather than dropping is the point: a silent drop is exactly how LoGeR* would degrade
invisibly, and a future upstream config gaining a second forward-only key should stop the run,
not be guessed at.

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
| — | — | `intrinsics` | fitted K broadcast to (N,3,3) |
| — | — | `intrinsics_downsampled` | alias to the same K — `_raw_to_world_points` expects the key (`vggtx.py:304`) |

LoGeR's native `points` is **discarded**, not retained. An earlier draft kept it as a
parity-test reference, which contradicts this spec's own memory analysis: another (N,H,W,3)
float32 array is ~3 MB/frame resident for the entire run, purely to serve a `slow`-marked test
that is skipped by default. The parity test runs the model itself and reads `points` there.

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

**Decision: use `_raw_to_world_points`, matching vggtx and vggt_omega.** Consistency wins
because the residual between the two clouds is measurable, and the parity test measures
precisely it:

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

LoGeR's remaining knobs are `LoGeRCreator.__init__` parameters and therefore also settable from
the block. Their defaults come from three different places, which the implementation must not
conflate: `window_size` / `overlap_size` / `reset_every` / `num_iterations` from the vendored
`original_config.yaml`'s `training_settings`; `pixel_limit = 255000` from `run_loger.py`'s
function default; `use_multiview_confidence = False` is ours, matching VGGT-X and Omega, and
appears in no LoGeR config at all.

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
2. **Anisotropic focal is recovered, not flattened.** Build a scene with `fx/fy = 1.10` — well
   past `_snap_square_pixels`' 2% tolerance — and assert both come back distinct. This is the
   test that protects the aspect-ratio argument under `_preprocess`; a "simplification" to
   `fx == fy` must fail loudly here.
3. **The median is robust.** Corrupt 30% of points *with high confidence* and assert the
   estimate still holds. A weighted median survives this; a least-squares fit would not. Written
   against median semantics deliberately — the estimator is a weighted median, not a fit.
4. **Degenerate input raises** rather than falling back: empty mask, NaN, negative depth.
5. **`original_coords`** is `[0,0,w,h,w,h]` per frame and round-trips through
   `_rescale_reconstruction_to_original_dimensions` to original-resolution K — with `fx != fy`
   preserved, since that function must scale x and y separately.
6. **c2w → w2c inversion.** Fake `camera_poses`; assert `result.extrinsics` is world-to-camera,
   not the raw model output. The single easiest thing to get backwards.
7. **`FeedforwardResult` field contract** — shapes and dtypes, `depth` is (N,H,W) not
   (N,H,W,1), colors uint8.
8. **Refusals**: `extract_intermediate_features` raises `NotImplementedError`; LC plus `loger`
   raises `ValueError`; a `model:` key that is neither `se3` nor a `Pi3.__init__` parameter
   raises.
9. **Registry**: `get_creator("loger")` returns `LoGeRCreator` when the vendored tree is
   present. (Note there is no `list_creators()` — `pointcloud/__init__.py` exposes only
   `get_creator` and `make_creator`.)

**Integration — the parity check.** Marked `@pytest.mark.slow`, skipped without weights:

Run the model, then unproject `depth` with the fitted K and `invert_poses(camera_poses)` and
compare against LoGeR's native `points`, read from the model output inside the test. Assert
median per-point error under a scene-scale-relative tolerance, and **log the residual** — it is
open item 3, not merely a pass/fail.

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
