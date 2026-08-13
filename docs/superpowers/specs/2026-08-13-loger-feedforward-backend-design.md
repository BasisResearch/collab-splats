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
- A **fitted** K cannot fail that way — it is centre-principal and square-pixel by construction,
  so that entire class of bug is unreachable. It fails differently: it can be
  plausibly-but-globally-wrong if the pointmap is scaled oddly or the confidence mask is
  unrepresentative. That is a *silent* failure, so it needs a numeric check, not a range guard.

### The fit

`_estimate_shared_focal(local_points, conf)`, ported from `run_loger.py`'s
`estimate_focal_lengths` / `_focal_from_frame` / `_snap_square_pixels`.

`local_points[..., 2]` **is** depth — the model literally constructs it as
`torch.cat([xy * z, z], dim=-1)` (`pi3.py:770,775`). So each valid pixel gives one equation of
the pinhole model, and a confidence-weighted least squares over all frames yields one focal.
Shared across frames is correct for our inputs: `frames.zarr` comes from a single video, one
physical camera, and the config exposes no zoom.

Roughly 40 lines. Justified because there is no alternative — the model emits no K, and we
reject the `1.2 * max(W, H)` heuristic (`localization/localizer.py:24 seed_intrinsics`) because a
wrong-but-plausible K against real depth is precisely the `be24be2` mesh-intrinsics regression.

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
| mod | `collab_splats/wrapper/reconstructor.py` | `_FEEDFORWARD_BACKENDS` (L43), `creator_map` (L189), LC refusal, `max_frames` warning |
| mod | `configs/base.yaml` | one word in one comment (see below) |
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
`finally`. Read `ckpts/{variant}/original_config.yaml` from the vendored tree. Filter its
`model:` keys against `inspect.signature(Pi3.__init__)` — the yaml carries training-only keys
`Pi3` will not accept — and re-parse `ttt_insert_after` / `attn_insert_after` when they arrive as
`"[4,8]"` strings. Weights via `hf_hub_download("Junyi42/LoGeR", f"{variant}/latest.pt")`,
`torch.load(map_location="cpu")`, unwrap `model_state_dict`, strip any `module.` prefix,
`load_state_dict(strict=True)`, `.eval().to(device)`. Stash forward kwargs from the yaml's
`training_settings`, overridden by constructor arguments.

### `_preprocess(frames, frame_idxs) -> (views, image_paths, original_coords)`

Synthetic `frame_{idx:06d}` labels, the VGGT-Omega convention — the frame store is the only IO
path and there are no real filenames. Target size from LoGeR's own rule:
`scale = sqrt(pixel_limit / (W * H))`, round each axis to a multiple of 14, shrink the longer
axis until under budget. PIL LANCZOS → `(N, 3, H, W)` float in `[0, 1]`, fed through
`frames_as_pil_source` so nothing touches disk.

LoGeR derives the target size from frame 0 alone. Rather than inherit that silent assumption we
assert frame-size uniformity and raise otherwise.

`original_coords` is where LoGeR is simpler than Omega: pure resize, no crop, so each row is
`[0, 0, orig_w, orig_h, orig_w, orig_h]` and `_rescale_reconstruction_to_original_dimensions`
scales K uniformly.

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
`unproject_and_filter_points(...)` → `FeedforwardResult`. One saving: `world_points` is LoGeR's
native `points`, not a recompute. `depth` is stored as (N,H,W) — trailing axis squeezed — to
match the field contract.

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

## Configuration

`configs/base.yaml` changes by **one word in one comment**:

```yaml
pointcloud:
  backend: vggt_omega   # vggt_omega | vggtx | mapanything | loger  (feedforward only)
```

No `pointcloud.loger:` block. No backend today has a per-backend settings block — all three are
constructed identically at `reconstructor.py:195` as
`creator_map[backend](max_points=max_points)`. Inventing that pattern for one backend is not
minimal. LoGeR's knobs (`variant`, `window_size`, `overlap_size`, `reset_every`, `pixel_limit`)
become `LoGeRCreator.__init__` defaults seeded from the vendored `original_config.yaml`,
reachable by anyone constructing the creator directly — the same status every other creator's
tunables have. If a knob later proves it needs run-level control, the block gets added then,
with a reason.

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
3. Loop closure calibration for the LoGeR backbone — separate work, refused until then.
4. Sparse `FeedforwardResult` for long sequences — rejected here for consistency; would need
   its own design covering all four backends.
