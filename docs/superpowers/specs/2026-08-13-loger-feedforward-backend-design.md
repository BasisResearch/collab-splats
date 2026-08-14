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
`VGGTSparkCreator._load_model` (`feedforward/vggt_spark_creator.py`).

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

- A **regressed** K can be geometrically invalid. `VGGTOmegaCreator._postprocess`
  (`feedforward/vggt_omega.py`) carries a comment naming
  exactly this: it is the "root cause of `cx > model_W` in `result.intrinsics`". The network is
  free to emit a principal point outside the image.
- A **fitted** K is centre-principal by construction, so `cx`/`cy` cannot land outside the
  image and that class of bug is unreachable. It fails differently: it can be
  plausibly-but-globally-wrong if the pointmap is scaled oddly or the confidence mask is
  unrepresentative. That is a *silent* failure, so it needs a numeric check, not a range guard.

Note it is **not** square-pixel: `fx` and `fy` are estimated independently and kept that way.
That matters — see the aspect-ratio discussion under `_preprocess`.

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

`_estimate_shared_intrinsics(local_points, conf) -> np.ndarray` returns a `(3, 3)` `float32` K.
It is **ported**, and from the fork we do *not* vendor — so the attribution has to be exact.
Source: PolyCam `LoGeR` @ `5d7c1a7`, `run_loger.py`, four functions collapsed into one:

| Upstream | Line | Fate |
|---|---|---|
| `estimate_focal_lengths(local_points, conf, shared=True)` | 206 | merged; `shared=False` branch and the `1.2·max(W,H)` fallback dropped |
| `_focal_from_frame(local_pts, conf_frame, uu, vv, H, W)` | 180 | **inlined** — single call site once `shared=False` is gone; the per-pixel inversion and all three guards preserved |
| `_weighted_median(values, w, max_n=50000)` | 167 | kept as the one helper, verbatim |
| `_snap_square_pixels(fx, fy, tol=0.02)` | 195 | **not ported** — harmful at model resolution, see below |

**Why it returns K and not `(fx, fy)`.** The principal point is not a separate decision the
caller gets to make — it is *already inside the estimator*. The pixel grid is built centred,
`u_centered = np.arange(W, dtype=np.float32) - (W - 1) / 2.0` (github.com/PolyCam/LoGeR @
5d7c1a7, `run_loger.py:228-229`), so every per-pixel focal
`uu * Z / X` is conditioned on `cx = (W - 1) / 2`. Returning only the focals splits one
calibration across two places and obliges the caller to independently rediscover that
convention. A caller that reasonably writes `W / 2.0` instead introduces a half-pixel principal
offset that nothing in the pipeline would flag — it is within the noise of the parity test and
would surface only as a small systematic reprojection bias in localization. Returning the
assembled matrix makes the convention unstatable-wrongly: the estimator owns the grid, so the
estimator owns `cx`, `cy`. The caller's remaining job is one broadcast to `(N, 3, 3)`, since the
fit is shared across frames.

`build_forward_kwargs` (L149) is **read for its behaviour, not ported** — the forward kwargs come
from the vendored yaml directly.

**The resize rule is cited to the vendored tree, not to PolyCam.** github.com/PolyCam/LoGeR @
5d7c1a7, `run_loger.py:117` (`load_images`) implements it, but so does
github.com/Junyi42/LoGeR @ 7685b7a, `loger/utils/basic.py:55-61` inside the package we
actually ship (`load_images_as_tensor`, signature at `basic.py:11`), with identical arithmetic:
`scale = sqrt(PIXEL_LIMIT / (W*H))`, independent `round(·/14)` per axis, then the shrink loop.
Porting ~10 lines out of a file we vendor is better provenance than copying the same lines from
a fork we discard.

**Neither repository ships a LICENSE file** — checked both clones. So the ported helper carries a
docstring naming the source repo, commit, file, and the four functions, and `third_party/README.md`
records the same. The absence is worth a flag to the user before this ships, since we are copying
~40 lines rather than only calling a vendored tree. The rest of `run_loger.py` — CLI, PLY export,
disk IO — is discarded.

**It is a weighted median, not a least-squares fit.** Each valid pixel yields its own focal
estimate directly by inverting the pinhole model — `fx_pp = uu * Z / X`, `fy_pp = vv * Z / Y`,
where `local_points[..., 2]` is `Z` per the construction above — and the estimate is the
confidence-weighted median over all of them. That is a deliberate robustness property: a median
is unmoved by the outliers a squared-error fit would chase, which matters because the ray field
is unconstrained at depth discontinuities.

Three behaviours to preserve, all in `_focal_from_frame`:

- validity mask `Z > 1e-3`, `|X| > 1e-6`, `|Y| > 1e-6`, `conf > 0.1` — the last is a hard
  threshold, not a weight, **and it is a threshold on sigmoid-activated confidence** (below)
- sanity bounds `W*0.1 < fx < W*10` and `H*0.1 < fy < H*10` applied before the median
- `_weighted_median` subsamples above 50 000 samples with a seeded RNG (`default_rng(42)`), so
  the result is deterministic

**The sigmoid is the caller's job, and the estimator depends on it having run.** LoGeR's
`conf_head` is a bare `LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)`
(github.com/Junyi42/LoGeR @ 7685b7a, `loger/models/pi3.py:172`) with **no output activation** —
the model emits logits. Upstream applies the activation outside the model, at
github.com/PolyCam/LoGeR @ 5d7c1a7, `run_loger.py:481`:
`preds["conf"] = torch.sigmoid(preds["conf"])`. So `_forward` must call `torch.sigmoid` on the
raw head output before anything else touches it. Skipping it is not a scaling nuisance that
washes out downstream: the `conf > 0.1` gate is calibrated against a probability, and on raw
logits `> 0.1` admits roughly half of all pixels instead of a deliberate ~10% floor, quietly
feeding the median the low-confidence pixels the gate exists to exclude. The fit would still
return a plausible number, which is what makes it worth naming here rather than leaving to be
inferred from upstream. Recorded as a hard ordering requirement in the `_forward` table.

**`fx` and `fy` are estimated independently and stay independent.** Load-bearing — see
`_preprocess`.

Shared across frames is correct for our inputs: `frames.zarr` comes from a single video, one
physical camera, and the config exposes no zoom.

#### What we drop, and what we keep

The four upstream functions total ~78 lines. Three cuts bring it to **two functions, ~35 lines**:

1. **`estimate_focal_lengths(shared=False)`** — the per-frame branch. We always want shared, and
   a parameter that is always its default should not exist.
2. **The `max(W, H) * 1.2` fallback** on fit failure (the same heuristic as
   `seed_intrinsics` in `collab_splats/localization/localizer.py`). We raise instead: a wrong-but-plausible K
   against real depth is precisely the `be24be2` mesh-intrinsics regression.
3. **`_snap_square_pixels` entirely** (10 lines) — and the reason is stronger than an earlier
   draft claimed. That draft said no consumer observes it. In fact, **at model resolution it
   injects error**, precisely in the band where it fires.

   Take a physical camera with square pixels, `fx_orig = fy_orig = f`. Our resize scales the
   axes by *different* factors `s_x ≠ s_y` (independent `round(·/14)`), so the true
   model-resolution focals are `f·s_x` and `f·s_y` — genuinely unequal, by the aspect
   distortion, typically a few percent. The estimator recovers both correctly. Then:

   - **snapped:** both become `f(s_x + s_y)/2`, and
     `_rescale_reconstruction_to_original_dimensions` divides by `s_x` and `s_y` separately,
     giving `f(s_x + s_y)/(2 s_x) ≠ f`. Residual error ≈ `(s_y/s_x − 1)/2`, up to ~1%.
   - **not snapped:** `f·s_x / s_x = f`, exact.

   The 2% snap tolerance is the same order as the anisotropy the resize introduces, so the snap
   fires exactly when it destroys a correction that was right. It makes sense upstream, where
   `run_loger.py` exports at model resolution and never rescales. It does not survive contact
   with our rescale step.

   Deleting it also makes the anisotropy argument categorical rather than conditional: `fx` and
   `fy` are *never* merged.

   **There is a version of the idea that is useful**, and it is worth recording rather than
   losing. If the camera really does have square pixels, then `fx_m/s_x` and `fy_m/s_y` are two
   independent estimates of the same scalar `f`, so averaging them halves the variance. The
   snap belongs in **original-resolution space, after the scales are divided out** — not at
   model resolution. Left out of the first cut: it would need `s_x`/`s_y` threaded into the
   estimator (or a hook after `_rescale_...`, which is base-class code we do not own), and the
   gain is only worth having if the estimator's own variance exceeds the ~1% it would trade
   away. That is measurable from the parity test, so it becomes a follow-up with a number
   behind it rather than a guess.

With `shared=False` gone, **`_focal_from_frame` is called exactly once**, so it is inlined — a
six-parameter helper with a single call site is a function boundary earning nothing.

**`_weighted_median` stays, subsampling included**, and that is a considered keep rather than an
oversight. The pooled sample count is `H*W*N`: at the 255k pixel budget and 300 frames that is
76.5M values, and a weighted median needs a full `argsort` — roughly 600 MB of workspace and
seconds of wall-clock. At the 1000-frame sequences this backend exists to enable it is 255M.
The 50k cap is a memory guard at exactly the frame counts LoGeR is for, and the seeded RNG is
what keeps it reproducible.

Justified because there is no alternative — the model emits no K.

### When to choose LoGeR over the alternatives

**Choose it for:** sequences past the ~300-frame ceiling where VGGT-Omega OOMs; long captures
where drift accumulates, since the TTT memory is designed to carry state across the sequence.

**Avoid it for:** short sequences (<100 frames), where the set-based VGGT models see every frame
jointly and LoGeR's windowing buys nothing; anything needing loop closure (see below);
captures where intrinsics genuinely vary (zoom), which the shared-K fit cannot represent;
**unordered image collections** — LoGeR's windows are sequential, whereas the VGGT family is
set-based and has no ordering requirement.

## Footprint

**3 new files, 8 modified**, zero new pip deps.

The vendored tree `third_party/LoGeR/` (upstream @ `7685b7a`) is itself gitignored
(`.gitignore:7`), but it does not arrive by hand. `third_party/README.md` states the policy:
"Setup scripts own the clone + patch lifecycle" and "New entries: add a row to the table below
and wire up in the relevant setup script." So the clone is a tracked script, not an instruction
in a doc.

| | Path | Why |
|---|---|---|
| new | `collab_splats/pointcloud/feedforward/loger.py` | `LoGeRCreator` + one module-level helper. Bare-name file, matching `vggtx.py` / `vggt_omega.py` / `mapanything.py` — not the `vggt_spark_creator.py` outlier |
| new | `tests/pointcloud/test_loger_creator.py` | mirrors `test_vggt_omega_creator.py` |
| new | `setup/loger.sh` | clones upstream @ `7685b7a` into `third_party/LoGeR/`, idempotent `if [ ! -d ... ]` guard following `setup/vggt_slam.sh`. Its own script because `setup/feedforward.sh` — named in `third_party/README.md` — does not exist |
| mod | `third_party/README.md` | table row: path, upstream, populated-by, used-by; plus the pinned commit and the missing-LICENSE note |
| mod | `collab_splats/pointcloud/feedforward/__init__.py` | guarded export, `try/except ImportError`, same shape as `VGGTOmegaCreator` |
| mod | `collab_splats/pointcloud/__init__.py` | `_LOGER_AVAILABLE` guard + `_REGISTRY["loger"]`, mirroring `_OMEGA_AVAILABLE` at lines 8–28 |
| mod | `collab_splats/wrapper/reconstructor.py` | `_FEEDFORWARD_BACKENDS` (L43), `creator_map` (L189), per-backend kwargs passthrough (L195), LC refusal, `max_frames` warning |
| mod | `configs/base.yaml` | backend comment + `pointcloud.loger:` kwargs block (see below) |
| mod | `configs/README.md` | backend row, intrinsics note, when-to-choose, `max_frames` guidance |
| mod | `docs/source/api/pointcloud.rst` | one `automodule` stanza for `feedforward.loger` |

**No `docs/source/conf.py` change.** An earlier draft added `"loger"` to `autodoc_mock_imports`.
Unnecessary: the vendored import happens *inside* `_load_model` behind the `sys.path` insert, so
autodoc never imports `loger` at all. The list needs `vggt` and `mapanything` precisely because
`vggtx.py` and `mapanything.py` import them at module level. Following SPARK's deferred-import
pattern buys the docs build for free.

### Reused unchanged — no new code

Every one of these is called, not copied. Sources given so the plan cannot silently
re-implement one:

| Function | Source | Note |
|---|---|---|
| `unproject_and_filter_points` | `feedforward/vggtx.py` | cross-backend import; `vggt_omega.py` already does exactly `from .vggtx import unproject_and_filter_points`, so this is established precedent, not a new coupling |
| `_raw_to_world_points` | `feedforward/base.py` | called with `subsample=1`; the signature default is `8` |
| `compute_multiview_depth_confidence` | `feedforward/base.py` | |
| `build_pycolmap_reconstruction` | `feedforward/base.py` | invoked by the base template, not by us |
| `_rescale_reconstruction_to_original_dimensions` | `feedforward/base.py` | |
| `invert_poses` | `geometry/transforms.py` | |
| `FeedforwardResult` + its zarr IO | `feedforward/base.py` | |

**Not reused, deliberately — two:**

- `_decode_verify_geometry` (in `feedforward/base.py`). VGGT-Omega imports it, so copying Omega's import
  block wholesale would pull it in — but it exists only to serve `_verify_loop_candidate`, and
  LoGeR refuses loop closure.
- **`frames_as_pil_source` (in `feedforward/base.py`)**, which an earlier draft listed as reuse. It does not
  fit, and the reason is worth stating so nobody re-adds it. That helper monkeypatches the
  process-global `PIL.Image.open` in order to drive a **path-based** upstream loader in memory —
  that is why VGGT-X, Omega, and MapAnything need it. LoGeR's loader,
  `load_images_as_tensor` (`loger/utils/basic.py:11`), takes a **directory or an .mp4** and
  enumerates it with `os.listdir`; patching `Image.open` cannot drive it, and running it as
  written would require real files on disk, breaking the rule that the frame store is the sole
  IO path.

  So `_preprocess` resizes the decoded frames with PIL directly: compute the target size, then
  `PIL.Image.fromarray(f).resize(..., LANCZOS)` per frame, stack, scale to `[0, 1]`. About six
  lines, and it drops a global-state monkeypatch and its single-threaded caveat rather than
  adding them. This is a simplification, not a compromise.

### Genuinely new, and why

1. `_estimate_shared_intrinsics(local_points, conf)` plus `_weighted_median` — justified above.
   Two functions, ~35 lines, down from the four functions / ~78 lines upstream.
2. `LoGeRCreator` itself.

`loger.py` follows the sibling modules' shape, which is a house style rather than an accident:
a module docstring opening with a `Provides:` block that lists the public names (see
`feedforward/vggt_omega.py`'s module docstring), `########` section dividers between constants / inference utilities /
creator, module-level `_LOGER_*` constants for the repo id and defaults, and `logger =
logging.getLogger(__name__)`.

**No `_loger_original_coords` helper.** An earlier draft proposed one. It is unnecessary: LoGeR
crops nothing, so every row is `[0, 0, orig_w, orig_h, orig_w, orig_h]` and the whole thing is
two lines inline in `_preprocess`. A named function for `np.tile` of a constant row is the kind
of premature abstraction the code style rules out. (Contrast `_compute_omega_original_coords`,
which exists because Omega's crop arithmetic is genuinely non-trivial.)

### Deliberately not implemented

- `extract_intermediate_features` raises `NotImplementedError` naming LoGeR's native windowed
  TTT memory as the reason.
- No loop closure in the first cut.

### One reduction considered and rejected

`_reproject` is byte-identical between `vggtx.py` and `vggt_omega.py` — both are a docstring
plus one `unproject_and_filter_points` call — and LoGeR's would be a third copy. Hoisting a
concrete default onto `BaseFeedforwardCreator` would delete ~30 lines across the tree, since
only MapAnything genuinely differs (it consumes `list[dict]` camera-frame points and would keep
its override).

**Not doing it here.** It converts an abstract method to concrete on the shared base and edits
two shipping backends' BA path, for zero LoGeR benefit — regression risk in exchange for line
count, on a spec whose whole premise is minimal additions. Worth its own small refactor later;
recorded so the duplication is a known choice rather than an oversight.

### Pre-existing gaps, noted not fixed

- `vggt_spark` is in `_REGISTRY` (`collab_splats/pointcloud/__init__.py`) but absent from
  `_FEEDFORWARD_BACKENDS` and `creator_map` (both in `collab_splats/wrapper/reconstructor.py`,
  the latter inside `_run_feedforward`), so it is unreachable from `Reconstructor`. `loger` is
  wired into both, since the requirement is that it work like any other backend.
- Neither `vggt_omega` nor `vggt_spark` appears in `docs/source/api/pointcloud.rst`, which
  documents only `base`, `sfm`, `feedforward.base`, `vggtx`, and `mapanything`. Optional
  vendored backends have simply never been added. `loger` gets its stanza; back-filling the
  other two is a separate, unrelated change.

## Data flow: the template methods

`BaseFeedforwardCreator` (`collab_splats/pointcloud/feedforward/base.py`) declares **six**
`@abstractmethod`s, not five: `_load_model`, `_preprocess`, `_forward`, `_postprocess`,
`extract_intermediate_features`, and `_reproject`. This matters twice over: LoGeR's `NotImplementedError` for
`extract_intermediate_features` is the ABC contract being satisfied, not a courtesy stub — the
class will not instantiate without it; and the base class docstring's own "5-step pipeline /
four abstract methods" wording is stale. Pre-existing, out of scope, noted so this spec does not
inherit the error.

### `_load_model(device)`

Follow VGGT-SPARK's `sys.path` idiom exactly (the module-level root constant plus the
try/finally in `VGGTSparkCreator._load_model`, `feedforward/vggt_spark_creator.py`), not a
looser version of it:

- module-level constant `_LOGER_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "LoGeR"`
- inside `_load_model`, a `_patched = _LOGER_ROOT not in sys.path` flag guards the insert, and
  the `finally` removes the path **only if this call added it**

The flag is the part worth copying. Without it, a re-entrant or nested load pops a path the
caller installed. Import is `from loger.models.pi3 import Pi3` with `# noqa: PLC0415`, matching
SPARK.

Read `ckpts/{variant}/original_config.yaml` from the vendored tree.

**`se3` lives in the `model:` block but is a forward kwarg.** `LoGeR_star` sets `se3: true`
under `model:`, yet `se3` is not a `Pi3.__init__` parameter — it is popped inside `forward`
(github.com/Junyi42/LoGeR @ 7685b7a, `pi3.py:589`, mutually exclusive with `sim3`). A naive
`inspect.signature(Pi3.__init__)` filter would silently discard it and run LoGeR* in the wrong
alignment mode.

Verified against `build_forward_kwargs` in github.com/PolyCam/LoGeR @ 5d7c1a7, `run_loger.py`,
`se3` is the **only** such intruder —
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
path and there are no real filenames. Target size from LoGeR's own rule
(github.com/Junyi42/LoGeR @ 7685b7a, `loger/utils/basic.py:55-61`):
`scale = sqrt(pixel_limit / (W * H))`, round each axis to a multiple of 14, shrink the longer
axis until under budget. Then PIL LANCZOS per frame → `(N, 3, H, W)` float in `[0, 1]`. No
monkeypatch — see the `frames_as_pil_source` note above.

LoGeR derives the target size from frame 0 alone. Rather than inherit that silent assumption we
assert frame-size uniformity and raise otherwise.

`original_coords` is where LoGeR is simpler than Omega: pure resize, no crop, so each row is
`[0, 0, orig_w, orig_h, orig_w, orig_h]`.

#### The resize is mildly anisotropic, and that is fine — conditionally

`k = round(W_t/14)` and `m = round(H_t/14)` round **independently**, and the shrink loop can
remove a whole 14 px from one axis. So aspect ratio is not exactly preserved; the image is
stretched by up to a few percent on one axis.

This does not distort the pointcloud, but only because **three** separate places all keep `fx`
and `fy` distinct. Every one is a place where an isotropic "simplification" would bake the error
in:

1. `_estimate_shared_intrinsics` fits `fx` and `fy` **separately** and never merges them, so the
   anisotropy is absorbed into the K rather than corrupting the geometry. (Upstream's
   `_snap_square_pixels` would have merged them under 2%; it is not ported — see the fit
   section.)
2. `_rescale_reconstruction_to_original_dimensions` scales with separate `scale_x` and `scale_y`,
   so the original-resolution K recovers the true aspect.
3. **`LoGeRCreator.camera_model` must be `"PINHOLE"`.** This one is easy to miss and would be
   silent. `build_pycolmap_reconstruction` (in `feedforward/base.py`) branches on it:

   ```python
   if camera_model == "PINHOLE":
       params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
   else:  # SIMPLE_PINHOLE
       params = [(K[0, 0] + K[1, 1]) / 2.0, K[0, 2], K[1, 2]]
   ```

   `SIMPLE_PINHOLE` averages `fx` and `fy` away on export — discarding exactly the anisotropy
   points 1 and 2 worked to preserve, with no error and no warning. And this is not a
   hypothetical: `VGGTXCreator` (`feedforward/vggtx.py`) sets `camera_model = "SIMPLE_PINHOLE"`
   today. Copying VGGT-X's class body would inherit it. `BaseFeedforwardCreator`
   (`feedforward/base.py`) defaults to `"PINHOLE"` and `VGGTOmegaCreator` sets it explicitly;
   LoGeR sets it explicitly too, with a comment pointing at this paragraph.

The model is also trained with this exact preprocessing, so the stretch is in-distribution.

Cropping to a multiple of 14 instead would be worse: it discards field of view, and
`original_coords` would need Omega's crop arithmetic back.

### `_forward(model, views)`

| LoGeR output | shape | our key | transform |
|---|---|---|---|
| `local_points[..., 2:3]` | (N,H,W,1) | `depth` | already depth; model builds `cat([xy*z, z])` |
| `conf` | (N,H,W,1) | `depth_conf` | **`torch.sigmoid` first** (head emits logits, `pi3.py:172`), then squeeze trailing axis |
| `camera_poses` | (N,4,4) **c2w** | `extrinsic` | `invert_poses(...)[:, :3, :]` → w2c (N,3,4) |
| — | — | `intrinsics` | `_estimate_shared_intrinsics(...)` → (3,3), broadcast to (N,3,3) |
| — | — | `intrinsics_downsampled` | alias to the same K — `_raw_to_world_points` expects the key (as `VGGTXCreator._postprocess` in `feedforward/vggtx.py` also supplies it) |

Ordering is load-bearing: sigmoid runs **before** the K fit, because the fit's `conf > 0.1` gate
is a threshold on a probability.

#### `conf_threshold` means a percentile here, not a value

`unproject_and_filter_points` overloads its threshold argument (in `feedforward/vggtx.py`):

```python
if conf_threshold > 1.0:
    threshold_val = float(np.percentile(depth_conf, conf_threshold))
else:
    threshold_val = float(conf_threshold)
```

`> 1.0` is a **percentile**; `<= 1.0` is a **raw confidence value**. VGGT's confidence is
exponential and unbounded above, so its backends pass percentiles (`VGGTXCreator.conf_threshold`
= 35.0, `VGGTOmegaCreator.conf_threshold` = 50.0) and never touch the raw branch. LoGeR's
confidence is sigmoid, so it
lands in `[0, 1]` and *both* branches are now reachable in a way they were not before — a raw
threshold is meaningful for the first time.

`LoGeRCreator.conf_threshold` is a **percentile**, default `50.0`, matching `vggt_omega`.
(Upstream's own PLY export uses `--conf_percentile 20.0`, github.com/PolyCam/LoGeR @ 5d7c1a7,
`run_loger.py:54` — a looser cut for a
visualisation dump. Sibling consistency wins here; the parity task records what the residual
looks like at both.) The
percentile branch is scale-free, so it transfers across the confidence-distribution change
without recalibration, whereas a raw value tuned on sigmoid confidence would be a new number with
no evidence behind it. The class docstring says which convention the default uses, because
somebody will otherwise read `50.0` as a confidence and lower it to `0.5` — which is a legal
value that silently switches semantics from "drop the bottom half" to "drop everything below
0.5", a very different mask.

Note the resulting asymmetry inside this one backend, since it is genuinely confusing on a first
read: the K fit gates on a **raw** `conf > 0.1` (upstream's, preserved verbatim), while the
pointcloud mask gates on a **percentile**. They are different thresholds serving different jobs —
the first is a fixed floor on what may inform a calibration, the second is a density knob on the
exported cloud — and both are correct as written. Called out in an inline comment at the fit's
mask so the two are not "unified" by a later reader.

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

All three wired backends populate the field: `VGGTXCreator._postprocess` and
`VGGTOmegaCreator._postprocess` derive it via
`_raw_to_world_points(raw_outputs, subsample=1)` inside a try/except that falls back to `None`;
`MapAnythingCreator._postprocess` uses `stacked_pts3d`. LoGeR has a second candidate the others lack — its
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

Byte-for-byte VGGT-Omega's implementation (`VGGTOmegaCreator._reproject`, ~15 lines): delegate to
`unproject_and_filter_points` with the refined extrinsics and intrinsics, returning
`(pts3d, colors)` and dropping `pixel_indices`.

## Error handling

Fail loudly, no silent fallbacks — consistent with "hard imports, no stub backends".

| Condition | Where | Behaviour |
|---|---|---|
| `third_party/LoGeR` absent | `feedforward/__init__.py` | `ImportError` swallowed by the guarded export, so `loger` never enters `_REGISTRY` and `make_creator("loger")` raises unknown-backend. Same as Omega/SPARK. |
| `loop_closure` truthy with `backend: loger` | `_run_feedforward` | `ValueError`. LC thresholds are per-backbone and uncalibrated here — see below. |
| Frame sizes non-uniform | `_preprocess` | `ValueError`. LoGeR sizes from frame 0 only; refuse rather than silently mis-resize the rest. |
| `frame_idxs` not strictly ascending | `_preprocess` | `ValueError`. LoGeR's windows and overlap stitching assume temporal order; out-of-order input degrades quality with no error. `frames.zarr` is ordered by construction today, so this is a one-line assert protecting an assumption the other backends do not make. |
| Focal fit degenerate (empty conf mask, non-finite, `f <= 0`) | `_estimate_shared_intrinsics` | `RuntimeError` naming frame count and mask survivors. **No `1.2 * max(W,H)` fallback** — see the intrinsics section. |
| RGB outside `[0, 1]` | `_forward` | `AssertionError`. Guards the `a157421` `[0,255]` bug class at the source. LoGeR's `ToTensor` gives `[0,1]`, matching VGGT-X, but this is asserted rather than assumed. |
| `extract_intermediate_features` called | creator | `NotImplementedError` naming LoGeR's windowed TTT memory. |

### Why no loop closure in the first cut

LoGeR's TTT plus windowed memory overlaps functionally with our `LoopClosure` wrapper, and every
backbone so far has needed its own retrieval-layer and threshold sweep: SPARK 0.95 native,
VGGT-X L10/1.17, Omega L13/1.55, MapAnything L4/1.46. Shipping uncalibrated thresholds would
produce quietly worse trajectories. Refuse instead. Calibration is a separate piece of work with
its own sweep.

**The refusal has to be at the `Reconstructor` level, and that is not arbitrary.**
`_verify_loop_candidate` is **concrete on the base class** (in `feedforward/base.py`), not abstract, so
`LoGeRCreator` inherits a working one — and it calls `extract_intermediate_features`, which
LoGeR raises from. Without the up-front `ValueError`, enabling LC on `loger` would not fail at
the config boundary; it would run preprocessing and the full forward pass, then die with a bare
`NotImplementedError` from inside the LC loop. Same outcome, far worse diagnostics, minutes of
GPU time later. The guard converts that into an immediate, named refusal.

LoGeR therefore also skips the LC `ClassVar` calibration block that `VGGTOmegaCreator`
carries (`_lc_layer_index`, `default_verify_match_ratio`). Inventing values for a refused
feature would be dead configuration.

**Decided 2026-08-14 (user): the refusal is the final state, not a first cut.** "we dont need
loop closure for this". The LC calibration sweep is therefore **not owed** — it has been removed
from the owed list, and the section title's "in the first cut" is retained only as the historical
record of how the decision was reached. The `ValueError` guard and the missing `ClassVar` block
stay exactly as shipped; no follow-up work is pending against them.

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
    overlap_size: 3
```

### Per-backend kwargs passthrough

**This section describes the codebase as it stood before implementation; the change below has
since landed.** At the time of writing, `_run_feedforward`
(`collab_splats/wrapper/reconstructor.py`) did not see `pc_cfg` — it took seven explicit
scalars, and its call site inside `Reconstructor.build_pointcloud` unpacked the config itself. So the
change was two-part, not one line:

```python
# _run_feedforward signature — new keyword-only parameter, defaulted so existing callers still work
def _run_feedforward(..., max_points: int, creator_kwargs: dict | None = None):
    ...
    # the creator_map lookup inside _run_feedforward — max_points stays explicit;
    # the block supplies the rest
    creator = creator_map[backend](max_points=max_points, **(creator_kwargs or {}))

# the _run_feedforward call site in Reconstructor.pointcloud — reads the per-backend block
    creator_kwargs=pc_cfg.get(pc_cfg["backend"], {}),
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
the block. **Their defaults do not come from the shipped yaml, and an earlier draft said they
did.** Checked directly: both `ckpts/LoGeR/original_config.yaml` and
`ckpts/LoGeR_star/original_config.yaml` contain **exactly one top-level key, `model:`**. There is
no `training_settings` block and no `num_iterations` key in either file.

That matters because `build_forward_kwargs` (github.com/PolyCam/LoGeR @ 5d7c1a7,
`run_loger.py:149-164`) reads them as
`training.get(...)` against `config.get("training_settings", {})` — an empty dict for both
shipped checkpoints. So *every* fallback in that function is the value that actually runs, and
the real defaults are:

All `run_loger.py` line numbers below are github.com/PolyCam/LoGeR @ 5d7c1a7.

| Knob | Effective default | Where it truly comes from |
|---|---|---|
| `window_size` | `32` | `run_loger.py:47` argparse default (`args.window_size or ...` short-circuits before the yaml lookup) |
| `overlap_size` | `3` | `run_loger.py:49` argparse default, same short-circuit |
| `reset_every` | `0` (never reset) | `training.get("reset_every", 0)` fallback |
| `num_iterations` | `1` | `config.get("num_iterations", 1)` fallback |
| `pixel_limit` | `255000` | `run_loger.py:117` `load_images` function default |
| `use_multiview_confidence` | `False` | **ours**, matching VGGT-X and Omega; in no LoGeR config |

`LoGeRCreator` hard-codes these as dataclass field defaults with a comment citing the line above,
rather than reading a `training_settings` block that does not exist. Reading the yaml is still
required — but only for the `model:` block, which is genuinely per-variant.

One consequence for the config example below: **`overlap_size` defaults to 3, not 8.**

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

Flat functions, mirroring `test_vggt_omega_creator.py`. Every fast test uses a fake model — no
weights, no GPU, no network.

**Not all of it goes in one new file.** Three of these have established homes, and a monolithic
`test_loger_creator.py` would duplicate them:

| Test | Goes in | Why |
|---|---|---|
| 4 (`original_coords` → original-res K) | `tests/pointcloud/test_feedforward_intrinsics.py` | that file already owns "result.intrinsics at model resolution" across backends, including `_compute_vggtx_crop_coords` coverage |
| 8 (registry) | `tests/pointcloud/test_registry.py` | already holds one `test_get_creator_<backend>` per backend |
| vendored tree absent → guarded import | `tests/pointcloud/feedforward/test_spark_load_guard.py` pattern, as a new sibling | exact precedent exists |

The rest go in `tests/pointcloud/test_loger_creator.py`.

One caveat on test 8: `test_registry.py` covers `colmap`, `hloc`, `mapanything`, `vggtx` and
deliberately omits `vggt_omega`/`vggt_spark`, because optional vendored backends are absent in a
bare checkout. So the `loger` case needs a `skipif` on availability, or it follows the same
omission. Prefer the guarded test — it is the only thing that proves the registry wiring.

**Unit, against a synthetic pinhole scene:**

1. **The fit recovers a known K**, `@pytest.mark.parametrize`d over `(fx, fy)` — one isotropic
   case and one at `fx/fy = 1.10`. Build `local_points` from a synthetic depth map and the known
   focals; assert `_estimate_shared_intrinsics` returns a `(3, 3)` matrix whose `fx`/`fy` match
   within tolerance, stay distinct in the anisotropic case, **and whose `cx`/`cy` equal
   `((W-1)/2, (H-1)/2)`**. Ground truth is exact because no model is involved. One parametrized
   test rather than two near-duplicate ones. The anisotropic row protects the aspect-ratio
   argument under `_preprocess` — any "simplification" to `fx == fy` fails here — and the
   principal-point assertion is what makes the centred-grid convention a tested contract rather
   than a comment, since the synthetic scene is generated about that same centre.
2. **The median is robust.** Corrupt 30% of points *with high confidence* and assert the
   estimate still holds. A weighted median survives this; a least-squares fit would not. Written
   against median semantics deliberately — the estimator is a weighted median, not a fit.
3. **Degenerate input raises** rather than falling back: empty mask, NaN, negative depth.
4. **`original_coords`** is `[0,0,w,h,w,h]` per frame and round-trips through
   `_rescale_reconstruction_to_original_dimensions` to original-resolution K — with `fx != fy`
   preserved, since that function must scale x and y separately.
5. **c2w → w2c inversion.** Fake `camera_poses`; assert `result.extrinsics` is world-to-camera,
   not the raw model output. The single easiest thing to get backwards.
6. **`FeedforwardResult` field contract** — shapes and dtypes, `depth` is (N,H,W) not
   (N,H,W,1), colors uint8. Includes `camera_model == "PINHOLE"`, so the
   `(fx + fy) / 2` collapse under `SIMPLE_PINHOLE` cannot creep back in, and
   `0 <= depth_conf <= 1`, which is the cheapest available proof that the sigmoid ran — feed the
   fake model logits outside `[0, 1]` so an un-activated path cannot pass.
7. **Refusals**: `extract_intermediate_features` raises `NotImplementedError`; LC plus `loger`
   raises `ValueError`; a `model:` key that is neither `se3` nor a `Pi3.__init__` parameter
   raises.
8. **Registry**: `get_creator("loger")` returns `LoGeRCreator` when the vendored tree is
   present. (Note there is no `list_creators()` — `pointcloud/__init__.py` exposes only
   `get_creator` and `make_creator`.)

**Integration — the parity check.** Marked `@pytest.mark.slow`, skipped without weights:

Run the model, then unproject `depth` with the fitted K and `invert_poses(camera_poses)` and
compare against LoGeR's native `points`, read from the model output inside the test. Assert
median per-point error under a scene-scale-relative tolerance, and **log the residual** — it is
open item 3, not merely a pass/fail.

This one assertion proves the focal fit, the depth extraction, and that `_raw_to_world_points`
is a legitimate reuse — because LoGeR itself computes
`points = camera_poses @ homogenize(local_points)`, so any of those three being wrong breaks it.
It is the reason reusing the existing unprojection is both the minimal choice and the verifiable
one.

**Two corrections to the paragraph above (2026-08-13, measured).** It named
`unproject_and_filter_points`; that function is real (`vggtx.py`) but is NOT what this test
exercises. The test calls `_raw_to_world_points`, which performs no filtering at all — it builds a
dense meshgrid and runs its own unprojection loop (`feedforward/base.py`). Corrected inline above.
It also claimed "zero new production code carries the validation", which is no longer true:
`LoGeRCreator._forward_kwargs()` was extracted from `_forward` specifically so the parity test and
production issue the identical model call, because a drifting duplicate would make `native`
silently wrong. That is a deliberate and small addition, not an accident — but it is not zero.

**Correction (2026-08-13, measured).** This originally claimed a fourth thing, the c2w->w2c pose
inversion. It does not cover it. `_forward` inverts `camera_poses` and `_raw_to_world_points`
inverts it straight back, so the inversion cancels inside our side of the comparison: patching
`invert_poses` to the identity leaves the test passing at an unchanged residual. The inversion is
pinned by `test_forward_inverts_camera_poses_to_world_to_camera` instead. The second forward pass
is still justified, but for a different reason than originally written — it takes `native` from
LoGeR's own `preds["points"]` rather than from our postprocessing, keeping the two sides
independent.

**Smoke test, first implementation task, gates everything else:** import `Pi3` under torch
2.5.1+cu121 and run one forward pass on 8 frames. LoGeR pins torch 2.6.0 and we run 2.5.1; the
versions are close but the gap is unverified, and nothing else in the plan is worth starting
until it passes.

## Open items carried into the plan

1. Torch 2.5.1 vs LoGeR's pinned 2.6.0 — smoke test gates the work.
2. The `max_frames` ceiling for `loger` is unmeasured. Sweep, then document.
3. **The pinhole residual is MEASURED (2026-08-13): median 0.272%, p95 0.687%, p99 0.943% of
   scene scale.** Fixture: 8 frames at fixed indices `range(0, 192, 24)` (~1 fps) out of the
   committed `data/tutorial/tutorial_example-video.mp4`, centre-cropped 810x1080 (3:4) out of
   the 1080x1920 portrait source, run at the 434x574 model resolution, `LoGeR_star`, gating on
   `depth_conf > LOGER_CONF_THRESHOLD` (74.1% of pixels pass; the conf band is
   [0.0001, 0.9842]). Scene scale, the 95th percentile point norm, is 1.234. Measured by
   `test_pinhole_residual_against_logers_native_pointcloud`
   (`tests/pointcloud/test_loger_creator.py`), bit-reproducible across runs. The crop is
   load-bearing: uncropped, `_compute_target_size` maps 1080x1920 to 378x672 at
   sx == sy == 0.35 exactly, so fx and fy cannot disagree and the measurement is vacuous.

   **Read this as "the median is tight, the tail is not negligible", not as a clean bill of
   health.** Calibration by mutation: multiplying the fitted `fx` by 1.05 moves the median to
   0.493%, so the observed p99 of 0.943% is of the same order as a ~5-10% focal error on the
   worst pixels. The median says the shared-K fit reproduces LoGeR's own `points` to a quarter
   of a percent for a typical pixel; the p99 says roughly one pixel in a hundred carries an
   error a TSDF or a BA residual would see, and those pixels are not randomly placed — the
   non-pinhole freedom in LoGeR's `xy` field is largest off-axis, which is exactly where a
   mesh shows it. Not yet measured: whether that tail actually degrades mesh extent or BA
   convergence. Until it is, treat shared-K as adequate for the point cloud and UNVERIFIED for
   the mesh and BA paths, and re-measure on a wide-FOV or distorted-lens capture before
   assuming it carries.

   The 0.4% assertion threshold is itself calibrated by mutation, not chosen: the 2% the
   design originally proposed was measured INERT — a +5% fx error passed it.

   **Attribution measured 2026-08-14: the residual is LoGeR's, not ours.** The paragraph above
   asserts the residual comes from LoGeR's `xy` field being non-pinhole; that attribution was
   itself untested, so it was decomposed on the same fixture into (A) a genuinely non-affine ray
   field, (B) error in our `estimate_intrinsics_from_points` fit, and (C) the cost of sharing one
   K across frames. The decisive comparison is done entirely in the camera frame, with poses
   excluded, so it cannot be confounded by the pose inversion:

   | K used to rebuild `local_points` | median | p95 | p99 |
   | --- | --- | --- | --- |
   | our shared fitted K | 0.2148% | 0.5411% | 0.7434% |
   | each frame's own least-squares optimal K | 0.2231% | 0.6397% | 0.9991% |

   **Giving every frame its own optimal K does not reduce the residual** — it is marginally worse
   than our single shared K. That rules out (B) and (C) as sources. What remains is (A), measured
   directly: least-squares fitting `xy ~ [u, v, 1]` per frame leaves a non-affine residual of
   median 2.388e-3 against a p95 ray spread of 0.3546, i.e. **0.674% of ray magnitude that no
   pinhole K of any kind can represent**. The affine cross terms are ~1e-6, so the field has no
   skew and our zero-skew K is structurally right. Supporting numbers: our K sits within 0.26%
   (fx), 1.08% (fy), 0.70% (cx), 0.47% (cy) of the mean per-frame optimum; the per-frame optimal
   fx varies only 2.383% across the eight frames, which is why sharing one K costs nothing. The
   non-affine residual is anisotropic — vertical rms 5.42e-3 is ~2.6x horizontal 2.08e-3.

   Consequence for the UNVERIFIED mesh/BA verdict above: the tail is not a fit artifact we could
   tighten, so a better estimator would not remove it. If the tail turns out to matter downstream
   the fix has to be a non-pinhole camera model, not a better K.

   **Checked against upstream's own estimator (2026-08-14).** The claim "a better estimator would
   not remove it" was tested against the only estimator with a real claim to authority — LoGeR's.
   Upstream fits a focal **only in its eval scripts**, never on the demo/viser path, which
   hardcodes a 60 degree FOV and comments that the missing intrinsics are "a limitation"
   (`github.com/Junyi42/LoGeR @ 7685b7a` — `loger/utils/viser_utils.py:445-449`). The eval path is
   dust3r's `estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")` with
   `pp = (W // 2, H // 2)` (`eval/relpose/launch.py:528-534`, likewise `:721-730` and
   `eval/video_depth/launch.py:530-537`). Reimplemented from dust3r's published algorithm
   (`github.com/naver/dust3r` — `dust3r/post_process.py`) and run on the same fixture:

   | K estimator | median | p95 | p99 |
   | --- | --- | --- | --- |
   | ours — conf-weighted median, fx != fy, `pp=(W-1)/2` | 0.2148% | 0.5411% | 0.7434% |
   | upstream dust3r Weiszfeld — one focal, `pp=(W//2, H//2)` | 0.2263% | 0.5541% | 0.8180% |
   | upstream fallback when dust3r is absent, `f = max(H, W) = 574` | 2.7619% | 4.9562% | 6.8999% |

   Two independent methods on independent implementations agree on the focal to **0.29%**
   (upstream 708.44, ours 710.49), and ours is marginally the better fit. Ablations attribute the
   gap to the two deliberate differences: `pp=(W-1)/2` rather than `W//2` is worth 0.0084pp of
   median and **0.066pp of p99** — the half-pixel offset costs most in the tail, which is the part
   that gates mesh and BA — and fitting fx and fy separately is worth 0.0038pp.

   Note the third row: dust3r is **not** a LoGeR dependency, so upstream's own eval silently
   degrades to a constant `max(H, W)` focal that is 13x worse than either real fit. Not our
   problem, but it means published LoGeR eval numbers are not a reference for intrinsics quality.

   Two model-level assumptions were verified against source at the same time, both correct:
   `local_points = torch.cat([xy * z, z], dim=-1)` with `z = torch.exp(z)`, so channel 2 is z-depth
   along the optical axis and not ray distance (`loger/models/pi3.py:772-775`, the `pi3x=False`
   branch we take); and `points = einsum('bnij,bnhwj->bnhwi', camera_poses, homogenize(local_points))`
   (`:807`). `xy` is two free channels off a single `nn.Linear` + `pixel_shuffle`
   (`loger/models/layers/transformer_head.py:58-81`) with no distortion model anywhere in `loger/`,
   which is the mechanism behind the 0.674%. `_preprocess` and `_forward_kwargs` were also compared
   against `demo_viser.py` and the `LoGeR_star` yaml and match; ImageNet normalisation happens
   inside the model (`loger/models/pi3.py:672`), so our `[0, 1]` input is correct.
4. ~~Loop closure calibration for the LoGeR backbone.~~ **Closed 2026-08-14 by user decision** —
   "we dont need loop closure for this". The `Reconstructor`-level refusal is the final state;
   no sweep is owed. See "Why no loop closure in the first cut".
5. Multiview confidence calibration — owned by
   `2026-08-12-multiview-confidence-all-models-design.md`; `loger` should be added to its scope.
6. Sparse `FeedforwardResult` for long sequences — rejected here for consistency; would need
   its own design covering all four backends.
7. **Neither LoGeR repository ships a LICENSE file.** Vendoring an unlicensed tree into
   `third_party/` matches what we already do for other backends, but this design additionally
   *copies* ~35 lines out of PolyCam's `run_loger.py`. Flagged for the user's call before the
   port lands; the fallback is to reimplement the estimator from the pinhole identity, which is
   a handful of lines of standard geometry, and cite the original only as prior art.
8. **Square-pixel averaging in original-resolution space.** Dropped from the first cut with a
   reason (see the fit section) rather than dismissed. Revisit only if the parity test shows the
   estimator's own spread exceeds the ~1% the model-resolution version would cost.
