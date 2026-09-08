# Sky segmentation as a general util — design

Date: 2026-09-07
Branch: `feat/sky-segmentation` (worktree `.worktrees/sky-mask`, forked at `clean/final` `b11cd016`)
Status: design approved, not implemented

## Problem

Feedforward depth models assign a depth to every pixel, sky included. Sky has no surface,
so those pixels fuse a spurious backdrop into the TSDF and seed floaters around it. The
pipeline has no notion of sky today: `mesh.conf_percentile` is the only per-pixel depth
gate, and confident-but-meaningless sky depth survives it.

Goal: a reusable sky-segmentation util, plus a mesh-stage consumer, plus a measurement of
whether masking sky improves the fused mesh.

## Source and attribution

Ported from `facebookresearch/vggt` at pin `a288dd0f14786c93483e45524328726ab7b1b4ce`:

- `visual_util.py:365-393` — `segment_sky`
- `visual_util.py:396-434` — `run_skyseg` (preprocessing + inference)
- `demo_viser.py:270-305` — how upstream applies it

Model: `JianyuanWang/skyseg` → `skyseg.onnx` (public, ungated, 176 MB). Input
`[1,3,320,320]` float, output `[1,1,320,320]` float.

The port site carries this attribution inline, per the repo convention.

## Measured findings

Probed directly against the pinned model before designing. These numbers, not upstream's
comments, are what the design is built on.

- output: already sigmoid-terminated, range `[0.000, 1.000]`. Not an unbounded logit.
- polarity: **high = sky**. Tutorial frame top rows mean 0.662, ground rows 0.000.
- threshold sensitivity: low. Tutorial sky fraction 0.237 at 0.125 → 0.115 at 0.9.
- speed: **632 ms/frame** ONNX on CPU (`onnxruntime 1.26.0` here has no
  `CUDAExecutionProvider`). Resize costs 1 ms down, 30 ms up. 300 frames ≈ 3.2 min.
- mask quality, `data/tutorial` frame: clean. Sky filled, thin branches and trunks
  correctly punched through — the hard case works.
- mask quality, GH010229: **12.1% and 1.8% of pixels falsely masked**. Diagnosed: the
  model calls the blown-out overexposed road surface sky. Real geometry, valid depth.

### Two upstream defects not inherited

1. **Polarity comment is wrong.** `visual_util.py:388` says "model outputs low values for
   sky", then sets `result < 32` → 255 and calls 255 non-sky. `demo_viser.py:300` keeps
   `result < 32`. Only consistent if the model emits high values for sky — which the probe
   confirms. The comment is wrong; the usage is right.
2. **Per-image min-max rescale is dropped.** `run_skyseg` does `(x-min)/(max-min)*255`.
   The output is already `[0,1]`, so this is near-identity in the common case — but on a
   frame whose maximum is 0.3 it stretches that to 1.0 and manufactures sky from nothing.
   We threshold the raw probability instead.

Everything else is byte-for-byte upstream: BGR→RGB, 320×320 square resize (aspect
deliberately ignored), ImageNet mean/std, NCHW float32, bilinear resize back to full frame.

## Design

### The util — `collab_splats/semantics/segmentation/sky.py`

Registered backend, not a loose function, so a second sky model is a config string rather
than a consumer rewrite:

```python
@BaseSegmentation.register("skyseg")
class SkySegmentation(BaseSegmentation):
    """
    Sky mask from VGGT's skyseg ONNX model.

    - one binary mask per image, unlike the object backends' N masks
    - the raw probability map rides along so a threshold sweep needs no second forward pass

    Args:
        threshold: probability above which a pixel is sky.
        model_path: local skyseg.onnx; None downloads it from the Hub.
    """

    def __init__(self, threshold: float = 0.5, model_path: Path | None = None) -> None: ...

    def segment(self, image: np.ndarray | Image.Image) -> tuple[torch.Tensor, dict]:
        """
        Sky mask for one frame.

        - True is sky, inverted from upstream's 255-is-not-sky convention

        Args:
            image: the frame to segment, coerced through `utils.image.open_image`.

        Returns:
            (mask, metadata) — mask (1, H, W) bool; metadata {'raw': (H, W) float32}.
        """
```

`segment` widens to `np.ndarray | Image.Image` to match `BaseSegmentation.segment`'s
declared type — `read_frames` hands us ndarrays and `open_image` coerces both.

Both sketches above are written to the docstring contract deliberately: single-line
summaries that do not restate their own names, bulleted bodies, every parameter
annotated and named under `Args:`, and an explicit `-> None` on `__init__`.

- `segment_with_text` inherits the base's `NotImplementedError` — skyseg takes no prompt.
- metadata carries the **raw** probability map, so threshold calibration needs no second
  forward pass.
- weights via `hf_hub_download(repo_id="JianyuanWang/skyseg", filename="skyseg.onnx")` —
  what `vda.py`, `loger.py` and `vggt_omega.py` all do, not upstream's hand-rolled
  redirect follower. See the reuse section on why the `load_hf_weights` wrapper is skipped.
- `threshold` is a **probability**, default 0.5. Upstream's `<32/255` is ≈0.125.

Batching and caching stay out of the class — every existing backend is per-image, and a
`segment_batch` on this one alone would make it the odd backend. One module-level function
beside it carries the pipeline concern:

```python
def sky_masks(
    images_dir: Path | str,
    idxs: Sequence[int] | None = None,
    cache_dir: Path | str | None = None,
    backend: str = "skyseg",
) -> np.ndarray
```

- returns `(N, H, W)` bool, True = sky.
- `idxs` mirrors `frames.read_frames`'s contract **exactly**: SOURCE frame indices in the
  order given; `None` reads every frame in filename order.
- resolves the backend through `BaseSegmentation.get(backend)` — the seam SAM3 slots into.
  `RegistryMixin.get` already raises `ValueError` listing the registered names, so an
  unknown backend needs no hand-written check.

### Cache

`<output_path>/sky/frame_NNNNNN.png`, uint8, **255 = sky**. Default `cache_dir` is
`Path(images_dir).parent / "sky"`.

- mirrors the `images/` naming so `frames.frame_paths` and `frame_idx_from_path` work on
  it unchanged, and a human can open one and look — mask quality is the main risk here.
- per-file existence check, same shape as upstream's `<folder>_sky_masks/`.
- **polarity is inverted from upstream**, which stores 255 = *non*-sky. Documented at the
  writer, the reader, and in the test suite.
- caching earns its place at 632 ms/frame; it would not at the ~ms/frame this was assumed
  to cost before measurement.

### Consumption — `_run_tsdf_mesh`, `wrapper/reconstructor.py:580`

Applied at **frame resolution, after `upsample_depths`** — the same grid as `rgbs`:

```python
depths = np.where(sky, 0.0, depths)
```

plus a shape guard and a "% of depth pixels dropped" log line mirroring the
`conf_percentile` one.

**Frame ordering.** The two source arms order frames differently:

- feedforward: `frames.read_frames(images_dir)` → filename order.
- splats: `read_frames(images_dir, [int(i) for i in image_ids])` → checkpoint order, by
  source `frame_idx`.

A single mask stack computed in filename order misaligns silently on the splats path —
wrong mask on wrong frame, no error raised. So `render_tsdf_inputs` returns `image_ids` as
a fifth element and the splats arm passes it as `idxs`.

### Config

`mesh.mask_sky: false` in `configs/base.yaml`, bool-validated with the other mesh keys,
threaded `Reconstructor.mesh()` → `_run_tsdf_mesh`.

Ships **off**, matching `use_multiview_confidence`, `undistort` and
`geometric_verification` — the measured false-positive rate earns that default.

The config comment must state an asymmetry a reader will otherwise assume away:
`mesh.conf_percentile` **also** masks splats depth targets
(`reconstructor.py:1497`); `mesh.mask_sky` **does not**.

## Non-goals

- **Pointcloud-stage gating.** Masking at mesh leaves sky points in `pointcloud.zarr` and
  `sparse_pc.ply`, which feed splats init and depth alignment. That is probably the larger
  win, and it is a separate spec — gating there makes every A/B arm a full pointcloud
  re-run and destroys attribution of the mesh delta.
- **Sky-masking splats depth targets.** Frame-res mask vs model-res targets behind the
  crop boxes; separate spec.
- **SAM3.** Access is confirmed (HF account `tbotch`, `facebook/sam3` gated=manual,
  `model_info` succeeds), but running it needs either `transformers>=5.0` (repo pins
  `<5.0`, installed 4.57.6) or a `--no-deps` install of the standalone package to avoid
  downgrading numpy 2.1.3→<2 and ftfy 6.3.1→6.1.1. Not worth it for a sky mask. Note that
  `semantics/segmentation/sam3.py` imports `sam3.model_builder`, which is not installed —
  that backend is currently dead code on every branch.
- **Mask dilation.** The tutorial frame shows branches correctly preserved; dilating to
  clean silhouette bleed would eat the thin geometry this scene exists to test.

## Repository conventions and reuse

The new file is a sibling of `sam3.py` / `mobile_sam.py` / `insid3.py` and must be
indistinguishable from them in shape.

### The style is machine-enforced, not advisory

`tests/test_docstring_contract.py` has `PACKAGES = ("preproc", "semantics", "pointcloud")`
and globs `collab_splats/<pkg>/**/*.py`. **`semantics/segmentation/sky.py` is inside that
glob**, so it is linted the moment it exists. What the lint actually asserts:

- module, public class and public function each carry a docstring.
- summary is a **single line ≤100 chars** and must not start with the def's own name.
- everything between summary and the first `Args:`/`Returns:`/`Raises:`/`Yields:`/`Notes:`/
  `Example:` section is `- ` bullets. One prose paragraph fails the file.
- every parameter is annotated in the signature **and** matched in the docstring by
  `^\s*\*{0,2}<name>\b` — a name mentioned only inside another parameter's description
  does not count.
- the return is annotated **even when it is `None`** (`node.returns is None` is a
  violation), and a non-`None` return needs a `Returns:` section.
- any run of **3+ consecutive `#` lines** must have a header line followed by a `- ` bullet
  on the next line. Runs led by `-`, `*`, `─`, `#` or `=` are exempt, which is why the
  boxed dividers pass.

So the plan writes `sky.py` to the contract from the first commit, and runs
`pytest tests/test_docstring_contract.py` as part of every task's gate — not just the
feature tests.

### File structure — match `semantics/segmentation/`, not `mesh/`

Two divider styles coexist on `clean/final`. `mesh/` and `preproc/` use terse
`######## Name`; `semantics/` and `utils/` use the boxed form. The new file takes the
**boxed** form, because that is what its neighbours use:

```
########################################################
########## Sky segmentation backend ####################
########################################################
```

- `from __future__ import annotations` first, then stdlib / third-party / local imports.
- `logger = logging.getLogger(__name__)` after imports.
- constructor params documented under `Args:` on the **class** docstring, as `sam3.py`
  does — `__init__` is private to the lint and is not checked itself.
- registration in `segmentation/__init__.py` goes in the existing
  `── Concrete backends ──` import block and under the `# backends` comment in `__all__`.
- `isort` carries `profile="black"` at width 88 — write long imports parenthesized.
- `CLAUDE.md`'s architecture map lists `segmentation/  # BaseSegmentation; registered
  insid3, mobilesamv2, sam3` — the new backend is added there too, or the map goes stale
  on the same commit that lands the code.

### Reuse — do not re-declare what already exists

| Need | Existing code to use | Instead of |
|---|---|---|
| ImageNet mean/std | `utils/image.py` `IMAGENET_MEAN` / `IMAGENET_STD` | upstream's hardcoded literals |
| weight download | raw `hf_hub_download`, matching `vda.py:54`, `vggt_omega.py:194`, `loger.py:287` | upstream's hand-rolled redirect-following `download_file_from_url` |
| input coercion | `utils/image.py` `open_image` | a per-backend isinstance ladder |
| frame listing / ordering | `preproc/frames.py` `frame_paths`, `frame_idx_from_path`, `IMAGE_EXTS` | a fresh `sorted(glob(...))` |
| PNG write level | `preproc/frames.py` `_PNG_COMPRESSION`, **promoted to public `PNG_COMPRESSION`** | a second hardcoded `1` |
| mask QA renders | `utils/visualization.py:162` `overlay_masks` | a hand-rolled overlay blend |
| A/B component stats | `mesh/clean.py` `get_scene_scale` + the `cluster_connected_triangles` pattern in `remove_floaters` | duplicating the clustering maths in an eval script |

**`load_hf_weights` is deliberately not used.** `utils/torch_utils.py:74` defines it, but a
repo-wide grep finds **zero callers** — it is re-exported from `utils/__init__.py` and
nothing else. All three real weight downloads call `hf_hub_download` directly, and the
wrapper's whole body is `return hf_hub_download(repo_id=repo_id, filename=filename)`.
Reusing it would follow a function's existence rather than the repo's actual practice, and
CLAUDE.md rules out "wrapper layers that add no value". Verify usage, not just presence.

`overlay_masks` is the opposite case: also unreferenced from module code, but the
2026-07-11 geometry-cleanup spec listed it for deletion and then explicitly kept it on a
notebook gate, and `tests/test_visualization.py` covers it. It is notebook-facing on
purpose. Check it renders a single-mask tensor sensibly before relying on it — every
existing caller passes N object masks, not one binary mask.

`_PNG_COMPRESSION` becoming public is the one edit to existing code this spec asks for
beyond the two wiring changes: it stops being private the moment a second writer needs the
same level, and the alternative is a duplicated literal whose justification ("level 9 costs
10x the time for 11% of the size") lives in another file.

## Tests

`tests/semantics/test_sky_segmentation.py` — flat, matching `test_insid3_segmentation.py`.
ONNX session monkeypatched, no network:

- **polarity**: raw map high in the top half → mask True there. Guards the inversion, and
  would have caught upstream's wrong comment.
- **on-disk polarity**: cached PNG is 255 where sky. In-memory correctness does not prove
  the artifact boundary.
- **cache hit**: a second call runs the session zero times.
- **ordering**: a permuted `idxs` returns permuted masks. Discriminating case per arm, not
  a smoke test.
- **no rescale**: a raw map maxing at 0.3 yields an **empty** mask. This is the test that
  pins the dropped-rescale fix.

`tests/wrapper/test_reconstructor.py` — both source arms get a case, since the mask lands
after the if/else but the shapes come from different producers:

- feedforward and splats: masked pixels zeroed, unmasked pixels bit-identical.
- `mask_sky: false`: no `sky/` directory, no model load, depth untouched.
- shape mismatch raises.

`tests/mesh/test_io.py` — `render_tsdf_inputs` returns `image_ids` matching the frames it read.

`tests/test_docstring_contract.py` — no new case to write, but it starts covering `sky.py`
the moment the file lands, so it joins the per-task gate alongside the feature tests.

## A/B protocol

Per scene: one pointcloud run, then two `--stages mesh` runs against the **same**
`pointcloud.zarr`, differing only in `mask_sky`.

- **`data/tutorial`** — in-repo, fast, sky through bare branches. The real test.
- **GH010229** — the control. Needs a full preproc + pointcloud rebuild from the local
  746 MB mp4; `images/` and `pointcloud.zarr` are gone.

Reported per arm: connected-component count, main-component vertex fraction, vertex and
triangle counts, % of depth pixels dropped, and fixed-camera renders. Component stats come
from what `mesh/clean.py` already computes, so the numbers stay comparable to the
floater-cleanup and banded-fusion runs.

Results land in `docs/superpowers/specs/2026-09-07-sky-mask-measured-report.md`, matching
the multiview-confidence report convention.

**Gate: tutorial must improve AND GH010229 must not regress.** On the frame evidence the
control is expected to regress. If it does, the honest outcome is "ships off, documented,
needs a depth-agreement gate before it can be a default" — not a green tick. A follow-up
option, deliberately not designed here: AND the sky mask with a far-depth test, since the
false positives are near surfaces.

## Amended during planning

Three things in the sections above were checked against the tree while writing
`docs/superpowers/plans/2026-09-07-sky-segmentation.md` and did not survive. The plan is
the authority on all three.

- **Mask rank is `(H, W)` bool, not `(1, H, W)`.** `BaseSegmentation.segment`'s docstring
  enumerates a rank per backend, and insid3 — the one other single-mask backend — already
  returns `(H, W)`. Matching it removes a squeeze at every call site.
- **`mesh.mask_sky` is not bool-validated.** The Config section says "bool-validated with
  the other mesh keys"; no mesh key is bool-validated. `validate_config` checks
  `mesh.source` membership and nothing else, and `texture: false` has no validation at all.
- **The mesh-wiring tests live in a new `tests/wrapper/test_mask_sky.py`**, not in
  `tests/wrapper/test_reconstructor.py`. That file is already ~1400 lines of config and
  orchestration coverage, and `tests/wrapper/` is otherwise one file per seam.

One reuse target also moved: `evals/scripts/analyze_splats.py::mesh_stats` reports
`largest_component_fraction` as a fraction of TRIANGLES, where the A/B protocol section
says "main-component vertex fraction". The existing function is reused as-is and the report
says triangle fraction.

## Risks

- **False sky on blown-out bright surfaces** — measured at 12.1% on one control frame.
  The main risk, and the reason for the default-off ship and the pass/fail control.
- **CPU-only inference** — 632 ms/frame. Acceptable behind a cache; would not be if the
  mask were recomputed per stage.
- **Stale cache** — a mask cached at one threshold is reused after the threshold changes.
  Mitigation: threshold is not a config key in this spec, so it cannot drift between runs
  without a code change.
- **Worktree PYTHONPATH** — a bare `pytest` inside a worktree tests `/workspace/collab-splats`
  and reports a false green. Every run is `cd <worktree> && PYTHONPATH=<worktree> ...`
  with a printed `collab_splats.__file__` proof line. `third_party/*` is gitignored, so it
  is symlinked into the worktree; otherwise guarded tests skip instead of failing.
