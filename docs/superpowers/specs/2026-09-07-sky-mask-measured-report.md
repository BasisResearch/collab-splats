# mesh.mask_sky — measured A/B report

Measured 2026-09-07 on branch `feat/sky-segmentation` (worktree
`/workspace/collab-splats/.worktrees/sky-mask`, forked at `clean/final`), against
[the design](2026-09-07-sky-segmentation-design.md) and
[the plan](../plans/2026-09-07-sky-segmentation.md) Task 8.

**Verdict up front: the gate fails on its tutorial half. `mesh.mask_sky` ships `false`.**

---

## What was run

Both arms of each scene run the mesh stage twice against the SAME `pointcloud.zarr`, so
`mesh.mask_sky` is the only difference. Backend is `skywater`
(`Realcat/skywater_seg`, SegFormer MiT-B2), the default that
`collab_splats.semantics.segmentation.sky.DEFAULT_BACKEND` names.

| scene | source | voxel_size | depth_trunc | bands | conf_percentile | frames |
|---|---|---|---|---|---|---|
| `tutorial_example-video` | feedforward | 0.0025 | 1.5 | null | 20 | 193 |
| `GH010229` | splats | 0.2 | 120.0 | null | null | 853 |

Both use a SINGLE TSDF volume (`bands: null`), per the standing instruction not to band by
default. Both ran with `semantics: {enabled: false}`.

## Tutorial (`data/tutorial/tutorial_example-video.mp4`)

`mesh.mask_sky: dropped 13.49% of valid depth pixels as sky`

| metric | mask_sky off | mask_sky on | delta |
|---|---|---|---|
| vertices | 857511 | 914203 | +56692 |
| triangles | 1599985 | 1702662 | +102677 |
| components | 1189 | 1267 | +78 |
| largest_component_fraction | 0.9106 | 0.9099 | -0.0008 |

Behind the cleaned numbers, from `remove_floaters` / `fill_holes`:

| | off | on |
|---|---|---|
| pre-clean components | 211,523 | 110,544 |
| scene_scale | 2.876 | 2.907 |
| raw triangles (pre `fill_holes`) | 1,559,554 | 1,658,169 |

## GH010229

`mesh.mask_sky: dropped 0.61% of valid depth pixels as sky`

| metric | mask_sky off | mask_sky on | delta |
|---|---|---|---|
| vertices | 294491 | 250625 | -43866 |
| triangles | 519074 | 442652 | -76422 |
| components | 571 | 483 | -88 |
| largest_component_fraction | 0.9479 | 0.9540 | +0.0061 |

| | off | on |
|---|---|---|
| pre-clean components | 61,215 | 59,309 |
| scene_scale | 144.149 | 137.175 |
| raw triangles (pre `fill_holes`) | 478,420 | 405,735 |

## Artifacts

Written by `evals/scripts/eval_sky_mask.py` into this session's scratchpad
(`/tmp/claude-0/-workspace-collab-splats/a253b673-.../scratchpad/sky_ab/`), which is
session-local and will not survive; regenerate with the same script, the mask cache under
`<scene>/sky/skywater/` makes the rerun cheap.

- `tutorial_skywater/sky_masks.png`, `gh010229/sky_masks.png` — every 10th frame with its
  mask composited 75% red and labelled with its own sky fraction. The tutorial sheet shows
  the mask stopping at the tree line and passing between bare branches, not swallowing
  them; GH010229's sheet is 0.000 on almost every tile, which is correct — that scene is
  shot downward over a fenced yard and holds sky only in a few frame corners.
- `tutorial_skywater/mesh_renders.png`, `gh010229/mesh_renders.png` — both arms from ONE
  shared camera framed on the off-arm mesh, so any reframing between the two tiles would be
  a bug in `render_meshes`, not a result. On GH010229 the on-arm visibly loses a bright
  white blob beside the dumpster, which is the sky-carved geometry going away. On the
  tutorial the two tiles are near-identical apart from extra spiky structure low-left in the
  on-arm.

## Reading the tutorial result

Masking halves the tutorial's pre-clean fragment count (211,523 -> 110,544, -47.7%) and yet
the post-clean kept count RISES (1,189 -> 1,267). That is not a contradiction and it is not
a threshold artifact in the masking's favour:

- `remove_floaters` thresholds are scale-relative (`get_scene_scale`), and masking moves the
  bounding box. Here scene_scale RISES 2.876 -> 2.907, so the floater threshold gets looser,
  not tighter — the extra survivors clear a slightly easier bar.
- raw triangles also rise, 1,559,554 -> 1,658,169. Sky depth was carving free-space through
  real geometry; removing it lets that surface close. More real surface means more surviving
  components, and `largest_component_fraction` dips because the main body grows more slowly
  than the newly-surviving small pieces.

So the tutorial's mask is doing something defensible to the volume while still moving both
gate metrics the wrong way. The gate is on the metrics, not on the story.

On GH010229 the same check runs the other way: scene_scale SHRINKS 144.149 -> 137.175 (the
sky backdrop was inflating the bounding box, so the floater bar gets HARDER), and the kept
count still drops 571 -> 483. That improvement survives the scale confound.

## Verdict against the gate

Gate: **the tutorial scene must improve AND GH010229 must not regress.**

- **GH010229: PASS.** -88 components, +0.0061 largest-fraction, both in the good direction,
  and it drops only 0.61% of valid depth pixels.
- **Tutorial: FAIL.** +78 components and -0.0008 largest-fraction. Both gate metrics move
  the wrong way, at a cost of 13.49% of valid depth pixels.

The gate is an AND, so the gate fails. `mesh.mask_sky` stays `false` in `configs/base.yaml`,
documented in `docs/mesh.md` as an opt-in flag with this report as its evidence.

Note the control expectation in the plan was inverted by the model swap: the plan expected
GH010229 to regress on skyseg's false positives, and the `skywater` backend fixed exactly
that half. What fails now is the half that was expected to pass.

## For the record: the skyseg arms

`skyseg` (VGGT's model, still registered for parity) was measured first and is what moved
the default to `skywater`. It calls bright desaturated ground sky — 39.4% of one GH010229
frame that holds no sky at all — and raising its threshold 0.5 -> 0.999 cut the false
positives 3.6x but the true sky 8.3x.

- tutorial, skyseg on-arm: 9.55% dropped; 883,771v / 1,648,308t / 1,232c / 0.9015
  (pre-clean 140,724). Same direction of failure as skywater, smaller magnitude.
- GH010229 BANDED (`mesh.bands` on, not the single-fuse config above), skyseg:
  off 2,803,585v / 5,254,968t / 716c / 0.9460; on 2,801,091v / 5,253,285t / 706c / 0.9430.
  Archived under `gh010229_banded/`; kept only because it is the run the plan was written
  against, superseded by the single-fuse numbers above.

## Follow-ups, none designed

- **A depth-agreement gate.** The spec deliberately left undesigned an AND of the sky mask
  with a far-depth test, on the theory that false positives are near surfaces. That still
  buys the skyseg failure mode, but skywater no longer needs it: its GH010229 false-positive
  rate is already 0.61%. It would not fix the tutorial half, where the mask is CORRECT and
  the fragment accounting is what moves.
- **A metric that separates real surface from floaters.** The tutorial result exposes that
  `components` and `largest_component_fraction` cannot tell "one more real twig survived"
  from "one more floater survived". Until they can, a scene whose masked volume genuinely
  improves can still fail this gate.
- **Scale-normalised floater thresholds.** Both scenes' `scene_scale` moved under masking,
  in opposite directions. Any future A/B that touches the volume's extent should either pin
  the threshold across arms or report the scale alongside the count.
