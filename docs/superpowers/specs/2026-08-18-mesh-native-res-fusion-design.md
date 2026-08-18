# Mesh Native-Resolution TSDF Fusion — Design

**Date:** 2026-08-18
**Status:** Draft — awaiting user review
**Scope:** `collab_splats/mesh/` (utils, tsdf), `collab_splats/wrapper/reconstructor.py` (mesh stage wiring), `configs/base.yaml`

## Problem

Two fidelity gaps in the TSDF mesh, both rooted in what enters `volume.integrate`:

1. **Floaters.** TSDF fuses **raw** model depth. The pointcloud path filters points by learned
   confidence before anyone sees them — the mesh path applies no mask at all. Every
   low-confidence depth pixel becomes a voxel, and single-view fabrications become the speckle
   components `clean_repair` has to mop up afterwards (240k components on the current params).
2. **Soft colors and staircase silhouettes.** Depth, RGB and K all come off the model forward
   pass at model resolution (~518 px wide). Vertex colors are integrated from ~518 px RGB, and
   object outlines carry model-res aliasing, even though the original frames (frames.zarr) and
   original-resolution intrinsics (COLMAP camera) both already exist on disk.

Upsampling adds **no geometric detail** — the depth ceiling stays model resolution. The wins
are sharper vertex colors, silhouettes snapped to image edges, and denser ray sampling per
voxel at the shipping `voxel_size: 0.0025` (model-res pixel footprint at 1 m is coarser than
the voxel grid).

## Design

Two independent features behind two config keys, applied in order. Both live in the
feedforward→TSDF adapter layer; `Open3DTSDFFusion.create` stays a dumb fuser.

### 1. Confidence masking (floaters)

New adapter step in `_feedforward_to_tsdf_inputs`: pixels whose learned confidence falls below
a global percentile get depth set to 0 (0 = "no observation" to Open3D's RGBD integration —
same convention `depth_trunc` already relies on).

- Source: `FeedforwardResult.confidence` (N, H, W) — already persisted in feedforward.zarr and
  loaded by default (`load_confidence=True`); `_run_tsdf_mesh` gets it for free.
- Threshold: the **same rule the pointcloud path uses** — `subsample_points`' global
  percentile cutoff (strict `>`, keep-all when confidence is uniform), extracted into a shared
  `confidence_mask(conf, percentile)` helper in `pointcloud/utils.py` that both callers use.
  No mask is persisted: raw confidence already lives in feedforward.zarr, the mask is a
  3-line derivation, and keeping the percentile a mesh-time knob means sweeping it never
  re-runs the feedforward pass. Config `mesh.conf_percentile: float | null`, default `null`
  (off — shipping output byte-identical).
- If `confidence is None` (old zarr) and `conf_percentile` is set → hard error, not silent
  skip. A user who asked for masking must know it did not run.
- Masking happens **before** upsampling — never amplify pixels you are about to delete.

### 2. Native-resolution fusion (colors + silhouettes)

Config `mesh.native_resolution: bool`, default `false`. When true, the mesh stage fuses at the
original frame resolution:

- **RGB:** read original frames from `frames.zarr` via `FrameStore.images()` — uint8
  (N, H, W, 3), decode-once store, row order matches the reconstruction's frame order
  (both derive from the same selection; guard asserts `len(store) == N`).
- **Depth:** upsample each model-res depth map to its crop region in original pixels using
  `original_coords` (tl/cr give the crop box, orig_w/h the canvas), paste into a zero-filled
  (orig_h, orig_w) canvas. Outside the crop = 0 = no observation.
- **Upsample method: joint (guided) filter, not bilinear.** Bilinear across a depth
  discontinuity fabricates depth on no surface — the exact failure mode rejected in the
  multiview-confidence work — and would *add* skirts. Pipeline per frame:
  1. nearest-neighbour resize model depth → crop size (blocky but never fabricates),
  2. guided filter with the original-res RGB crop as guide, radius ~2× the upsample factor,
     validity-weighted so masked/zero pixels do not bleed in,
  3. re-zero every output pixel whose nearest source pixel was masked (the guide sharpens
     edges; it must never resurrect deleted depth).
  `cv2.ximgproc` is **not installed** — implement the classic box-filter guided filter
  (~30 lines, `cv2.boxFilter`, O(1) per pixel) as `guided_upsample_depth` in `mesh/utils.py`.
- **Intrinsics:** the COLMAP camera — `PointcloudResult.intrinsics` is original-resolution by
  contract (`build_colmap` rescales it; see 2026-08-11 mesh-intrinsics spec). The model-res K
  from the zarr is not used on this path. This deliberately re-creates the resolution pairing
  the 2026-08-11 guards police — the guards must learn that (original-res K, original-res
  depth) is a *consistent* pairing: the frame-count guard stays; add a principal-point-inside-
  depth-grid check (same check the multiview work uses) so a mismatched pairing still fails
  loudly instead of producing a collapsed mesh.
- **Memory:** RGB stays **uint8** end-to-end. `Open3DTSDFFusion.create` gains uint8
  passthrough (`rgbs.dtype == np.uint8` → skip the ×255; the existing [0,1] float contract and
  its >1.5 guard are unchanged for float input). At 4K × 300 frames float32 RGB would be
  ~30 GB — uint8 is a quarter of that, and depth (float32 canvas) dominates instead.
  Per-frame integration already streams; only the input arrays are resident.

### Wiring

`_run_tsdf_mesh` grows `conf_percentile`, `native_resolution`, and a `frames_zarr: Path`
argument (the Reconstructor knows the scene layout; the mesh module does not discover paths).
`pointcloud_to_mesh` / `_feedforward_to_tsdf_inputs` take the new options and return
(depths, rgbs, c2w, intrinsics) exactly as today — resolution is a property of the arrays,
not a new interface.

`configs/base.yaml`:

```yaml
mesh:
  conf_percentile: null    # mask depth below this global confidence percentile (null = off)
  native_resolution: false # fuse at original frame resolution (frames.zarr RGB + upsampled depth)
```

Both default off → every existing config reproduces today's output byte-identically.

### Cost

Native-resolution integration scales with pixel count: ~16× at 518→2000 px. On the reference
scene (100 frames) TSDF integration goes from ~40 s to an estimated ~10 min; acceptable for an
opt-in flag. The guided filter is O(pixels) per frame and negligible next to integration.

## Verification

Prototype-before-polish, on the same scene as the clean_repair work
(`/workspace/outputs/2026_07_15-Goprosplat-GH010229`, vggt_omega):

1. **Masking A/B:** `conf_percentile ∈ {null, 20, 40}` → component count before clean_repair,
   vertex count, visual floater check. Success = large component-count drop at ≤ moderate
   surface loss.
2. **Native-res A/B:** same scene, `native_resolution: true` → vertex-color crop comparison
   and silhouette overlay vs model-res mesh; wall-clock + peak RSS recorded (46.6 GB cap).
3. **Regression:** defaults-off run byte-compares mesh.ply against current output.

## Testing

- `guided_upsample_depth`: shape/crop placement; a step-edge depth map upsamples without
  intermediate fabricated values (the anti-bilinear property, asserted on the edge profile);
  masked source pixels stay zero after filtering.
- Confidence masking: synthetic result with known confidence → exactly the sub-percentile
  pixels zeroed; `confidence is None` + percentile set → raises.
- uint8 passthrough: `create()` fuses uint8 RGB identically to the same data as [0,1] float
  (small synthetic scene, byte-compare meshes).
- Guards: frame-count mismatch and principal-point-outside-grid raise on the native path.
- Adapter defaults: options off → outputs identical to today (existing tests keep passing).

## Alternatives considered

- **Ball pivoting on the (cleaner) pointcloud** — rejected. The pointcloud only looks better
  because it is confidence-filtered; BPA adds no averaging, so multi-frame pose/scale noise
  shingles into double surfaces, plus normal estimation, per-scene radius tuning, holes at
  density gaps, and 5M+-point runtimes. Masking gives TSDF the same filter without losing its
  noise-averaging.
- **Poisson on the filtered cloud** — already exists (`mesh/poisson.py`); better than BPA for
  noise but hallucinates surface in unobserved space. Remains available; not this spec.
- **Naive bilinear upsample** — fabricates depth across discontinuities; rejected outright.
- **Depend on `opencv-contrib` for `cv2.ximgproc`** — swaps the whole opencv wheel for one
  30-line algorithm; rejected.
- **TSDF weight-threshold extraction (tensor API port)** — complementary floater lever, out of
  scope; revisit only if masking leaves speckle.

## Implementation principles

- Reuse: `FrameStore` for RGB, `original_coords` for crop mapping, COLMAP K as-is,
  existing `depth == 0` no-observation convention. No new dependencies.
- Config surface: two keys, both default-off, byte-identical shipping output.
- The adapter owns resolution policy; `Open3DTSDFFusion.create` stays resolution-agnostic.
- No pose-dependent work — pose refinement is parked; nothing here waits on it.
