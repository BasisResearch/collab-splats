Mesh
====

TSDF meshing from posed depth + RGB, then cleaning, decimation and texturing. Every entry point
takes plain arrays, never a ``PointcloudResult``: only the caller knows which resolution grid
it is on. ``collab_splats.mesh`` exports ``compute_tsdf_voxel_size``, ``create_tsdf_mesh``,
``clean_repair_mesh``, ``prepare_mesh`` and ``create_texture_mesh``. The pipeline runs it as
the ``mesh`` stage (``mesh:`` in the yaml, ``--stages mesh``; keys in :doc:`/configuration`);
the ``04_mesh`` tutorial walks through fusion, cleaning and texturing.

Fusion
------

.. code-block:: python

   import open3d as o3d

   from collab_splats.mesh import (
       clean_repair_mesh,
       compute_tsdf_voxel_size,
       create_tsdf_mesh,
       prepare_mesh,
   )

   voxel_size = compute_tsdf_voxel_size(depths, c2w, K, depth_fx=model_fx)
   mesh = create_tsdf_mesh(
       depths,  # (N, H, W) float32, 0 = no observation
       rgbs,  # (N, H, W, 3) uint8
       c2w,  # (N, 4, 4) camera-to-world
       K,  # (N, 3, 3) pixel-center, at the depth resolution
       voxel_size=voxel_size,
   )
   cleaned, real = clean_repair_mesh(mesh, c2w)  # real: floater-cut surface, no hull or fill
   prepared = prepare_mesh(cleaned, voxel_size=voxel_size, max_faces=1_500_000)
   o3d.io.write_triangle_mesh("scene/mesh.ply", prepared)

- No file IO in ``collab_splats.mesh`` except the texture writer. The stage deletes a stale
  ``mesh.ply`` before fusing and writes it last, after texturing: ``mesh.ply`` is the done
  marker, so a run that fails partway leaves none and is re-run, never skipped.
- The voxel is derived, never set. The world has no fixed scale (feedforward depth is roughly
  unit scale, a COLMAP-scale reconstruction one to two orders of magnitude larger), so an
  absolute voxel or depth cutoff tuned on one scene empties or explodes the next.
- ``compute_tsdf_voxel_size``: ``depth_px`` (``mesh.voxel_depth_px``, ``4``) depth-pixel
  footprints, ``depth / depth_fx``, at the ``ref_percentile`` depth
  (``mesh.voxel_ref_percentile``, ``50``). ``depth_fx`` is the focal of the grid the depth was
  predicted on: the model grid for feedforward, since depth upsampled to the frame grid carries
  no finer detail. It then counts the 16³ surface blocks the voxel would allocate and coarsens
  by the square root of the overshoot until they fit ``max_gb`` (``8``).
- On GH010229 the default landed at 0.0023 on the 294-frame run (hand-tuned value 0.0025) and
  0.0056 on the 1039-frame run, whose depth is 2.4× larger.
- Depth cutoff: the stage zeroes depth beyond the ``mesh.depth_trunc_percentile`` (``99``,
  required, in (0, 99]) percentile of positive depth; left uncut, a far depth tail crashed
  Open3D's extraction. The cut happens before the voxel is sized, so depth that will not be
  fused never coarsens the voxel. ``create_tsdf_mesh``'s own ``depth_trunc=None`` keeps every
  depth.
- ``sdf_trunc`` defaults to ``4 × voxel_size``; the stage passes
  ``mesh.sdf_trunc_mult × voxel_size`` (``6.0``). A band narrower than one voxel raises.
- ``create_tsdf_mesh`` runs on CUDA when Open3D has it, else on the CPU, and raises rather than
  fusing quietly wrong (``validate_views``): non-uint8 RGB, view-count or resolution
  mismatches, and a principal point outside the depth grid. Pairing one grid's depth with the
  other grid's ``K`` otherwise collapses the mesh instead of failing (the 2026-08-11
  regression). An out-of-scale ``depth_trunc`` is not checked: Open3D treats it as a
  legitimately empty volume.

.. automodule:: collab_splats.mesh.tsdf
   :members:
   :show-inheritance:

Composing the inputs
--------------------

``mesh.source`` picks the depth: ``feedforward`` (default) or ``splats``.

**Feedforward, at frame resolution.** ``frame_depths`` (``collab_splats.pointcloud.utils``)
confidence-masks the model-grid depth and guided-upsamples it onto the frames, so the full-res
``result.intrinsics`` is the ``K`` to fuse with:

.. code-block:: python

   from collab_splats.geometry.transforms import invert_poses
   from collab_splats.pointcloud.utils import frame_depths
   from collab_splats.preproc import frame_idx_from_path, read_frames

   image_ids = [frame_idx_from_path(p) for p in result.image_paths]
   rgbs = read_frames(images_dir, image_ids)
   depths = frame_depths(result, rgbs, conf_percentile=10)
   c2w = invert_poses(result.extrinsics)
   K = result.intrinsics  # full-res; depth_fx from result.model_intrinsics

- ``result`` is the ``PointcloudResult`` from ``pointcloud.zarr``. It stores ``K`` twice:
  ``intrinsics`` on the frame grid and ``model_intrinsics`` on the model grid. Mixing one
  grid's depth with the other grid's ``K`` is the collapse bug above.
- ``conf_percentile`` is ``mesh.conf_percentile`` (``10``; ``null`` = off). A reconstruction
  with no confidence array (``sfm``) fuses unmasked and logs it.
- At model resolution (notebooks, evals): ``result.depth``, ``result.images`` transposed to
  ``(N, H, W, 3)`` and scaled to uint8, ``invert_poses(result.extrinsics)`` and
  ``result.model_intrinsics``. No lift.

**From a trained splat.** ``render_tsdf_inputs`` (``collab_splats.splats.checkpoint``) renders
every training camera from the splats stage's ``ckpt.pt``, at frame resolution, with the poses
they were rendered from, pose-opt deltas included. Nothing to lift or re-pose, and
``splats.zarr`` is not an input. The stage never auto-runs the splats stage; a missing
checkpoint raises.

.. code-block:: python

   from collab_splats.splats.checkpoint import render_tsdf_inputs

   depths, rgbs, c2w, K, image_ids = render_tsdf_inputs(scene_dir / "splats" / "ckpt.pt", images_dir)

- ``image_ids``: the source frame index of each row, in the checkpoint's render order (not
  filename order), the same values ``read_frames`` takes; per-view artifacts (sky masks,
  per-frame scores) line up through it.
- ``K`` comes back pixel-center; depth is zeroed where alpha is 0.
- ``depth_source`` picks the 2dgs depth: ``"expected"`` (default, used by the stage) is the
  alpha-weighted ``depth``, defined wherever anything contributes, at the cost of smearing
  across depth discontinuities; ``"median"`` is ``median_depth``, sharper but blank wherever no
  gaussian crosses the median, which opens holes on grazing ground. On GH010229 (853 views,
  voxel 0.10) ``"median"`` fused fewer vertices in far fewer components (3.02M / 94,937 against
  4.0M / 426,695), yet the ``"expected"`` mesh looks better: fragmentation is not the metric.
  A 3dgs checkpoint has only ``depth``.
- ``images_dir`` swaps the rendered RGB for the source keyframes matched by image id, as the
  stage does; omit it to fuse the render's own color.

**Masking sky.** Sky has no surface, but every depth source assigns it one, so it fuses as a
backdrop and seeds floaters. ``mesh.mask_sky`` (``true``) zeroes depth wherever the sky
segmenter fires, before fusion, on both sources:

.. code-block:: python

   from collab_splats.semantics.segmentation import sky_masks

   masks = sky_masks(images_dir, idxs=image_ids, cache_dir=scene_dir / "sky")  # (N, H, W) bool
   depths[masks] = 0.0

- ``idxs`` takes source frame indices in the order wanted; the stage passes each source's
  ``image_ids``. Sky probabilities cache as PNGs under ``<scene>/sky/``, so a re-run pays for
  inference once.
- Backend: ``skywater``, a SegFormer MiT-B2 registered as ``BaseSegmentation.get("skywater")``.
- Masking applies to mesh depth only, never to the splats stage's depth targets, which do take
  ``mesh.conf_percentile``: the asymmetry is deliberate.

Cleaning
--------

``clean_repair_mesh(mesh, c2w=None, ..., use_convex_hull=False)`` returns ``(cleaned, real)``:
``remove_floaters`` (in place), optionally ``make_convex_hull``, then ``fill_holes``. ``real``
is the mesh right after the floater cut, the texture occluder. Cleaning is not optional in the
pipeline; ``remove_floaters``, ``make_convex_hull``, ``fill_holes``, ``trim_mesh_edges`` and
``bridge_mesh_edges`` are public for callers who want one step alone.

Every threshold is relative to the mesh's own extent, never a world distance, so one default
works on a metric scan and on a scale-free feedforward reconstruction alike.
``get_scene_scale`` is that extent: the diagonal of the 1st-99th-percentile bounding box,
which ignores a stray far component.

- ``min_area_frac`` (``6e-6``): components with area under this × scene_scale² go; scale is
  taken from the largest component alone.
- ``max_gap_frac`` (``0.01``): components whose centroid is further than this × scene_scale
  from the main body go, however large.
- ``max_hole_perimeter_ratio`` (``0.014``): holes with a perimeter under this × scene_scale are
  filled; anything bigger is a real opening (a doorway, the missing back of the scene). On
  GH010229 (scene scale 181.4) that is a 2.5-unit perimeter.
- ``subdivide_fill`` (``True``): patches are triangulated, then one meshlib ``subdivideMesh``
  over all patches (to the mesh's edge length) and one cotan ``positionVertsSmoothly`` of their
  new vertices, so a patch does not read as a flat fan. ``False`` leaves a flat lid.
- ``max_plain_edges`` (``8``, ``fill_holes`` only): loops with at most this many edges take a
  plain flat lid; ``0`` subdivides every patch.
- Batching matters: per-hole ``fillHoleNicely`` costs ~40 ms a call at 0.0025 m, since each
  call scales with the whole mesh; on GH010229 the batched fill cut the clean stage's fill from
  159 s to 18 s at sub-millimeter p99 difference.
- A component's outer rim is never filled, whatever the bound: its widest loop, when that loop
  spans at least half the component's bounding box. A sheet's rim spans all of it; a hole in a
  closed surface spans a fraction, so a sphere with one hole still gets it filled.

**Convex hull** (``mesh.use_convex_hull``, ``true`` in ``configs/base.yaml``; set ``false``
indoors and for objects). ``make_convex_hull(mesh, c2w)`` trims the ragged outer edge of a
fused scene and patches the ground out to a rounded convex hull:

1. Frame: the expected up is the normal of the plane the camera centers span (the mean camera
   up when the path is close to a line). The ground is the first of up to 3 RANSAC planes
   (``fit_dominant_plane``, fixed seed, each failed plane's inliers removed before the next
   fit) with ``|normal · expected up| >= min_up_cos`` (``0.7``); up points to the cameras'
   side. On GH010229 (300 frames) the camera-path normal is within 2° of the ground, while the
   mean camera up is 33° off (the GoPro pitches down).
2. Hull cut: faces outside the rounded convex hull of the mesh's top-down coverage go.
3. Outline trim (``trim_mesh_edges``): faces outside the smoothed outline that connect to the
   outer rim go; interior patches stay.
4. Ground patch: uncovered hull cells get a triangulated sheet, pinned to the rim and pulled
   toward the eroded ground height away from the mesh; rim triangles reaching up a wall or a
   spike are dropped.
5. Join: patch faces that would make a non-manifold or fold-over edge are dropped, then
   ``make_manifold``.
6. Neck bridges (``bridge_mesh_edges``): rim vertices close in 3D but far apart along the
   outer loop are bridged, so inlets become interior holes ``fill_holes`` then closes.

Lengths are in cells of the top-down image, one median edge length each (about one TSDF
voxel). It assumes a ground-dominated height field; a mesh with no dominant ground raises
``ValueError`` rather than inventing one.

Preparing mesh.ply
------------------

The stage always ships the prepared mesh, textured or not. ``prepare_mesh(mesh, *,
voxel_size, ...)`` returns a new mesh and leaves its input unmodified:

1. ``fill_holes`` at full density with ``max_hole_perimeter_ratio`` (``3.9``): outer rims stay
   open by rule, so the bound only stops the largest interior openings (on GH010229, one hole
   at 4.0 × scene scale). With the convex hull on, more inlets are interior holes by now.
2. ``decimate_mesh``: meshlib QEM to an error bound in voxels (``decimate_max_error``, ``0.5``
   × ``voxel_size``), so the budget follows the fusion resolution, not a triangle count; past
   ``max_faces`` (``mesh.max_faces``, ``1500000``) a second pass decimates on to that count
   whatever error it costs, so texturing cost stays bounded.
3. ``make_manifold``: split non-manifold vertices, drop degenerate, duplicate and fold-over
   faces.
4. ``fill_holes`` again with ``subdivide_fill=False``, then ``make_manifold``: flat lids over
   the pinholes decimation and repair open, then a repair of what the lids fold.
5. ``smooth_iterations`` > 0 only (``mesh.smooth_iterations``, ``30``; function default ``0``):
   meshlib ``relaxKeepVolume``, then ``make_manifold``. Last, so decimation never sees it;
   vertices move, faces stay. GH010229 300 frames at ``sdf_trunc_mult`` 6: 30 passes take 10 s,
   face-to-neighbor normal angle 11.3° -> 2.2° (Taubin 10 reached 3.8°).

Filling only after decimation leaves fold-over faces; this order was the best of four on
GH010229 (16.73 dB reprojection PSNR).

.. automodule:: collab_splats.mesh.clean
   :members:
   :show-inheritance:

Texturing
---------

``create_texture_mesh(mesh, occluder, out_dir, rgbs, c2w, K, *, voxel_size)`` bakes the views
into one albedo atlas (``mesh.texture``, ``true``; CUDA only, the stage refuses before fusion
without it). ``mesh`` is the prepared mesh, unwrapped as is; ``occluder`` is the ``real`` mesh
``clean_repair_mesh`` returns, so hole-fill patches never hide a real surface. The stage writes
to ``<backend>/texture/``.

1. ``unwrap_view_charts``: the source images are the charts. Each face takes the view in which
   it owns at least 75% (``frac``) of its projected pixels in that view's nvdiffrast face-id
   buffer, smoothed toward its neighbors' views; its UVs are its pixel coordinates there. The
   z-buffer gives each pixel to one face, so faces in one camera chart cannot overlap. Unseen
   faces get one flat patch per connected group. Charts are tiled and packed; flipped or
   overwritten faces regroup into flat patches and are repacked (~23 s on 1.1M faces, against
   ~30 min for the UVAtlas it replaced).
2. ``project_images_to_texture``: nvdiffrast rasterizes the atlas into per-texel world position
   and normal. Occlusion is a per-view nvdiffrast depth render of the occluder: a texel is
   hidden when its nearest raster pixel holds a surface nearer by more than a voxel. Bilinear
   samples (``grid_sample``) are weighted by pixel size at the texel. Each image splits into a
   Gaussian-blurred base (``blur_pix_sigma=16`` source pixels) and its detail; detail comes
   only from views that resolve the texel nearly as finely as its best (``view_ratio=1.5``),
   the base averages every view that sees it, so per-frame exposure does not step where the
   sharp views change. This replaced a per-view gain solve whose gains drifted along the
   sequence (0.59-1.61 on GH010229). ``fill_missing_pixels`` (push-pull) fills every texel no
   view reached.
3. ``_dilate_chart_gutters``: chart edge texels grow into their own padding, so bilinear
   lookups never pull in a neighboring chart.
4. ``write_textured_obj`` (``collab_splats.utils.io``): ``mesh.obj`` + ``mesh.mtl`` +
   ``albedo.png``. Each position and smooth normal is written once and faces index
   ``v``/``vt``/``vn`` separately, at 6 decimals (~2.8× smaller on GH010229: 409 -> 146 MB).
   UV seams add ``vt`` entries, never positions: the OBJ's ``v`` lines are the prepared mesh's
   vertices. White diffuse (``Kd 1 1 1``): a grey ``Kd`` darkens the texture in every viewer.

Open3D's projection path was tried first and does not work here: CPU-only, 24 s per view,
OOM-killed at 8 views. An NVIDIA Warp ray-cast replaced it, then gave way to the depth-render
test: same held-out scores on GH010229, one dependency fewer.

.. automodule:: collab_splats.mesh.texture
   :members:
   :show-inheritance:

Vertex features
---------------

The semantics stage (after mesh) lifts features straight onto the vertices with
``collab_splats.semantics.lifting.lift_features``, passing a result whose ``points`` are the
vertices and no ``pixel_indices`` (decision 021): each vertex projects into every frame and
averages the depth-consistent samples; a vertex no frame sees stays zero (unobserved). They are
stored as ``vertex_features`` in ``<scene>/<backend>/semantics/<extractor>_lifted.zarr``, which
``python -m collab_splats.viewer <scene>/<backend>`` shows on the mesh.

- ``transfer_features(targets, points, features, k=5, max_dist=0.03)``: Gaussian-weighted
  k-NN scatter of point features onto targets; pass the points themselves to smooth.
- ``collab_splats.semantics.utils.cluster_points``, given the vertex positions, groups
  high-scoring vertices into spatially connected clusters.

Utilities
---------

``to_meshlib`` / ``from_meshlib`` convert to and from meshlib; ``face_edge_ids``,
``adjacent_face_pairs``, ``face_components`` and ``face_areas`` give face connectivity;
``validate_views`` is the view-array contract shared by ``create_tsdf_mesh`` and
``create_texture_mesh``.

.. automodule:: collab_splats.mesh.utils
   :members:
   :show-inheritance:
