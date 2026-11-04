Mesh
====

TSDF meshing from posed depth + RGB, then cleaning, decimation and texturing. Every entry point
takes plain arrays, never a ``PointcloudResult``: only the caller knows which resolution grid
it is on. ``collab_splats.mesh`` exports ``compute_tsdf_voxel_size``, ``calibrate_confidence``,
``pixel_weights``, ``lift_weights``, ``build_volume_ground``, ``solve_ground``,
``VolumeGround``, ``create_tsdf_mesh``, ``clean_repair_mesh``, ``prepare_mesh`` and ``create_texture_mesh``. The pipeline runs it as
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
       weights=weights,  # (N, H, W) float32, 0 = skip the pixel
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
- Depth cutoff: before sizing the voxel, the stage zeroes depth beyond the
  ``mesh.depth_trunc_percentile`` (``99``, in (0, 99]) percentile of positive depth; a far tail crashes Open3D.
- ``sdf_trunc`` defaults to ``4 × voxel_size``; the stage passes
  ``mesh.sdf_trunc_mult × voxel_size`` (``6.0``). A band narrower than one voxel raises.
- Fusion is weighted per pixel. Each sample adds its weight to the voxel; tsdf and color are
  weighted running averages, and surface is extracted only where the summed weight exceeds
  ``w_min`` (``1.0``). One view at weight 1 is not enough on its own.
- Weights come from ``pixel_weights``: ``(conf / c*)^(2γ)`` above the gate ``c*``, 0 below it,
  0 where depth is 0, and 0 on depth edges (3x3 relative jump above ``edge_rtol`` ``0.03``).
  Without confidence (sfm, splats renders) every pixel off an edge weighs 1.
- ``c*`` and ``γ`` are fitted per scene by ``calibrate_confidence`` and are never configured.
  The fit uses cross-view depth error: the median residual against views ±2, ±5 and ±10, with
  occluded samples dropped, binned by confidence and fitted as a log-log line. ``c*`` is where
  the error equals the run's ``sdf_trunc``, and ``γ`` is the negated slope. So the voxel size is
  computed first. Projection runs at full fp32 matmul precision (a global TF32 would round the
  residuals). Percentile gates were rejected: they do not equalize scenes (GH010238's p20 is
  confidence 29).
- With ``ground=`` (from ``build_volume_ground``), voxels inside the ground hull that the views
  left at summed weight ``<= w_min`` take the ground's signed height distance at weight
  ``1.01 w_min``. The ground is an obstacle membrane: the lowest surface under every observed
  height (10th percentile per 4-voxel cell), pushed up into the gaps by a pressure
  ``kappa / scene scale`` (``kappa`` 30). Observed voxels are never overwritten, so scene and
  ground come out of marching cubes as one surface, with no stitching. The stage builds it only
  when ``mesh.use_convex_hull`` is on.
- ``create_tsdf_mesh`` runs on CUDA when Open3D has it, else on the CPU, and raises rather than
  fusing quietly wrong (``validate_views``): non-uint8 RGB, view-count or resolution
  mismatches, and a principal point outside the depth grid. Pairing one grid's depth with the
  other grid's ``K`` otherwise collapses the mesh instead of failing. An out-of-scale
  ``depth_trunc`` is not checked: Open3D treats it as a legitimately empty volume.

Composing the inputs
~~~~~~~~~~~~~~~~~~~~

``mesh.source`` picks the depth: ``feedforward`` (default) or ``splats``.

**Feedforward, at frame resolution.** ``frame_depths`` (``collab_splats.pointcloud.utils``)
guided-upsamples the model-grid depth onto the frames (``conf_percentile=None``: every pixel),
so the full-res ``result.intrinsics`` is the ``K`` to fuse with. The weights are computed on the
model grid and lifted with ``lift_weights`` (nearest, through each crop box):

.. code-block:: python

   from collab_splats.geometry.transforms import invert_poses
   from collab_splats.mesh import (
       build_volume_ground,
       calibrate_confidence,
       lift_weights,
       pixel_weights,
   )
   from collab_splats.pointcloud.utils import frame_depths
   from collab_splats.preproc import frame_idx_from_path, read_frames

   image_ids = [frame_idx_from_path(p) for p in result.image_paths]
   rgbs = read_frames(images_dir, image_ids)
   depths = frame_depths(result, rgbs, conf_percentile=None)
   c2w = invert_poses(result.extrinsics)
   K = result.intrinsics  # full-res; depth_fx from result.model_intrinsics
   # voxel_size: compute_tsdf_voxel_size above; 6.0 is mesh.sdf_trunc_mult
   c_star, gamma = calibrate_confidence(
       result.depth, result.confidence, result.extrinsics, result.model_intrinsics,
       sdf_trunc=6.0 * voxel_size,
   )
   weights = pixel_weights(result.depth, result.confidence, c_star=c_star, gamma=gamma)
   weights = lift_weights(weights, result.original_coords, depths.shape[1:])
   ground = build_volume_ground(
       depths, weights, rgbs, c2w, K,
       voxel_size=voxel_size, sdf_trunc=6.0 * voxel_size,
   )

Pass ``weights=weights`` and ``ground=ground`` to ``create_tsdf_mesh``. The stage orders this as:
sky zeroes depth, depth is cut at ``depth_trunc_percentile``, the voxel is sized, then
calibration (on the model grid, sky included), weights, sky zeroed on the model grid, lift,
ground, fuse.

- ``result`` is the ``PointcloudResult`` from ``pointcloud.zarr``. It stores ``K`` twice:
  ``intrinsics`` on the frame grid and ``model_intrinsics`` on the model grid. Mixing one
  grid's depth with the other grid's ``K`` is the collapse above.
- ``mesh.conf_percentile`` is gone: a config that still sets it has it dropped on load with a
  warning. A reconstruction with no confidence array (``sfm``) fuses with unit weights and logs
  it.
- With ``mesh.mask_sky`` the stage zeroes the sky weights on the model grid before the lift, so
  sky fuses nothing; calibration still sees it.
- At model resolution (evals): ``result.depth``, ``result.images`` transposed to
  ``(N, H, W, 3)`` and scaled to uint8, ``invert_poses(result.extrinsics)`` and
  ``result.model_intrinsics``. No lift: ``pixel_weights`` runs directly on the model grid and
  its weights need no lift either.

**From a trained splat.** ``render_tsdf_inputs`` (``collab_splats.splats.checkpoint``) renders
every training camera from the splats stage's ``ckpt.pt``, at frame resolution, with the poses
they were rendered from, pose-opt deltas included. Nothing to lift or re-pose. The stage never auto-runs the splats stage; a missing
checkpoint raises.

.. code-block:: python

   from collab_splats.splats.checkpoint import render_tsdf_inputs

   depths, rgbs, c2w, K, image_ids = render_tsdf_inputs(backend_dir / "splats" / "ckpt.pt", images_dir)

- ``image_ids``: the source frame index of each row, in the checkpoint's render order (not
  filename order), the same values ``read_frames`` takes; per-view artifacts (sky masks,
  per-frame scores) line up through it.
- ``K`` comes back pixel-center; depth is zeroed where alpha is 0.
- ``depth_source`` picks the 2dgs depth: ``"expected"`` (default, used by the stage) is the
  alpha-weighted ``depth``, defined wherever anything contributes but smeared across depth
  discontinuities; ``"median"`` is ``median_depth``, sharper but blank wherever no gaussian
  crosses the median, which opens holes on grazing ground. A 3dgs checkpoint has only
  ``depth``.
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
- Masking applies to mesh depth only, never to the splats stage's depth targets, which take
  ``splats.depth_conf_percentile``: the asymmetry is deliberate.

.. automodule:: collab_splats.mesh.tsdf
   :members:
   :show-inheritance:

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
  filled; anything bigger is a real opening (a doorway, the missing back of the scene).
- ``subdivide_fill`` (``True``): patches are triangulated, then one meshlib ``subdivideMesh``
  over all patches (to the mesh's edge length) and one cotan ``positionVertsSmoothly`` of their
  new vertices, so a patch does not read as a flat fan. ``False`` leaves a flat lid.
- ``max_plain_edges`` (``8``, ``fill_holes`` only): loops with at most this many edges take a
  plain flat lid; ``0`` subdivides every patch.
- A component's outer rim is never filled, whatever the bound: its widest loop, when that loop
  spans at least half the component's bounding box. A sheet's rim spans all of it; a hole in a
  closed surface spans a fraction, so a sphere with one hole still gets it filled.

**Convex hull** (``mesh.use_convex_hull``, ``true`` in ``configs/base.yaml``; set ``false``
indoors and for objects). ``make_convex_hull(mesh, c2w)`` finds the ground plane from the camera
path and right vectors. When no RANSAC plane faces the camera up, the largest plane is the
ground and a warning is logged; steps are in its docstring below.

- ``make_convex_hull`` cuts the mesh to a rounded convex hull of its ground, trims the ragged
  outer edge, drops small pieces, makes it manifold and bridges inlets. It adds no vertices: the
  ground inside the hull comes from the volume ground at fusion. A stitched ground patch was
  rejected; it attached the ground to random high points.

Preparing mesh.ply
~~~~~~~~~~~~~~~~~~

The stage always ships the prepared mesh, textured or not. ``prepare_mesh(mesh, *,
voxel_size, ...)`` returns a new mesh and leaves its input unmodified:

1. ``fill_holes`` at full density with ``max_hole_perimeter_ratio`` (``3.9``): outer rims stay
   open by rule, so the bound only stops the largest interior openings. With the convex hull
   on, more inlets are interior holes by now.
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
   vertices move, faces stay.

Filling only after decimation leaves fold-over faces.

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
   overwritten faces regroup into flat patches and are repacked.
2. ``project_images_to_texture``: nvdiffrast rasterizes the atlas into per-texel world position
   and normal. Occlusion is a per-view nvdiffrast depth render of the occluder: a texel is
   hidden when its nearest raster pixel holds a surface nearer by more than a voxel. Bilinear
   samples (``grid_sample``) are weighted by pixel size at the texel. Each image splits into a
   Gaussian-blurred base (``blur_pix_sigma=16`` source pixels) and its detail; detail comes
   only from views that resolve the texel nearly as finely as its best (``view_ratio=1.5``),
   the base averages every view that sees it, so per-frame exposure does not step where the
   sharp views change. ``_color_correction`` then brightens night footage: a bake whose
   seen-texel median luminance is below ``mesh.texture_brightness`` (``0.4``; ``null`` = off) is
   raised by one luminance gamma onto it, RGB scaled by the luminance ratio, in float before
   quantization. It is a floor, not a standard: brighter bakes are never darkened, because a
   daytime median (0.34-0.56 across our scenes) is surface color, and a gamma above 1 crushes
   shadows. It runs at the bake, not on the frames: lifting C0125's frames before the pipeline
   left depth confidence unchanged and made cross-frame depth agreement slightly worse.
   ``fill_missing_pixels`` (push-pull) fills every texel no view reached.
3. ``_dilate_chart_gutters``: chart edge texels grow into their own padding, so bilinear
   lookups never pull in a neighboring chart.
4. ``write_textured_obj`` (``collab_splats.utils.io``): ``mesh.obj`` + ``mesh.mtl`` +
   ``albedo.png``. Each position and smooth normal is written once and faces index
   ``v``/``vt``/``vn`` separately, at 6 decimals. UV seams add ``vt`` entries, never
   positions: the OBJ's ``v`` lines are the prepared mesh's vertices. White diffuse
   (``Kd 1 1 1``): a grey ``Kd`` darkens the texture in every viewer.

.. automodule:: collab_splats.mesh.texture
   :members:
   :show-inheritance:

Vertex features
---------------

The semantics stage (after mesh) lifts features straight onto the vertices with
``collab_splats.semantics.lifting.lift_features``, passing a result whose ``points`` are the
vertices and no ``pixel_indices``: each vertex projects into every frame and
averages the depth-consistent samples; a vertex no frame sees stays zero (unobserved). They are
stored as ``vertex_features`` in ``<scene>/<backend>/semantics/<extractor>_lifted.zarr``, which
``python -m collab_splats.viewer local <scene>/<backend>`` shows on the mesh.

- ``collab_splats.semantics.lifting.transfer_features(targets, points, features, k=5, max_dist=0.03)``: Gaussian-weighted
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
