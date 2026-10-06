Geometry
========

Pose and geometry backend: transforms, bundle adjustment, scene metrics, and
loop closure.

Transforms
----------

Pose conversions, Umeyama alignment and intrinsics helpers, including
``project_to_so3``, ``decompose_camera`` and ``intrinsics_4x4``.

.. automodule:: collab_splats.geometry.transforms
   :members:
   :show-inheritance:

Bundle adjustment
-----------------

``BundleAdjustment.refine`` takes arrays (images, confidence, world points,
extrinsics, the result's ``model_intrinsics``) and returns ``(extrinsics, intrinsics)``, K
on the model grid.

.. automodule:: collab_splats.geometry.bundle_adjustment
   :members:
   :show-inheritance:

Projection
----------

Pinhole unprojection and projection, ``unproject_frames`` (batched numpy
unprojection on the device), the cross-view depth residual, and
``multiview_depth_confidence``, the per-pixel agree/seen counts the
feedforward multiview filter thresholds.

.. automodule:: collab_splats.geometry.projection
   :members:
   :show-inheritance:

Metrics
-------

``compute_reconstruction_quality`` is the entry point; the Reconstructor stage owns
the zarr load and the ``reconstruction_quality_report.json`` write.

.. automodule:: collab_splats.geometry.metrics
   :members:
   :show-inheritance:

Report columns
~~~~~~~~~~~~~~

Report-only, no verdicts; scale-free (1 recon unit is not 1 meter). The model grid
is the backbone's depth grid; the original grid is the source image. ``null``
marks a value that does not exist, never a failure.

.. list-table:: ``frames`` — one row per reconstruction frame
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``frame_idx``
     - source video frame index; null off the ``frame_{idx:06d}`` stem contract
   * - ``covered_fraction``
     - crop area / original canvas area, 0..1, original grid
   * - ``median_abs_rel_depth_error``
     - median ``|s - 1|`` over the depth pairs touching the frame, model grid
   * - ``multiview_agreement``
     - share of the frame's seen pixels that at least one other view agrees
       with at ``rel_thresh`` 0.05, model grid; null when no other view sees it
   * - ``confidence_median``
     - median per-pixel confidence, model grid; backbone-native, not comparable
       across backbones; null without a confidence array

.. list-table:: ``depth_pairs`` — model grid, one row per ordered direction
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``idx1``, ``idx2``
     - reconstruction frame positions
   * - ``n_pixels``
     - model-grid pixels compared
   * - ``median_depth``
     - median depth of the compared pixels, recon units, scale-free
   * - ``median_rel_depth_error``
     - signed scale offset ``s - 1``, dimensionless
   * - ``iqr_rel_depth_error``
     - spread of the relative residual with the bias removed, dimensionless
   * - ``median_parallax_deg``
     - median parallax of the compared pixels, degrees

``depth_residual_histogram`` (model grid): ``counts`` and ``bin_edges`` over
``r / (1 + |r|)``; invert an edge with ``u / (1 - |u|)``.

.. list-table:: ``photometric_pairs`` — original grid, one row per pair ``i < j``; null without ``images/``
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``idx1``, ``idx2``
     - reconstruction frame positions
   * - ``photometric_ncc``
     - zero-mean NCC of the warped RGB, -1..1
   * - ``n_pixels``
     - original-grid pixels compared

Loop closure
------------

The ``collab_splats.geometry.loop_closure`` package exports ``LoopClosure``,
``LoopClosureConfig``, ``PoseGraph`` and ``Submap``; import anything else from
its submodules.

.. automodule:: collab_splats.geometry.loop_closure.graph
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.loop_closure.submap
   :members:
   :show-inheritance:

Quickstart
~~~~~~~~~~

.. code-block:: python

   from pathlib import Path

   from collab_splats.pointcloud import get_creator
   from collab_splats.geometry.loop_closure import LoopClosure, LoopClosureConfig

   base = get_creator("vggtx")()
   lc = LoopClosure(base, config=LoopClosureConfig())
   result = lc.create_pointcloud(Path("path/to/images"), Path("path/to/out"), Path("path/to/out/colmap/sparse/0"))

See ``docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`` for the full
walkthrough (candidate matching, plots).

.. automodule:: collab_splats.geometry.loop_closure.wrapper
   :members:
   :show-inheritance:
