Preprocessing
=============

Video preprocessing: decode-once frame extraction, quality gating, and the
canonical keyframe store.

.. automodule:: collab_splats.preproc.frame_store
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.sampling
   :members:
   :show-inheritance:

Quickstart
----------

The preprocess stage decodes the source video exactly once and writes
``output_path/frames.zarr`` — chunked-per-frame RGB images, columnar
selection records, and provenance attrs. This is the sole persistent frame
artifact; there is no ``output_path/images/`` JPG directory. Every pixel
consumer (pointcloud, semantics, localization, dashboard) reads frames from
this store instead of re-decoding the video. Path-locked consumers (e.g.
model preprocessing that requires a real image directory) use
``FrameStore.export()`` to materialize a transient JPG dir that the caller
deletes after use.

.. code-block:: python

   from collab_splats.preproc import FrameStore

   store = FrameStore.open(run_dir / "frames.zarr")
   img = store.image_by_frame_idx(120)   # (H,W,3) RGB, by source video index
   paths = store.export(tmp_dir)          # transient JPGs for path-locked tools

``feedforward.zarr`` (written by the pointcloud stage) keeps its own
model-resolution ``images`` tensor — that is not a duplicate of
``frames.zarr``; it is used directly by mesh extraction, bundle adjustment,
and point colorization at the model's inference resolution.

See ``docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`` for
the full frame-selection walkthrough (uniform vs. optical-flow sampling,
blur/exposure gating).

Known limitations
------------------

- **transforms.json still points at ``../images/``.** ``Reconstructor``
  writes nerfstudio's ``file_path: "../images/{name}"`` for each frame
  (`collab_splats/wrapper/reconstructor.py`) even though that directory no
  longer exists. This is write-only and cosmetic — nothing in this repo
  reads ``transforms.json`` back (``ns-train`` is driven from its own
  images dir, not this file). Left dangling by design; repoint it only if
  a workflow ever runs ``ns-train`` directly against a backend output dir.
- **Dashboard localization reference thumbnails are not yet migrated.**
  ``collab_splats/dashboard/pipeline.py``'s ``_local_ref_paths`` still reads
  reference-frame thumbnails from an ``out_dir/frames/`` JPG dir written by
  ``_write_frames_jpegs``, rather than from ``frames.zarr``. Tracked as a
  follow-up in the frame-store work.
