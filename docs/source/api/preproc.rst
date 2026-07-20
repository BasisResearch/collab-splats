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

- **``frames/`` JPG dir still exists, by design.** ``_write_frames_jpegs``
  (`collab_splats/dashboard/pipeline.py`) still writes an
  ``out_dir/frames/`` directory because ``creator.setup_inference`` and
  semantic feature extraction are path-locked consumers that require a real
  on-disk image directory. Filenames use the source ``frame_idx`` (matching
  ``FrameStore.frame_idx_from_path``'s convention), so they stay addressable
  from both the JPG dir and ``frames.zarr``. Dashboard localization
  reference thumbnails (``_build_result_figures`` in
  ``collab_splats/dashboard/localize.py``) read pixels from ``frames.zarr``
  via ``plot_correspondences(..., frames_zarr=...)`` for reconstruction-
  sourced frames; ``localized`` frames (appended post-hoc, never written to
  ``frames.zarr``) still read from ``localized_frames/`` on disk.
