Preprocessing
=============

Video preprocessing: the canonical keyframe store, ffmpeg decode and video
metadata (``video``), and quality gating plus keyframe selection (``sampling``).

.. automodule:: collab_splats.preproc.frame_store
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.video
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

- **No ``frames/`` JPG dir is written.** ``frames.zarr`` is the sole frame
  store: ``creator.setup_inference`` takes the store path directly, and
  semantic feature extraction streams from it via
  ``BaseFeatureExtractor.extract_and_cache_from_zarr`` — neither consumer is
  path-locked any more, so the JPG export (and the ``_write_frames_jpegs``
  helper that produced it) is gone. Path-locked third-party tools get
  transient JPGs on demand from ``FrameStore.export(tmp_dir)``. Dashboard
  localization reference thumbnails (``_build_result_figures`` in
  ``collab_splats/dashboard/localize.py``) read pixels from ``frames.zarr``
  via ``plot_correspondences(..., frames_zarr=...)`` for reconstruction-
  sourced frames; ``localized`` frames (appended post-hoc, never written to
  ``frames.zarr``) still read from ``localized_frames/`` on disk.
