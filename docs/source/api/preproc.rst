Preprocessing
=============

Video preprocessing: the canonical keyframe store (``frames``), PyAV decode and
video metadata (``video``), capture-quality measurement (``qa``), keyframe
selection (``sampling``) and lens calibration (``undistort``).

Quickstart
----------

The preprocess stage decodes the video once into ``<output_path>/images/frame_NNNNNN.png``:
lossless PNGs named by source frame index. Selection settings land in
``<backend>/run_config.yaml``, per-frame quality in ``video_quality_report.json``. Every pixel
consumer reads this directory; nothing re-decodes the video.

.. code-block:: python

   from collab_splats.preproc import frame_paths, read_frames

   paths = frame_paths(run_dir / "images")       # sorted, one path per selected frame
   imgs = read_frames(run_dir / "images")        # (N, H, W, 3) uint8 RGB

Frame arrays crossing the public API are RGB; ``analysis_gray`` takes BGR.

See :doc:`../tutorials/01_preprocessing/preprocessing` for the frame-selection walkthrough
(fps vs. uniform vs. optical-flow sampling, blur/exposure gating). Sampling plots live in
``collab_splats.preproc.viz`` (notebook-only, not documented here).

.. automodule:: collab_splats.preproc.frames
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.video
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.qa
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.sampling
   :members:
   :show-inheritance:

.. automodule:: collab_splats.preproc.undistort
   :members:
   :show-inheritance:
