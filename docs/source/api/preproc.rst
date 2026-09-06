Preprocessing
=============

Video preprocessing: the canonical keyframe store (``frames``), ffmpeg decode and
video metadata (``video``), capture-quality measurement (``qa``), and keyframe
selection (``sampling``).

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

Quickstart
----------

The preprocess stage decodes the source video exactly once and writes
``output_path/images/frame_NNNNNN.png`` — a COLMAP-style directory of lossless
PNGs named by SOURCE video frame index — beside ``output_path/frames.json``,
which holds the selection records and the provenance COLMAP has no slot for.
This is the sole persistent frame artifact. Every pixel consumer (pointcloud,
semantics, localization, dashboard) reads the directory instead of re-decoding
the video, and path-locked consumers take the directory itself, so nothing
stages a second copy.

.. code-block:: python

   from collab_splats.preproc import frame_paths, read_frames, read_manifest

   paths = frame_paths(run_dir / "images")       # sorted, one path per selected frame
   imgs = read_frames(run_dir / "images")        # (N, H, W, 3) uint8 RGB
   manifest = read_manifest(run_dir / "images")  # selection records + provenance

Every function in this module takes and returns **RGB**; the BGR conversions
``cv2`` needs happen inside ``frames.py`` and nowhere else.

``pointcloud.zarr`` (written by the pointcloud stage) keeps its own
model-resolution ``images`` tensor — that is not a duplicate of the keyframe
directory; it is used directly by mesh extraction, bundle adjustment, and point
colorization at the model's inference resolution.

See ``docs/source/tutorials/01_preprocessing/keyframe_extraction.ipynb`` for
the full frame-selection walkthrough (uniform vs. optical-flow sampling,
blur/exposure gating).

Migration
---------

Scenes written before this format hold a ``frames.zarr`` store and no ``images/``
directory. ``read_manifest`` raises ``FileNotFoundError`` on them rather than
falling back to the old store. Convert one without re-decoding the video::

   python scripts/migrate_frames_zarr.py <scene_dir> [<scene_dir> ...]
   python scripts/migrate_frames_zarr.py --all <processed_root>

Known limitations
------------------

- **Dashboard localization thumbnails read two directories.** Reference frames
  that came from the reconstruction live in ``images/``; ``localized`` frames are
  appended post-hoc and are never written there, so they are read from
  ``localized_frames/`` instead (``_build_result_figures`` in
  ``collab_splats/dashboard/localize.py``).
