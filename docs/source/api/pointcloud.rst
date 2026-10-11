Point Cloud
===========

``collab_splats.pointcloud`` builds a ``PointcloudResult`` (points, poses, depth, per-pixel
provenance) from the scene's ``images/`` keyframes through one of two backend families, picked by
``pointcloud.method``: ``feedforward`` (``vggt_omega`` default, ``vggtx``, ``mapanything``,
``loger``) or ``sfm`` (``instantsfm``, ``colmap``, ``hloc``, each with VDA metric depth). The
``pointcloud`` stage writes ``<backend>/pointcloud.zarr``, the COLMAP model under
``<backend>/colmap/sparse/0`` and ``<backend>/sparse_pc.ply``. See :doc:`/configuration` for the
full config table.

.. automodule:: collab_splats.pointcloud.base
   :members:
   :show-inheritance:

Bundle adjustment with loop closure
-----------------------------------

Feedforward only; the solver itself is in :doc:`geometry`. With ``pointcloud.loop_closure`` and
``pointcloud.bundle_adjustment.enabled`` both on, ``pointcloud.bundle_adjustment.strategy`` picks
the solve:

- ``global`` (default): plain LC windows, then the ``refine`` stage solves every frame at once
- ``window``: BA inside each loop-closure window, right after the window's forward pass and before
  its submap enters the pose graph; ``refine`` is dropped from the default stage set, and naming it
  explicitly raises ``ValueError``
- ``window+global``: both, in that order

Window solves:

- the first window refined with ``refine_focal`` sets the focal; later windows hold it fixed
- a window whose solve raises ``ValueError`` keeps its feedforward poses, with a warning
- each window's record (``start``, ``n_frames``, ``ok``, ``alignment_scale``, ``loss_final``,
  ``focal``, ``seconds``, ``gpu_max_mib``) lands in the ``pointcloud.zarr`` attrs under
  ``window_ba``
- the LC store carries model-grid images and per-point source pixels from each frame's owning
  submap, so ``refine`` reads it like a single-pass store

Depth alignment (sfm path)
--------------------------

``depth.py`` supplies the sfm backends' dense depth:

- ``estimate_depth`` runs Video-Depth-Anything per keyframe at ``depth_width`` 518 (the
  feedforward model grid), cached as ``<backend>/depth_vda/images/npy/<stem>.npy``; a complete
  cache is loaded as is, a partial one is deleted and recomputed
- ``align_depth`` scales each frame's depth to the COLMAP world: median ratio of COLMAP to VDA
  depth over the frame's track observations; frames with fewer than ``min_obs`` (20) use the
  global median
- the result has no ``confidence`` (``None``): sfm has no per-pixel confidence
- weights: **Metric-Video-Depth-Anything-Large**, **cc-by-nc-4.0** (non-commercial); Metric-Small
  (Apache-2.0) is the commercial-safe alternative, not taken here

.. automodule:: collab_splats.pointcloud.depth
   :members:
   :show-inheritance:

Backends
--------

Feedforward backends are registered on ``BaseFeedforwardCreator``; their config block is
``pointcloud.<backend>``, passed as creator keyword arguments.

.. automodule:: collab_splats.pointcloud.feedforward.base
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.feedforward.vggtx
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.feedforward.vggt_omega
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.feedforward.mapanything
   :members:
   :show-inheritance:

``pointcloud.method: sfm`` dispatches ``pointcloud.backend`` through ``SFM_CREATORS``:
``instantsfm`` (global), ``colmap`` and ``hloc`` (incremental). Every sfm backend:

- is experimental: the stage emits a ``UserWarning``, and its numbers are not yet measured
- refuses ``loop_closure`` (``ValueError`` at config validation: no sequential submaps)
- skips bundle adjustment: ``refine`` is dropped from the default stage set (the mapper runs its
  own BA), and naming it explicitly raises ``ValueError``
- reads the scene-root ``images/`` store in place, so COLMAP image names are the keyframe stems
- shares ``pairing``, ``overlap`` (10), ``num_retrieved`` (20), ``num_threads`` (8) and
  ``min_registered_frac`` (0.5) from ``BaseSfmCreator``

Downstream stages (``mesh``, ``splats``, ``semantics``, ``localize``,
``reconstruction_quality_report``) read ``pointcloud.zarr`` unchanged. Consumers of
``confidence`` handle its absence: mesh and splats depth targets keep every pixel, with a log line
when ``mesh.conf_percentile`` is set; the feature lift uses uniform weights.

.. automodule:: collab_splats.pointcloud.sfm
   :no-members:

.. autodata:: collab_splats.pointcloud.sfm.SFM_CREATORS
   :no-value:

   Backend name to creator class; ``Reconstructor.pointcloud`` dispatches on it.

The ``loger`` backend
~~~~~~~~~~~~~~~~~~~~~

LoGeR is a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window inference.
Requires ``bash setup/loger.sh`` once (vendors ``Junyi42/LoGeR`` @ ``7685b7a``); weights download
from HuggingFace on first use.

- **choose it for:** sequences past the ~300-frame ceiling where ``vggt_omega`` OOMs, and long
  captures where drift accumulates; the TTT memory carries state across the sequence
- **avoid it for:** short sequences (reasoning, not measured: below a couple of windows the
  set-based VGGT models see every frame jointly; no crossover swept); anything needing loop
  closure, which ``loger`` refuses (no LC thresholds calibrated for it); captures whose
  intrinsics genuinely vary, e.g. zoom; unordered image collections (LoGeR's windows are
  sequential, the VGGT family is set-based)

**Intrinsics differ from every other backend.** VGGT-family backends and MapAnything *predict* K.
LoGeR does not: ``estimate_intrinsics_from_points`` *solves* one K from the camera-frame pointmap
by a confidence-weighted median pinhole fit, shared across all frames. A predicted K can be
geometrically invalid (principal point outside the image); a fitted K is center-principal by
construction but can be plausibly-but-globally wrong. There is no fallback focal: a degenerate
fit raises ``RuntimeError``.

**preproc.max_frames is not tuned for LoGeR.** The default 300 is ``vggt_omega``'s GPU limit and
lives in the preproc stage, which runs first; raise it to use LoGeR's windowing. The real ceiling
is ``PointcloudResult``, which holds dense per-frame images, world points, depth and confidence:
8.13 MB/frame at LoGeR's default ``pixel_limit`` of 255,000, against the 46.6 GB container cap.
The cap binds every backend; LoGeR is merely the first able to feed it enough frames to matter.

.. automodule:: collab_splats.pointcloud.feedforward.loger
   :members:
   :show-inheritance:

The ``instantsfm`` backend
~~~~~~~~~~~~~~~~~~~~~~~~~~

Global SfM instead of a feedforward model: a pycolmap SIFT database (``sift_db.py``, replacing
upstream's own database step), then InstantSfM's global mapper (rotation averaging, global
positioning, global BA) with VDA depth as priors. A split solve keeps cluster 0 only.

.. code-block:: yaml

   pointcloud:
     method: sfm
     backend: instantsfm
     instantsfm:
       pairing: exhaustive          # this backend's default; see SIFT database and pairing
       retriangulation: false       # retriangulate + re-run BA after the global solve
       random_seed: null            # null = unseeded, runs differ
       min_num_view_per_track: null # null = upstream's 3; higher lowers BA memory
       min_registered_frac: 0.5

**Install.** ``setup.sh`` installs ``instantsfm`` from a pinned git commit (``d3e599e``) with
``--no-deps`` (upstream pins ``numpy==1.26.4``; the lock runs numpy 2.x), plus ``pyceres==2.3``,
``scikit-sparse==0.4.15`` (needs ``libsuitesparse-dev``) and ``easydict==1.13``. It clones
Video-Depth-Anything into ``third_party/Video-Depth-Anything`` at ``4f5ae23`` (source only). The
metric checkpoint (``depth-anything/Metric-Video-Depth-Anything-Large``, ~1.5 GB) is pulled from
the Hugging Face hub on first use and cached under ``HF_HOME``; with no network then, the first
sfm run fails with a ``RuntimeError`` naming the repo. Plain ``uv sync`` prunes the
``--no-deps`` packages; re-run that ``setup.sh`` block afterwards.

**Licenses.** InstantSfM is CC-BY-NC-4.0 (research use only). VDA code is Apache-2.0; the VDA
metric weights are CC-BY-NC-4.0.

**Output layout:**

.. code-block:: text

   <scene>/instantsfm/
     depth_vda/images/npy/<stem>.npy  VDA metric depth, 518-wide (reused when complete)
     colmap/instantsfm.db             SIFT database (local build artifact, not pushed)
     colmap/sparse/0/*.bin            global-mapper model, image names = frame stems
     pointcloud.zarr                  model-res depth/images/poses/K; no confidence (absent, never zeros);
                                      attrs: method, backend, registered_frames, total_frames, depth alignment
     sparse_pc.ply

``colmap/instantsfm.db`` has its own name so it never collides with ``colmap/colmap.db``; both are
in ``PUSH_EXCLUDES`` and stay local.

.. automodule:: collab_splats.pointcloud.sfm.instantsfm
   :members:
   :show-inheritance:

The ``colmap`` backend
~~~~~~~~~~~~~~~~~~~~~~

Classical incremental SfM (decision 018, ``docs/superpowers/decisions/018-sfm-backends.md``):

- features + matches: the pycolmap SIFT database (see SIFT database and pairing)
- mapping: ``pycolmap.incremental_mapping`` in a scratch ``colmap/mapper/`` dir; the largest model
  is kept, with a warning when the scene splits
- one shared SIMPLE_RADIAL camera, refined by the mapper; VDA metric depth as for instantsfm

.. code-block:: yaml

   pointcloud:
     method: sfm
     backend: colmap
     colmap:
       pairing: sequential+retrieval  # sequential | retrieval | sequential+retrieval | exhaustive
       overlap: 10
       num_retrieved: 20
       num_threads: 8
       min_registered_frac: 0.5

**Install.** Nothing beyond the instantsfm prerequisites (the VDA clone and checkpoint).

**Output layout:**

.. code-block:: text

   <scene>/colmap/
     depth_vda/images/npy/<stem>.npy  VDA metric depth (as instantsfm)
     colmap/colmap.db                 SIFT database (local build artifact, not pushed)
     colmap/sparse/0/*.bin            largest incremental model, image names = frame stems
     pointcloud.zarr                  attrs: method, backend, registered_frames, total_frames, depth alignment
     sparse_pc.ply

.. automodule:: collab_splats.pointcloud.sfm.colmap
   :members:
   :show-inheritance:

SIFT database and pairing
~~~~~~~~~~~~~~~~~~~~~~~~~

``instantsfm`` and ``colmap`` build their SIFT database through ``ensure_sift_database``:

- SIFT runs on GPU when pycolmap is the CUDA build (``pycolmap-cuda12``) and torch sees a GPU,
  otherwise on CPU capped at ``num_threads``
- retrieval pairings cap the vocab-tree threads at ``num_threads`` even on GPU (uncapped, OpenBLAS
  can crash on many cores)
- any ``*retrieval`` pairing fetches COLMAP's FAISS-format
  ``vocab_tree_faiss_flickr100K_words32K.bin`` (9.5 MB, sha256-pinned) once into
  ``~/.cache/collab_splats/``, only on a database rebuild; a failed download is a ``RuntimeError``
  naming URL and path

.. list-table::
   :header-rows: 1

   * - ``pairing``
     - pycolmap matcher
   * - ``sequential``
     - ``match_sequential``, ``overlap`` N, ``quadratic_overlap=False`` (i with i+1..i+N)
   * - ``retrieval``
     - ``match_vocabtree``, ``num_images`` = ``num_retrieved``
   * - ``sequential+retrieval``
     - ``match_sequential`` as above + ``loop_detection=True``
       (``loop_detection_num_images`` = ``num_retrieved``)
   * - ``exhaustive``
     - ``match_exhaustive``

- COLMAP's loop detection fires every ``loop_detection_period`` (10) frames, not per frame, so its
  ``sequential+retrieval`` is not the same pair set as hloc's

**Caching.** The database is reused only while its image set AND its matching params
(``pairing``, plus ``overlap`` for sequential modes and ``num_retrieved`` for retrieval modes)
match; the params live in a ``collab_params`` table inside the DB, written after a successful
build. A knob the pairing ignores is not recorded, so changing it keeps the DB.

.. automodule:: collab_splats.pointcloud.sfm.sift_db
   :members:
   :show-inheritance:

The ``hloc`` backend
~~~~~~~~~~~~~~~~~~~~

Learned-feature incremental SfM through hloc (``cvg/Hierarchical-Localization`` @ ``c13273b``):
local features + matches from hloc, mapping via ``hloc.reconstruction.main`` (pycolmap), same
single SIMPLE_RADIAL camera and VDA depth.

.. code-block:: yaml

   pointcloud:
     method: sfm
     backend: hloc
     hloc:
       pairing: sequential+retrieval
       overlap: 10
       num_retrieved: 20
       retrieval_conf: netvlad
       feature_conf: superpoint_max
       matcher_conf: superpoint+lightglue
       num_threads: 8
       min_registered_frac: 0.5

**Install.** hloc is the optional ``hloc`` extra, an editable uv path source on
``third_party/hloc``:

- ``bash setup/hloc.sh`` clones it (``--recursive``) and pins ``c13273b``; ``setup.sh`` calls it
  before the sync, and ``setup/hloc.sh --prefetch`` caches the netvlad / SuperPoint / LightGlue
  weights
- the clone must exist before ``uv lock`` / ``uv sync`` resolve
- without it, ``HlocCreator`` raises ``ImportError`` naming ``setup/hloc.sh``

**Licenses.** SuperPoint/SuperGlue weights are Magic Leap **non-commercial**; LightGlue is
Apache-2.0; netvlad weights come from the original authors (research use).

.. list-table::
   :header-rows: 1

   * - ``pairing``
     - hloc
   * - ``sequential``
     - in-repo ``sequential_pairs``: frame i with i+1..i+N (``overlap``)
   * - ``retrieval``
     - ``pairs_from_retrieval`` on ``retrieval_conf`` descriptors, top ``num_retrieved`` (clamped
       to N-1)
   * - ``sequential+retrieval``
     - union of both pair sets, deduplicated
   * - ``exhaustive``
     - ``pairs_from_exhaustive``

**Caching.** hloc's h5 features and matches are reused by its own skip; the mapper dir
(``colmap/hloc/sfm/``) is rebuilt every run.

**Output layout:**

.. code-block:: text

   <scene>/hloc/
     depth_vda/images/npy/<stem>.npy  VDA metric depth (as instantsfm)
     colmap/hloc/                     h5 features/matches, pairs-*.txt, sfm/ mapper dir (not pushed)
     colmap/sparse/0/*.bin            hloc's largest model, image names = frame stems
     pointcloud.zarr                  attrs: method, backend, registered_frames, total_frames, depth alignment
     sparse_pc.ply

.. automodule:: collab_splats.pointcloud.sfm.hloc
   :members:
   :show-inheritance:

Registered subset
~~~~~~~~~~~~~~~~~

Every sfm backend may leave frames unregistered:

- below ``min_registered_frac`` of the keyframes registered: ``RuntimeError`` with N/M
- above it: depth, names and keyframes are filtered to the registered stems, with a warning
- ``registered_frames`` / ``total_frames`` in the zarr attrs record the split
- ``images/`` still holds every keyframe; downstream stages read only the frames
  ``pointcloud.zarr`` names in its ``image_paths`` attr, joined on frame index
- the 2D semantics cache ``semantics/<extractor>_codes.zarr`` stays scene-level (every
  ``images/`` frame); the lift picks the pointcloud's rows out of it

Cleaning and utilities
----------------------

``clean_pointcloud`` (result in, result out), ``outlier_mask``, ``confidence_mask``,
``subsample_points`` and ``frame_depths`` (the confidence-masked depth targets mesh and splats
read).

.. automodule:: collab_splats.pointcloud.utils
   :members:
   :show-inheritance:
