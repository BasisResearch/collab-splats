Point Cloud
===========

``collab_splats.pointcloud`` builds a ``PointcloudResult`` (points, poses, depth, per-pixel
provenance) from the scene's ``images/`` keyframes through one of two backend families, picked by
``pointcloud.method``: ``feedforward`` (``vggt_omega`` default, ``vggtx``, ``mapanything``,
``loger``) or ``sfm`` (``instantsfm``, ``colmap``, ``hloc``, each with VDA metric depth). Config
keys: :doc:`/configuration`.

Output layout, per backend:

.. code-block:: text

   <scene>/<backend>/
     pointcloud.zarr                  depth, images, poses, K; attrs (sfm: method, backend,
                                      registered_frames, total_frames, depth alignment)
     colmap/sparse/0/*.bin            COLMAP model, image names = frame stems
     sparse_pc.ply
     depth_vda/images/npy/<stem>.npy  sfm only: VDA metric depth, 518-wide
     colmap/instantsfm.db             instantsfm only: SIFT database (not pushed)
     colmap/colmap.db                 colmap only: SIFT database (not pushed)
     colmap/hloc/                     hloc only: h5 features/matches, pairs, sfm/ mapper dir (not pushed)

.. automodule:: collab_splats.pointcloud.base
   :members:
   :show-inheritance:

Bundle adjustment with loop closure
-----------------------------------

Feedforward only; the solver itself is in :doc:`geometry`. With ``pointcloud.loop_closure`` and
``pointcloud.bundle_adjustment.enabled`` both on, ``pointcloud.bundle_adjustment.strategy`` picks
the solve:

- ``global`` (default): plain LC windows, then the ``refine`` stage solves every frame at once
- ``window``: BA inside each loop-closure window, before its submap enters the pose graph;
  ``refine`` is dropped from the default stage set, and naming it explicitly raises ``ValueError``
- ``window+global``: both, in that order

Window solves:

- the first window refined with ``refine_focal`` sets the focal; later windows hold it fixed
- a window whose solve raises ``ValueError`` keeps its feedforward poses, with a warning
- per-window records (``start``, ``n_frames``, ``ok``, ``alignment_scale``, ``loss_final``,
  ``focal``, ``seconds``, ``gpu_max_mib``) land in the ``pointcloud.zarr`` attrs under
  ``window_ba``

Depth alignment (sfm path)
--------------------------

- ``estimate_depth``: Video-Depth-Anything metric depth per keyframe at ``depth_width`` 518,
  cached under ``depth_vda/``; a complete cache is loaded, a partial one recomputed
- ``align_depth``: per-frame median COLMAP / VDA depth ratio over track observations; frames
  under ``min_obs`` (20) take the median scale of the fitted frames
- no ``confidence`` (``None``): sfm has no per-pixel confidence
- weights: Metric-Video-Depth-Anything-Large, **CC-BY-NC-4.0** (non-commercial); VDA code
  Apache-2.0

.. automodule:: collab_splats.pointcloud.depth
   :members:
   :show-inheritance:

Feedforward backends
--------------------

Registered on ``BaseFeedforwardCreator``; the ``pointcloud.<backend>`` block is passed as creator
keyword arguments.

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

The ``loger`` backend
~~~~~~~~~~~~~~~~~~~~~

LoGeR: a Pi3 backbone plus a TTT fast-weight memory, run with sliding-window inference. Install
with ``bash setup/loger.sh`` (vendors ``Junyi42/LoGeR`` @ ``7685b7a``); weights download from
HuggingFace on first use.

- for long, ordered captures; refuses ``loop_closure``
- K is fitted, not predicted: ``estimate_intrinsics_from_points`` solves one shared K from the
  pointmap; a degenerate fit raises ``RuntimeError``
- ``preproc.max_frames`` (default 300) caps the frame count; raise it to use LoGeR's windowing

.. automodule:: collab_splats.pointcloud.feedforward.loger
   :members:
   :show-inheritance:

SfM backends
------------

``pointcloud.method: sfm`` dispatches ``pointcloud.backend`` through ``SFM_CREATORS``:
``instantsfm`` (global), ``colmap`` and ``hloc`` (incremental). Every sfm backend:

- is experimental: the stage emits a ``UserWarning``
- refuses ``loop_closure`` (``ValueError`` at config validation)
- skips bundle adjustment: ``refine`` is dropped from the default stage set (the mapper runs its
  own BA), and naming it explicitly raises ``ValueError``
- reads the scene-root ``images/`` store in place, so COLMAP image names are the keyframe stems
- shares ``pairing``, ``overlap``, ``num_retrieved``, ``num_threads`` and ``min_registered_frac``
  from ``BaseSfmCreator``

Downstream stages read ``pointcloud.zarr`` unchanged; with no ``confidence``, mesh and splats
depth targets keep every pixel and the feature lift uses uniform weights.

Registered subset:

- below ``min_registered_frac`` of the keyframes registered: ``RuntimeError``
- above it: depth, names and keyframes are filtered to the registered stems, with a warning
- downstream stages read only the frames named in the zarr's ``image_paths`` attr

.. automodule:: collab_splats.pointcloud.sfm
   :no-members:

.. autodata:: collab_splats.pointcloud.sfm.SFM_CREATORS
   :no-value:

   Backend name to creator class; ``Reconstructor.pointcloud`` dispatches on it.

.. automodule:: collab_splats.pointcloud.sfm.base
   :members:
   :show-inheritance:

The ``instantsfm`` backend
~~~~~~~~~~~~~~~~~~~~~~~~~~

Global SfM: a pycolmap SIFT database, then InstantSfM's global mapper (rotation averaging, global
positioning, global BA) with VDA depth as priors. A split solve keeps cluster 0 only.

- install: the locked ``instantsfm`` extra, installed by ``setup.sh`` (``uv sync --all-extras``);
  ``scikit-sparse`` needs apt ``libsuitesparse-dev``
- licenses: InstantSfM **CC-BY-NC-4.0** (research use only)

.. automodule:: collab_splats.pointcloud.sfm.instantsfm
   :members:
   :show-inheritance:

The ``colmap`` backend
~~~~~~~~~~~~~~~~~~~~~~

Classical incremental SfM: the pycolmap SIFT database, then ``pycolmap.incremental_mapping``;
the largest model is kept (with a warning on a split). One shared SIMPLE_RADIAL camera, refined
by the mapper.

.. automodule:: collab_splats.pointcloud.sfm.colmap
   :members:
   :show-inheritance:

SIFT database and pairing
~~~~~~~~~~~~~~~~~~~~~~~~~

``instantsfm`` and ``colmap`` build their SIFT database through ``ensure_sift_database``:

- GPU SIFT with the CUDA pycolmap build (``pycolmap-cuda12``) and a visible GPU, else CPU capped
  at ``num_threads``
- ``*retrieval`` pairings fetch COLMAP's vocab tree once into ``~/.cache/collab_splats/``
  (sha256-pinned)
- reused while its image set and matching params match; otherwise rebuilt

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

- loop detection fires every ``loop_detection_period`` frames (pycolmap's default, 10), so
  ``sequential+retrieval`` is not the same pair set as hloc's

.. automodule:: collab_splats.pointcloud.sfm.sift_db
   :members:
   :show-inheritance:

The ``hloc`` backend
~~~~~~~~~~~~~~~~~~~~

Learned-feature incremental SfM through hloc (``cvg/Hierarchical-Localization`` @ ``c13273b``):
local features and matches from hloc, mapping via ``hloc.reconstruction.main``, same single
SIMPLE_RADIAL camera and VDA depth.

- install: the optional ``hloc`` extra; ``setup.sh`` runs ``setup/hloc.sh`` (clone + weight
  prefetch); without it ``HlocCreator`` raises ``ImportError``
- licenses: SuperPoint/SuperGlue weights Magic Leap **non-commercial**; LightGlue Apache-2.0;
  netvlad weights research use

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

.. automodule:: collab_splats.pointcloud.sfm.hloc
   :members:
   :show-inheritance:

Cleaning and utilities
----------------------

``clean_pointcloud`` (result in, result out), ``outlier_mask``, ``confidence_mask``,
``subsample_points`` and ``frame_depths`` (the confidence-masked depth targets mesh and splats
read).

.. automodule:: collab_splats.pointcloud.utils
   :members:
   :show-inheritance:
