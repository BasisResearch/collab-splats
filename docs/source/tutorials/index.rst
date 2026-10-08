Tutorials
=========

The pages share one scene, built from ``data/tutorial/tutorial_example-video.mp4`` into
``data/tutorial_scene/``. Each page runs top to bottom on its own: it builds the pipeline stages
it needs that are not on disk yet, then shows its subject. Opened in order, later pages reuse what
earlier pages built.

After pulling code changes, delete ``data/tutorial_scene/`` so the stages are rebuilt.

The scene uses a small profile (96 frames, short training); production values are in
``configs/base.yaml``.

=================================== ===============================
Stage                               Page
=================================== ===============================
``preproc``                         01 · Preprocessing
``pointcloud``                      02 · Reconstruction
``reconstruction_quality_report``   02 · Reconstruction
``refine``                          02 · Refinement
``splats``                          03 · Train splats
``mesh``                            04 · Mesh
``semantics``                       05 · Lifting and query, OCR lens
``localize``                        06 · Localization
=================================== ===============================

.. toctree::
   :maxdepth: 1
   :caption: 01 · Preprocessing

   01_preprocessing/preprocessing

.. toctree::
   :maxdepth: 1
   :caption: 02 · Pointcloud

   02_pointcloud/reconstruction
   02_pointcloud/refinement

.. toctree::
   :maxdepth: 1
   :caption: 03 · Splats

   03_splats/train_splats

.. toctree::
   :maxdepth: 1
   :caption: 04 · Mesh

   04_mesh/mesh

.. toctree::
   :maxdepth: 1
   :caption: 05 · Semantics

   05_semantics/feature_extraction
   05_semantics/segmentation
   05_semantics/lifting_and_query
   05_semantics/ocr_lens

.. toctree::
   :maxdepth: 1
   :caption: 06 · Localization

   06_localization/localization
