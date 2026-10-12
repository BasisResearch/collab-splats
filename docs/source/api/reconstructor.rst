Reconstructor
=============

Config-driven stage pipeline for one scene; ``reconstruct local|remote`` runs it per input.

- one output tree per scene; a run config merges over ``configs/base.yaml``
- stages run in ``STAGES`` order: preproc, pointcloud, refine, splats, mesh, semantics, localize,
  reconstruction_quality_report
- leaf stages re-run alone with ``--stages``
- every config key: :doc:`/configuration`

.. automodule:: collab_splats.reconstructor
   :members:
   :show-inheritance:

Command line
------------

.. automodule:: collab_splats.__main__
   :members:

Remote scenes
-------------

.. automodule:: collab_splats.remote
   :members:
   :show-inheritance:
