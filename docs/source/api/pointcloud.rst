Point Cloud
===========

Structure-from-motion and feedforward reconstruction.

.. automodule:: collab_splats.pointcloud.base
   :members:
   :show-inheritance:

SfM backends
------------

``pointcloud.method: sfm`` dispatches ``pointcloud.backend`` through ``SFM_CREATORS``:
``instantsfm`` (global), ``colmap`` and ``hloc`` (incremental). See
:doc:`/configuration` for the config blocks.

.. automodule:: collab_splats.pointcloud.sfm
   :no-members:

.. autodata:: collab_splats.pointcloud.sfm.SFM_CREATORS
   :no-value:

   Backend name to creator class; ``Reconstructor.pointcloud`` dispatches on it.

.. automodule:: collab_splats.pointcloud.sfm.colmap
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.sfm.hloc
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.sfm.instantsfm
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.sfm.sift_db
   :members:
   :show-inheritance:

Depth and feedforward
---------------------

.. automodule:: collab_splats.pointcloud.depth
   :members:
   :show-inheritance:

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

.. automodule:: collab_splats.pointcloud.feedforward.loger
   :members:
   :show-inheritance:

.. automodule:: collab_splats.pointcloud.utils
   :members:
   :show-inheritance:
