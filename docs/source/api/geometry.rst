Geometry
========

Pose and geometry backend: transforms, bundle adjustment, and loop closure.

.. automodule:: collab_splats.geometry.transforms
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.bundle_adjustment
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.loop_closure.closure
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.loop_closure.graph
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.loop_closure.submap
   :members:
   :show-inheritance:

.. automodule:: collab_splats.geometry.loop_closure.eval
   :members:
   :show-inheritance:

Quickstart
----------

.. code-block:: python

   from collab_splats.pointcloud import get_creator
   from collab_splats.geometry.loop_closure import LoopClosure, LoopClosureConfig

   base = get_creator("vggtx")()
   lc = LoopClosure(base, config=LoopClosureConfig())
   result = lc.reconstruct(image_dir="path/to/images", output_dir="path/to/out")

See ``docs/source/tutorials/02_pointcloud/slam_loop_closure.ipynb`` for the full
walkthrough (candidate matching, plots).

.. automodule:: collab_splats.geometry.loop_closure.wrapper
   :members:
   :show-inheritance:
