Localization
============

Query image to camera pose in a known reconstruction: retrieval, local matching, PnP.

- retrieval: ``DinoSaladExtractor`` / ``MegaLocExtractor`` rank reference frames
- matching: ``LocalMatcher`` (xfeat | loma) matches query to reference pixels
- pose: ``CameraLocalizer`` looks matched pixels up in ``world_points`` and solves PnP with pycolmap
- the localize stage caches reference features in ``pointcloud.zarr``; config keys
  ``localization.matcher`` and ``localization.retrieval`` (default megaloc)

.. automodule:: collab_splats.localization.retrieval
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.extractors
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.localizer
   :members:
   :show-inheritance:

.. automodule:: collab_splats.localization.viz
   :members:
