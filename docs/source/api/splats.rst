Splats
======

Gaussian-splat training on upstream `gsplat <https://github.com/nerfstudio-project/gsplat>`_
(pinned at ``GSPLAT_COMMIT``) from an existing pointcloud stage. Training runs in the COLMAP
camera convention: poses are ``world_to_cam``, intrinsics are pixel-center at frame resolution,
and the seed points come from ``pointcloud.zarr``. The pipeline runs this stage for you
(``splats:`` in the yaml, ``--stages splats``); the ``train_splats`` tutorial calls ``train``
directly. ``SplatsConfig`` mirrors the ``splats:`` yaml block minus ``enabled``; the keys are in
``configs/README.md``.

``train`` builds one model, ``Gaussians`` or ``Scaffold`` (chosen by ``representation``), and one
``CameraOpt``. The two model classes expose the same members, so the loop does not branch on the
representation. ``CameraOpt`` holds a pose delta and a per-image color affine, toggled by
``pose_opt`` / ``appearance_opt``; a half that is off passes its input through. Each step refines
the camera, renders, applies the color affine, composites over a random background, computes the
losses, steps every optimizer and scheduler, then lets the model densify itself.

Primitive and representation
----------------------------

The two axes are independent; every combination trains.

- ``primitive: 3dgs`` (default): ``gsplat.rasterization``, antialiased, densified by MCMC under a
  ``cap_max`` budget. Better thin structure and speed.
- ``primitive: 2dgs``: ``gsplat.rasterization_2dgs``, which also returns rendered normals, a
  distortion map and the median depth; densified by ``DefaultStrategy`` under ``grow_grad2d``.
  Better surfaces.
- ``representation: vanilla`` (default): one parameter set per Gaussian (means, quats, log-scales,
  logit-opacities, SH bands); ``sh_degree`` unlocks one band every ``sh_degree_interval`` steps.
- ``representation: scaffold``: anchors, each with a feature vector and ``n_offsets`` learned
  offsets. Small MLP heads decode opacity, covariance and color per view, and densification grows
  and prunes anchors. ``sh_degree`` / ``sh_degree_interval`` are vanilla-only: set deliberately
  under ``scaffold``, they raise. The anchor model reimplements
  `Scaffold-GS <https://github.com/city-super/Scaffold-GS>`_ @ ``59c833b5`` (Inria non-commercial
  license); Scaffold x 2DGS follows `GS-SR <https://github.com/yanxian-ll/GS-SR>`_ @ ``566359be``
  (no LICENSE file upstream). No code is vendored.

``scaffold.appearance_dim`` (a per-image embedding fed to the MLP heads) and ``appearance_opt``
(a per-image color affine on the render) are separate appearance models; either, both or neither.

Outputs
-------

``train`` writes three files into ``out_dir``:

- ``splats.ply``: binary Gaussian ply. Under ``scaffold`` it is baked, each anchor offset decoded
  once at that anchor's mean observed view direction, so it is a fixed-view snapshot.
- ``ckpt.pt``: the reloadable model, with ``config``, ``cam_to_world``, ``intrinsics``,
  ``image_ids``, ``image_size``, ``appearance`` and, under ``scaffold``, the MLP heads and voxel
  size. Pose deltas are folded into ``cam_to_world`` before the write, so reloading cannot apply
  them twice. ``load_checkpoint`` plus ``render_views`` reproduce every training-view render;
  the mesh stage and the eval scripts consume the model this way.
- ``splats_quality_report.json``: mean and per-frame PSNR / SSIM, wall time, the final-step loss
  snapshot and ``n_gaussians`` (the anchor count under ``scaffold``).

Losses
------

``losses:`` is a schedule, ``name: {weight[, start, end, end_weight]}``. A loss contributes when
its weight at the step is above 0; with ``end`` the weight decays log-linearly from ``weight`` at
``start`` to ``end_weight`` at ``end``. The photometric loss (0.8 L1 + 0.2 (1 - SSIM)) is always
on. Registered: ``depth``, ``normal_consistency``, ``distortion``, ``opacity_reg``,
``scale_reg``, ``appearance_reg``. Under ``scaffold`` the two regularizers read the decoded
opacities and log-scales off the render.

Modules
-------

.. automodule:: collab_splats.splats.trainer
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.checkpoint
   :members:

.. automodule:: collab_splats.splats.gaussian
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.scaffold
   :members:
   :show-inheritance:

.. automodule:: collab_splats.splats.losses
   :members:

.. automodule:: collab_splats.splats.rendering
   :members:

.. automodule:: collab_splats.splats.cameras
   :members:
