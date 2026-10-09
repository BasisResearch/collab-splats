Semantics
=========

Feature extraction, lifting, compression and segmentation. Everything public is importable from
``collab_splats.semantics``. The pipeline's ``semantics:`` block (``extractors``, ``n_components``,
``target_cosine``, ``max_epochs``) drives the stage: it runs after the mesh stage, builds each
listed extractor with its defaults in order (default ``[talk2dino, ocr_lens]``), one model
resident at a time, and writes one set of stores per extractor. The ``05_semantics`` tutorials walk through extraction, lifting and
query, the OCR lens and segmentation.

Feature extractors
------------------

Four extractors are registered: ``dinov2``, ``maskclip``, ``talk2dino`` and
``ocr_lens``. ``BaseFeatureExtractor.get(name)(device=...)`` builds one; ``forward`` takes a
list of images and returns a list of unit-norm (D, H_p, W_p) float32 CPU tensors.

- ``preprocess`` is shared: ``resize_mode="max_size"`` resizes the longest edge
  proportionally, ``"square"`` center-crops first; either way the size is rounded to the nearest
  multiple of ``patch_size``. Defaults: 800 (dinov2), 1024 (maskclip), 512 (talk2dino);
  ``ocr_lens`` is fixed at ``max_size`` 672.
- ``maskclip`` and ``talk2dino`` are ``BaseQueryableExtractor``: ``score_queries(features,
  positive, negative=None, temperature=0.05, reduction="max")`` embeds text into the feature space
  and returns (H_p, W_p) scores for a map or (P,) for a (P, D) point array. ``reduction="max"``
  scores each positive against the negatives and keeps the best; ``"pool"`` averages the positives
  first; any other value raises. ``negative=None`` uses ``["object"]``; ``negative=[]`` returns
  raw similarity, outside [0, 1].
- ``debias`` projects out the positional component DINO-family features carry, estimated from a
  black image at the same patch grid; it takes and returns a list, L2-renormalized. Validated for
  dinov2 and talk2dino; other extractors log a warning. ``get_bias_visualization(H_p, W_p)``
  shows what was removed and needs a prior ``debias`` at that grid.
- ``features_to_rgb`` (PCA to RGB) is in ``collab_splats.utils.image``.

.. automodule:: collab_splats.semantics.features
   :members:
   :show-inheritance:

Lifting
-------

``lift_features(frame_features, result, *, depth_tol=0.05, num_classes=None)`` lifts per-frame
maps onto a ``PointcloudResult``'s points: multi-view, confidence-weighted and depth-gated,
falling back to the source pixel. ``frame_features`` is a callable returning frame i's map
(e.g. ``maps.__getitem__``), dense (D, H_p, W_p) or an indexed ``(ids, values)`` pair; the result
is (P, D) float32. ``transfer_features`` scatters point features onto other targets (mesh
vertices) by k-NN.

.. automodule:: collab_splats.semantics.lifting
   :members:
   :show-inheritance:

Compression
-----------

``FeatureAutoencoder(input_dim, latent_dim)`` compresses ViT-width features to the latent width
stored on disk. ``fit`` trains in place on a tensor, array or zarr array (streamed) and returns
nothing; ``recon_cosine``, ``recon_mse`` and ``epochs_run`` land on the instance and in the
checkpoint. They are measured on the training set, so they are optimistic at small N.
``encode`` works on patch maps or points (``per_point_encode``); decoding is per point only
(``per_point_decode``, ``iter_decode``): the pipeline encodes maps, lifts the latent maps, and
decodes per point on read. ``save`` takes a ``.pt`` file path; ``load`` is a classmethod.

.. automodule:: collab_splats.semantics.compression
   :members:
   :show-inheritance:

Storage
-------

Three zarr stores per extractor:

- ``<scene>/semantics/<extractor>_codes.zarr``: ``features`` (N, latent, H_p, W_p) fp16, one
  chunk per frame, plus ``autoencoder.pt``; attrs ``extractor``, ``patch_size``, ``n_frames``,
  ``extractor_kwargs``, ``latent_dim``. Written by ``write_feature_cache``.
  ``valid_feature_cache`` returns it for reuse only when the extractor, frame count,
  ``extractor_kwargs`` and ``latent_dim`` all match, so changing ``n_components`` re-extracts.
- ``<scene>/semantics/<extractor>_features.zarr``: full-width features, temporary, deleted once
  encoded and never pushed.
- ``<scene>/<backend>/semantics/<extractor>_lifted.zarr``: ``features`` (P, latent) fp16 and
  ``autoencoder.pt``; attrs ``input_dim``, ``latent_dim``, ``extractor``, ``extractor_kwargs``.
  When ``mesh.ply`` exists it also holds ``vertex_features`` (queryable extractors) or
  ``vertex_word_ids`` / ``vertex_word_probs`` with attr ``words`` (``ocr_lens``), and attr
  ``mesh_sha256``; a new mesh makes the stage re-run. Written by ``write_point_features``
  through a ``.tmp`` dir renamed into place, so a failed write leaves nothing behind.

``n_components: null`` keeps full-width features in ``_codes.zarr`` (``latent_dim`` null, no
autoencoder, no temporary store). ``read_point_features(store_path, name="features")`` decodes a
lifted store to (P, D) float32, L2-normalized; pass ``name="vertex_features"`` for the mesh
vertices. ``python -m collab_splats.viewer <scene>/<backend>`` shows the lifted semantics on the
mesh.

.. automodule:: collab_splats.semantics.store
   :members:
   :show-inheritance:

Segmentation
------------

Four backends behind ``BaseSegmentation.get(name)``:

- ``mobilesamv2``: class-agnostic masks; ``strategy="object"`` (YOLOv8 boxes to SAM) or
  ``"auto"``. ``segment(frame)`` returns (N, H, W) bool masks (N may be 0) and SAM's metadata.
- ``insid3``: training-free in-context segmentation on frozen DINOv2 features;
  ``set_context(ref_image, ref_mask)`` then ``segment``, or ``segment_with_mask``.
- ``sam3``: text-prompted; ``segment_with_text(img, prompt)`` returns masks, boxes and scores.
  It is a gated model: request access at https://huggingface.co/facebook/sam3, run
  ``huggingface-cli login``, and install from https://github.com/facebookresearch/sam3; until
  then construction raises ``ImportError`` with those steps.
- ``skywater``: SegFormer sky masks. ``sky_masks(images_dir, ...)`` returns (N, H, W) bool and
  caches probabilities in a sibling ``sky/`` dir; the mesh stage uses it for ``mesh.mask_sky``.

Backends without text support raise ``NotImplementedError`` from ``segment_with_text``.

.. automodule:: collab_splats.semantics.segmentation
   :members:
   :show-inheritance:
