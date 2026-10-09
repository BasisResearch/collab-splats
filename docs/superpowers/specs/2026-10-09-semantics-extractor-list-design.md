# Semantics: several extractors per config

**Status:** approved 2026-10-09 · **Branch:** `feat/semantics-list`

## Problem

- `semantics.extractor` names one extractor; talk2dino + ocr_lens on one scene needs two runs with overrides
- the AE fit streams the full-width zarr from disk every epoch: ocr_lens 10-20 s/epoch on `/workspace`
- the `target_cosine` stop ends fits early and costs query quality; 0.95 is unreachable for ocr_lens
- `extractor_kwargs` is never set by any config, and one shared kwargs dict cannot serve two extractors

## Measurements (GH010238, 300 frames, 30 eval frames vs full-width features)

| extractor | latent | stop | epoch | top-5% IoU / top-1 word |
|---|---|---|---|---|
| talk2dino | 64 | cos 0.95 (old default) | 6 | 0.66 |
| talk2dino | 128 | 30 epochs | 30 | **0.78** |
| ocr_lens | 64 | 20 epochs (0.95 unreached) | 20 | 0.74 |
| ocr_lens | 128 | cos 0.90 | 2 | 0.78 |
| ocr_lens | 128 | 20 epochs | 20 | **0.81** |

- GPU-resident 30-epoch fit at 128: talk2dino 28 s (0.2 GB), ocr_lens 57 s (3.1 GB, peak 5.1 GB VRAM)
- same final cosine as the streamed fit

## Design

```yaml
semantics:
  enabled: true
  extractors: [talk2dino, ocr_lens]
  n_components: 128
  target_cosine: null
  max_epochs: 30
```

- `extractors`: list, run in order, one at a time (extractor freed before the next); shared settings
- every run rebuilds each listed store; a valid codes cache skips that extractor's extraction and fit
- `extractor` and `extractor_kwargs` removed from the config; `validate_config` checks `extractors` is a non-empty
  list of registered names
- unknown `semantics` keys are not rejected: recorded `run_config.yaml` files still carry `extractor`, and the
  dashboard and CLI re-runs merge them back in
- extractors build with their defaults; stores keep `extractor_kwargs: {}` in attrs, so existing caches stay valid
- AE fit: the stage reads the temporary features zarr onto the AE device once, passes the tensor to `fit()`;
  `fit()` unchanged
- `Reconstructor.lifted_stores`: extractor -> `<backend>/semantics/<extractor>_lifted.zarr`
- `Reconstructor.fresh_lifted_stores()`: the listed stores on disk whose recorded mesh hash (if any) matches mesh.ply
- `outputs["semantics"]` is `<backend>/semantics`; `done("semantics")` needs every listed store fresh
- dashboard app shows the first listed extractor with a fresh store; tutorial helper sets `extractors: [name]`

## Out of scope

- per-extractor settings (one shared setting wins for both, measured above)
- patch subsampling for the fit (only if a scene runs out of VRAM)
- a CLI filter to run one listed extractor
