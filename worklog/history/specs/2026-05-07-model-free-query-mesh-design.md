# Design: model-free query_mesh via saved decoder state dict

**Date:** 2026-05-07  
**Branch:** refactor/core-modules

## Problem

`query_mesh()` calls `nerfstudio.utils.eval_utils.eval_setup()` to load the full pipeline.
`eval_setup` loads the model weights **and** the datamanager (all training images).
Two splatters queried in the same notebook = two full pipelines in memory → OOM crash.

## What `query_mesh` actually needs

1. **Decoder** (`TwoLayerMLP`) — a 2-layer MLP (~34K params, ~130KB state dict).  
   Transforms raw latent features from `mesh_features.pt` (dim=13) into the text-encoder feature space.
2. **Text encoder** (`BaseQueryableExtractor` subclass) — CLIP or Talk2DINO.  
   Already instantiable via `BaseFeatureExtractor.get(name)(device=...)` with no nerfstudio dependency.

Neither requires the datamanager, the renderer, or the Gaussian parameters.

## Key insight: state dict is self-describing

`TwoLayerMLP` Conv2d weights encode all needed dimensions:
- `hidden_conv.weight` shape `(H, I, 1, 1)` → `hidden_dim=H`, `input_dim=I`
- `feature_branch_dict.{name}.weight` shape `(O, H, 1, 1)` → `output_dim=O`, `feature_type=name`

No config YAML parsing needed. Feature type is whichever branch key appears in  
`_QUERYABLE_FEATURE_TYPES = {"maskclip", "samclip", "talk2dino"}`.  
Encoder registry name = `_TEXT_ENCODER_NAME.get(feature_type, feature_type)`.

## Design

### `_extract_mesh_features` — add 1 line

After saving `mesh_features.pt`, also save the decoder:

```python
torch.save(self.model.decoder.state_dict(), features_path.parent / "mesh_decoder.pt")
```

### `query_mesh` — replace `eval_setup` block

Fast path (when `mesh_decoder.pt` exists):

```python
decoder_path = self.config["mesh_info"]["mesh"].parent / "mesh_decoder.pt"
if decoder_path.exists():
    state = torch.load(decoder_path, map_location="cpu")
    input_dim  = state["hidden_conv.weight"].shape[1]
    hidden_dim = state["hidden_conv.weight"].shape[0]
    feat_dims  = {
        k.split(".")[1]: (v.shape[0], 1, 1)
        for k, v in state.items()
        if k.startswith("feature_branch_dict.") and k.endswith(".weight")
    }
    decoder = TwoLayerMLP(input_dim, hidden_dim, feat_dims)
    decoder.load_state_dict(state)
    feature_type = next(k for k in feat_dims if k in _QUERYABLE_FEATURE_TYPES)
    encoder_name = _TEXT_ENCODER_NAME.get(feature_type, feature_type)
    text_encoder = BaseFeatureExtractor.get(encoder_name)(device="cpu")
    self.model = SimpleNamespace(
        decoder=decoder,
        similarity_fx=text_encoder.score_queries,
        main_features_name=feature_type,
        device="cpu",
    )
else:
    # legacy fallback — eval_setup
    ...
```

### Backward compatibility

`mesh_decoder.pt` absent → fall back to existing `eval_setup` path (unchanged).  
Existing fieldwork data re-generates decoder file via `splatter.mesh(overwrite=True)` once.

## Files changed

| File | Change |
|------|--------|
| `collab_splats/wrapper/splatter.py` | `_extract_mesh_features`: +1 line; `query_mesh`: replace eval_setup block |
| `tests/wrapper/test_splatter_query.py` | Add tests for fast path and fallback |

## Verification

1. Run `splatter.query_mesh(positive_queries=["feeder"])` with `mesh_decoder.pt` present — verify no `eval_setup` call, returns `(N, 3)` array.
2. Run same call with `mesh_decoder.pt` absent — verify falls back to `eval_setup`.
3. Run both splatters in notebook without OOM.
4. Similarity scores match (fast path vs eval_setup path produce identical outputs).
