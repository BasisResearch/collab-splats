# Spec: VGGTx Feedforward Refactor — Structural Parity + Intermediate Feature Extraction

**Date:** 2026-05-23
**Status:** approved

## Goals

1. Make `VGGTXCreator` structurally parallel to `MapAnythingCreator` (file sections, docstrings, method naming).
2. Replace the class-level monkey-patch (`_patch_vggtx_compute_similarity`) with a clean per-call forward hook pattern.
3. Add `extract_intermediate_features(frames, layer_index)` as a general abstract method on `BaseFeedforwardCreator` so both backends expose the same capability — useful for loop closure verification, visualization, retrieval, and similarity search.

---

## Problem

### VGGTx monkey-patch

`_patch_vggtx_compute_similarity()` mutates the `Aggregator` class globally at load time:
- Replaces `Aggregator.forward` with a flag-checked wrapper.
- Replaces `VGGT.forward` with a `compute_similarity` parameter shim.
- Stores state on the instance (`_compute_similarity_requested`, `_last_image_match_ratio`).

Consequences: not thread-safe, not per-call, leaves class in mutated state for any other consumer of `vggt`.

### MapAnything workaround

`MapAnythingCreator._verify_loop_candidate` uses `self._lc_retrieval.embed_frames()` + `F.cosine_similarity` — the Stage 1 DINO retrieval model, not the cross-frame transformer. This is weaker than attention-based gating and inconsistent with the VGGTx path.

### No shared extraction interface

There is no common API for callers to ask either backend "give me intermediate cross-frame attention maps from block N." Both backends hard-code their own gating logic with no shared abstraction.

---

## Design

### Core principle

> Extracting attention maps is a general capability of these models.  
> Computing similarity for loop closure is specific logic that consumes that capability.

### 1. `cross_frame_attention_ratio` — in `pointcloud/utils.py`

Pure function. No state, no class dependency. Importable independently of any creator
for visualization, retrieval experiments, or analysis.

```python
def cross_frame_attention_ratio(
    k: torch.Tensor,
    q: torch.Tensor,
    token_offset: int = 5,
) -> float:
    """Cross-frame attention ratio between two frames' QKV tensors.

    Measures how much frame B's tokens attend to frame A.  Port of VGGT-SPARK
    get_similarity().  High value = frames share matching geometry/content.

    Args:
        k, q:         (B, heads, N_tokens, head_dim) from a 2-frame forward pass.
        token_offset: Skip leading non-patch tokens (camera + register tokens).

    Returns:
        Scalar in [0, 1].  Values >= 0.85 match the VGGT-SPARK acceptance threshold.
    """
```

Lives in `pointcloud/utils.py` (alongside `lift_features`, `reproject_pixels`) because
it is a pure utility with no creator dependency — not in `base.py` which is already
large and creator-focused.

### 2. `extract_intermediate_features` — abstract on `BaseFeedforwardCreator`

Each backend registers a per-call hook on its specific cross-frame transformer block,
runs a 2-frame forward, captures QKV, removes the hook, and returns a dict.
No shared boilerplate helper is needed — each implementation is ~15 lines and
self-contained. No persistent state is left on the model.

```python
@abstractmethod
def extract_intermediate_features(
    self, frames: torch.Tensor, layer_index: int = -1, **kwargs
) -> dict[str, torch.Tensor]:
    """Capture cross-frame transformer activations via a per-call forward hook.

    Register a forward hook on the QKV projection of block `layer_index` in the
    cross-frame attention stack, run the model on `frames`, capture activations,
    then remove the hook.  The hook is removed in a finally block — it is guaranteed
    to be cleaned up even if the forward raises.

    Args:
        frames:      (N, C, H, W) preprocessed frames on the correct device.
        layer_index: Which cross-frame block to tap.  -1 = last (default, matches
                     VGGT-SPARK).  Valid range: [-len(blocks), len(blocks)-1].
                     VGGTx indexes into aggregator.global_blocks;
                     MapAnything into info_sharing.self_attention_blocks.
        **kwargs:    Backend-specific forward kwargs forwarded to the model call.
                     MapAnything: minibatch_size (int, default 1),
                                  memory_efficient_inference (bool, default False).
                     VGGTx: unused.

    Returns:
        dict with at minimum:
          "q":  (B, heads, N_tokens, head_dim) — query projections
          "k":  (B, heads, N_tokens, head_dim) — key projections
        VGGTx additionally includes "poses": (2, 4, 4) float32 np.ndarray —
        pre-decoded camera extrinsics.  MapAnything omits "poses".
    """
```

#### VGGTx implementation

```python
def extract_intermediate_features(
    self, frames: torch.Tensor, layer_index: int = -1, **kwargs
) -> dict[str, torch.Tensor]:
    device = next(self.model.parameters()).device
    dtype = next(self.model.parameters()).dtype
    batch = frames.unsqueeze(0).to(device, dtype=dtype)

    block = self.model.aggregator.global_blocks[layer_index]
    C_nh = block.attn.num_heads
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, _inp, out):
        B, N, C3 = out.shape
        hd = (C3 // 3) // C_nh
        qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
        captured["q"], captured["k"] = qkv[0], qkv[1]

    hook = block.attn.qkv.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            predictions = self.model(batch)
    finally:
        hook.remove()

    # Decode (2, 4, 4) extrinsics here — base sees "poses", not raw "pose_enc"
    image_shape = (frames.shape[-2], frames.shape[-1])
    ext_3x4, _ = pose_encoding_to_extri_intri(
        predictions["pose_enc"].detach(), image_shape
    )
    ext_3x4 = ext_3x4.cpu().float().numpy().squeeze(0)  # (2, 3, 4)
    captured["poses"] = _extrinsics_3x4_to_4x4(ext_3x4)  # (2, 4, 4)
    return captured
```

#### MapAnything implementation

```python
def extract_intermediate_features(
    self, frames: torch.Tensor, layer_index: int = -1, **kwargs
) -> dict[str, torch.Tensor]:
    minibatch_size = kwargs.get("minibatch_size", 1)
    memory_efficient = kwargs.get("memory_efficient_inference", False)

    block = self.model.info_sharing.self_attention_blocks[layer_index]
    C_nh = block.attn.num_heads
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, _inp, out):
        B, N, C3 = out.shape
        hd = (C3 // 3) // C_nh
        qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
        captured["q"], captured["k"] = qkv[0], qkv[1]

    hook = block.attn.qkv.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            raw_views = [{"img": f.unsqueeze(0)} for f in frames.cpu()]
            views = preprocess_input_views_for_inference(raw_views)
            self.model.forward(
                views,
                memory_efficient_inference=memory_efficient,
                minibatch_size=minibatch_size,
            )
    finally:
        hook.remove()

    return captured
```

### 3. `_verify_loop_candidate` — concrete on `BaseFeedforwardCreator`

`_verify_loop_candidate` is concrete on base.  `base.py` imports nothing from `vggt`
and knows nothing about backend-specific keys.  Pose data (if any) arrives pre-decoded
as a `"poses"` key in the features dict — that is part of the shared return contract
of `extract_intermediate_features`.  No `_decode_lc_poses` override method is needed.

```python
def _verify_loop_candidate(
    self,
    frame1: torch.Tensor,
    frame2: torch.Tensor,
    verify_match_ratio: float = 0.85,
    layer_index: int = -1,
    **kwargs,
) -> tuple[bool, np.ndarray | None]:
    """Verify a loop closure candidate via cross-frame attention gate.

    Args:
        frame1, frame2:      Preprocessed frames (C, H, W).
        verify_match_ratio:  Accept threshold (default 0.85, matches VGGT-SPARK).
        layer_index:         Transformer block to tap (default -1 = last).
        **kwargs:            Forwarded to extract_intermediate_features (e.g.
                             minibatch_size=2 for MapAnything).

    Returns:
        (accepted, poses_or_None).  poses is (2, 4, 4) float32 np.ndarray when
        the backend includes a "poses" key; None otherwise — caller uses submap poses.
    """
    # Run the model once to capture cross-frame activations
    features = self.extract_intermediate_features(
        torch.stack([frame1, frame2]), layer_index=layer_index, **kwargs
    )
    # Compute the cross-frame attention ratio gate
    ratio = cross_frame_attention_ratio(features["k"], features["q"])
    if ratio < verify_match_ratio:
        return False, None
    # "poses" is optional — VGGTx includes it (pre-decoded), MapAnything does not
    return True, features.get("poses")
```

VGGTXCreator decodes poses *inside* `extract_intermediate_features` and returns them
under `"poses"`.  `pose_encoding_to_extri_intri` is imported at the top of `vggtx.py`
(already used in `_postprocess`).  MapAnythingCreator omits `"poses"` — base returns
`None` via `features.get("poses")`.

---

## What changes per file

### `pointcloud/utils.py`

- Add `cross_frame_attention_ratio(k, q, token_offset=5) -> float`

### `base.py`

- Add abstract method `extract_intermediate_features(frames, layer_index=-1) -> dict`
- Add `layer_index` param to `_verify_loop_candidate`; replace `NotImplementedError` body with concrete implementation calling `cross_frame_attention_ratio` (imported from `pointcloud/utils.py`)
- Update module docstring

### `vggtx.py`

- **Delete** `_patch_vggtx_compute_similarity()` and its call in `_load_model`
- **Delete** `_verify_loop_candidate` (now on base)
- **Add** concrete `extract_intermediate_features` (hook on `aggregator.global_blocks[layer_index].attn.qkv`)
- Update module docstring (remove patch from Provides list)
- Add descriptive docstrings throughout (matching MapAnything style)
- Section structure: Constants → Inference utilities → Creator (matching MapAnything)

### `mapanything.py`

- **Delete** `_verify_loop_candidate` (now on base)
- **Remove** `import torch.nn.functional as F` (was only for cosine_similarity)
- **Add** concrete `extract_intermediate_features` (hook on `info_sharing.self_attention_blocks[layer_index].attn.qkv`)

---

## What does NOT change

- `_lc_retrieval` stays on `BaseFeedforwardCreator`. It is Stage 1 (candidate retrieval) and the base already orchestrates LC — moving it to the `LoopClosure` wrapper would require method signature changes out of scope.
- The 5-step `BaseFeedforwardCreator` pipeline (`_load_model`, `_preprocess`, `_forward`, `_postprocess`, `build_colmap`) is unchanged.
- `FeedforwardResult` dataclass is unchanged.

---

## Parallel layer mapping

| | VGGTx | MapAnything |
|---|---|---|
| Cross-frame transformer | `model.aggregator` | `model.info_sharing` |
| Block list | `aggregator.global_blocks` | `info_sharing.self_attention_blocks` |
| Hook target | `global_blocks[i].attn.qkv` | `self_attention_blocks[i].attn.qkv` |
| Default index | `-1` (last block) | `-1` (last block) |
| Returns pose | Yes (`"poses"` key, pre-decoded (2,4,4)) | No (`"poses"` absent → None) |

---

## Testing

- `test_cross_frame_attention_ratio_high` — synthetic q/k where frame B tokens attend strongly to frame A; verify ratio >= 0.85.
- `test_cross_frame_attention_ratio_low` — random uncorrelated q/k; verify ratio < 0.5.
- `test_extract_intermediate_features_vggtx` — mock tiny `aggregator.global_blocks` (2 blocks with real `nn.Linear` qkv); call `extract_intermediate_features`; verify q/k shapes correct, hook removed after call, hook removed even when forward raises.
- `test_extract_intermediate_features_vggtx_layer_index` — 3-block aggregator; call with `layer_index=1`; verify middle block fires, not last.
- `test_extract_intermediate_features_mapanything` — mock `info_sharing.self_attention_blocks`; same shape + cleanup checks.
- `test_verify_loop_candidate_accepted_with_poses` — stub `extract_intermediate_features` returning high-ratio q/k + `"poses"` np.ndarray (2,4,4); verify (True, (2,4,4) poses).
- `test_verify_loop_candidate_accepted_no_poses` — stub returning high-ratio q/k, no `"poses"` key; verify (True, None).
- `test_verify_loop_candidate_rejected` — stub returning low-ratio q/k; verify (False, None).
- `test_verify_loop_candidate_layer_index_forwarded` — verify `layer_index` arg is passed through to `extract_intermediate_features`.
- `test_patch_vggtx_compute_similarity_deleted` — verify `_patch_vggtx_compute_similarity` does not exist in `vggtx` module.
