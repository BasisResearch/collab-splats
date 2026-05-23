# VGGTx Feedforward Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the VGGTx monkey-patch with per-call forward hooks, add `extract_intermediate_features(frames, layer_index, **kwargs)` as a shared abstract interface on both feedforward backends, move `cross_frame_attention_ratio` to `pointcloud/utils.py` as a general pure function, and make `_verify_loop_candidate` concrete on the base.

**Architecture:** `cross_frame_attention_ratio` lives in `pointcloud/utils.py` (no class dependency, importable anywhere). Each backend implements `extract_intermediate_features` independently (~15 lines each, per-call hook, self-contained). `_verify_loop_candidate` on base delegates pose decoding to `_decode_lc_poses` (non-abstract, default `None`; VGGTXCreator overrides). `base.py` has zero vggt imports.

**Tech Stack:** PyTorch `register_forward_hook`, `vggt.utils.pose_enc.pose_encoding_to_extri_intri`, `mapanything.utils.inference.preprocess_input_views_for_inference`, pytest with `unittest.mock`.

## Coding Principles (apply to every task)

1. **Imports at the top.** No lazy imports inside methods or functions — all `import` statements go at the module level. If a symbol is only used by one backend, it lives in that backend's file, not in base.
2. **Block-level inline comments.** Each logical block of code gets a short `#` comment explaining what it does and why. Comment at block level — one comment per logical block, not one per line.

---

## File Map

| File | Change |
|---|---|
| `collab_splats/pointcloud/utils.py` | Add `cross_frame_attention_ratio` |
| `collab_splats/pointcloud/feedforward/base.py` | Add abstract `extract_intermediate_features`; make `_verify_loop_candidate` concrete using `features.get("poses")`; no vggt imports |
| `collab_splats/pointcloud/feedforward/vggtx.py` | Delete `_patch_vggtx_compute_similarity` + call in `_load_model`; delete `_verify_loop_candidate`; add concrete `extract_intermediate_features`; update docstrings + section structure |
| `collab_splats/pointcloud/feedforward/mapanything.py` | Delete `_verify_loop_candidate`; remove `import torch.nn.functional as F`; add concrete `extract_intermediate_features` |
| `collab_splats/pointcloud/feedforward/__init__.py` | Remove `_patch_vggtx_compute_similarity` re-export |
| `tests/pointcloud/test_pointcloud_utils.py` | Add `cross_frame_attention_ratio` tests |
| `tests/pointcloud/test_feedforward_shared.py` | Add `_verify_loop_candidate` base tests |
| `tests/pointcloud/test_vggtx_creator.py` | Add `extract_intermediate_features` tests; add monkey-patch deletion test |
| `tests/pointcloud/test_mapanything_creator.py` | Add `extract_intermediate_features` tests |

---

## Task 1: `cross_frame_attention_ratio` in `pointcloud/utils.py`

**Files:**
- Modify: `tests/pointcloud/test_pointcloud_utils.py`
- Modify: `collab_splats/pointcloud/utils.py`

- [ ] **Step 1.1: Write failing tests**

Add to `tests/pointcloud/test_pointcloud_utils.py`:

```python
import torch
import pytest


def test_cross_frame_attention_ratio_returns_float():
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    result = cross_frame_attention_ratio(k, q, token_offset=0)
    assert isinstance(result, float)


def test_cross_frame_attention_ratio_similar_frames_high():
    """Identical content in both frame halves → ratio close to 1.0."""
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    torch.manual_seed(0)
    B, heads, hd = 1, 2, 4
    # N=20: 10 tokens per frame, both frames have identical feature vectors
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    ratio = cross_frame_attention_ratio(k, q, token_offset=0)
    assert ratio > 0.8


def test_cross_frame_attention_ratio_orthogonal_frames_low():
    """Second frame tokens orthogonal to first → low cross-frame attention ratio."""
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    B, heads, hd = 1, 1, 4
    N = 20
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    # First frame activations in dim 0
    k[:, :, :10, 0] = 10.0
    q[:, :, :10, 0] = 10.0
    # Second frame activations in dim 1 (orthogonal to first)
    k[:, :, 10:, 1] = 10.0
    q[:, :, 10:, 1] = 10.0
    ratio = cross_frame_attention_ratio(k, q, token_offset=0)
    assert ratio < 0.2


def test_cross_frame_attention_ratio_empty_returns_zero():
    """If token_offset >= tokens_per_img, k_first is empty → return 0.0."""
    from collab_splats.pointcloud.utils import cross_frame_attention_ratio
    B, heads, N, hd = 1, 2, 20, 4
    k = torch.randn(B, heads, N, hd)
    q = torch.randn(B, heads, N, hd)
    # token_offset=10 means k_first = k[:, :, 10:10, :] which is empty
    result = cross_frame_attention_ratio(k, q, token_offset=10)
    assert result == 0.0
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py::test_cross_frame_attention_ratio_returns_float tests/pointcloud/test_pointcloud_utils.py::test_cross_frame_attention_ratio_similar_frames_high -v
```

Expected: `ImportError: cannot import name 'cross_frame_attention_ratio'`

- [ ] **Step 1.3: Add function to `collab_splats/pointcloud/utils.py`**

Find the end of the existing functions section (after `voxel_downsample` or similar) and add:

```python
def cross_frame_attention_ratio(
    k: torch.Tensor,
    q: torch.Tensor,
    token_offset: int = 5,
) -> float:
    """Cross-frame attention ratio between two frames' QKV tensors.

    Measures how much frame B's tokens attend to frame A relative to frame A's
    self-attention.  Port of VGGT-SPARK get_similarity().  Used to gate loop
    closure candidate acceptance — high ratio means the two frames share coherent
    overlapping geometry.

    Args:
        k:            (B, heads, N_tokens, head_dim) key projections. N_tokens covers
                      both frames concatenated, so tokens_per_img = N_tokens // 2.
        q:            (B, heads, N_tokens, head_dim) query projections, same layout.
        token_offset: Skip the first N tokens per frame (camera + register tokens
                      that precede patch tokens in VGGT-style models).  Default 5.

    Returns:
        Scalar float in [0, ∞), 90th-percentile of the normalized cross-frame
        attention.  Values >= 0.85 match the VGGT-SPARK acceptance threshold.
        Returns 0.0 if token_offset >= tokens_per_img (no patch tokens to measure).
    """
    import numpy as np

    tokens_per_img = q.shape[2] // 2
    k_first = k[:, :, token_offset:tokens_per_img, :]
    if k_first.shape[2] == 0:
        return 0.0

    # Compute attention of all queries over first-frame patch keys
    attn = q @ k_first.transpose(-2, -1)   # (B, H, N_q, N_k_first)
    attn = attn.transpose(-2, -1)           # (B, H, N_k_first, N_q)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)                 # (B, N_k_first, N_q) — avg over heads

    # Split attention by frame destination
    attn_to_first = attn[..., :tokens_per_img]   # first-frame self-attention
    attn_to_second = attn[..., tokens_per_img:]  # cross-frame attention to second

    # Ratio: how much cross-frame attention relative to self-attention peak
    max_self = attn_to_first.max(dim=-1)[0]      # (B, N_k_first)
    normalized = attn_to_second / (max_self.unsqueeze(-1) + 1e-8)
    ratio = normalized.max(dim=1)[0]              # (B, N_second)

    ratio_np = ratio.cpu().float().numpy().ravel()
    if ratio_np.size == 0:
        return 0.0
    return float(np.percentile(ratio_np, 90))
```

- [ ] **Step 1.4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_pointcloud_utils.py -k "cross_frame" -v
```

Expected: 4 tests PASS

- [ ] **Step 1.5: Commit**

```bash
git add collab_splats/pointcloud/utils.py tests/pointcloud/test_pointcloud_utils.py
git commit -m "feat(pointcloud): add cross_frame_attention_ratio to pointcloud/utils.py"
```

---

## Task 2: Abstract `extract_intermediate_features` + concrete `_verify_loop_candidate` on base

**Files:**
- Modify: `tests/pointcloud/test_feedforward_shared.py`
- Modify: `collab_splats/pointcloud/feedforward/base.py`

- [ ] **Step 2.1: Write failing tests**

Add to `tests/pointcloud/test_feedforward_shared.py`:

```python
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock

from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator


# ── Stub creator ──────────────────────────────────────────────────────────────

class _StubCreator(BaseFeedforwardCreator):
    """Minimal concrete subclass of BaseFeedforwardCreator for base-method tests.

    Implements all abstract methods.  Instantiate via object.__new__ to bypass
    the dataclass __init__ (no real model or paths needed for these unit tests).
    Set creator._stubbed_features before calling _verify_loop_candidate.
    """

    def _load_model(self, device): return None
    def _preprocess(self, image_paths, **kwargs): pass
    def _forward(self, model, views, **kwargs): pass
    def _postprocess(self, raw_outputs, **kwargs): pass
    def _reproject_after_ba(self, raw_outputs, extrinsics_3x4, intrinsics): pass

    def extract_intermediate_features(self, frames, layer_index=-1):
        return self._stubbed_features  # set per-test as instance attribute


def _make_stub() -> _StubCreator:
    """Bypass the dataclass __init__; no real model or paths needed."""
    return object.__new__(_StubCreator)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_verify_loop_candidate_rejected():
    """Ratio below threshold → (False, None) regardless of pose_enc presence."""
    creator = _make_stub()
    B, heads, hd = 1, 1, 4
    N = 20
    # Orthogonal frames → low cross-frame ratio
    k = torch.zeros(B, heads, N, hd)
    q = torch.zeros(B, heads, N, hd)
    k[:, :, :10, 0] = 10.0
    q[:, :, :10, 0] = 10.0
    k[:, :, 10:, 1] = 10.0
    q[:, :, 10:, 1] = 10.0
    creator._stubbed_features = {"q": q, "k": k}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, poses = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.85, layer_index=-1
    )
    assert accepted is False
    assert poses is None


def test_verify_loop_candidate_accepted_no_poses():
    """Ratio above threshold but no 'poses' key → (True, None)."""
    creator = _make_stub()
    B, heads, hd = 1, 2, 4
    torch.manual_seed(0)
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    creator._stubbed_features = {"q": q, "k": k}  # no "poses" key
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, poses = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.5, layer_index=-1
    )
    assert accepted is True
    assert poses is None


def test_verify_loop_candidate_accepted_with_poses():
    """Ratio above threshold + 'poses' key present → (True, (2,4,4) array)."""
    creator = _make_stub()
    B, heads, hd = 1, 2, 4
    torch.manual_seed(0)
    half = torch.randn(B, heads, 10, hd) * 10.0
    k = torch.cat([half, half], dim=2)
    q = torch.cat([half, half], dim=2)
    fake_poses = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)  # (2, 4, 4)
    creator._stubbed_features = {"q": q, "k": k, "poses": fake_poses}
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    accepted, poses = creator._verify_loop_candidate(
        frame1, frame2, verify_match_ratio=0.5, layer_index=-1
    )
    assert accepted is True
    assert poses is not None
    assert poses.shape == (2, 4, 4)


def test_verify_loop_candidate_layer_index_forwarded():
    """layer_index is passed through to extract_intermediate_features."""
    creator = _make_stub()
    captured_index = []

    def _capture_extract(frames, layer_index=-1):
        captured_index.append(layer_index)
        B, heads, hd = 1, 2, 4
        torch.manual_seed(0)
        half = torch.randn(B, heads, 10, hd) * 10.0
        k = torch.cat([half, half], dim=2)
        q = torch.cat([half, half], dim=2)
        return {"q": q, "k": k}

    creator.extract_intermediate_features = _capture_extract
    frame1 = torch.zeros(3, 16, 16)
    frame2 = torch.zeros(3, 16, 16)
    creator._verify_loop_candidate(frame1, frame2, verify_match_ratio=0.5, layer_index=2)
    assert captured_index == [2]


def test_extract_intermediate_features_is_abstract():
    """BaseFeedforwardCreator cannot be instantiated without extract_intermediate_features."""
    import inspect
    from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
    abstracts = {
        name for name, _ in inspect.getmembers(BaseFeedforwardCreator)
        if getattr(getattr(BaseFeedforwardCreator, name, None), '__isabstractmethod__', False)
    }
    assert 'extract_intermediate_features' in abstracts
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py -k "verify_loop_candidate or extract_intermediate_features_is_abstract" -v
```

Expected: `AttributeError` or `NotImplementedError` — `_verify_loop_candidate` not concrete yet, `extract_intermediate_features` not abstract yet.

- [ ] **Step 2.3: Update `collab_splats/pointcloud/feedforward/base.py`**

**2.3a — Add import at top of `base.py`** (near existing imports):

```python
from collab_splats.pointcloud.utils import cross_frame_attention_ratio
```

**2.3b — Add abstract method** in `BaseFeedforwardCreator` after `_postprocess` (around line 491):

```python
    @abstractmethod
    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Capture cross-frame transformer activations via a per-call forward hook.

        Register a forward hook on the QKV projection of the cross-frame attention
        block at `layer_index`, run a 2-frame model forward on `frames`, capture
        the intermediate activations, then remove the hook.  The hook is removed
        in a finally block — guaranteed even if the forward raises.

        Args:
            frames:      (N, C, H, W) preprocessed frames on the correct device.
            layer_index: Cross-frame block index.  -1 = last block (default, matches
                         VGGT-SPARK).  Valid range: [-len(blocks), len(blocks)-1].
                         VGGTx indexes into aggregator.global_blocks;
                         MapAnything into info_sharing.self_attention_blocks.
            **kwargs:    Backend-specific forward kwargs.
                         MapAnything: minibatch_size (int), memory_efficient_inference (bool).
                         VGGTx: unused.

        Returns:
            dict with at minimum:
              "q":  (B, heads, N_tokens, head_dim) — query projections
              "k":  (B, heads, N_tokens, head_dim) — key projections
            VGGTx additionally includes:
              "poses": (2, 4, 4) float32 np.ndarray — pre-decoded camera extrinsics.
            MapAnything omits "poses" — _verify_loop_candidate returns None for poses.
        """
        ...
```

**2.3c — Replace `_verify_loop_candidate`** (currently raises `NotImplementedError`).

`base.py` must NOT import from `vggt` and must NOT inspect backend-specific keys.  Pose
data (if any) arrives pre-decoded as `"poses"` in the features dict — that is part of the
shared return contract of `extract_intermediate_features`.  No `_decode_lc_poses` override
method is needed.

```python
    def _verify_loop_candidate(
        self,
        frame1: Any,
        frame2: Any,
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
            the backend includes a "poses" key in its extract_intermediate_features
            return dict; None otherwise — caller uses submap poses.
        """
        # Run the model once to capture cross-frame activations
        features = self.extract_intermediate_features(
            torch.stack([frame1, frame2]), layer_index=layer_index, **kwargs
        )
        # Compute the cross-frame attention ratio gate
        ratio = cross_frame_attention_ratio(features["k"], features["q"])
        if ratio < verify_match_ratio:
            return False, None
        # "poses" is optional — VGGTx includes it, MapAnything does not
        return True, features.get("poses")
```

Remove the old body (the `raise NotImplementedError(...)` line).

**2.3d — Update module docstring** at top of `base.py`. Change the `Provides:` block to note `cross_frame_attention_ratio` is in `pointcloud/utils.py`, and update `BaseFeedforwardCreator` description to mention `extract_intermediate_features`.

- [ ] **Step 2.4: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py -k "verify_loop_candidate or extract_intermediate_features_is_abstract" -v
```

Expected: 4 tests PASS

- [ ] **Step 2.5: Run full suite to check for regressions**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_feedforward_shared.py tests/pointcloud/test_base.py -v
```

Expected: all PASS

- [ ] **Step 2.6: Commit**

```bash
git add collab_splats/pointcloud/feedforward/base.py tests/pointcloud/test_feedforward_shared.py
git commit -m "refactor(feedforward): add abstract extract_intermediate_features; make _verify_loop_candidate concrete on base"
```

---

## Task 3: VGGTx — `extract_intermediate_features` + delete monkey-patch

**Files:**
- Modify: `tests/pointcloud/test_vggtx_creator.py`
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`

- [ ] **Step 3.1: Write failing tests**

Add to `tests/pointcloud/test_vggtx_creator.py`:

```python
import torch
import torch.nn as nn
from unittest.mock import MagicMock


def _make_vggtx_creator_with_mock_model(num_heads=2, head_dim=4, n_blocks=2, n_tokens=10):
    """Build a VGGTXCreator backed by a mock model with real nn.Linear QKV layers."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator

    total_dim = num_heads * head_dim

    # Build real nn.Linear so register_forward_hook actually works
    qkv_linears = [nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)]

    blocks = []
    for qkv in qkv_linears:
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = qkv
        blocks.append(block)

    # Mock VGGT model: forward calls qkv_linears[-1] so the hook fires
    def mock_forward(batch):
        B = batch.shape[0]
        x = torch.randn(B, n_tokens, total_dim)
        for qkv in qkv_linears:
            qkv(x)  # triggers any registered hook
        pose_enc = torch.zeros(B, 2, 9)
        return {"pose_enc": pose_enc}

    mock_model = MagicMock()
    mock_model.aggregator.global_blocks = blocks
    mock_model.side_effect = mock_forward
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def test_vggtx_extract_intermediate_features_shapes():
    """Hook fires on last block; q/k have correct shape; poses decoded and present."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_creator_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    result = creator.extract_intermediate_features(frames, layer_index=-1)
    # Required keys: q, k (activations) and poses (decoded, not raw pose_enc)
    assert "q" in result and "k" in result
    assert "pose_enc" not in result  # raw encoding must not leak out
    assert "poses" in result
    # q/k: (B=1, heads, n_tokens, head_dim)
    assert result["q"].shape == (1, num_heads, n_tokens, head_dim)
    assert result["k"].shape == (1, num_heads, n_tokens, head_dim)
    # poses: (2, 4, 4) float32 numpy array
    assert result["poses"].shape == (2, 4, 4)
    assert result["poses"].dtype == np.float32


def test_vggtx_extract_intermediate_features_hook_removed():
    """Hook is removed after the call — repeated calls don't accumulate hooks."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_creator_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    qkv = blocks[-1].attn.qkv
    assert len(qkv._forward_hooks) == 0
    creator.extract_intermediate_features(frames, layer_index=-1)
    assert len(qkv._forward_hooks) == 0  # hook removed


def test_vggtx_extract_intermediate_features_hook_removed_on_error():
    """Hook is removed even when the model forward raises."""
    from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator
    import torch.nn as nn

    num_heads, head_dim, n_tokens = 2, 4, 10
    total_dim = num_heads * head_dim
    qkv = nn.Linear(total_dim, total_dim * 3, bias=False)

    block = MagicMock()
    block.attn.num_heads = num_heads
    block.attn.qkv = qkv

    def boom(batch):
        raise RuntimeError("simulated forward failure")

    mock_model = MagicMock()
    mock_model.aggregator.global_blocks = [block]
    mock_model.side_effect = boom
    param = nn.Parameter(torch.zeros(1))
    mock_model.parameters = lambda: iter([param])

    creator = VGGTXCreator.__new__(VGGTXCreator)
    creator.model = mock_model

    with pytest.raises(RuntimeError, match="simulated forward failure"):
        creator.extract_intermediate_features(torch.zeros(2, 3, 16, 16))

    assert len(qkv._forward_hooks) == 0  # hook still removed


def test_vggtx_extract_intermediate_features_layer_index():
    """Non-default layer_index taps the correct block."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_vggtx_creator_with_mock_model(n_blocks=3)
    frames = torch.zeros(2, 3, 16, 16)

    # Hook should fire on blocks[1].attn.qkv, not blocks[2]
    hooked_blocks = []
    orig_register = blocks[1].attn.qkv.register_forward_hook

    def spy_register(fn):
        hooked_blocks.append(1)
        return orig_register(fn)

    blocks[1].attn.qkv.register_forward_hook = spy_register
    creator.extract_intermediate_features(frames, layer_index=1)
    assert hooked_blocks == [1]


def test_patch_vggtx_compute_similarity_deleted():
    """_patch_vggtx_compute_similarity must not exist after refactor."""
    import collab_splats.pointcloud.feedforward.vggtx as vggtx_mod
    assert not hasattr(vggtx_mod, '_patch_vggtx_compute_similarity'), (
        "_patch_vggtx_compute_similarity still exists — delete it and its _load_model call"
    )
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py -k "extract_intermediate or patch_vggtx" -v
```

Expected: `AttributeError: type object 'VGGTXCreator' has no attribute 'extract_intermediate_features'` + `test_patch_vggtx_compute_similarity_deleted` FAILS (function still exists).

- [ ] **Step 3.3: Delete `_patch_vggtx_compute_similarity` from `vggtx.py`**

Delete the entire function `_patch_vggtx_compute_similarity` (lines 39–142 approximately — the full function body including the nested `_get_similarity`, `_patched_agg_forward`, `_patched_vggt_forward`, the idempotency guard, and the final monkey-patch assignments).

Also delete these imports that are only used by the patch:

```python
from vggt.models.aggregator import Aggregator
from vggt.models.vggt import VGGT
```

- [ ] **Step 3.4: Remove `_patch_vggtx_compute_similarity()` call from `_load_model`**

In `VGGTXCreator._load_model` (around line 254), find and delete:

```python
        # In-tree workaround: see _patch_vggtx_compute_similarity (VGGT-X missing gate).
        _patch_vggtx_compute_similarity()
```

- [ ] **Step 3.5: Delete `_verify_loop_candidate` from `VGGTXCreator`**

Delete the entire `_verify_loop_candidate` method (lines 387–411 approximately).

- [ ] **Step 3.6: Add concrete `extract_intermediate_features` to `VGGTXCreator`**

`pose_encoding_to_extri_intri` is already imported at the top of `vggtx.py`.
Pose decoding happens *inside* this method — the return dict exposes `"poses"` (already
decoded np.ndarray), not raw `"pose_enc"`.  No `_decode_lc_poses` override is needed.

Add after `_reproject_after_ba`:

```python
    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Hook aggregator.global_blocks[layer_index].attn.qkv; return {q, k, poses}.

        Runs a 2-frame VGGT-X forward with a per-call hook on the QKV projection of the
        specified global attention block.  Captures q and k tensors, decodes the VGGT-X
        pose encoding, and returns pre-decoded (2, 4, 4) camera poses so
        _verify_loop_candidate does not need a second forward pass.

        The hook is removed in a finally block — guaranteed cleanup even if the
        forward raises.

        Args:
            frames:      (2, C, H, W) preprocessed frames (float16/32 on CPU or GPU).
            layer_index: Which global attention block to tap.  -1 = last (default).

        Returns:
            dict with:
              "q":     (B, heads, N_tokens, head_dim) query projections
              "k":     (B, heads, N_tokens, head_dim) key projections
              "poses": (2, 4, 4) float32 np.ndarray — decoded camera extrinsics
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        # Add batch dimension; move to model device + dtype
        batch = frames.unsqueeze(0).to(device, dtype=dtype)

        # Register a per-call hook on the QKV projection of the chosen block
        block = self.model.aggregator.global_blocks[layer_index]
        C_nh = block.attn.num_heads
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, _inp, out):
            # out: (B, N, 3*C) — split into q/k/v, reshape to (B, heads, N, head_dim)
            B, N, C3 = out.shape
            hd = (C3 // 3) // C_nh
            qkv = out.detach().reshape(B, N, 3, C_nh, hd).permute(2, 0, 3, 1, 4)
            captured["q"], captured["k"] = qkv[0], qkv[1]

        hook = block.attn.qkv.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                predictions = self.model(batch)
        finally:
            # Always remove the hook — no persistent state left on the model
            hook.remove()

        # Decode (2, 4, 4) camera extrinsics from VGGT-X pose encoding
        image_shape = (frames.shape[-2], frames.shape[-1])
        ext_3x4, _ = pose_encoding_to_extri_intri(
            predictions["pose_enc"].detach(), image_shape
        )
        ext_3x4 = ext_3x4.cpu().float().numpy().squeeze(0)  # (2, 3, 4)
        captured["poses"] = _extrinsics_3x4_to_4x4(ext_3x4)  # (2, 4, 4)
        return captured
```

- [ ] **Step 3.7: Update module docstring in `vggtx.py`**

Replace the `Provides:` block at the top:

```python
"""VGGT-X feedforward backend: inference utilities and creator.

Provides:
  VGGTX_IMG_LOAD_RESOLUTION      — fixed inference resolution for VGGT-X
  unproject_and_filter_points    — depth → world-space point cloud with confidence filtering
  VGGTXCreator                   — feedforward creator using VGGT-X depth + pose estimation
"""
```

- [ ] **Step 3.8: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py -k "extract_intermediate or patch_vggtx" -v
```

Expected: 5 tests PASS

- [ ] **Step 3.9: Run broader suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_feedforward_shared.py -v
```

Expected: all PASS

- [ ] **Step 3.10: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py tests/pointcloud/test_vggtx_creator.py
git commit -m "refactor(vggtx): replace monkey-patch with per-call forward hook; add extract_intermediate_features"
```

---

## Task 4: MapAnything — `extract_intermediate_features` + remove cosine workaround

**Files:**
- Modify: `tests/pointcloud/test_mapanything_creator.py`
- Modify: `collab_splats/pointcloud/feedforward/mapanything.py`

- [ ] **Step 4.1: Write failing tests**

Add to `tests/pointcloud/test_mapanything_creator.py`:

```python
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import pytest


def _make_mapanything_creator_with_mock_model(num_heads=2, head_dim=4, n_blocks=2, n_tokens=10):
    """Build a MapAnythingCreator backed by a mock model with real nn.Linear QKV layers."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator

    total_dim = num_heads * head_dim

    qkv_linears = [nn.Linear(total_dim, total_dim * 3, bias=False) for _ in range(n_blocks)]

    blocks = []
    for qkv in qkv_linears:
        block = MagicMock()
        block.attn.num_heads = num_heads
        block.attn.qkv = qkv
        blocks.append(block)

    def mock_forward(views):
        # Simulate info_sharing calling QKV on all blocks
        x = torch.randn(1, n_tokens, total_dim)
        for qkv in qkv_linears:
            qkv(x)
        return [{}]  # MapAnything returns list of output dicts

    mock_model = MagicMock()
    mock_model.info_sharing.self_attention_blocks = blocks
    mock_model.side_effect = mock_forward

    creator = MapAnythingCreator.__new__(MapAnythingCreator)
    creator.model = mock_model
    return creator, blocks, n_tokens, num_heads, head_dim


def test_mapanything_extract_intermediate_features_shapes():
    """Hook fires on last block; q/k have correct shape; no pose_enc."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_creator_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    result = creator.extract_intermediate_features(frames, layer_index=-1)
    assert "q" in result and "k" in result
    assert "pose_enc" not in result  # MapAnything has no pose_enc
    assert result["q"].shape == (1, num_heads, n_tokens, head_dim)
    assert result["k"].shape == (1, num_heads, n_tokens, head_dim)


def test_mapanything_extract_intermediate_features_hook_removed():
    """Hook is removed after the call."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_creator_with_mock_model()
    frames = torch.zeros(2, 3, 16, 16)
    qkv = blocks[-1].attn.qkv
    assert len(qkv._forward_hooks) == 0
    creator.extract_intermediate_features(frames, layer_index=-1)
    assert len(qkv._forward_hooks) == 0


def test_mapanything_extract_intermediate_features_layer_index():
    """Non-default layer_index taps correct block."""
    creator, blocks, n_tokens, num_heads, head_dim = _make_mapanything_creator_with_mock_model(n_blocks=3)
    frames = torch.zeros(2, 3, 16, 16)

    hooked_blocks = []
    orig_register = blocks[0].attn.qkv.register_forward_hook

    def spy_register(fn):
        hooked_blocks.append(0)
        return orig_register(fn)

    blocks[0].attn.qkv.register_forward_hook = spy_register
    creator.extract_intermediate_features(frames, layer_index=0)
    assert hooked_blocks == [0]


def test_mapanything_verify_loop_candidate_removed():
    """MapAnythingCreator must not define its own _verify_loop_candidate."""
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator
    assert '_verify_loop_candidate' not in MapAnythingCreator.__dict__, (
        "_verify_loop_candidate still defined on MapAnythingCreator — it should be on the base only"
    )
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py -k "extract_intermediate or verify_loop_candidate_removed" -v
```

Expected: `AttributeError: type object 'MapAnythingCreator' has no attribute 'extract_intermediate_features'`

- [ ] **Step 4.3: Delete `_verify_loop_candidate` from `MapAnythingCreator`**

In `collab_splats/pointcloud/feedforward/mapanything.py`, delete the entire `_verify_loop_candidate` method (lines 227–244 approximately).

- [ ] **Step 4.4: Remove `import torch.nn.functional as F`**

In `mapanything.py`, delete line 16:

```python
import torch.nn.functional as F
```

- [ ] **Step 4.5: Add concrete `extract_intermediate_features` to `MapAnythingCreator`**

MapAnything's `_forward` calls `self.model.forward(self._processed_views, memory_efficient_inference=True, minibatch_size=...)`.  `_processed_views` are built during `_preprocess` by `preprocess_input_views_for_inference` from a list of raw view dicts `{"img": tensor}`.

For `extract_intermediate_features`, the input `frames` are already-preprocessed tensors `(2, C, H, W)`.  Wrap them into raw view dicts and call `preprocess_input_views_for_inference`, then forward.

Add after `_reproject_after_ba`:

```python
    def extract_intermediate_features(
        self, frames: torch.Tensor, layer_index: int = -1
    ) -> dict[str, torch.Tensor]:
        """Hook info_sharing.self_attention_blocks[layer_index].attn.qkv; return {q, k}.

        Wraps the 2 input frames into MapAnything's view format, runs a forward pass
        with a per-call hook on the cross-frame self-attention block at `layer_index`,
        then removes the hook.  MapAnything has no pose_enc, so only q and k are
        returned — _verify_loop_candidate returns None for fresh poses.

        Args:
            frames:      (2, C, H, W) preprocessed frames on CPU or GPU.
            layer_index: Which self_attention_block to tap.  -1 = last (default).

        Returns:
            dict with keys "q", "k" — (1, heads, N_tokens, head_dim) tensors.
        """
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
                # frames: (2, C, H, W) — wrap into the {"img": (1,C,H,W)} view dicts
                # that preprocess_input_views_for_inference expects, then forward.
                raw_views = [{"img": f.unsqueeze(0)} for f in frames.cpu()]
                views = preprocess_input_views_for_inference(raw_views)
                self.model.forward(
                    views, memory_efficient_inference=False, minibatch_size=1
                )
        finally:
            hook.remove()

        return captured
```

> **If the hook doesn't fire:** `self.model.info_sharing` may not be called with these 2-frame views via `preprocess_input_views_for_inference`. Open `_preprocess` in `mapanything.py` and trace what `preprocess_input_views_for_inference` returns — the output of that call is exactly what `_forward` passes to `self.model.forward`. If the raw view dict needs more keys (e.g. camera data), check `load_images` in the existing `_preprocess` body and match that format.

- [ ] **Step 4.6: Run tests**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py -k "extract_intermediate or verify_loop_candidate_removed" -v
```

Expected: 4 tests PASS

- [ ] **Step 4.7: Run broader suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_mapanything_creator.py tests/pointcloud/test_mapanything.py -v
```

Expected: all PASS

- [ ] **Step 4.8: Commit**

```bash
git add collab_splats/pointcloud/feedforward/mapanything.py tests/pointcloud/test_mapanything_creator.py
git commit -m "refactor(mapanything): add extract_intermediate_features hook; remove cosine-similarity workaround"
```

---

## Task 5: Clean up `feedforward/__init__.py`

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/__init__.py`

- [ ] **Step 5.1: Remove `_patch_vggtx_compute_similarity` re-export**

In `collab_splats/pointcloud/feedforward/__init__.py`, delete:

```python
# ── Compat patches (re-exported for tests that patch or call directly) ────────
from .vggtx import _patch_vggtx_compute_similarity
```

Also remove `_patch_vggtx_compute_similarity` from `__all__` if it appears there.

- [ ] **Step 5.2: Check for any test that imports `_patch_vggtx_compute_similarity`**

```bash
grep -r "_patch_vggtx_compute_similarity" /workspace/collab-splats/tests/
```

If any test imports it, update those tests to remove the import (the function is gone).

- [ ] **Step 5.3: Run full pointcloud test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/ -v
```

Expected: all previously-passing tests still PASS; `test_patch_vggtx_compute_similarity_deleted` now passes.

- [ ] **Step 5.4: Commit**

```bash
git add collab_splats/pointcloud/feedforward/__init__.py
git commit -m "refactor(feedforward): remove _patch_vggtx_compute_similarity re-export from __init__"
```

---

## Task 6: VGGTx structural parity + docstring cleanup

**Files:**
- Modify: `collab_splats/pointcloud/feedforward/vggtx.py`

No new tests needed (documentation + structure only).

- [ ] **Step 6.1: Verify section structure matches MapAnything**

`mapanything.py` structure (reference):
```
# ── Inference utilities ────────────────────────────────────────────────────────
def collect_pts3d_from_outputs(...)
...

# ── Creator ───────────────────────────────────────────────────────────────────
@dataclass
class MapAnythingCreator(BaseFeedforwardCreator):
    ...
```

`vggtx.py` should match:
```
# ── Constants ─────────────────────────────────────────────────────────────────
VGGTX_IMG_LOAD_RESOLUTION = 518

# ── Inference utilities ────────────────────────────────────────────────────────
def unproject_and_filter_points(...)

# ── Creator ───────────────────────────────────────────────────────────────────
@dataclass
class VGGTXCreator(BaseFeedforwardCreator):
    ...
```

Adjust `########`-style dividers to match the `──` style used in `mapanything.py`.

- [ ] **Step 6.2: Add descriptive docstrings to `VGGTXCreator` methods that lack them**

Each method should have a one-line summary + Args/Returns matching MapAnything's style.  Methods to check: `_load_model`, `_preprocess`, `_forward`, `_postprocess`, `_reproject_after_ba`.  The `extract_intermediate_features` docstring was written in Task 3.

- [ ] **Step 6.3: Run suite to verify no regressions from formatting changes**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/pointcloud/test_vggtx_creator.py tests/pointcloud/test_vggtx_preproc.py -v
```

Expected: all PASS

- [ ] **Step 6.4: Commit**

```bash
git add collab_splats/pointcloud/feedforward/vggtx.py
git commit -m "refactor(vggtx): structural parity with mapanything — section dividers and method docstrings"
```

---

## Task 7: Final regression check

- [ ] **Step 7.1: Run full test suite**

```bash
/opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: same pass count as before this branch started (check `worklog/known-test-failures.md` for pre-existing failures to exclude).

- [ ] **Step 7.2: Verify `cross_frame_attention_ratio` is importable from utils**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "from collab_splats.pointcloud.utils import cross_frame_attention_ratio; print('ok')"
```

Expected: `ok`

- [ ] **Step 7.3: Verify `extract_intermediate_features` is on both creators**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator
assert hasattr(VGGTXCreator, 'extract_intermediate_features')
assert hasattr(MapAnythingCreator, 'extract_intermediate_features')
print('both ok')
"
```

Expected: `both ok`

---

## Task 8: End-to-end package verification

- [ ] **Step 8.1: Verify the full `collab_splats` package imports cleanly**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import collab_splats
import collab_splats.pointcloud.feedforward
from collab_splats.pointcloud.feedforward import VGGTXCreator, MapAnythingCreator, BaseFeedforwardCreator
from collab_splats.pointcloud.utils import cross_frame_attention_ratio
import inspect
# Confirm extract_intermediate_features is abstract on base
abstracts = {n for n, _ in inspect.getmembers(BaseFeedforwardCreator)
             if getattr(getattr(BaseFeedforwardCreator, n, None), '__isabstractmethod__', False)}
assert 'extract_intermediate_features' in abstracts, 'not abstract on base'
# Confirm _patch_vggtx_compute_similarity is gone
import collab_splats.pointcloud.feedforward.vggtx as vmod
assert not hasattr(vmod, '_patch_vggtx_compute_similarity'), 'patch fn still exists'
# Confirm no torch.nn.functional.cosine_similarity import in mapanything
import collab_splats.pointcloud.feedforward.mapanything as mmod
import inspect as ins
src = ins.getsource(mmod)
assert 'cosine_similarity' not in src, 'cosine_similarity still imported in mapanything'
print('package e2e ok')
"
```

Expected: `package e2e ok`

- [ ] **Step 8.2: Verify `cross_frame_attention_ratio` runs correctly end-to-end**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
import torch
from collab_splats.pointcloud.utils import cross_frame_attention_ratio
B, heads, hd = 1, 2, 8
half = torch.randn(B, heads, 20, hd) * 10.0
k = torch.cat([half, half], dim=2)
q = torch.cat([half, half], dim=2)
ratio = cross_frame_attention_ratio(k, q, token_offset=0)
print(f'ratio={ratio:.3f}')
assert ratio > 0.5, f'expected high ratio for identical frames, got {ratio}'
print('cross_frame_attention_ratio e2e ok')
"
```

Expected: `ratio=<value>` then `cross_frame_attention_ratio e2e ok`

- [ ] **Step 8.3: Verify the loop closure pipeline imports correctly**

```bash
/opt/conda/envs/nerfstudio/bin/python -c "
from collab_splats.pointcloud.wrappers import LoopClosure
from collab_splats.pointcloud.loop_closure import LoopClosureConfig
print('loop closure imports ok')
"
```

Expected: `loop closure imports ok`

- [ ] **Step 8.4: Final commit**

```bash
git add -p  # review any unstaged formatting changes from Task 6
git commit -m "chore(feedforward): end-to-end verification pass — all package imports and cross_frame_attention_ratio confirmed"
```
