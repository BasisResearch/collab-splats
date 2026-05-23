# Model-Free query_mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate `eval_setup` (full nerfstudio pipeline load) from `query_mesh()` by saving the decoder state dict at extract time and reconstructing from it — removing training-image memory pressure entirely.

**Architecture:** `_extract_mesh_features` saves `mesh_decoder.pt` (decoder state dict, ~130KB) alongside `mesh_features.pt`. `query_mesh` detects this file and reconstructs `TwoLayerMLP` + text encoder from registry — no datamanager, no training images. `eval_setup` retained as fallback for legacy data.

**Tech Stack:** PyTorch (`torch.save`/`torch.load`), `types.SimpleNamespace`, existing `TwoLayerMLP`, `BaseFeatureExtractor` registry, `_QUERYABLE_FEATURE_TYPES`, `_TEXT_ENCODER_NAME` constants already in `rade_features.py`.

---

### Task 1: Save decoder state dict in `_extract_mesh_features`

**Files:**
- Modify: `collab_splats/wrapper/splatter.py` (`_extract_mesh_features`)
- Modify: `tests/wrapper/test_splatter_mesh.py`

- [ ] **Step 1: Write failing test**

Add to `tests/wrapper/test_splatter_mesh.py`:

```python
def test_extract_mesh_features_saves_decoder(tmp_path):
    """_extract_mesh_features must write mesh_decoder.pt next to mesh_features.pt."""
    import torch
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace

    # Build a minimal fake model with a decoder that has a state_dict
    fake_decoder = MagicMock()
    fake_decoder.state_dict.return_value = {"hidden_conv.weight": torch.zeros(64, 13, 1, 1)}

    fake_model = SimpleNamespace(
        decoder=fake_decoder,
        main_features_name="maskclip",
        device="cpu",
        means=torch.zeros(10, 3),
    )

    mesh_path = tmp_path / "mesh_tsdf_clean.ply"
    mesh_path.touch()

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {"mesh": mesh_path, "features": tmp_path / "mesh_features.pt"},
    }
    splatter.model = fake_model

    import numpy as np
    vertex_features = np.zeros((5, 13), dtype=np.float32)
    torch.save(torch.from_numpy(vertex_features), tmp_path / "mesh_features.pt")

    with patch("collab_splats.wrapper.splatter.features2vertex", return_value=vertex_features), \
         patch("collab_splats.wrapper.splatter.o3d") as mock_o3d:
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.zeros((5, 3))
        mock_o3d.io.read_triangle_mesh.return_value = mock_mesh
        splatter._extract_mesh_features(features_name="distill_features")

    assert (tmp_path / "mesh_decoder.pt").exists(), "mesh_decoder.pt not written"
    saved = torch.load(tmp_path / "mesh_decoder.pt", map_location="cpu")
    assert "hidden_conv.weight" in saved
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_mesh.py::test_extract_mesh_features_saves_decoder -v 2>&1 | tail -20
```

Expected: FAIL — `mesh_decoder.pt` not written / assertion error.

- [ ] **Step 3: Add the save line to `_extract_mesh_features`**

In `collab_splats/wrapper/splatter.py`, locate `_extract_mesh_features`. After the line:
```python
torch.save(torch.from_numpy(vertex_features).float(), features_path)
```
Add:
```python
torch.save(self.model.decoder.state_dict(), features_path.parent / "mesh_decoder.pt")
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_mesh.py::test_extract_mesh_features_saves_decoder -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/splatter.py tests/wrapper/test_splatter_mesh.py
git commit -m "feat(splatter): save mesh_decoder.pt in _extract_mesh_features"
```

---

### Task 2: Fast-path `query_mesh` — reconstruct decoder + text encoder without eval_setup

**Files:**
- Modify: `collab_splats/wrapper/splatter.py` (`query_mesh`)
- Modify: `tests/wrapper/test_splatter_query.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/wrapper/test_splatter_query.py`:

```python
def _make_decoder_state(input_dim=13, hidden_dim=4, output_dim=8, feature_type="maskclip"):
    """Build a minimal decoder state dict matching TwoLayerMLP structure."""
    import torch
    return {
        "hidden_conv.weight": torch.randn(hidden_dim, input_dim, 1, 1),
        "hidden_conv.bias":   torch.zeros(hidden_dim),
        f"feature_branch_dict.{feature_type}.weight": torch.randn(output_dim, hidden_dim, 1, 1),
        f"feature_branch_dict.{feature_type}.bias":   torch.zeros(output_dim),
    }


def test_query_mesh_fast_path_skips_eval_setup(tmp_path):
    """When mesh_decoder.pt exists, query_mesh must not call eval_setup."""
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch

    mesh_path = tmp_path / "mesh_tsdf_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, 13)
    torch.save(features, tmp_path / "mesh_features.pt")
    torch.save(_make_decoder_state(), tmp_path / "mesh_decoder.pt")

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_extractor = MagicMock()
    mock_extractor.score_queries.return_value = fake_scores

    with patch("collab_splats.wrapper.splatter.eval_setup") as mock_eval, \
         patch("collab_splats.wrapper.splatter.BaseFeatureExtractor") as mock_bfe:
        mock_bfe.get.return_value = lambda **kw: mock_extractor
        result = splatter.query_mesh(positive_queries=["feeder"])

    mock_eval.assert_not_called()
    assert result.shape == (5, 3)


def test_query_mesh_fast_path_decoder_reconstructed_from_state(tmp_path):
    """Decoder is correctly reconstructed: output dim inferred from weight shape."""
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch

    input_dim, hidden_dim, output_dim = 13, 4, 8
    mesh_path = tmp_path / "mesh_tsdf_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, input_dim)
    torch.save(features, tmp_path / "mesh_features.pt")
    torch.save(_make_decoder_state(input_dim, hidden_dim, output_dim), tmp_path / "mesh_decoder.pt")

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_extractor = MagicMock()
    mock_extractor.score_queries.return_value = fake_scores

    with patch("collab_splats.wrapper.splatter.eval_setup"), \
         patch("collab_splats.wrapper.splatter.BaseFeatureExtractor") as mock_bfe:
        mock_bfe.get.return_value = lambda **kw: mock_extractor
        splatter.query_mesh(positive_queries=["feeder"])

    # Check that score_queries was called with something shaped (output_dim, N, 1)
    call_args = mock_extractor.score_queries.call_args
    feat_arg = call_args[1]["features"] if "features" in call_args[1] else call_args[0][0]
    assert feat_arg.shape[0] == output_dim


def test_query_mesh_falls_back_to_eval_setup_when_no_decoder(tmp_path):
    """When mesh_decoder.pt is absent, query_mesh falls back to eval_setup."""
    import torch
    from unittest.mock import MagicMock, patch
    from types import SimpleNamespace

    mesh_path = tmp_path / "mesh_tsdf_clean.ply"
    mesh_path.touch()
    features = torch.zeros(5, 13)
    torch.save(features, tmp_path / "mesh_features.pt")
    # No mesh_decoder.pt written

    splatter = object.__new__(Splatter)
    splatter.config = {
        "model_config_path": "dummy/config.yml",
        "mesh_info": {
            "mesh": mesh_path,
            "features": tmp_path / "mesh_features.pt",
        },
    }

    fake_scores = torch.zeros(5)
    mock_model = MagicMock()
    mock_model.main_features_name = "maskclip"
    mock_model.device = "cpu"
    mock_model.decoder.per_gaussian_forward.return_value = {"maskclip": torch.zeros(5, 8)}
    mock_model.similarity_fx.return_value = fake_scores

    mock_pipeline = MagicMock()
    mock_pipeline.model = mock_model

    with patch("collab_splats.wrapper.splatter.eval_setup", return_value=(None, mock_pipeline, None, None)) as mock_eval:
        result = splatter.query_mesh(positive_queries=["feeder"])

    mock_eval.assert_called_once()
    assert result.shape == (5, 3)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_query.py::test_query_mesh_fast_path_skips_eval_setup tests/wrapper/test_splatter_query.py::test_query_mesh_fast_path_decoder_reconstructed_from_state tests/wrapper/test_splatter_query.py::test_query_mesh_falls_back_to_eval_setup_when_no_decoder -v 2>&1 | tail -20
```

Expected: FAIL.

- [ ] **Step 3: Implement fast path in `query_mesh`**

In `collab_splats/wrapper/splatter.py`, add to imports at top of file:
```python
from types import SimpleNamespace
```

Locate `query_mesh`. Replace the block:
```python
        if not self.config.get("model_config_path"):
            self._select_run()
        elif getattr(self, "model", None) is None:
            print(f"Loading model from {self.config['model_config_path']}")
            from nerfstudio.utils.eval_utils import eval_setup

            _, pipeline, _, _ = eval_setup(Path(self.config["model_config_path"]))
            self.model = pipeline.model
```

With:
```python
        if getattr(self, "model", None) is None:
            if not self.config.get("model_config_path"):
                self._select_run()
            mesh_dir = self.config["mesh_info"]["mesh"].parent
            decoder_path = mesh_dir / "mesh_decoder.pt"
            if decoder_path.exists():
                from collab_splats.nerfstudio.models.rade_features import (
                    TwoLayerMLP, _QUERYABLE_FEATURE_TYPES, _TEXT_ENCODER_NAME,
                )
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
                print(f"Loading model from {self.config['model_config_path']}")
                from nerfstudio.utils.eval_utils import eval_setup
                _, pipeline, _, _ = eval_setup(Path(self.config["model_config_path"]))
                self.model = pipeline.model
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/test_splatter_query.py -v 2>&1 | tail -20
```

Expected: all pass (including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add collab_splats/wrapper/splatter.py tests/wrapper/test_splatter_query.py
git commit -m "feat(splatter): model-free query_mesh fast path via mesh_decoder.pt"
```

---

### Task 3: Run full wrapper test suite and verify no regressions

**Files:**
- No changes — verification only.

- [ ] **Step 1: Run full wrapper tests**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/wrapper/ -v 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 2: Verify `mesh_decoder.pt` not written by `mesh()` else-branch (existing data path)**

The else-branch in `mesh()` only reads file paths — it does NOT call `_extract_mesh_features`, so it cannot write `mesh_decoder.pt`. Confirm by reading `mesh()` else-branch in `splatter.py` (lines ~466–479) and confirming no `_extract_mesh_features` call there. This is expected: existing data re-generates via `mesh(overwrite=True)`.

- [ ] **Step 3: Commit if any fixups needed; otherwise done**

```bash
git status  # expect clean
```
