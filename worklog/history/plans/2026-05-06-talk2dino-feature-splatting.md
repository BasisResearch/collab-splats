# Talk2DINO Feature Splatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Talk2DINO as a feature space for Gaussian splatting with text-query inference, gating segmentation exclusively to the new `"samclip"` mode (maskclip + SAM).

**Architecture:** Add `"samclip"` as a config-level alias in `FeatureSplattingDataManager` — internally it uses the `"maskclip"` extractor plus SAM segmentation. `"maskclip"` and `"talk2dino"` skip segmentation entirely. The model's `populate_text_encoder` uses an explicit allowlist + alias map instead of a fragile string check, enabling talk2dino to have `similarity_fx` wired up for text-query inference.

**Tech Stack:** PyTorch, nerfstudio, `collab_splats.semantics.features.BaseFeatureExtractor`, `collab_splats.semantics.segmentation.Segmentation`, pytest + `unittest.mock`

---

## File Map

| File | Change |
|---|---|
| `collab_splats/nerfstudio/datamanagers/features.py` | Add `"samclip"` literal + default, add `_EXTRACTOR_NAME`, gate segmentation |
| `collab_splats/nerfstudio/models/rade_features.py` | Add `_QUERYABLE_FEATURE_TYPES` + `_TEXT_ENCODER_NAME`, update `populate_text_encoder` |
| `tests/nerfstudio/test_datamanager_config.py` | Update existing default test, add `samclip` config test + segmentation gate test |
| `tests/test_models.py` | Add `test_populate_text_encoder_talk2dino` |

No changes needed to `collab_splats/nerfstudio/method_configs/rade_features.py` — its `FeatureSplattingDataManagerConfig(...)` instantiation doesn't pass `main_features`, so the new default `"samclip"` is picked up automatically.

---

### Task 1: Update config — add `"samclip"`, change default

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py:41-48`
- Modify: `tests/nerfstudio/test_datamanager_config.py`

- [ ] **Step 1: Update the failing test for the default value**

In `tests/nerfstudio/test_datamanager_config.py`, replace `test_main_features_accepts_maskclip` and add `test_main_features_accepts_samclip`:

```python
def test_main_features_default_is_samclip():
    """Default is samclip (maskclip + SAM segmentation)."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig()
    assert config.main_features == "samclip"


def test_main_features_accepts_samclip():
    """samclip is a valid main_features value."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(main_features="samclip")
    assert config.main_features == "samclip"


def test_main_features_accepts_maskclip():
    """maskclip (no segmentation) is a valid main_features value."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(main_features="maskclip")
    assert config.main_features == "maskclip"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py -v 2>&1 | tail -20
```

Expected: `test_main_features_default_is_samclip` FAILS (`assert 'maskclip' == 'samclip'`), `test_main_features_accepts_samclip` FAILS (invalid literal).

- [ ] **Step 3: Update `FeatureSplattingDataManagerConfig`**

In `collab_splats/nerfstudio/datamanagers/features.py`, replace lines 41–48:

```python
    main_features: Literal["maskclip", "samclip", "talk2dino"] = "samclip"
    """Feature extractor for main training features.

    "samclip": patch CLIP features masked by SAM segmentation (default).
        Use with regularization_features="dinov2".
    "maskclip": patch CLIP features, no segmentation.
        Use with regularization_features="dinov2".
    "talk2dino": DINOv3 + CLIP projection — structural grounding is built in;
        set regularization_features=None. Images are center-cropped to
        square during extraction (feature spatial coverage != full frame).
    """
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py -v 2>&1 | tail -20
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/nerfstudio/datamanagers/features.py tests/nerfstudio/test_datamanager_config.py
git commit -m "feat(datamanager): add samclip as default main_features, keep maskclip as no-seg option"
```

---

### Task 2: Gate segmentation on `"samclip"` in `extract_features`

**Files:**
- Modify: `collab_splats/nerfstudio/datamanagers/features.py`
- Modify: `tests/nerfstudio/test_datamanager_config.py`

- [ ] **Step 1: Write the failing segmentation gate test**

Append to `tests/nerfstudio/test_datamanager_config.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import torch


@pytest.mark.parametrize("main_features,expect_seg", [
    ("samclip", True),
    ("maskclip", False),
    ("talk2dino", False),
])
def test_segmentation_gate(main_features, expect_seg):
    """Segmentation is instantiated only when main_features='samclip'."""
    from torch import nn
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager

    config = MagicMock()
    config.main_features = main_features
    config.regularization_features = None
    config.obj_resolution = 4
    config.final_resolution = 4
    config.sam_resolution = 16
    config.segmentation_backend = "mobilesamv2"
    config.segmentation_strategy = "object"

    class _FakeImage:
        height = 16
        width = 16

    mock_extractor = MagicMock()
    mock_extractor.forward.return_value = [torch.zeros(8, 4, 4)]

    manager = object.__new__(FeatureSplattingDataManager)
    manager.config = config

    with (
        patch("collab_splats.nerfstudio.datamanagers.features.Image.open", return_value=_FakeImage()),
        patch(
            "collab_splats.nerfstudio.datamanagers.features.BaseFeatureExtractor.get",
            return_value=MagicMock(return_value=mock_extractor),
        ),
        patch("collab_splats.nerfstudio.datamanagers.features.Segmentation") as mock_seg_cls,
        patch("collab_splats.nerfstudio.datamanagers.features.resize_image", return_value=_FakeImage()),
        patch("collab_splats.nerfstudio.datamanagers.features.pytorch_gc"),
        patch("torch.cuda.empty_cache"),
        patch("gc.collect"),
    ):
        mock_seg_instance = MagicMock()
        mock_seg_instance.segment.return_value = None
        mock_seg_cls.return_value = mock_seg_instance

        manager.extract_features(["fake_path.jpg"])

    if expect_seg:
        mock_seg_cls.assert_called_once()
    else:
        mock_seg_cls.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py::test_segmentation_gate -v 2>&1 | tail -20
```

Expected: `samclip` PASSES (segmentation already called), `maskclip` and `talk2dino` FAIL (`assert_not_called` fails since Segmentation always instantiates now). Also `talk2dino` may error with `BaseFeatureExtractor` registry miss — that's expected.

- [ ] **Step 3: Add `_EXTRACTOR_NAME` and update `extract_features`**

In `collab_splats/nerfstudio/datamanagers/features.py`:

**After the imports block (after line 32), add at module level:**

```python
_EXTRACTOR_NAME: dict[str, str] = {"samclip": "maskclip"}
```

**Replace lines 178–239** (the main-features extraction block) with:

```python
        # Extract main features
        extractor_name = _EXTRACTOR_NAME.get(self.config.main_features, self.config.main_features)
        extractor = BaseFeatureExtractor.get(extractor_name)(device=device)
        use_seg = self.config.main_features == "samclip"
        if use_seg:
            segmentation = Segmentation(
                backend=self.config.segmentation_backend,
                strategy=self.config.segmentation_strategy,
                device=device,
            )

        # Add empty list for main features
        features_dict[self.config.main_features] = []

        for i in trange(
            len(image_filenames),
            desc=f"Extracting {self.config.main_features} features",
        ):
            # Load and process image
            image = Image.open(image_filenames[i])
            H, W = image.height, image.width

            # Calculate resolutions
            object_W = self.config.obj_resolution
            object_H = H * object_W // W
            final_W = self.config.final_resolution
            final_H = H * final_W // W

            # Extract features
            [features] = extractor.forward([image])

            if use_seg:
                # Prepare image for segmentation
                image = resize_image(image, self.config.sam_resolution)
                image = np.asarray(image)

                # Apply segmentation masks over features
                seg_outputs = segmentation.segment(image)

                # Add an all-zero tensor if no object is detected
                if seg_outputs is None:
                    features_dict[self.config.main_features].append(
                        torch.zeros((features.shape[0], final_H, final_W))
                    )
                    del features
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

                masks = seg_outputs[0]
                features = aggregate_masked_features(
                    features,
                    masks,
                    resolution=(object_H, object_W),
                    final_resolution=(final_H, final_W),
                )
                del masks

            features = features.detach().cpu()
            features_dict[self.config.main_features].append(features)

            # Clear memory after each image
            del features
            torch.cuda.empty_cache()
            gc.collect()

        del extractor
        if use_seg:
            del segmentation
        pytorch_gc()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/nerfstudio/test_datamanager_config.py -v 2>&1 | tail -20
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/nerfstudio/datamanagers/features.py tests/nerfstudio/test_datamanager_config.py
git commit -m "feat(datamanager): gate SAM segmentation on samclip only, add _EXTRACTOR_NAME alias"
```

---

### Task 3: Wire up `similarity_fx` for all queryable feature types in model

**Files:**
- Modify: `collab_splats/nerfstudio/models/rade_features.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_populate_text_encoder_talk2dino(scene_box):
    """populate_text_encoder wires up similarity_fx for talk2dino feature type."""
    from unittest.mock import patch, MagicMock
    import torch.nn as nn

    class _FakeEncoder(nn.Module):
        def score_queries(self, *args, **kwargs):
            pass

    mock_encoder_cls = MagicMock(return_value=_FakeEncoder())

    cfg = RadegsFeaturesModelConfig(output_depth_during_training=False)
    cfg.sh_degree = 0

    metadata = {
        "feature_type": "talk2dino",
        "feature_dims": {
            "talk2dino": (8, 4, 4),
        },
    }

    with patch(
        "collab_splats.nerfstudio.models.rade_features.BaseFeatureExtractor.get",
        return_value=mock_encoder_cls,
    ):
        model = RadegsFeaturesModel(
            cfg,
            scene_box=scene_box,
            num_train_data=1,
            metadata=metadata,
        )

    assert model.similarity_fx is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_models.py::test_populate_text_encoder_talk2dino -v 2>&1 | tail -20
```

Expected: FAIL — `assert None is not None` (talk2dino doesn't contain "clip" so `similarity_fx = None`).

- [ ] **Step 3: Add module-level constants and update `populate_text_encoder`**

In `collab_splats/nerfstudio/models/rade_features.py`, after the imports block (after `from collab_splats.semantics.features import BaseFeatureExtractor`, before `class TwoLayerMLP`), add:

```python
_QUERYABLE_FEATURE_TYPES: frozenset[str] = frozenset({"maskclip", "samclip", "talk2dino"})
_TEXT_ENCODER_NAME: dict[str, str] = {"samclip": "maskclip"}
```

Replace `populate_text_encoder` (lines 154–171):

```python
    def populate_text_encoder(self):
        feature_type = self.kwargs["metadata"]["feature_type"]
        if feature_type in _QUERYABLE_FEATURE_TYPES:
            encoder_name = _TEXT_ENCODER_NAME.get(feature_type, feature_type)
            self.text_encoder = BaseFeatureExtractor.get(encoder_name)(device=self.device)
            self.add_module("text_encoder", self.text_encoder)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.similarity_fx = self.text_encoder.score_queries
        else:
            self.similarity_fx = None
```

- [ ] **Step 4: Run all model tests to verify they pass**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/test_models.py -v 2>&1 | tail -20
```

Expected: all tests PASS including `test_populate_text_encoder_talk2dino` and existing `test_radegs_features_model` (maskclip still works).

- [ ] **Step 5: Run full test suite**

```bash
cd /workspace/collab-splats && /opt/conda/envs/nerfstudio/bin/python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /workspace/collab-splats && git add collab_splats/nerfstudio/models/rade_features.py tests/test_models.py
git commit -m "feat(model): wire similarity_fx for talk2dino via _QUERYABLE_FEATURE_TYPES allowlist"
```
