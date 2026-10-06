from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from collab_splats.localization import (
    BaseRetrievalExtractor,
    DinoSaladExtractor,
    PECLIPExtractor,
)
from collab_splats.utils.image import IMAGENET_MEAN, IMAGENET_STD


def test_registry_get_dino_salad():
    cls = BaseRetrievalExtractor.get("dino-salad")
    assert cls is DinoSaladExtractor


def test_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        BaseRetrievalExtractor.get("nonexistent-model")


def test_base_extractor_forward_abstract():
    with pytest.raises(TypeError):
        BaseRetrievalExtractor()


########################################################
########## PECLIPExtractor ############################
########################################################


def test_registry_get_pe_clip():
    cls = BaseRetrievalExtractor.get("pe-clip")
    assert cls is PECLIPExtractor


def test_pe_clip_forward_shape_and_norm():
    """forward() returns (1, 1024) unit-norm tensor without loading real weights."""
    fake_img_emb = torch.randn(1, 1024)
    fake_img_emb = fake_img_emb / fake_img_emb.norm(dim=-1, keepdim=True)

    mock_model = MagicMock()
    mock_model.encode_image.return_value = fake_img_emb
    mock_model.context_length = 32

    mock_preprocess = MagicMock(return_value=torch.zeros(3, 336, 336))

    with patch("collab_splats.localization.retrieval.open_clip") as mock_oc:
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, mock_preprocess)
        mock_oc.get_tokenizer.return_value = MagicMock(return_value=torch.zeros(1, 32, dtype=torch.long))
        extractor = PECLIPExtractor(device="cpu")

    img = Image.new("RGB", (336, 336))
    result = extractor([img])

    assert result.shape == (1, 1024)
    norms = result.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(1), atol=1e-5, rtol=0)


def test_pe_clip_encode_text_shape_and_norm():
    """encode_text() returns (2, 1024) unit-norm tensor."""
    fake_text_emb = torch.randn(2, 1024)
    fake_text_emb = fake_text_emb / fake_text_emb.norm(dim=-1, keepdim=True)

    mock_model = MagicMock()
    mock_model.encode_text.return_value = fake_text_emb
    mock_model.context_length = 32

    with patch("collab_splats.localization.retrieval.open_clip") as mock_oc:
        mock_oc.create_model_and_transforms.return_value = (mock_model, None, MagicMock())
        mock_tokenizer = MagicMock(return_value=torch.zeros(2, 32, dtype=torch.long))
        mock_oc.get_tokenizer.return_value = mock_tokenizer
        extractor = PECLIPExtractor(device="cpu")

    result = extractor.encode_text(["a cat", "a dog"])

    assert result.shape == (2, 1024)
    norms = result.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=0)


def test_dino_salad_tensor_input_matches_pil_input():
    """A [0, 1] tensor batch reaches the backbone normalized exactly like the PIL path."""
    # Extractor without weights; the backbone records what it receives
    extractor = DinoSaladExtractor.__new__(DinoSaladExtractor)
    torch.nn.Module.__init__(extractor)
    extractor._device = "cpu"
    extractor._transform = T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    seen = []
    extractor.backbone = lambda x: seen.append(x) or x
    extractor.aggregator = lambda x: x.flatten(1)

    # Same 224x224 image as PIL and as a [0, 1] tensor
    rgb = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    extractor([Image.fromarray(rgb)])
    extractor(torch.as_tensor(rgb).permute(2, 0, 1)[None].float() / 255.0)

    torch.testing.assert_close(seen[1], seen[0], atol=1e-5, rtol=0)
