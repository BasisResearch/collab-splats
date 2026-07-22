import pytest
import torch

from collab_splats.localization import BaseRetrievalExtractor, DinoSaladExtractor


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
    from collab_splats.localization import BaseRetrievalExtractor, PECLIPExtractor

    cls = BaseRetrievalExtractor.get("pe-clip")
    assert cls is PECLIPExtractor


def test_pe_clip_forward_shape_and_norm():
    """forward() returns (1, 1024) unit-norm tensor without loading real weights."""
    from unittest.mock import MagicMock, patch

    import torch
    from PIL import Image

    from collab_splats.localization import PECLIPExtractor

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
    from unittest.mock import MagicMock, patch

    import torch

    from collab_splats.localization import PECLIPExtractor

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
