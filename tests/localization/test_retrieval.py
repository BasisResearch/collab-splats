import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from collab_splats.localization import (
    BaseRetrievalExtractor,
    DinoSaladExtractor,
    MegaLocExtractor,
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


def test_megaloc_is_registered():
    assert BaseRetrievalExtractor.get("megaloc") is MegaLocExtractor


def test_pe_clip_is_gone():
    with pytest.raises(ValueError, match="Unknown"):
        BaseRetrievalExtractor.get("pe-clip")


@pytest.mark.slow
def test_megaloc_descriptors_are_unit_norm_on_cpu():
    extractor = MegaLocExtractor(device="cpu")
    images = torch.rand(2, 3, 294, 518)

    desc = extractor(images)

    assert desc.shape == (2, 8448)
    assert torch.allclose(desc.norm(dim=-1), torch.ones(2), atol=1e-5)


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
