"""Tests for unified extractor preprocessing utilities and interface."""
import torch
import torch.nn.functional as F
import pytest
from PIL import Image

from collab_splats.semantics.utils import _tokens_to_feature_map


def test_tokens_to_feature_map_shape():
    patch_size = 14
    H, W = 196, 280  # multiples of 14
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D)
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    assert out.shape == (D, ph, pw)


def test_tokens_to_feature_map_l2_normalized():
    patch_size = 14
    H, W = 196, 196
    D = 8
    ph, pw = H // patch_size, W // patch_size
    tokens = torch.randn(ph * pw, D) * 10  # large values
    out = _tokens_to_feature_map(tokens, H, W, patch_size)
    # F.normalize(feat, dim=0) normalizes along channel dim per spatial position
    norms = out.norm(dim=0)  # (H_p, W_p)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tokens_to_feature_map_wrong_count_raises():
    with pytest.raises(AssertionError):
        _tokens_to_feature_map(torch.randn(99, 8), 196, 196, 14)


from unittest.mock import MagicMock, patch
import torchvision.transforms as T


def _make_fake_dino(resize_mode="max_size", image_resolution=224, patch_size=14):
    """Build a DINOFeatureExtractor with a mocked backbone (no weights download)."""
    from collab_splats.semantics.features.dino import DINOFeatureExtractor

    mock_model = MagicMock()
    mock_model.config.patch_size = patch_size

    def fake_forward(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        n_tokens = 1 + ph * pw
        result = MagicMock()
        result.last_hidden_state = torch.randn(B, n_tokens, 8)
        return result

    mock_model.side_effect = fake_forward

    with patch("collab_splats.semantics.features.dino.AutoModel.from_pretrained", return_value=mock_model):
        ext = DINOFeatureExtractor(
            resize_mode=resize_mode,
            image_resolution=image_resolution,
            device="cpu",
        )
    ext.model = mock_model
    return ext


def test_dino_preprocess_returns_tensor():
    ext = _make_fake_dino()
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 3  # (C, H, W)
    assert t.shape[0] == 3


def test_dino_preprocess_patch_aligned():
    patch_size = 14
    ext = _make_fake_dino(patch_size=patch_size)
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % patch_size == 0
    assert W % patch_size == 0


def test_dino_preprocess_imagenet_normalization():
    """White image green channel should be >1.5 under ImageNet stats, ~1.0 under [0.5] stats."""
    ext = _make_fake_dino(image_resolution=224)
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    t = ext.preprocess(img)
    # Green channel: (1.0 - 0.456) / 0.224 ≈ 2.43 under ImageNet; 1.0 under [0.5] norm
    assert t[1].mean().item() > 1.5, "Expected ImageNet normalization (green channel > 1.5 for white img)"


def test_dino_forward_returns_list_of_maps():
    ext = _make_fake_dino(image_resolution=196, patch_size=14)
    imgs = [Image.new("RGB", (196, 196)) for _ in range(2)]

    fake_features = MagicMock()
    # last_hidden_state (B, 1+N, D): 1 CLS + 14*14=196 patches
    fake_features.last_hidden_state = torch.randn(2, 197, 8)
    ext.model.side_effect = None
    ext.model.return_value = fake_features

    out = ext.forward(imgs)
    assert len(out) == 2
    assert out[0].shape == (8, 14, 14)  # (D, H_p, W_p)


def _make_fake_maskclip(resize_mode="max_size", image_resolution=336, patch_size=14):
    """Build MaskCLIPExtractor with mocked model (no weights download)."""
    from unittest.mock import MagicMock, patch
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor

    mock_model = MagicMock()
    mock_model.visual.patch_size = patch_size

    def fake_patch_encodings(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        return torch.randn(B, ph * pw, 8)

    mock_model.get_patch_encodings.side_effect = fake_patch_encodings

    mock_maskclip_onnx = MagicMock()
    mock_maskclip_onnx.clip.load.return_value = (mock_model, MagicMock())

    with patch.dict("sys.modules", {"maskclip_onnx": mock_maskclip_onnx}):
        with patch("collab_splats.semantics.features.maskclip.maskclip_onnx", mock_maskclip_onnx, create=True):
            ext = MaskCLIPExtractor(
                resize_mode=resize_mode,
                image_resolution=image_resolution,
                device="cpu",
            )
    ext.model = mock_model
    ext._maskclip_onnx = mock_maskclip_onnx
    return ext


def test_maskclip_preprocess_clip_normalization():
    """White image green channel should be ~(1.0-0.4578275)/0.26130258 ≈ 2.07 under CLIP stats."""
    ext = _make_fake_maskclip(image_resolution=336)
    img = Image.new("RGB", (336, 336), color=(255, 255, 255))
    t = ext.preprocess(img)
    # CLIP green: (1.0 - 0.4578275) / 0.26130258 ≈ 2.075
    # ImageNet green: (1.0 - 0.456) / 0.224 ≈ 2.429
    assert 1.9 < t[1].mean().item() < 2.2, (
        f"Expected CLIP normalization (~2.07), got {t[1].mean().item():.3f}"
    )


def test_maskclip_preprocess_patch_aligned():
    ext = _make_fake_maskclip(patch_size=14)
    img = Image.new("RGB", (400, 300))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % 14 == 0 and W % 14 == 0


def test_maskclip_forward_returns_list_of_maps():
    ext = _make_fake_maskclip(image_resolution=196, patch_size=14)
    imgs = [Image.new("RGB", (196, 196)) for _ in range(3)]
    out = ext.forward(imgs)
    assert len(out) == 3
    assert out[0].shape == (8, 14, 14)


def _make_fake_talk2dino(resize_mode="max_size", image_resolution=512, patch_size=14):
    """Build Talk2DinoExtractor with mocked backbone (no weights download)."""
    import torchvision.transforms as T
    from unittest.mock import MagicMock, patch
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    mock_backbone = MagicMock()

    def fake_forward_features(batch):
        B, _, H, W = batch.shape
        ph, pw = H // patch_size, W // patch_size
        # Returns (B, 5 + H_p*W_p, D): 5 register/CLS tokens + patch tokens
        return torch.randn(B, 5 + ph * pw, 8)

    mock_backbone.forward_features.side_effect = fake_forward_features

    mock_model = MagicMock()
    mock_model.model = mock_backbone
    mock_backbone.patch_embed.proj.stride = (patch_size, patch_size)

    # Provide a real Normalize as image_transforms.transforms[-1]
    mock_model.image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    with patch("collab_splats.semantics.features.talk2dino.AutoModel.from_pretrained", return_value=mock_model):
        ext = Talk2DinoExtractor(
            resize_mode=resize_mode,
            image_resolution=image_resolution,
            device="cpu",
        )
    ext._model = mock_model
    return ext


def test_talk2dino_preprocess_returns_tensor():
    ext = _make_fake_talk2dino()
    img = Image.new("RGB", (320, 240))
    t = ext.preprocess(img)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 3


def test_talk2dino_preprocess_patch_aligned():
    ext = _make_fake_talk2dino(patch_size=14)
    img = Image.new("RGB", (400, 300))
    t = ext.preprocess(img)
    _, H, W = t.shape
    assert H % 14 == 0 and W % 14 == 0


def test_talk2dino_forward_single_path():
    """forward() must work for both resize modes without _forward_max_size/_forward_square."""
    for mode in ("max_size", "square"):
        ext = _make_fake_talk2dino(resize_mode=mode, image_resolution=196, patch_size=14)
        imgs = [Image.new("RGB", (196, 196)) for _ in range(2)]
        out = ext.forward(imgs)
        assert len(out) == 2
        assert out[0].shape[1] == out[0].shape[2]  # square feature map for square input


def test_talk2dino_no_dual_forward_methods():
    """_forward_max_size and _forward_square must not exist after refactor."""
    from collab_splats.semantics.features import talk2dino as t2d_mod
    src = open(t2d_mod.__file__).read()
    assert "_forward_max_size" not in src
    assert "_forward_square" not in src


########################################################################
# Shared preprocess
########################################################################


def test_preprocess_is_defined_once_on_the_base():
    """All three extractors share BaseFeatureExtractor.preprocess — no per-backend copies."""
    from collab_splats.semantics.features.base import BaseFeatureExtractor
    from collab_splats.semantics.features.dino import DINOFeatureExtractor
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    for cls in (DINOFeatureExtractor, MaskCLIPExtractor, Talk2DinoExtractor):
        assert "preprocess" not in vars(cls), f"{cls.__name__} still overrides preprocess"
        assert cls.preprocess is BaseFeatureExtractor.preprocess


def test_each_extractor_keeps_its_own_default_resolution():
    """Hoisting the plumbing must not flatten the per-backbone defaults."""
    import inspect

    from collab_splats.semantics.features.dino import DINOFeatureExtractor
    from collab_splats.semantics.features.maskclip import MaskCLIPExtractor
    from collab_splats.semantics.features.talk2dino import Talk2DinoExtractor

    defaults = {
        DINOFeatureExtractor: 800,
        MaskCLIPExtractor: 1024,
        Talk2DinoExtractor: 512,
    }
    for cls, expected in defaults.items():
        params = inspect.signature(cls.__init__).parameters
        assert params["image_resolution"].default == expected
        assert params["resize_mode"].default == "max_size"
        assert "kwargs" not in params, f"{cls.__name__} still takes **kwargs"


def test_svd_components_still_reaches_the_base():
    """insid3 constructs DINOFeatureExtractor(svd_components=...) — the path must survive."""
    import inspect

    from collab_splats.semantics.features.dino import DINOFeatureExtractor

    assert inspect.signature(DINOFeatureExtractor.__init__).parameters["svd_components"].default == 500
