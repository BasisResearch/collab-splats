"""Tests for FeatureSplattingDataManagerConfig."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch


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


def test_main_features_accepts_talk2dino():
    """talk2dino is a valid main_features value."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(main_features="talk2dino")
    assert config.main_features == "talk2dino"


def test_regularization_features_can_be_none():
    """regularization_features=None is valid (used with talk2dino, no DINOv2 needed)."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManagerConfig
    config = FeatureSplattingDataManagerConfig(
        main_features="talk2dino",
        regularization_features=None,
    )
    assert config.regularization_features is None



@pytest.mark.parametrize("main_features,expect_seg", [
    ("samclip", True),
    ("maskclip", False),
    ("talk2dino", False),
])
def test_segmentation_gate(main_features, expect_seg):
    """Segmentation is instantiated only when main_features='samclip'."""
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
        patch("collab_splats.nerfstudio.datamanagers.features.MobileSAMSegmentation") as mock_seg_cls,
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


# ── Cache stem helper ─────────────────────────────────────────────────────────

def test_cache_filenames_returns_sorted_names():
    from collab_splats.nerfstudio.datamanagers.features import _cache_filenames
    paths = [
        "/data/preproc/images_2/frame_002.jpg",
        "/data/preproc/images_2/frame_001.jpg",
    ]
    assert _cache_filenames(paths) == ["frame_001.jpg", "frame_002.jpg"]


def test_cache_filenames_strips_downscale_folder():
    from collab_splats.nerfstudio.datamanagers.features import _cache_filenames
    training_paths = ["/data/preproc/images_2/frame_001.jpg"]
    inference_paths = ["/data/preproc/images/frame_001.jpg"]
    assert _cache_filenames(training_paths) == _cache_filenames(inference_paths)


def test_cache_filenames_strips_absolute_prefix():
    from collab_splats.nerfstudio.datamanagers.features import _cache_filenames
    abs_paths = ["/machine_a/data/images/frame_001.jpg"]
    rel_paths = ["../../data/images/frame_001.jpg"]
    assert _cache_filenames(abs_paths) == _cache_filenames(rel_paths)


# ── setup() cache hit/miss ────────────────────────────────────────────────────

def _make_setup_manager(tmp_path, main_features="samclip", data_path=None):
    """Build a minimal FeatureSplattingDataManager with mocked datasets."""
    from collab_splats.nerfstudio.datamanagers.features import (
        FeatureSplattingDataManager,
        FeatureSplattingDataManagerConfig,
    )
    config = FeatureSplattingDataManagerConfig()
    config.main_features = main_features
    config.enable_cache = True
    config.dataparser = MagicMock()
    config.dataparser.data = data_path if data_path is not None else tmp_path

    manager = object.__new__(FeatureSplattingDataManager)
    manager.config = config
    return manager


def test_setup_cache_hit_same_stems(tmp_path):
    """Cache is returned when stems match, even if full paths differ (downscale folder)."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager

    cached_filenames = [tmp_path / "images_2" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    manager = _make_setup_manager(tmp_path)
    # Inference-mode paths have no downscale folder
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_001.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    result = manager.setup()

    # torch.load deserializes a new object, so use value equality not identity
    assert set(result.keys()) == set(cached_features.keys())
    assert torch.equal(result["samclip"], cached_features["samclip"])


def test_setup_cache_miss_different_stems(tmp_path):
    """Cache is invalidated when the image set has changed."""
    from collab_splats.nerfstudio.datamanagers.features import FeatureSplattingDataManager

    cached_filenames = [tmp_path / "images" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    manager = _make_setup_manager(tmp_path)
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_999.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    extracted = {"samclip": torch.ones(1, 8, 8, 8)}
    with patch.object(manager, "extract_features", return_value=extracted) as mock_extract:
        result = manager.setup()

    assert result is extracted
    mock_extract.assert_called_once()


def test_setup_cache_path_resolved(tmp_path, monkeypatch):
    """cache_dir resolves to absolute — relative data path still finds the cache."""
    # Write the cache under tmp_path
    cached_filenames = [tmp_path / "images" / "frame_001.jpg"]
    cached_features = {"samclip": torch.zeros(1, 8, 8, 8)}
    cache_path = tmp_path / "feature-splatting_samclip-features.pt"
    torch.save({"image_filenames": cached_filenames, "features_dict": cached_features}, cache_path)

    # Change CWD to tmp_path.parent so that tmp_path.name is a valid relative path
    monkeypatch.chdir(tmp_path.parent)
    relative_data = Path(tmp_path.name)  # resolves to tmp_path from new CWD

    manager = _make_setup_manager(tmp_path, data_path=relative_data)
    manager.train_dataset = MagicMock(image_filenames=[tmp_path / "images" / "frame_001.jpg"])
    manager.eval_dataset = MagicMock(image_filenames=[])

    result = manager.setup()

    # torch.load deserializes a new object, so use value equality not identity
    assert set(result.keys()) == set(cached_features.keys())
    assert torch.equal(result["samclip"], cached_features["samclip"])
