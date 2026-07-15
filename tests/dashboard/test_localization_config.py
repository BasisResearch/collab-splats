"""LocalizationConfig defaults and construction."""
from collab_splats.dashboard.config import LocalizationConfig


def test_defaults():
    cfg = LocalizationConfig()
    assert cfg.extractor == "loma-g"
    assert cfg.append_to_db is True
    assert cfg.top_k_viz == 3
    assert cfg.calibration_path is None


def test_override():
    cfg = LocalizationConfig(extractor="disk", append_to_db=False)
    assert cfg.extractor == "disk"
    assert cfg.append_to_db is False
