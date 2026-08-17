"""LocalizationConfig defaults and construction."""

from collab_splats.dashboard.config import LocalizationConfig


def test_localizationconfig_defaults():
    cfg = LocalizationConfig()
    assert cfg.matcher == "loma"
    assert cfg.append_to_db is True
    assert cfg.top_k_viz == 3
    assert cfg.calibration_path is None
    assert cfg.max_pairs == 200


def test_localizationconfig_override():
    cfg = LocalizationConfig(matcher="xfeat", append_to_db=False)
    assert cfg.matcher == "xfeat"
    assert cfg.append_to_db is False
