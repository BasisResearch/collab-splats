"""Every creator must forward its own mv fields, not a surviving hardcoded literal."""

import importlib

import pytest

# VGGT-SPARK is deliberately absent: it inherits the mv machinery but is not wired into
# the reconstructor's creator_map, so it has no mv fields of its own to check.
CREATORS = [
    ("collab_splats.pointcloud.feedforward.vggtx", "VGGTXCreator"),
    ("collab_splats.pointcloud.feedforward.vggt_omega", "VGGTOmegaCreator"),
    ("collab_splats.pointcloud.feedforward.mapanything", "MapAnythingCreator"),
    ("collab_splats.pointcloud.feedforward.loger", "LoGeRCreator"),
]


def _creator(module_path: str, cls_name: str):
    """Instantiate a creator by dotted module path, skipping optional absent backends."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # optional backend not installed in this environment
        pytest.skip(f"{cls_name} unavailable: {exc}")
    return getattr(module, cls_name)()


@pytest.mark.parametrize("module_path,cls_name", CREATORS)
def test_creator_exposes_min_views_and_rel_thresh(module_path, cls_name):
    """mv_conf_threshold is gone; min_views and mv_conf_rel_thresh are real fields."""
    creator = _creator(module_path, cls_name)
    assert isinstance(creator.min_views, int)
    assert isinstance(creator.mv_conf_rel_thresh, float)
    assert not hasattr(creator, "mv_conf_threshold"), "mv_conf_threshold must be removed"


def test_mapanything_ships_min_views_one():
    """K=1 is exactly the old threshold=0.0, so MapAnything's behaviour is preserved."""
    creator = _creator("collab_splats.pointcloud.feedforward.mapanything", "MapAnythingCreator")
    assert creator.min_views == 1
    assert creator.use_multiview_confidence is True
    assert creator.mv_conf_abs_thresh == 0.02
    assert creator.mv_conf_rel_thresh == 0.02


@pytest.mark.parametrize(
    "module_path,cls_name",
    [c for c in CREATORS if c[1] != "MapAnythingCreator"],
)
def test_non_metric_backends_keep_abs_thresh_zero(module_path, cls_name):
    """abs_thresh must stay 0.0 for non-metric depth or scale invariance breaks."""
    creator = _creator(module_path, cls_name)
    assert creator.mv_conf_abs_thresh == 0.0
