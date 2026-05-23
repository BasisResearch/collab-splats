import pytest
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.sfm import ColmapCreator, HlocCreator
from collab_splats.pointcloud.feedforward import MapAnythingCreator, VGGTXCreator
from collab_splats.pointcloud.base import BasePointcloudCreator


def test_get_creator_colmap():
    assert get_creator("colmap") is ColmapCreator


def test_get_creator_hloc():
    assert get_creator("hloc") is HlocCreator


def test_get_creator_mapanything():
    assert get_creator("mapanything") is MapAnythingCreator


def test_get_creator_vggtx():
    assert get_creator("vggtx") is VGGTXCreator


def test_get_creator_unknown_raises():
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator("nonexistent")


def test_all_creators_are_instantiable():
    for name in ("colmap", "hloc", "mapanything", "vggtx"):
        cls = get_creator(name)
        instance = cls()
        assert isinstance(instance, BasePointcloudCreator)


def test_old_keys_removed():
    with pytest.raises(KeyError):
        get_creator("sfm")
    with pytest.raises(KeyError):
        get_creator("feedforward")
