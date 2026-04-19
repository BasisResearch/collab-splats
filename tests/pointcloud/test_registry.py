import pytest
from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.sfm import NerfstudioSfmCreator
from collab_splats.pointcloud.feedforward import MapAnythingCreator
from collab_splats.pointcloud.base import BasePointcloudCreator


def test_get_creator_sfm():
    assert get_creator("sfm") is NerfstudioSfmCreator


def test_get_creator_feedforward():
    assert get_creator("feedforward") is MapAnythingCreator


def test_get_creator_unknown_raises():
    with pytest.raises(KeyError, match="unknown pointcloud backend"):
        get_creator("nonexistent")


def test_creator_is_instantiable():
    cls = get_creator("sfm")
    instance = cls()
    assert isinstance(instance, BasePointcloudCreator)
