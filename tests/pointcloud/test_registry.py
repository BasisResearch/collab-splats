import pytest

from collab_splats.pointcloud import get_creator
from collab_splats.pointcloud.base import BasePointcloudCreator
from collab_splats.pointcloud.feedforward import (
    BaseFeedforwardCreator,
    MapAnythingCreator,
    VGGTXCreator,
)


def test_get_creator_mapanything():
    assert get_creator("mapanything") is MapAnythingCreator


def test_get_creator_vggtx():
    assert get_creator("vggtx") is VGGTXCreator


def test_get_creator_loger():
    # NOT guarded on third_party/LoGeR being present: the vendored import is deferred
    # into _load_model, so the registry entry exists in a bare checkout too. Guarding
    # this would skip it in exactly the environment where the wiring can break.
    from collab_splats.pointcloud.feedforward import LoGeRCreator

    assert get_creator("loger") is LoGeRCreator


def test_get_creator_unknown_raises():
    with pytest.raises(ValueError, match="Unknown 'nonexistent'"):
        get_creator("nonexistent")


def test_all_creators_are_instantiable():
    for name in ("mapanything", "vggtx"):
        cls = get_creator(name)
        instance = cls()
        assert isinstance(instance, BasePointcloudCreator)


def test_registry_holds_exactly_the_feedforward_backends():
    assert set(BaseFeedforwardCreator._registry) == {"loger", "mapanything", "vggt_omega", "vggtx"}
