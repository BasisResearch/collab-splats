"""Tests for pointcloud density alignment across feedforward backends."""
import dataclasses

from collab_splats.pointcloud.feedforward.base import BaseFeedforwardCreator
from collab_splats.pointcloud.feedforward.vggtx import VGGTXCreator


def test_base_has_max_points_field():
    fields = {f.name: f for f in dataclasses.fields(BaseFeedforwardCreator)}
    assert "max_points" in fields
    assert fields["max_points"].default == 500_000


def test_vggtx_conf_threshold_default():
    fields = {f.name: f for f in dataclasses.fields(VGGTXCreator)}
    assert fields["conf_threshold"].default == 35.0


def test_vggtx_inherits_max_points():
    fields = {f.name: f for f in dataclasses.fields(VGGTXCreator)}
    assert fields["max_points"].default == 500_000


def test_mapanything_inherits_max_points():
    from collab_splats.pointcloud.feedforward.mapanything import MapAnythingCreator
    fields = {f.name: f for f in dataclasses.fields(MapAnythingCreator)}
    assert fields["max_points"].default == 500_000


def test_mapanything_no_collect_pts3d_import():
    import collab_splats.pointcloud.feedforward.mapanything as m
    assert not hasattr(m, "collect_pts3d_from_outputs")
