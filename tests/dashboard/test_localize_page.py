"""Pure-logic tests for the localize page helpers."""

import numpy as np

from collab_splats.dashboard.localize import (
    SceneCache,
    camera_centers,
    preselect_method,
    subsample_step,
)


def test_camera_centers_inverts_world_to_camera():
    # Camera at world (1, 2, 3), identity rotation: extrinsic t = -R @ C = -C
    ext = np.eye(4, dtype=np.float32)[None]
    ext[0, :3, 3] = [-1.0, -2.0, -3.0]
    centers = camera_centers(ext)
    np.testing.assert_allclose(centers[0], [1.0, 2.0, 3.0], atol=1e-6)


def test_subsample_step_thresholds():
    assert subsample_step(30) == 1
    assert subsample_step(60) == 1
    assert subsample_step(61) == 3
    assert subsample_step(300) == 3


def test_preselect_existing_db_wins():
    options, value = preselect_method(["disk"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "disk"
    assert options == ["disk", "xfeat", "loma", "loma-g"]


def test_preselect_defaults_to_loma_g_when_no_db():
    _, value = preselect_method([], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


def test_preselect_prefers_loma_g_among_multiple_dbs():
    _, value = preselect_method(["disk", "loma-g"], ["disk", "xfeat", "loma", "loma-g"])
    assert value == "loma-g"


def test_scene_cache_roundtrip():
    cache = SceneCache()
    assert cache.get(("s", "v"), "mesh") is None
    cache.put(("s", "v"), "mesh", object())
    assert cache.get(("s", "v"), "mesh") is not None
    cache.clear()
    assert cache.get(("s", "v"), "mesh") is None


def test_scene_cache_drop_kind_prefix():
    cache = SceneCache()
    cache.put(("s", "v"), "mesh", object())
    cache.put(("s", "v"), "localizer:loma-g", object())
    cache.put(("s", "w"), "localizer:disk", object())
    cache.drop_kind("localizer")
    assert cache.get(("s", "v"), "localizer:loma-g") is None
    assert cache.get(("s", "w"), "localizer:disk") is None
    assert cache.get(("s", "v"), "mesh") is not None  # CPU loads survive


def test_scene_cache_evicts_oldest_mesh_beyond_keep():
    cache = SceneCache()
    for i in range(5):  # _KIND_KEEP["mesh"] == 3
        cache.put(("s", f"v{i}"), "mesh", f"m{i}")
    assert cache.get(("s", "v0"), "mesh") is None
    assert cache.get(("s", "v1"), "mesh") is None
    assert cache.get(("s", "v4"), "mesh") == "m4"


def test_scene_cache_unbounded_kinds_untouched():
    cache = SceneCache()
    for i in range(5):
        cache.put(("s", f"v{i}"), "localizer:disk", i)
    assert cache.get(("s", "v0"), "localizer:disk") == 0
