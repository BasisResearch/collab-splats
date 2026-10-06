"""GraphMap (ported from VGGT-SLAM) — submap collection + scene access."""

import numpy as np

from collab_splats.geometry.loop_closure.map import GraphMap
from collab_splats.geometry.loop_closure.submap import Submap


def _submap(sid, is_lc=False, k=2):
    return Submap(
        submap_id=sid,
        frames=None,
        poses=np.tile(np.eye(4, dtype=np.float32), (k, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (k, 1, 1)),
        retrieval_vectors=np.zeros((k, 8), dtype=np.float32),
        image_paths=[f"s{sid}_{i}.jpg" for i in range(k)],
        is_lc_submap=is_lc,
        frame_start=sid * k,
    )


def test_add_and_len():
    m = GraphMap()
    assert len(m) == 0
    m.add_submap(_submap(0))
    m.add_submap(_submap(1))
    assert len(m) == 2


def test_ordered_submaps_by_key():
    m = GraphMap()
    for sid in (2, 0, 1):
        m.add_submap(_submap(sid))
    assert [s.submap_id for s in m.ordered_submaps_by_key()] == [0, 1, 2]
