"""The lifted features.zarr must never be mistaken for the extractor's 2D patch cache.

semantics_dir is flat: it holds BOTH {extractor_name}.zarr (2D patch cache, (N, D, H_p, W_p))
and features.zarr (lifted per-point codes, (P, latent)). Path.glob yields in os.scandir order,
so an unfiltered wildcard picks either one nondeterministically.
"""

from pathlib import Path

import numpy as np
import zarr

from collab_splats.dashboard import pipeline, viewer


def _write_both_stores(tmp_path):
    """Write the 2D patch cache and the lifted per-point store into one flat dir."""
    cache = zarr.open(str(tmp_path / "talk2dino.zarr"), mode="w")
    cache["features"] = np.zeros((2, 4, 3, 3), dtype=np.float32)
    lifted = zarr.open(str(tmp_path / "features.zarr"), mode="w")
    lifted["features"] = np.zeros((7, 4), dtype=np.float32)


def _force_features_first(monkeypatch):
    """Pin the adversarial glob order: features.zarr yielded first.

    Path.glob follows os.scandir order, which is filesystem-dependent — on this box it
    happens to yield the extractor store first, so an unfiltered wildcard would pass by
    luck. Pinning the order makes these tests actually discriminate.
    """
    real_glob = Path.glob

    def ordered_glob(self, pattern, *args, **kwargs):
        return iter(sorted(real_glob(self, pattern, *args, **kwargs), key=lambda p: p.name != "features.zarr"))

    monkeypatch.setattr(Path, "glob", ordered_glob)


def test_load_feature_maps_ignores_lifted_features_store(tmp_path, monkeypatch):
    """features.zarr must never be mistaken for the extractor's 2D patch cache."""
    _write_both_stores(tmp_path)
    _force_features_first(monkeypatch)

    maps = pipeline.load_feature_maps(tmp_path)
    # 2 frames of (D=4, H_p=3, W_p=3) — not 7 points of (4,)
    assert len(maps) == 2
    assert tuple(maps[0].shape) == (4, 3, 3)


def test_viewer_lift_routes_through_the_shared_loader(tmp_path, monkeypatch):
    """The viewer's legacy lift shares ONE loader definition, so it shares the filter."""
    from unittest.mock import patch

    _write_both_stores(tmp_path)
    _force_features_first(monkeypatch)

    seen = {}

    def fake_lift(feature_maps, result):
        seen["shapes"] = [tuple(m.shape) for m in feature_maps]
        import torch

        return torch.zeros((7, 4))

    with patch("collab_splats.pointcloud.utils.lift_features", fake_lift):
        viewer.lift_point_features(object(), tmp_path)
    assert seen["shapes"] == [(4, 3, 3), (4, 3, 3)]
