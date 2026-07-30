"""resolve_semantics_dir resolves the FLAT dashboard layout — the only one the dashboard reads.

The dashboard writes and reads flat: ``{scene}/semantics/<extractor>_lifted.zarr``, matching the flat
``{scene}/feedforward.zarr`` its loader gates on. The Reconstructor (the published output
contract) writes everything two levels deeper — ``{scene}/{backend}/...`` — and the dashboard
cannot browse such a scene at all: it fails on the pointcloud long before semantics. Resolving
a backend-keyed semantics dir therefore had no reachable caller, and no writer produces the
hybrid tree (flat pointcloud + nested semantics) it would have served.
"""

from pathlib import Path

import numpy as np
import zarr

from collab_splats.dashboard.pipeline import resolve_semantics_dir


def _write_features(sem_dir: Path) -> Path:
    """Create a semantics dir holding a minimal lifted per-point store."""
    sem_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(sem_dir / "talk2dino_lifted.zarr"), mode="w")
    store["features"] = np.zeros((3, 4), dtype=np.float32)
    return sem_dir


def test_resolve_finds_the_flat_dashboard_layout(tmp_path):
    """A dashboard-produced scene resolves to its own flat semantics dir."""
    flat = _write_features(tmp_path / "semantics")
    assert resolve_semantics_dir(tmp_path) == flat


def test_resolve_returns_none_when_no_semantics_dir_exists(tmp_path):
    """No flat dir -> None; every caller tolerates a missing semantics dir."""
    (tmp_path / "feedforward.zarr").mkdir(parents=True)
    assert resolve_semantics_dir(tmp_path) is None


def test_resolve_keeps_flat_dir_holding_only_the_2d_cache(tmp_path):
    """A flat dir with the 2D patch cache but no lifted store must still resolve.

    That is exactly the legacy scene the viewer's on-demand lift exists for — returning
    None here would strand it with no semantics forever.
    """
    sem = tmp_path / "semantics"
    sem.mkdir(parents=True)
    zarr.open(str(sem / "talk2dino.zarr"), mode="w")["features"] = np.zeros((2, 4, 3, 3), dtype=np.float32)
    assert resolve_semantics_dir(tmp_path) == sem


def test_resolve_ignores_a_backend_keyed_tree(tmp_path):
    """A published (Reconstructor) scene is not browsable by the dashboard, so do not pretend.

    Reporting a semantics dir for a scene whose flat feedforward.zarr does not exist would only
    put the loader one step further into a failure it cannot recover from.
    """
    _write_features(tmp_path / "vggt_omega" / "semantics")
    assert resolve_semantics_dir(tmp_path) is None
