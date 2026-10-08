"""
Tests for the reconstruction quality report plots.
"""

from collab_splats.geometry.viz import plot_photometric_ncc


def test_plot_photometric_ncc_writes_a_png(tmp_path):
    pairs = {"idx1": [0, 0, 1], "idx2": [1, 2, 2], "photometric_ncc": [0.9, 0.5, 0.8], "n_pixels": [64, 64, 64]}
    path = plot_photometric_ncc(pairs, tmp_path / "sub" / "ncc.png")
    assert path == tmp_path / "sub" / "ncc.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_photometric_ncc_skips_missing_or_empty_tables(tmp_path):
    empty = {"idx1": [], "idx2": [], "photometric_ncc": [], "n_pixels": []}
    assert plot_photometric_ncc(None, tmp_path / "a.png") is None
    assert plot_photometric_ncc(empty, tmp_path / "b.png") is None
    assert not list(tmp_path.iterdir())
