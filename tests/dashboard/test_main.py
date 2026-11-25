"""collab-dashboard CLI: argument contract."""

import sys

import pytest

from collab_splats.dashboard.__main__ import main


def test_main_without_base_dir_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["collab-dashboard"])

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    assert "--base-dir" in capsys.readouterr().err
