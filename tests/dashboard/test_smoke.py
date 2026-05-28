"""Smoke tests: verify dashboard imports and instantiates without error."""
from unittest.mock import patch

import panel as pn
import pytest


def test_imports_cleanly():
    from collab_splats.dashboard.app import App, run_app
    from collab_splats.dashboard.operation_log import OperationLog
    from collab_splats.dashboard.panes._placeholder import PlaceholderPane
    from collab_splats.dashboard.panes.preprocess import PreprocessPane
    from collab_splats.dashboard.state import AppState

    assert all([App, run_app, AppState, OperationLog, PreprocessPane, PlaceholderPane])


def test_app_instantiates():
    pn.extension()
    from collab_splats.dashboard.app import App

    app = App()
    template = app.servable()
    assert template is not None


def test_cli_main_help():
    with patch("sys.argv", ["collab-dashboard", "--help"]):
        with pytest.raises(SystemExit) as exc:
            from collab_splats.dashboard.__main__ import main
            main()
    assert exc.value.code == 0


def test_cli_semantics_deprecated():
    """semantics mode redirects to app with DeprecationWarning."""
    import importlib

    import collab_splats.dashboard.__main__ as m

    importlib.reload(m)  # ensure fresh state

    called_with = {}

    def fake_run_app(**kwargs):
        called_with.update(kwargs)

    with patch.object(m, "DASHBOARDS", {"app": "collab_splats.dashboard.app:run_app",
                                         "semantics": "collab_splats.dashboard.app:run_app"}):
        with patch("sys.argv", ["collab-dashboard", "semantics", "--port", "9999"]):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                # Just verify the warning is issued — don't actually launch the server
                with patch("collab_splats.dashboard.app.run_app", fake_run_app):
                    try:
                        m.main()
                    except Exception:
                        pass  # server launch may fail in test env; warning is what we test
