"""Shell composition and page-split contracts."""

import panel as pn

from collab_splats.dashboard.app import SplatsApp
from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.sources import SessionSource


class _NoopSource(SessionSource):
    def __init__(self):
        self._client = None  # degrade: listings fail soft, nothing remote


def _app(tmp_path):
    return SplatsApp(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())


def test_splats_page_exposes_sidebar_and_main(tmp_path):
    app = _app(tmp_path)
    assert isinstance(app.sidebar(), pn.Column)
    main = app.main()
    assert isinstance(main, pn.Column)


def test_splats_page_view_still_returns_template(tmp_path):
    app = _app(tmp_path)
    tpl = app.view()
    assert isinstance(tpl, pn.template.MaterialTemplate)


def test_shell_builds_tabs_with_two_pages(tmp_path):
    from collab_splats.dashboard.shell import DashboardShell

    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())
    tpl = shell.view()
    assert isinstance(tpl, pn.template.MaterialTemplate)
    assert len(shell._tabs) == 2
    assert [t for t in shell._tabs._names] == ["Splats", "Localize"]


def test_inactive_tab_main_built_lazily(tmp_path, monkeypatch):
    """The Localize page's main() (VTK plotter) is not built until its tab is first shown."""
    import collab_splats.dashboard.shell as shell_mod

    built = {"n": 0}
    real_main = shell_mod.LocalizePage.main

    def counting_main(self):
        built["n"] += 1
        return real_main(self)

    monkeypatch.setattr(shell_mod.LocalizePage, "main", counting_main)

    shell = shell_mod.DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())
    shell.view()
    assert built["n"] == 0  # localize main deferred
    shell._on_tab(type("E", (), {"new": 1, "old": 0})())
    assert built["n"] == 1  # built exactly once on first activation
    shell._on_tab(type("E", (), {"new": 0, "old": 1})())
    shell._on_tab(type("E", (), {"new": 1, "old": 0})())
    assert built["n"] == 1  # NOT rebuilt on subsequent activations


def test_shell_sidebar_swaps_on_tab_change(tmp_path):
    from collab_splats.dashboard.shell import DashboardShell

    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=OperationLog())
    shell.view()
    splats_sidebar = shell._splats.sidebar()
    localize_sidebar = shell._localize.sidebar()
    assert shell._sidebar_holder[0] is splats_sidebar
    shell._tabs.active = 1
    assert shell._sidebar_holder[0] is localize_sidebar
    shell._tabs.active = 0
    assert shell._sidebar_holder[0] is splats_sidebar
