"""Shell composition and page-split contracts."""

from types import SimpleNamespace

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


def test_localize_tab_builds_and_logs(tmp_path):
    """First activation builds the page (inline when no doc) and logs the build time."""
    from collab_splats.dashboard.shell import DashboardShell

    op_log = OperationLog()
    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=op_log)
    shell.view()
    event = SimpleNamespace(new=1, old=0)
    shell._on_tab(event)
    assert shell._localize_built
    assert len(shell._localize_holder) == 1
    assert any("localize page built" in line for line in op_log.log_lines)


def test_localize_tab_build_failure_surfaces_error(tmp_path, monkeypatch):
    """A failing localize build renders an error pane into the holder and logs an ERROR line."""
    import collab_splats.dashboard.shell as shell_mod

    def boom(self):
        raise RuntimeError("no GL context")

    monkeypatch.setattr(shell_mod.LocalizePage, "main", boom)

    op_log = OperationLog()
    shell = shell_mod.DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=op_log)
    shell.view()
    shell._on_tab(SimpleNamespace(new=1, old=0))
    assert len(shell._localize_holder) == 1
    assert "failed to build" in shell._localize_holder[0].object
    assert op_log.log_lines[-1].startswith("ERROR")


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


def test_shell_console_lives_outside_tabs(tmp_path):
    """The op-log console is template-level (visible on both tabs), not per-tab content."""
    from collab_splats.dashboard.shell import DashboardShell

    op_log = OperationLog()
    shell = DashboardShell(base_dir=tmp_path, source=_NoopSource(), op_log=op_log)
    tpl = shell.view()
    assert shell._progress is not None
    # Console updates via the shell's own version-gated tick.
    op_log.append_line("ping")
    shell._on_progress_tick()
    assert "ping" in shell._progress.object
    # Neither page's main content carries its own strip anymore (single console).
    splats_main = shell._splats.main()
    assert all("render_html" not in str(type(c)) for c in splats_main)  # sanity: no HTML strip pane
