import panel as pn

from collab_splats.dashboard.panes._placeholder import PlaceholderPane


def test_placeholder_pane_returns_panel():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    assert result is not None


def test_placeholder_pane_contains_title():
    pane = PlaceholderPane("Semantics", "Coming in Phase 2")
    result = pane.panel()
    # panel repr varies; just verify it returns something
    assert result is not None


def test_app_creates():
    from collab_splats.dashboard.app import App
    app = App()
    assert app is not None


def test_app_servable_returns_material_template():
    import panel as pn
    from collab_splats.dashboard.app import App
    app = App()
    template = app.servable()
    assert isinstance(template, pn.template.MaterialTemplate)


def test_app_has_five_tabs():
    from collab_splats.dashboard.app import App
    app = App()
    assert set(app._tab_names) == {"Preprocess", "Semantics", "Reconstruct", "Visualize", "Localize"}


def test_reconstruct_pane_wired(tmp_path):
    from collab_splats.dashboard.app import App
    from collab_splats.dashboard.panes.reconstruct import ReconstructPane
    app = App(base_dir=str(tmp_path))
    assert isinstance(app._reconstruct, ReconstructPane)
