# tests/dashboard/test_localize.py
def test_appstate_has_localize_fields():
    from collab_splats.dashboard.state import AppState
    s = AppState()
    assert s.localize_method == ""
    assert s.localize_extractor == "DISK+LightGlue"

def test_appstate_localize_method_is_watchable():
    from collab_splats.dashboard.state import AppState
    s = AppState()
    seen = []
    s.param.watch(lambda e: seen.append(e.new), ["localize_method"])
    s.localize_method = "vggtx"
    assert seen == ["vggtx"]
