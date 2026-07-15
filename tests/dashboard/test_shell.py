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
