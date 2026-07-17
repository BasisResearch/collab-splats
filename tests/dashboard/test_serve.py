"""Fast-binding server: factory gating, warm progress, light-import guarantee."""

import subprocess
import sys

import panel as pn

from collab_splats.dashboard.operation_log import OperationLog
from collab_splats.dashboard.serve import ServerState, make_factory, warm


def test_factory_serves_loading_page_until_ready(tmp_path):
    state = ServerState()
    op_log = OperationLog()
    factory = make_factory(tmp_path, state, op_log)
    tmpl = factory()
    # Loading page is a template with no tabs — cheap to build, no heavy imports.
    assert isinstance(tmpl, pn.template.MaterialTemplate)
    assert not state.ready


def test_warm_flips_ready_and_streams_progress():
    state = ServerState()
    op_log = OperationLog()
    # Stand-in modules + finalize keep the test light (no torch import, no Xvfb).
    warm(state, op_log, modules=(("json", "json"), ("math", "math")), finalize=lambda s: None)
    assert state.ready and not state.failed
    joined = "\n".join(op_log.log_lines)
    assert "import json…" in joined
    assert "import json done (" in joined
    assert not op_log.is_running  # finish_op ran


def test_warm_core_import_failure_marks_failed():
    state = ServerState()
    op_log = OperationLog()
    # A missing module whose name ends in dashboard.app is load-bearing -> failed.
    warm(state, op_log, modules=(("no.such.dashboard.app", "core"),), finalize=lambda s: None)
    assert state.failed and not state.ready
    assert op_log.log_lines[-1].startswith("ERROR")


def test_warm_optional_import_failure_tolerated():
    state = ServerState()
    op_log = OperationLog()
    warm(state, op_log, modules=(("no.such.optional.stack", "optional"),), finalize=lambda s: None)
    assert state.ready and not state.failed


def test_light_import_path_stays_light():
    """serve.py + OperationLog must not pull torch — the fast bind depends on it."""
    code = (
        "import sys; import collab_splats.dashboard.serve; "
        "assert 'torch' not in sys.modules, 'serve.py import pulled torch'; "
        "assert 'pyvista' not in sys.modules, 'serve.py import pulled pyvista'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
