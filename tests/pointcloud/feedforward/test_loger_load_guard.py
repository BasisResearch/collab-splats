"""LoGeR reaches the vendored tree at _load_model time, not import time — the guard
that keeps the registry importable in a bare checkout with no third_party/LoGeR."""
import ast
import inspect

from collab_splats.pointcloud.feedforward import loger as loger_mod
from collab_splats.pointcloud.feedforward.loger import _LOGER_ROOT


def test_module_does_not_import_the_vendored_tree_at_module_level():
    # loger.py must reach `loger.models.pi3` only from inside _load_model, behind its
    # sys.path insert. A module-level import would break `import collab_splats.pointcloud`
    # outright in a bare checkout rather than just omitting the backend — and the registry
    # entry added alongside this test is what makes that failure mode reachable.
    #
    # This is a source check rather than the more direct `"loger" not in sys.modules`,
    # because that assertion is BOTH vacuous and order-dependent here. Measured:
    #   * "pi3" never appears in sys.modules at all — not even after a real
    #     `from loger.models.pi3 import Pi3`, which registers `loger`, `loger.models`,
    #     `loger.models.pi3` and 24 more, but no bare `pi3`. Nothing in the vendored tree
    #     imports `pi3` unqualified (all its imports are relative or `loger.`-prefixed),
    #     so asserting on that key can never fail and would pin nothing.
    #   * `loger` and `loger.*` DO appear — but test_target_size_matches_the_vendored_loader
    #     (in tests/pointcloud/test_loger_creator.py) imports the vendored loader for
    #     real and drops only its sys.path entry, never the sys.modules keys. Measured: it
    #     leaks `loger`, `loger.utils`, `loger.utils.basic` for the rest of the session. So
    #     a sys.modules assertion passes or fails on collection order, and pytest-randomly
    #     is installed. The AST carries the same property and no shared state.
    tree = ast.parse(inspect.getsource(loger_mod))

    # Module-level imports only — the nested one inside _load_model is the point of the
    # design, so walking the whole tree would flag the correct implementation.
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "loger"]
        # level == 0 excludes the relative `from .base import ...` / `from ...geometry`
        # imports, which resolve inside collab_splats and share only the bare name.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            offenders += [node.module] if node.module.split(".")[0] == "loger" else []

    assert not offenders, f"vendored tree imported at module level: {offenders}"


def test_root_points_at_third_party():
    # _LOGER_ROOT is built by walking parents[3] up from this file, so moving loger.py
    # between package levels silently retargets it at a directory that does not exist.
    assert _LOGER_ROOT.name == "LoGeR"
    assert _LOGER_ROOT.parent.name == "third_party"
