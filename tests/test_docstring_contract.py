"""
The documentation contract for the cleaned-up packages.

- a lint, not a behavior test: it exists because the alternative is a slow drift
  back to paragraphs that describe neither the inputs nor the outputs
- every module, public class and public function carries a bulleted docstring
- every parameter and return is annotated in the SIGNATURE and named (without a
  duplicated type) in the docstring
- multi-line ``#`` comment runs state the problem, then bullet the detail
- extend PACKAGES as the remaining modules are cleaned up
"""

import ast
import pathlib
import re

import pytest

PACKAGES = ("preproc", "semantics", "pointcloud")

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Sections that terminate the bulleted body; `Attributes:` is deliberately NOT one of
# them, so prose trailing a class's attribute block is still caught
SECTION_RE = re.compile(r"^\s*(Args|Returns|Raises|Yields|Notes?|Example)s?:")

# A comment run led by one of these is a divider, a legend or already bulleted
RUN_SKIP_PREFIXES = ("-", "*", "─", "#", "=")

MIN_RUN = 3


def _sources() -> list[pathlib.Path]:
    """
    Every .py file in the packages the contract covers.

    Returns:
        Paths, sorted, relative to the repo root.
    """
    out = []
    for pkg in PACKAGES:
        out += sorted((ROOT / "collab_splats" / pkg).rglob("*.py"))
    return out


SOURCES = _sources()
SOURCE_IDS = [str(p.relative_to(ROOT)) for p in SOURCES]


def _documented_defs(path: pathlib.Path) -> list[tuple[ast.AST, str]]:
    """
    Public module-level defs and classes, plus the public methods of those classes.

    Args:
        path: source file to parse.

    Returns:
        (node, label) pairs; the label names the def for the assertion message.
    """
    tree = ast.parse(path.read_text())
    out: list[tuple[ast.AST, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            out.append((node, node.name))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out.append((node, node.name))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                    out.append((sub, f"{node.name}.{sub.name}"))
    return out


def _check_body(doc: str, where: str) -> list[str]:
    """
    The summary and bullets-not-prose rules, shared by modules, classes and functions.

    Args:
        doc: the docstring, already dedented by ast.get_docstring.
        where: label used in the failure messages.

    Returns:
        One message per violation; empty when the docstring conforms.
    """
    bad = []
    paragraphs = doc.split("\n\n")
    summary = paragraphs[0]

    # The summary is one line, short enough to read at a glance
    if "\n" in summary:
        bad.append(f"{where}: summary spans multiple lines")
    if len(summary) > 100:
        bad.append(f"{where}: summary is {len(summary)} chars (max 100)")

    # Everything between the summary and the first named section is bullets. Only a
    # paragraph's first line decides: a wrapped bullet's continuation lines are
    # indented plain text and would otherwise read as prose.
    for para in paragraphs[1:]:
        if SECTION_RE.match(para):
            break
        head = next((ln for ln in para.splitlines() if ln.strip()), "")
        if head and not head.lstrip().startswith(("-", "*")) and not head.rstrip().endswith(":"):
            bad.append(f"{where}: prose paragraph, not bullets — {head.strip()[:60]!r}")
            break

    return bad


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_module_docstring_is_a_bulleted_summary(path):
    doc = ast.get_docstring(ast.parse(path.read_text()))
    assert doc, f"{path.name} has no module docstring"
    assert not _check_body(doc, path.name)


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_public_defs_are_documented_and_annotated(path):
    bad: list[str] = []

    for node, label in _documented_defs(path):
        where = f"{path.name}:{node.lineno} {label}"
        doc = ast.get_docstring(node)
        if not doc:
            bad.append(f"{where}: no docstring")
            continue

        bad += _check_body(doc, where)
        if doc.split("\n\n")[0].startswith(label.split(".")[-1]):
            bad.append(f"{where}: summary restates its own name")

        if isinstance(node, ast.ClassDef):
            continue

        # Every parameter is annotated in the signature and named in Args:
        params = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
        if node.args.vararg:
            params.append(node.args.vararg.arg)
        if node.args.kwarg:
            params.append(node.args.kwarg.arg)
        if params and "Args:" not in doc:
            bad.append(f"{where}: takes {params} and documents none of them")
        elif params:
            # Anchored to a line start, so the name inside prose or inside another
            # parameter's description does not count as documentation
            undocumented = [p for p in params if not re.search(rf"^\s*\*{{0,2}}{re.escape(p)}\b", doc, re.M)]
            if undocumented:
                bad.append(f"{where}: does not document {undocumented}")
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.arg not in ("self", "cls") and arg.annotation is None:
                bad.append(f"{where}: parameter '{arg.arg}' is unannotated")

        # A generator documents its stream with Yields:, everything else with Returns:
        is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
        returns_something = node.returns is None or ast.unparse(node.returns) != "None"
        if returns_something and "Returns:" not in doc and not (is_generator and "Yields:" in doc):
            bad.append(f"{where}: returns something and documents nothing")
        if node.returns is None:
            bad.append(f"{where}: return type is unannotated")

    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", SOURCES, ids=SOURCE_IDS)
def test_comment_runs_state_the_problem_then_bullet_it(path):
    lines = path.read_text().splitlines()
    bad = []

    # Walk maximal runs of consecutive `#` lines; a run of MIN_RUN or more is a block
    # comment and must read as a header followed by bullets
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("#"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        run = [lines[k].strip().lstrip("#").strip() for k in range(i, j)]
        if j - i >= MIN_RUN and run[0] and not run[0].startswith(RUN_SKIP_PREFIXES):
            if not run[1].startswith("- "):
                bad.append(f"{path.name}:{i + 1}: {j - i}-line comment is prose — {run[0][:60]!r}")
        i = j

    assert not bad, "\n".join(bad)
