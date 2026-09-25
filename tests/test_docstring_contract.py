"""
The documentation contract for the cleaned-up packages.

- a lint, not a behavior test: it exists because the alternative is a slow drift
  back to paragraphs that describe neither the inputs nor the outputs
- every module, public class and public function carries a bulleted docstring
- every parameter and return is annotated in the SIGNATURE and named (without a
  duplicated type) in the docstring
- multi-line ``#`` comment runs state the problem, then bullet the detail
- extend PACKAGES as the remaining modules are cleaned up
- release checks (017) run on every file and must pass for packages in RELEASED
"""

import ast
import io
import pathlib
import re
import tokenize

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


########################################################################
# Release checks (decision 017) — fixtures prove each check can fail
########################################################################

# Packages that finished their release cleanup; the rest xfail the release checks
RELEASED: frozenset[str] = frozenset({"preproc", "semantics"})

# Module-level numbers that are facts, not tunables
FIXED_FACTS = frozenset({"SCHEMA_VERSION"})

UPPER_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
BANNED_RE = re.compile(r"\bmeasured\b|\bhypothesis\b|\d+(\.\d+)?x faster|ffmpeg pipe|replaces the old", re.I)
SCENE_ID_RE = re.compile(r"\b(GH\d{6}|C\d{4})\b")
MAX_BULLETS = 6
MAX_COMMENT_LINES = 4


def _numeric_constants(src: str) -> list[str]:
    """
    Module-level UPPER_CASE names bound to a bare number, minus FIXED_FACTS.

    Args:
        src: module source.

    Returns:
        "<line>: <NAME> = <value>" per offender.
    """
    out = []
    for node in ast.parse(src).body:
        # Every bound name: chained `A = B = 5` binds two, annotated `A: int = 5` one
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue

        # Flag a bare (optionally negated) int/float; bool is an int subclass, skip it
        num = node.value.operand if isinstance(node.value, ast.UnaryOp) else node.value
        if not isinstance(num, ast.Constant) or not isinstance(num.value, (int, float)) or isinstance(num.value, bool):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and UPPER_RE.match(target.id) and target.id not in FIXED_FACTS:
                out.append(f"{node.lineno}: {target.id} = {ast.unparse(node.value)}")
    return out


def _comments(src: str) -> list[tuple[int, str]]:
    """
    (line, text) of every `#` comment, via tokenize so `#` inside strings is ignored.

    Args:
        src: module source.

    Returns:
        One entry per comment token, text without the leading `#`.
    """
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    return [(t.start[0], t.string.lstrip("#").strip()) for t in toks if t.type == tokenize.COMMENT]


def _docstrings(src: str) -> list[tuple[str, str]]:
    """
    (label, docstring) for the module and every def and class, private ones included.

    Args:
        src: module source.

    Returns:
        One entry per documented node.
    """
    tree = ast.parse(src)
    nodes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return [(getattr(n, "name", "module"), ast.get_docstring(n)) for n in nodes if ast.get_docstring(n)]


def _banned_words(src: str) -> list[str]:
    """
    Lore words and scene ids in docstrings and comments; paths are exempt.

    Args:
        src: module source.

    Returns:
        "docstring <label>: <word>" or "<line>: <word>" per hit.
    """
    texts = [(f"docstring {label}", doc) for label, doc in _docstrings(src)]
    texts += [(str(line), text) for line, text in _comments(src)]
    out = []
    for where, text in texts:
        text = re.sub(r"\S*/\S*", "", text)
        for rx in (BANNED_RE, SCENE_ID_RE):
            out += [f"{where}: {m.group(0)}" for m in rx.finditer(text)]
    return out


def _bullet_overflow(src: str) -> list[str]:
    """
    Docstrings with more than MAX_BULLETS top-level bullets before the first section.

    Args:
        src: module source.

    Returns:
        "<label>: <n> bullets (max 6)" per offender.
    """
    out = []
    for label, doc in _docstrings(src):
        body = []
        for line in doc.splitlines()[1:]:
            if SECTION_RE.match(line):
                break
            body.append(line)
        n = sum(1 for line in body if line.startswith(("- ", "* ")))
        if n > MAX_BULLETS:
            out.append(f"{label}: {n} bullets (max {MAX_BULLETS})")
    return out


def _long_comment_runs(src: str) -> list[str]:
    """
    Runs of consecutive comment lines longer than MAX_COMMENT_LINES; divider lines do not count.

    Args:
        src: module source.

    Returns:
        "<first line>: <n>-line comment run (max 4)" per offender.
    """
    lines = src.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("#"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("#"):
            j += 1
        text = [k for k in range(i, j) if lines[k].strip().strip("#").strip()]
        if len(text) > MAX_COMMENT_LINES:
            out.append(f"{text[0] + 1}: {len(text)}-line comment run (max {MAX_COMMENT_LINES})")
        i = j
    return out


def _silent_fallbacks(src: str) -> list[str]:
    """
    `x or <number>` expressions and `except Exception:` handlers that never re-raise.

    Args:
        src: module source.

    Returns:
        "<line>: <idiom>" per offender.
    """
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            last = node.values[-1]
            if (
                isinstance(last, ast.Constant)
                and isinstance(last.value, (int, float))
                and not isinstance(last.value, bool)
            ):
                out.append(f"{node.lineno}: `or {last.value!r}` fallback")
        if isinstance(node, ast.ExceptHandler):
            broad = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
            )
            if broad and not any(isinstance(n, ast.Raise) for n in ast.walk(node)):
                out.append(f"{node.lineno}: except Exception without re-raise")
    return out


RELEASE_CHECKS = {
    "numeric-constant": _numeric_constants,
    "banned-word": _banned_words,
    "bullet-cap": _bullet_overflow,
    "comment-cap": _long_comment_runs,
    "silent-fallback": _silent_fallbacks,
}


def _release_params() -> list:
    """
    One param per (file, check); files outside RELEASED are known failures.

    Returns:
        pytest params with ids "<file>::<check>".
    """
    params = []
    for path, pid in zip(SOURCES, SOURCE_IDS):
        pkg = path.relative_to(ROOT / "collab_splats").parts[0]
        marks = [] if pkg in RELEASED else [pytest.mark.xfail(strict=False, reason=f"{pkg}: release cleanup pending")]
        for name in RELEASE_CHECKS:
            params.append(pytest.param(path, name, marks=marks, id=f"{pid}::{name}"))
    return params


@pytest.mark.parametrize(("path", "check"), _release_params())
def test_release_rules(path, check):
    bad = RELEASE_CHECKS[check](path.read_text())
    assert not bad, "\n".join(bad)


def test_numeric_constant_check_flags_a_tunable_and_spares_a_fact():
    src = "SCHEMA_VERSION = 2\n_LEVEL = 1\nNEG = -0.5\nIMAGE_EXTS = ('.png',)\nFLAG = True\n"
    assert _numeric_constants(src) == ["2: _LEVEL = 1", "3: NEG = -0.5"]
    assert _numeric_constants("A = B = 5\nC: int = 3\n") == ["1: A = 5", "1: B = 5", "2: C = 3"]


def test_banned_word_check_flags_lore_and_ignores_paths():
    src = (
        '"""\nSummary.\n\n- measured 3x faster on GH010229\n"""\n'
        "# see docs/superpowers/specs/2026-08-20-video-quality-report-measured.md\n"
        "# HYPOTHESIS: the ffmpeg pipe\n"
    )
    hits = _banned_words(src)
    assert hits == [
        "docstring module: measured",
        "docstring module: 3x faster",
        "docstring module: GH010229",
        "7: HYPOTHESIS",
        "7: ffmpeg pipe",
    ]


def test_bullet_cap_flags_a_seventh_bullet():
    body = "\n".join(f"    - b{i}" for i in range(7))
    src = f'def f() -> None:\n    """\n    Summary.\n\n{body}\n    """\n'
    assert _bullet_overflow(src) == ["f: 7 bullets (max 6)"]
    assert _bullet_overflow(src.replace("    - b6\n", "")) == []
    assert _bullet_overflow(src.replace("    - ", "    * ")) == ["f: 7 bullets (max 6)"]


def test_comment_run_cap_ignores_dividers_and_flags_a_fifth_line():
    ok = "#####\n# a\n# - b\n# - c\n# - d\n#####\nx = 1\n"
    bad = "# a\n# - b\n# - c\n# - d\n# - e\nx = 1\n"
    assert _long_comment_runs(ok) == []
    assert _long_comment_runs(bad) == ["1: 5-line comment run (max 4)"]


def test_silent_fallback_check_flags_or_number_and_swallowed_exception():
    src = (
        "fps = info['fps'] or 30.0\n"
        "name = x or 'default'\n"
        "try:\n    pass\nexcept Exception:\n    pass\n"
        "try:\n    pass\nexcept Exception:\n    raise\n"
        "try:\n    pass\nexcept ValueError:\n    pass\n"
    )
    assert _silent_fallbacks(src) == ["1: `or 30.0` fallback", "5: except Exception without re-raise"]
