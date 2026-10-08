"""
Contract gate for the tutorial notebooks.

- shared scene: pages reach the scene only through tutorial_scene
- independence: no notebook may depend on another notebook having run
- API truth: every collab_splats symbol a notebook imports must resolve
- no new code: a notebook demonstrates the package, it does not define helpers
- figures: at most MPL_CAP hand-written matplotlib calls per page
- executed: outputs committed
- readability: prose and line-length caps
"""

import ast
import importlib
import json
import re
from pathlib import Path

import pytest

TUTORIALS = Path(__file__).resolve().parents[2] / "docs" / "source" / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.rglob("*.ipynb"))

# Names that bypass the shared scene: the retired cache, private scratch, stage rewrites
BANNED_TOKENS = {
    "data/outputs": "the retired shared output directory",
    "OUTPUT_DIR": "shared output dir from the old tutorial_config",
    "IMAGES_DIR": "shared keyframe dir from the old tutorial_config",
    "TUTORIAL_CACHE": "cross-notebook scratch cache",
    "sys.path.insert": "import hack; the tutorial imports installed packages only",
    "mkdtemp": "pages share data/tutorial_scene through tutorial_scene()",
    "overwrite=True": "would rewrite a stage other pages read",
    "autoreload": "development magic, not tutorial content",
}

# "run 02_pointcloud/feedforward_methods.ipynb first" and friends
RUN_FIRST = re.compile(r"run\s+\S*\d{2}[_/]\S*\.ipynb", re.IGNORECASE)

FIRST_PARTY = {"collab_splats", "evals"}

# Pages authored against today's API but executed only once their blocker lands on clean/final
GATED: dict[str, str] = {}

# Hand-written matplotlib calls allowed per page; layout around package plots only
MPL_CAP = 10
MPL_CALL = re.compile(r"\bplt\.|\bax\.|\baxes\[")
MPL_OVERRIDES: dict[str, int] = {}

# Prose markers of plan or draft text
BANNED_PROSE = {"§": "use '## 1. Title' headings", "Package gap": "plan language, not tutorial text"}

# Readability caps per page
EM_DASH_CAP = 3
BOLD_CAP = 2
LINE_CAP = 100
BOLD = re.compile(r"\*\*[^*\n]+\*\*")


def _code_sources(nb_path: Path) -> list[str]:
    """
    Source text of every code cell in a notebook.
    """
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown_sources(nb_path: Path) -> list[str]:
    """
    Source text of every markdown cell in a notebook.
    """
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]


def _strip_magics(src: str) -> str:
    """
    Blank out IPython magic and shell lines so the cell parses as Python.
    """
    return "\n".join("" if ln.lstrip().startswith(("%", "!", "?")) else ln for ln in src.split("\n"))


def _parsed_cells(nb_path: Path):
    """
    Every code cell that parses as Python, as an AST module.
    """
    for src in _code_sources(nb_path):
        try:
            yield ast.parse(_strip_magics(src))
        except SyntaxError:
            continue


PAGES = [
    "01_preprocessing/preprocessing.ipynb",
    "02_pointcloud/reconstruction.ipynb",
    "02_pointcloud/refinement.ipynb",
    "03_splats/train_splats.ipynb",
    "04_mesh/mesh.ipynb",
    "05_semantics/feature_extraction.ipynb",
    "05_semantics/lifting_and_query.ipynb",
    "05_semantics/ocr_lens.ipynb",
    "05_semantics/segmentation.ipynb",
    "06_localization/localization.ipynb",
]


def _rel(nb: Path) -> str:
    """
    Notebook path relative to the tutorials root, posix.
    """
    return nb.relative_to(TUTORIALS).as_posix()


def test_notebook_set_is_the_ten_pages():
    """
    The set itself is part of the contract; a stray notebook is an ungated page.
    """
    assert sorted(_rel(p) for p in NOTEBOOKS) == PAGES


def test_index_lists_every_page():
    """
    A page missing from the toctree is unreachable in the built docs.
    """
    index = (TUTORIALS / "index.rst").read_text()
    missing = [p for p in PAGES if p.removesuffix(".ipynb") not in index]
    assert not missing, f"index.rst does not list {missing}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_matplotlib_cap(nb: Path):
    """
    Figures come from package plotters; hand-written matplotlib is layout only.
    """
    n = len(MPL_CALL.findall("\n".join(_code_sources(nb))))
    cap = MPL_OVERRIDES.get(_rel(nb), MPL_CAP)
    assert n <= cap, f"{nb.name} has {n} hand-written matplotlib calls (cap {cap}); use a package plotter"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_outputs_present(nb: Path):
    """
    A write-now page is committed executed; a gated page waits for its blocker.
    """
    if _rel(nb) in GATED:
        pytest.skip(f"gated on {GATED[_rel(nb)]}")

    cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
    unexecuted = [i for i, c in enumerate(cells) if c.get("execution_count") is None]
    assert not unexecuted, f"{nb.name}: code cells {unexecuted} were never executed"
    assert any(c.get("outputs") for c in cells), f"{nb.name}: no cell has outputs"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_shared_cache(nb: Path):
    """
    A page may not name the retired cross-notebook cache.
    """
    text = "\n".join(_code_sources(nb))
    hits = [f"{tok} ({why})" for tok, why in BANNED_TOKENS.items() if tok in text]
    assert not hits, f"{nb.name} still references: {'; '.join(hits)}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_dependency_on_another_notebook(nb: Path):
    """
    A page may not tell the reader to go run a different page first.
    """
    match = RUN_FIRST.search("\n".join(_code_sources(nb)))
    assert match is None, f"{nb.name} depends on another notebook: {match.group(0)!r}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_defines_no_helpers(nb: Path):
    """
    Notebooks demonstrate the package; helper code belongs in collab_splats or tutorial.py.
    """
    defined = [
        node.name
        for tree in _parsed_cells(nb)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not defined, f"{nb.name} defines {defined} — call the package instead"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_imports_resolve(nb: Path):
    """
    Every first-party symbol a page imports exists on this branch.
    """
    missing: list[str] = []
    for tree in _parsed_cells(nb):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if not node.module or node.module.split(".")[0] not in FIRST_PARTY:
                    continue
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as exc:
                    missing.append(f"{node.module} ({exc})")
                    continue
                missing += [f"{node.module}.{a.name}" for a in node.names if not hasattr(mod, a.name)]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in FIRST_PARTY:
                        continue
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as exc:
                        missing.append(f"{alias.name} ({exc})")
    assert not missing, f"{nb.name} imports names that do not exist: {missing}"


########################################################################
# Readability
########################################################################


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_draft_prose(nb: Path):
    """
    Section marks and plan language do not belong in a tutorial.
    """
    text = "\n".join(_markdown_sources(nb) + _code_sources(nb))
    hits = [f"{tok} ({why})" for tok, why in BANNED_PROSE.items() if tok in text]
    assert not hits, f"{nb.name}: {'; '.join(hits)}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_em_dash_and_bold_caps(nb: Path):
    """
    Em-dashes and bold phrases are capped per page, prose and code comments both.
    """
    text = "\n".join(_markdown_sources(nb) + _code_sources(nb))
    dashes = text.count("—")
    bold = len(BOLD.findall("\n".join(_markdown_sources(nb))))
    assert dashes <= EM_DASH_CAP, f"{nb.name} has {dashes} em-dashes (cap {EM_DASH_CAP})"
    assert bold <= BOLD_CAP, f"{nb.name} has {bold} bold phrases (cap {BOLD_CAP})"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_code_line_length(nb: Path):
    """
    Code lines stay readable in the rendered docs.
    """
    lines = "\n".join(_code_sources(nb)).split("\n")
    long = [ln for ln in lines if len(ln) > LINE_CAP]
    assert not long, f"{nb.name} has {len(long)} code lines over {LINE_CAP} chars: {long[0]!r}"
