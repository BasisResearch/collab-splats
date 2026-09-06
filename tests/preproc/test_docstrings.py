"""
The docstring contract for preproc's public surface.

Every public function documents its inputs and its outputs. This is a lint, not
a behaviour test — it exists because the alternative is a slow drift back to
paragraphs that describe neither.
"""

import inspect
import re

import pytest

from collab_splats import preproc

PUBLIC = [(name, getattr(preproc, name)) for name in preproc.__all__ if inspect.isfunction(getattr(preproc, name))]


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_public_function_documents_its_inputs_and_outputs(name, fn):
    doc = inspect.getdoc(fn)
    assert doc, f"{name} has no docstring"

    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    if params:
        assert "Args:" in doc, f"{name} takes {params} and documents none of them"
        for p in params:
            # Anchored to the start of a line: a bare `f"{p}:" in doc` also matches the
            # name inside prose or inside another parameter's description, so a function
            # could pass this lint while documenting nothing.
            assert re.search(rf"^\s*{re.escape(p)}:", doc, re.M), f"{name} does not document '{p}'"

    if "-> None" not in str(inspect.signature(fn)):
        assert "Returns:" in doc, f"{name} returns something and documents nothing"


@pytest.mark.parametrize("name,fn", PUBLIC, ids=[n for n, _ in PUBLIC])
def test_summary_is_one_line(name, fn):
    doc = inspect.getdoc(fn)
    summary = doc.split("\n\n")[0]

    assert "\n" not in summary, f"{name}'s summary spans multiple lines"
    assert len(summary) <= 100, f"{name}'s summary is {len(summary)} chars"
    assert not summary.startswith(name), f"{name}'s summary restates its own name"
