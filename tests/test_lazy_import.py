"""The model stack must not be imported until an embed actually happens.

Importing torch costs seconds, so ``kb --help``, course creation, and the dummy path
all have to stay clear of it. Each check runs in a fresh subprocess and reports which
heavy modules ended up in ``sys.modules``.

``mcp`` and its transport dependencies are on the same list. Nothing in the package
imports :mod:`course_kb.mcp_server`, and that has to stay true — it is the only reason
the ``[mcp]`` extra can pull starlette and uvicorn without the CLI paying for them.

These are only meaningful when the ``[local]`` extra is installed — otherwise the
modules are absent no matter what the code does — so they skip without it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from course_kb.embedding.sentence_transformer import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="vacuous without the [local] extra installed"
)

HEAVY = ("torch", "sentence_transformers", "transformers", "mcp", "starlette", "uvicorn")

_REPORT = f"""
import sys
leaked = sorted(m for m in {HEAVY!r} if m in sys.modules)
print("LEAKED:" + ",".join(leaked))
"""


def _heavy_modules_after(code: str) -> list[str]:
    """Run ``code`` in a fresh interpreter; return the heavy modules it imported."""
    result = subprocess.run(
        [sys.executable, "-c", code + _REPORT],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("LEAKED:"))
    return [m for m in line.removeprefix("LEAKED:").split(",") if m]


def test_help_does_not_import_the_model_stack():
    assert (
        _heavy_modules_after(
            """
from course_kb.cli import main
try:
    main(["--help"])
except SystemExit:
    pass
"""
        )
        == []
    )


def test_constructing_the_local_embedder_does_not_import_the_model_stack():
    # dims comes from a known-widths table, so init-course can size a table without
    # loading a model.
    assert (
        _heavy_modules_after(
            """
from course_kb.config import Config
from course_kb.embedding import get_embedder
emb = get_embedder("local", Config())
assert emb.dims == 384, emb.dims
"""
        )
        == []
    )


def test_search_help_does_not_import_the_model_stack():
    """Reranking added a second model; --help must still cost nothing."""
    assert (
        _heavy_modules_after(
            """
from course_kb.cli import main
try:
    main(["search", "--help"])
except SystemExit:
    pass
"""
        )
        == []
    )


def test_constructing_the_reranker_does_not_import_the_model_stack():
    # Naming a reranker, or having one configured but never asking for it, is free.
    assert (
        _heavy_modules_after(
            """
from course_kb.config import Config
from course_kb.reranking import get_reranker
r = get_reranker("local", Config())
assert r.model_id.startswith("cross-encoder/"), r.model_id
"""
        )
        == []
    )


def test_dense_search_without_rerank_never_loads_the_cross_encoder():
    """The default path must stay exactly as cheap as it was before Phase 4."""
    assert (
        _heavy_modules_after(
            """
import sys, tempfile, pathlib
from course_kb.cli import main
tmp = tempfile.mkdtemp()
import os; os.chdir(tmp)
pathlib.Path("config.toml").write_text('embedder = "dummy"\\nreranker = "dummy"\\n')
pathlib.Path("notes.txt").write_text("Skip lists use coin flips to pick tower height.\\n")
assert main(["init-course", "C"]) == 0
assert main(["ingest", "C", "notes.txt", "--category", "lecture"]) == 0
assert main(["search", "C", "tower height", "--json"]) == 0
"""
        )
        == []
    )


def test_the_cli_never_imports_the_mcp_sdk():
    """A dense search is the CLI's heaviest path; the server's SDK is not part of it."""
    assert (
        _heavy_modules_after(
            """
import pathlib, tempfile, os
from course_kb.cli import main
os.chdir(tempfile.mkdtemp())
pathlib.Path("config.toml").write_text('embedder = "dummy"\\n')
pathlib.Path("notes.txt").write_text("Skip lists use coin flips to pick tower height.\\n")
assert main(["init-course", "C"]) == 0
assert main(["ingest", "C", "notes.txt", "--category", "lecture"]) == 0
assert main(["search", "C", "tower height", "--json"]) == 0
"""
        )
        == []
    )


def test_dummy_path_never_touches_the_model_stack():
    assert (
        _heavy_modules_after(
            """
from course_kb.config import Config
from course_kb.embedding import get_embedder
vectors = get_embedder("dummy", Config()).embed(["some text"])
assert len(vectors[0]) == 64
"""
        )
        == []
    )


def test_kb_show_never_loads_a_model(tmp_path):
    """Reading a document back compares no vectors, so it must cost no model at all.

    This is the whole reason ``cmd_show`` opens the store directly instead of going
    through ``_open_for_search``: that helper constructs an embedder to run the
    dims-match guard, which would pull in torch for a path that never embeds anything.
    """
    assert (
        _heavy_modules_after(
            """
import pathlib, tempfile, os
from course_kb.cli import main
os.chdir(tempfile.mkdtemp())
pathlib.Path("config.toml").write_text('embedder = "dummy"\\n')
pathlib.Path("notes.txt").write_text("Skip lists use coin flips to pick tower height.\\n")
assert main(["init-course", "C"]) == 0
assert main(["ingest", "C", "notes.txt", "--category", "lecture"]) == 0
assert main(["show", "C", "notes.txt"]) == 0
"""
        )
        == []
    )


def test_read_document_never_loads_a_model():
    """Same guarantee through the MCP tool.

    ``mcp`` itself is expected here — importing the server is the point — so this asserts
    on the *model* stack specifically rather than the whole heavy list.
    """
    leaked = _heavy_modules_after(
        """
import pathlib, tempfile, os
from course_kb.cli import main
root = tempfile.mkdtemp()
os.chdir(root)
pathlib.Path("config.toml").write_text('embedder = "dummy"\\n')
pathlib.Path("notes.txt").write_text("Skip lists use coin flips to pick tower height.\\n")
assert main(["init-course", "C"]) == 0
assert main(["ingest", "C", "notes.txt", "--category", "lecture"]) == 0
os.environ["COURSE_KB_ROOT"] = str(pathlib.Path(root) / "course-kb")
from course_kb.mcp_server import read_document
out = read_document("C", "notes.txt")
assert "coin flips" in out, out
"""
    )

    assert [m for m in leaked if m in ("torch", "sentence_transformers", "transformers")] == []
