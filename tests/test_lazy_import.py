"""The model stack must not be imported until an embed actually happens.

Importing torch costs seconds, so ``kb --help``, course creation, and the dummy path
all have to stay clear of it. Each check runs in a fresh subprocess and reports which
heavy modules ended up in ``sys.modules``.

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

HEAVY = ("torch", "sentence_transformers", "transformers")

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
