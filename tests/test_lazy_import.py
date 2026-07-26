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
emb = get_embedder("minilm", Config())
assert emb.dims == 384, emb.dims
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
