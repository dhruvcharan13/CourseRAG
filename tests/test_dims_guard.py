"""Switching embedding model on an existing course fails loudly and changes nothing.

A course's vector width is fixed when its table is created, and vectors from different
models are not comparable even at the same width. Both cases must be refused before
anything is parsed, embedded, or written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from courserag.cli import main
from courserag.manifest import read_manifest
from courserag.store import CourseStore


class _FakeEmbedder:
    """Stands in for a differently-configured embedder. Embedding is unreachable."""

    def __init__(self, model_id: str, dims: int) -> None:
        self.model_id = model_id
        self.dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("embed must not be reached when the embedder mismatches")


def _course_dir(tmp_path: Path, course: str = "C") -> Path:
    return tmp_path / "course-kb" / "courses" / course


def _ingested_course(tmp_path, monkeypatch) -> Path:
    """A dummy-64 course with one file already ingested."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    note = tmp_path / "note.txt"
    note.write_text("An AVL tree self-balances after each insertion.\n", encoding="utf-8")
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    return _course_dir(tmp_path)


# --------------------------------------------------------------------------- #
# Store-level guard
# --------------------------------------------------------------------------- #


def test_reopening_a_table_at_a_different_width_raises(tmp_path):
    course_dir = tmp_path / "C"
    CourseStore.open_or_create(course_dir, 64)

    with pytest.raises(ValueError, match=r"stores 64-dim vectors.*384-dim"):
        CourseStore.open_or_create(course_dir, 384)


def test_reopening_a_table_at_the_same_width_is_fine(tmp_path):
    course_dir = tmp_path / "C"
    CourseStore.open_or_create(course_dir, 64)
    assert CourseStore.open_or_create(course_dir, 64).dims == 64


# --------------------------------------------------------------------------- #
# CLI-level guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model_id,dims",
    [
        ("sentence-transformers/all-MiniLM-L6-v2", 384),  # different width
        ("some-other-model-64", 64),  # same width, different model
    ],
)
def test_ingest_refuses_a_different_embedder_and_writes_nothing(
    tmp_path, monkeypatch, capsys, dummy_config, model_id, dims
):
    course_dir = _ingested_course(tmp_path, monkeypatch)
    before_manifest = json.loads((course_dir / "manifest.json").read_text(encoding="utf-8"))
    before_rows = CourseStore.open_or_create(course_dir, 64).count()
    capsys.readouterr()

    monkeypatch.setattr(
        "courserag.ingest.get_embedder", lambda name, cfg: _FakeEmbedder(model_id, dims)
    )
    other = tmp_path / "other.txt"
    other.write_text("A red-black tree is a balanced binary search tree.\n", encoding="utf-8")

    assert main(["ingest", "C", str(other), "--category", "notes"]) == 1

    err = capsys.readouterr().err
    assert "embedder mismatch" in err
    assert "dummy-hash-64" in err and model_id in err
    assert "rm -rf" in err and "kb init-course C" in err
    assert "note.txt" in err  # the files that would need re-ingesting

    # Nothing changed: no rows, no manifest edits, no half-written table.
    assert CourseStore.open_or_create(course_dir, 64).count() == before_rows
    after_manifest = json.loads((course_dir / "manifest.json").read_text(encoding="utf-8"))
    assert after_manifest == before_manifest
    assert "other.txt" not in after_manifest["files"]


def test_matching_embedder_still_ingests(tmp_path, monkeypatch, dummy_config):
    # The guard compares model id and width, so the normal path is unaffected.
    course_dir = _ingested_course(tmp_path, monkeypatch)
    manifest = read_manifest(course_dir)
    assert manifest.embedding_model == "dummy-hash-64"
    assert manifest.dims == 64
    assert CourseStore.open_or_create(course_dir, manifest.dims).count() >= 1
