"""Deleting a document: scoped to one source file, no embedding, manifest stays true.

Deletion is the cheapest operation in the system — a vector is a pure function of chunk
text, so removing rows costs no model load. These tests pin that down along with the
derived-state cleanup that is easy to get wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from course_kb.cli import main
from course_kb.manifest import read_manifest
from course_kb.store import CourseStore

pytestmark = pytest.mark.usefixtures("dummy_config")


def _course_dir(tmp_path: Path, course: str = "C") -> Path:
    return tmp_path / "course-kb" / "courses" / course


def _store(tmp_path: Path, course: str = "C") -> CourseStore:
    course_dir = _course_dir(tmp_path, course)
    return CourseStore.open_or_create(course_dir, read_manifest(course_dir).dims)


def _two_file_course(tmp_path, monkeypatch) -> Path:
    """A course with notes.txt (category 'notes') and slides.txt (category 'lecture')."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0

    notes = tmp_path / "notes.txt"
    notes.write_text("An AVL tree self-balances after each insertion.\n", encoding="utf-8")
    assert main(["ingest", "C", str(notes), "--category", "notes"]) == 0

    slides = tmp_path / "slides.txt"
    slides.write_text("Outline: skip lists, move-to-front, optimal ordering.\n", encoding="utf-8")
    assert main(["ingest", "C", str(slides), "--category", "lecture"]) == 0

    return _course_dir(tmp_path)


def test_delete_removes_only_the_named_files_chunks(tmp_path, monkeypatch):
    _two_file_course(tmp_path, monkeypatch)
    before = _store(tmp_path).count()

    assert main(["delete", "C", "notes.txt"]) == 0

    store = _store(tmp_path)
    assert store.count() < before
    assert store.source_files() == ["slides.txt"]  # the other file is untouched
    assert all(r.source_file == "slides.txt" for r in store.get_all())


def test_delete_recomputes_manifest_files_and_categories(tmp_path, monkeypatch):
    course_dir = _two_file_course(tmp_path, monkeypatch)
    before = read_manifest(course_dir)
    assert sorted(before.categories) == ["lecture", "notes"]

    assert main(["delete", "C", "notes.txt"]) == 0

    after = read_manifest(course_dir)
    assert after.files == ["slides.txt"]
    # 'notes' was used only by the deleted file, so it must go; 'lecture' stays.
    assert after.categories == ["lecture"]
    # A deletion is not an indexing event.
    assert after.last_indexed == before.last_indexed
    assert (after.embedding_model, after.dims) == (before.embedding_model, before.dims)


def test_delete_reports_unknown_file_and_changes_nothing(tmp_path, monkeypatch, capsys):
    course_dir = _two_file_course(tmp_path, monkeypatch)
    before_rows = _store(tmp_path).count()
    before_manifest = json.loads((course_dir / "manifest.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    assert main(["delete", "C", "typo.txt"]) == 1

    err = capsys.readouterr().err
    assert "no chunks from 'typo.txt'" in err
    assert "notes.txt" in err and "slides.txt" in err  # tells you the valid names
    assert _store(tmp_path).count() == before_rows
    assert json.loads((course_dir / "manifest.json").read_text(encoding="utf-8")) == before_manifest


def test_delete_handles_quotes_in_filenames(tmp_path, monkeypatch):
    # The filter is a typed expression, not an SQL string: a quote in the name must
    # match literally rather than malform the predicate.
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    for name in ("Chapter 1's Notes.txt", "a'; DROP TABLE index; --.txt", "plain.txt"):
        path = tmp_path / name
        path.write_text(f"Content of {name}, discussing balanced trees.\n", encoding="utf-8")
        assert main(["ingest", "C", str(path), "--category", "notes"]) == 0

    assert main(["delete", "C", "Chapter 1's Notes.txt"]) == 0
    assert main(["delete", "C", "a'; DROP TABLE index; --.txt"]) == 0

    store = _store(tmp_path)
    assert store.source_files() == ["plain.txt"]
    assert store.count() >= 1


def test_reingesting_after_delete_restores_the_chunks(tmp_path, monkeypatch):
    # Dedup is driven by live rows, so a deleted file is genuinely re-ingestable.
    _two_file_course(tmp_path, monkeypatch)
    full = _store(tmp_path).count()

    assert main(["delete", "C", "notes.txt"]) == 0
    assert main(["ingest", "C", str(tmp_path / "notes.txt"), "--category", "notes"]) == 0

    store = _store(tmp_path)
    assert store.count() == full  # no duplicates, no missing rows
    assert sorted(store.source_files()) == ["notes.txt", "slides.txt"]
    assert sorted(read_manifest(_course_dir(tmp_path)).categories) == ["lecture", "notes"]


def test_delete_needs_no_embedder(tmp_path, monkeypatch):
    """A course must be prunable even when its embedding model cannot be loaded."""
    _two_file_course(tmp_path, monkeypatch)

    def _unavailable(name, cfg):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr("course_kb.cli.get_embedder", _unavailable)

    assert main(["delete", "C", "notes.txt"]) == 0
    assert _store(tmp_path).source_files() == ["slides.txt"]


def test_delete_reclaims_disk_instead_of_growing_it(tmp_path):
    """Compaction alone makes a delete *worse*; old versions must be pruned too.

    Measured on 4k rows: delete + plain optimize() went 5.9MB -> 8.9MB, because the
    merged files land beside a pre-delete version that is retained for 7 days by
    default. This guards against that regression.
    """
    from course_kb.records import ChunkRecord

    course_dir = tmp_path / "sized"
    store = CourseStore.open_or_create(course_dir, 64)
    store.add(
        [
            ChunkRecord(
                id=f"C::{name}::p{i}::c0",
                text=f"chunk {i} of {name}" * 20,
                course="C",
                source_file=name,
                category="notes",
                content_hash=f"h-{name}-{i}",
                added_at="t",
                vector=[0.1] * 64,
            )
            for name in ("a.pdf", "b.pdf")
            for i in range(400)
        ]
    )

    def disk_bytes() -> int:
        return sum(p.stat().st_size for p in course_dir.rglob("*") if p.is_file())

    before = disk_bytes()
    assert store.delete_source("a.pdf") == 400

    assert disk_bytes() < before  # space actually given back
    assert len(store._table.list_versions()) == 1  # pre-delete versions pruned
    assert store.count() == 400
    assert store.source_files() == ["b.pdf"]


def test_delete_on_uninitialized_course(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["delete", "NOPE", "x.txt"]) == 1
    assert "not initialized" in capsys.readouterr().err


def test_delete_clears_a_stale_manifest_entry(tmp_path, monkeypatch, capsys):
    # Self-healing: a file listed in the manifest with no rows can still be cleaned up.
    course_dir = _two_file_course(tmp_path, monkeypatch)
    manifest = read_manifest(course_dir)
    manifest.files.append("ghost.txt")
    from course_kb.manifest import write_manifest

    write_manifest(course_dir, manifest)
    capsys.readouterr()

    assert main(["delete", "C", "ghost.txt"]) == 0
    assert "stale manifest entry" in capsys.readouterr().out
    assert read_manifest(course_dir).files == ["notes.txt", "slides.txt"]
