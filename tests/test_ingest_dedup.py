"""Ingest wiring: content-hash dedup (re-ingest is a no-op) + store read-back."""

from __future__ import annotations

from pathlib import Path

from course_kb.cli import main
from course_kb.manifest import read_manifest
from course_kb.store import CourseStore

FIXTURES = Path(__file__).parent / "fixtures"


def _open_store(tmp_path: Path, course: str) -> CourseStore:
    course_dir = tmp_path / "course-kb" / "courses" / course
    manifest = read_manifest(course_dir)
    return CourseStore.open_or_create(course_dir, manifest.dims)


def test_reingesting_unchanged_file_adds_no_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    note = tmp_path / "note.txt"
    note.write_text("An AVL tree self-balances after each insertion.\n", encoding="utf-8")

    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    first = _open_store(tmp_path, "C").count()
    assert first >= 1

    capsys.readouterr()  # drop the first ingest's output
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    assert "No new chunks" in capsys.readouterr().out
    assert _open_store(tmp_path, "C").count() == first  # unchanged


def test_changed_file_adds_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    note = tmp_path / "note.txt"

    note.write_text("Version one content about balanced trees.\n", encoding="utf-8")
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    before = _open_store(tmp_path, "C").count()

    note.write_text("Version two is entirely different prose discussing graphs.\n", encoding="utf-8")
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    assert _open_store(tmp_path, "C").count() > before


def test_identical_text_in_different_files_is_kept(tmp_path, monkeypatch):
    # Dedup is scoped per source file, so a slide shared across two files is
    # stored once for each (preserving per-file provenance).
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    shared = "Outline: skip lists, MTF, optimal static ordering.\n"

    a = tmp_path / "module05.txt"
    a.write_text(shared, encoding="utf-8")
    assert main(["ingest", "C", str(a), "--category", "lecture"]) == 0

    b = tmp_path / "module06.txt"
    b.write_text(shared, encoding="utf-8")
    assert main(["ingest", "C", str(b), "--category", "lecture"]) == 0

    store = _open_store(tmp_path, "C")
    assert store.count() == 2  # not deduped across files
    assert {r.source_file for r in store.get_all()} == {"module05.txt", "module06.txt"}


def test_pdf_ingest_and_read_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0

    assert main(["ingest", "C", str(FIXTURES / "slides.pdf"), "--category", "slides"]) == 0
    store = _open_store(tmp_path, "C")
    assert store.count() == 3  # slide deck -> one chunk per page

    records = store.get_all()
    assert len(records) == 3
    r = sorted(records, key=lambda rec: rec.id)[0]
    assert r.id.startswith("C::slides.pdf::p")
    assert r.page in (1, 2, 3)
    assert r.title
    assert r.char_range is not None
    assert r.content_hash
    assert r.vector is not None and len(r.vector) == 64  # dummy embedder dims survive


def test_chunks_command_prints_json_without_storing(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.chdir(tmp_path)
    assert main(["chunks", "C", str(FIXTURES / "slides.pdf")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 3
    assert all({"id", "page", "title", "char_range", "content_hash", "text"} <= set(c) for c in payload)
    # No course was initialized and nothing was written.
    assert not (tmp_path / "course-kb" / "courses" / "C").exists()
