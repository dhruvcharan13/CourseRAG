"""Ingest wiring: content-hash dedup (re-ingest is a no-op) + store read-back."""

from __future__ import annotations

import re

import pytest
from pathlib import Path

from course_kb.cli import main
from course_kb.manifest import read_manifest
from course_kb.store import CourseStore

FIXTURES = Path(__file__).parent / "fixtures"


def _open_store(tmp_path: Path, course: str) -> CourseStore:
    course_dir = tmp_path / "course-kb" / "courses" / course
    manifest = read_manifest(course_dir)
    return CourseStore.open_or_create(course_dir, manifest.dims)


def test_reingesting_unchanged_file_adds_no_rows(tmp_path, monkeypatch, capsys, dummy_config):
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


def test_changed_file_adds_rows(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    note = tmp_path / "note.txt"

    note.write_text("Version one content about balanced trees.\n", encoding="utf-8")
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    before = _open_store(tmp_path, "C").count()

    note.write_text("Version two is entirely different prose discussing graphs.\n", encoding="utf-8")
    assert main(["ingest", "C", str(note), "--category", "notes"]) == 0
    assert _open_store(tmp_path, "C").count() > before


def test_identical_text_in_different_files_is_kept(tmp_path, monkeypatch, dummy_config):
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


def test_duplicate_chunks_within_one_file_are_deduped_on_first_ingest(
    tmp_path, monkeypatch, dummy_config
):
    """Two identical chunks in the SAME file, in ONE run, must store once.

    Regression: dedup compared against a snapshot of what was already stored, so a
    document with two identical pages (a slide deck's repeated "Intentionally blank."
    or a bare "UML Diagram" caption) stored both on the first ingest. Re-ingest then
    reported a clean no-op, because by then both were stored — so the index looked
    deduped while holding duplicate vectors competing for the same top-k slots.
    Found on a real 34-document course, where 6 chunks were affected.
    """
    fitz = pytest.importorskip("fitz")
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0

    # A slide deck chunks one-per-page, so two identical pages give two identical
    # chunks in a single run — the shape a text file cannot produce, because prose
    # mode merges short paragraphs into one chunk.
    repeated = "Intentionally blank."
    path = tmp_path / "deck.pdf"
    doc = fitz.open()
    for title in (repeated, "Iterator Pattern", repeated):
        page = doc.new_page()
        page.insert_text((72, 96), title, fontsize=28, fontname="hebo")
    doc.save(path)
    doc.close()

    assert main(["ingest", "C", str(path), "--category", "lecture"]) == 0

    records = _open_store(tmp_path, "C").get_all()
    hashes = [r.content_hash for r in records]
    assert len(hashes) == len(set(hashes)), "first ingest stored duplicate chunks"
    assert sum(1 for r in records if repeated in r.text) == 1


def test_pdf_ingest_and_read_back(tmp_path, monkeypatch, dummy_config):
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


def test_ingesting_a_file_with_no_extractable_text_reports_clearly(
    tmp_path, monkeypatch, capsys, dummy_config
):
    # An image-only scan yields zero chunks. That is not "already up to date" —
    # it needs to be distinguishable from a successful no-op re-ingest.
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0

    assert main(["ingest", "C", str(FIXTURES / "scanned.pdf"), "--category", "slides"]) == 1

    err = capsys.readouterr().err
    assert "no chunks produced" in err
    assert "OCR" in err
    assert "2 page(s) parsed, 2 dropped" in err

    # Nothing recorded: no rows, no file entry, still never indexed.
    manifest = read_manifest(tmp_path / "course-kb" / "courses" / "C")
    assert manifest.files == []
    assert manifest.last_indexed is None
    assert _open_store(tmp_path, "C").count() == 0


def test_chunks_report_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["chunks", "C", str(FIXTURES / "scanned.pdf"), "--report"]) == 0
    out = capsys.readouterr().out
    assert "mode:" in out and "chunk chars:" not in out  # empty file: no length line
    assert "2 dropped" in out  # both near-empty pages dropped
    assert re.search(r"chunks:\s+0", out)


def test_oversize_chunk_warns_and_stores_untruncated(tmp_path, monkeypatch, capsys, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    big = "```\n" + "\n".join(f"row_{i} = compute({i})" for i in range(300)) + "\n```"
    listing = tmp_path / "listing.txt"
    listing.write_text(big, encoding="utf-8")

    assert main(["ingest", "C", str(listing), "--category", "code"]) == 0
    assert "exceed" in capsys.readouterr().err  # oversize warning on stderr

    texts = [r.text for r in _open_store(tmp_path, "C").get_all()]
    assert any(len(t) > 1000 for t in texts)  # the big block stored whole, not truncated
    assert all(t.count("```") == 2 for t in texts if "```" in t)


def test_chunks_command_prints_json_without_storing(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.chdir(tmp_path)
    assert main(["chunks", "C", str(FIXTURES / "slides.pdf")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 3
    assert all({"id", "page", "title", "char_range", "content_hash", "text"} <= set(c) for c in payload)
    # No course was initialized and nothing was written.
    assert not (tmp_path / "course-kb" / "courses" / "C").exists()
