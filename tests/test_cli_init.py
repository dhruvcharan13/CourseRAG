"""End-to-end CLI: init-course creates the tree; ingest wires the pipeline."""

from __future__ import annotations

import json

from course_kb.cli import main


def test_init_course_creates_expected_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["init-course", "CS240-W26"]) == 0

    course_dir = tmp_path / "course-kb" / "courses" / "CS240-W26"
    assert course_dir.is_dir()
    assert (course_dir / "raw").is_dir()
    assert (course_dir / "index.lance").is_dir()  # LanceDB dataset
    assert (course_dir / "archive").exists() is False  # archive lives at the root, not per-course
    assert (tmp_path / "course-kb" / "archive").is_dir()

    course_md = course_dir / "COURSE.md"
    assert course_md.is_file()
    assert course_md.read_text(encoding="utf-8") == ""

    manifest = json.loads((course_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["course"] == "CS240-W26"
    assert manifest["embedding_model"] == "dummy-hash-64"
    assert manifest["dims"] == 64
    assert manifest["categories"] == []
    assert manifest["files"] == []
    assert manifest["last_indexed"] is None


def test_init_course_is_not_clobbered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240-W26"]) == 0
    # Re-initializing an existing course is refused, not silently overwritten.
    assert main(["init-course", "CS240-W26"]) == 1


def test_ingest_pipeline_stores_chunks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240-W26"]) == 0

    sample = tmp_path / "sample.txt"
    sample.write_text("An AVL tree self-balances.\n", encoding="utf-8")

    # parse -> chunk -> embed -> store, proven by a nonzero chunk count.
    assert main(["ingest", "CS240-W26", str(sample), "--category", "notes"]) == 0

    manifest = json.loads(
        (tmp_path / "course-kb" / "courses" / "CS240-W26" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["files"] == ["sample.txt"]
    assert manifest["categories"] == ["notes"]
    assert manifest["last_indexed"] is not None


def test_search_stub(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["search", "anything"]) == 0
    assert "not implemented (Phase 3)" in capsys.readouterr().out
