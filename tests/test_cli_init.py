"""End-to-end CLI: init-course creates the tree; ingest wires the pipeline."""

from __future__ import annotations

import json

import pytest

from courserag.cli import main


def test_init_course_creates_expected_paths(tmp_path, monkeypatch, dummy_config):
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


def test_init_course_is_not_clobbered(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS240-W26"]) == 0
    # Re-initializing an existing course is refused, not silently overwritten.
    assert main(["init-course", "CS240-W26"]) == 1


def test_ingest_pipeline_stores_chunks(tmp_path, monkeypatch, dummy_config):
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


def test_unavailable_embedder_creates_nothing(tmp_path, monkeypatch, capsys):
    # An unusable embedder (missing [local] extra, unknown name) must not leave a
    # manifest-less course behind for `kb list` to show.
    monkeypatch.chdir(tmp_path)

    def _unavailable(name, cfg):
        raise ImportError('sentence-transformers is not installed. pip install -e ".[local]"')

    # init-course resolves its own embedder (there is no manifest to match against yet),
    # so the seam here is the CLI's, not courserag.ingest's.
    monkeypatch.setattr("courserag.cli.get_embedder", _unavailable)

    assert main(["init-course", "CS240-W26"]) == 1
    assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "course-kb" / "courses" / "CS240-W26").exists()


def test_search_needs_a_course_and_a_query(tmp_path, monkeypatch, capsys):
    """``search`` is no longer a stub; it takes a course and refuses without one.

    Behaviour lives in tests/test_search_cli.py — this only pins the argument contract.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["search", "anything"])
    assert exc.value.code == 2
