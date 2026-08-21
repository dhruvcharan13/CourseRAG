"""Exporting a course and importing it somewhere else.

The archive ships sources, not vectors, on the claim that an importer running the same
model rebuilds an identical index. The round-trip test asserts exactly that — same chunk
ids, same text, same provenance, same vectors — because if it were only approximately
true the feature would be quietly worthless: two people would get different answers to
the same question and have no way to notice.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from courserag.cli import main
from courserag.config import Config, load_config
from courserag.ingest import open_course
from courserag.sync import apply_sync, source_root
from courserag.transfer import (
    FORMAT,
    METADATA_NAME,
    TransferError,
    export_course,
    import_course,
    read_metadata,
)


def write(cfg: Config, course: str, rel: str, text: str) -> None:
    path = source_root(cfg, course) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _absolute(cfg: Config, root: Path) -> Config:
    """Pin a config to an absolute root.

    The default root is *relative*, resolved against the cwd on every use — so a test
    holding two roots at once has to pin them, or the second fixture's ``chdir`` silently
    repoints the first one's config at the wrong knowledge base. This is the same hazard
    ``COURSE_KB_ROOT`` exists for, met from the other side.
    """
    return replace(cfg, root=root.resolve())


@pytest.fixture
def sender(tmp_path, monkeypatch, dummy_config):
    """A populated course in its own data root."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "CS101"]) == 0
    cfg = _absolute(load_config(), tmp_path / "course-kb")
    write(cfg, "CS101", "lecture/01.txt", "Skip lists pick tower height by coin flip.\n")
    write(cfg, "CS101", "lecture/02.txt", "Red-black trees rebalance with rotations.\n")
    write(cfg, "CS101", "exam/final.txt", "Question 1: analyse quickselect.\n")
    apply_sync(cfg, "CS101")
    return cfg


@pytest.fixture
def receiver(tmp_path, monkeypatch):
    """A separate, empty data root standing in for a friend's machine."""
    home = tmp_path / "friend"
    home.mkdir()
    (home / "config.toml").write_text('embedder = "dummy"\n', encoding="utf-8")
    monkeypatch.chdir(home)
    return _absolute(load_config(), home / "course-kb")


def chunks_of(cfg: Config, course: str) -> dict:
    _manifest, store = open_course(cfg, course)
    return {r.id: r for r in store.get_all()}


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


def test_an_imported_course_is_identical_to_the_exported_one(sender, receiver, tmp_path):
    """The whole premise: same sources + same model => same index, vector for vector."""
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    before = chunks_of(sender, "CS101")

    report = import_course(receiver, archive)
    after = chunks_of(receiver, "CS101")

    assert report.exact
    assert set(before) == set(after)
    assert all(before[k].text == after[k].text for k in before)
    assert all(before[k].vector == after[k].vector for k in before)
    assert all(
        (before[k].source_file, before[k].page, before[k].category)
        == (after[k].source_file, after[k].page, after[k].category)
        for k in before
    )


def test_category_folders_survive_the_trip(sender, receiver, tmp_path):
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    import_course(receiver, archive)

    root = source_root(receiver, "CS101")
    assert (root / "lecture" / "01.txt").exists()
    assert (root / "exam" / "final.txt").exists()
    _manifest, store = open_course(receiver, "CS101")
    assert set(store.categories()) == {"lecture", "exam"}


def test_course_notes_travel_with_the_course(sender, receiver, tmp_path):
    (sender.courses_dir / "CS101" / "COURSE.md").write_text("Prof: Ada.\n", encoding="utf-8")
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)

    import_course(receiver, archive)

    assert (receiver.courses_dir / "CS101" / "COURSE.md").read_text() == "Prof: Ada.\n"


def test_the_archive_carries_sources_not_the_index(sender, tmp_path):
    """Shipping index.lance would tie two machines' LanceDB versions together."""
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert not any("index.lance" in n for n in names), names
    assert sorted(n for n in names if not n.startswith("raw/")) == sorted(
        [METADATA_NAME, "COURSE.md"]
    )


def test_the_imported_course_can_be_searched(sender, receiver, tmp_path):
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    import_course(receiver, archive)

    assert main(["search", "CS101", "tower height", "--json"]) == 0


def test_an_imported_course_syncs_clean(sender, receiver, tmp_path):
    """Import must leave the sidecar consistent, or the next sync churns the index."""
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    import_course(receiver, archive)

    from courserag.sync import plan_sync

    assert plan_sync(receiver, "CS101").is_empty


# --------------------------------------------------------------------------- #
# Naming and collisions
# --------------------------------------------------------------------------- #


def test_importing_over_an_existing_course_is_refused(sender, receiver, tmp_path):
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    import_course(receiver, archive)

    with pytest.raises(TransferError, match="already exists"):
        import_course(receiver, archive)


def test_force_replaces_an_existing_course(sender, receiver, tmp_path):
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)
    import_course(receiver, archive)

    report = import_course(receiver, archive, force=True)

    assert report.exact


def test_as_imports_under_a_different_name(sender, receiver, tmp_path):
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)

    report = import_course(receiver, archive, as_course="CS101-fromDana")

    assert report.course_id == "CS101-fromDana"
    assert (receiver.courses_dir / "CS101-fromDana" / "manifest.json").exists()
    assert not (receiver.courses_dir / "CS101").exists()


# --------------------------------------------------------------------------- #
# Rejecting archives that cannot be trusted
# --------------------------------------------------------------------------- #


def test_a_plain_zip_is_not_a_course_export(tmp_path):
    bogus = tmp_path / "holiday-photos.zip"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("beach.jpg", b"\xff\xd8\xff")

    with pytest.raises(TransferError, match="not a CourseRAG course export"):
        read_metadata(bogus)


def test_a_future_format_version_is_refused(tmp_path):
    future = tmp_path / "next.zip"
    with zipfile.ZipFile(future, "w") as zf:
        zf.writestr(METADATA_NAME, json.dumps({"format": "courserag-course/99", "course": "X"}))

    with pytest.raises(TransferError, match="declares format"):
        read_metadata(future)


def test_a_corrupt_file_is_refused(tmp_path):
    corrupt = tmp_path / "truncated.zip"
    corrupt.write_bytes(b"PK\x03\x04 this is not a zip")

    with pytest.raises(TransferError, match="not a readable zip"):
        read_metadata(corrupt)


def test_a_missing_archive_is_refused(tmp_path):
    with pytest.raises(TransferError, match="No such archive"):
        read_metadata(tmp_path / "nope.zip")


def test_an_archive_with_no_sources_is_refused(receiver, tmp_path):
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr(
            METADATA_NAME,
            json.dumps(
                {
                    "format": FORMAT,
                    "course": "GHOST",
                    "embedding_model": "dummy-hash-64",
                    "dims": 64,
                    "chunk_count": 0,
                }
            ),
        )

    with pytest.raises(TransferError, match="no source files"):
        import_course(receiver, empty)


def test_exporting_a_course_with_no_files_is_refused(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "EMPTY"]) == 0

    with pytest.raises(TransferError, match="no files"):
        export_course(load_config(), "EMPTY", tmp_path / "empty.zip")


# --------------------------------------------------------------------------- #
# A hostile archive
# --------------------------------------------------------------------------- #


def _malicious(path: Path, member: str) -> Path:
    """An otherwise-valid export whose one source file has a hostile path."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            METADATA_NAME,
            json.dumps(
                {
                    "format": FORMAT,
                    "course": "EVIL",
                    "embedding_model": "dummy-hash-64",
                    "dims": 64,
                    "chunk_count": 1,
                }
            ),
        )
        zf.writestr(member, "Skip lists pick tower height by coin flip.\n")
    return path


HOSTILE = [
    "raw/../../../../../../tmp/pwned.txt",
    "raw/../../pwned.txt",
    "raw/lecture/../../../pwned.txt",
    "/raw/pwned.txt",
    "raw/....//....//pwned.txt",
    "raw/..\\..\\pwned.txt",
    "raw/./../../pwned.txt",
]


@pytest.mark.parametrize("member", HOSTILE)
def test_no_hostile_path_can_resolve_outside_the_course_folder(tmp_path, member):
    """Zip stores whatever path string the writer chose; ``..`` in one is 'zip slip'.

    Asserted on the destination path itself rather than by looking for stray files
    afterwards: a filesystem sweep can only prove nothing escaped *this* time, and
    would quietly pass if the payload happened to land somewhere the sweep did not
    look. This is the invariant the extraction loop depends on.
    """
    from courserag.transfer import _safe_members

    archive = _malicious(tmp_path / "evil.zip", member)
    root = (tmp_path / "root").resolve()

    with zipfile.ZipFile(archive) as zf:
        for _info, rel in _safe_members(zf):
            assert ".." not in rel.parts
            assert not rel.is_absolute()
            assert root in (root / Path(*rel.parts)).resolve().parents


@pytest.mark.parametrize("member", HOSTILE)
def test_a_hostile_archive_imports_with_its_payload_contained(receiver, tmp_path, member):
    """The traversal is stripped rather than rejected, so the file still arrives — inside."""
    archive = _malicious(tmp_path / "evil.zip", member)

    import_course(receiver, archive)

    root = source_root(receiver, "EVIL").resolve()
    landed = [p for p in root.rglob("*") if p.is_file()]
    assert landed, "the payload vanished entirely"
    assert all(root in p.resolve().parents for p in landed)
    # And nothing was written above the course's own folder.
    assert not (receiver.courses_dir / "EVIL" / "pwned.txt").exists()
    assert not (receiver.root / "pwned.txt").exists()


# --------------------------------------------------------------------------- #
# The embedding model
# --------------------------------------------------------------------------- #


def test_an_unavailable_model_is_reported_before_anything_is_written(
    receiver, tmp_path, monkeypatch
):
    # Stubbed rather than named-and-fetched: a real unknown HuggingFace id would send
    # this test to the network, where it would fail slowly, differently offline, and
    # for reasons unrelated to what is being checked.
    def _unavailable(name, cfg):
        raise ImportError('sentence-transformers is not installed. pip install -e ".[local]"')

    monkeypatch.setattr("courserag.embedding.get_embedder", _unavailable)

    archive = tmp_path / "exotic.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            METADATA_NAME,
            json.dumps(
                {
                    "format": FORMAT,
                    "course": "EXOTIC",
                    "embedding_model": "some-lab/not-a-real-model",
                    "dims": 1024,
                    "chunk_count": 5,
                }
            ),
        )
        zf.writestr("raw/notes.txt", "Skip lists.\n")

    with pytest.raises(TransferError, match="cannot be loaded here"):
        import_course(receiver, archive)

    assert not (receiver.courses_dir / "EXOTIC").exists()


def test_a_dims_disagreement_is_refused(receiver, tmp_path):
    """Same model name, different width, means the vectors would not be comparable."""
    archive = tmp_path / "wide.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            METADATA_NAME,
            json.dumps(
                {
                    "format": FORMAT,
                    "course": "WIDE",
                    "embedding_model": "dummy-hash-64",
                    "dims": 384,  # the dummy is 64 here
                    "chunk_count": 1,
                }
            ),
        )
        zf.writestr("raw/notes.txt", "Skip lists.\n")

    with pytest.raises(TransferError, match="Refusing to build"):
        import_course(receiver, archive)


def test_the_archives_model_wins_over_the_importers_config(sender, tmp_path, monkeypatch):
    """A friend whose config says something else must still get the sender's course."""
    archive = tmp_path / "CS101.zip"
    export_course(sender, "CS101", archive)

    home = tmp_path / "other"
    home.mkdir()
    # This config would otherwise build a 384-dim course.
    (home / "config.toml").write_text('embedder = "minilm"\n', encoding="utf-8")
    monkeypatch.chdir(home)

    report = import_course(load_config(), archive)

    assert report.embedding_model == "dummy-hash-64"
    _manifest, store = open_course(load_config(), "CS101")
    assert store.dims == 64


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #


def test_export_then_import_through_the_cli(sender, tmp_path, monkeypatch, capsys):
    assert main(["export", "CS101", "-o", str(tmp_path / "out.zip")]) == 0
    assert "Exported 'CS101'" in capsys.readouterr().out

    home = tmp_path / "cli-friend"
    home.mkdir()
    (home / "config.toml").write_text('embedder = "dummy"\n', encoding="utf-8")
    monkeypatch.chdir(home)

    assert main(["import", str(tmp_path / "out.zip")]) == 0
    assert "an exact match" in capsys.readouterr().out


def test_export_defaults_to_a_named_file_in_the_cwd(sender, tmp_path, capsys):
    assert main(["export", "CS101"]) == 0
    assert (tmp_path / "CS101-courserag.zip").is_file()


def test_export_into_a_directory_that_does_not_exist_yet(sender, tmp_path, capsys):
    """A trailing slash means 'into here', even before the directory exists."""
    assert main(["export", "CS101", "-o", f"{tmp_path / 'share'}/"]) == 0

    assert (tmp_path / "share" / "CS101-courserag.zip").is_file()


def test_export_refuses_to_clobber_without_force(sender, tmp_path, capsys):
    dest = tmp_path / "out.zip"
    dest.write_bytes(b"precious")

    assert main(["export", "CS101", "-o", str(dest)]) == 1
    assert dest.read_bytes() == b"precious"
    assert main(["export", "CS101", "-o", str(dest), "--force"]) == 0


def test_importing_an_unreadable_archive_exits_nonzero(receiver, tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip at all")

    assert main(["import", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err


def test_exporting_an_unknown_course_exits_nonzero(sender, capsys):
    assert main(["export", "NOPE"]) == 1
    assert "not initialized" in capsys.readouterr().err
