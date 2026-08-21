"""The folder is the source of truth; sync is what makes that true.

These cover the four transitions a mirror has to get right — added, changed, removed,
untouched — plus the two guards that keep "the folder is empty" from being read as
"delete everything".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courserag.cli import main
from courserag.config import Config, load_config
from courserag.ingest import open_course
from courserag.store import CourseStore
from courserag.sync import (
    RenameRefused,
    SyncRefused,
    apply_sync,
    clean_category,
    forget_source,
    import_paths,
    load_state,
    plan_import,
    plan_sync,
    rename_source,
    scan_sources,
    set_category,
    source_root,
)


@pytest.fixture
def course(tmp_path, monkeypatch, dummy_config):
    """An initialized dummy-embedder course, with the cwd inside its data root."""
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    return load_config()


def write(cfg: Config, rel: str, text: str) -> None:
    """Write a source file into the course's raw/ folder, creating parents."""
    path = source_root(cfg, "C") / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def counts(cfg: Config) -> dict[str, int]:
    _manifest, store = open_course(cfg, "C")
    return store.source_file_counts()


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #


def test_a_subfolder_names_the_category_and_loose_files_take_the_default(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "syllabus.txt", "Grades: 40% assignments, 60% exams.\n")

    scan = scan_sources(course, "C")

    assert scan.files["05.txt"].category == "lecture"
    assert scan.files["syllabus.txt"].category == "notes"  # Config.default_category


def test_nested_folders_join_into_one_category_path(course):
    write(course, "slides/week1/intro.txt", "Course overview and logistics.\n")
    assert scan_sources(course, "C").files["intro.txt"].category == "slides/week1"


def test_unparseable_extensions_and_dotfiles_are_skipped_not_failed(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    write(course, "archive.zip", "not really a zip\n")
    write(course, ".hidden.txt", "editor scratch\n")
    (source_root(course, "C") / ".DS_Store").write_bytes(b"\x00")

    scan = scan_sources(course, "C")

    assert set(scan.files) == {"notes.txt"}
    assert [p.name for p in scan.unsupported] == ["archive.zip"]


def test_two_files_sharing_a_basename_are_refused_rather_than_guessed_at(course):
    """Chunks are keyed by basename, so both would claim the same identity."""
    write(course, "lecture/05.txt", "Skip lists and their expected height.\n")
    write(course, "tutorial/05.txt", "Tutorial worksheet on skip lists.\n")

    scan = scan_sources(course, "C")

    assert "05.txt" not in scan.files
    assert sorted(scan.duplicates["05.txt"]) == ["lecture/05.txt", "tutorial/05.txt"]


# --------------------------------------------------------------------------- #
# The four transitions
# --------------------------------------------------------------------------- #


def test_a_new_file_is_added(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")

    report = apply_sync(course, "C")

    assert [r.source_file for r in report.added] == ["05.txt"]
    assert counts(course)["05.txt"] > 0
    assert load_state(course, "C")["05.txt"]["category"] == "lecture"


def test_an_untouched_file_is_left_alone(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")

    report = apply_sync(course, "C")

    assert not report.changed
    assert report.unchanged == ["notes.txt"]


def test_an_edited_file_is_reindexed_rather_than_appended_to(course):
    """Content-hash dedup alone would add the new chunks and keep the stale ones.

    That is the whole reason sync tracks digests: without a delete-then-ingest the
    index would accumulate every draft of a file that was ever synced, and searches
    would cite text no longer in the document.
    """
    write(course, "notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    write(course, "notes.txt", "Red-black trees rebalance with rotations and recoloring.\n")

    report = apply_sync(course, "C")

    assert [r.source_file for r in report.reindexed] == ["notes.txt"]
    # A reindex's internal delete is a step, not a user-facing removal.
    assert report.removed == []
    _manifest, store = open_course(course, "C")
    stored = " ".join(r.text for r in store.get_all())
    assert "rotations" in stored and "coin flip" not in stored


def test_a_file_deleted_from_disk_is_removed_from_the_index(course):
    write(course, "keep.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "drop.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "drop.txt").unlink()

    report = apply_sync(course, "C")

    assert [r.source_file for r in report.removed] == ["drop.txt"]
    assert set(counts(course)) == {"keep.txt"}
    _manifest, _store = open_course(course, "C")
    assert "drop.txt" not in load_state(course, "C")


def test_moving_a_file_between_category_folders_recategorizes_it(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    root = source_root(course, "C")
    (root / "tutorial").mkdir()
    (root / "lecture" / "05.txt").rename(root / "tutorial" / "05.txt")

    apply_sync(course, "C")

    _manifest, store = open_course(course, "C")
    assert {r.category for r in store.get_all()} == {"tutorial"}


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_an_empty_folder_does_not_silently_clear_the_index(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "notes.txt").unlink()

    with pytest.raises(SyncRefused, match="almost always a missing folder"):
        apply_sync(course, "C")

    assert counts(course)["notes.txt"] > 0  # nothing was touched


def test_force_clears_it_anyway(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "notes.txt").unlink()

    report = apply_sync(course, "C", force=True)

    assert [r.source_file for r in report.removed] == ["notes.txt"]
    assert counts(course) == {}


def test_a_missing_source_folder_is_refused_separately_from_an_empty_one(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    import shutil

    shutil.rmtree(source_root(course, "C"))

    with pytest.raises(SyncRefused, match="does not exist"):
        apply_sync(course, "C")


def test_deleting_only_some_files_is_not_treated_as_a_wipe(course):
    """The guard is about a vanished folder, not about deleting things on purpose."""
    write(course, "keep.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "drop.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "drop.txt").unlink()

    report = apply_sync(course, "C")  # no force needed

    assert [r.source_file for r in report.removed] == ["drop.txt"]


def test_one_unparseable_file_does_not_abort_the_batch(course, monkeypatch):
    write(course, "good.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "empty.txt", "")

    report = apply_sync(course, "C")

    assert [r.source_file for r in report.added] == ["good.txt"]
    assert [name for name, _ in report.failed] == ["empty.txt"]


# --------------------------------------------------------------------------- #
# Adoption: courses that predate sync
# --------------------------------------------------------------------------- #


def test_files_ingested_before_sync_existed_are_adopted_not_reingested(course):
    """`kb ingest` from anywhere, then drop the same file in raw/: no double index."""
    external = source_root(course, "C").parent.parent.parent / "outside.txt"
    external.write_text("Skip lists pick tower height by coin flip.\n", encoding="utf-8")
    assert main(["ingest", "C", str(external), "--category", "lecture"]) == 0
    before = counts(course)["outside.txt"]

    import_paths(course, "C", [external], category="lecture")
    report = apply_sync(course, "C")

    assert report.unchanged == ["outside.txt"]
    assert counts(course)["outside.txt"] == before
    # It is tracked from now on, so a later edit is detected.
    assert "outside.txt" in load_state(course, "C")


def test_adopted_files_are_reindexed_once_they_actually_change(course):
    write(course, "notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    # Simulate a course that predates the sidecar entirely.
    (source_root(course, "C").parent / "sync.json").unlink()
    write(course, "notes.txt", "Red-black trees rebalance with rotations.\n")

    first = apply_sync(course, "C")
    assert first.unchanged == ["notes.txt"]  # nothing to compare against yet

    write(course, "notes.txt", "Now it changes against a recorded digest.\n")
    second = apply_sync(course, "C")
    assert [r.source_file for r in second.reindexed] == ["notes.txt"]


# --------------------------------------------------------------------------- #
# Importing into raw/
# --------------------------------------------------------------------------- #


def test_importing_a_folder_makes_it_a_category(course, tmp_path):
    src = tmp_path / "incoming" / "lecture"
    src.mkdir(parents=True)
    (src / "05.txt").write_text("Skip lists pick tower height.\n", encoding="utf-8")

    import_paths(course, "C", [src])

    assert (source_root(course, "C") / "lecture" / "05.txt").exists()
    assert scan_sources(course, "C").files["05.txt"].category == "lecture"


def test_importing_an_indexed_file_files_it_under_its_recorded_category(course, tmp_path):
    """Adopting a pre-sync course must not flatten its hand-assigned categories."""
    external = tmp_path / "corpus" / "05.txt"
    external.parent.mkdir(parents=True)
    external.write_text("Skip lists pick tower height by coin flip.\n", encoding="utf-8")
    assert main(["ingest", "C", str(external), "--category", "exam"]) == 0

    # The source directory is named "corpus", but the index says this is an exam.
    import_paths(course, "C", [external.parent])

    assert (source_root(course, "C") / "exam" / "05.txt").exists()
    assert plan_sync(course, "C").is_empty


def test_an_explicit_category_still_wins_over_the_recorded_one(course, tmp_path):
    external = tmp_path / "05.txt"
    external.write_text("Skip lists pick tower height by coin flip.\n", encoding="utf-8")
    assert main(["ingest", "C", str(external), "--category", "exam"]) == 0

    import_paths(course, "C", [external], category="lecture")

    assert (source_root(course, "C") / "lecture" / "05.txt").exists()


def test_deleting_a_file_also_removes_it_from_the_folder(course):
    """Otherwise the next sync puts it straight back and the delete looks broken."""
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")

    assert main(["delete", "C", "notes.txt"]) == 0

    assert not (source_root(course, "C") / "notes.txt").exists()
    assert plan_sync(course, "C").is_empty


def test_forget_source_is_safe_when_the_file_is_already_gone(course):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "notes.txt").unlink()

    assert forget_source(course, "C", "notes.txt") is None


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #


def test_dry_run_reports_the_plan_and_changes_nothing(course, capsys):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")

    assert main(["sync", "C", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "+ lecture/05.txt" in out and "[lecture]" in out
    assert counts(course) == {}


def test_dry_run_does_not_copy_either(course, tmp_path, capsys):
    """Copying into raw/ is a write, and --dry-run promises not to write."""
    src = tmp_path / "incoming"
    src.mkdir()
    (src / "notes.txt").write_text("Skip lists pick tower height.\n", encoding="utf-8")

    assert main(["sync", "C", "--from", str(src), "--dry-run"]) == 0

    assert "would copy into raw/: incoming/notes.txt" in capsys.readouterr().out
    assert not (source_root(course, "C") / "incoming").exists()
    assert counts(course) == {}


def test_sync_from_copies_then_indexes(course, tmp_path, capsys):
    src = tmp_path / "incoming"
    src.mkdir()
    (src / "notes.txt").write_text("Skip lists pick tower height.\n", encoding="utf-8")

    assert main(["sync", "C", "--from", str(src)]) == 0

    assert "notes.txt" in counts(course)
    assert scan_sources(course, "C").files["notes.txt"].category == "incoming"


def test_the_wipe_guard_surfaces_as_a_nonzero_exit(course, capsys):
    write(course, "notes.txt", "A red-black tree is a balanced BST.\n")
    assert main(["sync", "C"]) == 0
    (source_root(course, "C") / "notes.txt").unlink()

    assert main(["sync", "C"]) == 1
    assert "Pass force if you meant it" in capsys.readouterr().err
    assert main(["sync", "C", "--force"]) == 0


def test_sync_needs_no_embedder_when_only_deleting(course, monkeypatch):
    """Removals compare no vectors, so a course whose model is gone can still shrink."""
    write(course, "keep.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "drop.txt", "A red-black tree is a balanced BST.\n")
    apply_sync(course, "C")
    (source_root(course, "C") / "drop.txt").unlink()

    def _unavailable(name, cfg):
        raise ImportError("sentence-transformers is not installed")

    monkeypatch.setattr("courserag.ingest.get_embedder", _unavailable)

    report = apply_sync(course, "C")
    assert [r.source_file for r in report.removed] == ["drop.txt"]


def test_sync_on_an_uninitialized_course_explains_itself(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["sync", "NOPE"]) == 1
    assert "not initialized" in capsys.readouterr().err


def test_the_store_counts_chunks_per_file(course):
    write(course, "a.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "b.txt", "A red-black tree is a balanced BST with colored nodes.\n")
    apply_sync(course, "C")

    _manifest, store = open_course(course, "C")
    per_file = store.source_file_counts()

    assert set(per_file) == {"a.txt", "b.txt"}
    assert sum(per_file.values()) == store.count()


def test_reopening_the_store_after_sync_sees_the_same_rows(course):
    write(course, "notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    manifest, _store = open_course(course, "C")

    reopened = CourseStore.open_or_create(course.courses_dir / "C", manifest.dims)

    assert reopened.source_files() == ["notes.txt"]


# --------------------------------------------------------------------------- #
# Changing a category
# --------------------------------------------------------------------------- #


def test_setting_a_category_moves_the_file_and_the_next_sync_reindexes_it(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    set_category(course, "C", "05.txt", "exam")

    # The move alone does not touch the index; sync is what rewrites the chunks.
    assert (source_root(course, "C") / "exam" / "05.txt").exists()
    report = apply_sync(course, "C")
    assert [r.source_file for r in report.reindexed] == ["05.txt"]
    _manifest, store = open_course(course, "C")
    assert store.source_categories() == {"05.txt": "exam"}


def test_an_emptied_category_folder_is_removed(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    set_category(course, "C", "05.txt", "exam")

    assert not (source_root(course, "C") / "lecture").exists()


def test_a_category_folder_still_holding_files_is_kept(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "lecture/06.txt", "Red-black trees rebalance with rotations.\n")
    apply_sync(course, "C")

    set_category(course, "C", "05.txt", "exam")

    assert (source_root(course, "C") / "lecture" / "06.txt").exists()


def test_the_default_category_means_the_top_of_raw(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    set_category(course, "C", "05.txt", "notes")

    assert (source_root(course, "C") / "05.txt").exists()
    apply_sync(course, "C")
    _manifest, store = open_course(course, "C")
    assert store.source_categories() == {"05.txt": "notes"}


def test_setting_the_same_category_is_a_no_op(course):
    write(course, "lecture/05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    set_category(course, "C", "05.txt", "lecture")

    assert plan_sync(course, "C").is_empty


def test_setting_a_category_on_a_missing_file_raises(course):
    with pytest.raises(KeyError):
        set_category(course, "C", "ghost.txt", "exam")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("lecture", "lecture"),
        ("  lecture  ", "lecture"),
        ("slides/week1", "slides/week1"),
        ("../../escape", "escape"),
        ("/absolute", "absolute"),
        (".hidden", "notes"),
        ("", "notes"),
        (None, "notes"),
        ("..", "notes"),
    ],
)
def test_a_category_name_cannot_become_a_path_that_escapes(course, raw, expected):
    assert clean_category(raw, course) == expected


def test_a_hostile_category_lands_inside_raw(course):
    write(course, "05.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    root = source_root(course, "C").resolve()

    dest = set_category(course, "C", "05.txt", "../../../pwned")

    assert root in dest.resolve().parents


# --------------------------------------------------------------------------- #
# Renaming a file
# --------------------------------------------------------------------------- #


def test_renaming_moves_the_chunks_to_the_new_name(course):
    write(course, "lecture/old.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    rename_source(course, "C", "old.txt", "new.txt")
    report = apply_sync(course, "C")

    # Chunks are keyed by basename, so a rename is a delete plus a fresh ingest.
    assert [r.source_file for r in report.removed] == ["old.txt"]
    assert [r.source_file for r in report.added] == ["new.txt"]
    assert set(counts(course)) == {"new.txt"}


def test_renaming_keeps_the_category(course):
    write(course, "lecture/old.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    rename_source(course, "C", "old.txt", "new.txt")
    apply_sync(course, "C")

    assert (source_root(course, "C") / "lecture" / "new.txt").exists()
    _manifest, store = open_course(course, "C")
    assert store.source_categories() == {"new.txt": "lecture"}


def test_renaming_the_only_file_does_not_trip_the_wipe_guard(course):
    """One file removed and one added is not an emptied folder."""
    write(course, "only.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    rename_source(course, "C", "only.txt", "renamed.txt")

    report = apply_sync(course, "C")  # no force
    assert set(counts(course)) == {"renamed.txt"}
    assert report.total_chunks > 0


def test_a_rename_that_changes_nothing_is_a_no_op(course):
    write(course, "same.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    rename_source(course, "C", "same.txt", "same.txt")

    assert plan_sync(course, "C").is_empty


def test_renaming_onto_an_existing_name_is_refused(course):
    """Two files sharing a basename means scan_sources indexes neither."""
    write(course, "lecture/a.txt", "Skip lists pick tower height by coin flip.\n")
    write(course, "exam/b.txt", "Red-black trees rebalance with rotations.\n")
    apply_sync(course, "C")

    with pytest.raises(RenameRefused, match="already has a file named"):
        rename_source(course, "C", "a.txt", "b.txt")

    # Nothing moved, so both are still indexed.
    assert set(counts(course)) == {"a.txt", "b.txt"}


def test_renaming_away_a_supported_extension_is_refused(course):
    """Otherwise the next sync drops the chunks and indexes nothing in their place."""
    write(course, "notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    with pytest.raises(RenameRefused, match="no supported extension"):
        rename_source(course, "C", "notes.txt", "notes.zip")

    assert (source_root(course, "C") / "notes.txt").exists()
    assert set(counts(course)) == {"notes.txt"}


@pytest.mark.parametrize("bad", ["", "   ", ".hidden.txt", "/", "..", "."])
def test_an_unusable_new_name_is_refused(course, bad):
    write(course, "notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    with pytest.raises(RenameRefused):
        rename_source(course, "C", "notes.txt", bad)


@pytest.mark.parametrize(
    "attempt", ["../escaped.txt", "../../escaped.txt", "/etc/escaped.txt", "sub/dir/x.txt"]
)
def test_a_rename_cannot_relocate_a_file(course, attempt):
    """A rename changes a name, never a location — the path part is stripped."""
    write(course, "lecture/notes.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")
    root = source_root(course, "C").resolve()

    dest = rename_source(course, "C", "notes.txt", attempt)

    assert dest.resolve().parent == root / "lecture"   # same folder as before
    assert root in dest.resolve().parents
    assert dest.name == Path(attempt).name            # only the basename was taken


def test_renaming_a_file_that_is_not_on_disk_raises(course):
    with pytest.raises(KeyError):
        rename_source(course, "C", "ghost.txt", "new.txt")


def test_the_sidecar_entry_follows_the_rename(course):
    write(course, "old.txt", "Skip lists pick tower height by coin flip.\n")
    apply_sync(course, "C")

    rename_source(course, "C", "old.txt", "new.txt")

    state = load_state(course, "C")
    assert "old.txt" not in state and "new.txt" in state


# --------------------------------------------------------------------------- #
# Importing a folder with subfolders
# --------------------------------------------------------------------------- #


@pytest.fixture
def tree(tmp_path):
    """An incoming folder two levels deep, with a repeated basename across branches."""
    root = tmp_path / "incoming"
    (root / "week1" / "mon").mkdir(parents=True)
    (root / "week2").mkdir(parents=True)
    (root / "top.txt").write_text("Top level content.\n", encoding="utf-8")
    (root / "week1" / "notes.txt").write_text("WEEK ONE about skip lists.\n", encoding="utf-8")
    (root / "week1" / "mon" / "deep.txt").write_text("Deep Monday content.\n", encoding="utf-8")
    (root / "week2" / "notes.txt").write_text("WEEK TWO about trees.\n", encoding="utf-8")
    return root


def test_a_folder_keeps_its_shape(course, tree):
    import_paths(course, "C", [tree])

    root = source_root(course, "C")
    assert (root / "incoming" / "top.txt").exists()
    assert (root / "incoming" / "week1" / "notes.txt").exists()
    assert (root / "incoming" / "week1" / "mon" / "deep.txt").exists()
    assert (root / "incoming" / "week2" / "notes.txt").exists()


def test_nesting_becomes_nested_categories(course, tree):
    import_paths(course, "C", [tree])

    scan = scan_sources(course, "C")
    assert scan.files["deep.txt"].category == "incoming/week1/mon"
    assert scan.files["top.txt"].category == "incoming"


def test_two_branches_sharing_a_basename_both_survive_the_copy(course, tree):
    """Flattening used to overwrite one with the other, silently losing a document."""
    import_paths(course, "C", [tree])

    root = source_root(course, "C")
    assert (root / "incoming" / "week1" / "notes.txt").read_text().startswith("WEEK ONE")
    assert (root / "incoming" / "week2" / "notes.txt").read_text().startswith("WEEK TWO")


def test_the_basename_clash_is_reported_rather_than_indexed(course, tree):
    """Both files are kept on disk; neither is indexed, and the plan says why."""
    import_paths(course, "C", [tree])

    plan = plan_sync(course, "C")

    assert sorted(plan.duplicates["notes.txt"]) == [
        "incoming/week1/notes.txt",
        "incoming/week2/notes.txt",
    ]
    assert {s.name for s in plan.add} == {"top.txt", "deep.txt"}


def test_an_explicit_category_flattens_the_tree(course, tree):
    """Asking for one category means one folder, whatever the source looked like."""
    import_paths(course, "C", [tree], category="lecture")

    root = source_root(course, "C")
    assert (root / "lecture" / "deep.txt").exists()
    assert not (root / "lecture" / "week1").exists()


def test_flattening_reports_a_collision_instead_of_overwriting(course, tree):
    report = import_paths(course, "C", [tree], category="lecture")

    assert len(report.collisions) == 1
    kept, dropped = report.collisions[0]
    assert "week1/notes.txt" in kept and "week2/notes.txt" in dropped
    # The first file written wins; the second is not silently copied over it.
    assert (source_root(course, "C") / "lecture" / "notes.txt").read_text().startswith("WEEK ONE")


def test_reimporting_the_same_tree_changes_nothing(course, tree):
    import_paths(course, "C", [tree])

    report = import_paths(course, "C", [tree])

    assert report.copied == []
    assert len(report.skipped) == 4


def test_an_updated_file_is_replaced_and_reported(course, tree):
    import_paths(course, "C", [tree])
    (tree / "top.txt").write_text("Top level content, revised and longer.\n", encoding="utf-8")

    report = import_paths(course, "C", [tree])

    assert report.replaced == ["incoming/top.txt"]
    assert "revised" in (source_root(course, "C") / "incoming" / "top.txt").read_text()


def test_a_dry_run_reports_the_same_shape_without_writing(course, tree):
    planned = plan_import(course, "C", [tree])

    assert "incoming/week1/mon/deep.txt" in planned.copied
    assert not source_root(course, "C").joinpath("incoming").exists()


def test_hidden_directories_in_the_tree_are_skipped(course, tmp_path):
    root = tmp_path / "incoming"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config.txt").write_text("not course material\n", encoding="utf-8")
    (root / "real.txt").write_text("Skip lists and tower height.\n", encoding="utf-8")

    import_paths(course, "C", [root])

    assert set(scan_sources(course, "C").files) == {"real.txt"}


def test_syncing_a_deep_tree_indexes_every_level(course, tree):
    (tree / "week2" / "notes.txt").unlink()  # remove the clash so both index
    import_paths(course, "C", [tree])

    report = apply_sync(course, "C")

    assert {r.source_file for r in report.added} == {"top.txt", "notes.txt", "deep.txt"}
    _manifest, store = open_course(course, "C")
    assert set(store.source_categories().values()) == {
        "incoming",
        "incoming/week1",
        "incoming/week1/mon",
    }
