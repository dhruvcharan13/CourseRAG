"""Per-course notes: the one thing an agent may write.

The behaviour that matters is not "a file gets appended to" — it is that notes reach
every read of the course, that a repeated fact does not accumulate, and that nothing
here can destroy what a term of notes has collected.
"""

from __future__ import annotations

from datetime import date

import pytest

from courserag.cli import main
from courserag.config import Config, load_config
from courserag.memory import (
    MAX_NOTE_CHARS,
    NoteRefused,
    append_note,
    for_prompt,
    notes_path,
    read_notes,
    strip_boilerplate,
)


@pytest.fixture
def course(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    return load_config()


# --------------------------------------------------------------------------- #
# Reading and appending
# --------------------------------------------------------------------------- #


def test_a_fresh_course_has_no_notes(course):
    assert read_notes(course, "C") == ""
    assert for_prompt(course, "C") == ""


def test_a_note_is_stored_dated(course):
    result = append_note(course, "C", "The prof writes n, never N.", on=date(2026, 3, 1))

    assert result.added
    assert "- 2026-03-01: The prof writes n, never N." in read_notes(course, "C")


def test_notes_accumulate_in_order(course):
    append_note(course, "C", "First fact.", on=date(2026, 3, 1))
    append_note(course, "C", "Second fact.", on=date(2026, 3, 2))

    body = read_notes(course, "C")
    assert body.index("First fact.") < body.index("Second fact.")


def test_the_same_fact_is_not_stored_twice(course):
    """An agent re-learning something every session would otherwise grow the file."""
    append_note(course, "C", "The prof writes n, never N.")

    again = append_note(course, "C", "The prof writes n, never N.")

    assert not again.added
    assert read_notes(course, "C").count("never N") == 1


def test_a_repeat_on_a_later_day_is_still_recognised(course):
    """Dedup compares the fact, not the dated line it was written on."""
    append_note(course, "C", "Module 7 was skipped.", on=date(2026, 3, 1))

    again = append_note(course, "C", "Module 7 was skipped.", on=date(2026, 9, 9))

    assert not again.added


def test_a_multiline_note_is_flattened_to_one_line(course):
    """One note is one list item; a stray newline must not break the file's shape."""
    append_note(course, "C", "A fact\nsplit over\nlines.", on=date(2026, 3, 1))

    body = read_notes(course, "C")
    assert "- 2026-03-01: A fact split over lines." in body


@pytest.mark.parametrize("bad", ["", "   ", "\n\n", "\t"])
def test_an_empty_note_is_refused(course, bad):
    with pytest.raises(NoteRefused, match="cannot be empty"):
        append_note(course, "C", bad)


def test_a_note_the_size_of_a_document_is_refused(course):
    """Notes are facts; a pasted transcript belongs in an ingested file."""
    with pytest.raises(NoteRefused, match="single facts"):
        append_note(course, "C", "x" * (MAX_NOTE_CHARS + 1))

    assert read_notes(course, "C") == ""


def test_appending_preserves_notes_a_human_wrote_by_hand(course):
    """The file is meant to be edited directly; appends must not reformat it."""
    handwritten = "## My own structure\n\nSome prose I typed.\n"
    notes_path(course, "C").write_text(handwritten, encoding="utf-8")

    append_note(course, "C", "An appended fact.", on=date(2026, 3, 1))

    body = read_notes(course, "C")
    assert "## My own structure" in body and "Some prose I typed." in body
    assert body.endswith("- 2026-03-01: An appended fact.")


def test_unreadable_notes_read_as_absent_rather_than_raising(course):
    """Notes decorate every read path, so they can never be the thing that breaks it."""
    notes_path(course, "C").write_bytes(b"\xff\xfe\x00 not valid utf-8")

    assert read_notes(course, "C") == ""
    assert for_prompt(course, "C") == ""


# --------------------------------------------------------------------------- #
# What rides along on a read
# --------------------------------------------------------------------------- #


def test_the_generated_header_is_dropped_from_the_prompt_form(course):
    """It explains the file to a human editor; on every search it is pure noise."""
    append_note(course, "C", "A fact.", on=date(2026, 3, 1))

    prompt = for_prompt(course, "C")

    assert "# Course notes" not in prompt
    assert "documents do not state" not in prompt
    assert "- 2026-03-01: A fact." in prompt


def test_a_users_own_heading_is_kept(course):
    notes_path(course, "C").write_text("## Marking\n\n- A4 is 20%.\n", encoding="utf-8")

    assert "## Marking" in for_prompt(course, "C")


def test_strip_boilerplate_leaves_ordinary_notes_alone():
    assert strip_boilerplate("- one\n- two") == "- one\n- two"


def test_long_notes_are_truncated_visibly_with_a_pointer(course):
    for i in range(200):
        append_note(course, "C", f"Fact number {i} about this course.", on=date(2026, 3, 1))

    prompt = for_prompt(course, "C", budget=400)

    assert len(prompt) < 700
    assert "notes truncated" in prompt
    assert "course_info" in prompt  # names where to get the rest


def test_truncation_cuts_at_a_line_boundary(course):
    for i in range(50):
        append_note(course, "C", f"Fact {i}.", on=date(2026, 3, 1))

    prompt = for_prompt(course, "C", budget=200)

    body = [ln for ln in prompt.splitlines() if ln.startswith("- ")]
    assert all(ln.endswith(".") for ln in body), body  # no half-written note


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #


def test_kb_notes_prints_guidance_when_there_are_none(course, capsys):
    assert main(["notes", "C"]) == 0

    out = capsys.readouterr().out
    assert "No notes for 'C' yet" in out
    assert "--add" in out and "--edit" in out


def test_kb_notes_adds_and_prints(course, capsys):
    assert main(["notes", "C", "--add", "The prof writes n, never N."]) == 0
    capsys.readouterr()

    assert main(["notes", "C"]) == 0
    assert "never N" in capsys.readouterr().out


def test_kb_notes_reports_a_duplicate_without_failing(course, capsys):
    main(["notes", "C", "--add", "A fact."])
    capsys.readouterr()

    assert main(["notes", "C", "--add", "A fact."]) == 0
    assert "Already recorded" in capsys.readouterr().out


def test_kb_notes_refuses_an_empty_note(course, capsys):
    assert main(["notes", "C", "--add", "   "]) == 1
    assert "cannot be empty" in capsys.readouterr().err


def test_kb_notes_on_an_unknown_course_exits_nonzero(course, capsys):
    assert main(["notes", "NOPE"]) == 1
    assert "not initialized" in capsys.readouterr().err


def test_kb_notes_edit_opens_the_file_in_the_editor(course, monkeypatch, tmp_path):
    """--edit is the escape hatch for changing or removing, which appends cannot do."""
    seen = {}
    monkeypatch.setenv("EDITOR", "my-editor --wait")

    def fake_call(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("courserag.cli.subprocess.call", fake_call)

    assert main(["notes", "C", "--edit"]) == 0

    assert seen["argv"][:2] == ["my-editor", "--wait"]
    assert seen["argv"][-1] == str(notes_path(course, "C"))
    assert notes_path(course, "C").exists()  # created if it was missing
