"""Per-course notes: the standing facts retrieval cannot supply.

``COURSE.md`` has existed in every course folder since the first commit, created empty
and never read — the same state ``raw/`` was in before ``kb sync`` gave it a meaning.
This module gives it one. It holds what is true about a *course* rather than what is
written in its documents: that the professor writes ``n`` for input size and never
``N``, that module 7 was skipped this term, that citations should use module numbers
rather than filenames, that assignment 4 is worth 20%.

Retrieval cannot produce any of that. It is either nowhere in the slides or scattered
across an aside in one lecture, and no ranking recovers it reliably. So it is kept
beside the index as plain markdown, and surfaced to every read of the course.

Appends are the only programmatic write. An agent that could rewrite the file could
also erase a term's worth of accumulated notes with one bad call, and the failure would
be silent — nothing else reads the file, so nobody would notice until the notes were
needed. Editing and deleting are deliberately left to a text editor and ``kb notes``,
where a human sees what they are removing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from courserag.config import Config
from courserag.ingest import course_dir

NOTES_NAME = "COURSE.md"

#: Cap on how much of the notes ride along on *every* read of a course. Notes are
#: prepended to search results, so an unbounded file would tax every query; past this
#: the text is cut at a line boundary and the reader is pointed at course_info for the
#: whole thing. Generous enough that a normal term's notes are never truncated.
PROMPT_BUDGET_CHARS = 2000

#: Refuse a single note longer than this. A note is a fact, not a document — something
#: this size is a pasted transcript, and the place for that is an ingested file.
MAX_NOTE_CHARS = 2000

_HEADER = "# Course notes\n\nFacts about this course that its documents do not state.\n"


class NoteRefused(Exception):
    """A note was not worth storing, with a reason to hand back."""


@dataclass
class AppendResult:
    note: str
    #: False when an identical note was already present.
    added: bool
    #: The whole file after the append.
    text: str


def notes_path(cfg: Config, course_id: str) -> Path:
    """Where a course's notes live."""
    return course_dir(cfg, course_id) / NOTES_NAME


def read_notes(cfg: Config, course_id: str) -> str:
    """The course's notes, or an empty string if there are none.

    Never raises for a missing or unreadable file: notes are an enrichment on every
    read path, and a course with no notes must behave exactly like one whose notes
    file happens to be absent.
    """
    path = notes_path(cfg, course_id)
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def strip_boilerplate(notes: str) -> str:
    """Drop the header this module writes into a fresh notes file.

    The header explains the file to a human opening it in an editor, and is pure noise
    once the notes are being pasted above every search result. Only the exact generated
    lines are removed — anything a user wrote themselves, heading or not, is kept.
    """
    boilerplate = {ln.strip() for ln in _HEADER.strip().splitlines() if ln.strip()}
    lines = notes.splitlines()
    while lines and (not lines[0].strip() or lines[0].strip() in boilerplate):
        lines.pop(0)
    return "\n".join(lines).strip()


def for_prompt(cfg: Config, course_id: str, budget: int = PROMPT_BUDGET_CHARS) -> str:
    """Notes formatted to sit above a tool's output, or "" when there are none.

    Truncation is visible and names where to get the rest, for the same reason
    ``read_document``'s is: silently dropping half the standing instructions about a
    course is worse than saying they were dropped.
    """
    notes = strip_boilerplate(read_notes(cfg, course_id))
    if not notes:
        return ""
    if len(notes) > budget:
        kept = notes[:budget].rsplit("\n", 1)[0]
        notes = (
            f"{kept}\n[notes truncated at {budget} characters; "
            f"call course_info({course_id!r}) for all of them]"
        )
    return f"Notes on {course_id} (recorded by you or the user, not from its documents):\n{notes}"


def append_note(
    cfg: Config, course_id: str, note: str, *, on: date | None = None
) -> AppendResult:
    """Add one note to a course, dated. Returns what the file now says.

    An exact duplicate is not appended — an agent re-learning the same fact across
    sessions would otherwise grow the file without adding anything, and the notes ride
    along on every read.

    Raises:
        NoteRefused: for an empty note, or one long enough to be a document.
    """
    text = " ".join(str(note).split())  # collapse newlines: one note is one line
    if not text:
        raise NoteRefused("A note cannot be empty.")
    if len(text) > MAX_NOTE_CHARS:
        raise NoteRefused(
            f"That note is {len(text)} characters; the limit is {MAX_NOTE_CHARS}. Notes "
            f"are single facts about the course — ingest a document if you want its "
            f"whole contents searchable."
        )

    path = notes_path(cfg, course_id)
    existing = read_notes(cfg, course_id)
    stamp = (on or date.today()).isoformat()
    line = f"- {stamp}: {text}"

    # Compare on the note body, not the whole line, so the same fact recorded on a
    # different day is still recognised as already known.
    if any(ln.split(": ", 1)[-1].strip() == text for ln in existing.splitlines()):
        return AppendResult(note=text, added=False, text=existing)

    # A blank line after the header paragraph, so the notes are a real markdown list
    # rather than a run-on under the prose.
    body = existing if existing else _HEADER.rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated = f"{body}\n{line}\n"
    path.write_text(updated, encoding="utf-8")
    return AppendResult(note=text, added=True, text=updated.strip())
