"""ChunkRecord <-> dict round-trip."""

from __future__ import annotations

from course_kb.records import ChunkRecord


def _full_record() -> ChunkRecord:
    return ChunkRecord(
        id="CS240-W26::lec06.pdf::p12::c3",
        text="An AVL tree is a self-balancing binary search tree.",
        course="CS240-W26",
        source_file="lec06.pdf",
        category="notes",
        content_hash=ChunkRecord.hash_text("An AVL tree is a self-balancing binary search tree."),
        added_at="2026-07-22T00:00:00+00:00",
        vector=[0.1, -0.2, 0.3],
        title="Balanced BSTs",
        module="Trees",
        page=12,
        char_range=(0, 51),
    )


def test_round_trip_full_record():
    record = _full_record()
    assert ChunkRecord.from_dict(record.to_dict()) == record


def test_round_trip_minimal_record():
    record = ChunkRecord(
        id="c::f::p0::c0",
        text="hello",
        course="c",
        source_file="f",
        category="notes",
        content_hash=ChunkRecord.hash_text("hello"),
        added_at="2026-07-22T00:00:00+00:00",
    )
    # Optional fields default to None; the round-trip must preserve that.
    restored = ChunkRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.vector is None
    assert restored.char_range is None
    assert restored.page is None


def test_to_dict_flattens_char_range():
    record = _full_record()
    d = record.to_dict()
    assert d["char_start"] == 0
    assert d["char_end"] == 51
    assert "char_range" not in d


def test_hash_text_is_stable():
    assert ChunkRecord.hash_text("hello") == ChunkRecord.hash_text("hello")
    assert ChunkRecord.hash_text("hello") != ChunkRecord.hash_text("world")
