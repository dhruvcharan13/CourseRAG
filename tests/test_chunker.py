"""Chunker boundaries on hand-built ParsedDocuments (no PDF needed)."""

from __future__ import annotations

from course_kb.chunker import chunk_document
from course_kb.config import Config
from course_kb.parsing import ParsedDocument, ParsedElement

_KW = dict(course="C", source_file="f.pdf", category="notes", added_at="2026-01-01T00:00:00+00:00")


def _doc(elements: list[ParsedElement]) -> ParsedDocument:
    return ParsedDocument(source_file="f.pdf", elements=elements)


def test_slide_deck_is_one_chunk_per_page():
    doc = _doc(
        [
            ParsedElement("Balanced Trees\n\nkeeps height low", page=1, title="Balanced Trees"),
            ParsedElement("Rotations\n\nrestore the invariant", page=2, title="Rotations"),
            ParsedElement("Complexity\n\nO(log n) lookups", page=3, title="Complexity"),
        ]
    )
    recs = chunk_document(doc, cfg=Config(), **_KW)

    assert len(recs) == 3
    assert [r.id for r in recs] == ["C::f.pdf::p1::c0", "C::f.pdf::p2::c0", "C::f.pdf::p3::c0"]
    assert [r.page for r in recs] == [1, 2, 3]
    # The slide title is already the first line, so it is not duplicated.
    assert recs[0].text.startswith("Balanced Trees")
    assert recs[0].text.count("Balanced Trees") == 1
    for r in recs:
        assert r.title and r.page and r.content_hash
        assert r.char_range is not None


def test_prose_chunks_overlap_and_have_unique_ids():
    body = "1. Introduction\n" + ("This section introduces the topic in careful detail. " * 20)
    doc = _doc([ParsedElement(body, page=1, title="1. Introduction")])
    cfg = Config(chunk_size=300, min_chunk_chars=20)

    recs = chunk_document(doc, cfg=cfg, **_KW)

    assert len(recs) >= 3
    ids = [r.id for r in recs]
    assert ids == sorted(set(ids), key=ids.index)  # unique, in order
    assert ids[:2] == ["C::f.pdf::p1::c0", "C::f.pdf::p1::c1"]
    # Every chunk carries its section title.
    assert all(r.title == "1. Introduction" for r in recs)
    # Consecutive chunks overlap in the source, and the overlap sits in the target band.
    for a, b in zip(recs, recs[1:]):
        (a_start, a_end), (b_start, _b_end) = a.char_range, b.char_range
        overlap = a_end - b_start
        assert overlap > 0
        assert overlap <= 0.30 * cfg.chunk_size


def test_fenced_code_block_is_never_split():
    code = "```\n" + "\n".join(f"line_{i} = compute({i})" for i in range(30)) + "\n```"
    body = (
        "2. Algorithm\n"
        + ("Preamble discussion sentence here. " * 10)
        + "\n\n"
        + code
        + "\n\n"
        + ("Trailing discussion sentence here. " * 10)
    )
    doc = _doc([ParsedElement(body, page=2, title="2. Algorithm")])

    recs = chunk_document(doc, cfg=Config(chunk_size=200, min_chunk_chars=20), **_KW)

    holders = [r for r in recs if "```" in r.text]
    assert len(holders) == 1  # the whole block lands in exactly one chunk
    holder = holders[0]
    assert holder.text.count("```") == 2  # both fences present -> not split
    for i in range(30):
        assert f"line_{i} = compute({i})" in holder.text


def test_proof_block_is_never_split():
    proof = "Proof. Assume the height is h. " + ("We bound each level carefully. " * 12) + "∎"
    body = ("Lemma one holds. " * 12) + "\n\n" + proof + "\n\n" + ("Corollary follows. " * 12)
    doc = _doc([ParsedElement(body, page=1, title="Theorem")])

    recs = chunk_document(doc, cfg=Config(chunk_size=200, min_chunk_chars=20), **_KW)

    holders = [r for r in recs if "Proof." in r.text and "∎" in r.text]
    assert len(holders) == 1  # proof stays intact in a single chunk


def test_chunking_is_deterministic():
    doc = _doc([ParsedElement("1. Alpha\n" + ("word " * 300), page=1, title="1. Alpha")])
    cfg = Config(chunk_size=250)

    def fingerprint(records):
        return [(r.id, r.content_hash, r.char_range, r.text) for r in records]

    assert fingerprint(chunk_document(doc, cfg=cfg, **_KW)) == fingerprint(
        chunk_document(doc, cfg=cfg, **_KW)
    )


def test_empty_document_yields_no_chunks():
    doc = _doc([ParsedElement("   \n\n  ", page=1, title=None)])
    assert chunk_document(doc, cfg=Config(), **_KW) == []
