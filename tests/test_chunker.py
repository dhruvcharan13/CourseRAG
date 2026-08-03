"""Chunker boundaries on hand-built ParsedDocuments (no PDF needed)."""

from __future__ import annotations

from courserag.chunker import (
    chunk_document,
    estimate_tokens,
    is_slide_deck,
    keep_elements,
    nonspace_len,
)
from courserag.config import Config
from courserag.parsing import ParsedDocument, ParsedElement

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


def test_short_title_case_lines_are_not_headings():
    # Figure labels / street names must NOT each become their own micro-section.
    body = (
        "2.1 Streets\n"
        + ("The city map induces a graph on its intersections and connections. " * 8)
        + "\nMain Street\nKing Street\nQueen Street\n"
        + ("Each street segment becomes an edge between two intersection vertices. " * 8)
    )
    doc = _doc([ParsedElement(body, page=1, title="2.1 Streets")])

    recs = chunk_document(doc, cfg=Config(chunk_size=400, min_chunk_chars=20), **_KW)

    assert {r.title for r in recs} == {"2.1 Streets"}  # only the numbered heading titles sections
    labels = {"Main Street", "King Street", "Queen Street"}
    assert not any(r.text.strip() in labels for r in recs)  # a label is never its own chunk


def test_numbered_headings_still_split_sections():
    body = (
        "1. First\n" + ("Alpha discussion sentence here. " * 15)
        + "\n2. Second\n" + ("Beta discussion sentence here. " * 15)
    )
    doc = _doc([ParsedElement(body, page=1, title="1. First")])

    titles = [r.title for r in chunk_document(doc, cfg=Config(chunk_size=1000), **_KW)]

    assert "1. First" in titles and "2. Second" in titles


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


def test_near_empty_elements_are_dropped():
    doc = _doc(
        [
            ParsedElement("", page=1, title=None),  # scanned / image-only page
            ParsedElement("42", page=2, title=None),  # stray page-number leftover
            ParsedElement("A real slide with plenty of words to keep.", page=3, title="Real"),
        ]
    )
    recs = chunk_document(doc, cfg=Config(), **_KW)  # default min_element_chars = 10

    assert [r.page for r in recs] == [3]  # only the real page survives
    assert all(nonspace_len(r.text) >= 10 for r in recs)


def test_char_range_indexes_body_not_prepended_text():
    element = ParsedElement(
        "1. Intro\n" + ("This section explains the idea in careful detail. " * 20),
        page=1,
        title="1. Intro",
    )
    recs = chunk_document(_doc([element]), cfg=Config(chunk_size=300, min_chunk_chars=20), **_KW)

    assert len(recs) >= 3  # includes title-prepended continuation chunks
    for r in recs:
        start, end = r.char_range
        body = element.text[start:end]
        assert body and body == body.strip()  # range is the exact, tight body slice
        assert r.text.endswith(body)  # stored text = optional title + "\n\n" + body


def test_large_atomic_block_is_one_oversize_chunk():
    code = "```\n" + "\n".join(f"row_{i} = compute({i})" for i in range(200)) + "\n```"
    recs = chunk_document(_doc([ParsedElement(code, page=1, title="Listing")]),
                          cfg=Config(chunk_size=1000), **_KW)

    code_chunks = [r for r in recs if "```" in r.text]
    assert len(code_chunks) == 1
    big = code_chunks[0]
    assert len(big.text) > 1000  # overshoots target rather than being split or truncated
    assert big.text.count("```") == 2
    assert estimate_tokens(big.text) > 256  # would be flagged oversize


def test_mixed_mode_deck_is_a_known_limitation():
    """Characterization test: pins TODAY's behaviour on a sparse-deck-plus-dense-appendix.

    Mode is chosen once per document from the MEDIAN words/page, so 40 sparse slides
    outvote a 5-page dense appendix and the whole file is chunked as slides — each dense
    appendix page collapsing into one oversize chunk. Per-page mode selection is the
    eventual fix; when it lands, this test should fail and be rewritten.
    """
    dense_para = "The appendix restates every result in full detail with all hypotheses. " * 30
    elements = [
        ParsedElement(f"Topic {i}\n\nA single sparse bullet", page=i, title=f"Topic {i}")
        for i in range(1, 41)
    ] + [
        ParsedElement(f"A.{i} Appendix\n\n{dense_para}", page=40 + i, title=f"A.{i} Appendix")
        for i in range(1, 6)
    ]
    cfg = Config()

    assert is_slide_deck(keep_elements(elements, cfg))  # the sparse majority decides
    recs = chunk_document(_doc(elements), cfg=cfg, **_KW)

    assert len(recs) == 45  # one chunk per page, dense pages included
    dense = [r for r in recs if r.page > 40]
    assert len(dense) == 5  # each dense page is a SINGLE chunk, not windowed
    assert all(estimate_tokens(r.text) > cfg.warn_chunk_tokens for r in dense)  # all oversize
    assert max(len(r.text) for r in dense) > 2 * cfg.chunk_size  # ~2.1k chars vs a 1k target


def test_nonspace_len_and_estimate_tokens():
    assert nonspace_len("  a b\tc\n") == 3
    assert nonspace_len("") == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # ceil(5 / 4)
