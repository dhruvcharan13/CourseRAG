"""PdfParser: registration + page/title extraction from committed fixtures."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from course_kb.chunker import nonspace_len
from course_kb.parsing import ParsedDocument, get_parser_for
from course_kb.parsing.pdf import _FOLD, _PAGE_NUMBER, PdfParser, _is_two_column

FIXTURES = Path(__file__).parent / "fixtures"


def test_pdf_extension_resolves_to_pdf_parser():
    assert isinstance(get_parser_for(Path("lec06.pdf")), PdfParser)
    assert isinstance(get_parser_for(Path("LEC06.PDF")), PdfParser)  # case-insensitive


def test_slides_pdf_page_count_and_titles():
    path = FIXTURES / "slides.pdf"
    doc = get_parser_for(path).parse(path)

    assert isinstance(doc, ParsedDocument)
    assert doc.source_file == str(path)
    assert [e.page for e in doc.elements] == [1, 2, 3]
    assert [e.title for e in doc.elements] == [
        "Balanced Search Trees",
        "AVL Rotations",
        "Amortized Analysis",
    ]
    assert all(e.text.strip() for e in doc.elements)


def test_prose_pdf_page_count_and_titles():
    path = FIXTURES / "prose.pdf"
    doc = get_parser_for(path).parse(path)

    assert [e.page for e in doc.elements] == [1, 2]
    assert doc.elements[0].title == "1. Introduction"
    assert doc.elements[1].title == "2. Rotations and Invariants"
    # Prose pages carry substantially more text than a slide.
    assert all(len(e.text.split()) > 100 for e in doc.elements)


def test_running_headers_and_page_numbers_are_stripped():
    path = FIXTURES / "notes.pdf"
    doc = get_parser_for(path).parse(path)

    full = "\n".join(e.text for e in doc.elements)
    assert "MATH 239 COURSE NOTES" not in full  # header repeated on every page -> dropped
    assert "Page 2 of 3" not in full  # "Page n of m" footer dropped even in large font
    # A numbered heading is detected by its font-size jump; a heading-less page gets no
    # title (its only large line, the page-number footer, is never chosen as a title).
    assert doc.elements[0].title == "4.1 Definitions"
    assert doc.elements[1].title is None
    assert not any(e.title and re.match(r"Page \d+ of \d+", e.title) for e in doc.elements)


def test_page_number_pattern_matches_footers_not_headings():
    for hit in ("5", "42", "5 / 44", "iv", "Page 2", "Page 2 of 28", "PAGE 12 OF 44"):
        assert _PAGE_NUMBER.match(hit), hit
    for miss in ("Page Layout", "Section 2", "5 pages", "Chapter 4"):
        assert not _PAGE_NUMBER.match(miss), miss


def test_scanned_pages_are_near_empty():
    path = FIXTURES / "scanned.pdf"
    doc = get_parser_for(path).parse(path)

    assert len(doc.elements) == 2  # image-only page + stray-number page
    assert all(nonspace_len(e.text) < 10 for e in doc.elements)


def test_two_column_reading_order_is_column_major():
    path = FIXTURES / "twocol.pdf"
    doc = get_parser_for(path).parse(path)

    text = doc.elements[0].text
    # The whole left column must precede the whole right column (no interleaving).
    assert text.index("Alpha8") < text.index("Beta1")


def test_single_column_page_keeps_top_down_order():
    # A near-full-width title whose bbox centre lands right of the page midpoint must
    # NOT be treated as a second column and shuffled to the bottom of the page.
    path = FIXTURES / "onecol_wide.pdf"
    doc = get_parser_for(path).parse(path)

    text = doc.elements[0].text
    assert text.index("Development Process Overview") < text.index("The body text")
    assert not _is_two_column(_text_blocks(path), 595.0 / 2)  # page is single-column


def test_two_column_detection_requires_a_clean_gutter():
    mid = 100.0
    straddling = [{"bbox": (10.0, 0.0, 190.0, 20.0)}, {"bbox": (10.0, 30.0, 80.0, 50.0)}]
    clean = [{"bbox": (10.0, 0.0, 90.0, 200.0)}, {"bbox": (110.0, 0.0, 190.0, 200.0)}]
    one_sided = [{"bbox": (10.0, 0.0, 90.0, 20.0)}, {"bbox": (10.0, 30.0, 90.0, 50.0)}]

    assert not _is_two_column(straddling, mid)  # a full-width block => single column
    assert _is_two_column(clean, mid)  # two blocks, clean gutter => two columns
    assert not _is_two_column(one_sided, mid)  # everything in one column
    assert not _is_two_column(clean[:1], mid)  # a lone block is never two columns


def test_ligatures_and_curly_quotes_are_folded_to_ascii():
    folded = "deﬁnition ﬂow eﬀect ﬃ ﬄ “quoted” it’s".translate(_FOLD)

    assert folded == "definition flow effect ffi ffl \"quoted\" it's"
    # Math and dashes are deliberately preserved — NFKC would rewrite some of them.
    assert "∑ ∏ ∫ µ — –".translate(_FOLD) == "∑ ∏ ∫ µ — –"


def _text_blocks(path: Path) -> list[dict]:
    with fitz.open(path) as doc:
        return [b for b in doc[0].get_text("dict")["blocks"] if b.get("type", 0) == 0]
