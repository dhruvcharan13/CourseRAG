"""PdfParser: registration + page/title extraction from committed fixtures."""

from __future__ import annotations

from pathlib import Path

from course_kb.parsing import ParsedDocument, get_parser_for
from course_kb.parsing.pdf import PdfParser

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
