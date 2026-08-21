"""Rendering a page of a source document as an image.

This exists to cover the one thing text retrieval structurally cannot: a page whose
content is drawn. The tests build real PDFs with pymupdf rather than mocking it, since
the interesting behaviour — page bounds, resolution clamping, what counts as a drawing
— is all in the library's actual output.
"""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")

from courserag.cli import main  # noqa: E402
from courserag.config import load_config  # noqa: E402
from courserag.rendering import (  # noqa: E402
    MAX_DPI,
    MAX_PIXELS,
    MIN_DPI,
    RenderError,
    page_stats,
    render_page,
    resolve_source,
)
from courserag.sync import apply_sync, source_root  # noqa: E402


def make_pdf(path, pages=3, draw=False, size=(612, 792)):
    """A small PDF: text on every page, optionally with vector drawing too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_text((72, 72), f"Page {n + 1} about skip lists and tower height.")
        if draw:
            for i in range(20):
                page.draw_circle(pymupdf.Point(100 + i * 12, 300), 8)
                page.draw_line(pymupdf.Point(100, 300), pymupdf.Point(100 + i * 12, 400))
    doc.save(path)
    doc.close()


@pytest.fixture
def course(tmp_path, monkeypatch, dummy_config):
    monkeypatch.chdir(tmp_path)
    assert main(["init-course", "C"]) == 0
    cfg = load_config()
    make_pdf(source_root(cfg, "C") / "lecture" / "deck.pdf", pages=3, draw=True)
    (source_root(cfg, "C") / "notes.txt").write_text("Plain text, no pages.\n", encoding="utf-8")
    apply_sync(cfg, "C")
    return cfg


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def test_a_page_renders_to_a_png(course):
    result = render_page(course, "C", "deck.pdf", 1)

    assert result.png.startswith(b"\x89PNG\r\n")
    assert result.page == 1 and result.page_count == 3
    assert result.width > 0 and result.height > 0


def test_pages_are_numbered_from_one_to_match_citations(course):
    """Search cites "deck.pdf p2"; that must be the page you get back."""
    first = render_page(course, "C", "deck.pdf", 1)
    second = render_page(course, "C", "deck.pdf", 2)

    assert first.png != second.png
    # Page 1 is the first page, not the second — an off-by-one here would be invisible.
    with pymupdf.open(source_root(course, "C") / "lecture" / "deck.pdf") as doc:
        assert doc[0].get_text().strip().startswith("Page 1")
    assert "Page 1" in _text_of(course, 1)
    assert "Page 2" in _text_of(course, 2)


def _text_of(cfg, page):
    with pymupdf.open(source_root(cfg, "C") / "lecture" / "deck.pdf") as doc:
        return doc[page - 1].get_text()


@pytest.mark.parametrize("page", [0, -1, 4, 999])
def test_a_page_out_of_range_is_refused_with_the_real_count(course, page):
    with pytest.raises(RenderError, match="3 pages"):
        render_page(course, "C", "deck.pdf", page)


def test_a_non_pdf_says_so_rather_than_failing_obscurely(course):
    with pytest.raises(RenderError, match="not a PDF"):
        render_page(course, "C", "notes.txt", 1)


def test_an_unknown_file_lists_what_the_course_has(course):
    with pytest.raises(RenderError, match="deck.pdf"):
        render_page(course, "C", "ghost.pdf", 1)


def test_a_citation_that_lost_its_capitalization_still_resolves(course):
    """A model retyping "DECK.pdf" from a citation should not hit a dead end."""
    assert render_page(course, "C", "DECK.pdf", 1).source_file == "deck.pdf"


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_higher_dpi_produces_a_bigger_image(course):
    low = render_page(course, "C", "deck.pdf", 1, dpi=72)
    high = render_page(course, "C", "deck.pdf", 1, dpi=200)

    assert high.width > low.width and high.height > low.height


@pytest.mark.parametrize("asked,expected", [(1, MIN_DPI), (10_000, MAX_DPI)])
def test_dpi_is_clamped_rather_than_rejected(course, asked, expected):
    # A small page, so the clamp is what is being measured and not the pixel ceiling —
    # a letter-sized page at MAX_DPI is 8.4M pixels and gets scaled back down.
    make_pdf(source_root(course, "C") / "small.pdf", pages=1, size=(300, 300))
    apply_sync(course, "C")

    assert render_page(course, "C", "small.pdf", 1, dpi=asked).dpi == expected


def test_an_enormous_page_is_scaled_down_instead_of_exhausting_memory(course, tmp_path):
    """A poster-sized page at 300dpi would otherwise be hundreds of megabytes."""
    make_pdf(source_root(course, "C") / "poster.pdf", pages=1, size=(5000, 5000))
    apply_sync(course, "C")

    result = render_page(course, "C", "poster.pdf", 1, dpi=MAX_DPI)

    assert result.width * result.height <= MAX_PIXELS
    assert result.dpi < MAX_DPI  # it was scaled, not refused


# --------------------------------------------------------------------------- #
# What is on the page
# --------------------------------------------------------------------------- #


def test_a_drawn_page_reports_its_drawings(course):
    result = render_page(course, "C", "deck.pdf", 1)

    assert result.drawings > 0
    assert result.text_chars > 0
    assert "vector drawings" in result.summary()


def test_page_stats_covers_every_page_without_rendering(course):
    stats = page_stats(course, "C", "deck.pdf")

    assert [s.page for s in stats] == [1, 2, 3]
    assert all(s.drawings > 0 for s in stats)
    assert all(s.spans >= 1 for s in stats)


def test_page_stats_on_a_non_pdf_is_empty_rather_than_an_error(course):
    assert page_stats(course, "C", "notes.txt") == []


# --------------------------------------------------------------------------- #
# Reach
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "attack",
    ["../../../../etc/passwd", "/etc/passwd", "manifest.json", "../manifest.json", "sync.json"],
)
def test_rendering_cannot_reach_outside_the_course_folder(course, attack):
    """Names resolve through the course's own scan, so a path simply matches nothing."""
    with pytest.raises(RenderError):
        resolve_source(course, "C", attack)


def test_resolution_is_limited_to_files_sync_found(course):
    """A file sitting beside raw/ is not part of the course and must not be readable."""
    stray = course.courses_dir / "C" / "secret.pdf"
    make_pdf(stray, pages=1)

    with pytest.raises(RenderError):
        resolve_source(course, "C", "secret.pdf")


# --------------------------------------------------------------------------- #
# Through the CLI
# --------------------------------------------------------------------------- #


def test_kb_page_writes_a_png(course, tmp_path, capsys):
    assert main(["page", "C", "deck.pdf", "2", "-o", str(tmp_path / "out.png")]) == 0

    assert (tmp_path / "out.png").read_bytes().startswith(b"\x89PNG")
    assert "page 2 of 3" in capsys.readouterr().out


def test_kb_page_names_the_file_itself_by_default(course, tmp_path):
    assert main(["page", "C", "deck.pdf", "2"]) == 0
    assert (tmp_path / "deck-p2.png").is_file()


def test_kb_page_overview_lists_every_page(course, capsys):
    assert main(["page", "C", "deck.pdf", "--overview"]) == 0

    out = capsys.readouterr().out
    assert "3 pages" in out
    assert out.count("\n") >= 5


def test_kb_page_without_a_page_number_explains_itself(course, capsys):
    assert main(["page", "C", "deck.pdf"]) == 1
    assert "--overview" in capsys.readouterr().err


def test_kb_page_on_an_unknown_course_exits_nonzero(course, capsys):
    assert main(["page", "NOPE", "deck.pdf", "1"]) == 1
    assert "not initialized" in capsys.readouterr().err


def test_kb_page_out_of_range_exits_nonzero(course, capsys):
    assert main(["page", "C", "deck.pdf", "99"]) == 1
    assert "out of range" in capsys.readouterr().err
