"""PDF parser backed by PyMuPDF (``fitz``).

Emits one :class:`ParsedElement` per page. Page structure detection (headings,
sections, slide-vs-prose) is intentionally NOT done here — that is the chunker's
job, which keeps the chunker a pure function of a :class:`ParsedDocument`. This
parser's only structural contribution is a per-page ``title``, detected with a
pragmatic font-size / boldness heuristic over PyMuPDF's span info.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from course_kb.parsing import ParsedDocument, ParsedElement

# A line is treated as a heading if its font is at least this many points larger
# than the document's body text, or if it is bold and no smaller than body text.
_HEADING_SIZE_DELTA = 1.5
# PyMuPDF span flag bit for bold (2**4).
_FLAG_BOLD = 1 << 4
# A heading title is a short line, not a run-on sentence.
_MAX_TITLE_CHARS = 120
_MAX_TITLE_WORDS = 14


class PdfParser:
    """Extracts one whole-page element per PDF page, with a detected title."""

    extensions: tuple[str, ...] = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        with fitz.open(path) as doc:
            pages = [page.get_text("dict") for page in doc]

        body_size = _body_font_size(pages)
        elements: list[ParsedElement] = []
        for pno, page_dict in enumerate(pages, start=1):
            lines = _page_lines(page_dict)
            text = _reconstruct_text(lines)
            title = _detect_title(lines, body_size)
            elements.append(ParsedElement(text=text, page=pno, title=title))

        return ParsedDocument(source_file=str(path), elements=elements)


# --------------------------------------------------------------------------- #
# Helpers (module-level, pure) — a "line" is (text, max_span_size, is_bold).
# --------------------------------------------------------------------------- #


def _page_lines(page_dict: dict) -> list[tuple[str, float, bool]]:
    """Flatten a page's text blocks into reading-order lines with font info."""
    out: list[tuple[str, float, bool]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type", 0) != 0:  # skip image / non-text blocks
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(span.get("text", "") for span in spans).rstrip()
            max_size = max((span.get("size", 0.0) for span in spans), default=0.0)
            is_bold = any(span.get("flags", 0) & _FLAG_BOLD for span in spans)
            out.append((text, round(max_size * 2) / 2, is_bold))
        out.append(("\x00BLOCK\x00", 0.0, False))  # block separator sentinel
    return out


def _reconstruct_text(lines: list[tuple[str, float, bool]]) -> str:
    """Join lines in reading order: lines with ``\\n``, blocks with a blank line."""
    parts: list[str] = []
    block: list[str] = []
    for text, _size, _bold in lines:
        if text == "\x00BLOCK\x00":
            if block:
                parts.append("\n".join(block))
                block = []
            continue
        block.append(text)
    if block:
        parts.append("\n".join(block))
    return "\n\n".join(p for p in parts if p.strip())


def _body_font_size(pages: list[dict]) -> float:
    """Most common (char-weighted) rounded span size across the document."""
    weighted: Counter[float] = Counter()
    for page_dict in pages:
        for _text, size, _bold in _page_lines(page_dict):
            if size <= 0:
                continue
            weighted[size] += 1
    if not weighted:
        return 12.0
    return weighted.most_common(1)[0][0]


def _detect_title(lines: list[tuple[str, float, bool]], body_size: float) -> str | None:
    """First short reading-order line that stands out by size or boldness."""
    for text, size, is_bold in lines:
        if text == "\x00BLOCK\x00" or not text.strip():
            continue
        stripped = text.strip()
        is_big = size >= body_size + _HEADING_SIZE_DELTA
        is_bold_heading = is_bold and size >= body_size
        if (is_big or is_bold_heading) and _looks_like_title(stripped):
            return stripped
    return None


def _looks_like_title(text: str) -> bool:
    return 0 < len(text) <= _MAX_TITLE_CHARS and len(text.split()) <= _MAX_TITLE_WORDS
