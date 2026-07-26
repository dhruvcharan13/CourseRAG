"""PDF parser backed by PyMuPDF (``fitz``).

Emits one :class:`ParsedElement` per page. Page structure detection (headings,
sections, slide-vs-prose) is intentionally NOT done here — that is the chunker's
job, which keeps the chunker a pure function of a :class:`ParsedDocument`. This
parser's contributions are pragmatic clean-ups over PyMuPDF's raw extraction:

* reading order: two-column pages are sorted column-major, everything else top-down;
* a per-page ``title``, detected from a genuine font-size jump over body text;
* de-boilerplating: running headers/footers (lines repeated across many pages)
  and standalone page-number lines are dropped so they don't pollute chunks;
* a narrow character fold: typographic ligatures and curly quotes become ASCII.

None of this is a layout analyzer — they are simple position/frequency/size
heuristics, each chosen because it measurably helped on real course PDFs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from course_kb.parsing import ParsedDocument, ParsedElement

# A line is a title only if its font is at least this many points larger than the
# document's body text. (Boldness alone is not enough: prose emphasises terms
# in-line with bold, which would otherwise flag ordinary body lines.)
_HEADING_SIZE_DELTA = 1.5
# A title is a short line, not a run-on sentence.
_MAX_TITLE_CHARS = 120
_MAX_TITLE_WORDS = 14
# PyMuPDF span flag bit for bold (2**4).
_FLAG_BOLD = 1 << 4

# De-boilerplating: a short line repeated verbatim on at least this many pages is
# treated as a running header/footer and dropped. Long lines are never dropped
# (so a genuinely repeated body sentence survives).
_BOILERPLATE_MIN_PAGES = 3
_MAX_BOILERPLATE_CHARS = 90
# A standalone page-number line is always dropped: a bare number/roman numeral,
# "n / m", or "Page n [of m]" (the last one otherwise gets grabbed as a title).
_PAGE_NUMBER = re.compile(
    r"^(?:page\s+\d{1,4}(?:\s+of\s+\d{1,4})?|\d{1,4}(?:\s*/\s*\d{1,4})?|[ivxlcdm]{1,7})$",
    re.IGNORECASE,
)

_BLOCK = "\x00BLOCK\x00"  # block-separator sentinel in a line list

# PDF extraction preserves typographic ligatures verbatim, so a typeset "definition"
# comes back as "deﬁnition" (U+FB01) and never matches a query typed in plain ASCII —
# 270 occurrences in one set of course notes. Fold the Latin ligature block and curly
# quotes only. Full NFKC would also do this, but it rewrites µ -> μ and ¯ -> a combining
# macron, so math and accents in course notes are left exactly as the author set them.
_FOLD = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


class PdfParser:
    """Extracts one whole-page element per PDF page, with a detected title."""

    extensions: tuple[str, ...] = (".pdf",)

    def parse(self, path: Path) -> ParsedDocument:
        with fitz.open(path) as doc:
            per_page_lines = [_page_lines(page.get_text("dict")) for page in doc]

        body_size = _body_font_size(per_page_lines)
        boilerplate = _boilerplate_lines(per_page_lines)

        elements: list[ParsedElement] = []
        for pno, lines in enumerate(per_page_lines, start=1):
            kept = [ln for ln in lines if not _is_boilerplate(ln[0], boilerplate)]
            text = _reconstruct_text(kept)
            title = _detect_title(kept, body_size)
            elements.append(ParsedElement(text=text, page=pno, title=title))

        return ParsedDocument(source_file=str(path), elements=elements)


# --------------------------------------------------------------------------- #
# Helpers (module-level, pure) — a "line" is (text, max_span_size, is_bold).
# --------------------------------------------------------------------------- #


def _sorted_blocks(page_dict: dict) -> list[dict]:
    """Text blocks in reading order: top-down, or column-major on two-column pages.

    PyMuPDF's raw block order can interleave the columns of a two-column layout, so
    those pages are sorted by (column, y). Every other page keeps plain top-to-bottom
    order — see :func:`_is_two_column` for why that fallback has to be the default.
    """
    blocks = [b for b in page_dict.get("blocks", []) if b.get("type", 0) == 0]
    mid = float(page_dict.get("width") or 0.0) / 2
    if mid and _is_two_column(blocks, mid):
        return sorted(blocks, key=lambda b: (_column(b, mid), round(_bbox(b)[1])))
    return sorted(blocks, key=lambda b: round(_bbox(b)[1]))


def _bbox(block: dict) -> tuple[float, float, float, float]:
    return block.get("bbox", (0.0, 0.0, 0.0, 0.0))


def _column(block: dict, mid: float) -> int:
    """0 if the block's horizontal centre sits left of ``mid``, else 1."""
    x0, _y0, x1, _y1 = _bbox(block)
    return 0 if (x0 + x1) / 2 < mid else 1


def _is_two_column(blocks: list[dict], mid: float) -> bool:
    """True only when blocks separate cleanly into two columns across ``mid``.

    A single block straddling the midpoint means the page is single-column however
    its bbox centre happens to land, and column-sorting would then move that block to
    the end of the page. Real decks are full of ~95%-width blocks whose centres fall
    either side of the fold by a fraction of a point, so an unguarded (column, y) sort
    reorders single-column pages far more often than it fixes two-column ones.
    """
    if len(blocks) < 2:
        return False
    if any(x0 < mid < x1 for x0, _y0, x1, _y1 in map(_bbox, blocks)):
        return False
    return len({_column(b, mid) for b in blocks}) == 2


def _page_lines(page_dict: dict) -> list[tuple[str, float, bool]]:
    """Flatten a page's text blocks into reading-order lines with font info."""
    out: list[tuple[str, float, bool]] = []
    for block in _sorted_blocks(page_dict):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # Fold here, the single point where text enters the pipeline, so the string
            # char_range indexes is the same one every later stage sees.
            text = "".join(span.get("text", "") for span in spans).translate(_FOLD).rstrip()
            max_size = max((span.get("size", 0.0) for span in spans), default=0.0)
            is_bold = any(span.get("flags", 0) & _FLAG_BOLD for span in spans)
            out.append((text, round(max_size * 2) / 2, is_bold))
        out.append((_BLOCK, 0.0, False))  # block separator sentinel
    return out


def _reconstruct_text(lines: list[tuple[str, float, bool]]) -> str:
    """Join lines in reading order: lines with ``\\n``, blocks with a blank line."""
    parts: list[str] = []
    block: list[str] = []
    for text, _size, _bold in lines:
        if text == _BLOCK:
            if block:
                parts.append("\n".join(block))
                block = []
            continue
        block.append(text)
    if block:
        parts.append("\n".join(block))
    return "\n\n".join(p for p in parts if p.strip())


def _body_font_size(per_page_lines: list[list[tuple[str, float, bool]]]) -> float:
    """Most common rounded span size across the document (the body text size)."""
    weighted: Counter[float] = Counter()
    for lines in per_page_lines:
        for text, size, _bold in lines:
            if text == _BLOCK or size <= 0:
                continue
            weighted[size] += 1
    if not weighted:
        return 12.0
    return weighted.most_common(1)[0][0]


def _boilerplate_lines(per_page_lines: list[list[tuple[str, float, bool]]]) -> set[str]:
    """Normalized short lines that recur on many pages (running headers/footers)."""
    page_counts: Counter[str] = Counter()
    for lines in per_page_lines:
        seen: set[str] = set()
        for text, _size, _bold in lines:
            if text == _BLOCK:
                continue
            norm = " ".join(text.split())
            if norm and len(norm) <= _MAX_BOILERPLATE_CHARS:
                seen.add(norm)
        page_counts.update(seen)
    return {norm for norm, count in page_counts.items() if count >= _BOILERPLATE_MIN_PAGES}


def _is_boilerplate(text: str, boilerplate: set[str]) -> bool:
    if text == _BLOCK:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if _PAGE_NUMBER.match(stripped):
        return True
    return " ".join(stripped.split()) in boilerplate


def _detect_title(lines: list[tuple[str, float, bool]], body_size: float) -> str | None:
    """First short reading-order line whose font is clearly larger than body text."""
    for text, size, _is_bold in lines:
        if text == _BLOCK or not text.strip():
            continue
        stripped = text.strip()
        if size >= body_size + _HEADING_SIZE_DELTA and _looks_like_title(stripped):
            return stripped
    return None


def _looks_like_title(text: str) -> bool:
    return 0 < len(text) <= _MAX_TITLE_CHARS and len(text.split()) <= _MAX_TITLE_WORDS
