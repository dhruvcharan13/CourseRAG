"""Structure-aware, rule-based chunking of a :class:`ParsedDocument`.

Pure and deterministic — no I/O, no embeddings, no network. Two document shapes
are handled:

* **Slide decks** (little text per page): one chunk per page — the natural unit.
* **Prose / notes**: split each page into heading-delimited sections, then pack
  each section into ``chunk_size``-char windows with ~12% overlap so a definition
  straddling a window boundary is not orphaned. Fenced code and proof/math blocks
  are never split mid-block, even if that overshoots the target.

Every chunk carries its section/slide ``title`` (prepended to the stored text),
its ``page``, a ``char_range`` into the source page text, a ``content_hash``, and
a deterministic id ``course::file::pNN::cK``.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from courserag.config import Config
from courserag.parsing import ParsedDocument, ParsedElement
from courserag.records import ChunkRecord

# A page whose median word count is at or below this is treated as a slide.
SLIDE_WORDS_PER_PAGE = 100
# Prose overlap as a fraction of chunk_size (the approved ~10-15% target).
OVERLAP_RATIO = 0.12


def nonspace_len(text: str) -> int:
    """Number of non-whitespace characters in ``text``."""
    return len("".join((text or "").split()))


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for token-budget warnings."""
    return (len(text) + 3) // 4


def keep_elements(elements: list[ParsedElement], cfg: Config) -> list[ParsedElement]:
    """Elements with enough real text to be worth chunking.

    A scanned or image-only page extracts to "" or a few stray characters, which would
    otherwise become a junk chunk and embed to a meaningless vector. Shared with the
    ``--report`` gate so the report can't drift from what the chunker actually keeps.
    """
    return [e for e in elements if nonspace_len(e.text or "") >= cfg.min_element_chars]


# Spans that must never be split mid-block. (pattern, flags) pairs.
_PROTECTED_PATTERNS: list[tuple[str, int]] = [
    (r"```.*?```", re.DOTALL),  # fenced code
    (r"\\begin\{proof\}.*?\\end\{proof\}", re.DOTALL),  # LaTeX proof env
    (r"^[ \t]*Proof[.:].*?(?:∎|□|QED|\\qed|q\.e\.d\.)", re.DOTALL | re.MULTILINE),
    (r"\$\$.*?\$\$", re.DOTALL),  # display math
    (r"\\\[.*?\\\]", re.DOTALL),  # display math (bracket form)
]

_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")
# A heading is a short label. This length cap distinguishes a section heading
# ("4.1 Definitions") from a numbered list/problem item ("1. For the graphs ...")
# and from a table-of-contents leader line ("4.10 Bridges . . . . 121").
_MAX_HEADING_CHARS = 80
_MAX_HEADING_WORDS = 12
# A sentence end needs a letter before the punctuation so enumerators ("2.") and
# decimals ("3.14") are not mistaken for sentence boundaries.
_SENTENCE_END = re.compile(r"(?<=[A-Za-z])[.!?][)\"']?\s")


@dataclass
class _Section:
    """A contiguous span of one page's text, with a title."""

    source: str  # the full page text this section is carved from
    start: int
    end: int
    page: int | None
    title: str | None


def chunk_document(
    doc: ParsedDocument,
    *,
    course: str,
    source_file: str,
    category: str,
    cfg: Config,
    added_at: str,
) -> list[ChunkRecord]:
    """Turn a parsed document into deterministic, fully-populated chunk records."""
    elements = keep_elements(doc.elements, cfg)
    if not elements:
        return []

    if is_slide_deck(elements):
        sections = [
            _Section(source=e.text, start=0, end=len(e.text), page=e.page, title=e.title)
            for e in elements
        ]
        windows_per_section = [[(0, len(e.text))] for e in elements]
    else:
        sections = [sec for e in elements for sec in _split_sections(e)]
        windows_per_section = [_window(sec.source[sec.start : sec.end], cfg) for sec in sections]

    records: list[ChunkRecord] = []
    per_page_counter: dict[int | None, int] = {}
    for sec, windows in zip(sections, windows_per_section):
        for ws, we in windows:
            body, char_range = _tighten(sec.source, sec.start + ws, sec.start + we)
            if not body.strip():
                continue
            final_text = _prepend_title(sec.title, body)
            k = per_page_counter.get(sec.page, 0)
            per_page_counter[sec.page] = k + 1
            page_part = f"p{sec.page}" if sec.page is not None else "p0"
            records.append(
                ChunkRecord(
                    id=f"{course}::{source_file}::{page_part}::c{k}",
                    text=final_text,
                    course=course,
                    source_file=source_file,
                    category=category,
                    content_hash=ChunkRecord.hash_text(final_text),
                    added_at=added_at,
                    title=sec.title,
                    module=None,
                    page=sec.page,
                    char_range=char_range,
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Mode decision
# --------------------------------------------------------------------------- #


def is_slide_deck(elements) -> bool:
    """True when the document reads as a slide deck (little text per page)."""
    words_per_page = [len((e.text or "").split()) for e in elements]
    return statistics.median(words_per_page) <= SLIDE_WORDS_PER_PAGE


# --------------------------------------------------------------------------- #
# Prose: page -> sections
# --------------------------------------------------------------------------- #


def _split_sections(element) -> list[_Section]:
    """Carve a page's text into heading-delimited sections (contiguous spans)."""
    text = element.text
    boundaries: list[tuple[int, str]] = []  # (offset, heading text)
    offset = 0
    for line in text.split("\n"):
        if _is_heading(line):
            boundaries.append((offset, line.strip()))
        offset += len(line) + 1  # +1 for the split "\n"

    sections: list[_Section] = []
    # Leading content before the first heading keeps the page's detected title.
    lead_end = boundaries[0][0] if boundaries else len(text)
    if text[:lead_end].strip():
        sections.append(_Section(text, 0, lead_end, element.page, element.title))
    for i, (off, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        if text[off:end].strip():
            sections.append(_Section(text, off, end, element.page, title))
    return sections


def _is_heading(line: str) -> bool:
    """A heading is a numbered (``1.2 Foo``) or markdown (``## Foo``) line.

    Free-form Title-Case / ALL-CAPS detection was deliberately dropped: on real
    PDFs it fired on figure labels ("Main St") and running headers, sharding pages
    into micro-sections. Un-numbered headings are instead surfaced via the parser's
    font-based per-page title, which seeds the leading section.
    """
    s = line.strip()
    if not s or len(s) > _MAX_HEADING_CHARS or len(s.split()) > _MAX_HEADING_WORDS:
        return False
    return bool(_MARKDOWN_HEADING.match(s) or _NUMBERED_HEADING.match(s))


# --------------------------------------------------------------------------- #
# Prose: section -> windows (with overlap, protecting code/proof blocks)
# --------------------------------------------------------------------------- #


def _window(sub: str, cfg: Config) -> list[tuple[int, int]]:
    """Split ``sub`` into windows of ~``chunk_size`` chars.

    Protected blocks (code / proof / math) are emitted as standalone windows so
    they are never split; the plain-text runs between them are windowed with
    ~12% overlap so a definition straddling a boundary is not orphaned.
    """
    n = len(sub)
    if n == 0:
        return []
    windows: list[tuple[int, int]] = []
    for start, end, is_protected in _atomize(n, _protected_spans(sub)):
        if is_protected:
            windows.append((start, end))
        else:
            windows.extend(_window_plain(sub, start, end, cfg))
    return windows


def _atomize(n: int, protected: list[tuple[int, int]]) -> list[tuple[int, int, bool]]:
    """Cover ``[0, n)`` in order as alternating plain / protected segments."""
    segments: list[tuple[int, int, bool]] = []
    cursor = 0
    for s, e in protected:
        if s > cursor:
            segments.append((cursor, s, False))
        segments.append((s, e, True))
        cursor = e
    if cursor < n:
        segments.append((cursor, n, False))
    return segments


def _window_plain(sub: str, lo: int, hi: int, cfg: Config) -> list[tuple[int, int]]:
    """Window a protected-block-free span ``[lo, hi)`` with overlap."""
    target = max(1, cfg.chunk_size)
    overlap = max(1, round(OVERLAP_RATIO * target))
    windows: list[tuple[int, int]] = []
    pos = lo
    while pos < hi:
        if hi - pos <= target:
            windows.append((pos, hi))
            break
        end = _natural_boundary(sub, pos, min(pos + target, hi))
        if end <= pos:
            end = min(pos + target, hi)
        windows.append((pos, end))
        start = end - overlap
        if start <= pos:  # window shorter than the overlap: advance fully, no overlap
            start = end
        else:
            space = sub.rfind(" ", pos, start)
            if space > pos:
                start = space + 1
        pos = start
    return _merge_small_tail(windows, cfg.min_chunk_chars)


def _natural_boundary(sub: str, pos: int, hi: int) -> int:
    """Best split point in the back half of ``(pos, hi]``.

    Restricting the split to the back half keeps a window at least half-full, so a
    heading (or any short line) followed by a blank line is never isolated as its
    own micro-chunk. Preference: paragraph break > sentence end > word > hard cut.
    """
    seg = sub[pos:hi]
    floor = max(1, len(seg) // 2)

    cut = seg.rfind("\n\n", floor)
    if cut != -1:
        return pos + cut + 2

    last = None
    for m in _SENTENCE_END.finditer(seg):
        if m.start() >= floor:
            last = m
    if last is not None:
        return pos + last.end()

    space = seg.rfind(" ", floor)
    return pos + space + 1 if space != -1 else hi


def _protected_spans(sub: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern, flags in _PROTECTED_PATTERNS:
        for m in re.finditer(pattern, sub, flags):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _merge_small_tail(windows: list[tuple[int, int]], min_chars: int) -> list[tuple[int, int]]:
    if len(windows) >= 2:
        ls, le = windows[-1]
        if le - ls < min_chars:
            ps, _pe = windows[-2]
            windows[-2] = (ps, le)
            windows.pop()
    return windows


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _tighten(source: str, start: int, end: int) -> tuple[str, tuple[int, int]]:
    """Trim surrounding whitespace so ``char_range`` matches the stored body exactly."""
    seg = source[start:end]
    start += len(seg) - len(seg.lstrip())
    end -= len(seg) - len(seg.rstrip())
    return source[start:end], (start, end)


def _prepend_title(title: str | None, body: str) -> str:
    """Prepend the title unless the body already opens with it."""
    if not title:
        return body
    title = title.strip()
    if body.lstrip().startswith(title):
        return body
    return f"{title}\n\n{body}"
