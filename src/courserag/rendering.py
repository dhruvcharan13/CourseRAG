"""Render one page of a source document as an image.

Retrieval over a slide deck has a blind spot that no amount of better ranking fixes:
an AVL rotation, a graph traversal, an ER diagram or a UML class figure carries its
meaning in *drawn* content, and the text layer around it is often a title and a caption.
Search can find the right page and still hand back almost nothing.

So the answer is not to make search see pictures — it is to let a caller that already
knows which page it wants ask for that page as an image. Search cites ``module05.pdf
p30``; this turns that citation into something an agent can actually look at. Selective
by construction: nothing is rendered at ingest, nothing is stored, and a page costs
about 8ms only when somebody asks for it.

Reading the page also reports what is *on* it — how much text, how many vector drawings,
how many embedded images — so a caller can tell "this page is a diagram" from "this page
is prose", and stop fetching images for pages that are only words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from courserag.config import Config
from courserag.sync import scan_sources

#: Rendering resolution. 150 is a readable slide at a sane size (~70KB for a 4:3
#: deck); below ~100 small axis labels and subscripts start to go.
DEFAULT_DPI = 150
MIN_DPI, MAX_DPI = 40, 300

#: Hard ceiling on the rendered bitmap. A page defined in a huge coordinate space —
#: a poster, a plotter drawing — would otherwise turn a 150-dpi request into hundreds
#: of megabytes. The dpi is scaled down to fit rather than the request being refused.
MAX_PIXELS = 4_000_000

# Deliberately no "is this a figure" threshold.
#
# Measured across three real decks, the obvious signals do not survive contact with
# them. CS240's module05 runs a median of 65 vector drawings per page (max 120); CS348's
# ER-diagram deck — 50 pages that are almost entirely diagrams — runs a median of 5, max
# 13. An absolute cutoff tuned on the first flags nothing on the second, and text length
# does not separate them either, because an ER diagram's entity and attribute labels
# count as text.
#
# What *is* stable is the comparison within one document: module05's figure pages sit at
# 65-120 against its own floor of 4, and the ER deck's diagram pages carry many short
# label spans and little prose against its own baseline. So the numbers are reported
# per page alongside the document's median, and the judgement is left to the caller —
# which, for the MCP server, is a model that can also just look at the rendered page.


class RenderError(Exception):
    """A page could not be rendered, in a way worth explaining to the caller."""


@dataclass
class PageImage:
    """A rendered page, and what was found on it."""

    png: bytes
    course: str
    source_file: str
    page: int
    page_count: int
    width: int
    height: int
    dpi: int
    text_chars: int
    drawings: int
    images: int

    def summary(self) -> str:
        """One line describing the page, for a caller to show beside the image."""
        return (
            f"{self.source_file} page {self.page} of {self.page_count} "
            f"({self.width}x{self.height}px at {self.dpi}dpi). This page has "
            f"{self.text_chars} chars of text, {self.drawings} vector drawings and "
            f"{self.images} embedded images."
        )


@dataclass
class PageStats:
    """Per-page signals, cheap enough to compute for a whole document."""

    page: int
    text_chars: int
    #: Text runs on the page. Many short ones with little total text is the shape of a
    #: labelled diagram; few long ones is prose.
    spans: int
    short_spans: int
    drawings: int
    images: int


def resolve_source(cfg: Config, course: str, source_file: str) -> Path:
    """Find a course's document on disk by the name the index cites it under.

    Looked up in the course's scan by basename rather than joined onto a path, so the
    only readable files are ones sync already found under ``raw/`` — the same rule the
    web viewer uses, for the same reason.

    Raises:
        RenderError: if the course holds no such file.
    """
    files = scan_sources(cfg, course).files
    wanted = Path(source_file).name
    match = files.get(wanted)
    if match is None:
        folded = [f for name, f in files.items() if name.lower() == wanted.lower()]
        if len(folded) != 1:
            known = ", ".join(sorted(files)) or "(none)"
            raise RenderError(
                f"'{course}' has no source file named '{wanted}'. Files: {known}"
            )
        match = folded[0]
    return match.path


def render_page(
    cfg: Config, course: str, source_file: str, page: int, dpi: int = DEFAULT_DPI
) -> PageImage:
    """Render one 1-based page of a document to PNG.

    Raises:
        RenderError: for an unknown file, a non-PDF, or a page out of range.
    """
    path = resolve_source(cfg, course, source_file)
    if path.suffix.lower() != ".pdf":
        raise RenderError(
            f"{path.name} is not a PDF, so it has no pages to render. Read it with "
            f"read_document instead."
        )

    import pymupdf  # local: rendering is the only thing in the package that needs it here

    dpi = max(MIN_DPI, min(int(dpi), MAX_DPI))
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - the file is user-supplied
        raise RenderError(f"Could not open {path.name}: {exc}") from exc

    with doc:
        if not 1 <= page <= doc.page_count:
            raise RenderError(
                f"{path.name} has {doc.page_count} pages; page {page} is out of range. "
                f"Pages are numbered from 1, matching the citations search returns."
            )
        target = doc[page - 1]

        # Scale the request down rather than refusing it, so an unusually large page
        # still renders — just not at the resolution that would blow up memory.
        #
        # This deliberately ignores MIN_DPI. The floor exists so a *readable* request
        # stays readable; the ceiling exists so no request can allocate a gigabyte.
        # Clamping back up to the floor here would let a 5000pt-square page through at
        # 7.7M pixels, which is the whole thing the cap is for — safety wins over
        # legibility, and the returned dpi says plainly what happened.
        box = target.rect
        estimated = (box.width * dpi / 72) * (box.height * dpi / 72)
        if estimated > MAX_PIXELS:
            dpi = max(1, int(dpi * (MAX_PIXELS / estimated) ** 0.5))

        pixmap = target.get_pixmap(dpi=dpi)
        return PageImage(
            png=pixmap.tobytes("png"),
            course=course,
            source_file=path.name,
            page=page,
            page_count=doc.page_count,
            width=pixmap.width,
            height=pixmap.height,
            dpi=dpi,
            text_chars=len(target.get_text().strip()),
            drawings=len(target.get_drawings()),
            images=len(target.get_images()),
        )


def page_stats(cfg: Config, course: str, source_file: str) -> list[PageStats]:
    """Per-page signals for a whole document, rendering nothing.

    Lets a caller decide which pages are worth fetching as images without paging
    through a 41-slide deck one render at a time. It reports counts and states no
    verdict — see the note above on why a fixed "this is a figure" rule does not
    survive contact with real decks.

    Raises:
        RenderError: if the course holds no such file.
    """
    path = resolve_source(cfg, course, source_file)
    if path.suffix.lower() != ".pdf":
        return []

    import pymupdf

    out: list[PageStats] = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, 1):
            spans = [
                s
                for block in page.get_text("dict")["blocks"]
                if block.get("type") == 0
                for line in block["lines"]
                for s in line["spans"]
            ]
            out.append(
                PageStats(
                    page=number,
                    text_chars=len(page.get_text().strip()),
                    spans=len(spans),
                    short_spans=sum(1 for s in spans if len(s["text"].strip()) <= 24),
                    drawings=len(page.get_drawings()),
                    images=len(page.get_images()),
                )
            )
    return out
