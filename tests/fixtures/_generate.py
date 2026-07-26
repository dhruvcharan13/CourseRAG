"""Regenerate the committed fixture PDFs with PyMuPDF.

Run from the repo root: ``python tests/fixtures/_generate.py``. The PDFs it emits
are committed so tests need no PDF toolchain at run time; this script exists only
to document how they were made and to recreate them if needed.
"""

from __future__ import annotations

from pathlib import Path

import fitz

FIXTURES = Path(__file__).parent

# (title, [body lines]) per page. Large bold title + small body => slide-style.
_SLIDES = [
    ("Balanced Search Trees", ["AVL trees keep height balanced", "Rotations restore the invariant", "Lookups run in O(log n)"]),
    ("AVL Rotations", ["Left and right rotations", "Rebalance after insert or delete", "Constant work per rotation"]),
    ("Amortized Analysis", ["Aggregate method", "Banker's and physicist's views", "Splay trees amortize to O(log n)"]),
]

# Prose page: a numbered heading + several sentences (well over 100 words/page).
_PARA = (
    "A self-balancing binary search tree keeps its height logarithmic in the "
    "number of stored keys so that search, insertion, and deletion all run in "
    "logarithmic time. The balance is maintained by local restructuring "
    "operations called rotations, which preserve the in-order sequence of keys "
    "while reducing the height of the affected subtree. "
)
_PROSE = [
    ("1. Introduction", _PARA * 3),
    ("2. Rotations and Invariants", _PARA * 3),
]


def _write_slides(path: Path) -> None:
    doc = fitz.open()
    for title, lines in _SLIDES:
        page = doc.new_page()
        page.insert_text((72, 96), title, fontsize=22, fontname="hebo")
        y = 150
        for line in lines:
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 24
    doc.save(path, deflate=True)
    doc.close()


def _write_prose(path: Path) -> None:
    doc = fitz.open()
    for heading, para in _PROSE:
        page = doc.new_page()
        page.insert_text((72, 90), heading, fontsize=14, fontname="hebo")
        rect = fitz.Rect(72, 110, 523, 740)
        page.insert_textbox(rect, para, fontsize=11, fontname="helv")
    doc.save(path, deflate=True)
    doc.close()


# Prose notes with a repeated running header + page numbers, to exercise the
# parser's de-boilerplating. Each page carries the same header; bodies differ.
_HEADER = "MATH 239 COURSE NOTES"
_NOTES_BODY = [
    "A graph is a finite nonempty set of vertices together with a set of edges, "
    "each edge being an unordered pair of distinct vertices. Two vertices joined "
    "by an edge are adjacent, and the edge is incident with each of them. " * 3,
    "The degree of a vertex is the number of edges incident with it. Summing the "
    "degrees over all vertices counts every edge twice, which gives the handshake "
    "identity relating the degree sum to the number of edges. " * 3,
    "A walk alternates vertices and edges; a path is a walk whose vertices are all "
    "distinct. A graph is connected when every pair of vertices is joined by a path, "
    "and a shortest such path realises the distance between them. " * 3,
]


def _write_notes(path: Path) -> None:
    doc = fitz.open()
    for pno, para in enumerate(_NOTES_BODY, start=1):
        page = doc.new_page()
        page.insert_text((72, 40), _HEADER, fontsize=11, fontname="helv")  # running header
        y = 90
        if pno == 1:
            page.insert_text((72, y), "4.1 Definitions", fontsize=14, fontname="hebo")
            y = 120
        page.insert_textbox(fitz.Rect(72, y, 523, 720), para, fontsize=11, fontname="helv")
        # Large "Page n of 3" footer: on heading-less pages it would be grabbed as the
        # title unless page-number lines are excluded (parser fix #5).
        page.insert_text((410, 760), f"Page {pno} of 3", fontsize=16, fontname="helv")
    doc.save(path, deflate=True)
    doc.close()


def _write_scanned(path: Path) -> None:
    """A scanned/image-only deck: page 1 has only a drawing, page 2 a stray number."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.draw_rect(fitz.Rect(120, 120, 480, 400), fill=(0.8, 0.8, 0.8))  # image stand-in, no text
    p2 = doc.new_page()
    p2.draw_rect(fitz.Rect(120, 120, 480, 400), fill=(0.8, 0.8, 0.8))
    p2.insert_text((300, 760), "42", fontsize=11, fontname="helv")  # stray page number only
    doc.save(path, deflate=True)
    doc.close()


def _write_twocol(path: Path) -> None:
    """Two-column page whose RIGHT column is written first (scrambles raw block order)."""
    doc = fitz.open()
    page = doc.new_page()
    right = "\n".join(f"Beta{i}" for i in range(1, 9))
    left = "\n".join(f"Alpha{i}" for i in range(1, 9))
    page.insert_textbox(fitz.Rect(300, 90, 520, 700), right, fontsize=11, fontname="helv")
    page.insert_textbox(fitz.Rect(72, 90, 290, 700), left, fontsize=11, fontname="helv")
    doc.save(path, deflate=True)
    doc.close()


def _write_onecol_wide(path: Path) -> None:
    """Single-column page whose full-width title sits RIGHT of the page midpoint.

    Reproduces the shape of a real lecture handout: near-full-width blocks whose bbox
    centres land either side of the fold by a hair. Sorting such a page by (column, y)
    without a two-column guard moves the title to the bottom of the page.
    """
    doc = fitz.open()
    page = doc.new_page()  # 595 x 842, midpoint x = 297.5
    # Title crosses the midpoint but its centre lands just RIGHT of it -> "column 1".
    page.insert_text((200, 80), "Development Process Overview", fontsize=16, fontname="hebo")
    # Body also crosses the midpoint, but its centre lands LEFT of it -> "column 0",
    # so an unguarded (column, y) sort emits the body before the title.
    page.insert_textbox(
        fitz.Rect(72, 130, 460, 400),
        "The body text follows the title in ordinary top-to-bottom reading order. "
        "Nothing on this page is laid out in columns.",
        fontsize=11,
        fontname="helv",
    )
    doc.save(path, deflate=True)
    doc.close()


# A mixed-mode deck: many sparse slides plus a dense appendix. The chunker picks
# slide-vs-prose mode once per document from the MEDIAN words/page, so the sparse
# majority wins and each dense appendix page collapses into one oversize chunk.
_MIXED_SPARSE_PAGES = 40
_MIXED_DENSE_PAGES = 5
_MIXED_DENSE_PARA = (
    "The appendix restates each result in full detail, including the hypotheses that "
    "the lecture slides left implicit and the boundary cases that arise when the input "
    "is empty or contains a single element. Each proof proceeds by induction on the "
    "size of the structure under consideration. "
)


def _write_mixed(path: Path) -> None:
    doc = fitz.open()
    for i in range(1, _MIXED_SPARSE_PAGES + 1):
        page = doc.new_page()
        page.insert_text((72, 96), f"Topic {i}", fontsize=22, fontname="hebo")
        page.insert_text((72, 150), f"A single sparse bullet for topic {i}", fontsize=11, fontname="helv")
    for i in range(1, _MIXED_DENSE_PAGES + 1):
        page = doc.new_page()
        page.insert_text((72, 90), f"A.{i} Appendix", fontsize=14, fontname="hebo")
        page.insert_textbox(
            fitz.Rect(72, 110, 523, 740), _MIXED_DENSE_PARA * 6, fontsize=11, fontname="helv"
        )
    doc.save(path, deflate=True)
    doc.close()


_WRITERS = {
    "slides.pdf": _write_slides,
    "prose.pdf": _write_prose,
    "notes.pdf": _write_notes,
    "scanned.pdf": _write_scanned,
    "twocol.pdf": _write_twocol,
    "onecol_wide.pdf": _write_onecol_wide,
    "mixed.pdf": _write_mixed,
}


if __name__ == "__main__":
    for name, write in _WRITERS.items():
        write(FIXTURES / name)
    print(f"Wrote {', '.join(_WRITERS)} to {FIXTURES}")
