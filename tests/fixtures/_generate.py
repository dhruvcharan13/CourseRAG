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


if __name__ == "__main__":
    _write_slides(FIXTURES / "slides.pdf")
    _write_prose(FIXTURES / "prose.pdf")
    print(f"Wrote {FIXTURES / 'slides.pdf'} and {FIXTURES / 'prose.pdf'}")
