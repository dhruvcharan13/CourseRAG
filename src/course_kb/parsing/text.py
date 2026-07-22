"""Reference parser for plain-text and Markdown files.

Trivial by design: it reads the whole file into a single element. Its job is to
prove the parse -> chunk -> embed -> store pipeline connects, not to be smart.
"""

from __future__ import annotations

from pathlib import Path

from course_kb.parsing import ParsedDocument, ParsedElement


class TextParser:
    """Reads a ``.txt`` / ``.md`` file as one whole-file element."""

    extensions: tuple[str, ...] = (".txt", ".md")

    def parse(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        element = ParsedElement(text=text, page=None, title=path.stem)
        return ParsedDocument(source_file=str(path), elements=[element])
