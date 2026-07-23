"""Document parsing contract and registry.

A :class:`Parser` turns a file into a :class:`ParsedDocument` (a list of
:class:`ParsedElement`). Parsers register the extensions they handle;
:func:`get_parser_for` resolves the right one for a path.

Phase 0 ships one reference parser (:class:`~course_kb.parsing.text.TextParser`
for ``.txt`` / ``.md``) purely to prove the wiring. Real PDF/PPTX/DOCX parsers
arrive in Phase 1 and register themselves the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = [
    "ParsedElement",
    "ParsedDocument",
    "Parser",
    "register_parser",
    "get_parser_for",
]


@dataclass
class ParsedElement:
    """A single unit of extracted content (e.g. a page, slide, or section)."""

    text: str
    page: int | None = None
    title: str | None = None


@dataclass
class ParsedDocument:
    """The full result of parsing one source file."""

    source_file: str
    elements: list[ParsedElement]


@runtime_checkable
class Parser(Protocol):
    """Parses files with one of its declared ``extensions`` into a document."""

    extensions: tuple[str, ...]

    def parse(self, path: Path) -> ParsedDocument: ...


# Extension (lowercased, dot-prefixed) -> parser instance.
_REGISTRY: dict[str, Parser] = {}


def register_parser(parser: Parser) -> None:
    """Register ``parser`` for each of its declared extensions.

    Extensions are matched case-insensitively. Registering a parser for an
    extension that is already handled overwrites the previous entry.
    """
    for ext in parser.extensions:
        _REGISTRY[ext.lower()] = parser


def get_parser_for(path: Path) -> Parser:
    """Return the parser for ``path``'s extension.

    Raises:
        ValueError: if no parser is registered for the extension.
    """
    ext = path.suffix.lower()
    parser = _REGISTRY.get(ext)
    if parser is None:
        supported = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"No parser registered for '{ext}' (supported: {supported})")
    return parser


# Register the built-in reference parser on import so the registry is usable
# out of the box. Imported here (bottom of module) to avoid a circular import,
# since text.py imports the names defined above.
from course_kb.parsing.pdf import PdfParser  # noqa: E402
from course_kb.parsing.text import TextParser  # noqa: E402

register_parser(TextParser())
register_parser(PdfParser())
