"""The row schema shared by every phase: :class:`ChunkRecord`.

One ``ChunkRecord`` is one stored chunk. ``to_dict`` / ``from_dict`` convert to
and from the flat dict shape LanceDB stores (see :mod:`courserag.store`), and are
exact inverses so ``ChunkRecord.from_dict(r.to_dict()) == r``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkRecord:
    """A single chunk of a source document.

    Required fields describe identity and provenance; the optional fields are
    filled progressively — ``vector`` stays ``None`` until the chunk is embedded.
    """

    # Required.
    id: str  # e.g. "CS240-W26::lec06.pdf::p12::c3"
    text: str
    course: str
    source_file: str
    category: str
    content_hash: str
    added_at: str  # ISO 8601

    # Optional / filled later.
    vector: list[float] | None = None  # None until embedded
    title: str | None = None
    module: str | None = None
    page: int | None = None
    # Offsets of the chunk BODY within the source element's text, i.e.
    # element_text[char_start:char_end] == the body. Note ``text`` additionally has
    # the section/slide title prepended, so char_range does NOT index ``text``.
    char_range: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a flat dict matching the LanceDB column layout.

        ``char_range`` is split into nullable ``char_start`` / ``char_end`` and
        ``vector`` is emitted as a plain list, keeping the stored schema flat and
        query-friendly.
        """
        char_start, char_end = self.char_range if self.char_range is not None else (None, None)
        return {
            "id": self.id,
            "text": self.text,
            "vector": list(self.vector) if self.vector is not None else None,
            "course": self.course,
            "source_file": self.source_file,
            "category": self.category,
            "title": self.title,
            "module": self.module,
            "page": self.page,
            "char_start": char_start,
            "char_end": char_end,
            "content_hash": self.content_hash,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChunkRecord:
        """Rebuild a record from the flat dict produced by :meth:`to_dict`."""
        char_start = d.get("char_start")
        char_end = d.get("char_end")
        char_range = (int(char_start), int(char_end)) if char_start is not None and char_end is not None else None

        vector = d.get("vector")
        page = d.get("page")
        return cls(
            id=d["id"],
            text=d["text"],
            course=d["course"],
            source_file=d["source_file"],
            category=d["category"],
            content_hash=d["content_hash"],
            added_at=d["added_at"],
            vector=list(vector) if vector is not None else None,
            title=d.get("title"),
            module=d.get("module"),
            page=int(page) if page is not None else None,
            char_range=char_range,
        )

    @staticmethod
    def hash_text(text: str) -> str:
        """Stable content hash for a chunk's text (sha256 hex digest)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
