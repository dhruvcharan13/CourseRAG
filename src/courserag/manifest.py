"""Read/write helpers for a course's ``manifest.json``.

The manifest records what a course store contains and how it was built, so later
phases (and other tools) can inspect a course without opening the LanceDB table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    """Course metadata persisted to ``manifest.json``."""

    course: str
    embedding_model: str
    dims: int
    categories: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    last_indexed: str | None = None


def manifest_path(course_dir: Path) -> Path:
    """Path to the manifest inside a course directory."""
    return course_dir / MANIFEST_NAME


def read_manifest(course_dir: Path) -> Manifest:
    """Load and return the manifest for a course.

    Raises:
        FileNotFoundError: if the course has no manifest.
    """
    path = manifest_path(course_dir)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Manifest(
        course=data["course"],
        embedding_model=data["embedding_model"],
        dims=int(data["dims"]),
        categories=list(data.get("categories", [])),
        files=list(data.get("files", [])),
        last_indexed=data.get("last_indexed"),
    )


def write_manifest(course_dir: Path, manifest: Manifest) -> None:
    """Write ``manifest`` to the course's ``manifest.json`` (pretty-printed)."""
    path = manifest_path(course_dir)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, indent=2, sort_keys=True)
        fh.write("\n")
