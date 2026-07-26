"""Configuration for course-kb.

A ``Config`` dataclass with sane defaults, optionally overridden by a
``config.toml`` loaded with the stdlib ``tomllib``. Works with no config file and
no environment variables — the zero-config default path.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    """Runtime configuration.

    ``root`` is the on-disk data root that holds ``courses/`` and ``archive/``.
    The chunking params are placeholders wired through now so later phases can
    read them without a config migration.
    """

    root: Path = Path("course-kb")
    embedder: str = "dummy"

    chunk_size: int = 1000
    # Currently unread: the chunker derives prose overlap from its own OVERLAP_RATIO
    # (~12% of chunk_size) rather than an absolute char count. Kept for config stability.
    chunk_overlap: int = 200
    min_chunk_chars: int = 50
    # Drop parsed elements with fewer non-whitespace chars than this (scanned /
    # image-only pages extract to a few stray chars and would become junk chunks).
    min_element_chars: int = 10
    # Chunks whose estimated token count exceeds this are flagged (a real embedder
    # like MiniLM truncates at ~256 tokens); tokens are approximated as chars / 4.
    warn_chunk_tokens: int = 256

    @property
    def courses_dir(self) -> Path:
        """Directory holding one subfolder per active course."""
        return self.root / "courses"

    @property
    def archive_dir(self) -> Path:
        """Directory holding retired courses (populated in a later phase)."""
        return self.root / "archive"


def load_config(path: Path | None = None) -> Config:
    """Load configuration.

    If ``path`` is given it is read as TOML. Otherwise, a ``config.toml`` in the
    current working directory is read if present. If neither exists, defaults are
    returned. Unknown keys in the file are ignored so the config surface can grow
    without breaking older files.
    """
    toml_path = path if path is not None else Path("config.toml")
    if not toml_path.exists():
        return Config()

    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)

    known = {f.name for f in fields(Config)}
    kwargs: dict[str, object] = {k: v for k, v in data.items() if k in known}
    if "root" in kwargs:
        kwargs["root"] = Path(str(kwargs["root"]))
    return Config(**kwargs)
