"""Configuration for course-kb.

A ``Config`` dataclass with sane defaults, optionally overridden by a
``config.toml`` loaded with the stdlib ``tomllib``. Works with no config file and
no environment variables — the zero-config default path.

The one environment variable, ``COURSE_KB_ROOT``, exists because the default ``root``
is *relative* and every command resolves it against the current working directory.
That is fine for a CLI you run from your project, and wrong for anything launched by
something else. See :func:`load_config`.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

#: Environment variable that overrides the data root with an absolute path.
ROOT_ENV_VAR = "COURSE_KB_ROOT"


@dataclass
class Config:
    """Runtime configuration.

    ``root`` is the on-disk data root that holds ``courses/`` and ``archive/``.
    The chunking params are placeholders wired through now so later phases can
    read them without a config migration.
    """

    root: Path = Path("course-kb")
    # "auto" picks the real local model when sentence-transformers is installed and
    # falls back to the dummy otherwise, so both installs work with no config file.
    # Also accepts "dummy", "local"/"bge", "minilm", or a HuggingFace model id.
    embedder: str = "auto"
    # Texts per forward pass when embedding a course's worth of chunks.
    embed_batch_size: int = 32
    # Where downloaded embedding models are cached; defaults to ``root / "models"``.
    cache_dir: Path | None = None

    # Which reranker to use *when one is asked for*. Naming it here costs nothing; only
    # `kb search --rerank` / `kb eval --rerank` construct it. "auto" picks the local
    # cross-encoder when sentence-transformers is installed and the dummy otherwise.
    reranker: str = "auto"
    # How many dense candidates the reranker rescores. A cross-encoder runs one forward
    # pass per candidate, so this is the knob that trades latency for the chance to
    # recover a low-ranked answer; it is also a hard ceiling on what reranking can fix.
    rerank_candidates: int = 20

    # Character budget for a whole-document read (``read_document`` / ``kb show``) when
    # no explicit page range is asked for. Documents in a slide-deck corpus run 7k-35k
    # chars, so this returns most of them entire and paginates only the long tail. It is
    # a *soft* cap: going over truncates visibly and names the range to ask for next,
    # never silently.
    max_document_chars: int = 20000

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

    @property
    def models_dir(self) -> Path:
        """Cache directory for downloaded embedding models."""
        return self.cache_dir if self.cache_dir is not None else self.root / "models"


def load_config(path: Path | None = None) -> Config:
    """Load configuration.

    If ``path`` is given it is read as TOML. Otherwise, a ``config.toml`` in the
    current working directory is read if present. If neither exists, defaults are
    returned. Unknown keys in the file are ignored so the config surface can grow
    without breaking older files.

    ``root`` resolves in this order, most explicit first::

        COURSE_KB_ROOT (env)  ->  absolute, via .resolve()
        config.toml `root =`  ->  as written
        neither               ->  Path("course-kb"), relative to the cwd

    The environment wins because it is the per-process signal, and because it is the
    only one a *caller* can set: a long-running process started by another program —
    the MCP server, launched by an editor or agent — inherits that program's working
    directory, not the repository's. Without an absolute root it would resolve
    ``course-kb`` against some unrelated directory, find no courses, and report an
    empty knowledge base as though that were the truth. Unset, nothing changes.
    """
    toml_path = path if path is not None else Path("config.toml")
    kwargs: dict[str, object] = {}
    if toml_path.exists():
        with toml_path.open("rb") as fh:
            data = tomllib.load(fh)
        known = {f.name for f in fields(Config)}
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("root", "cache_dir"):
            if kwargs.get(key) is not None:
                kwargs[key] = Path(str(kwargs[key])).expanduser()

    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        kwargs["root"] = Path(env_root).expanduser().resolve()
    return Config(**kwargs)
