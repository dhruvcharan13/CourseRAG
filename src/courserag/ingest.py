"""Ingest and delete, as library calls rather than CLI commands.

``cmd_ingest`` grew the whole parse -> chunk -> dedup -> embed -> store pipeline
inline, which was fine while the CLI was its only caller. ``kb sync`` and the web UI
are two more, and none of them should re-derive the dedup rule or the
embedder-mismatch guard. The pipeline lives here and returns a result object; the
front-ends decide how to say it.

Nothing in this module prints or exits. Failures that a caller must explain — an
uninitialized course, an embedder that does not match the one the course was built
with — raise :class:`IngestError` subclasses carrying the facts needed to write that
explanation, so the CLI's wording is unchanged and the web UI can render its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from courserag.chunker import chunk_document, estimate_tokens, keep_elements
from courserag.config import Config
from courserag.embedding import Embedder, get_embedder
from courserag.manifest import Manifest, manifest_path, read_manifest, write_manifest
from courserag.parsing import get_parser_for
from courserag.records import ChunkRecord
from courserag.store import CourseStore


def now_iso() -> str:
    """UTC timestamp in the format the manifest stores."""
    return datetime.now(timezone.utc).isoformat()


def course_dir(cfg: Config, course_id: str) -> Path:
    """Directory holding one course's store, manifest and source files."""
    return cfg.courses_dir / course_id


def course_exists(cfg: Config, course_id: str) -> bool:
    """Whether ``course_id`` has been initialized (has a manifest)."""
    return manifest_path(course_dir(cfg, course_id)).exists()


def list_courses(cfg: Config) -> list[str]:
    """Active course ids, sorted."""
    base = cfg.courses_dir
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class IngestError(Exception):
    """Base for failures a front-end is expected to report to a human."""


class CourseNotInitialized(IngestError):
    def __init__(self, course_id: str) -> None:
        super().__init__(f"Course '{course_id}' is not initialized.")
        self.course_id = course_id


class EmbedderMismatch(IngestError):
    """The configured embedder is not the one the course was built with.

    Carries both sides so the caller can print the full remediation. Raised *before*
    anything is parsed, embedded, or opened for write, so nothing on disk changed.
    """

    def __init__(self, course_id: str, manifest: Manifest, embedder: Embedder) -> None:
        super().__init__(
            f"embedder mismatch for course '{course_id}': course was built with "
            f"{manifest.embedding_model} (dims={manifest.dims}), config selects "
            f"{embedder.model_id} (dims={embedder.dims})"
        )
        self.course_id = course_id
        self.manifest = manifest
        self.embedder = embedder


class UnsupportedFile(IngestError):
    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(detail)
        self.path = path


class NoChunksProduced(IngestError):
    """The file parsed but yielded nothing storable — usually an image-only scan.

    Distinct from "already up to date": both store zero new chunks, and conflating
    them is how a scanned PDF silently indexes as an empty document.
    """

    def __init__(self, path: Path, pages: int, dropped: int, min_element_chars: int) -> None:
        self.path = path
        self.pages = pages
        self.dropped = dropped
        self.min_element_chars = min_element_chars
        self.hint = (
            "An image-only or scanned PDF needs OCR before it can be indexed."
            if dropped == pages
            else "Text was parsed, but no chunk survived the chunker's minimum size."
        )
        super().__init__(
            f"no chunks produced from {path.name}: {pages} page(s) parsed, {dropped} "
            f"dropped as near-empty (< {min_element_chars} non-space chars). {self.hint}"
        )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class IngestResult:
    """What one :func:`ingest_file` call did."""

    source_file: str
    category: str
    chunks_added: int
    total_chunks: int
    #: Chunks the embedder will truncate: ``(record, token_count_or_None)``. A count of
    #: ``None`` means it came from the chars/4 estimate rather than a real tokenizer.
    oversize: list[tuple[ChunkRecord, int | None]] = field(default_factory=list)
    #: True when every chunk was already stored — a re-ingest no-op.
    already_up_to_date: bool = False


@dataclass
class DeleteResult:
    source_file: str
    chunks_removed: int
    total_chunks: int
    #: True when no rows matched but a stale manifest entry was cleaned up.
    manifest_only: bool = False


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def open_course(cfg: Config, course_id: str) -> tuple[Manifest, CourseStore]:
    """Open a course for reading or deleting. Loads no embedding model.

    Deletion and inspection compare no vectors, so they must work on a course whose
    embedding model is unavailable — see :func:`delete_file`.
    """
    cdir = course_dir(cfg, course_id)
    if not manifest_path(cdir).exists():
        raise CourseNotInitialized(course_id)
    manifest = read_manifest(cdir)
    return manifest, CourseStore.open_or_create(cdir, manifest.dims)


def resolve_embedder(cfg: Config, course_id: str) -> tuple[Manifest, Embedder]:
    """Return the course manifest and an embedder that matches it.

    Raises:
        CourseNotInitialized: if the course has no manifest.
        EmbedderMismatch: if the configured embedder differs from the course's.
    """
    cdir = course_dir(cfg, course_id)
    if not manifest_path(cdir).exists():
        raise CourseNotInitialized(course_id)
    manifest = read_manifest(cdir)
    embedder = get_embedder(cfg.embedder, cfg)
    if (embedder.model_id, embedder.dims) != (manifest.embedding_model, manifest.dims):
        raise EmbedderMismatch(course_id, manifest, embedder)
    return manifest, embedder


def find_oversize(
    embedder: Embedder, records: list[ChunkRecord], cfg: Config
) -> list[tuple[ChunkRecord, int | None]]:
    """Chunks whose tails ``embedder`` will drop.

    An embedder that can tokenize gives an exact count; without one, fall back to
    ``estimate_tokens`` and report ``None`` for the count so the caller can label it as
    the estimate it is. Measured over 6.7k real chunks that proxy misses ~94% of actual
    truncations — see docs/chunking-robustness.md.
    """
    count_tokens = getattr(embedder, "count_tokens", None)
    if count_tokens is None:
        return [(r, None) for r in records if estimate_tokens(r.text) > cfg.warn_chunk_tokens]
    limit = embedder.max_input_tokens
    counts = count_tokens([r.text for r in records])
    return [(r, n) for r, n in zip(records, counts) if n > limit]


def ingest_file(
    cfg: Config,
    course_id: str,
    path: Path,
    category: str,
    *,
    manifest: Manifest | None = None,
    embedder: Embedder | None = None,
) -> IngestResult:
    """Parse, chunk, embed and store one file. Returns what changed.

    ``manifest`` and ``embedder`` let a batch caller (``kb sync``, the web UI) resolve
    the embedder once and reuse it across many files instead of paying the mismatch
    check — and, for the real model, the load — per file. When they are passed the
    caller is responsible for having validated them with :func:`resolve_embedder`;
    when they are not, this does it.

    Raises:
        CourseNotInitialized, EmbedderMismatch, UnsupportedFile, NoChunksProduced.
    """
    if manifest is None or embedder is None:
        manifest, embedder = resolve_embedder(cfg, course_id)

    if not path.is_file():
        raise UnsupportedFile(path, f"No such file: {path}")
    try:
        parser = get_parser_for(path)
    except ValueError as exc:
        raise UnsupportedFile(path, str(exc)) from exc

    cdir = course_dir(cfg, course_id)
    store = CourseStore.open_or_create(cdir, manifest.dims)
    doc = parser.parse(path)

    added_at = now_iso()
    records = chunk_document(
        doc,
        course=course_id,
        source_file=path.name,
        category=category,
        cfg=cfg,
        added_at=added_at,
    )
    if not records:
        kept = keep_elements(doc.elements, cfg)
        raise NoChunksProduced(
            path, len(doc.elements), len(doc.elements) - len(kept), cfg.min_element_chars
        )

    # Skip chunks already stored for THIS file (by content hash) so re-ingest is a
    # no-op, while identical slides shared across files are kept per file. The seen-set
    # carries forward through this batch, not just the stored snapshot: a deck with two
    # identical pages yields two identical chunks in ONE run, and checking only what was
    # already stored would let both through.
    seen = store.existing_hashes(path.name)
    new_records: list[ChunkRecord] = []
    for record in records:
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        new_records.append(record)

    if not new_records:
        return IngestResult(
            source_file=path.name,
            category=category,
            chunks_added=0,
            total_chunks=store.count(),
            already_up_to_date=True,
        )

    oversize = find_oversize(embedder, new_records, cfg)

    vectors = embedder.embed([r.text for r in new_records])
    for record, vector in zip(new_records, vectors):
        record.vector = vector
    store.add(new_records)

    if category not in manifest.categories:
        manifest.categories.append(category)
    if path.name not in manifest.files:
        manifest.files.append(path.name)
    manifest.last_indexed = added_at
    write_manifest(cdir, manifest)

    return IngestResult(
        source_file=path.name,
        category=category,
        chunks_added=len(new_records),
        total_chunks=store.count(),
        oversize=oversize,
    )


def delete_file(cfg: Config, course_id: str, source_file: str) -> DeleteResult:
    """Remove every chunk that came from ``source_file``.

    Constructs no embedder: a vector is a pure function of chunk text, so deletion
    needs no model, and a course can be pruned even when its embedding model is
    unavailable.

    Raises:
        CourseNotInitialized: if the course has no manifest.
        KeyError: if neither a stored row nor a manifest entry names ``source_file``.
    """
    manifest, store = open_course(cfg, course_id)

    removed = store.delete_source(source_file)
    if removed == 0 and source_file not in manifest.files:
        raise KeyError(source_file)

    # The manifest's file and category lists are derived from the rows, so recompute
    # rather than patch: a category may have been used only by the deleted file.
    # Filtering (instead of rebuilding) preserves the existing ingest order.
    remaining_categories = set(store.categories())
    manifest.files = [f for f in manifest.files if f != source_file]
    manifest.categories = [c for c in manifest.categories if c in remaining_categories]
    # last_indexed records when content was last *indexed*; a deletion does not index.
    write_manifest(course_dir(cfg, course_id), manifest)

    return DeleteResult(
        source_file=source_file,
        chunks_removed=removed,
        total_chunks=store.count(),
        manifest_only=removed == 0,
    )
