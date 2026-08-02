"""Expose the course knowledge base to an agent over MCP.

Three read-only tools — list the courses, describe one, search one — as thin wrappers
over :func:`course_kb.retrieval.retrieve` and the manifest. Retrieval behaviour is
identical to ``kb search``; this module owns presentation and error handling, nothing
else. Ingestion stays in the CLI, so an agent can read the knowledge base but never
rewrite it.

Two things are load-bearing and easy to get wrong:

**The data root must be absolute.** A stdio server inherits the *client's* working
directory, and ``Config.root`` defaults to a relative ``course-kb``. Set
``COURSE_KB_ROOT`` when registering the server (see the module docstring's usage
example below); without it the server resolves the root against whatever directory the
editor happened to launch from, finds nothing, and reports an empty knowledge base as
though that were true. :func:`course_kb.config.load_config` implements the precedence.

**Nothing here may write to stdout.** Under the stdio transport, stdout *is* the
JSON-RPC channel — a stray ``print`` corrupts the stream and surfaces as an
unexplained client-side protocol error. That is why these tools wrap ``retrieval`` and
the manifest directly rather than reusing the ``cmd_*`` functions in
:mod:`course_kb.cli`, all of which print.

Register with::

    claude mcp add course-kb \\
      -e COURSE_KB_ROOT=/abs/path/to/course-kb \\
      -- /abs/path/to/.venv/bin/python -m course_kb.mcp_server
"""

from __future__ import annotations

import os
import re

# Before anything can import the model stack. The embedding and reranking models are
# already on disk by the time a course is searchable — you cannot index a course
# without having downloaded its embedder — so the Hub round trip at load time only
# checks for updates nobody asked for. Skipping it makes a cold search 3.9s -> 2.5s
# and a cold reranked search 2.1s -> 0.2s, and means a background server started with
# no network works instead of stalling. ``setdefault``, so HF_HUB_OFFLINE=0 still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from mcp.server import MCPServer  # noqa: E402

from course_kb.config import Config, load_config  # noqa: E402
from course_kb.embedding import Embedder, get_embedder  # noqa: E402
from course_kb.manifest import Manifest, manifest_path, read_manifest  # noqa: E402
from course_kb.records import ChunkRecord  # noqa: E402
from course_kb.retrieval import SearchResult, retrieve  # noqa: E402
from course_kb.store import CourseStore  # noqa: E402

server = MCPServer(
    name="course-kb",
    instructions=(
        "A local, per-course knowledge base over the user's own course materials "
        "(lecture slides, assignments, readings, syllabi). Use search_course to answer "
        "questions about a specific course, and always cite the source file and page "
        "number that each passage came from. Call list_courses first if you are not "
        "sure of the exact course id."
    ),
)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


class CourseUnavailable(Exception):
    """A course cannot be queried, with a message written for the model to act on.

    Raised rather than returned so each tool has one exit for every way resolution can
    fail. Tools catch it and hand the text back as the result: an agent can recover
    from "no course named X; here are the real ones" on its next call, but a traceback
    only ends the turn.
    """


def _course_ids(cfg: Config) -> list[str]:
    """Every initialized course id, sorted."""
    base = cfg.courses_dir
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if manifest_path(p).exists())


def _resolve(cfg: Config, course: str) -> Manifest:
    """The manifest for ``course``, or an explanation of why there isn't one."""
    course_dir = cfg.courses_dir / course
    if not manifest_path(course_dir).exists():
        known = _course_ids(cfg)
        available = ", ".join(known) if known else "none"
        raise CourseUnavailable(
            f"No course named {course!r} in the knowledge base at {cfg.root}.\n"
            f"Available courses: {available}"
        )
    return read_manifest(course_dir)


def _open_store(cfg: Config, course: str) -> tuple[CourseStore, Manifest]:
    """Open a course for *reading*, with no embedder involved.

    Reading a document back compares no vectors, so it needs neither a model nor the
    embedder-match guard that :func:`_open` applies. Keeping the two paths separate is
    what lets ``read_document`` stay free of torch, and lets it still work on a course
    whose embedder is missing or has since changed — the text and its citations are
    just as valid either way.
    """
    manifest = _resolve(cfg, course)
    store = CourseStore.open_or_create(cfg.courses_dir / course, manifest.dims)
    if store.count() == 0:
        raise CourseUnavailable(
            f"Course {course!r} exists but has no documents indexed yet. The user "
            f"needs to run: kb ingest {course} <file> --category <category>"
        )
    return store, manifest


def _open(cfg: Config, course: str) -> tuple[CourseStore, Embedder, Manifest]:
    """Open a course for querying, or explain why it cannot be queried.

    The embedder check is the invariant ingest enforces, applied at the other end. A
    query embedded by a different model than the passages lands in a different vector
    space, where the scores are not merely worse but meaningless — so this refuses
    instead of returning numbers that look like relevance.
    """
    manifest = _resolve(cfg, course)
    embedder = get_embedder(cfg.embedder, cfg)
    if (embedder.model_id, embedder.dims) != (manifest.embedding_model, manifest.dims):
        raise CourseUnavailable(
            f"Cannot search {course!r}: it was indexed with "
            f"{manifest.embedding_model} ({manifest.dims}-dim) but this server is "
            f"configured for {embedder.model_id} ({embedder.dims}-dim). Vectors from "
            f"different models are not comparable. The user needs to rebuild the "
            f"course or restore the original embedder; this is not something you can "
            f"work around."
        )
    store = CourseStore.open_or_create(cfg.courses_dir / course, manifest.dims)
    if store.count() == 0:
        raise CourseUnavailable(
            f"Course {course!r} exists but has no documents indexed yet. The user "
            f"needs to run: kb ingest {course} <file> --category <category>"
        )
    return store, embedder, manifest


def _resolve_source(store: CourseStore, course: str, source_file: str) -> str:
    """Match ``source_file`` against the course's documents, or explain what exists.

    Matching is exact first, then case-insensitive, because a model reading a citation
    out of a previous result reproduces the name but not always its capitalization. An
    ambiguous fold (two files differing only in case) is refused rather than guessed —
    picking one silently would cite the wrong document.
    """
    known = store.source_files()
    if source_file in known:
        return source_file

    folded = [name for name in known if name.lower() == source_file.lower()]
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1:
        raise CourseUnavailable(
            f"{source_file!r} matches more than one document in {course} "
            f"({', '.join(folded)}). Ask for one of those exactly."
        )
    listing = "\n".join(f"  {name}" for name in known)
    raise CourseUnavailable(
        f"No document named {source_file!r} in {course}.\nDocuments in {course}:\n{listing}"
    )


def _parse_pages(pages: str) -> tuple[int, int | None]:
    """Parse ``"12"``, ``"1-20"`` or ``"25-"`` into an inclusive ``(first, last)``.

    Raises:
        CourseUnavailable: on anything else, so a malformed range comes back as a
            correctable sentence rather than a traceback that ends the turn.
    """
    text = pages.strip()
    match = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+)?)?", text)
    if not match:
        raise CourseUnavailable(
            f"Could not read {pages!r} as a page range. Use a single page (\"12\"), a "
            f"range (\"1-20\"), or an open end (\"25-\")."
        )
    first = int(match.group(1))
    if match.group(0).find("-") == -1:
        return first, first
    last = int(match.group(2)) if match.group(2) else None
    if last is not None and last < first:
        raise CourseUnavailable(f"Page range {pages!r} ends before it starts.")
    return first, last


def _format_document(records: list[ChunkRecord]) -> str:
    """Passages in reading order, each headed by its page.

    Deliberately unlike :func:`_format`: there is no rank and no score, because nothing
    here was ranked. Showing a relevance number on a straight document read would
    invite treating reading order as a relevance order.
    """
    blocks = []
    for chunk in records:
        head = f"[p{chunk.page}]" if chunk.page is not None else "[page unknown]"
        if chunk.title:
            head += f" {chunk.title}"
        blocks.append(f"{head}\n{chunk.text}")
    return "\n\n".join(blocks)


def _is_dummy(manifest: Manifest) -> bool:
    """Whether a course was built with placeholder vectors rather than a real model."""
    return manifest.embedding_model.startswith("dummy-")


def _format(results: list[SearchResult]) -> str:
    """Ranked passages, each headed by the citation the model should quote.

    The citation leads and the score trails: putting the file and page first is what
    makes a model attribute the passage correctly instead of paraphrasing it unsourced.
    Scores are shown but are informational — no threshold is applied, because a
    meaningful relevance floor is model-specific and has to be calibrated from measured
    data rather than guessed.
    """
    blocks = []
    for rank, r in enumerate(results, start=1):
        chunk = r.chunk
        page = f"p{chunk.page}" if chunk.page is not None else "page unknown"
        score = (
            f"relevance {r.score:.3f}"
            if r.rerank_score is None
            else f"rerank {r.rerank_score:+.2f}, dense {r.score:.3f}"
        )
        head = f"[{rank}] {chunk.source_file} {page} ({chunk.category}; {score})"
        if chunk.title:
            head += f"\n{chunk.title}"
        blocks.append(f"{head}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@server.tool()
def list_courses() -> str:
    """List every course in the knowledge base with its document and passage counts.

    Call this first when you are not certain of the exact course id, rather than
    guessing one and getting an error.
    """
    cfg = load_config()
    ids = _course_ids(cfg)
    if not ids:
        return (
            f"The knowledge base at {cfg.root} has no courses yet. The user creates one "
            f"with: kb init-course <COURSE_ID>"
        )

    lines = [f"{len(ids)} course(s) in {cfg.root}:", ""]
    for cid in ids:
        manifest = read_manifest(cfg.courses_dir / cid)
        store = CourseStore.open_or_create(cfg.courses_dir / cid, manifest.dims)
        note = "  [NOT SEARCHABLE: placeholder vectors, results would be noise]" if (
            _is_dummy(manifest)
        ) else ""
        lines.append(
            f"  {cid} — {len(manifest.files)} document(s), {store.count()} passage(s), "
            f"categories: {', '.join(manifest.categories) or 'none'}{note}"
        )
    return "\n".join(lines)


@server.tool()
def course_info(course: str) -> str:
    """Show what one course contains: its source documents, categories, and size.

    Useful before searching, to see which materials exist and what a citation like
    "module05.pdf" actually refers to.

    Args:
        course: The course id, exactly as returned by list_courses (e.g. "CS240").
    """
    cfg = load_config()
    try:
        manifest = _resolve(cfg, course)
    except CourseUnavailable as exc:
        return str(exc)

    store = CourseStore.open_or_create(cfg.courses_dir / course, manifest.dims)
    lines = [
        f"Course:      {manifest.course}",
        f"Passages:    {store.count()}",
        f"Categories:  {', '.join(manifest.categories) or 'none'}",
        f"Indexed by:  {manifest.embedding_model} ({manifest.dims}-dim)",
        f"Updated:     {manifest.last_indexed or 'never'}",
        "",
        f"Documents ({len(manifest.files)}):",
    ]
    lines += [f"  {name}" for name in manifest.files] or ["  (none)"]
    if _is_dummy(manifest):
        lines += [
            "",
            "WARNING: this course was indexed with placeholder vectors, not a real "
            "embedding model. Searching it returns noise. Tell the user it needs "
            "rebuilding.",
        ]
    return "\n".join(lines)


@server.tool()
def search_course(
    course: str, query: str, k: int = 5, rerank: bool = False, source_file: str = ""
) -> str:
    """Search one course's materials and return the most relevant passages.

    Each passage comes with its source file and page number — cite them. Ask a full
    question ("how is a skip list's tower height chosen?") rather than keywords; the
    index is semantic, so a well-formed question retrieves better than a bag of terms.

    Args:
        course: The course id, exactly as returned by list_courses (e.g. "CS240").
        query: The question to answer, in natural language.
        k: How many passages to return. Defaults to 5; raise it when a question spans
            several slides or documents.
        rerank: Run a slower, more accurate second pass that re-reads each candidate
            against the question. Costs roughly 60ms. Worth setting when the default
            results are on the right topic but do not actually answer what was asked.
        source_file: Restrict the search to one document, named as course_info lists it
            (e.g. "lect-SQL-handout.pdf"). Use this when you already know which document
            should hold the answer; without it a question competes against the whole
            course, and a related document can outrank the right one. To read a document
            straight through rather than search it, use read_document instead.
    """
    cfg = load_config()
    try:
        store, embedder, manifest = _open(cfg, course)
        scope = _resolve_source(store, course, source_file) if source_file else None
    except CourseUnavailable as exc:
        return str(exc)

    if k < 1:
        return f"k must be at least 1, got {k}."

    results, _ = retrieve(cfg, store, embedder, query, k, use_rerank=rerank, source_file=scope)
    if not results:
        where = f"{course}/{scope}" if scope else course
        return f"No passages found in {where} for {query!r}."

    within = f" in {scope}" if scope else ""
    header = f"Top {len(results)} passage(s) from {course}{within} for {query!r}:"
    if _is_dummy(manifest):
        header = (
            f"WARNING: {course} was indexed with placeholder vectors, so the ranking "
            f"below is random and must not be presented as an answer.\n\n" + header
        )
    return f"{header}\n\n{_format(results)}"


@server.tool()
def read_document(course: str, source_file: str, pages: str = "") -> str:
    """Read one document straight through, in page order, rather than searching it.

    This is the tool for "explain module 5 to me" or "summarize this assignment" — a
    question about a *document* rather than a question the document happens to answer.
    search_course cannot do this: it ranks by similarity across the whole course, so it
    returns scattered passages and misses most of any one document.

    Long documents come back truncated at a page boundary, with a marker naming the
    range to ask for next. Truncation is always visible; you will never silently get a
    partial document.

    Args:
        course: The course id, exactly as returned by list_courses (e.g. "CS348").
        source_file: The document, named as course_info lists it (e.g. "lect-ER-handout.pdf").
        pages: Optional page range — "12", "1-20", or "25-" for page 25 onward. Omit it
            to read from the start.
    """
    cfg = load_config()
    try:
        store, manifest = _open_store(cfg, course)
        name = _resolve_source(store, course, source_file)
        window = _parse_pages(pages) if pages else None
    except CourseUnavailable as exc:
        return str(exc)

    records = store.read_source(name)
    if not records:
        return f"{name} is listed in {course} but has no stored passages."

    total_pages = max((r.page for r in records if r.page is not None), default=0)
    if window is not None:
        first, last = window
        records = [
            r for r in records
            if r.page is not None and r.page >= first and (last is None or r.page <= last)
        ]
        if not records:
            return (
                f"No passages in {name} within pages {pages!r}. The document runs to "
                f"page {total_pages}."
            )

    # Truncate on a page boundary rather than mid-page: a page split across the cut
    # would look complete in one call and be silently duplicated in the next.
    kept: list[ChunkRecord] = []
    used = 0
    for record in records:
        if kept and used + len(record.text) > cfg.max_document_chars:
            boundary = record.page
            kept = [r for r in kept if r.page != boundary] or kept
            break
        kept.append(record)
        used += len(record.text)

    shown = [r.page for r in kept if r.page is not None]
    body = _format_document(kept)
    header = (
        f"{name} — {course} ({len(kept)} of {len(records)} passage(s), "
        f"pages {min(shown, default='?')}-{max(shown, default='?')} of {total_pages})"
    )
    if len(kept) < len(records):
        resume = max(shown, default=0) + 1
        body += (
            f"\n\n[truncated at the {cfg.max_document_chars}-character budget: "
            f"{len(kept)} of {len(records)} passages shown. "
            f'Request the rest with pages="{resume}-".]'
        )
    if _is_dummy(manifest):
        header = (
            f"NOTE: {course} was indexed with placeholder vectors. The text below is "
            f"still the real document — only its search ranking is meaningless.\n\n" + header
        )
    return f"{header}\n\n{body}"


def main() -> None:
    """Run the server on stdio. Entry point for ``python -m course_kb.mcp_server``."""
    server.run()


if __name__ == "__main__":
    main()
