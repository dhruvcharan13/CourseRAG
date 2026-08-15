"""Expose the course knowledge base to an agent over MCP.

Seven tools — list the courses, describe one, search one, read a document straight
through, look at one of its pages as an image, preview what is on each page, and record
a lasting note about a course — as thin wrappers over
:func:`courserag.retrieval.retrieve`, the manifest, :mod:`courserag.rendering` and
:mod:`courserag.memory`. Retrieval behaviour is identical to ``kb search``; this module
owns presentation and error handling, nothing else.

**Six of the seven are read-only, and the seventh writes only notes.** ``remember``
appends to a course's ``COURSE.md``; nothing here can ingest a document, delete a chunk,
or touch the index. Ingestion stays in the CLI. That boundary is the point: an agent may
accumulate what it learns *about* a course, and cannot alter the course itself. Appends
rather than rewrites for the same reason — see :mod:`courserag.memory`.

Every read of a course carries its notes at the top, because the things notes hold (the
professor's notation, material that was skipped, how the user wants citations) change
how the passages below should be read. Memory nobody sees until they think to look it up
is not memory.

``page_image`` exists because retrieval over slides has a blind spot text ranking
cannot close: a tree rotation, a graph traversal or an ER diagram carries its meaning
in drawn content, and the text layer around it is a title and a few labels. Search
finds the right page and returns almost nothing. Rendering that page on request turns
a citation the other tools already produce into something a model can look at.

Two things are load-bearing and easy to get wrong:

**The data root must be absolute.** A stdio server inherits the *client's* working
directory, and ``Config.root`` defaults to a relative ``course-kb``. Set
``COURSE_KB_ROOT`` when registering the server (see the module docstring's usage
example below); without it the server resolves the root against whatever directory the
editor happened to launch from, finds nothing, and reports an empty knowledge base as
though that were true. :func:`courserag.config.load_config` implements the precedence.

**Nothing here may write to stdout.** Under the stdio transport, stdout *is* the
JSON-RPC channel — a stray ``print`` corrupts the stream and surfaces as an
unexplained client-side protocol error. That is why these tools wrap ``retrieval`` and
the manifest directly rather than reusing the ``cmd_*`` functions in
:mod:`courserag.cli`, all of which print.

Register with::

    claude mcp add courserag \\
      -e COURSE_KB_ROOT=/abs/path/to/course-kb \\
      -- /abs/path/to/.venv/bin/python -m courserag.mcp_server
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
from mcp.server.mcpserver.utilities.types import Image  # noqa: E402

from courserag.config import Config, load_config  # noqa: E402
from courserag.embedding import Embedder, get_embedder  # noqa: E402
from courserag.manifest import Manifest, manifest_path, read_manifest  # noqa: E402
from courserag.memory import NoteRefused, append_note, for_prompt, read_notes  # noqa: E402
from courserag.records import ChunkRecord  # noqa: E402
from courserag.rendering import (  # noqa: E402
    DEFAULT_DPI,
    RenderError,
    page_stats,
    render_page,
)
from courserag.retrieval import SearchResult, retrieve  # noqa: E402
from courserag.store import CourseStore  # noqa: E402

server = MCPServer(
    name="courserag",
    instructions=(
        "A local, per-course knowledge base over the user's own course materials "
        "(lecture slides, assignments, readings, syllabi). Use search_course to answer "
        "questions about a specific course, and always cite the source file and page "
        "number that each passage came from. Call list_courses first if you are not "
        "sure of the exact course id.\n\n"
        "Much of this material is slides, where the substance of a page is often drawn "
        "rather than written: trees and their rotations, graphs and traversals, ER and "
        "UML diagrams, plots, circuits, state machines, memory layouts. The extracted "
        "text of such a page is not merely thin, it is misleading — an AVL tree comes "
        "back as a bare run of numbers like '14 4 10 3 6 2 4 1', which is the node keys "
        "and heights with the structure that connects them stripped out. Answering from "
        "that produces confident nonsense.\n\n"
        "So when a question turns on structure or on a figure, and when a retrieved "
        "passage reads as scattered labels, stray numbers, or a caption with nothing "
        "under it, call page_image(course, source_file, page) on the page the citation "
        "already gives you and read the image instead. Use page_overview(course, "
        "source_file) to find the figures in a document you do not know without "
        "fetching every page. Prefer looking to guessing: these are the user's own "
        "course notes, and a wrong description of a diagram is worse than none.\n\n"
        "Each course also has notes, which arrive at the top of every search and read. "
        "They hold what the documents do not say — the notation the professor uses, "
        "material that was skipped, how the user wants things cited, a mark scheme "
        "corrected out loud. Treat them as standing instructions about that course and "
        "let them override the documents where they conflict. When the user tells you "
        "something durable of that kind, call remember(course, note) so the next session "
        "starts with it. One fact per call, phrased to survive without today's context. "
        "Do not record what is already in the documents, and do not use it as a "
        "scratchpad: notes are permanent and shown on every later read."
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


def _with_notes(cfg: Config, course: str, body: str) -> str:
    """Put a course's notes above a tool's output.

    Notes ride along on *every* read rather than waiting to be asked for, because the
    things they hold — the professor's notation, material that was skipped, how the user
    wants citations formatted — change how the passages below should be read. A note
    nobody sees until they think to look it up is not memory, it is a file.
    """
    notes = for_prompt(cfg, course)
    return f"{notes}\n\n{body}" if notes else body


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
    if notes := read_notes(cfg, course):
        lines += ["", "Notes on this course (recorded by you or the user):", notes]
    else:
        lines += [
            "",
            "No notes recorded for this course yet. Use remember() for facts its "
            "documents do not state — notation the professor uses, material that was "
            "skipped, how the user wants things cited.",
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
    return _with_notes(cfg, course, f"{header}\n\n{_format(results)}")


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
    return _with_notes(cfg, course, f"{header}\n\n{body}")


@server.tool()
def page_image(course: str, source_file: str, page: int, dpi: int = DEFAULT_DPI) -> list:
    """Look at one page of a document as an image, instead of reading its text.

    Use this when the answer is *drawn* rather than written — an AVL rotation, a graph
    traversal, a B-tree split, an ER or UML diagram, a plotted curve, a circuit. The
    text layer of such a page is usually a title and a few scattered labels, so
    search_course and read_document can return the right page and still tell you almost
    nothing. They cite pages as "module05.pdf p30"; pass that file and page here.

    Typical use: search_course first to find *which* page discusses the thing, then this
    to actually see it. Call page_overview if you need to find the diagrams in a
    document without fetching every page.

    Args:
        course: Course id, as list_courses reports it.
        source_file: Document name, exactly as a citation gives it.
        page: 1-based page number, matching the citations search returns.
        dpi: Resolution, 40-300. The default is readable for slides; raise it for a
            dense figure with small labels, lower it if you only need the layout.
    """
    cfg = load_config()
    try:
        _resolve(cfg, course)
        rendered = render_page(cfg, course, source_file, page, dpi)
    except (CourseUnavailable, RenderError) as exc:
        # Handed back as the result, like every other tool here: an agent can recover
        # from "no course named X; here are the real ones" on its next call, but an
        # exception only ends the turn.
        return [str(exc)]
    # A list of [text, image] reaches the caller as two content blocks: the caption
    # says what was rendered, the image is the page itself.
    return [rendered.summary(), Image(data=rendered.png, format="png")]


@server.tool()
def page_overview(course: str, source_file: str) -> str:
    """Show what is on each page of a document, so you can pick which to look at.

    Renders nothing and costs no image tokens. Use it to locate the diagrams in a long
    deck before calling page_image, rather than fetching pages one at a time.

    Read the numbers *relative to the rest of the document*, not against any absolute
    bar: what counts as drawing-heavy differs enormously between one course's slides and
    another's. A page well above its document's median drawing count, or one carrying
    many short label-like text runs and little prose, is usually a figure.
    """
    cfg = load_config()
    try:
        _resolve(cfg, course)
        stats = page_stats(cfg, course, source_file)
    except (CourseUnavailable, RenderError) as exc:
        return str(exc)
    if not stats:
        return (
            f"{source_file} is not a PDF, so it has no pages to preview. "
            f"Read it with read_document."
        )

    drawings = sorted(s.drawings for s in stats)
    median = drawings[len(drawings) // 2]
    lines = [
        f"{source_file} in {course}: {len(stats)} pages. "
        f"Median vector drawings per page: {median}. Pages well above that, or with "
        f"many short text runs and few characters, are likely diagrams.",
        "",
        f"{'page':>5}  {'chars':>6}  {'drawings':>8}  {'images':>6}  {'text runs':>9}",
    ]
    for s in stats:
        lines.append(
            f"{s.page:>5}  {s.text_chars:>6}  {s.drawings:>8}  {s.images:>6}  "
            f"{s.spans:>4} ({s.short_spans} short)"
        )
    return "\n".join(lines)


@server.tool()
def remember(course: str, note: str) -> str:
    """Record one lasting fact about a course, for every future session to see.

    This is the only tool here that writes anything, and it writes notes — never
    documents, never the index. Use it for what the course materials do not state and
    retrieval therefore cannot find:

    * notation and conventions ("the professor writes n for input size, never N")
    * what the course actually covered ("module 7 was skipped this term")
    * the user's standing preferences ("cite by module number, not filename")
    * facts stated once out loud ("A4 is worth 20%, not the 15% on the syllabus")

    Do not use it for anything already in the documents — that is what search_course is
    for, and duplicating a passage here only makes every future read longer. Do not use
    it as a scratchpad for the current turn; notes are permanent and shown on every
    subsequent search of this course.

    One fact per call, phrased so it still makes sense months from now, in a session
    with none of this context. Notes are appended and dated; an exact repeat of an
    existing note is recognised and not stored twice. Editing or removing a note is
    deliberately not possible here — the user does that with `kb notes` or an editor.
    """
    cfg = load_config()
    try:
        _resolve(cfg, course)
    except CourseUnavailable as exc:
        return str(exc)

    try:
        result = append_note(cfg, course, note)
    except NoteRefused as exc:
        return f"Not recorded: {exc}"
    except OSError as exc:
        return f"Could not write the notes file for {course}: {exc}"

    if not result.added:
        return (
            f"Already known for {course}, so nothing was added:\n  {result.note}\n\n"
            f"Current notes:\n{result.text}"
        )
    return f"Recorded for {course}:\n  {result.note}\n\nCurrent notes:\n{result.text}"


def main() -> None:
    """Run the server on stdio. Entry point for ``python -m courserag.mcp_server``."""
    server.run()


if __name__ == "__main__":
    main()
