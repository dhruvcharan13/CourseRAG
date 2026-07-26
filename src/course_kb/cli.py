"""The ``kb`` command-line interface.

Wires the Phase 0 contracts together end to end. ``init-course`` and ``ingest``
are fully working (ingest proves parse -> chunk -> embed -> store); ``list`` and
``info`` inspect existing courses; ``search`` is a Phase 3 stub.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from course_kb import __version__
from course_kb.chunker import chunk_document, estimate_tokens, is_slide_deck, keep_elements
from course_kb.config import Config, load_config
from course_kb.embedding import Embedder, get_embedder
from course_kb.manifest import Manifest, manifest_path, read_manifest, write_manifest
from course_kb.parsing import ParsedDocument, get_parser_for
from course_kb.records import ChunkRecord
from course_kb.store import CourseStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _course_dir(cfg: Config, course_id: str) -> Path:
    return cfg.courses_dir / course_id


def _is_dummy(model_id: str) -> bool:
    return model_id.startswith("dummy-")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_init_course(args: argparse.Namespace) -> int:
    cfg = load_config()
    course_id = args.course_id
    course_dir = _course_dir(cfg, course_id)

    if manifest_path(course_dir).exists():
        print(f"Course '{course_id}' already exists at {course_dir}", file=sys.stderr)
        return 1

    # Resolve the embedder first: it fixes the course's vector width, and if it is
    # unavailable (missing extra, unknown name) nothing should be created at all.
    embedder = get_embedder(cfg.embedder, cfg)

    # Folder tree.
    (course_dir / "raw").mkdir(parents=True, exist_ok=True)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    # Empty per-course memory file.
    (course_dir / "COURSE.md").write_text("", encoding="utf-8")

    # Empty LanceDB table, sized to the embedder's dimensions.
    CourseStore.open_or_create(course_dir, embedder.dims)

    # Manifest recording how this course was built.
    manifest = Manifest(
        course=course_id,
        embedding_model=embedder.model_id,
        dims=embedder.dims,
        categories=[],
        files=[],
        last_indexed=None,
    )
    write_manifest(course_dir, manifest)

    print(f"Initialized course '{course_id}' at {course_dir}")
    print(f"  embedder: {embedder.model_id} (dims={embedder.dims})")
    if _is_dummy(embedder.model_id):
        print(
            "note: dummy vectors carry no semantic meaning. For real embeddings:\n"
            '  pip install -e ".[local]"   (or set embedder = "local" in config.toml)',
            file=sys.stderr,
        )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = load_config()
    course_id = args.course_id
    course_dir = _course_dir(cfg, course_id)

    if not manifest_path(course_dir).exists():
        print(f"Course '{course_id}' is not initialized. Run: kb init-course {course_id}", file=sys.stderr)
        return 1

    path = Path(args.path)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    # A course's embedding model is fixed at init-course: vectors from different
    # models are not comparable, and a table's vector width cannot change. Check
    # this before parsing, embedding, or opening the table, so a mismatched run
    # changes nothing on disk.
    manifest = read_manifest(course_dir)
    embedder = get_embedder(cfg.embedder, cfg)
    if (embedder.model_id, embedder.dims) != (manifest.embedding_model, manifest.dims):
        _print_embedder_mismatch(course_id, course_dir, manifest, embedder)
        return 1

    store = CourseStore.open_or_create(course_dir, manifest.dims)

    parser = get_parser_for(path)
    doc = parser.parse(path)

    added_at = _now_iso()
    records = chunk_document(
        doc,
        course=course_id,
        source_file=path.name,
        category=args.category,
        cfg=cfg,
        added_at=added_at,
    )

    # A file that yields nothing is not "already up to date" — say so, since the
    # usual cause is an image-only scan that silently indexes as an empty document.
    if not records:
        kept = keep_elements(doc.elements, cfg)
        dropped = len(doc.elements) - len(kept)
        hint = (
            "An image-only or scanned PDF needs OCR before it can be indexed."
            if not kept
            else "Text was parsed, but no chunk survived the chunker's minimum size."
        )
        print(
            f"error: no chunks produced from {path.name} — nothing was ingested.\n"
            f"  {len(doc.elements)} page(s) parsed, {dropped} dropped as near-empty "
            f"(< {cfg.min_element_chars} non-space chars).\n"
            f"  {hint}",
            file=sys.stderr,
        )
        return 1

    # Skip chunks already stored for THIS file (by content hash) so re-ingest is
    # a no-op, while identical slides shared across files are kept per file.
    existing = store.existing_hashes(path.name)
    new_records = [r for r in records if r.content_hash not in existing]
    if not new_records:
        print(
            f"No new chunks from {path.name}; already up to date in "
            f"'{course_id}' (total: {store.count()})"
        )
        return 0

    # Warn (non-fatal) about chunks the embedder will truncate.
    _warn_oversize(embedder, new_records, cfg)

    # Embed (one vector per chunk), then store.
    vectors = embedder.embed([r.text for r in new_records])
    for record, vector in zip(new_records, vectors):
        record.vector = vector
    store.add(new_records)

    # Update the manifest (read before the embedder check above).
    if args.category not in manifest.categories:
        manifest.categories.append(args.category)
    if path.name not in manifest.files:
        manifest.files.append(path.name)
    manifest.last_indexed = added_at
    write_manifest(course_dir, manifest)

    print(
        f"Ingested {len(new_records)} chunk(s) from {path.name} into "
        f"'{course_id}' (total: {store.count()})"
    )
    return 0


def _shown_ids(records: list[ChunkRecord], limit: int = 5) -> str:
    return ", ".join(r.id for r in records[:limit]) + (" ..." if len(records) > limit else "")


def _warn_oversize(embedder: Embedder, records: list[ChunkRecord], cfg: Config) -> None:
    """Warn about chunks whose tails the embedder will drop.

    An embedder that can tokenize (the real local model) gives an exact count, and the
    consequence is concrete: an oversize chunk's vector is identical to the vector of
    its head, so its tail is unsearchable even though the stored text and citations
    cover the whole chunk.

    Without a tokenizer, fall back to ``estimate_tokens`` (~4 chars/token). Measured over
    6.7k real chunks that proxy misses ~94% of actual truncations, because notation-dense
    text (math, SQL, code) can run past 1 token per character — so it is reported as the
    estimate it is. See docs/chunking-robustness.md.
    """
    count_tokens = getattr(embedder, "count_tokens", None)
    if count_tokens is None:
        oversize = [r for r in records if estimate_tokens(r.text) > cfg.warn_chunk_tokens]
        if oversize:
            print(
                f"warning: {len(oversize)} chunk(s) exceed ~{cfg.warn_chunk_tokens} tokens by "
                f"character estimate and may be truncated by a token-limited embedder "
                f"(estimate only; under-counts math/code): {_shown_ids(oversize)}",
                file=sys.stderr,
            )
        return

    limit = embedder.max_input_tokens
    counts = count_tokens([r.text for r in records])
    oversize = [(r, n) for r, n in zip(records, counts) if n > limit]
    if not oversize:
        return
    worst = max(n for _, n in oversize)
    print(
        f"warning: {len(oversize)} of {len(records)} chunk(s) exceed the model's "
        f"{limit}-token limit (largest: {worst} tokens). Only their first ~{limit} tokens "
        f"are embedded, so the tail of each is not searchable; the full text is still "
        f"stored and cited: {_shown_ids([r for r, _ in oversize])}",
        file=sys.stderr,
    )


def _print_embedder_mismatch(
    course_id: str, course_dir: Path, manifest: Manifest, embedder: Embedder
) -> None:
    """Explain an embedder/course mismatch and how to fix it. Nothing was written."""
    print(
        f"error: embedder mismatch for course '{course_id}' — nothing was ingested.\n"
        f"  course was built with: {manifest.embedding_model} (dims={manifest.dims})\n"
        f"  current config selects: {embedder.model_id} (dims={embedder.dims})\n"
        f"Vectors from different models are not comparable, and a course's vector width "
        f"is fixed when its table is created.\n"
        f"Either restore the original embedder in config.toml, or rebuild the course "
        f"from your source files:\n"
        f"  rm -rf {course_dir}\n"
        f"  kb init-course {course_id}\n"
        f"  kb ingest {course_id} <file> --category <category>   # for each file",
        file=sys.stderr,
    )
    if manifest.files:
        print(f"Files to re-ingest: {', '.join(manifest.files)}", file=sys.stderr)


def cmd_delete(args: argparse.Namespace) -> int:
    """Remove every chunk that came from one source file.

    Deliberately does **not** construct an embedder: a vector is a pure function of
    chunk text, so deletion needs no model. That also means a course can be pruned even
    when its embedding model is unavailable (missing extra, or a course built with a
    model the current config no longer selects).
    """
    cfg = load_config()
    course_id = args.course_id
    course_dir = _course_dir(cfg, course_id)

    if not manifest_path(course_dir).exists():
        print(f"Course '{course_id}' is not initialized.", file=sys.stderr)
        return 1

    manifest = read_manifest(course_dir)
    store = CourseStore.open_or_create(course_dir, manifest.dims)

    removed = store.delete_source(args.file)
    if removed == 0 and args.file not in manifest.files:
        known = ", ".join(manifest.files) or "(none)"
        print(
            f"error: no chunks from '{args.file}' in course '{course_id}'. "
            f"Names must match exactly.\nFiles in this course: {known}",
            file=sys.stderr,
        )
        return 1

    # The manifest's file and category lists are derived from the rows, so recompute
    # rather than patch: a category may have been used only by the deleted file.
    # Filtering (instead of rebuilding) preserves the existing ingest order.
    remaining_categories = set(store.categories())
    manifest.files = [f for f in manifest.files if f != args.file]
    manifest.categories = [c for c in manifest.categories if c in remaining_categories]
    # last_indexed records when content was last *indexed*; a deletion does not index.
    write_manifest(course_dir, manifest)

    if removed == 0:
        print(
            f"No stored chunks from {args.file}; removed its stale manifest entry "
            f"in '{course_id}' (total: {store.count()})"
        )
        return 0

    print(
        f"Deleted {removed} chunk(s) from {args.file} in '{course_id}' "
        f"(remaining: {store.count()})"
    )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_config()

    def _course_ids(base: Path) -> list[str]:
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    active = _course_ids(cfg.courses_dir)
    archived = _course_ids(cfg.archive_dir)

    print("Active courses:")
    if active:
        for cid in active:
            print(f"  {cid}")
    else:
        print("  (none)")

    print("Archived courses:")
    if archived:
        for cid in archived:
            print(f"  {cid}")
    else:
        print("  (none)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = load_config()
    course_id = args.course_id
    course_dir = _course_dir(cfg, course_id)

    if not manifest_path(course_dir).exists():
        print(f"Course '{course_id}' is not initialized.", file=sys.stderr)
        return 1

    manifest = read_manifest(course_dir)
    store = CourseStore.open_or_create(course_dir, manifest.dims)

    print(f"Course:          {manifest.course}")
    print(f"Embedding model: {manifest.embedding_model}")
    print(f"Dimensions:      {manifest.dims}")
    print(f"Categories:      {', '.join(manifest.categories) or '(none)'}")
    print(f"Files:           {', '.join(manifest.files) or '(none)'}")
    print(f"Last indexed:    {manifest.last_indexed or '(never)'}")
    print(f"Chunk count:     {store.count()}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    print("not implemented (Phase 3)")
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    """Parse + chunk a file and print the chunks as JSON — no embed, no store."""
    cfg = load_config()
    path = Path(args.path)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1

    parser = get_parser_for(path)
    doc = parser.parse(path)
    records = chunk_document(
        doc,
        course=args.course_id,
        source_file=path.name,
        category="(dry-run)",
        cfg=cfg,
        added_at=_now_iso(),
    )

    if args.report:
        return _print_chunk_report(path, doc, records, cfg)

    payload = [
        {
            "id": r.id,
            "page": r.page,
            "title": r.title,
            "char_range": list(r.char_range) if r.char_range is not None else None,
            "content_hash": r.content_hash,
            "chars": len(r.text),
            "text": r.text,
        }
        for r in records
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n{len(records)} chunk(s) from {path.name}", file=sys.stderr)
    return 0


def _print_chunk_report(
    path: Path, doc: ParsedDocument, records: list[ChunkRecord], cfg: Config
) -> int:
    """Print a health summary for a parsed+chunked file — the 'is this file healthy' gate."""
    kept = keep_elements(doc.elements, cfg)
    dropped = len(doc.elements) - len(kept)
    lengths = sorted(len(r.text) for r in records)
    titled = sum(1 for r in records if r.title)
    oversize = sum(1 for r in records if estimate_tokens(r.text) > cfg.warn_chunk_tokens)
    mode = "slide" if kept and is_slide_deck(kept) else ("prose" if kept else "empty")
    pct_titled = round(100 * titled / len(records)) if records else 0

    print(f"file:              {path.name}")
    print(f"mode:              {mode}")
    print(f"pages parsed:      {len(doc.elements)}")
    print(f"near-empty pages:  {dropped} dropped (< {cfg.min_element_chars} non-space chars)")
    print(f"chunks:            {len(records)}")
    print(f"titled chunks:     {titled}/{len(records)} ({pct_titled}%)")
    if lengths:
        print(f"chunk chars:       min {lengths[0]} / median {int(statistics.median(lengths))} / max {lengths[-1]}")
    print(
        f"oversize chunks:   {oversize} (est. > ~{cfg.warn_chunk_tokens} tokens from chars/4; "
        f"a rough hint — ingest counts real tokens)"
    )
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="Per-course RAG knowledge base.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-course", help="Create an isolated store for a course.")
    p_init.add_argument("course_id")
    p_init.set_defaults(func=cmd_init_course)

    p_ingest = sub.add_parser("ingest", help="Parse, chunk, embed, and store a file.")
    p_ingest.add_argument("course_id")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--category", required=True, help="Category label for the chunks (e.g. notes).")
    p_ingest.set_defaults(func=cmd_ingest)

    p_delete = sub.add_parser(
        "delete", help="Remove all chunks that came from one source file."
    )
    p_delete.add_argument("course_id")
    p_delete.add_argument("file", help="Source file name as stored (see kb info).")
    p_delete.set_defaults(func=cmd_delete)

    p_list = sub.add_parser("list", help="List active and archived courses.")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="Show a course's manifest and chunk count.")
    p_info.add_argument("course_id")
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="Retrieve chunks (Phase 3).")
    p_search.add_argument("query", nargs="*", help="Search query (ignored in Phase 0).")
    p_search.set_defaults(func=cmd_search)

    p_chunks = sub.add_parser(
        "chunks", help="Parse + chunk a file and print chunks as JSON (no embed/store)."
    )
    p_chunks.add_argument("course_id")
    p_chunks.add_argument("path")
    p_chunks.add_argument(
        "--report",
        action="store_true",
        help="Print a health summary instead of per-chunk JSON.",
    )
    p_chunks.set_defaults(func=cmd_chunks)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, NotImplementedError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
