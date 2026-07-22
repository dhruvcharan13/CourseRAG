"""The ``kb`` command-line interface.

Wires the Phase 0 contracts together end to end. ``init-course`` and ``ingest``
are fully working (ingest proves parse -> chunk -> embed -> store); ``list`` and
``info`` inspect existing courses; ``search`` is a Phase 3 stub.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from course_kb import __version__
from course_kb.config import Config, load_config
from course_kb.embedding import get_embedder
from course_kb.manifest import Manifest, manifest_path, read_manifest, write_manifest
from course_kb.parsing import get_parser_for
from course_kb.records import ChunkRecord
from course_kb.store import CourseStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _course_dir(cfg: Config, course_id: str) -> Path:
    return cfg.courses_dir / course_id


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

    # Folder tree.
    (course_dir / "raw").mkdir(parents=True, exist_ok=True)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    # Empty per-course memory file.
    (course_dir / "COURSE.md").write_text("", encoding="utf-8")

    # Empty LanceDB table, sized to the embedder's dimensions.
    embedder = get_embedder(cfg.embedder, cfg)
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

    parser = get_parser_for(path)
    doc = parser.parse(path)

    added_at = _now_iso()
    records: list[ChunkRecord] = []
    for i, element in enumerate(doc.elements):
        page_part = f"p{element.page}" if element.page is not None else "p0"
        records.append(
            ChunkRecord(
                id=f"{course_id}::{path.name}::{page_part}::c{i}",
                text=element.text,
                course=course_id,
                source_file=path.name,
                category=args.category,
                content_hash=ChunkRecord.hash_text(element.text),
                added_at=added_at,
                title=element.title,
                module=None,
                page=element.page,
                char_range=(0, len(element.text)),
            )
        )

    # Embed (one vector per chunk), then store.
    embedder = get_embedder(cfg.embedder, cfg)
    vectors = embedder.embed([r.text for r in records])
    for record, vector in zip(records, vectors):
        record.vector = vector

    store = CourseStore.open_or_create(course_dir, embedder.dims)
    store.add(records)

    # Update the manifest.
    manifest = read_manifest(course_dir)
    if args.category not in manifest.categories:
        manifest.categories.append(args.category)
    if path.name not in manifest.files:
        manifest.files.append(path.name)
    manifest.last_indexed = added_at
    write_manifest(course_dir, manifest)

    print(f"Ingested {len(records)} chunk(s) from {path.name} into '{course_id}' (total: {store.count()})")
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

    p_list = sub.add_parser("list", help="List active and archived courses.")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="Show a course's manifest and chunk count.")
    p_info.add_argument("course_id")
    p_info.set_defaults(func=cmd_info)

    p_search = sub.add_parser("search", help="Retrieve chunks (Phase 3).")
    p_search.add_argument("query", nargs="*", help="Search query (ignored in Phase 0).")
    p_search.set_defaults(func=cmd_search)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
