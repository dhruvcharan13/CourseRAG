"""The ``kb`` command-line interface.

``ingest`` runs parse -> chunk -> embed -> store; ``search`` runs the same pipeline
in reverse, embedding a query and returning cited chunks; ``eval`` scores that search
against a hand-labelled question set. ``list``/``info``/``chunks`` inspect.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from course_kb import __version__
from course_kb.chunker import chunk_document, estimate_tokens, is_slide_deck, keep_elements
from course_kb.config import Config, load_config
from course_kb.embedding import Embedder, get_embedder
from course_kb.evaluation import (
    EVAL_DIR,
    EVAL_K,
    QueryOutcome,
    first_hit_rank,
    load_eval_set,
    score_run,
    score_spread,
)
from course_kb.manifest import Manifest, manifest_path, read_manifest, write_manifest
from course_kb.parsing import ParsedDocument, get_parser_for
from course_kb.records import ChunkRecord
from course_kb.reranking import get_reranker
from course_kb.retrieval import SearchResult, retrieve
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
        _print_embedder_mismatch(
            course_id, course_dir, manifest, embedder, "nothing was ingested"
        )
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
    #
    # The seen-set carries forward through this batch, not just the stored snapshot:
    # a deck with two identical pages ("Intentionally blank.", a bare "UML Diagram"
    # caption) yields two identical chunks in ONE run, and checking only what was
    # already stored would let both through. Re-ingest hid it — the second run found
    # them stored and reported no-op — so the index looked deduped while holding
    # duplicate vectors that compete for the same top-k slots.
    seen = store.existing_hashes(path.name)
    new_records = []
    for record in records:
        if record.content_hash in seen:
            continue
        seen.add(record.content_hash)
        new_records.append(record)
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
    course_id: str, course_dir: Path, manifest: Manifest, embedder: Embedder, consequence: str
) -> None:
    """Explain an embedder/course mismatch and how to fix it. Nothing was written.

    Shared by ingest and search because the failure is the same one at both ends: a
    query embedded by a different model than the passages lands in a different space,
    where similarity scores are meaningless rather than merely worse. ``consequence``
    is the caller's half of the first line ("nothing was ingested" / "refusing to
    search").
    """
    print(
        f"error: embedder mismatch for course '{course_id}' — {consequence}.\n"
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


def _open_for_search(cfg: Config, course_id: str) -> tuple[CourseStore, Embedder] | None:
    """Resolve a course for querying, or explain why it cannot be queried and return None.

    The embedder check is the same invariant ingest enforces, applied at the other end:
    a query embedded by a different model than the passages lands in a different vector
    space, where the scores are not merely worse but meaningless. Refusing is the only
    honest option — there is no partial answer to give.
    """
    course_dir = _course_dir(cfg, course_id)
    if not manifest_path(course_dir).exists():
        print(f"Course '{course_id}' is not initialized.", file=sys.stderr)
        return None

    manifest = read_manifest(course_dir)
    embedder = get_embedder(cfg.embedder, cfg)
    if (embedder.model_id, embedder.dims) != (manifest.embedding_model, manifest.dims):
        _print_embedder_mismatch(
            course_id, course_dir, manifest, embedder, "refusing to search"
        )
        return None

    if _is_dummy(manifest.embedding_model):
        print(
            "warning: this course was built with dummy vectors, which are random hashes "
            "with no semantic meaning. These rankings are noise, not retrieval.",
            file=sys.stderr,
        )

    return CourseStore.open_or_create(course_dir, manifest.dims), embedder


def _match_source(store: CourseStore, source_file: str) -> str | None:
    """Resolve a document name against the store, or explain what exists on stderr.

    Case-insensitive fallback for the same reason the MCP server has one: names get
    retyped from a citation. Returns ``None`` when the caller should exit non-zero.
    """
    known = store.source_files()
    if source_file in known:
        return source_file
    folded = [name for name in known if name.lower() == source_file.lower()]
    if len(folded) == 1:
        return folded[0]
    if len(folded) > 1:
        print(f"'{source_file}' matches {', '.join(folded)}. Name one exactly.", file=sys.stderr)
        return None
    print(f"No document '{source_file}' in this course. Documents:", file=sys.stderr)
    for name in known:
        print(f"  {name}", file=sys.stderr)
    return None


def cmd_show(args: argparse.Namespace) -> int:
    """Print one document's stored text in page order — a read, not a search.

    Loads no embedding model: reading text back compares no vectors, so this works on a
    course whose embedder is unavailable, and costs nothing to start up.
    """
    cfg = load_config()
    course_dir = cfg.courses_dir / args.course_id
    if not manifest_path(course_dir).exists():
        print(f"No course '{args.course_id}'. Run: kb list", file=sys.stderr)
        return 1

    manifest = read_manifest(course_dir)
    store = CourseStore.open_or_create(course_dir, manifest.dims)
    name = _match_source(store, args.file)
    if name is None:
        return 1

    records = store.read_source(name)
    if args.pages:
        bounds = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+)?)?", args.pages.strip())
        if not bounds:
            print(f"Could not read '{args.pages}' as a page range.", file=sys.stderr)
            return 1
        first = int(bounds.group(1))
        last = int(bounds.group(2)) if bounds.group(2) else (None if "-" in args.pages else first)
        records = [
            r for r in records
            if r.page is not None and r.page >= first and (last is None or r.page <= last)
        ]

    if not records:
        print(f"No passages in {name} for that range.", file=sys.stderr)
        return 1

    print(f"{name} — {args.course_id} ({len(records)} passage(s))\n")
    for record in records:
        page = f"p{record.page}" if record.page is not None else "p?"
        head = f"[{page}] {record.title}" if record.title else f"[{page}]"
        print(head)
        print(record.text)
        print()
    return 0


def _snippet(text: str, width: int = 160) -> str:
    """One-line preview of a chunk: collapsed whitespace, truncated."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _print_results(results: list[SearchResult]) -> None:
    """Ranked, human-readable hits — score first, then the citation, then the text."""
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        page = f"p{chunk.page}" if chunk.page is not None else "p?"
        # Both scales when reranked: the cross-encoder logit decided the order, the
        # cosine says where the dense stage had put it.
        scores = (
            f"{result.score:.3f}"
            if result.rerank_score is None
            else f"ce {result.rerank_score:+.2f} | cos {result.score:.3f}"
        )
        print(f"{rank}. [{scores}] {chunk.source_file} {page}")
        if chunk.title:
            print(f"   {chunk.title}")
        print(f"   {_snippet(chunk.text)}")
        print()


def cmd_search(args: argparse.Namespace) -> int:
    """Embed a query and return the course's top-k most similar chunks, with citations."""
    cfg = load_config()
    opened = _open_for_search(cfg, args.course_id)
    if opened is None:
        return 1
    store, embedder = opened

    if store.count() == 0:
        print(f"Course '{args.course_id}' has no chunks yet. Run: kb ingest ...", file=sys.stderr)
        return 1

    scope = None
    if args.file:
        scope = _match_source(store, args.file)
        if scope is None:
            return 1

    results, _ = retrieve(
        cfg, store, embedder, args.query, args.k,
        use_rerank=args.rerank, candidates=args.candidates or cfg.rerank_candidates,
        source_file=scope,
    )

    if args.json:
        payload = [r.to_dict(rank) for rank, r in enumerate(results, start=1)]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n{len(results)} result(s) for {args.query!r}", file=sys.stderr)
        return 0

    if not results:
        print(f"No results for {args.query!r}.")
        return 0

    print(f"{len(results)} result(s) for {args.query!r}:\n")
    _print_results(results)
    # Scores are informational: no relevance threshold is applied. A meaningful floor is
    # model-specific and has to be calibrated from eval data, not guessed. See `kb eval`.
    return 0


def _print_ab(
    dense: list[QueryOutcome], reranked: list[QueryOutcome], k: int, seconds: list[float]
) -> None:
    """Dense-only vs reranked, arranged so a regression cannot hide behind gross fixes.

    The headline is NET, not the count of things that improved. A reranker that fixes
    four queries and breaks three is worth +1, and the +1 is what gets reported.
    """
    before = score_run(dense, k=k)
    after = score_run(reranked, k=k)
    pairs = list(zip(dense, reranked))
    n = len(pairs)

    fixed = [d.query.id for d, r in pairs if d.rank != 1 and r.rank == 1]
    broke = [d.query.id for d, r in pairs if d.rank == 1 and r.rank != 1]
    both_hit = sum(1 for d, r in pairs if d.rank == 1 and r.rank == 1)
    neither = sum(1 for d, r in pairs if d.rank != 1 and r.rank != 1)

    print(f"NET rank-1 change: {len(fixed) - len(broke):+d}   "
          f"(fixed {len(fixed)}, broke {len(broke)})")
    print()
    print("  movement matrix — every query lands in exactly one cell")
    print(f"    {'':<22}{'reranked rank-1':>16}{'reranked not':>15}")
    print(f"    {'dense rank-1':<22}{both_hit:>16}{len(broke):>15}   <- broke")
    print(f"    {'dense not rank-1':<22}{len(fixed):>16}{neither:>15}")
    print(f"    {'':<22}{'^ fixed':>16}")
    print()

    print(f"  {'metric':<12}{'dense':>9}{'reranked':>11}{'delta':>9}")
    for cutoff in sorted(before.recall):
        b, a = before.recall[cutoff], after.recall[cutoff]
        print(f"  {'recall@' + str(cutoff):<12}{b:>8.1%}{a:>11.1%}{a - b:>+9.1%}")
    print(f"  {'MRR@' + str(k):<12}{before.mrr:>8.3f}{after.mrr:>11.3f}{after.mrr - before.mrr:>+9.3f}")
    print()

    # Regressions first: a demotion is the expensive kind of change, so it is not
    # allowed to sit below a list of wins where it can be skimmed past.
    def delta(d: QueryOutcome, r: QueryOutcome) -> int:
        lo = d.rank if d.rank is not None else k + 1
        hi = r.rank if r.rank is not None else k + 1
        return hi - lo

    moved = [(d, r) for d, r in pairs if d.rank != r.rank]
    if moved:
        print(f"  rank movements ({len(moved)} of {n} queries), worst regression first:")
        for d, r in sorted(moved, key=lambda t: -delta(*t)):
            arrow = "WORSE" if delta(d, r) > 0 else "better"
            print(f"    {d.query.id:<6}{str(d.rank or '>k'):>4} -> {str(r.rank or '>k'):<4} {arrow}")
    else:
        print("  no query changed rank.")
    print()

    if seconds:
        ordered = sorted(seconds)
        print(f"  latency/query: median {1000 * ordered[len(ordered) // 2]:.0f} ms, "
              f"max {1000 * ordered[-1]:.0f} ms, total {sum(seconds):.1f} s "
              f"(includes the one-off model load)")
        print()


def cmd_eval(args: argparse.Namespace) -> int:
    """Score search against a hand-labelled eval set: recall@k, MRR, and every miss."""
    cfg = load_config()
    course_id = args.course_id

    eval_path = Path(args.set) if args.set else Path(EVAL_DIR) / f"{course_id}.json"
    if not eval_path.is_file():
        print(
            f"No eval set at {eval_path}. Write one, or pass --set PATH.",
            file=sys.stderr,
        )
        return 1
    eval_set = load_eval_set(eval_path)

    if eval_set.course != course_id:
        print(
            f"error: {eval_path} is an eval set for '{eval_set.course}', not '{course_id}'.",
            file=sys.stderr,
        )
        return 1

    opened = _open_for_search(cfg, course_id)
    if opened is None:
        return 1
    store, embedder = opened

    k = args.k
    candidates = args.candidates or cfg.rerank_candidates
    # Every query runs once at the deepest cutoff; the shallower ones are prefixes of
    # the same ranking, so recall@1..@k costs no extra searches.
    #
    # With --rerank the dense ranking is stage one anyway, so both rankings come out of
    # the same pass and the A/B is free. Scoring both is the whole point: a reranker is
    # only worth keeping net of what it demotes, and that is invisible from the
    # reranked numbers alone.
    outcomes, dense_outcomes = [], []
    rerank_seconds = []
    for q in eval_set.queries:
        started = time.perf_counter()
        final, dense = retrieve(
            cfg, store, embedder, q.query, k,
            use_rerank=args.rerank, candidates=candidates,
        )
        rerank_seconds.append(time.perf_counter() - started)
        outcomes.append(QueryOutcome(query=q, results=final, rank=first_hit_rank(final, q.gold)))
        dense_outcomes.append(
            QueryOutcome(query=q, results=dense, rank=first_hit_rank(dense, q.gold))
        )
    report = score_run(outcomes, k=k)

    print(f"Eval:            {eval_path}")
    print(f"Course:          {course_id} ({store.count()} chunks)")
    print(f"Embedding model: {embedder.model_id}")
    if args.rerank:
        print(f"Reranker:        {get_reranker(cfg.reranker, cfg).model_id} "
              f"(top-{max(candidates, k)} candidates)")
    print(f"Queries:         {len(outcomes)}")
    print()

    if args.rerank:
        _print_ab(dense_outcomes, outcomes, k, rerank_seconds)
    for cutoff, value in sorted(report.recall.items()):
        hits = round(value * len(outcomes))
        print(f"  recall@{cutoff:<3}{value:6.1%}   ({hits}/{len(outcomes)})")
    print(f"  MRR@{k:<5}{report.mrr:6.3f}")
    print()
    print("  recall@k here is a hit-rate: each query has one intended answer, labelled")
    print("  with the page(s) it may legitimately appear on. MRR is truncated at k, so")
    print(f"  an answer ranked below {k} scores 0 rather than 1/rank.")
    print()

    # Per-query ranks, not just the aggregate: a query answered at rank 3 is a success by
    # recall@5 and a near-failure in practice, and the summary alone cannot tell them
    # apart. This is also the per-query score data a Phase 4 threshold gets fitted to.
    print("Per query (rank of the correct chunk, its score, and the top-1 score):")
    print(f"  {'id':<5}{'rank':>5}{'gold':>8}{'top1':>8}   query")
    for outcome in outcomes:
        rank = str(outcome.rank) if outcome.hit_at(k) else f">{k}"
        gold_score = f"{outcome.gold_score:.3f}" if outcome.hit_at(k) else "—"
        top_score = f"{outcome.top_score:.3f}" if outcome.top_score is not None else "—"
        print(
            f"  {outcome.query.id:<5}{rank:>5}{gold_score:>8}{top_score:>8}   "
            f"{_snippet(outcome.query.query, 60)}"
        )
    print()

    hit_top = [o.top_score for o in outcomes if o.hit_at(k) and o.top_score is not None]
    miss_top = [o.top_score for o in outcomes if not o.hit_at(k) and o.top_score is not None]
    gold = [o.gold_score for o in outcomes if o.hit_at(k) and o.gold_score is not None]
    print("Scores (min / median / max) — reported, not thresholded:")
    print(f"  top-1, hits:   {score_spread(hit_top)}")
    print(f"  top-1, misses: {score_spread(miss_top)}")
    print(f"  correct chunk: {score_spread(gold)}")
    print()

    if not report.misses:
        print("No misses.")
        return 0

    print(f"Misses ({len(report.misses)}/{len(outcomes)}):")
    for outcome in report.misses:
        expected = ", ".join(f"{f} p{p}" for f, p in sorted(outcome.query.gold))
        print(f"\n  [{outcome.query.id}] {outcome.query.query}")
        print(f"    expected: {expected}")
        if outcome.query.note:
            print(f"    note:     {outcome.query.note}")
        print(f"    got:      (top {min(args.show, len(outcome.results))} of {len(outcome.results)})")
        for rank, result in enumerate(outcome.results[: args.show], start=1):
            chunk = result.chunk
            page = f"p{chunk.page}" if chunk.page is not None else "p?"
            print(
                f"      {rank}. [{result.score:.3f}] {chunk.source_file} {page} — "
                f"{_snippet(chunk.text, 80)}"
            )
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


def _add_rerank_flags(parser: argparse.ArgumentParser) -> None:
    """Opt-in reranking, shared by ``search`` and ``eval``.

    Off by default: dense-only is the measured baseline, and it stays the default until
    the numbers say otherwise. The flag is the switch; ``config.reranker`` only chooses
    which model runs when the switch is on.
    """
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rescore the top dense candidates with a cross-encoder (needs the [local] extra).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Dense candidates to rerank (default: config rerank_candidates, 20).",
    )


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

    p_search = sub.add_parser("search", help="Retrieve a course's chunks by similarity.")
    p_search.add_argument("course_id")
    p_search.add_argument("query", help="What to search for, in natural language.")
    p_search.add_argument("-k", type=int, default=5, help="How many results to return (default: 5).")
    p_search.add_argument("--json", action="store_true", help="Emit JSON on stdout for piping.")
    p_search.add_argument(
        "--file", help="Restrict the search to one source document (as shown by kb info)."
    )
    _add_rerank_flags(p_search)
    p_search.set_defaults(func=cmd_search)

    p_show = sub.add_parser("show", help="Print one document's stored text in page order.")
    p_show.add_argument("course_id")
    p_show.add_argument("file", help="Source document, as shown by kb info.")
    p_show.add_argument("--pages", help='Page range: "12", "1-20", or "25-".')
    p_show.set_defaults(func=cmd_show)

    p_eval = sub.add_parser("eval", help="Score search against a labelled eval set.")
    p_eval.add_argument("course_id")
    p_eval.add_argument("--set", help=f"Eval set path (default: {EVAL_DIR}/<course>.json).")
    p_eval.add_argument("-k", type=int, default=EVAL_K, help=f"Ranking depth (default: {EVAL_K}).")
    p_eval.add_argument(
        "--show", type=int, default=3, help="Results to print per miss (default: 3)."
    )
    _add_rerank_flags(p_eval)
    p_eval.set_defaults(func=cmd_eval)

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
