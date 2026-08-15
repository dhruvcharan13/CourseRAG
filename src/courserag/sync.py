"""Reconcile a course's source folder with its index.

Each course owns a ``raw/`` directory — created by ``init-course`` since the first
commit, and until now never written to. This module makes it the source of truth:
what is in ``raw/`` is what is in the index, in both directions. Drop a PDF in and
sync ingests it; delete one and sync removes its chunks. That is the whole reason a
file manager — Finder, the web UI, anything — can serve as the front-end: none of
them need to know what a chunk is, only how to move files.

Layout inside ``raw/`` carries one piece of meaning: a subdirectory names the
category its files belong to. ``raw/lecture/05.pdf`` ingests as category ``lecture``;
a file loose at the top gets ``cfg.default_category``. Nothing else about the tree is
interpreted.

Change detection lives in a sidecar ``sync.json`` rather than the manifest, because it
is bookkeeping about the folder, not metadata about the course — and because the
manifest's ``files`` list is a documented shape other tools read. Without it a mirror
would be only half honest: content-hash dedup makes re-ingesting an *edited* file add
its new chunks while its stale ones linger, so a changed file has to be dropped and
re-ingested, and that requires knowing it changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from courserag.config import Config
from courserag.ingest import (
    DeleteResult,
    IngestError,
    IngestResult,
    course_dir,
    delete_file,
    ingest_file,
    open_course,
    resolve_embedder,
)
from courserag.parsing import supported_extensions

SOURCE_DIR_NAME = "raw"
STATE_NAME = "sync.json"

#: Files a file manager leaves lying around that are never course material.
_IGNORED_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


def source_root(cfg: Config, course_id: str) -> Path:
    """The folder whose contents define what the course indexes."""
    return course_dir(cfg, course_id) / SOURCE_DIR_NAME


# --------------------------------------------------------------------------- #
# Scanning the folder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceFile:
    """One file found under ``raw/``."""

    path: Path
    #: Name as the index knows it. Chunks are keyed by basename, not by path.
    name: str
    #: Path relative to ``raw/``, for display.
    rel: str
    category: str
    size: int
    mtime_ns: int

    def digest(self) -> str:
        """SHA-256 of the file's bytes."""
        h = hashlib.sha256()
        with self.path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()


@dataclass
class ScanResult:
    """What is currently under ``raw/``, plus what could not be used."""

    files: dict[str, SourceFile] = field(default_factory=dict)
    #: Extensions with no registered parser: reported, then skipped.
    unsupported: list[Path] = field(default_factory=list)
    #: Basename collisions across category folders, as ``name -> [rel, rel, ...]``.
    #: Chunks are keyed by basename, so two files sharing one would silently share an
    #: identity in the index; both are skipped rather than guessed at.
    duplicates: dict[str, list[str]] = field(default_factory=dict)


def scan_sources(cfg: Config, course_id: str) -> ScanResult:
    """Walk ``raw/`` and classify everything in it."""
    root = source_root(cfg, course_id)
    result = ScanResult()
    if not root.is_dir():
        return result

    extensions = supported_extensions()
    seen: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Skip dotfiles and anything inside a dot-directory: editors, sync clients and
        # version control all keep state there, and none of it is course material.
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts) or path.name in _IGNORED_NAMES:
            continue
        if path.suffix.lower() not in extensions:
            result.unsupported.append(path)
            continue
        seen.setdefault(path.name, []).append(path)

    for name, paths in seen.items():
        if len(paths) > 1:
            result.duplicates[name] = [str(p.relative_to(root)) for p in paths]
            continue
        path = paths[0]
        rel = path.relative_to(root)
        parent = rel.parent
        category = cfg.default_category if parent == Path(".") else parent.as_posix()
        stat = path.stat()
        result.files[name] = SourceFile(
            path=path,
            name=name,
            rel=rel.as_posix(),
            category=category,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )
    return result


# --------------------------------------------------------------------------- #
# Sidecar state
# --------------------------------------------------------------------------- #


def state_path(cfg: Config, course_id: str) -> Path:
    return course_dir(cfg, course_id) / STATE_NAME


def load_state(cfg: Config, course_id: str) -> dict[str, dict]:
    """Per-file sync bookkeeping, keyed by source file name. Missing file -> empty."""
    path = state_path(cfg, course_id)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Bookkeeping, not truth. A corrupt sidecar costs a re-hash, not a failure.
        return {}
    entries = data.get("files", {})
    return entries if isinstance(entries, dict) else {}


def save_state(cfg: Config, course_id: str, entries: dict[str, dict]) -> None:
    path = state_path(cfg, course_id)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"files": entries}, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _entry_for(src: SourceFile, digest: str, chunks: int | None = None) -> dict:
    entry = {
        "rel": src.rel,
        "category": src.category,
        "size": src.size,
        "mtime_ns": src.mtime_ns,
        "digest": digest,
    }
    if chunks is not None:
        entry["chunks"] = chunks
    return entry


def _unchanged(src: SourceFile, entry: dict | None) -> bool:
    """Whether ``src`` can be assumed identical to what was last synced.

    Size and mtime are checked first purely to avoid re-reading megabytes of PDF on
    every sync; the digest is what actually decides, and is only computed when the
    cheap check is inconclusive.
    """
    if not entry:
        return False
    if entry.get("size") == src.size and entry.get("mtime_ns") == src.mtime_ns:
        return True
    return entry.get("digest") == src.digest()


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


@dataclass
class SyncPlan:
    """What a sync would do. Computing one changes nothing."""

    course_id: str
    root: Path
    add: list[SourceFile] = field(default_factory=list)
    #: On disk and indexed, but the bytes changed: delete the old chunks, ingest again.
    reindex: list[SourceFile] = field(default_factory=list)
    #: Indexed but gone from disk.
    remove: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    #: Category moved (file dragged between folders) — recategorized via reindex.
    unsupported: list[Path] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.add or self.reindex or self.remove)

    @property
    def wipes_everything(self) -> bool:
        """True when applying this would empty a non-empty index.

        The signature of a source folder that vanished — an unmounted drive, a
        half-synced cloud directory, a course whose ``raw/`` was never populated —
        rather than of someone deliberately clearing a course.
        """
        return bool(self.remove) and not self.add and not self.reindex and not self.unchanged


def plan_sync(cfg: Config, course_id: str) -> SyncPlan:
    """Diff ``raw/`` against the index. Loads no embedding model.

    Raises:
        CourseNotInitialized: if the course has no manifest.
    """
    manifest, _store = open_course(cfg, course_id)
    scan = scan_sources(cfg, course_id)
    state = load_state(cfg, course_id)
    indexed = set(manifest.files)

    plan = SyncPlan(
        course_id=course_id,
        root=source_root(cfg, course_id),
        unsupported=scan.unsupported,
        duplicates=scan.duplicates,
    )

    for name, src in sorted(scan.files.items()):
        if name not in indexed:
            plan.add.append(src)
            continue
        entry = state.get(name)
        # A file the index knows but the sidecar does not is an adoption: it was
        # ingested before sync existed, or by `kb ingest` from somewhere else. Treat it
        # as unchanged and record it, rather than re-ingesting the whole corpus once.
        if entry is None:
            plan.unchanged.append(name)
        elif _unchanged(src, entry) and entry.get("category") == src.category:
            plan.unchanged.append(name)
        else:
            plan.reindex.append(src)

    plan.remove = sorted(indexed - set(scan.files))
    return plan


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #


class SyncRefused(Exception):
    """A guard stopped the sync before it changed anything."""


@dataclass
class SyncReport:
    """What a sync actually did."""

    course_id: str
    added: list[IngestResult] = field(default_factory=list)
    reindexed: list[IngestResult] = field(default_factory=list)
    removed: list[DeleteResult] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    #: Files that raised during ingest, as ``(name, message)``. One bad PDF does not
    #: abort the batch — the other twenty still index, and the failures are reported.
    failed: list[tuple[str, str]] = field(default_factory=list)
    unsupported: list[Path] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    total_chunks: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.reindexed or self.removed)


#: Called as ``progress(stage, name, index, total)`` before each file is processed,
#: so a UI can show what is happening during a slow batch. ``stage`` is one of
#: ``"ingest"``, ``"reindex"``, ``"remove"``.
ProgressFn = Callable[[str, str, int, int], None]


def apply_sync(
    cfg: Config,
    course_id: str,
    plan: SyncPlan | None = None,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> SyncReport:
    """Make the index match ``raw/``.

    Removals happen first: they need no model, they free the names that a rename would
    otherwise collide with, and doing them before the slow half means an interrupted
    sync leaves the index smaller rather than doubled.

    Raises:
        SyncRefused: if the source folder is missing, or the plan would empty a
            non-empty index, and ``force`` is not set.
        CourseNotInitialized, EmbedderMismatch: from the underlying course open.
    """
    if plan is None:
        plan = plan_sync(cfg, course_id)

    root = source_root(cfg, course_id)
    if not root.is_dir() and not force:
        raise SyncRefused(
            f"source folder {root} does not exist — refusing to treat that as "
            f"'every file was deleted'. Create it, or pass force to sync anyway."
        )
    if plan.wipes_everything and not force:
        raise SyncRefused(
            f"syncing would remove all {len(plan.remove)} indexed file(s) from "
            f"'{course_id}' and add nothing, because {root} is empty. That is almost "
            f"always a missing folder rather than an intentional wipe. Pass force if "
            f"you meant it."
        )

    report = SyncReport(
        course_id=course_id,
        unchanged=list(plan.unchanged),
        unsupported=list(plan.unsupported),
        duplicates=dict(plan.duplicates),
    )
    state = load_state(cfg, course_id)

    # Deletions need no embedder, so do them before resolving one: a course whose
    # model is unavailable can still be pruned.
    #
    # A reindex is a delete followed by an ingest, because content-hash dedup would
    # otherwise add an edited file's new chunks while leaving its stale ones behind.
    # That delete is a step of the reindex, not a removal in its own right, so it is
    # not reported as one — the file is still in the course when this finishes.
    gone = set(plan.remove)
    removals = plan.remove + [s.name for s in plan.reindex]
    for i, name in enumerate(removals, 1):
        if progress:
            progress("remove", name, i, len(removals))
        try:
            result = delete_file(cfg, course_id, name)
        except KeyError:
            pass  # Already absent; the mirror wanted it gone either way.
        else:
            if name in gone:
                report.removed.append(result)
        state.pop(name, None)
    if removals:
        save_state(cfg, course_id, state)

    # Reindexed files were just deleted above, so they re-enter as plain ingests.
    incoming = plan.add + plan.reindex
    if incoming:
        manifest, embedder = resolve_embedder(cfg, course_id)
        reindexed_names = {s.name for s in plan.reindex}
        for i, src in enumerate(incoming, 1):
            if progress:
                progress("reindex" if src.name in reindexed_names else "ingest", src.name, i, len(incoming))
            try:
                result = ingest_file(
                    cfg, course_id, src.path, src.category, manifest=manifest, embedder=embedder
                )
            except IngestError as exc:
                report.failed.append((src.name, str(exc)))
                continue
            state[src.name] = _entry_for(src, src.digest(), result.chunks_added)
            if src.name in reindexed_names:
                report.reindexed.append(result)
            else:
                report.added.append(result)
        save_state(cfg, course_id, state)

    # Record adopted files (indexed before sync existed) so the next run can detect
    # edits to them instead of forever treating them as untracked.
    adopted = False
    scan = scan_sources(cfg, course_id)
    for name in plan.unchanged:
        if name not in state and (src := scan.files.get(name)):
            state[name] = _entry_for(src, src.digest())
            adopted = True
    if adopted:
        save_state(cfg, course_id, state)

    _manifest, store = open_course(cfg, course_id)
    report.total_chunks = store.count()
    return report


# --------------------------------------------------------------------------- #
# Getting files into raw/
# --------------------------------------------------------------------------- #


def forget_source(cfg: Config, course_id: str, name: str) -> Path | None:
    """Delete a file from ``raw/`` and drop its sync state. Returns the path removed.

    The counterpart to indexing it. Once the folder is the source of truth, removing a
    document from the index without removing the file is not a smaller version of the
    same operation — it is a no-op, because the very next sync sees an unindexed file
    sitting in ``raw/`` and puts it straight back.
    """
    removed: Path | None = None
    for src in scan_sources(cfg, course_id).files.values():
        if src.name == name:
            src.path.unlink(missing_ok=True)
            removed = src.path
            break

    state = load_state(cfg, course_id)
    if state.pop(name, None) is not None:
        save_state(cfg, course_id, state)
    return removed


@dataclass
class ImportReport:
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def plan_import(
    cfg: Config,
    course_id: str,
    paths: Iterable[Path],
    *,
    category: str | None = None,
) -> ImportReport:
    """What :func:`import_paths` would copy, without copying it."""
    return import_paths(cfg, course_id, paths, category=category, dry_run=True)


def import_paths(
    cfg: Config,
    course_id: str,
    paths: Iterable[Path],
    *,
    category: str | None = None,
    dry_run: bool = False,
) -> ImportReport:
    """Copy files and folders into a course's ``raw/``, without indexing anything.

    A directory is copied as a category folder named after itself, so
    ``import_paths(cfg, "CS247", [Path("~/slides/lecture")])`` lands its contents in
    ``raw/lecture/`` and they ingest as category ``lecture``. An explicit ``category``
    overrides that for everything in the call.

    A file the course *already* indexes keeps the category it was indexed under,
    whatever folder the copy came from. This is what makes adopting a pre-sync course
    lossless: pointing ``--from`` at the directory the files were originally ingested
    from files each one back where the index says it belongs, instead of flattening
    four hand-assigned categories into one named after the source directory.

    Separating this from :func:`apply_sync` keeps one rule in one place: the folder is
    the source of truth, and *everything* — a CLI ``--from``, a browser upload, a drag
    in Finder — is just a way of writing to it. Call sync afterwards to index.
    """
    root = source_root(cfg, course_id)
    extensions = supported_extensions()
    report = ImportReport()

    try:
        _manifest, store = open_course(cfg, course_id)
        indexed_categories = store.source_categories()
    except IngestError:
        indexed_categories = {}

    def place(src: Path, sub: str | None) -> None:
        if src.name in _IGNORED_NAMES or src.name.startswith("."):
            return
        if src.suffix.lower() not in extensions:
            report.unsupported.append(src.name)
            return
        if category is None and src.name in indexed_categories:
            sub = indexed_categories[src.name]
        dest_dir = root / sub if sub else root
        dest = dest_dir / src.name
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            report.skipped.append(src.name)
            return
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        report.copied.append(str(dest.relative_to(root)))

    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            sub = category if category is not None else path.name
            for child in sorted(path.rglob("*")):
                if child.is_file() and not any(p.startswith(".") for p in child.parts):
                    place(child, sub)
        elif path.is_file():
            place(path, category)
        else:
            report.unsupported.append(str(path))
    return report
