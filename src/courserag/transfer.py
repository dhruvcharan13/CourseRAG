"""Export a course to a single file, and import one somebody sent you.

The archive carries a course's **sources and metadata, not its index**. That is not a
size decision — the index is the small half (4.4MB against 21MB of PDFs for CS240) —
it is a correctness one. A vector is a pure function of chunk text and embedding model,
so an importer who runs the same model rebuilds an index identical to the exporter's,
chunk ids and all. Shipping ``index.lance`` instead would carry a LanceDB on-disk format
between two machines whose library versions nobody controls, to save a few minutes of
embedding, and would still need the sources alongside it for the recipient to re-sync.

So an import is: unpack ``raw/``, recreate the course with the *recorded* embedder, and
run the ordinary sync. Nothing about the receiving end is special-cased, which is why
the resulting course is indistinguishable from one built locally.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from courserag.config import Config
from courserag.ingest import course_dir, course_exists, open_course
from courserag.manifest import Manifest, write_manifest
from courserag.store import CourseStore
from courserag.sync import SOURCE_DIR_NAME, apply_sync, source_root

#: Bumped when the archive layout changes in a way an older importer cannot read.
FORMAT = "courserag-course/1"
METADATA_NAME = "courserag-export.json"


class TransferError(Exception):
    """An archive could not be written, read, or trusted."""


@dataclass
class ExportReport:
    path: Path
    course_id: str
    files: int
    chunks: int
    bytes: int


@dataclass
class ImportReport:
    course_id: str
    files: int
    #: Chunks the archive said the exporter had.
    expected_chunks: int
    #: Chunks actually rebuilt here.
    chunks: int
    embedding_model: str
    #: Files in the archive that failed to index on this machine.
    failed: list[tuple[str, str]]

    @property
    def exact(self) -> bool:
        """Whether the rebuilt index matches the exporter's, chunk for chunk."""
        return self.chunks == self.expected_chunks and not self.failed


def default_export_name(course_id: str) -> str:
    return f"{course_id}-courserag.zip"


def selector_for(model_id: str) -> str:
    """The ``get_embedder`` selector that reproduces a recorded model id.

    A manifest records what a model *is* (``dummy-hash-64``,
    ``BAAI/bge-small-en-v1.5``); ``get_embedder`` takes what to *ask for* (``dummy``,
    ``bge``, or a HuggingFace id). The two vocabularies coincide for HuggingFace models,
    because their id is also a valid selector, and diverge for the dummy — which has no
    slash and so is not recognized as an id. Passing a recorded id straight through
    therefore works for every real course and fails for dummy ones, which is exactly
    the kind of gap that only shows up on the receiving end.
    """
    return "dummy" if model_id.startswith("dummy-") else model_id


def export_course(cfg: Config, course_id: str, dest: Path) -> ExportReport:
    """Write ``course_id`` to a zip at ``dest``.

    Raises:
        CourseNotInitialized: if the course has no manifest.
        TransferError: if the course has no source files to send.
    """
    manifest, store = open_course(cfg, course_id)
    cdir = course_dir(cfg, course_id)
    root = source_root(cfg, course_id)

    sources = sorted(
        p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")
    ) if root.is_dir() else []
    if not sources:
        raise TransferError(
            f"Course '{course_id}' has no files in {root} to export. An index alone "
            f"cannot be rebuilt from — the archive carries sources, not vectors."
        )

    metadata = {
        "format": FORMAT,
        "course": course_id,
        "embedding_model": manifest.embedding_model,
        "dims": manifest.dims,
        "categories": manifest.categories,
        "files": manifest.files,
        "chunk_count": store.count(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(METADATA_NAME, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        course_md = cdir / "COURSE.md"
        if course_md.exists():
            zf.write(course_md, "COURSE.md")
        for path in sources:
            # Store paths relative to raw/, so the category folders travel with the files.
            zf.write(path, str(PurePosixPath(SOURCE_DIR_NAME) / path.relative_to(root)))

    return ExportReport(
        path=dest,
        course_id=course_id,
        files=len(sources),
        chunks=store.count(),
        bytes=dest.stat().st_size,
    )


def read_metadata(archive: Path) -> dict:
    """Read and validate an archive's manifest without extracting anything.

    Lets a caller show what is inside — and refuse an incompatible or hand-made zip —
    before it writes a single file to disk.
    """
    if not archive.is_file():
        raise TransferError(f"No such archive: {archive}")
    try:
        with zipfile.ZipFile(archive) as zf:
            with zf.open(METADATA_NAME) as fh:
                metadata = json.load(fh)
    except KeyError:
        raise TransferError(
            f"{archive.name} has no {METADATA_NAME} — it is not a CourseRAG course export."
        ) from None
    except (zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise TransferError(f"{archive.name} is not a readable zip archive: {exc}") from exc

    fmt = metadata.get("format")
    if fmt != FORMAT:
        raise TransferError(
            f"{archive.name} declares format {fmt!r}, but this version reads {FORMAT!r}."
        )
    if not metadata.get("course"):
        raise TransferError(f"{archive.name} does not name a course.")
    return metadata


def _safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Archive entries under ``raw/``, with their destination paths inside the course.

    A zip stores whatever path string the writer put in it, including ``../..`` and
    absolute roots — extracting one blindly writes outside the destination directory
    ("zip slip"). Each name is rebuilt from its own parts here, dropping anything that
    could climb, rather than trusted and checked afterwards.
    """
    out = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = [p for p in PurePosixPath(info.filename.replace("\\", "/")).parts if p != "/"]
        if not parts or parts[0] != SOURCE_DIR_NAME:
            continue  # metadata and COURSE.md are handled separately; ignore the rest
        clean = [p for p in parts[1:] if p not in ("", ".", "..") and not p.startswith(".")]
        if not clean:
            continue
        out.append((info, PurePosixPath(*clean)))
    return out


def import_course(
    cfg: Config,
    archive: Path,
    *,
    as_course: str | None = None,
    force: bool = False,
) -> ImportReport:
    """Create a course from ``archive`` and index it.

    The course is built with the embedding model the archive records, not the one this
    machine's config happens to select — otherwise an importer with a different default
    would silently produce a course whose vectors are incomparable to the exporter's.

    Raises:
        TransferError: for an unreadable archive, a name collision without ``force``,
            or an embedding model this machine cannot load.
    """
    metadata = read_metadata(archive)
    course_id = as_course or metadata["course"]
    model_id = metadata["embedding_model"]

    if course_exists(cfg, course_id) and not force:
        raise TransferError(
            f"Course '{course_id}' already exists. Import under a different name, or "
            f"force to replace it."
        )

    # Resolve the archive's model before writing anything, so an importer missing the
    # [local] extra gets told that instead of a half-created course.
    from courserag.embedding import get_embedder

    selector = selector_for(model_id)
    try:
        embedder = get_embedder(selector, cfg)
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad. The model name comes out of an archive someone else
        # wrote, so it can fail in ways this end does not enumerate — a missing extra,
        # an unknown alias, or a HuggingFace id that 404s on download. All of them mean
        # the same thing to the person importing, and none of them should surface as a
        # library traceback.
        raise TransferError(
            f"{archive.name} was built with '{model_id}', which cannot be loaded here "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    if (embedder.model_id, embedder.dims) != (model_id, metadata["dims"]):
        raise TransferError(
            f"'{selector}' resolves to {embedder.model_id} ({embedder.dims}-dim) here, "
            f"but the archive records {model_id} ({metadata['dims']}-dim). Refusing to "
            f"build a course whose vectors would not match the one exported."
        )

    cdir = course_dir(cfg, course_id)
    root = cdir / SOURCE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    cfg.archive_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        members = _safe_members(zf)
        if not members:
            raise TransferError(f"{archive.name} contains no source files under raw/.")
        for info, rel in members:
            dest = root / Path(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as out:
                while block := src.read(1 << 20):
                    out.write(block)
        try:
            with zf.open("COURSE.md") as fh:
                (cdir / "COURSE.md").write_bytes(fh.read())
        except KeyError:
            (cdir / "COURSE.md").write_text("", encoding="utf-8")

    # An empty course at the archive's width, then the ordinary sync fills it. Files
    # start empty so sync sees every unpacked file as new.
    CourseStore.open_or_create(cdir, metadata["dims"])
    write_manifest(
        cdir,
        Manifest(
            course=course_id,
            embedding_model=model_id,
            dims=metadata["dims"],
            categories=[],
            files=[],
            last_indexed=None,
        ),
    )

    # Pin the embedder for this run: the recipient's config is irrelevant to what the
    # archive says the course is.
    report = apply_sync(replace(cfg, embedder=selector), course_id, force=True)

    return ImportReport(
        course_id=course_id,
        files=len(members),
        expected_chunks=int(metadata.get("chunk_count", 0)),
        chunks=report.total_chunks,
        embedding_model=model_id,
        failed=report.failed,
    )
