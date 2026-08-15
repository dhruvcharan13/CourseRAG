"""A local web UI for putting files into courses and taking them out again.

The whole app is a front-end for one idea that lives in :mod:`courserag.sync`: a
course's ``raw/`` folder is the source of truth, and indexing is a reconciliation.
Dropping files here writes them into that folder and enqueues a sync — exactly what
``kb sync --from`` does, and exactly what dragging them in Finder and running
``kb sync`` does. No API endpoint knows what a chunk is.

Bound to localhost and unauthenticated, because it reads and writes a knowledge base
built from the user's own files on the user's own machine. It is not a service; the
one hardening it does need is against a *file name* escaping the course folder, since
browsers will happily send ``../`` in a directory upload — see :func:`_safe_relpath`.

Imported lazily by ``kb web`` and by nothing else in the package, so the ``[web]``
extra never costs the CLI anything.
"""

from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from courserag.config import Config, load_config
from courserag.ingest import (
    CourseNotInitialized,
    IngestError,
    course_dir,
    course_exists,
    delete_file,
    list_courses,
    open_course,
)
from courserag.manifest import Manifest, manifest_path
from courserag.parsing import supported_extensions
from courserag.store import CourseStore
from courserag.sync import (
    SOURCE_DIR_NAME,
    RenameRefused,
    SyncRefused,
    apply_sync,
    clean_category,
    forget_source,
    plan_sync,
    rename_source,
    scan_sources,
    set_category,
    source_root,
)
from courserag.transfer import TransferError, default_export_name, export_course
from courserag.web.jobs import Job, JobQueue

STATIC_DIR = Path(__file__).parent / "static"


def _safe_relpath(raw: str) -> PurePosixPath:
    """Turn a browser-supplied relative path into one that cannot leave ``raw/``.

    A directory upload sends ``webkitRelativePath`` verbatim, which is attacker- (or
    just typo-) controlled text: ``../../../.ssh/config`` is a perfectly well-formed
    value for it. Absolute roots, ``..`` segments and dot-directories are dropped
    rather than rejected, so an odd folder name still uploads — it just lands flat.
    """
    parts = [
        part
        for part in PurePosixPath(raw.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/") and not part.startswith(".")
    ]
    return PurePosixPath(*parts) if parts else PurePosixPath("")


def _course_payload(cfg: Config, course_id: str, jobs: JobQueue) -> dict:
    """Everything the UI shows about one course.

    Opens the store but never an embedder: listing files compares no vectors, so the
    page still renders for a course whose model is unavailable.
    """
    manifest, store = open_course(cfg, course_id)
    counts = store.source_file_counts()
    scan = scan_sources(cfg, course_id)

    files = []
    for name in sorted(set(manifest.files) | set(counts) | set(scan.files)):
        src = scan.files.get(name)
        files.append(
            {
                "name": name,
                "chunks": counts.get(name, 0),
                "category": src.category if src else None,
                "rel": src.rel if src else None,
                "size": src.size if src else None,
                # On disk but unindexed (dropped, not yet synced), or indexed but the
                # file is gone (deleted in Finder, not yet synced). Either way the UI
                # marks it pending rather than pretending the two agree.
                "on_disk": src is not None,
                "indexed": counts.get(name, 0) > 0 or name in manifest.files,
            }
        )

    active = jobs.active_for(course_id)
    return {
        "id": course_id,
        "embedding_model": manifest.embedding_model,
        "dims": manifest.dims,
        "categories": manifest.categories,
        "last_indexed": manifest.last_indexed,
        "chunks": store.count(),
        "files": files,
        "source_dir": str(source_root(cfg, course_id)),
        "duplicates": scan.duplicates,
        "unsupported": [p.name for p in scan.unsupported],
        "job": active.as_dict() if active else None,
    }


def create_app(cfg: Config | None = None) -> FastAPI:
    """Build the app. ``cfg`` is resolved once here, not per request."""
    config = cfg if cfg is not None else load_config()
    jobs = JobQueue()
    app = FastAPI(title="CourseRAG", docs_url=None, redoc_url=None)

    def _require_course(course_id: str) -> None:
        if not course_exists(config, course_id):
            raise HTTPException(404, f"No course '{course_id}'.")

    def _submit_sync(course_id: str, label: str) -> Job:
        """Queue a reconciliation of one course, logging each file as it lands."""

        def run(job: Job) -> None:
            plan = plan_sync(config, course_id)
            if plan.is_empty:
                job.log("Already up to date.")
                return
            total = len(plan.add) + len(plan.reindex) + len(plan.remove)
            job.log(
                f"{len(plan.add)} to add, {len(plan.reindex)} changed, "
                f"{len(plan.remove)} to remove."
            )

            seen = 0

            def progress(stage: str, name: str, index: int, count: int) -> None:
                nonlocal seen
                seen += 1
                job.progress = min(seen / total, 0.99) if total else None
                job.log(f"{stage}: {name}")

            try:
                report = apply_sync(config, course_id, plan, progress=progress)
            except SyncRefused as exc:
                # The empty-folder guard. Surfaced as a message rather than a crash,
                # because from the UI it is a normal thing to hit: a course whose
                # files you just deleted looks exactly like a course whose folder
                # went missing.
                raise RuntimeError(str(exc)) from exc

            for result in report.added + report.reindexed:
                job.log(f"  {result.source_file}: {result.chunks_added} chunks")
            for removed in report.removed:
                job.log(f"  {removed.source_file}: removed {removed.chunks_removed} chunks")
            for name, message in report.failed:
                job.log(f"  failed {name}: {message}")
            job.log(f"Done — {report.total_chunks} chunks.")

        return jobs.submit(course_id, label, run)

    # ----------------------------------------------------------------- pages #

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ------------------------------------------------------------------- api #

    @app.get("/api/courses")
    def get_courses() -> dict:
        """Course list for the sidebar. Cheap: manifest and row count only."""
        out = []
        for course_id in list_courses(config):
            cdir = course_dir(config, course_id)
            if not manifest_path(cdir).exists():
                continue
            manifest, store = open_course(config, course_id)
            active = jobs.active_for(course_id)
            out.append(
                {
                    "id": course_id,
                    "chunks": store.count(),
                    "files": len(manifest.files),
                    "busy": active is not None,
                }
            )
        return {"courses": out, "extensions": sorted(supported_extensions())}

    @app.post("/api/courses")
    def create_course(course_id: str = Form(...)) -> dict:
        """Create a course. Mirrors ``kb init-course``, including its embedder check."""
        name = course_id.strip()
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(400, "A course id cannot be empty, hidden, or contain a slash.")
        if course_exists(config, name):
            raise HTTPException(409, f"Course '{name}' already exists.")

        # Imported here: constructing an embedder is the one thing on this path that
        # can pull the model stack, and only course creation needs it.
        from courserag.embedding import get_embedder
        from courserag.manifest import write_manifest

        try:
            embedder = get_embedder(config.embedder, config)
        except (ImportError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        cdir = course_dir(config, name)
        (cdir / SOURCE_DIR_NAME).mkdir(parents=True, exist_ok=True)
        config.archive_dir.mkdir(parents=True, exist_ok=True)
        (cdir / "COURSE.md").write_text("", encoding="utf-8")
        CourseStore.open_or_create(cdir, embedder.dims)
        write_manifest(
            cdir,
            Manifest(
                course=name,
                embedding_model=embedder.model_id,
                dims=embedder.dims,
                categories=[],
                files=[],
                last_indexed=None,
            ),
        )
        return {"id": name, "embedding_model": embedder.model_id, "dims": embedder.dims}

    @app.get("/api/courses/{course_id}")
    def get_course(course_id: str) -> dict:
        _require_course(course_id)
        return _course_payload(config, course_id, jobs)

    @app.post("/api/courses/{course_id}/files")
    async def upload(
        course_id: str,
        files: list[UploadFile] = File(...),
        category: str | None = Form(None),
        paths: list[str] | None = Form(None),
    ) -> dict:
        """Write dropped files into ``raw/`` and queue a sync.

        ``paths`` carries each file's path relative to the folder that was dropped, so
        a dropped directory keeps its shape and therefore its category. It is mirrored
        into ``raw/`` verbatim (after sanitizing) rather than interpreted here — the
        directory tree means the same thing to the UI as it does to ``kb sync``.
        """
        _require_course(course_id)
        root = source_root(config, course_id)
        extensions = supported_extensions()

        written: list[str] = []
        skipped: list[str] = []
        for i, upload_file in enumerate(files):
            supplied = paths[i] if paths and i < len(paths) else (upload_file.filename or "")
            rel = _safe_relpath(supplied)
            name = rel.name or Path(upload_file.filename or "").name
            if not name:
                continue
            if Path(name).suffix.lower() not in extensions:
                skipped.append(name)
                continue
            sub = rel.parent if str(rel.parent) != "." else PurePosixPath("")
            # An explicit category overrides the dropped folder's own shape.
            dest_dir = root / category if category else root / sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / name
            with dest.open("wb") as fh:
                while chunk := await upload_file.read(1 << 20):
                    fh.write(chunk)
            written.append(str(dest.relative_to(root)))

        if not written:
            return {"written": [], "skipped": skipped, "job": None}
        job = _submit_sync(course_id, f"Indexing {len(written)} file(s)")
        return {"written": written, "skipped": skipped, "job": job.as_dict()}

    @app.get("/api/courses/{course_id}/export")
    def export_archive(course_id: str) -> FileResponse:
        """Download the course as a shareable zip.

        Built into a temp file and deleted once the response is sent, so a browser
        download costs no permanent disk and two people exporting at once cannot
        collide on a name.
        """
        _require_course(course_id)
        fd, tmp = tempfile.mkstemp(suffix=".zip", prefix=f"{course_id}-")
        os.close(fd)
        try:
            export_course(config, course_id, Path(tmp))
        except (IngestError, TransferError) as exc:
            os.unlink(tmp)
            raise HTTPException(400, str(exc)) from exc
        return FileResponse(
            tmp,
            media_type="application/zip",
            filename=default_export_name(course_id),
            background=BackgroundTask(os.unlink, tmp),
        )

    @app.get("/api/courses/{course_id}/raw/{name:path}")
    def file_bytes(course_id: str, name: str) -> FileResponse:
        """Serve one source file back, for the in-page viewer.

        The requested name is never joined onto a path. It is looked up by basename in
        the course's own scan, so the only files reachable are ones sync already found
        under ``raw/`` — a traversal string simply matches nothing. That is the same
        reason chunks are keyed by basename, reused as an access rule.

        ``inline`` disposition so the browser renders it (every current browser has a
        built-in PDF viewer) rather than downloading it.
        """
        _require_course(course_id)
        wanted = Path(name).name
        match = scan_sources(config, course_id).files.get(wanted)
        if match is None:
            raise HTTPException(404, f"No file '{wanted}' in '{course_id}'.")
        media_type, _ = mimetypes.guess_type(match.path.name)
        return FileResponse(
            match.path,
            media_type=media_type or "application/octet-stream",
            headers={"Content-Disposition": f'inline; filename="{wanted}"'},
        )

    @app.post("/api/courses/{course_id}/sync")
    def sync_course(course_id: str) -> dict:
        """Reconcile after the folder was changed outside the browser."""
        _require_course(course_id)
        if (active := jobs.active_for(course_id)) is not None:
            return {"job": active.as_dict()}
        return {"job": _submit_sync(course_id, "Syncing").as_dict()}

    @app.patch("/api/courses/{course_id}/files/{name:path}")
    def update_file(
        course_id: str,
        name: str,
        category: str | None = Form(None),
        new_name: str | None = Form(None),
    ) -> dict:
        """Recategorize and/or rename one file, then reindex it.

        Both edits in one call, and one sync at the end, because each is a move on disk
        and syncing between them would index the file twice for no reason.

        The sync is not optional bookkeeping: a chunk stores both its category and its
        source file name, so until the file is reindexed the folder and the index
        disagree — search would still report the old ones.
        """
        _require_course(course_id)
        safe = Path(name).name
        current = safe
        changes = []

        try:
            if new_name is not None and Path(str(new_name).strip()).name != safe:
                current = rename_source(config, course_id, safe, new_name).name
                changes.append(f"renamed to {current}")
            if category is not None:
                target = clean_category(category, config)
                # Rename first, so this moves the file under whatever it is now called.
                set_category(config, course_id, current, target)
                changes.append(f"category {target}")
        except KeyError:
            raise HTTPException(404, f"No file '{safe}' on disk in '{course_id}'.") from None
        except RenameRefused as exc:
            raise HTTPException(409, str(exc)) from exc

        # Report the resulting state, not the requested one: with two edits in one call
        # the caller should not have to work out what it ended up with.
        landed = scan_sources(config, course_id).files.get(current)
        settled = landed.category if landed else None

        if not changes:
            return {"name": current, "category": settled, "job": None}

        job = _submit_sync(course_id, f"{safe}: {', '.join(changes)}")
        return {"name": current, "category": settled, "job": job.as_dict()}

    @app.delete("/api/courses/{course_id}/files/{name:path}")
    def remove_file(course_id: str, name: str) -> dict:
        """Remove a document from the index *and* from ``raw/``.

        Both, because the folder is the source of truth: dropping only the chunks
        would leave the file sitting there for the next sync to re-ingest, which reads
        as the delete having silently failed.
        """
        _require_course(course_id)
        safe = Path(name).name  # chunks are keyed by basename; never a path
        try:
            result = delete_file(config, course_id, safe)
        except CourseNotInitialized as exc:
            raise HTTPException(404, str(exc)) from exc
        except KeyError:
            # Not indexed, but possibly dropped-and-not-yet-synced: still remove the file.
            if forget_source(config, course_id, safe) is None:
                raise HTTPException(404, f"No file '{safe}' in '{course_id}'.") from None
            return {"name": safe, "chunks_removed": 0, "removed_from_disk": True}

        removed_path = forget_source(config, course_id, safe)
        return {
            "name": safe,
            "chunks_removed": result.chunks_removed,
            "total_chunks": result.total_chunks,
            "removed_from_disk": removed_path is not None,
        }

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        return job.as_dict()

    @app.exception_handler(IngestError)
    def _ingest_error(_request, exc: IngestError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, cfg: Config | None = None) -> None:
    """Run the UI. Blocks until interrupted."""
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
