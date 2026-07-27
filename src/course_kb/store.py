"""Per-course vector store backed by LanceDB.

Each course is one LanceDB dataset in its own directory. ``lancedb.connect``
treats the course directory as the database and a table named ``index``
materializes as ``<course_dir>/index.lance/`` — matching the on-disk layout.

:meth:`CourseStore.search` is the retrieval entry point: an exact cosine top-k over
the course's vectors. Embedding the query is the caller's job (see
:mod:`course_kb.retrieval`), so the store never needs to know which model built it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import lancedb
import pyarrow as pa
from lancedb.expr import col, lit

from course_kb.records import ChunkRecord
from course_kb.retrieval import SearchResult

TABLE_NAME = "index"


def _build_schema(dims: int) -> pa.Schema:
    """Arrow schema mirroring :meth:`ChunkRecord.to_dict`.

    The vector is a fixed-size ``float32`` list of length ``dims``; every other
    field is a scalar, with the optional ones nullable.
    """
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), dims), nullable=False),
            pa.field("course", pa.string(), nullable=False),
            pa.field("source_file", pa.string(), nullable=False),
            pa.field("category", pa.string(), nullable=False),
            pa.field("title", pa.string(), nullable=True),
            pa.field("module", pa.string(), nullable=True),
            pa.field("page", pa.int64(), nullable=True),
            pa.field("char_start", pa.int64(), nullable=True),
            pa.field("char_end", pa.int64(), nullable=True),
            pa.field("content_hash", pa.string(), nullable=False),
            pa.field("added_at", pa.string(), nullable=False),
        ]
    )


class CourseStore:
    """A LanceDB-backed store for a single course."""

    def __init__(self, table: "lancedb.table.Table", dims: int) -> None:
        self._table = table
        self.dims = dims

    @classmethod
    def open_or_create(cls, course_dir: Path, dims: int) -> "CourseStore":
        """Open the course's ``index`` table, creating it empty if absent.

        Raises:
            ValueError: if an existing table's vector width differs from ``dims``.
                A table's width is fixed at creation, so this means the caller is
                using a different embedding model than the one the course was built
                with — caught here rather than as an opaque Arrow error mid-insert.
        """
        course_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(course_dir)
        # A course DB only ever holds the single "index" table, so reading the
        # (unpaginated) table list here is safe.
        if TABLE_NAME in db.list_tables().tables:
            table = db.open_table(TABLE_NAME)
            stored_dims = table.schema.field("vector").type.list_size
            if stored_dims != dims:
                raise ValueError(
                    f"Course table at {course_dir} stores {stored_dims}-dim vectors, but "
                    f"{dims}-dim vectors were requested. Vector width is fixed when the "
                    f"table is created; rebuild the course to change embedding model."
                )
        else:
            table = db.create_table(TABLE_NAME, schema=_build_schema(dims))
        return cls(table, dims)

    def add(self, records: list[ChunkRecord]) -> None:
        """Insert records. Every record must be embedded (``vector`` set)."""
        if not records:
            return
        for r in records:
            if r.vector is None:
                raise ValueError(f"ChunkRecord {r.id!r} has no vector; embed before storing.")
            if len(r.vector) != self.dims:
                raise ValueError(
                    f"ChunkRecord {r.id!r} vector has {len(r.vector)} dims, expected {self.dims}."
                )
        rows = [r.to_dict() for r in records]
        arrow_table = pa.Table.from_pylist(rows, schema=_build_schema(self.dims))
        self._table.add(arrow_table)

    def count(self) -> int:
        """Number of stored chunks."""
        return self._table.count_rows()

    def existing_hashes(self, source_file: str | None = None) -> set[str]:
        """Stored ``content_hash`` values, optionally scoped to one source file.

        Scoping to ``source_file`` keeps dedup per-file: re-ingesting a file is a
        no-op, while an identical slide shared across different files is retained
        in each (preserving per-file page provenance).
        """
        if self._table.count_rows() == 0:
            return set()
        table = self._table.to_arrow()
        hashes = table.column("content_hash").to_pylist()
        if source_file is None:
            return set(hashes)
        files = table.column("source_file").to_pylist()
        return {h for h, f in zip(hashes, files) if f == source_file}

    def get_all(self) -> list[ChunkRecord]:
        """Read every stored row back as a :class:`ChunkRecord`."""
        return [ChunkRecord.from_dict(row) for row in self._table.to_arrow().to_pylist()]

    def _scan(self, columns: list[str]) -> pa.Table:
        """Read whole columns without materializing the vector column.

        ``search()`` with no query vector is a plain scan, and ``limit(0)`` means no
        limit (the default is 10). Projecting matters because the vector column dwarfs
        every other field — 0.7MB of a 1.2MB scan on a 483-row course.
        """
        return self._table.search().select(columns).limit(0).to_arrow()

    def source_files(self) -> list[str]:
        """Distinct ``source_file`` values currently stored, in first-seen order."""
        return list(dict.fromkeys(self._scan(["source_file"]).column("source_file").to_pylist()))

    def categories(self) -> list[str]:
        """Distinct ``category`` values currently stored, in first-seen order."""
        return list(dict.fromkeys(self._scan(["category"]).column("category").to_pylist()))

    def delete_source(self, source_file: str) -> int:
        """Delete every chunk that came from ``source_file``; return rows removed.

        Vectors are a pure function of chunk text, so removing a document costs no
        embedding work at all — no model is loaded on this path.

        The filter is a typed expression rather than an SQL string, so a filename
        containing a quote (``Chapter 1's Notes.pdf``) is matched correctly instead of
        malforming the predicate.
        """
        before = self._table.count_rows()
        self._table.delete(col("source_file") == lit(source_file))
        removed = before - self._table.count_rows()
        if removed:
            # A delete only tombstones rows. Compacting alone makes things *worse* —
            # it writes merged files while the pre-delete version is still retained
            # (measured: 5.9MB -> 8.9MB) — so prune old versions too and actually give
            # the space back (-> 3.0MB). The cost is that the table can no longer be
            # rolled back to before the delete; re-ingesting the source file restores
            # it exactly, since chunk ids and vectors are derived from text.
            self._table.optimize(cleanup_older_than=timedelta(0))
        return removed

    def search(self, query_vector: list[float], k: int = 5) -> list["SearchResult"]:
        """Top-``k`` chunks by cosine similarity to ``query_vector``, best first.

        Vectors are unit-normalized at embed time, so cosine similarity equals the dot
        product; either way the ranking is the same. No ANN index is ever built on these
        tables, so this is an exact brute-force scan — correct at any corpus size, and
        fast enough while courses are thousands of chunks rather than millions.

        LanceDB returns a *distance*; the similarity callers want is ``1 - distance``.
        That conversion is pinned by a test on hand-computed vectors rather than trusted,
        since the formula lives in Lance's Rust core and is not part of its Python API.

        Raises:
            ValueError: if ``query_vector`` is not this table's width — otherwise the
                mismatch surfaces as an opaque error from deep inside the query engine.
        """
        if len(query_vector) != self.dims:
            raise ValueError(
                f"Query vector has {len(query_vector)} dims, but this course stores "
                f"{self.dims}-dim vectors. The query must be embedded by the same model "
                f"as the course."
            )
        if k <= 0 or self._table.count_rows() == 0:
            return []

        # Project the vector column away: it is not needed to rank (LanceDB already
        # did that) or to cite, and it is far larger than every other field combined.
        # ChunkRecord.from_dict reads "vector" with .get(), so the records rebuild fine.
        # "_distance" is named explicitly: Lance auto-adds it to a projected search today
        # but warns that it will stop doing so.
        columns = [f.name for f in self._table.schema if f.name != "vector"]
        columns.append("_distance")
        rows = (
            self._table.search(query_vector)
            .metric("cosine")
            .select(columns)
            .limit(k)
            .to_arrow()
            .to_pylist()
        )
        return [
            SearchResult(chunk=ChunkRecord.from_dict(row), score=1.0 - float(row["_distance"]))
            for row in rows
        ]
