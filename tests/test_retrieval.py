"""Retrieval: the ranking is correct, the scores are real cosines, citations survive."""

from __future__ import annotations

import pytest

from course_kb.embedding.dummy import DummyEmbedder
from course_kb.records import ChunkRecord
from course_kb.retrieval import embed_query, retrieve
from course_kb.store import CourseStore


def _record(
    rid: str, vector: list[float], *, text: str | None = None, page: int = 1,
    source_file: str = "notes.pdf", char_start: int | None = None,
) -> ChunkRecord:
    # char_range is what the record carries; the store splits it into char_start /
    # char_end columns, which is what read_source orders on.
    text = text if text is not None else rid
    return ChunkRecord(
        id=rid,
        text=text,
        course="C",
        source_file=source_file,
        category="lecture",
        content_hash=ChunkRecord.hash_text(text),
        added_at="2026-07-26T00:00:00+00:00",
        vector=vector,
        title=f"Title {rid}",
        page=page,
        char_range=None if char_start is None else (char_start, char_start + len(text)),
    )


@pytest.fixture
def unit_store(tmp_path):
    """Four 3-dim unit vectors whose cosines against [1,0,0] are known exactly."""
    store = CourseStore.open_or_create(tmp_path / "C", 3)
    store.add(
        [
            _record("exact", [1.0, 0.0, 0.0], page=1),
            _record("close", [0.6, 0.8, 0.0], page=2),
            _record("orthogonal", [0.0, 1.0, 0.0], page=3),
            _record("opposite", [-1.0, 0.0, 0.0], page=4),
        ]
    )
    return store


def test_ranks_by_cosine_and_reports_exact_scores(unit_store):
    """Pins ``score = 1 - _distance``.

    Lance defines its distance metrics in Rust and does not document the formula in the
    Python API, so the conversion is asserted against cosines computed by hand rather
    than trusted. If Lance ever changes the definition this fails loudly, instead of
    silently reporting numbers that look plausible and rank subtly wrong.
    """
    results = unit_store.search([1.0, 0.0, 0.0], k=4)

    assert [r.chunk.id for r in results] == ["exact", "close", "orthogonal", "opposite"]
    assert [round(r.score, 6) for r in results] == [1.0, 0.6, 0.0, -1.0]


def test_results_carry_citations_but_not_vectors(unit_store):
    result = unit_store.search([1.0, 0.0, 0.0], k=1)[0]

    assert (result.chunk.source_file, result.chunk.page) == ("notes.pdf", 1)
    assert result.chunk.title == "Title exact"
    assert result.chunk.text == "exact"
    # The vector column is projected away: it is not provenance, and it is the largest
    # field by far. from_dict tolerates its absence.
    assert result.chunk.vector is None


def test_to_dict_is_citation_shaped(unit_store):
    payload = unit_store.search([1.0, 0.0, 0.0], k=1)[0].to_dict(rank=1)

    assert payload["rank"] == 1
    assert payload["score"] == 1.0
    assert payload["source_file"] == "notes.pdf"
    assert payload["page"] == 1
    assert "vector" not in payload


def test_k_larger_than_corpus_returns_everything(unit_store):
    assert len(unit_store.search([1.0, 0.0, 0.0], k=99)) == 4


def test_non_positive_k_and_empty_store_return_nothing(unit_store, tmp_path):
    assert unit_store.search([1.0, 0.0, 0.0], k=0) == []
    empty = CourseStore.open_or_create(tmp_path / "empty", 3)
    assert empty.search([1.0, 0.0, 0.0], k=5) == []


def test_wrong_width_query_is_refused(unit_store):
    """Caught here rather than as an opaque error from inside the query engine."""
    with pytest.raises(ValueError, match="2 dims.*stores 3-dim"):
        unit_store.search([1.0, 0.0], k=1)


def test_known_chunk_is_rank_one_end_to_end(tmp_path):
    """The whole path on a tiny store: embed a query, get its chunk back first.

    Uses DummyEmbedder so this runs offline with no model. Dummy vectors are seeded by
    the text's hash, so a query equal to a chunk's text embeds to that chunk's exact
    vector — self-similarity dominates and the right chunk must rank 1.
    """
    embedder = DummyEmbedder()
    store = CourseStore.open_or_create(tmp_path / "C", embedder.dims)
    texts = [
        "Skip lists are a hierarchy of ordered linked lists.",
        "An AVL tree is a height-balanced binary search tree.",
        "Move-to-front moves an accessed item to the head of the list.",
    ]
    vectors = embedder.embed(texts)
    store.add(
        [_record(f"c{i}", v, text=t, page=i) for i, (t, v) in enumerate(zip(texts, vectors))]
    )

    results = store.search(embed_query(embedder, texts[2]), k=3)

    assert results[0].chunk.text == texts[2]
    assert results[0].score > results[1].score


# --------------------------------------------------------------------------- #
# Document scoping
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_doc_store(tmp_path):
    """A store where one document outranks the other on every query.

    ``near.pdf`` sits almost on the query axis and ``far.pdf`` well off it, so an
    unscoped top-k is all ``near.pdf``. That asymmetry is the point: it is what makes
    the pre-filter assertion below able to fail.
    """
    store = CourseStore.open_or_create(tmp_path / "C", 3)
    store.add(
        [_record(f"n{i}", [1.0, 0.05 * i, 0.0], source_file="near.pdf", page=i)
         for i in range(1, 6)]
        + [_record(f"f{i}", [0.1, 1.0, 0.0], source_file="far.pdf", page=i)
           for i in range(1, 4)]
    )
    return store


def test_search_scoped_to_a_document_returns_only_that_document(two_doc_store):
    results = two_doc_store.search([1.0, 0.0, 0.0], k=3, source_file="far.pdf")

    assert {r.chunk.source_file for r in results} == {"far.pdf"}


def test_scoping_pre_filters_rather_than_filtering_the_top_k(two_doc_store):
    """The bug this exists to catch, and it fails silently without ``prefilter=True``.

    Every ``far.pdf`` chunk ranks below every ``near.pdf`` chunk, so a post-filter would
    take the course-wide top 3 (all ``near.pdf``), drop them for not matching, and
    return **nothing** — indistinguishable from an honest "this document has no
    relevant passages". Pre-filtering ranks inside the document, so k means k.
    """
    unscoped = two_doc_store.search([1.0, 0.0, 0.0], k=3)
    assert {r.chunk.source_file for r in unscoped} == {"near.pdf"}, "fixture no longer asymmetric"

    scoped = two_doc_store.search([1.0, 0.0, 0.0], k=3, source_file="far.pdf")

    assert len(scoped) == 3


def test_scoping_to_an_absent_document_returns_nothing(two_doc_store):
    assert two_doc_store.search([1.0, 0.0, 0.0], k=3, source_file="missing.pdf") == []


def test_unscoped_search_is_unchanged_by_the_feature(two_doc_store):
    """The default path has to stay byte-identical; the committed baseline rests on it."""
    assert two_doc_store.search([1.0, 0.0, 0.0], k=8) == two_doc_store.search(
        [1.0, 0.0, 0.0], k=8, source_file=None
    )


def test_read_source_returns_the_whole_document_in_page_order(tmp_path):
    """Reading order, not relevance order — and total, since pages hold several chunks."""
    store = CourseStore.open_or_create(tmp_path / "C", 3)
    store.add(
        [
            _record("b", [1.0, 0.0, 0.0], source_file="d.pdf", page=2, char_start=0),
            _record("a2", [1.0, 0.0, 0.0], source_file="d.pdf", page=1, char_start=500),
            _record("a1", [1.0, 0.0, 0.0], source_file="d.pdf", page=1, char_start=0),
            _record("other", [1.0, 0.0, 0.0], source_file="e.pdf", page=1, char_start=0),
        ]
    )

    records = store.read_source("d.pdf")

    assert [r.id for r in records] == ["a1", "a2", "b"]
    assert {r.source_file for r in records} == {"d.pdf"}


def test_read_source_of_an_absent_document_is_empty(tmp_path):
    store = CourseStore.open_or_create(tmp_path / "C", 3)
    store.add([_record("a", [1.0, 0.0, 0.0], source_file="d.pdf")])

    assert store.read_source("nope.pdf") == []


def test_retrieve_threads_the_scope_through_the_rerank_candidate_pool(two_doc_store):
    """Scoping has to reach the candidate search, not the slice after reranking.

    Filtering afterwards would rescore course-wide candidates and then discard most of
    them, returning fewer than k from a stage that had k available.
    """
    from course_kb.config import Config
    from course_kb.embedding.dummy import DummyEmbedder as _D

    class _Q:
        dims, model_id = 3, "stub"

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    cfg = Config(reranker="dummy", rerank_candidates=20)
    final, dense = retrieve(
        cfg, two_doc_store, _Q(), "anything", 3, use_rerank=True, source_file="far.pdf"
    )

    assert len(final) == 3
    assert {r.chunk.source_file for r in final} == {"far.pdf"}
    assert {r.chunk.source_file for r in dense} == {"far.pdf"}
