"""Retrieval: the ranking is correct, the scores are real cosines, citations survive."""

from __future__ import annotations

import pytest

from course_kb.embedding.dummy import DummyEmbedder
from course_kb.records import ChunkRecord
from course_kb.retrieval import embed_query
from course_kb.store import CourseStore


def _record(rid: str, vector: list[float], *, text: str | None = None, page: int = 1) -> ChunkRecord:
    text = text if text is not None else rid
    return ChunkRecord(
        id=rid,
        text=text,
        course="C",
        source_file="notes.pdf",
        category="lecture",
        content_hash=ChunkRecord.hash_text(text),
        added_at="2026-07-26T00:00:00+00:00",
        vector=vector,
        title=f"Title {rid}",
        page=page,
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
