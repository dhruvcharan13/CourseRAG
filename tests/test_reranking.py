"""The reranker contract, the pure reorder step, and the real cross-encoder."""

from __future__ import annotations

import pytest

from courserag.config import Config
from courserag.records import ChunkRecord
from courserag.reranking import Reranker, get_reranker, resolve_reranker_name
from courserag.reranking.cross_encoder import DEFAULT_MODEL_ID, CrossEncoderReranker, is_available
from courserag.reranking.dummy import DummyReranker
from courserag.retrieval import SearchResult, rerank


def _result(rid: str, text: str, score: float) -> SearchResult:
    chunk = ChunkRecord(
        id=rid,
        text=text,
        course="C",
        source_file="deck.pdf",
        category="lecture",
        content_hash=ChunkRecord.hash_text(text),
        added_at="2026-07-27T00:00:00+00:00",
        title=f"Title {rid}",
        page=int(rid[-1]),
    )
    return SearchResult(chunk=chunk, score=score)


CANDIDATES = [
    _result("c1", "Skip lists are a hierarchy of ordered linked lists.", 0.90),
    _result("c2", "Move-to-front moves an accessed item to the head.", 0.80),
    _result("c3", "An AVL tree is a height-balanced search tree.", 0.70),
]


# --------------------------------------------------------------------------- #
# Contract and factory
# --------------------------------------------------------------------------- #


def test_dummy_satisfies_the_protocol():
    """runtime_checkable verifies every member, so a missing one fails here."""
    assert isinstance(DummyReranker(), Reranker)


def test_factory_dispatches_by_name():
    assert isinstance(get_reranker("dummy", Config()), DummyReranker)


def test_unknown_reranker_name_is_refused():
    with pytest.raises(NotImplementedError, match="Unknown reranker"):
        get_reranker("no-such-reranker", Config())


def test_auto_resolves_without_importing_a_model():
    assert resolve_reranker_name("auto") in {"local", "dummy"}
    assert resolve_reranker_name("dummy") == "dummy"


def test_dummy_is_deterministic_and_exactly_reverses():
    """It is a wiring probe, not a ranking strategy — reversal makes that unmistakable."""
    reranker = DummyReranker()
    scores = reranker.score("anything", ["a", "b", "c"])

    assert scores == reranker.score("a different query", ["a", "b", "c"])
    assert scores == sorted(scores)  # ascending => ranking by score reverses input


# --------------------------------------------------------------------------- #
# The pure reorder step
# --------------------------------------------------------------------------- #


def test_rerank_reorders_by_reranker_score():
    out = rerank(DummyReranker(), "query", CANDIDATES)
    assert [r.chunk.id for r in out] == ["c3", "c2", "c1"]


def test_rerank_keeps_both_scores_and_all_citation_fields():
    """Provenance must survive stage two, and the dense score must not be overwritten."""
    out = rerank(DummyReranker(), "query", CANDIDATES)
    top = out[0]

    assert top.chunk.id == "c3"
    assert top.score == 0.70  # the ORIGINAL cosine, not the rerank score
    assert top.rerank_score == 2.0
    assert (top.chunk.source_file, top.chunk.page, top.chunk.title) == ("deck.pdf", 3, "Title c3")
    assert top.chunk.text == CANDIDATES[2].chunk.text


def test_rerank_does_not_mutate_its_input():
    rerank(DummyReranker(), "query", CANDIDATES)
    assert [r.chunk.id for r in CANDIDATES] == ["c1", "c2", "c3"]
    assert all(r.rerank_score is None for r in CANDIDATES)


def test_rerank_of_nothing_is_nothing():
    """An empty candidate list must not load a model."""

    class _Exploding:
        model_id = "boom"

        def score(self, query, passages):
            raise AssertionError("must not be called for an empty candidate list")

    assert rerank(_Exploding(), "query", []) == []


def test_ties_fall_back_to_the_dense_order():
    """An indifferent reranker degrades to dense ranking, not to an arbitrary one."""

    class _Flat:
        model_id = "flat"

        def score(self, query, passages):
            return [1.0] * len(passages)

    out = rerank(_Flat(), "query", CANDIDATES)
    assert [r.chunk.id for r in out] == ["c1", "c2", "c3"]


def test_to_dict_carries_both_scores():
    payload = rerank(DummyReranker(), "query", CANDIDATES)[0].to_dict(rank=1)
    assert payload["score"] == 0.7
    assert payload["rerank_score"] == 2.0
    assert payload["source_file"] == "deck.pdf"


def test_to_dict_reports_null_rerank_score_when_dense_only():
    assert CANDIDATES[0].to_dict(rank=1)["rerank_score"] is None


# --------------------------------------------------------------------------- #
# The real cross-encoder
# --------------------------------------------------------------------------- #

local_only = pytest.mark.skipif(
    not is_available(), reason='requires the [local] extra: pip install -e ".[local]"'
)


@local_only
def test_cross_encoder_prefers_the_passage_that_answers_the_question():
    """A margin, not an absolute floor — cross-encoder logits are not a calibrated scale."""
    reranker = CrossEncoderReranker(DEFAULT_MODEL_ID, cache_dir=Config().models_dir)
    query = "What problem does the Decorator pattern solve?"

    answer, unrelated = reranker.score(
        query,
        [
            "The Decorator pattern gives objects new responsibilities without changing "
            "the underlying classes, keeping them open for extension but closed for "
            "modification.",
            "A skip list is a hierarchy of ordered linked lists with sentinel nodes.",
        ],
    )

    assert answer > unrelated
    assert answer - unrelated > 2.0


@local_only
def test_cross_encoder_separates_a_question_about_a_topic_from_its_answer():
    """The whole premise of Phase 4, as a unit test.

    A bi-encoder scores both of these as "about move-to-front". The measured failure is
    that the assignment *asking* about it outranked the slide *defining* it, so the
    reranker has to prefer the definition when the query asks what MTF does.
    """
    reranker = CrossEncoderReranker(DEFAULT_MODEL_ID, cache_dir=Config().models_dir)

    defines, asks = reranker.score(
        "What does the move-to-front heuristic do after a successful search?",
        [
            "Move-To-Front heuristic (MTF): upon a successful search, move the accessed "
            "item to the front of the list.",
            "Question 2. Analysis of self-organizing search [4+2+1+4+3 marks]. In this "
            "problem, we analyse the move-to-front strategy for linear search.",
        ],
    )

    assert defines > asks


@local_only
def test_pair_token_counts_are_available_for_truncation_reporting():
    reranker = CrossEncoderReranker(DEFAULT_MODEL_ID, cache_dir=Config().models_dir)
    counts = reranker.count_pair_tokens("a query", ["short", "a much longer passage " * 20])

    assert counts[1] > counts[0]
    assert reranker.max_input_tokens == 512
