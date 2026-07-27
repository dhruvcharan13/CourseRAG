"""Eval-set parsing and the metric math, checked against rankings computed by hand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from course_kb.evaluation import (
    EvalQuery,
    QueryOutcome,
    first_hit_rank,
    load_eval_set,
    score_run,
    score_spread,
)
from course_kb.records import ChunkRecord
from course_kb.retrieval import SearchResult

REPO_ROOT = Path(__file__).resolve().parent.parent


def _result(source_file: str, page: int, score: float) -> SearchResult:
    chunk = ChunkRecord(
        id=f"{source_file}::p{page}",
        text="...",
        course="C",
        source_file=source_file,
        category="lecture",
        content_hash="h",
        added_at="2026-07-26T00:00:00+00:00",
        page=page,
    )
    return SearchResult(chunk=chunk, score=score)


def _outcome(gold_pages: set[tuple[str, int]], ranking: list[tuple[str, int]]) -> QueryOutcome:
    """Build an outcome from a hand-written ranking of (file, page) pairs."""
    results = [_result(f, p, 0.9 - 0.01 * i) for i, (f, p) in enumerate(ranking)]
    query = EvalQuery(id="q", query="?", gold=gold_pages)
    return QueryOutcome(query=query, results=results, rank=first_hit_rank(results, gold_pages))


# --------------------------------------------------------------------------- #
# Rank finding
# --------------------------------------------------------------------------- #


def test_first_hit_rank_is_one_based_and_takes_the_earliest_match():
    outcome = _outcome({("a.pdf", 2)}, [("a.pdf", 9), ("a.pdf", 2), ("a.pdf", 2)])
    assert outcome.rank == 2


def test_a_hit_needs_both_file_and_page_to_match():
    """The right page number in the wrong document is not a hit."""
    outcome = _outcome({("a.pdf", 2)}, [("b.pdf", 2), ("a.pdf", 3)])
    assert outcome.rank is None
    assert not outcome.hit


def test_any_labelled_page_counts_as_the_same_answer():
    """``expected`` enumerates acceptable locations of one answer, not several answers."""
    outcome = _outcome({("a.pdf", 21), ("a.pdf", 30)}, [("a.pdf", 30)])
    assert outcome.rank == 1


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_recall_and_mrr_on_a_hand_computed_run():
    """Golds at ranks 1, 2 and 4, plus one miss.

    recall@1 = 1/4, recall@3 = 2/4, recall@5 = 3/4.
    MRR = (1/1 + 1/2 + 1/4 + 0) / 4 = 1.75 / 4 = 0.4375.
    """
    outcomes = [
        _outcome({("a.pdf", 1)}, [("a.pdf", 1)]),
        _outcome({("a.pdf", 2)}, [("x.pdf", 9), ("a.pdf", 2)]),
        _outcome({("a.pdf", 3)}, [("x.pdf", 9)] * 3 + [("a.pdf", 3)]),
        _outcome({("a.pdf", 4)}, [("x.pdf", 9)] * 5),
    ]

    report = score_run(outcomes, k=10)

    assert report.recall[1] == 0.25
    assert report.recall[3] == 0.5
    assert report.recall[5] == 0.75
    assert report.recall[10] == 0.75
    assert report.mrr == pytest.approx(0.4375)
    assert [o.query.id for o in report.misses] == ["q"]
    assert len(report.misses) == 1


def test_mrr_is_truncated_at_k_rather_than_credited():
    """A gold at rank 11 scores 0, not 1/11 — the metric is MRR@k, and says so."""
    outcome = _outcome({("a.pdf", 1)}, [("x.pdf", 9)] * 10 + [("a.pdf", 1)])
    assert outcome.rank == 11

    report = score_run([outcome], k=10)

    assert report.mrr == 0.0
    assert report.recall[10] == 0.0
    assert report.misses == [outcome]


def test_recall_cutoffs_deeper_than_k_are_not_reported():
    """Reporting recall@10 from a k=3 run would be a claim the run cannot support."""
    report = score_run([_outcome({("a.pdf", 1)}, [("a.pdf", 1)])], k=3)
    assert sorted(report.recall) == [1, 3]


def _run_from_ranks(ranks: list[int | None], k: int = 10) -> list[QueryOutcome]:
    """Synthesize outcomes that place the gold at each given rank (None = never)."""
    outcomes = []
    for i, rank in enumerate(ranks):
        gold = {("gold.pdf", 1)}
        ranking = [("filler.pdf", 9)] * k
        if rank is not None and rank <= k:
            ranking[rank - 1] = ("gold.pdf", 1)
        outcome = _outcome(gold, ranking)
        outcome.query.id = f"q{i}"
        outcomes.append(outcome)
    return outcomes


def test_metrics_at_thirty_queries_match_a_hand_computation():
    """The real eval-set scale, with every number computed independently by hand.

    The harness had two bugs at 12 queries that only testing caught, so its arithmetic
    is pinned at the size it actually runs at rather than assumed to generalize. These
    are the exact ranks the CS240 baseline produced: 24 firsts, two seconds, and one
    each at 3, 5 and 6, plus one query that never finds its answer.

    recall@1 = 24/30, @3 = 27/30, @5 = 28/30, @10 = 29/30.
    MRR@10 = (24 + 2*(1/2) + 1/3 + 1/5 + 1/6) / 30 = 257/300.
    """
    ranks = [1] * 24 + [2, 2, 3, 5, 6, None]
    assert len(ranks) == 30

    report = score_run(_run_from_ranks(ranks), k=10)

    assert report.recall[1] == pytest.approx(24 / 30)
    assert report.recall[3] == pytest.approx(27 / 30)
    assert report.recall[5] == pytest.approx(28 / 30)
    assert report.recall[10] == pytest.approx(29 / 30)
    assert report.mrr == pytest.approx(257 / 300)
    assert len(report.misses) == 1


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_miss_count_always_agrees_with_recall(k):
    """The invariant the ``hit_at`` bug broke: misses and recall must describe one run.

    Previously ``misses`` ignored the cutoff, so a gold at rank 11 scored 0 for MRR
    while not appearing in the miss list — the report contradicted its own number. Any
    future divergence between the two views fails here.
    """
    ranks = [1] * 24 + [2, 2, 3, 5, 6, None]
    report = score_run(_run_from_ranks(ranks), k=k)

    assert len(report.misses) + round(report.recall[k] * len(ranks)) == len(ranks)
    assert all(not o.hit_at(k) for o in report.misses)


def test_scoring_an_empty_run_is_an_error():
    with pytest.raises(ValueError, match="empty run"):
        score_run([], k=10)


def test_outcome_exposes_top_and_gold_scores_separately():
    """A miss still has a top-1 score — that is the number a threshold gets calibrated on."""
    hit = _outcome({("a.pdf", 2)}, [("x.pdf", 9), ("a.pdf", 2)])
    assert hit.top_score == pytest.approx(0.90)
    assert hit.gold_score == pytest.approx(0.89)

    miss = _outcome({("a.pdf", 2)}, [("x.pdf", 9)])
    assert miss.top_score == pytest.approx(0.90)
    assert miss.gold_score is None


def test_score_spread_formats_min_median_max():
    assert score_spread([0.5, 0.9, 0.7]) == "0.500 / 0.700 / 0.900"
    assert score_spread([]) == "(none)"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _write(tmp_path, payload):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_expands_pages_into_a_gold_set(tmp_path):
    path = _write(
        tmp_path,
        {
            "course": "CS240",
            "default_k": 7,
            "queries": [
                {
                    "id": "m3",
                    "query": "how is tower height chosen",
                    "expected": [
                        {"source_file": "module05.pdf", "pages": [21, 30]},
                        {"source_file": "a4.pdf", "pages": [1]},
                    ],
                    "note": "prose slide and pseudocode slide",
                }
            ],
        },
    )

    eval_set = load_eval_set(path)

    assert eval_set.course == "CS240"
    assert eval_set.default_k == 7
    assert eval_set.queries[0].gold == {
        ("module05.pdf", 21),
        ("module05.pdf", 30),
        ("a4.pdf", 1),
    }
    assert eval_set.queries[0].note == "prose slide and pseudocode slide"


def test_a_query_labelling_no_pages_is_rejected(tmp_path):
    """An unlabelled query would silently count as a permanent miss and drag the score."""
    path = _write(
        tmp_path,
        {
            "course": "C",
            "queries": [{"id": "bad", "query": "?", "expected": [{"source_file": "a.pdf", "pages": []}]}],
        },
    )
    with pytest.raises(ValueError, match="labels no expected pages"):
        load_eval_set(path)


def test_an_empty_eval_set_is_rejected(tmp_path):
    path = _write(tmp_path, {"course": "C", "queries": []})
    with pytest.raises(ValueError, match="no queries"):
        load_eval_set(path)


def test_the_committed_cs240_set_is_valid_and_blind_authored():
    """The real eval set parses and holds the ~25-30 questions the baseline is scored on."""
    eval_set = load_eval_set(REPO_ROOT / "evals" / "CS240.json")

    assert eval_set.course == "CS240"
    assert 25 <= len(eval_set.queries) <= 30
    assert len({q.id for q in eval_set.queries}) == len(eval_set.queries)
    for query in eval_set.queries:
        assert query.query.strip()
        assert query.gold
        assert query.note.strip(), f"{query.id} has no note explaining why the label is fair"


def test_the_committed_cs240_set_spans_every_material_type():
    """A baseline drawn from one document type would not generalize to the course."""
    eval_set = load_eval_set(REPO_ROOT / "evals" / "CS240.json")
    labelled = {f for q in eval_set.queries for f, _ in q.gold}

    lectures = {f for f in labelled if f.startswith("module")}
    assignments = {f for f in labelled if f.startswith("a") and f[1:2].isdigit()}

    assert len(lectures) >= 3, f"only {lectures} lecture decks are labelled"
    assert len(assignments) >= 5, f"only {assignments} assignments are labelled"
    assert "tut01sol.pdf" in labelled, "no tutorial question is labelled"


@pytest.mark.parametrize("course", ["CS240", "CS247"])
def test_committed_labels_point_at_real_pages(course):
    """Every gold page must exist in its source PDF.

    A typo'd page number is invisible in the summary — it just looks like a retrieval
    miss forever, quietly dragging the baseline Phase 4 is measured against.
    """
    fitz = pytest.importorskip("fitz")
    corpus = REPO_ROOT / "corpus" / course
    if not corpus.is_dir():
        pytest.skip(f"corpus/{course} not present (source PDFs are gitignored)")

    eval_set = load_eval_set(REPO_ROOT / "evals" / f"{course}.json")
    pages = {}
    for query in eval_set.queries:
        for source_file, page in sorted(query.gold):
            path = corpus / source_file
            assert path.is_file(), f"{query.id} labels missing file {source_file}"
            if source_file not in pages:
                with fitz.open(path) as doc:
                    pages[source_file] = len(doc)
            assert 1 <= page <= pages[source_file], (
                f"{query.id} labels {source_file} p{page}, but it has {pages[source_file]} pages"
            )


def test_the_committed_cs247_set_is_valid_and_covers_every_question_shape():
    """CS247 exists to give an independent read, so it must not clone CS240's set."""
    cs247 = load_eval_set(REPO_ROOT / "evals" / "CS247.json")
    cs240 = load_eval_set(REPO_ROOT / "evals" / "CS240.json")

    assert cs247.course == "CS247"
    assert 25 <= len(cs247.queries) <= 30
    assert len({q.id for q in cs247.queries}) == len(cs247.queries)
    for query in cs247.queries:
        assert query.query.strip() and query.gold and query.note.strip()

    # No question is shared with CS240 — a ported question measures the same thing twice.
    assert not ({q.query for q in cs247.queries} & {q.query for q in cs240.queries})

    # All four designed shapes are present, keyed by the id prefix convention.
    prefixes = {q.id[0] for q in cs247.queries}
    assert prefixes == {"e", "p", "c", "s"}

    labelled = {f for q in cs247.queries for f, _ in q.gold}
    assert sum(1 for f in labelled if f[0].isdigit()) >= 6, "too few lecture decks labelled"
    assert any(f.startswith("A4Q") for f in labelled), "no assignment spec labelled"
    assert any("Project" in f for f in labelled), "no project spec labelled"
